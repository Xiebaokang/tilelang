"""Compile a folder of OverlapPlan JSON files and write CUDA sources."""

from __future__ import annotations

import math
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import tilelang
import torch
from tvm.target import Target

from OverlapPlaner.arch.hopper import HOPPER_CUDA_TARGET
from OverlapPlaner.planner import use_schedule_planner
from OverlapPlaner.serialization import load_plan_json
from OverlapPlaner.tune.operators.workloads import OperatorSpec, SearchWorkload


def list_plan_files(plan_dir: Path) -> tuple[Path, ...]:
    """Return schedule JSON files in ``plan_dir``, excluding manifests."""

    if not plan_dir.is_dir():
        raise FileNotFoundError(f"plan directory does not exist: {plan_dir}")
    files = tuple(
        sorted(
            path
            for path in plan_dir.iterdir()
            if path.is_file()
            and path.suffix == ".json"
            and path.name != "manifest.json"
            and not path.name.endswith(".error.json")
        )
    )
    if not files:
        raise ValueError(f"no schedule JSON files found in {plan_dir}")
    return files


def rank_results(
    results: Sequence[Mapping[str, Any]], n_rank: int
) -> list[dict[str, Any]]:
    """Return the top ``n_rank`` successful results by TFLOPS."""

    if n_rank < 1:
        raise ValueError("n_rank must be positive")
    ordered = sorted(results, key=lambda item: item["tflops"], reverse=True)
    ranked = []
    for rank, item in enumerate(ordered[:n_rank], start=1):
        payload = dict(item)
        payload["rank"] = rank
        ranked.append(payload)
    return ranked


def _require_hopper_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    major, minor = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError(
            f"OverlapPlan requires a Hopper GPU, found sm_{major}{minor}"
        )


def _cuda_target() -> Target:
    return HOPPER_CUDA_TARGET


def _schedule_index(path: Path, fallback: int) -> int:
    prefix = "schedule_"
    if path.stem.startswith(prefix) and path.stem[len(prefix) :].isdigit():
        return int(path.stem[len(prefix) :])
    return fallback


def _compile_and_benchmark(
    workload: SearchWorkload,
    tile: Mapping[str, int],
    source_path: str | Path | None,
    warmup: int,
    rep: int,
    phase_callback: Callable[[str], None] | None,
) -> dict[str, Any]:
    target = _cuda_target()
    if phase_callback is not None:
        phase_callback("compilation")
    with target:
        compiled = tilelang.compile(
            workload.prim_func,
            out_idx=list(workload.out_idx),
            target=target,
            execution_backend="cython",
            pass_configs=dict(workload.pass_configs),
        )

    source_file = None
    if source_path is not None:
        source_path = Path(source_path)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(compiled.get_kernel_source(), encoding="utf-8")
        source_file = str(source_path)

    profiler = compiled.get_profiler()
    inputs = (
        None
        if workload.input_tensors is None
        else list(workload.input_tensors)
    )
    if phase_callback is not None:
        phase_callback("correctness_validation")
    profiler.assert_allclose(
        workload.reference_program,
        input_tensors=inputs,
        rtol=0.01,
        atol=0.01,
    )
    if phase_callback is not None:
        phase_callback("benchmark")
    latency_ms = float(
        profiler.do_bench_iterations(
            warmup=warmup, rep=rep, input_tensors=inputs
        )
    )
    if not math.isfinite(latency_ms) or latency_ms <= 0:
        raise RuntimeError(f"invalid benchmark latency {latency_ms}")
    return {
        "source_file": source_file,
        "latency_ms": latency_ms,
        "tflops": workload.total_flops / latency_ms * 1e-9,
        "tile": dict(tile),
    }


def evaluate(
    operator: OperatorSpec,
    tile: Mapping[str, int],
    options: Mapping[str, Any],
    schedule_path: str | Path,
    source_path: str | Path | None = None,
    warmup: int = 100,
    rep: int = 400,
    phase_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compile one searched plan, optionally save CUDA source, then benchmark."""

    if warmup < 0 or rep < 1:
        raise ValueError("warmup must be non-negative and rep must be positive")
    _require_hopper_gpu()
    schedule_path = Path(schedule_path).resolve()
    if not schedule_path.is_file():
        raise FileNotFoundError(f"schedule JSON does not exist: {schedule_path}")
    plan = load_plan_json(schedule_path)
    torch.manual_seed(0)
    workload = operator.build(options, dict(tile))
    auto_overlap = (
        workload.prim_func.attrs.get("tl.auto_overlap")
        if workload.prim_func.attrs is not None
        else None
    )
    if auto_overlap is None or int(auto_overlap) == 0:
        raise ValueError(
            f"{operator.name} must set @T.prim_func(auto_overlap=True); "
            "OverlapPlan has no default schedule"
        )

    def planner(_symbol, _function, _compile_target):
        return plan

    target = _cuda_target()
    with use_schedule_planner(planner), target:
        result = _compile_and_benchmark(
            workload,
            tile,
            source_path,
            warmup,
            rep,
            phase_callback,
        )
    return {
        **result,
        "backend": "overlap_plan",
        "schedule_file": str(schedule_path),
        "num_groups": len(plan.groups),
        "effective_threads": sum(
            int(group.warp_count) * 32 for group in plan.groups
        ),
    }


def evaluate_native(
    operator: OperatorSpec,
    tile: Mapping[str, int],
    options: Mapping[str, Any],
    source_path: str | Path | None = None,
    warmup: int = 100,
    rep: int = 400,
    phase_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compile and benchmark one tile with TileLang's native lowering."""

    if warmup < 0 or rep < 1:
        raise ValueError("warmup must be non-negative and rep must be positive")
    _require_hopper_gpu()
    torch.manual_seed(0)
    workload = operator.build_native(options, dict(tile))
    auto_overlap = (
        workload.prim_func.attrs.get("tl.auto_overlap")
        if workload.prim_func.attrs is not None
        else None
    )
    if auto_overlap is not None and int(auto_overlap) != 0:
        raise ValueError(
            f"{operator.name} native workload still enables tl.auto_overlap"
        )
    result = _compile_and_benchmark(
        workload,
        tile,
        source_path,
        warmup,
        rep,
        phase_callback,
    )
    return {**result, "backend": "native"}


def search(
    operator: OperatorSpec,
    tile: Mapping[str, int],
    options: Mapping[str, Any],
    plan_dir: str | Path,
    source_dir: str | Path,
    warmup: int = 100,
    rep: int = 400,
    *,
    skip_schedule_indices: Sequence[int] = (),
    schedule_indices: Sequence[int] | None = None,
    state_callback: Callable[[Mapping[str, Any]], None] | None = None,
    result_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile every plan JSON and save CUDA sources next to the tile."""

    plan_dir = Path(plan_dir).resolve()
    source_dir = Path(source_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    plan_files = list_plan_files(plan_dir)

    print(f"operator={operator.name}")
    print(f"tile={dict(tile)}")
    print(f"plan_dir={plan_dir}")
    print(f"source_dir={source_dir}")
    print(f"plans={len(plan_files)}", flush=True)

    successful: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped = set(skip_schedule_indices)
    selected = None if schedule_indices is None else set(schedule_indices)
    examined = 0
    for index, schedule_path in enumerate(plan_files):
        source_path = source_dir / f"{schedule_path.stem}.cu"
        schedule_index = _schedule_index(schedule_path, index)
        if schedule_index in skipped or (
            selected is not None and schedule_index not in selected
        ):
            continue
        examined += 1
        print(
            f"[{index + 1}/{len(plan_files)}] compiling {schedule_path.name}",
            flush=True,
        )
        if source_path.exists():
            source_path.unlink()
        current_phase = "compilation"

        def update_phase(phase: str) -> None:
            nonlocal current_phase
            current_phase = phase
            if state_callback is not None:
                state_callback(
                    {
                        "schedule_index": schedule_index,
                        "phase": phase,
                        "schedule_file": str(schedule_path),
                    }
                )

        try:
            result = evaluate(
                operator,
                tile,
                options,
                schedule_path,
                source_path,
                warmup,
                rep,
                phase_callback=update_phase,
            )
            result["schedule_index"] = schedule_index
            successful.append(result)
            if result_callback is not None:
                result_callback("successful", result)
            update_phase("completed")
            print(
                f"[{index + 1}/{len(plan_files)}] "
                f"latency={result['latency_ms']:.4f} ms "
                f"tflops={result['tflops']:.3f}",
                flush=True,
            )
        except Exception as error:
            failure = {
                "schedule_index": schedule_index,
                "schedule_file": str(schedule_path),
                "source_file": str(source_path) if source_path.exists() else None,
                "phase": current_phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            if result_callback is not None:
                result_callback("failure", failure)
            update_phase("failed")
            print(
                f"[{index + 1}/{len(plan_files)}] failed: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    return {
        "operator": operator.name,
        "tile": dict(tile),
        "enumerated": len(plan_files),
        "examined": examined,
        "successful": successful,
        "failures": failures,
    }
