from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Callable, Mapping

import torch


def scratch_allocator(size: int, align: int, stream):
    """Allocator required by in-kernel tensor descriptors."""
    del align, stream
    return torch.empty(size, device="cuda", dtype=torch.int8)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OVERLAP_RESULTS = ROOT / "OverlapPlaner" / "tune" / "results"


def benchmark_cuda(
    fn: Callable[[], object],
    *,
    warmup: int = 100,
    rep: int = 400,
    trials: int = 5,
) -> tuple[float, list[float]]:
    """Return median kernel latency in milliseconds.

    Outputs are allocated by callers. CUDA events therefore cover GPU work and
    launch ordering, without charging either framework for tensor allocation.
    """

    fn()  # Compile and autotune outside the timed region.
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / rep)
    return statistics.median(samples), samples


def controlled_stage_sweep(
    launch: Callable[[int, bool], object],
    config: dict[str, object],
    *,
    warmup: int,
    rep: int,
    trials: int,
    stages: tuple[int, ...] = (1, 2, 3, 4),
) -> list[dict[str, object]]:
    """Measure stage depth and WS while holding tile and input warps fixed."""

    variants: list[dict[str, object]] = []
    for warp_specialize in (False, True):
        for num_stages in stages:
            record: dict[str, object] = {
                "name": f"controlled_stage_{num_stages}_{'ws' if warp_specialize else 'no_ws'}",
                "warp_specialize": warp_specialize,
                "num_stages": num_stages,
                "descriptor_loads": True,
                "triton_config": {**config, "num_stages": num_stages},
            }
            try:
                latency, samples = benchmark_cuda(
                    lambda ns=num_stages, ws=warp_specialize: launch(ns, ws),
                    warmup=warmup,
                    rep=rep,
                    trials=trials,
                )
                record.update(status="ok", latency_ms=latency, samples_ms=samples)
            except Exception as error:
                record.update(
                    status="unsupported",
                    error=f"{type(error).__name__}: {error}",
                )
            variants.append(record)
    return variants


def benchmark_autotuned_variant(
    launch: Callable[[], object],
    kernel,
    *,
    warp_specialize: bool,
    warmup: int,
    rep: int,
    trials: int,
    descriptor_loads: bool = True,
) -> dict[str, object]:
    """Benchmark one broad-autotune branch and preserve its selected config."""

    record: dict[str, object] = {
        "name": f"autotuned_{'ws' if warp_specialize else 'no_ws'}",
        "warp_specialize": warp_specialize,
        "descriptor_loads": descriptor_loads,
        "search_kind": "broad_autotune",
    }
    try:
        latency, samples = benchmark_cuda(
            launch, warmup=warmup, rep=rep, trials=trials
        )
        record.update(
            status="ok",
            latency_ms=latency,
            samples_ms=samples,
            triton_config=triton_config(kernel),
        )
    except Exception as error:
        record.update(
            status="unsupported",
            error=f"{type(error).__name__}: {error}",
        )
    return record


def triton_config(kernel) -> dict[str, object] | None:
    config = getattr(kernel, "best_config", None)
    if config is None:
        return None
    return {
        "kwargs": dict(config.kwargs),
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "num_ctas": config.num_ctas,
        "maxnreg": config.maxnreg,
    }


def overlap_result(
    operator: str,
    workload: Mapping[str, object],
    root: Path = DEFAULT_OVERLAP_RESULTS,
) -> dict:
    path = root / operator / "top30.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = next(
        row
        for row in payload["top_results"]
        if all(row.get("workload", {}).get(key) == value for key, value in workload.items())
    )
    return {
        "latency_ms": best["latency_ms"],
        "native_latency_ms": best["native_latency_ms"],
        "speedup_vs_native": best["speedup_vs_native"],
        "tile": best["tile"],
        "schedule_index": best["schedule_index"],
        "source_file": str(path.parent / best["source_file"]),
        "result_file": str(path),
    }


def comparison(
    operator: str,
    latency_ms: float,
    *,
    total_flops: float,
    workload: Mapping[str, object],
) -> dict:
    result = {
        "operator": operator,
        "triton_latency_ms": latency_ms,
        "triton_tflops": total_flops / latency_ms * 1.0e-9,
    }
    try:
        overlap = overlap_result(operator, workload)
    except (FileNotFoundError, KeyError, IndexError, StopIteration):
        return result
    result.update(
        overlapplaner_reference=overlap,
        overlapplaner_speedup_vs_triton=latency_ms / overlap["latency_ms"],
    )
    return result
