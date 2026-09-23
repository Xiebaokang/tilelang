"""Run measured Overlaper searches for an explicitly imported operator list.

Edit :data:`SEARCH_OPERATORS` below to choose the operators to search. New
operators only need to expose the ``OPERATOR`` interface documented in
``operators/README.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from tvm.target import Target
from tilelang.carver.arch.driver import get_max_dynamic_shared_size_bytes

from history.overlaper.candidates import (
    GuidedSearchConfig,
    Schedule,
    enumerate_guided_schedules,
    enumerate_schedules,
    load_schedule_json,
    schedule_to_dict,
)
from history.overlaper.parse import extract_dataflow_graph
from history.overlaper.tune.operators.convolution import OPERATOR as CONVOLUTION
from history.overlaper.tune.operators.fa3 import OPERATOR as FA3
from history.overlaper.tune.operators.gemm import OPERATOR as GEMM
from history.overlaper.tune.operators.gemm_fp8 import OPERATOR as GEMM_FP8
from history.overlaper.tune.operators.gqa import OPERATOR as GQA
from history.overlaper.tune.operators.gqa_bwd import OPERATOR as GQA_BWD
from history.overlaper.tune.operators.mamba_scan import OPERATOR as MAMBA_SCAN
from history.overlaper.tune.operators.mha_bwd import OPERATOR as MHA_BWD
from history.overlaper.tune.operators.mla import OPERATOR as MLA
from history.overlaper.tune.operators.workloads import OperatorSpec
from history.overlaper.tune.search import search_schedules


# This is the only list that needs editing to select searches.
SEARCH_OPERATORS: list[OperatorSpec] = [
    # FA3,
    # GQA,
    # MAMBA_SCAN,
    MLA,
    # MHA_BWD,
    # GQA_BWD,
    # GEMM,
    # CONVOLUTION,
    # GEMM_FP8,
]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except FileNotFoundError:
        return []


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _cuda_target(
    arch: str, max_shared_memory_per_block: int = 253952
) -> Target:
    return Target(
        {
            "kind": "cuda",
            "arch": arch,
            "max_threads_per_block": 1024,
            "max_shared_memory_per_block": max_shared_memory_per_block,
        }
    )


def _config_name(config: Mapping[str, int]) -> str:
    aliases = {"block_m": "m", "block_n": "n", "block_k": "k"}
    return "_".join(
        f"{aliases.get(key, key)}{value}" for key, value in config.items()
    )


def _validate_operators(
    operators: Sequence[OperatorSpec],
) -> tuple[OperatorSpec, ...]:
    result = tuple(operators)
    if not result:
        raise ValueError("SEARCH_OPERATORS cannot be empty")
    if any(not isinstance(operator, OperatorSpec) for operator in result):
        raise TypeError("SEARCH_OPERATORS must contain OperatorSpec objects")
    names = tuple(operator.name for operator in result)
    if len(set(names)) != len(names):
        raise ValueError("SEARCH_OPERATORS cannot contain duplicate operators")
    return result


def _load_operator(module_name: str, expected_name: str) -> OperatorSpec:
    operator = getattr(import_module(module_name), "OPERATOR", None)
    if not isinstance(operator, OperatorSpec) or operator.name != expected_name:
        raise RuntimeError(
            f"{module_name} must export OPERATOR named {expected_name!r}"
        )
    return operator


def _run_worker(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    operator = _load_operator(job["operator_module"], job["operator_name"])
    workload = operator.build(job["options"], job["config"])
    output = Path(job["output_directory"])
    torch.manual_seed(job["seed"])
    shared_memory_limit = job.get("max_shared_memory_per_block")
    target = (
        _cuda_target(job["arch"])
        if shared_memory_limit is None
        else _cuda_target(job["arch"], shared_memory_limit)
    )
    search_schedules(
        workload.prim_func,
        target=target,
        # OperatorSpec keeps this immutable, while TileLang's adapter API
        # intentionally accepts only ``list[int]`` (or a single integer).
        out_idx=list(workload.out_idx),
        total_flops=workload.total_flops,
        reference_program=workload.reference_program,
        input_tensors=(
            None
            if workload.input_tensors is None
            else list(workload.input_tensors)
        ),
        output_directory=output,
        schedule_candidates=_load_schedule_candidates(
            output / "candidates", job["candidate_count"]
        ),
        warmup=job["warmup"],
        rep=job["rep"],
        start_schedule_index=job["start_schedule_index"],
        max_schedules=job["max_schedules"],
        rtol=job["rtol"],
        atol=job["atol"],
        execution_backend=job["execution_backend"],
        pass_configs=dict(workload.pass_configs),
    )


def _load_schedule_candidates(
    candidate_dir: Path, count: int
) -> tuple[Schedule, ...]:
    return tuple(
        load_schedule_json(candidate_dir / f"schedule_{index:05d}.json")
        for index in range(count)
    )


def _terminate(process: subprocess.Popen, grace: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    except KeyboardInterrupt:
        # A second Ctrl+C must not leave a detached worker process group alive.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def _record_process_failure(
    config_dir: Path,
    state: Mapping[str, Any],
    *,
    kind: str,
    message: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    index = int(state.get("schedule_index", -1))
    failures = config_dir / "failures"
    failures.mkdir(parents=True, exist_ok=True)
    saved_schedule = None
    relative_source = state.get("schedule_file")
    if relative_source:
        source = config_dir / str(relative_source)
        if source.exists():
            saved_schedule = failures / f"{kind}_schedule_{index:05d}.json"
            shutil.copyfile(source, saved_schedule)
    detail = {
        "schedule_index": index,
        "phase": state.get("phase", "unknown"),
        "error_type": kind,
        "error": message,
        "schedule_file": (
            None
            if saved_schedule is None
            else str(saved_schedule.relative_to(config_dir))
        ),
        **dict(extra or {}),
    }
    _write_json(failures / f"{kind}_schedule_{index:05d}.error.json", detail)
    with (config_dir / "failures.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(detail, sort_keys=True) + "\n")
    print(
        f"schedule={index} failed phase={detail['phase']} "
        f"type={kind} saved={detail['schedule_file']}",
        flush=True,
    )


def _serializable_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, list)) or value is None
    }


def _make_job(
    args: argparse.Namespace,
    operator: OperatorSpec,
    config: Mapping[str, int],
    config_dir: Path,
    candidate_count: int,
    start: int,
) -> dict[str, Any]:
    return {
        "operator_name": operator.name,
        "operator_module": operator.workload_factory.__module__,
        "config": dict(config),
        "options": _serializable_options(args),
        "output_directory": str(config_dir.resolve()),
        "candidate_count": candidate_count,
        "start_schedule_index": start,
        "max_schedules": args.max_schedules_per_config,
        "warmup": args.warmup,
        "rep": args.rep,
        "seed": args.seed,
        "rtol": args.rtol,
        "atol": args.atol,
        "arch": args.arch,
        "max_shared_memory_per_block": getattr(
            args, "max_shared_memory_per_block", 253952
        ),
        "execution_backend": args.execution_backend,
    }


def _prepare_schedule_candidates(
    args: argparse.Namespace,
    operator: OperatorSpec,
    config: Mapping[str, int],
    candidate_dir: Path,
) -> int:
    """Materialize the selected schedule frontier for one concrete tile."""

    workload = operator.build(vars(args), config)
    graph = extract_dataflow_graph(
        workload.prim_func,
        target=_cuda_target(args.arch, args.max_shared_memory_per_block),
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for path in candidate_dir.glob("schedule_*.json"):
        path.unlink()
    count = 0
    if args.search_mode == "guided":
        iterator = enumerate_guided_schedules(
            graph,
            config=GuidedSearchConfig(
                groups_per_count=args.guided_groups_per_count,
                structures=args.guided_structures,
                schedules=args.guided_schedules,
            ),
        )
    else:
        iterator = enumerate_schedules(graph)
    for index, schedule in enumerate(iterator):
        _write_json(
            candidate_dir / f"schedule_{index:05d}.json",
            schedule_to_dict(graph, schedule),
        )
        count = index + 1
    if count == 0:
        raise RuntimeError("Overlaper did not enumerate a feasible schedule")
    _write_json(
        candidate_dir / "manifest.json",
        {
            "operator": operator.name,
            "tile": dict(config),
            "schedule_count": count,
            "options": _serializable_options(args),
        },
    )
    return count


def _completed_config(
    config_dir: Path,
    config: Mapping[str, int],
    max_schedules: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]] | None:
    """Load a tile only when every requested schedule was already examined."""

    manifest = _read_json(config_dir / "candidates" / "manifest.json")
    if manifest is None or manifest.get("tile") != dict(config):
        return None
    try:
        candidate_count = int(manifest["schedule_count"])
    except (KeyError, TypeError, ValueError):
        return None
    if candidate_count < 1:
        return None

    search_count = (
        candidate_count
        if max_schedules is None
        else min(candidate_count, max_schedules)
    )
    expected = set(range(search_count))

    successful_by_index = {
        item["schedule_index"]: item
        for item in _read_jsonl(config_dir / "results.jsonl")
        if isinstance(item.get("schedule_index"), int)
        and item["schedule_index"] in expected
    }
    failed_by_index = {
        item["schedule_index"]: item
        for item in _read_jsonl(config_dir / "failures.jsonl")
        if isinstance(item.get("schedule_index"), int)
        and item["schedule_index"] in expected
    }
    examined = successful_by_index.keys() | failed_by_index.keys()
    if not expected.issubset(examined):
        return None

    # A successful retry takes precedence over stale failure metadata.
    failed_indices = failed_by_index.keys() - successful_by_index.keys()
    failures = [failed_by_index[index] for index in failed_indices]
    results = [
        successful_by_index[index] for index in sorted(successful_by_index)
    ]
    return results, {
        "enumerated_schedules": candidate_count,
        "examined_schedules": search_count,
        "successful_schedules": len(results),
        "failed_schedules": len(failures),
        "timeout_schedules": sum(
            item.get("error_type") == "timeout" for item in failures
        ),
        "crashed_schedules": sum(
            item.get("error_type") == "worker_exit" for item in failures
        ),
    }


def _supervise_config(
    args: argparse.Namespace,
    operator: OperatorSpec,
    config: Mapping[str, int],
    config_dir: Path,
    candidate_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "failures").mkdir(exist_ok=True)
    (config_dir / "sources").mkdir(exist_ok=True)
    for directory in (config_dir / "failures", config_dir / "sources"):
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()
    for name in (
        "results.jsonl",
        "failures.jsonl",
        "candidate_state.json",
        "job.json",
    ):
        path = config_dir / name
        if path.exists():
            path.unlink()
    start = timeouts = crashes = 0

    while True:
        state: dict[str, Any] = {
            "schedule_index": start,
            "phase": "startup",
            "schedule_file": None,
        }
        _write_json(config_dir / "candidate_state.json", state)
        job_path = config_dir / "job.json"
        _write_json(
            job_path,
            _make_job(
                args,
                operator,
                config,
                config_dir,
                candidate_count,
                start,
            ),
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "overlaper.tune.run",
                "--worker-job",
                str(job_path.resolve()),
            ],
            start_new_session=True,
        )
        active_phase = None
        phase_started = time.monotonic()
        timed_out = False
        try:
            while process.poll() is None:
                updated = _read_json(config_dir / "candidate_state.json")
                if updated is not None:
                    state = updated
                    phase = (state.get("schedule_index"), state.get("phase"))
                    if phase != active_phase:
                        active_phase = phase
                        phase_started = time.monotonic()
                timeout = (
                    args.execution_timeout
                    if state.get("phase")
                    in {"correctness_validation", "benchmark"}
                    else args.compile_timeout
                )
                if time.monotonic() - phase_started > timeout:
                    timed_out = True
                    _terminate(process, args.kill_grace)
                    _record_process_failure(
                        config_dir,
                        state,
                        kind="timeout",
                        message=f"phase exceeded {timeout:g} seconds",
                        extra={"timeout_seconds": timeout},
                    )
                    timeouts += 1
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            print(
                "\nInterrupt received; terminating active worker...",
                file=sys.stderr,
                flush=True,
            )
            _terminate(process, args.kill_grace)
            raise

        if not timed_out and process.returncode == 0:
            break
        if not timed_out:
            _record_process_failure(
                config_dir,
                state,
                kind="worker_exit",
                message=f"worker exited with status {process.returncode}",
                extra={"returncode": process.returncode},
            )
            crashes += 1

        stuck = int(state.get("schedule_index", -1))
        if stuck < 0 or not state.get("schedule_file"):
            raise RuntimeError(
                "worker failed before checkpointing a schedule; see its job "
                "and enumeration error in the tile directory"
            )
        start = stuck + 1
        limit = args.max_schedules_per_config or candidate_count
        if limit and start >= limit:
            break

    all_results = _read_jsonl(config_dir / "results.jsonl")
    failed = len(_read_jsonl(config_dir / "failures.jsonl"))
    for name in ("candidate_state.json", "job.json"):
        (config_dir / name).unlink(missing_ok=True)
    return all_results, {
        "enumerated_schedules": candidate_count,
        "examined_schedules": len(all_results) + failed,
        "successful_schedules": len(all_results),
        "failed_schedules": failed,
        "timeout_schedules": timeouts,
        "crashed_schedules": crashes,
    }


def _write_top30(
    operator_dir: Path,
    operator: OperatorSpec,
    results: Sequence[Mapping[str, Any]],
    configurations: Sequence[Mapping[str, Any]],
) -> None:
    ranked = sorted(results, key=lambda item: item["tflops"], reverse=True)[:30]
    top_results = []
    for rank, result in enumerate(ranked, start=1):
        top_results.append(
            {
                "rank": rank,
                "tile": result["tile"],
                "schedule_index": result["schedule_index"],
                "latency_ms": result["latency_ms"],
                "tflops": result["tflops"],
                "schedule_file": result["schedule_file"],
                "source_file": result["source_file"],
            }
        )
    _write_json(
        operator_dir / "top30.json",
        {
            "operator": operator.name,
            "description": operator.description,
            "configurations": list(configurations),
            "top_results": top_results,
        },
    )
    print(f"\n=== {operator.name} Top-30 ===", flush=True)
    for item in top_results:
        print(
            f"#{item['rank']:02d} tile={item['tile']} "
            f"schedule={item['schedule_index']} "
            f"latency={item['latency_ms']:.4f} ms "
            f"tflops={item['tflops']:.2f}",
            flush=True,
        )


def run(
    operators: Sequence[OperatorSpec],
    args: argparse.Namespace,
) -> None:
    selected = _validate_operators(operators)
    args.output.mkdir(parents=True, exist_ok=True)
    search_summary = []
    for operator in selected:
        operator_dir = args.output / operator.name
        operator_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        config_summaries: list[dict[str, Any]] = []
        for config in operator.configurations(vars(args)):
            config_name = _config_name(config)
            config_dir = operator_dir / config_name
            completed = _completed_config(
                config_dir, config, args.max_schedules_per_config
            )
            print(
                f"\n=== operator={operator.name} tile={config} ===",
                flush=True,
            )
            if completed is not None:
                found, summary = completed
                print(
                    f"Skipping completed tile: examined "
                    f"{summary['examined_schedules']} schedules.",
                    flush=True,
                )
                config_summaries.append({"tile": config, **summary})
                for item in found:
                    normalized = {"tile": config, **item}
                    for key in ("schedule_file", "source_file"):
                        normalized[key] = os.path.normpath(
                            str(Path(config_name) / str(normalized[key]))
                        )
                    results.append(normalized)
                _write_top30(
                    operator_dir, operator, results, config_summaries
                )
                continue
            try:
                candidate_count = _prepare_schedule_candidates(
                    args, operator, config, config_dir / "candidates"
                )
                search_count = (
                    candidate_count
                    if args.max_schedules_per_config is None
                    else min(
                        candidate_count, args.max_schedules_per_config
                    )
                )
                print(
                    f"Generated {candidate_count} schedule candidates; "
                    f"searching {search_count}.",
                    flush=True,
                )
                found, summary = _supervise_config(
                    args,
                    operator,
                    config,
                    config_dir,
                    candidate_count,
                )
                config_summaries.append({"tile": config, **summary})
                for item in found:
                    normalized = {
                        "tile": config,
                        **item,
                    }
                    for key in ("schedule_file", "source_file"):
                        normalized[key] = os.path.normpath(
                            str(Path(config_name) / str(normalized[key]))
                        )
                    results.append(normalized)
            except Exception as error:
                config_summaries.append(
                    {
                        "tile": config,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                print(
                    f"tile failed: {type(error).__name__}: {error}",
                    flush=True,
                )
            _write_top30(operator_dir, operator, results, config_summaries)

        search_summary.append(
            {
                "operator": operator.name,
                "configurations": config_summaries,
                "top30_file": str(
                    (operator_dir / "top30.json").relative_to(args.output)
                ),
            }
        )
        _write_json(args.output / "summary.json", search_summary)


def make_parser(
    operators: Sequence[OperatorSpec] | None = None,
) -> argparse.ArgumentParser:
    selected = _validate_operators(
        SEARCH_OPERATORS if operators is None else operators
    )
    parser = argparse.ArgumentParser(
        description="Search the OperatorSpec list imported in overlaper.tune.run"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("overlaper_tune_results")
    )
    parser.add_argument(
        "--warmup", type=int, default=100, help="kernel warmup iterations"
    )
    parser.add_argument(
        "--rep", type=int, default=400, help="measured kernel iterations"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--arch", default="sm_90a")
    parser.add_argument(
        "--max-shared-memory-per-block",
        type=int,
        help="override CUDA's opt-in dynamic shared-memory limit in bytes",
    )
    parser.add_argument("--compile-timeout", type=float, default=30.0)
    parser.add_argument("--execution-timeout", type=float, default=15.0)
    parser.add_argument("--kill-grace", type=float, default=2.0)
    parser.add_argument("--execution-backend", default="cython")
    parser.add_argument("--max-schedules-per-config", type=int)
    parser.add_argument(
        "--search-mode",
        choices=("guided", "exhaustive"),
        default="guided",
        help="budgeted hardware-aware search or the complete schedule space",
    )
    parser.add_argument("--guided-groups-per-count", type=int, default=16)
    parser.add_argument("--guided-structures", type=int, default=2048)
    parser.add_argument("--guided-schedules", type=int, default=128)
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    for operator in selected:
        operator.add_arguments(parser)
    return parser


def main(
    operators: Sequence[OperatorSpec] | None = None,
    argv: list[str] | None = None,
) -> None:
    selected = tuple(SEARCH_OPERATORS if operators is None else operators)
    parser = make_parser(selected)
    args = parser.parse_args(argv)
    if args.worker_job is not None:
        try:
            _run_worker(args.worker_job)
        except Exception as error:
            print(
                f"worker failed: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1) from None
        return
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("Overlaper tuning requires a Hopper CUDA GPU")
    if args.max_shared_memory_per_block is None:
        args.max_shared_memory_per_block = get_max_dynamic_shared_size_bytes()
    if (
        args.max_shared_memory_per_block is None
        or args.max_shared_memory_per_block <= 0
    ):
        parser.error("cannot determine a positive shared-memory limit")
    if min(args.compile_timeout, args.execution_timeout) <= 0:
        parser.error("timeouts must be positive")
    if args.warmup < 0 or args.rep < 1:
        parser.error("warmup must be non-negative and rep must be positive")
    if (
        args.max_schedules_per_config is not None
        and args.max_schedules_per_config < 1
    ):
        parser.error("max schedules must be positive")
    if min(
        args.guided_groups_per_count,
        args.guided_structures,
        args.guided_schedules,
    ) < 1:
        parser.error("guided search budgets must be positive")
    try:
        run(selected, args)
    except KeyboardInterrupt:
        print(
            "Search interrupted; active worker terminated.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()


"""
PYTHONPATH="$PWD/3rdparty/tvm/python:$PWD" \
TVM_LIBRARY_PATH="$PWD/build/lib" \
python -u -m overlaper.tune.run --output results > log 2>&1

"""
