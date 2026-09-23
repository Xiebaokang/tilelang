"""Compile, validate, and benchmark Overlaper schedule candidates."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tilelang
from tvm import tirx
from tvm.target import Target

from ..candidates import Schedule, enumerate_schedules, schedule_to_dict
from ..integration import use_schedule_planner
from ..parse import DataflowGraph


class PtxasPerformanceLoss(RuntimeError):
    """ptxas reported a Potential Performance Loss; skip kernel execution."""


def ptxas_performance_loss_warnings(log: str) -> tuple[str, ...]:
    """Return ptxas lines that declare a Potential Performance Loss."""

    return tuple(
        line.strip()
        for line in log.splitlines()
        if "Potential Performance Loss" in line
    )


def _replay_captured_log(log: str) -> None:
    if not log:
        return
    sys.stderr.write(log if log.endswith("\n") else log + "\n")
    sys.stderr.flush()


@contextmanager
def _capture_stdio() -> Iterator[list[str]]:
    """Capture fd 1/2 so nvcc/ptxas output is visible to the search loop."""

    chunks: list[str] = []
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    with tempfile.TemporaryFile() as buffer:
        try:
            os.dup2(buffer.fileno(), 1)
            os.dup2(buffer.fileno(), 2)
            yield chunks
        finally:
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:
                    pass
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
            buffer.seek(0)
            chunks.append(buffer.read().decode("utf-8", errors="replace"))


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    schedule_index: int
    latency_ms: float
    tflops: float
    num_groups: int
    num_stages_by_region: dict[int, int]
    effective_threads: int
    schedule_file: str
    source_file: str


@dataclass(frozen=True, slots=True)
class SearchSummary:
    enumerated_schedules: int
    examined_schedules: int
    successful_schedules: int
    failed_schedules: int
    output_directory: str


def _write_json(path: Path, payload: Any) -> None:
    """Atomically publish JSON so an external watchdog never reads half a file."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        output.flush()


def _error_summary(error: Exception, limit: int = 240) -> str:
    """Keep exhaustive search logs readable; artifacts retain the full error."""

    first_line = str(error).splitlines()[0] if str(error) else type(error).__name__
    return first_line if len(first_line) <= limit else first_line[: limit - 3] + "..."


def _graph_fingerprint(graph: DataflowGraph) -> tuple[Any, ...]:
    return (
        tuple(graph.region_kinds),
        tuple(
            (node.node_id, node.region_id, node.name, node.reads, node.writes)
            for node in graph.nodes
        ),
        tuple(
            (buffer.buffer_id, buffer.name, buffer.scope, buffer.nbytes)
            for buffer in graph.buffers
        ),
    )


def _state(
    output: Path,
    schedule_index: int,
    phase: str,
    schedule_file: str | None = None,
) -> None:
    _write_json(
        output / "candidate_state.json",
        {
            "schedule_index": schedule_index,
            "phase": phase,
            "schedule_file": schedule_file,
        },
    )


def search_schedules(
    prim_func: tirx.PrimFunc,
    *,
    target: Target,
    out_idx: Sequence[int] | int,
    total_flops: float,
    reference_program: Callable[..., Any],
    input_tensors: list[Any] | None = None,
    output_directory: str | Path = "overlaper_search",
    schedule_candidates: Sequence[Schedule] | None = None,
    warmup: int = 15,
    rep: int = 40,
    start_schedule_index: int = 0,
    max_schedules: int | None = None,
    original_threads: int | None = None,
    rtol: float = 0.01,
    atol: float = 0.01,
    execution_backend: str = "cython",
    pass_configs: dict[str, Any] | None = None,
) -> SearchSummary:
    """Run one resumable search segment.

    This function deliberately does not catch process hangs. It checkpoints the
    active schedule and phase; :mod:`overlaper.tune.run` supervises it
    from a separate process and can safely kill a deadlocked CUDA context.
    """

    if not isinstance(prim_func, tirx.PrimFunc):
        raise TypeError("prim_func must be a TIRX PrimFunc")
    if total_flops <= 0 or rep < 1 or warmup < 0:
        raise ValueError("total_flops and rep must be positive")
    if start_schedule_index < 0:
        raise ValueError("start_schedule_index cannot be negative")
    if max_schedules is not None and max_schedules < 1:
        raise ValueError("max_schedules must be positive or None")
    if schedule_candidates is not None and not schedule_candidates:
        raise ValueError("schedule_candidates cannot be empty")

    output = Path(output_directory)
    candidates_dir = output / "candidates"
    sources_dir = output / "sources"
    failures_dir = output / "failures"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    failures_path = output / "failures.jsonl"

    schedules = list(schedule_candidates or ())
    graph_ref: DataflowGraph | None = None
    fingerprint: tuple[Any, ...] | None = None
    current_index = start_schedule_index
    schedule_file: str | None = None

    def planner(_symbol: str, graph: DataflowGraph, _target: Target) -> Schedule:
        nonlocal graph_ref, fingerprint, schedule_file
        current = _graph_fingerprint(graph)
        if graph_ref is None:
            graph_ref = graph
            fingerprint = current
            if not schedules:
                schedules.extend(
                    enumerate_schedules(graph, original_threads=original_threads)
                )
            if not schedules:
                raise RuntimeError("Overlaper did not enumerate a feasible schedule")
        elif current != fingerprint:
            raise RuntimeError("LayoutReducer produced unstable operation IDs")
        if current_index >= len(schedules):
            raise IndexError("schedule index exceeds the enumerated search space")
        schedule = schedules[current_index]
        schedule_path = candidates_dir / f"schedule_{current_index:05d}.json"
        if not schedule_path.exists():
            _write_json(schedule_path, schedule_to_dict(graph, schedule))
        schedule_file = str(schedule_path.relative_to(output))
        _state(output, current_index, "compilation", schedule_file)
        return schedule

    examined = successful = failed = 0
    while not schedules or current_index < len(schedules):
        if max_schedules is not None and current_index >= max_schedules:
            break
        examined += 1
        schedule_file = None
        source_file: str | None = None
        phase = "compilation"
        _state(output, current_index, phase)
        try:
            candidate_prim = prim_func.with_attr(
                "tl.program_schedule.request",
                f"overlaper-search-v1={current_index}",
            )
            compile_log = ""
            captured: list[str] = []
            try:
                with (
                    use_schedule_planner(planner),
                    target,
                    _capture_stdio() as captured,
                ):
                    compiled = tilelang.compile(
                        candidate_prim,
                        out_idx=out_idx,
                        target=target,
                        execution_backend=execution_backend,
                        pass_configs=pass_configs,
                    )
            finally:
                compile_log = captured[0] if captured else ""
                _replay_captured_log(compile_log)
            assert graph_ref is not None
            schedule = schedules[current_index]
            assert schedule_file is not None
            source_path = sources_dir / f"schedule_{current_index:05d}.cu"
            source_path.write_text(compiled.get_kernel_source(), encoding="utf-8")
            source_file = str(source_path.relative_to(output))

            phase = "ptxas_check"
            _state(output, current_index, phase, schedule_file)
            warnings = ptxas_performance_loss_warnings(compile_log)
            if warnings:
                raise PtxasPerformanceLoss(
                    "skipping execution after ptxas Potential Performance Loss:\n"
                    + "\n".join(warnings)
                )

            phase = "correctness_validation"
            _state(output, current_index, phase, schedule_file)
            profiler = compiled.get_profiler()
            profiler.assert_allclose(
                reference_program,
                input_tensors=input_tensors,
                rtol=rtol,
                atol=atol,
            )

            phase = "benchmark"
            _state(output, current_index, phase, schedule_file)
            latency_ms = float(
                profiler.do_bench_iterations(
                    warmup=warmup,
                    rep=rep,
                    input_tensors=input_tensors,
                )
            )
            tflops = total_flops / latency_ms * 1e-9
            if not math.isfinite(latency_ms) or latency_ms <= 0:
                raise RuntimeError(f"invalid benchmark latency {latency_ms}")
            result = BenchmarkResult(
                schedule_index=current_index,
                latency_ms=latency_ms,
                tflops=tflops,
                num_groups=schedule.num_groups,
                num_stages_by_region={
                    region_id: max(stages.values()) + 1
                    for region_id, stages in schedule.stages_by_region.items()
                },
                effective_threads=schedule.warp_allocation.effective_threads,
                schedule_file=schedule_file,
                source_file=source_file,
            )
            _append_jsonl(results_path, asdict(result))
            successful += 1
            print(
                f"schedule={current_index} latency={latency_ms:.4f} ms "
                f"throughput={tflops:.2f} TFLOPS",
                flush=True,
            )
            _state(
                output, current_index, "completed", schedule_file
            )
        except Exception as error:
            if graph_ref is None:
                raise
            if not schedules:
                detail = {
                    "schedule_index": None,
                    "phase": "enumeration",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "schedule_file": None,
                }
                _write_json(output / "enumeration.error.json", detail)
                _append_jsonl(failures_path, detail)
                raise
            failed += 1
            schedule = schedules[current_index]
            failure_schedule = failures_dir / f"schedule_{current_index:05d}.json"
            _write_json(failure_schedule, schedule_to_dict(graph_ref, schedule))
            detail = {
                "schedule_index": current_index,
                "phase": phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "schedule_file": str(failure_schedule.relative_to(output)),
            }
            if source_file is not None:
                detail["source_file"] = source_file
            _write_json(
                failures_dir / f"schedule_{current_index:05d}.error.json",
                detail,
            )
            _append_jsonl(failures_path, detail)
            print(
                f"schedule={current_index} failed in {phase}: "
                f"{type(error).__name__}: {_error_summary(error)}",
                flush=True,
            )
            _state(output, current_index, "failed", detail["schedule_file"])

        current_index += 1

    return SearchSummary(
        enumerated_schedules=len(schedules),
        examined_schedules=examined,
        successful_schedules=successful,
        failed_schedules=failed,
        output_directory=str(output.resolve()),
    )
