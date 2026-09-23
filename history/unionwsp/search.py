"""Compile, validate, benchmark, and retain the fastest UnionWSP schedules."""

from __future__ import annotations

import heapq
import json
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tilelang
from tvm import tirx
from tvm.target import Target

from . import WSPSchedule, enumerate_wsp_schedules
from .integration import use_schedule_planner
from .parseIR.graph import DataflowGraph, DependencyKind
from .physical import GroupWarpAllocation, SharedMemoryPlan, WarpAllocation
from .schedule import (
    SynchronizationChannel,
    SynchronizationChannelKind,
    SynchronizationScope,
)


@dataclass(frozen=True)
class WSPBenchmarkResult:
    """Measured performance of one successfully compiled schedule."""

    schedule_index: int
    latency_ms: float
    tflops: float
    num_groups: int
    num_stages_by_region: dict[int, int]
    effective_threads: int


@dataclass(frozen=True)
class WSPSearchSummary:
    """Final exhaustive-search counters and descending Top-K results."""

    enumerated_schedules: int
    examined_schedules: int
    successful_schedules: int
    failed_schedules: int
    top_results: tuple[WSPBenchmarkResult, ...]
    output_directory: str


def schedule_to_dict(
    graph: DataflowGraph,
    schedule: WSPSchedule,
) -> dict[str, Any]:
    """Return stable JSON metadata sufficient to review one schedule."""

    return {
        "stages_by_region": {
            str(region_id): {str(node_id): stage for node_id, stage in stages.items()}
            for region_id, stages in schedule.stages_by_region.items()
        },
        "groups": {str(node_id): group for node_id, group in schedule.groups.items()},
        "orders": {
            str(region_id): {
                str(group_id): {
                    str(node_id): position for node_id, position in local_order.items()
                }
                for group_id, local_order in group_orders.items()
            }
            for region_id, group_orders in schedule.orders.items()
        },
        "buffer_versions": {
            graph.buffer_for_id(buffer_id).name: version
            for buffer_id, version in schedule.buffer_versions.items()
        },
        "synchronization": [
            {
                "kind": channel.kind.value,
                "scope": channel.scope.value,
                "producer_id": channel.producer_id,
                "consumer_id": channel.consumer_id,
                "producer_group": channel.producer_group,
                "consumer_group": channel.consumer_group,
                "iteration_distance": channel.iteration_distance,
                "effective_stage_distance": channel.effective_stage_distance,
                "slot_count": channel.slot_count,
                "buffer": (
                    None
                    if channel.buffer_id is None
                    else graph.buffer_for_id(channel.buffer_id).name
                ),
                "dependency_kinds": sorted(
                    kind.value for kind in channel.dependency_kinds
                ),
            }
            for channel in schedule.synchronization
        ],
        "warp_allocation": {
            "effective_threads": schedule.warp_allocation.effective_threads,
            "groups": [asdict(group) for group in schedule.warp_allocation.groups],
            "register_counts": schedule.warp_allocation.register_counts,
            "register_is_increase": schedule.warp_allocation.register_is_increase,
        },
        "shared_memory": {
            "shared_buffer_bytes": schedule.shared_memory.shared_buffer_bytes,
            "synchronization_bytes": schedule.shared_memory.synchronization_bytes,
            "merged_shared_bytes": schedule.shared_memory.merged_shared_bytes,
        },
        "operations": [
            {
                "node_id": node.node_id,
                "name": node.name,
                "region_id": node.region_id,
                "instruction_kind": node.instruction_kind.value,
            }
            for node in graph.nodes
        ],
    }


def schedule_from_dict(
    graph: DataflowGraph,
    payload: dict[str, Any],
) -> WSPSchedule:
    """Rebuild one exact schedule previously emitted by ``schedule_to_dict``.

    Node IDs are meaningful only for the LayoutReducer output from which they
    were extracted.  Validate the recorded operation table before accepting
    the schedule so a stale failure JSON cannot silently modify the wrong IR.
    """

    expected_operations = [
        {
            "node_id": node.node_id,
            "name": node.name,
            "region_id": node.region_id,
            "instruction_kind": node.instruction_kind.value,
        }
        for node in graph.nodes
    ]
    if payload.get("operations") != expected_operations:
        raise ValueError(
            "recorded schedule operations do not match the current "
            "LayoutReducer graph"
        )

    stages_by_region = {
        int(region_id): {
            int(node_id): int(stage)
            for node_id, stage in stages.items()
        }
        for region_id, stages in payload["stages_by_region"].items()
    }
    groups = {
        int(node_id): int(group)
        for node_id, group in payload["groups"].items()
    }
    orders = {
        int(region_id): {
            int(group_id): {
                int(node_id): int(position)
                for node_id, position in local_order.items()
            }
            for group_id, local_order in group_orders.items()
        }
        for region_id, group_orders in payload["orders"].items()
    }

    buffers_by_name = {buffer.name: buffer for buffer in graph.buffers}
    if len(buffers_by_name) != len(graph.buffers):
        raise ValueError("cannot replay a schedule with duplicate buffer names")
    recorded_versions = payload["buffer_versions"]
    if set(recorded_versions) != set(buffers_by_name):
        raise ValueError(
            "recorded buffer versions do not match the current graph"
        )
    buffer_versions = {
        buffer.buffer_id: int(recorded_versions[name])
        for name, buffer in buffers_by_name.items()
    }

    synchronization = tuple(
        SynchronizationChannel(
            kind=SynchronizationChannelKind(item["kind"]),
            scope=SynchronizationScope(item["scope"]),
            producer_id=int(item["producer_id"]),
            consumer_id=int(item["consumer_id"]),
            producer_group=int(item["producer_group"]),
            consumer_group=int(item["consumer_group"]),
            iteration_distance=int(item["iteration_distance"]),
            effective_stage_distance=(
                None
                if item["effective_stage_distance"] is None
                else int(item["effective_stage_distance"])
            ),
            slot_count=int(item["slot_count"]),
            buffer_id=(
                None
                if item["buffer"] is None
                else buffers_by_name[item["buffer"]].buffer_id
            ),
            dependency_kinds=frozenset(
                DependencyKind(kind) for kind in item["dependency_kinds"]
            ),
        )
        for item in payload["synchronization"]
    )

    allocation_payload = payload["warp_allocation"]
    register_counts = allocation_payload.get("register_counts")
    register_is_increase = allocation_payload.get("register_is_increase")
    warp_allocation = WarpAllocation(
        groups=tuple(
            GroupWarpAllocation(
                group_id=int(item["group_id"]),
                first_warp=int(item["first_warp"]),
                warp_count=int(item["warp_count"]),
            )
            for item in allocation_payload["groups"]
        ),
        effective_threads=int(allocation_payload["effective_threads"]),
        register_counts=(
            None
            if register_counts is None
            else tuple(int(value) for value in register_counts)
        ),
        register_is_increase=(
            None
            if register_is_increase is None
            else tuple(bool(value) for value in register_is_increase)
        ),
    )

    shared_payload = payload["shared_memory"]
    if graph.hardware is None:
        raise ValueError("schedule replay requires graph.hardware")
    shared_memory = SharedMemoryPlan(
        shared_buffer_bytes=int(shared_payload["shared_buffer_bytes"]),
        synchronization_bytes=int(shared_payload["synchronization_bytes"]),
        merged_shared_bytes=int(shared_payload["merged_shared_bytes"]),
        shared_memory_capacity_bytes=(
            graph.hardware.shared_memory_capacity_bytes
        ),
        # Physical byte offsets are not consumed by IR schedule application;
        # the backend shared-memory reuse pass recomputes them from the IR.
        shared_allocations=(),
    )

    return WSPSchedule(
        stages_by_region=stages_by_region,
        groups=groups,
        orders=orders,
        buffer_versions=buffer_versions,
        synchronization=synchronization,
        warp_allocation=warp_allocation,
        shared_memory=shared_memory,
    )


def _print_schedule(
    graph: DataflowGraph,
    schedule: WSPSchedule,
    schedule_index: int,
    schedule_count: int,
) -> None:
    """Print one complete schedule before compiling its generated kernel."""

    print(
        f"\n=== Schedule [{schedule_index + 1}/{schedule_count}] ===",
        flush=True,
    )
    print(
        json.dumps(
            schedule_to_dict(graph, schedule),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def _print_generated_code(
    source: str,
    schedule_index: int,
    schedule_count: int,
) -> None:
    """Print generated device code before correctness and performance runs."""

    print(
        f"\n=== Generated code [{schedule_index + 1}/{schedule_count}] ===",
        flush=True,
    )
    print(source, flush=True)


def _print_performance(
    latency_ms: float,
    tflops: float,
    schedule_index: int,
    schedule_count: int,
) -> None:
    """Print performance only after the candidate has executed successfully."""

    print(
        f"\n=== Performance [{schedule_index + 1}/{schedule_count}] ===",
        flush=True,
    )
    print(f"latency: {latency_ms:.4f} ms", flush=True)
    print(f"throughput: {tflops:.2f} TFLOPS", flush=True)


def _describe_input_tensors(input_tensors: list[Any] | None) -> list[dict[str, Any]]:
    """Return JSON-friendly tensor metadata without recording tensor contents."""

    if input_tensors is None:
        return []
    descriptions = []
    for index, tensor in enumerate(input_tensors):
        shape = getattr(tensor, "shape", None)
        descriptions.append(
            {
                "index": index,
                "type": type(tensor).__name__,
                "shape": None if shape is None else [str(value) for value in shape],
                "dtype": str(getattr(tensor, "dtype", "unknown")),
                "device": str(getattr(tensor, "device", "unknown")),
            }
        )
    return descriptions


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON checkpoint visible to a supervisor process."""

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_candidate_state(
    output_directory: Path,
    schedule_index: int,
    phase: str,
    schedule_count: int,
    *,
    schedule_file: str | None = None,
    source_file: str | None = None,
) -> None:
    """Publish the active phase so an external watchdog can detect a hang."""

    _write_json_atomic(
        output_directory / "candidate_state.json",
        {
            "schedule_index": schedule_index,
            "phase": phase,
            "enumerated_schedule_count": schedule_count,
            "schedule_file": schedule_file,
            "source_file": source_file,
        },
    )


def _write_failure_artifacts(
    output_directory: Path,
    failure: dict[str, Any],
    kernel_source: str | None,
) -> dict[str, Any]:
    """Persist a detailed failure record and any successfully generated source."""

    failure_directory = output_directory / "failures"
    failure_directory.mkdir(parents=True, exist_ok=True)
    schedule_index = int(failure["schedule_index"])
    stem = f"schedule_{schedule_index:05d}"

    record = dict(failure)
    if kernel_source is None:
        record["source_file"] = None
    else:
        source_path = failure_directory / f"{stem}.cu"
        source_path.write_text(kernel_source, encoding="utf-8")
        record["source_file"] = str(source_path.relative_to(output_directory))

    detail_path = failure_directory / f"{stem}.json"
    record["detail_file"] = str(detail_path.relative_to(output_directory))
    serialized = json.dumps(record, indent=2, sort_keys=True, default=str)
    detail_path.write_text(serialized, encoding="utf-8")
    with (output_directory / "failures.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return record


def _print_failure(record: dict[str, Any], schedule_count: int) -> None:
    """Print an explicit failure block with links to its saved artifacts."""

    schedule_index = int(record["schedule_index"])
    count = str(schedule_count) if schedule_count else "?"
    print(f"\n=== Rejected [{schedule_index + 1}/{count}] ===", flush=True)
    print(f"phase: {record['phase']}", flush=True)
    print(f"error_type: {record['error_type']}", flush=True)
    print(f"error: {record['error']}", flush=True)
    print(f"detail_log: {record['detail_file']}", flush=True)
    if record["source_file"] is not None:
        print(f"generated_source: {record['source_file']}", flush=True)


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


def _write_top_results(
    output_directory: Path,
    heap: list[tuple[float, int, WSPBenchmarkResult, str, dict[str, Any]]],
    counters: dict[str, int],
    *,
    write_sources: bool = False,
) -> None:
    ranked = sorted(heap, key=lambda item: (-item[0], item[1]))
    payload = {
        **counters,
        "top_results": [
            {
                "rank": rank,
                **asdict(result),
                "schedule": schedule,
                "source_file": f"schedule_{result.schedule_index:05d}.cu",
            }
            for rank, (_, _, result, _, schedule) in enumerate(ranked, start=1)
        ],
    }
    (output_directory / "top20.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    if write_sources:
        for _, _, result, source, _ in ranked:
            (
                output_directory / f"schedule_{result.schedule_index:05d}.cu"
            ).write_text(source, encoding="utf-8")


def search_wsp_schedules(
    prim_func: tirx.PrimFunc,
    *,
    target: Target,
    out_idx: Sequence[int] | int,
    total_flops: float,
    reference_program: Callable[..., Any],
    input_tensors: list[Any] | None = None,
    output_directory: str | Path = "unionwsp_top20",
    top_k: int = 20,
    warmup: int = 25,
    rep: int = 100,
    original_threads: int | None = None,
    num_stages: int | None = None,
    num_groups: int | None = None,
    start_schedule_index: int = 0,
    max_schedules: int | None = None,
    validate_each_schedule: bool = True,
    execution_backend: str = "cython",
    pass_configs: dict[str, Any] | None = None,
) -> WSPSearchSummary:
    """Exhaustively compile schedules and save the fastest ``top_k``.

    Enumeration happens inside the first LayoutReducer callback, guaranteeing
    that the graph is extracted from the exact IR boundary consumed by WSP.
    Compilation or correctness failures reject only that candidate.
    """

    if not isinstance(prim_func, tirx.PrimFunc):
        raise TypeError("prim_func must be a TIRX PrimFunc")
    if total_flops <= 0:
        raise ValueError("total_flops must be positive")
    if top_k < 1 or warmup < 0 or rep < 1:
        raise ValueError("top_k and rep must be positive; warmup cannot be negative")
    if max_schedules is not None and max_schedules < 1:
        raise ValueError("max_schedules must be positive or None")
    if start_schedule_index < 0:
        raise ValueError("start_schedule_index cannot be negative")
    if max_schedules is not None and start_schedule_index >= max_schedules:
        raise ValueError("start_schedule_index must be smaller than max_schedules")

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    schedules: list[WSPSchedule] = []
    canonical_graph: DataflowGraph | None = None
    fingerprint: tuple[Any, ...] | None = None
    requested_index = start_schedule_index
    printed_schedule_indices: set[int] = set()
    candidate_directory = output_path / "candidates"
    candidate_directory.mkdir(parents=True, exist_ok=True)
    active_schedule_file: str | None = None
    active_source_file: str | None = None

    def planner(_symbol: str, graph: DataflowGraph, _target: Target) -> WSPSchedule:
        nonlocal canonical_graph, fingerprint, active_schedule_file
        current_fingerprint = _graph_fingerprint(graph)
        if canonical_graph is None:
            threads = original_threads or graph.kernel_threads
            if threads is None:
                raise ValueError(
                    "cannot infer the input CTA size after LayoutReducer; "
                    "pass original_threads explicitly"
                )
            initial_schedules = list(
                enumerate_wsp_schedules(
                    graph,
                    original_threads=threads,
                    num_stages=num_stages,
                    num_groups=num_groups,
                )
            )
            if not initial_schedules:
                raise RuntimeError("UnionWSP did not enumerate a feasible schedule")
            canonical_graph = graph
            fingerprint = current_fingerprint
            schedules.extend(initial_schedules)
            _write_json_atomic(
                output_path / "enumeration.json",
                {"enumerated_schedule_count": len(schedules)},
            )
        elif current_fingerprint != fingerprint:
            raise RuntimeError("LayoutReducer produced unstable operation IDs")
        schedule = schedules[requested_index]
        schedule_path = candidate_directory / f"schedule_{requested_index:05d}.json"
        _write_json_atomic(schedule_path, schedule_to_dict(graph, schedule))
        active_schedule_file = str(schedule_path.relative_to(output_path))
        _write_candidate_state(
            output_path,
            requested_index,
            "compilation",
            len(schedules),
            schedule_file=active_schedule_file,
        )
        if requested_index not in printed_schedule_indices:
            _print_schedule(graph, schedule, requested_index, len(schedules))
            printed_schedule_indices.add(requested_index)
        return schedule

    top_heap: list[
        tuple[float, int, WSPBenchmarkResult, str, dict[str, Any]]
    ] = []
    successful = 0
    failed = 0
    examined = 0
    failure_log = output_path / "failures.jsonl"
    failure_log.write_text("", encoding="utf-8")

    while True:
        if schedules and requested_index >= len(schedules):
            break
        if max_schedules is not None and requested_index >= max_schedules:
            break
        examined += 1
        # TileLang's persistent cache key is formed before the LayoutReducer
        # callback, so make the candidate identity part of the input PrimFunc.
        candidate_prim = prim_func.with_attr(
            "tl.program_schedule.request",
            # Bump this identity whenever schedule application/lowering changes.
            # TileLang forms its persistent cache key before the planner callback,
            # so the explicit schema version prevents reuse of stale kernels.
            f"unionwsp-search-v2={requested_index}",
        )
        kernel_source: str | None = None
        failure_phase = "compilation"
        active_schedule_file = None
        active_source_file = None
        _write_candidate_state(
            output_path,
            requested_index,
            failure_phase,
            len(schedules),
        )
        try:
            # After the first candidate, enumeration is already complete and
            # the selected schedule can be shown before entering compilation.
            # The first candidate is printed by ``planner`` after LayoutReducer.
            if (
                canonical_graph is not None
                and requested_index < len(schedules)
                and requested_index not in printed_schedule_indices
            ):
                # _print_schedule(
                #     canonical_graph,
                #     schedules[requested_index],
                #     requested_index,
                #     len(schedules),
                # )
                printed_schedule_indices.add(requested_index)
                schedule_path = (
                    candidate_directory / f"schedule_{requested_index:05d}.json"
                )
                _write_json_atomic(
                    schedule_path,
                    schedule_to_dict(canonical_graph, schedules[requested_index]),
                )
                active_schedule_file = str(schedule_path.relative_to(output_path))
                _write_candidate_state(
                    output_path,
                    requested_index,
                    failure_phase,
                    len(schedules),
                    schedule_file=active_schedule_file,
                )
            with use_schedule_planner(planner), target:
                compiled = tilelang.compile(
                    candidate_prim,
                    out_idx=out_idx,
                    target=target,
                    execution_backend=execution_backend,
                    pass_configs=pass_configs,
                )
            failure_phase = "source_extraction"
            kernel_source = compiled.get_kernel_source()
            source_path = candidate_directory / f"schedule_{requested_index:05d}.cu"
            source_path.write_text(kernel_source, encoding="utf-8")
            active_source_file = str(source_path.relative_to(output_path))
            # _print_generated_code(
            #     kernel_source,
            #     requested_index,
            #     len(schedules),
            # )
            failure_phase = "profiler_initialization"
            _write_candidate_state(
                output_path,
                requested_index,
                failure_phase,
                len(schedules),
                schedule_file=active_schedule_file,
                source_file=active_source_file,
            )
            profiler = compiled.get_profiler()
            if validate_each_schedule:
                failure_phase = "correctness_validation"
                _write_candidate_state(
                    output_path,
                    requested_index,
                    failure_phase,
                    len(schedules),
                    schedule_file=active_schedule_file,
                    source_file=active_source_file,
                )
                profiler.assert_allclose(
                    reference_program,
                    input_tensors=input_tensors,
                    rtol=0.01,
                    atol=0.01,
                )
            failure_phase = "benchmark"
            _write_candidate_state(
                output_path,
                requested_index,
                failure_phase,
                len(schedules),
                schedule_file=active_schedule_file,
                source_file=active_source_file,
            )
            latency_ms = float(
                profiler.do_bench(
                    warmup=warmup,
                    rep=rep,
                    input_tensors=input_tensors,
                )
            )
            tflops = total_flops / latency_ms * 1e-9
            failure_phase = "result_recording"
            schedule = schedules[requested_index]
            result = WSPBenchmarkResult(
                schedule_index=requested_index,
                latency_ms=latency_ms,
                tflops=tflops,
                num_groups=schedule.num_groups,
                num_stages_by_region=schedule.num_stages_by_region,
                effective_threads=schedule.warp_allocation.effective_threads,
            )
            item = (
                tflops,
                requested_index,
                result,
                kernel_source,
                schedule_to_dict(canonical_graph, schedule),
            )
            if len(top_heap) < top_k:
                heapq.heappush(top_heap, item)
            elif item[0] > top_heap[0][0]:
                heapq.heapreplace(top_heap, item)
            successful += 1
            _print_performance(
                latency_ms,
                tflops,
                requested_index,
                len(schedules),
            )
            _write_candidate_state(
                output_path,
                requested_index,
                "completed",
                len(schedules),
                schedule_file=active_schedule_file,
                source_file=active_source_file,
            )
        except Exception as error:  # Candidate rejection is intentional here.
            failed += 1
            failure: dict[str, Any] = {
                "schedule_index": requested_index,
                "phase": failure_phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "generated_source_available": kernel_source is not None,
                "enumerated_schedule_count": len(schedules),
                "search_configuration": {
                    "target": str(target),
                    "out_idx": out_idx,
                    "total_flops": total_flops,
                    "warmup": warmup,
                    "rep": rep,
                    "original_threads": original_threads,
                    "num_stages": num_stages,
                    "num_groups": num_groups,
                    "validate_each_schedule": validate_each_schedule,
                    "execution_backend": execution_backend,
                    "pass_configs": pass_configs,
                },
                "input_tensors": _describe_input_tensors(input_tensors),
            }
            # Compilation can fail after enumeration but before benchmarking.
            # Keep the rejected schedule in the log so layout/resource/sync
            # failures can be reproduced without relying on a mutable index.
            if canonical_graph is not None and requested_index < len(schedules):
                try:
                    failure["schedule"] = schedule_to_dict(
                        canonical_graph, schedules[requested_index]
                    )
                except Exception as serialization_error:
                    failure["schedule_serialization_error"] = (
                        f"{type(serialization_error).__name__}: {serialization_error}"
                    )
            record = _write_failure_artifacts(
                output_path,
                failure,
                kernel_source,
            )
            _print_failure(record, len(schedules))
            _write_candidate_state(
                output_path,
                requested_index,
                "failed",
                len(schedules),
                schedule_file=active_schedule_file,
                source_file=active_source_file,
            )
            if canonical_graph is None:
                raise

        counters = {
            "enumerated_schedules": len(schedules),
            "examined_schedules": examined,
            "successful_schedules": successful,
            "failed_schedules": failed,
        }
        _write_top_results(output_path, top_heap, counters)
        requested_index += 1

    ranked = tuple(
        item[2] for item in sorted(top_heap, key=lambda item: (-item[0], item[1]))
    )
    _write_top_results(
        output_path,
        top_heap,
        {
            "enumerated_schedules": len(schedules),
            "examined_schedules": examined,
            "successful_schedules": successful,
            "failed_schedules": failed,
        },
        write_sources=True,
    )
    return WSPSearchSummary(
        enumerated_schedules=len(schedules),
        examined_schedules=examined,
        successful_schedules=successful,
        failed_schedules=failed,
        top_results=ranked,
        output_directory=str(output_path.resolve()),
    )
