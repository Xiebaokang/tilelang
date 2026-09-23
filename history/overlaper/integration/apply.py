"""Encode an Overlaper schedule for the C++ lowering contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from tvm import IRModule, tirx

from ..analysis.physical import SharedMemoryPlan, WarpAllocation
from ..analysis.schedule import (
    ProgramOrders,
    Synchronization,
    SynchronizationKind,
    SynchronizationScope,
)
from ..analysis.schedule.multi_version import _validate_inputs
from ..parse import DataflowGraph, DependencyKind, RegionKind


class ScheduleLike(Protocol):
    """Fields required to lower one complete Overlaper schedule."""

    stages_by_region: Mapping[int, Mapping[int, int]]
    groups: Mapping[int, int]
    orders: ProgramOrders
    buffer_versions: Mapping[int, int]
    synchronizations: tuple[Synchronization, ...]
    warp_allocation: WarpAllocation
    shared_memory: SharedMemoryPlan


_COMMUNICATION_EXTERNAL = 0
_COMMUNICATION_PRIVATE = 1
_COMMUNICATION_SHARED = 2
_COMMUNICATION_TMEM = 3
_COMMUNICATION_MATERIALIZED_SHARED = 4

_COMPLETION_THREAD_ARRIVE = 0
_COMPLETION_ASYNC_TRANSACTION = 1


def _validate_schedule(graph: DataflowGraph, schedule: ScheduleLike) -> None:
    _validate_inputs(
        graph,
        schedule.stages_by_region,
        schedule.groups,
        schedule.orders,
    )
    if set(schedule.buffer_versions) != {
        buffer.buffer_id for buffer in graph.buffers
    } or any(version < 1 for version in schedule.buffer_versions.values()):
        raise ValueError("buffer versions must cover every buffer positively")

    group_ids = tuple(sorted(set(schedule.groups.values())))
    allocation = schedule.warp_allocation
    if tuple(item.group_id for item in allocation.groups) != group_ids:
        raise ValueError("warp allocation must cover every group")
    expected_first_warp = 0
    for item in allocation.groups:
        if item.first_warp != expected_first_warp or item.warp_count < 1:
            raise ValueError("warp intervals must be positive and contiguous")
        expected_first_warp = item.warp_stop
    assert graph.hardware is not None
    if allocation.effective_threads != (
        expected_first_warp * graph.hardware.device_resource.warp_size
    ):
        raise ValueError("effective_threads disagrees with warp allocation")
    if allocation.setmaxnreg_enabled:
        register_counts = allocation.register_counts
        register_actions = allocation.register_is_increase
        assert register_counts is not None
        if (
            register_actions is None
            or len(register_counts) != len(allocation.groups)
            or len(register_actions) != len(allocation.groups)
        ):
            raise ValueError("register allocation must cover every group")
        resource = graph.hardware.device_resource
        register_usage = sum(
            item.warp_count
            * resource.warp_size
            * register_counts[item.group_id]
            for item in allocation.groups
        )
        # Defend the lowering path against older or hand-written schedules
        # that saturate the pool and can block in setmaxnreg.inc.
        if register_usage >= resource.register_file_capacity:
            raise ValueError(
                "register allocation must leave register-file headroom"
            )
    if not schedule.shared_memory.fits:
        raise ValueError("schedule exceeds shared-memory capacity")

    for item in schedule.synchronizations:
        if (
            schedule.groups[item.producer_id] != item.producer_group
            or schedule.groups[item.consumer_id] != item.consumer_group
        ):
            raise ValueError("synchronization group is stale")
        graph.buffer_for_id(item.buffer_id)


def _buffer_communication(
    graph: DataflowGraph,
    schedule: ScheduleLike,
    buffer_id: int,
) -> int:
    buffer = graph.buffer_for_id(buffer_id)
    if buffer.scope in ("", "global"):
        return _COMMUNICATION_EXTERNAL
    if "tmem" in buffer.scope:
        return _COMMUNICATION_TMEM
    if buffer.scope.startswith("shared"):
        return _COMMUNICATION_SHARED
    users = {
        schedule.groups[node.node_id]
        for node in graph.nodes
        if buffer_id in (*node.reads, *node.writes)
    }
    if buffer.scope == "local.fragment" or len(users) <= 1:
        return _COMMUNICATION_PRIVATE
    if buffer.nbytes is None:
        raise ValueError(f"cross-group buffer {buffer.name} has dynamic size")
    return _COMMUNICATION_MATERIALIZED_SHARED


def _dependency_mask(
    graph: DataflowGraph,
    synchronization: Synchronization,
) -> int:
    if synchronization.kind == SynchronizationKind.BUFFER_REUSE:
        return 8
    kinds = {
        kind
        for edge in graph.edges
        if edge.producer_id == synchronization.producer_id
        and edge.consumer_id == synchronization.consumer_id
        and edge.buffer_id == synchronization.buffer_id
        and edge.iteration_distance == synchronization.iteration_distance
        for kind in edge.dependency_kinds
    }
    if not kinds:
        raise ValueError("forward synchronization has no matching graph edge")
    bits = {
        DependencyKind.RAW: 1,
        DependencyKind.WAR: 2,
        DependencyKind.WAW: 4,
    }
    return sum(bits[kind] for kind in kinds)


def _completion_mode(
    graph: DataflowGraph,
    synchronization: Synchronization,
) -> int:
    if synchronization.kind != SynchronizationKind.FORWARD_DEPENDENCY:
        return _COMPLETION_THREAD_ARRIVE
    producer = graph.node_for_id(synchronization.producer_id)
    if producer.instruction.name != "tma":
        return _COMPLETION_THREAD_ARRIVE
    output = graph.buffer_for_id(synchronization.buffer_id)
    reads_global = any(
        graph.buffer_for_id(buffer_id).scope in ("", "global")
        for buffer_id in producer.reads
    )
    if (
        reads_global
        and synchronization.buffer_id in producer.writes
        and output.scope.startswith("shared")
    ):
        return _COMPLETION_ASYNC_TRANSACTION
    return _COMPLETION_THREAD_ARRIVE


def _completion_modes(
    graph: DataflowGraph,
    synchronizations: tuple[Synchronization, ...],
) -> tuple[int, ...]:
    result = tuple(_completion_mode(graph, item) for item in synchronizations)
    transaction_producers = [
        item.producer_id
        for item, mode in zip(synchronizations, result)
        if mode == _COMPLETION_ASYNC_TRANSACTION
    ]
    return tuple(
        mode
        if transaction_producers.count(item.producer_id) == 1
        else _COMPLETION_THREAD_ARRIVE
        for item, mode in zip(synchronizations, result)
    )


def _buffer_offset_names(buffer) -> tuple[str, ...]:
    names = [buffer.name]
    data = getattr(buffer.buffer, "data", None)
    hint = getattr(data, "name_hint", None)
    if hint is not None:
        hint = str(hint)
        if hint not in names:
            names.append(hint)
    return tuple(names)


def _shared_memory_ir_attrs(
    graph: DataflowGraph, schedule: ScheduleLike
) -> tuple[dict[str, int], int, tuple[int, ...]]:
    """Name-keyed offsets that survive FinalizeProgramSchedule into merge."""

    by_id = {
        item.buffer_id: item
        for item in schedule.shared_memory.shared_allocations
    }
    offset_map: dict[str, int] = {}
    offsets: list[int] = []
    for buffer in graph.buffers:
        item = by_id.get(buffer.buffer_id)
        if item is None:
            offsets.append(-1)
            continue
        offsets.append(item.byte_offset)
        for name in (*_buffer_offset_names(buffer), item.name):
            offset_map[name] = item.byte_offset
    return (
        offset_map,
        schedule.shared_memory.merged_shared_bytes,
        tuple(offsets),
    )


def _register_domains(
    graph: DataflowGraph,
    allocation: WarpAllocation,
) -> dict[str, tuple[int, ...]]:
    empty = {
        "register_domain_groups": (),
        "register_domain_first_warps": (),
        "register_domain_warp_counts": (),
        "register_domain_register_counts": (),
        "register_domain_is_increase": (),
    }
    if not allocation.setmaxnreg_enabled:
        return empty
    assert allocation.register_counts is not None
    assert allocation.register_is_increase is not None
    assert graph.hardware is not None
    granule = graph.hardware.device_resource.specialized_group_warp_multiple
    if granule != 4:
        raise ValueError("program-schedule v3 requires four-warp domains")

    domains = [
        (group, first_warp)
        for group in allocation.groups
        for first_warp in range(group.first_warp, group.warp_stop, granule)
    ]
    if any(group.warp_count % granule for group in allocation.groups):
        raise ValueError("setmaxnreg groups must contain whole domains")
    return {
        "register_domain_groups": tuple(group.group_id for group, _ in domains),
        "register_domain_first_warps": tuple(first for _, first in domains),
        "register_domain_warp_counts": (granule,) * len(domains),
        "register_domain_register_counts": tuple(
            allocation.register_counts[group.group_id] for group, _ in domains
        ),
        "register_domain_is_increase": tuple(
            int(allocation.register_is_increase[group.group_id])
            for group, _ in domains
        ),
    }


def build_ir_plan(
    graph: DataflowGraph,
    schedule: ScheduleLike,
) -> dict[str, int | tuple[int, ...]]:
    """Convert a complete schedule to version-3 dense C++ arrays."""

    _validate_schedule(graph, schedule)
    synchronizations = tuple(schedule.synchronizations)
    communications = tuple(
        _buffer_communication(graph, schedule, buffer.buffer_id)
        for buffer in graph.buffers
    )
    scope_codes = {
        SynchronizationScope.ONCE: 0,
        SynchronizationScope.PER_ITERATION: 1,
        SynchronizationScope.REGION_BOUNDARY: 2,
    }
    kind_codes = {
        SynchronizationKind.FORWARD_DEPENDENCY: 0,
        SynchronizationKind.BUFFER_REUSE: 1,
    }
    allocation = schedule.warp_allocation
    plan: dict[str, int | tuple[int, ...]] = {
        "operation_groups": tuple(
            schedule.groups[node.node_id] for node in graph.nodes
        ),
        "operation_regions": tuple(node.region_id for node in graph.nodes),
        "operation_stages": tuple(
            -1
            if node.region_id not in schedule.stages_by_region
            else schedule.stages_by_region[node.region_id][node.node_id]
            for node in graph.nodes
        ),
        "operation_local_orders": tuple(
            schedule.orders[node.region_id][schedule.groups[node.node_id]][
                node.node_id
            ]
            for node in graph.nodes
        ),
        "region_num_stages": tuple(
            0
            if kind == RegionKind.SERIAL
            else max(schedule.stages_by_region[region_id].values()) + 1
            for region_id, kind in enumerate(graph.region_kinds)
        ),
        "group_first_warps": tuple(item.first_warp for item in allocation.groups),
        "group_warp_counts": tuple(item.warp_count for item in allocation.groups),
        "effective_threads": allocation.effective_threads,
        "buffer_versions": tuple(
            schedule.buffer_versions[buffer.buffer_id] for buffer in graph.buffers
        ),
        "buffer_communications": communications,
        "buffer_requires_new_allocation": tuple(
            int(
                schedule.buffer_versions[buffer.buffer_id] > 1
                or communications[buffer.buffer_id]
                == _COMMUNICATION_MATERIALIZED_SHARED
            )
            for buffer in graph.buffers
        ),
        "sync_producers": tuple(item.producer_id for item in synchronizations),
        "sync_consumers": tuple(item.consumer_id for item in synchronizations),
        "sync_producer_groups": tuple(
            item.producer_group for item in synchronizations
        ),
        "sync_consumer_groups": tuple(
            item.consumer_group for item in synchronizations
        ),
        "sync_producer_regions": tuple(
            graph.node_for_id(item.producer_id).region_id
            for item in synchronizations
        ),
        "sync_consumer_regions": tuple(
            graph.node_for_id(item.consumer_id).region_id
            for item in synchronizations
        ),
        "sync_iteration_distances": tuple(
            item.iteration_distance for item in synchronizations
        ),
        "sync_effective_stage_distances": tuple(
            -1
            if item.effective_stage_distance is None
            else item.effective_stage_distance
            for item in synchronizations
        ),
        "sync_slot_counts": tuple(item.slot_count for item in synchronizations),
        "sync_scopes": tuple(scope_codes[item.scope] for item in synchronizations),
        "sync_kinds": tuple(kind_codes[item.kind] for item in synchronizations),
        "sync_buffer_ids": tuple(item.buffer_id for item in synchronizations),
        "sync_dependency_masks": tuple(
            _dependency_mask(graph, item) for item in synchronizations
        ),
        "sync_completion_modes": _completion_modes(graph, synchronizations),
        "setmaxnreg_enabled": int(allocation.setmaxnreg_enabled),
    }
    _, merged_shared_bytes, shared_byte_offsets = _shared_memory_ir_attrs(
        graph, schedule
    )
    plan["shared_byte_offsets"] = shared_byte_offsets
    plan["merged_shared_bytes"] = merged_shared_bytes
    plan.update(_register_domains(graph, allocation))
    return plan


def _resolve_global_symbol(
    mod: IRModule,
    graph: DataflowGraph,
    requested: str | None,
) -> str:
    if requested is not None:
        global_var = mod.get_global_var(requested)
        if not isinstance(mod[global_var], tirx.PrimFunc):
            raise ValueError(f"{requested!r} is not a PrimFunc")
        if graph.prim_func is not None and not mod[global_var].same_as(
            graph.prim_func
        ):
            raise ValueError("graph does not describe the requested PrimFunc")
        return requested
    matches = [
        global_var.name_hint
        for global_var, function in mod.functions.items()
        if isinstance(function, tirx.PrimFunc)
        and (graph.prim_func is None or function.same_as(graph.prim_func))
    ]
    if len(matches) != 1:
        raise ValueError("cannot uniquely match graph to an IRModule PrimFunc")
    return matches[0]


def apply_schedule_to_ir(
    mod: IRModule,
    graph: DataflowGraph,
    schedule: ScheduleLike,
    *,
    global_symbol: str | None = None,
) -> IRModule:
    """Attach a complete schedule to the exact PrimFunc it describes."""

    if not isinstance(mod, IRModule):
        raise TypeError(f"mod must be IRModule, got {type(mod).__name__}")
    symbol = _resolve_global_symbol(mod, graph, global_symbol)
    plan = build_ir_plan(graph, schedule)
    offset_map, merged_shared_bytes, _ = _shared_memory_ir_attrs(graph, schedule)
    rewritten = mod.clone()
    global_var = rewritten.get_global_var(symbol)
    function = rewritten[global_var].with_attr("tl.program_schedule.version", 3)
    for name, value in plan.items():
        function = function.with_attr(
            f"tl.program_schedule.{name}",
            value if isinstance(value, int) else list(value),
        )
    if offset_map:
        function = function.with_attr("tl.smem_offset_map", offset_map)
        function = function.with_attr(
            "tl.smem_planned_arena_bytes", int(merged_shared_bytes)
        )
    rewritten.update_func(global_var, function)
    return rewritten
