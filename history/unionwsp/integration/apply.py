"""Encode a UnionWSP schedule for the existing C++ lowering contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from tvm import IRModule, tirx

from ..hardware.spec import InstructionKind
from ..parseIR.graph import DataflowGraph, DependencyKind, RegionKind
from ..schedule.synchronization import (
    SynchronizationChannel,
    SynchronizationChannelKind,
    SynchronizationScope,
)

if TYPE_CHECKING:
    from .. import WSPSchedule


_COMMUNICATION_EXTERNAL = 0
_COMMUNICATION_PRIVATE = 1
_COMMUNICATION_SHARED = 2
_COMMUNICATION_TMEM = 3
_COMMUNICATION_MATERIALIZED_SHARED = 4

_COMPLETION_THREAD_ARRIVE = 0
_COMPLETION_ASYNC_TRANSACTION = 1


def _sync_completion_mode(
    graph: DataflowGraph, channel: SynchronizationChannel
) -> int:
    """Select how one non-pipeline producer completes its mbarrier event."""

    if channel.kind != SynchronizationChannelKind.FORWARD_DEPENDENCY:
        return _COMPLETION_THREAD_ARRIVE
    producer = graph.node_for_id(channel.producer_id)
    if (
        producer.instruction_kind != InstructionKind.TMA
        or graph.region_kinds[producer.region_id] != RegionKind.SERIAL
        or channel.scope == SynchronizationScope.PER_ITERATION
        or channel.buffer_id is None
    ):
        return _COMPLETION_THREAD_ARRIVE
    output = graph.buffer_for_id(channel.buffer_id)
    reads_global = any(
        graph.buffer_for_id(buffer_id).scope in ("", "global")
        for buffer_id in producer.reads
    )
    writes_output = channel.buffer_id in producer.writes
    if reads_global and writes_output and output.scope.startswith("shared"):
        return _COMPLETION_ASYNC_TRANSACTION
    return _COMPLETION_THREAD_ARRIVE


def _sync_completion_modes(
    graph: DataflowGraph,
    channels: tuple[SynchronizationChannel, ...],
) -> tuple[int, ...]:
    modes = tuple(_sync_completion_mode(graph, channel) for channel in channels)
    transaction_producers = [
        channel.producer_id
        for channel, mode in zip(channels, modes)
        if mode == _COMPLETION_ASYNC_TRANSACTION
    ]
    return tuple(
        mode
        if transaction_producers.count(channel.producer_id) == 1
        else _COMPLETION_THREAD_ARRIVE
        for channel, mode in zip(channels, modes)
    )


def _validate_schedule(graph: DataflowGraph, schedule: WSPSchedule) -> None:
    node_ids = set(range(len(graph.nodes)))
    buffer_ids = set(range(len(graph.buffers)))
    pipeline_regions = {
        region_id
        for region_id, kind in enumerate(graph.region_kinds)
        if kind == RegionKind.PIPELINE
    }
    if set(schedule.groups) != node_ids:
        raise ValueError("schedule groups must cover every operation")
    group_ids = set(schedule.groups.values())
    if group_ids != set(range(len(group_ids))):
        raise ValueError("schedule group IDs must be dense")
    if set(schedule.stages_by_region) != pipeline_regions:
        raise ValueError("schedule stages must cover every pipeline region")
    for region_id in pipeline_regions:
        expected = {node.node_id for node in graph.nodes_for_region(region_id)}
        stages = schedule.stages_by_region[region_id]
        if set(stages) != expected or set(stages.values()) != set(
            range(max(stages.values(), default=0) + 1)
        ):
            raise ValueError("pipeline stages must densely cover region nodes")
    if set(schedule.orders) != set(range(len(graph.region_kinds))):
        raise ValueError("schedule orders must cover every region")
    for region_id, group_orders in schedule.orders.items():
        if set(group_orders) != group_ids:
            raise ValueError("every region order must cover every group")
        for group_id, local_order in group_orders.items():
            expected = {
                node.node_id
                for node in graph.nodes_for_region(region_id)
                if schedule.groups[node.node_id] == group_id
            }
            if set(local_order) != expected or set(local_order.values()) != set(
                range(len(expected))
            ):
                raise ValueError("group-local order is not a dense permutation")
    if set(schedule.buffer_versions) != buffer_ids:
        raise ValueError("buffer versions must cover every buffer")
    if any(version < 1 for version in schedule.buffer_versions.values()):
        raise ValueError("buffer version counts must be positive")
    allocations = schedule.warp_allocation.groups
    if tuple(item.group_id for item in allocations) != tuple(sorted(group_ids)):
        raise ValueError("warp allocation must cover every group")
    if not schedule.shared_memory.fits:
        raise ValueError("schedule exceeds shared-memory capacity")


def _buffer_communication(
    graph: DataflowGraph,
    schedule: WSPSchedule,
    buffer_id: int,
) -> int:
    buffer = graph.buffer_for_id(buffer_id)
    scope = buffer.scope
    users = {
        schedule.groups[node.node_id]
        for node in graph.nodes
        if buffer_id in node.reads or buffer_id in node.writes
    }
    if scope in ("", "global"):
        return _COMMUNICATION_EXTERNAL
    if "tmem" in scope:
        return _COMMUNICATION_TMEM
    if scope.startswith("shared"):
        return _COMMUNICATION_SHARED
    # The C++ pass recognizes RAW synchronization on a private fragment and
    # creates an edge-local shared handoff buffer and export/import copies.
    if scope == "local.fragment" or len(users) <= 1:
        return _COMMUNICATION_PRIVATE
    if buffer.nbytes is None:
        raise ValueError(
            f"cross-group private buffer {buffer.name} has dynamic size"
        )
    return _COMMUNICATION_MATERIALIZED_SHARED


def _register_domains(schedule: WSPSchedule) -> dict[str, tuple[int, ...]]:
    allocation = schedule.warp_allocation
    if not allocation.setmaxnreg_enabled:
        return {
            "register_domain_groups": (),
            "register_domain_first_warps": (),
            "register_domain_warp_counts": (),
            "register_domain_register_counts": (),
            "register_domain_is_increase": (),
        }
    assert allocation.register_counts is not None
    assert allocation.register_is_increase is not None
    domain_groups = []
    first_warps = []
    warp_counts = []
    register_counts = []
    is_increase = []
    for group in allocation.groups:
        if group.warp_count % 4:
            raise ValueError(
                "the current C++ setmaxnreg contract requires four-warp domains"
            )
        for first_warp in range(
            group.first_warp, group.warp_stop, 4
        ):
            domain_groups.append(group.group_id)
            first_warps.append(first_warp)
            warp_counts.append(4)
            register_counts.append(allocation.register_counts[group.group_id])
            is_increase.append(
                int(allocation.register_is_increase[group.group_id])
            )
    return {
        "register_domain_groups": tuple(domain_groups),
        "register_domain_first_warps": tuple(first_warps),
        "register_domain_warp_counts": tuple(warp_counts),
        "register_domain_register_counts": tuple(register_counts),
        "register_domain_is_increase": tuple(is_increase),
    }


def build_ir_plan(
    graph: DataflowGraph,
    schedule: WSPSchedule,
) -> dict[str, int | tuple[int, ...]]:
    """Convert one ID-based UnionWSP schedule to dense C++ pass arrays."""

    _validate_schedule(graph, schedule)
    stages = []
    local_orders = []
    for node in graph.nodes:
        region_stages = schedule.stages_by_region.get(node.region_id)
        stages.append(-1 if region_stages is None else region_stages[node.node_id])
        local_orders.append(
            schedule.orders[node.region_id][schedule.groups[node.node_id]][
                node.node_id
            ]
        )

    region_num_stages = tuple(
        0
        if kind == RegionKind.SERIAL
        else max(schedule.stages_by_region[region_id].values(), default=0) + 1
        for region_id, kind in enumerate(graph.region_kinds)
    )
    communications = tuple(
        _buffer_communication(graph, schedule, buffer.buffer_id)
        for buffer in graph.buffers
    )
    dependency_bits = {
        DependencyKind.RAW: 1,
        DependencyKind.WAR: 2,
        DependencyKind.WAW: 4,
    }
    scope_codes = {
        SynchronizationScope.ONCE: 0,
        SynchronizationScope.PER_ITERATION: 1,
        SynchronizationScope.REGION_BOUNDARY: 2,
    }
    kind_codes = {
        SynchronizationChannelKind.FORWARD_DEPENDENCY: 0,
        SynchronizationChannelKind.BUFFER_REUSE: 1,
    }
    allocation = schedule.warp_allocation
    plan: dict[str, int | tuple[int, ...]] = {
        "operation_groups": tuple(schedule.groups[node.node_id] for node in graph.nodes),
        "operation_regions": tuple(node.region_id for node in graph.nodes),
        "operation_stages": tuple(stages),
        "operation_local_orders": tuple(local_orders),
        "region_num_stages": region_num_stages,
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
        "sync_producers": tuple(item.producer_id for item in schedule.synchronization),
        "sync_consumers": tuple(item.consumer_id for item in schedule.synchronization),
        "sync_producer_groups": tuple(
            item.producer_group for item in schedule.synchronization
        ),
        "sync_consumer_groups": tuple(
            item.consumer_group for item in schedule.synchronization
        ),
        "sync_producer_regions": tuple(
            graph.node_for_id(item.producer_id).region_id
            for item in schedule.synchronization
        ),
        "sync_consumer_regions": tuple(
            graph.node_for_id(item.consumer_id).region_id
            for item in schedule.synchronization
        ),
        "sync_iteration_distances": tuple(
            item.iteration_distance for item in schedule.synchronization
        ),
        "sync_effective_stage_distances": tuple(
            -1
            if item.effective_stage_distance is None
            else item.effective_stage_distance
            for item in schedule.synchronization
        ),
        "sync_slot_counts": tuple(
            item.slot_count for item in schedule.synchronization
        ),
        "sync_scopes": tuple(
            scope_codes[item.scope] for item in schedule.synchronization
        ),
        "sync_kinds": tuple(
            kind_codes[item.kind] for item in schedule.synchronization
        ),
        "sync_buffer_ids": tuple(
            -1 if item.buffer_id is None else item.buffer_id
            for item in schedule.synchronization
        ),
        "sync_dependency_masks": tuple(
            sum(dependency_bits[kind] for kind in item.dependency_kinds)
            if item.kind == SynchronizationChannelKind.FORWARD_DEPENDENCY
            else 8
            for item in schedule.synchronization
        ),
        "sync_completion_modes": _sync_completion_modes(
            graph, schedule.synchronization
        ),
        "setmaxnreg_enabled": int(allocation.setmaxnreg_enabled),
    }
    plan.update(_register_domains(schedule))
    return plan


def _resolve_global_symbol(
    mod: IRModule,
    graph: DataflowGraph,
    requested_symbol: str | None,
) -> str:
    if requested_symbol is not None:
        global_var = mod.get_global_var(requested_symbol)
        if not isinstance(mod[global_var], tirx.PrimFunc):
            raise ValueError(f"{requested_symbol!r} is not a PrimFunc")
        if graph.prim_func is not None and not mod[global_var].same_as(graph.prim_func):
            raise ValueError("graph was not extracted from the requested PrimFunc")
        return requested_symbol
    if graph.prim_func is None:
        functions = [
            var.name_hint
            for var, func in mod.functions.items()
            if isinstance(func, tirx.PrimFunc)
        ]
    else:
        functions = [
            var.name_hint
            for var, func in mod.functions.items()
            if isinstance(func, tirx.PrimFunc) and func.same_as(graph.prim_func)
        ]
    if len(functions) != 1:
        raise ValueError("cannot uniquely match the graph to an IRModule PrimFunc")
    return functions[0]


def _attach_plan(
    function: tirx.PrimFunc,
    plan: Mapping[str, int | tuple[int, ...]],
) -> tirx.PrimFunc:
    rewritten = function.with_attr("tl.program_schedule.version", 3)
    for name, value in plan.items():
        rewritten = rewritten.with_attr(
            f"tl.program_schedule.{name}",
            value if isinstance(value, int) else list(value),
        )
    return rewritten


def apply_schedule_to_ir(
    mod: IRModule,
    graph: DataflowGraph,
    schedule: WSPSchedule,
    *,
    global_symbol: str | None = None,
) -> IRModule:
    """Attach ``schedule`` to the exact LayoutReducer output it describes."""

    if not isinstance(mod, IRModule):
        raise TypeError(f"mod must be IRModule, got {type(mod).__name__}")
    symbol = _resolve_global_symbol(mod, graph, global_symbol)
    plan = build_ir_plan(graph, schedule)
    rewritten = mod.clone()
    global_var = rewritten.get_global_var(symbol)
    rewritten.update_func(global_var, _attach_plan(rewritten[global_var], plan))
    return rewritten
