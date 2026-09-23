"""L3: bind a realized structure onto the typed OverlapPlan contract."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

from tvm import IRModule, tirx
from tvm.target import Target
from tvm.tirx import transform as tirx_transform

import tilelang

from OverlapPlaner.arch import HOPPER, HOPPER_CUDA_TARGET, Architecture, ClassifiedGraph
from OverlapPlaner.facts import DependencyKind, FactGraph, extract_fact_graph
from OverlapPlaner.ir import BufferPlan, GroupPlan, OperationPlacement, OverlapPlan, SyncEdge
from OverlapPlaner.physical.model import PhysicalPlan
from OverlapPlaner.structure import (
    SearchBudget,
    SynchronizationKind,
    SynchronizationScope,
    SyncSkeleton,
    enumerate_structures,
)
from OverlapPlaner.structure.model import validate_stage_group_order

_COMMUNICATION_EXTERNAL = 0
_COMMUNICATION_PRIVATE = 1
_COMMUNICATION_SHARED = 2
_COMMUNICATION_TMEM = 3
_COMMUNICATION_MATERIALIZED_SHARED = 4
_COMPLETION_THREAD_ARRIVE = 0
_COMPLETION_ASYNC_TRANSACTION = 1
_KIND_FORWARD = 0
_KIND_REUSE = 1
_SCOPE_CODES = {
    SynchronizationScope.ONCE: 0,
    SynchronizationScope.PER_ITERATION: 1,
    SynchronizationScope.REGION_BOUNDARY: 2,
}
_DEPENDENCY_BITS = {
    DependencyKind.RAW: 1,
    DependencyKind.WAR: 2,
    DependencyKind.WAW: 4,
}


def layout_reduced_module(
    prim_func: tirx.PrimFunc, target: Target | None = None
) -> tuple[IRModule, Target]:
    """Run the CUDA prologue through LayoutReducer, matching compile time."""

    target = target or HOPPER_CUDA_TARGET
    mod = IRModule({"main": prim_func})
    with target, tilelang.transform.PassContext(config={}):
        mod = tirx_transform.BindTarget(target)(mod)
        mod = tilelang.transform.MaterializeKernelLaunch()(mod)
        mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
        mod = tilelang.transform.LegalizeNegativeIndex()(mod)
        mod = tilelang.transform.InjectAssumes()(mod)
        mod = tilelang.transform.Simplify()(mod)
        mod = tilelang.transform.LayoutReducer()(mod)
    return mod, target


def layout_reduced_prim_func(
    prim_func: tirx.PrimFunc, target: Target | None = None
) -> tirx.PrimFunc:
    """Return the layout-reduced kernel PrimFunc."""

    mod, _ = layout_reduced_module(prim_func, target)
    for global_var, function in mod.functions.items():
        if global_var.name_hint == "main" and isinstance(function, tirx.PrimFunc):
            return function
    functions = [
        function
        for function in mod.functions.values()
        if isinstance(function, tirx.PrimFunc)
    ]
    if len(functions) != 1:
        raise ValueError("layout-reduced module must contain one PrimFunc")
    return functions[0]


def _buffer_communication(
    graph: FactGraph, groups: dict[int, int], buffer_id: int
) -> int:
    buffer = graph.buffer_for_id(buffer_id)
    if buffer.scope in ("", "global"):
        return _COMMUNICATION_EXTERNAL
    if "tmem" in buffer.scope:
        return _COMMUNICATION_TMEM
    if buffer.scope.startswith("shared"):
        return _COMMUNICATION_SHARED
    users = {
        groups[node.node_id]
        for node in graph.nodes
        if buffer_id in (*node.reads, *node.writes)
    }
    if buffer.scope == "local.fragment" or len(users) <= 1:
        return _COMMUNICATION_PRIVATE
    if buffer.nbytes is None:
        raise ValueError(f"cross-group buffer {buffer.name} has dynamic size")
    return _COMMUNICATION_MATERIALIZED_SHARED


def _dependency_mask(graph: FactGraph, sync: SyncSkeleton) -> int:
    if sync.kind == SynchronizationKind.BUFFER_REUSE:
        return 8
    kinds = {
        kind
        for edge in graph.edges
        if edge.producer_id == sync.producer_id
        and edge.consumer_id == sync.consumer_id
        and edge.buffer_id == sync.buffer_id
        and edge.iteration_distance == sync.iteration_distance
        for kind in edge.dependency_kinds
    }
    if not kinds:
        raise ValueError("forward synchronization has no matching graph edge")
    return sum(_DEPENDENCY_BITS[kind] for kind in kinds)


def _is_async_smem_load(
    classified: ClassifiedGraph, node_id: int, buffer_id: int
) -> bool:
    graph = classified.graph
    producer = graph.node_for_id(node_id)
    output = graph.buffer_for_id(buffer_id)
    return (
        classified.traits_for(node_id).async_completion
        and buffer_id in producer.writes
        and output.scope.startswith("shared")
        and any(
            graph.buffer_for_id(item).scope in ("", "global")
            for item in producer.reads
        )
    )


def _completion_modes(
    classified: ClassifiedGraph, sync_edges: tuple[SyncSkeleton, ...]
) -> tuple[int, ...]:
    modes = []
    for item in sync_edges:
        if (
            item.kind == SynchronizationKind.FORWARD_DEPENDENCY
            and _is_async_smem_load(classified, item.producer_id, item.buffer_id)
        ):
            modes.append(_COMPLETION_ASYNC_TRANSACTION)
        else:
            modes.append(_COMPLETION_THREAD_ARRIVE)
    # One asynchronous copy may feed several consumers.  Those consumers wait
    # on the same transaction event as long as their ring-buffer protocol is
    # identical.  A single TMA operation cannot target multiple, incompatible
    # barriers, so retain the conservative thread-arrive fallback in that case.
    signatures_by_producer: dict[
        int, set[tuple[int, SynchronizationScope, int, int]]
    ] = {}
    for item, mode in zip(sync_edges, modes):
        if mode != _COMPLETION_ASYNC_TRANSACTION:
            continue
        signatures_by_producer.setdefault(item.producer_id, set()).add(
            (
                item.buffer_id,
                item.scope,
                item.iteration_distance,
                item.slot_count,
            )
        )
    return tuple(
        mode
        if len(signatures_by_producer.get(item.producer_id, ())) == 1
        else _COMPLETION_THREAD_ARRIVE
        for item, mode in zip(sync_edges, modes)
    )


def to_overlap_plan(
    classified: ClassifiedGraph, physical: PhysicalPlan
) -> OverlapPlan:
    """Stamp a realized structure onto Stmt/Buffer identity in OverlapPlan."""

    graph = classified.graph
    structure = physical.structure
    allocation = physical.warp_allocation
    validate_stage_group_order(
        graph, structure.stages_by_region, structure.groups, structure.orders
    )
    if len(allocation.groups) != structure.num_groups:
        raise ValueError("warp allocation must cover every logical group")

    groups = []
    for item in allocation.groups:
        register_count = None
        register_increase = None
        if allocation.setmaxnreg_enabled:
            assert allocation.register_counts is not None
            assert allocation.register_is_increase is not None
            register_count = int(allocation.register_counts[item.group_id])
            register_increase = int(
                allocation.register_is_increase[item.group_id]
            )
        groups.append(
            GroupPlan(
                warp_count=int(item.warp_count),
                register_count=register_count,
                register_increase=register_increase,
            )
        )

    operations = []
    for node in graph.nodes:
        stage = None
        if node.region_id in structure.stages_by_region:
            stage = int(structure.stages_by_region[node.region_id][node.node_id])
        operations.append(
            OperationPlacement(
                operation_id=int(node.node_id),
                statement=node.statement,
                group_id=int(structure.groups[node.node_id]),
                stage=stage,
                order=int(
                    structure.orders[node.region_id][
                        structure.groups[node.node_id]
                    ][node.node_id]
                ),
            )
        )

    offset_by_id = {
        item.buffer_id: item.byte_offset
        for item in physical.shared_memory.shared_allocations
    }
    handoff_offset_by_channel = {
        item.channel_id: item.byte_offset
        for item in physical.shared_memory.handoff_allocations
    }
    buffers = []
    for buffer in graph.buffers:
        buffers.append(
            BufferPlan(
                buffer_id=int(buffer.buffer_id),
                buffer=buffer.buffer,
                version_count=int(structure.buffer_versions[buffer.buffer_id]),
                communication=_buffer_communication(
                    graph, structure.groups, buffer.buffer_id
                ),
                byte_offset=offset_by_id.get(buffer.buffer_id),
            )
        )

    modes = _completion_modes(classified, structure.sync_edges)
    sync_edges = []
    for channel, (item, mode) in enumerate(zip(structure.sync_edges, modes)):
        sync_edges.append(
            SyncEdge(
                producer_id=int(item.producer_id),
                consumer_id=int(item.consumer_id),
                buffer_id=int(item.buffer_id),
                kind=(
                    _KIND_REUSE
                    if item.kind == SynchronizationKind.BUFFER_REUSE
                    else _KIND_FORWARD
                ),
                scope=int(_SCOPE_CODES[item.scope]),
                iteration_distance=int(item.iteration_distance),
                slot_count=int(item.slot_count),
                dependency_kind=int(_dependency_mask(graph, item)),
                completion_mode=int(mode),
                byte_offset=handoff_offset_by_channel.get(channel),
            )
        )

    return OverlapPlan(
        groups=groups,
        operations=operations,
        buffers=buffers,
        sync_edges=sync_edges,
        # Mbarriers live in static shared memory after lowering.  The planned
        # dynamic arena contains only buffers and fragment handoffs.
        shared_arena_bytes=int(physical.shared_memory.shared_buffer_bytes),
    )


def enumerate_overlap_plans(
    prim_func: tirx.PrimFunc,
    *,
    arch: Architecture | None = None,
    budget: SearchBudget | None = None,
    target: Target | None = None,
    reduce_ir: bool = True,
) -> Iterator[OverlapPlan]:
    """Yield typed OverlapPlans for one kernel, after optional layout reduce.

    ``budget.max_structures`` counts realized plans. L1 may enumerate more
    structures so infeasible warp/smem candidates do not exhaust the cutoff.
    """

    arch = arch or HOPPER
    budget = budget or SearchBudget()
    if reduce_ir:
        prim_func = layout_reduced_prim_func(prim_func, target)
    classified = arch.classify(extract_fact_graph(prim_func))
    l1_budget = replace(
        budget,
        max_structures=max(budget.max_structures * 32, 256),
    )
    realized = 0
    for structure in enumerate_structures(classified, l1_budget):
        for physical in arch.realize(classified, structure):
            yield to_overlap_plan(classified, physical)
            realized += 1
            if realized >= budget.max_structures:
                return
