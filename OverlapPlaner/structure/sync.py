"""Build the synchronization implied by a structure, without an ISA."""

from __future__ import annotations

from collections.abc import Mapping

from OverlapPlaner.arch import ClassifiedGraph
from OverlapPlaner.facts import BufferRangeAccess, FactEdge, FactGraph, RegionKind
from OverlapPlaner.structure.model import (
    ProgramOrders,
    SynchronizationCycleError,
    SynchronizationKind,
    SynchronizationScope,
    SyncSkeleton,
    validate_stage_group_order,
)
from OverlapPlaner.structure.stage import effective_stage_distance
from OverlapPlaner.structure.version import ranges_may_overlap, region_buffer_accesses


def _scope(
    graph: FactGraph, producer_id: int, consumer_id: int
) -> SynchronizationScope:
    producer_region = graph.node_for_id(producer_id).region_id
    consumer_region = graph.node_for_id(consumer_id).region_id
    if producer_region != consumer_region:
        return SynchronizationScope.REGION_BOUNDARY
    if graph.region_kinds[producer_region] == RegionKind.PIPELINE:
        return SynchronizationScope.PER_ITERATION
    return SynchronizationScope.ONCE


def _is_intra_group_async_handoff(
    classified: ClassifiedGraph, edge: FactEdge
) -> bool:
    """Same-group async copy still needs a completion barrier."""

    graph = classified.graph
    producer = graph.node_for_id(edge.producer_id)
    output = graph.buffer_for_id(edge.buffer_id)
    return (
        classified.traits_for(edge.producer_id).async_completion
        and edge.buffer_id in producer.writes
        and output.scope.startswith("shared")
        and any(
            graph.buffer_for_id(buffer_id).scope in ("", "global")
            for buffer_id in producer.reads
        )
    )


def _forward_slot_count(
    graph: FactGraph,
    edge: FactEdge,
    groups: Mapping[int, int],
    effective_distance: int | None,
    versions: Mapping[int, int],
    scope: SynchronizationScope,
) -> int:
    """Return ring slots for one forward edge.

    Fragment buffers stay PRIVATE; ``buffer_versions`` is register ping-pong,
    not the shared handoff. Cross-group warp-specialized groups run together,
    so a one-stage delay needs two handoff slots: producer iteration ``k+1``
    overlaps consumer iteration ``k``.
    """

    if scope != SynchronizationScope.PER_ITERATION:
        return 1
    if graph.buffer_for_id(edge.buffer_id).scope != "local.fragment":
        return max(1, versions[edge.buffer_id])
    if effective_distance is None:
        raise ValueError("per-iteration fragment handoff needs a stage distance")
    if groups[edge.producer_id] == groups[edge.consumer_id]:
        return max(1, effective_distance)
    # Cross-group producer and consumer execute independently. Retain the
    # value for the effective pipeline distance plus the slot the producer may
    # already be filling. The distance includes a loop-carried dependency.
    return max(1, effective_distance + 1)


def _forward_synchronizations(
    classified: ClassifiedGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
) -> list[SyncSkeleton]:
    graph = classified.graph
    result = []
    # A same-group wait on the current-iteration value already completes the
    # asynchronous shared-memory write for every later use in that group's
    # program order. Dependence extraction may additionally report a
    # conservative loop-carried RAW edge for the same producer/buffer/consumer.
    # A second completion channel for that duplicate edge can deadlock when it
    # is waited in the following iteration.
    current_async_handoffs = {
        (edge.producer_id, edge.consumer_id, edge.buffer_id)
        for edge in graph.edges
        if edge.iteration_distance == 0
        and groups[edge.producer_id] == groups[edge.consumer_id]
        and _is_intra_group_async_handoff(classified, edge)
    }
    for edge in graph.edges:
        producer_group = groups[edge.producer_id]
        consumer_group = groups[edge.consumer_id]
        if producer_group == consumer_group:
            if not _is_intra_group_async_handoff(classified, edge):
                continue
            if (
                edge.iteration_distance > 0
                and (edge.producer_id, edge.consumer_id, edge.buffer_id)
                in current_async_handoffs
            ):
                continue

        scope = _scope(graph, edge.producer_id, edge.consumer_id)
        distance = None
        stages = None
        if scope == SynchronizationScope.PER_ITERATION:
            region_id = graph.node_for_id(edge.producer_id).region_id
            stages = stages_by_region[region_id]
            distance = effective_stage_distance(edge, stages)
        result.append(
            SyncSkeleton(
                kind=SynchronizationKind.FORWARD_DEPENDENCY,
                scope=scope,
                producer_id=edge.producer_id,
                consumer_id=edge.consumer_id,
                producer_group=producer_group,
                consumer_group=consumer_group,
                buffer_id=edge.buffer_id,
                iteration_distance=edge.iteration_distance,
                effective_stage_distance=distance,
                slot_count=_forward_slot_count(
                    graph, edge, groups, distance, versions, scope
                ),
            )
        )
    # A wait before the first read in one ordered group also protects later
    # reads of the same producer's tile. Keep separate channels when the
    # iteration/stage distance differs, since those waits refer to different
    # values in a pipeline ring.
    earliest: dict[tuple, SyncSkeleton] = {}
    for sync in result:
        consumer_region = graph.node_for_id(sync.consumer_id).region_id
        key = (
            sync.producer_id,
            sync.buffer_id,
            sync.consumer_group,
            consumer_region,
            sync.scope,
            sync.iteration_distance,
            sync.effective_stage_distance,
            sync.slot_count,
        )
        previous = earliest.get(key)
        if previous is None or (
            orders[consumer_region][sync.consumer_group][sync.consumer_id]
            < orders[consumer_region][sync.consumer_group][previous.consumer_id]
        ):
            earliest[key] = sync
    retained = set(earliest.values())
    return [sync for sync in result if sync in retained]


def _latest_conflicting_accessor(
    conflicts: list[tuple[int, BufferRangeAccess | None]],
    stages: Mapping[int, int],
    order: Mapping[int, int],
) -> int:
    return max(
        conflicts,
        key=lambda item: (stages[item[0]], order[item[0]]),
    )[0]


def _reuse_synchronizations(
    classified: ClassifiedGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
) -> list[SyncSkeleton]:
    graph = classified.graph
    result = []
    for region_id, stages in stages_by_region.items():
        for buffer in graph.buffers:
            if buffer.scope in ("", "global", "local.fragment"):
                continue
            accesses, writers = region_buffer_accesses(
                graph, region_id, buffer.buffer_id
            )
            for writer_id, writer_access in writers:
                writer_group = groups[writer_id]
                conflicts_by_group: dict[
                    int, list[tuple[int, BufferRangeAccess | None]]
                ] = {}
                for accessor_id, accessor_access in accesses:
                    accessor_group = groups[accessor_id]
                    # Program order is not sufficient when an elected thread
                    # launches an asynchronous shared-memory write.  It may
                    # re-arm and overwrite the tile while slower threads in
                    # the same group still consume the previous iteration.
                    # Keep the usual cross-group back-pressure, and add the
                    # same protocol for an intra-group async shared writer.
                    intra_group_async_reuse = (
                        accessor_group == writer_group
                        and classified.traits_for(writer_id).async_completion
                        and buffer.scope.startswith("shared")
                    )
                    if (
                        accessor_group == writer_group
                        and not intra_group_async_reuse
                    ):
                        continue
                    if not ranges_may_overlap(accessor_access, writer_access):
                        continue
                    conflicts_by_group.setdefault(accessor_group, []).append(
                        (accessor_id, accessor_access)
                    )

                for accessor_group, conflicts in conflicts_by_group.items():
                    accessor_id = _latest_conflicting_accessor(
                        conflicts,
                        stages,
                        orders[region_id][accessor_group],
                    )
                    version_count = versions[buffer.buffer_id]
                    result.append(
                        SyncSkeleton(
                            kind=SynchronizationKind.BUFFER_REUSE,
                            scope=SynchronizationScope.PER_ITERATION,
                            producer_id=accessor_id,
                            consumer_id=writer_id,
                            producer_group=accessor_group,
                            consumer_group=writer_group,
                            buffer_id=buffer.buffer_id,
                            iteration_distance=version_count,
                            effective_stage_distance=(
                                version_count
                                + stages[writer_id]
                                - stages[accessor_id]
                            ),
                            slot_count=version_count,
                        )
                    )
    # A downstream group cannot release the slot before the async write has
    # completed and the data has been consumed. Its back-pressure channel
    # therefore also prevents this writer from overwriting its own in-flight
    # slot; a separate writer-to-itself channel adds only barriers.
    cross_group_releases = {
        (sync.consumer_id, sync.buffer_id)
        for sync in result
        if sync.producer_group != sync.consumer_group
    }
    return [
        sync for sync in result
        if sync.producer_id != sync.consumer_id
        or (sync.consumer_id, sync.buffer_id) not in cross_group_releases
    ]


def _reject_same_epoch_cycle(
    graph: FactGraph,
    orders: ProgramOrders,
    synchronizations: tuple[SyncSkeleton, ...],
) -> None:
    successors = {node.node_id: set() for node in graph.nodes}
    incoming = {node.node_id: 0 for node in graph.nodes}

    def add_edge(producer_id: int, consumer_id: int) -> None:
        if producer_id == consumer_id or consumer_id in successors[producer_id]:
            return
        successors[producer_id].add(consumer_id)
        incoming[consumer_id] += 1

    for region_orders in orders.values():
        for local_order in region_orders.values():
            nodes = sorted(local_order, key=local_order.__getitem__)
            for producer_id, consumer_id in zip(nodes, nodes[1:]):
                add_edge(producer_id, consumer_id)
    for item in synchronizations:
        if (
            item.scope != SynchronizationScope.PER_ITERATION
            or item.effective_stage_distance == 0
        ):
            add_edge(item.producer_id, item.consumer_id)

    ready = [node_id for node_id, count in incoming.items() if count == 0]
    visited = 0
    while ready:
        node_id = ready.pop()
        visited += 1
        for consumer_id in successors[node_id]:
            incoming[consumer_id] -= 1
            if incoming[consumer_id] == 0:
                ready.append(consumer_id)
    if visited != len(graph.nodes):
        raise SynchronizationCycleError(
            "same-epoch synchronization and group order form a cycle"
        )


def build_synchronizations(
    classified: ClassifiedGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
) -> tuple[SyncSkeleton, ...]:
    """Return the deadlock-free cross-group and async-copy sync skeleton."""

    graph = classified.graph
    validate_stage_group_order(graph, stages_by_region, groups, orders)
    if set(versions) != {buffer.buffer_id for buffer in graph.buffers}:
        raise ValueError("versions must cover every buffer exactly once")
    if any(version < 1 for version in versions.values()):
        raise ValueError("every buffer version count must be positive")
    result = _forward_synchronizations(
        classified, stages_by_region, groups, orders, versions
    )
    result.extend(
        _reuse_synchronizations(
            classified, stages_by_region, groups, orders, versions
        )
    )
    synchronizations = tuple(dict.fromkeys(result))
    _reject_same_epoch_cycle(graph, orders, synchronizations)
    return synchronizations
