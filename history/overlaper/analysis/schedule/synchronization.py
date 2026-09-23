"""Build the synchronization needed by communication across groups."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from ...parse.graph import (
    BufferRangeAccess,
    DataflowEdge,
    DataflowGraph,
    RegionKind,
)
from .multi_version import (
    _ranges_may_overlap,
    _region_buffer_accesses,
    _validate_inputs,
)
from .order import ProgramOrders
from .stage import effective_stage_distance


class SynchronizationKind(str, Enum):
    """Why one group must notify another group."""

    FORWARD_DEPENDENCY = "forward_dependency"
    BUFFER_REUSE = "buffer_reuse"


class SynchronizationScope(str, Enum):
    """How often one synchronization relation is executed."""

    ONCE = "once"
    PER_ITERATION = "per_iteration"
    REGION_BOUNDARY = "region_boundary"


class SynchronizationCycleError(ValueError):
    """The same-epoch waits of a candidate schedule form a cycle."""


@dataclass(frozen=True, slots=True)
class Synchronization:
    """One logical arrive/wait relation for a cross-group or TMA handoff."""

    kind: SynchronizationKind
    scope: SynchronizationScope
    producer_id: int
    consumer_id: int
    producer_group: int
    consumer_group: int
    buffer_id: int
    iteration_distance: int
    effective_stage_distance: int | None
    slot_count: int

    def __post_init__(self) -> None:
        if min(self.producer_id, self.consumer_id, self.buffer_id) < 0:
            raise ValueError("synchronization IDs must be non-negative")
        if (
            self.producer_group == self.consumer_group
            and self.kind != SynchronizationKind.FORWARD_DEPENDENCY
        ):
            raise ValueError(
                "only a TMA forward dependency may synchronize within a group"
            )
        if self.iteration_distance < 0:
            raise ValueError("iteration_distance must be non-negative")
        if self.slot_count < 1:
            raise ValueError("slot_count must be positive")
        if self.scope == SynchronizationScope.PER_ITERATION:
            if (
                self.effective_stage_distance is None
                or self.effective_stage_distance < 0
            ):
                raise ValueError(
                    "per-iteration synchronization needs a non-negative "
                    "effective stage distance"
                )
        elif self.effective_stage_distance is not None:
            raise ValueError(
                "non-pipeline synchronization has no stage distance"
            )


def _scope(
    graph: DataflowGraph,
    producer_id: int,
    consumer_id: int,
) -> SynchronizationScope:
    producer_region = graph.node_for_id(producer_id).region_id
    consumer_region = graph.node_for_id(consumer_id).region_id
    if producer_region != consumer_region:
        return SynchronizationScope.REGION_BOUNDARY
    if graph.region_kinds[producer_region] == RegionKind.PIPELINE:
        return SynchronizationScope.PER_ITERATION
    return SynchronizationScope.ONCE


def _is_intra_group_tma_handoff(
    graph: DataflowGraph,
    edge: DataflowEdge,
) -> bool:
    """Whether one same-group edge needs a TMA completion barrier.

    Ordinary same-group dependencies are ordered by the group's local order
    and ThreadSync.  A global-to-shared TMA copy is different: its producer
    thread returns before the asynchronous transfer becomes visible, so its
    consumer needs a per-copy mbarrier even within the same group.
    """

    producer = graph.node_for_id(edge.producer_id)
    output = graph.buffer_for_id(edge.buffer_id)
    return (
        producer.instruction.name == "tma"
        and edge.buffer_id in producer.writes
        and output.scope.startswith("shared")
        and any(
            graph.buffer_for_id(buffer_id).scope in ("", "global")
            for buffer_id in producer.reads
        )
    )


def _forward_synchronizations(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    versions: Mapping[int, int],
) -> list[Synchronization]:
    result = []
    for edge in graph.edges:
        producer_group = groups[edge.producer_id]
        consumer_group = groups[edge.consumer_id]
        if (
            producer_group == consumer_group
            and not _is_intra_group_tma_handoff(graph, edge)
        ):
            continue

        scope = _scope(graph, edge.producer_id, edge.consumer_id)
        distance = None
        if scope == SynchronizationScope.PER_ITERATION:
            region_id = graph.node_for_id(edge.producer_id).region_id
            distance = effective_stage_distance(
                edge, stages_by_region[region_id]
            )
        result.append(
            Synchronization(
                kind=SynchronizationKind.FORWARD_DEPENDENCY,
                scope=scope,
                producer_id=edge.producer_id,
                consumer_id=edge.consumer_id,
                producer_group=producer_group,
                consumer_group=consumer_group,
                buffer_id=edge.buffer_id,
                iteration_distance=edge.iteration_distance,
                effective_stage_distance=distance,
                slot_count=(
                    versions[edge.buffer_id]
                    if scope == SynchronizationScope.PER_ITERATION
                    else 1
                ),
            )
        )
    return result


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
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
) -> list[Synchronization]:
    result = []
    for region_id, stages in stages_by_region.items():
        for buffer in graph.buffers:
            if buffer.scope in ("", "global", "local.fragment"):
                continue
            accesses, writers = _region_buffer_accesses(
                graph, region_id, buffer.buffer_id
            )
            for writer_id, writer_access in writers:
                writer_group = groups[writer_id]
                conflicts_by_group: dict[
                    int, list[tuple[int, BufferRangeAccess | None]]
                ] = {}
                for accessor_id, accessor_access in accesses:
                    accessor_group = groups[accessor_id]
                    if accessor_group == writer_group:
                        continue
                    if not _ranges_may_overlap(accessor_access, writer_access):
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
                        Synchronization(
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
    return result


def _validate_versions(
    graph: DataflowGraph,
    versions: Mapping[int, int],
) -> None:
    if set(versions) != {buffer.buffer_id for buffer in graph.buffers}:
        raise ValueError("versions must cover every buffer exactly once")
    if any(version < 1 for version in versions.values()):
        raise ValueError("every buffer version count must be positive")


def _reject_same_epoch_cycle(
    graph: DataflowGraph,
    orders: ProgramOrders,
    synchronizations: tuple[Synchronization, ...],
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
    for synchronization in synchronizations:
        if (
            synchronization.scope != SynchronizationScope.PER_ITERATION
            or synchronization.effective_stage_distance == 0
        ):
            add_edge(
                synchronization.producer_id,
                synchronization.consumer_id,
            )

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
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
) -> tuple[Synchronization, ...]:
    """Return the complete, deadlock-free cross-group synchronization plan."""

    _validate_inputs(graph, stages_by_region, groups, orders)
    _validate_versions(graph, versions)
    result = _forward_synchronizations(
        graph, stages_by_region, groups, versions
    )
    result.extend(
        _reuse_synchronizations(
            graph, stages_by_region, groups, orders, versions
        )
    )
    synchronizations = tuple(dict.fromkeys(result))
    _reject_same_epoch_cycle(graph, orders, synchronizations)
    return synchronizations
