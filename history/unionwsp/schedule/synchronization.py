"""Build logical cross-group synchronization for a realized schedule."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from ..parseIR.graph import (
    BufferRangeAccess,
    DataflowGraph,
    DependencyKind,
    RegionKind,
)
from .multi_version import (
    _ranges_may_overlap,
    _region_buffer_accesses,
    _validate_inputs,
)
from .order import ProgramOrders
from .stage import effective_stage_distance


class SynchronizationChannelKind(str, Enum):
    """The correctness relation implemented by one logical channel."""

    FORWARD_DEPENDENCY = "forward_dependency"
    BUFFER_REUSE = "buffer_reuse"


class SynchronizationScope(str, Enum):
    """How often a synchronization event occurs."""

    ONCE = "once"
    PER_ITERATION = "per_iteration"
    REGION_BOUNDARY = "region_boundary"


class SynchronizationCycleError(ValueError):
    """A candidate schedule deadlocks in its same-epoch wait graph."""


@dataclass(frozen=True)
class SynchronizationChannel:
    """One logical arrive/wait relation, before physical warp allocation."""

    kind: SynchronizationChannelKind
    scope: SynchronizationScope
    producer_id: int
    consumer_id: int
    producer_group: int
    consumer_group: int
    iteration_distance: int
    effective_stage_distance: int | None
    slot_count: int
    buffer_id: int | None = None
    dependency_kinds: frozenset[DependencyKind] = field(
        default_factory=frozenset
    )

    def __post_init__(self) -> None:
        if self.producer_id < 0 or self.consumer_id < 0:
            raise ValueError("synchronization endpoints must be non-negative")
        if self.producer_group == self.consumer_group:
            raise ValueError("a synchronization channel must cross groups")
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
                "non-pipeline synchronization has no effective stage distance"
            )
        if self.kind == SynchronizationChannelKind.FORWARD_DEPENDENCY:
            if not self.dependency_kinds:
                raise ValueError("a forward channel needs dependency kinds")
        elif self.buffer_id is None or self.dependency_kinds:
            raise ValueError(
                "a reuse channel needs only its buffer dependency"
            )


def _scope_for_nodes(
    graph: DataflowGraph,
    producer_id: int,
    consumer_id: int,
) -> SynchronizationScope:
    producer = graph.node_for_id(producer_id)
    consumer = graph.node_for_id(consumer_id)
    if producer.region_id == consumer.region_id:
        if graph.region_kinds[producer.region_id] == RegionKind.PIPELINE:
            return SynchronizationScope.PER_ITERATION
        return SynchronizationScope.ONCE
    return SynchronizationScope.REGION_BOUNDARY


def _pipeline_slot_count(
    stages: Mapping[int, int],
    buffer_id: int | None,
    versions: Mapping[int, int],
) -> int:
    """Match buffer-backed slots to storage versions.

    A reuse channel prevents a producer from overwriting a live version, so a
    buffer-backed channel needs exactly one synchronization slot per physical
    version.  Only a pure control channel has no storage ring to provide that
    index and therefore uses the pipeline stage count.
    """

    num_stages = max(stages.values(), default=0) + 1
    if buffer_id is None:
        return num_stages
    return versions[buffer_id]


def _build_forward_channels(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    versions: Mapping[int, int],
) -> list[SynchronizationChannel]:
    channels = []
    for edge in graph.edges:
        if groups[edge.producer_id] == groups[edge.consumer_id]:
            continue
        scope = _scope_for_nodes(
            graph, edge.producer_id, edge.consumer_id
        )
        distance = None
        slots = 1
        if scope == SynchronizationScope.PER_ITERATION:
            region_id = graph.node_for_id(edge.producer_id).region_id
            stages = stages_by_region[region_id]
            distance = effective_stage_distance(edge, stages)
            slots = _pipeline_slot_count(stages, edge.buffer_id, versions)
        channels.append(
            SynchronizationChannel(
                kind=SynchronizationChannelKind.FORWARD_DEPENDENCY,
                scope=scope,
                producer_id=edge.producer_id,
                consumer_id=edge.consumer_id,
                producer_group=groups[edge.producer_id],
                consumer_group=groups[edge.consumer_id],
                iteration_distance=edge.iteration_distance,
                effective_stage_distance=distance,
                slot_count=slots,
                buffer_id=edge.buffer_id,
                dependency_kinds=edge.dependency_kinds,
            )
        )
    return channels


def _build_reuse_channels(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
) -> list[SynchronizationChannel]:
    channels = []
    seen = set()
    for region_id, stages in stages_by_region.items():
        group_orders = orders[region_id]
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
                    if accessor_group == writer_group or not _ranges_may_overlap(
                        accessor_access, writer_access
                    ):
                        continue
                    conflicts_by_group.setdefault(accessor_group, []).append(
                        (accessor_id, accessor_access)
                    )

                for accessor_group, conflicts in conflicts_by_group.items():
                    accessor_id, _ = max(
                        conflicts,
                        key=lambda item: (
                            stages[item[0]],
                            group_orders[accessor_group][item[0]],
                        ),
                    )
                    key = (region_id, buffer.buffer_id, accessor_id, writer_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    version_count = versions[buffer.buffer_id]
                    distance = (
                        version_count
                        + stages[writer_id]
                        - stages[accessor_id]
                    )
                    channels.append(
                        SynchronizationChannel(
                            kind=SynchronizationChannelKind.BUFFER_REUSE,
                            scope=SynchronizationScope.PER_ITERATION,
                            producer_id=accessor_id,
                            consumer_id=writer_id,
                            producer_group=accessor_group,
                            consumer_group=writer_group,
                            iteration_distance=version_count,
                            effective_stage_distance=distance,
                            slot_count=_pipeline_slot_count(
                                stages, buffer.buffer_id, versions
                            ),
                            buffer_id=buffer.buffer_id,
                        )
                    )
    return channels


def _validate_versions(
    graph: DataflowGraph, versions: Mapping[int, int]
) -> None:
    buffer_ids = {buffer.buffer_id for buffer in graph.buffers}
    if set(versions) != buffer_ids:
        raise ValueError("versions must cover every buffer exactly once")
    if any(version < 1 for version in versions.values()):
        raise ValueError("every buffer version count must be positive")


def _validate_zero_epoch_graph(
    graph: DataflowGraph,
    orders: ProgramOrders,
    channels: tuple[SynchronizationChannel, ...],
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
            ordered = sorted(local_order, key=local_order.__getitem__)
            for producer_id, consumer_id in zip(ordered, ordered[1:]):
                add_edge(producer_id, consumer_id)
    for channel in channels:
        if (
            channel.scope != SynchronizationScope.PER_ITERATION
            or channel.effective_stage_distance == 0
        ):
            add_edge(channel.producer_id, channel.consumer_id)

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
            "same-epoch synchronization and group-local order form a cycle"
        )


def _build_channels(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
) -> tuple[SynchronizationChannel, ...]:
    channels = _build_forward_channels(
        graph, stages_by_region, groups, versions
    )
    channels.extend(
        _build_reuse_channels(
            graph, stages_by_region, groups, orders, versions
        )
    )
    return tuple(channels)


def build_synchronization_channels(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
) -> tuple[SynchronizationChannel, ...]:
    """Return complete logical synchronization for one realized schedule."""

    _validate_inputs(graph, stages_by_region, groups, orders)
    _validate_versions(graph, versions)
    channels = _build_channels(
        graph, stages_by_region, groups, orders, versions
    )
    if len(set(channels)) != len(channels):
        raise ValueError("synchronization plan contains duplicate channels")
    _validate_zero_epoch_graph(graph, orders, channels)
    return channels


def validate_synchronization_channels(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    channels: tuple[SynchronizationChannel, ...],
) -> None:
    """Reject stale, incomplete, duplicate, or deadlocking channels."""

    _validate_inputs(graph, stages_by_region, groups, orders)
    _validate_versions(graph, versions)
    expected = _build_channels(
        graph, stages_by_region, groups, orders, versions
    )
    if channels != expected:
        raise ValueError("synchronization channels are stale or incomplete")
    if len(set(channels)) != len(channels):
        raise ValueError("synchronization plan contains duplicate channels")
    _validate_zero_epoch_graph(graph, orders, channels)
