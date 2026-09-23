"""Target-independent synchronization plans for warp specialization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .buffer_planning import BufferReuseConstraint
from ..analysis.core import DataflowNode
from ..analysis.program import (
    ProgramDataflowAnalysis,
    ProgramRegionKind,
)
from ..scheduling.order import ProgramRegionSchedule
from ..scheduling.warp_specialization import (
    ProgramScheduleCandidate,
    RegionDependencyScope,
    build_cross_group_dependencies,
)


class SynchronizationChannelKind(str, Enum):
    """Correctness relation implemented by an abstract channel."""

    FORWARD_DEPENDENCY = "forward_dependency"
    BUFFER_REUSE = "buffer_reuse"


@dataclass(frozen=True)
class SynchronizationChannel:
    """One uncoalesced producer-to-consumer dependency channel."""

    producer_group: int
    consumer_group: int
    producer_operation_id: int
    consumer_operation_id: int
    iteration_distance: int
    effective_stage_distance: int | None
    dependency_kinds: frozenset[str]
    kind: SynchronizationChannelKind = SynchronizationChannelKind.FORWARD_DEPENDENCY
    buffer_id: int | None = None
    producer_region_id: int | None = None
    consumer_region_id: int | None = None
    dependency_scope: RegionDependencyScope | None = None


def _operation_ids(
    analysis: ProgramDataflowAnalysis,
) -> dict[DataflowNode, int]:
    return {
        operation.node: operation.operation_id
        for operation in analysis.operations
    }


def _program_schedules_by_region(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramScheduleCandidate,
) -> dict[int, ProgramRegionSchedule]:
    """Validate region coverage and return schedules indexed by region ID."""

    nodes = set(analysis.nodes)
    if set(candidate.partition.node_groups) != nodes:
        raise ValueError("candidate partition must cover the analyzed operations")
    expected_group_ids = set(range(candidate.partition.num_groups))
    schedules_by_region = {
        schedule.region_id: schedule for schedule in candidate.region_schedules
    }
    expected_region_ids = {region.region_id for region in analysis.regions}
    if len(schedules_by_region) != len(candidate.region_schedules):
        raise ValueError("candidate contains duplicate region schedules")
    if set(schedules_by_region) != expected_region_ids:
        raise ValueError("candidate must provide one schedule per program region")

    covered_nodes: set[DataflowNode] = set()
    for region in analysis.regions:
        schedule = schedules_by_region[region.region_id]
        region_nodes = set(analysis.nodes_for_region(region))
        expected_stage_nodes = (
            region_nodes if region.kind == ProgramRegionKind.PIPELINE else set()
        )
        if region.kind == ProgramRegionKind.SERIAL:
            if schedule.num_stages != 0:
                raise ValueError("serial region must use zero pipeline stages")
        elif schedule.num_stages < 1:
            raise ValueError("pipeline region must use at least one stage")
        elif (
            not region.auto_schedule
            and region.num_stages is not None
            and schedule.num_stages != region.num_stages
        ):
            raise ValueError(
                f"region {region.region_id} must use its fixed pipeline depth"
            )
        if set(schedule.stages) != expected_stage_nodes:
            raise ValueError(
                f"region {region.region_id} stages cover the wrong operations"
            )
        if schedule.stages and (
            min(schedule.stages.values()) != 0
            or max(schedule.stages.values()) != schedule.num_stages - 1
        ):
            raise ValueError(
                f"region {region.region_id} stage assignment does not span "
                "its selected pipeline depth"
            )
        if set(schedule.group_orders) != expected_group_ids:
            raise ValueError(
                f"region {region.region_id} must provide one order per logical group"
            )
        for group_id, local_order in schedule.group_orders.items():
            expected_nodes = {
                node
                for node in region_nodes
                if candidate.partition.node_groups[node] == group_id
            }
            if set(local_order) != expected_nodes:
                raise ValueError(
                    f"region {region.region_id}, group {group_id} order covers "
                    "the wrong operations"
                )
            if set(local_order.values()) != set(range(len(local_order))):
                raise ValueError(
                    f"region {region.region_id}, group {group_id} order must be "
                    "contiguous"
                )
            covered_nodes.update(local_order)
    if covered_nodes != nodes:
        raise ValueError("candidate region orders must cover every operation once")
    return schedules_by_region


def build_synchronization_channels(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramScheduleCandidate,
) -> tuple[SynchronizationChannel, ...]:
    """Build an uncoalesced arrive/wait plan for every cross-group edge.

    A distinct channel per dependency is intentional at this stage: merging
    channels can change phase and arrival-count semantics, so it belongs in a
    later target-aware lowering pass.
    """

    _program_schedules_by_region(analysis, candidate)
    expected_dependencies = build_cross_group_dependencies(
        analysis, candidate.partition, candidate.region_schedules
    )
    if candidate.cross_group_dependencies != expected_dependencies:
        raise ValueError("candidate cross-group dependencies are stale or incomplete")

    operation_ids = _operation_ids(analysis)
    channels: list[SynchronizationChannel] = []
    for dependency in expected_dependencies:
        producer_id = operation_ids[dependency.producer]
        consumer_id = operation_ids[dependency.consumer]
        channels.append(
            SynchronizationChannel(
                producer_group=dependency.producer_group,
                consumer_group=dependency.consumer_group,
                producer_operation_id=producer_id,
                consumer_operation_id=consumer_id,
                iteration_distance=dependency.iteration_distance,
                effective_stage_distance=dependency.effective_stage_distance,
                dependency_kinds=dependency.dependency_kinds,
                buffer_id=dependency.buffer_id,
                producer_region_id=dependency.producer_region,
                consumer_region_id=dependency.consumer_region,
                dependency_scope=dependency.scope,
            )
        )
    return tuple(channels)


def add_buffer_reuse_channels(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramScheduleCandidate,
    channels: tuple[SynchronizationChannel, ...],
    constraints: tuple[BufferReuseConstraint, ...],
) -> tuple[SynchronizationChannel, ...]:
    """Add consumer-to-producer backpressure before a buffer slot is reused."""

    _program_schedules_by_region(analysis, candidate)
    nodes_by_id = {
        operation.operation_id: operation.node
        for operation in analysis.operations
    }
    channels = list(channels)
    for constraint in constraints:
        if constraint.region_id is None:
            raise ValueError("program buffer reuse constraint needs a region ID")
        region = analysis.regions[constraint.region_id]
        if region.kind != ProgramRegionKind.PIPELINE:
            raise ValueError("buffer reuse synchronization must belong to a pipeline")
        accessor_node = nodes_by_id[constraint.accessor_operation_id]
        writer_node = nodes_by_id[constraint.writer_operation_id]
        if (
            analysis.operation_for(accessor_node).region_id != constraint.region_id
            or analysis.operation_for(writer_node).region_id != constraint.region_id
        ):
            raise ValueError("buffer reuse endpoints must belong to its region")
        channels.append(
            SynchronizationChannel(
                producer_group=constraint.accessor_group,
                consumer_group=constraint.writer_group,
                producer_operation_id=constraint.accessor_operation_id,
                consumer_operation_id=constraint.writer_operation_id,
                iteration_distance=constraint.iteration_distance,
                effective_stage_distance=constraint.effective_stage_distance,
                dependency_kinds=frozenset({"REUSE"}),
                kind=SynchronizationChannelKind.BUFFER_REUSE,
                buffer_id=constraint.buffer_id,
                producer_region_id=constraint.region_id,
                consumer_region_id=constraint.region_id,
                dependency_scope=RegionDependencyScope.PER_ITERATION,
            )
        )
    return tuple(channels)


def _validate_zero_distance_graph(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramScheduleCandidate,
    channels: tuple[SynchronizationChannel, ...],
) -> None:
    """Reject cycles among region-local orders and same-epoch synchronization."""

    schedules_by_region = _program_schedules_by_region(analysis, candidate)
    operation_ids = _operation_ids(analysis)
    successors = {operation_id: [] for operation_id in operation_ids.values()}
    incoming = {operation_id: 0 for operation_id in operation_ids.values()}

    def add_edge(producer: int, consumer: int) -> None:
        if consumer in successors[producer]:
            return
        successors[producer].append(consumer)
        incoming[consumer] += 1

    for schedule in schedules_by_region.values():
        for local_order in schedule.group_orders.values():
            ordered_nodes = sorted(local_order, key=local_order.__getitem__)
            for earlier, later in zip(ordered_nodes, ordered_nodes[1:]):
                add_edge(operation_ids[earlier], operation_ids[later])
    for channel in channels:
        if (
            channel.dependency_scope != RegionDependencyScope.PER_ITERATION
            or channel.effective_stage_distance == 0
        ):
            add_edge(channel.producer_operation_id, channel.consumer_operation_id)

    ready = [operation_id for operation_id, count in incoming.items() if count == 0]
    visited = 0
    while ready:
        operation_id = ready.pop()
        visited += 1
        for consumer in successors[operation_id]:
            incoming[consumer] -= 1
            if incoming[consumer] == 0:
                ready.append(consumer)
    if visited != len(incoming):
        raise ValueError("same-epoch synchronization and local order form a cycle")


def validate_synchronization_channels(
    channels: tuple[SynchronizationChannel, ...],
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramScheduleCandidate,
) -> None:
    """Validate channel consistency and same-tick deadlock freedom."""

    for channel in channels:
        if channel.producer_group == channel.consumer_group:
            raise ValueError("a synchronization channel must cross groups")
        if channel.iteration_distance < 0 or (
            channel.effective_stage_distance is not None
            and channel.effective_stage_distance < 0
        ):
            raise ValueError("synchronization distances cannot be negative")
        if channel.dependency_scope == RegionDependencyScope.PER_ITERATION:
            if channel.effective_stage_distance is None:
                raise ValueError(
                    "per-iteration synchronization requires a stage distance"
                )
            if channel.producer_region_id != channel.consumer_region_id:
                raise ValueError(
                    "per-iteration synchronization cannot cross program regions"
                )
        if channel.dependency_scope == RegionDependencyScope.REGION_BOUNDARY:
            if channel.producer_region_id == channel.consumer_region_id:
                raise ValueError(
                    "region-boundary synchronization must cross program regions"
                )
    _program_schedules_by_region(analysis, candidate)
    _validate_zero_distance_graph(analysis, candidate, channels)
