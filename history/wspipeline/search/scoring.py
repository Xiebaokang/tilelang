"""Ranking heuristics for logical and realized WSP schedules."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tvm.target import Target

from ..analysis.core import ExecutionKind, HardwareUnit
from ..analysis.program import ProgramDataflowAnalysis, ProgramRegionKind
from ..physical.synchronization import SynchronizationChannelKind
from ..scheduling.warp_specialization import LogicalWarpPartition

if TYPE_CHECKING:
    from .realization import ProgramRealizedScheduleCandidate


def _operation_work(node) -> float:
    profile = node.profile
    return max(
        1.0,
        profile.memory_bytes / 128.0
        + profile.mma_ops * 4.0
        + profile.alu_ops / 32.0
        + profile.sfu_ops / 8.0
        + profile.cast_ops / 64.0,
    )


def _independent_units(left: HardwareUnit, right: HardwareUnit) -> bool:
    if left == right:
        return False
    scalar_units = {HardwareUnit.ALU, HardwareUnit.SFU}
    return not (left in scalar_units and right in scalar_units)


def score_stage_dependency(producer, consumer, producer_stage: int, consumer_stage: int):
    """Return lexicographic costs used to order stage assignments."""

    is_split = producer_stage != consumer_stage
    return (
        int(producer.unit == consumer.unit and is_split),
        int(producer.unit != consumer.unit and not is_split),
        consumer_stage - producer_stage,
    )


def score_order_choice(node, original_index: dict, previous_node):
    """Rank one ready node without removing legal local orders."""

    hardware_switch = int(previous_node is not None and previous_node.unit != node.unit)
    return hardware_switch, original_index[node]


def score_stage_partition(
    analysis: ProgramDataflowAnalysis,
    stage_assignments: tuple[dict, ...],
    partition: LogicalWarpPartition,
    target: Target | None = None,
) -> float:
    """Rank a logical partition. Lower is better; legality is checked elsewhere."""

    del target
    stages_by_region = {
        region.region_id: stage_assignments[region.region_id]
        for region in analysis.regions
    }
    overlap_gain = 0.0
    missed_overlap = 0.0
    synchronization = 0.0
    communication = 0.0
    same_unit_split = 0.0
    for edge in analysis.edges:
        producer_group = partition.node_groups[edge.producer]
        consumer_group = partition.node_groups[edge.consumer]
        crosses_group = producer_group != consumer_group
        producer_region = analysis.operation_for(edge.producer).region_id
        consumer_region = analysis.operation_for(edge.consumer).region_id
        distance = None
        pipeline_depth = 1
        if (
            producer_region == consumer_region
            and analysis.regions[producer_region].kind == ProgramRegionKind.PIPELINE
        ):
            stages = stages_by_region[producer_region]
            pipeline_depth = max(stages.values(), default=0) + 1
            distance = (
                edge.iteration_distance
                + stages[edge.consumer]
                - stages[edge.producer]
            )
        work = min(_operation_work(edge.producer), _operation_work(edge.consumer))
        has_overlap = (
            distance is not None
            and distance > 0
            and pipeline_depth > 1
        )
        independent = _independent_units(edge.producer.unit, edge.consumer.unit)
        if independent and has_overlap:
            if crosses_group:
                overlap_gain += work
            else:
                missed_overlap += work
        if not crosses_group:
            continue
        synchronization += 2.0 if distance == 0 else 0.5
        if edge.producer.unit == edge.consumer.unit:
            same_unit_split += work * 0.25
        if edge.buffer_id is not None:
            descriptor = analysis.buffer_for_id(edge.buffer_id)
            if descriptor.scope == "local.fragment":
                communication += (descriptor.nbytes or 4096) / 1024.0
            elif not (
                descriptor.scope.startswith("shared")
                or descriptor.scope in ("", "global")
            ):
                communication += (descriptor.nbytes or 1024) / 2048.0

    workloads = [0.0 for _ in range(partition.num_groups)]
    group_units = [set() for _ in range(partition.num_groups)]
    group_has_async_memory = [False for _ in range(partition.num_groups)]
    for node in analysis.nodes:
        group_id = partition.node_groups[node]
        workloads[group_id] += _operation_work(node)
        group_units[group_id].add(node.unit)
        group_has_async_memory[group_id] |= node.execution_kind == ExecutionKind.ASYNC_MEMORY
    scalar_units = {HardwareUnit.ALU, HardwareUnit.SFU}
    mixed_engine = 0.0
    for units, has_async_memory in zip(group_units, group_has_async_memory):
        if has_async_memory and HardwareUnit.MMA in units:
            mixed_engine += 512.0
        if has_async_memory and units & scalar_units:
            mixed_engine += 256.0
    average = sum(workloads) / len(workloads)
    imbalance = max(workloads, default=0.0) - average
    return (
        missed_overlap
        + synchronization
        + communication
        + same_unit_split
        + mixed_engine
        + imbalance * 0.1
        - overlap_gain
    )


def score_realized_schedule(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramRealizedScheduleCandidate,
    target: Target,
) -> float:
    """Return the final analytical cost for a valid realized schedule."""

    del target
    logical_cost = score_stage_partition(
        analysis,
        tuple(schedule.stages for schedule in candidate.logical.region_schedules),
        candidate.logical.partition,
    )
    synchronization_cost = sum(
        2.0
        if channel.effective_stage_distance == 0
        else 1.0
        if channel.kind == SynchronizationChannelKind.FORWARD_DEPENDENCY
        else 1.5
        for channel in candidate.channels
    )
    communication_cost = sum(
        buffer.additional_shared_bytes for buffer in candidate.buffers.buffers
    ) / 4096.0
    shared_limit = candidate.buffers.target_shared_memory_limit
    shared_ratio = (
        candidate.buffers.total_shared_bytes / shared_limit if shared_limit else 0.0
    )
    register_cost = sum(
        25.0 * max(0.0, group.estimated_limit_ratio - 0.75) ** 2
        for group in candidate.register_pressure.groups
    )
    thread_cost = max(
        0.0,
        candidate.warp_allocation.effective_threads
        / max(candidate.warp_allocation.original_threads, 1)
        - 1.0,
    )
    pipeline_depth_cost = 0.25 * sum(
        max(0, schedule.num_stages - 2)
        for schedule in candidate.logical.region_schedules
    )
    total = (
        logical_cost
        + synchronization_cost
        + communication_cost
        + 20.0 * shared_ratio * shared_ratio
        + register_cost
        + thread_cost
        + pipeline_depth_cost
    )
    if not math.isfinite(total):
        raise ValueError("realized schedule score must be finite")
    return total
