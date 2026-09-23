"""Unified hard-correctness validation for realized WSP schedules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tvm.target import Target

from .buffer_planning import BufferCommunicationKind, validate_buffer_plan
from ..analysis.program import ProgramDataflowAnalysis, ProgramRegionKind
from .register_pressure import (
    build_register_pressure_plan,
    validate_register_pressure,
)
from ..scheduling.stage import effective_stage_distance
from .synchronization import (
    SynchronizationChannelKind,
    validate_synchronization_channels,
)
from .warp_allocation import (
    build_register_plan,
    target_execution_profile,
    validate_physical_warp_allocation,
)
from ..scheduling.warp_specialization import build_cross_group_dependencies

if TYPE_CHECKING:
    from ..search.realization import ProgramRealizedScheduleCandidate


def _validate_temporal_dependencies(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramRealizedScheduleCandidate,
) -> None:
    logical = candidate.logical
    schedules = {item.region_id: item for item in logical.region_schedules}
    if set(logical.partition.node_groups) != set(analysis.nodes):
        raise ValueError("candidate partition does not cover the analyzed operations")
    if set(schedules) != {region.region_id for region in analysis.regions}:
        raise ValueError("candidate does not provide exactly one schedule per region")

    expected_cross_group = build_cross_group_dependencies(
        analysis, logical.partition, logical.region_schedules
    )
    if expected_cross_group != logical.cross_group_dependencies:
        raise ValueError("candidate cross-group dependencies are stale or incomplete")

    forward_channels = {
        (
            channel.producer_operation_id,
            channel.consumer_operation_id,
            channel.producer_group,
            channel.consumer_group,
            channel.buffer_id,
            channel.producer_region_id,
            channel.consumer_region_id,
        ): channel
        for channel in candidate.channels
        if channel.kind == SynchronizationChannelKind.FORWARD_DEPENDENCY
    }
    for edge in analysis.edges:
        producer = analysis.operation_for(edge.producer)
        consumer = analysis.operation_for(edge.consumer)
        producer_group = logical.partition.node_groups[edge.producer]
        consumer_group = logical.partition.node_groups[edge.consumer]
        effective_distance = None
        if producer.region_id == consumer.region_id:
            region = analysis.regions[producer.region_id]
            schedule = schedules[producer.region_id]
            if region.kind == ProgramRegionKind.PIPELINE:
                effective_distance = effective_stage_distance(edge, schedule.stages)
                if effective_distance < 0:
                    raise ValueError(
                        f"dependency {producer.operation_id}->{consumer.operation_id} "
                        "has a negative effective stage distance"
                    )
            if producer_group == consumer_group and edge.producer != edge.consumer:
                producer_order = schedule.group_orders[producer_group][edge.producer]
                consumer_order = schedule.group_orders[consumer_group][edge.consumer]
                if (effective_distance is None or effective_distance == 0) and (
                    producer_order >= consumer_order
                ):
                    raise ValueError(
                        f"dependency {producer.operation_id}->{consumer.operation_id} "
                        "is reversed by its group-local order"
                    )
        elif producer.region_id > consumer.region_id:
            raise ValueError("a dependency cannot execute backwards across program regions")

        if producer_group != consumer_group:
            key = (
                producer.operation_id,
                consumer.operation_id,
                producer_group,
                consumer_group,
                edge.buffer_id,
                producer.region_id,
                consumer.region_id,
            )
            channel = forward_channels.pop(key, None)
            if channel is None:
                raise ValueError(
                    f"cross-group dependency {producer.operation_id}->"
                    f"{consumer.operation_id} has no forward synchronization channel"
                )
            if channel.dependency_kinds != edge.dependency_kinds:
                raise ValueError("synchronization dependency kinds do not match the edge")
            if channel.effective_stage_distance != effective_distance:
                raise ValueError("synchronization stage distance does not match the edge")
    if forward_channels:
        raise ValueError("synchronization plan contains stale forward channels")


def _validate_buffers_and_layouts(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramRealizedScheduleCandidate,
) -> None:
    validate_buffer_plan(candidate.buffers)
    buffers = {item.buffer_id: item for item in candidate.buffers.buffers}
    group_warps = {
        item.group_id: item.warp_count for item in candidate.warp_allocation.groups
    }
    reuse_channels = {
        (
            channel.buffer_id,
            channel.producer_operation_id,
            channel.consumer_operation_id,
            channel.producer_group,
            channel.consumer_group,
        )
        for channel in candidate.channels
        if channel.kind == SynchronizationChannelKind.BUFFER_REUSE
    }
    for constraint in candidate.buffers.reuse_constraints:
        key = (
            constraint.buffer_id,
            constraint.accessor_operation_id,
            constraint.writer_operation_id,
            constraint.accessor_group,
            constraint.writer_group,
        )
        if key not in reuse_channels:
            raise ValueError("buffer reuse constraint has no backpressure channel")
    if len(reuse_channels) != len(candidate.buffers.reuse_constraints):
        raise ValueError("synchronization plan contains stale buffer-reuse channels")

    for dependency in candidate.logical.cross_group_dependencies:
        if dependency.buffer_id is None or "RAW" not in dependency.dependency_kinds:
            continue
        descriptor = analysis.buffer_for_id(dependency.buffer_id)
        realized = buffers[dependency.buffer_id]
        producer_warps = group_warps[dependency.producer_group]
        consumer_warps = group_warps[dependency.consumer_group]
        if descriptor.scope == "local.fragment":
            if producer_warps != consumer_warps:
                raise ValueError(
                    f"fragment handoff {descriptor.name} needs layout resharding "
                    "between unequal-width groups"
                )
        elif realized.communication in {
            BufferCommunicationKind.EXTERNAL,
            BufferCommunicationKind.SHARED,
            BufferCommunicationKind.MATERIALIZED_SHARED,
        }:
            pass
        elif producer_warps != consumer_warps:
            raise ValueError(
                f"buffer {descriptor.name} has no layout boundary between "
                "unequal-width groups"
            )


def validate_realized_schedule(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramRealizedScheduleCandidate,
    target: Target,
) -> None:
    """Recheck every hard constraint required by lowering."""

    profile = target_execution_profile(
        target, enable_setmaxnreg=candidate.warp_allocation.setmaxnreg_enabled
    )
    validate_physical_warp_allocation(
        candidate.logical, candidate.warp_allocation, profile
    )
    expected_registers = build_register_plan(
        candidate.warp_allocation, candidate.logical, profile.warpgroup_warps
    )
    if expected_registers != candidate.register_allocation:
        raise ValueError("register allocation plan is stale or inconsistent")
    validate_register_pressure(candidate.register_pressure)
    expected_pressure = build_register_pressure_plan(
        analysis,
        candidate.logical,
        candidate.warp_allocation,
        candidate.register_allocation,
        target,
    )
    if expected_pressure != candidate.register_pressure:
        raise ValueError("register-pressure plan is stale or inconsistent")
    validate_synchronization_channels(
        candidate.channels, analysis, candidate.logical
    )
    _validate_temporal_dependencies(analysis, candidate)
    _validate_buffers_and_layouts(analysis, candidate)
