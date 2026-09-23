"""Enumerate and build target-valid warp-specialized schedules."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from tvm.target import Target

from ..physical.buffer_planning import (
    BufferPlanningError,
    BufferRealizationPlan,
    BufferVersionPolicy,
    RegionOverlapCache,
    enumerate_buffer_plans,
)
from ..analysis.core import HardwareUnit
from ..analysis.program import ProgramDataflowAnalysis
from ..physical.register_pressure import (
    RegisterPressureError,
    RegisterPressurePlan,
    build_register_pressure_plan,
)
from ..physical.synchronization import (
    SynchronizationChannel,
    add_buffer_reuse_channels,
    build_synchronization_channels,
)
from ..physical.warp_allocation import (
    PhysicalWarpAllocation,
    RegisterAllocationPlan,
    build_register_plan,
    enumerate_warp_allocations,
    target_execution_profile,
)
from ..scheduling.warp_specialization import (
    ProgramScheduleCandidate,
    enumerate_logical_schedules,
)


@dataclass(frozen=True)
class ProgramRealizedScheduleCandidate:
    """A logical schedule with every target-dependent decision fixed."""

    logical: ProgramScheduleCandidate
    warp_allocation: PhysicalWarpAllocation
    register_allocation: RegisterAllocationPlan
    register_pressure: RegisterPressurePlan
    buffers: BufferRealizationPlan
    channels: tuple[SynchronizationChannel, ...]


def _infer_group_counts(
    analysis: ProgramDataflowAnalysis,
    target: Target,
) -> tuple[int, ...]:
    roles = {
        "memory"
        if node.unit in {HardwareUnit.LOAD_STORE, HardwareUnit.TMA, HardwareUnit.TMEM}
        else "mma"
        if node.unit == HardwareUnit.MMA
        else "scalar"
        for node in analysis.nodes
    }
    profile = target_execution_profile(target)
    group_warps = max(1, profile.logical_group_warp_multiple)
    target_limit = profile.max_threads_per_block // profile.warp_size // group_warps
    maximum = min(len(analysis.nodes), max(1, len(roles)), max(1, target_limit))
    return tuple(range(maximum, 0, -1))


def build_realized_schedule(
    analysis: ProgramDataflowAnalysis,
    logical: ProgramScheduleCandidate,
    allocation: PhysicalWarpAllocation,
    buffers: BufferRealizationPlan,
    target: Target,
) -> ProgramRealizedScheduleCandidate:
    """Build and validate one complete target-dependent schedule."""

    profile = target_execution_profile(
        target, enable_setmaxnreg=allocation.setmaxnreg_enabled
    )
    channels = add_buffer_reuse_channels(
        analysis,
        logical,
        build_synchronization_channels(analysis, logical),
        buffers.reuse_constraints,
    )
    registers = build_register_plan(
        allocation, logical, profile.warpgroup_warps
    )
    pressure = build_register_pressure_plan(
        analysis, logical, allocation, registers, target
    )
    candidate = ProgramRealizedScheduleCandidate(
        logical, allocation, registers, pressure, buffers, channels
    )

    from ..physical.validation import validate_realized_schedule

    validate_realized_schedule(analysis, candidate, target)
    return candidate


def _enumerate_group_schedules(
    analysis: ProgramDataflowAnalysis,
    target: Target,
    group_count: int,
    limit: int | None,
    overlap_cache: RegionOverlapCache,
) -> Iterator[ProgramRealizedScheduleCandidate]:
    examined = 0
    for logical in enumerate_logical_schedules(analysis, group_count, target=target):
        try:
            allocations = tuple(
                enumerate_warp_allocations(analysis, logical, target)
            )
            for buffers in enumerate_buffer_plans(
                analysis,
                logical,
                target,
                version_policy=BufferVersionPolicy.ENUMERATE,
                overlap_cache=overlap_cache,
            ):
                for allocation in allocations:
                    if limit is not None and examined >= limit:
                        return
                    examined += 1
                    try:
                        yield build_realized_schedule(
                            analysis, logical, allocation, buffers, target
                        )
                    except (BufferPlanningError, RegisterPressureError, ValueError):
                        continue
        except (BufferPlanningError, ValueError):
            continue


def enumerate_realized_schedules(
    analysis: ProgramDataflowAnalysis,
    target: Target,
    examined_limit: int | None = None,
) -> Iterator[ProgramRealizedScheduleCandidate]:
    """Enumerate valid schedules fairly across inferred logical-group counts."""

    if examined_limit is not None and examined_limit <= 0:
        raise ValueError("examined_limit must be positive or None")
    group_counts = _infer_group_counts(analysis, target)
    if examined_limit is None:
        budgets = (None,) * len(group_counts)
    else:
        base, remainder = divmod(examined_limit, len(group_counts))
        budgets = tuple(
            base + int(index < remainder) for index in range(len(group_counts))
        )

    overlap_cache: RegionOverlapCache = {}
    active = [
        _enumerate_group_schedules(
            analysis,
            target,
            group_count,
            budget,
            overlap_cache,
        )
        for group_count, budget in zip(group_counts, budgets)
        if budget != 0
    ]
    while active:
        remaining = []
        for generator in active:
            try:
                candidate = next(generator)
            except StopIteration:
                continue
            remaining.append(generator)
            yield candidate
        active = remaining
