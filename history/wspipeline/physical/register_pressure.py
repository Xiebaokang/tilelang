"""Static register-pressure bounds for fully specified WSP candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math

from tvm.target import Target

from ..analysis.program import ProgramDataflowAnalysis
from .warp_allocation import (
    PhysicalWarpAllocation,
    RegisterAllocationPlan,
    target_execution_profile,
)
from ..scheduling.warp_specialization import ProgramScheduleCandidate


class RegisterPressureError(ValueError):
    """A candidate exceeds a provable architectural register bound."""


@dataclass(frozen=True)
class GroupRegisterPressure:
    """Per-thread register lower bound and performance estimate for one group."""

    group_id: int
    thread_count: int
    mandatory_registers_per_thread: int
    estimated_registers_per_thread: int
    register_limit_per_thread: int

    @property
    def estimated_limit_ratio(self) -> float:
        return self.estimated_registers_per_thread / self.register_limit_per_thread


@dataclass(frozen=True)
class RegisterPressurePlan:
    """Candidate-wide static register feasibility and scoring information."""

    groups: tuple[GroupRegisterPressure, ...]
    register_file_size: int
    mandatory_registers_per_cta: int
    estimated_registers_per_cta: int


def _register_file_size(target: Target) -> int:
    for key in ("registers_per_sm", "max_registers_per_sm"):
        value = target.attrs.get(key, None)
        if value is not None:
            return int(value)
    # NVIDIA Hopper and Blackwell expose 64K 32-bit registers per SM. Keep
    # the same architectural budget used by the setmaxnreg planner.
    return 65536


def _per_thread_limit(
    group_id: int,
    register_allocation: RegisterAllocationPlan,
) -> int:
    limits = {
        domain.register_count
        for domain in register_allocation.domains
        if domain.group_id == group_id
    }
    if not limits:
        return 255
    if len(limits) != 1:
        raise RegisterPressureError(
            f"group {group_id} has inconsistent setmaxnreg domain limits"
        )
    return next(iter(limits))


def _buffer_registers_per_thread(
    scope: str,
    nbytes: int | None,
    thread_count: int,
) -> int | None:
    if not scope.startswith("local"):
        return 0
    if nbytes is None:
        return None
    register_words = math.ceil(nbytes / 4)
    if scope == "local.fragment":
        return math.ceil(register_words / thread_count)
    # Ordinary local allocations are private arrays replicated by each thread.
    return register_words


def build_register_pressure_plan(
    analysis: ProgramDataflowAnalysis,
    logical: ProgramScheduleCandidate,
    warp_allocation: PhysicalWarpAllocation,
    register_allocation: RegisterAllocationPlan,
    target: Target,
) -> RegisterPressurePlan:
    """Estimate pressure and retain only conservative facts as hard bounds."""

    profile = target_execution_profile(
        target, enable_setmaxnreg=warp_allocation.setmaxnreg_enabled
    )
    accesses = analysis.operation_accesses
    groups = []
    for allocation in warp_allocation.groups:
        group_id = allocation.group_id
        thread_count = allocation.warp_count * profile.warp_size
        operation_ids = tuple(
            operation.operation_id
            for operation in analysis.operations
            if logical.partition.node_groups[operation.node] == group_id
        )
        used_buffer_ids = sorted(
            {
                buffer_id
                for operation_id in operation_ids
                for buffer_id in (
                    accesses[operation_id].read_buffer_ids
                    + accesses[operation_id].write_buffer_ids
                )
                if analysis.buffer_for_id(buffer_id).scope.startswith("local")
            }
        )
        footprints = {}
        for buffer_id in used_buffer_ids:
            descriptor = analysis.buffer_for_id(buffer_id)
            footprint = _buffer_registers_per_thread(
                descriptor.scope, descriptor.nbytes, thread_count
            )
            if footprint is None:
                continue
            else:
                footprints[buffer_id] = footprint

        mandatory_count = 0
        for operation_id in operation_ids:
            operation_buffers = tuple(
                sorted(
                    {
                        buffer_id
                        for buffer_id in (
                            accesses[operation_id].read_buffer_ids
                            + accesses[operation_id].write_buffer_ids
                        )
                        if buffer_id in footprints
                    }
                )
            )
            count = sum(footprints[buffer_id] for buffer_id in operation_buffers)
            if count > mandatory_count:
                mandatory_count = count

        live_ranges = {}
        for buffer_id in footprints:
            buffer_operations = [
                operation_id
                for operation_id in operation_ids
                if buffer_id in accesses[operation_id].read_buffer_ids
                or buffer_id in accesses[operation_id].write_buffer_ids
            ]
            live_ranges[buffer_id] = (
                min(buffer_operations),
                max(buffer_operations),
            )
        estimated_count = 0
        for operation_id in operation_ids:
            live_buffers = tuple(
                buffer_id
                for buffer_id, (start, stop) in live_ranges.items()
                if start <= operation_id <= stop
            )
            count = sum(footprints[buffer_id] for buffer_id in live_buffers)
            if count > estimated_count:
                estimated_count = count

        scalar_work = max(
            (
                analysis.operations[operation_id].node.profile.alu_ops
                + analysis.operations[operation_id].node.profile.sfu_ops
                + analysis.operations[operation_id].node.profile.cast_ops
                for operation_id in operation_ids
            ),
            default=0,
        )
        temporary_estimate = max(8, min(32, math.ceil(scalar_work / thread_count)))
        estimated_count = max(mandatory_count, estimated_count) + temporary_estimate
        limit = _per_thread_limit(group_id, register_allocation)
        groups.append(
            GroupRegisterPressure(
                group_id,
                thread_count,
                mandatory_count,
                estimated_count,
                limit,
            )
        )

    mandatory_per_cta = sum(
        group.mandatory_registers_per_thread * group.thread_count for group in groups
    )
    estimated_per_cta = sum(
        group.estimated_registers_per_thread * group.thread_count for group in groups
    )
    plan = RegisterPressurePlan(
        tuple(groups),
        _register_file_size(target),
        mandatory_per_cta,
        estimated_per_cta,
    )
    return plan


def validate_register_pressure(plan: RegisterPressurePlan) -> None:
    """Reject only violations established by the conservative lower bound."""

    if tuple(group.group_id for group in plan.groups) != tuple(range(len(plan.groups))):
        raise ValueError("register-pressure groups must use dense IDs")
    if plan.register_file_size <= 0:
        raise ValueError("register-file size must be positive")
    for group in plan.groups:
        if group.thread_count <= 0 or group.register_limit_per_thread <= 0:
            raise ValueError("register-pressure thread counts and limits must be positive")
        if group.mandatory_registers_per_thread > group.estimated_registers_per_thread:
            raise ValueError("estimated pressure cannot be below its mandatory bound")
        if group.mandatory_registers_per_thread > group.register_limit_per_thread:
            raise RegisterPressureError(
                f"group {group.group_id} requires at least "
                f"{group.mandatory_registers_per_thread} registers per thread, "
                f"exceeding its limit {group.register_limit_per_thread}"
            )
    expected_mandatory = sum(
        group.mandatory_registers_per_thread * group.thread_count
        for group in plan.groups
    )
    expected_estimated = sum(
        group.estimated_registers_per_thread * group.thread_count
        for group in plan.groups
    )
    if plan.mandatory_registers_per_cta != expected_mandatory:
        raise ValueError("mandatory CTA register accounting is inconsistent")
    if plan.estimated_registers_per_cta != expected_estimated:
        raise ValueError("estimated CTA register accounting is inconsistent")
    if plan.mandatory_registers_per_cta > plan.register_file_size:
        raise RegisterPressureError(
            f"candidate requires at least {plan.mandatory_registers_per_cta} "
            f"registers per CTA, exceeding the register file size "
            f"{plan.register_file_size}"
        )
