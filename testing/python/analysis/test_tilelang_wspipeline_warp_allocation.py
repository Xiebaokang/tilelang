import tilelang  # noqa: F401
import pytest
from tvm.target import Target

from history.wspipeline.analysis.core import DataflowNode, HardwareUnit, InstructionKind
from history.wspipeline.physical.warp_allocation import (
    CUDAArchitectureFamily,
    WarpAllocationMode,
    build_register_plan,
    enumerate_physical_warp_allocations,
    target_execution_profile,
)
from history.wspipeline.scheduling.order import ProgramRegionSchedule
from history.wspipeline.scheduling.warp_specialization import (
    LogicalWarpPartition,
    ProgramScheduleCandidate,
)


def _memory_compute_candidate(
    compute_instruction: InstructionKind,
) -> ProgramScheduleCandidate:
    producer = DataflowNode(
        "producer",
        HardwareUnit.TMA,
        instruction_kind=InstructionKind.TMA,
    )
    consumer = DataflowNode(
        "consumer",
        HardwareUnit.MMA,
        instruction_kind=compute_instruction,
    )
    partition = LogicalWarpPartition(
        {producer: 0, consumer: 1},
        2,
    )
    return ProgramScheduleCandidate(
        partition=partition,
        region_schedules=(
            ProgramRegionSchedule(
                region_id=0,
                num_stages=0,
                stages={},
                group_orders={0: {producer: 0}, 1: {consumer: 0}},
            ),
        ),
        cross_group_dependencies=(),
    )


def _target(arch: str) -> Target:
    return Target(
        {
            "kind": "cuda",
            "arch": arch,
            "max_threads_per_block": 1024,
        }
    )


def test_target_profiles_select_architecture_group_granularity():
    hopper = target_execution_profile(_target("sm_90"))
    blackwell = target_execution_profile(_target("sm_100"))
    blackwell_with_register_reallocation = target_execution_profile(
        _target("sm_100"), enable_setmaxnreg=True
    )

    assert hopper.family == CUDAArchitectureFamily.HOPPER
    assert hopper.logical_group_warp_multiple == 4
    assert hopper.setmaxnreg_enabled
    assert blackwell.family == CUDAArchitectureFamily.BLACKWELL
    assert blackwell.logical_group_warp_multiple == 1
    assert not blackwell.setmaxnreg_enabled
    assert blackwell_with_register_reallocation.logical_group_warp_multiple == 4
    assert blackwell_with_register_reallocation.setmaxnreg_enabled


def test_hopper_and_blackwell_allocate_different_group_granularity():
    candidate = _memory_compute_candidate(InstructionKind.TCGEN05_MMA)
    hopper_profile = target_execution_profile(_target("sm_90"))
    hopper_allocations = tuple(
        enumerate_physical_warp_allocations(
            candidate,
            original_threads=128,
            profile=hopper_profile,
        )
    )
    hopper = hopper_allocations[0]
    assert [group.warp_count for group in hopper.groups] == [4, 4]
    assert [group.warp_count for group in hopper_allocations[1].groups] == [
        4,
        8,
    ]
    assert all(
        allocation.groups[0].warp_count == 4
        for allocation in hopper_allocations
    )
    assert hopper.setmaxnreg_enabled
    hopper_register_plan = build_register_plan(hopper, candidate)
    assert [domain.group_id for domain in hopper_register_plan.domains] == [0, 1]
    assert [domain.register_count for domain in hopper_register_plan.domains] == [
        24,
        240,
    ]
    assert [domain.is_increase for domain in hopper_register_plan.domains] == [
        False,
        True,
    ]

    wide_hopper = next(
        enumerate_physical_warp_allocations(
            candidate,
            original_threads=128,
            mode=WarpAllocationMode.EXPLICIT,
            explicit_group_warps={0: 4, 1: 8},
            profile=hopper_profile,
        )
    )
    wide_register_plan = build_register_plan(
        wide_hopper, candidate
    )
    assert [domain.group_id for domain in wide_register_plan.domains] == [
        0,
        1,
        1,
    ]
    assert [domain.first_warp for domain in wide_register_plan.domains] == [
        0,
        4,
        8,
    ]

    with pytest.raises(ValueError, match="violates warp requirement"):
        next(
            enumerate_physical_warp_allocations(
                candidate,
                original_threads=128,
                mode=WarpAllocationMode.EXPLICIT,
                explicit_group_warps={0: 8, 1: 4},
                profile=hopper_profile,
            )
        )

    blackwell_profile = target_execution_profile(_target("sm_100"))
    blackwell = next(
        enumerate_physical_warp_allocations(
            candidate,
            original_threads=128,
            profile=blackwell_profile,
        )
    )
    assert [group.warp_count for group in blackwell.groups] == [1, 4]
    assert not blackwell.setmaxnreg_enabled
    assert not build_register_plan(blackwell, candidate).domains

    blackwell_register_profile = target_execution_profile(
        _target("sm_100"), enable_setmaxnreg=True
    )
    blackwell_with_register_reallocation = next(
        enumerate_physical_warp_allocations(
            candidate,
            original_threads=128,
            profile=blackwell_register_profile,
        )
    )
    assert [
        group.warp_count
        for group in blackwell_with_register_reallocation.groups
    ] == [4, 4]
    assert blackwell_with_register_reallocation.setmaxnreg_enabled


def test_wgmma_expands_compute_budget_to_warpgroup_requirement():
    candidate = _memory_compute_candidate(InstructionKind.WGMMA)
    blackwell_profile = target_execution_profile(_target("sm_100"))

    allocation = next(
        enumerate_physical_warp_allocations(
            candidate,
            original_threads=32,
            profile=blackwell_profile,
        )
    )
    assert [group.warp_count for group in allocation.groups] == [1, 4]
