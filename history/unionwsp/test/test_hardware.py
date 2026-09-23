import pytest
from tvm.target import Target

from history.unionwsp.hardware import (
    HOPPER,
    HardwareSpec,
    InstructionKind,
    OperationInfo,
    resolve_hardware,
)


def test_only_tma_and_wgmma_outputs_support_extra_versions() -> None:
    assert HOPPER.supports_multiversion_output(InstructionKind.TMA)
    assert HOPPER.supports_multiversion_output(InstructionKind.WGMMA)
    assert all(
        not HOPPER.supports_multiversion_output(kind)
        for kind in HOPPER.supported_instructions
        if kind not in {InstructionKind.TMA, InstructionKind.WGMMA}
    )


def test_hopper_classifies_execution_units() -> None:
    assert HOPPER.classify(
        OperationInfo(
            "tl.tileop.copy",
            read_scopes=("global",),
            write_scopes=("shared.dyn",),
        )
    ) == InstructionKind.TMA
    assert HOPPER.classify(
        OperationInfo("parallel", call_names=("tl.tileop.gemm",))
    ) == InstructionKind.WGMMA
    assert HOPPER.classify(
        OperationInfo("parallel", call_names=("tirx.exp2",))
    ) == InstructionKind.FUNCTION
    assert HOPPER.classify(
        OperationInfo("parallel", call_names=("tirx.add",))
    ) == InstructionKind.GENERIC


def test_hopper_defines_natural_group_boundaries() -> None:
    assert HOPPER.allows_cross_group_dependency(
        InstructionKind.TMA, InstructionKind.WGMMA
    )
    assert HOPPER.allows_cross_group_dependency(
        InstructionKind.GSCP, InstructionKind.GENERIC
    )
    assert HOPPER.allows_cross_group_dependency(
        InstructionKind.RSCP, InstructionKind.TMA
    )
    assert HOPPER.can_own_standalone_group(InstructionKind.TMA)
    assert not HOPPER.can_own_standalone_group(InstructionKind.RSCP)
    assert not HOPPER.allows_cross_group_dependency(
        InstructionKind.WGMMA, InstructionKind.FUNCTION
    )
    assert not HOPPER.allows_cross_group_dependency(
        InstructionKind.WGMMA, InstructionKind.TMA
    )


def test_cross_group_dependency_pairs_are_directional() -> None:
    for producer_kind in HOPPER.supported_instructions:
        for consumer_kind in HOPPER.supported_instructions:
            assert HOPPER.allows_cross_group_dependency(
                producer_kind, consumer_kind
            ) == (
                (producer_kind, consumer_kind)
                in HOPPER.cross_group_dependency_pairs
            )


def test_hopper_defines_physical_warp_properties() -> None:
    assert HOPPER.warp_size == 32
    assert HOPPER.warpgroup_warps == 4
    assert HOPPER.max_threads_per_block == 1024
    assert HOPPER.specialized_group_warp_multiple == 4
    assert HOPPER.setmaxnreg_required_for_specialization
    assert HOPPER.warpgroup_collective_kinds == frozenset(
        {InstructionKind.WGMMA}
    )
    assert HOPPER.warpgroup_collective_max_warps is None
    assert HOPPER.warpgroup_collective_tile_rows == 64
    assert HOPPER.collective_max_warps(64) == 4
    assert HOPPER.collective_max_warps(128) == 8
    assert HOPPER.collective_max_warps(192) == 12
    assert InstructionKind.TMA in HOPPER.fixed_one_granule_kinds
    assert HOPPER.register_receiver_kinds == frozenset(
        {InstructionKind.WGMMA}
    )
    assert HOPPER.setmaxnreg_min_registers == 24
    assert HOPPER.setmaxnreg_max_registers == 240
    assert HOPPER.setmaxnreg_granularity == 8
    assert HOPPER.shared_memory_capacity_bytes == 232448
    assert HOPPER.register_file_capacity == 64512
    assert HOPPER.max_registers_per_thread == 255
    assert HOPPER.mbarrier_bytes == 8


def test_resolve_hardware_from_target() -> None:
    assert resolve_hardware(
        Target({"kind": "cuda", "arch": "sm_90a"})
    ) is HOPPER

    with pytest.raises(ValueError, match="unsupported UnionWSP target"):
        resolve_hardware(Target({"kind": "cuda", "arch": "sm_100a"}))
    with pytest.raises(ValueError, match="unsupported UnionWSP target"):
        resolve_hardware(Target({"kind": "cuda", "arch": "sm_900"}))


def test_stage_splittable_pairs_are_directional_and_default_to_false() -> None:
    hardware = HardwareSpec(
        name="test",
        supported_instructions=frozenset(
            {InstructionKind.TMA, InstructionKind.WGMMA}
        ),
        classifier=lambda operation: InstructionKind.TMA,
        issue_priorities={
            InstructionKind.TMA: 1,
            InstructionKind.WGMMA: 0,
        },
        stage_splittable_pairs=frozenset(
            {(InstructionKind.TMA, InstructionKind.WGMMA)}
        ),
    )

    assert hardware.allows_stage_split(
        InstructionKind.TMA, InstructionKind.WGMMA
    )
    assert not hardware.allows_stage_split(
        InstructionKind.WGMMA, InstructionKind.TMA
    )
    assert not hardware.allows_stage_split(
        InstructionKind.TMA, InstructionKind.TMA
    )


def test_instruction_properties_belong_to_each_hardware() -> None:
    hardware = HardwareSpec(
        name="different-properties",
        supported_instructions=frozenset({InstructionKind.TMA}),
        classifier=lambda operation: InstructionKind.TMA,
        issue_priorities={InstructionKind.TMA: 0},
    )

    assert HOPPER.issue_priority(InstructionKind.TMA) == 6
    assert HOPPER.is_async(InstructionKind.TMA)
    assert not hardware.is_async(InstructionKind.TMA)
    assert not hardware.supports_multiversion_output(InstructionKind.TMA)

    with pytest.raises(ValueError, match="issue_priorities must cover"):
        HardwareSpec(
            name="missing-priority",
            supported_instructions=frozenset({InstructionKind.TMA}),
            classifier=lambda operation: InstructionKind.TMA,
        )


def test_hopper_stage_splits_use_an_explicit_directional_whitelist() -> None:
    for producer_kind in HOPPER.supported_instructions:
        for consumer_kind in HOPPER.supported_instructions:
            assert HOPPER.allows_stage_split(
                producer_kind, consumer_kind
            ) == (
                (producer_kind, consumer_kind)
                in HOPPER.stage_splittable_pairs
            )

    assert {
        (InstructionKind.TMA, InstructionKind.WGMMA),
        (InstructionKind.RRCP, InstructionKind.WGMMA),
        (InstructionKind.GENERIC, InstructionKind.WGMMA),
        (InstructionKind.WGMMA, InstructionKind.GENERIC),
    } <= HOPPER.stage_splittable_pairs
    assert not any(
        producer_kind == consumer_kind
        for producer_kind, consumer_kind in HOPPER.stage_splittable_pairs
    )
    assert not HOPPER.allows_stage_split(
        InstructionKind.RRCP, InstructionKind.GENERIC
    )
