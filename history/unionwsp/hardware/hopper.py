"""Hopper operation classification for UnionWSP."""

from __future__ import annotations

from .spec import HardwareSpec, InstructionKind, OperationInfo


_SFU_OPS = {
    "tirx.exp",
    "tirx.exp2",
    "tirx.log",
    "tirx.log2",
    "tirx.sin",
    "tirx.cos",
    "tirx.tanh",
    "tirx.sqrt",
}


def _is_shared(scope: str) -> bool:
    return scope.startswith("shared") and scope != "shared.tmem"


def _is_fragment(scope: str) -> bool:
    return scope == "local.fragment"


def _classify_copy(operation: OperationInfo) -> InstructionKind:
    scopes = set(operation.read_scopes + operation.write_scopes)
    if operation.name == "tl.tileop.tma_copy":
        return InstructionKind.TMA
    if operation.name == "tl.tileop.async_copy":
        return InstructionKind.GSCP
    if "global" in scopes and any(_is_shared(scope) for scope in scopes):
        return InstructionKind.TMA
    if "global" in scopes and any(_is_fragment(scope) for scope in scopes):
        return InstructionKind.GRCP
    if any(_is_fragment(scope) for scope in scopes) and any(
        _is_shared(scope) for scope in scopes
    ):
        return InstructionKind.RSCP
    if scopes and all(_is_fragment(scope) for scope in scopes):
        return InstructionKind.RRCP
    return InstructionKind.GENERIC


def classify_hopper_operation(operation: OperationInfo) -> InstructionKind:
    name = operation.name.lower()
    call_names = tuple(call_name.lower() for call_name in operation.call_names)
    if name in {
        "tl.tileop.copy",
        "tl.tileop.async_copy",
        "tl.tileop.tma_copy",
    }:
        return _classify_copy(operation)
    if any(
        "gemm" in call_name
        or "mma" in call_name
        or "wgmma" in call_name
        for call_name in (name, *call_names)
    ):
        return InstructionKind.WGMMA
    if any("tma" in call_name for call_name in (name, *call_names)):
        return InstructionKind.TMA
    if any(call_name in _SFU_OPS for call_name in call_names):
        return InstructionKind.FUNCTION
    return InstructionKind.GENERIC

_ISSUE_PRIORITIES: dict[InstructionKind, int] = {
    InstructionKind.GENERIC: 0,
    InstructionKind.FUNCTION: 0,
    InstructionKind.RRCP: 1,
    InstructionKind.RSCP: 2,
    InstructionKind.GRCP: 3,
    InstructionKind.GSCP: 4,
    InstructionKind.WGMMA: 5,
    InstructionKind.TMA: 6,
}

_ROLES_CLASSIFY: dict[InstructionKind, int] = {
    InstructionKind.GENERIC: 0,
    InstructionKind.FUNCTION: 0,
    InstructionKind.RRCP: 0,
    InstructionKind.RSCP: 0,
    InstructionKind.GRCP: 0,
    InstructionKind.GSCP: 0,
    InstructionKind.WGMMA: 1,
    InstructionKind.TMA: 2,
}

_SUPPORTED_INSTRUCTIONS = frozenset(_ISSUE_PRIORITIES)
HOPPER = HardwareSpec(
    name="hopper",
    supported_instructions=_SUPPORTED_INSTRUCTIONS,
    classifier=classify_hopper_operation,
    issue_priorities=_ISSUE_PRIORITIES,
    roles_classify=_ROLES_CLASSIFY,

    async_instruction_kinds=frozenset(
        {
            InstructionKind.GSCP,
            InstructionKind.WGMMA,
            InstructionKind.TMA,
        }
    ),
    multiversion_output_instruction_kinds=frozenset(
        {InstructionKind.WGMMA, InstructionKind.TMA}
    ),
    stage_splittable_pairs=frozenset(
        {
            (InstructionKind.TMA, InstructionKind.WGMMA),
            (InstructionKind.WGMMA, InstructionKind.RRCP),
            (InstructionKind.RRCP, InstructionKind.WGMMA),
            (InstructionKind.FUNCTION, InstructionKind.GENERIC),
            (InstructionKind.GENERIC, InstructionKind.FUNCTION),
            (InstructionKind.GENERIC, InstructionKind.WGMMA),
            (InstructionKind.WGMMA, InstructionKind.GENERIC),
            (InstructionKind.WGMMA, InstructionKind.FUNCTION),
            (InstructionKind.FUNCTION, InstructionKind.WGMMA),
        }
    ),
    cross_group_dependency_pairs=frozenset(
        {
            (InstructionKind.TMA, InstructionKind.WGMMA),
            (InstructionKind.GSCP, InstructionKind.WGMMA),
            (InstructionKind.TMA, InstructionKind.GENERIC),
            (InstructionKind.GSCP, InstructionKind.GENERIC),
            (InstructionKind.RSCP, InstructionKind.TMA),
            (InstructionKind.TMA, InstructionKind.RSCP),
            (InstructionKind.GENERIC, InstructionKind.WGMMA),
        }
    ),
    standalone_group_kinds=frozenset({InstructionKind.TMA}),
    warp_size=32,
    max_threads_per_block=1024,
    warpgroup_warps=4,
    specialized_group_warp_multiple=4,
    warpgroup_collective_kinds=frozenset({InstructionKind.WGMMA}),
    # One 64-row output tile is assigned to one four-warp warpgroup.  The
    # effective collective-group limit is derived from each WGMMA output.
    warpgroup_collective_max_warps=None,
    warpgroup_collective_tile_rows=64,
    fixed_one_granule_kinds=frozenset(
        {
            InstructionKind.TMA,
            InstructionKind.GSCP,
            InstructionKind.GRCP,
            InstructionKind.RSCP,
            InstructionKind.RRCP,
        }
    ),
    register_receiver_kinds=frozenset(
        {
            InstructionKind.WGMMA,
            # InstructionKind.GENERIC,
            # InstructionKind.FUNCTION,
        }
    ),
    setmaxnreg_required_for_specialization=True,
    setmaxnreg_min_registers=24,
    setmaxnreg_max_registers=240,
    setmaxnreg_granularity=8,
    shared_memory_capacity_bytes=253952,
    register_file_capacity=64512,
    max_registers_per_thread=255,
    mbarrier_bytes=8,
)
