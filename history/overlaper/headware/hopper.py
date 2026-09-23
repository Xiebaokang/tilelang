"""Hopper resources, instructions, and operation classification."""

from __future__ import annotations

from .spec import (
    HardwareSpec,
    Instruction,
    InstructionType,
    OperationInfo,
    Resource,
)


class HopperResource(Resource):
    """Physical limits of an NVIDIA Hopper streaming multiprocessor."""

    def __init__(self) -> None:
        super().__init__(
            name="hopper",
            warp_size=32,
            max_threads_per_block=1024,
            register_file_capacity=64512,
            shared_memory_capacity_bytes=253952,
            max_registers_per_thread=255,
            specialized_group_warp_multiple=4,
        )


HOPPER_RESOURCE = HopperResource()


class HopperInstruction(Instruction):
    """Base type for instructions supported by Hopper."""


class GenericInstruction(HopperInstruction):
    def __init__(self) -> None:
        super().__init__("generic")


class FunctionInstruction(HopperInstruction):
    def __init__(self) -> None:
        super().__init__("function")


class RegisterToRegisterCopyInstruction(HopperInstruction):
    def __init__(self) -> None:
        super().__init__("rrcp", InstructionType.COMPUTE, 1)


class RegisterToSharedCopyInstruction(HopperInstruction):
    def __init__(self) -> None:
        super().__init__("rscp", InstructionType.MEMORY, 2)


class GlobalToRegisterCopyInstruction(HopperInstruction):
    def __init__(self) -> None:
        super().__init__("grcp", InstructionType.MEMORY, 3)


class GlobalToSharedCopyInstruction(HopperInstruction):
    def __init__(self) -> None:
        super().__init__("gscp", InstructionType.MEMORY, 4)


class WgmmaInstruction(HopperInstruction):
    def __init__(self) -> None:
        super().__init__("wgmma", issue_priority=5)


class TmaInstruction(HopperInstruction):
    def __init__(self) -> None:
        super().__init__("tma", InstructionType.MEMORY, 6)


GENERIC = GenericInstruction()
FUNCTION = FunctionInstruction()
RRCP = RegisterToRegisterCopyInstruction()
RSCP = RegisterToSharedCopyInstruction()
GRCP = GlobalToRegisterCopyInstruction()
GSCP = GlobalToSharedCopyInstruction()
WGMMA = WgmmaInstruction()
TMA = TmaInstruction()

_SFU_OPS = frozenset(
    {
        "tirx.exp",
        "tirx.exp2",
        "tirx.log",
        "tirx.log2",
        "tirx.sin",
        "tirx.cos",
        "tirx.tanh",
        "tirx.sqrt",
    }
)


def _is_shared(scope: str) -> bool:
    return scope.startswith("shared") and scope != "shared.tmem"


def _is_fragment(scope: str) -> bool:
    return scope == "local.fragment"


def _classify_copy(operation: OperationInfo) -> Instruction:
    scopes = set(operation.read_scopes + operation.write_scopes)
    if operation.name == "tl.tileop.tma_copy":
        return TMA
    if operation.name == "tl.tileop.async_copy":
        return GSCP
    if "global" in scopes and any(_is_shared(scope) for scope in scopes):
        return TMA
    if "global" in scopes and any(_is_fragment(scope) for scope in scopes):
        return GRCP
    if any(_is_fragment(scope) for scope in scopes) and any(
        _is_shared(scope) for scope in scopes
    ):
        return RSCP
    if scopes and all(_is_fragment(scope) for scope in scopes):
        return RRCP
    return GENERIC


def classify_hopper_operation(operation: OperationInfo) -> Instruction:
    """Return the Hopper execution resource used by ``operation``."""

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
        return WGMMA
    if any("tma" in call_name for call_name in (name, *call_names)):
        return TMA
    if any(call_name in _SFU_OPS for call_name in call_names):
        return FUNCTION
    return GENERIC


HOPPER = HardwareSpec(
    name="hopper",
    classifier=classify_hopper_operation,
    supported_instructions=(
        GENERIC,
        FUNCTION,
        RRCP,
        RSCP,
        GRCP,
        GSCP,
        WGMMA,
        TMA,
    ),
    device_resource=HOPPER_RESOURCE,
)
