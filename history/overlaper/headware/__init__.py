"""Hardware descriptions available to Overlaper."""

from .hopper import HOPPER
from .spec import (
    HardwareSpec,
    Instruction,
    InstructionType,
    OperationInfo,
    Resource,
)


def __getattr__(name: str):
    if name == "resolve_hardware":
        from .resolve import resolve_hardware

        return resolve_hardware
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HOPPER",
    "HardwareSpec",
    "Instruction",
    "InstructionType",
    "OperationInfo",
    "Resource",
    "resolve_hardware",
]
