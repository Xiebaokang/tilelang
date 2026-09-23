"""Hardware descriptions available to UnionWSP."""

from .hopper import HOPPER
from .spec import (
    HardwareSpec,
    InstructionKind,
    OperationInfo,
)


def __getattr__(name: str):
    if name == "resolve_hardware":
        from .resolve import resolve_hardware

        return resolve_hardware
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "HOPPER",
    "HardwareSpec",
    "InstructionKind",
    "OperationInfo",
    "resolve_hardware",
]
