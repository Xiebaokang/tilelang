"""Target hardware descriptions used by Overlaper.

The description deliberately keeps scheduling behaviour on instructions rather
than in target-wide pair whitelists.  This makes a new target a small set of
resource and instruction definitions plus an operation classifier.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True, slots=True)
class Resource:
    """Physical resource limits and optional capabilities of one target."""

    name: str
    warp_size: int
    max_threads_per_block: int
    register_file_capacity: int
    shared_memory_capacity_bytes: int
    max_registers_per_thread: int
    specialized_group_warp_multiple: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("resource name must not be empty")
        if self.warp_size < 1:
            raise ValueError("warp_size must be positive")
        if (
            self.max_threads_per_block < self.warp_size
            or self.max_threads_per_block % self.warp_size
        ):
            raise ValueError("max_threads_per_block must contain whole warps")
        if self.specialized_group_warp_multiple < 1:
            raise ValueError("specialized_group_warp_multiple must be positive")
        if (
            self.register_file_capacity < 0
            or self.shared_memory_capacity_bytes < 0
            or self.max_registers_per_thread < 1
        ):
            raise ValueError("resource capacities must be non-negative")

    @property
    def max_warps_per_block(self) -> int:
        """Maximum number of whole warps that fit in one thread block."""

        return self.max_threads_per_block // self.warp_size


class InstructionType(Enum):
    """Broad target-independent instruction categories."""

    MEMORY = 0
    COMPUTE = 1


@dataclass(frozen=True, slots=True)
class Instruction:
    """Behavior shared by one target-specific instruction kind."""

    name: str
    type: InstructionType = InstructionType.COMPUTE
    issue_priority: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("instruction name must not be empty")
        if not isinstance(self.issue_priority, int):
            raise TypeError("instruction issue_priority must be an integer")


@dataclass(frozen=True, slots=True)
class OperationInfo:
    """Target-neutral facts supplied to a hardware classifier."""

    name: str
    call_names: tuple[str, ...] = ()
    read_scopes: tuple[str, ...] = ()
    write_scopes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    """A validated collection of target-specific resources and instructions."""

    name: str
    classifier: Callable[[OperationInfo], Instruction] = field(
        repr=False, compare=False
    )
    supported_instructions: tuple[Instruction, ...]
    device_resource: Resource

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("hardware name must not be empty")
        if not callable(self.classifier):
            raise TypeError("classifier must be callable")
        if not isinstance(self.device_resource, Resource):
            raise TypeError("device_resource must be a Resource")
        if not self.supported_instructions:
            raise ValueError("hardware must support at least one instruction")
        if any(
            not isinstance(instruction, Instruction)
            for instruction in self.supported_instructions
        ):
            raise TypeError("supported_instructions must contain Instructions")
        names = [instruction.name for instruction in self.supported_instructions]
        if len(names) != len(set(names)):
            raise ValueError("supported instruction names must be unique")

    def classify(self, operation: OperationInfo) -> Instruction:
        """Classify an operation and ensure the target supports the result."""

        if not isinstance(operation, OperationInfo):
            raise TypeError("operation must be an OperationInfo")
        instruction = self.classifier(operation)
        if not isinstance(instruction, Instruction):
            raise TypeError("classifier must return an Instruction")
        if instruction not in self.supported_instructions:
            raise ValueError(
                f"hardware {self.name} classified {operation.name} as "
                f"unsupported instruction {instruction.name}"
            )
        return instruction
