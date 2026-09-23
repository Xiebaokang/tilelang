"""Target-independent data types for pipeline scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum


class HardwareUnit(str, Enum):
    """Coarse hardware resources used by pipeline analysis."""

    LOAD_STORE = "LOAD/STORE"
    TMA = "TMA"
    MMA = "MMA"
    TMEM = "TMEM"
    SFU = "SFU"
    ALU = "ALU"


class ExecutionKind(IntEnum):
    """Coarse synchronous/asynchronous execution behavior of an operation."""

    SYNC_COMPUTE = 0
    SYNC_MEMORY = 1
    ASYNC_COMPUTE = 2
    ASYNC_MEMORY = 3


class InstructionKind(str, Enum):
    """Target-relevant operation semantics without fixing a physical target."""

    GENERIC = "generic"
    GENERIC_MMA = "generic_mma"
    WGMMA = "wgmma"
    TCGEN05_MMA = "tcgen05_mma"
    TCGEN05_CP = "tcgen05_cp"
    TCGEN05_LD = "tcgen05_ld"
    TCGEN05_ST = "tcgen05_st"
    TMA = "tma"


_DEFAULT_EXECUTION_KIND = {
    HardwareUnit.LOAD_STORE: ExecutionKind.SYNC_MEMORY,
    HardwareUnit.TMA: ExecutionKind.ASYNC_MEMORY,
    HardwareUnit.MMA: ExecutionKind.SYNC_COMPUTE,
    HardwareUnit.TMEM: ExecutionKind.ASYNC_MEMORY,
    HardwareUnit.SFU: ExecutionKind.SYNC_COMPUTE,
    HardwareUnit.ALU: ExecutionKind.SYNC_COMPUTE,
}


@dataclass(frozen=True)
class OperationProfile:
    """Target-independent structural work contained in one IR operation."""

    logical_elements: int = 1
    alu_ops: int = 0
    sfu_ops: int = 0
    cast_ops: int = 0
    mma_ops: int = 0
    memory_bytes: int = 0
    source_scopes: tuple[str, ...] = ()
    destination_scopes: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class DataflowNode:
    """One identity-distinct operation extracted from a pipelined TIR loop."""

    name: str
    unit: HardwareUnit
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    execution_kind: ExecutionKind | None = None
    profile: OperationProfile = field(default_factory=OperationProfile)
    instruction_kind: InstructionKind = InstructionKind.GENERIC

    def __post_init__(self) -> None:
        if self.execution_kind is None:
            object.__setattr__(
                self, "execution_kind", _DEFAULT_EXECUTION_KIND[self.unit]
            )


@dataclass
class DataflowEdge:
    """One precedence edge carrying every memory hazard for its relation."""

    producer: DataflowNode
    consumer: DataflowNode
    iteration_distance: int = 0
    dependency_kinds: frozenset[str] = frozenset({"RAW"})
    buffer: str | None = None
    buffer_id: int | None = None

    def __post_init__(self) -> None:
        self.dependency_kinds = (
            frozenset({self.dependency_kinds})
            if isinstance(self.dependency_kinds, str)
            else frozenset(self.dependency_kinds)
        )
        if not self.dependency_kinds:
            raise ValueError("dependency_kinds cannot be empty")
        unsupported = self.dependency_kinds - {"RAW", "WAR", "WAW"}
        if unsupported:
            raise ValueError(
                f"unsupported dataflow dependency kinds: {sorted(unsupported)}"
            )

    @property
    def is_loop_carried(self) -> bool:
        return self.iteration_distance > 0
