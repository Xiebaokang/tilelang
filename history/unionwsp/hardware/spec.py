"""Small hardware-description types shared by UnionWSP backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType



class InstructionKind(str, Enum):
    """Execution units currently understood by UnionWSP."""

    GENERIC = "generic"
    FUNCTION = "function"
    WGMMA = "wgmma"
    TMA = "tma"
    GSCP = "gscp"
    GRCP = "grcp"
    RSCP = "rscp"
    RRCP = "rrcp"


@dataclass(frozen=True)
class OperationInfo:
    """Target-neutral facts supplied to a hardware classifier."""

    name: str
    call_names: tuple[str, ...] = ()
    read_scopes: tuple[str, ...] = ()
    write_scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class HardwareSpec:
    """One hardware backend's supported units and classification rule."""

    name: str
    supported_instructions: frozenset[InstructionKind]
    classifier: Callable[[OperationInfo], InstructionKind] = field(repr=False)
    issue_priorities: Mapping[InstructionKind, int] = field(
        default_factory=dict, repr=False, hash=False
    )
    roles_classify: Mapping[InstructionKind, int] = field(
        default_factory=dict, repr=False, hash=False
    )

    async_instruction_kinds: frozenset[InstructionKind] = frozenset()
    multiversion_output_instruction_kinds: frozenset[
        InstructionKind
    ] = frozenset()
    stage_splittable_pairs: frozenset[
        tuple[InstructionKind, InstructionKind]
    ] = frozenset()
    cross_group_dependency_pairs: frozenset[
        tuple[InstructionKind, InstructionKind]
    ] = frozenset()
    standalone_group_kinds: frozenset[InstructionKind] = frozenset()

    warp_size: int = 32
    max_threads_per_block: int = 1024
    warpgroup_warps: int = 1
    specialized_group_warp_multiple: int = 1
    warpgroup_collective_kinds: frozenset[InstructionKind] = frozenset()
    warpgroup_collective_max_warps: int | None = None
    warpgroup_collective_tile_rows: int | None = None
    fixed_one_granule_kinds: frozenset[InstructionKind] = frozenset()
    register_receiver_kinds: frozenset[InstructionKind] = frozenset()
    setmaxnreg_required_for_specialization: bool = False
    setmaxnreg_min_registers: int = 24
    setmaxnreg_max_registers: int = 240
    setmaxnreg_granularity: int = 8
    shared_memory_capacity_bytes: int = 0
    register_file_capacity: int = 0
    max_registers_per_thread: int = 255
    mbarrier_bytes: int = 0

    def __post_init__(self) -> None:
        priorities = dict(self.issue_priorities)
        priority_kinds = set(priorities)
        if priority_kinds != set(self.supported_instructions):
            missing = self.supported_instructions - priority_kinds
            extra = priority_kinds - self.supported_instructions
            details = []
            if missing:
                details.append(
                    "missing=" + ",".join(sorted(kind.value for kind in missing))
                )
            if extra:
                details.append(
                    "unsupported=" + ",".join(sorted(kind.value for kind in extra))
                )
            raise ValueError(
                "issue_priorities must cover supported instructions exactly: "
                + " ".join(details)
            )
        if any(
            not isinstance(priority, int)
            for priority in priorities.values()
        ):
            raise ValueError("instruction priorities must be integers")
        object.__setattr__(
            self, "issue_priorities", MappingProxyType(priorities)
        )

        unsupported = {
            kind
            for pair in (
                self.stage_splittable_pairs
                | self.cross_group_dependency_pairs
            )
            for kind in pair
            if kind not in self.supported_instructions
        }
        unsupported.update(
            self.standalone_group_kinds - self.supported_instructions
        )
        unsupported.update(
            self.async_instruction_kinds - self.supported_instructions
        )
        unsupported.update(
            self.multiversion_output_instruction_kinds
            - self.supported_instructions
        )
        unsupported.update(
            self.warpgroup_collective_kinds - self.supported_instructions
        )
        unsupported.update(
            self.fixed_one_granule_kinds - self.supported_instructions
        )
        unsupported.update(
            self.register_receiver_kinds - self.supported_instructions
        )
        if unsupported:
            names = ", ".join(sorted(kind.value for kind in unsupported))
            raise ValueError(
                f"hardware {self.name} uses unsupported instructions in "
                f"scheduling pairs: {names}"
            )
        if (
            self.warp_size < 1
            or self.warpgroup_warps < 1
            or self.specialized_group_warp_multiple < 1
        ):
            raise ValueError("warp and warpgroup sizes must be positive")
        if (
            self.max_threads_per_block < self.warp_size
            or self.max_threads_per_block % self.warp_size
        ):
            raise ValueError(
                "max_threads_per_block must contain whole warps"
            )
        if self.warpgroup_collective_max_warps is not None and (
            self.warpgroup_collective_max_warps < self.warpgroup_warps
            or self.warpgroup_collective_max_warps % self.warpgroup_warps
            or self.warpgroup_collective_max_warps
            > self.max_threads_per_block // self.warp_size
        ):
            raise ValueError(
                "warpgroup_collective_max_warps must contain whole "
                "warpgroups within the CTA limit"
            )
        if (
            self.warpgroup_collective_tile_rows is not None
            and self.warpgroup_collective_tile_rows < 1
        ):
            raise ValueError(
                "warpgroup_collective_tile_rows must be positive or None"
            )
        if self.setmaxnreg_required_for_specialization:
            if (
                self.specialized_group_warp_multiple
                % self.warpgroup_warps
            ):
                raise ValueError(
                    "setmaxnreg groups must contain whole warpgroups"
                )
        if (
            self.setmaxnreg_min_registers < 1
            or self.setmaxnreg_max_registers
            < self.setmaxnreg_min_registers
            or self.setmaxnreg_granularity < 1
            or self.setmaxnreg_min_registers
            % self.setmaxnreg_granularity
            or self.setmaxnreg_max_registers
            % self.setmaxnreg_granularity
        ):
            raise ValueError("invalid setmaxnreg limits")
        if (
            self.shared_memory_capacity_bytes < 0
            or self.register_file_capacity < 0
            or self.max_registers_per_thread < 1
            or self.mbarrier_bytes < 0
        ):
            raise ValueError("hardware resource limits cannot be negative")

    def classify(self, operation: OperationInfo) -> InstructionKind:
        kind = self.classifier(operation)
        if kind not in self.supported_instructions:
            raise ValueError(
                f"hardware {self.name} classified {operation.name} as "
                f"unsupported instruction {kind.value}"
            )
        return kind

    def issue_priority(self, kind: InstructionKind) -> int:
        """Return this hardware's ready-list priority for an instruction."""

        self._require_supported(kind)
        return self.issue_priorities[kind]

    def collective_max_warps(self, tile_rows: int | None = None) -> int | None:
        """Return the fixed and tile-derived collective-group warp limit."""

        maximum = self.warpgroup_collective_max_warps
        if self.warpgroup_collective_tile_rows is None or tile_rows is None:
            return maximum
        tile_maximum = (
            tile_rows
            // self.warpgroup_collective_tile_rows
            * self.warpgroup_warps
        )
        return tile_maximum if maximum is None else min(maximum, tile_maximum)

    def is_async(self, kind: InstructionKind) -> bool:
        """Return whether this hardware executes the instruction asynchronously."""

        self._require_supported(kind)
        return kind in self.async_instruction_kinds

    def supports_multiversion_output(self, kind: InstructionKind) -> bool:
        """Return whether outputs may independently use an extra version."""

        self._require_supported(kind)
        return kind in self.multiversion_output_instruction_kinds

    def _require_supported(self, kind: InstructionKind) -> None:
        if kind not in self.supported_instructions:
            raise ValueError(
                f"hardware {self.name} does not support instruction: "
                f"{kind.value}"
            )

    def allows_stage_split(
        self,
        producer_kind: InstructionKind,
        consumer_kind: InstructionKind,
    ) -> bool:
        """Whether this ordered direct-dependency pair may cross stages."""

        unsupported = {
            kind
            for kind in (producer_kind, consumer_kind)
            if kind not in self.supported_instructions
        }
        if unsupported:
            names = ", ".join(sorted(kind.value for kind in unsupported))
            raise ValueError(
                f"hardware {self.name} does not support instructions: {names}"
            )
        return (
            producer_kind,
            consumer_kind,
        ) in self.stage_splittable_pairs

    def allows_cross_group_dependency(
        self,
        producer_kind: InstructionKind,
        consumer_kind: InstructionKind,
    ) -> bool:
        """Whether this ordered direct-dependency pair may cross groups."""

        unsupported = {
            kind
            for kind in (producer_kind, consumer_kind)
            if kind not in self.supported_instructions
        }
        if unsupported:
            names = ", ".join(sorted(kind.value for kind in unsupported))
            raise ValueError(
                f"hardware {self.name} does not support instructions: {names}"
            )
        return (
            producer_kind,
            consumer_kind,
        ) in self.cross_group_dependency_pairs

    def can_own_standalone_group(self, kind: InstructionKind) -> bool:
        """Whether a serial operation of this kind may create a group role."""

        self._require_supported(kind)
        return kind in self.standalone_group_kinds

    def execution_role(self, kind: InstructionKind) -> int:
        """Return the coarse hardware role used to bound stage/group search."""

        self._require_supported(kind)
        role_id = self.roles_classify.get(kind)
        if role_id is None:
            raise ValueError(
                f"hardware {self.name} does not support instruction: "
                f"{kind.value}"
            )
        return role_id
