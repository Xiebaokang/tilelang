"""Physical resources bound by ``Architecture.realize()``."""

from __future__ import annotations

from dataclasses import dataclass

from OverlapPlaner.structure.model import Structure


@dataclass(frozen=True, slots=True)
class GroupWarpAllocation:
    """One contiguous warp interval assigned to a logical group."""

    group_id: int
    first_warp: int
    warp_count: int

    def __post_init__(self) -> None:
        if self.group_id < 0:
            raise ValueError("group_id must be non-negative")
        if self.first_warp < 0:
            raise ValueError("first_warp must be non-negative")
        if self.warp_count < 1:
            raise ValueError("warp_count must be positive")

    @property
    def warp_stop(self) -> int:
        return self.first_warp + self.warp_count


@dataclass(frozen=True, slots=True)
class WarpAllocation:
    """Warp widths and optional per-group register redistribution."""

    groups: tuple[GroupWarpAllocation, ...]
    effective_threads: int
    register_counts: tuple[int, ...] | None = None
    register_is_increase: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("warp allocation requires at least one group")
        if tuple(item.group_id for item in self.groups) != tuple(
            range(len(self.groups))
        ):
            raise ValueError("warp groups must use dense ordered IDs")
        if self.effective_threads < 1:
            raise ValueError("effective_threads must be positive")
        if (self.register_counts is None) != (self.register_is_increase is None):
            raise ValueError("register counts and actions must be set together")
        if self.register_counts is not None:
            if len(self.register_counts) != len(self.groups):
                raise ValueError("register_counts must cover every group")
            assert self.register_is_increase is not None
            if len(self.register_is_increase) != len(self.groups):
                raise ValueError("register_is_increase must cover every group")

    @property
    def total_warps(self) -> int:
        return sum(group.warp_count for group in self.groups)

    @property
    def setmaxnreg_enabled(self) -> bool:
        return self.register_counts is not None


@dataclass(frozen=True, slots=True)
class WarpRequirement:
    """Legal warp-count interval for one logical group."""

    minimum: int
    maximum: int
    multiple: int

    def __post_init__(self) -> None:
        if self.minimum < 1 or self.multiple < 1:
            raise ValueError("warp requirement bounds must be positive")


@dataclass(frozen=True, slots=True)
class SetMaxNRegPolicy:
    """Optional per-architecture register redistribution quanta."""

    min_count: int
    max_count: int
    granularity: int

    def __post_init__(self) -> None:
        if min(self.min_count, self.max_count, self.granularity) < 1:
            raise ValueError("setmaxnreg quanta must be positive")
        if self.min_count > self.max_count:
            raise ValueError("setmaxnreg min_count cannot exceed max_count")


@dataclass(frozen=True, slots=True)
class SharedBufferAllocation:
    """One versioned shared buffer placed in the merged arena."""

    buffer_id: int
    name: str
    start: int
    end: int
    size_bytes: int
    alignment: int
    byte_offset: int


@dataclass(frozen=True, slots=True)
class FragmentHandoffAllocation:
    """One C++-inserted fragment↔smem copy buffer placed in the merged arena."""

    channel_id: int
    source_buffer_id: int
    name: str
    size_bytes: int
    alignment: int
    byte_offset: int


@dataclass(frozen=True, slots=True)
class SharedMemoryPlan:
    """Shared buffers, fragment handoffs, and synchronization barriers."""

    shared_buffer_bytes: int
    synchronization_bytes: int
    merged_shared_bytes: int
    shared_memory_capacity_bytes: int
    shared_allocations: tuple[SharedBufferAllocation, ...]
    handoff_allocations: tuple[FragmentHandoffAllocation, ...] = ()

    @property
    def total_shared_bytes(self) -> int:
        return self.merged_shared_bytes

    @property
    def fits(self) -> bool:
        return self.total_shared_bytes <= self.shared_memory_capacity_bytes


@dataclass(frozen=True, slots=True)
class PhysicalPlan:
    """One structure plus the resources ``realize()`` bound to it."""

    structure: Structure
    warp_allocation: WarpAllocation
    shared_memory: SharedMemoryPlan
