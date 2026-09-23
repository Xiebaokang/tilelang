"""Physical realization of L1 structures."""

from .model import (
    FragmentHandoffAllocation,
    GroupWarpAllocation,
    PhysicalPlan,
    SetMaxNRegPolicy,
    SharedBufferAllocation,
    SharedMemoryPlan,
    WarpAllocation,
    WarpRequirement,
)
from .shared_memory import analyze_shared_memory
from .warp import (
    enumerate_warp_allocations,
    estimate_group_registers_per_thread,
    warp_requirements_are_feasible,
)

__all__ = [
    "FragmentHandoffAllocation",
    "GroupWarpAllocation",
    "PhysicalPlan",
    "SetMaxNRegPolicy",
    "SharedBufferAllocation",
    "SharedMemoryPlan",
    "WarpAllocation",
    "WarpRequirement",
    "analyze_shared_memory",
    "enumerate_warp_allocations",
    "estimate_group_registers_per_thread",
    "warp_requirements_are_feasible",
]
