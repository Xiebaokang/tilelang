"""UnionWSP schedule enumeration."""

from .multi_version import (
    analyze_buffer_versions,
    enumerate_buffer_version_plans,
    multiversion_buffers,
    multiversion_candidate_buffers,
)
from .order import ProgramOrders, build_program_orders
from .stage import (
    effective_stage_distance,
    enumerate_stage_assignments,
    infer_max_stages,
)
from .synchronization import (
    SynchronizationChannel,
    SynchronizationChannelKind,
    SynchronizationCycleError,
    SynchronizationScope,
    build_synchronization_channels,
    validate_synchronization_channels,
)
from .warp_specialization import (
    build_group_components,
    build_ws_opportunities,
    enumerate_group_assignments,
    infer_max_groups,
    is_legal_group_cut,
)

__all__ = [
    "ProgramOrders",
    "SynchronizationChannel",
    "SynchronizationChannelKind",
    "SynchronizationCycleError",
    "SynchronizationScope",
    "analyze_buffer_versions",
    "build_group_components",
    "build_program_orders",
    "build_synchronization_channels",
    "build_ws_opportunities",
    "effective_stage_distance",
    "enumerate_buffer_version_plans",
    "enumerate_group_assignments",
    "enumerate_stage_assignments",
    "infer_max_groups",
    "infer_max_stages",
    "is_legal_group_cut",
    "multiversion_buffers",
    "multiversion_candidate_buffers",
    "validate_synchronization_channels",
]
