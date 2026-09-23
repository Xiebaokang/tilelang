"""Schedule-space enumeration for Overlaper."""

from .group import (
    MAX_NUM_GROUPS,
    build_group_components,
    build_group_opportunities,
    enumerate_bounded_group_assignments,
    enumerate_group_assignments,
    is_group_opportunity,
    is_group_visible_scope,
)
from .guided import (
    score_group_assignment,
    score_stage_assignment,
    score_structure,
)
from .multi_version import (
    analyze_buffer_versions,
    enumerate_buffer_version_plans,
    multiversion_buffers,
    multiversion_candidate_buffers,
)
from .order import ProgramOrders, build_program_orders
from .stage import (
    NUM_STAGES,
    can_split_stages,
    effective_stage_distance,
    enumerate_bounded_stage_assignments,
    enumerate_stage_assignments,
)
from .synchronization import (
    Synchronization,
    SynchronizationCycleError,
    SynchronizationKind,
    SynchronizationScope,
    build_synchronizations,
)

__all__ = [
    "MAX_NUM_GROUPS",
    "NUM_STAGES",
    "ProgramOrders",
    "Synchronization",
    "SynchronizationCycleError",
    "SynchronizationKind",
    "SynchronizationScope",
    "analyze_buffer_versions",
    "build_group_components",
    "build_group_opportunities",
    "build_program_orders",
    "build_synchronizations",
    "can_split_stages",
    "effective_stage_distance",
    "enumerate_buffer_version_plans",
    "enumerate_bounded_group_assignments",
    "enumerate_group_assignments",
    "enumerate_bounded_stage_assignments",
    "score_group_assignment",
    "score_stage_assignment",
    "score_structure",
    "enumerate_stage_assignments",
    "is_group_opportunity",
    "is_group_visible_scope",
    "multiversion_buffers",
    "multiversion_candidate_buffers",
]
