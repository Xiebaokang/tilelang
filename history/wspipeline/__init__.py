"""Automatic warp-specialized pipeline scheduling."""

from ._bootstrap import ensure_tvm_importable

ensure_tvm_importable()
del ensure_tvm_importable

from .analysis.extractor import analyze_program_dataflow
from .integration.capture import apply_active_schedule, use_schedule_planner
from .search.realization import (
    ProgramRealizedScheduleCandidate,
    enumerate_realized_schedules,
)
from .search.scoring import score_realized_schedule

__all__ = [
    "ProgramRealizedScheduleCandidate",
    "analyze_program_dataflow",
    "apply_active_schedule",
    "enumerate_realized_schedules",
    "score_realized_schedule",
    "use_schedule_planner",
]
