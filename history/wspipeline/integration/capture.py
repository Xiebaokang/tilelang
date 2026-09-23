"""Adapters for capture and planning at the CUDA LayoutReducer boundary."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

from tvm import IRModule, tirx
from tvm.target import Target

from ..analysis.extractor import (
    _auto_wsp_enabled,
    _find_pipelined_loops,
    analyze_program_dataflow,
)
from ..analysis.program import ProgramDataflowAnalysis
from ..search.realization import ProgramRealizedScheduleCandidate

SchedulePlanner = Callable[
    [str, ProgramDataflowAnalysis, Target],
    ProgramRealizedScheduleCandidate | None,
]
_ACTIVE_SCHEDULE_PLANNER: ContextVar[SchedulePlanner | None] = (
    ContextVar("tilelang_wsp_schedule_planner", default=None)
)


def _analyze_auto_program_module(
    mod: IRModule,
) -> dict[str, ProgramDataflowAnalysis]:
    """Analyze only PrimFuncs that explicitly request automatic WSP."""

    functions: dict[str, ProgramDataflowAnalysis] = {}
    for global_var, func in mod.functions.items():
        if not isinstance(func, tirx.PrimFunc):
            continue
        loops = _find_pipelined_loops(func)
        if not loops or not any(map(_auto_wsp_enabled, loops)):
            continue
        functions[global_var.name_hint] = analyze_program_dataflow(func)
    return functions


def _module_requests_auto_wsp(mod: IRModule) -> bool:
    return any(
        isinstance(func, tirx.PrimFunc)
        and any(_auto_wsp_enabled(loop) for loop in _find_pipelined_loops(func))
        for func in mod.functions.values()
    )


def _select_auto_schedule(
    analysis: ProgramDataflowAnalysis,
    target: Target,
) -> ProgramRealizedScheduleCandidate:
    from ..search.realization import enumerate_realized_schedules
    from ..search.scoring import score_realized_schedule

    candidates = enumerate_realized_schedules(
        analysis, target, examined_limit=1000
    )
    try:
        return min(
            candidates,
            key=lambda candidate: score_realized_schedule(
                analysis, candidate, target
            ),
        )
    except ValueError as error:
        raise RuntimeError(
            "automatic WSP was requested, but no valid schedule was found"
        ) from error


def apply_active_schedule(mod: IRModule, target: Target) -> IRModule:
    """Run the active planner and return the optionally annotated IR."""

    if not _module_requests_auto_wsp(mod):
        return mod
    planner = _ACTIVE_SCHEDULE_PLANNER.get()

    from .apply import apply_schedule_to_ir

    analyses = _analyze_auto_program_module(mod)
    rewritten = mod
    for global_symbol, analysis in analyses.items():
        auto_requested = any(region.auto_schedule for region in analysis.regions)
        if not auto_requested:
            continue
        if planner is None:
            schedule = _select_auto_schedule(analysis, target)
        else:
            schedule = planner(global_symbol, analysis, target)
        if schedule is None:
            continue
        rewritten = apply_schedule_to_ir(
            rewritten,
            analysis,
            schedule,
            target,
            global_symbol=global_symbol,
        )
    return rewritten


@contextmanager
def use_schedule_planner(
    planner: SchedulePlanner,
) -> Iterator[None]:
    """Attach planner-selected whole-program schedules during lowering."""

    if _ACTIVE_SCHEDULE_PLANNER.get() is not None:
        raise RuntimeError("a program schedule planner is already active")
    token = _ACTIVE_SCHEDULE_PLANNER.set(planner)
    try:
        yield
    finally:
        _ACTIVE_SCHEDULE_PLANNER.reset(token)
