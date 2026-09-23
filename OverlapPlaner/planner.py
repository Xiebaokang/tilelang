"""Install an OverlapPlan provider at the LayoutReducer pipeline boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from tvm import IRModule, tirx
from tvm.target import Target

from .apply import apply_plan_to_ir
from .ir import OverlapPlan

OverlapPlanner = Callable[[str, tirx.PrimFunc, Target], OverlapPlan | None]
_ACTIVE_PLANNER: ContextVar[OverlapPlanner | None] = ContextVar(
    "overlap_plan_planner", default=None
)


def schedule_planner_is_active() -> bool:
    return _ACTIVE_PLANNER.get() is not None


def _requests_auto_overlap(function: tirx.PrimFunc) -> bool:
    if function.attrs is None:
        return False
    value = function.attrs.get("tl.auto_overlap")
    if value is None:
        return False
    try:
        return int(value) != 0
    except (TypeError, ValueError):
        return bool(value)


def apply_active_schedule(mod: IRModule, target: Target) -> IRModule:
    """Apply the active planner to PrimFuncs that request overlap scheduling."""

    planner = _ACTIVE_PLANNER.get()
    if planner is None:
        return mod
    rewritten = mod
    for global_var, function in tuple(mod.functions.items()):
        if not isinstance(function, tirx.PrimFunc):
            continue
        if not _requests_auto_overlap(function):
            continue
        plan = planner(global_var.name_hint, function, target)
        if plan is not None:
            rewritten = apply_plan_to_ir(
                rewritten, plan, global_symbol=global_var.name_hint
            )
    return rewritten


@contextmanager
def use_schedule_planner(planner: OverlapPlanner) -> Iterator[None]:
    """Install one OverlapPlan provider for the duration of compilation."""

    if not callable(planner):
        raise TypeError("planner must be callable")
    if _ACTIVE_PLANNER.get() is not None:
        raise RuntimeError("an OverlapPlan planner is already active")
    token = _ACTIVE_PLANNER.set(planner)
    try:
        yield
    finally:
        _ACTIVE_PLANNER.reset(token)
