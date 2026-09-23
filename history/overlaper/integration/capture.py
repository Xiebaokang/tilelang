"""Invoke a caller-selected Overlaper planner at the LayoutReducer boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from tvm import IRModule, tirx
from tvm.target import Target
from tvm.tirx.stmt_functor import post_order_visit

from ..parse import DataflowGraph, extract_dataflow_graph
from .apply import ScheduleLike, apply_schedule_to_ir


SchedulePlanner = Callable[
    [str, DataflowGraph, Target],
    ScheduleLike | None,
]
_ACTIVE_PLANNER: ContextVar[SchedulePlanner | None] = ContextVar(
    "overlaper_schedule_planner", default=None
)


def schedule_planner_is_active() -> bool:
    return _ACTIVE_PLANNER.get() is not None


def _requests_auto_schedule(function: tirx.PrimFunc) -> bool:
    requested = False

    def visit(node) -> None:
        nonlocal requested
        if isinstance(node, tirx.For):
            value = node.annotations.get("tl.wsp.auto_schedule")
            requested = requested or (value is not None and int(value) != 0)

    post_order_visit(function.body, visit)
    return requested


def apply_active_schedule(mod: IRModule, target: Target) -> IRModule:
    """Apply the active planner to PrimFuncs that request overlap scheduling."""

    planner = _ACTIVE_PLANNER.get()
    if planner is None:
        return mod
    rewritten = mod
    for global_var, function in tuple(mod.functions.items()):
        if not isinstance(function, tirx.PrimFunc):
            continue
        if not _requests_auto_schedule(function):
            continue
        graph = extract_dataflow_graph(function, target=target)
        schedule = planner(global_var.name_hint, graph, target)
        if schedule is not None:
            rewritten = apply_schedule_to_ir(
                rewritten,
                graph,
                schedule,
                global_symbol=global_var.name_hint,
            )
    return rewritten


@contextmanager
def use_schedule_planner(planner: SchedulePlanner) -> Iterator[None]:
    """Install one planner for the duration of compilation."""

    if not callable(planner):
        raise TypeError("planner must be callable")
    if _ACTIVE_PLANNER.get() is not None:
        raise RuntimeError("an Overlaper planner is already active")
    token = _ACTIVE_PLANNER.set(planner)
    try:
        yield
    finally:
        _ACTIVE_PLANNER.reset(token)
