"""Invoke a caller-selected UnionWSP schedule at the LayoutReducer boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from tvm import IRModule, tirx
from tvm.target import Target
from tvm.tirx.stmt_functor import post_order_visit

from ..parseIR import extract_dataflow_graph
from ..parseIR.graph import DataflowGraph
from .apply import apply_schedule_to_ir

if TYPE_CHECKING:
    from .. import WSPSchedule

SchedulePlanner = Callable[
    [str, DataflowGraph, Target],
    "WSPSchedule | None",
]
_ACTIVE_PLANNER: ContextVar[SchedulePlanner | None] = ContextVar(
    "unionwsp_schedule_planner", default=None
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
    """Apply the active planner after LayoutReducer; otherwise do nothing."""

    planner = _ACTIVE_PLANNER.get()
    if planner is None:
        return mod
    rewritten = mod
    for global_var, function in tuple(mod.functions.items()):
        if not isinstance(function, tirx.PrimFunc) or not _requests_auto_schedule(
            function
        ):
            continue
        graph = extract_dataflow_graph(function, target=target)
        schedule = planner(global_var.name_hint, graph, target)
        if schedule is None:
            continue
        rewritten = apply_schedule_to_ir(
            rewritten,
            graph,
            schedule,
            global_symbol=global_var.name_hint,
        )
    return rewritten


@contextmanager
def use_schedule_planner(planner: SchedulePlanner) -> Iterator[None]:
    """Install one UnionWSP planner for the duration of compilation."""

    if _ACTIVE_PLANNER.get() is not None:
        raise RuntimeError("a UnionWSP schedule planner is already active")
    token = _ACTIVE_PLANNER.set(planner)
    try:
        yield
    finally:
        _ACTIVE_PLANNER.reset(token)
