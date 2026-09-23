"""Attach a typed OverlapPlan to a PrimFunc."""

from __future__ import annotations

from tvm import IRModule, tirx

from .ir import OverlapPlan


def apply_plan_to_ir(
    mod: IRModule,
    plan: OverlapPlan,
    *,
    global_symbol: str | None = None,
) -> IRModule:
    """Attach a typed OverlapPlan to one PrimFunc."""

    if not isinstance(mod, IRModule):
        raise TypeError(f"mod must be IRModule, got {type(mod).__name__}")
    if not isinstance(plan, OverlapPlan):
        raise TypeError(f"plan must be OverlapPlan, got {type(plan).__name__}")
    if global_symbol is None:
        matches = [
            global_var.name_hint
            for global_var, function in mod.functions.items()
            if isinstance(function, tirx.PrimFunc)
        ]
        if len(matches) != 1:
            raise ValueError("cannot uniquely match a PrimFunc; pass global_symbol")
        global_symbol = matches[0]
    rewritten = mod.clone()
    global_var = rewritten.get_global_var(global_symbol)
    function = rewritten[global_var]
    if not isinstance(function, tirx.PrimFunc):
        raise TypeError(f"{global_symbol!r} is not a PrimFunc")
    rewritten.update_func(
        global_var, function.with_attr("tl.overlap_plan", plan)
    )
    return rewritten
