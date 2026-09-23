"""Coverage for the migrated GQA and MHA backward kernels."""

from __future__ import annotations

import tilelang

from OverlapPlaner.apply import apply_plan_to_ir
from OverlapPlaner.contract import enumerate_overlap_plans, layout_reduced_module
from OverlapPlaner.structure import SearchBudget
from OverlapPlaner.tune.operators import get_operator
from OverlapPlaner.tune.operators.gqa_bwd import OPERATOR as GQA_BWD
from OverlapPlaner.tune.operators.mha_bwd import OPERATOR as MHA_BWD
from OverlapPlaner.tune.run import SEARCH_OPERATORS


CASES = (
    (
        GQA_BWD,
        {
            "gqa_bwd_batch": 1,
            "gqa_bwd_heads": 4,
            "gqa_bwd_groups": 2,
            "gqa_bwd_seq": 128,
            "gqa_bwd_dim_qk": 64,
            "gqa_bwd_dim_v": 64,
            "gqa_bwd_causal": False,
        },
        {"block_m": 128, "block_n": 32},
    ),
    (
        MHA_BWD,
        {
            "mha_bwd_batch": 1,
            "mha_bwd_heads": 4,
            "mha_bwd_seq": 128,
            "mha_bwd_dim": 64,
            "mha_bwd_causal": False,
        },
        {"block_m": 128, "block_n": 32},
    ),
)


def test_attention_backward_operators_are_registered() -> None:
    assert get_operator("gqa_bwd") is GQA_BWD
    assert get_operator("mha_bwd") is MHA_BWD
    selected = {operator.name for operator in SEARCH_OPERATORS}
    assert {"gqa_bwd", "mha_bwd"} <= selected


def test_attention_backward_native_path_removes_search_marker() -> None:
    for operator, options, tile in CASES:
        searched = operator.build(options, tile)
        native = operator.build_native(options, tile)
        assert int(searched.prim_func.attrs["tl.auto_overlap"]) == 1
        assert native.prim_func.attrs.get("tl.auto_overlap") is None
        # dQ remains an explicitly supplied atomic-add buffer. dK/dV are the
        # deterministic outputs used by correctness validation.
        assert searched.out_idx == (7, 8)
        assert native.out_idx == searched.out_idx


def test_attention_backward_enumerates_and_lowers() -> None:
    for operator, options, tile in CASES:
        workload = operator.build(options, tile)
        mod, target = layout_reduced_module(workload.prim_func)
        function = mod[mod.get_global_var("main")]
        plan = next(
            enumerate_overlap_plans(
                function,
                budget=SearchBudget(
                    max_groups=1,
                    max_stages=2,
                    max_structures=1,
                ),
                target=target,
                reduce_ir=False,
            )
        )
        annotated = apply_plan_to_ir(mod, plan)
        with target, tilelang.transform.PassContext(config={}):
            lowered = tilelang.transform.LowerOverlapPlan()(annotated)
        lowered_function = lowered[lowered.get_global_var("main")]
        assert lowered_function.attrs.get("tl.overlap_plan") is not None
