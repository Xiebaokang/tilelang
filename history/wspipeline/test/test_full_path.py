import inspect

import pytest
import torch

import history.wspipeline as wsp
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit

from history.wspipeline.test.fa3_kernel import (
    main,
    make_cuda_target,
    make_fa3_prim_func,
    make_program_schedule_planner,
)
from history.wspipeline.scheduling.stage import infer_max_stages


def _has_hopper_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def test_automatic_entry_has_no_manual_schedule_size_controls():
    closed_parameters = {
        "num_stages",
        "num_groups",
        "max_num_versions",
        "mma_warpgroups",
    }
    assert closed_parameters.isdisjoint(inspect.signature(main).parameters)
    assert closed_parameters.isdisjoint(
        inspect.signature(make_program_schedule_planner).parameters
    )
    assert closed_parameters.isdisjoint(
        inspect.signature(wsp.enumerate_realized_schedules).parameters
    )


def test_auto_wsp_annotation_replaces_manual_pipeline_depth():
    loops = []

    def collect(node):
        if isinstance(node, tirx.For) and "tl.wsp.auto_schedule" in node.annotations:
            loops.append(node)

    post_order_visit(make_fa3_prim_func().body, collect)
    assert len(loops) == 1
    assert int(loops[0].annotations["tl.wsp.auto_schedule"]) == 1
    assert "num_stages" not in loops[0].annotations


def test_automatic_search_includes_single_stage_and_single_group_baseline():
    analysis = wsp.analyze_program_dataflow(make_fa3_prim_func())
    pipeline_region = analysis.regions[1]
    assert infer_max_stages(analysis, pipeline_region) == 2

    candidates = tuple(
        wsp.enumerate_realized_schedules(
            analysis,
            make_cuda_target(),
            examined_limit=120,
        )
    )
    combinations = {
        (
            candidate.logical.partition.num_groups,
            candidate.logical.region_schedules[1].num_stages,
        )
        for candidate in candidates
    }
    assert (1, 1) in combinations
    assert {groups for groups, _ in combinations} == {1, 2, 3}
    assert {stages for _, stages in combinations} == {1, 2}


def test_inferred_stage_bound_uses_program_roles_and_static_loop_extent():
    analysis = wsp.analyze_program_dataflow(
        make_fa3_prim_func(seq_kv=8192)
    )
    assert infer_max_stages(analysis, analysis.regions[1]) == 4


@pytest.mark.skipif(not _has_hopper_gpu(), reason="requires a Hopper CUDA GPU")
def test_automatic_fa3_search_lowering_and_execution():
    """Exercise extraction, hierarchical ranking, realization, and lowering."""

    compiled = main(
        examined_limit=256,
    )
    assert compiled is not None
