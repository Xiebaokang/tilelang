"""Backend selection must survive serialization and control generated code."""

import json

import pytest
import tilelang
from tvm.target import Target
import tilelang.language as T
import torch

from OverlapPlaner.arch import HOPPER
from OverlapPlaner.arch.hopper import optional_tma_copy_ids
from OverlapPlaner.contract import enumerate_overlap_plans, layout_reduced_prim_func
from OverlapPlaner.facts import extract_fact_graph
from OverlapPlaner.serialization import plan_from_dict, plan_to_dict, save_plan_json
from OverlapPlaner.structure import SearchBudget
from OverlapPlaner.tune.copy_search import (
    classified_for_plan, realize_copy_configuration, selected_optional_tma_copies,
)
from OverlapPlaner.tune.run import generate_plans
from OverlapPlaner.tune.search import evaluate
from OverlapPlaner.tune.operators.workloads import OperatorSpec, SearchWorkload


def build(options, tile):
    @T.prim_func(auto_overlap=True)
    def kernel(A: T.Tensor((1024,), T.float16), B: T.Tensor((1024,), T.float16)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((1024,), T.float16)
            T.copy(A, shared)
            T.copy(shared, B)
    return SearchWorkload(prim_func=kernel, out_idx=(1,), total_flops=1024,
                          reference_program=lambda a: a)


OPERATOR = OperatorSpec(name="copy_backend_test", description="load/store",
                        add_cli_arguments=lambda p: None,
                        configuration_factory=lambda o: [{}], workload_factory=build)


def candidates():
    function = layout_reduced_prim_func(build({}, {}).prim_func)
    classified = HOPPER.classify(extract_fact_graph(function))
    parent = plan_to_dict(next(enumerate_overlap_plans(
        function, reduce_ir=False, budget=SearchBudget(max_groups=1, max_structures=1)
    )))
    return classified, parent


@pytest.mark.parametrize("selected", [set(), {0}, {1}, {0, 1}])
def test_load_and_store_choices_roundtrip_and_lower(selected):
    from tilelang.engine import lower
    from OverlapPlaner.planner import use_schedule_planner

    classified, parent = candidates()
    assert set(optional_tma_copy_ids(classified)) == {0, 1}
    payload = realize_copy_configuration(classified, parent, selected)
    assert payload is not None
    assert payload["groups"] == parent["groups"]
    assert selected_optional_tma_copies(classified, payload) == selected
    assert plan_to_dict(plan_from_dict(payload)) == payload
    traits = classified_for_plan(classified, payload)
    assert traits.traits_for(0).async_completion == (0 in selected)
    assert not traits.traits_for(1).async_completion  # store waits locally
    assert all(e["producer_id"] != 1 or e["completion_mode"] == 0
               for e in payload["sync_edges"])
    target = Target({"kind": "cuda", "arch": "sm_90a"})
    with target, use_schedule_planner(lambda *_: plan_from_dict(payload)):
        artifact = lower(build({}, {}).prim_func, target=target)
    source = artifact.kernel_source
    assert ("tl::tma_load(" in source) == (0 in selected)
    assert ("tl::tma_store(" in source) == (1 in selected)
    if 1 in selected:
        assert "tma_store_wait" in source


def test_copy_seeds_obey_total_pool_budget(tmp_path):
    generate_plans(OPERATOR, tmp_path, {}, {}, budget=SearchBudget(max_structures=16))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["base_schedule_count"] <= 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("selected", [set(), {0}, {1}, {0, 1}])
def test_copy_backends_gpu(tmp_path, selected):
    classified, parent = candidates()
    payload = realize_copy_configuration(classified, parent, selected)
    schedule = tmp_path / "schedule.json"
    save_plan_json(schedule, plan_from_dict(payload))
    result = evaluate(OPERATOR, {}, {}, schedule, tmp_path / "source.cu", warmup=1, rep=2)
    assert result["latency_ms"] > 0


def build_loop(options, tile):
    @T.prim_func(auto_overlap=True)
    def kernel(A: T.Tensor((8, 64, 64), T.float16), B: T.Tensor((8, 64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64, 64), T.float16)
            for k in T.Pipelined(8, num_stages=2):
                T.copy(A[k, :, :], shared)
                T.copy(shared, B[k, :, :])
    return SearchWorkload(prim_func=kernel, out_idx=(1,), total_flops=8192,
                          reference_program=lambda a: a)


LOOP_OPERATOR = OperatorSpec(name="copy_loop_test", description="ring load/store",
                             add_cli_arguments=lambda p: None,
                             configuration_factory=lambda o: [{}], workload_factory=build_loop)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("selected", [set(), {0}, {1}, {0, 1}])
@pytest.mark.parametrize("num_groups", [1, 2])
def test_cross_group_store_waits_before_shared_reuse_gpu(tmp_path, selected, num_groups):
    function = layout_reduced_prim_func(build_loop({}, {}).prim_func)
    classified = HOPPER.classify(extract_fact_graph(function))
    parent = next(plan_to_dict(p) for p in enumerate_overlap_plans(
        function, reduce_ir=False,
        budget=SearchBudget(max_groups=2, max_stages=1, max_structures=32),
    ) if len(p.groups) == num_groups)
    payload = realize_copy_configuration(classified, parent, selected)
    assert payload is not None
    schedule = tmp_path / "schedule.json"
    save_plan_json(schedule, plan_from_dict(payload))
    result = evaluate(LOOP_OPERATOR, {}, {}, schedule, tmp_path / "source.cu", warmup=1, rep=2)
    assert result["latency_ms"] > 0
