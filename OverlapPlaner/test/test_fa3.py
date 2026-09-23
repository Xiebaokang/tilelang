"""Native FA3 compile path for OverlapPlan."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import tilelang
import torch

from OverlapPlaner.apply import apply_plan_to_ir
from OverlapPlaner.arch.hopper import HOPPER_CUDA_TARGET
from OverlapPlaner.contract import enumerate_overlap_plans, layout_reduced_module
from OverlapPlaner.planner import use_schedule_planner
from OverlapPlaner.serialization import load_plan_json, plan_from_dict, plan_to_dict
from OverlapPlaner.structure import SearchBudget
from OverlapPlaner.tune.operators.fa3 import OPERATOR as FA3
from OverlapPlaner.tune.operators.fa3 import build as build_fa3
from OverlapPlaner.tune.run import SEARCH_OPERATORS, generate_plans
from OverlapPlaner.tune.search import evaluate
from tilelang.engine.lower import lower


def _has_hopper_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


_NATIVE_FA3_OPTIONS = {
    "fa3_batch": 1,
    "fa3_heads": 1,
    "fa3_seq_q": 256,
    "fa3_seq_kv": 256,
    "fa3_dim": 128,
    "fa3_causal": False,
}
_NATIVE_FA3_TILE = {"block_m": 128, "block_n": 128}
_NATIVE_FA3_BUDGET = SearchBudget(max_groups=2, max_structures=4)


def _native_fa3_workload():
    return build_fa3(_NATIVE_FA3_OPTIONS, _NATIVE_FA3_TILE)


def _lower_native_plan(prim, plan):
    mod, target = layout_reduced_module(prim)
    annotated = apply_plan_to_ir(mod, plan)
    with target, tilelang.transform.PassContext(config={}):
        return tilelang.transform.LowerOverlapPlan()(annotated), target


def test_native_fa3_enumerates_and_lowers() -> None:
    workload = _native_fa3_workload()
    mod, target = layout_reduced_module(workload.prim_func)
    function = mod[mod.get_global_var("main")]
    plan = next(
        enumerate_overlap_plans(
            function,
            budget=_NATIVE_FA3_BUDGET,
            target=target,
            reduce_ir=False,
        )
    )
    assert plan.operations
    assert plan.groups
    annotated = apply_plan_to_ir(mod, plan)
    with target, tilelang.transform.PassContext(config={}):
        lowered = tilelang.transform.LowerOverlapPlan()(annotated)
    function = lowered[lowered.get_global_var("main")]
    assert function.attrs.get("tl.overlap_plan") is not None
    assert int(function.attrs["tl.smem_planned_arena_bytes"]) == int(
        plan.shared_arena_bytes
    )


def test_native_fa3_handle_free_plan_lowers() -> None:
    workload = _native_fa3_workload()
    plan = plan_from_dict(
        plan_to_dict(
            next(
                enumerate_overlap_plans(
                    workload.prim_func, budget=_NATIVE_FA3_BUDGET
                )
            )
        )
    )
    assert plan.operations[0].statement is None
    lowered, _ = _lower_native_plan(workload.prim_func, plan)
    function = lowered[lowered.get_global_var("main")]
    assert int(function.attrs["tl.smem_planned_arena_bytes"]) == int(
        plan.shared_arena_bytes
    )


def test_fa3_serial_shared_output_store_split_reaches_cuda_source() -> None:
    workload = _native_fa3_workload()
    plan = next(
        plan
        for plan in enumerate_overlap_plans(
            workload.prim_func,
            budget=SearchBudget(max_groups=2, max_stages=1, max_structures=32),
        )
        if int(plan.operations[20].group_id)
        != int(plan.operations[21].group_id)
    )
    assert any(
        int(edge.producer_id) == 20 and int(edge.consumer_id) == 21
        for edge in plan.sync_edges
    )
    detached = plan_from_dict(plan_to_dict(plan))
    with use_schedule_planner(lambda *_: detached), HOPPER_CUDA_TARGET:
        artifact = lower(workload.prim_func, target=HOPPER_CUDA_TARGET)
    assert "overlap_plan_mbar" in artifact.kernel_source
    assert "tl::tma_store(" in artifact.kernel_source


def test_search_operators_includes_fa3() -> None:
    assert any(operator.name == "fa3" for operator in SEARCH_OPERATORS)


def test_native_fa3_generate_plans(tmp_path: Path) -> None:
    plan_dir = tmp_path / "candidates"
    generate_plans(
        FA3,
        plan_dir,
        _NATIVE_FA3_TILE,
        _NATIVE_FA3_OPTIONS,
        budget=SearchBudget(max_groups=1, max_structures=2),
    )
    files = sorted(plan_dir.glob("schedule_*.json"))
    assert files
    plan = load_plan_json(files[0])
    assert plan.operations
    assert plan.groups
    manifest = json.loads((plan_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["operator"] == "fa3"
    assert manifest["schedule_count"] >= 1


@pytest.mark.skipif(not _has_hopper_gpu(), reason="requires a Hopper CUDA GPU")
def test_native_fa3_runtime_correctness(tmp_path: Path) -> None:
    plan_dir = tmp_path / "candidates"
    generate_plans(
        FA3,
        plan_dir,
        _NATIVE_FA3_TILE,
        _NATIVE_FA3_OPTIONS,
        budget=SearchBudget(max_groups=1, max_stages=1, max_structures=1),
    )
    schedule_path = next(iter(sorted(plan_dir.glob("schedule_*.json"))))
    result = evaluate(
        FA3,
        _NATIVE_FA3_TILE,
        _NATIVE_FA3_OPTIONS,
        schedule_path,
        tmp_path / "sources" / "schedule_00000.cu",
        warmup=10,
        rep=10,
    )
    assert result["tflops"] > 0
    assert math.isfinite(result["latency_ms"]) and result["latency_ms"] > 0
    source = Path(result["source_file"]).read_text(encoding="utf-8")
    assert source.strip()
    print(
        "\nFA3 native OverlapPlan correctness: PASS, "
        f"latency={result['latency_ms']:.6f} ms, "
        f"throughput={result['tflops']:.3f} TFLOPS"
    )
