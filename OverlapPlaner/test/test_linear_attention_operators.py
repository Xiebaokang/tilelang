"""OverlapPlan coverage for the migrated linear-attention examples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pytest
import tilelang
import torch

from OverlapPlaner.apply import apply_plan_to_ir
from OverlapPlaner.contract import enumerate_overlap_plans, layout_reduced_module
from OverlapPlaner.structure import SearchBudget
from OverlapPlaner.tune.operators.linear_attn_fwd import OPERATOR as LINEAR_ATTN_FWD
from OverlapPlaner.tune.operators.mamba_chunk_scan import OPERATOR as MAMBA_CHUNK_SCAN
from OverlapPlaner.tune.operators.mamba_chunk_state import OPERATOR as MAMBA_CHUNK_STATE
from OverlapPlaner.tune.run import SEARCH_OPERATORS, _supervise_native, generate_plans
from OverlapPlaner.tune.search import evaluate, evaluate_native


CASES = (
    (
        LINEAR_ATTN_FWD,
        {
            "linear_attn_batch": 1,
            "linear_attn_seq": 128,
            "linear_attn_heads": 1,
            "linear_attn_key_dim": 64,
            "linear_attn_value_dim": 64,
        },
        {"block_k": 64, "block_v": 64},
    ),
    (
        MAMBA_CHUNK_STATE,
        {
            "mamba_state_batch": 1,
            "mamba_state_heads": 4,
            "mamba_state_groups": 1,
            "mamba_state_seq": 128,
            "mamba_state_chunk": 64,
            "mamba_state_dim": 64,
            "mamba_state_dstate": 64,
        },
        {"block_m": 64, "block_n": 64, "block_k": 64},
    ),
    (
        MAMBA_CHUNK_SCAN,
        {
            "mamba_scan_batch": 1,
            "mamba_scan_heads": 4,
            "mamba_scan_groups": 1,
            "mamba_scan_seq": 128,
            "mamba_scan_chunk": 64,
            "mamba_scan_dim": 64,
            "mamba_scan_dstate": 64,
        },
        {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "block_dstate": 64,
        },
    ),
)


def _has_hopper_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.parametrize(
    "operator,options,tile", CASES, ids=lambda value: getattr(value, "name", None)
)
def test_operator_enumerates_and_lowers(operator, options, tile) -> None:
    workload = operator.build(options, tile)
    mod, target = layout_reduced_module(workload.prim_func)
    function = mod[mod.get_global_var("main")]
    plan = next(
        enumerate_overlap_plans(
            function,
            budget=SearchBudget(max_groups=1, max_stages=2, max_structures=1),
            target=target,
            reduce_ir=False,
        )
    )
    annotated = apply_plan_to_ir(mod, plan)
    with target, tilelang.transform.PassContext(config={}):
        lowered = tilelang.transform.LowerOverlapPlan()(annotated)
    lowered_function = lowered[lowered.get_global_var("main")]
    assert lowered_function.attrs.get("tl.overlap_plan") is not None
    assert int(lowered_function.attrs["tl.smem_planned_arena_bytes"]) == int(
        plan.shared_arena_bytes
    )


def test_operators_are_registered_for_search() -> None:
    registered = {operator.name for operator in SEARCH_OPERATORS}
    assert {
        "linear_attn_fwd",
        "mamba_chunk_scan",
        "mamba_chunk_state",
    } <= registered


def test_mamba_scan_default_tiles_are_overlap_plan_feasible() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    MAMBA_CHUNK_SCAN.add_arguments(parser)
    options = vars(parser.parse_args([]))
    configurations = MAMBA_CHUNK_SCAN.configurations(options)

    assert len(configurations) == 14
    assert all("num_stages" not in tile for tile in configurations)
    for tile in configurations:
        workload = MAMBA_CHUNK_SCAN.build(options, tile)
        next(
            enumerate_overlap_plans(
                workload.prim_func,
                budget=SearchBudget(max_structures=1),
            )
        )


def test_mamba_state_does_not_duplicate_tiles_by_native_pipeline_depth() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    MAMBA_CHUNK_STATE.add_arguments(parser)
    options = vars(parser.parse_args([]))
    configurations = MAMBA_CHUNK_STATE.configurations(options)

    assert "mamba_state_num_stages" not in options
    assert len(configurations) == 6
    assert all("num_stages" not in tile for tile in configurations)


@pytest.mark.parametrize(
    "operator,options,tile", CASES, ids=lambda value: getattr(value, "name", None)
)
def test_native_workload_disables_overlap_plan(operator, options, tile) -> None:
    searched = operator.build(options, tile)
    native = operator.build_native(options, tile)
    assert int(searched.prim_func.attrs["tl.auto_overlap"]) == 1
    assert native.prim_func.attrs.get("tl.auto_overlap") is None
    assert native.out_idx == searched.out_idx
    assert native.total_flops == searched.total_flops
    assert native.pass_configs == searched.pass_configs


@pytest.mark.parametrize(
    "operator,options,tile", CASES, ids=lambda value: getattr(value, "name", None)
)
def test_generate_plans(operator, options, tile, tmp_path: Path) -> None:
    plan_dir = tmp_path / operator.name / "candidates"
    generate_plans(
        operator,
        plan_dir,
        tile,
        options,
        budget=SearchBudget(max_groups=1, max_stages=2, max_structures=1),
    )
    assert (plan_dir / "schedule_00000.json").is_file()


@pytest.mark.skipif(not _has_hopper_gpu(), reason="requires a Hopper CUDA GPU")
@pytest.mark.parametrize(
    "operator,options,tile", CASES, ids=lambda value: getattr(value, "name", None)
)
def test_runtime_correctness(operator, options, tile, tmp_path: Path) -> None:
    root = tmp_path / operator.name
    generate_plans(
        operator,
        root / "candidates",
        tile,
        options,
        budget=SearchBudget(max_groups=1, max_stages=2, max_structures=1),
    )
    result = evaluate(
        operator,
        tile,
        options,
        root / "candidates" / "schedule_00000.json",
        root / "sources" / "schedule_00000.cu",
        warmup=2,
        rep=3,
    )
    assert math.isfinite(result["latency_ms"]) and result["latency_ms"] > 0
    assert result["tflops"] > 0
    assert Path(result["source_file"]).read_text(encoding="utf-8").strip()


@pytest.mark.skipif(not _has_hopper_gpu(), reason="requires a Hopper CUDA GPU")
@pytest.mark.parametrize(
    "operator,options,tile", CASES, ids=lambda value: getattr(value, "name", None)
)
def test_native_runtime_correctness(operator, options, tile, tmp_path: Path) -> None:
    source = tmp_path / operator.name / "native" / "source.cu"
    result = evaluate_native(
        operator,
        tile,
        options,
        source,
        warmup=2,
        rep=3,
    )
    assert result["backend"] == "native"
    assert math.isfinite(result["latency_ms"]) and result["latency_ms"] > 0
    assert result["tflops"] > 0
    assert source.read_text(encoding="utf-8").strip()


@pytest.mark.skipif(not _has_hopper_gpu(), reason="requires a Hopper CUDA GPU")
def test_native_supervisor_writes_replayable_artifacts(tmp_path: Path) -> None:
    operator, options, tile = CASES[0]
    config_dir = tmp_path / operator.name / "block_k64_block_v64"
    payload = _supervise_native(
        operator,
        tile,
        options,
        config_dir,
        warmup=1,
        rep=2,
        compile_timeout=30,
        execution_timeout=15,
        kill_grace=2,
    )
    assert payload["failure"] is None
    assert payload["result"]["backend"] == "native"
    persisted = config_dir / "native" / "result.json"
    result = json.loads(persisted.read_text(encoding="utf-8"))
    assert result["source_file"] == "native/source.cu"
    assert (config_dir / result["source_file"]).is_file()
