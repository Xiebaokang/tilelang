"""Structural coverage for example-derived research operators."""

from __future__ import annotations

import argparse

import pytest

from OverlapPlaner.contract import enumerate_overlap_plans
from OverlapPlaner.facts import extract_fact_graph
from OverlapPlaner.structure import SearchBudget
from OverlapPlaner.tune.operators import get_operator
from OverlapPlaner.tune.run import SEARCH_OPERATORS


NAMES = (
    "gdn_chunk_o_bwd",
    "gdn_chunk_delta_bwd",
    "kda_chunk_bwd_intra",
    "dequant_gemm_fp4",
    "kda_wy_fast_bwd",
    "fused_moe",
)


@pytest.mark.parametrize("name", NAMES)
def test_research_operator_builds_and_enumerates(name: str) -> None:
    operator = get_operator(name)
    parser = argparse.ArgumentParser(add_help=False)
    operator.add_arguments(parser)
    options = vars(parser.parse_args([]))
    tile = operator.configurations(options)[0]
    workload = operator.build(options, tile)
    assert int(workload.prim_func.attrs["tl.auto_overlap"]) == 1
    plan = next(
        enumerate_overlap_plans(
            workload.prim_func,
            budget=SearchBudget(
                max_groups=1,
                max_stages=2,
                max_structures=1,
            ),
        )
    )
    assert plan.operations


def test_research_operators_are_selectable_in_run() -> None:
    selected = {operator.name for operator in SEARCH_OPERATORS}
    assert set(NAMES) <= selected


def test_gdn_o_deduplicates_same_group_async_completion() -> None:
    operator = get_operator("gdn_chunk_o_bwd")
    parser = argparse.ArgumentParser(add_help=False)
    operator.add_arguments(parser)
    options = vars(parser.parse_args([]))
    workload = operator.build(options, {"block_dk": 32, "block_dv": 32})
    plan = next(
        enumerate_overlap_plans(
            workload.prim_func,
            budget=SearchBudget(max_groups=1, max_stages=1, max_structures=1),
        )
    )
    group_by_operation = {
        int(item.operation_id): int(item.group_id) for item in plan.operations
    }
    current = {
        (int(edge.producer_id), int(edge.consumer_id), int(edge.buffer_id))
        for edge in plan.sync_edges
        if int(edge.iteration_distance) == 0
        and group_by_operation[int(edge.producer_id)]
        == group_by_operation[int(edge.consumer_id)]
    }
    assert not any(
        int(edge.iteration_distance) > 0
        and (int(edge.producer_id), int(edge.consumer_id), int(edge.buffer_id))
        in current
        for edge in plan.sync_edges
    )


def test_default_workloads_are_production_scale() -> None:
    expected = {
        "fa3": {"fa3_seq_q": 8192, "fa3_seq_kv": 8192},
        "mla": {"mla_seq": 8192},
        "gqa": {"gqa_seq": 8192},
        "gqa_bwd": {"gqa_bwd_seq": 8192},
        "mha_bwd": {"mha_bwd_seq": 8192},
        "linear_attn_fwd": {"linear_attn_seq": 8192},
        "mamba_chunk_scan": {"mamba_scan_seq": 8192},
        "mamba_chunk_state": {"mamba_state_seq": 8192},
        "kda_wy_fast_bwd": {"kda_wy_bwd_seq": 8192},
        "gdn_chunk_o_bwd": {"gdn_o_bwd_seq": 8192},
        "gdn_chunk_delta_bwd": {"gdn_delta_bwd_seq": 8192},
        "kda_chunk_bwd_intra": {"kda_intra_seq": 8192},
        "fused_moe": {"fused_moe_tokens": 8192},
        "gemm": {"gemm_m": 4096, "gemm_n": 4096, "gemm_k": 4096},
        "gemm_fp8": {
            "gemm_fp8_m": 4096,
            "gemm_fp8_n": 4096,
            "gemm_fp8_k": 4096,
        },
        "dequant_gemm_fp4": {
            "dequant_fp4_m": 4096,
            "dequant_fp4_n": 4096,
            "dequant_fp4_k": 4096,
        },
    }
    for name, values in expected.items():
        operator = get_operator(name)
        parser = argparse.ArgumentParser(add_help=False)
        operator.add_arguments(parser)
        options = vars(parser.parse_args([]))
        assert {key: options[key] for key in values} == values


@pytest.mark.parametrize(
    "name,tile",
    (
        ("fa3", {"block_m": 128, "block_n": 128}),
        ("mla", {"block_h": 64, "block_n": 64}),
        ("gemm", {"block_m": 128, "block_n": 256, "block_k": 64}),
        ("gemm_fp8", {"block_m": 128, "block_n": 128, "block_k": 128}),
        ("dequant_gemm_fp4", {"block_m": 128, "block_n": 128, "block_k": 256}),
        ("convolution", {"block_m": 128, "block_n": 256, "block_k": 64}),
        ("mamba_chunk_scan", {"block_m": 64, "block_n": 64, "block_k": 64, "block_dstate": 128}),
        ("mamba_chunk_state", {"block_m": 64, "block_n": 128, "block_k": 64}),
        ("kda_wy_fast_bwd", {"block_dk": 32, "block_dv": 32}),
        ("gdn_chunk_o_bwd", {"block_dk": 32, "block_dv": 128}),
        ("gdn_chunk_delta_bwd", {"block_dv": 128}),
        ("kda_chunk_bwd_intra", {"block_dk": 128}),
    ),
)
def test_default_tiles_cover_example_best_known_shapes(name, tile) -> None:
    operator = get_operator(name)
    parser = argparse.ArgumentParser(add_help=False)
    operator.add_arguments(parser)
    options = vars(parser.parse_args([]))
    assert tile in operator.configurations(options)


@pytest.mark.parametrize(
    "name,tile,threads",
    (
        ("fa3", {"block_m": 64, "block_n": 128}, 128),
        ("fa3", {"block_m": 128, "block_n": 64}, 256),
        ("mla", {"block_h": 32, "block_n": 64}, 128),
        ("mla", {"block_h": 64, "block_n": 64}, 256),
        (
            "dequant_gemm_fp4",
            {"block_m": 64, "block_n": 128, "block_k": 256},
            256,
        ),
        ("kda_wy_fast_bwd", {"block_dk": 32, "block_dv": 128}, 256),
        ("gdn_chunk_o_bwd", {"block_dk": 32, "block_dv": 32}, 128),
        ("gdn_chunk_delta_bwd", {"block_dv": 128}, 256),
        ("kda_chunk_bwd_intra", {"block_dk": 128}, 256),
        (
            "mamba_chunk_scan",
            {
                "block_m": 256,
                "block_n": 32,
                "block_k": 128,
                "block_dstate": 128,
            },
            512,
        ),
        (
            "mamba_chunk_state",
            {"block_m": 64, "block_n": 128, "block_k": 64},
            256,
        ),
        ("linear_attn_fwd", {"block_k": 128, "block_v": 64}, 256),
        (
            "fused_moe",
            {"block_token": 64, "block_hidden": 128, "block_expert": 128},
            128,
        ),
    ),
)
def test_thread_count_is_derived_from_tile(name, tile, threads) -> None:
    operator = get_operator(name)
    parser = argparse.ArgumentParser(add_help=False)
    operator.add_arguments(parser)
    options = vars(parser.parse_args([]))
    graph = extract_fact_graph(operator.build(options, tile).prim_func)
    assert graph.kernel_threads == threads
