"""Native MLA compile and runtime path for OverlapPlan."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import tilelang
import torch

from OverlapPlaner.apply import apply_plan_to_ir
from OverlapPlaner.arch import HOPPER
from OverlapPlaner.contract import (
    enumerate_overlap_plans,
    layout_reduced_module,
    to_overlap_plan,
)
from OverlapPlaner.facts import extract_fact_graph
from OverlapPlaner.physical import estimate_group_registers_per_thread
from OverlapPlaner.serialization import load_plan_json, plan_from_dict, plan_to_dict
from OverlapPlaner.structure import SearchBudget
from OverlapPlaner.structure.group import enumerate_group_assignments
from OverlapPlaner.structure.model import Structure
from OverlapPlaner.structure.order import build_program_orders
from OverlapPlaner.structure.stage import enumerate_program_stages
from OverlapPlaner.structure.sync import build_synchronizations
from OverlapPlaner.structure.version import analyze_buffer_versions
from OverlapPlaner.tune.dynamic import classified_plan_features, producer_copy_count
from OverlapPlaner.tune.order_search import realize_order_swap
from OverlapPlaner.tune.operators.gemm import build as build_gemm
from OverlapPlaner.tune.operators.mla import OPERATOR as MLA
from OverlapPlaner.tune.operators.mla import build as build_mla
from OverlapPlaner.tune.run import SEARCH_OPERATORS, generate_plans
from OverlapPlaner.tune.search import evaluate


def _has_hopper_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


_NATIVE_MLA_OPTIONS = {
    "mla_batch": 1,
    "mla_heads": 64,
    "mla_kv_heads": 1,
    "mla_seq": 256,
    "mla_dim": 512,
    "mla_pe_dim": 64,
}
_NATIVE_MLA_TILE = {"block_h": 64, "block_n": 64}
_NATIVE_MLA_BUDGET = SearchBudget(max_groups=2, max_stages=2, max_structures=8)


def _native_mla_workload():
    return build_mla(_NATIVE_MLA_OPTIONS, _NATIVE_MLA_TILE)


def test_native_mla_enumerates_fullcol_plan_and_lowers() -> None:
    workload = _native_mla_workload()
    mod, target = layout_reduced_module(workload.prim_func)
    function = mod[mod.get_global_var("main")]
    plans = list(
        enumerate_overlap_plans(
            function,
            budget=_NATIVE_MLA_BUDGET,
            target=target,
            reduce_ir=False,
        )
    )
    assert plans
    one_group = [plan for plan in plans if len(plan.groups) == 1]
    assert one_group
    assert all(int(plan.groups[0].warp_count) == 8 for plan in one_group)
    completion_transactions = {}
    for edge in one_group[0].sync_edges:
        if int(edge.completion_mode) != 1:
            continue
        key = (int(edge.producer_id), int(edge.buffer_id),
               int(one_group[0].operations[int(edge.consumer_id)].group_id))
        completion_transactions.setdefault(key, []).append(edge)
    assert completion_transactions
    assert all(len(edges) == 1 for edges in completion_transactions.values())
    annotated = apply_plan_to_ir(mod, one_group[0])
    with target, tilelang.transform.PassContext(config={}):
        lowered = tilelang.transform.LowerOverlapPlan()(annotated)
    lowered_function = lowered[lowered.get_global_var("main")]
    assert int(lowered_function.attrs["tl.smem_planned_arena_bytes"]) == int(
        one_group[0].shared_arena_bytes
    )


def test_cross_group_shared_buffer_can_use_three_slots_without_stage_cut() -> None:
    workload = build_gemm(
        {"gemm_m": 256, "gemm_n": 256, "gemm_k": 256},
        {"block_m": 128, "block_n": 128, "block_k": 64},
    )
    mod, target = layout_reduced_module(workload.prim_func)
    function = mod[mod.get_global_var("main")]
    classified = HOPPER.classify(extract_fact_graph(function))
    shared_id = next(
        buffer.buffer_id
        for buffer in classified.graph.buffers
        if buffer.name == "b_shared"
    )
    pairs = {}
    for plan in enumerate_overlap_plans(
        function,
        budget=SearchBudget(max_structures=20),
        target=target,
        reduce_ir=False,
    ):
        if len(plan.groups) != 2:
            continue
        key = tuple(
            (
                int(op.group_id),
                None if op.stage is None else int(op.stage),
                int(op.order),
            )
            for op in plan.operations
        )
        pairs.setdefault(key, {})[int(plan.buffers[shared_id].version_count)] = plan
    base, expanded = next(
        (variants[2], variants[3])
        for variants in pairs.values()
        if 2 in variants and 3 in variants
    )
    b_writer = next(
        node.node_id
        for node in classified.graph.nodes
        if shared_id in node.writes
    )
    b_reader = next(
        node.node_id
        for node in classified.graph.nodes
        if shared_id in node.reads
    )
    assert int(expanded.operations[b_writer].group_id) != int(
        expanded.operations[b_reader].group_id
    )
    assert int(expanded.operations[b_writer].stage) == int(
        expanded.operations[b_reader].stage
    )
    assert base.sync_edges != expanded.sync_edges
    assert int(expanded.shared_arena_bytes) > int(base.shared_arena_bytes)
    with target, tilelang.transform.PassContext(config={}):
        lowered = tilelang.transform.LowerOverlapPlan()(
            apply_plan_to_ir(mod, expanded)
        )
    assert lowered[lowered.get_global_var("main")]


def test_mla_native_like_producer_partition_survives_resource_filter() -> None:
    mod, _ = layout_reduced_module(_native_mla_workload().prim_func)
    classified = HOPPER.classify(
        extract_fact_graph(mod[mod.get_global_var("main")])
    )
    groups = next(
        item
        for item in enumerate_group_assignments(classified, 2)
        if item[5] == item[6] != item[7] == item[8] == item[19]
        and item[2] == item[7]
        and item[3] == item[7]
        and item[4] == item[7]
    )
    stages = next(enumerate_program_stages(classified, SearchBudget()))
    orders = build_program_orders(classified, stages, groups)
    versions = analyze_buffer_versions(
        classified.graph, stages, groups, orders
    )
    structure = Structure(
        stages,
        groups,
        orders,
        versions,
        build_synchronizations(
            classified, stages, groups, orders, versions
        ),
    )
    physical = next(HOPPER.realize(classified, structure))
    plan = to_overlap_plan(classified, physical)
    offsets = {
        buffer.name: int(buffer_plan.byte_offset)
        for buffer, buffer_plan in zip(classified.graph.buffers, plan.buffers)
        if buffer_plan.byte_offset is not None
    }
    assert offsets["Q_shared"] == offsets["O_shared"]
    assert [int(group.warp_count) for group in plan.groups] == [8, 4]
    assert physical.shared_memory.shared_buffer_bytes < (
        physical.shared_memory.merged_shared_bytes
    )
    assert int(plan.shared_arena_bytes) == (
        physical.shared_memory.shared_buffer_bytes
    )
    assert producer_copy_count(plan_to_dict(plan), classified) == 2

    # Splitting the two producer copies into separate four-warp groups raises
    # the CTA width to 512.  The compute group's explicit fragments alone
    # exceed the uniform register ceiling imposed by that launch bound.
    three_groups = {**groups, 6: 2}
    three_orders = build_program_orders(classified, stages, three_groups)
    three_versions = analyze_buffer_versions(
        classified.graph, stages, three_groups, three_orders
    )
    compute_registers = estimate_group_registers_per_thread(
        classified.graph,
        0,
        three_groups,
        three_orders,
        three_versions,
        8,
        HOPPER.resource(),
    )
    assert compute_registers > (
        HOPPER.resource().register_file_capacity // 512
    )
    three_structure = Structure(
        stages,
        three_groups,
        three_orders,
        three_versions,
        build_synchronizations(
            classified, stages, three_groups, three_orders, three_versions
        ),
    )
    assert not list(HOPPER.realize(classified, three_structure))


def test_mla_all_tma_loads_share_one_producer_partition() -> None:
    mod, _ = layout_reduced_module(_native_mla_workload().prim_func)
    classified = HOPPER.classify(
        extract_fact_graph(mod[mod.get_global_var("main")])
    )
    graph = classified.graph
    producer_names = {
        "copy_Q_to_Q_shared",
        "copy_Q_pe_to_Q_pe_shared",
        "copy_KV_to_KV_shared",
        "copy_K_pe_to_K_pe_shared",
    }
    groups = {
        node.node_id: int(node.name in producer_names) for node in graph.nodes
    }
    stages = next(enumerate_program_stages(classified, SearchBudget()))
    orders = build_program_orders(classified, stages, groups)
    versions = analyze_buffer_versions(graph, stages, groups, orders)
    structure = Structure(
        stages,
        groups,
        orders,
        versions,
        build_synchronizations(
            classified, stages, groups, orders, versions
        ),
    )
    physical = next(HOPPER.realize(classified, structure))
    offsets = {
        placed.name: placed.byte_offset
        for placed in physical.shared_memory.shared_allocations
    }
    assert offsets["Q_shared"] == offsets["O_shared"]
    assert physical.shared_memory.fits
    assert [
        item.warp_count for item in physical.warp_allocation.groups
    ] == [8, 4]


def test_mla_default_pool_contains_native_stage_group_order_and_warps() -> None:
    """The search cutoff must retain the native WS execution arrangement."""

    mod, target = layout_reduced_module(_native_mla_workload().prim_func)
    function = mod[mod.get_global_var("main")]
    plans = enumerate_overlap_plans(
        function,
        budget=SearchBudget(max_structures=128),
        target=target,
        reduce_ir=False,
    )
    for plan in plans:
        if len(plan.groups) != 2:
            continue
        if [(int(group.warp_count), int(group.register_count))
            for group in plan.groups] != [(4, 24), (8, 240)]:
            continue
        operations = {int(op.operation_id): op for op in plan.operations}
        if any(int(operations[node_id].group_id) != 0 for node_id in (5, 6)):
            continue
        if any(int(operations[node_id].group_id) != 1
               for node_id in (7, 8, 15, 16, 17, 18, 19)):
            continue
        if any(int(operations[node_id].stage) != 0
               for node_id in (5, 6, 7, 8, 15, 16, 17, 18, 19)):
            continue
        if not all(
            int(operations[left].order) < int(operations[right].order)
            for left, right in ((5, 6), (15, 16), (16, 17), (17, 18), (18, 19))
        ):
            continue
        assert int(plan.buffers[8].version_count) == 2
        assert int(plan.buffers[10].version_count) == 2
        assert sum(int(edge.slot_count) for edge in plan.sync_edges) == 10
        annotated = apply_plan_to_ir(mod, plan)
        with target, tilelang.transform.PassContext(config={}):
            lowered = tilelang.transform.LowerOverlapPlan()(annotated)
        assert lowered[lowered.get_global_var("main")]
        classified = HOPPER.classify(extract_fact_graph(function))
        neighbor = realize_order_swap(
            classified, plan_to_dict(plan), (1, 1, 15, 16)
        )
        assert neighbor is not None
        assert classified_plan_features(neighbor, classified) != (
            classified_plan_features(plan_to_dict(plan), classified)
        )
        with target, tilelang.transform.PassContext(config={}):
            neighbor_ir = tilelang.transform.LowerOverlapPlan()(
                apply_plan_to_ir(mod, plan_from_dict(neighbor))
            )
        assert neighbor_ir[neighbor_ir.get_global_var("main")]
        break
    else:
        pytest.fail("default MLA pool omitted the native WS arrangement")


def test_search_operators_includes_mla() -> None:
    assert any(operator.name == "mla" for operator in SEARCH_OPERATORS)


def test_native_mla_generate_plans(tmp_path: Path) -> None:
    plan_dir = tmp_path / "candidates"
    generate_plans(
        MLA,
        plan_dir,
        _NATIVE_MLA_TILE,
        _NATIVE_MLA_OPTIONS,
        budget=SearchBudget(max_groups=1, max_stages=2, max_structures=4),
    )
    files = sorted(plan_dir.glob("schedule_*.json"))
    assert files
    plan = load_plan_json(files[0])
    assert len(plan.groups) == 1
    assert int(plan.groups[0].warp_count) == 8
    manifest = json.loads((plan_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["operator"] == "mla"
    assert manifest["schedule_count"] >= 1


@pytest.mark.skipif(not _has_hopper_gpu(), reason="requires a Hopper CUDA GPU")
def test_native_mla_runtime_correctness(tmp_path: Path) -> None:
    plan_dir = tmp_path / "candidates"
    generate_plans(
        MLA,
        plan_dir,
        _NATIVE_MLA_TILE,
        _NATIVE_MLA_OPTIONS,
        budget=SearchBudget(max_groups=1, max_stages=3, max_structures=32),
    )
    schedules = tuple(sorted(plan_dir.glob("schedule_*.json")))
    plans = tuple((path, load_plan_json(path)) for path in schedules)
    schedule_path, plan = max(plans, key=lambda item: item[1].shared_arena_bytes)
    # This plan leaves only 2 KiB below Hopper's dynamic-SMEM limit. Reduction
    # workspaces introduced after planning must therefore be liveness-packed;
    # concatenating all eight 1 KiB buffers makes compilation fail.
    assert plan.shared_arena_bytes == 230400
    result = evaluate(
        MLA,
        _NATIVE_MLA_TILE,
        _NATIVE_MLA_OPTIONS,
        schedule_path,
        tmp_path / "sources" / "schedule_00000.cu",
        warmup=5,
        rep=5,
    )
    assert result["tflops"] > 0
    assert math.isfinite(result["latency_ms"]) and result["latency_ms"] > 0
    assert Path(result["source_file"]).read_text(encoding="utf-8").strip()
