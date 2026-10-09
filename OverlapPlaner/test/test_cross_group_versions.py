"""Single-slot shared handoffs retain reuse protection and run-ahead options."""

import random

import pytest
import tilelang

from OverlapPlaner.apply import apply_plan_to_ir
from OverlapPlaner.arch import HOPPER
from OverlapPlaner.contract import enumerate_overlap_plans, layout_reduced_module
from OverlapPlaner.facts import OpKind, RegionKind, extract_fact_graph
from OverlapPlaner.structure import SearchBudget
from OverlapPlaner.structure.enumerate import _version_variants
from OverlapPlaner.structure.model import SynchronizationKind
from OverlapPlaner.structure.order import build_program_orders
from OverlapPlaner.structure.sync import build_synchronizations
from OverlapPlaner.structure.version import analyze_buffer_versions, cross_group_version_buffers
from OverlapPlaner.tune.operators.gemm import build as build_gemm
from OverlapPlaner.tune.global_search import sample_structure
from OverlapPlaner.tune.joint_search import adjacent_joint_moves, realize_joint_move
from OverlapPlaner.serialization import plan_to_dict


@pytest.fixture
def gemm():
    return build_gemm(
        {"gemm_m": 256, "gemm_n": 256, "gemm_k": 256},
        {"block_m": 128, "block_n": 128, "block_k": 64},
    ).prim_func


def shared_handoff(classified, consumer_stage=0):
    graph = classified.graph
    groups = {node.node_id: int(node.kind == OpKind.GEMM) for node in graph.nodes}
    stages = {
        region.region_id: {
            node.node_id: consumer_stage if node.kind == OpKind.GEMM else 0
            for node in graph.nodes_for_region(region.region_id)
        }
        for region in graph.regions if region.kind == RegionKind.PIPELINE
    }
    orders = build_program_orders(classified, stages, groups)
    versions = analyze_buffer_versions(graph, stages, groups, orders)
    return stages, groups, orders, versions


@pytest.mark.parametrize("stage,minimum", [(0, 1), (1, 1), (2, 2)])
def test_cross_group_shared_minimum_and_reuse_protection(gemm, stage, minimum):
    classified = HOPPER.classify(extract_fact_graph(gemm))
    stages, groups, orders, versions = shared_handoff(classified, stage)
    buffers = cross_group_version_buffers(classified.graph, groups)
    assert len(buffers) == 2
    syncs = build_synchronizations(classified, stages, groups, orders, versions)
    for buffer_id in buffers:
        assert versions[buffer_id] == minimum
        forward = next(s for s in syncs if s.buffer_id == buffer_id
                       and s.kind == SynchronizationKind.FORWARD_DEPENDENCY)
        reuse = next(s for s in syncs if s.buffer_id == buffer_id
                     and s.kind == SynchronizationKind.BUFFER_REUSE)
        assert forward.slot_count == reuse.slot_count == minimum
        assert reuse.iteration_distance == minimum
        assert reuse.effective_stage_distance == minimum - stage
        assert reuse.producer_id == forward.consumer_id
        assert reuse.consumer_id == forward.producer_id


@pytest.mark.parametrize("extra,counts", [(0, {1}), (1, {1, 2, 3}), (2, {1, 2, 3, 4})])
def test_cross_group_version_search_retains_run_ahead(gemm, extra, counts):
    classified = HOPPER.classify(extract_fact_graph(gemm))
    _, groups, _, minimum = shared_handoff(classified)
    buffers = cross_group_version_buffers(classified.graph, groups)
    variants = list(_version_variants(
        classified, groups, minimum,
        SearchBudget(extra_shared_versions=extra, max_version_variants=32),
    ))
    assert variants[0] == minimum
    assert len({tuple(sorted(v.items())) for v in variants}) == len(variants)
    for buffer_id in buffers:
        assert {v[buffer_id] for v in variants} == counts


def test_single_slot_cross_group_plan_lowers(gemm):
    mod, target = layout_reduced_module(gemm)
    function = mod[mod.get_global_var("main")]
    plan = next(
        plan for plan in enumerate_overlap_plans(
            function, target=target, reduce_ir=False,
            budget=SearchBudget(max_groups=2, max_stages=1, max_structures=128,
                                extra_shared_versions=0),
        )
        if len(plan.groups) == 2
        and all(int(buffer.version_count) == 1 for buffer in plan.buffers)
    )
    pipeline_syncs = [edge for edge in plan.sync_edges if int(edge.scope) == 1]
    assert pipeline_syncs
    assert all(int(edge.slot_count) == 1 for edge in pipeline_syncs)
    assert any(int(edge.kind) == 1 and int(edge.iteration_distance) == 1
               for edge in pipeline_syncs)
    with target, tilelang.transform.PassContext(config={}):
        lowered = tilelang.transform.LowerOverlapPlan()(apply_plan_to_ir(mod, plan))
    assert "overlap_plan_mbar" in lowered.script()


@pytest.mark.parametrize("extra,counts", [(0, {1, 2}), (1, {1, 2, 3}), (2, {1, 2, 3, 4})])
def test_global_sampling_covers_source_version_choices(gemm, extra, counts):
    classified = HOPPER.classify(extract_fact_graph(gemm))
    budget = SearchBudget(max_groups=2, max_stages=1, extra_shared_versions=extra)
    sampled = set()
    for seed in range(128):
        structure = sample_structure(classified, budget, random.Random(seed))
        if structure is None or structure.num_groups != 2:
            continue
        for buffer_id in cross_group_version_buffers(classified.graph, structure.groups):
            sampled.add(structure.buffer_versions[buffer_id])
    assert sampled == counts


@pytest.mark.parametrize("extra,upper", [(0, 2), (1, 3), (2, 4)])
def test_local_version_moves_walk_full_range_and_rebuild_sync(gemm, extra, upper):
    mod, target = layout_reduced_module(gemm)
    function = mod[mod.get_global_var("main")]
    classified = HOPPER.classify(extract_fact_graph(function))
    plan = next(
        p for p in enumerate_overlap_plans(
            function, target=target, reduce_ir=False,
            budget=SearchBudget(max_groups=2, max_stages=1, max_structures=128,
                                extra_shared_versions=0),
        ) if len(p.groups) == 2
        and all(int(b.version_count) == 1 for b in p.buffers)
    )
    payload = plan_to_dict(plan)
    _, groups, _, _ = shared_handoff(classified)
    buffer_id = cross_group_version_buffers(classified.graph, groups)[0]
    for current in list(range(1, upper + 1)) + list(range(upper - 1, 0, -1)):
        if payload["buffers"][buffer_id]["version_count"] != current:
            payload = realize_joint_move(classified, payload, ("version", buffer_id, current, 0, 0))
            assert payload is not None
        destinations = {
            move[2] for move in adjacent_joint_moves(classified, payload, extra_shared_versions=extra)
            if move[:2] == ("version", buffer_id)
        }
        assert destinations == {v for v in (current - 1, current + 1) if 1 <= v <= upper}
        channels = [s for s in payload["sync_edges"] if s["buffer_id"] == buffer_id]
        assert channels and all(s["slot_count"] == current for s in channels)
