"""Tests for L3 PhysicalPlan → OverlapPlan conversion."""

from pathlib import Path

import tilelang

import tilelang.language as T

from OverlapPlaner.apply import apply_plan_to_ir
from OverlapPlaner.arch import HOPPER, ResourceKind
from OverlapPlaner.arch.hopper import HOPPER_CUDA_TARGET
from OverlapPlaner.contract import (
    enumerate_overlap_plans,
    layout_reduced_module,
    to_overlap_plan,
)
from OverlapPlaner.facts import DependencyKind, OpKind, RegionKind, extract_fact_graph
from OverlapPlaner.planner import use_schedule_planner
from OverlapPlaner.serialization import load_plan_json, plan_from_dict, plan_to_dict
from OverlapPlaner.structure import SearchBudget, enumerate_structures
from OverlapPlaner.structure.model import Structure
from OverlapPlaner.structure.order import build_program_orders
from OverlapPlaner.structure.stage import enumerate_program_stages
from OverlapPlaner.structure.sync import build_synchronizations
from OverlapPlaner.structure.version import analyze_buffer_versions
from OverlapPlaner.tune.operators.fa3 import build as build_fa3
from OverlapPlaner.tune.operators.gemm import OPERATOR as GEMM
from OverlapPlaner.tune.operators.gemm import build as build_gemm
from OverlapPlaner.tune.operators.mamba_chunk_scan import build as build_mamba_scan
from OverlapPlaner.tune.run import generate_plans
from tilelang.engine.lower import lower


def _gemm_prim():
    return build_gemm(
        {"gemm_m": 256, "gemm_n": 256, "gemm_k": 256},
        {"block_m": 128, "block_n": 128, "block_k": 64},
    ).prim_func


def _fa3_prim():
    return build_fa3(
        {
            "fa3_batch": 1,
            "fa3_heads": 1,
            "fa3_seq_q": 256,
            "fa3_seq_kv": 256,
            "fa3_dim": 128,
            "fa3_causal": False,
        },
        {"block_m": 128, "block_n": 128},
    ).prim_func


@T.prim_func(auto_overlap=True)
def _shared_waw_prim(
    A: T.Tensor((128, 128), T.float16),
    B: T.Tensor((128, 128), T.float16),
    C: T.Tensor((128, 128), T.float16),
):
    with T.Kernel(1, threads=128):
        a_shared = T.alloc_shared((128, 64), T.float16)
        b_shared = T.alloc_shared((64, 128), T.float16)
        c_local = T.alloc_fragment((128, 128), T.float32)
        T.clear(c_local)
        T.copy(A[0, 0], a_shared)
        for k in T.Pipelined(2, num_stages=2):
            T.copy(B[0, k * 64], a_shared)
            T.copy(B[k * 64, 0], b_shared)
            T.gemm(a_shared, b_shared, c_local)
        T.copy(c_local, C)


def test_shared_waw_can_split_groups_and_lower() -> None:
    mod, _ = layout_reduced_module(_shared_waw_prim)
    classified = HOPPER.classify(extract_fact_graph(mod[mod.get_global_var("main")]))
    graph = classified.graph
    waw = next(
        edge
        for edge in graph.edges
        if edge.dependency_kinds == frozenset({DependencyKind.WAW})
        and graph.buffer_for_id(edge.buffer_id).scope.startswith("shared")
    )
    groups = {
        node.node_id: (1 if node.node_id == waw.producer_id else 0)
        for node in graph.nodes
    }
    stages = next(enumerate_program_stages(classified, SearchBudget(max_stages=1)))
    orders = build_program_orders(classified, stages, groups)
    versions = analyze_buffer_versions(graph, stages, groups, orders)
    sync = build_synchronizations(classified, stages, groups, orders, versions)
    structure = Structure(stages, groups, orders, versions, sync)
    physical = next(HOPPER.realize(classified, structure))
    plan = to_overlap_plan(classified, physical)
    assert any(
        int(edge.producer_id) == waw.producer_id
        and int(edge.consumer_id) == waw.consumer_id
        and int(edge.dependency_kind) == 4
        and int(edge.completion_mode) == 1
        for edge in plan.sync_edges
    )
    detached = plan_from_dict(plan_to_dict(plan))
    with (
        use_schedule_planner(lambda *_: detached),
        HOPPER_CUDA_TARGET,
        tilelang.transform.PassContext(opt_level=3),
    ):
        artifact = lower(_shared_waw_prim, target=HOPPER_CUDA_TARGET)
    assert "overlap_plan_mbar" in artifact.kernel_source


def test_mamba_three_stage_serial_shared_group_handoffs_lower() -> None:
    workload = build_mamba_scan(
        {
            "mamba_scan_batch": 1,
            "mamba_scan_heads": 16,
            "mamba_scan_groups": 2,
            "mamba_scan_seq": 2048,
            "mamba_scan_chunk": 256,
            "mamba_scan_dim": 64,
            "mamba_scan_dstate": 128,
        },
        {"block_m": 64, "block_n": 64, "block_k": 64, "block_dstate": 128},
    )
    mod, target = layout_reduced_module(workload.prim_func)
    classified = HOPPER.classify(extract_fact_graph(mod[mod.get_global_var("main")]))
    stages = {
        1: {
            8: 0, 9: 1, 10: 0, 11: 1, 12: 2, 13: 0,
            14: 1, 15: 2, 16: 2, 17: 1, 18: 2,
        }
    }
    producer_nodes = {0, 4, 5, 8, 10, 13, 17, 20, 24}
    groups = {
        node.node_id: int(node.node_id in producer_nodes)
        for node in classified.graph.nodes
    }
    orders = build_program_orders(classified, stages, groups)
    versions = analyze_buffer_versions(classified.graph, stages, groups, orders)
    synchronizations = build_synchronizations(
        classified, stages, groups, orders, versions
    )
    structure = Structure(stages, groups, orders, versions, synchronizations)
    plan = to_overlap_plan(classified, next(HOPPER.realize(classified, structure)))
    detached = plan_from_dict(plan_to_dict(plan))
    with use_schedule_planner(lambda *_: detached), target:
        artifact = lower(workload.prim_func, target=target)
    assert "warpgroup_reg_alloc" in artifact.kernel_source
    assert "overlap_plan_mbar" in artifact.kernel_source


def _physical(prim, **budget):
    classified = HOPPER.classify(extract_fact_graph(prim))
    structure = next(
        enumerate_structures(classified, SearchBudget(**budget))
    )
    physical = next(HOPPER.realize(classified, structure))
    return classified, physical


def test_overlap_plan_stamps_statement_and_buffer_identity() -> None:
    classified, physical = _physical(_gemm_prim(), max_groups=1, max_structures=1)
    plan = to_overlap_plan(classified, physical)
    graph = classified.graph
    assert len(plan.operations) == len(graph.nodes)
    assert len(plan.buffers) == len(graph.buffers)
    assert len(plan.groups) == physical.structure.num_groups
    for node, operation in zip(graph.nodes, plan.operations):
        assert int(operation.operation_id) == node.node_id
        assert operation.statement.same_as(node.statement)
        assert int(operation.group_id) == physical.structure.groups[node.node_id]
        if graph.region_for_id(node.region_id).kind == RegionKind.SERIAL:
            assert operation.stage is None
        else:
            region_stages = physical.structure.stages_by_region[node.region_id]
            assert int(operation.stage) == region_stages[node.node_id]
    for buffer, buffer_plan in zip(graph.buffers, plan.buffers):
        assert int(buffer_plan.buffer_id) == buffer.buffer_id
        assert buffer_plan.buffer.same_as(buffer.buffer)
        if buffer.scope.startswith("shared") and "tmem" not in buffer.scope:
            assert buffer_plan.byte_offset is not None
        else:
            assert buffer_plan.byte_offset is None
    assert int(plan.shared_arena_bytes) == physical.shared_memory.shared_buffer_bytes


def test_async_in_loop_copy_uses_transaction_completion() -> None:
    classified, physical = _physical(_gemm_prim(), max_groups=1, max_structures=4)
    plan = to_overlap_plan(classified, physical)
    graph = classified.graph
    async_copies = [
        node.node_id
        for node in graph.nodes
        if node.kind == OpKind.COPY
        and classified.traits_for(node.node_id).async_completion
        and classified.traits_for(node.node_id).kind == ResourceKind.MEMORY
    ]
    assert async_copies
    modes = {
        int(edge.producer_id): int(edge.completion_mode) for edge in plan.sync_edges
    }
    assert any(modes.get(node_id) == 1 for node_id in async_copies)


def test_json_round_trip_drops_handles() -> None:
    classified, physical = _physical(_gemm_prim(), max_groups=1, max_structures=1)
    plan = to_overlap_plan(classified, physical)
    loaded = plan_from_dict(plan_to_dict(plan))
    assert loaded.operations[0].statement is None
    assert loaded.buffers[0].buffer is None
    assert len(loaded.operations) == len(plan.operations)
    assert len(loaded.sync_edges) == len(plan.sync_edges)
    assert int(loaded.shared_arena_bytes) == int(plan.shared_arena_bytes)


def _lower(mod, plan, target):
    annotated = apply_plan_to_ir(mod, plan)
    with target, tilelang.transform.PassContext(config={}):
        return tilelang.transform.LowerOverlapPlan()(annotated)


def test_stamped_plan_lowers_on_layout_reduced_ir() -> None:
    prim = _gemm_prim()
    mod, target = layout_reduced_module(prim)
    function = mod[mod.get_global_var("main")]
    classified, physical = _physical(function, max_groups=1, max_structures=1)
    plan = to_overlap_plan(classified, physical)
    lowered = _lower(mod, plan, target)
    function = lowered[lowered.get_global_var("main")]
    assert function.attrs.get("tl.overlap_plan") is not None
    assert int(function.attrs["tl.smem_planned_arena_bytes"]) == int(
        plan.shared_arena_bytes
    )


def test_handle_free_plan_lowers_on_layout_reduced_ir() -> None:
    prim = _gemm_prim()
    mod, target = layout_reduced_module(prim)
    function = mod[mod.get_global_var("main")]
    classified, physical = _physical(function, max_groups=1, max_structures=1)
    plan = plan_from_dict(plan_to_dict(to_overlap_plan(classified, physical)))
    lowered = _lower(mod, plan, target)
    function = lowered[lowered.get_global_var("main")]
    assert int(function.attrs["tl.smem_planned_arena_bytes"]) == int(
        plan.shared_arena_bytes
    )


def test_fa3_multi_group_plan_lowers() -> None:
    prim = _fa3_prim()
    mod, target = layout_reduced_module(prim)
    function = mod[mod.get_global_var("main")]
    classified = HOPPER.classify(extract_fact_graph(function))
    physical = None
    for structure in enumerate_structures(
        classified,
        SearchBudget(max_groups=3, max_structures=256),
        arch=HOPPER,
    ):
        if structure.num_groups < 2:
            continue
        realized = list(HOPPER.realize(classified, structure))
        if realized:
            physical = realized[0]
            break
    assert physical is not None
    plan = to_overlap_plan(classified, physical)
    lowered = _lower(mod, plan, target)
    function = lowered[lowered.get_global_var("main")]
    script = function.script()
    assert "tl.overlap_plan.group_scope" in script
    assert "set_max_nreg" in script or "SetMaxNReg" in script
    if physical.shared_memory.handoff_allocations:
        offsets = function.attrs.get("tl.smem_offset_map")
        assert offsets is not None
        for handoff in physical.shared_memory.handoff_allocations:
            assert int(offsets[handoff.name]) == int(handoff.byte_offset)
            edge = plan.sync_edges[handoff.channel_id]
            assert int(edge.byte_offset) == int(handoff.byte_offset)


def test_generate_plans_writes_native_candidates(tmp_path: Path) -> None:
    plan_dir = tmp_path / "candidates"
    generate_plans(
        GEMM,
        plan_dir,
        {"block_m": 128, "block_n": 128, "block_k": 64},
        {"gemm_m": 256, "gemm_n": 256, "gemm_k": 256},
        budget=SearchBudget(max_groups=1, max_structures=2),
    )
    manifest = (plan_dir / "manifest.json").read_text(encoding="utf-8")
    files = sorted(plan_dir.glob("schedule_*.json"))
    assert files
    plan = load_plan_json(files[0])
    assert plan.operations
    assert plan.groups
    assert '"schedule_count"' in manifest
    assert plan.operations[0].statement is None


@T.prim_func
def fragment_stage_kernel(
    A: T.Tensor((64,), T.float16),
    B: T.Tensor((64,), T.float16),
):
    with T.Kernel(1, threads=128):
        a_shared = T.alloc_shared((64,), T.float16)
        acc = T.alloc_fragment((64,), T.float32)
        for _k in T.Pipelined(8):
            T.copy(A, a_shared)
            T.copy(a_shared, acc)
            for i in T.Parallel(64):
                acc[i] = T.exp(acc[i])
        T.copy(acc, B)


def test_fragment_pingpong_lowers_as_distinct_buffers() -> None:
    mod, target = layout_reduced_module(fragment_stage_kernel)
    function = mod[mod.get_global_var("main")]
    plan = None
    for candidate in enumerate_overlap_plans(
        function,
        budget=SearchBudget(max_groups=1, max_stages=2, max_structures=32),
        target=target,
        reduce_ir=False,
    ):
        if any(
            int(buffer.version_count) > 1
            and buffer.buffer is not None
            and buffer.buffer.scope() == "local.fragment"
            for buffer in candidate.buffers
        ):
            plan = candidate
            break
    assert plan is not None
    lowered = _lower(mod, plan, target)
    script = lowered[lowered.get_global_var("main")].script()
    assert "_v0" in script
    assert "_v1" in script
    assert "local.fragment" in script
