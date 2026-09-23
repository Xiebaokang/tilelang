"""Validate UnionWSP schedule application before performance search."""

import math
from functools import partial

import pytest
import tilelang
import torch
from tvm import IRModule, tirx
from tvm.tirx.stmt_functor import post_order_visit

import history.unionwsp as unionwsp
from history.unionwsp.parseIR import extract_dataflow_graph
from history.wspipeline.test.fa3_kernel import (
    make_cuda_target,
    make_fa3_prim_func,
    ref_program,
)


def _layout_reduced_fa3(prim_func=None):
    target = make_cuda_target()
    mod = IRModule(
        {"main": make_fa3_prim_func() if prim_func is None else prim_func}
    )
    with target, tilelang.transform.PassContext(config={}):
        mod = tirx.transform.BindTarget(target)(mod)
        mod = tilelang.transform.MaterializeKernelLaunch()(mod)
        mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
        mod = tilelang.transform.LegalizeNegativeIndex()(mod)
        mod = tilelang.transform.InjectAssumes()(mod)
        mod = tilelang.transform.Simplify()(mod)
        mod = tilelang.transform.LayoutReducer()(mod)
    return mod, target


def _first_specialized_schedule(graph):
    assert graph.kernel_threads == 256
    return next(
        unionwsp.enumerate_wsp_schedules(
            graph,
            original_threads=graph.kernel_threads,
            num_stages=2,
            num_groups=2,
        )
    )


def _versioned_v_schedule(graph):
    buffer_id = next(
        buffer.buffer_id for buffer in graph.buffers if buffer.name == "V_shared"
    )
    node_ids = {node.name: node.node_id for node in graph.nodes}
    copy_v = node_ids["copy_V_to_V_shared"]
    gemm_v = node_ids["gemm_acc_o"]
    copy_k = node_ids["copy_K_to_K_shared"]
    for schedule in unionwsp.enumerate_wsp_schedules(
        graph,
        original_threads=graph.kernel_threads,
        num_stages=2,
    ):
        if (
            schedule.buffer_versions[buffer_id] == 2
            and schedule.groups[copy_v] == schedule.groups[gemm_v]
            and schedule.groups[copy_k] == schedule.groups[gemm_v]
        ):
            return schedule
    raise AssertionError("FA3 must provide an in-group two-version V schedule")


def _q_loader_specialized_schedule(graph):
    node_ids = {node.name: node.node_id for node in graph.nodes}
    copy_q = node_ids["copy_Q_to_Q_shared"]
    gemm_qk = node_ids["gemm_acc_s"]
    for schedule in unionwsp.enumerate_wsp_schedules(
        graph,
        original_threads=graph.kernel_threads,
        num_stages=3,
        num_groups=2,
    ):
        if schedule.groups[copy_q] != schedule.groups[gemm_qk]:
            return schedule
    raise AssertionError("FA3 must provide a specialized Q-loader schedule")


def test_fa3_schedule_is_materialized_by_cpp_pass() -> None:
    """Check the exact LayoutReducer -> UnionWSP -> C++ pass boundary."""

    mod, target = _layout_reduced_fa3()
    graph = extract_dataflow_graph(mod, target=target)
    schedule = _first_specialized_schedule(graph)
    annotated = unionwsp.apply_schedule_to_ir(mod, graph, schedule)

    function = annotated[annotated.get_global_var("main")]
    assert int(function.attrs["tl.program_schedule.version"]) == 3
    assert tuple(function.attrs["tl.program_schedule.operation_groups"]) == tuple(
        schedule.groups[node.node_id] for node in graph.nodes
    )
    assert int(function.attrs["tl.program_schedule.effective_threads"]) == (
        schedule.warp_allocation.effective_threads
    )
    assert tuple(function.attrs["tl.program_schedule.sync_slot_counts"]) == tuple(
        channel.slot_count for channel in schedule.synchronization
    )

    with target, tilelang.transform.PassContext(config={}):
        lowered = tilelang.transform.LowerProgramSchedule()(annotated)
    function = lowered[lowered.get_global_var("main")]
    assert int(function.attrs["tl.program_schedule.groups_lowered"]) == 1
    assert int(function.attrs["tl.program_schedule.buffers_lowered"]) == 1
    assert int(function.attrs["tl.program_schedule.synchronization_lowered"]) == 1

    operation_ids = []
    group_scopes = []
    pipeline_loops = []

    def collect(node) -> None:
        if isinstance(node, tirx.AttrStmt):
            if node.attr_key == "tl.program_schedule.operation":
                operation_ids.append(int(node.node))
            elif node.attr_key == "tl.program_schedule.group_scope":
                group_scopes.append(int(node.value))
        elif isinstance(node, tirx.For) and "tl_pipeline_stage" in node.annotations:
            pipeline_loops.append(node)

    post_order_visit(function.body, collect)
    assert sorted(operation_ids) == list(range(len(graph.nodes)))
    assert set(group_scopes) == {0, 1}
    assert pipeline_loops
    assert all("tl_pipeline_order" in loop.annotations for loop in pipeline_loops)


def test_region_boundary_wait_is_anchored_to_first_q_consumer() -> None:
    """A one-shot Q wait must not block unrelated pipeline preparation."""

    mod, target = _layout_reduced_fa3()
    graph = extract_dataflow_graph(mod, target=target)
    schedule = _q_loader_specialized_schedule(graph)
    node_ids = {node.name: node.node_id for node in graph.nodes}
    consumer_id = node_ids["gemm_acc_s"]
    mod = unionwsp.apply_schedule_to_ir(mod, graph, schedule)

    function = mod[mod.get_global_var("main")]
    assert tuple(function.attrs["tl.program_schedule.sync_completion_modes"]) == (
        1,
    )

    with target, tilelang.transform.PassContext(config={}):
        mod = tilelang.transform.LowerProgramSchedule()(mod)

    script = mod[mod.get_global_var("main")].script()
    operation_marker = (
        f'with T.attr({consumer_id}, "tl.program_schedule.operation"'
    )
    consumer_start = script.index(operation_marker)
    wait = "T.mbarrier_wait_parity(program_schedule_mbar"
    wait_index = script.index(wait)
    gemm_index = script.index("T.gemm", consumer_start)

    assert script.count(wait) == 1
    assert consumer_start < wait_index < gemm_index
    assert "if k == 0:" in script[consumer_start:wait_index]
    assert "T.tma_copy" in script
    assert "T.ptx_arrive_barrier(program_schedule_mbar" not in script


def test_fa3_explicit_schedule_reaches_tma_pipeline_rewrite() -> None:
    """UnionWSP's explicit plan must not lower its TMA nodes as cp.async."""

    mod, target = _layout_reduced_fa3()
    graph = extract_dataflow_graph(mod, target=target)
    schedule = _first_specialized_schedule(graph)
    mod = unionwsp.apply_schedule_to_ir(mod, graph, schedule)

    with target, tilelang.transform.PassContext(config={}):
        mod = tilelang.transform.LowerProgramSchedule()(mod)
        mod = tilelang.transform.PipelinePlanning()(mod)

    pipeline_loops = []

    def collect(node) -> None:
        if (
            isinstance(node, tirx.For)
            and "software_pipeline_stage" in node.annotations
        ):
            pipeline_loops.append(node)

    post_order_visit(mod[mod.get_global_var("main")].body, collect)
    assert pipeline_loops
    assert any(
        "software_pipeline_tma_copies" in loop.annotations
        for loop in pipeline_loops
    )

    with target, tilelang.transform.PassContext(config={}):
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    script = mod[mod.get_global_var("main")].script()
    assert "T.tma_copy" in script
    assert "T.mbarrier_wait_parity" in script

    with target, tilelang.transform.PassContext(config={}):
        mod = tilelang.transform.Simplify()(mod)
        mod = tilelang.transform.LayoutInference()(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
    script = mod[mod.get_global_var("main")].script()
    assert "ptx_cp_async" not in script
    assert "tma_load" in script


def test_versioned_v_wgmma_descriptor_uses_dynamic_slot_offset() -> None:
    """The WGMMA consumer must read the same V_shared slot written by TMA."""

    mod, target = _layout_reduced_fa3(make_fa3_prim_func(seq_kv=1024))
    graph = extract_dataflow_graph(mod, target=target)
    schedule = _versioned_v_schedule(graph)
    mod = unionwsp.apply_schedule_to_ir(mod, graph, schedule)

    with target, tilelang.transform.PassContext(config={}):
        mod = tilelang.transform.LowerProgramSchedule()(mod)
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
        mod = tilelang.transform.Simplify()(mod)
        mod = tilelang.transform.LayoutInference()(mod)
        mod = tilelang.transform.LowerTileOp()(mod)

    function = mod[mod.get_global_var("main")]
    offsets = []

    def collect_descriptor_offsets(node) -> None:
        if (
            isinstance(node, tirx.Call)
            and hasattr(node.op, "name")
            and node.op.name == "tl.increase_descriptor_offset"
        ):
            offsets.append(node.args[1])

    post_order_visit(function.body, collect_descriptor_offsets)
    assert offsets
    assert any(tirx.analysis.undefined_vars(offset, []) for offset in offsets)


def _has_hopper_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.skipif(not _has_hopper_gpu(), reason="requires a Hopper CUDA GPU")
def test_one_applied_fa3_schedule_is_numerically_correct() -> None:
    """Compile one real UnionWSP schedule and compare it with PyTorch."""

    batch, heads, seq_q, seq_kv, dim = 1, 1, 256, 256, 64
    prim_func = make_fa3_prim_func(batch, heads, seq_q, seq_kv, dim)
    target = make_cuda_target()
    selected = []

    def planner(_symbol, graph, _target):
        schedule = _first_specialized_schedule(graph)
        selected.append(schedule)
        return schedule

    with unionwsp.use_schedule_planner(planner), target:
        compiled = tilelang.compile(
            prim_func.with_attr("tl.program_schedule.request", "unionwsp-correctness-v1"),
            out_idx=[3],
            target=target,
            execution_backend="cython",
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
    assert len(selected) == 1
    compiled.get_profiler().assert_allclose(
        partial(ref_program, is_causal=False),
        rtol=0.01,
        atol=0.01,
    )
    assert math.isfinite(
        compiled.get_profiler().do_bench(warmup=10, rep=10)
    )
