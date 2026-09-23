"""Overlaper-to-LowerProgramSchedule integration tests."""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import tilelang
import torch
import torch.nn.functional as F
from tvm import IRModule, tirx
from tvm.target import Target
from tvm.tirx.stmt_functor import post_order_visit

from history.overlaper import load_schedule_json
from history.overlaper.analysis.physical import (
    WarpAllocation,
    analyze_shared_memory,
    enumerate_warp_allocations,
)
from history.overlaper.analysis.schedule import (
    analyze_buffer_versions,
    build_program_orders,
    build_synchronizations,
    enumerate_group_assignments,
)
from history.overlaper.integration import (
    apply_active_schedule,
    apply_schedule_to_ir,
    build_ir_plan,
    schedule_planner_is_active,
    use_schedule_planner,
)
from history.overlaper.parse import extract_dataflow_graph
from history.overlaper.test.test_extractor import make_fa3_prim_func
from history.overlaper.test.test_stage import (
    FA3_THREE_STAGE_ASSIGNMENT,
    FA3_TRANSACTION_STAGE_ASSIGNMENT,
    find_fa3_stage_assignment,
)


def _layout_reduced_fa3(seq_len=256, **kernel_kwargs):
    target = Target({"kind": "cuda", "arch": "sm_90a"})
    mod = IRModule({"main": make_fa3_prim_func(seq_len=seq_len, **kernel_kwargs)})
    with target, tilelang.transform.PassContext(config={}):
        mod = tirx.transform.BindTarget(target)(mod)
        mod = tilelang.transform.MaterializeKernelLaunch()(mod)
        mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
        mod = tilelang.transform.LegalizeNegativeIndex()(mod)
        mod = tilelang.transform.InjectAssumes()(mod)
        mod = tilelang.transform.Simplify()(mod)
        mod = tilelang.transform.LayoutReducer()(mod)
    return mod, target


def _fa3_schedule(
    graph,
    stage_assignment=FA3_THREE_STAGE_ASSIGNMENT,
    group_index=1,
):
    stages = {1: find_fa3_stage_assignment(graph, stage_assignment)}
    groups = tuple(enumerate_group_assignments(graph, 2))[group_index]
    orders = build_program_orders(graph, stages, groups)
    versions = analyze_buffer_versions(graph, stages, groups, orders)
    synchronizations = build_synchronizations(
        graph, stages, groups, orders, versions
    )
    allocation = next(
        enumerate_warp_allocations(graph, groups, orders, versions)
    )
    shared_memory = analyze_shared_memory(
        graph, groups, orders, versions, synchronizations
    )
    return SimpleNamespace(
        stages_by_region=stages,
        groups=groups,
        orders=orders,
        buffer_versions=versions,
        synchronizations=synchronizations,
        warp_allocation=allocation,
        shared_memory=shared_memory,
    )


def compile_fa3_schedule_json(path: str | Path) -> IRModule:
    """Load one FA3 schedule JSON file and lower it into compiled schedule IR."""

    mod, target = _layout_reduced_fa3()
    graph = extract_dataflow_graph(mod, target=target)
    schedule = load_schedule_json(path)
    annotated = apply_schedule_to_ir(mod, graph, schedule)
    with target, tilelang.transform.PassContext(config={}):
        return tilelang.transform.LowerProgramSchedule()(annotated)


def _fa3_reference(q, k, v):
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale
    probabilities = F.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", probabilities, v.float()).half()


def _has_hopper_gpu() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
    )


def test_fa3_schedule_json_file_compiles() -> None:
    schedule_path = Path(__file__).parent / "data" / "fa3_schedule.json"
    lowered = compile_fa3_schedule_json(schedule_path)
    function = lowered[lowered.get_global_var("main")]

    assert int(function.attrs["tl.program_schedule.version"]) == 3
    assert int(function.attrs["tl.program_schedule.groups_lowered"]) == 1
    assert int(function.attrs["tl.program_schedule.buffers_lowered"]) == 1
    assert int(
        function.attrs["tl.program_schedule.synchronization_lowered"]
    ) == 1
    print(lowered)


def test_fa3_same_group_qk_handoffs_lower_to_tma() -> None:
    """Same-group Q/K handoffs use transaction TMA, not SIMT copies."""
    mod, target = _layout_reduced_fa3()
    graph = extract_dataflow_graph(mod, target=target)
    schedule = _fa3_schedule(graph)
    annotated = apply_schedule_to_ir(mod, graph, schedule)

    with target, tilelang.transform.PassContext(config={}):
        lowered = tilelang.transform.LowerProgramSchedule()(annotated)
        lowered = tilelang.transform.PipelinePlanning()(lowered)
        lowered = tilelang.transform.InjectSoftwarePipeline()(lowered)
        lowered = tilelang.transform.Simplify()(lowered)
        lowered = tilelang.transform.LayoutInference()(lowered)
        lowered = tilelang.transform.LowerTileOp()(lowered)

    script = lowered[lowered.get_global_var("main")].script()
    assert script.count("T.tma_load") >= 3
    assert "Q.data" in script
    assert "K.data" in script
    assert "V.data" in script
    assert "T.mbarrier_wait_parity(program_schedule_mbar" in script
    assert "pipeline_mbar" not in script


def test_program_schedule_tma_elects_group_local_first_warp() -> None:
    """Group TMA must elect local warp 0, not every Nth warp of the block."""

    mod, target = _layout_reduced_fa3()
    graph = extract_dataflow_graph(mod, target=target)
    schedule = _fa3_schedule(graph)
    assert schedule.warp_allocation.groups[0].warp_count == 8
    assert schedule.warp_allocation.groups[1].first_warp == 8
    annotated = apply_schedule_to_ir(mod, graph, schedule)

    with target, tilelang.transform.PassContext(config={}):
        lowered = tilelang.transform.LowerProgramSchedule()(annotated)
        lowered = tilelang.transform.PipelinePlanning()(lowered)
        lowered = tilelang.transform.InjectSoftwarePipeline()(lowered)
        lowered = tilelang.transform.Simplify()(lowered)
        lowered = tilelang.transform.LayoutInference()(lowered)
        lowered = tilelang.transform.LowerTileOp()(lowered)

    conditions = []

    def collect(node):
        if isinstance(node, tirx.IfThenElse):
            conditions.append(str(node.condition))

    post_order_visit(lowered[lowered.get_global_var("main")].body, collect)
    elect_conditions = [
        condition
        for condition in conditions
        if "tl_shuffle_elect" in condition or "shuffle_elect" in condition
    ]
    assert elect_conditions
    assert any(
        "// 32" in condition or "floordiv" in condition.lower()
        for condition in elect_conditions
    ), elect_conditions


@pytest.mark.skipif(not _has_hopper_gpu(), reason="requires a Hopper CUDA GPU")
def test_fa3_schedule_json_runtime_correctness_and_performance() -> None:
    schedule_path = Path(__file__).parent / "data" / "fa3_schedule.json"
    target = Target({"kind": "cuda", "arch": "sm_90a"})
    loaded_schedules = []

    def planner(_symbol, _graph, _target):
        schedule = load_schedule_json(schedule_path)
        loaded_schedules.append(schedule)
        return schedule

    prim_func = make_fa3_prim_func().with_attr(
        "tl.program_schedule.request", "overlaper-fa3-json-correctness-v1"
    )
    with use_schedule_planner(planner), target:
        compiled = tilelang.compile(
            prim_func,
            out_idx=[3],
            target=target,
            execution_backend="cython",
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
    assert len(loaded_schedules) == 1

    torch.manual_seed(0)
    shape = (1, 1, 256, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    actual = compiled(q, k, v)
    expected = _fa3_reference(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)

    max_abs_error = float((actual.float() - expected.float()).abs().max())
    latency_ms = float(
        compiled.get_profiler().do_bench(
            warmup=10,
            rep=10,
            input_tensors=[q, k, v],
        )
    )
    total_flops = 4.0 * 1 * 1 * 256 * 256 * 64
    tflops = total_flops / latency_ms * 1e-9
    assert math.isfinite(latency_ms) and latency_ms > 0
    assert math.isfinite(tflops) and tflops > 0
    print(
        "\nFA3 JSON schedule correctness: PASS, "
        f"max_abs_error={max_abs_error:.6f}, "
        f"latency={latency_ms:.6f} ms, throughput={tflops:.3f} TFLOPS"
    )


def test_fa3_ir_plan_is_accepted_by_cpp_lowering() -> None:
    mod, target = _layout_reduced_fa3()
    graph = extract_dataflow_graph(mod, target=target)
    schedule = _fa3_schedule(graph)

    plan = build_ir_plan(graph, schedule)
    assert plan["region_num_stages"] == (0, 3, 0)
    assert plan["sync_dependency_masks"] == (1, 1, 1, 8)
    # The Q/V producer handoffs may now use per-iteration TMA transaction
    # completion.  The buffer-reuse channel remains a thread-arrive barrier.
    assert plan["sync_completion_modes"] == (1, 1, 1, 0)
    assert plan["group_warp_counts"] == (8, 4)
    assert plan["register_domain_groups"] == (0, 0, 1)

    annotated = apply_schedule_to_ir(mod, graph, schedule)
    function = annotated[annotated.get_global_var("main")]
    assert int(function.attrs["tl.program_schedule.version"]) == 3

    with target, tilelang.transform.PassContext(config={}):
        lowered = tilelang.transform.LowerProgramSchedule()(annotated)
    function = lowered[lowered.get_global_var("main")]
    assert int(function.attrs["tl.program_schedule.groups_lowered"]) == 1
    assert int(function.attrs["tl.program_schedule.buffers_lowered"]) == 1
    assert int(
        function.attrs["tl.program_schedule.synchronization_lowered"]
    ) == 1


def test_apply_schedule_attaches_shared_memory_plan() -> None:
    mod, target = _layout_reduced_fa3()
    graph = extract_dataflow_graph(mod, target=target)
    schedule = _fa3_schedule(graph)
    plan = build_ir_plan(graph, schedule)
    annotated = apply_schedule_to_ir(mod, graph, schedule)
    function = annotated[annotated.get_global_var("main")]

    assert plan["merged_shared_bytes"] == schedule.shared_memory.merged_shared_bytes
    assert int(function.attrs["tl.smem_planned_arena_bytes"]) == (
        schedule.shared_memory.merged_shared_bytes
    )
    offset_map = {
        str(name): int(value)
        for name, value in function.attrs["tl.smem_offset_map"].items()
    }
    for item in schedule.shared_memory.shared_allocations:
        assert offset_map[item.name] == item.byte_offset
    by_id = {
        item.buffer_id: item.byte_offset
        for item in schedule.shared_memory.shared_allocations
    }
    for buffer, offset in zip(graph.buffers, plan["shared_byte_offsets"]):
        if buffer.buffer_id in by_id:
            assert offset == by_id[buffer.buffer_id]
        else:
            assert offset == -1


def test_ir_plan_rejects_register_file_saturation() -> None:
    mod, target = _layout_reduced_fa3()
    graph = extract_dataflow_graph(mod, target=target)
    schedule = _fa3_schedule(graph)
    allocation = schedule.warp_allocation

    schedule.warp_allocation = WarpAllocation(
        groups=allocation.groups,
        effective_threads=allocation.effective_threads,
        register_counts=(240, 24),
        register_is_increase=(True, False),
    )

    with pytest.raises(
        ValueError, match="must leave register-file headroom"
    ):
        build_ir_plan(graph, schedule)


def test_planner_context_is_scoped_and_only_visits_auto_schedule() -> None:
    mod, target = _layout_reduced_fa3()
    calls = []

    def planner(symbol, graph, received_target):
        calls.append((symbol, len(graph.nodes), received_target.kind.name))
        return None

    assert not schedule_planner_is_active()
    with use_schedule_planner(planner):
        assert schedule_planner_is_active()
        assert apply_active_schedule(mod, target).same_as(mod)
    assert not schedule_planner_is_active()
    assert calls == [("main", 22, "cuda")]


def test_tma_handoffs_use_transaction_completion() -> None:
    mod, target = _layout_reduced_fa3()
    graph = extract_dataflow_graph(mod, target=target)
    schedule = _fa3_schedule(
        graph,
        stage_assignment=FA3_TRANSACTION_STAGE_ASSIGNMENT,
        group_index=7,
    )

    plan = build_ir_plan(graph, schedule)

    assert plan["sync_producers"] == (0, 4, 17)
    assert plan["sync_consumers"] == (6, 6, 18)
    assert plan["sync_completion_modes"] == (1, 1, 1)


def test_group_pipeline_stages_are_normalized_after_partition() -> None:
    """A group's common leading stage must not become idle pipeline phases."""

    mod, target = _layout_reduced_fa3(seq_len=8192)
    graph = extract_dataflow_graph(mod, target=target)
    shifted = {
        4: 0,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
        11: 1,
        12: 1,
        13: 1,
        14: 1,
        15: 1,
        16: 1,
        17: 1,
        18: 2,
    }
    stages = {1: find_fa3_stage_assignment(graph, shifted)}
    groups = {
        node.node_id: (
            1
            if node.node_id in {0, 17}
            else 2
            if node.node_id in {4, 21}
            else 0
        )
        for node in graph.nodes
    }
    orders = build_program_orders(graph, stages, groups)
    versions = analyze_buffer_versions(graph, stages, groups, orders)
    synchronizations = build_synchronizations(
        graph, stages, groups, orders, versions
    )
    schedule = SimpleNamespace(
        stages_by_region=stages,
        groups=groups,
        orders=orders,
        buffer_versions=versions,
        synchronizations=synchronizations,
        warp_allocation=next(
            enumerate_warp_allocations(graph, groups, orders, versions)
        ),
        shared_memory=analyze_shared_memory(
            graph, groups, orders, versions, synchronizations
        ),
    )
    annotated = apply_schedule_to_ir(mod, graph, schedule)

    with target, tilelang.transform.PassContext(config={}):
        lowered = tilelang.transform.LowerProgramSchedule()(annotated)

    script = lowered[lowered.get_global_var("main")].script()
    # The compute group owns stages {1, 2}; after group partition they become
    # {0, 1}.  No group-local annotation may retain the leading offset.
    assert '"tl_pipeline_stage": [1, 1' not in script
    assert '"tl_pipeline_stage": [0, 0' in script

    with target, tilelang.transform.PassContext(config={}):
        lowered = tilelang.transform.PipelinePlanning()(lowered)
        lowered = tilelang.transform.InjectSoftwarePipeline()(lowered)
        lowered = tilelang.transform.Simplify()(lowered)

    pipelined_script = lowered[lowered.get_global_var("main")].script()
    assert "for k in T.serial(63," in pipelined_script
    assert "for k in T.serial(62," not in pipelined_script


"""
PYTHONPATH="$PWD/3rdparty/tvm/python:$PWD" \
TVM_LIBRARY_PATH="$PWD/build/lib" \
python -m pytest \
overlaper/test/test_integration.py::test_fa3_schedule_json_file_compiles \
-q -s
"""
