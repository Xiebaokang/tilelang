"""Contract checks for OverlapPlan matching and lowering."""

from __future__ import annotations

import pytest
import tilelang
from tvm.error import TVMError

from OverlapPlaner.apply import apply_plan_to_ir
from OverlapPlaner.contract import enumerate_overlap_plans, layout_reduced_module
from OverlapPlaner.ir import (
    BufferPlan,
    OperationPlacement,
    OverlapPlan,
    SyncEdge,
)
from OverlapPlaner.serialization import plan_from_dict, plan_to_dict
from OverlapPlaner.structure import SearchBudget
from OverlapPlaner.tune.operators.fa3 import build as build_fa3
from OverlapPlaner.tune.operators.gemm import build as build_gemm
from OverlapPlaner.tune.operators.mamba_chunk_scan import build as build_mamba_scan


def _layout_reduced_gemm():
    prim = build_gemm(
        {"gemm_m": 256, "gemm_n": 256, "gemm_k": 256},
        {"block_m": 128, "block_n": 128, "block_k": 64},
    ).prim_func
    return layout_reduced_module(prim)


def _gemm_plan(mod, target):
    function = mod[mod.get_global_var("main")]
    return next(
        enumerate_overlap_plans(
            function,
            budget=SearchBudget(max_groups=1, max_structures=4),
            target=target,
            reduce_ir=False,
        )
    )


def _optional_int(value):
    return None if value is None else int(value)


def _clone_operation(operation, **overrides):
    return OperationPlacement(
        overrides.get("operation_id", int(operation.operation_id)),
        overrides["statement"] if "statement" in overrides else operation.statement,
        overrides.get("group_id", int(operation.group_id)),
        overrides["stage"] if "stage" in overrides else _optional_int(operation.stage),
        overrides.get("order", int(operation.order)),
    )


def _clone_buffer(buffer_plan, **overrides):
    return BufferPlan(
        overrides.get("buffer_id", int(buffer_plan.buffer_id)),
        overrides["buffer"] if "buffer" in overrides else buffer_plan.buffer,
        overrides.get("version_count", int(buffer_plan.version_count)),
        overrides.get("communication", int(buffer_plan.communication)),
        overrides["byte_offset"]
        if "byte_offset" in overrides
        else _optional_int(buffer_plan.byte_offset),
    )


def _clone_sync_edge(edge):
    return SyncEdge(
        int(edge.producer_id),
        int(edge.consumer_id),
        _optional_int(edge.buffer_id),
        int(edge.kind),
        int(edge.scope),
        int(edge.iteration_distance),
        int(edge.slot_count),
        int(edge.dependency_kind),
        int(edge.completion_mode),
        _optional_int(edge.byte_offset),
    )


def _rebuild_plan(plan, operations=None, buffers=None, sync_edges=None):
    return OverlapPlan(
        groups=list(plan.groups),
        operations=list(plan.operations) if operations is None else operations,
        buffers=list(plan.buffers) if buffers is None else buffers,
        sync_edges=list(plan.sync_edges) if sync_edges is None else sync_edges,
        shared_arena_bytes=_optional_int(plan.shared_arena_bytes),
    )


def _bump_first_stage(plan, stage):
    operations = []
    bumped = False
    for operation in plan.operations:
        next_stage = _optional_int(operation.stage)
        if not bumped and next_stage is not None:
            next_stage = stage
            bumped = True
        operations.append(_clone_operation(operation, stage=next_stage))
    assert bumped
    return operations


def _lower(mod, plan, target):
    annotated = apply_plan_to_ir(mod, plan)
    with target, tilelang.transform.PassContext(config={}):
        return tilelang.transform.LowerOverlapPlan()(annotated)


def test_wrong_operation_id_is_rejected() -> None:
    mod, target = _layout_reduced_gemm()
    plan = _gemm_plan(mod, target)
    operations = [_clone_operation(operation) for operation in plan.operations]
    operations[0] = _clone_operation(operations[0], operation_id=999)
    with pytest.raises(TVMError, match="operation_id must equal"):
        _lower(mod, _rebuild_plan(plan, operations=operations), target)


def test_mixed_statement_handles_are_rejected() -> None:
    mod, target = _layout_reduced_gemm()
    plan = _gemm_plan(mod, target)
    operations = [_clone_operation(operation) for operation in plan.operations]
    operations[0] = _clone_operation(
        operations[0], statement=plan.operations[1].statement
    )
    operations[1] = _clone_operation(
        operations[1], statement=plan.operations[0].statement
    )
    operations[2] = _clone_operation(operations[2], statement=None)
    with pytest.raises(TVMError, match="mixed matching is not supported"):
        _lower(mod, _rebuild_plan(plan, operations=operations), target)


def test_mixed_buffer_handles_are_rejected() -> None:
    mod, target = _layout_reduced_gemm()
    plan = _gemm_plan(mod, target)
    buffers = [_clone_buffer(buffer) for buffer in plan.buffers]
    buffers[0] = _clone_buffer(buffers[0], buffer=None)
    with pytest.raises(TVMError, match="mixed matching is not supported"):
        _lower(mod, _rebuild_plan(plan, buffers=buffers), target)


def test_auto_overlap_may_change_pipeline_depth() -> None:
    mod, target = _layout_reduced_gemm()
    plan = _gemm_plan(mod, target)
    lowered = _lower(
        mod, _rebuild_plan(plan, operations=_bump_first_stage(plan, 9)), target
    )
    function = lowered[lowered.get_global_var("main")]
    assert "tl.overlap_plan.group_scope" in function.script()


def test_manual_pipeline_depth_must_match_without_auto_overlap() -> None:
    mod, target = _layout_reduced_gemm()
    function = mod[mod.get_global_var("main")]
    mod.update_func(
        mod.get_global_var("main"), function.without_attr("tl.auto_overlap")
    )
    plan = _gemm_plan(mod, target)
    with pytest.raises(TVMError, match="stage count does not match"):
        _lower(
            mod, _rebuild_plan(plan, operations=_bump_first_stage(plan, 9)), target
        )


def test_lowered_schedule_carries_planned_offsets_and_stages() -> None:
    from tvm import tirx
    from tvm.tirx.stmt_functor import post_order_visit

    mod, target = _layout_reduced_gemm()
    plan = _gemm_plan(mod, target)
    lowered = _lower(mod, plan, target)
    function = lowered[lowered.get_global_var("main")]

    offset_map = function.attrs.get("tl.smem_offset_map")
    assert offset_map is not None
    for buffer_plan in plan.buffers:
        if buffer_plan.byte_offset is None or buffer_plan.buffer is None:
            continue
        expected = int(buffer_plan.byte_offset)
        assert int(offset_map[buffer_plan.buffer.name]) == expected
        assert int(offset_map[buffer_plan.buffer.data.name]) == expected
    assert int(function.attrs["tl.smem_planned_arena_bytes"]) == int(
        plan.shared_arena_bytes
    )

    planned_stages = {
        int(operation.stage)
        for operation in plan.operations
        if operation.stage is not None
    }
    staged_loops = []

    def visit(node):
        if isinstance(node, tirx.For) and "tl_pipeline_stage" in node.annotations:
            staged_loops.append(
                [int(stage) for stage in node.annotations["tl_pipeline_stage"]]
            )

    post_order_visit(function.body, visit)
    assert staged_loops
    assert planned_stages
    for stages in staged_loops:
        assert max(stages) + 1 <= max(planned_stages) + 1


def test_sync_events_are_marked_per_channel() -> None:
    from tvm import tirx
    from tvm.tirx.stmt_functor import post_order_visit

    prim = build_fa3(
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
    mod, target = layout_reduced_module(prim)
    function = mod[mod.get_global_var("main")]
    plan = None
    for candidate in enumerate_overlap_plans(
        function,
        budget=SearchBudget(max_groups=3, max_structures=256),
        target=target,
        reduce_ir=False,
    ):
        if len(candidate.groups) < 2:
            continue
        if any(int(edge.completion_mode) == 0 for edge in candidate.sync_edges):
            plan = candidate
            break
    assert plan is not None
    sync_edges = [_clone_sync_edge(edge) for edge in plan.sync_edges]
    thread_edge = next(
        edge for edge in plan.sync_edges if int(edge.completion_mode) == 0
    )
    sync_edges.append(_clone_sync_edge(thread_edge))
    lowered = _lower(mod, _rebuild_plan(plan, sync_edges=sync_edges), target)
    function = lowered[lowered.get_global_var("main")]

    events = []

    def visit(node):
        if (
            isinstance(node, tirx.AttrStmt)
            and node.attr_key == "tl.overlap_plan.sync_event"
        ):
            events.append((int(node.node), int(node.value)))

    post_order_visit(function.body, visit)
    for channel, edge in enumerate(sync_edges):
        if int(edge.completion_mode) == 1:
            same_event = [
                index
                for index, candidate in enumerate(sync_edges)
                if int(candidate.completion_mode) == 1
                and int(candidate.producer_id) == int(edge.producer_id)
            ]
            expected_arrives = 1 if channel == min(same_event) else 0
            assert events.count((channel, 2)) == expected_arrives
        else:
            assert events.count((channel, 0)) == 1
        assert events.count((channel, 1)) == 1

    with target, tilelang.transform.PassContext(config={}):
        finalized = tilelang.transform.FinalizeOverlapPlan()(lowered)
    finalized_function = finalized[finalized.get_global_var("main")]
    assert "tl.overlap_plan.sync_event" not in finalized_function.script()


def test_region_boundary_fragment_handoff_is_guarded_once() -> None:
    """A pipeline accumulator crosses each region boundary exactly once."""

    prim = build_mamba_scan(
        {
            "mamba_scan_batch": 1,
            "mamba_scan_heads": 4,
            "mamba_scan_groups": 1,
            "mamba_scan_seq": 256,
            "mamba_scan_chunk": 256,
            "mamba_scan_dim": 64,
            "mamba_scan_dstate": 128,
        },
        {
            "block_m": 128,
            "block_n": 32,
            "block_k": 64,
            "block_dstate": 128,
        },
    ).prim_func
    mod, target = layout_reduced_module(prim)
    function = mod[mod.get_global_var("main")]
    selected = None
    for candidate in enumerate_overlap_plans(
        function,
        budget=SearchBudget(max_groups=3, max_stages=1, max_structures=256),
        target=target,
        reduce_ir=False,
    ):
        groups = {
            int(operation.operation_id): int(operation.group_id)
            for operation in candidate.operations
        }
        # acc crosses serial prologue -> pipeline -> serial epilogue.
        if groups[7] != groups[18] and groups[18] != groups[22]:
            selected = candidate
            break
    assert selected is not None

    script = _lower(mod, selected, target)[
        mod.get_global_var("main")
    ].script()
    assert "if ik == 0:" in script
    assert "if ik == by // 2 * 2 + 2 - 1:" in script
    assert "acc_wsp_handoff" in script


def test_handle_free_native_plan_lowers_with_shared_offsets() -> None:
    mod, target = _layout_reduced_gemm()
    handle_plan = _gemm_plan(mod, target)
    plan = plan_from_dict(plan_to_dict(handle_plan))
    assert all(operation.statement is None for operation in plan.operations)
    assert all(buffer.buffer is None for buffer in plan.buffers)

    lowered = _lower(mod, plan, target)
    function = lowered[lowered.get_global_var("main")]
    offsets = function.attrs.get("tl.smem_offset_map")
    assert offsets is not None
    assert int(function.attrs["tl.smem_planned_arena_bytes"]) == int(
        plan.shared_arena_bytes
    )
