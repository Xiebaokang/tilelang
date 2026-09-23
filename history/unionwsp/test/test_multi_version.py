"""Analyze and print multiversion buffers for every FA3 final schedule."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "3rdparty" / "tvm" / "python"))
os.environ.setdefault("TVM_LIBRARY_PATH", str(PROJECT_ROOT / "build" / "lib"))

from history.unionwsp.hardware import HOPPER, InstructionKind
from history.unionwsp.parseIR import (
    BufferDescriptor,
    DataflowEdge,
    DataflowGraph,
    DataflowNode,
    RegionKind,
)
from history.unionwsp.schedule import (
    analyze_buffer_versions,
    enumerate_buffer_version_plans,
    multiversion_buffers,
    multiversion_candidate_buffers,
)


def _buffer(buffer_id: int, name: str, scope: str) -> BufferDescriptor:
    return BufferDescriptor(buffer_id, name, scope, 128, object())


def test_order_can_reduce_the_required_version_count() -> None:
    graph = DataflowGraph(
        buffers=(_buffer(0, "shared", "shared"),),
        nodes=(
            DataflowNode(
                0,
                0,
                "writer",
                writes=(0,),
                instruction_kind=InstructionKind.TMA,
            ),
            DataflowNode(
                1,
                0,
                "accessor",
                reads=(0,),
                instruction_kind=InstructionKind.WGMMA,
            ),
        ),
        edges=(DataflowEdge(0, 1, buffer_id=0),),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
    )
    stages = {0: {0: 0, 1: 1}}
    groups = {0: 0, 1: 0}

    writer_first = {0: {0: {0: 0, 1: 1}}}
    accessor_first = {0: {0: {1: 0, 0: 1}}}

    assert analyze_buffer_versions(
        graph, stages, groups, writer_first
    ) == {0: 2}
    assert analyze_buffer_versions(
        graph, stages, groups, accessor_first
    ) == {0: 1}
    assert multiversion_candidate_buffers(graph, stages) == (0,)
    assert tuple(
        enumerate_buffer_version_plans(
            graph, stages, groups, accessor_first
        )
    ) == ({0: 1}, {0: 2})


def test_candidates_are_independent_and_exclude_loop_carried_state() -> None:
    graph = DataflowGraph(
        buffers=(
            _buffer(0, "tma_output", "shared"),
            _buffer(1, "mma_output", "local.fragment"),
            _buffer(2, "ordinary", "shared"),
            _buffer(3, "recurrence", "local.fragment"),
        ),
        nodes=(
            DataflowNode(
                0,
                0,
                "tma",
                writes=(0,),
                instruction_kind=InstructionKind.TMA,
            ),
            DataflowNode(
                1,
                0,
                "mma",
                writes=(1, 3),
                instruction_kind=InstructionKind.WGMMA,
            ),
            DataflowNode(
                2,
                0,
                "generic",
                writes=(2,),
                instruction_kind=InstructionKind.GENERIC,
            ),
            DataflowNode(3, 0, "consume", reads=(0, 1, 2, 3)),
        ),
        edges=(
            DataflowEdge(0, 3, buffer_id=0),
            DataflowEdge(1, 3, buffer_id=1),
            DataflowEdge(2, 3, buffer_id=2),
            DataflowEdge(1, 3, buffer_id=3),
            DataflowEdge(1, 1, iteration_distance=1, buffer_id=3),
        ),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
    )
    stages = {0: {0: 0, 1: 0, 2: 0, 3: 1}}
    groups = {node_id: 0 for node_id in range(4)}
    orders = {0: {0: {node_id: node_id for node_id in range(4)}}}

    minimum = analyze_buffer_versions(graph, stages, groups, orders)
    assert minimum == {0: 2, 1: 2, 2: 2, 3: 2}
    assert multiversion_candidate_buffers(graph, stages) == (0,)
    assert tuple(
        enumerate_buffer_version_plans(graph, stages, groups, orders)
    ) == ()


def test_fa3_independently_expands_tma_and_mma_buffers() -> None:
    from history.unionwsp.test.test_order import enumerate_fa3_ordered_schedules

    graph, _, schedules = enumerate_fa3_ordered_schedules()
    all_candidate_ids = set()
    for _, _, stages_by_region, groups, orders in schedules:
        minimum = analyze_buffer_versions(
            graph, stages_by_region, groups, orders
        )
        candidate_ids = multiversion_candidate_buffers(graph, stages_by_region)
        all_candidate_ids.update(candidate_ids)
        plans = tuple(
            enumerate_buffer_version_plans(
                graph, stages_by_region, groups, orders
            )
        )
        has_unrealizable_fragment = any(
            buffer.scope == "local.fragment"
            and minimum[buffer.buffer_id] > 1
            for buffer in graph.buffers
        )
        if has_unrealizable_fragment:
            assert plans == ()
            continue
        assert len(plans) == 1 << len(candidate_ids)
        assert plans[0] == minimum
        for plan in plans:
            assert all(
                plan[buffer_id] == minimum[buffer_id]
                for buffer_id in minimum
                if buffer_id not in candidate_ids
            )
            assert all(
                plan[buffer_id]
                in {minimum[buffer_id], minimum[buffer_id] + 1}
                for buffer_id in candidate_ids
            )
    assert all_candidate_ids == {6, 13}


def enumerate_fa3_version_plans():
    from history.unionwsp.test.test_order import enumerate_fa3_ordered_schedules

    graph, stage_assignments, schedules = enumerate_fa3_ordered_schedules()
    plans = tuple(
        (
            result_index,
            stage_index,
            group_index,
            multiversion_buffers(
                graph,
                stages_by_region,
                groups,
                orders,
            ),
        )
        for result_index, (
            stage_index,
            group_index,
            stages_by_region,
            groups,
            orders,
        ) in enumerate(schedules, start=1)
    )
    return graph, stage_assignments, schedules, plans


def print_fa3_version_plans(graph, stage_assignments, plans) -> None:
    print(
        f"\nFA3 multiversion analysis: "
        f"stage_assignments={len(stage_assignments)}, schedules={len(plans)}"
    )
    for result_index, stage_index, group_index, versions in plans:
        buffers = ", ".join(
            f"{buffer_id}:{graph.buffer_for_id(buffer_id).name}x{count}"
            for buffer_id, count in versions.items()
        )
        print(
            f"result {result_index:03d}: stage_assignment={stage_index:03d}, "
            f"group_assignment={group_index:02d}, "
            f"multiversion=[{buffers}]"
        )


def print_fa3_enumerated_version_plans(
    schedule_limit: int | None = None,
) -> int:
    """Print every independent TMA/MMA version choice for FA3 schedules."""

    from history.unionwsp.test.test_order import enumerate_fa3_ordered_schedules

    graph, stage_assignments, schedules = enumerate_fa3_ordered_schedules()
    schedule_plans = []
    all_candidate_ids = set()
    total_plans = 0
    for schedule in schedules:
        _, _, stages_by_region, groups, orders = schedule
        minimum = analyze_buffer_versions(
            graph, stages_by_region, groups, orders
        )
        candidate_ids = multiversion_candidate_buffers(graph, stages_by_region)
        plans = tuple(
            enumerate_buffer_version_plans(
                graph, stages_by_region, groups, orders
            )
        )
        schedule_plans.append((schedule, candidate_ids, plans))
        all_candidate_ids.update(candidate_ids)
        total_plans += len(plans)
    selected_schedule_plans = (
        schedule_plans
        if schedule_limit is None
        else schedule_plans[:schedule_limit]
    )
    candidate_names = ", ".join(
        graph.buffer_for_id(buffer_id).name
        for buffer_id in sorted(all_candidate_ids)
    )
    print(
        f"\nFA3 independent multiversion enumeration: "
        f"stage_assignments={len(stage_assignments)}, "
        f"schedules={len(schedules)}, "
        f"eligible_buffers=[{candidate_names}], "
        f"version_plans={total_plans}"
    )

    printed = 0
    for schedule_index, (
        schedule,
        candidate_ids,
        plans,
    ) in enumerate(selected_schedule_plans, start=1):
        stage_index, group_index, _, _, _ = schedule
        plans_per_schedule = len(plans)
        for plan_index, versions in enumerate(plans, start=1):
            candidate_versions = ", ".join(
                f"{buffer_id}:{graph.buffer_for_id(buffer_id).name}x"
                f"{versions[buffer_id]}"
                for buffer_id in candidate_ids
            )
            other_multiversion = ", ".join(
                f"{buffer_id}:{graph.buffer_for_id(buffer_id).name}x{count}"
                for buffer_id, count in versions.items()
                if buffer_id not in candidate_ids and count > 1
            )
            print(
                f"schedule {schedule_index:03d}: "
                f"stage_assignment={stage_index:03d}, "
                f"group_assignment={group_index:02d}, "
                f"version_plan={plan_index:02d}/{plans_per_schedule:02d}, "
                f"candidate_versions=[{candidate_versions}], "
                f"other_multiversion=[{other_multiversion}]"
            )
            printed += 1
    return printed


def test_every_fa3_schedule_reports_its_multiversion_buffers() -> None:
    graph, stage_assignments, schedules, plans = enumerate_fa3_version_plans()
    print_fa3_version_plans(graph, stage_assignments, plans)

    assert stage_assignments
    assert len(schedules) == len(plans) > 0
    assert any(plan for _, _, _, plan in plans)
    assert all(
        count == 2
        for _, _, _, plan in plans
        for count in plan.values()
    )


if __name__ == "__main__":
    print_fa3_enumerated_version_plans()
