"""Test deterministic priority order selection, including the full FA3 path."""

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
from history.unionwsp.schedule import build_program_orders, effective_stage_distance


def _buffer(buffer_id: int, name: str, scope: str) -> BufferDescriptor:
    return BufferDescriptor(buffer_id, name, scope, 128, object())


def test_ready_nodes_use_priority_but_serial_keeps_ir_order() -> None:
    graph = DataflowGraph(
        buffers=(_buffer(0, "shared", "shared"),),
        nodes=(
            DataflowNode(0, 0, "generic"),
            DataflowNode(
                1,
                0,
                "load",
                writes=(0,),
                instruction_kind=InstructionKind.TMA,
            ),
            DataflowNode(
                2,
                0,
                "mma",
                reads=(0,),
                instruction_kind=InstructionKind.WGMMA,
            ),
            DataflowNode(3, 1, "serial_generic"),
            DataflowNode(
                4,
                1,
                "serial_tma",
                instruction_kind=InstructionKind.TMA,
            ),
        ),
        edges=(DataflowEdge(1, 2, buffer_id=0),),
        region_kinds=(RegionKind.PIPELINE, RegionKind.SERIAL),
        hardware=HOPPER,
    )
    stages = {0: {0: 0, 1: 0, 2: 0}}
    groups = {node.node_id: 0 for node in graph.nodes}

    orders = build_program_orders(graph, stages, groups)

    assert orders == {
        0: {0: {1: 0, 2: 1, 0: 2}},
        1: {0: {3: 0, 4: 1}},
    }


def enumerate_fa3_ordered_schedules():
    from history.unionwsp.test.test_warp_specialization import (
        enumerate_fa3_group_assignments,
    )

    graph, stage_assignments, feasible_groups = (
        enumerate_fa3_group_assignments()
    )
    schedules = tuple(
        (
            stage_index,
            group_index,
            stages_by_region,
            groups,
            build_program_orders(graph, stages_by_region, groups),
        )
        for stage_index, group_index, stages_by_region, groups in feasible_groups
    )
    return graph, stage_assignments, schedules


def print_fa3_ordered_schedules(graph, stage_assignments, schedules) -> None:
    print(
        f"\nFA3 final schedules: stage_assignments={len(stage_assignments)}, "
        f"ordered_schedules={len(schedules)}"
    )
    for result_index, (
        stage_index,
        group_index,
        stages_by_region,
        groups,
        orders,
    ) in enumerate(schedules, start=1):
        print(
            f"result {result_index:03d}: stage_assignment={stage_index:03d}, "
            f"group_assignment={group_index:02d}"
        )
        for region_id, group_orders in orders.items():
            region_stages = stages_by_region.get(region_id, {})
            for group_id, local_order in group_orders.items():
                operations = ", ".join(
                    f"{order}:{node_id}:{graph.node_for_id(node_id).name}"
                    f"[{graph.node_for_id(node_id).instruction_kind.value},"
                    f"priority={graph.issue_priority(node_id)},"
                    f"stage={region_stages.get(node_id, '-')}]"
                    for node_id, order in sorted(
                        local_order.items(), key=lambda item: item[1]
                    )
                )
                print(
                    f"  region {region_id} group {group_id}=[{operations}]"
                )


def _combined_region_order_is_acyclic(
    graph,
    region_id,
    stages_by_region,
    group_orders,
) -> bool:
    nodes = graph.nodes_for_region(region_id)
    node_ids = {node.node_id for node in nodes}
    predecessors = {node_id: set() for node_id in node_ids}
    for edge in graph.edges:
        if edge.producer_id not in node_ids or edge.consumer_id not in node_ids:
            continue
        if graph.region_kinds[region_id] == RegionKind.PIPELINE:
            distance = effective_stage_distance(
                edge,
                stages_by_region[region_id],
            )
            if distance != 0:
                continue
        if edge.producer_id != edge.consumer_id:
            predecessors[edge.consumer_id].add(edge.producer_id)
    for local_order in group_orders.values():
        ordered = sorted(local_order, key=local_order.__getitem__)
        for earlier, later in zip(ordered, ordered[1:]):
            predecessors[later].add(earlier)

    pending = dict(predecessors)
    while pending:
        ready = [node_id for node_id, required in pending.items() if not required]
        if not ready:
            return False
        for node_id in ready:
            del pending[node_id]
        for required in pending.values():
            required.difference_update(ready)
    return True


def test_fa3_stage_group_then_priority_order() -> None:
    graph, stage_assignments, schedules = enumerate_fa3_ordered_schedules()
    print_fa3_ordered_schedules(graph, stage_assignments, schedules)

    assert stage_assignments
    assert schedules
    for _, _, stages_by_region, groups, orders in schedules:
        assert set(orders) == set(range(len(graph.region_kinds)))
        for region_id, group_orders in orders.items():
            assert set(group_orders) == set(groups.values())
            region_node_ids = {
                node.node_id for node in graph.nodes_for_region(region_id)
            }
            covered = set()
            for group_id, local_order in group_orders.items():
                expected = {
                    node_id
                    for node_id in region_node_ids
                    if groups[node_id] == group_id
                }
                assert set(local_order) == expected
                assert set(local_order.values()) == set(range(len(expected)))
                covered.update(local_order)
            assert covered == region_node_ids
            assert _combined_region_order_is_acyclic(
                graph,
                region_id,
                stages_by_region,
                group_orders,
            )


if __name__ == "__main__":
    test_fa3_stage_group_then_priority_order()
