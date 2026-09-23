"""Multiversion regression tests for cross-group buffer edges."""

from history.overlaper.analysis.schedule import (
    analyze_buffer_versions,
    build_program_orders,
    enumerate_buffer_version_plans,
    enumerate_group_assignments,
    multiversion_candidate_buffers,
)
from history.overlaper.headware import HOPPER
from history.overlaper.headware.hopper import TMA, WGMMA
from history.overlaper.parse import (
    BufferDescriptor,
    DataflowEdge,
    DataflowGraph,
    DataflowNode,
    RegionKind,
    extract_dataflow_graph,
)
from history.overlaper.test.test_extractor import make_fa3_prim_func
from history.overlaper.test.test_stage import find_fa3_stage_assignment


def _buffer(buffer_id: int, name: str, scope: str) -> BufferDescriptor:
    return BufferDescriptor(buffer_id, name, scope, 128, object())


def _two_node_graph() -> DataflowGraph:
    return DataflowGraph(
        buffers=(_buffer(0, "shared", "shared"),),
        nodes=(
            DataflowNode(0, 0, "writer", TMA, writes=(0,)),
            DataflowNode(1, 0, "reader", WGMMA, reads=(0,)),
        ),
        edges=(DataflowEdge(0, 1, buffer_id=0),),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
    )


def test_cross_stage_same_group_does_not_enumerate_minimum_plus_one() -> None:
    graph = _two_node_graph()
    stages = {0: {0: 0, 1: 1}}
    groups = {0: 0, 1: 0}
    orders = build_program_orders(graph, stages, groups)

    minimum = analyze_buffer_versions(graph, stages, groups, orders)
    assert minimum == {0: 2}
    assert multiversion_candidate_buffers(graph, groups) == ()
    assert tuple(
        enumerate_buffer_version_plans(graph, stages, groups, orders)
    ) == ({0: 2},)


def test_cross_group_same_stage_enumerates_minimum_plus_one() -> None:
    graph = _two_node_graph()
    stages = {0: {0: 0, 1: 0}}
    groups = {0: 0, 1: 1}
    orders = build_program_orders(graph, stages, groups)

    minimum = analyze_buffer_versions(graph, stages, groups, orders)
    assert minimum == {0: 2}
    assert multiversion_candidate_buffers(graph, groups) == (0,)
    assert tuple(
        enumerate_buffer_version_plans(graph, stages, groups, orders)
    ) == ({0: 2}, {0: 3})


def test_fa3_cross_group_multiversion_candidates() -> None:
    graph = extract_dataflow_graph(make_fa3_prim_func(), hardware=HOPPER)
    group_assignments = tuple(enumerate_group_assignments(graph, 2))

    all_candidates = {
        buffer_id
        for groups in group_assignments
        for buffer_id in multiversion_candidate_buffers(graph, groups)
    }
    assert all_candidates == {6, 13}

    # This combination crosses only the serial O_shared edge. The communication
    # still needs a region-boundary synchronization, but O_shared cannot be
    # pipeline-multiversioned.
    stages_by_region = {1: find_fa3_stage_assignment(graph)}
    groups = group_assignments[0]
    orders = build_program_orders(graph, stages_by_region, groups)
    minimum = analyze_buffer_versions(
        graph, stages_by_region, groups, orders
    )
    candidates = multiversion_candidate_buffers(graph, groups)
    plans = tuple(
        enumerate_buffer_version_plans(
            graph, stages_by_region, groups, orders
        )
    )

    print(
        f"\nFA3 multiversion: candidates={candidates}, "
        f"minimum={minimum}, plans={plans}"
    )
    assert candidates == ()
    assert minimum[14] == 1
    assert plans == (minimum,)


def test_serial_cross_group_buffer_is_not_multiversioned() -> None:
    graph = DataflowGraph(
        buffers=(_buffer(0, "shared", "shared"),),
        nodes=(
            DataflowNode(0, 0, "writer", TMA, writes=(0,)),
            DataflowNode(1, 0, "reader", WGMMA, reads=(0,)),
        ),
        edges=(DataflowEdge(0, 1, buffer_id=0),),
        region_kinds=(RegionKind.SERIAL,),
        hardware=HOPPER,
    )
    groups = {0: 0, 1: 1}
    orders = build_program_orders(graph, {}, groups)

    assert multiversion_candidate_buffers(graph, groups) == ()
    assert analyze_buffer_versions(graph, {}, groups, orders) == {0: 1}
