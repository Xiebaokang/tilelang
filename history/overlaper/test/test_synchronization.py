"""Cross-group synchronization tests, including an FA3 schedule."""

import pytest

from history.overlaper.analysis.schedule import (
    SynchronizationCycleError,
    SynchronizationKind,
    SynchronizationScope,
    build_program_orders,
    build_synchronizations,
    enumerate_buffer_version_plans,
    enumerate_group_assignments,
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


def _two_node_graph() -> DataflowGraph:
    return DataflowGraph(
        buffers=(BufferDescriptor(0, "shared", "shared", 128, object()),),
        nodes=(
            DataflowNode(0, 0, "writer", TMA, writes=(0,)),
            DataflowNode(1, 0, "reader", WGMMA, reads=(0,)),
        ),
        edges=(DataflowEdge(0, 1, buffer_id=0),),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
    )


def test_same_group_non_tma_handoff_needs_no_synchronization() -> None:
    graph = _two_node_graph()
    stages = {0: {0: 0, 1: 1}}
    groups = {0: 0, 1: 0}
    orders = build_program_orders(graph, stages, groups)

    assert build_synchronizations(
        graph, stages, groups, orders, versions={0: 2}
    ) == ()


def test_same_group_tma_handoff_has_forward_synchronization() -> None:
    """A same-group TMA copy still needs async completion visibility."""
    graph = DataflowGraph(
        buffers=(
            BufferDescriptor(0, "input", "global", 128, object()),
            BufferDescriptor(1, "shared", "shared", 128, object()),
        ),
        nodes=(
            DataflowNode(0, 0, "copy", TMA, reads=(0,), writes=(1,)),
            DataflowNode(1, 0, "consumer", WGMMA, reads=(1,)),
        ),
        edges=(DataflowEdge(0, 1, buffer_id=1),),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
    )
    stages = {0: {0: 0, 1: 1}}
    groups = {0: 0, 1: 0}
    orders = build_program_orders(graph, stages, groups)

    synchronizations = build_synchronizations(
        graph, stages, groups, orders, versions={0: 1, 1: 1}
    )

    assert len(synchronizations) == 1
    item = synchronizations[0]
    assert item.kind == SynchronizationKind.FORWARD_DEPENDENCY
    assert item.scope == SynchronizationScope.PER_ITERATION
    assert (item.producer_group, item.consumer_group) == (0, 0)
    assert item.buffer_id == 1


def test_cross_group_has_forward_and_reuse_synchronization() -> None:
    graph = _two_node_graph()
    stages = {0: {0: 0, 1: 0}}
    groups = {0: 0, 1: 1}
    orders = build_program_orders(graph, stages, groups)

    synchronizations = build_synchronizations(
        graph, stages, groups, orders, versions={0: 1}
    )

    assert tuple(item.kind for item in synchronizations) == (
        SynchronizationKind.FORWARD_DEPENDENCY,
        SynchronizationKind.BUFFER_REUSE,
    )
    assert all(
        item.scope == SynchronizationScope.PER_ITERATION
        for item in synchronizations
    )
    assert (
        synchronizations[0].producer_id,
        synchronizations[0].consumer_id,
        synchronizations[0].effective_stage_distance,
    ) == (0, 1, 0)
    assert (
        synchronizations[1].producer_id,
        synchronizations[1].consumer_id,
        synchronizations[1].effective_stage_distance,
    ) == (1, 0, 1)


def test_same_epoch_cross_group_cycle_is_rejected() -> None:
    graph = DataflowGraph(
        buffers=(
            BufferDescriptor(0, "first", "shared", 128, object()),
            BufferDescriptor(1, "second", "shared", 128, object()),
        ),
        nodes=(
            DataflowNode(0, 0, "first_writer", TMA, writes=(0,)),
            DataflowNode(1, 0, "second_writer", TMA, writes=(1,)),
            DataflowNode(2, 0, "reader", WGMMA, reads=(0, 1)),
        ),
        edges=(
            DataflowEdge(0, 2, buffer_id=0),
            DataflowEdge(1, 2, buffer_id=1),
        ),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
    )
    stages = {0: {0: 0, 1: 1, 2: 1}}
    groups = {0: 0, 1: 0, 2: 1}
    orders = build_program_orders(graph, stages, groups)

    # Same epoch: reuse 2->0, local order 0->1, and forward dependency 1->2.
    with pytest.raises(SynchronizationCycleError):
        build_synchronizations(
            graph, stages, groups, orders, versions={0: 1, 1: 1}
        )


def _format_synchronization(graph, item) -> str:
    buffer = graph.buffer_for_id(item.buffer_id)
    distance = (
        "-"
        if item.effective_stage_distance is None
        else item.effective_stage_distance
    )
    return (
        f"{item.kind.value}: {item.producer_id}(g{item.producer_group}) -> "
        f"{item.consumer_id}(g{item.consumer_group}), "
        f"buffer={buffer.name}, scope={item.scope.value}, "
        f"iteration={item.iteration_distance}, distance={distance}, "
        f"slots={item.slot_count}"
    )


def test_fa3_cross_group_synchronization() -> None:
    graph = extract_dataflow_graph(make_fa3_prim_func(), hardware=HOPPER)
    stages = {1: find_fa3_stage_assignment(graph)}
    # This assignment isolates the second K/V TMA producer (node 17), making
    # K/V_shared cross a group boundary inside the pipeline region.
    groups = tuple(enumerate_group_assignments(graph, 2))[1]
    orders = build_program_orders(graph, stages, groups)
    versions = next(
        enumerate_buffer_version_plans(graph, stages, groups, orders)
    )
    synchronizations = build_synchronizations(
        graph, stages, groups, orders, versions
    )

    forward = tuple(
        item
        for item in synchronizations
        if item.kind == SynchronizationKind.FORWARD_DEPENDENCY
    )
    # Besides the V cross-group handoff, same-group Q/K TMA copies need an
    # async-completion channel before WGMMA can consume their shared outputs.
    assert {
        (item.producer_id, item.consumer_id, item.buffer_id)
        for item in forward
    } == {
        (0, 6, 1),
        (4, 6, 6),
        (17, 18, 13),
    }
    assert all(
        item.producer_group != item.consumer_group
        for item in synchronizations
        if item.kind == SynchronizationKind.BUFFER_REUSE
    )
    assert {
        (
            item.kind,
            item.producer_id,
            item.consumer_id,
            item.buffer_id,
            item.effective_stage_distance,
        )
        for item in synchronizations
    } == {
        (SynchronizationKind.FORWARD_DEPENDENCY, 0, 6, 1, None),
        (SynchronizationKind.FORWARD_DEPENDENCY, 4, 6, 6, 2),
        (SynchronizationKind.FORWARD_DEPENDENCY, 17, 18, 13, 1),
        (SynchronizationKind.BUFFER_REUSE, 18, 17, 13, 1),
    }

    print("\nFA3 cross-group synchronization:")
    for item in synchronizations:
        print("  " + _format_synchronization(graph, item))
