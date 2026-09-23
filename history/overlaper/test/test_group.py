"""FA3 group-enumeration regression tests for Overlaper."""

import pytest

from history.overlaper.analysis.schedule import (
    MAX_NUM_GROUPS,
    build_group_components,
    build_group_opportunities,
    enumerate_bounded_group_assignments,
    enumerate_group_assignments,
    is_group_opportunity,
)
from history.overlaper.headware import HOPPER
from history.overlaper.parse import extract_dataflow_graph
from history.overlaper.test.test_extractor import make_fa3_prim_func


def print_group_assignments(graph, assignments_by_count) -> None:
    opportunities = build_group_opportunities(graph)
    print(
        f"\nFA3 whole-graph group enumeration: "
        f"components={len(build_group_components(graph))}, "
        f"opportunities={len(opportunities)}, "
        f"max_num_groups={MAX_NUM_GROUPS}"
    )
    print("opportunity edges:")
    for edge in opportunities:
        buffer = graph.buffer_for_id(edge.buffer_id)
        print(
            f"  {edge.producer_id}:{graph.node_for_id(edge.producer_id).name} "
            f"-> {edge.consumer_id}:{graph.node_for_id(edge.consumer_id).name} "
            f"buffer={buffer.buffer_id}:{buffer.name}[{buffer.scope}]"
        )

    for num_groups, assignments in assignments_by_count.items():
        print(f"\nnum_groups={num_groups}, assignments={len(assignments)}")
        for index, assignment in enumerate(assignments, start=1):
            groups = []
            for group_id in range(num_groups):
                operations = ", ".join(
                    f"{node.node_id}:{node.name}[{node.instruction.name}]"
                    for node in graph.nodes
                    if assignment[node.node_id] == group_id
                )
                groups.append(f"group {group_id}=[{operations}]")
            print(f"assignment {index:02d}: " + " | ".join(groups))


def test_extract_and_enumerate_fa3_groups() -> None:
    graph = extract_dataflow_graph(make_fa3_prim_func(), hardware=HOPPER)
    opportunities = build_group_opportunities(graph)
    components = build_group_components(graph)
    assignments_by_count = {
        num_groups: tuple(enumerate_group_assignments(graph, num_groups))
        for num_groups in range(1, MAX_NUM_GROUPS + 1)
    }
    print_group_assignments(graph, assignments_by_count)

    assert tuple(
        (
            edge.producer_id,
            edge.consumer_id,
            graph.buffer_for_id(edge.buffer_id).name,
        )
        for edge in opportunities
    ) == (
        (0, 6, "Q_shared"),
        (4, 6, "K_shared"),
        (17, 18, "V_shared"),
        (20, 21, "O_shared"),
    )
    assert components == (
        (0,),
        (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20),
        (4,),
        (17,),
        (21,),
    )
    assert {
        num_groups: len(assignments)
        for num_groups, assignments in assignments_by_count.items()
    } == {1: 1, 2: 15, 3: 25}

    all_node_ids = {node.node_id for node in graph.nodes}
    for num_groups, assignments in assignments_by_count.items():
        for assignment in assignments:
            assert set(assignment) == all_node_ids
            assert set(assignment.values()) == set(range(num_groups))
            for edge in graph.edges:
                if not is_group_opportunity(graph, edge):
                    assert (
                        assignment[edge.producer_id]
                        == assignment[edge.consumer_id]
                    )
            if num_groups > 1:
                participating_groups = {
                    assignment[node_id]
                    for edge in opportunities
                    if assignment[edge.producer_id]
                    != assignment[edge.consumer_id]
                    for node_id in (edge.producer_id, edge.consumer_id)
                }
                assert participating_groups == set(range(num_groups))


def test_group_count_is_bounded_by_three() -> None:
    graph = extract_dataflow_graph(make_fa3_prim_func(), hardware=HOPPER)
    with pytest.raises(ValueError, match="between 1 and 3"):
        tuple(enumerate_group_assignments(graph, 0))
    with pytest.raises(ValueError, match="between 1 and 3"):
        tuple(enumerate_group_assignments(graph, 4))


def test_bounded_group_enumeration_obeys_beam_and_constraints() -> None:
    graph = extract_dataflow_graph(make_fa3_prim_func(), hardware=HOPPER)
    assignments = tuple(
        enumerate_bounded_group_assignments(
            graph,
            3,
            beam_width=6,
            score=lambda groups: sum(groups.values()),
        )
    )

    assert 0 < len(assignments) <= 6
    assert all(set(item.values()) == {0, 1, 2} for item in assignments)
