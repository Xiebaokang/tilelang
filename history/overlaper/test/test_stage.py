"""FA3 stage-enumeration regression tests for Overlaper."""

from history.overlaper.analysis.schedule import (
    NUM_STAGES,
    can_split_stages,
    enumerate_bounded_stage_assignments,
    enumerate_stage_assignments,
)
from history.overlaper.headware import HOPPER
from history.overlaper.headware.hopper import FUNCTION, GENERIC, GSCP, RRCP, TMA, WGMMA
from history.overlaper.headware.spec import InstructionType
from history.overlaper.parse import RegionKind, extract_dataflow_graph
from history.overlaper.test.test_extractor import make_fa3_prim_func


FA3_THREE_STAGE_ASSIGNMENT = {
    4: 0,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 1,
    18: 2,
}

FA3_TRANSACTION_STAGE_ASSIGNMENT = {
    4: 0,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 1,
    12: 1,
    13: 2,
    14: 2,
    15: 1,
    16: 1,
    17: 0,
    18: 1,
}

FA3_QK_PV_STAGE_ASSIGNMENT = {
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 1,
    18: 1,
}


def find_fa3_stage_assignment(graph, expected=FA3_THREE_STAGE_ASSIGNMENT):
    """Find a named FA3 fixture without depending on enumeration order."""

    return next(
        assignment
        for assignment in enumerate_stage_assignments(graph, 1)
        if assignment == expected
    )


def print_stage_assignments(graph, region_id, assignments) -> None:
    nodes = graph.nodes_for_region(region_id)
    print(
        f"\nFA3 pipeline region {region_id}: {len(nodes)} nodes, "
        f"num_stages={NUM_STAGES}, assignments={len(assignments)}"
    )
    for index, assignment in enumerate(assignments, start=1):
        stages = []
        for stage in range(max(assignment.values()) + 1):
            operations = ", ".join(
                f"{node.node_id}:{node.name}[{node.instruction.name}]"
                for node in nodes
                if assignment[node.node_id] == stage
            )
            stages.append(f"stage {stage}=[{operations}]")
        print(f"assignment {index:02d}: " + " | ".join(stages))


def test_instruction_stage_split_rules() -> None:
    assert RRCP.type == InstructionType.COMPUTE
    assert can_split_stages(TMA, WGMMA)
    assert can_split_stages(WGMMA, TMA)
    assert can_split_stages(TMA, GSCP)
    assert can_split_stages(GSCP, TMA)
    assert can_split_stages(TMA, RRCP)
    assert can_split_stages(RRCP, TMA)
    assert can_split_stages(GENERIC, FUNCTION)
    assert can_split_stages(FUNCTION, GENERIC)

    assert can_split_stages(RRCP, WGMMA)
    assert can_split_stages(WGMMA, RRCP)
    assert can_split_stages(GENERIC, RRCP)
    assert can_split_stages(RRCP, GENERIC)
    assert can_split_stages(RRCP, FUNCTION)
    assert can_split_stages(FUNCTION, RRCP)
    assert not can_split_stages(RRCP, RRCP)
    assert not can_split_stages(GENERIC, GENERIC)
    assert not can_split_stages(WGMMA, WGMMA)


def test_enumerate_fa3_stage_assignments() -> None:
    graph = extract_dataflow_graph(make_fa3_prim_func(), hardware=HOPPER)
    pipeline_regions = tuple(
        region_id
        for region_id, kind in enumerate(graph.region_kinds)
        if kind == RegionKind.PIPELINE
    )
    assert pipeline_regions == (1,)

    region_id = pipeline_regions[0]
    assignments = list(enumerate_stage_assignments(graph, region_id))
    print_stage_assignments(graph, region_id, assignments)

    nodes = graph.nodes_for_region(region_id)
    node_ids = frozenset(node.node_id for node in nodes)
    edges = tuple(
        edge
        for edge in graph.edges
        if edge.producer_id in node_ids and edge.consumer_id in node_ids
    )

    assert NUM_STAGES == 3
    assignments_by_count = {
        num_stages: [
            assignment
            for assignment in assignments
            if len(set(assignment.values())) == num_stages
        ]
        for num_stages in range(1, NUM_STAGES + 1)
    }
    assert {
        num_stages: len(stage_assignments)
        for num_stages, stage_assignments in assignments_by_count.items()
    } == {1: 1, 2: 38, 3: 408}
    assert len(assignments) == 447
    assert tuple(enumerate_stage_assignments(graph, region_id, 1)) == tuple(
        assignments_by_count[1]
    )
    assert tuple(enumerate_stage_assignments(graph, region_id, 2)) == tuple(
        assignments_by_count[1] + assignments_by_count[2]
    )
    for assignment in assignments:
        assert set(assignment) == node_ids
        used_stages = set(assignment.values())
        assert used_stages == set(range(len(used_stages)))
        for edge in edges:
            producer_stage = assignment[edge.producer_id]
            consumer_stage = assignment[edge.consumer_id]
            assert producer_stage <= (
                consumer_stage + edge.iteration_distance
            )
            if producer_stage != consumer_stage:
                producer = graph.node_for_id(edge.producer_id).instruction
                consumer = graph.node_for_id(edge.consumer_id).instruction
                assert can_split_stages(producer, consumer)

    # TMA/WGMMA and different compute instructions can cross a stage.
    assert any(assignment[4] != assignment[6] for assignment in assignments)
    assert any(assignment[11] != assignment[14] for assignment in assignments)

    # RRCP/non-RRCP compute dependencies can now cross; same-kind compute
    # dependencies remain in one stage.
    assert any(assignment[7] != assignment[8] for assignment in assignments)
    # Register initialization must remain with its direct consumer.
    assert all(assignment[5] == assignment[6] for assignment in assignments)
    assert all(assignment[8] == assignment[9] for assignment in assignments)
    assert any(assignment[15] != assignment[18] for assignment in assignments)

    # The FA3 schedule with K/QK/softmax in stage 0 and V/PV in stage 1 is
    # reachable after allowing RRCP-to-WGMMA stage boundaries.
    assert FA3_QK_PV_STAGE_ASSIGNMENT in assignments


def test_bounded_stage_enumeration_obeys_beam_and_constraints() -> None:
    graph = extract_dataflow_graph(make_fa3_prim_func(), hardware=HOPPER)
    assignments = tuple(
        enumerate_bounded_stage_assignments(
            graph,
            1,
            beam_width=12,
            score=lambda stages: sum(stages.values()),
        )
    )

    assert 0 < len(assignments) <= 3 * 12
    assert all(
        set(item.values()) == set(range(len(set(item.values()))))
        for item in assignments
    )


"""
PYTHONPATH="../../3rdparty/tvm/python:../.." TVM_LIBRARY_PATH="../../build/lib" python -m pytest test_stage.py -q -s > log 2>&1
"""
