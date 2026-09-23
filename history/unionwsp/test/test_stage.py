"""Extract the FA3 graph, enumerate every stage assignment, and print it."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "3rdparty" / "tvm" / "python"))
os.environ.setdefault("TVM_LIBRARY_PATH", str(PROJECT_ROOT / "build" / "lib"))

from tvm.target import Target

from history.unionwsp.hardware import HOPPER
from history.unionwsp.parseIR import (
    DataflowGraph,
    RegionKind,
    extract_dataflow_graph,
)
from history.unionwsp.schedule import enumerate_stage_assignments
from history.unionwsp.test.test_extractor import flashattn


NUM_STAGES = 3


def make_fa3_prim_func():
    return flashattn.jit_impl.get_tir(
        1,
        1,
        256,
        256,
        64,
        False,
        block_M=128,
        block_N=128,
        threads=256,
        auto_wsp=True,
    )


def hopper_target() -> Target:
    return Target({"kind": "cuda", "arch": "sm_90a"})


def print_stage_assignments(graph, region_id, assignments) -> None:
    nodes = graph.nodes_for_region(region_id)
    print(
        f"\nFA3 pipeline region {region_id}: {len(nodes)} nodes, "
        f"num_stages={NUM_STAGES}, assignments={len(assignments)}"
    )
    for index, assignment in enumerate(assignments, start=1):
        groups = []
        for stage in range(NUM_STAGES):
            operations = ", ".join(
                f"{node.node_id}:{node.name}[{node.instruction_kind.value}]"
                for node in nodes
                if assignment[node.node_id] == stage
            )
            groups.append(f"stage {stage}=[{operations}]")
        print(f"assignment {index:03d}: " + " | ".join(groups))


def test_enumerate_all_fa3_stage_assignments() -> None:
    graph = extract_dataflow_graph(
        make_fa3_prim_func(), target=hopper_target()
    )
    pipeline_regions = tuple(
        region_id
        for region_id, kind in enumerate(graph.region_kinds)
        if kind == RegionKind.PIPELINE
    )
    assert pipeline_regions == (1,)

    region_id = pipeline_regions[0]
    assignments = list(
        enumerate_stage_assignments(graph, region_id, NUM_STAGES)
    )

    print_stage_assignments(graph, region_id, assignments)

    node_ids = {
        node.node_id for node in graph.nodes_for_region(region_id)
    }
    edges = tuple(
        edge
        for edge in graph.edges
        if edge.producer_id in node_ids and edge.consumer_id in node_ids
    )
    assert assignments
    for assignment in assignments:
        assert set(assignment) == node_ids
        assignment_depth = max(assignment.values()) + 1
        assert set(assignment.values()) == set(range(assignment_depth))
        for boundary in range(assignment_depth - 1):
            assert any(
                min(
                    assignment[edge.producer_id],
                    assignment[edge.consumer_id],
                )
                <= boundary
                < max(
                    assignment[edge.producer_id],
                    assignment[edge.consumer_id],
                )
                and (
                    graph.is_async(edge.producer_id)
                    or graph.is_async(edge.consumer_id)
                )
                for edge in edges
            )
        for edge in edges:
            producer_stage = assignment[edge.producer_id]
            consumer_stage = assignment[edge.consumer_id]
            assert producer_stage <= (
                consumer_stage + edge.iteration_distance
            )

            producer = graph.node_for_id(edge.producer_id)
            consumer = graph.node_for_id(edge.consumer_id)
            if producer_stage != consumer_stage:
                assert graph.hardware is not None
                assert graph.hardware.allows_stage_split(
                    producer.instruction_kind,
                    consumer.instruction_kind,
                )


if __name__ == "__main__":
    test_enumerate_all_fa3_stage_assignments()
