"""Test whole-program FA3 warp-group enumeration and print every result."""

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
    build_group_components,
    build_ws_opportunities,
    enumerate_group_assignments,
    enumerate_stage_assignments,
    is_legal_group_cut,
    infer_max_groups
)


def _buffer(buffer_id: int, name: str, scope: str) -> BufferDescriptor:
    return BufferDescriptor(buffer_id, name, scope, 128, object())


def _whole_program_graph(
    loop_carried: bool = False,
    producer_kind: InstructionKind = InstructionKind.TMA,
) -> DataflowGraph:
    return DataflowGraph(
        buffers=(
            _buffer(0, "shared", "shared"),
            _buffer(1, "fragment", "local.fragment"),
            _buffer(2, "output", "global"),
        ),
        nodes=(
            DataflowNode(0, 0, "fill_acc", writes=(1,)),
            DataflowNode(
                1,
                1,
                "load",
                writes=(0,),
                instruction_kind=producer_kind,
            ),
            DataflowNode(
                2,
                1,
                "mma",
                reads=(0, 1),
                writes=(1,),
                instruction_kind=InstructionKind.WGMMA,
            ),
            DataflowNode(3, 2, "store", reads=(1,), writes=(2,)),
            DataflowNode(4, 0, "independent_serial"),
        ),
        edges=(
            DataflowEdge(0, 2, buffer_id=1),
            DataflowEdge(
                1,
                2,
                iteration_distance=int(loop_carried),
                buffer_id=0,
            ),
            DataflowEdge(2, 3, buffer_id=1),
        ),
        region_kinds=(
            RegionKind.SERIAL,
            RegionKind.PIPELINE,
            RegionKind.SERIAL,
        ),
        hardware=HOPPER,
    )


def test_pipeline_opportunity_creates_groups_and_serial_ops_attach() -> None:
    graph = _whole_program_graph()
    stages = {1: {1: 0, 2: 1}}

    assert build_ws_opportunities(graph, stages) == ((1, 2),)
    assert build_group_components(graph) == ((0, 2, 3), (1,), (4,))
    assert list(enumerate_group_assignments(graph, stages, 1)) == [
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    ]
    assert list(enumerate_group_assignments(graph, stages, 2)) == [
        {0: 0, 1: 1, 2: 0, 3: 0, 4: 0}
    ]


def test_same_stage_async_producer_can_create_specialized_group() -> None:
    graph = _whole_program_graph()
    stages = {1: {1: 0, 2: 0}}

    assert build_ws_opportunities(graph, stages) == ((1, 2),)
    assert list(enumerate_group_assignments(graph, stages, 2)) == [
        {0: 0, 1: 1, 2: 0, 3: 0, 4: 0}
    ]


def test_same_stage_synchronous_producer_cannot_justify_a_group() -> None:
    graph = _whole_program_graph(producer_kind=InstructionKind.GENERIC)
    stages = {1: {1: 0, 2: 0}}

    assert is_legal_group_cut(graph, graph.edges[1])
    assert build_ws_opportunities(graph, stages) == ()
    assert list(enumerate_group_assignments(graph, stages, 2)) == []


def test_different_stage_nodes_without_an_edge_are_not_opportunities() -> None:
    graph = DataflowGraph(
        buffers=(),
        nodes=(
            DataflowNode(
                0,
                0,
                "independent_load",
                instruction_kind=InstructionKind.TMA,
            ),
            DataflowNode(
                1,
                0,
                "independent_mma",
                instruction_kind=InstructionKind.WGMMA,
            ),
        ),
        edges=(),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
    )
    stages = {0: {0: 0, 1: 1}}

    assert build_ws_opportunities(graph, stages) == ()
    assert list(enumerate_group_assignments(graph, stages, 2)) == []


def test_cross_stage_edge_must_be_a_legal_group_cut() -> None:
    def make_graph(
        producer_kind: InstructionKind,
        consumer_kind: InstructionKind,
        scope: str,
    ) -> DataflowGraph:
        return DataflowGraph(
            buffers=(_buffer(0, "value", scope),),
            nodes=(
                DataflowNode(
                    0,
                    0,
                    "producer",
                    writes=(0,),
                    instruction_kind=producer_kind,
                ),
                DataflowNode(
                    1,
                    0,
                    "consumer",
                    reads=(0,),
                    instruction_kind=consumer_kind,
                ),
            ),
            edges=(DataflowEdge(0, 1, buffer_id=0),),
            region_kinds=(RegionKind.PIPELINE,),
            hardware=HOPPER,
        )

    stages = {0: {0: 0, 1: 1}}
    unsupported_pair = make_graph(
        InstructionKind.WGMMA,
        InstructionKind.FUNCTION,
        "shared",
    )
    invisible_buffer = make_graph(
        InstructionKind.TMA,
        InstructionKind.WGMMA,
        "local.fragment",
    )

    for graph in (unsupported_pair, invisible_buffer):
        assert not is_legal_group_cut(graph, graph.edges[0])
        assert build_ws_opportunities(graph, stages) == ()
        assert build_group_components(graph) == ((0, 1),)


def test_group_split_need_not_realize_every_stage_boundary() -> None:
    graph = DataflowGraph(
        buffers=(
            _buffer(0, "shared", "shared"),
            _buffer(1, "fragment", "local.fragment"),
        ),
        nodes=(
            DataflowNode(
                0,
                0,
                "load",
                writes=(0,),
                instruction_kind=InstructionKind.TMA,
            ),
            DataflowNode(
                1,
                0,
                "mma",
                reads=(0,),
                writes=(1,),
                instruction_kind=InstructionKind.WGMMA,
            ),
            DataflowNode(
                2,
                0,
                "consumer",
                reads=(1,),
                instruction_kind=InstructionKind.GENERIC,
            ),
        ),
        edges=(
            DataflowEdge(0, 1, buffer_id=0),
            DataflowEdge(1, 2, buffer_id=1),
        ),
        region_kinds=(RegionKind.PIPELINE,),
        hardware=HOPPER,
    )
    stages = {0: {0: 0, 1: 0, 2: 1}}

    assert build_ws_opportunities(graph, stages) == ((0, 1),)
    assert build_group_components(graph) == ((0,), (1, 2))
    assert list(enumerate_group_assignments(graph, stages, 2)) == [
        {0: 1, 1: 0, 2: 0}
    ]


def test_loop_carried_edge_can_create_opportunity_at_the_same_stage() -> None:
    graph = _whole_program_graph(loop_carried=True)
    stages = {1: {1: 0, 2: 0}}

    assert build_ws_opportunities(graph, stages) == ((1, 2),)
    assert list(enumerate_group_assignments(graph, stages, 2)) == [
        {0: 0, 1: 1, 2: 0, 3: 0, 4: 0}
    ]


def test_serial_tma_store_can_create_a_standalone_group() -> None:
    graph = DataflowGraph(
        buffers=(
            _buffer(0, "input_shared", "shared"),
            _buffer(1, "acc", "local.fragment"),
            _buffer(2, "output_shared", "shared"),
            _buffer(3, "output", "global"),
        ),
        nodes=(
            DataflowNode(
                0,
                0,
                "load",
                writes=(0,),
                instruction_kind=InstructionKind.TMA,
            ),
            DataflowNode(
                1,
                0,
                "mma",
                reads=(0,),
                writes=(1,),
                instruction_kind=InstructionKind.WGMMA,
            ),
            DataflowNode(
                2,
                1,
                "copy_to_shared",
                reads=(1,),
                writes=(2,),
                instruction_kind=InstructionKind.RSCP,
            ),
            DataflowNode(
                3,
                1,
                "store",
                reads=(2,),
                writes=(3,),
                instruction_kind=InstructionKind.TMA,
            ),
        ),
        edges=(
            DataflowEdge(0, 1, buffer_id=0),
            DataflowEdge(1, 2, buffer_id=1),
            DataflowEdge(2, 3, buffer_id=2),
        ),
        region_kinds=(RegionKind.PIPELINE, RegionKind.SERIAL),
        hardware=HOPPER,
    )
    stages = {0: {0: 0, 1: 0}}

    assert build_ws_opportunities(graph, stages) == ((0, 1),)
    assert is_legal_group_cut(graph, graph.edges[2])
    assert build_group_components(graph) == ((0,), (1, 2), (3,))
    assignments = list(enumerate_group_assignments(graph, stages, 2))
    assert {tuple(assignment.items()) for assignment in assignments} == {
        tuple({0: 0, 1: 0, 2: 0, 3: 1}.items()),
        tuple({0: 1, 1: 0, 2: 0, 3: 0}.items()),
        tuple({0: 1, 1: 0, 2: 0, 3: 1}.items()),
    }


def enumerate_fa3_group_assignments():
    from history.unionwsp.parseIR import extract_dataflow_graph
    from history.unionwsp.test.test_stage import hopper_target, make_fa3_prim_func

    graph = extract_dataflow_graph(
        make_fa3_prim_func(), target=hopper_target()
    )
    stage_assignments = tuple(enumerate_stage_assignments(graph, 1, 3))
    # print(infer_max_groups(graph))
    feasible = []
    for stage_index, stages in enumerate(stage_assignments, start=1):
        stages_by_region = {1: stages}
        groups_for_stage = tuple(
            enumerate_group_assignments(graph, stages_by_region, 3)
        )
        for group_index, groups in enumerate(groups_for_stage, start=1):
            feasible.append(
                (stage_index, group_index, stages_by_region, groups)
            )
    return graph, stage_assignments, tuple(feasible)


def print_fa3_group_assignments(graph, stage_assignments, feasible) -> None:
    print(
        f"\nFA3 whole-program group enumeration: "
        f"stage_assignments={len(stage_assignments)}, "
        f"feasible_group_assignments={len(feasible)}, num_groups=2"
    )
    for result_index, (
        stage_index,
        group_index,
        stages_by_region,
        groups,
    ) in enumerate(feasible, start=1):
        print(
            f"result {result_index:03d}: stage_assignment={stage_index:03d}, "
            f"group_assignment={group_index:02d}"
        )

        def stage_label(node):
            return stages_by_region.get(node.region_id, {}).get(node.node_id, "-")

        for group_id in range(2):
            operations = ", ".join(
                f"{node.node_id}:{node.name}"
                f"[{node.instruction_kind.value},region={node.region_id},"
                f"stage={stage_label(node)}]"
                for node in graph.nodes
                if groups[node.node_id] == group_id
            )
            print(f"  group {group_id}=[{operations}]")


def test_fa3_stage_assignments_feed_whole_program_groups() -> None:
    graph, stage_assignments, feasible = enumerate_fa3_group_assignments()
    print_fa3_group_assignments(graph, stage_assignments, feasible)

    assert stage_assignments
    assert feasible
    all_node_ids = {node.node_id for node in graph.nodes}
    for _, _, stages_by_region, groups in feasible:
        assert set(groups) == all_node_ids
        assert set(groups.values()) == {0, 1}
        opportunities = build_ws_opportunities(graph, stages_by_region)
        participating_groups = {
            groups[node_id]
            for left, right in opportunities
            if groups[left] != groups[right]
            for node_id in (left, right)
        }
        pipeline_groups = {
            groups[node.node_id]
            for node in graph.nodes
            if graph.region_kinds[node.region_id] == RegionKind.PIPELINE
        }
        standalone_groups = {
            group_id
            for group_id in range(2)
            if group_id not in pipeline_groups
            and any(
                groups[node.node_id] == group_id
                and graph.region_kinds[node.region_id] == RegionKind.SERIAL
                and graph.hardware is not None
                and graph.hardware.can_own_standalone_group(
                    node.instruction_kind
                )
                for node in graph.nodes
            )
        }
        assert 1 in participating_groups | standalone_groups
        for edge in graph.edges:
            if groups[edge.producer_id] != groups[edge.consumer_id]:
                assert is_legal_group_cut(graph, edge)


if __name__ == "__main__":
    test_fa3_stage_assignments_feed_whole_program_groups()
