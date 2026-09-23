import pytest
from tvm import ir, tirx

from history.unionwsp.hardware import HOPPER
from history.unionwsp.parseIR.graph import (
    BufferAccessKind,
    BufferDescriptor,
    BufferRangeAccess,
    DataflowEdge,
    DataflowGraph,
    DataflowNode,
    DependencyKind,
    InstructionKind,
    RegionKind,
)


def _buffer(buffer_id: int, name: str) -> BufferDescriptor:
    buffer = tirx.decl_buffer((16,), "float16", name=name, scope="shared")
    return BufferDescriptor(buffer_id, name, "shared", 32, buffer)


def test_graph_uses_operation_ids_and_buffer_ids() -> None:
    ranges = (ir.Range.from_min_extent(0, 16),)
    graph = DataflowGraph(
        buffers=(_buffer(0, "shared"),),
        nodes=(
            DataflowNode(0, 1, "load", writes=(0,), instruction_kind=InstructionKind.TMA),
            DataflowNode(1, 1, "compute", reads=(0,), instruction_kind=InstructionKind.WGMMA),
        ),
        edges=(DataflowEdge(0, 1, buffer_id=0),),
        region_kinds=(RegionKind.SERIAL, RegionKind.PIPELINE, RegionKind.SERIAL),
        buffer_accesses=(
            BufferRangeAccess(0, 0, BufferAccessKind.WRITE, ranges),
            BufferRangeAccess(1, 0, BufferAccessKind.READ, ranges),
        ),
    )

    assert graph.node_for_id(1).name == "compute"
    assert graph.buffer_for_id(0).name == "shared"
    assert tuple(node.node_id for node in graph.nodes_for_region(1)) == (0, 1)
    assert graph.accesses_for_node(1) == (graph.buffer_accesses[1],)
    assert graph.topological_order() == (0, 1)


def test_issue_priority_orders_only_ready_nodes() -> None:
    nodes = (
        DataflowNode(0, 0, "generic"),
        DataflowNode(1, 0, "tma", instruction_kind=InstructionKind.TMA),
        DataflowNode(2, 0, "wgmma", instruction_kind=InstructionKind.WGMMA),
    )
    graph = DataflowGraph(
        (), nodes, (), (RegionKind.PIPELINE,), hardware=HOPPER
    )

    assert graph.issue_priority(1) > graph.issue_priority(2)
    assert graph.topological_order() == (1, 2, 0)

    constrained_nodes = (
        DataflowNode(0, 0, "generic", writes=(0,)),
        DataflowNode(
            1,
            0,
            "tma",
            reads=(0,),
            instruction_kind=InstructionKind.TMA,
        ),
        DataflowNode(
            2,
            0,
            "wgmma",
            instruction_kind=InstructionKind.WGMMA,
        ),
    )
    constrained = DataflowGraph(
        (_buffer(0, "shared"),),
        constrained_nodes,
        (DataflowEdge(0, 1, buffer_id=0),),
        (RegionKind.PIPELINE,),
        hardware=HOPPER,
    )
    assert constrained.topological_order() == (2, 0, 1)


def test_instruction_kind_defines_async_behavior() -> None:
    async_kinds = {
        kind
        for kind in InstructionKind
        if kind in HOPPER.supported_instructions and HOPPER.is_async(kind)
    }

    assert async_kinds == {
        InstructionKind.WGMMA,
        InstructionKind.TMA,
        InstructionKind.GSCP,
    }


def test_loop_carried_edge_does_not_create_same_iteration_cycle() -> None:
    graph = DataflowGraph(
        (_buffer(0, "state"),),
        (
            DataflowNode(0, 0, "read", reads=(0,)),
            DataflowNode(1, 0, "write", writes=(0,)),
        ),
        (
            DataflowEdge(0, 1, buffer_id=0),
            DataflowEdge(
                1,
                0,
                iteration_distance=1,
                dependency_kinds=frozenset({DependencyKind.RAW}),
                buffer_id=0,
            ),
        ),
        (RegionKind.PIPELINE,),
    )

    assert graph.topological_order() == (0, 1)
    assert graph.edges[1].is_loop_carried


def test_graph_rejects_invalid_references_and_cycles() -> None:
    with pytest.raises(ValueError, match="unknown buffer"):
        DataflowGraph(
            (),
            (DataflowNode(0, 0, "bad", reads=(0,)),),
            (),
            (RegionKind.PIPELINE,),
        )

    with pytest.raises(ValueError, match="disagrees with node reads/writes"):
        DataflowGraph(
            (_buffer(0, "shared"),),
            (DataflowNode(0, 0, "read", reads=(0,)),),
            (),
            (RegionKind.PIPELINE,),
            (
                BufferRangeAccess(
                    0,
                    0,
                    BufferAccessKind.WRITE,
                    (ir.Range.from_min_extent(0, 16),),
                ),
            ),
        )

    with pytest.raises(ValueError, match="cycle"):
        DataflowGraph(
            (_buffer(0, "shared"),),
            (
                DataflowNode(0, 0, "a", reads=(0,), writes=(0,)),
                DataflowNode(1, 0, "b", reads=(0,), writes=(0,)),
            ),
            (
                DataflowEdge(0, 1, buffer_id=0),
                DataflowEdge(1, 0, buffer_id=0),
            ),
            (RegionKind.PIPELINE,),
        )
