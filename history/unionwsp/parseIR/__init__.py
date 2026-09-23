"""IR extraction and graph definitions for UnionWSP."""

from .graph import (
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


def __getattr__(name: str):
    if name == "extract_dataflow_graph":
        from .extractor import extract_dataflow_graph

        return extract_dataflow_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BufferAccessKind",
    "BufferDescriptor",
    "BufferRangeAccess",
    "DataflowEdge",
    "DataflowGraph",
    "DataflowNode",
    "DependencyKind",
    "InstructionKind",
    "RegionKind",
    "extract_dataflow_graph",
]
