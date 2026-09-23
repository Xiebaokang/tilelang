"""Architecture-free IR facts for OverlapPlan search."""

from .extract import extract_fact_graph
from .graph import (
    BufferAccessKind,
    BufferFact,
    BufferRangeAccess,
    DependencyKind,
    FactEdge,
    FactGraph,
    FactNode,
    GemmFact,
    OpKind,
    ReduceFact,
    RegionFact,
    RegionKind,
    is_group_visible_scope,
)

__all__ = [
    "BufferAccessKind",
    "BufferFact",
    "BufferRangeAccess",
    "DependencyKind",
    "FactEdge",
    "FactGraph",
    "FactNode",
    "GemmFact",
    "OpKind",
    "ReduceFact",
    "RegionFact",
    "RegionKind",
    "extract_fact_graph",
    "is_group_visible_scope",
]
