"""Build one deterministic priority-driven order for fixed stages and groups."""

from __future__ import annotations

import heapq
from collections.abc import Mapping

from ..parseIR.graph import DataflowEdge, DataflowGraph, RegionKind
from .stage import effective_stage_distance


ProgramOrders = dict[int, dict[int, dict[int, int]]]


def _validate_inputs(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
) -> tuple[dict[int, dict[int, int]], tuple[int, ...]]:
    node_ids = {node.node_id for node in graph.nodes}
    if set(groups) != node_ids:
        raise ValueError("groups must cover every node exactly once")
    if any(group_id < 0 for group_id in groups.values()):
        raise ValueError("group IDs must be non-negative")
    group_ids = tuple(sorted(set(groups.values())))
    if group_ids != tuple(range(len(group_ids))):
        raise ValueError("group IDs must be dense and non-empty")

    pipeline_regions = {
        region_id
        for region_id, kind in enumerate(graph.region_kinds)
        if kind == RegionKind.PIPELINE
    }
    if set(stages_by_region) != pipeline_regions:
        raise ValueError("stages must cover every pipeline region exactly once")
    stages: dict[int, dict[int, int]] = {}
    for region_id in sorted(pipeline_regions):
        region_stages = dict(stages_by_region[region_id])
        region_node_ids = {
            node.node_id for node in graph.nodes_for_region(region_id)
        }
        if set(region_stages) != region_node_ids:
            raise ValueError("stages must cover every pipeline node exactly once")
        if any(stage < 0 for stage in region_stages.values()):
            raise ValueError("stage IDs must be non-negative")
        stages[region_id] = region_stages
    return stages, group_ids


def _region_edges(
    graph: DataflowGraph,
    node_ids: frozenset[int],
) -> tuple[DataflowEdge, ...]:
    return tuple(
        edge
        for edge in graph.edges
        if edge.producer_id in node_ids and edge.consumer_id in node_ids
    )


def _priority_pipeline_order(
    graph: DataflowGraph,
    region_id: int,
    stages: Mapping[int, int],
) -> tuple[int, ...]:
    """Topologically order zero-distance work, choosing highest priority first."""

    nodes = graph.nodes_for_region(region_id)
    node_ids = frozenset(node.node_id for node in nodes)
    predecessors: dict[int, set[int]] = {
        node_id: set() for node_id in node_ids
    }
    successors: dict[int, set[int]] = {
        node_id: set() for node_id in node_ids
    }
    for edge in _region_edges(graph, node_ids):
        distance = effective_stage_distance(edge, stages)
        if distance < 0:
            raise ValueError("stages violate a pipeline dependency")
        if distance == 0 and edge.producer_id != edge.consumer_id:
            predecessors[edge.consumer_id].add(edge.producer_id)
            successors[edge.producer_id].add(edge.consumer_id)

    ready = [
        graph.issue_order_key(node_id)
        for node_id, required in predecessors.items()
        if not required
    ]
    heapq.heapify(ready)
    remaining = {
        node_id: len(required) for node_id, required in predecessors.items()
    }
    order: list[int] = []
    while ready:
        _, node_id = heapq.heappop(ready)
        order.append(node_id)
        for successor_id in successors[node_id]:
            remaining[successor_id] -= 1
            if remaining[successor_id] == 0:
                heapq.heappush(
                    ready,
                    graph.issue_order_key(successor_id),
                )
    if len(order) != len(nodes):
        raise ValueError("zero-distance pipeline dependencies contain a cycle")
    return tuple(order)


def _project_group_orders(
    node_order: tuple[int, ...],
    groups: Mapping[int, int],
    group_ids: tuple[int, ...],
) -> dict[int, dict[int, int]]:
    result: dict[int, dict[int, int]] = {}
    for group_id in group_ids:
        local_nodes = [
            node_id for node_id in node_order if groups[node_id] == group_id
        ]
        result[group_id] = {
            node_id: order for order, node_id in enumerate(local_nodes)
        }
    return result


def _serial_order(graph: DataflowGraph, region_id: int) -> tuple[int, ...]:
    """Return original IR order and verify that it respects dependencies."""

    node_order = tuple(
        node.node_id for node in graph.nodes_for_region(region_id)
    )
    positions = {node_id: index for index, node_id in enumerate(node_order)}
    node_ids = frozenset(node_order)
    for edge in _region_edges(graph, node_ids):
        if (
            edge.producer_id != edge.consumer_id
            and positions[edge.producer_id] >= positions[edge.consumer_id]
        ):
            raise ValueError("serial dependency is reversed by original IR order")
    return node_order


def build_program_orders(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
) -> ProgramOrders:
    """Build one group-local order for every region without enumeration.

    Pipeline regions use a single priority-driven global topological order and
    project it onto groups. Serial regions preserve original operation order.
    """

    stages, group_ids = _validate_inputs(graph, stages_by_region, groups)
    result: ProgramOrders = {}
    for region_id, kind in enumerate(graph.region_kinds):
        if kind == RegionKind.PIPELINE:
            node_order = _priority_pipeline_order(
                graph,
                region_id,
                stages[region_id],
            )
        else:
            node_order = _serial_order(graph, region_id)
        result[region_id] = _project_group_orders(
            node_order,
            groups,
            group_ids,
        )
    return result
