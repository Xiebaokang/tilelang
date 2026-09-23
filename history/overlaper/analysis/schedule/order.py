"""Build one priority-driven order for fixed stages and groups."""

from __future__ import annotations

import heapq
from collections.abc import Mapping

from ...parse.graph import DataflowEdge, DataflowGraph, RegionKind
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


def _reachable_issue_priority(
    graph: DataflowGraph,
    successors: Mapping[int, set[int]],
    remaining: set[int],
    node_id: int,
) -> int:
    """Return the best issue priority still reachable from ``node_id``."""

    best = graph.issue_priority(node_id)
    seen = {node_id}
    stack = [node_id]
    while stack:
        current = stack.pop()
        for successor_id in successors[current]:
            if successor_id in remaining and successor_id not in seen:
                seen.add(successor_id)
                best = max(best, graph.issue_priority(successor_id))
                stack.append(successor_id)
    return best


def _pipeline_ready_key(
    graph: DataflowGraph,
    stages: Mapping[int, int],
    successors: Mapping[int, set[int]],
    remaining: set[int],
    node_id: int,
) -> tuple[int, int, int, int]:
    """Prefer unlocking high-priority work, then the lower stage."""

    return (
        -_reachable_issue_priority(graph, successors, remaining, node_id),
        stages[node_id],
        -graph.issue_priority(node_id),
        node_id,
    )


def _priority_pipeline_order(
    graph: DataflowGraph,
    region_id: int,
    stages: Mapping[int, int],
) -> tuple[int, ...]:
    """Order ready nodes by unlocked priority, then by lower stage.

    A ready node is scored by the highest issue priority still reachable
    from it on the zero-distance graph.  That keeps a stage-0 WGMMA such
    as QK ahead of a ready stage-1 WGMMA such as PV, because the initializer
    that unlocks QK shares PV's priority but sits at the lower stage.
    After those high-priority ops are issued, leftover lower-priority work
    (softmax) no longer wins against PV: a naive "always emit the lowest
    ready stage" rule would drain softmax first.
    """

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

    remaining = set(node_ids)
    pending = {
        node_id: set(required) for node_id, required in predecessors.items()
    }
    ready = {node_id for node_id, required in pending.items() if not required}
    order: list[int] = []
    while ready:
        node_id = min(
            ready,
            key=lambda item: _pipeline_ready_key(
                graph, stages, successors, remaining, item
            ),
        )
        order.append(node_id)
        ready.remove(node_id)
        remaining.remove(node_id)
        for successor_id in successors[node_id]:
            pending[successor_id].discard(node_id)
            if not pending[successor_id] and successor_id in remaining:
                ready.add(successor_id)

    if len(order) != len(nodes):
        raise ValueError("zero-distance pipeline dependencies contain a cycle")
    return tuple(order)


def _serial_order(graph: DataflowGraph, region_id: int) -> tuple[int, ...]:
    """Topologically order serial work, choosing highest priority first."""

    nodes = graph.nodes_for_region(region_id)
    node_ids = frozenset(node.node_id for node in nodes)
    predecessors: dict[int, set[int]] = {
        node_id: set() for node_id in node_ids
    }
    successors: dict[int, set[int]] = {
        node_id: set() for node_id in node_ids
    }
    for edge in _region_edges(graph, node_ids):
        if edge.producer_id != edge.consumer_id:
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
                heapq.heappush(ready, graph.issue_order_key(successor_id))

    if len(order) != len(nodes):
        raise ValueError("serial dependencies contain a cycle")
    return tuple(order)


def _project_group_orders(
    node_order: tuple[int, ...],
    groups: Mapping[int, int],
    group_ids: tuple[int, ...],
) -> dict[int, dict[int, int]]:
    """Project one global acyclic order onto each parallel group."""

    result: dict[int, dict[int, int]] = {}
    for group_id in group_ids:
        group_nodes = [
            node_id for node_id in node_order if groups[node_id] == group_id
        ]
        result[group_id] = {
            node_id: position for position, node_id in enumerate(group_nodes)
        }
    return result


def build_program_orders(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
) -> ProgramOrders:
    """Build one deterministic group-local order for every graph region.

    For pipeline regions, only zero-effective-distance edges constrain the
    current iteration. When several nodes are ready in parallel, the retained
    choice is the one that unlocks the highest remaining issue priority;
    ties prefer the lower stage, then the node's own issue priority, then
    the original node ID. Serial regions keep the priority-first topological
    rule over all of their internal dependencies. The selected global order
    is then projected into each group.
    """

    stages, group_ids = _validate_inputs(graph, stages_by_region, groups)
    result: ProgramOrders = {}
    for region_id, kind in enumerate(graph.region_kinds):
        if kind == RegionKind.PIPELINE:
            node_order = _priority_pipeline_order(
                graph, region_id, stages[region_id]
            )
        else:
            node_order = _serial_order(graph, region_id)
        result[region_id] = _project_group_orders(
            node_order, groups, group_ids
        )
    return result
