"""Build priority and source-order schedules for fixed stages and groups."""

from __future__ import annotations

import heapq
from collections.abc import Iterator, Mapping

from OverlapPlaner.arch import ClassifiedGraph
from OverlapPlaner.facts import FactGraph, RegionKind
from OverlapPlaner.structure.model import (
    ProgramOrders,
    region_edges,
    validate_stage_group_order,
)
from OverlapPlaner.structure.stage import effective_stage_distance


def _issue_priority(classified: ClassifiedGraph, node_id: int) -> int:
    return classified.traits_for(node_id).issue_priority


def _issue_order_key(classified: ClassifiedGraph, node_id: int) -> tuple[int, int]:
    return (-_issue_priority(classified, node_id), node_id)


def _reachable_issue_priority(
    classified: ClassifiedGraph,
    successors: Mapping[int, set[int]],
    remaining: set[int],
    node_id: int,
) -> int:
    best = _issue_priority(classified, node_id)
    seen = {node_id}
    stack = [node_id]
    while stack:
        current = stack.pop()
        for successor_id in successors[current]:
            if successor_id in remaining and successor_id not in seen:
                seen.add(successor_id)
                best = max(best, _issue_priority(classified, successor_id))
                stack.append(successor_id)
    return best


def _pipeline_ready_key(
    classified: ClassifiedGraph,
    stages: Mapping[int, int],
    successors: Mapping[int, set[int]],
    remaining: set[int],
    node_id: int,
) -> tuple[int, int, int, int]:
    return (
        -_reachable_issue_priority(classified, successors, remaining, node_id),
        stages[node_id],
        -_issue_priority(classified, node_id),
        node_id,
    )


def _priority_pipeline_order(
    classified: ClassifiedGraph,
    region_id: int,
    stages: Mapping[int, int],
) -> tuple[int, ...]:
    graph = classified.graph
    nodes = graph.nodes_for_region(region_id)
    node_ids = frozenset(node.node_id for node in nodes)
    predecessors: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
    successors: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
    for edge in region_edges(graph, node_ids):
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
                classified, stages, successors, remaining, item
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


def _serial_order(classified: ClassifiedGraph, region_id: int) -> tuple[int, ...]:
    graph = classified.graph
    nodes = graph.nodes_for_region(region_id)
    node_ids = frozenset(node.node_id for node in nodes)
    predecessors: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
    successors: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
    for edge in region_edges(graph, node_ids):
        if edge.producer_id != edge.consumer_id:
            predecessors[edge.consumer_id].add(edge.producer_id)
            successors[edge.producer_id].add(edge.consumer_id)

    ready = [
        _issue_order_key(classified, node_id)
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
                heapq.heappush(ready, _issue_order_key(classified, successor_id))

    if len(order) != len(nodes):
        raise ValueError("serial dependencies contain a cycle")
    return tuple(order)


def _project_group_orders(
    node_order: tuple[int, ...],
    groups: Mapping[int, int],
    group_ids: tuple[int, ...],
) -> dict[int, dict[int, int]]:
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
    classified: ClassifiedGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
) -> ProgramOrders:
    """Build one deterministic group-local order for every graph region."""

    graph: FactGraph = classified.graph
    stages, group_ids = validate_stage_group_order(
        graph, stages_by_region, groups
    )
    result: ProgramOrders = {}
    for region_id, kind in enumerate(graph.region_kinds):
        if kind == RegionKind.PIPELINE:
            node_order = _priority_pipeline_order(
                classified, region_id, stages[region_id]
            )
        else:
            node_order = _serial_order(classified, region_id)
        result[region_id] = _project_group_orders(node_order, groups, group_ids)
    return result


def _source_pipeline_order(
    classified: ClassifiedGraph,
    region_id: int,
    stages: Mapping[int, int],
) -> tuple[int, ...]:
    """Follow the original operation order where zero-distance edges allow it."""

    graph = classified.graph
    node_ids = frozenset(node.node_id for node in graph.nodes_for_region(region_id))
    predecessors: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
    successors: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
    for edge in region_edges(graph, node_ids):
        distance = effective_stage_distance(edge, stages)
        if distance < 0:
            raise ValueError("stages violate a pipeline dependency")
        if distance == 0 and edge.producer_id != edge.consumer_id:
            predecessors[edge.consumer_id].add(edge.producer_id)
            successors[edge.producer_id].add(edge.consumer_id)
    ready = [node_id for node_id, required in predecessors.items() if not required]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        node_id = heapq.heappop(ready)
        order.append(node_id)
        for successor_id in successors[node_id]:
            predecessors[successor_id].remove(node_id)
            if not predecessors[successor_id]:
                heapq.heappush(ready, successor_id)
    if len(order) != len(node_ids):
        raise ValueError("zero-distance pipeline dependencies contain a cycle")
    return tuple(order)


def enumerate_program_orders(
    classified: ClassifiedGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
) -> Iterator[ProgramOrders]:
    """Include the input program order as well as the issue-priority order."""

    try:
        priority = build_program_orders(classified, stages_by_region, groups)
    except ValueError:
        return
    yield priority
    graph = classified.graph
    _, group_ids = validate_stage_group_order(graph, stages_by_region, groups)
    source: ProgramOrders = {}
    try:
        for region_id, kind in enumerate(graph.region_kinds):
            if kind == RegionKind.PIPELINE:
                node_order = _source_pipeline_order(
                    classified, region_id, stages_by_region[region_id]
                )
            else:
                node_order = tuple(
                    node.node_id for node in graph.nodes_for_region(region_id)
                )
            source[region_id] = _project_group_orders(
                node_order, groups, group_ids
            )
    except ValueError:
        return
    if source != priority:
        yield source
