"""Enumerate pipeline stages from classified traits."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import product

from OverlapPlaner.arch import ClassifiedGraph, can_split_stages
from OverlapPlaner.facts import FactEdge, FactGraph, RegionKind
from OverlapPlaner.structure.model import SearchBudget, region_edges


def effective_stage_distance(edge: FactEdge, stages: Mapping[int, int]) -> int:
    """Return the pipeline-time distance from producer to consumer."""

    return (
        edge.iteration_distance
        + stages[edge.consumer_id]
        - stages[edge.producer_id]
    )


def _correctness_constraint(edge: FactEdge, stages: Mapping[int, int]) -> bool:
    return stages[edge.producer_id] <= (
        stages[edge.consumer_id] + edge.iteration_distance
    )


def _performance_constraint(
    classified: ClassifiedGraph,
    edge: FactEdge,
    stages: Mapping[int, int],
) -> bool:
    if stages[edge.producer_id] == stages[edge.consumer_id]:
        return True
    if classified.graph.is_initializer(edge.producer_id):
        return False
    return can_split_stages(
        classified.traits_for(edge.producer_id),
        classified.traits_for(edge.consumer_id),
    )


def _region_order(graph: FactGraph, region_id: int) -> tuple[int, ...]:
    node_ids = {node.node_id for node in graph.nodes_for_region(region_id)}
    return tuple(
        node_id for node_id in graph.topological_order() if node_id in node_ids
    )


def _incident_edges(
    graph: FactGraph, node_ids: frozenset[int]
) -> dict[int, list[FactEdge]]:
    edges = region_edges(graph, node_ids)
    incident: dict[int, list[FactEdge]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        incident[edge.producer_id].append(edge)
        if edge.consumer_id != edge.producer_id:
            incident[edge.consumer_id].append(edge)
    return incident


def enumerate_stage_assignments(
    classified: ClassifiedGraph,
    region_id: int,
    num_stages: int,
) -> Iterator[dict[int, int]]:
    """Enumerate legal dense assignments using at most ``num_stages`` stages."""

    if num_stages < 1:
        raise ValueError("num_stages must be at least 1")
    graph = classified.graph
    nodes = graph.nodes_for_region(region_id)
    if graph.region_kinds[region_id] != RegionKind.PIPELINE:
        yield {node.node_id: 0 for node in nodes}
        return
    if not nodes:
        return

    node_ids = frozenset(node.node_id for node in nodes)
    order = _region_order(graph, region_id)
    incident = _incident_edges(graph, node_ids)
    stages: dict[int, int] = {}

    def visit(index: int, exact_num_stages: int) -> Iterator[dict[int, int]]:
        if index == len(order):
            if set(stages.values()) == set(range(exact_num_stages)):
                yield dict(sorted(stages.items()))
            return

        node_id = order[index]
        for stage in range(exact_num_stages):
            stages[node_id] = stage
            completed = (
                edge
                for edge in incident[node_id]
                if edge.producer_id in stages and edge.consumer_id in stages
            )
            if all(
                _correctness_constraint(edge, stages)
                and _performance_constraint(classified, edge, stages)
                for edge in completed
            ):
                yield from visit(index + 1, exact_num_stages)
            del stages[node_id]

    for exact_num_stages in range(1, num_stages + 1):
        yield from visit(0, exact_num_stages)


def enumerate_bounded_stage_assignments(
    classified: ClassifiedGraph,
    region_id: int,
    *,
    beam_width: int,
    num_stages: int,
) -> Iterator[dict[int, int]]:
    """Enumerate legal stage assignments with bounded intermediate states."""

    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if num_stages < 1:
        raise ValueError("num_stages must be at least 1")
    graph = classified.graph
    nodes = graph.nodes_for_region(region_id)
    if graph.region_kinds[region_id] != RegionKind.PIPELINE:
        yield {node.node_id: 0 for node in nodes}
        return
    if not nodes:
        return

    node_ids = frozenset(node.node_id for node in nodes)
    order = _region_order(graph, region_id)
    incident = _incident_edges(graph, node_ids)
    results = []
    for exact_num_stages in range(1, num_stages + 1):
        states: list[dict[int, int]] = [{}]
        for node_id in order:
            expanded = []
            for current in states:
                for stage in range(exact_num_stages):
                    updated = {**current, node_id: stage}
                    completed = (
                        edge
                        for edge in incident[node_id]
                        if edge.producer_id in updated
                        and edge.consumer_id in updated
                    )
                    if not all(
                        _correctness_constraint(edge, updated)
                        and _performance_constraint(classified, edge, updated)
                        for edge in completed
                    ):
                        continue
                    expanded.append((tuple(sorted(updated.items())), updated))
            expanded.sort(key=lambda item: item[0])
            next_states = []
            per_reached = max(
                1, (beam_width + exact_num_stages - 1) // exact_num_stages
            )
            for reached in range(1, exact_num_stages + 1):
                matching = [
                    assignment
                    for _, assignment in expanded
                    if len(set(assignment.values())) == reached
                ]
                next_states.extend(matching[:per_reached])
            states = next_states[:beam_width]
        for assignment in states:
            if set(assignment.values()) == set(range(exact_num_stages)):
                result = dict(sorted(assignment.items()))
                results.append((tuple(result.items()), result))

    results.sort(key=lambda item: item[0])
    for _, assignment in results:
        yield assignment


def enumerate_program_stages(
    classified: ClassifiedGraph,
    budget: SearchBudget,
) -> Iterator[dict[int, dict[int, int]]]:
    """Yield whole-program pipeline stage maps."""

    graph = classified.graph
    pipeline_ids = [
        region.region_id
        for region in graph.regions
        if region.kind == RegionKind.PIPELINE
    ]
    if not pipeline_ids:
        yield {}
        return

    per_region: list[list[dict[int, int]]] = []
    for region_id in pipeline_ids:
        # TileLang predicates the prologue and epilogue when a loop executes
        # fewer iterations than its stage depth. A minimum-trip-count cutoff
        # would incorrectly remove Mamba's valid three-stage schedules.
        node_count = len(graph.nodes_for_region(region_id))
        if node_count <= 8:
            assignments = list(
                enumerate_stage_assignments(
                    classified, region_id, budget.max_stages
                )
            )
        else:
            assignments = list(
                enumerate_bounded_stage_assignments(
                    classified,
                    region_id,
                    beam_width=budget.stage_beam,
                    num_stages=budget.max_stages,
                )
            )
        if not assignments:
            return
        per_region.append(assignments)

    for combo in product(*per_region):
        yield {
            region_id: assignment
            for region_id, assignment in zip(pipeline_ids, combo)
        }
