"""Enumerate pipeline stages under correctness and instruction constraints."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

from ...headware.spec import Instruction, InstructionType
from ...parse.graph import DataflowEdge, DataflowGraph, RegionKind


NUM_STAGES = 3


def can_split_stages(
    producer: Instruction,
    consumer: Instruction,
) -> bool:
    """Return whether an edge between two instructions may cross stages.

    Memory instructions may split from memory or compute instructions. Two
    compute instructions can split when they are different instruction kinds.
    RRCP follows the same rule: it may split from any non-RRCP compute
    instruction, while two adjacent RRCP instructions must remain together.
    """

    if (
        producer.type == InstructionType.COMPUTE
        and consumer.type == InstructionType.COMPUTE
    ):
        return producer.name != consumer.name
    return True


def _correctness_constraint(
    edge: DataflowEdge,
    stages: Mapping[int, int],
) -> bool:
    """Check producer/consumer time order, matching the UnionWSP rule."""

    return stages[edge.producer_id] <= (
        stages[edge.consumer_id] + edge.iteration_distance
    )


def _performance_constraint(
    graph: DataflowGraph,
    edge: DataflowEdge,
    stages: Mapping[int, int],
) -> bool:
    """Check whether the edge's instructions permit a stage boundary."""

    if stages[edge.producer_id] == stages[edge.consumer_id]:
        return True
    producer_node = graph.node_for_id(edge.producer_id)
    if _is_register_initializer(graph, edge.producer_id):
        return False
    consumer_node = graph.node_for_id(edge.consumer_id)
    return can_split_stages(
        producer_node.instruction, consumer_node.instruction
    )


def _is_register_initializer(graph: DataflowGraph, node_id: int) -> bool:
    """Return whether a node initializes fragment state for its consumers."""

    node = graph.node_for_id(node_id)
    return node.name.startswith(("fill_", "clear_")) or (
        not node.reads
        and any(
            graph.buffer_for_id(buffer_id).scope == "local.fragment"
            for buffer_id in node.writes
        )
    )


def _region_edges(
    graph: DataflowGraph,
    node_ids: frozenset[int],
) -> tuple[DataflowEdge, ...]:
    return tuple(
        edge
        for edge in graph.edges
        if edge.producer_id in node_ids and edge.consumer_id in node_ids
    )


def enumerate_stage_assignments(
    graph: DataflowGraph,
    region_id: int,
    num_stages: int = NUM_STAGES,
) -> Iterator[dict[int, int]]:
    """Enumerate legal assignments using at most ``num_stages`` stages.

    ``num_stages`` is an upper bound, like the group-count bound: passing 3
    enumerates assignments that use exactly ``{0}``, ``{0, 1}``, or
    ``{0, 1, 2}``. Stage numbers are always dense. Correctness requires
    ``stage(producer) <= stage(consumer) + iteration_distance``; whenever an
    edge crosses a stage boundary, :func:`can_split_stages` must allow its
    instruction pair. Register fill/clear operations and other write-only
    fragment initializers always remain with their direct consumers.
    """

    if not 1 <= num_stages <= NUM_STAGES:
        raise ValueError(
            f"num_stages must be in [1, {NUM_STAGES}], got {num_stages}"
        )

    nodes = graph.nodes_for_region(region_id)
    if graph.region_kinds[region_id] != RegionKind.PIPELINE:
        yield {node.node_id: 0 for node in nodes}
        return
    if not nodes:
        return

    node_ids = frozenset(node.node_id for node in nodes)
    order = tuple(
        node_id for node_id in graph.topological_order() if node_id in node_ids
    )
    edges = _region_edges(graph, node_ids)
    incident_edges: dict[int, list[DataflowEdge]] = {
        node_id: [] for node_id in node_ids
    }
    for edge in edges:
        incident_edges[edge.producer_id].append(edge)
        if edge.consumer_id != edge.producer_id:
            incident_edges[edge.consumer_id].append(edge)

    stages: dict[int, int] = {}

    def visit(index: int, exact_num_stages: int) -> Iterator[dict[int, int]]:
        if index == len(order):
            if set(stages.values()) == set(range(exact_num_stages)):
                yield dict(sorted(stages.items()))
            return

        node_id = order[index]
        for stage in range(exact_num_stages):
            stages[node_id] = stage
            completed_edges = (
                edge
                for edge in incident_edges[node_id]
                if edge.producer_id in stages and edge.consumer_id in stages
            )
            if all(
                _correctness_constraint(edge, stages)
                and _performance_constraint(graph, edge, stages)
                for edge in completed_edges
            ):
                yield from visit(index + 1, exact_num_stages)
            del stages[node_id]

    for exact_num_stages in range(1, num_stages + 1):
        yield from visit(0, exact_num_stages)


def enumerate_bounded_stage_assignments(
    graph: DataflowGraph,
    region_id: int,
    *,
    beam_width: int,
    score: Callable[[Mapping[int, int]], float],
    num_stages: int = NUM_STAGES,
) -> Iterator[dict[int, int]]:
    """Enumerate legal stage assignments with bounded intermediate states."""

    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if not 1 <= num_stages <= NUM_STAGES:
        raise ValueError(
            f"num_stages must be in [1, {NUM_STAGES}], got {num_stages}"
        )
    nodes = graph.nodes_for_region(region_id)
    if graph.region_kinds[region_id] != RegionKind.PIPELINE:
        yield {node.node_id: 0 for node in nodes}
        return
    if not nodes:
        return

    node_ids = frozenset(node.node_id for node in nodes)
    order = tuple(
        node_id for node_id in graph.topological_order() if node_id in node_ids
    )
    edges = _region_edges(graph, node_ids)
    incident_edges: dict[int, list[DataflowEdge]] = {
        node_id: [] for node_id in node_ids
    }
    for edge in edges:
        incident_edges[edge.producer_id].append(edge)
        if edge.consumer_id != edge.producer_id:
            incident_edges[edge.consumer_id].append(edge)

    results = []
    for exact_num_stages in range(1, num_stages + 1):
        states: list[dict[int, int]] = [{}]
        for node_id in order:
            expanded = []
            for stages in states:
                for stage in range(exact_num_stages):
                    updated = {**stages, node_id: stage}
                    completed_edges = (
                        edge
                        for edge in incident_edges[node_id]
                        if edge.producer_id in updated
                        and edge.consumer_id in updated
                    )
                    if not all(
                        _correctness_constraint(edge, updated)
                        and _performance_constraint(graph, edge, updated)
                        for edge in completed_edges
                    ):
                        continue
                    full = {
                        candidate_id: updated.get(candidate_id, 0)
                        for candidate_id in order
                    }
                    expanded.append(
                        (score(full), tuple(sorted(updated.items())), updated)
                    )
            expanded.sort(key=lambda item: (-item[0], item[1]))
            # Keep paths with different numbers of stages already reached.
            next_states = []
            per_reached = max(
                1, (beam_width + exact_num_stages - 1) // exact_num_stages
            )
            for reached in range(1, exact_num_stages + 1):
                matching = [
                    stages
                    for _, _, stages in expanded
                    if len(set(stages.values())) == reached
                ]
                next_states.extend(matching[:per_reached])
            states = next_states[:beam_width]
        for stages in states:
            if set(stages.values()) == set(range(exact_num_stages)):
                result = dict(sorted(stages.items()))
                results.append((score(result), tuple(result.items()), result))

    results.sort(key=lambda item: (-item[0], item[1]))
    for _, _, assignment in results:
        yield assignment


def effective_stage_distance(
    edge: DataflowEdge,
    stages: Mapping[int, int],
) -> int:
    """Return the pipeline-time distance from producer to consumer."""

    return (
        edge.iteration_distance
        + stages[edge.consumer_id]
        - stages[edge.producer_id]
    )
