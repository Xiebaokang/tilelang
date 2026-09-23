"""Enumerate pipeline stages under correctness and performance constraints."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from ..parseIR.graph import DataflowEdge, DataflowGraph, RegionKind


def infer_max_stages(graph: DataflowGraph, region_id: int) -> int:
    """Infer a bounded useful depth from hardware execution roles."""

    nodes = graph.nodes_for_region(region_id)
    if graph.region_kinds[region_id] != RegionKind.PIPELINE or not nodes:
        return 1
    if graph.hardware is None:
        raise ValueError("stage inference requires graph.hardware")

    kinds = {node.instruction_kind for node in nodes}
    roles = {
        graph.hardware.execution_role(kind)
        for kind in kinds
    }
    same_kind_splits = sum(
        graph.hardware.allows_stage_split(kind, kind) for kind in kinds
    )
    return max(1, min(len(nodes), len(roles) + same_kind_splits))


def _correctness_constraint(
    edge: DataflowEdge,
    stages: Mapping[int, int],
) -> bool:
    """Check producer/consumer time order for one completed edge."""

    return stages[edge.producer_id] <= (
        stages[edge.consumer_id] + edge.iteration_distance
    )


def _performance_constraint(
    graph: DataflowGraph,
    edge: DataflowEdge,
    stages: Mapping[int, int],
) -> bool:
    """Check whether hardware permits this ordered pair to split."""

    producer = graph.node_for_id(edge.producer_id)
    consumer = graph.node_for_id(edge.consumer_id)
    if stages[producer.node_id] == stages[consumer.node_id]:
        return True
    assert graph.hardware is not None
    return graph.hardware.allows_stage_split(
        producer.instruction_kind,
        consumer.instruction_kind,
    )


def _region_edges(
    graph: DataflowGraph, node_ids: frozenset[int]
) -> tuple[DataflowEdge, ...]:
    return tuple(
        edge
        for edge in graph.edges
        if edge.producer_id in node_ids and edge.consumer_id in node_ids
    )


def _has_useful_stage_boundaries(
    graph: DataflowGraph,
    edges: tuple[DataflowEdge, ...],
    stages: Mapping[int, int],
    num_stages: int,
) -> bool:
    """Require every boundary to expose an asynchronous overlap."""

    for boundary in range(num_stages - 1):
        # if not any(
        if any(
            min(stages[edge.producer_id], stages[edge.consumer_id])
            <= boundary
            < max(stages[edge.producer_id], stages[edge.consumer_id])
            and (
                graph.is_async(edge.producer_id)
                or graph.is_async(edge.consumer_id)
            )
            for edge in edges
        ):
            # return False
    # return True
            return True
    return False

def enumerate_stage_assignments(
    graph: DataflowGraph,
    region_id: int,
    num_stages: int,
) -> Iterator[dict[int, int]]:
    """Enumerate normalized stage assignments for one pipeline region.

    Correctness requires ``stage(producer) <= stage(consumer) + distance``.
    A direct dependency may cross stages only when its ordered instruction-kind
    pair is present in the selected hardware's whitelist. Results span
    ``[0, num_stages - 1]`` so they use the requested pipeline depth.
    """

    if num_stages < 1:
        raise ValueError("num_stages must be positive")
    nodes = graph.nodes_for_region(region_id)
    if graph.region_kinds[region_id] != RegionKind.PIPELINE:
        if num_stages != 1:
            raise ValueError("a serial region must use exactly one stage")
        yield {node.node_id: 0 for node in nodes}
        return
    if graph.hardware is None:
        raise ValueError("stage enumeration requires graph.hardware")
    if not nodes:
        if num_stages == 1:
            yield {}
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

    def visit(index: int) -> Iterator[dict[int, int]]:
        if index == len(order):
            for i in range(num_stages):
                if i not in stages.values():
                    return
            if not _has_useful_stage_boundaries(
                graph, edges, stages, num_stages
            ):
                return
            yield dict(sorted(stages.items()))
            return

        node_id = order[index]
        for stage in range(num_stages):
            stages[node_id] = stage
            completed = (
                edge
                for edge in incident_edges[node_id]
                if edge.producer_id in stages and edge.consumer_id in stages
            )
            if all(
                _correctness_constraint(edge, stages)
                and _performance_constraint(graph, edge, stages)
                for edge in completed
            ):
                yield from visit(index + 1)
            del stages[node_id]

    yield from visit(0)


def effective_stage_distance(
    edge: DataflowEdge, stages: Mapping[int, int]
) -> int:
    """Return the pipeline-time distance from producer to consumer."""

    return (
        edge.iteration_distance
        + stages[edge.consumer_id]
        - stages[edge.producer_id]
    )
