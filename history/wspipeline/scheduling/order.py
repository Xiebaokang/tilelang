"""Correctness-only local-order enumeration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterator

from ..analysis.core import DataflowEdge, DataflowNode
from ..analysis.extractor import RegionOverlap, analyze_region_overlap
from ..analysis.program import (
    BufferAccessKind,
    ProgramDataflowAnalysis,
    ProgramRegion,
    ProgramRegionKind,
)
from .stage import effective_stage_distance


@dataclass(frozen=True)
class ProgramRegionSchedule:
    """The stage assignment and group-local orders of one program region."""

    region_id: int
    num_stages: int
    stages: dict[DataflowNode, int]
    group_orders: dict[int, dict[DataflowNode, int]]


def build_fragment_reuse_pairs(
    analysis: ProgramDataflowAnalysis,
) -> dict[int, tuple[tuple[DataflowNode, DataflowNode], ...]]:
    """Precompute next-iteration fragment access/writer conflicts by region."""

    nodes_by_id = {
        operation.operation_id: operation.node
        for operation in analysis.operations
    }
    result = {}
    for region in analysis.regions:
        if region.kind != ProgramRegionKind.PIPELINE or region.loop is None:
            continue
        region_operation_ids = set(region.operation_ids)
        pairs = []
        for descriptor in analysis.buffers:
            if descriptor.scope != "local.fragment":
                continue
            accesses = tuple(
                access
                for access in analysis.region_accesses
                if access.buffer_id == descriptor.buffer_id
                and access.operation_id in region_operation_ids
            )
            writers = tuple(
                access
                for access in accesses
                if access.kind == BufferAccessKind.WRITE
            )
            for access in accesses:
                for writer in writers:
                    if analyze_region_overlap(
                        access,
                        writer,
                        region.loop.loop_var,
                        iteration_delta=1,
                    ) == RegionOverlap.DISJOINT:
                        continue
                    pairs.append(
                        (
                            nodes_by_id[access.operation_id],
                            nodes_by_id[writer.operation_id],
                        )
                    )
        result[region.region_id] = tuple(pairs)
    return result


def build_order_predecessors(
    nodes: list[DataflowNode],
    edges: list[DataflowEdge],
    stages: dict[DataflowNode, int],
) -> dict[DataflowNode, set[DataflowNode]]:
    """Build the correctness precedence constraints for order search.

    For ``producer(k) -> consumer(k + d)``, the effective pipeline distance is
    ``d + S(consumer) - S(producer)``. A zero distance requires the producer
    to precede the consumer in ``software_pipeline_order``. A negative distance
    means that the stage assignment is invalid. No resource, execution-kind,
    or original-IR-order preference is added here; those belong to scoring.
    """

    if set(stages) != set(nodes):
        raise ValueError("stages must assign every node exactly once")

    predecessors: dict[DataflowNode, set[DataflowNode]] = {
        node: set() for node in nodes
    }
    for edge in edges:
        gap = effective_stage_distance(edge, stages)
        if gap < 0:
            raise ValueError(
                "stage assignment violates dependency "
                f"{edge.producer.name}->{edge.consumer.name}"
            )
        if gap == 0 and edge.producer != edge.consumer:
            predecessors[edge.consumer].add(edge.producer)

    return predecessors


def enumerate_feasible_orders(
    nodes: list[DataflowNode],
    predecessors: dict[DataflowNode, set[DataflowNode]],
) -> Iterator[dict[DataflowNode, int]]:
    """Enumerate every topological order permitted by correctness constraints."""

    if set(predecessors) != set(nodes):
        raise ValueError("predecessors must contain every node exactly once")
    node_set = set(nodes)
    if any(not required <= node_set for required in predecessors.values()):
        raise ValueError("predecessors contains a node outside the payload graph")

    original_index = {node: index for index, node in enumerate(nodes)}
    successors: dict[DataflowNode, list[DataflowNode]] = {
        node: [] for node in nodes
    }
    remaining = {node: len(predecessors[node]) for node in nodes}
    for consumer, producers in predecessors.items():
        for producer in producers:
            successors[producer].append(consumer)

    ready = [node for node in nodes if remaining[node] == 0]
    orders: dict[DataflowNode, int] = {}

    from ..search.scoring import score_order_choice

    def visit(
        available: list[DataflowNode], previous_node: DataflowNode | None
    ) -> Iterator[dict[DataflowNode, int]]:
        if len(orders) == len(nodes):
            yield dict(orders)
            return

        choices = sorted(
            available,
            key=lambda node: score_order_choice(
                node, original_index, previous_node
            ),
        )
        for node in choices:
            next_available = [
                candidate for candidate in available if candidate != node
            ]
            orders[node] = len(orders)
            newly_ready: list[DataflowNode] = []
            for consumer in successors[node]:
                remaining[consumer] -= 1
                if remaining[consumer] == 0:
                    newly_ready.append(consumer)
            yield from visit(next_available + newly_ready, node)
            for consumer in successors[node]:
                remaining[consumer] += 1
            orders.pop(node)

    yield from visit(ready, None)


def _group_orders_are_deadlock_free(
    nodes: list[DataflowNode],
    global_predecessors: dict[DataflowNode, set[DataflowNode]],
    group_orders: dict[int, dict[DataflowNode, int]],
) -> bool:
    """Check the combined dependency and group-order graph for cycles."""

    predecessors = {node: set(global_predecessors[node]) for node in nodes}
    for local_order in group_orders.values():
        ordered_nodes = sorted(local_order, key=local_order.__getitem__)
        for earlier, later in zip(ordered_nodes, ordered_nodes[1:]):
            predecessors[later].add(earlier)

    successors: dict[DataflowNode, list[DataflowNode]] = {
        node: [] for node in nodes
    }
    remaining = {node: len(predecessors[node]) for node in nodes}
    for consumer, producers in predecessors.items():
        for producer in producers:
            successors[producer].append(consumer)
    ready = [node for node in nodes if remaining[node] == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for consumer in successors[node]:
            remaining[consumer] -= 1
            if remaining[consumer] == 0:
                ready.append(consumer)
    return visited == len(nodes)


def _add_fragment_import_lifetime_predecessors(
    analysis: ProgramDataflowAnalysis,
    region: ProgramRegion,
    edges: list[DataflowEdge],
    stages: dict[DataflowNode, int],
    node_groups: Mapping[DataflowNode, int],
    predecessors: dict[DataflowNode, set[DataflowNode]],
) -> bool:
    """Order a prior fragment access before the next edge-local import."""

    region_operation_ids = set(region.operation_ids)
    nodes_by_id = {
        operation.operation_id: operation.node
        for operation in analysis.operations
    }
    seen_imports: set[tuple[int, DataflowNode, int]] = set()
    import_stages: dict[tuple[int, int], int] = {}
    for edge in edges:
        if edge.buffer_id is None or "RAW" not in edge.dependency_kinds:
            continue
        descriptor = analysis.buffer_for_id(edge.buffer_id)
        if descriptor.scope != "local.fragment":
            continue
        producer_group = node_groups[edge.producer]
        consumer_group = node_groups[edge.consumer]
        if producer_group == consumer_group:
            continue
        import_key = (edge.buffer_id, edge.consumer, consumer_group)
        if import_key in seen_imports:
            continue
        seen_imports.add(import_key)
        import_node = edge.consumer
        import_stage = stages[import_node]
        stage_key = (edge.buffer_id, consumer_group)
        previous_stage = import_stages.setdefault(stage_key, import_stage)
        if previous_stage != import_stage:
            # Lowering inserts a write to the same consumer-side fragment at
            # every import. Different stages overlap across iterations and
            # therefore require unsupported fragment multiversioning.
            return False
        for access in analysis.region_accesses:
            if (
                access.buffer_id != edge.buffer_id
                or access.operation_id not in region_operation_ids
            ):
                continue
            accessor = nodes_by_id[access.operation_id]
            if node_groups[accessor] != consumer_group:
                continue
            effective_distance = 1 + import_stage - stages[accessor]
            if effective_distance < 0:
                return False
            if effective_distance == 0 and accessor != import_node:
                predecessors[import_node].add(accessor)
    return True


def _add_fragment_reuse_predecessors(
    pairs: tuple[tuple[DataflowNode, DataflowNode], ...],
    stages: dict[DataflowNode, int],
    node_groups: Mapping[DataflowNode, int],
    predecessors: dict[DataflowNode, set[DataflowNode]],
) -> bool:
    """Enforce single-version fragment reuse before physical planning."""

    for accessor, writer in pairs:
        if node_groups[accessor] != node_groups[writer]:
            continue
        effective_distance = 1 + stages[writer] - stages[accessor]
        if effective_distance < 0:
            return False
        if effective_distance == 0 and accessor != writer:
            predecessors[writer].add(accessor)
    return True


def enumerate_region_group_orders(
    analysis: ProgramDataflowAnalysis,
    region: ProgramRegion,
    nodes: list[DataflowNode],
    edges: list[DataflowEdge],
    stages: dict[DataflowNode, int],
    node_groups: Mapping[DataflowNode, int],
    group_ids: tuple[int, ...],
    fragment_reuse_pairs: tuple[tuple[DataflowNode, DataflowNode], ...],
) -> Iterator[dict[int, dict[DataflowNode, int]]]:
    """Enumerate loop-local orders while allowing groups empty in this region."""

    global_predecessors = build_order_predecessors(nodes, edges, stages)
    if not _add_fragment_import_lifetime_predecessors(
        analysis,
        region,
        edges,
        stages,
        node_groups,
        global_predecessors,
    ):
        return
    if not _add_fragment_reuse_predecessors(
        fragment_reuse_pairs,
        stages,
        node_groups,
        global_predecessors,
    ):
        return
    inputs = {}
    for group_id in group_ids:
        local_nodes = [node for node in nodes if node_groups[node] == group_id]
        local_set = set(local_nodes)
        inputs[group_id] = (
            local_nodes,
            {
                node: global_predecessors[node] & local_set
                for node in local_nodes
            },
        )

    selected: dict[int, dict[DataflowNode, int]] = {}

    def visit(index: int) -> Iterator[dict[int, dict[DataflowNode, int]]]:
        if index == len(group_ids):
            candidate = {
                group_id: dict(order) for group_id, order in selected.items()
            }
            if _group_orders_are_deadlock_free(
                nodes, global_predecessors, candidate
            ):
                yield candidate
            return
        group_id = group_ids[index]
        local_nodes, local_predecessors = inputs[group_id]
        for order in enumerate_feasible_orders(local_nodes, local_predecessors):
            selected[group_id] = order
            yield from visit(index + 1)
        selected.pop(group_id, None)

    yield from visit(0)


def enumerate_region_schedules_for_stages(
    analysis: ProgramDataflowAnalysis,
    region: ProgramRegion,
    stages: dict[DataflowNode, int],
    node_groups: Mapping[DataflowNode, int],
    group_ids: tuple[int, ...],
    fragment_reuse_pairs: tuple[tuple[DataflowNode, DataflowNode], ...],
) -> Iterator[ProgramRegionSchedule]:
    """Enumerate group-local orders for one fixed stage assignment."""

    nodes = list(analysis.nodes_for_region(region))
    if region.kind == ProgramRegionKind.SERIAL:
        fixed_orders = {}
        for group_id in group_ids:
            local_nodes = [
                node for node in nodes if node_groups[node] == group_id
            ]
            fixed_orders[group_id] = {
                node: order for order, node in enumerate(local_nodes)
            }
        yield ProgramRegionSchedule(region.region_id, 0, {}, fixed_orders)
        return

    for group_orders in enumerate_region_group_orders(
        analysis,
        region,
        nodes,
        list(analysis.edges_for_region(region)),
        stages,
        node_groups,
        group_ids,
        fragment_reuse_pairs,
    ):
        yield ProgramRegionSchedule(
            region.region_id,
            max(stages.values()) + 1,
            dict(stages),
            group_orders,
        )
