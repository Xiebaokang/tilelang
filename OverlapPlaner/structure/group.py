"""Enumerate CTA-partition groups from group-opportunity edges."""

from __future__ import annotations

from collections.abc import Iterator

from OverlapPlaner.arch.api import ClassifiedGraph, is_group_opportunity
from OverlapPlaner.facts import FactEdge


def build_group_opportunities(classified: ClassifiedGraph) -> tuple[FactEdge, ...]:
    return tuple(
        edge
        for edge in classified.graph.edges
        if is_group_opportunity(classified, edge)
    )


def connected_components(classified: ClassifiedGraph) -> tuple[tuple[int, ...], ...]:
    """Contract nodes joined by an edge that cannot cross groups."""

    graph = classified.graph
    parents = list(range(len(graph.nodes)))

    def find(node_id: int) -> int:
        while parents[node_id] != node_id:
            parents[node_id] = parents[parents[node_id]]
            node_id = parents[node_id]
        return node_id

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for edge in graph.edges:
        if not is_group_opportunity(classified, edge):
            union(edge.producer_id, edge.consumer_id)

    members: dict[int, list[int]] = {}
    for node in graph.nodes:
        members.setdefault(find(node.node_id), []).append(node.node_id)
    return tuple(
        sorted(
            (tuple(component) for component in members.values()),
            key=lambda component: component[0],
        )
    )


def enumerate_group_assignments(
    classified: ClassifiedGraph,
    num_groups: int,
) -> Iterator[dict[int, int]]:
    """Yield canonical partitions that use exactly ``num_groups`` groups.

    Non-opportunity dependencies are contracted first. A split is kept only
    when every group participates in at least one cut opportunity. Group IDs
    use restricted-growth order so label permutations are not enumerated.
    """

    if num_groups < 1:
        raise ValueError("num_groups must be at least 1")
    graph = classified.graph
    if not graph.nodes:
        return
    if num_groups == 1:
        yield {node.node_id: 0 for node in graph.nodes}
        return

    components = connected_components(classified)
    component_index = {
        node_id: component_id
        for component_id, component in enumerate(components)
        for node_id in component
    }
    opportunity_components = tuple(
        (component_index[edge.producer_id], component_index[edge.consumer_id])
        for edge in build_group_opportunities(classified)
        if component_index[edge.producer_id] != component_index[edge.consumer_id]
    )
    candidate_components = {
        component_id
        for opportunity in opportunity_components
        for component_id in opportunity
    }
    if num_groups > len(candidate_components):
        return

    baseline = max(
        candidate_components,
        key=lambda component_id: (
            len(components[component_id]),
            -components[component_id][0],
        ),
    )
    variables = tuple(sorted(candidate_components - {baseline}))
    assigned: dict[int, int] = {baseline: 0}

    def visit(index: int, maximum_group: int) -> Iterator[dict[int, int]]:
        if index == len(variables):
            if set(assigned.values()) != set(range(num_groups)):
                return
            participating_groups = {
                assigned[left]
                for left, right in opportunity_components
                if assigned[left] != assigned[right]
            } | {
                assigned[right]
                for left, right in opportunity_components
                if assigned[left] != assigned[right]
            }
            if participating_groups != set(range(num_groups)):
                return
            result = {
                node_id: assigned.get(component_id, 0)
                for component_id, component in enumerate(components)
                for node_id in component
            }
            yield dict(sorted(result.items()))
            return

        component_id = variables[index]
        largest_choice = min(maximum_group + 1, num_groups - 1)
        for group_id in range(largest_choice + 1):
            assigned[component_id] = group_id
            yield from visit(index + 1, max(maximum_group, group_id))
        assigned.pop(component_id, None)

    yield from visit(0, 0)
