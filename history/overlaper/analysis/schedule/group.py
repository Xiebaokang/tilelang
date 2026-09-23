"""Enumerate warp-group partitions from group-visible graph edges."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

from ...parse.graph import DataflowEdge, DataflowGraph


MAX_NUM_GROUPS = 3


def is_group_visible_scope(scope: str) -> bool:
    """Return whether a buffer can communicate between warp groups."""

    return (
        scope in ("", "global")
        or scope.startswith("shared")
        or "tmem" in scope
    )


def is_group_opportunity(graph: DataflowGraph, edge: DataflowEdge) -> bool:
    """Return whether ``edge`` can be cut to expose a group opportunity."""

    return edge.producer_id != edge.consumer_id and is_group_visible_scope(
        graph.buffer_for_id(edge.buffer_id).scope
    )


def build_group_opportunities(
    graph: DataflowGraph,
) -> tuple[DataflowEdge, ...]:
    """Return all whole-graph edges backed by group-visible buffers."""

    return tuple(edge for edge in graph.edges if is_group_opportunity(graph, edge))


def build_group_components(
    graph: DataflowGraph,
) -> tuple[tuple[int, ...], ...]:
    """Contract nodes joined by an edge that cannot cross warp groups."""

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
        if not is_group_opportunity(graph, edge):
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
    graph: DataflowGraph,
    num_groups: int,
) -> Iterator[dict[int, int]]:
    """Enumerate canonical partitions using exactly ``num_groups`` groups.

    Non-visible dependencies have already been contracted into indivisible
    components. A group split is retained only when every resulting group
    participates in at least one cut opportunity edge. Group IDs use restricted
    growth order, which removes equivalent permutations of group labels.
    """

    if num_groups < 1 or num_groups > MAX_NUM_GROUPS:
        raise ValueError(
            f"num_groups must be between 1 and {MAX_NUM_GROUPS}, "
            f"got {num_groups}"
        )
    if not graph.nodes:
        return
    if num_groups == 1:
        yield {node.node_id: 0 for node in graph.nodes}
        return

    components = build_group_components(graph)
    component_index = {
        node_id: component_id
        for component_id, component in enumerate(components)
        for node_id in component
    }
    opportunity_components = tuple(
        (component_index[edge.producer_id], component_index[edge.consumer_id])
        for edge in build_group_opportunities(graph)
        if component_index[edge.producer_id]
        != component_index[edge.consumer_id]
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


def enumerate_bounded_group_assignments(
    graph: DataflowGraph,
    num_groups: int,
    *,
    beam_width: int,
    score: Callable[[Mapping[int, int]], float],
) -> Iterator[dict[int, int]]:
    """Enumerate promising canonical partitions without visiting ``k**N``.

    The beam operates on contracted group components.  Unassigned components
    temporarily stay in the baseline group, allowing a graph-aware scorer to
    estimate every partial partition.  Final candidates obey the same exact
    participation constraints as :func:`enumerate_group_assignments`.
    """

    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if num_groups < 1 or num_groups > MAX_NUM_GROUPS:
        raise ValueError(
            f"num_groups must be between 1 and {MAX_NUM_GROUPS}, got {num_groups}"
        )
    if num_groups == 1:
        if graph.nodes:
            yield {node.node_id: 0 for node in graph.nodes}
        return

    components = build_group_components(graph)
    component_index = {
        node_id: component_id
        for component_id, component in enumerate(components)
        for node_id in component
    }
    opportunities = tuple(
        (component_index[edge.producer_id], component_index[edge.consumer_id])
        for edge in build_group_opportunities(graph)
        if component_index[edge.producer_id] != component_index[edge.consumer_id]
    )
    candidate_components = {
        component_id for edge in opportunities for component_id in edge
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
    degree = {
        component_id: sum(component_id in edge for edge in opportunities)
        for component_id in candidate_components
    }
    variables = tuple(
        sorted(
            candidate_components - {baseline},
            key=lambda component_id: (
                -degree[component_id],
                components[component_id][0],
            ),
        )
    )
    states: list[tuple[dict[int, int], int]] = [({baseline: 0}, 0)]

    def materialize(assigned: Mapping[int, int]) -> dict[int, int]:
        return {
            node_id: assigned.get(component_id, 0)
            for component_id, component in enumerate(components)
            for node_id in component
        }

    for component_id in variables:
        expanded = []
        for assigned, maximum_group in states:
            largest_choice = min(maximum_group + 1, num_groups - 1)
            for group_id in range(largest_choice + 1):
                updated = {**assigned, component_id: group_id}
                full = materialize(updated)
                expanded.append(
                    (
                        score(full),
                        tuple(sorted(updated.items())),
                        updated,
                        max(maximum_group, group_id),
                    )
                )
        expanded.sort(key=lambda item: (-item[0], item[1]))
        # Retain some paths for every number of groups reached so far.
        next_states = []
        per_reached = max(1, (beam_width + num_groups - 1) // num_groups)
        for reached in range(num_groups):
            matching = [
                (assigned, maximum_group)
                for _, _, assigned, maximum_group in expanded
                if maximum_group == reached
            ]
            next_states.extend(matching[:per_reached])
        states = next_states[:beam_width]

    results = []
    for assigned, _ in states:
        if set(assigned.values()) != set(range(num_groups)):
            continue
        participating = {
            assigned[left]
            for left, right in opportunities
            if assigned[left] != assigned[right]
        } | {
            assigned[right]
            for left, right in opportunities
            if assigned[left] != assigned[right]
        }
        if participating != set(range(num_groups)):
            continue
        full = dict(sorted(materialize(assigned).items()))
        results.append((score(full), tuple(full.items()), full))
    results.sort(key=lambda item: (-item[0], item[1]))
    for _, _, assignment in results:
        yield assignment
