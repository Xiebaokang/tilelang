"""Enumerate whole-program warp groups from software-pipeline opportunities."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from ..parseIR.graph import DataflowEdge, DataflowGraph, DependencyKind, RegionKind
from .stage import effective_stage_distance


def infer_max_groups(graph: DataflowGraph) -> int:
    """Infer a useful logical-group bound from roles and hardware capacity."""

    if graph.hardware is None:
        raise ValueError("group inference requires graph.hardware")
    if not graph.nodes:
        return 1
    hardware = graph.hardware
    roles = set(
        hardware.execution_role(node.instruction_kind)
        for node in graph.nodes
    )
    physical_limit = (
        hardware.max_threads_per_block
        // hardware.warp_size
        // hardware.warpgroup_warps
    )
    return max(
        1,
        min(
            len(build_group_components(graph)),
            len(roles),
            physical_limit,
        ),
    )


def _is_group_visible_scope(scope: str) -> bool:
    return scope in ("", "global") or scope.startswith("shared") or "tmem" in scope


def _is_register_initializer(graph: DataflowGraph, node_id: int) -> bool:
    node = graph.node_for_id(node_id)
    return node.name.startswith(("fill_", "clear_")) or (
        not node.reads
        and any(
            graph.buffer_for_id(buffer_id).scope == "local.fragment"
            for buffer_id in node.writes
        )
    )


def _exports_register_value(graph: DataflowGraph, node_id: int) -> bool:
    node = graph.node_for_id(node_id)
    return any(
        graph.buffer_for_id(buffer_id).scope == "local.fragment"
        for buffer_id in node.reads
    ) and any(
        _is_group_visible_scope(graph.buffer_for_id(buffer_id).scope)
        for buffer_id in node.writes
    )


def _imports_register_value(graph: DataflowGraph, node_id: int) -> bool:
    node = graph.node_for_id(node_id)
    return any(
        _is_group_visible_scope(graph.buffer_for_id(buffer_id).scope)
        for buffer_id in node.reads
    ) and any(
        graph.buffer_for_id(buffer_id).scope == "local.fragment"
        for buffer_id in node.writes
    )


def _validated_stages(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
) -> dict[int, dict[int, int]]:
    """Validate and copy one stage assignment for every pipeline region."""

    pipeline_region_ids = {
        region_id
        for region_id, kind in enumerate(graph.region_kinds)
        if kind == RegionKind.PIPELINE
    }
    if set(stages_by_region) != pipeline_region_ids:
        raise ValueError("stages must cover every pipeline region exactly once")

    result: dict[int, dict[int, int]] = {}
    for region_id in sorted(pipeline_region_ids):
        stages = dict(stages_by_region[region_id])
        node_ids = {node.node_id for node in graph.nodes_for_region(region_id)}
        if set(stages) != node_ids:
            raise ValueError("stages must cover every pipeline node exactly once")
        if any(stage < 0 for stage in stages.values()):
            raise ValueError("stage IDs must be non-negative")
        result[region_id] = stages

    for edge in graph.edges:
        producer = graph.node_for_id(edge.producer_id)
        consumer = graph.node_for_id(edge.consumer_id)
        if (
            producer.region_id == consumer.region_id
            and producer.region_id in result
            and effective_stage_distance(edge, result[producer.region_id]) < 0
        ):
            raise ValueError("stages violate a pipeline dependency")
    return result


def is_legal_group_cut(graph: DataflowGraph, edge: DataflowEdge) -> bool:
    """Return whether one direct dependency can be communicated across groups."""

    if graph.hardware is None:
        raise ValueError("group enumeration requires graph.hardware")
    if edge.producer_id == edge.consumer_id:
        return False
    producer = graph.node_for_id(edge.producer_id)
    consumer = graph.node_for_id(edge.consumer_id)
    if not graph.hardware.allows_cross_group_dependency(
        producer.instruction_kind,
        consumer.instruction_kind,
    ):
        return False
    if edge.buffer_id is None:
        return True
    return _is_group_visible_scope(graph.buffer_for_id(edge.buffer_id).scope)


def build_ws_opportunities(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return opportunities induced by legal direct dependencies."""

    stages = _validated_stages(graph, stages_by_region)
    if graph.hardware is None:
        raise ValueError("group enumeration requires graph.hardware")
    result: set[tuple[int, int]] = set()
    for edge in graph.edges:
        producer = graph.node_for_id(edge.producer_id)
        consumer = graph.node_for_id(edge.consumer_id)
        if producer.region_id != consumer.region_id:
            continue
        region_stages = stages.get(producer.region_id)
        if region_stages is None or not is_legal_group_cut(graph, edge):
            continue
        crosses_stage = (
            region_stages[producer.node_id]
            != region_stages[consumer.node_id]
        )
        same_stage_async = (
            not crosses_stage
            and graph.is_async(producer.node_id)
        )
        loop_carried_overlap = (
            edge.is_loop_carried
            and effective_stage_distance(edge, region_stages) > 0
        )
        if crosses_stage or same_stage_async or loop_carried_overlap:
            result.add((edge.producer_id, edge.consumer_id))
    return tuple(sorted(result))


def build_group_components(graph: DataflowGraph) -> tuple[tuple[int, ...], ...]:
    """Build whole-program components that must remain in one group."""

    if graph.hardware is None:
        raise ValueError("group enumeration requires graph.hardware")
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

    register_successors: dict[int, dict[int, set[int]]] = {}
    for edge in graph.edges:
        if not is_legal_group_cut(graph, edge):
            union(edge.producer_id, edge.consumer_id)

        if edge.buffer_id is None:
            continue
        buffer = graph.buffer_for_id(edge.buffer_id)
        if buffer.scope != "local.fragment":
            continue
        if DependencyKind.RAW in edge.dependency_kinds:
            register_successors.setdefault(buffer.buffer_id, {}).setdefault(
                edge.producer_id, set()
            ).add(edge.consumer_id)
        if (
            _is_register_initializer(graph, edge.producer_id)
            or _exports_register_value(graph, edge.consumer_id)
            or _imports_register_value(graph, edge.producer_id)
        ):
            union(edge.producer_id, edge.consumer_id)

    # A register RAW cycle cannot be split without versioned fragment handoff.
    for successors in register_successors.values():
        reachable: dict[int, set[int]] = {}
        for start in successors:
            visited: set[int] = set()
            pending = [start]
            while pending:
                current = pending.pop()
                for successor in successors.get(current, ()):
                    if successor not in visited:
                        visited.add(successor)
                        pending.append(successor)
            reachable[start] = visited
        for left, right_nodes in reachable.items():
            for right in right_nodes:
                if left in reachable.get(right, ()):
                    union(left, right)

    members: dict[int, list[int]] = {}
    for node in graph.nodes:
        members.setdefault(find(node.node_id), []).append(node.node_id)
    return tuple(
        sorted(
            (tuple(component) for component in members.values()),
            key=lambda component: component[0],
        )
    )


def _component_neighbors(
    graph: DataflowGraph,
    component_index: Mapping[int, int],
    component_count: int,
) -> tuple[frozenset[int], ...]:
    neighbors: list[set[int]] = [set() for _ in range(component_count)]
    for edge in graph.edges:
        left = component_index[edge.producer_id]
        right = component_index[edge.consumer_id]
        if left != right:
            neighbors[left].add(right)
            neighbors[right].add(left)
    return tuple(frozenset(items) for items in neighbors)


def _serial_component_choices(
    neighbors: tuple[frozenset[int], ...],
    core_groups: Mapping[int, int],
    serial_component: int,
) -> tuple[int, ...]:
    """Find existing pipeline roles reachable from one serial component."""

    choices: set[int] = set()
    visited = {serial_component}
    pending = [serial_component]
    while pending:
        current = pending.pop(0)
        for neighbor in neighbors[current]:
            if neighbor in core_groups:
                choices.add(core_groups[neighbor])
            elif neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return tuple(sorted(choices)) or (0,)


def _is_standalone_serial_component(
    graph: DataflowGraph,
    component: tuple[int, ...],
) -> bool:
    """Return whether hardware permits this serial component to own a role."""

    assert graph.hardware is not None
    return all(
        graph.region_kinds[graph.node_for_id(node_id).region_id]
        == RegionKind.SERIAL
        for node_id in component
    ) and any(
        graph.hardware.can_own_standalone_group(
            graph.node_for_id(node_id).instruction_kind
        )
        for node_id in component
    )


def enumerate_group_assignments(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    num_groups: int,
) -> Iterator[dict[int, int]]:
    """Enumerate canonical whole-program groups for fixed pipeline stages.

    A non-baseline group must either cut a stage-derived pipeline opportunity
    or contain a hardware-approved standalone serial component. Other serial
    components only attach to roles created by one of those two mechanisms.
    """

    if num_groups < 1:
        raise ValueError("num_groups must be positive")
    stages = _validated_stages(graph, stages_by_region)
    if not graph.nodes:
        return
    if num_groups == 1:
        yield {node.node_id: 0 for node in graph.nodes}
        return
    if graph.hardware is None:
        raise ValueError("group enumeration requires graph.hardware")

    opportunities = build_ws_opportunities(graph, stages)
    components = build_group_components(graph)
    component_index = {
        node_id: index
        for index, component in enumerate(components)
        for node_id in component
    }
    pipeline_components = tuple(
        index  # components id
        for index, component in enumerate(components)
        if any(
            graph.region_kinds[graph.node_for_id(node_id).region_id]
            == RegionKind.PIPELINE
            for node_id in component
        )
    )
    if not pipeline_components:
        return
    standalone_components = tuple(
        index
        for index, component in enumerate(components)
        if _is_standalone_serial_component(graph, component)
    )
    baseline_component = max(
        pipeline_components,
        key=lambda index: (len(components[index]), -components[index][0]),
    )
    role_component_set = set(pipeline_components) | set(standalone_components)
    role_components = (baseline_component,) + tuple(
        index
        for index in sorted(role_component_set)
        if index != baseline_component
    )
    ordinary_serial_components = tuple(
        index
        for index in range(len(components))
        if index not in role_component_set
    )
    if num_groups > len(role_components):
        return

    opportunity_components = tuple(
        (component_index[left], component_index[right])
        for left, right in opportunities
        if component_index[left] != component_index[right]
    )
    neighbors = _component_neighbors(graph, component_index, len(components))
    assigned: dict[int, int] = {baseline_component: 0}

    def groups_are_justified() -> bool:
        justified = {0}
        pipeline_groups = {
            assigned[component_id]
            for component_id in pipeline_components
        }
        for left, right in opportunity_components:
            left_group = assigned[left]
            right_group = assigned[right]
            if left_group != right_group:
                justified.update((left_group, right_group))
        for component_id in standalone_components:
            group_id = assigned[component_id]
            if group_id not in pipeline_groups and any(
                assigned[neighbor] != group_id
                for neighbor in neighbors[component_id]
            ):
                justified.add(group_id)
        return justified == set(range(num_groups))

    def attach_serial(
        index: int,
        core_groups: Mapping[int, int],
    ) -> Iterator[dict[int, int]]:
        if index == len(ordinary_serial_components):
            if not groups_are_justified():
                return
            result = {
                node_id: assigned[component_id]
                for component_id, component in enumerate(components)
                for node_id in component
            }
            yield dict(sorted(result.items()))
            return
        component_id = ordinary_serial_components[index]
        for group_id in _serial_component_choices(
            neighbors,
            core_groups,
            component_id,
        ):
            assigned[component_id] = group_id
            yield from attach_serial(index + 1, core_groups)
        assigned.pop(component_id, None)

    def visit(index: int, maximum_group: int) -> Iterator[dict[int, int]]:
        if index == len(role_components):
            if maximum_group + 1 != num_groups:
                return
            core_groups = {
                component_id: assigned[component_id]
                for component_id in role_components
            }
            yield from attach_serial(0, core_groups)
            return

        remaining_after_current = len(role_components) - index - 1
        largest_choice = min(maximum_group + 1, num_groups - 1)
        component_id = role_components[index]
        for group_id in range(largest_choice + 1):
            next_maximum = max(maximum_group, group_id)
            missing_groups = num_groups - (next_maximum + 1)
            if missing_groups > remaining_after_current:
                continue
            assigned[component_id] = group_id
            yield from visit(index + 1, next_maximum)
        assigned.pop(component_id, None)

    yield from visit(1, 0)
