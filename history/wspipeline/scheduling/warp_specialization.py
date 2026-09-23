"""Logical warp-specialization partition and program-schedule composition."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum
from collections.abc import Callable, Mapping
from typing import Iterator

from tvm.target import Target

from ..analysis.core import (
    DataflowNode,
    HardwareUnit,
)
from .order import (
    ProgramRegionSchedule,
    build_fragment_reuse_pairs,
    enumerate_region_schedules_for_stages,
)
from .stage import (
    effective_stage_distance,
    enumerate_stage_assignments,
)
from ..analysis.program import (
    ProgramRegionKind,
    ProgramDataflowAnalysis,
)

_STAGE_ASSIGNMENT_SEED_COUNT = 12


@dataclass
class LogicalWarpPartition:
    """A label-canonical assignment of every operation to one logical group."""

    node_groups: dict[DataflowNode, int]
    num_groups: int


class RegionDependencyScope(str, Enum):
    """How often a cross-group dependency is synchronized."""

    ONCE = "once"
    PER_ITERATION = "per_iteration"
    REGION_BOUNDARY = "region_boundary"


@dataclass(frozen=True)
class ProgramCrossGroupDependency:
    """A dependency between groups in the whole-scope region model."""

    producer: DataflowNode
    consumer: DataflowNode
    producer_group: int
    consumer_group: int
    producer_region: int
    consumer_region: int
    scope: RegionDependencyScope
    iteration_distance: int
    effective_stage_distance: int | None
    dependency_kinds: frozenset[str]
    buffer_id: int | None = None


@dataclass(frozen=True)
class ProgramScheduleCandidate:
    """A whole-scope partition plus one temporal schedule per region."""

    partition: LogicalWarpPartition
    region_schedules: tuple[ProgramRegionSchedule, ...]
    cross_group_dependencies: tuple[ProgramCrossGroupDependency, ...]


def build_logical_partition(
    nodes: list[DataflowNode],
    node_groups: dict[DataflowNode, int],
    num_groups: int,
) -> LogicalWarpPartition:
    if set(node_groups) != set(nodes):
        raise ValueError("partition must cover every node exactly once")
    if set(node_groups.values()) != set(range(num_groups)):
        raise ValueError("partition group IDs must be dense and non-empty")
    return LogicalWarpPartition(dict(node_groups), num_groups)


def _is_register_fragment_scope(scope: str) -> bool:
    """Return whether a buffer is a register-backed tile fragment."""

    return scope == "local.fragment"


def _is_register_initializer(node: DataflowNode) -> bool:
    """Return whether an extracted operation initializes a register value."""

    return node.name.startswith(("fill_", "clear_")) or (
        node.unit == HardwareUnit.ALU
        and not node.reads
        and "local.fragment" in node.profile.destination_scopes
    )


def _is_group_visible_scope(scope: str) -> bool:
    return scope in ("", "global") or scope.startswith("shared") or "tmem" in scope


def _exports_register_value(node: DataflowNode) -> bool:
    return (
        node.unit
        in {HardwareUnit.LOAD_STORE, HardwareUnit.TMA, HardwareUnit.TMEM}
        and "local.fragment" in node.profile.source_scopes
        and any(
            _is_group_visible_scope(scope) or scope == "local.fragment"
            for scope in node.profile.destination_scopes
        )
    )


def _imports_register_value(node: DataflowNode) -> bool:
    return (
        node.unit
        in {HardwareUnit.LOAD_STORE, HardwareUnit.TMA, HardwareUnit.TMEM}
        and "local.fragment" in node.profile.destination_scopes
        and any(
            _is_group_visible_scope(scope) for scope in node.profile.source_scopes
        )
    )


def build_colocation_groups(
    analysis: ProgramDataflowAnalysis,
) -> tuple[tuple[DataflowNode, ...], ...]:
    """Build local register-boundary placement components.

    A ``fill``/``clear`` stays with its first direct users. An explicit register
    export stays with its last producer, and an explicit import stays with its
    first consumer. Other register RAW edges may cross groups and are realized
    later with an edge-local shared-memory handoff.
    """

    node_count = len(analysis.nodes)
    parents = list(range(node_count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    register_buffer_ids = {
        descriptor.buffer_id
        for descriptor in analysis.buffers
        if _is_register_fragment_scope(descriptor.scope)
    }
    operation_ids = {
        operation.node: operation.operation_id
        for operation in analysis.operations
    }
    register_raw_successors: dict[int, dict[int, set[int]]] = {}
    for edge in analysis.edges:
        if (
            edge.buffer_id not in register_buffer_ids
            or edge.producer == edge.consumer
        ):
            continue
        producer_id = operation_ids[edge.producer]
        consumer_id = operation_ids[edge.consumer]
        if "RAW" in edge.dependency_kinds:
            register_raw_successors.setdefault(edge.buffer_id, {}).setdefault(
                producer_id, set()
            ).add(consumer_id)
        if not (
            _is_register_initializer(edge.producer)
            or _exports_register_value(edge.consumer)
            or _imports_register_value(edge.producer)
        ):
            continue
        union(producer_id, consumer_id)

    for buffer_id, successors in register_raw_successors.items():
        reachable = {}
        for start in successors:
            visited = set()
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

    members_by_root: dict[int, list[int]] = {}
    for operation_id in range(node_count):
        members_by_root.setdefault(find(operation_id), []).append(operation_id)

    constraints = []
    for operation_ids in members_by_root.values():
        if len(operation_ids) < 2:
            continue
        constraints.append(tuple(analysis.nodes[index] for index in operation_ids))
    return tuple(constraints)


def _make_colocation_partition_checker(
    constraints: tuple[tuple[DataflowNode, ...], ...],
) -> Callable[[Mapping[DataflowNode, int], int], bool]:
    constraints_by_node: dict[DataflowNode, list[tuple[DataflowNode, ...]]] = {}
    for constraint in constraints:
        for node in constraint:
            constraints_by_node.setdefault(node, []).append(constraint)

    def is_feasible(
        assignments: Mapping[DataflowNode, int], _created_groups: int
    ) -> bool:
        touched_constraints = {
            constraint
            for node in assignments
            for constraint in constraints_by_node.get(node, ())
        }
        return all(
            len(
                {
                    assignments[node]
                    for node in constraint
                    if node in assignments
                }
            )
            <= 1
            for constraint in touched_constraints
        )

    return is_feasible


def _preferred_role_partition(
    nodes: list[DataflowNode],
    num_groups: int,
    constraints: tuple[tuple[DataflowNode, ...], ...],
) -> LogicalWarpPartition | None:
    """Build a GEMM/compute/memory seed without replacing full enumeration."""

    if num_groups not in (2, 3):
        return None
    node_indices = {node: index for index, node in enumerate(nodes)}
    parents = list(range(len(nodes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for constraint in constraints:
        first = node_indices[constraint[0]]
        for node in constraint[1:]:
            union(first, node_indices[node])

    members: dict[int, list[DataflowNode]] = {}
    for node in nodes:
        members.setdefault(find(node_indices[node]), []).append(node)

    component_roles = {}
    for root, component_nodes in members.items():
        if any(node.unit == HardwareUnit.MMA for node in component_nodes):
            role = 0
        elif num_groups == 3 and any(
            node.unit in {HardwareUnit.ALU, HardwareUnit.SFU}
            for node in component_nodes
        ):
            role = 1
        else:
            role = num_groups - 1
        component_roles[root] = role

    raw_groups = {
        node: component_roles[find(node_indices[node])] for node in nodes
    }
    if set(raw_groups.values()) != set(range(num_groups)):
        return None
    canonical_ids: dict[int, int] = {}
    node_groups = {}
    for node in nodes:
        role = raw_groups[node]
        canonical_ids.setdefault(role, len(canonical_ids))
        node_groups[node] = canonical_ids[role]
    return build_logical_partition(nodes, node_groups, num_groups)


def _partition_key(partition: LogicalWarpPartition, nodes: list[DataflowNode]) -> tuple:
    return tuple(partition.node_groups[node] for node in nodes)


def enumerate_logical_warp_partitions(
    nodes: list[DataflowNode],
    num_groups: int,
    partial_assignment_is_feasible: Callable[
        [Mapping[DataflowNode, int], int], bool
    ]
    | None = None,
) -> Iterator[LogicalWarpPartition]:
    """Enumerate unlabeled, non-empty logical partitions into ``num_groups``.

    Restricted-growth group IDs remove permutations that differ only by group
    labels. Node order is used solely to canonicalize labels and is not a
    scheduling or performance constraint. Physical thread counts are absent.

    ``partial_assignment_is_feasible`` is called after every assignment with
    the currently assigned node groups and the number of groups created so
    far. Returning false prunes the whole DFS subtree. The predicate must only
    reject assignments whose every completion is illegal.
    """

    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes must not contain duplicates")
    if num_groups < 1:
        raise ValueError("num_groups must be positive")
    if num_groups > len(nodes):
        return
    if not nodes:
        return

    assignments: dict[DataflowNode, int] = {nodes[0]: 0}
    if partial_assignment_is_feasible is not None and not (
        partial_assignment_is_feasible(assignments, 1)
    ):
        return

    def visit(index: int, maximum_group: int) -> Iterator[LogicalWarpPartition]:
        if index == len(nodes):
            if maximum_group + 1 == num_groups:
                yield build_logical_partition(
                    nodes, assignments, num_groups
                )
            return

        remaining_after_current = len(nodes) - index - 1
        largest_choice = min(maximum_group + 1, num_groups - 1)
        for group_id in range(largest_choice + 1):
            next_maximum = max(maximum_group, group_id)
            missing_groups = num_groups - (next_maximum + 1)
            if missing_groups > remaining_after_current:
                continue
            assignments[nodes[index]] = group_id
            if partial_assignment_is_feasible is not None and not (
                partial_assignment_is_feasible(assignments, next_maximum + 1)
            ):
                continue
            yield from visit(index + 1, next_maximum)
        assignments.pop(nodes[index], None)

    yield from visit(1, 0)


def build_cross_group_dependencies(
    analysis: ProgramDataflowAnalysis,
    partition: LogicalWarpPartition,
    schedules: tuple[ProgramRegionSchedule, ...],
) -> tuple[ProgramCrossGroupDependency, ...]:
    """Build cross-group dependencies for a whole-scope schedule."""

    schedules_by_region = {
        schedule.region_id: schedule for schedule in schedules
    }
    dependencies = []
    for edge in analysis.edges:
        producer_group = partition.node_groups[edge.producer]
        consumer_group = partition.node_groups[edge.consumer]
        if producer_group == consumer_group:
            continue
        producer_region = analysis.operation_for(edge.producer).region_id
        consumer_region = analysis.operation_for(edge.consumer).region_id
        region = analysis.regions[producer_region]
        if (
            producer_region == consumer_region
            and region.kind == ProgramRegionKind.PIPELINE
        ):
            scope = RegionDependencyScope.PER_ITERATION
            distance = effective_stage_distance(
                edge, schedules_by_region[producer_region].stages
            )
        elif producer_region == consumer_region:
            scope = RegionDependencyScope.ONCE
            distance = None
        else:
            scope = RegionDependencyScope.REGION_BOUNDARY
            distance = None
        dependencies.append(
            ProgramCrossGroupDependency(
                edge.producer,
                edge.consumer,
                producer_group,
                consumer_group,
                producer_region,
                consumer_region,
                scope,
                edge.iteration_distance,
                distance,
                edge.dependency_kinds,
                edge.buffer_id,
            )
        )
    return tuple(dependencies)


def enumerate_logical_schedules(
    analysis: ProgramDataflowAnalysis,
    num_groups: int,
    target: Target | None = None,
    partition_is_feasible: Callable[[LogicalWarpPartition], bool] | None = None,
    partition_seed_count: int = 32,
    partition_scan_limit: int = 256,
) -> Iterator[ProgramScheduleCandidate]:
    """Enumerate stages, logical groups, then group-local orders.

    Register initialization and explicit import/export boundaries remain on
    the register-owning side. Other cross-group fragment RAW edges require an
    edge-local shared-memory handoff during physical realization.
    """

    nodes = list(analysis.nodes)
    colocation_constraints = build_colocation_groups(analysis)
    colocation_is_feasible = _make_colocation_partition_checker(
        colocation_constraints
    )
    if partition_seed_count <= 0 or partition_scan_limit <= 0:
        raise ValueError("partition ranking limits must be positive")
    role_partition = _preferred_role_partition(
        nodes, num_groups, colocation_constraints
    )
    fragment_reuse_pairs = build_fragment_reuse_pairs(analysis)

    def schedules_for_stages(
        stage_assignments: tuple[dict[DataflowNode, int], ...],
    ) -> Iterator[ProgramScheduleCandidate]:
        def partition_space() -> Iterator[LogicalWarpPartition]:
            yield from enumerate_logical_warp_partitions(
                nodes,
                num_groups,
                partial_assignment_is_feasible=colocation_is_feasible,
            )

        seeds = []
        if role_partition is not None:
            seeds.append(role_partition)
        if target is not None:
            from ..search.scoring import score_stage_partition

            scanned = {
                _partition_key(partition, nodes): partition
                for partition in itertools.islice(
                    partition_space(), partition_scan_limit
                )
            }
            if role_partition is not None:
                scanned[_partition_key(role_partition, nodes)] = role_partition
            seeds = sorted(
                scanned.values(),
                key=lambda partition: score_stage_partition(
                    analysis, stage_assignments, partition, target
                ),
            )[:partition_seed_count]

        emitted_keys = set()
        for partition in itertools.chain(seeds, partition_space()):
            partition_key = _partition_key(partition, nodes)
            if partition_key in emitted_keys:
                continue
            emitted_keys.add(partition_key)
            if partition_is_feasible is not None and not partition_is_feasible(
                partition
            ):
                continue
            selected: list[ProgramRegionSchedule] = []

            def visit(
                region_index: int,
            ) -> Iterator[ProgramScheduleCandidate]:
                if region_index == len(analysis.regions):
                    schedules = tuple(selected)
                    yield ProgramScheduleCandidate(
                        partition,
                        schedules,
                        build_cross_group_dependencies(
                            analysis, partition, schedules
                        ),
                    )
                    return
                region = analysis.regions[region_index]
                for schedule in enumerate_region_schedules_for_stages(
                    analysis,
                    region,
                    stage_assignments[region_index],
                    partition.node_groups,
                    tuple(range(partition.num_groups)),
                    fragment_reuse_pairs.get(region.region_id, ()),
                ):
                    selected.append(schedule)
                    yield from visit(region_index + 1)
                    selected.pop()

            yield from visit(0)

    stage_space = iter(enumerate_stage_assignments(analysis))
    active = [
        schedules_for_stages(stage_assignments)
        for stage_assignments in itertools.islice(
            stage_space, _STAGE_ASSIGNMENT_SEED_COUNT
        )
    ]
    while active:
        remaining = []
        for generator in active:
            try:
                candidate = next(generator)
            except StopIteration:
                continue
            remaining.append(generator)
            yield candidate
        active = remaining
    for stage_assignments in stage_space:
        yield from schedules_for_stages(stage_assignments)
