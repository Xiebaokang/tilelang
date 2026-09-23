"""L1 overlap structure: stages, groups, orders, versions, and sync skeleton."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from OverlapPlaner.facts import FactEdge, FactGraph, RegionKind

ProgramOrders = dict[int, dict[int, dict[int, int]]]


class SynchronizationKind(str, Enum):
    FORWARD_DEPENDENCY = "forward_dependency"
    BUFFER_REUSE = "buffer_reuse"


class SynchronizationScope(str, Enum):
    ONCE = "once"
    PER_ITERATION = "per_iteration"
    REGION_BOUNDARY = "region_boundary"


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Bounds for structure search. These are budgets, not architecture constants."""

    max_groups: int = 3
    max_stages: int = 3
    stage_beam: int = 64
    max_structures: int = 128
    extra_shared_versions: int = 1
    max_version_variants: int = 8

    def __post_init__(self) -> None:
        if self.max_groups < 1 or self.max_stages < 1:
            raise ValueError("max_groups and max_stages must be positive")
        if self.stage_beam < 1 or self.max_structures < 1:
            raise ValueError("stage_beam and max_structures must be positive")
        if self.extra_shared_versions < 0:
            raise ValueError("extra_shared_versions must be non-negative")
        if self.max_version_variants < 1:
            raise ValueError("max_version_variants must be positive")


@dataclass(frozen=True, slots=True)
class SyncSkeleton:
    """One arrive/wait relation implied by a structure, without an ISA."""

    kind: SynchronizationKind
    scope: SynchronizationScope
    producer_id: int
    consumer_id: int
    producer_group: int
    consumer_group: int
    buffer_id: int
    iteration_distance: int
    effective_stage_distance: int | None
    slot_count: int

    def __post_init__(self) -> None:
        if min(self.producer_id, self.consumer_id, self.buffer_id) < 0:
            raise ValueError("synchronization IDs must be non-negative")
        # Same-group channels are valid for both async completion and shared
        # buffer reuse.  The latter is a block-wide back-pressure event: all
        # participating threads finish reading before the elected async-copy
        # thread may overwrite the slot in the next iteration.
        if self.iteration_distance < 0:
            raise ValueError("iteration_distance must be non-negative")
        if self.slot_count < 1:
            raise ValueError("slot_count must be positive")
        if self.scope == SynchronizationScope.PER_ITERATION:
            if (
                self.effective_stage_distance is None
                or self.effective_stage_distance < 0
            ):
                raise ValueError(
                    "per-iteration synchronization needs a non-negative "
                    "effective stage distance"
                )
        elif self.effective_stage_distance is not None:
            raise ValueError("non-pipeline synchronization has no stage distance")


@dataclass(frozen=True, slots=True)
class Structure:
    """One architecture-free overlap structure."""

    stages_by_region: dict[int, dict[int, int]]
    groups: dict[int, int]
    orders: ProgramOrders
    buffer_versions: dict[int, int]
    sync_edges: tuple[SyncSkeleton, ...]

    @property
    def num_groups(self) -> int:
        return len(set(self.groups.values()))


class SynchronizationCycleError(ValueError):
    """Same-epoch waits and group order form a cycle."""


def pipeline_region_ids(graph: FactGraph) -> tuple[int, ...]:
    return tuple(
        region.region_id
        for region in graph.regions
        if region.kind == RegionKind.PIPELINE
    )


def region_edges(graph: FactGraph, node_ids: frozenset[int]) -> tuple[FactEdge, ...]:
    return tuple(
        edge
        for edge in graph.edges
        if edge.producer_id in node_ids and edge.consumer_id in node_ids
    )


def validate_groups(graph: FactGraph, groups: Mapping[int, int]) -> tuple[int, ...]:
    node_ids = {node.node_id for node in graph.nodes}
    if set(groups) != node_ids:
        raise ValueError("groups must cover every node exactly once")
    if any(group_id < 0 for group_id in groups.values()):
        raise ValueError("group IDs must be non-negative")
    group_ids = tuple(sorted(set(groups.values())))
    if group_ids != tuple(range(len(group_ids))):
        raise ValueError("group IDs must be dense and non-empty")
    return group_ids


def validate_stage_group_order(
    graph: FactGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders | None = None,
) -> tuple[dict[int, dict[int, int]], tuple[int, ...]]:
    group_ids = validate_groups(graph, groups)
    expected_pipeline = set(pipeline_region_ids(graph))
    if set(stages_by_region) != expected_pipeline:
        raise ValueError("stages must cover every pipeline region exactly once")

    stages: dict[int, dict[int, int]] = {}
    for region_id in sorted(expected_pipeline):
        region_stages = dict(stages_by_region[region_id])
        region_node_ids = {
            node.node_id for node in graph.nodes_for_region(region_id)
        }
        if set(region_stages) != region_node_ids:
            raise ValueError("stages must cover every pipeline node exactly once")
        if any(stage < 0 for stage in region_stages.values()):
            raise ValueError("stage IDs must be non-negative")
        stages[region_id] = region_stages

    if orders is None:
        return stages, group_ids
    if set(orders) != set(range(len(graph.regions))):
        raise ValueError("orders must cover every region exactly once")
    for region_id, region in enumerate(graph.regions):
        region_node_ids = {
            node.node_id for node in graph.nodes_for_region(region.region_id)
        }
        if set(orders[region_id]) != set(group_ids):
            raise ValueError("each region order must cover every group")
        for group_id, local_order in orders[region_id].items():
            expected = {
                node_id
                for node_id in region_node_ids
                if groups[node_id] == group_id
            }
            if set(local_order) != expected:
                raise ValueError("group order covers the wrong region nodes")
            if set(local_order.values()) != set(range(len(expected))):
                raise ValueError("group order positions must be dense")
    return stages, group_ids


def group_local_stages(
    graph: FactGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
) -> tuple[tuple[int, int, int], ...]:
    """Subtract each region/group's leading idle stages."""

    normalized = []
    for region_id, region_stages in sorted(stages_by_region.items()):
        region_groups = {
            groups[node.node_id] for node in graph.nodes_for_region(region_id)
        }
        offsets = {
            group_id: min(
                stage
                for node_id, stage in region_stages.items()
                if groups[node_id] == group_id
            )
            for group_id in region_groups
        }
        normalized.extend(
            (region_id, node_id, stage - offsets[groups[node_id]])
            for node_id, stage in sorted(region_stages.items())
        )
    return tuple(normalized)


def structure_key(
    graph: FactGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
) -> tuple:
    """Identity of a structure before versions and synchronization."""

    return (
        group_local_stages(graph, stages_by_region, groups),
        tuple(sorted(groups.items())),
        tuple(
            (region_id, group_id, tuple(sorted(order.items())))
            for region_id, group_orders in sorted(orders.items())
            for group_id, order in sorted(group_orders.items())
        ),
    )
