"""Legal local moves over stages, groups, orders, buffers, and copy backends."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from OverlapPlaner.arch import HOPPER, ClassifiedGraph, is_group_opportunity
from OverlapPlaner.arch.hopper import optional_tma_copy_ids
from OverlapPlaner.contract import to_overlap_plan
from OverlapPlaner.facts import RegionKind
from OverlapPlaner.serialization import plan_to_dict
from OverlapPlaner.structure.group import connected_components
from OverlapPlaner.structure.model import Structure
from OverlapPlaner.structure.order import enumerate_program_orders
from OverlapPlaner.structure.stage import (
    _correctness_constraint,
    _performance_constraint,
)
from OverlapPlaner.structure.sync import build_synchronizations
from OverlapPlaner.structure.version import (
    analyze_buffer_versions,
    cross_group_version_buffers,
)
from OverlapPlaner.tune.copy_search import (
    classified_for_plan,
    realize_copy_move,
    selected_optional_tma_copies,
)
from OverlapPlaner.tune.order_search import (
    _placement_maps,
    adjacent_order_swaps,
    realize_order_swap,
)

# (dimension, region/component/buffer, node/destination/count, value, order variant)
JointMove = tuple[str, int, int, int, int]


def adjacent_joint_moves(
    classified: ClassifiedGraph,
    payload: Mapping[str, Any],
    *,
    max_stages: int = 3,
    max_groups: int = 3,
) -> Iterator[JointMove]:
    """Offer legal stage, group, order, copy, and shared-version changes."""

    graph = classified.graph
    groups, stages, orders = _placement_maps(classified, payload)
    for region_id, assignment in stages.items():
        if graph.region_for_id(region_id).kind != RegionKind.PIPELINE:
            continue
        for node_id, current in assignment.items():
            for destination in (current - 1, current + 1):
                if 0 <= destination < max_stages:
                    for variant in (0, 1):
                        yield "stage", region_id, node_id, destination, variant

    group_ids = set(groups.values())
    for component in connected_components(classified):
        source = groups[component[0]]
        if any(groups[node_id] != source for node_id in component):
            continue
        if sum(group_id == source for group_id in groups.values()) == len(component):
            continue  # Keep group IDs dense.
        destinations = group_ids - {source}
        if len(group_ids) < max_groups:
            destinations = destinations | {len(group_ids)}
        for destination in sorted(destinations):
            for variant in (0, 1):
                yield "group", component[0], destination, 0, variant

    for region_id, group_id, left, right in adjacent_order_swaps(classified, payload):
        yield "order", region_id, group_id, left, right

    selected_copies = selected_optional_tma_copies(classified, payload)
    for node_id in optional_tma_copy_ids(classified):
        yield "copy", node_id, int(node_id not in selected_copies), 0, 0

    minimum = analyze_buffer_versions(graph, stages, groups, orders)
    counts = {
        int(buffer["buffer_id"]): int(buffer["version_count"])
        for buffer in payload["buffers"]
    }
    for buffer_id in cross_group_version_buffers(graph, groups):
        if not graph.buffer_for_id(buffer_id).scope.startswith("shared"):
            continue
        if counts[buffer_id] == minimum[buffer_id]:
            yield "version", buffer_id, minimum[buffer_id] + 1, 0, 0
        elif counts[buffer_id] == minimum[buffer_id] + 1:
            yield "version", buffer_id, minimum[buffer_id], 0, 0


def realize_joint_move(
    classified: ClassifiedGraph,
    payload: Mapping[str, Any],
    move: JointMove,
) -> dict[str, Any] | None:
    """Recompute versions, synchronization, and resources after a move."""

    dimension, first, second, value, variant = move
    if dimension == "copy":
        if value != 0 or variant != 0 or second not in (0, 1):
            return None
        return realize_copy_move(classified, payload, first, bool(second))
    if dimension == "order":
        return realize_order_swap(
            classified_for_plan(classified, payload),
            payload,
            (first, second, value, variant),
        )
    if dimension not in {"stage", "group", "version"} or variant not in (0, 1):
        return None
    classified = classified_for_plan(classified, payload)
    graph = classified.graph
    groups, stages, parent_orders = _placement_maps(classified, payload)
    if dimension == "stage":
        if first not in stages or second not in stages[first]:
            return None
        previous = stages[first][second]
        if abs(value - previous) != 1 or value < 0:
            return None
        stages[first][second] = value
        if set(stages[first].values()) != set(range(max(stages[first].values()) + 1)):
            return None
        for edge in graph.edges:
            if (
                edge.producer_id not in stages[first]
                or edge.consumer_id not in stages[first]
            ):
                continue
            if not _correctness_constraint(
                edge, stages[first]
            ) or not _performance_constraint(
                classified, edge, stages[first]
            ):
                return None
    elif dimension == "group":
        components = {
            component[0]: component for component in connected_components(classified)
        }
        component = components.get(first)
        group_ids = set(groups.values())
        if component is None or second not in group_ids | {len(group_ids)}:
            return None
        source = groups[component[0]]
        if source == second or any(groups[node_id] != source for node_id in component):
            return None
        if sum(group_id == source for group_id in groups.values()) == len(component):
            return None
        for node_id in component:
            groups[node_id] = second
        if any(
            groups[edge.producer_id] != groups[edge.consumer_id]
            and not is_group_opportunity(classified, edge)
            for edge in graph.edges
        ):
            return None

    try:
        if dimension == "version":
            selected = parent_orders
        else:
            orders = list(enumerate_program_orders(classified, stages, groups))
            if variant >= len(orders):
                return None
            selected = orders[variant]
        versions = analyze_buffer_versions(graph, stages, groups, selected)
        minimum_versions = dict(versions)
        for buffer in payload["buffers"]:
            buffer_id = int(buffer["buffer_id"])
            if graph.buffer_for_id(buffer_id).scope.startswith("shared"):
                versions[buffer_id] = max(
                    versions[buffer_id], int(buffer["version_count"])
                )
        if dimension == "version":
            if first not in cross_group_version_buffers(graph, groups):
                return None
            if not graph.buffer_for_id(first).scope.startswith("shared"):
                return None
            if (
                abs(second - versions[first]) != 1
                or second < minimum_versions[first]
            ):
                return None
            versions[first] = second
        sync = build_synchronizations(classified, stages, groups, selected, versions)
        structure = Structure(stages, groups, selected, versions, sync)
        parent_warps = tuple(int(group["warp_count"]) for group in payload["groups"])
        fallback = None
        for physical in HOPPER.realize(classified, structure):
            candidate = plan_to_dict(to_overlap_plan(classified, physical))
            if (
                tuple(item.warp_count for item in physical.warp_allocation.groups)
                == parent_warps
            ):
                return candidate
            if fallback is None:
                fallback = candidate
        return fallback
    except ValueError:
        return None
