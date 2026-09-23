"""Legal, measurement-driven local moves in the operation-order space."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from OverlapPlaner.arch import HOPPER, ClassifiedGraph
from OverlapPlaner.contract import to_overlap_plan
from OverlapPlaner.facts import RegionKind
from OverlapPlaner.serialization import plan_to_dict
from OverlapPlaner.structure.model import Structure, validate_stage_group_order
from OverlapPlaner.structure.stage import effective_stage_distance
from OverlapPlaner.structure.sync import build_synchronizations
from OverlapPlaner.structure.version import analyze_buffer_versions

OrderSwap = tuple[int, int, int, int]  # region, group, earlier node, later node


def _placement_maps(classified: ClassifiedGraph, payload: Mapping[str, Any]):
    graph = classified.graph
    placements = {int(item["operation_id"]): item for item in payload["operations"]}
    if set(placements) != {node.node_id for node in graph.nodes}:
        raise ValueError("plan operations do not match the current fact graph")
    groups = {node_id: int(item["group_id"]) for node_id, item in placements.items()}
    stages = {
        region.region_id: {
            node.node_id: int(placements[node.node_id]["stage"])
            for node in graph.nodes_for_region(region.region_id)
        }
        for region in graph.regions
        if region.kind == RegionKind.PIPELINE
    }
    group_ids = tuple(sorted(set(groups.values())))
    orders = {
        region.region_id: {
            group_id: {
                node.node_id: int(placements[node.node_id]["order"])
                for node in graph.nodes_for_region(region.region_id)
                if groups[node.node_id] == group_id
            }
            for group_id in group_ids
        }
        for region in graph.regions
    }
    validate_stage_group_order(graph, stages, groups, orders)
    return groups, stages, orders


def adjacent_order_swaps(
    classified: ClassifiedGraph, payload: Mapping[str, Any]
) -> Iterator[OrderSwap]:
    """Yield adjacent group-local swaps not forbidden by a same-epoch edge."""

    graph = classified.graph
    _, stages, orders = _placement_maps(classified, payload)
    for region_id, region_orders in orders.items():
        region = graph.region_for_id(region_id)
        for group_id, local_order in region_orders.items():
            sequence = sorted(local_order, key=local_order.__getitem__)
            for left, right in zip(sequence, sequence[1:]):
                dependent = any(
                    edge.producer_id == left
                    and edge.consumer_id == right
                    and (
                        region.kind != RegionKind.PIPELINE
                        or effective_stage_distance(edge, stages[region_id]) == 0
                    )
                    for edge in graph.edges
                )
                if not dependent:
                    yield region_id, group_id, left, right


def realize_order_swap(
    classified: ClassifiedGraph,
    payload: Mapping[str, Any],
    swap: OrderSwap,
) -> dict[str, Any] | None:
    """Recompute all order-dependent state, retaining the parent's warp widths."""

    graph = classified.graph
    groups, stages, orders = _placement_maps(classified, payload)
    if swap not in set(adjacent_order_swaps(classified, payload)):
        return None
    region_id, group_id, left, right = swap
    orders[region_id][group_id][left], orders[region_id][group_id][right] = (
        orders[region_id][group_id][right],
        orders[region_id][group_id][left],
    )
    try:
        versions = analyze_buffer_versions(graph, stages, groups, orders)
        sync_edges = build_synchronizations(
            classified, stages, groups, orders, versions
        )
        structure = Structure(stages, groups, orders, versions, sync_edges)
        parent_warps = tuple(int(group["warp_count"]) for group in payload["groups"])
        for physical in HOPPER.realize(classified, structure):
            if tuple(
                group.warp_count for group in physical.warp_allocation.groups
            ) == parent_warps:
                return plan_to_dict(to_overlap_plan(classified, physical))
    except ValueError:
        pass
    return None
