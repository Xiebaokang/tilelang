"""Schedule-local TMA alternatives for capability-checked small copies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from OverlapPlaner.arch import HOPPER
from OverlapPlaner.arch.api import ClassifiedGraph
from OverlapPlaner.arch.hopper import optional_tma_copy_ids
from OverlapPlaner.contract import to_overlap_plan
from OverlapPlaner.serialization import plan_to_dict
from OverlapPlaner.structure.model import Structure
from OverlapPlaner.structure.sync import build_synchronizations
from OverlapPlaner.tune.order_search import _placement_maps


def selected_optional_tma_copies(
    classified: ClassifiedGraph, payload: Mapping[str, Any]
) -> frozenset[int]:
    """Recover copy choices from transaction-completion channels."""

    eligible = set(optional_tma_copy_ids(classified))
    return frozenset(
        int(edge["producer_id"])
        for edge in payload["sync_edges"]
        if int(edge["completion_mode"]) == 1
        and int(edge["producer_id"]) in eligible
    )


def classified_for_plan(
    classified: ClassifiedGraph, payload: Mapping[str, Any]
) -> ClassifiedGraph:
    """Use a plan's copy choices while recomputing its other dimensions."""

    selected = selected_optional_tma_copies(classified, payload)
    if not selected:
        return classified
    traits = list(classified.traits)
    for node_id in selected:
        traits[node_id] = replace(traits[node_id], async_completion=True)
    return ClassifiedGraph(classified.graph, tuple(traits))


def realize_copy_move(
    classified: ClassifiedGraph,
    payload: Mapping[str, Any],
    node_id: int,
    enable_tma: bool,
) -> dict[str, Any] | None:
    """Rebuild barriers and resources while retaining placement and warp widths."""

    eligible = optional_tma_copy_ids(classified)
    if node_id not in eligible:
        return None
    chosen = set(selected_optional_tma_copies(classified, payload))
    if (node_id in chosen) == enable_tma:
        return None
    if enable_tma:
        chosen.add(node_id)
    else:
        chosen.remove(node_id)
    traits = list(classified.traits)
    for selected_id in chosen:
        traits[selected_id] = replace(traits[selected_id], async_completion=True)
    variant = ClassifiedGraph(classified.graph, tuple(traits))
    try:
        groups, stages, orders = _placement_maps(classified, payload)
        versions = {
            int(buffer["buffer_id"]): int(buffer["version_count"])
            for buffer in payload["buffers"]
        }
        sync = build_synchronizations(variant, stages, groups, orders, versions)
        structure = Structure(stages, groups, orders, versions, sync)
        parent_warps = tuple(int(group["warp_count"]) for group in payload["groups"])
        for physical in HOPPER.realize(variant, structure):
            if tuple(
                group.warp_count for group in physical.warp_allocation.groups
            ) == parent_warps:
                proposal = plan_to_dict(to_overlap_plan(variant, physical))
                has_tma = node_id in selected_optional_tma_copies(
                    classified, proposal
                )
                if has_tma == enable_tma:
                    return proposal
    except ValueError:
        pass
    return None
