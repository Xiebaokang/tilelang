"""Explicit load/store backend choices and their synchronization realization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from OverlapPlaner.arch import HOPPER
from OverlapPlaner.arch.api import ClassifiedGraph
from OverlapPlaner.arch.hopper import optional_tma_copy_ids
from OverlapPlaner.contract import to_overlap_plan
from OverlapPlaner.physical.warp import _register_estimates, _registers_fit
from OverlapPlaner.serialization import plan_to_dict
from OverlapPlaner.structure.model import Structure
from OverlapPlaner.structure.sync import build_synchronizations
from OverlapPlaner.structure.version import analyze_buffer_versions
from OverlapPlaner.tune.order_search import _placement_maps


def selected_optional_tma_copies(
    classified: ClassifiedGraph, payload: Mapping[str, Any]
) -> frozenset[int]:
    """Read explicit choices; infer transaction loads only for legacy plans."""

    eligible = set(optional_tma_copy_ids(classified))
    legacy_transactions = {
        int(edge["producer_id"]) for edge in payload["sync_edges"]
        if int(edge["completion_mode"]) == 1
    }
    return frozenset(
        int(op["operation_id"]) for op in payload["operations"]
        if int(op["operation_id"]) in eligible
        and (op.get("copy_backend") == "tma" or (
            op.get("copy_backend") is None
            and int(op["operation_id"]) in legacy_transactions
        ))
    )


def classified_for_plan(
    classified: ClassifiedGraph, payload: Mapping[str, Any]
) -> ClassifiedGraph:
    """Carry explicit backend choices through stage/group/order mutations."""

    selected = selected_optional_tma_copies(classified, payload)
    eligible = set(optional_tma_copy_ids(classified))
    traits = list(classified.traits)
    for op in payload["operations"]:
        node_id = int(op["operation_id"])
        backend = op.get("copy_backend")
        if backend is None and node_id in eligible:
            backend = "tma" if node_id in selected else "simt"
        if backend is None:
            continue
        if backend not in ("simt", "tma"):
            raise ValueError("invalid copy backend")
        node = classified.graph.node_for_id(node_id)
        is_load = any(
            classified.graph.buffer_for_id(i).scope.startswith("shared")
            for i in node.writes
        )
        # TMA stores use a local commit/wait and are complete at the operation
        # boundary. Only loads require transaction completion channels.
        traits[node_id] = replace(
            traits[node_id], copy_backend=backend,
            async_completion=backend == "tma" and is_load,
        )
    return ClassifiedGraph(classified.graph, tuple(traits))


def realize_copy_move(
    classified: ClassifiedGraph,
    payload: Mapping[str, Any],
    node_id: int,
    enable_tma: bool,
) -> dict[str, Any] | None:
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
    return realize_copy_configuration(classified, payload, chosen)


def realize_copy_configuration(
    classified: ClassifiedGraph,
    payload: Mapping[str, Any],
    tma_node_ids: set[int] | frozenset[int],
) -> dict[str, Any] | None:
    """Change only backends and required communication/resource consequences.

    Warp widths and register quotas/actions are retained exactly. A proposal
    requiring a different allocation is rejected rather than silently turning
    a copy mutation into a register-allocation mutation.
    """

    eligible = set(optional_tma_copy_ids(classified))
    chosen = set(tma_node_ids)
    if not chosen.issubset(eligible):
        return None
    selected_payload = {**payload, "operations": [
        {**op, "copy_backend": "tma" if int(op["operation_id"]) in chosen else "simt"}
        if int(op["operation_id"]) in eligible else dict(op)
        for op in payload["operations"]
    ]}
    variant = classified_for_plan(classified, selected_payload)
    try:
        groups, stages, orders = _placement_maps(variant, payload)
        versions = analyze_buffer_versions(variant.graph, stages, groups, orders)
        for buffer in payload["buffers"]:
            bid = int(buffer["buffer_id"])
            versions[bid] = max(versions[bid], int(buffer["version_count"]))
        sync = build_synchronizations(variant, stages, groups, orders, versions)
        structure = Structure(stages, groups, orders, versions, sync)
        parent_warps = tuple(int(g["warp_count"]) for g in payload["groups"])
        quotas = tuple(g.get("register_count") for g in payload["groups"])
        actions = tuple(g.get("register_increase") for g in payload["groups"])
        if any(x is not None for x in quotas):
            if any(x is None for x in (*quotas, *actions)):
                return None
            estimates = _register_estimates(
                variant.graph, groups, orders, versions, parent_warps, HOPPER.resource()
            )
            if not _registers_fit(HOPPER.resource(), parent_warps, quotas):
                return None
            if any(need > quota for need, quota in zip(estimates, quotas)):
                return None
        for physical in HOPPER.realize(variant, structure):
            if tuple(g.warp_count for g in physical.warp_allocation.groups) != parent_warps:
                continue
            allocation = replace(
                physical.warp_allocation,
                register_counts=None if all(x is None for x in quotas) else quotas,
                register_is_increase=None if all(x is None for x in quotas) else tuple(bool(x) for x in actions),
            )
            proposal = plan_to_dict(to_overlap_plan(variant, replace(physical, warp_allocation=allocation)))
            assert proposal["groups"] == payload["groups"]
            return proposal
    except ValueError:
        pass
    return None
