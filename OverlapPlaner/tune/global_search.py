"""Bounded, parent-independent sampling of the legal schedule space.

This is not uniform sampling and does not guarantee an optimum. Each attempt
starts afresh; neither the initial pool nor the stage beam limits its choices.
"""

from __future__ import annotations

import random

from OverlapPlaner.arch import HOPPER
from OverlapPlaner.arch.api import ClassifiedGraph
from OverlapPlaner.arch.hopper import optional_tma_copy_ids
from OverlapPlaner.contract import to_overlap_plan
from OverlapPlaner.facts import RegionKind
from OverlapPlaner.serialization import plan_to_dict
from OverlapPlaner.structure.group import connected_components, build_group_opportunities
from OverlapPlaner.structure.model import SearchBudget, Structure, region_edges
from OverlapPlaner.structure.order import _project_group_orders
from OverlapPlaner.structure.stage import (
    _correctness_constraint,
    _performance_constraint,
    effective_stage_distance,
    region_stage_limit,
)
from OverlapPlaner.structure.sync import build_synchronizations
from OverlapPlaner.structure.version import analyze_buffer_versions, cross_group_version_buffers
from OverlapPlaner.tune.copy_search import classified_for_plan


def sample_structure(
    classified: ClassifiedGraph, budget: SearchBudget, rng: random.Random
) -> Structure | None:
    """One bounded attempt; abandon dead ends instead of exponential backtracking."""
    graph = classified.graph
    components = connected_components(classified)
    if not components:
        return None
    count = rng.randint(1, min(budget.max_groups, len(components)))
    labels = [rng.randrange(count) for _ in components]
    if set(labels) != set(range(count)):
        return None
    groups = {
        node: label
        for component, label in zip(components, labels)
        for node in component
    }
    if count > 1:
        participating = set()
        for edge in build_group_opportunities(classified):
            left, right = groups[edge.producer_id], groups[edge.consumer_id]
            if left != right:
                participating.update((left, right))
        if participating != set(range(count)):
            return None

    stages = {}
    orders = {}
    for region in graph.regions:
        nodes = frozenset(n.node_id for n in graph.nodes_for_region(region.region_id))
        edges = region_edges(graph, nodes)
        pipeline = region.kind == RegionKind.PIPELINE
        if pipeline:
            depth = rng.randint(
                1,
                min(
                    region_stage_limit(region, budget.max_stages),
                    max(1, len(nodes)),
                ),
            )
            assignment = {}
            for node in graph.topological_order():
                if node not in nodes:
                    continue
                choices = list(range(depth))
                rng.shuffle(choices)
                for stage in choices:
                    assignment[node] = stage
                    completed = [
                        e for e in edges
                        if e.producer_id in assignment and e.consumer_id in assignment
                    ]
                    if all(
                        _correctness_constraint(e, assignment)
                        and _performance_constraint(classified, e, assignment)
                        for e in completed
                    ):
                        break
                else:
                    return None
            if set(assignment.values()) != set(range(depth)):
                return None
            stages[region.region_id] = assignment
        pending = {node: set() for node in nodes}
        for edge in edges:
            if edge.producer_id != edge.consumer_id and (
                not pipeline or effective_stage_distance(edge, stages[region.region_id]) == 0
            ):
                pending[edge.consumer_id].add(edge.producer_id)
        order = []
        while pending:
            ready = sorted(node for node, deps in pending.items() if not deps)
            if not ready:
                return None
            node = rng.choice(ready)
            order.append(node)
            del pending[node]
            for deps in pending.values():
                deps.discard(node)
        orders[region.region_id] = _project_group_orders(
            tuple(order), groups, tuple(range(count))
        )
    versions = analyze_buffer_versions(graph, stages, groups, orders)
    for bid in cross_group_version_buffers(graph, groups):
        if graph.buffer_for_id(bid).scope.startswith("shared"):
            choices = [versions[bid]]
            choices.extend(
                max(versions[bid], 2) + extra
                for extra in range(budget.extra_shared_versions + 1)
            )
            versions[bid] = rng.choice(tuple(dict.fromkeys(choices)))
    sync = build_synchronizations(classified, stages, groups, orders, versions)
    return Structure(stages, groups, orders, versions, sync)


def sample_global_plan(classified: ClassifiedGraph, budget: SearchBudget, seed: int):
    """Sample backends before legality, then jointly sample structure and resources."""
    rng = random.Random(seed)
    eligible = optional_tma_copy_ids(classified)
    # Give both endpoints non-vanishing probability even with many copies.
    mode = rng.randrange(3)
    operations = [
        {
            "operation_id": node,
            "copy_backend": (
                "tma" if mode == 2 or (mode == 1 and rng.randrange(2)) else "simt"
            ),
        }
        for node in eligible
    ]
    variant = classified_for_plan(classified, {"operations": operations, "sync_edges": []})
    try:
        structure = sample_structure(variant, budget, rng)
        if structure is None:
            return None
        # Reservoir sampling avoids always selecting the first warp/quota choice.
        # Physical choices are bounded by the hardware warp count.
        selected = None
        for count, physical in enumerate(HOPPER.realize(variant, structure), 1):
            if rng.randrange(count) == 0:
                selected = physical
        if selected is not None:
            return plan_to_dict(to_overlap_plan(variant, selected))
    except ValueError:
        # Structural/resource illegality is an expected rejected proposal.
        return None
    return None
