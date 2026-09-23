"""Hardware-aware priorities for budgeted schedule enumeration.

The scores in this module are deliberately analytical: they only use the
dataflow graph and target instruction classes.  Benchmark results are never an
input.  A score is a search priority, not a correctness condition; all emitted
candidates still pass the exact version, synchronization, register, and shared
memory analyses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from ...headware.spec import InstructionType
from ...parse.graph import DataflowEdge, DataflowGraph, RegionKind


_EXPENSIVE_INSTRUCTIONS = frozenset({"tma", "wgmma", "function"})


def _edge_weight(graph: DataflowGraph, edge: DataflowEdge) -> float:
    """Return a bounded traffic proxy for an edge.

    Buffer sizes span several orders of magnitude, so a logarithm prevents one
    large conservative allocation from hiding every other scheduling signal.
    """

    nbytes = graph.buffer_for_id(edge.buffer_id).nbytes
    if not nbytes:
        return 1.0
    return max(1.0, min(4.0, math.log2(nbytes + 1) / 5.0))


def _cut_value(graph: DataflowGraph, edge: DataflowEdge) -> float:
    producer = graph.node_for_id(edge.producer_id).instruction
    consumer = graph.node_for_id(edge.consumer_id).instruction
    weight = _edge_weight(graph, edge)
    if producer.name == "tma" and consumer.name == "wgmma":
        return 3.0 * weight
    if producer.type != consumer.type:
        return 1.5 * weight
    if producer.name != consumer.name:
        return 0.5 * weight
    return -0.75 * weight


def score_group_assignment(
    graph: DataflowGraph,
    groups: Mapping[int, int],
) -> float:
    """Prioritize group cuts that expose independent hardware engines."""

    group_ids = set(groups.values())
    score = -0.75 * (len(group_ids) - 1)
    # A single 128-thread consumer warpgroup already occupies the whole
    # original CTA for an m64 WGMMA tile.  Adding producer groups in this case
    # usually costs more occupancy than the exposed overlap can recover.
    if graph.kernel_threads is not None and graph.kernel_threads <= 128:
        score -= 20.0 * (len(group_ids) - 1)
    for edge in graph.edges:
        if groups[edge.producer_id] != groups[edge.consumer_id]:
            score += _cut_value(graph, edge)

    tma_groups = set()
    for group_id in group_ids:
        names = {
            node.instruction.name
            for node in graph.nodes
            if groups[node.node_id] == group_id
        }
        if "tma" in names:
            tma_groups.add(group_id)
        if names == {"tma"}:
            score += 2.0
        elif "tma" in names and "wgmma" in names:
            score -= 3.0
        if names == {"generic"}:
            score -= 1.0
    # Multiple producer-only groups contend for the same TMA issue engine.
    score -= max(0, len(tma_groups) - 1) * 0.75
    return score


def _normalized_stage(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
) -> dict[int, int]:
    result: dict[int, int] = {}
    for region_id, stages in stages_by_region.items():
        for group_id in set(groups.values()):
            node_ids = [
                node.node_id
                for node in graph.nodes_for_region(region_id)
                if groups[node.node_id] == group_id
            ]
            if not node_ids:
                continue
            offset = min(stages[node_id] for node_id in node_ids)
            result.update(
                (node_id, stages[node_id] - offset) for node_id in node_ids
            )
    return result


def score_stage_assignment(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
) -> float:
    """Prioritize useful engine overlap and reject idle stage decoration."""

    stages = _normalized_stage(graph, stages_by_region, groups)
    # Every delayed node lengthens a prologue/epilogue or creates a separate
    # issue segment.  A split therefore has to earn back this base cost through
    # an expensive edge or engine-diversity benefit below.
    score = -0.9 * sum(stages.values())
    for region_id in stages_by_region:
        if graph.region_kinds[region_id] != RegionKind.PIPELINE:
            continue
        for group_id in set(groups.values()):
            node_ids = [
                node.node_id
                for node in graph.nodes_for_region(region_id)
                if groups[node.node_id] == group_id
            ]
            used = {stages[node_id] for node_id in node_ids}
            score -= 0.75 * max(0, len(used) - 1)
            expensive_stages = {
                stages[node_id]
                for node_id in node_ids
                if graph.node_for_id(node_id).instruction.name
                in _EXPENSIVE_INSTRUCTIONS
            }
            score += 1.75 * max(0, len(expensive_stages) - 1)

    for edge in graph.edges:
        if edge.producer_id not in stages or edge.consumer_id not in stages:
            continue
        if groups[edge.producer_id] != groups[edge.consumer_id]:
            continue
        producer_stage = stages[edge.producer_id]
        consumer_stage = stages[edge.consumer_id]
        if producer_stage == consumer_stage:
            continue
        producer = graph.node_for_id(edge.producer_id).instruction
        consumer = graph.node_for_id(edge.consumer_id).instruction
        value = _cut_value(graph, edge)
        # Forward stage distance can hide producer latency.  A backward split
        # is useful primarily for a loop-carried dependence whose effective
        # distance becomes zero.
        if consumer_stage > producer_stage:
            score += value
        elif edge.iteration_distance + consumer_stage - producer_stage == 0:
            score += 0.75 * max(0.0, value)
        else:
            score -= abs(value)
        if (
            producer.name not in _EXPENSIVE_INSTRUCTIONS
            and consumer.name not in _EXPENSIVE_INSTRUCTIONS
        ):
            score -= 0.5
    return score


def score_structure(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
) -> float:
    """Return the priority of a stage/group combination."""

    return score_group_assignment(graph, groups) + score_stage_assignment(
        graph, stages_by_region, groups
    )


__all__ = [
    "score_group_assignment",
    "score_stage_assignment",
    "score_structure",
]
