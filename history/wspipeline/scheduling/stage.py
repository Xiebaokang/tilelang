"""Pipeline stage assignment search with correctness and structural priority."""

from __future__ import annotations

import heapq
import itertools
from typing import Iterator

from tvm import tirx

from ..analysis.core import (
    DataflowEdge,
    DataflowNode,
    ExecutionKind,
    HardwareUnit,
)
from ..analysis.program import (
    ProgramDataflowAnalysis,
    ProgramRegion,
    ProgramRegionKind,
)


def _execution_role(node: DataflowNode) -> str:
    if node.unit in {
        HardwareUnit.LOAD_STORE,
        HardwareUnit.TMA,
        HardwareUnit.TMEM,
    }:
        return "memory"
    if node.unit == HardwareUnit.MMA:
        return "mma"
    return "scalar"


def infer_max_stages(
    analysis: ProgramDataflowAnalysis,
    region: ProgramRegion,
) -> int:
    """Infer a finite useful stage-depth bound from the program structure."""

    nodes = analysis.nodes_for_region(region)
    if region.kind != ProgramRegionKind.PIPELINE or not nodes:
        return 1
    roles = {_execution_role(node) for node in nodes}
    async_roles = {
        _execution_role(node)
        for node in nodes
        if node.execution_kind
        in {ExecutionKind.ASYNC_COMPUTE, ExecutionKind.ASYNC_MEMORY}
    }
    maximum = min(len(nodes), len(roles) + len(async_roles))
    if region.loop is not None and isinstance(region.loop.extent, tirx.IntImm):
        maximum = min(maximum, max(1, int(region.loop.extent)))
    return max(1, maximum)


def _priority_predecessors(
    nodes: list[DataflowNode], edges: list[DataflowEdge]
) -> dict[DataflowNode, tuple[DataflowNode, ...]]:
    """Return unique same-iteration predecessors used to rank stage splits."""

    predecessors: dict[DataflowNode, set[DataflowNode]] = {
        node: set() for node in nodes
    }
    for edge in edges:
        if edge.is_loop_carried or edge.producer == edge.consumer:
            continue
        predecessors[edge.consumer].add(edge.producer)
    node_positions = {node: index for index, node in enumerate(nodes)}
    return {
        node: tuple(sorted(items, key=node_positions.__getitem__))
        for node, items in predecessors.items()
    }


def topological_order(
    nodes: list[DataflowNode], edges: list[DataflowEdge]
) -> list[DataflowNode]:
    """Return a stable topological order and reject cyclic graphs."""

    incoming_count = {node: 0 for node in nodes}
    users: dict[DataflowNode, list[DataflowNode]] = {node: [] for node in nodes}
    for edge in edges:
        if edge.is_loop_carried:
            continue
        incoming_count[edge.consumer] += 1
        users[edge.producer].append(edge.consumer)

    ready = [node for node in nodes if incoming_count[node] == 0]
    order: list[DataflowNode] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for user in users[node]:
            incoming_count[user] -= 1
            if incoming_count[user] == 0:
                ready.append(user)
    if len(order) != len(nodes):
        raise ValueError("the payload graph contains a cycle")
    return order


def enumerate_feasible_stages(
    nodes: list[DataflowNode], edges: list[DataflowEdge], budget: int
) -> Iterator[dict[DataflowNode, int]]:
    """Enumerate every translation-normalized legal stage assignment.

    Intra-iteration dependencies require ``S(producer) <= S(consumer)``.
    A loop-carried dependency of distance ``d`` requires
    ``S(producer) <= S(consumer) + d``. Requiring the assigned stages to span
    exactly ``[0, budget]`` keeps one translation-normalized representative
    and ensures that the configured pipeline depth is actually used.

    Legal assignments are ordered by three structural preferences: minimize
    same-hardware-unit splits, minimize unsplit cross-unit dependencies, then
    minimize the total same-iteration stage gap. These preferences never
    remove a legal assignment. Loop-carried edges do not participate in the
    ranking because their stage distance represents inter-iteration timing.
    """

    if budget < 0:
        raise ValueError("the stage budget cannot be negative")

    order = topological_order(nodes, edges)
    if not order:
        return
    incoming: dict[DataflowNode, list[DataflowEdge]] = {node: [] for node in nodes}
    for edge in edges:
        if not edge.is_loop_carried:
            incoming[edge.consumer].append(edge)

    priority_predecessors = _priority_predecessors(nodes, edges)
    # Imported lazily to keep the correctness search independent from the
    # higher-level scoring module during package initialization.
    from ..search.scoring import score_stage_dependency

    positions = {node: index for index, node in enumerate(order)}
    incoming_positions = {
        node: tuple(positions[edge.producer] for edge in incoming[node])
        for node in order
    }
    predecessor_positions = {
        node: tuple(positions[item] for item in priority_predecessors[node])
        for node in order
    }
    completed_constraints: list[list[tuple[int, int, int, bool]]] = [
        [] for _ in order
    ]
    for edge in edges:
        producer = positions[edge.producer]
        consumer = positions[edge.consumer]
        completed_constraints[max(producer, consumer)].append(
            (producer, consumer, edge.iteration_distance, edge.is_loop_carried)
        )

    counter = itertools.count()
    queue: list[tuple[tuple[int, int, int], int, tuple[int, ...]]] = [
        ((0, 0, 0), next(counter), ())
    ]
    while queue:
        score, _, assigned = heapq.heappop(queue)
        index = len(assigned)
        if index == len(order):
            if min(assigned) == 0 and max(assigned) == budget:
                yield dict(zip(order, assigned))
            continue

        node = order[index]
        first_stage = max(
            (assigned[position] for position in incoming_positions[node]),
            default=0,
        )
        for stage in range(first_stage, budget + 1):
            next_assigned = assigned + (stage,)
            if any(
                (
                    next_assigned[producer]
                    > next_assigned[consumer]
                    + (distance if is_loop_carried else 0)
                )
                for producer, consumer, distance, is_loop_carried
                in completed_constraints[index]
            ):
                continue
            added = [0, 0, 0]
            for predecessor_position in predecessor_positions[node]:
                dependency_score = score_stage_dependency(
                    order[predecessor_position],
                    node,
                    assigned[predecessor_position],
                    stage,
                )
                for score_index, value in enumerate(dependency_score):
                    added[score_index] += value
            next_score = tuple(
                current + increment
                for current, increment in zip(score, added)
            )
            heapq.heappush(
                queue,
                (next_score, next(counter), next_assigned),
            )


def _enumerate_region_stages(
    analysis: ProgramDataflowAnalysis,
    region: ProgramRegion,
) -> Iterator[dict[DataflowNode, int]]:
    """Enumerate the stage assignment for one program region."""

    nodes = list(analysis.nodes_for_region(region))
    if region.kind == ProgramRegionKind.SERIAL:
        yield {}
        return

    if not nodes:
        yield {}
        return
    if region.auto_schedule:
        stage_counts = range(1, infer_max_stages(analysis, region) + 1)
    elif region.num_stages is not None:
        stage_counts = (region.num_stages,)
    else:
        raise ValueError("pipeline region must provide num_stages or enable auto_wsp")
    edges = list(analysis.edges_for_region(region))
    active = [
        enumerate_feasible_stages(nodes, edges, num_stages - 1)
        for num_stages in stage_counts
    ]
    while active:
        remaining = []
        for generator in active:
            try:
                stages = next(generator)
            except StopIteration:
                continue
            remaining.append(generator)
            yield stages
        active = remaining


def enumerate_stage_assignments(
    analysis: ProgramDataflowAnalysis,
) -> Iterator[tuple[dict[DataflowNode, int], ...]]:
    """Enumerate one stage assignment for every ordered program region."""

    selected: list[dict[DataflowNode, int]] = []

    def visit(
        region_index: int,
    ) -> Iterator[tuple[dict[DataflowNode, int], ...]]:
        if region_index == len(analysis.regions):
            yield tuple(dict(stages) for stages in selected)
            return
        for stages in _enumerate_region_stages(
            analysis, analysis.regions[region_index]
        ):
            selected.append(stages)
            yield from visit(region_index + 1)
            selected.pop()

    yield from visit(0)


def effective_stage_distance(
    edge: DataflowEdge, stages: dict[DataflowNode, int]
) -> int:
    """Return the pipeline-time distance from producer to consumer."""

    return (
        edge.iteration_distance
        + stages[edge.consumer]
        - stages[edge.producer]
    )
