"""Enumerate L1 overlap structures from a classified fact graph."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from itertools import combinations, permutations

from OverlapPlaner.arch.api import ClassifiedGraph
from OverlapPlaner.structure.group import enumerate_group_assignments
from OverlapPlaner.structure.model import (
    SearchBudget,
    Structure,
    SynchronizationCycleError,
    structure_key,
)
from OverlapPlaner.structure.order import enumerate_program_orders
from OverlapPlaner.structure.stage import enumerate_program_stages
from OverlapPlaner.structure.sync import build_synchronizations
from OverlapPlaner.structure.version import (
    analyze_buffer_versions,
    cross_group_version_buffers,
)


def _group_maps(
    classified: ClassifiedGraph,
    budget: SearchBudget,
) -> dict[int, Callable[[], Iterator[dict[int, int]]]]:
    # A newly exposed group opportunity can multiply the number of partitions.
    # Generate each group-count bucket only as the structure budget consumes it.
    def labeled(num_groups: int) -> Iterator[dict[int, int]]:
        for canonical in enumerate_group_assignments(classified, num_groups):
            # Group IDs determine physical warp intervals in the current plan
            # contract. A canonical partition alone cannot represent, e.g.,
            # the producer-first warp layout used by native Hopper kernels.
            for labels in permutations(range(num_groups)):
                yield {
                    node_id: labels[group_id]
                    for node_id, group_id in canonical.items()
                }

    return {
        num_groups: lambda count=num_groups: labeled(count)
        for num_groups in range(1, budget.max_groups + 1)
    }


def _stage_depth(stages_by_region: dict[int, dict[int, int]]) -> int:
    return max(
        (
            max(stages.values(), default=0) + 1
            for stages in stages_by_region.values()
        ),
        default=1,
    )


def _diverse_stage_group_pairs(
    group_maps: (
        Mapping[int, Callable[[], Iterator[dict[int, int]]]]
        | list[dict[int, int]]
    ),
    stage_maps: list[dict[int, dict[int, int]]],
) -> Iterator[tuple[dict[int, int], dict[int, dict[int, int]]]]:
    """Round-robin canonical buckets without predicting performance."""

    if isinstance(group_maps, Mapping):
        groups_by_count = group_maps
    else:
        lists: dict[int, list[dict[int, int]]] = {}
        for groups in group_maps:
            lists.setdefault(len(set(groups.values())), []).append(groups)
        groups_by_count = lists
    stages_by_depth: dict[int, list[dict[int, dict[int, int]]]] = {}
    for stages in stage_maps:
        stages_by_depth.setdefault(_stage_depth(stages), []).append(stages)

    def diagonal_pairs(groups, stages):
        # A Cartesian-product iterator exhausts many stage maps for the first
        # group map before seeing any later group maps.  The realized-plan
        # budget can then hide valid producer/consumer partitions entirely.
        source = iter(groups)
        cached = []
        exhausted = False
        diagonal = 0
        while stages:
            last_group = diagonal
            while len(cached) <= last_group and not exhausted:
                try:
                    cached.append(next(source))
                except StopIteration:
                    exhausted = True
            if exhausted and diagonal >= len(cached) + len(stages) - 1:
                return
            first_group = max(0, diagonal - len(stages) + 1)
            for group_index in range(
                min(last_group, len(cached) - 1), first_group - 1, -1
            ):
                yield cached[group_index], stages[diagonal - group_index]
            diagonal += 1

    iterators = {
        (group_count, stage_depth): diagonal_pairs(
            groups() if callable(groups) else groups, stages
        )
        for group_count, groups in sorted(groups_by_count.items())
        for stage_depth, stages in sorted(stages_by_depth.items())
    }
    active = list(iterators)
    while active:
        remaining = []
        for key in active:
            try:
                yield next(iterators[key])
                remaining.append(key)
            except StopIteration:
                pass
        active = remaining


def _version_variants(
    classified: ClassifiedGraph,
    groups: dict[int, int],
    minimum: dict[int, int],
    budget: SearchBudget,
) -> Iterator[dict[int, int]]:
    """Try extra ring slots for cross-group shared buffers after the minimum."""

    yield minimum
    if budget.extra_shared_versions == 0:
        return
    graph = classified.graph
    eligible = tuple(
        buffer_id
        for buffer_id in cross_group_version_buffers(graph, groups)
        if graph.buffer_for_id(buffer_id).scope.startswith("shared")
    )
    emitted = 0
    # Single-buffer changes isolate their effect; larger subsets can allow
    # several independent producers to run farther ahead at once.
    for extra in range(1, budget.extra_shared_versions + 1):
        for subset_size in range(1, len(eligible) + 1):
            for subset in combinations(eligible, subset_size):
                yield {
                    **minimum,
                    **{buffer_id: minimum[buffer_id] + extra for buffer_id in subset},
                }
                emitted += 1
                if emitted >= budget.max_version_variants:
                    return


def enumerate_structures(
    classified: ClassifiedGraph,
    budget: SearchBudget | None = None,
    *,
    arch=None,
) -> Iterator[Structure]:
    """Yield legal ``{stages, groups, orders, versions, sync}`` structures.

    Warp allocation, shared-memory packing, and ISA binding are L2.
    Enumeration includes source and issue-priority orders and all physical
    group labelings, without a performance score. Search policies consume the
    feasible plans produced downstream.
    """

    del arch  # Kept for source compatibility; selection no longer uses it.
    budget = budget or SearchBudget()
    graph = classified.graph
    seen: set[tuple] = set()
    count = 0
    stage_maps = list(
        enumerate_program_stages(
            classified,
            budget,
        )
    )
    pairs = _diverse_stage_group_pairs(
        _group_maps(classified, budget), stage_maps
    )
    for groups, stages_by_region in pairs:
        for orders in enumerate_program_orders(
            classified, stages_by_region, groups
        ):
            key = structure_key(graph, stages_by_region, groups, orders)
            if key in seen:
                continue
            seen.add(key)
            try:
                minimum = analyze_buffer_versions(
                    graph, stages_by_region, groups, orders
                )
            except ValueError:
                continue
            for versions in _version_variants(classified, groups, minimum, budget):
                try:
                    sync_edges = build_synchronizations(
                        classified, stages_by_region, groups, orders, versions
                    )
                except (ValueError, SynchronizationCycleError):
                    continue
                yield Structure(
                    stages_by_region=stages_by_region,
                    groups=groups,
                    orders=orders,
                    buffer_versions=versions,
                    sync_edges=sync_edges,
                )
                count += 1
                if count >= budget.max_structures:
                    return
