"""Enumerate and serialize complete Overlaper schedule candidates."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

from .analysis.physical import (
    GroupWarpAllocation,
    SharedBufferAllocation,
    SharedMemoryPlan,
    WarpAllocation,
    analyze_shared_memory,
    enumerate_warp_allocations,
    warp_requirements_are_feasible,
)
from .analysis.schedule import (
    MAX_NUM_GROUPS,
    ProgramOrders,
    Synchronization,
    SynchronizationCycleError,
    SynchronizationKind,
    SynchronizationScope,
    build_program_orders,
    build_synchronizations,
    enumerate_buffer_version_plans,
    enumerate_bounded_group_assignments,
    enumerate_bounded_stage_assignments,
    enumerate_group_assignments,
    enumerate_stage_assignments,
    multiversion_candidate_buffers,
    score_group_assignment,
    score_structure,
)
from .parse import DataflowGraph, RegionKind


@dataclass(frozen=True, slots=True)
class Schedule:
    """One complete, resource-feasible Overlaper schedule."""

    stages_by_region: dict[int, dict[int, int]]
    groups: dict[int, int]
    orders: ProgramOrders
    buffer_versions: dict[int, int]
    synchronizations: tuple[Synchronization, ...]
    warp_allocation: WarpAllocation
    shared_memory: SharedMemoryPlan

    @property
    def num_groups(self) -> int:
        return len(set(self.groups.values()))


@dataclass(frozen=True, slots=True)
class GuidedSearchConfig:
    """Budgets for hardware-aware schedule enumeration."""

    groups_per_count: int = 16
    structures: int = 2048
    schedules: int = 128

    def __post_init__(self) -> None:
        if min(self.groups_per_count, self.structures, self.schedules) < 1:
            raise ValueError("guided search budgets must be positive")


def _group_local_stages(
    graph: DataflowGraph,
    stages_by_region: dict[int, dict[int, int]],
    groups: dict[int, int],
) -> tuple[tuple[int, int, int], ...]:
    """Return stages after removing each region/group's leading idle stages."""

    normalized = []
    for region_id, stages in sorted(stages_by_region.items()):
        offsets = {
            group_id: min(
                stage
                for node_id, stage in stages.items()
                if groups[node_id] == group_id
            )
            for group_id in {
                groups[node.node_id]
                for node in graph.nodes_for_region(region_id)
            }
        }
        normalized.extend(
            (region_id, node_id, stage - offsets[groups[node_id]])
            for node_id, stage in sorted(stages.items())
        )
    return tuple(normalized)


def _schedule_equivalence_key(
    graph: DataflowGraph,
    schedule: Schedule,
) -> tuple[Any, ...]:
    """Return the code-generation identity of a complete schedule.

    Absolute stage offsets disappear after groups are split.  Effective stage
    distance is analysis metadata; synchronization lowering is determined by
    the channel's iteration distance and slot count.
    """

    synchronization_key = tuple(
        (
            item.kind.value,
            item.scope.value,
            item.producer_id,
            item.consumer_id,
            item.producer_group,
            item.consumer_group,
            item.buffer_id,
            item.iteration_distance,
            item.slot_count,
        )
        for item in schedule.synchronizations
    )
    warp_key = (
        tuple(
            (item.group_id, item.first_warp, item.warp_count)
            for item in schedule.warp_allocation.groups
        ),
        schedule.warp_allocation.effective_threads,
        schedule.warp_allocation.register_counts,
        schedule.warp_allocation.register_is_increase,
    )
    return (
        _group_local_stages(graph, schedule.stages_by_region, schedule.groups),
        tuple(sorted(schedule.groups.items())),
        tuple(
            (region_id, group_id, tuple(sorted(order.items())))
            for region_id, group_orders in sorted(schedule.orders.items())
            for group_id, order in sorted(group_orders.items())
        ),
        tuple(sorted(schedule.buffer_versions.items())),
        synchronization_key,
        warp_key,
        tuple(
            (
                item.buffer_id,
                item.start,
                item.end,
                item.size_bytes,
                item.alignment,
                item.byte_offset,
            )
            for item in schedule.shared_memory.shared_allocations
        ),
        schedule.shared_memory.synchronization_bytes,
        schedule.shared_memory.merged_shared_bytes,
    )


def _stage_group_order_equivalence_key(
    graph: DataflowGraph,
    stages_by_region: dict[int, dict[int, int]],
    groups: dict[int, int],
    orders: ProgramOrders,
) -> tuple[Any, ...]:
    """Return the pre-version identity after removing group-local idle stages."""

    return (
        _group_local_stages(graph, stages_by_region, groups),
        tuple(sorted(groups.items())),
        tuple(
            (region_id, group_id, tuple(sorted(order.items())))
            for region_id, group_orders in sorted(orders.items())
            for group_id, order in sorted(group_orders.items())
        ),
    )


def _program_stage_assignments(
    graph: DataflowGraph,
) -> Iterator[dict[int, dict[int, int]]]:
    pipeline_regions = tuple(
        region_id
        for region_id, kind in enumerate(graph.region_kinds)
        if kind == RegionKind.PIPELINE
    )
    choices = tuple(
        tuple(enumerate_stage_assignments(graph, region_id))
        for region_id in pipeline_regions
    )
    for assignment in product(*choices):
        yield {
            region_id: stages
            for region_id, stages in zip(pipeline_regions, assignment)
        }


def _guided_program_stage_assignments(
    graph: DataflowGraph,
    limit: int,
) -> tuple[dict[int, dict[int, int]], ...]:
    """Build a bounded cross-region stage frontier."""

    pipeline_regions = tuple(
        region_id
        for region_id, kind in enumerate(graph.region_kinds)
        if kind == RegionKind.PIPELINE
    )
    if not pipeline_regions:
        return ({},)
    baseline_groups = {node.node_id: 0 for node in graph.nodes}
    states: list[dict[int, dict[int, int]]] = [{}]
    for region_id in pipeline_regions:
        choices = tuple(
            enumerate_bounded_stage_assignments(
                graph,
                region_id,
                beam_width=limit,
                score=lambda stages, selected_region=region_id: (
                    score_structure(
                        graph,
                        {selected_region: dict(stages)},
                        baseline_groups,
                    )
                ),
            )
        )
        expanded = []
        for current in states:
            for stages in choices:
                updated = {**current, region_id: stages}
                key = tuple(
                    (candidate_region, tuple(sorted(values.items())))
                    for candidate_region, values in sorted(updated.items())
                )
                expanded.append(
                    (
                        score_structure(graph, updated, baseline_groups),
                        key,
                        updated,
                    )
                )
        expanded.sort(key=lambda item: (-item[0], item[1]))
        states = [item[2] for item in expanded[:limit]]
    return tuple(states)


def enumerate_schedules(
    graph: DataflowGraph,
    original_threads: int | None = None,
) -> Iterator[Schedule]:
    """Yield every complete candidate allowed by the current fixed bounds."""

    if graph.hardware is None:
        raise ValueError("schedule enumeration requires graph.hardware")
    threads = graph.kernel_threads if original_threads is None else original_threads
    if threads is None:
        raise ValueError("original thread count is unknown")

    seen_stage_group_orders: set[tuple[Any, ...]] = set()
    seen_schedules: set[tuple[Any, ...]] = set()
    for stages_by_region in _program_stage_assignments(graph):
        for num_groups in range(1, MAX_NUM_GROUPS + 1):
            for groups in enumerate_group_assignments(graph, num_groups):
                if not warp_requirements_are_feasible(graph, groups):
                    continue
                orders = build_program_orders(graph, stages_by_region, groups)
                stage_group_order_key = _stage_group_order_equivalence_key(
                    graph, stages_by_region, groups, orders
                )
                if stage_group_order_key in seen_stage_group_orders:
                    continue
                seen_stage_group_orders.add(stage_group_order_key)
                for versions in enumerate_buffer_version_plans(
                    graph, stages_by_region, groups, orders
                ):
                    try:
                        synchronizations = build_synchronizations(
                            graph,
                            stages_by_region,
                            groups,
                            orders,
                            versions,
                        )
                    except SynchronizationCycleError:
                        continue
                    shared_memory = analyze_shared_memory(
                        graph,
                        groups,
                        orders,
                        versions,
                        synchronizations,
                    )
                    if not shared_memory.fits:
                        continue
                    for warp_allocation in enumerate_warp_allocations(
                        graph,
                        groups,
                        orders,
                        versions,
                        original_threads=threads,
                    ):
                        schedule = Schedule(
                            stages_by_region={
                                region_id: dict(stages)
                                for region_id, stages in stages_by_region.items()
                            },
                            groups=dict(groups),
                            orders={
                                region_id: {
                                    group_id: dict(order)
                                    for group_id, order in group_orders.items()
                                }
                                for region_id, group_orders in orders.items()
                            },
                            buffer_versions=dict(versions),
                            synchronizations=synchronizations,
                            warp_allocation=warp_allocation,
                            shared_memory=shared_memory,
                        )
                        key = _schedule_equivalence_key(graph, schedule)
                        if key in seen_schedules:
                            continue
                        seen_schedules.add(key)
                        yield schedule


def _ranked_group_assignments(
    graph: DataflowGraph,
    per_count: int,
) -> tuple[dict[int, int], ...]:
    """Keep a small, diverse set of analytically promising partitions."""

    selected = []
    for num_groups in range(1, MAX_NUM_GROUPS + 1):
        ranked = sorted(
            enumerate_bounded_group_assignments(
                graph,
                num_groups,
                beam_width=max(64, per_count * 8),
                score=lambda groups: score_group_assignment(graph, groups),
            ),
            key=lambda groups: (
                -score_group_assignment(graph, groups),
                tuple(sorted(groups.items())),
            ),
        )
        selected.extend(ranked[:per_count])
    return tuple(selected)


def _full_schedule_score(graph: DataflowGraph, schedule: Schedule) -> float:
    """Refine structural priority with exact physical-analysis results."""

    score = score_structure(
        graph, schedule.stages_by_region, schedule.groups
    )
    for buffer_id in multiversion_candidate_buffers(graph, schedule.groups):
        versions = schedule.buffer_versions[buffer_id]
        nbytes = graph.buffer_for_id(buffer_id).nbytes or 0
        traffic = max(1.0, min(4.0, (nbytes.bit_length() - 1) / 5.0))
        if versions == 2:
            score += 0.75 * traffic
        elif versions == 3:
            score += 0.55 * traffic
        else:
            score -= (versions - 3) * traffic

    resource = graph.hardware.device_resource
    score -= 0.35 * max(
        0,
        schedule.warp_allocation.effective_threads
        // resource.specialized_group_warp_multiple
        // resource.warp_size
        - 1,
    )
    shared_ratio = (
        schedule.shared_memory.merged_shared_bytes
        + schedule.shared_memory.synchronization_bytes
    ) / resource.shared_memory_capacity_bytes
    if shared_ratio > 0.85:
        score -= 8.0 * (shared_ratio - 0.85)

    allocation = schedule.warp_allocation
    if allocation.register_counts is not None:
        register_usage = sum(
            item.warp_count * resource.warp_size * registers
            for item, registers in zip(
                allocation.groups, allocation.register_counts
            )
        )
        score -= 2.0 * register_usage / resource.register_file_capacity
    score -= 0.1 * len(schedule.synchronizations)
    return score


def enumerate_guided_schedules(
    graph: DataflowGraph,
    original_threads: int | None = None,
    *,
    config: GuidedSearchConfig | None = None,
) -> Iterator[Schedule]:
    """Yield a budgeted schedule set ranked by graph and hardware features.

    Correctness is unchanged from :func:`enumerate_schedules`.  The only
    difference is that low-priority stage/group products are discarded before
    version, synchronization, shared-memory, and warp-allocation expansion.
    """

    if graph.hardware is None:
        raise ValueError("schedule enumeration requires graph.hardware")
    threads = graph.kernel_threads if original_threads is None else original_threads
    if threads is None:
        raise ValueError("original thread count is unknown")
    policy = GuidedSearchConfig() if config is None else config

    groups = _ranked_group_assignments(graph, policy.groups_per_count)
    structures: list[
        tuple[
            float,
            tuple[Any, ...],
            dict[int, dict[int, int]],
            dict[int, int],
            ProgramOrders,
        ]
    ] = []
    seen_structures: set[tuple[Any, ...]] = set()
    stage_frontier = _guided_program_stage_assignments(
        graph, min(policy.structures, 512)
    )
    for stages_by_region in stage_frontier:
        for group_assignment in groups:
            if not warp_requirements_are_feasible(graph, group_assignment):
                continue
            orders = build_program_orders(
                graph, stages_by_region, group_assignment
            )
            key = _stage_group_order_equivalence_key(
                graph, stages_by_region, group_assignment, orders
            )
            if key in seen_structures:
                continue
            seen_structures.add(key)
            structures.append(
                (
                    score_structure(
                        graph, stages_by_region, group_assignment
                    ),
                    key,
                    stages_by_region,
                    group_assignment,
                    orders,
                )
            )
    # Preserve exploration across group/stage depths.  A single high-scoring
    # but infeasible family must not consume the complete structural budget.
    buckets = defaultdict(list)
    for item in structures:
        _, _, stages_by_region, group_assignment, _ = item
        local_stages = _group_local_stages(
            graph, stages_by_region, group_assignment
        )
        stage_depth = max((stage for _, _, stage in local_stages), default=0) + 1
        buckets[(len(set(group_assignment.values())), stage_depth)].append(item)
    if not buckets:
        return
    per_bucket = max(1, math.ceil(policy.structures / len(buckets)))
    selected_structures = []
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (-item[0], item[1]))
        selected_structures.extend(bucket[:per_bucket])
    selected_structures.sort(key=lambda item: (-item[0], item[1]))
    structures = selected_structures[: policy.structures]

    candidates: list[tuple[float, tuple[Any, ...], Schedule]] = []
    seen_schedules: set[tuple[Any, ...]] = set()
    for _, _, stages_by_region, group_assignment, orders in structures:
        for versions in enumerate_buffer_version_plans(
            graph, stages_by_region, group_assignment, orders
        ):
            try:
                synchronizations = build_synchronizations(
                    graph,
                    stages_by_region,
                    group_assignment,
                    orders,
                    versions,
                )
            except SynchronizationCycleError:
                continue
            shared_memory = analyze_shared_memory(
                graph,
                group_assignment,
                orders,
                versions,
                synchronizations,
            )
            if not shared_memory.fits:
                continue
            for warp_allocation in enumerate_warp_allocations(
                graph,
                group_assignment,
                orders,
                versions,
                original_threads=threads,
            ):
                schedule = Schedule(
                    stages_by_region={
                        region_id: dict(stages)
                        for region_id, stages in stages_by_region.items()
                    },
                    groups=dict(group_assignment),
                    orders={
                        region_id: {
                            group_id: dict(order)
                            for group_id, order in group_orders.items()
                        }
                        for region_id, group_orders in orders.items()
                    },
                    buffer_versions=dict(versions),
                    synchronizations=synchronizations,
                    warp_allocation=warp_allocation,
                    shared_memory=shared_memory,
                )
                key = _schedule_equivalence_key(graph, schedule)
                if key in seen_schedules:
                    continue
                seen_schedules.add(key)
                candidates.append(
                    (_full_schedule_score(graph, schedule), key, schedule)
                )

    candidates.sort(key=lambda item: (-item[0], item[1]))
    for _, _, schedule in candidates[: policy.schedules]:
        yield schedule


def schedule_to_dict(
    graph: DataflowGraph,
    schedule: Schedule,
) -> dict[str, Any]:
    """Return stable JSON-compatible metadata for one candidate."""

    return {
        "stages_by_region": {
            str(region_id): {
                str(node_id): stage for node_id, stage in stages.items()
            }
            for region_id, stages in schedule.stages_by_region.items()
        },
        "groups": {
            str(node_id): group_id
            for node_id, group_id in schedule.groups.items()
        },
        "orders": {
            str(region_id): {
                str(group_id): {
                    str(node_id): position
                    for node_id, position in order.items()
                }
                for group_id, order in group_orders.items()
            }
            for region_id, group_orders in schedule.orders.items()
        },
        "buffer_versions": [
            {
                "buffer_id": buffer.buffer_id,
                "name": buffer.name,
                "versions": schedule.buffer_versions[buffer.buffer_id],
            }
            for buffer in graph.buffers
        ],
        "synchronizations": [
            {
                "kind": item.kind.value,
                "scope": item.scope.value,
                "producer_id": item.producer_id,
                "consumer_id": item.consumer_id,
                "producer_group": item.producer_group,
                "consumer_group": item.consumer_group,
                "buffer_id": item.buffer_id,
                "buffer": graph.buffer_for_id(item.buffer_id).name,
                "iteration_distance": item.iteration_distance,
                "effective_stage_distance": item.effective_stage_distance,
                "slot_count": item.slot_count,
            }
            for item in schedule.synchronizations
        ],
        "warp_allocation": {
            "effective_threads": schedule.warp_allocation.effective_threads,
            "groups": [
                asdict(group) for group in schedule.warp_allocation.groups
            ],
            "register_counts": schedule.warp_allocation.register_counts,
            "register_is_increase": (
                schedule.warp_allocation.register_is_increase
            ),
        },
        "shared_memory": {
            "shared_buffer_bytes": schedule.shared_memory.shared_buffer_bytes,
            "synchronization_bytes": (
                schedule.shared_memory.synchronization_bytes
            ),
            "merged_shared_bytes": schedule.shared_memory.merged_shared_bytes,
            "capacity_bytes": (
                schedule.shared_memory.shared_memory_capacity_bytes
            ),
            "allocations": [
                asdict(allocation)
                for allocation in schedule.shared_memory.shared_allocations
            ],
        },
    }


def schedule_from_dict(data: dict[str, Any]) -> Schedule:
    """Restore one schedule serialized by :func:`schedule_to_dict`."""

    required = {
        "stages_by_region",
        "groups",
        "orders",
        "buffer_versions",
        "synchronizations",
        "warp_allocation",
        "shared_memory",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(
            "schedule JSON is missing fields: " + ", ".join(sorted(missing))
        )

    stages_by_region = {
        int(region_id): {
            int(node_id): int(stage) for node_id, stage in stages.items()
        }
        for region_id, stages in data["stages_by_region"].items()
    }
    groups = {
        int(node_id): int(group_id)
        for node_id, group_id in data["groups"].items()
    }
    orders = {
        int(region_id): {
            int(group_id): {
                int(node_id): int(position)
                for node_id, position in order.items()
            }
            for group_id, order in group_orders.items()
        }
        for region_id, group_orders in data["orders"].items()
    }
    buffer_versions = {
        int(item["buffer_id"]): int(item["versions"])
        for item in data["buffer_versions"]
    }
    synchronizations = tuple(
        Synchronization(
            kind=SynchronizationKind(item["kind"]),
            scope=SynchronizationScope(item["scope"]),
            producer_id=int(item["producer_id"]),
            consumer_id=int(item["consumer_id"]),
            producer_group=int(item["producer_group"]),
            consumer_group=int(item["consumer_group"]),
            buffer_id=int(item["buffer_id"]),
            iteration_distance=int(item["iteration_distance"]),
            effective_stage_distance=(
                None
                if item["effective_stage_distance"] is None
                else int(item["effective_stage_distance"])
            ),
            slot_count=int(item["slot_count"]),
        )
        for item in data["synchronizations"]
    )

    warp_data = data["warp_allocation"]
    register_counts = warp_data["register_counts"]
    register_is_increase = warp_data["register_is_increase"]
    warp_allocation = WarpAllocation(
        groups=tuple(
            GroupWarpAllocation(
                group_id=int(item["group_id"]),
                first_warp=int(item["first_warp"]),
                warp_count=int(item["warp_count"]),
            )
            for item in warp_data["groups"]
        ),
        effective_threads=int(warp_data["effective_threads"]),
        register_counts=(
            None
            if register_counts is None
            else tuple(int(value) for value in register_counts)
        ),
        register_is_increase=(
            None
            if register_is_increase is None
            else tuple(bool(value) for value in register_is_increase)
        ),
    )

    shared_data = data["shared_memory"]
    shared_memory = SharedMemoryPlan(
        shared_buffer_bytes=int(shared_data["shared_buffer_bytes"]),
        synchronization_bytes=int(shared_data["synchronization_bytes"]),
        merged_shared_bytes=int(shared_data["merged_shared_bytes"]),
        shared_memory_capacity_bytes=int(shared_data["capacity_bytes"]),
        shared_allocations=tuple(
            SharedBufferAllocation(
                buffer_id=int(item["buffer_id"]),
                name=str(item["name"]),
                start=int(item["start"]),
                end=int(item["end"]),
                size_bytes=int(item["size_bytes"]),
                alignment=int(item["alignment"]),
                byte_offset=int(item["byte_offset"]),
            )
            for item in shared_data["allocations"]
        ),
    )
    return Schedule(
        stages_by_region=stages_by_region,
        groups=groups,
        orders=orders,
        buffer_versions=buffer_versions,
        synchronizations=synchronizations,
        warp_allocation=warp_allocation,
        shared_memory=shared_memory,
    )


def schedule_from_json(
    text: str,
    *,
    candidate_index: int = 0,
) -> Schedule:
    """Read one schedule from candidate JSON text.

    The input may be either a single object returned by ``schedule_to_dict``
    or the wrapper returned by ``schedules_to_dict``/``schedules_to_json``.
    """

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("schedule JSON root must be an object")
    if "candidates" in data:
        candidates = data["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("schedule JSON contains no candidates")
        try:
            data = candidates[candidate_index]
        except IndexError as error:
            raise ValueError(
                f"candidate_index {candidate_index} is out of range"
            ) from error
    if not isinstance(data, dict):
        raise ValueError("selected schedule candidate must be an object")
    return schedule_from_dict(data)


def load_schedule_json(
    path: str | Path,
    *,
    candidate_index: int = 0,
) -> Schedule:
    """Load one serialized schedule candidate from a UTF-8 JSON file."""

    return schedule_from_json(
        Path(path).read_text(encoding="utf-8"),
        candidate_index=candidate_index,
    )


def schedules_to_dict(
    graph: DataflowGraph,
    schedules: Sequence[Schedule],
) -> dict[str, Any]:
    """Serialize graph metadata once followed by all schedule candidates."""

    return {
        "graph": {
            "operations": [
                {
                    "node_id": node.node_id,
                    "name": node.name,
                    "region_id": node.region_id,
                    "instruction": node.instruction.name,
                }
                for node in graph.nodes
            ],
            "buffers": [
                {
                    "buffer_id": buffer.buffer_id,
                    "name": buffer.name,
                    "scope": buffer.scope,
                    "nbytes": buffer.nbytes,
                }
                for buffer in graph.buffers
            ],
        },
        "candidate_count": len(schedules),
        "candidates": [schedule_to_dict(graph, item) for item in schedules],
    }


def schedules_to_json(
    graph: DataflowGraph,
    schedules: Sequence[Schedule],
    *,
    indent: int | None = 2,
) -> str:
    """Return all candidates as deterministic JSON text."""

    return json.dumps(
        schedules_to_dict(graph, schedules),
        indent=indent,
        sort_keys=True,
    )
