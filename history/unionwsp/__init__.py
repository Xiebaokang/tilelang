"""Complete UnionWSP schedule enumeration entry point."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .parseIR.graph import DataflowGraph, RegionKind
from .physical import (
    SharedMemoryPlan,
    WarpAllocation,
    enumerate_feasible_shared_memory_plans,
    enumerate_warp_allocations,
)
from .schedule import (
    ProgramOrders,
    SynchronizationChannel,
    SynchronizationCycleError,
    build_program_orders,
    build_synchronization_channels,
    enumerate_buffer_version_plans,
    enumerate_group_assignments,
    enumerate_stage_assignments,
    infer_max_groups,
    infer_max_stages,
)


@dataclass(frozen=True)
class WSPSchedule:
    """One fully specified and resource-feasible WSP schedule."""

    stages_by_region: dict[int, dict[int, int]]
    groups: dict[int, int]
    orders: ProgramOrders
    buffer_versions: dict[int, int]
    synchronization: tuple[SynchronizationChannel, ...]
    warp_allocation: WarpAllocation
    shared_memory: SharedMemoryPlan

    @property
    def num_stages_by_region(self) -> dict[int, int]:
        return {
            region_id: max(stages.values(), default=0) + 1
            for region_id, stages in self.stages_by_region.items()
        }

    @property
    def num_groups(self) -> int:
        return len(set(self.groups.values()))


def _round_robin(iterators):
    active = [iter(iterator) for iterator in iterators]
    while active:
        remaining = []
        for iterator in active:
            try:
                item = next(iterator)
            except StopIteration:
                continue
            remaining.append(iterator)
            yield item
        active = remaining


def _region_stage_assignments(
    graph: DataflowGraph,
    region_id: int,
    num_stages: int | None,
) -> Iterator[dict[int, int]]:
    stage_counts = (
        range(1, infer_max_stages(graph, region_id) + 1)
        if num_stages is None
        else (num_stages,)
    )
    yield from _round_robin(
        enumerate_stage_assignments(graph, region_id, stage_count)
        for stage_count in stage_counts
    )


def _program_stage_assignments(
    graph: DataflowGraph,
    num_stages: int | None,
) -> Iterator[dict[int, dict[int, int]]]:
    pipeline_regions = tuple(
        region_id
        for region_id, kind in enumerate(graph.region_kinds)
        if kind == RegionKind.PIPELINE
    )
    selected: dict[int, dict[int, int]] = {}

    def visit(index: int) -> Iterator[dict[int, dict[int, int]]]:
        if index == len(pipeline_regions):
            yield {
                region_id: dict(stages)
                for region_id, stages in selected.items()
            }
            return
        region_id = pipeline_regions[index]
        for stages in _region_stage_assignments(
            graph, region_id, num_stages
        ):
            selected[region_id] = stages
            yield from visit(index + 1)
        selected.pop(region_id, None)

    yield from visit(0)


def _enumerate_group_count(
    graph: DataflowGraph,
    original_threads: int,
    group_count: int,
    num_stages: int | None,
) -> Iterator[WSPSchedule]:
    for stages_by_region in _program_stage_assignments(graph, num_stages):
        for groups in enumerate_group_assignments(
            graph, stages_by_region, group_count
        ):
            orders = build_program_orders(graph, stages_by_region, groups)
            for versions in enumerate_buffer_version_plans(
                graph,
                stages_by_region,
                groups,
                orders,
            ):
                try:
                    synchronization = build_synchronization_channels(
                        graph,
                        stages_by_region,
                        groups,
                        orders,
                        versions,
                    )
                except SynchronizationCycleError:
                    continue
                allocations = enumerate_warp_allocations(
                    graph,
                    groups,
                    orders,
                    versions,
                    original_threads,
                )
                feasible = enumerate_feasible_shared_memory_plans(
                    graph,
                    groups,
                    orders,
                    versions,
                    synchronization,
                    allocations,
                )
                for allocation, shared_memory in feasible:
                    yield WSPSchedule(
                        stages_by_region={
                            region_id: dict(stages)
                            for region_id, stages in stages_by_region.items()
                        },
                        groups=dict(groups),
                        orders={
                            region_id: {
                                group_id: dict(local_order)
                                for group_id, local_order in group_orders.items()
                            }
                            for region_id, group_orders in orders.items()
                        },
                        buffer_versions=dict(versions),
                        synchronization=synchronization,
                        warp_allocation=allocation,
                        shared_memory=shared_memory,
                    )


def enumerate_wsp_schedules(
    graph: DataflowGraph,
    original_threads: int,
    num_stages: int | None = None,
    num_groups: int | None = None,
) -> Iterator[WSPSchedule]:
    """Enumerate every complete WSP schedule that fits the target hardware.

    ``None`` infers and enumerates the useful range for that dimension; a
    positive integer fixes it.
    """

    if graph.hardware is None:
        raise ValueError("WSP enumeration requires graph.hardware")
    if original_threads < 1:
        raise ValueError("original_threads must be positive")
    if num_stages is not None and num_stages < 1:
        raise ValueError("num_stages must be positive or None")
    if num_groups is not None and num_groups < 1:
        raise ValueError("num_groups must be positive or None")
    group_counts = (
        tuple(range(infer_max_groups(graph), 0, -1))
        if num_groups is None
        else (num_groups,)
    )
    yield from _round_robin(
        _enumerate_group_count(
            graph,
            original_threads,
            group_count,
            num_stages,
        )
        for group_count in group_counts
    )


from .integration import (
    apply_schedule_to_ir,
    build_ir_plan,
    use_schedule_planner,
)
from .search import (
    WSPBenchmarkResult,
    WSPSearchSummary,
    schedule_from_dict,
    schedule_to_dict,
    search_wsp_schedules,
)

__all__ = [
    "WSPSchedule",
    "WSPBenchmarkResult",
    "WSPSearchSummary",
    "apply_schedule_to_ir",
    "build_ir_plan",
    "enumerate_wsp_schedules",
    "schedule_from_dict",
    "schedule_to_dict",
    "search_wsp_schedules",
    "use_schedule_planner",
]
