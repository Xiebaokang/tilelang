"""Plan shared-memory reuse and prune schedules that exceed its capacity."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import heapq

from ..parseIR.graph import DataflowGraph, RegionKind
from ..schedule.order import ProgramOrders
from ..schedule.synchronization import SynchronizationChannel
from .warp_allocation import WarpAllocation


@dataclass(frozen=True)
class SharedBufferAllocation:
    """One versioned shared buffer placed in the merged arena."""

    buffer_id: int
    name: str
    start: int
    end: int
    size_bytes: int
    alignment: int
    byte_offset: int


@dataclass(frozen=True)
class SharedMemoryPlan:
    """Versioned buffers and synchronization placed in one shared arena."""

    shared_buffer_bytes: int
    synchronization_bytes: int
    merged_shared_bytes: int
    shared_memory_capacity_bytes: int
    shared_allocations: tuple[SharedBufferAllocation, ...]

    @property
    def total_shared_bytes(self) -> int:
        return self.merged_shared_bytes

    @property
    def fits(self) -> bool:
        return self.total_shared_bytes <= self.shared_memory_capacity_bytes


def _validate_versions(
    graph: DataflowGraph, versions: Mapping[int, int]
) -> None:
    buffer_ids = {buffer.buffer_id for buffer in graph.buffers}
    if set(versions) != buffer_ids:
        raise ValueError("versions must cover every buffer exactly once")
    if any(version < 1 for version in versions.values()):
        raise ValueError("every buffer version count must be positive")


def _validate_orders(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
) -> None:
    group_ids = set(groups.values())
    if set(orders) != set(range(len(graph.region_kinds))):
        raise ValueError("orders must cover every region exactly once")
    for region_id in range(len(graph.region_kinds)):
        if set(orders[region_id]) != group_ids:
            raise ValueError("each region order must cover every group")
        for group_id, local_order in orders[region_id].items():
            expected = {
                node.node_id
                for node in graph.nodes_for_region(region_id)
                if groups[node.node_id] == group_id
            }
            if set(local_order) != expected:
                raise ValueError("group order covers the wrong region nodes")
            if set(local_order.values()) != set(range(len(expected))):
                raise ValueError("local order positions must be dense")


@dataclass(frozen=True)
class _SharedInterval:
    buffer_id: int
    name: str
    start: int
    end: int
    size_bytes: int
    alignment: int


def _align_up(value: int, alignment: int) -> int:
    remainder = value % alignment
    return value if remainder == 0 else value + alignment - remainder


def _schedule_touches(
    graph: DataflowGraph,
    orders: ProgramOrders,
) -> tuple[dict[int, list[int]], dict[int, tuple[int, int]]]:
    """Return shared-buffer touches and node lifetime spans."""

    buffer_touches: dict[int, list[int]] = {}
    node_spans: dict[int, tuple[int, int]] = {}
    position = 0
    for region_id, kind in enumerate(graph.region_kinds):
        nonempty_groups = sum(
            bool(local_order)
            for local_order in orders[region_id].values()
        )
        if kind == RegionKind.PIPELINE or nonempty_groups > 1:
            for node in graph.nodes_for_region(region_id):
                node_spans[node.node_id] = (position, position + 2)
                for buffer_id in (*node.reads, *node.writes):
                    buffer = graph.buffer_for_id(buffer_id)
                    if (
                        buffer.scope.startswith("shared")
                        and "tmem" not in buffer.scope
                    ):
                        buffer_touches.setdefault(buffer_id, []).extend(
                            (position, position + 1)
                        )
            position += 2
            continue

        for group_id in sorted(orders[region_id]):
            local_order = orders[region_id][group_id]
            for node_id in sorted(local_order, key=local_order.__getitem__):
                node = graph.node_for_id(node_id)
                node_spans[node_id] = (position, position + 1)
                for buffer_id in (*node.reads, *node.writes):
                    buffer = graph.buffer_for_id(buffer_id)
                    if (
                        buffer.scope.startswith("shared")
                        and "tmem" not in buffer.scope
                    ):
                        buffer_touches.setdefault(buffer_id, []).append(
                            position
                        )
                position += 1
    return buffer_touches, node_spans


def _shared_intervals(
    graph: DataflowGraph,
    versions: Mapping[int, int],
    alignment: int,
    touches: Mapping[int, Sequence[int]],
) -> tuple[_SharedInterval, ...]:
    """Build first/last-touch intervals like the backend reuse pass."""

    intervals = []
    for buffer in graph.buffers:
        if not buffer.scope.startswith("shared") or "tmem" in buffer.scope:
            continue
        if buffer.nbytes is None:
            raise ValueError(f"shared buffer {buffer.name} has a dynamic size")
        buffer_touches = touches.get(buffer.buffer_id)
        if not buffer_touches:
            continue
        intervals.append(
            _SharedInterval(
                buffer.buffer_id,
                buffer.name,
                min(buffer_touches),
                max(buffer_touches) + 1,
                buffer.nbytes * versions[buffer.buffer_id],
                alignment,
            )
        )
    return tuple(intervals)


def _synchronization_intervals(
    synchronization: Sequence[SynchronizationChannel],
    node_spans: Mapping[int, tuple[int, int]],
    mbarrier_bytes: int,
    alignment: int,
) -> tuple[_SharedInterval, ...]:
    intervals = []
    for index, channel in enumerate(synchronization):
        producer_span = node_spans[channel.producer_id]
        consumer_span = node_spans[channel.consumer_id]
        intervals.append(
            _SharedInterval(
                -index - 1,
                f"__mbarrier_{index}",
                min(producer_span[0], consumer_span[0]),
                max(producer_span[1], consumer_span[1]),
                channel.slot_count * mbarrier_bytes,
                alignment,
            )
        )
    return tuple(intervals)


def _insert_free_block(
    blocks: list[tuple[int, int]], offset: int, size: int
) -> None:
    """Insert and coalesce one free block like the C++ FreeList."""

    if size == 0:
        return
    blocks.append((offset, size))
    blocks.sort()
    merged: list[tuple[int, int]] = []
    for block_offset, block_size in blocks:
        if merged and merged[-1][0] + merged[-1][1] >= block_offset:
            previous_offset, previous_size = merged[-1]
            merged[-1] = (
                previous_offset,
                max(
                    previous_offset + previous_size,
                    block_offset + block_size,
                )
                - previous_offset,
            )
        else:
            merged.append((block_offset, block_size))
    blocks[:] = merged


def _allocate_best_fit(
    blocks: list[tuple[int, int]], size: int, alignment: int
) -> int | None:
    best_index = None
    best_waste = None
    for index, (offset, block_size) in enumerate(blocks):
        aligned = _align_up(offset, alignment)
        head = aligned - offset
        if head <= block_size and block_size - head >= size:
            waste = block_size - size
            if best_waste is None or waste < best_waste:
                best_index = index
                best_waste = waste
    if best_index is None:
        return None
    offset, block_size = blocks.pop(best_index)
    aligned = _align_up(offset, alignment)
    head = aligned - offset
    tail = block_size - head - size
    _insert_free_block(blocks, aligned + size, tail)
    _insert_free_block(blocks, offset, head)
    return aligned


def _allocate_from_tail(
    blocks: list[tuple[int, int]],
    size: int,
    alignment: int,
    arena_top: int,
) -> int | None:
    if not blocks:
        return None
    offset, block_size = blocks[-1]
    if offset + block_size != arena_top:
        return None
    aligned = _align_up(offset, alignment)
    if aligned >= arena_top:
        return None
    blocks.pop()
    _insert_free_block(blocks, offset, aligned - offset)
    return aligned


def _plan_shared_arena(
    intervals: Sequence[_SharedInterval], alignment: int
) -> tuple[int, tuple[SharedBufferAllocation, ...]]:
    """Port the backend pass's LinearScanPack allocation policy."""

    ordered = sorted(
        intervals,
        key=lambda interval: (
            interval.start,
            -interval.size_bytes,
            interval.name,
            interval.buffer_id,
        ),
    )
    active: list[tuple[int, int, int, int]] = []
    free_blocks: list[tuple[int, int]] = []
    arena_top = 0
    allocations = []
    for interval in ordered:
        while active and active[0][0] <= interval.start:
            _, offset, size, _ = heapq.heappop(active)
            _insert_free_block(free_blocks, offset, size)

        offset = _allocate_best_fit(
            free_blocks, interval.size_bytes, interval.alignment
        )
        if offset is None:
            offset = _allocate_from_tail(
                free_blocks,
                interval.size_bytes,
                interval.alignment,
                arena_top,
            )
            if offset is not None:
                arena_top = offset + interval.size_bytes
        if offset is None:
            offset = _align_up(arena_top, interval.alignment)
            _insert_free_block(free_blocks, arena_top, offset - arena_top)
            arena_top = offset + interval.size_bytes

        heapq.heappush(
            active,
            (interval.end, offset, interval.size_bytes, interval.buffer_id),
        )
        allocations.append(
            SharedBufferAllocation(
                interval.buffer_id,
                interval.name,
                interval.start,
                interval.end,
                interval.size_bytes,
                interval.alignment,
                offset,
            )
        )
    return _align_up(arena_top, alignment), tuple(allocations)


def analyze_shared_memory(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    synchronization: Sequence[SynchronizationChannel],
) -> SharedMemoryPlan:
    """Return shared-memory usage after the full schedule is fixed."""

    if graph.hardware is None:
        raise ValueError("shared-memory analysis requires graph.hardware")
    hardware = graph.hardware
    if hardware.shared_memory_capacity_bytes < 1:
        raise ValueError("shared-memory capacity must be configured")
    _validate_versions(graph, versions)
    _validate_orders(graph, groups, orders)
    shared_alignment = 16
    touches, node_spans = _schedule_touches(graph, orders)
    shared_intervals = _shared_intervals(
        graph, versions, shared_alignment, touches
    )
    synchronization_intervals = _synchronization_intervals(
        synchronization,
        node_spans,
        hardware.mbarrier_bytes,
        shared_alignment,
    )
    shared_buffer_bytes, _ = _plan_shared_arena(
        shared_intervals,
        shared_alignment,
    )
    merged_shared_bytes, merged_allocations = _plan_shared_arena(
        (*shared_intervals, *synchronization_intervals),
        shared_alignment,
    )
    shared_allocations = tuple(
        item for item in merged_allocations if item.buffer_id >= 0
    )
    return SharedMemoryPlan(
        shared_buffer_bytes=shared_buffer_bytes,
        synchronization_bytes=sum(
            channel.slot_count * hardware.mbarrier_bytes
            for channel in synchronization
        ),
        merged_shared_bytes=merged_shared_bytes,
        shared_memory_capacity_bytes=hardware.shared_memory_capacity_bytes,
        shared_allocations=shared_allocations,
    )


def enumerate_feasible_shared_memory_plans(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    synchronization: Sequence[SynchronizationChannel],
    allocations: Iterable[WarpAllocation],
) -> Iterator[tuple[WarpAllocation, SharedMemoryPlan]]:
    """Yield allocations whose merged shared-memory arena fits the target."""

    plan = analyze_shared_memory(
        graph,
        groups,
        orders,
        versions,
        synchronization,
    )
    if plan.fits:
        for allocation in allocations:
            yield allocation, plan
