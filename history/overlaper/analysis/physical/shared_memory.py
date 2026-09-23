"""Plan shared-memory reuse with the backend's linear-scan policy."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...parse.graph import DataflowGraph, RegionKind
from ..schedule.multi_version import _validate_groups
from ..schedule.order import ProgramOrders
from ..schedule.synchronization import Synchronization


_WS_GROUP_NONE = 0
_BASE_ALIGNMENT = 16
_FALLBACK_TMA_ALIGNMENT = 1024
_MBARRIER_BYTES = 8


def _is_exclusive_ws_group(mask: int) -> bool:
    return mask != _WS_GROUP_NONE and (mask & (mask - 1)) == 0


def _ws_group_conflict(lhs: int, rhs: int) -> bool:
    if lhs == _WS_GROUP_NONE or rhs == _WS_GROUP_NONE:
        return False
    return not (_is_exclusive_ws_group(lhs) and lhs == rhs)


def _buffer_ws_group(
    graph: DataflowGraph, groups: Mapping[int, int], buffer_id: int
) -> int:
    mask = _WS_GROUP_NONE
    for group in {
        groups[node.node_id]
        for node in graph.nodes
        if buffer_id in (*node.reads, *node.writes)
    }:
        mask |= 1 << group
    return mask


@dataclass(frozen=True, slots=True)
class SharedBufferAllocation:
    """One versioned shared buffer placed in the merged arena."""

    buffer_id: int
    name: str
    start: int
    end: int
    size_bytes: int
    alignment: int
    byte_offset: int


@dataclass(frozen=True, slots=True)
class SharedMemoryPlan:
    """Shared buffers and synchronization barriers in one arena."""

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


@dataclass(frozen=True, slots=True)
class _Interval:
    buffer_id: int
    name: str
    start: int
    end: int
    size_bytes: int
    alignment: int
    ws_group: int = 0


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _is_shared(scope: str) -> bool:
    return scope.startswith("shared") and "tmem" not in scope


def _validate_inputs(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
) -> None:
    group_ids = _validate_groups(graph, groups)
    if set(versions) != {buffer.buffer_id for buffer in graph.buffers} or any(
        version < 1 for version in versions.values()
    ):
        raise ValueError("versions must cover every buffer with positive counts")
    if set(orders) != set(range(len(graph.region_kinds))):
        raise ValueError("orders must cover every region exactly once")
    for region_id, group_orders in orders.items():
        if set(group_orders) != group_ids:
            raise ValueError("each region order must cover every group")
        region_nodes = {
            node.node_id for node in graph.nodes_for_region(region_id)
        }
        for group_id, order in group_orders.items():
            expected = {
                node_id
                for node_id in region_nodes
                if groups[node_id] == group_id
            }
            if set(order) != expected or set(order.values()) != set(
                range(len(expected))
            ):
                raise ValueError("group order must be a dense region permutation")


def _explicit_alignments(graph: DataflowGraph) -> dict[str, int]:
    if graph.prim_func is None or graph.prim_func.attrs is None:
        return {}
    alignment_map = graph.prim_func.attrs.get("tl.smem_alignment_map")
    if alignment_map is None:
        return {}
    return {str(name): int(alignment) for name, alignment in alignment_map.items()}


def _buffer_alignment(
    graph: DataflowGraph,
    buffer_id: int,
    explicit: Mapping[str, int],
) -> int:
    buffer = graph.buffer_for_id(buffer_id)
    data = getattr(buffer.buffer, "data", None)
    data_name = getattr(data, "name_hint", None)
    required = explicit.get(str(data_name), explicit.get(buffer.name))
    if required is not None:
        return max(_BASE_ALIGNMENT, required)
    if any(
        node.instruction.name in ("tma", "wgmma")
        and buffer_id in (*node.reads, *node.writes)
        for node in graph.nodes
    ):
        return _FALLBACK_TMA_ALIGNMENT
    return _BASE_ALIGNMENT


def _schedule_touches(
    graph: DataflowGraph,
    _groups: Mapping[int, int],
    orders: ProgramOrders,
) -> tuple[dict[int, list[int]], dict[int, tuple[int, int]]]:
    """Linearize region accesses for the arena packer.

    Cross-group aliasing is applied later by ``_buffer_ws_group`` / ``_pack``.
    """

    touches: dict[int, list[int]] = {}
    node_spans: dict[int, tuple[int, int]] = {}
    position = 0
    for region_id, kind in enumerate(graph.region_kinds):
        group_orders = orders[region_id]
        concurrent = kind == RegionKind.PIPELINE or sum(
            bool(order) for order in group_orders.values()
        ) > 1
        if concurrent:
            for node in graph.nodes_for_region(region_id):
                node_spans[node.node_id] = (position, position + 2)
                for buffer_id in (*node.reads, *node.writes):
                    if _is_shared(graph.buffer_for_id(buffer_id).scope):
                        touches.setdefault(buffer_id, []).extend(
                            (position, position + 1)
                        )
            position += 2
            continue

        for order in group_orders.values():
            for node_id in sorted(order, key=order.__getitem__):
                node = graph.node_for_id(node_id)
                node_spans[node_id] = (position, position + 1)
                for buffer_id in (*node.reads, *node.writes):
                    if _is_shared(graph.buffer_for_id(buffer_id).scope):
                        touches.setdefault(buffer_id, []).append(position)
                position += 1
    return touches, node_spans


def _buffer_intervals(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    versions: Mapping[int, int],
    touches: Mapping[int, Sequence[int]],
) -> tuple[_Interval, ...]:
    explicit = _explicit_alignments(graph)
    result = []
    for buffer in graph.buffers:
        if not _is_shared(buffer.scope):
            continue
        if buffer.nbytes is None:
            raise ValueError(f"shared buffer {buffer.name} has dynamic size")
        buffer_touches = touches.get(buffer.buffer_id)
        if not buffer_touches:
            continue
        result.append(
            _Interval(
                buffer.buffer_id,
                buffer.name,
                min(buffer_touches),
                max(buffer_touches) + 1,
                buffer.nbytes * versions[buffer.buffer_id],
                _buffer_alignment(graph, buffer.buffer_id, explicit),
                _buffer_ws_group(graph, groups, buffer.buffer_id),
            )
        )
    return tuple(result)


def _synchronization_intervals(
    synchronizations: Sequence[Synchronization],
    node_spans: Mapping[int, tuple[int, int]],
) -> tuple[_Interval, ...]:
    return tuple(
        _Interval(
            -index - 1,
            f"__mbarrier_{index}",
            min(
                node_spans[item.producer_id][0],
                node_spans[item.consumer_id][0],
            ),
            max(
                node_spans[item.producer_id][1],
                node_spans[item.consumer_id][1],
            ),
            item.slot_count * _MBARRIER_BYTES,
            _BASE_ALIGNMENT,
            (1 << item.producer_group) | (1 << item.consumer_group),
        )
        for index, item in enumerate(synchronizations)
    )


def _insert_free_block(
    blocks: list[tuple[int, int]],
    offset: int,
    size: int,
) -> None:
    """Insert and coalesce exactly like the backend FreeList."""

    if size == 0:
        return
    blocks.append((offset, size))
    blocks.sort()
    merged = []
    for block_offset, block_size in blocks:
        if merged and merged[-1][0] + merged[-1][1] >= block_offset:
            old_offset, old_size = merged[-1]
            merged[-1] = (
                old_offset,
                max(
                    old_offset + old_size,
                    block_offset + block_size,
                )
                - old_offset,
            )
        else:
            merged.append((block_offset, block_size))
    blocks[:] = merged


def _allocate_best_fit(
    blocks: list[tuple[int, int]],
    size: int,
    alignment: int,
) -> int | None:
    best = None
    best_waste = None
    for index, (offset, block_size) in enumerate(blocks):
        aligned = _align_up(offset, alignment)
        head = aligned - offset
        if head > block_size or block_size - head < size:
            continue
        waste = block_size - size
        if best_waste is None or waste < best_waste:
            best = index
            best_waste = waste
    if best is None:
        return None
    offset, block_size = blocks.pop(best)
    aligned = _align_up(offset, alignment)
    head = aligned - offset
    _insert_free_block(blocks, offset, head)
    _insert_free_block(blocks, aligned + size, block_size - head - size)
    return aligned


def _allocate_from_tail(
    blocks: list[tuple[int, int]],
    alignment: int,
    arena_top: int,
) -> int | None:
    if not blocks:
        return None
    offset, size = blocks[-1]
    if offset + size != arena_top:
        return None
    aligned = _align_up(offset, alignment)
    if aligned >= arena_top:
        return None
    blocks.pop()
    _insert_free_block(blocks, offset, aligned - offset)
    return aligned


def _pack(
    intervals: Sequence[_Interval],
) -> tuple[int, tuple[SharedBufferAllocation, ...]]:
    """Port TileLang's ``LinearScanPack`` allocation order and FreeList."""

    ordered = sorted(
        intervals,
        key=lambda item: (item.start, -item.size_bytes, item.name),
    )
    active: list[tuple[int, int, int, int, int]] = []
    free_blocks: list[tuple[int, int]] = []
    arena_top = 0
    allocations = []
    has_ws_groups = any(item.ws_group != _WS_GROUP_NONE for item in ordered)
    for interval in ordered:
        if has_ws_groups:
            remaining = []
            for item in active:
                end, offset, size, _buffer_id, ws_group = item
                if end <= interval.start and not _ws_group_conflict(
                    ws_group, interval.ws_group
                ):
                    _insert_free_block(free_blocks, offset, size)
                else:
                    remaining.append(item)
            active = remaining
        else:
            while active and active[0][0] <= interval.start:
                _, offset, size, _, _ = heapq.heappop(active)
                _insert_free_block(free_blocks, offset, size)

        offset = _allocate_best_fit(
            free_blocks, interval.size_bytes, interval.alignment
        )
        if offset is None:
            offset = _allocate_from_tail(
                free_blocks, interval.alignment, arena_top
            )
            if offset is not None:
                arena_top = offset + interval.size_bytes
        if offset is None:
            offset = _align_up(arena_top, interval.alignment)
            _insert_free_block(free_blocks, arena_top, offset - arena_top)
            arena_top = offset + interval.size_bytes

        placed = (
            interval.end,
            offset,
            interval.size_bytes,
            interval.buffer_id,
            interval.ws_group,
        )
        if has_ws_groups:
            active.append(placed)
        else:
            heapq.heappush(active, placed)
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
    return _align_up(arena_top, _BASE_ALIGNMENT), tuple(allocations)


def analyze_shared_memory(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    synchronizations: Sequence[Synchronization],
) -> SharedMemoryPlan:
    """Return backend-equivalent merged shared-memory usage."""

    _validate_inputs(graph, groups, orders, versions)
    if graph.hardware is None:
        raise ValueError("shared-memory analysis requires graph.hardware")
    capacity = graph.hardware.device_resource.shared_memory_capacity_bytes
    if capacity < 1:
        raise ValueError("shared-memory capacity must be positive")

    touches, node_spans = _schedule_touches(graph, groups, orders)
    buffer_intervals = _buffer_intervals(graph, groups, versions, touches)
    synchronization_intervals = _synchronization_intervals(
        synchronizations, node_spans
    )
    shared_buffer_bytes, _ = _pack(buffer_intervals)
    merged_shared_bytes, allocations = _pack(
        (*buffer_intervals, *synchronization_intervals)
    )
    return SharedMemoryPlan(
        shared_buffer_bytes,
        sum(item.slot_count * _MBARRIER_BYTES for item in synchronizations),
        merged_shared_bytes,
        capacity,
        tuple(item for item in allocations if item.buffer_id >= 0),
    )
