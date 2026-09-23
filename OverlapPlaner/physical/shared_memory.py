"""Plan shared-memory reuse with the backend's linear-scan policy."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from OverlapPlaner.arch.api import DeviceResource
from OverlapPlaner.facts import DependencyKind, FactGraph, RegionKind
from OverlapPlaner.physical.model import (
    FragmentHandoffAllocation,
    SharedBufferAllocation,
    SharedMemoryPlan,
)
from OverlapPlaner.structure.model import (
    Structure,
    SynchronizationKind,
    SynchronizationScope,
    SyncSkeleton,
    validate_stage_group_order,
)

_WS_GROUP_NONE = 0
_BASE_ALIGNMENT = 16
_MBARRIER_BYTES = 8
_HANDOFF_NAME_SUFFIX = "_wsp_handoff_"
_HANDOFF_BUFFER_ID_BASE = -1_000_000


def _ws_group_conflict(
    left: int,
    right: int,
    left_pipeline_writer: bool,
    right_pipeline_writer: bool,
    left_accesses: tuple[int, ...],
    right_accesses: tuple[int, ...],
    happens_before: frozenset[tuple[int, int]],
) -> bool:
    """Return whether two disjoint lifetimes still may execute concurrently.

    A different or multi-group mask is not itself a conflict when the
    materialized schedule proves that every access to the retired buffer
    happens before every access to the new buffer.  Pipeline-written buffers
    remain conservative: their dynamic iterations and outstanding async
    transactions need an iteration-aware completion proof.
    """

    if left == _WS_GROUP_NONE or right == _WS_GROUP_NONE:
        return False
    if left_pipeline_writer or right_pipeline_writer:
        return True
    if left == right and left & (left - 1) == 0:
        return False
    if not left_accesses or not right_accesses:
        return True
    return not all(
        (left_node, right_node) in happens_before
        for left_node in left_accesses
        for right_node in right_accesses
    )


def fragment_handoff_buffer_name(source_name: str, channel_id: int) -> str:
    """Return the C++-allocated shared buffer name for one fragment RAW edge."""

    return f"{source_name}{_HANDOFF_NAME_SUFFIX}{channel_id}"


def _handoff_buffer_id(channel_id: int) -> int:
    return _HANDOFF_BUFFER_ID_BASE - channel_id



def _is_shared(scope: str) -> bool:
    return scope.startswith("shared") and "tmem" not in scope


def _buffer_ws_group(
    graph: FactGraph, groups: Mapping[int, int], buffer_id: int
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
class _Interval:
    buffer_id: int
    name: str
    start: int
    end: int
    size_bytes: int
    alignment: int
    ws_group: int = 0
    pipeline_writer: bool = False
    access_node_ids: tuple[int, ...] = ()


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _explicit_alignments(graph: FactGraph) -> dict[str, int]:
    if graph.prim_func is None or graph.prim_func.attrs is None:
        return {}
    alignment_map = graph.prim_func.attrs.get("tl.smem_alignment_map")
    if alignment_map is None:
        return {}
    return {str(name): int(alignment) for name, alignment in alignment_map.items()}


def default_buffer_alignment(graph: FactGraph, buffer_id: int) -> int:
    buffer = graph.buffer_for_id(buffer_id)
    explicit = _explicit_alignments(graph)
    data = getattr(buffer.buffer, "data", None)
    data_name = getattr(data, "name_hint", None)
    required = explicit.get(str(data_name), explicit.get(buffer.name))
    if required is not None:
        return max(_BASE_ALIGNMENT, required)
    return _BASE_ALIGNMENT


def _schedule_touches(
    graph: FactGraph,
    orders: Mapping[int, Mapping[int, Mapping[int, int]]],
) -> tuple[dict[int, list[int]], dict[int, tuple[int, int]]]:
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
                node_spans[node.node_id] = (position, position + 1)
                for buffer_id in (*node.reads, *node.writes):
                    if _is_shared(graph.buffer_for_id(buffer_id).scope):
                        touches.setdefault(buffer_id, []).append(position)
                position += 1
    return touches, node_spans


def _schedule_happens_before(
    graph: FactGraph, structure: Structure
) -> frozenset[tuple[int, int]]:
    """Return operation pairs ordered by the materialized schedule.

    Group-local order extends across regions because each physical warp group
    executes its own prefix, pipeline, and epilogue in program order.  A
    zero-distance or non-pipeline synchronization adds the corresponding
    cross-group completion edge.  Transitive closure then proves phase
    boundaries such as ``Q TMA completion -> Q consumer -> O_shared write``.
    """

    successors: dict[int, set[int]] = {
        node.node_id: set() for node in graph.nodes
    }
    group_ids = tuple(sorted(set(structure.groups.values())))
    for group_id in group_ids:
        sequence: list[int] = []
        for region_id in range(len(graph.regions)):
            local_order = structure.orders[region_id][group_id]
            sequence.extend(sorted(local_order, key=local_order.__getitem__))
        for producer_id, consumer_id in zip(sequence, sequence[1:]):
            successors[producer_id].add(consumer_id)

    for sync in structure.sync_edges:
        if (
            sync.scope != SynchronizationScope.PER_ITERATION
            or sync.effective_stage_distance == 0
        ):
            successors[sync.producer_id].add(sync.consumer_id)

    ordered: set[tuple[int, int]] = set()
    for source in successors:
        stack = list(successors[source])
        visited: set[int] = set()
        while stack:
            destination = stack.pop()
            if destination in visited:
                continue
            visited.add(destination)
            ordered.add((source, destination))
            stack.extend(successors[destination])
    return frozenset(ordered)


def _buffer_intervals(
    graph: FactGraph,
    groups: Mapping[int, int],
    versions: Mapping[int, int],
    touches: Mapping[int, Sequence[int]],
    alignment_for_buffer: Callable[[int], int],
) -> tuple[_Interval, ...]:
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
                alignment_for_buffer(buffer.buffer_id),
                _buffer_ws_group(graph, groups, buffer.buffer_id),
                any(
                    buffer.buffer_id in node.writes
                    and graph.region_for_id(node.region_id).kind
                    == RegionKind.PIPELINE
                    for node in graph.nodes
                ),
                tuple(
                    node.node_id
                    for node in graph.nodes
                    if buffer.buffer_id in (*node.reads, *node.writes)
                ),
            )
        )
    return tuple(result)


def _is_fragment_raw_handoff(graph: FactGraph, item: SyncSkeleton) -> bool:
    """Match LowerOverlapPlan: FORWARD RAW on a local.fragment buffer."""

    if item.kind != SynchronizationKind.FORWARD_DEPENDENCY:
        return False
    buffer = graph.buffer_for_id(item.buffer_id)
    if buffer.scope != "local.fragment":
        return False
    return any(
        edge.producer_id == item.producer_id
        and edge.consumer_id == item.consumer_id
        and edge.buffer_id == item.buffer_id
        and DependencyKind.RAW in edge.dependency_kinds
        for edge in graph.edges
    )


def _fragment_handoff_intervals(
    graph: FactGraph,
    structure: Structure,
    node_spans: Mapping[int, tuple[int, int]],
) -> tuple[_Interval, ...]:
    result = []
    for channel_id, item in enumerate(structure.sync_edges):
        if not _is_fragment_raw_handoff(graph, item):
            continue
        buffer = graph.buffer_for_id(item.buffer_id)
        if buffer.nbytes is None:
            raise ValueError(f"fragment handoff {buffer.name} has dynamic size")
        slots = item.slot_count
        result.append(
            _Interval(
                _handoff_buffer_id(channel_id),
                fragment_handoff_buffer_name(buffer.name, channel_id),
                min(
                    node_spans[item.producer_id][0],
                    node_spans[item.consumer_id][0],
                ),
                max(
                    node_spans[item.producer_id][1],
                    node_spans[item.consumer_id][1],
                ),
                buffer.nbytes * slots,
                _BASE_ALIGNMENT,
                (1 << item.producer_group) | (1 << item.consumer_group),
                True,
                (item.producer_id, item.consumer_id),
            )
        )
    return tuple(result)


def _synchronization_intervals(
    synchronizations: Sequence[SyncSkeleton],
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
            True,
            (item.producer_id, item.consumer_id),
        )
        for index, item in enumerate(synchronizations)
    )


def _insert_free_block(
    blocks: list[tuple[int, int]],
    offset: int,
    size: int,
) -> None:
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
                max(old_offset + old_size, block_offset + block_size) - old_offset,
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
    happens_before: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[int, tuple[SharedBufferAllocation, ...]]:
    ordered = sorted(
        intervals,
        key=lambda item: (item.start, -item.size_bytes, item.name),
    )
    active: list[
        tuple[int, int, int, int, int, bool, tuple[int, ...]]
    ] = []
    free_blocks: list[tuple[int, int]] = []
    arena_top = 0
    allocations = []
    has_ws_groups = any(item.ws_group != _WS_GROUP_NONE for item in ordered)
    for interval in ordered:
        if has_ws_groups:
            free_blocks = []
            remaining = []
            for item in active:
                if (
                    item[0] <= interval.start
                    and not _ws_group_conflict(
                        item[4], interval.ws_group, item[5],
                        interval.pipeline_writer,
                        item[6], interval.access_node_ids,
                        happens_before,
                    )
                ):
                    _insert_free_block(free_blocks, item[1], item[2])
                else:
                    remaining.append(item)
            active = remaining
        else:
            while active and active[0][0] <= interval.start:
                _, offset, size, _, _, _, _ = heapq.heappop(active)
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
            interval.pipeline_writer,
            interval.access_node_ids,
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
    graph: FactGraph,
    structure: Structure,
    resource: DeviceResource,
    *,
    alignment_for_buffer: Callable[[int], int] | None = None,
) -> SharedMemoryPlan:
    """Return backend-equivalent merged shared-memory usage."""

    validate_stage_group_order(
        graph, structure.stages_by_region, structure.groups, structure.orders
    )
    if resource.shared_memory_capacity_bytes < 1:
        raise ValueError("shared-memory capacity must be positive")
    align = alignment_for_buffer or (
        lambda buffer_id: default_buffer_alignment(graph, buffer_id)
    )
    touches, node_spans = _schedule_touches(graph, structure.orders)
    buffer_intervals = _buffer_intervals(
        graph,
        structure.groups,
        structure.buffer_versions,
        touches,
        align,
    )
    handoff_intervals = _fragment_handoff_intervals(
        graph, structure, node_spans
    )
    synchronization_intervals = _synchronization_intervals(
        structure.sync_edges, node_spans
    )
    happens_before = _schedule_happens_before(graph, structure)
    shared_buffer_bytes, allocations = _pack(
        (*buffer_intervals, *handoff_intervals), happens_before
    )
    merged_shared_bytes, _ = _pack(
        (*buffer_intervals, *handoff_intervals, *synchronization_intervals),
        happens_before,
    )
    placed = {item.buffer_id: item for item in allocations}
    handoff_allocations = []
    for channel_id, item in enumerate(structure.sync_edges):
        placed_handoff = placed.get(_handoff_buffer_id(channel_id))
        if placed_handoff is None:
            continue
        handoff_allocations.append(
            FragmentHandoffAllocation(
                channel_id,
                item.buffer_id,
                placed_handoff.name,
                placed_handoff.size_bytes,
                placed_handoff.alignment,
                placed_handoff.byte_offset,
            )
        )
    return SharedMemoryPlan(
        shared_buffer_bytes,
        sum(item.slot_count * _MBARRIER_BYTES for item in structure.sync_edges),
        merged_shared_bytes,
        resource.shared_memory_capacity_bytes,
        tuple(item for item in allocations if item.buffer_id >= 0),
        tuple(handoff_allocations),
    )
