"""Determine buffer versions after stage, group, and order are fixed."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import product

from ...parse.graph import (
    BufferAccessKind,
    BufferRangeAccess,
    DataflowGraph,
    RegionKind,
)
from .order import ProgramOrders


def _ranges_may_overlap(
    accessor: BufferRangeAccess | None,
    writer: BufferRangeAccess | None,
) -> bool:
    """Conservatively test whether two physical buffer regions may overlap."""

    if accessor is None or writer is None:
        return True
    if not accessor.is_exact or not writer.is_exact:
        return True
    if len(accessor.ranges) != len(writer.ranges):
        return True

    from tvm import arith

    analyzer = arith.Analyzer()
    for left, right in zip(accessor.ranges, writer.ranges):
        left_end = left.min + left.extent
        right_end = right.min + right.extent
        if analyzer.can_prove(left_end <= right.min) or analyzer.can_prove(
            right_end <= left.min
        ):
            return False
    return True


def _region_buffer_accesses(
    graph: DataflowGraph,
    region_id: int,
    buffer_id: int,
) -> tuple[
    tuple[tuple[int, BufferRangeAccess | None], ...],
    tuple[tuple[int, BufferRangeAccess | None], ...],
]:
    """Return all accesses and writes, falling back to node metadata."""

    node_ids = {
        node.node_id for node in graph.nodes_for_region(region_id)
    }
    exact_accesses = tuple(
        access
        for access in graph.buffer_accesses
        if access.node_id in node_ids and access.buffer_id == buffer_id
    )
    if exact_accesses:
        accesses = tuple((access.node_id, access) for access in exact_accesses)
        writers = tuple(
            (access.node_id, access)
            for access in exact_accesses
            if access.kind == BufferAccessKind.WRITE
        )
        return accesses, writers

    nodes = graph.nodes_for_region(region_id)
    accesses = tuple(
        (node.node_id, None)
        for node in nodes
        if buffer_id in node.reads or buffer_id in node.writes
    )
    writers = tuple(
        (node.node_id, None)
        for node in nodes
        if buffer_id in node.writes
    )
    return accesses, writers


def _validate_groups(
    graph: DataflowGraph,
    groups: Mapping[int, int],
) -> set[int]:
    node_ids = {node.node_id for node in graph.nodes}
    if set(groups) != node_ids:
        raise ValueError("groups must cover every node exactly once")
    group_ids = set(groups.values())
    if group_ids != set(range(len(group_ids))):
        raise ValueError("group IDs must be dense and non-empty")
    return group_ids


def _validate_inputs(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
) -> None:
    group_ids = _validate_groups(graph, groups)
    pipeline_regions = {
        region_id
        for region_id, kind in enumerate(graph.region_kinds)
        if kind == RegionKind.PIPELINE
    }
    if set(stages_by_region) != pipeline_regions:
        raise ValueError("stages must cover every pipeline region exactly once")
    if set(orders) != set(range(len(graph.region_kinds))):
        raise ValueError("orders must cover every region exactly once")

    for region_id, kind in enumerate(graph.region_kinds):
        region_node_ids = {
            node.node_id for node in graph.nodes_for_region(region_id)
        }
        if kind == RegionKind.PIPELINE:
            stages = stages_by_region[region_id]
            if set(stages) != region_node_ids:
                raise ValueError("stages must cover every pipeline node")
            if any(stage < 0 for stage in stages.values()):
                raise ValueError("stage IDs must be non-negative")
        if set(orders[region_id]) != group_ids:
            raise ValueError("each region order must cover every group")
        for group_id, local_order in orders[region_id].items():
            expected = {
                node_id
                for node_id in region_node_ids
                if groups[node_id] == group_id
            }
            if set(local_order) != expected:
                raise ValueError("group order covers the wrong region nodes")
            if set(local_order.values()) != set(range(len(expected))):
                raise ValueError("group order positions must be dense")


def _minimum_region_versions(
    graph: DataflowGraph,
    region_id: int,
    buffer_id: int,
    stages: Mapping[int, int],
    groups: Mapping[int, int],
    group_orders: Mapping[int, Mapping[int, int]],
    maximum: int,
) -> int:
    accesses, writers = _region_buffer_accesses(graph, region_id, buffer_id)
    if not accesses or not writers:
        return 1

    for versions in range(1, maximum + 1):
        safe = True
        for writer_id, writer_access in writers:
            for accessor_id, accessor_access in accesses:
                if not _ranges_may_overlap(accessor_access, writer_access):
                    continue
                reuse_distance = (
                    versions + stages[writer_id] - stages[accessor_id]
                )
                if reuse_distance < 0:
                    safe = False
                    break
                if (
                    reuse_distance == 0
                    and groups[writer_id] == groups[accessor_id]
                    and group_orders[groups[accessor_id]][accessor_id]
                    >= group_orders[groups[writer_id]][writer_id]
                ):
                    safe = False
                    break
            if not safe:
                break
        if safe:
            return versions

    buffer = graph.buffer_for_id(buffer_id)
    raise ValueError(f"buffer {buffer.name} needs more than {maximum} versions")


def analyze_buffer_versions(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
) -> dict[int, int]:
    """Return the minimum correctness and cross-group overlap versions.

    In addition to the pipeline reuse minimum, every buffer eligible for
    cross-group multiversioning has a lower bound of two versions. A single
    physical slot would serialize its producer and consumer groups and cannot
    realize overlap.
    """

    _validate_inputs(graph, stages_by_region, groups, orders)
    versions = {buffer.buffer_id: 1 for buffer in graph.buffers}
    for region_id, kind in enumerate(graph.region_kinds):
        if kind != RegionKind.PIPELINE:
            continue
        stages = stages_by_region[region_id]
        maximum = max(stages.values(), default=0) + 1
        for buffer in graph.buffers:
            if buffer.scope in ("", "global"):
                continue
            required = _minimum_region_versions(
                graph,
                region_id,
                buffer.buffer_id,
                stages,
                groups,
                orders[region_id],
                maximum,
            )
            versions[buffer.buffer_id] = max(
                versions[buffer.buffer_id], required
            )
    for buffer_id in multiversion_candidate_buffers(graph, groups):
        versions[buffer_id] = max(versions[buffer_id], 2)
    return versions


def multiversion_candidate_buffers(
    graph: DataflowGraph,
    groups: Mapping[int, int],
) -> tuple[int, ...]:
    """Return buffers eligible for independent ``minimum + 1`` choices.

    Eligibility comes only from a dependency edge that crosses groups inside
    one pipeline region. A stage boundary by itself does not make a buffer a
    candidate. Serial and region-boundary communication is synchronized, but
    it cannot use pipeline multiversioning. Global buffers, register fragments,
    and loop-carried state are excluded as in UnionWSP.
    """

    _validate_groups(graph, groups)
    loop_carried_buffers = {
        edge.buffer_id for edge in graph.edges if edge.is_loop_carried
    }
    candidates = {
        edge.buffer_id
        for edge in graph.edges
        if groups[edge.producer_id] != groups[edge.consumer_id]
        and graph.node_for_id(edge.producer_id).region_id
        == graph.node_for_id(edge.consumer_id).region_id
        and graph.region_kinds[
            graph.node_for_id(edge.producer_id).region_id
        ]
        == RegionKind.PIPELINE
        and edge.buffer_id not in loop_carried_buffers
        and graph.buffer_for_id(edge.buffer_id).scope not in ("", "global")
        and graph.buffer_for_id(edge.buffer_id).scope != "local.fragment"
    }
    return tuple(sorted(candidates))


def enumerate_buffer_version_plans(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
) -> Iterator[dict[int, int]]:
    """Enumerate ``minimum``/``minimum + 1`` for cross-group buffers.

    The minimum of every candidate is already at least two, as established by
    :func:`analyze_buffer_versions`.
    """

    minimum = analyze_buffer_versions(
        graph, stages_by_region, groups, orders
    )
    if any(
        buffer.scope == "local.fragment"
        and minimum[buffer.buffer_id] > 1
        for buffer in graph.buffers
    ):
        return

    candidates = multiversion_candidate_buffers(graph, groups)
    for expand in product((False, True), repeat=len(candidates)):
        plan = dict(minimum)
        for buffer_id, should_expand in zip(candidates, expand):
            if should_expand:
                plan[buffer_id] += 1
        yield plan


def multiversion_buffers(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
) -> dict[int, int]:
    """Return only buffers whose correctness minimum exceeds one."""

    return {
        buffer_id: count
        for buffer_id, count in analyze_buffer_versions(
            graph, stages_by_region, groups, orders
        ).items()
        if count > 1
    }
