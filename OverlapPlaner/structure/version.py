"""Minimum buffer versions after stage, group, and order are fixed."""

from __future__ import annotations

from collections.abc import Mapping

from OverlapPlaner.facts import (
    BufferAccessKind,
    BufferRangeAccess,
    FactGraph,
    RegionKind,
)
from OverlapPlaner.structure.model import ProgramOrders, validate_stage_group_order
from OverlapPlaner.structure.stage import effective_stage_distance


def ranges_may_overlap(
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


def region_buffer_accesses(
    graph: FactGraph,
    region_id: int,
    buffer_id: int,
) -> tuple[
    tuple[tuple[int, BufferRangeAccess | None], ...],
    tuple[tuple[int, BufferRangeAccess | None], ...],
]:
    node_ids = {node.node_id for node in graph.nodes_for_region(region_id)}
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
        (node.node_id, None) for node in nodes if buffer_id in node.writes
    )
    return accesses, writers


def _minimum_region_versions(
    graph: FactGraph,
    region_id: int,
    buffer_id: int,
    stages: Mapping[int, int],
    groups: Mapping[int, int],
    group_orders: Mapping[int, Mapping[int, int]],
    maximum: int,
) -> int:
    accesses, writers = region_buffer_accesses(graph, region_id, buffer_id)
    if not accesses or not writers:
        return 1

    for versions in range(1, maximum + 1):
        safe = True
        for writer_id, writer_access in writers:
            for accessor_id, accessor_access in accesses:
                if not ranges_may_overlap(accessor_access, writer_access):
                    continue
                reuse_distance = versions + stages[writer_id] - stages[accessor_id]
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
    graph: FactGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
) -> dict[int, int]:
    """Return the minimum correctness and cross-group overlap versions.

    Eligible cross-group pipeline buffers have a lower bound of two versions.
    Optional slots above the minimum are enumerated by the structure search.

    ``local.fragment`` participates in the pipeline reuse minimum. A same-group
    stage cut can require two or more register tiles. Lowering realizes that as
    distinct fragment allocations, not a leading ``fragment[k % V]`` index.
    Other ``local`` buffers stay unversioned. Cross-group fragment traffic uses
    PRIVATE copies plus a shared handoff ring whose depth includes both stage
    separation and loop-carried distance.
    """

    validate_stage_group_order(graph, stages_by_region, groups, orders)
    versions = {buffer.buffer_id: 1 for buffer in graph.buffers}
    for region in graph.regions:
        if region.kind != RegionKind.PIPELINE:
            continue
        stages = stages_by_region[region.region_id]
        maximum = max(stages.values(), default=0) + 1
        for buffer in graph.buffers:
            if buffer.scope in ("", "global"):
                continue
            if (
                buffer.scope.startswith("local")
                and buffer.scope != "local.fragment"
            ):
                continue
            required = _minimum_region_versions(
                graph,
                region.region_id,
                buffer.buffer_id,
                stages,
                groups,
                orders[region.region_id],
                maximum,
            )
            versions[buffer.buffer_id] = max(versions[buffer.buffer_id], required)
    # A cross-group fragment keeps one PRIVATE allocation per warp group.
    # The shared handoff ring protects communication between groups, but it
    # cannot prevent two pipeline stages in the consumer group from reusing
    # the same private registers. Mirror the handoff's stage-delay depth in
    # the fragment allocation so producer k+1 and consumer k stay distinct.
    for edge in graph.edges:
        buffer = graph.buffer_for_id(edge.buffer_id)
        if buffer.scope != "local.fragment":
            continue
        if groups[edge.producer_id] == groups[edge.consumer_id]:
            continue
        producer = graph.node_for_id(edge.producer_id)
        consumer = graph.node_for_id(edge.consumer_id)
        if producer.region_id != consumer.region_id:
            continue
        if graph.region_kinds[producer.region_id] != RegionKind.PIPELINE:
            continue
        stages = stages_by_region[producer.region_id]
        stage_delay = effective_stage_distance(edge, stages)
        if stage_delay > 0:
            versions[edge.buffer_id] = max(
                versions[edge.buffer_id], stage_delay + 1
            )
    for buffer_id in cross_group_version_buffers(graph, groups):
        versions[buffer_id] = max(versions[buffer_id], 2)
    return versions


def cross_group_version_buffers(
    graph: FactGraph,
    groups: Mapping[int, int],
) -> tuple[int, ...]:
    """Buffers whose producer and consumer sit in different groups."""

    loop_carried = {edge.buffer_id for edge in graph.edges if edge.is_loop_carried}
    candidates = {
        edge.buffer_id
        for edge in graph.edges
        if groups[edge.producer_id] != groups[edge.consumer_id]
        and graph.node_for_id(edge.producer_id).region_id
        == graph.node_for_id(edge.consumer_id).region_id
        and graph.region_kinds[graph.node_for_id(edge.producer_id).region_id]
        == RegionKind.PIPELINE
        and edge.buffer_id not in loop_carried
        and graph.buffer_for_id(edge.buffer_id).scope not in ("", "global")
        and not graph.buffer_for_id(edge.buffer_id).scope.startswith("local")
    }
    return tuple(sorted(candidates))
