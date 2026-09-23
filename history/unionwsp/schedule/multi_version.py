"""Determine minimum buffer versions after stage, group, and order are fixed."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import product

from ..parseIR.graph import (
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
    """Return all accesses and writes, falling back to node-level metadata."""

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


def _validate_inputs(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
) -> None:
    node_ids = {node.node_id for node in graph.nodes}
    if set(groups) != node_ids:
        raise ValueError("groups must cover every node exactly once")
    group_ids = set(groups.values())
    if group_ids != set(range(len(group_ids))):
        raise ValueError("group IDs must be dense and non-empty")
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


def _minimum_region_versions(
    graph: DataflowGraph,
    region_id: int,
    buffer_id: int,
    stages: Mapping[int, int],
    groups: Mapping[int, int],
    group_orders: Mapping[int, Mapping[int, int]],
    maximum: int,
) -> int:
    accesses, writers = _region_buffer_accesses(
        graph,
        region_id,
        buffer_id,
    )
    if not accesses or not writers:
        return 1

    for versions in range(1, maximum + 1):
        safe = True
        for writer_id, writer_access in writers:
            for accessor_id, accessor_access in accesses:
                if not _ranges_may_overlap(accessor_access, writer_access):
                    continue
                reuse_distance = (
                    versions
                    + stages[writer_id]
                    - stages[accessor_id]
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
    raise ValueError(
        f"buffer {buffer.name} needs more than {maximum} versions"
    )


def analyze_buffer_versions(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
) -> dict[int, int]:
    """Return the minimum correctness version count for every buffer.

    Cross-group reuse is assumed to receive a later consumer-to-producer
    synchronization channel. Versioning and that backpressure are separate
    decisions: a larger finite ring cannot by itself bound an independent
    producer group.
    """

    _validate_inputs(graph, stages_by_region, groups, orders)
    versions = {buffer.buffer_id: 1 for buffer in graph.buffers}

    for region_id, kind in enumerate(graph.region_kinds):
        if kind != RegionKind.PIPELINE:
            continue
        stages = stages_by_region[region_id]
        region_maximum = max(stages.values(), default=0) + 1
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
                region_maximum,
            )
            versions[buffer.buffer_id] = max(
                versions[buffer.buffer_id],
                required,
            )
    return versions


def multiversion_candidate_buffers(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
) -> tuple[int, ...]:
    """Return buffers eligible for independent ``min``/``min + 1`` choices.

    A candidate has an overlapping writer/accessor pair at different stages,
    and that exact writer's hardware instruction supports multiversion output.
    Loop-carried state is deliberately excluded: rotating recurrence values
    requires recurrence-aware lowering rather than ordinary ring buffering.
    """

    if graph.hardware is None:
        raise ValueError("multiversion analysis requires graph.hardware")
    pipeline_regions = {
        region_id
        for region_id, kind in enumerate(graph.region_kinds)
        if kind == RegionKind.PIPELINE
    }
    if set(stages_by_region) != pipeline_regions:
        raise ValueError("stages must cover every pipeline region exactly once")
    for region_id in pipeline_regions:
        node_ids = {node.node_id for node in graph.nodes_for_region(region_id)}
        stages = stages_by_region[region_id]
        if set(stages) != node_ids or any(
            stage < 0 for stage in stages.values()
        ):
            raise ValueError("stages must cover every pipeline node")
    loop_carried_buffers = {
        edge.buffer_id
        for edge in graph.edges
        if edge.is_loop_carried and edge.buffer_id is not None
    }
    candidates = []
    for buffer in graph.buffers:
        if (
            buffer.scope in ("", "global")
            or buffer.scope == "local.fragment"
            or buffer.buffer_id in loop_carried_buffers
        ):
            continue
        eligible = False
        for region_id in sorted(pipeline_regions):
            stages = stages_by_region[region_id]
            accesses, writers = _region_buffer_accesses(
                graph, region_id, buffer.buffer_id
            )
            for writer_id, writer_access in writers:
                writer = graph.node_for_id(writer_id)
                if not graph.hardware.supports_multiversion_output(
                    writer.instruction_kind
                ):
                    continue
                if any(
                    stages[writer_id] != stages[accessor_id]
                    and _ranges_may_overlap(accessor_access, writer_access)
                    for accessor_id, accessor_access in accesses
                ):
                    eligible = True
                    break
            if eligible:
                break
        if eligible:
            candidates.append(buffer.buffer_id)
    return tuple(candidates)


def enumerate_buffer_version_plans(
    graph: DataflowGraph,
    stages_by_region: Mapping[int, Mapping[int, int]],
    groups: Mapping[int, int],
    orders: ProgramOrders,
) -> Iterator[dict[int, int]]:
    """Enumerate independent minimum/expanded choices for eligible buffers.

    Every yielded plan covers every graph buffer. Non-candidates always use
    their correctness minimum. Each candidate independently uses either its
    minimum or one extra version. Hardware resource analysis later removes
    expanded choices that do not fit the target.
    """

    minimum = analyze_buffer_versions(
        graph,
        stages_by_region,
        groups,
        orders,
    )
    # LayoutInference describes a fragment's logical rank explicitly.  The
    # current lowering cannot prepend a physical version axis to that layout.
    # If correctness already needs multiple fragment versions, the schedule is
    # unrealizable; otherwise the fragment remains fixed at one version.
    if any(
        buffer.scope == "local.fragment"
        and minimum[buffer.buffer_id] > 1
        for buffer in graph.buffers
    ):
        return
    candidates = multiversion_candidate_buffers(graph, stages_by_region)
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
    """Return only buffers whose minimum version count exceeds one."""

    return {
        buffer_id: count
        for buffer_id, count in analyze_buffer_versions(
            graph,
            stages_by_region,
            groups,
            orders,
        ).items()
        if count > 1
    }
