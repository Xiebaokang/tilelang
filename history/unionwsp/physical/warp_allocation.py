"""Jointly enumerate physical warps and register redistribution."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from ..parseIR.graph import DataflowGraph
from ..schedule.order import ProgramOrders


@dataclass(frozen=True)
class GroupWarpAllocation:
    """One contiguous physical warp interval owned by a logical group."""

    group_id: int
    first_warp: int
    warp_count: int

    @property
    def warp_stop(self) -> int:
        return self.first_warp + self.warp_count


@dataclass(frozen=True)
class WarpAllocation:
    """One joint warp-count and per-group setmaxnreg assignment."""

    groups: tuple[GroupWarpAllocation, ...]
    effective_threads: int
    register_counts: tuple[int, ...] | None = None
    register_is_increase: tuple[bool, ...] | None = None

    @property
    def setmaxnreg_enabled(self) -> bool:
        return self.register_counts is not None

    @property
    def total_warps(self) -> int:
        return sum(group.warp_count for group in self.groups)


@dataclass(frozen=True)
class _GroupWarpRequirement:
    minimum_warps: int
    warp_multiple: int
    maximum_warps: int


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def _validate_inputs(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
) -> tuple[int, ...]:
    node_ids = {node.node_id for node in graph.nodes}
    if set(groups) != node_ids:
        raise ValueError("groups must cover every node exactly once")
    group_ids = tuple(sorted(set(groups.values())))
    if group_ids != tuple(range(len(group_ids))):
        raise ValueError("group IDs must be dense and non-empty")
    if graph.hardware is None:
        raise ValueError("warp allocation requires graph.hardware")
    if graph.hardware.register_file_capacity < 1:
        raise ValueError("register file capacity must be configured")
    if set(versions) != {buffer.buffer_id for buffer in graph.buffers}:
        raise ValueError("versions must cover every buffer exactly once")
    if any(version < 1 for version in versions.values()):
        raise ValueError("every buffer version count must be positive")
    if set(orders) != set(range(len(graph.region_kinds))):
        raise ValueError("orders must cover every region exactly once")
    for region_id in range(len(graph.region_kinds)):
        if set(orders[region_id]) != set(group_ids):
            raise ValueError("each region order must cover every group")
        region_nodes = {node.node_id for node in graph.nodes_for_region(region_id)}
        for group_id, local_order in orders[region_id].items():
            expected = {
                node_id
                for node_id in region_nodes
                if groups[node_id] == group_id
            }
            if set(local_order) != expected:
                raise ValueError("group order covers the wrong region nodes")
            if set(local_order.values()) != set(range(len(expected))):
                raise ValueError("local order positions must be dense")
    return group_ids


def register_receiver_groups(
    graph: DataflowGraph, groups: Mapping[int, int]
) -> frozenset[int]:
    """Return groups containing hardware-preferred register receivers."""

    if graph.hardware is None:
        raise ValueError("register role analysis requires graph.hardware")
    group_ids = tuple(sorted(set(groups.values())))
    return frozenset(
        group_id
        for group_id in group_ids
        if any(
            groups[node.node_id] == group_id
            and node.instruction_kind
            in graph.hardware.register_receiver_kinds
            for node in graph.nodes
        )
    )


def _group_sequence(
    orders: ProgramOrders, group_id: int
) -> tuple[int, ...]:
    result = []
    for region_id in sorted(orders):
        local_order = orders[region_id][group_id]
        result.extend(sorted(local_order, key=local_order.__getitem__))
    return tuple(result)


def _registers_per_thread(
    scope: str,
    nbytes: int,
    versions: int,
    thread_count: int,
) -> int:
    register_words = math.ceil(nbytes * versions / 4)
    if scope == "local.fragment":
        return math.ceil(register_words / thread_count)
    return register_words


def estimate_group_registers_per_thread(
    graph: DataflowGraph,
    group_id: int,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    warp_count: int,
) -> int:
    """Estimate peak explicit local-buffer registers for one group thread."""

    if graph.hardware is None:
        raise ValueError("register estimation requires graph.hardware")
    if warp_count < 1:
        raise ValueError("warp_count must be positive")
    thread_count = warp_count * graph.hardware.warp_size
    sequence = _group_sequence(orders, group_id)
    positions = {node_id: position for position, node_id in enumerate(sequence)}
    live_ranges = []
    for buffer in graph.buffers:
        if not buffer.scope.startswith("local"):
            continue
        users = tuple(
            node.node_id
            for node in graph.nodes
            if buffer.buffer_id in node.reads or buffer.buffer_id in node.writes
        )
        if not users:
            continue
        user_groups = {groups[node_id] for node_id in users}
        if len(user_groups) != 1:
            raise ValueError(f"local buffer {buffer.name} crosses physical groups")
        if group_id not in user_groups:
            continue
        if buffer.nbytes is None:
            raise ValueError(f"local buffer {buffer.name} has a dynamic size")
        live_ranges.append(
            (
                min(positions[node_id] for node_id in users),
                max(positions[node_id] for node_id in users),
                _registers_per_thread(
                    buffer.scope,
                    buffer.nbytes,
                    versions[buffer.buffer_id],
                    thread_count,
                ),
            )
        )
    return max(
        (
            sum(
                register_count
                for start, stop, register_count in live_ranges
                if start <= position <= stop
            )
            for position in range(len(sequence))
        ),
        default=0,
    )


def _estimate_register_counts(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    warp_counts: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        estimate_group_registers_per_thread(
            graph, group_id, groups, orders, versions, warp_count
        )
        for group_id, warp_count in enumerate(warp_counts)
    )


def _register_usage_fits(
    graph: DataflowGraph,
    warp_counts: tuple[int, ...],
    estimates: tuple[int, ...],
    register_counts: tuple[int, ...] | None,
) -> bool:
    assert graph.hardware is not None
    hardware = graph.hardware
    if len(estimates) != len(warp_counts):
        return False
    if register_counts is None:
        if any(
            estimate > hardware.max_registers_per_thread
            for estimate in estimates
        ):
            return False
        budget_counts = estimates
    else:
        if len(register_counts) != len(warp_counts) or any(
            estimate > limit
            for estimate, limit in zip(estimates, register_counts)
        ):
            return False
        budget_counts = register_counts
    return (
        sum(
            warp_count * hardware.warp_size * count
            for warp_count, count in zip(warp_counts, budget_counts)
        )
        <= hardware.register_file_capacity
    )


def _group_warp_requirements(
    graph: DataflowGraph,
    groups: Mapping[int, int],
) -> tuple[_GroupWarpRequirement, ...]:
    assert graph.hardware is not None
    hardware = graph.hardware
    maximum_warps = hardware.max_threads_per_block // hardware.warp_size
    requirements = []
    for group_id in range(len(set(groups.values()))):
        kinds = {
            node.instruction_kind
            for node in graph.nodes
            if groups[node.node_id] == group_id
        }
        warp_multiple = hardware.specialized_group_warp_multiple
        has_collective = bool(kinds & hardware.warpgroup_collective_kinds)
        if has_collective:
            warp_multiple = math.lcm(
                warp_multiple, hardware.warpgroup_warps
            )
        minimum_warps = warp_multiple
        fixed = bool(kinds) and kinds <= hardware.fixed_one_granule_kinds
        group_maximum_warps = minimum_warps if fixed else maximum_warps
        if (
            has_collective
            and hardware.warpgroup_collective_max_warps is not None
        ):
            group_maximum_warps = min(
                group_maximum_warps,
                hardware.warpgroup_collective_max_warps,
            )
        if (
            has_collective
            and hardware.warpgroup_collective_tile_rows is not None
        ):
            tile_limits = []
            for node in graph.nodes:
                if (
                    groups[node.node_id] != group_id
                    or node.instruction_kind
                    not in hardware.warpgroup_collective_kinds
                ):
                    continue
                output_rows = []
                for buffer_id in node.writes:
                    buffer = graph.buffer_for_id(buffer_id)
                    shape = getattr(buffer.buffer, "shape", ())
                    if buffer.scope == "local.fragment" and shape:
                        first_extent = shape[0]
                        try:
                            output_rows.append(int(first_extent))
                        except (TypeError, ValueError):
                            pass
                if output_rows:
                    rows = min(output_rows)
                    tile_limit = hardware.collective_max_warps(rows)
                    if tile_limit is not None:
                        tile_limits.append(tile_limit)
            if tile_limits:
                group_maximum_warps = min(
                    group_maximum_warps, min(tile_limits)
                )
        requirements.append(
            _GroupWarpRequirement(
                minimum_warps,
                warp_multiple,
                group_maximum_warps,
            )
        )
    return tuple(requirements)


def _minimum_register_count(estimated: int, graph: DataflowGraph) -> int | None:
    assert graph.hardware is not None
    hardware = graph.hardware
    count = _align_up(
        max(estimated, hardware.setmaxnreg_min_registers),
        hardware.setmaxnreg_granularity,
    )
    maximum = _align_down(
        min(
            hardware.setmaxnreg_max_registers,
            hardware.max_registers_per_thread,
        ),
        hardware.setmaxnreg_granularity,
    )
    return count if count <= maximum else None


def _select_receiver_groups(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    minimum_counts: tuple[int, ...],
) -> frozenset[int] | None:
    """Select unique dynamic roles after warp-dependent demands are known."""

    assert graph.hardware is not None
    hardware = graph.hardware
    group_ids = frozenset(range(len(minimum_counts)))
    preferred = register_receiver_groups(graph, groups)
    if not preferred:
        return None
    if preferred != group_ids:
        return preferred
    if len(group_ids) < 2:
        return None

    def donor_key(group_id: int) -> tuple[bool, int, int]:
        kinds = {
            node.instruction_kind
            for node in graph.nodes
            if groups[node.node_id] == group_id
        }
        return (
            bool(kinds & hardware.warpgroup_collective_kinds),
            minimum_counts[group_id],
            group_id,
        )

    donor = min(group_ids, key=donor_key)
    return group_ids - {donor}


def _solve_register_assignment(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    warp_counts: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[bool, ...]] | None:
    assert graph.hardware is not None
    hardware = graph.hardware
    group_ids = frozenset(range(len(warp_counts)))

    estimates = _estimate_register_counts(
        graph, groups, orders, versions, warp_counts
    )
    minimum_counts = [
        _minimum_register_count(estimate, graph) for estimate in estimates
    ]
    if any(count is None for count in minimum_counts):
        return None
    minimum = tuple(int(count) for count in minimum_counts)
    receivers = _select_receiver_groups(graph, groups, minimum)
    if not receivers or receivers == group_ids:
        return None
    counts = list(minimum)
    used = sum(
        warp_count * hardware.warp_size * count
        for warp_count, count in zip(warp_counts, counts)
    )
    if used > hardware.register_file_capacity:
        return None

    maximum_count = _align_down(
        min(
            hardware.setmaxnreg_max_registers,
            hardware.max_registers_per_thread,
        ),
        hardware.setmaxnreg_granularity,
    )
    while True:
        eligible = []
        for group_id in receivers:
            cost = (
                warp_counts[group_id]
                * hardware.warp_size
                * hardware.setmaxnreg_granularity
            )
            if (
                counts[group_id] + hardware.setmaxnreg_granularity
                <= maximum_count
                and used + cost <= hardware.register_file_capacity
            ):
                eligible.append(group_id)
        if not eligible:
            break
        group_id = min(
            eligible,
            key=lambda candidate: (
                counts[candidate] / minimum[candidate],
                candidate,
            ),
        )
        counts[group_id] += hardware.setmaxnreg_granularity
        used += (
            warp_counts[group_id]
            * hardware.warp_size
            * hardware.setmaxnreg_granularity
        )
    actions = tuple(group_id in receivers for group_id in range(len(counts)))
    return tuple(counts), actions


def _make_allocation(
    warp_counts: tuple[int, ...],
    warp_size: int,
    register_counts: tuple[int, ...] | None,
    register_is_increase: tuple[bool, ...] | None,
) -> WarpAllocation:
    first_warp = 0
    allocations = []
    for group_id, warp_count in enumerate(warp_counts):
        allocations.append(GroupWarpAllocation(group_id, first_warp, warp_count))
        first_warp += warp_count
    return WarpAllocation(
        tuple(allocations),
        first_warp * warp_size,
        register_counts,
        register_is_increase,
    )


def _enumerate_joint_assignments(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
) -> Iterator[
    tuple[
        tuple[int, ...],
        tuple[int, ...] | None,
        tuple[bool, ...] | None,
    ]
]:
    """Jointly enumerate group widths and their feasible register limits."""

    assert graph.hardware is not None
    hardware = graph.hardware
    requirements = _group_warp_requirements(graph, groups)
    maximum_warps = hardware.max_threads_per_block // hardware.warp_size
    selected: list[int] = []

    def visit(index: int, used_warps: int):
        if index == len(requirements):
            warp_counts = tuple(selected)
            if hardware.setmaxnreg_required_for_specialization:
                assignment = _solve_register_assignment(
                    graph, groups, orders, versions, warp_counts
                )
                if assignment is None:
                    return
                counts, actions = assignment
                yield warp_counts, counts, actions
            else:
                estimates = _estimate_register_counts(
                    graph, groups, orders, versions, warp_counts
                )
                if _register_usage_fits(
                    graph, warp_counts, estimates, None
                ):
                    yield warp_counts, None, None
            return
        requirement = requirements[index]
        future_minimum_warps = sum(
            item.minimum_warps for item in requirements[index + 1 :]
        )
        maximum = min(
            requirement.maximum_warps,
            maximum_warps - used_warps - future_minimum_warps,
        )
        for warp_count in range(
            requirement.minimum_warps,
            maximum + 1,
            requirement.warp_multiple,
        ):
            selected.append(warp_count)
            yield from visit(index + 1, used_warps + warp_count)
            selected.pop()

    yield from visit(0, 0)


def enumerate_warp_allocations(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    original_threads: int,
) -> Iterator[WarpAllocation]:
    """Enumerate joint warp widths and register limits for one schedule."""

    group_ids = _validate_inputs(graph, groups, orders, versions)
    assert graph.hardware is not None
    hardware = graph.hardware
    if original_threads < 1 or original_threads % hardware.warp_size:
        raise ValueError("original_threads must contain whole warps")
    if original_threads > hardware.max_threads_per_block:
        raise ValueError("original_threads exceeds the hardware limit")

    if len(group_ids) == 1:
        warp_counts = (original_threads // hardware.warp_size,)
        requirement = _group_warp_requirements(graph, groups)[0]
        warp_count = warp_counts[0]
        if (
            warp_count < requirement.minimum_warps
            or warp_count > requirement.maximum_warps
            or warp_count % requirement.warp_multiple
        ):
            return
        estimates = _estimate_register_counts(
            graph, groups, orders, versions, warp_counts
        )
        if not _register_usage_fits(graph, warp_counts, estimates, None):
            return
        allocation = _make_allocation(
            warp_counts,
            hardware.warp_size,
            None,
            None,
        )
        validate_warp_allocation(
            graph, groups, orders, versions, original_threads, allocation
        )
        yield allocation
        return

    assignments = _enumerate_joint_assignments(
        graph, groups, orders, versions
    )
    for warp_counts, register_counts, register_actions in assignments:
        allocation = _make_allocation(
            warp_counts,
            hardware.warp_size,
            register_counts,
            register_actions,
        )
        validate_warp_allocation(
            graph, groups, orders, versions, original_threads, allocation
        )
        yield allocation


def validate_warp_allocation(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    original_threads: int,
    allocation: WarpAllocation,
) -> None:
    """Validate warp intervals and their estimated register allocation."""

    group_ids = _validate_inputs(graph, groups, orders, versions)
    assert graph.hardware is not None
    hardware = graph.hardware
    if original_threads < 1 or original_threads % hardware.warp_size:
        raise ValueError("original_threads must contain whole warps")
    if tuple(group.group_id for group in allocation.groups) != group_ids:
        raise ValueError("allocation must cover every logical group")

    requirements = _group_warp_requirements(graph, groups)
    expected_first_warp = 0
    for group in allocation.groups:
        if group.first_warp != expected_first_warp or group.warp_count < 1:
            raise ValueError("physical warp intervals must be contiguous")
        expected_first_warp = group.warp_stop
        requirement = requirements[group.group_id]
        if (
            group.warp_count < requirement.minimum_warps
            or group.warp_count > requirement.maximum_warps
            or group.warp_count % requirement.warp_multiple
        ):
            raise ValueError("group violates its hardware warp requirement")
    if allocation.effective_threads != expected_first_warp * hardware.warp_size:
        raise ValueError("effective_threads disagrees with warp intervals")
    if allocation.effective_threads > hardware.max_threads_per_block:
        raise ValueError("allocation exceeds the hardware thread limit")

    if len(group_ids) == 1:
        if allocation.effective_threads != original_threads:
            raise ValueError("a single group must preserve the original CTA")
        if allocation.setmaxnreg_enabled:
            raise ValueError("a single group must not use setmaxnreg")
        warp_counts = tuple(group.warp_count for group in allocation.groups)
        estimates = _estimate_register_counts(
            graph, groups, orders, versions, warp_counts
        )
        if not _register_usage_fits(graph, warp_counts, estimates, None):
            raise ValueError("single-group register usage exceeds the budget")
        return

    registers = allocation.register_counts
    actions = allocation.register_is_increase
    if hardware.setmaxnreg_required_for_specialization:
        if registers is None or actions is None:
            raise ValueError("specialized groups require setmaxnreg")
    elif registers is not None or actions is not None:
        raise ValueError("hardware does not require specialized setmaxnreg")
    else:
        warp_counts = tuple(group.warp_count for group in allocation.groups)
        estimates = _estimate_register_counts(
            graph, groups, orders, versions, warp_counts
        )
        if not _register_usage_fits(graph, warp_counts, estimates, None):
            raise ValueError("register usage exceeds the hardware budget")
        return
    if len(registers) != len(group_ids) or len(actions) != len(group_ids):
        raise ValueError("register controls must cover every group")

    warp_counts = tuple(group.warp_count for group in allocation.groups)
    estimates = _estimate_register_counts(
        graph, groups, orders, versions, warp_counts
    )
    minimum_counts = tuple(
        _minimum_register_count(estimate, graph) for estimate in estimates
    )
    if any(count is None for count in minimum_counts):
        raise ValueError("estimated register demand exceeds the thread limit")
    minimum = tuple(int(count) for count in minimum_counts)
    receivers = _select_receiver_groups(graph, groups, minimum)
    if not receivers or len(receivers) == len(group_ids):
        raise ValueError("setmaxnreg needs donor and receiver roles")
    if tuple(group_id in receivers for group_id in group_ids) != actions:
        raise ValueError("setmaxnreg actions disagree with hardware roles")

    maximum_count = min(
        hardware.setmaxnreg_max_registers,
        hardware.max_registers_per_thread,
    )
    for group, register_count, estimate in zip(
        allocation.groups, registers, estimates
    ):
        if (
            register_count < hardware.setmaxnreg_min_registers
            or register_count > maximum_count
            or register_count % hardware.setmaxnreg_granularity
        ):
            raise ValueError("invalid setmaxnreg count")
        if register_count < estimate:
            raise ValueError("setmaxnreg is below estimated register demand")
        if (
            group.first_warp % hardware.warpgroup_warps
            or group.warp_count % hardware.warpgroup_warps
        ):
            raise ValueError("setmaxnreg domains must be warpgroup aligned")
    if not _register_usage_fits(
        graph, warp_counts, estimates, registers
    ):
        raise ValueError("setmaxnreg allocation exceeds register demand or budget")
