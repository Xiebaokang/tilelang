"""Enumerate warp widths and optional register redistribution."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping

from OverlapPlaner.arch.api import ClassifiedGraph, DeviceResource
from OverlapPlaner.facts import FactGraph
from OverlapPlaner.physical.model import (
    GroupWarpAllocation,
    SetMaxNRegPolicy,
    WarpAllocation,
    WarpRequirement,
)
from OverlapPlaner.structure.model import ProgramOrders, Structure, validate_stage_group_order

_REGISTER_BYTES = 4


def estimate_group_registers_per_thread(
    graph: FactGraph,
    group_id: int,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    warp_count: int,
    resource: DeviceResource,
) -> int:
    """Estimate peak explicit local-buffer registers for one group thread.

    A fragment that is handed off across groups stays PRIVATE in each group,
    so only this group's users contribute to the live range. Pipeline versions
    of a fragment are distinct register tiles, so ``nbytes`` is scaled by
    ``versions``.
    """

    if warp_count < 1:
        raise ValueError("warp_count must be positive")
    sequence = tuple(
        node_id
        for region_id in sorted(orders)
        for node_id in sorted(
            orders[region_id][group_id],
            key=orders[region_id][group_id].__getitem__,
        )
    )
    positions = {node_id: position for position, node_id in enumerate(sequence)}
    live_ranges = []
    for buffer in graph.buffers:
        if not buffer.scope.startswith("local"):
            continue
        all_users = tuple(
            node.node_id
            for node in graph.nodes
            if buffer.buffer_id in (*node.reads, *node.writes)
        )
        if not all_users:
            continue
        user_groups = {groups[node_id] for node_id in all_users}
        if len(user_groups) > 1 and buffer.scope != "local.fragment":
            raise ValueError(f"local buffer {buffer.name} crosses groups")
        users = tuple(
            node_id for node_id in all_users if groups[node_id] == group_id
        )
        if not users:
            continue
        if buffer.nbytes is None:
            raise ValueError(f"local buffer {buffer.name} has dynamic size")
        registers = math.ceil(
            buffer.nbytes * versions[buffer.buffer_id] / _REGISTER_BYTES
        )
        if buffer.scope == "local.fragment":
            registers = math.ceil(registers / (warp_count * resource.warp_size))
        live_ranges.append(
            (
                min(positions[node_id] for node_id in users),
                max(positions[node_id] for node_id in users),
                registers,
            )
        )
    return max(
        (
            sum(
                registers
                for start, stop, registers in live_ranges
                if start <= position <= stop
            )
            for position in range(len(sequence))
        ),
        default=0,
    )


def warp_requirements_are_feasible(requirements: tuple[WarpRequirement, ...]) -> bool:
    return all(item.minimum <= item.maximum for item in requirements)


def _register_estimates(
    graph: FactGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    warp_counts: tuple[int, ...],
    resource: DeviceResource,
) -> tuple[int, ...]:
    return tuple(
        estimate_group_registers_per_thread(
            graph, group_id, groups, orders, versions, warp_count, resource
        )
        for group_id, warp_count in enumerate(warp_counts)
    )


def _register_usage(
    resource: DeviceResource,
    warp_counts: tuple[int, ...],
    register_counts: tuple[int, ...],
) -> int:
    return sum(
        warps * resource.warp_size * registers
        for warps, registers in zip(warp_counts, register_counts)
    )


def _registers_fit(
    resource: DeviceResource,
    warp_counts: tuple[int, ...],
    register_counts: tuple[int, ...],
) -> bool:
    return all(
        count <= resource.max_registers_per_thread for count in register_counts
    ) and _register_usage(
        resource, warp_counts, register_counts
    ) <= resource.register_file_capacity


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _minimum_register_count(
    estimate: int, resource: DeviceResource, policy: SetMaxNRegPolicy
) -> int | None:
    count = _align_up(max(estimate, policy.min_count), policy.granularity)
    maximum = min(policy.max_count, resource.max_registers_per_thread)
    return count if count <= maximum else None


def _receiver_groups(
    receivers: frozenset[int],
    minimum_counts: tuple[int, ...],
) -> frozenset[int] | None:
    group_ids = frozenset(range(len(minimum_counts)))
    if not receivers:
        return None
    if receivers != group_ids:
        return receivers
    if len(group_ids) == 1:
        return None
    donor = min(group_ids, key=lambda group_id: (minimum_counts[group_id], group_id))
    return group_ids - {donor}


def _register_assignment(
    graph: FactGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    warp_counts: tuple[int, ...],
    resource: DeviceResource,
    receivers: frozenset[int],
    policy: SetMaxNRegPolicy,
) -> tuple[tuple[int, ...], tuple[bool, ...]] | None:
    estimates = _register_estimates(
        graph, groups, orders, versions, warp_counts, resource
    )
    aligned = tuple(
        _minimum_register_count(estimate, resource, policy) for estimate in estimates
    )
    if any(count is None for count in aligned):
        return None
    minimum = tuple(int(count) for count in aligned)
    assigned_receivers = _receiver_groups(receivers, minimum)
    if not assigned_receivers or not _registers_fit(resource, warp_counts, minimum):
        return None

    counts = list(minimum)
    used = _register_usage(resource, warp_counts, minimum)
    maximum = min(policy.max_count, resource.max_registers_per_thread)
    while True:
        eligible = [
            group_id
            for group_id in assigned_receivers
            if counts[group_id] + policy.granularity <= maximum
            and used
            + warp_counts[group_id] * resource.warp_size * policy.granularity
            <= resource.register_file_capacity
        ]
        if not eligible:
            break
        group_id = min(
            eligible,
            key=lambda candidate: (
                counts[candidate] / minimum[candidate],
                candidate,
            ),
        )
        counts[group_id] += policy.granularity
        used += warp_counts[group_id] * resource.warp_size * policy.granularity
    return (
        tuple(counts),
        tuple(group_id in assigned_receivers for group_id in range(len(counts))),
    )


def _make_allocation(
    warp_counts: tuple[int, ...],
    resource: DeviceResource,
    registers: tuple[int, ...] | None = None,
    actions: tuple[bool, ...] | None = None,
) -> WarpAllocation:
    first_warp = 0
    groups = []
    for group_id, warp_count in enumerate(warp_counts):
        groups.append(GroupWarpAllocation(group_id, first_warp, warp_count))
        first_warp += warp_count
    return WarpAllocation(
        tuple(groups), first_warp * resource.warp_size, registers, actions
    )


def _enumerate_warp_counts(
    requirements: tuple[WarpRequirement, ...],
    maximum_warps: int,
) -> Iterator[tuple[int, ...]]:
    selected: list[int] = []

    def visit(index: int, used: int) -> Iterator[tuple[int, ...]]:
        if index == len(requirements):
            yield tuple(selected)
            return
        requirement = requirements[index]
        reserved = sum(item.minimum for item in requirements[index + 1 :])
        maximum = min(requirement.maximum, maximum_warps - used - reserved)
        for count in range(requirement.minimum, maximum + 1, requirement.multiple):
            selected.append(count)
            yield from visit(index + 1, used + count)
            selected.pop()

    yield from visit(0, 0)


def enumerate_warp_allocations(
    classified: ClassifiedGraph,
    structure: Structure,
    resource: DeviceResource,
    requirements: tuple[WarpRequirement, ...],
    *,
    register_receivers: frozenset[int] = frozenset(),
    setmaxnreg: SetMaxNRegPolicy | None = None,
) -> Iterator[WarpAllocation]:
    """Enumerate feasible warp widths for one structure."""

    graph = classified.graph
    validate_stage_group_order(
        graph, structure.stages_by_region, structure.groups, structure.orders
    )
    group_ids = tuple(range(structure.num_groups))
    if len(requirements) != len(group_ids):
        raise ValueError("warp requirements must cover every group")
    threads = graph.kernel_threads
    if threads is None:
        raise ValueError("original thread count is unknown")
    if threads < 1 or threads % resource.warp_size:
        raise ValueError("kernel_threads must contain whole warps")
    if threads > resource.max_threads_per_block:
        raise ValueError("kernel_threads exceeds the hardware limit")

    if len(group_ids) == 1:
        warp_count = threads // resource.warp_size
        requirement = requirements[0]
        if not (
            requirement.minimum <= warp_count <= requirement.maximum
            and warp_count % requirement.multiple == 0
        ):
            return
        estimates = _register_estimates(
            graph,
            structure.groups,
            structure.orders,
            structure.buffer_versions,
            (warp_count,),
            resource,
        )
        if _registers_fit(resource, (warp_count,), estimates):
            yield _make_allocation((warp_count,), resource)
        return

    for warp_counts in _enumerate_warp_counts(
        requirements, resource.max_warps_per_block
    ):
        estimates = _register_estimates(
            graph,
            structure.groups,
            structure.orders,
            structure.buffer_versions,
            warp_counts,
            resource,
        )
        # ptxas applies launch_bounds to the whole CTA before runtime
        # setmaxnreg redistributes registers between warp groups.  A kernel
        # that needs more static registers in any group than the uniform CTA
        # budget cannot compile, even if its summed per-group usage fits.
        static_limit = min(
            resource.max_registers_per_thread,
            resource.register_file_capacity
            // (sum(warp_counts) * resource.warp_size),
        )
        if max(estimates, default=0) > static_limit:
            continue
        # Hopper setmaxnreg is a warpgroup collective. MMA-only partitions may
        # legally use a one-warp granularity, but those allocations must use
        # the ordinary register limit rather than executing setmaxnreg on an
        # incomplete four-warp domain.
        can_redistribute = setmaxnreg is not None and all(
            count % resource.partition_warp_multiple == 0
            for count in warp_counts
        )
        if not can_redistribute:
            if _registers_fit(resource, warp_counts, estimates):
                yield _make_allocation(warp_counts, resource)
            continue
        assignment = _register_assignment(
            graph,
            structure.groups,
            structure.orders,
            structure.buffer_versions,
            warp_counts,
            resource,
            register_receivers,
            setmaxnreg,
        )
        if assignment is None:
            continue
        registers, actions = assignment
        yield _make_allocation(warp_counts, resource, registers, actions)
