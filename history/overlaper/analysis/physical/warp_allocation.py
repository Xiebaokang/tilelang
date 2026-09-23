"""Enumerate physical warps and Hopper register redistribution."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from ...headware.spec import InstructionType, Resource
from ...parse.graph import DataflowGraph
from ..schedule.multi_version import _validate_groups
from ..schedule.order import ProgramOrders


_REGISTER_BYTES = 4
_SETMAXNREG_MIN = 24
_SETMAXNREG_MAX = 240
_SETMAXNREG_GRANULARITY = 8
_WGMMA_TILE_ROWS = 64


@dataclass(frozen=True, slots=True)
class GroupWarpAllocation:
    """One contiguous warp interval assigned to a logical group."""

    group_id: int
    first_warp: int
    warp_count: int

    @property
    def warp_stop(self) -> int:
        return self.first_warp + self.warp_count


@dataclass(frozen=True, slots=True)
class WarpAllocation:
    """Warp widths and optional per-group ``setmaxnreg`` controls."""

    groups: tuple[GroupWarpAllocation, ...]
    effective_threads: int
    register_counts: tuple[int, ...] | None = None
    register_is_increase: tuple[bool, ...] | None = None

    @property
    def total_warps(self) -> int:
        return sum(group.warp_count for group in self.groups)

    @property
    def setmaxnreg_enabled(self) -> bool:
        return self.register_counts is not None


@dataclass(frozen=True, slots=True)
class _WarpRequirement:
    minimum: int
    maximum: int
    multiple: int


def _resource(graph: DataflowGraph) -> Resource:
    if graph.hardware is None:
        raise ValueError("warp allocation requires graph.hardware")
    resource = graph.hardware.device_resource
    if resource.register_file_capacity < 1:
        raise ValueError("register file capacity must be positive")
    return resource


def _validate_inputs(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
) -> tuple[int, ...]:
    group_ids = tuple(sorted(_validate_groups(graph, groups)))
    buffer_ids = {buffer.buffer_id for buffer in graph.buffers}
    if set(versions) != buffer_ids or any(
        version < 1 for version in versions.values()
    ):
        raise ValueError("versions must cover every buffer with positive counts")
    if set(orders) != set(range(len(graph.region_kinds))):
        raise ValueError("orders must cover every region exactly once")
    for region_id, group_orders in orders.items():
        if set(group_orders) != set(group_ids):
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
    _resource(graph)
    return group_ids


def register_receiver_groups(
    graph: DataflowGraph,
    groups: Mapping[int, int],
) -> frozenset[int]:
    """Return groups whose WGMMA work receives redistributed registers."""

    _resource(graph)
    return frozenset(
        groups[node.node_id]
        for node in graph.nodes
        if node.instruction.name == "wgmma"
    )


def _group_sequence(
    orders: ProgramOrders,
    group_id: int,
) -> tuple[int, ...]:
    return tuple(
        node_id
        for region_id in sorted(orders)
        for node_id in sorted(
            orders[region_id][group_id],
            key=orders[region_id][group_id].__getitem__,
        )
    )


def estimate_group_registers_per_thread(
    graph: DataflowGraph,
    group_id: int,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    warp_count: int,
) -> int:
    """Estimate peak explicit local-buffer registers for one group thread."""

    resource = _resource(graph)
    if warp_count < 1:
        raise ValueError("warp_count must be positive")
    sequence = _group_sequence(orders, group_id)
    positions = {node_id: position for position, node_id in enumerate(sequence)}
    live_ranges = []
    for buffer in graph.buffers:
        if not buffer.scope.startswith("local"):
            continue
        users = tuple(
            node.node_id
            for node in graph.nodes
            if buffer.buffer_id in (*node.reads, *node.writes)
        )
        if not users:
            continue
        user_groups = {groups[node_id] for node_id in users}
        if len(user_groups) != 1:
            raise ValueError(f"local buffer {buffer.name} crosses groups")
        if group_id not in user_groups:
            continue
        if buffer.nbytes is None:
            raise ValueError(f"local buffer {buffer.name} has dynamic size")
        registers = math.ceil(
            buffer.nbytes * versions[buffer.buffer_id] / _REGISTER_BYTES
        )
        if buffer.scope == "local.fragment":
            registers = math.ceil(
                registers / (warp_count * resource.warp_size)
            )
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


def _register_estimates(
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


def _register_usage(
    resource: Resource,
    warp_counts: tuple[int, ...],
    register_counts: tuple[int, ...],
) -> int:
    return sum(
        warps * resource.warp_size * registers
        for warps, registers in zip(warp_counts, register_counts)
    )


def _registers_fit(
    resource: Resource,
    warp_counts: tuple[int, ...],
    register_counts: tuple[int, ...],
) -> bool:
    # Do not consume the entire modeled CTA register pool.  A Hopper
    # setmaxnreg.inc can otherwise wait forever for registers that no donor
    # warpgroup can release.  Counts and warp widths are quantized below, so
    # the strict comparison leaves at least one allocation quantum unused.
    return all(
        count <= resource.max_registers_per_thread
        for count in register_counts
    ) and _register_usage(
        resource, warp_counts, register_counts
    ) < resource.register_file_capacity


def _wgmma_required_warps(
    graph: DataflowGraph,
    group_id: int,
    groups: Mapping[int, int],
    granule: int,
) -> int | None:
    requirements = []
    for node in graph.nodes:
        if groups[node.node_id] != group_id or node.instruction.name != "wgmma":
            continue
        rows = []
        for buffer_id in node.writes:
            buffer = graph.buffer_for_id(buffer_id)
            shape = getattr(buffer.buffer, "shape", ())
            if buffer.scope == "local.fragment" and shape:
                try:
                    rows.append(int(shape[0]))
                except (TypeError, ValueError):
                    pass
        if rows:
            requirements.append(min(rows) // _WGMMA_TILE_ROWS * granule)
    if not requirements:
        return None
    # Every WGMMA in one logical group executes on the same physical warp
    # interval.  Differing fragment-row requirements therefore cannot share
    # one allocation.
    return requirements[0] if len(set(requirements)) == 1 else 0


def _warp_requirements(
    graph: DataflowGraph,
    groups: Mapping[int, int],
) -> tuple[_WarpRequirement, ...]:
    resource = _resource(graph)
    granule = resource.specialized_group_warp_multiple
    hardware_maximum = resource.max_warps_per_block
    requirements = []
    for group_id in range(len(set(groups.values()))):
        instructions = tuple(
            node.instruction
            for node in graph.nodes
            if groups[node.node_id] == group_id
        )
        memory_only = instructions and all(
            instruction.type == InstructionType.MEMORY
            for instruction in instructions
        )
        has_wgmma = any(
            instruction.name == "wgmma" for instruction in instructions
        )
        # Producer / softmax groups without WGMMA used to grow to the CTA
        # warp limit.  That inflated launch_bounds until ptxas sat below the
        # WGMMA register floor, broke TMA swizzle bijections, and deadlocked
        # mbarriers.  Keep them at one warpgroup; only WGMMA groups may grow.
        maximum = granule if memory_only or not has_wgmma else hardware_maximum
        wgmma_required = _wgmma_required_warps(
            graph, group_id, groups, granule
        )
        minimum = granule
        if wgmma_required is not None:
            minimum = max(minimum, wgmma_required)
            maximum = min(maximum, wgmma_required)
        requirements.append(_WarpRequirement(minimum, maximum, granule))
    return tuple(requirements)


def warp_requirements_are_feasible(
    graph: DataflowGraph,
    groups: Mapping[int, int],
) -> bool:
    """Return whether every group has a non-empty legal warp-count interval."""

    requirements = _warp_requirements(graph, groups)
    return all(item.minimum <= item.maximum for item in requirements)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _minimum_register_count(estimate: int, resource: Resource) -> int | None:
    count = _align_up(
        max(estimate, _SETMAXNREG_MIN), _SETMAXNREG_GRANULARITY
    )
    maximum = min(_SETMAXNREG_MAX, resource.max_registers_per_thread)
    return count if count <= maximum else None


def _receiver_groups(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    minimum_counts: tuple[int, ...],
) -> frozenset[int] | None:
    group_ids = frozenset(range(len(minimum_counts)))
    receivers = register_receiver_groups(graph, groups)
    if not receivers:
        return None
    if receivers != group_ids:
        return receivers
    if len(group_ids) == 1:
        return None
    donor = min(group_ids, key=lambda group_id: (minimum_counts[group_id], group_id))
    return group_ids - {donor}


def _register_assignment(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    warp_counts: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[bool, ...]] | None:
    resource = _resource(graph)
    estimates = _register_estimates(
        graph, groups, orders, versions, warp_counts
    )
    aligned = tuple(
        _minimum_register_count(estimate, resource) for estimate in estimates
    )
    if any(count is None for count in aligned):
        return None
    minimum = tuple(int(count) for count in aligned)
    receivers = _receiver_groups(graph, groups, minimum)
    if not receivers or not _registers_fit(resource, warp_counts, minimum):
        return None

    counts = list(minimum)
    used = _register_usage(resource, warp_counts, minimum)
    maximum = min(_SETMAXNREG_MAX, resource.max_registers_per_thread)
    while True:
        eligible = [
            group_id
            for group_id in receivers
            if counts[group_id] + _SETMAXNREG_GRANULARITY <= maximum
            # Preserve one register-allocation quantum for setmaxnreg.inc.
            and used
            + warp_counts[group_id]
            * resource.warp_size
            * _SETMAXNREG_GRANULARITY
            < resource.register_file_capacity
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
        counts[group_id] += _SETMAXNREG_GRANULARITY
        used += (
            warp_counts[group_id]
            * resource.warp_size
            * _SETMAXNREG_GRANULARITY
        )
    return (
        tuple(counts),
        tuple(group_id in receivers for group_id in range(len(counts))),
    )


def _make_allocation(
    warp_counts: tuple[int, ...],
    resource: Resource,
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
    requirements: tuple[_WarpRequirement, ...],
    maximum_warps: int,
) -> Iterator[tuple[int, ...]]:
    selected = []

    def visit(index: int, used: int) -> Iterator[tuple[int, ...]]:
        if index == len(requirements):
            yield tuple(selected)
            return
        requirement = requirements[index]
        reserved = sum(item.minimum for item in requirements[index + 1 :])
        maximum = min(requirement.maximum, maximum_warps - used - reserved)
        for count in range(
            requirement.minimum, maximum + 1, requirement.multiple
        ):
            selected.append(count)
            yield from visit(index + 1, used + count)
            selected.pop()

    yield from visit(0, 0)


def enumerate_warp_allocations(
    graph: DataflowGraph,
    groups: Mapping[int, int],
    orders: ProgramOrders,
    versions: Mapping[int, int],
    original_threads: int | None = None,
) -> Iterator[WarpAllocation]:
    """Enumerate feasible warp widths and register limits for one schedule."""

    group_ids = _validate_inputs(graph, groups, orders, versions)
    resource = _resource(graph)
    threads = graph.kernel_threads if original_threads is None else original_threads
    if threads is None:
        raise ValueError("original thread count is unknown")
    if threads < 1 or threads % resource.warp_size:
        raise ValueError("original_threads must contain whole warps")
    if threads > resource.max_threads_per_block:
        raise ValueError("original_threads exceeds the hardware limit")

    requirements = _warp_requirements(graph, groups)

    if len(group_ids) == 1:
        warp_count = threads // resource.warp_size
        requirement = requirements[0]
        if not (
            requirement.minimum <= warp_count <= requirement.maximum
            and warp_count % requirement.multiple == 0
        ):
            return
        estimates = _register_estimates(
            graph, groups, orders, versions, (warp_count,)
        )
        if _registers_fit(resource, (warp_count,), estimates):
            yield _make_allocation((warp_count,), resource)
        return

    for warp_counts in _enumerate_warp_counts(
        requirements, resource.max_warps_per_block
    ):
        register_assignment = _register_assignment(
            graph, groups, orders, versions, warp_counts
        )
        if register_assignment is None:
            continue
        registers, actions = register_assignment
        yield _make_allocation(
            warp_counts, resource, registers, actions
        )
