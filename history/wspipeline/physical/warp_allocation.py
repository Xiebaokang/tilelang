"""Physical warp allocation for logical warp-specialized schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Mapping

from tvm.target import Target

from ..analysis.core import DataflowNode, HardwareUnit, InstructionKind
from ..analysis.program import ProgramDataflowAnalysis
from ..scheduling.warp_specialization import ProgramScheduleCandidate


class WarpAllocationMode(str, Enum):
    """Relationship between logical groups and the original CTA size."""

    PRESERVE_CTA = "preserve_cta"
    EXPAND_CTA = "expand_cta"
    EXPLICIT = "explicit"


class CUDAArchitectureFamily(str, Enum):
    """CUDA architecture family relevant to warp-specialization lowering."""

    GENERIC = "generic"
    HOPPER = "hopper"
    BLACKWELL = "blackwell"


@dataclass(frozen=True)
class TargetExecutionProfile:
    """Target-specific execution and warp-specialization granularity."""

    architecture: str
    family: CUDAArchitectureFamily
    warp_size: int
    max_threads_per_block: int
    warpgroup_warps: int
    logical_group_warp_multiple: int
    setmaxnreg_enabled: bool


@dataclass(frozen=True)
class _OperationWarpRequirement:
    """Physical participation required by one operation on one target."""

    minimum_warps: int
    warp_multiple: int
    reason: str


@dataclass(frozen=True)
class _GroupWarpRequirement:
    """Legal physical-width interval for one logical group."""

    group_id: int
    minimum_warps: int
    warp_multiple: int
    maximum_warps: int
    scalable: bool
    reason: str


@dataclass(frozen=True)
class GroupWarpAllocation:
    """A contiguous physical warp interval assigned to one logical group."""

    group_id: int
    warp_count: int
    first_warp: int

    @property
    def warp_stop(self) -> int:
        return self.first_warp + self.warp_count


@dataclass(frozen=True)
class PhysicalWarpAllocation:
    """One legal physical realization of a logical partition."""

    original_threads: int
    effective_threads: int
    groups: tuple[GroupWarpAllocation, ...]
    setmaxnreg_enabled: bool = False

    @property
    def total_warps(self) -> int:
        return sum(group.warp_count for group in self.groups)

@dataclass(frozen=True)
class RegisterWarpgroupDomain:
    """One hardware warpgroup that must share setmaxnreg control flow."""

    group_id: int
    first_warp: int
    warp_count: int
    register_count: int
    is_increase: bool


@dataclass(frozen=True)
class RegisterAllocationPlan:
    """Concrete register-control actions for hardware warpgroups."""

    setmaxnreg_enabled: bool
    warpgroup_warps: int
    domains: tuple[RegisterWarpgroupDomain, ...]


def build_register_plan(
    allocation: PhysicalWarpAllocation,
    logical: ProgramScheduleCandidate,
    warpgroup_warps: int = 4,
) -> RegisterAllocationPlan:
    """Materialize one concrete setmaxnreg action per hardware warpgroup.

    Pure-memory groups donate registers and groups containing compute receive
    the remaining register-file budget.  A plan is emitted only when both
    roles exist; homogeneous or mixed-only partitions do not benefit from
    dynamic register redistribution.
    """

    if warpgroup_warps <= 0:
        raise ValueError("warpgroup_warps must be positive")
    if not allocation.setmaxnreg_enabled:
        return RegisterAllocationPlan(False, warpgroup_warps, ())

    memory_units = {HardwareUnit.LOAD_STORE, HardwareUnit.TMA, HardwareUnit.TMEM}
    nodes_by_group = {
        group_id: tuple(
            node
            for node, assigned_group in logical.partition.node_groups.items()
            if assigned_group == group_id
        )
        for group_id in range(logical.partition.num_groups)
    }
    allocation_group_ids = {group.group_id for group in allocation.groups}
    if set(nodes_by_group) != allocation_group_ids:
        raise ValueError("logical and physical groups must match")

    producer_groups = {
        group_id
        for group_id, nodes in nodes_by_group.items()
        if nodes and all(node.unit in memory_units for node in nodes)
    }
    consumer_groups = allocation_group_ids - producer_groups
    if not producer_groups or not consumer_groups:
        return RegisterAllocationPlan(False, warpgroup_warps, ())

    # Match the established Hopper policy. Counts accepted by setmaxnreg are
    # multiples of eight; 64512 leaves the architectural bookkeeping reserve
    # used by the existing two-role lowering.
    register_file_budget = 64512
    producer_register_count = 24
    maximum_consumer_register_count = 240
    threads_per_warp = allocation.effective_threads // allocation.total_warps
    producer_threads = sum(
        group.warp_count * threads_per_warp
        for group in allocation.groups
        if group.group_id in producer_groups
    )
    consumer_threads = allocation.effective_threads - producer_threads
    consumer_register_count = min(
        maximum_consumer_register_count,
        (
            (register_file_budget - producer_register_count * producer_threads)
            // consumer_threads
            // 8
        )
        * 8,
    )
    if consumer_register_count <= producer_register_count:
        return RegisterAllocationPlan(False, warpgroup_warps, ())

    domains = []
    for group in allocation.groups:
        if (
            group.first_warp % warpgroup_warps
            or group.warp_count % warpgroup_warps
        ):
            raise ValueError(
                "setmaxnreg-enabled groups must start and end on "
                "hardware warpgroup boundaries"
            )
        for first_warp in range(
            group.first_warp,
            group.warp_stop,
            warpgroup_warps,
        ):
            domains.append(
                RegisterWarpgroupDomain(
                    group.group_id,
                    first_warp,
                    warpgroup_warps,
                    (
                        producer_register_count
                        if group.group_id in producer_groups
                        else consumer_register_count
                    ),
                    group.group_id in consumer_groups,
                )
            )
    return RegisterAllocationPlan(True, warpgroup_warps, tuple(domains))


def _target_architecture_number(architecture: str) -> int | None:
    digits = "".join(character for character in architecture if character.isdigit())
    return int(digits) if digits else None


def target_execution_profile(
    target: Target,
    enable_setmaxnreg: bool | None = None,
) -> TargetExecutionProfile:
    """Resolve target-specific physical scheduling policy.

    Hopper warp-specialized groups are warpgroup-granular. Blackwell defaults
    to warp-granular groups because tcgen05.mma/cp have single-thread issue
    semantics. Enabling setmaxnreg restores warpgroup alignment because every
    warp in a hardware warpgroup must execute the same setmaxnreg instruction.
    """

    warp_size = int(target.attrs.get("thread_warp_size", 32))
    maximum = target.attrs.get("max_threads_per_block", None)
    if maximum is None:
        maximum = target.attrs.get("max_num_threads", None)
    if maximum is None:
        if target.kind.name != "cuda":
            raise ValueError(
                "target must provide max_threads_per_block or max_num_threads"
            )
        maximum = 1024
    architecture = str(target.attrs.get("arch", ""))
    architecture_number = _target_architecture_number(architecture)
    if target.kind.name != "cuda" or architecture_number is None:
        family = CUDAArchitectureFamily.GENERIC
    elif architecture_number >= 100:
        family = CUDAArchitectureFamily.BLACKWELL
    elif architecture_number >= 90:
        family = CUDAArchitectureFamily.HOPPER
    else:
        family = CUDAArchitectureFamily.GENERIC

    if enable_setmaxnreg is None:
        setmaxnreg_enabled = family == CUDAArchitectureFamily.HOPPER
    else:
        setmaxnreg_enabled = bool(enable_setmaxnreg)
    warpgroup_warps = 4
    if family == CUDAArchitectureFamily.HOPPER or setmaxnreg_enabled:
        logical_group_warp_multiple = warpgroup_warps
    else:
        logical_group_warp_multiple = 1
    return TargetExecutionProfile(
        architecture,
        family,
        warp_size,
        int(maximum),
        warpgroup_warps,
        logical_group_warp_multiple,
        setmaxnreg_enabled,
    )


def _operation_warp_requirement(
    node: DataflowNode,
    profile: TargetExecutionProfile,
) -> _OperationWarpRequirement:
    """Resolve issue/collective requirements independently of group policy."""

    if node.instruction_kind == InstructionKind.WGMMA:
        return _OperationWarpRequirement(
            profile.warpgroup_warps,
            profile.warpgroup_warps,
            "wgmma is a warpgroup collective",
        )
    if node.instruction_kind in {
        InstructionKind.TCGEN05_MMA,
        InstructionKind.TCGEN05_CP,
        InstructionKind.TMA,
    }:
        return _OperationWarpRequirement(
            1,
            1,
            f"{node.instruction_kind.value} has single-thread issue semantics",
        )
    if node.instruction_kind in {
        InstructionKind.TCGEN05_LD,
        InstructionKind.TCGEN05_ST,
    }:
        return _OperationWarpRequirement(
            1,
            1,
            f"{node.instruction_kind.value} is a warp collective",
        )
    return _OperationWarpRequirement(
        1,
        1,
        "generic operation is warp-granular",
    )


def _infer_group_warp_requirements(
    candidate: ProgramScheduleCandidate,
    profile: TargetExecutionProfile,
) -> tuple[_GroupWarpRequirement, ...]:
    """Combine operation semantics with target group granularity.

    A group containing only single-thread asynchronous issue operations does
    not gain useful data parallelism from another copy of its scheduling
    granule.  Keep such a producer at one target group granule.  Cooperative
    memory and compute groups remain scalable; their final upper bound is the
    CTA thread limit and is explored by the physical allocator.
    """

    is_specialized = candidate.partition.num_groups > 1
    group_policy_multiple = (
        profile.logical_group_warp_multiple if is_specialized else 1
    )

    requirements = []
    for group_id in range(candidate.partition.num_groups):
        nodes = tuple(
            node
            for node, assigned_group in candidate.partition.node_groups.items()
            if assigned_group == group_id
        )
        operation_requirements = tuple(
            _operation_warp_requirement(node, profile) for node in nodes
        )
        warp_multiple = group_policy_multiple
        for operation_requirement in operation_requirements:
            warp_multiple = math.lcm(
                warp_multiple, operation_requirement.warp_multiple
            )
        minimum_warps = max(
            (item.minimum_warps for item in operation_requirements),
            default=1,
        )
        minimum_warps = max(minimum_warps, group_policy_multiple)
        minimum_warps = (
            (minimum_warps + warp_multiple - 1) // warp_multiple
        ) * warp_multiple
        fixed_single_thread_issue = bool(nodes) and all(
            node.instruction_kind
            in {InstructionKind.TMA, InstructionKind.TCGEN05_CP}
            for node in nodes
        )
        maximum_warps = (
            minimum_warps
            if fixed_single_thread_issue
            else profile.max_threads_per_block // profile.warp_size
        )
        reasons = [
            item.reason
            for item in operation_requirements
            if item.minimum_warps > 1 or item.warp_multiple > 1
        ]
        if group_policy_multiple > 1:
            reasons.append(
                f"{profile.family.value} specialization uses "
                f"{group_policy_multiple}-warp group alignment"
            )
        if fixed_single_thread_issue:
            reasons.append(
                "pure single-thread async issue group is fixed to one "
                "target scheduling granule"
            )
        requirements.append(
            _GroupWarpRequirement(
                group_id,
                minimum_warps,
                warp_multiple,
                maximum_warps,
                not fixed_single_thread_issue,
                "; ".join(dict.fromkeys(reasons))
                or "single-warp-granular logical group",
            )
        )
    return tuple(requirements)


def _aligned_values(requirement: _GroupWarpRequirement, maximum: int) -> range:
    legal_maximum = min(maximum, requirement.maximum_warps)
    if not requirement.scalable:
        legal_maximum = min(legal_maximum, requirement.minimum_warps)
    return range(
        requirement.minimum_warps,
        legal_maximum + 1,
        requirement.warp_multiple,
    )


def _enumerate_exact_total(
    requirements: tuple[_GroupWarpRequirement, ...],
    total_warps: int,
) -> Iterator[tuple[int, ...]]:
    selected: list[int] = []

    def visit(index: int, remaining: int) -> Iterator[tuple[int, ...]]:
        if index == len(requirements):
            if remaining == 0:
                yield tuple(selected)
            return
        requirement = requirements[index]
        minimum_rest = sum(
            item.minimum_warps for item in requirements[index + 1 :]
        )
        maximum = remaining - minimum_rest
        for count in _aligned_values(requirement, maximum):
            selected.append(count)
            yield from visit(index + 1, remaining - count)
            selected.pop()

    yield from visit(0, total_warps)


def _enumerate_feasible_totals(
    requirements: tuple[_GroupWarpRequirement, ...],
    minimum_total: int,
    maximum_total: int,
) -> Iterator[tuple[int, ...]]:
    """Enumerate every legal assignment, ordered by increasing CTA width."""

    for total_warps in range(minimum_total, maximum_total + 1):
        assignments = tuple(_enumerate_exact_total(requirements, total_warps))
        if assignments:
            yield from assignments


def _make_allocation(
    original_threads: int,
    warp_counts: tuple[int, ...],
    profile: TargetExecutionProfile,
) -> PhysicalWarpAllocation:
    total_warps = sum(warp_counts)
    effective_threads = total_warps * profile.warp_size
    if effective_threads > profile.max_threads_per_block:
        raise ValueError(
            f"allocation needs {effective_threads} threads, exceeding target limit "
            f"{profile.max_threads_per_block}"
        )
    first_warp = 0
    groups = []
    for group_id, count in enumerate(warp_counts):
        groups.append(GroupWarpAllocation(group_id, count, first_warp))
        first_warp += count
    specialized = len(groups) > 1
    return PhysicalWarpAllocation(
        original_threads,
        effective_threads,
        tuple(groups),
        profile.setmaxnreg_enabled and specialized,
    )


def enumerate_physical_warp_allocations(
    candidate: ProgramScheduleCandidate,
    original_threads: int,
    profile: TargetExecutionProfile,
    mode: WarpAllocationMode = WarpAllocationMode.EXPAND_CTA,
    producer_warps: int = 1,
    explicit_group_warps: Mapping[int, int] | None = None,
) -> Iterator[PhysicalWarpAllocation]:
    """Enumerate hard-legal physical warp allocations.

    ``original_threads`` is a baseline, not a maximum. In EXPAND_CTA mode the
    compute-capable groups enumerate every legal width from that baseline to
    the target block limit, while non-compute producers add their requested
    width. Pure single-thread issue groups cannot be widened. PRESERVE_CTA
    instead shares the original budget among all groups. EXPLICIT validates
    one caller-provided assignment.
    """

    if original_threads <= 0 or original_threads % profile.warp_size:
        raise ValueError("original_threads must be a positive whole number of warps")
    if producer_warps <= 0:
        raise ValueError("producer_warps must be positive")
    original_warps = original_threads // profile.warp_size
    requirements = _infer_group_warp_requirements(candidate, profile)

    if mode == WarpAllocationMode.EXPLICIT:
        if explicit_group_warps is None:
            raise ValueError("EXPLICIT mode requires explicit_group_warps")
        if set(explicit_group_warps) != set(range(len(requirements))):
            raise ValueError("explicit allocation must cover every logical group")
        counts = tuple(explicit_group_warps[index] for index in range(len(requirements)))
        for count, requirement in zip(counts, requirements):
            if (
                count < requirement.minimum_warps
                or count > requirement.maximum_warps
                or count % requirement.warp_multiple
            ):
                raise ValueError(
                    f"group {requirement.group_id} violates warp requirement: "
                    f"{requirement.reason}"
                )
        yield _make_allocation(original_threads, counts, profile)
        return

    if explicit_group_warps is not None:
        raise ValueError("explicit_group_warps is only valid in EXPLICIT mode")

    if mode == WarpAllocationMode.PRESERVE_CTA:
        for counts in _enumerate_exact_total(requirements, original_warps):
            yield _make_allocation(original_threads, counts, profile)
        return

    if mode != WarpAllocationMode.EXPAND_CTA:
        raise ValueError(f"unsupported allocation mode: {mode}")

    memory_units = {HardwareUnit.LOAD_STORE, HardwareUnit.TMA, HardwareUnit.TMEM}
    compute_group_ids = tuple(
        group_id
        for group_id in range(candidate.partition.num_groups)
        if any(
            node.unit not in memory_units
            and candidate.partition.node_groups[node] == group_id
            for node in candidate.partition.node_groups
        )
    )
    if not compute_group_ids:
        for counts in _enumerate_exact_total(requirements, original_warps):
            yield _make_allocation(original_threads, counts, profile)
        return

    producer_counts = {}
    for requirement in requirements:
        if requirement.group_id in compute_group_ids:
            continue
        requested_warps = max(producer_warps, requirement.minimum_warps)
        producer_count = (
            (requested_warps + requirement.warp_multiple - 1)
            // requirement.warp_multiple
        ) * requirement.warp_multiple
        if producer_count > requirement.maximum_warps:
            raise ValueError(
                f"group {requirement.group_id} cannot be expanded to "
                f"{producer_count} warps: {requirement.reason}"
            )
        producer_counts[requirement.group_id] = producer_count
    maximum_warps = profile.max_threads_per_block // profile.warp_size
    maximum_compute_warps = maximum_warps - sum(producer_counts.values())
    compute_requirements = tuple(requirements[index] for index in compute_group_ids)
    for compute_counts in _enumerate_feasible_totals(
        compute_requirements,
        original_warps,
        maximum_compute_warps,
    ):
        counts_by_group = dict(producer_counts)
        counts_by_group.update(zip(compute_group_ids, compute_counts))
        counts = tuple(counts_by_group[index] for index in range(len(requirements)))
        if sum(counts) * profile.warp_size > profile.max_threads_per_block:
            continue
        yield _make_allocation(original_threads, counts, profile)


def enumerate_warp_allocations(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramScheduleCandidate,
    target: Target,
    mode: WarpAllocationMode = WarpAllocationMode.EXPAND_CTA,
    producer_warps: int = 1,
    explicit_group_warps: Mapping[int, int] | None = None,
    enable_setmaxnreg: bool | None = None,
) -> Iterator[PhysicalWarpAllocation]:
    """Use analysis/target constraints to enumerate physical allocations."""

    if analysis.kernel_threads is None:
        raise ValueError(
            "the analyzed PrimFunc has no static threadIdx extent; "
            "pass original_threads to enumerate_physical_warp_allocations"
        )
    profile = target_execution_profile(target, enable_setmaxnreg)
    fragment_width_edges = []
    for dependency in candidate.cross_group_dependencies:
        if dependency.buffer_id is None or "RAW" not in dependency.dependency_kinds:
            continue
        if analysis.buffer_for_id(dependency.buffer_id).scope != "local.fragment":
            continue
        fragment_width_edges.append(
            (dependency.producer_group, dependency.consumer_group)
        )

    for allocation in enumerate_physical_warp_allocations(
        candidate,
        original_threads=analysis.kernel_threads,
        profile=profile,
        mode=mode,
        producer_warps=producer_warps,
        explicit_group_warps=explicit_group_warps,
    ):
        group_warps = {
            group.group_id: group.warp_count for group in allocation.groups
        }
        mismatch = next(
            (
                (producer_group, consumer_group)
                for producer_group, consumer_group in fragment_width_edges
                if group_warps[producer_group] != group_warps[consumer_group]
            ),
            None,
        )
        if mismatch is None:
            yield allocation
            continue
        if mode == WarpAllocationMode.EXPLICIT:
            producer_group, consumer_group = mismatch
            raise ValueError(
                "groups connected by a local.fragment RAW dependency must "
                "have the same warp count until fragment resharding is "
                f"implemented; groups {producer_group} and {consumer_group} "
                f"have {group_warps[producer_group]} and "
                f"{group_warps[consumer_group]} warps"
            )


def validate_physical_warp_allocation(
    candidate: ProgramScheduleCandidate,
    allocation: PhysicalWarpAllocation,
    profile: TargetExecutionProfile,
) -> None:
    """Check target, interval, and per-group hard constraints."""

    requirements = _infer_group_warp_requirements(candidate, profile)
    if len(allocation.groups) != len(requirements):
        raise ValueError("physical allocation must cover every logical group")
    expected_first_warp = 0
    for group, requirement in zip(allocation.groups, requirements):
        if group.group_id != requirement.group_id:
            raise ValueError("physical groups must use contiguous logical IDs")
        if group.first_warp != expected_first_warp:
            raise ValueError("physical warp intervals must be contiguous")
        if (
            group.warp_count < requirement.minimum_warps
            or group.warp_count > requirement.maximum_warps
            or group.warp_count % requirement.warp_multiple
        ):
            raise ValueError(f"group {group.group_id} violates its warp requirement")
        expected_first_warp = group.warp_stop
    expected_threads = expected_first_warp * profile.warp_size
    if allocation.effective_threads != expected_threads:
        raise ValueError("effective thread count does not match group allocations")
    if expected_threads > profile.max_threads_per_block:
        raise ValueError("physical allocation exceeds the target block limit")
