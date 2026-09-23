"""Candidate-dependent buffer versioning and communication realization."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import product

from tvm import tirx
from tvm.target import Target

from ..analysis.core import DataflowNode
from ..analysis.extractor import RegionOverlap, analyze_region_overlap
from ..analysis.program import (
    BufferAccessKind,
    BufferDescriptor,
    BufferRegionAccess,
    ProgramDataflowAnalysis,
    ProgramRegion,
    ProgramRegionKind,
)
from ..scheduling.order import ProgramRegionSchedule
from ..scheduling.warp_specialization import (
    ProgramScheduleCandidate,
    RegionDependencyScope,
)


class BufferCommunicationKind(str, Enum):
    """How a logical buffer is made visible to its assigned warp groups."""

    EXTERNAL = "external"
    PRIVATE = "private"
    SHARED = "shared"
    TMEM = "tmem"
    MATERIALIZED_SHARED = "materialized_shared"


class BufferPlanningError(ValueError):
    """A logical candidate cannot be given a proven buffer realization."""


class BufferVersionPolicy(str, Enum):
    """How legal physical buffer-version counts enter the search space."""

    MINIMUM = "minimum"
    ENUMERATE = "enumerate"


RegionOverlapCache = dict[tuple[int, int, int, int], RegionOverlap]


def _cached_region_overlap(
    cache: RegionOverlapCache,
    region: ProgramRegion,
    earlier: BufferRegionAccess,
    later: BufferRegionAccess,
    iteration_delta: int,
) -> RegionOverlap:
    """Reuse symbolic overlap proofs throughout one schedule search."""

    key = (
        region.region_id,
        id(earlier),
        id(later),
        iteration_delta,
    )
    result = cache.get(key)
    if result is None:
        if region.loop is None:
            raise BufferPlanningError("pipeline region is missing its loop")
        result = analyze_region_overlap(
            earlier,
            later,
            region.loop.loop_var,
            iteration_delta=iteration_delta,
        )
        cache[key] = result
    return result


def _supports_physical_multiversion(scope: str) -> bool:
    """Whether current IR/layout lowering supports an added version axis."""

    # Fragment layouts describe their logical rank explicitly. Adding a
    # leading physical axis before LayoutInference currently makes tile-op
    # accesses and the fragment layout disagree on rank. Keep those buffers
    # single-versioned; reuse synchronization remains available when legal.
    return scope != "local.fragment"


@dataclass(frozen=True)
class BufferReuseConstraint:
    """A consumer-group acknowledgement before a producer reuses a version."""

    buffer_id: int
    accessor_operation_id: int
    writer_operation_id: int
    accessor_group: int
    writer_group: int
    iteration_distance: int
    effective_stage_distance: int
    region_id: int | None = None


@dataclass(frozen=True)
class RealizedBuffer:
    """One buffer after version count and communication storage are fixed."""

    buffer_id: int
    name: str
    original_scope: str
    communication: BufferCommunicationKind
    writer_operation_ids: tuple[int, ...]
    version_count: int
    requires_new_allocation: bool
    total_shared_bytes: int
    additional_shared_bytes: int


@dataclass(frozen=True)
class BufferRealizationPlan:
    """Hard-feasibility buffer plan for one complete logical candidate."""

    buffers: tuple[RealizedBuffer, ...]
    reuse_constraints: tuple[BufferReuseConstraint, ...]
    total_shared_bytes: int
    additional_shared_bytes: int
    target_shared_memory_limit: int | None


def _target_shared_memory_limit(target: Target) -> int | None:
    for key in (
        "max_shared_memory_per_block",
        "max_shared_memory_per_block_optin",
        "max_shared_memory",
    ):
        value = target.attrs.get(key, None)
        if value is not None:
            return int(value)
    return None


def _target_arch_number(target: Target) -> int | None:
    arch = str(target.attrs.get("arch", ""))
    digits = "".join(character for character in arch if character.isdigit())
    return int(digits) if digits else None


def _operations_by_id(
    analysis: ProgramDataflowAnalysis,
) -> dict[int, DataflowNode]:
    return {
        operation.operation_id: operation.node
        for operation in analysis.operations
    }


def _buffer_users(
    analysis: ProgramDataflowAnalysis,
    buffer_id: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    readers = tuple(
        access.operation_id
        for access in analysis.operation_accesses
        if buffer_id in access.read_buffer_ids
    )
    writers = tuple(
        access.operation_id
        for access in analysis.operation_accesses
        if buffer_id in access.write_buffer_ids
    )
    return readers, writers


def _communication_kind(
    descriptor: BufferDescriptor,
    group_ids: tuple[int, ...],
    target: Target,
) -> BufferCommunicationKind:
    scope = descriptor.scope
    crosses_groups = len(group_ids) > 1
    if scope in ("global", ""):
        return BufferCommunicationKind.EXTERNAL
    if "tmem" in scope:
        arch = _target_arch_number(target)
        if crosses_groups and arch is not None and arch < 100:
            raise BufferPlanningError(
                f"buffer {descriptor.name} uses TMEM across groups on {target}"
            )
        return BufferCommunicationKind.TMEM
    if scope.startswith("shared"):
        return BufferCommunicationKind.SHARED
    if scope == "local.fragment":
        return BufferCommunicationKind.PRIVATE
    if not crosses_groups:
        return BufferCommunicationKind.PRIVATE
    if descriptor.nbytes is None:
        raise BufferPlanningError(
            f"cross-group private buffer {descriptor.name} has dynamic size"
        )
    return BufferCommunicationKind.MATERIALIZED_SHARED


def _fragment_handoff_bytes(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramScheduleCandidate,
) -> dict[int, int]:
    """Return conservative shared storage for cross-group fragment RAW edges."""

    result: dict[int, int] = {}
    for dependency in candidate.cross_group_dependencies:
        if dependency.buffer_id is None or "RAW" not in dependency.dependency_kinds:
            continue
        descriptor = analysis.buffer_for_id(dependency.buffer_id)
        if descriptor.scope != "local.fragment":
            continue
        if descriptor.nbytes is None:
            raise BufferPlanningError(
                f"fragment handoff {descriptor.name} has dynamic size"
            )
        slots = 1
        if dependency.scope == RegionDependencyScope.PER_ITERATION:
            schedule = next(
                schedule
                for schedule in candidate.region_schedules
                if schedule.region_id == dependency.producer_region
            )
            slots = max(
                1,
                schedule.stages[dependency.consumer]
                - schedule.stages[dependency.producer],
            )
        result[descriptor.buffer_id] = result.get(descriptor.buffer_id, 0) + (
            descriptor.nbytes * slots
        )
    return result


def _fragment_handoff_reuse_constraints(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramScheduleCandidate,
    schedules: Mapping[int, ProgramRegionSchedule],
) -> tuple[BufferReuseConstraint, ...]:
    """Protect minimal fragment-handoff slots with reverse backpressure."""

    operation_ids = {
        operation.node: operation.operation_id for operation in analysis.operations
    }
    constraints = []
    for dependency in candidate.cross_group_dependencies:
        if (
            dependency.scope != RegionDependencyScope.PER_ITERATION
            or dependency.buffer_id is None
            or "RAW" not in dependency.dependency_kinds
            or analysis.buffer_for_id(dependency.buffer_id).scope
            != "local.fragment"
        ):
            continue
        schedule = schedules[dependency.producer_region]
        slots = max(
            1,
            schedule.stages[dependency.consumer]
            - schedule.stages[dependency.producer],
        )
        constraints.append(
            BufferReuseConstraint(
                buffer_id=dependency.buffer_id,
                accessor_operation_id=operation_ids[dependency.consumer],
                writer_operation_id=operation_ids[dependency.producer],
                accessor_group=dependency.consumer_group,
                writer_group=dependency.producer_group,
                iteration_distance=slots,
                effective_stage_distance=(
                    slots
                    + schedule.stages[dependency.producer]
                    - schedule.stages[dependency.consumer]
                ),
                region_id=dependency.producer_region,
            )
        )
    return tuple(constraints)


def _validate_fragment_import_lifetimes(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramScheduleCandidate,
    schedules: Mapping[int, ProgramRegionSchedule],
) -> None:
    """Prove that a next-iteration import cannot clobber a live fragment."""

    nodes_by_id = _operations_by_id(analysis)
    node_groups = candidate.partition.node_groups
    seen_imports: set[tuple[int, int, int, int]] = set()
    import_stages: dict[tuple[int, int, int], int] = {}
    for dependency in candidate.cross_group_dependencies:
        if (
            dependency.scope != RegionDependencyScope.PER_ITERATION
            or dependency.buffer_id is None
            or "RAW" not in dependency.dependency_kinds
        ):
            continue
        descriptor = analysis.buffer_for_id(dependency.buffer_id)
        if descriptor.scope != "local.fragment":
            continue
        import_node = dependency.consumer
        import_operation_id = analysis.operation_for(import_node).operation_id
        import_key = (
            dependency.producer_region,
            dependency.buffer_id,
            dependency.consumer_group,
            import_operation_id,
        )
        if import_key in seen_imports:
            continue
        seen_imports.add(import_key)
        region = analysis.regions[dependency.consumer_region]
        schedule = schedules[region.region_id]
        import_stage = schedule.stages[import_node]
        stage_key = (
            dependency.consumer_region,
            dependency.buffer_id,
            dependency.consumer_group,
        )
        previous_stage = import_stages.setdefault(stage_key, import_stage)
        if previous_stage != import_stage:
            raise BufferPlanningError(
                f"fragment handoff imports for {descriptor.name} need "
                "consumer-side multiversioning across stages"
            )
        import_order = schedule.group_orders[dependency.consumer_group][
            import_node
        ]
        for access in _program_region_accesses_for_buffer(
            analysis, region, dependency.buffer_id
        ):
            accessor = nodes_by_id[access.operation_id]
            if node_groups[accessor] != dependency.consumer_group:
                continue
            effective_distance = 1 + import_stage - schedule.stages[accessor]
            if effective_distance < 0:
                raise BufferPlanningError(
                    f"fragment handoff import for {descriptor.name} needs "
                    "another consumer-side version before "
                    f"{accessor.name}"
                )
            if (
                effective_distance == 0
                and accessor != import_node
                and schedule.group_orders[dependency.consumer_group][accessor]
                >= import_order
            ):
                raise BufferPlanningError(
                    f"fragment handoff import for {descriptor.name} "
                    "overwrites the previous iteration before "
                    f"{accessor.name}"
                )


def _normalize_version_policy(
    policy: BufferVersionPolicy | str,
) -> BufferVersionPolicy:
    try:
        return BufferVersionPolicy(policy)
    except ValueError as error:
        raise ValueError(f"unsupported buffer version policy: {policy}") from error


def _program_schedules_by_region(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramScheduleCandidate,
) -> dict[int, ProgramRegionSchedule]:
    if set(candidate.partition.node_groups) != set(analysis.nodes):
        raise BufferPlanningError(
            "candidate partition must cover every analyzed operation"
        )
    schedules = {
        schedule.region_id: schedule for schedule in candidate.region_schedules
    }
    if len(schedules) != len(candidate.region_schedules):
        raise BufferPlanningError("candidate contains duplicate region schedules")
    if set(schedules) != {region.region_id for region in analysis.regions}:
        raise BufferPlanningError(
            "candidate must provide one schedule per program region"
        )
    expected_group_ids = set(range(candidate.partition.num_groups))
    for region in analysis.regions:
        schedule = schedules[region.region_id]
        region_nodes = analysis.nodes_for_region(region)
        expected_stage_nodes = (
            set(region_nodes)
            if region.kind == ProgramRegionKind.PIPELINE
            else set()
        )
        if set(schedule.stages) != expected_stage_nodes:
            raise BufferPlanningError(
                f"region {region.region_id} stages cover the wrong operations"
            )
        if set(schedule.group_orders) != expected_group_ids:
            raise BufferPlanningError(
                f"region {region.region_id} does not cover every logical group"
            )
        for group_id, order in schedule.group_orders.items():
            expected_nodes = {
                node
                for node in region_nodes
                if candidate.partition.node_groups[node] == group_id
            }
            if set(order) != expected_nodes:
                raise BufferPlanningError(
                    f"region {region.region_id}, group {group_id} order covers "
                    "the wrong operations"
                )
            if set(order.values()) != set(range(len(order))):
                raise BufferPlanningError(
                    f"region {region.region_id}, group {group_id} order is not "
                    "contiguous"
                )
    return schedules


def _program_region_accesses_for_buffer(
    analysis: ProgramDataflowAnalysis,
    region: ProgramRegion,
    buffer_id: int,
) -> tuple[BufferRegionAccess, ...]:
    return tuple(
        access
        for access in analysis.region_accesses
        if access.buffer_id == buffer_id
        and access.operation_id in region.operation_ids
    )


def _required_program_region_versions(
    analysis: ProgramDataflowAnalysis,
    region: ProgramRegion,
    descriptor: BufferDescriptor,
    schedule: ProgramRegionSchedule,
    candidate: ProgramScheduleCandidate,
    overlap_cache: RegionOverlapCache,
) -> int:
    """Return an order-independent safe bound for one pipeline region."""

    if region.kind != ProgramRegionKind.PIPELINE or region.loop is None:
        return 1
    accesses = _program_region_accesses_for_buffer(
        analysis, region, descriptor.buffer_id
    )
    writers = tuple(
        access for access in accesses if access.kind == BufferAccessKind.WRITE
    )
    if not accesses or not writers:
        return 1
    nodes_by_id = _operations_by_id(analysis)
    accessor_stages = [
        schedule.stages[nodes_by_id[access.operation_id]]
        for access in accesses
    ]
    guaranteed_upper_bound = max(accessor_stages) - min(accessor_stages) + 1
    for versions in range(1, guaranteed_upper_bound + 1):
        safe = True
        for access in accesses:
            access_stage = schedule.stages[nodes_by_id[access.operation_id]]
            for writer in writers:
                if descriptor.scope == "local.fragment":
                    accessor_node = nodes_by_id[access.operation_id]
                    writer_node = nodes_by_id[writer.operation_id]
                    if (
                        candidate.partition.node_groups[accessor_node]
                        != candidate.partition.node_groups[writer_node]
                    ):
                        continue
                if _cached_region_overlap(
                    overlap_cache,
                    region,
                    access,
                    writer,
                    versions,
                ) == RegionOverlap.DISJOINT:
                    continue
                writer_stage = schedule.stages[
                    nodes_by_id[writer.operation_id]
                ]
                if versions + writer_stage - access_stage <= 0:
                    safe = False
                    break
            if not safe:
                break
        if safe:
            return versions
    raise BufferPlanningError(
        f"cannot derive a safe version count for buffer {descriptor.name} "
        f"in region {region.region_id}"
    )


def _program_reuse_constraints_for_version(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramScheduleCandidate,
    schedules: Mapping[int, ProgramRegionSchedule],
    buffer: RealizedBuffer,
    version_count: int,
    overlap_cache: RegionOverlapCache,
) -> tuple[BufferReuseConstraint, ...]:
    """Validate one count in every pipeline region and build backpressure."""

    if version_count < 1:
        raise BufferPlanningError("buffer version count must be positive")
    if (
        not buffer.writer_operation_ids
        or buffer.communication == BufferCommunicationKind.EXTERNAL
    ):
        return ()

    nodes_by_id = _operations_by_id(analysis)
    node_groups = candidate.partition.node_groups
    constraints: list[BufferReuseConstraint] = []
    keys: set[tuple[int, int, int, int]] = set()
    for region in analysis.regions:
        if region.kind != ProgramRegionKind.PIPELINE or region.loop is None:
            continue
        accesses = _program_region_accesses_for_buffer(
            analysis, region, buffer.buffer_id
        )
        writers = tuple(
            access for access in accesses if access.kind == BufferAccessKind.WRITE
        )
        if not writers:
            continue
        schedule = schedules[region.region_id]
        for writer_region in writers:
            writer_id = writer_region.operation_id
            writer_node = nodes_by_id[writer_id]
            writer_group = node_groups[writer_node]
            conflicts_by_group: dict[int, list[int]] = {}
            for access in accesses:
                if _cached_region_overlap(
                    overlap_cache,
                    region,
                    access,
                    writer_region,
                    version_count,
                ) == RegionOverlap.DISJOINT:
                    continue
                accessor_id = access.operation_id
                accessor_node = nodes_by_id[accessor_id]
                accessor_group = node_groups[accessor_node]
                if (
                    buffer.original_scope == "local.fragment"
                    and accessor_group != writer_group
                ):
                    # Each group owns a distinct physical fragment instance.
                    # Cross-group value transfer and lifetime are carried by
                    # an edge-local shared handoff, not by reusing this private
                    # allocation across groups.
                    continue
                effective_distance = (
                    version_count
                    + schedule.stages[writer_node]
                    - schedule.stages[accessor_node]
                )
                if effective_distance < 0:
                    raise BufferPlanningError(
                        f"buffer {buffer.name} cannot be reused safely with "
                        f"{version_count} versions in region {region.region_id}"
                    )
                if accessor_group == writer_group:
                    if (
                        effective_distance == 0
                        and schedule.group_orders[accessor_group][accessor_node]
                        >= schedule.group_orders[writer_group][writer_node]
                    ):
                        raise BufferPlanningError(
                            f"buffer {buffer.name} needs another version for "
                            f"region {region.region_id} local order"
                        )
                    continue
                conflicts_by_group.setdefault(accessor_group, []).append(
                    accessor_id
                )

            for accessor_group, operation_ids in conflicts_by_group.items():
                accessor_id = max(
                    operation_ids,
                    key=lambda operation_id: (
                        schedule.stages[nodes_by_id[operation_id]],
                        schedule.group_orders[accessor_group][
                            nodes_by_id[operation_id]
                        ],
                    ),
                )
                key = (
                    region.region_id,
                    accessor_id,
                    writer_id,
                    buffer.buffer_id,
                )
                if key in keys:
                    continue
                keys.add(key)
                accessor_node = nodes_by_id[accessor_id]
                constraints.append(
                    BufferReuseConstraint(
                        buffer_id=buffer.buffer_id,
                        accessor_operation_id=accessor_id,
                        writer_operation_id=writer_id,
                        accessor_group=accessor_group,
                        writer_group=writer_group,
                        iteration_distance=version_count,
                        effective_stage_distance=(
                            version_count
                            + schedule.stages[writer_node]
                            - schedule.stages[accessor_node]
                        ),
                        region_id=region.region_id,
                    )
                )
    return tuple(constraints)


def enumerate_buffer_plans(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramScheduleCandidate,
    target: Target,
    *,
    max_num_versions: int | None = None,
    version_policy: BufferVersionPolicy | str = BufferVersionPolicy.MINIMUM,
    overlap_cache: RegionOverlapCache | None = None,
) -> Iterator[BufferRealizationPlan]:
    """Enumerate legal whole-kernel buffer plans for a program schedule."""

    if not analysis.buffers or not analysis.operation_accesses:
        raise BufferPlanningError(
            "analysis does not contain exact buffer access metadata"
        )
    if not analysis.region_accesses:
        raise BufferPlanningError(
            "analysis does not contain exact buffer region metadata"
        )
    if max_num_versions is not None and max_num_versions < 1:
        raise ValueError("max_num_versions must be positive or None")
    policy = _normalize_version_policy(version_policy)
    if overlap_cache is None:
        overlap_cache = {}
    schedules = _program_schedules_by_region(analysis, candidate)
    _validate_fragment_import_lifetimes(analysis, candidate, schedules)
    nodes_by_id = _operations_by_id(analysis)
    operation_regions = {
        operation.operation_id: operation.region_id
        for operation in analysis.operations
    }

    base_buffers: list[RealizedBuffer] = []
    version_spaces: list[tuple[int, ...]] = []
    for descriptor in analysis.buffers:
        readers, writers = _buffer_users(analysis, descriptor.buffer_id)
        operation_ids = tuple(sorted(set(readers) | set(writers)))
        group_ids = tuple(
            sorted(
                {
                    candidate.partition.node_groups[nodes_by_id[operation_id]]
                    for operation_id in operation_ids
                }
            )
        )
        region_ids = tuple(
            sorted({operation_regions[operation_id] for operation_id in operation_ids})
        )
        communication = _communication_kind(descriptor, group_ids, target)
        if communication in (
            BufferCommunicationKind.SHARED,
            BufferCommunicationKind.MATERIALIZED_SHARED,
        ) and descriptor.nbytes is None:
            raise BufferPlanningError(
                f"shared buffer {descriptor.name} has dynamic size"
            )
        base = RealizedBuffer(
            buffer_id=descriptor.buffer_id,
            name=descriptor.name,
            original_scope=descriptor.scope,
            communication=communication,
            writer_operation_ids=writers,
            version_count=1,
            requires_new_allocation=(
                communication == BufferCommunicationKind.MATERIALIZED_SHARED
            ),
            total_shared_bytes=0,
            additional_shared_bytes=0,
        )
        base_buffers.append(base)

        if (
            communication == BufferCommunicationKind.EXTERNAL
            or not writers
            or not readers
        ):
            version_spaces.append((1,))
            continue
        pipeline_region_ids = {
            region_id
            for region_id in region_ids
            if analysis.regions[region_id].kind == ProgramRegionKind.PIPELINE
        }
        safe_count = max(
            (
                _required_program_region_versions(
                    analysis,
                    analysis.regions[region_id],
                    descriptor,
                    schedules[region_id],
                    candidate,
                    overlap_cache,
                )
                for region_id in pipeline_region_ids
            ),
            default=1,
        )
        automatic_upper_bound = max(
            safe_count,
            max(
                (
                    schedules[region_id].num_stages or 1
                    for region_id in pipeline_region_ids
                ),
                default=1,
            ),
        )
        upper_bound = (
            max_num_versions
            if max_num_versions is not None
            else automatic_upper_bound
        )
        if not _supports_physical_multiversion(descriptor.scope):
            upper_bound = 1
        legal = []
        for version_count in range(1, upper_bound + 1):
            if version_count > 1 and len(pipeline_region_ids) != 1:
                continue
            try:
                _program_reuse_constraints_for_version(
                    analysis,
                    candidate,
                    schedules,
                    base,
                    version_count,
                    overlap_cache,
                )
            except BufferPlanningError:
                continue
            legal.append(version_count)
        if not legal:
            detail = (
                " because accesses span multiple pipeline regions"
                if len(pipeline_region_ids) > 1
                else ""
            )
            raise BufferPlanningError(
                f"buffer {descriptor.name} has no legal version count up to "
                f"{upper_bound}{detail}"
            )
        version_spaces.append(
            (legal[0],) if policy == BufferVersionPolicy.MINIMUM else tuple(legal)
        )

    descriptors = {item.buffer_id: item for item in analysis.buffers}
    fragment_handoff_bytes = _fragment_handoff_bytes(analysis, candidate)
    limit = _target_shared_memory_limit(target)
    for version_counts in product(*version_spaces):
        realized_buffers = []
        reuse_constraints = list(
            _fragment_handoff_reuse_constraints(
                analysis, candidate, schedules
            )
        )
        for base, version_count in zip(base_buffers, version_counts):
            descriptor = descriptors[base.buffer_id]
            if base.communication == BufferCommunicationKind.SHARED:
                total_shared = (descriptor.nbytes or 0) * version_count
                additional_shared = (descriptor.nbytes or 0) * (
                    version_count - 1
                )
            elif (
                base.communication
                == BufferCommunicationKind.MATERIALIZED_SHARED
            ):
                total_shared = (descriptor.nbytes or 0) * version_count
                additional_shared = total_shared
            else:
                total_shared = 0
                additional_shared = 0
            handoff_bytes = fragment_handoff_bytes.get(base.buffer_id, 0)
            total_shared += handoff_bytes
            additional_shared += handoff_bytes
            realized = RealizedBuffer(
                buffer_id=base.buffer_id,
                name=base.name,
                original_scope=base.original_scope,
                communication=base.communication,
                writer_operation_ids=base.writer_operation_ids,
                version_count=version_count,
                requires_new_allocation=(
                    version_count > 1
                    or base.communication
                    == BufferCommunicationKind.MATERIALIZED_SHARED
                ),
                total_shared_bytes=total_shared,
                additional_shared_bytes=additional_shared,
            )
            realized_buffers.append(realized)
            reuse_constraints.extend(
                _program_reuse_constraints_for_version(
                    analysis,
                    candidate,
                    schedules,
                    realized,
                    version_count,
                    overlap_cache,
                )
            )

        total_shared = sum(item.total_shared_bytes for item in realized_buffers)
        additional_shared = sum(
            item.additional_shared_bytes for item in realized_buffers
        )
        if total_shared and limit is None:
            raise BufferPlanningError(
                "target must provide max_shared_memory_per_block to prove "
                "shared-memory feasibility"
            )
        if limit is not None and total_shared > limit:
            continue
        yield BufferRealizationPlan(
            buffers=tuple(realized_buffers),
            reuse_constraints=tuple(reuse_constraints),
            total_shared_bytes=total_shared,
            additional_shared_bytes=additional_shared,
            target_shared_memory_limit=limit,
        )


def validate_buffer_plan(plan: BufferRealizationPlan) -> None:
    """Validate internal accounting and reuse-distance invariants."""

    if tuple(buffer.buffer_id for buffer in plan.buffers) != tuple(
        range(len(plan.buffers))
    ):
        raise ValueError("realized buffers must use dense IDs")
    if any(buffer.version_count < 1 for buffer in plan.buffers):
        raise ValueError("every buffer needs at least one version")
    if plan.total_shared_bytes != sum(
        buffer.total_shared_bytes for buffer in plan.buffers
    ):
        raise ValueError("total shared-memory accounting is inconsistent")
    if plan.additional_shared_bytes != sum(
        buffer.additional_shared_bytes for buffer in plan.buffers
    ):
        raise ValueError("additional shared-memory accounting is inconsistent")
    if (
        plan.target_shared_memory_limit is not None
        and plan.total_shared_bytes > plan.target_shared_memory_limit
    ):
        raise ValueError("buffer plan exceeds the target shared-memory limit")
    for constraint in plan.reuse_constraints:
        if constraint.accessor_group == constraint.writer_group:
            raise ValueError("reuse synchronization must cross logical groups")
        if constraint.iteration_distance < 1:
            raise ValueError("reuse synchronization must cross an iteration")
        if constraint.effective_stage_distance < 0:
            raise ValueError("reuse synchronization distance cannot be negative")
