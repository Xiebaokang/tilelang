"""Adapters that validate and apply selected pipeline schedules to IR."""

from __future__ import annotations

from tvm import IRModule, tirx
from tvm.target import Target

from ..analysis.program import ProgramDataflowAnalysis
from ..physical.buffer_planning import BufferCommunicationKind
from ..physical.synchronization import SynchronizationChannelKind
from ..physical.validation import validate_realized_schedule
from ..scheduling.warp_specialization import RegionDependencyScope
from ..search.realization import ProgramRealizedScheduleCandidate


def _resolve_global_symbol(
    mod: IRModule,
    analysis: ProgramDataflowAnalysis,
    requested_symbol: str | None,
) -> str:
    matches = [
        global_var.name_hint
        for global_var, func in mod.functions.items()
        if isinstance(func, tirx.PrimFunc) and func.same_as(analysis.prim_func)
    ]
    if len(matches) != 1:
        raise ValueError(
            "analysis.prim_func must be the exact PrimFunc in the input IRModule"
        )
    actual_symbol = matches[0]
    if requested_symbol is not None and requested_symbol != actual_symbol:
        raise ValueError(
            f"requested function {requested_symbol!r} does not match "
            f"analysis function {actual_symbol!r}"
        )
    return actual_symbol


def build_ir_plan(
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramRealizedScheduleCandidate,
) -> dict[str, int | tuple[int, ...]]:
    """Encode a fully realized program candidate using dense analysis IDs."""

    if not isinstance(candidate, ProgramRealizedScheduleCandidate):
        raise TypeError(
            "IR rewrite planning requires a ProgramRealizedScheduleCandidate"
        )
    logical = candidate.logical
    if set(logical.partition.node_groups) != set(analysis.nodes):
        raise ValueError("candidate partition does not match program analysis")
    schedules = {
        schedule.region_id: schedule
        for schedule in logical.region_schedules
    }
    if set(schedules) != {region.region_id for region in analysis.regions}:
        raise ValueError("candidate region schedules do not match program analysis")
    operations = tuple(
        sorted(analysis.operations, key=lambda operation: operation.operation_id)
    )
    if tuple(operation.operation_id for operation in operations) != tuple(
        range(len(operations))
    ):
        raise ValueError("program operation IDs must be dense")

    stages = []
    local_orders = []
    for operation in operations:
        group_id = logical.partition.node_groups[operation.node]
        schedule = schedules[operation.region_id]
        stages.append(schedule.stages.get(operation.node, -1))
        local_orders.append(schedule.group_orders[group_id][operation.node])

    scope_codes = {
        RegionDependencyScope.ONCE: 0,
        RegionDependencyScope.PER_ITERATION: 1,
        RegionDependencyScope.REGION_BOUNDARY: 2,
        None: -1,
    }
    kind_codes = {
        SynchronizationChannelKind.FORWARD_DEPENDENCY: 0,
        SynchronizationChannelKind.BUFFER_REUSE: 1,
    }
    communication_codes = {
        BufferCommunicationKind.EXTERNAL: 0,
        BufferCommunicationKind.PRIVATE: 1,
        BufferCommunicationKind.SHARED: 2,
        BufferCommunicationKind.TMEM: 3,
        BufferCommunicationKind.MATERIALIZED_SHARED: 4,
    }
    dependency_kind_bits = {
        "RAW": 1,
        "WAR": 2,
        "WAW": 4,
        "REUSE": 8,
    }
    channels = candidate.channels
    buffers = tuple(
        sorted(candidate.buffers.buffers, key=lambda buffer: buffer.buffer_id)
    )
    if tuple(buffer.buffer_id for buffer in buffers) != tuple(range(len(buffers))):
        raise ValueError("realized buffer IDs must be dense")
    allocations = tuple(
        sorted(
            candidate.warp_allocation.groups,
            key=lambda group: group.group_id,
        )
    )
    domains = candidate.register_allocation.domains
    return {
        "operation_groups": tuple(
            logical.partition.node_groups[operation.node]
            for operation in operations
        ),
        "operation_regions": tuple(
            operation.region_id for operation in operations
        ),
        "operation_stages": tuple(stages),
        "operation_local_orders": tuple(local_orders),
        "region_num_stages": tuple(
            schedules[region.region_id].num_stages for region in analysis.regions
        ),
        "group_first_warps": tuple(group.first_warp for group in allocations),
        "group_warp_counts": tuple(group.warp_count for group in allocations),
        "effective_threads": candidate.warp_allocation.effective_threads,
        "buffer_versions": tuple(buffer.version_count for buffer in buffers),
        "buffer_communications": tuple(
            communication_codes[buffer.communication] for buffer in buffers
        ),
        "buffer_requires_new_allocation": tuple(
            int(buffer.requires_new_allocation) for buffer in buffers
        ),
        "sync_producers": tuple(
            channel.producer_operation_id for channel in channels
        ),
        "sync_consumers": tuple(
            channel.consumer_operation_id for channel in channels
        ),
        "sync_producer_groups": tuple(
            channel.producer_group for channel in channels
        ),
        "sync_consumer_groups": tuple(
            channel.consumer_group for channel in channels
        ),
        "sync_producer_regions": tuple(
            channel.producer_region_id
            if channel.producer_region_id is not None
            else -1
            for channel in channels
        ),
        "sync_consumer_regions": tuple(
            channel.consumer_region_id
            if channel.consumer_region_id is not None
            else -1
            for channel in channels
        ),
        "sync_iteration_distances": tuple(
            channel.iteration_distance for channel in channels
        ),
        "sync_effective_stage_distances": tuple(
            channel.effective_stage_distance
            if channel.effective_stage_distance is not None
            else -1
            for channel in channels
        ),
        "sync_scopes": tuple(
            scope_codes[channel.dependency_scope] for channel in channels
        ),
        "sync_kinds": tuple(
            kind_codes[channel.kind] for channel in channels
        ),
        "sync_buffer_ids": tuple(
            channel.buffer_id if channel.buffer_id is not None else -1
            for channel in channels
        ),
        "sync_dependency_masks": tuple(
            sum(dependency_kind_bits[kind] for kind in channel.dependency_kinds)
            for channel in channels
        ),
        "register_domain_groups": tuple(domain.group_id for domain in domains),
        "register_domain_first_warps": tuple(
            domain.first_warp for domain in domains
        ),
        "register_domain_warp_counts": tuple(
            domain.warp_count for domain in domains
        ),
        "register_domain_register_counts": tuple(
            domain.register_count for domain in domains
        ),
        "register_domain_is_increase": tuple(
            int(domain.is_increase) for domain in domains
        ),
        "setmaxnreg_enabled": int(
            candidate.register_allocation.setmaxnreg_enabled
        ),
    }


def _attach_program_rewrite_plan(
    func: tirx.PrimFunc,
    plan: dict[str, int | tuple[int, ...]],
) -> tirx.PrimFunc:
    """Attach the versioned C++ lowering contract without rewriting bodies."""

    rewritten = func.with_attr("tl.program_schedule.version", 3)
    rewritten = rewritten.with_attr(
        "tl.program_schedule.effective_threads", plan["effective_threads"]
    )
    rewritten = rewritten.with_attr(
        "tl.program_schedule.setmaxnreg_enabled",
        plan["setmaxnreg_enabled"],
    )
    for suffix, values in plan.items():
        if suffix in {"effective_threads", "setmaxnreg_enabled"}:
            continue
        rewritten = rewritten.with_attr(
            f"tl.program_schedule.{suffix}", list(values)
        )
    return rewritten


def apply_schedule_to_ir(
    mod: IRModule,
    analysis: ProgramDataflowAnalysis,
    candidate: ProgramRealizedScheduleCandidate,
    target: Target,
    *,
    global_symbol: str | None = None,
) -> IRModule:
    """Attach a complete program schedule for decision-driven C++ lowering.

    This function deliberately does not perform structural warp-specialization
    rewriting in Python. The ``tl.program_schedule.*`` attributes form the
    checked hand-off contract for the C++ TIRX pass described in the package
    README.
    """

    if not isinstance(mod, IRModule):
        raise TypeError(f"mod must be IRModule, got {type(mod).__name__}")
    validate_realized_schedule(analysis, candidate, target)
    symbol = _resolve_global_symbol(mod, analysis, global_symbol)
    plan = build_ir_plan(analysis, candidate)
    rewritten = mod.clone()
    global_var = rewritten.get_global_var(symbol)
    rewritten.update_func(
        global_var,
        _attach_program_rewrite_plan(rewritten[global_var], plan),
    )
    return rewritten
