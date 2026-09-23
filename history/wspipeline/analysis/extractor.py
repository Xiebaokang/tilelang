"""Extract a pipeline dataflow graph and exact access metadata from TIR."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from tvm import arith, ir, tirx
from tvm.tirx.stmt_functor import post_order_visit, substitute

from .core import (
    DataflowEdge,
    DataflowNode,
    ExecutionKind,
    HardwareUnit,
    InstructionKind,
    OperationProfile,
)
from .program import (
    BufferAccessKind,
    BufferDescriptor,
    BufferRegionAccess,
    ProgramDataflowAnalysis,
    ProgramOperation,
    ProgramRegion,
    ProgramRegionKind,
)


@dataclass(frozen=True)
class _StatementPathStep:
    """One typed step from a PrimFunc body to an operation statement."""

    kind: str
    index: int = 0


class RegionOverlap(str, Enum):
    """A conservative three-state result for two symbolic regions."""

    DISJOINT = "disjoint"
    OVERLAP = "overlap"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class _OperationClassification:
    unit: HardwareUnit
    execution_kind: ExecutionKind
    instruction_kind: InstructionKind = InstructionKind.GENERIC


def _op_name(call: tirx.Call) -> str:
    return call.op.name if isinstance(call.op, ir.Op) else str(call.op)


def _buffer_name(buffer: tirx.Buffer) -> str:
    return str(buffer.name)


@dataclass(frozen=True)
class _RawRegionAccess:
    buffer: tirx.Buffer = field(compare=False, hash=False)
    kind: BufferAccessKind
    region: tuple[ir.Range, ...] = field(compare=False, hash=False)
    is_exact: bool


def _accessed_buffers(
    accesses: list[_RawRegionAccess],
) -> tuple[set[tirx.Buffer], set[tirx.Buffer]]:
    reads = {
        access.buffer
        for access in accesses
        if access.kind == BufferAccessKind.READ
    }
    writes = {
        access.buffer
        for access in accesses
        if access.kind == BufferAccessKind.WRITE
    }
    return reads, writes


def _whole_buffer_region(buffer: tirx.Buffer) -> tuple[ir.Range, ...]:
    return tuple(
        ir.Range.from_min_extent(tirx.IntImm("int32", 0), extent)
        for extent in buffer.shape
    )


def _parallel_domains(statement) -> dict[tirx.Var, arith.IntSet]:
    domains: dict[tirx.Var, arith.IntSet] = {}

    def visit(node) -> None:
        if not isinstance(node, tirx.For) or node.kind != tirx.ForKind.PARALLEL:
            return
        maximum = node.min + node.extent - 1
        domains[node.loop_var] = arith.IntervalSet(node.min, maximum)

    post_order_visit(statement, visit)
    return domains


def _interval_for_index(
    index: tirx.PrimExpr,
    domains: dict[tirx.Var, arith.IntSet],
) -> tuple[ir.Range | None, bool]:
    analyzer = arith.Analyzer()

    def dense_over_domains(expression: tirx.PrimExpr) -> bool:
        if not domains:
            return True
        variables = list(domains)
        coefficients = arith.detect_linear_equation(expression, variables)
        if len(coefficients) != len(variables) + 1:
            return False
        nonzero_coefficients = [
            coefficient
            for coefficient in coefficients[:-1]
            if not analyzer.can_prove_equal(coefficient, 0)
        ]
        return len(nonzero_coefficients) <= 1 and all(
            analyzer.can_prove_equal(coefficient, 1)
            or analyzer.can_prove_equal(coefficient, -1)
            for coefficient in nonzero_coefficients
        )

    if isinstance(index, tirx.Ramp):
        stride = index.stride
        if not isinstance(stride, tirx.IntImm):
            return None, False
        stride_value = int(stride)
        if stride_value == 0:
            return None, False
        base_set = analyzer.int_set(index.base, domains)
        if not isinstance(base_set, arith.IntervalSet):
            return None, False
        lane_span = (int(index.lanes) - 1) * stride_value
        minimum = base_set.min_value + min(0, lane_span)
        maximum = base_set.max_value + max(0, lane_span)
        return (
            ir.Range.from_min_extent(
                analyzer.simplify(minimum),
                analyzer.simplify(maximum - minimum + 1),
            ),
            abs(stride_value) == 1 and dense_over_domains(index.base),
        )

    index_set = analyzer.int_set(index, domains)
    if not isinstance(index_set, arith.IntervalSet):
        return None, False
    return (
        ir.Range.from_min_extent(
            analyzer.simplify(index_set.min_value),
            analyzer.simplify(index_set.max_value - index_set.min_value + 1),
        ),
        dense_over_domains(index),
    )


def _scalar_statement_region_accesses(statement) -> list[_RawRegionAccess]:
    domains = _parallel_domains(statement)
    accesses: list[_RawRegionAccess] = []

    def append(buffer: tirx.Buffer, indices, kind: BufferAccessKind) -> None:
        if len(indices) != len(buffer.shape):
            accesses.append(
                _RawRegionAccess(buffer, kind, _whole_buffer_region(buffer), False)
            )
            return
        region: list[ir.Range] = []
        exact = True
        for index in indices:
            interval, dimension_exact = _interval_for_index(index, domains)
            if interval is None:
                accesses.append(
                    _RawRegionAccess(
                        buffer, kind, _whole_buffer_region(buffer), False
                    )
                )
                return
            region.append(interval)
            exact = exact and dimension_exact
        accesses.append(_RawRegionAccess(buffer, kind, tuple(region), exact))

    def visit(node) -> None:
        if isinstance(node, tirx.BufferLoad):
            append(node.buffer, node.indices, BufferAccessKind.READ)
        elif isinstance(node, tirx.BufferStore):
            append(node.buffer, node.indices, BufferAccessKind.WRITE)

    post_order_visit(statement, visit)
    return accesses


def _tile_call_region_accesses(call: tirx.Call) -> list[_RawRegionAccess]:
    accesses: list[_RawRegionAccess] = []

    def visit(node) -> None:
        if not isinstance(node, tirx.Call) or _op_name(node) != "tl.tileop.region":
            return
        if len(node.args) < 2 or not isinstance(node.args[0], tirx.BufferLoad):
            return
        load = node.args[0]
        access_mask = int(node.args[1])
        extents = tuple(node.args[2:])
        exact = len(extents) == len(load.indices) == len(load.buffer.shape)
        if exact:
            region = tuple(
                ir.Range.from_min_extent(index, extent)
                for index, extent in zip(load.indices, extents)
            )
        else:
            region = _whole_buffer_region(load.buffer)
        if access_mask & 1:
            accesses.append(
                _RawRegionAccess(load.buffer, BufferAccessKind.READ, region, exact)
            )
        if access_mask & 2:
            accesses.append(
                _RawRegionAccess(load.buffer, BufferAccessKind.WRITE, region, exact)
            )

    post_order_visit(call, visit)
    return accesses


def _shift_expr(
    expression: tirx.PrimExpr,
    loop_var: tirx.Var,
    iteration_delta: int,
) -> tirx.PrimExpr:
    if iteration_delta == 0:
        return expression
    return substitute(
        expression,
        {loop_var: loop_var + iteration_delta},
    )


def analyze_region_overlap(
    earlier: BufferRegionAccess,
    later: BufferRegionAccess,
    loop_var: tirx.Var,
    iteration_delta: int = 0,
) -> RegionOverlap:
    """Compare ``earlier(i)`` with ``later(i + iteration_delta)``.

    ``UNKNOWN`` is intentionally distinct from overlap. Correctness callers
    must conservatively treat both OVERLAP and UNKNOWN as a possible conflict.
    """

    if earlier.buffer_id != later.buffer_id:
        return RegionOverlap.DISJOINT
    if iteration_delta < 0:
        raise ValueError("iteration_delta cannot be negative")
    if len(earlier.region) != len(later.region):
        return RegionOverlap.UNKNOWN
    analyzer = arith.Analyzer()
    all_dimensions_overlap = True
    for earlier_range, later_range in zip(earlier.region, later.region):
        earlier_min = earlier_range.min
        earlier_end = analyzer.simplify(earlier_min + earlier_range.extent)
        later_min = analyzer.simplify(
            _shift_expr(later_range.min, loop_var, iteration_delta)
        )
        later_extent = analyzer.simplify(
            _shift_expr(later_range.extent, loop_var, iteration_delta)
        )
        later_end = analyzer.simplify(later_min + later_extent)
        if analyzer.can_prove(earlier_end <= later_min) or analyzer.can_prove(
            later_end <= earlier_min
        ):
            return RegionOverlap.DISJOINT
        if not (
            analyzer.can_prove(earlier_min < later_end)
            and analyzer.can_prove(later_min < earlier_end)
        ):
            all_dimensions_overlap = False
    return (
        RegionOverlap.OVERLAP
        if all_dimensions_overlap and earlier.is_exact and later.is_exact
        else RegionOverlap.UNKNOWN
    )


def region_covers(
    outer: BufferRegionAccess,
    inner: BufferRegionAccess,
    loop_var: tirx.Var,
    inner_iteration_delta: int = 0,
) -> bool:
    """Return whether one interval region provably covers another."""

    if (
        outer.buffer_id != inner.buffer_id
        or len(outer.region) != len(inner.region)
        or not outer.is_exact
    ):
        return False
    analyzer = arith.Analyzer()
    for outer_range, inner_range in zip(outer.region, inner.region):
        outer_min = outer_range.min
        outer_end = analyzer.simplify(outer_min + outer_range.extent)
        inner_min = analyzer.simplify(
            _shift_expr(inner_range.min, loop_var, inner_iteration_delta)
        )
        inner_extent = analyzer.simplify(
            _shift_expr(inner_range.extent, loop_var, inner_iteration_delta)
        )
        inner_end = analyzer.simplify(inner_min + inner_extent)
        if not (
            analyzer.can_prove(outer_min <= inner_min)
            and analyzer.can_prove(inner_end <= outer_end)
        ):
            return False
    return True


def _build_direct_memory_dependencies(
    operation_count: int,
    region_accesses: tuple[BufferRegionAccess, ...],
    loop_var: tirx.Var,
    add_edge: Callable[[int, int, str, int], None],
) -> None:
    """Build direct same-iteration RAW/WAR/WAW precedence relations.

    Reaching writes are killed only when a later exact region provably covers
    them. Reads remain live until a later write provably covers their region.
    Unknown and partial coverage therefore stays conservative without adding
    every transitive pairwise conflict.
    """

    accesses_by_operation = {
        operation_id: tuple(
            access
            for access in region_accesses
            if access.operation_id == operation_id
        )
        for operation_id in range(operation_count)
    }
    reaching_writes: dict[int, list[BufferRegionAccess]] = {}
    live_reads: dict[int, list[BufferRegionAccess]] = {}

    def may_overlap(
        earlier: BufferRegionAccess, later: BufferRegionAccess
    ) -> bool:
        return (
            analyze_region_overlap(earlier, later, loop_var)
            != RegionOverlap.DISJOINT
        )

    for operation_id in range(operation_count):
        accesses = accesses_by_operation[operation_id]
        reads = tuple(
            access for access in accesses if access.kind == BufferAccessKind.READ
        )
        writes = tuple(
            access for access in accesses if access.kind == BufferAccessKind.WRITE
        )

        for read in reads:
            for writer in reversed(reaching_writes.get(read.buffer_id, ())):
                if may_overlap(writer, read):
                    add_edge(
                        writer.operation_id,
                        operation_id,
                        "RAW",
                        read.buffer_id,
                    )
                if region_covers(writer, read, loop_var):
                    break

        for write in writes:
            for reader in live_reads.get(write.buffer_id, ()):
                if may_overlap(reader, write):
                    add_edge(
                        reader.operation_id,
                        operation_id,
                        "WAR",
                        write.buffer_id,
                    )
            for writer in reversed(reaching_writes.get(write.buffer_id, ())):
                if may_overlap(writer, write):
                    add_edge(
                        writer.operation_id,
                        operation_id,
                        "WAW",
                        write.buffer_id,
                    )
                if region_covers(writer, write, loop_var):
                    break

        for read in reads:
            live_reads.setdefault(read.buffer_id, []).append(read)
        for write in writes:
            reaching_writes[write.buffer_id] = [
                writer
                for writer in reaching_writes.get(write.buffer_id, ())
                if not region_covers(write, writer, loop_var)
            ]
            reaching_writes[write.buffer_id].append(write)
            live_reads[write.buffer_id] = [
                reader
                for reader in live_reads.get(write.buffer_id, ())
                if not region_covers(write, reader, loop_var)
            ]


def _remove_transitive_ordering_edges(
    edges: list[DataflowEdge],
) -> list[DataflowEdge]:
    """Remove only ordering-only precedence implied through another operation.

    Parallel edges between the same endpoints may describe different buffers;
    they do not make each other transitive. Loop-carried edges are retained
    because reducing a weighted iteration-distance graph requires a separate
    proof. RAW edges are also retained: an alternative path through a reader
    can prove ordering, but it does not guarantee that every future group
    partition receives the producer's buffer value.
    """

    successors: dict[DataflowNode, set[DataflowNode]] = {}
    for edge in edges:
        if edge.iteration_distance == 0:
            successors.setdefault(edge.producer, set()).add(edge.consumer)

    endpoint_is_transitive: dict[tuple[DataflowNode, DataflowNode], bool] = {}
    for edge in edges:
        if edge.iteration_distance != 0:
            continue
        endpoint = (edge.producer, edge.consumer)
        if endpoint in endpoint_is_transitive:
            continue
        worklist = [
            successor
            for successor in successors.get(edge.producer, ())
            if successor != edge.consumer
        ]
        visited = set(worklist)
        transitive = False
        while worklist and not transitive:
            node = worklist.pop()
            for successor in successors.get(node, ()):
                if successor == edge.consumer:
                    transitive = True
                    break
                if successor not in visited:
                    visited.add(successor)
                    worklist.append(successor)
        endpoint_is_transitive[endpoint] = transitive

    return [
        edge
        for edge in edges
        if edge.iteration_distance != 0
        or "RAW" in edge.dependency_kinds
        or not endpoint_is_transitive[(edge.producer, edge.consumer)]
    ]


_SFU_OPS = {
    "tirx.exp",
    "tirx.exp2",
    "tirx.log",
    "tirx.log2",
    "tirx.sin",
    "tirx.cos",
    "tirx.tanh",
    "tirx.sqrt",
}

_ALU_EXPR_TYPES = (
    tirx.Add,
    tirx.Sub,
    tirx.Mul,
    tirx.Div,
    tirx.FloorDiv,
    tirx.FloorMod,
    tirx.Min,
    tirx.Max,
    tirx.EQ,
    tirx.NE,
    tirx.LT,
    tirx.LE,
    tirx.GT,
    tirx.GE,
    tirx.And,
    tirx.Or,
    tirx.Not,
    tirx.Select,
)


# -----------------------------------------------------------------------------
# Static per-operation profiling; profiles never become schedulable nodes
# -----------------------------------------------------------------------------


def _static_extent_product(extents) -> int | None:
    product = 1
    for extent in extents:
        if not isinstance(extent, tirx.IntImm):
            return None
        product *= int(extent)
    return product


def _buffer_elements(buffer: tirx.Buffer) -> int | None:
    return _static_extent_product(buffer.shape)


def _buffer_nbytes(buffer: tirx.Buffer) -> int | None:
    elements = _buffer_elements(buffer)
    if elements is None:
        return None
    bits = int(buffer.dtype.bits) * int(buffer.dtype.lanes)
    return (elements * bits + 7) // 8


def _buffer_scopes(buffers: set[tirx.Buffer]) -> tuple[str, ...]:
    return tuple(sorted({buffer.scope() for buffer in buffers}))


def _logical_transfer_size(
    reads: set[tirx.Buffer], writes: set[tirx.Buffer]
) -> int:
    sizes = [
        size
        for buffer in reads | writes
        if (size := _buffer_nbytes(buffer)) is not None
    ]
    return min(sizes, default=0)


def _logical_operation_elements(
    reads: set[tirx.Buffer], writes: set[tirx.Buffer]
) -> int:
    output_sizes = [
        size
        for buffer in writes
        if (size := _buffer_elements(buffer)) is not None
    ]
    if output_sizes:
        return max(output_sizes)
    input_sizes = [
        size
        for buffer in reads
        if (size := _buffer_elements(buffer)) is not None
    ]
    return max(input_sizes, default=1)


def _parallel_iterations(statement) -> int:
    extents = []

    def visit(node) -> None:
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL:
            extents.append(node.extent)

    post_order_visit(statement, visit)
    return _static_extent_product(extents) or 1


def _profile_statement(
    statement,
    reads: set[tirx.Buffer],
    writes: set[tirx.Buffer],
) -> OperationProfile:
    scalar_alu_ops = 0
    scalar_sfu_ops = 0
    scalar_cast_ops = 0

    def visit(node) -> None:
        nonlocal scalar_alu_ops, scalar_sfu_ops, scalar_cast_ops
        if isinstance(node, _ALU_EXPR_TYPES):
            scalar_alu_ops += 1
        elif isinstance(node, tirx.Cast):
            scalar_cast_ops += 1
        elif isinstance(node, tirx.Call):
            name = _op_name(node).lower()
            if name in _SFU_OPS:
                scalar_sfu_ops += 1
            elif name == "tirx.if_then_else":
                scalar_alu_ops += 1

    post_order_visit(statement, visit)
    logical_elements = _parallel_iterations(statement)
    return OperationProfile(
        logical_elements=logical_elements,
        alu_ops=scalar_alu_ops * logical_elements,
        sfu_ops=scalar_sfu_ops * logical_elements,
        cast_ops=scalar_cast_ops * logical_elements,
        source_scopes=_buffer_scopes(reads),
        destination_scopes=_buffer_scopes(writes),
    )


def _profile_call(
    call: tirx.Call,
    classification: _OperationClassification,
    reads: set[tirx.Buffer],
    writes: set[tirx.Buffer],
) -> OperationProfile:
    name = _op_name(call).lower()
    logical_elements = _logical_operation_elements(reads, writes)
    if "reduce" in name:
        logical_elements = max(
            (
                size
                for buffer in reads | writes
                if (size := _buffer_elements(buffer)) is not None
            ),
            default=logical_elements,
        )
    is_memory_operation = classification.unit in {
        HardwareUnit.LOAD_STORE,
        HardwareUnit.TMA,
        HardwareUnit.TMEM,
    }
    is_mma = classification.unit == HardwareUnit.MMA
    return OperationProfile(
        logical_elements=logical_elements,
        alu_ops=(
            logical_elements if not is_memory_operation and not is_mma else 0
        ),
        mma_ops=int(is_mma),
        memory_bytes=(
            _logical_transfer_size(reads, writes) if is_memory_operation else 0
        ),
        source_scopes=_buffer_scopes(reads),
        destination_scopes=_buffer_scopes(writes),
    )


def _single_buffer(buffers: set[tirx.Buffer]) -> tirx.Buffer | None:
    return next(iter(buffers)) if len(buffers) == 1 else None


def _is_shared_scope(scope: str) -> bool:
    return scope.startswith("shared") and scope != "shared.tmem"


def _classify_copy(
    name: str,
    reads: set[tirx.Buffer],
    writes: set[tirx.Buffer],
) -> _OperationClassification:
    src = _single_buffer(reads)
    dst = _single_buffer(writes)
    src_scope = src.scope() if src is not None else ""
    dst_scope = dst.scope() if dst is not None else ""

    # TMEM is the execution unit only for tcgen05.ld/st between tensor memory
    # and a fragment. An MMA that happens to access TMEM is classified as MMA.
    if src_scope == "shared.tmem" and dst_scope == "local.fragment":
        return _OperationClassification(
            HardwareUnit.TMEM,
            ExecutionKind.ASYNC_MEMORY,
            InstructionKind.TCGEN05_LD,
        )
    if src_scope == "local.fragment" and dst_scope == "shared.tmem":
        return _OperationClassification(
            HardwareUnit.TMEM,
            ExecutionKind.ASYNC_MEMORY,
            InstructionKind.TCGEN05_ST,
        )

    if name == "tl.tileop.tma_copy":
        return _OperationClassification(
            HardwareUnit.TMA,
            ExecutionKind.ASYNC_MEMORY,
            InstructionKind.TMA,
        )
    if name == "tl.tileop.async_copy":
        return _OperationClassification(
            HardwareUnit.LOAD_STORE,
            ExecutionKind.ASYNC_MEMORY,
        )

    # Preserve the analysis-level TMA prediction for a regular global/shared
    # tile copy. Exact backend eligibility still depends on target and layout.
    if "global" in {src_scope, dst_scope} and (
        _is_shared_scope(src_scope) or _is_shared_scope(dst_scope)
    ):
        return _OperationClassification(
            HardwareUnit.TMA,
            ExecutionKind.ASYNC_MEMORY,
            InstructionKind.TMA,
        )

    return _OperationClassification(
        HardwareUnit.LOAD_STORE, ExecutionKind.SYNC_MEMORY
    )


def _classify_call(
    call: tirx.Call,
    reads: set[tirx.Buffer] | None = None,
    writes: set[tirx.Buffer] | None = None,
) -> _OperationClassification:
    name = _op_name(call).lower()
    if name in {
        "tl.tileop.copy",
        "tl.tileop.async_copy",
        "tl.tileop.tma_copy",
    }:
        if reads is None or writes is None:
            reads, writes = _accessed_buffers(_tile_call_region_accesses(call))
        return _classify_copy(name, reads, writes)
    if "tcgen05_gemm" in name or "tcgen05_mma" in name:
        return _OperationClassification(
            HardwareUnit.MMA,
            ExecutionKind.ASYNC_COMPUTE,
            InstructionKind.TCGEN05_MMA,
        )
    if "tcgen05_cp" in name:
        return _OperationClassification(
            HardwareUnit.TMEM,
            ExecutionKind.ASYNC_MEMORY,
            InstructionKind.TCGEN05_CP,
        )
    if "tcgen05_ld" in name:
        return _OperationClassification(
            HardwareUnit.TMEM,
            ExecutionKind.ASYNC_MEMORY,
            InstructionKind.TCGEN05_LD,
        )
    if "tcgen05_st" in name:
        return _OperationClassification(
            HardwareUnit.TMEM,
            ExecutionKind.ASYNC_MEMORY,
            InstructionKind.TCGEN05_ST,
        )
    if "wgmma" in name:
        return _OperationClassification(
            HardwareUnit.MMA,
            ExecutionKind.ASYNC_COMPUTE,
            InstructionKind.WGMMA,
        )
    if "gemm" in name or "mma" in name:
        return _OperationClassification(
            HardwareUnit.MMA,
            ExecutionKind.SYNC_COMPUTE,
            InstructionKind.GENERIC_MMA,
        )
    if "tma" in name:
        return _OperationClassification(
            HardwareUnit.TMA,
            ExecutionKind.ASYNC_MEMORY,
            InstructionKind.TMA,
        )
    if name in _SFU_OPS:
        return _OperationClassification(HardwareUnit.SFU, ExecutionKind.SYNC_COMPUTE)
    return _OperationClassification(
        HardwareUnit.ALU, ExecutionKind.SYNC_COMPUTE
    )


def _classify_statement(statement) -> _OperationClassification:
    classifications: list[_OperationClassification] = []

    def visit(node) -> None:
        if isinstance(node, tirx.Call):
            classifications.append(_classify_call(node))

    post_order_visit(statement, visit)
    for unit in (
        HardwareUnit.TMEM,
        HardwareUnit.MMA,
        HardwareUnit.TMA,
        HardwareUnit.SFU,
    ):
        for classification in classifications:
            if classification.unit == unit:
                return classification
    return _OperationClassification(
        HardwareUnit.ALU, ExecutionKind.SYNC_COMPUTE
    )


@dataclass(frozen=True)
class _CollectedOperation:
    name: str
    classification: _OperationClassification
    reads: set[tirx.Buffer]
    writes: set[tirx.Buffer]
    profile: OperationProfile
    path: tuple[_StatementPathStep, ...]
    statement: object = field(compare=False, hash=False, repr=False)
    region_accesses: tuple[_RawRegionAccess, ...] = field(
        compare=False, hash=False, repr=False
    )


class _OperationCollector:
    def __init__(self) -> None:
        self.operations: list[_CollectedOperation] = []
        self.counts: dict[str, int] = {}

    def _append(
        self,
        base_name: str,
        classification: _OperationClassification,
        reads: set[tirx.Buffer],
        writes: set[tirx.Buffer],
        profile: OperationProfile,
        path: tuple[_StatementPathStep, ...],
        statement,
        region_accesses: list[_RawRegionAccess],
    ) -> None:
        index = self.counts.get(base_name, 0)
        self.counts[base_name] = index + 1
        name = base_name if index == 0 else f"{base_name}_{index + 1}"
        self.operations.append(
            _CollectedOperation(
                name,
                classification,
                reads,
                writes,
                profile,
                path,
                statement,
                tuple(region_accesses),
            )
        )

    @staticmethod
    def _operation_name(
        base_name: str, reads: set[tirx.Buffer], writes: set[tirx.Buffer]
    ) -> str:
        inputs = "_".join(sorted(_buffer_name(buffer) for buffer in reads))
        outputs = "_".join(sorted(_buffer_name(buffer) for buffer in writes))
        if base_name == "copy" and inputs and outputs:
            return f"{base_name}_{inputs}_to_{outputs}"
        if outputs:
            return f"{base_name}_{outputs}"
        if inputs:
            return f"{base_name}_{inputs}"
        return base_name

    def visit(
        self,
        statement,
        path: tuple[_StatementPathStep, ...] = (),
    ) -> None:
        if isinstance(statement, tirx.SeqStmt):
            for index, child in enumerate(statement.seq):
                self.visit(child, path + (_StatementPathStep("seq", index),))
            return
        if isinstance(statement, tirx.For):
            if statement.kind == tirx.ForKind.PARALLEL:
                region_accesses = _scalar_statement_region_accesses(statement)
                reads, writes = _accessed_buffers(region_accesses)
                classification = _classify_statement(statement)
                self._append(
                    self._operation_name("parallel", reads, writes),
                    classification,
                    reads,
                    writes,
                    _profile_statement(statement, reads, writes),
                    path,
                    statement,
                    region_accesses,
                )
            else:
                self.visit(
                    statement.body,
                    path
                    + (_StatementPathStep("for_body", int(statement.kind)),),
                )
            return
        if isinstance(statement, tirx.SBlockRealize):
            self.visit(
                statement.block.body,
                path + (_StatementPathStep("block_realize_body"),),
            )
            return
        if isinstance(statement, tirx.SBlock):
            self.visit(statement.body, path + (_StatementPathStep("block_body"),))
            return
        if isinstance(statement, tirx.AttrStmt):
            self.visit(statement.body, path + (_StatementPathStep("attr_body"),))
            return
        if isinstance(statement, tirx.IfThenElse):
            self.visit(statement.then_case, path + (_StatementPathStep("if_then"),))
            if statement.else_case is not None:
                self.visit(statement.else_case, path + (_StatementPathStep("if_else"),))
            return
        if isinstance(statement, tirx.Evaluate) and isinstance(
            statement.value, tirx.Call
        ):
            call = statement.value
            name = _op_name(call)
            if name.startswith("tl.tileop."):
                region_accesses = _tile_call_region_accesses(call)
                reads, writes = _accessed_buffers(region_accesses)
                base_name = name.removeprefix("tl.tileop.")
                classification = _classify_call(call, reads, writes)
                self._append(
                    self._operation_name(base_name, reads, writes),
                    classification,
                    reads,
                    writes,
                    _profile_call(call, classification, reads, writes),
                    path,
                    statement,
                    region_accesses,
                )
            return
        if isinstance(statement, tirx.BufferStore):
            region_accesses = _scalar_statement_region_accesses(statement)
            reads, writes = _accessed_buffers(region_accesses)
            classification = _classify_statement(statement)
            self._append(
                self._operation_name("store", reads, writes),
                classification,
                reads,
                writes,
                _profile_statement(statement, reads, writes),
                path,
                statement,
                region_accesses,
            )


def _dataflow_node(operation: _CollectedOperation) -> DataflowNode:
    return DataflowNode(
        operation.name,
        operation.classification.unit,
        tuple(sorted(_buffer_name(buffer) for buffer in operation.reads)),
        tuple(sorted(_buffer_name(buffer) for buffer in operation.writes)),
        operation.classification.execution_kind,
        operation.profile,
        operation.classification.instruction_kind,
    )


def _buffer_sort_key(buffer: tirx.Buffer) -> tuple:
    return (
        _buffer_name(buffer),
        buffer.scope(),
        str(buffer.dtype),
        tuple(map(str, buffer.shape)),
    )


def _collect_buffer_identities(
    operations: list[_CollectedOperation],
) -> tuple[dict[tirx.Buffer, int], list[tirx.Buffer]]:
    buffer_ids: dict[tirx.Buffer, int] = {}
    ordered_buffers: list[tirx.Buffer] = []
    for operation in operations:
        ordered_reads = sorted(operation.reads, key=_buffer_sort_key)
        ordered_writes = sorted(operation.writes, key=_buffer_sort_key)
        for buffer in (*ordered_reads, *ordered_writes):
            if buffer not in buffer_ids:
                buffer_ids[buffer] = len(ordered_buffers)
                ordered_buffers.append(buffer)
    return buffer_ids, ordered_buffers


def _auto_wsp_enabled(loop: tirx.For) -> bool:
    value = loop.annotations.get("tl.wsp.auto_schedule")
    return value is not None and int(value) != 0


def _is_pipeline_loop(loop: tirx.For) -> bool:
    return _auto_wsp_enabled(loop) or "num_stages" in loop.annotations


def _find_pipelined_loops(func: tirx.PrimFunc) -> list[tirx.For]:
    loops: list[tirx.For] = []

    def visit(node) -> None:
        if isinstance(node, tirx.For) and _is_pipeline_loop(node):
            loops.append(node)

    post_order_visit(func.body, visit)
    return loops


def _resolve_prim_func(prim: tirx.PrimFunc | ir.IRModule) -> tirx.PrimFunc:
    if isinstance(prim, tirx.PrimFunc):
        return prim
    if isinstance(prim, ir.IRModule):
        functions = [func for func in prim.functions.values() if isinstance(func, tirx.PrimFunc)]
        if len(functions) != 1:
            raise ValueError("expected an IRModule containing exactly one PrimFunc")
        return functions[0]
    raise TypeError(f"expected PrimFunc or IRModule, got {type(prim).__name__}")


def _kernel_thread_count(func: tirx.PrimFunc) -> int | None:
    """Return static threadIdx extents before or after launch materialization."""

    extents: dict[str, int] = {}
    dynamic = False

    def visit(node) -> None:
        nonlocal dynamic
        if isinstance(node, tirx.For):
            thread_binding = node.thread_binding
            if thread_binding is None:
                return
            thread_tag = thread_binding.thread_tag
            extent = node.extent
        elif isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent":
            thread_tag = getattr(node.node, "thread_tag", "")
            extent = node.value
        else:
            return
        if not thread_tag.startswith("threadIdx."):
            return
        if not isinstance(extent, tirx.IntImm):
            dynamic = True
            return
        value = int(extent)
        if thread_tag in extents and extents[thread_tag] != value:
            dynamic = True
            return
        extents[thread_tag] = value

    post_order_visit(func.body, visit)
    if dynamic or not extents:
        return None
    result = 1
    for extent in extents.values():
        result *= extent
    return result


def _pipelined_loop_paths(
    statement,
    path: tuple[_StatementPathStep, ...] = (),
) -> list[tuple[tirx.For, tuple[_StatementPathStep, ...]]]:
    """Return pipelined loops and their paths in lexical order."""

    result: list[tuple[tirx.For, tuple[_StatementPathStep, ...]]] = []
    if isinstance(statement, tirx.SeqStmt):
        for index, child in enumerate(statement.seq):
            result.extend(
                _pipelined_loop_paths(
                    child, path + (_StatementPathStep("seq", index),)
                )
            )
    elif isinstance(statement, tirx.For):
        if _is_pipeline_loop(statement):
            result.append((statement, path))
        if statement.kind != tirx.ForKind.PARALLEL:
            result.extend(
                _pipelined_loop_paths(
                    statement.body,
                    path
                    + (_StatementPathStep("for_body", int(statement.kind)),),
                )
            )
    elif isinstance(statement, tirx.SBlockRealize):
        result.extend(
            _pipelined_loop_paths(
                statement.block.body,
                path + (_StatementPathStep("block_realize_body"),),
            )
        )
    elif isinstance(statement, tirx.SBlock):
        result.extend(
            _pipelined_loop_paths(
                statement.body, path + (_StatementPathStep("block_body"),)
            )
        )
    elif isinstance(statement, tirx.AttrStmt):
        result.extend(
            _pipelined_loop_paths(
                statement.body, path + (_StatementPathStep("attr_body"),)
            )
        )
    elif isinstance(statement, tirx.IfThenElse):
        result.extend(
            _pipelined_loop_paths(
                statement.then_case, path + (_StatementPathStep("if_then"),)
            )
        )
        if statement.else_case is not None:
            result.extend(
                _pipelined_loop_paths(
                    statement.else_case, path + (_StatementPathStep("if_else"),)
                )
            )
    return result


def _path_has_prefix(
    path: tuple[_StatementPathStep, ...],
    prefix: tuple[_StatementPathStep, ...],
) -> bool:
    return len(path) >= len(prefix) and path[: len(prefix)] == prefix


def _validate_program_pipeline_structure(
    loop_sites: list[tuple[tirx.For, tuple[_StatementPathStep, ...]]],
) -> None:
    """Reject control flow that the ordered-region model cannot represent."""

    paths = [path for _, path in loop_sites]
    for loop_index, (_, path) in enumerate(loop_sites):
        if any(step.kind in {"if_then", "if_else"} for step in path):
            raise ValueError(
                "pipeline loops under conditional control flow are not supported "
                "by the ordered program-region model"
            )
        enclosing_pipeline_bodies = {
            outer_path
            + (_StatementPathStep("for_body", int(outer_loop.kind)),)
            for outer_index, (outer_loop, outer_path) in enumerate(loop_sites)
            if outer_index != loop_index
        }
        if any(
            step.kind == "for_body"
            and step.index == int(tirx.ForKind.SERIAL)
            and path[: path_index + 1] not in enclosing_pipeline_bodies
            for path_index, step in enumerate(path)
        ):
            raise ValueError(
                "pipeline loops nested in non-parallel loops are not supported "
                "by the ordered program-region model"
            )
    for outer_index, outer_path in enumerate(paths):
        outer_loop = loop_sites[outer_index][0]
        outer_body = outer_path + (
            _StatementPathStep("for_body", int(outer_loop.kind)),
        )
        if any(
            inner_index != outer_index
            and _path_has_prefix(inner_path, outer_body)
            for inner_index, inner_path in enumerate(paths)
        ):
            raise ValueError(
                "nested pipeline loops are not supported by the ordered "
                "program-region model"
            )


def analyze_program_dataflow(
    prim: tirx.PrimFunc | ir.IRModule,
) -> ProgramDataflowAnalysis:
    """Extract a whole-function graph segmented around all pipeline loops.

    Serial operations before, between, and after pipeline loops remain in the
    graph. Dependencies crossing region boundaries are conservative lexical
    memory dependencies and therefore never rely on a pipeline stage number.
    """

    func = _resolve_prim_func(prim)
    loop_sites = _pipelined_loop_paths(func.body)
    if not loop_sites:
        raise ValueError("expected at least one software-pipeline loop")
    auto_modes = tuple(_auto_wsp_enabled(loop) for loop, _ in loop_sites)
    if any(auto_modes) and not all(auto_modes):
        raise ValueError(
            "all pipeline loops in one PrimFunc must use the same auto_wsp mode"
        )
    _validate_program_pipeline_structure(loop_sites)

    collector = _OperationCollector()
    collector.visit(func.body)
    if not collector.operations:
        raise ValueError("the specialization scope contains no operations")

    nodes = [_dataflow_node(operation) for operation in collector.operations]
    raw_region_accesses: list[tuple[_RawRegionAccess, ...]] = []
    paths: list[tuple[_StatementPathStep, ...]] = []
    memberships: list[int | None] = []
    for operation in collector.operations:
        raw_region_accesses.append(operation.region_accesses)
        paths.append(operation.path)
        enclosing = [
            (len(loop_path), loop_index)
            for loop_index, (loop, loop_path) in enumerate(loop_sites)
            if _path_has_prefix(
                operation.path,
                loop_path
                + (_StatementPathStep("for_body", int(loop.kind)),),
            )
        ]
        memberships.append(max(enclosing)[1] if enclosing else None)

    for path, membership in zip(paths, memberships):
        if any(step.kind in {"if_then", "if_else"} for step in path):
            raise ValueError(
                "operations under conditional control flow are not supported "
                "by the ordered program-region model"
            )
        serial_loop_depth = sum(
            step.kind == "for_body" and step.index == int(tirx.ForKind.SERIAL)
            for step in path
        )
        expected_depth = 1 if membership is not None else 0
        if serial_loop_depth != expected_depth:
            raise ValueError(
                "operations in non-pipeline serial loops are not supported "
                "by the ordered program-region model"
            )

    region_keys: list[tuple[str, int]] = []
    operation_region_ids: list[int] = []
    serial_index = -1
    previous_membership: int | None | object = object()
    for membership in memberships:
        if membership is None:
            if previous_membership is not None:
                serial_index += 1
                region_keys.append((ProgramRegionKind.SERIAL.value, serial_index))
        elif membership != previous_membership:
            region_keys.append((ProgramRegionKind.PIPELINE.value, membership))
        operation_region_ids.append(len(region_keys) - 1)
        previous_membership = membership

    regions: list[ProgramRegion] = []
    for region_id, (kind_value, source_id) in enumerate(region_keys):
        kind = ProgramRegionKind(kind_value)
        region_operation_ids = tuple(
            operation_id
            for operation_id, assigned_region in enumerate(operation_region_ids)
            if assigned_region == region_id
        )
        if kind == ProgramRegionKind.PIPELINE:
            loop = loop_sites[source_id][0]
            regions.append(
                ProgramRegion(
                    region_id,
                    kind,
                    region_operation_ids,
                    loop=loop,
                    num_stages=(
                        None
                        if _auto_wsp_enabled(loop)
                        else int(loop.annotations["num_stages"])
                    ),
                    auto_schedule=_auto_wsp_enabled(loop),
                )
            )
        else:
            regions.append(ProgramRegion(region_id, kind, region_operation_ids))

    operations = tuple(
        ProgramOperation(index, node, operation_region_ids[index])
        for index, node in enumerate(nodes)
    )

    buffer_ids, ordered_buffers = _collect_buffer_identities(collector.operations)

    region_accesses = tuple(
        BufferRegionAccess(
            operation_id,
            buffer_ids[access.buffer],
            access.kind,
            access.region,
            access.is_exact,
        )
        for operation_id, operation_regions in enumerate(raw_region_accesses)
        for access in operation_regions
    )

    edges: list[DataflowEdge] = []
    edge_indices: dict[tuple[int, int, int, int | None], int] = {}

    def add_edge(
        producer_id: int,
        consumer_id: int,
        iteration_distance: int,
        dependency_kinds: str | frozenset[str],
        buffer: tirx.Buffer | None,
    ) -> None:
        buffer_id = buffer_ids[buffer] if buffer is not None else None
        key = (producer_id, consumer_id, iteration_distance, buffer_id)
        kinds = (
            frozenset({dependency_kinds})
            if isinstance(dependency_kinds, str)
            else dependency_kinds
        )
        if key in edge_indices:
            edge = edges[edge_indices[key]]
            edge.dependency_kinds |= kinds
            return
        edge_indices[key] = len(edges)
        edges.append(
            DataflowEdge(
                nodes[producer_id],
                nodes[consumer_id],
                iteration_distance,
                kinds,
                _buffer_name(buffer) if buffer is not None else None,
                buffer_id,
            )
        )

    _build_direct_memory_dependencies(
        len(nodes),
        region_accesses,
        loop_sites[0][0].loop_var,
        lambda producer, consumer, kind, buffer_id: add_edge(
            producer,
            consumer,
            0,
            kind,
            ordered_buffers[buffer_id],
        ),
    )

    # Add loop-carried RAW dependencies directly in dense program IDs.
    for loop_index, (loop, _) in enumerate(loop_sites):
        loop_operation_ids = {
            operation_id
            for operation_id, membership in enumerate(memberships)
            if membership == loop_index
        }
        loop_accesses = tuple(
            access
            for access in region_accesses
            if access.operation_id in loop_operation_ids
        )
        writes_by_buffer: dict[int, list[BufferRegionAccess]] = {}
        for access in loop_accesses:
            if access.kind == BufferAccessKind.WRITE:
                writes_by_buffer.setdefault(access.buffer_id, []).append(access)
        if isinstance(loop.extent, tirx.IntImm) and int(loop.extent) <= 1:
            continue
        for read in loop_accesses:
            if read.kind != BufferAccessKind.READ:
                continue
            writers = writes_by_buffer.get(read.buffer_id, ())
            earlier_writes = tuple(
                writer
                for writer in writers
                if writer.operation_id < read.operation_id
            )
            if any(
                region_covers(writer, read, loop.loop_var)
                for writer in earlier_writes
            ):
                continue
            for writer in reversed(writers):
                if analyze_region_overlap(
                    writer,
                    read,
                    loop.loop_var,
                    iteration_delta=1,
                ) != RegionOverlap.DISJOINT:
                    add_edge(
                        writer.operation_id,
                        read.operation_id,
                        1,
                        "RAW",
                        ordered_buffers[read.buffer_id],
                    )
                if region_covers(
                    writer,
                    read,
                    loop.loop_var,
                    inner_iteration_delta=1,
                ):
                    break

    edges = _remove_transitive_ordering_edges(edges)

    buffers = tuple(
        BufferDescriptor(
            buffer_id=buffer_id,
            name=_buffer_name(buffer),
            scope=buffer.scope(),
            nbytes=_buffer_nbytes(buffer),
            buffer=buffer,
        )
        for buffer_id, buffer in enumerate(ordered_buffers)
    )
    return ProgramDataflowAnalysis(
        prim_func=func,
        operations=operations,
        edges=tuple(edges),
        regions=tuple(regions),
        kernel_threads=_kernel_thread_count(func),
        buffers=buffers,
        region_accesses=region_accesses,
    )
