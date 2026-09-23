"""Extract an Overlaper dataflow graph from TileLang TIR."""

from __future__ import annotations

from dataclasses import dataclass, field

from tvm import arith, ir, tirx
from tvm.target import Target
from tvm.tirx.stmt_functor import post_order_visit, substitute

from ..headware.resolve import resolve_hardware
from ..headware.spec import HardwareSpec, Instruction, OperationInfo
from .graph import (
    BufferAccessKind,
    BufferDescriptor,
    BufferRangeAccess,
    DataflowEdge,
    DataflowGraph,
    DataflowNode,
    DependencyKind,
    RegionKind,
)


@dataclass(frozen=True, slots=True)
class _RawBufferAccess:
    buffer: tirx.Buffer = field(compare=False, hash=False)
    kind: BufferAccessKind
    ranges: tuple[ir.Range, ...] = field(compare=False, hash=False)
    is_exact: bool


@dataclass(frozen=True, slots=True)
class _RawOperation:
    name: str
    statement: tirx.Stmt = field(compare=False, hash=False, repr=False)
    accesses: tuple[_RawBufferAccess, ...] = field(compare=False, hash=False)
    pipeline_loop: tirx.For | None = field(
        compare=False, hash=False, repr=False
    )
    instruction: Instruction


def _op_name(call: tirx.Call) -> str:
    return call.op.name if isinstance(call.op, ir.Op) else str(call.op)


def _whole_buffer_ranges(buffer: tirx.Buffer) -> tuple[ir.Range, ...]:
    return tuple(ir.Range.from_min_extent(0, extent) for extent in buffer.shape)


def _accessed_buffers(
    accesses: tuple[_RawBufferAccess, ...] | list[_RawBufferAccess],
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


def _parallel_domains(statement: tirx.Stmt) -> dict[tirx.Var, arith.IntSet]:
    domains: dict[tirx.Var, arith.IntSet] = {}

    def visit(node) -> None:
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL:
            domains[node.loop_var] = arith.IntervalSet(
                node.min, node.min + node.extent - 1
            )

    post_order_visit(statement, visit)
    return domains


def _interval_for_index(
    index: tirx.PrimExpr,
    domains: dict[tirx.Var, arith.IntSet],
) -> tuple[ir.Range | None, bool]:
    analyzer = arith.Analyzer()

    def is_dense(expression: tirx.PrimExpr) -> bool:
        if not domains:
            return True
        variables = list(domains)
        coefficients = arith.detect_linear_equation(expression, variables)
        if len(coefficients) != len(variables) + 1:
            return False
        nonzero = [
            coefficient
            for coefficient in coefficients[:-1]
            if not analyzer.can_prove_equal(coefficient, 0)
        ]
        return len(nonzero) <= 1 and all(
            analyzer.can_prove_equal(coefficient, 1)
            or analyzer.can_prove_equal(coefficient, -1)
            for coefficient in nonzero
        )

    if isinstance(index, tirx.Ramp):
        if not isinstance(index.stride, tirx.IntImm) or int(index.stride) == 0:
            return None, False
        base_set = analyzer.int_set(index.base, domains)
        if not isinstance(base_set, arith.IntervalSet):
            return None, False
        stride = int(index.stride)
        lane_span = (int(index.lanes) - 1) * stride
        minimum = base_set.min_value + min(0, lane_span)
        maximum = base_set.max_value + max(0, lane_span)
        return (
            ir.Range.from_min_extent(
                analyzer.simplify(minimum),
                analyzer.simplify(maximum - minimum + 1),
            ),
            abs(stride) == 1 and is_dense(index.base),
        )

    index_set = analyzer.int_set(index, domains)
    if not isinstance(index_set, arith.IntervalSet):
        return None, False
    return (
        ir.Range.from_min_extent(
            analyzer.simplify(index_set.min_value),
            analyzer.simplify(index_set.max_value - index_set.min_value + 1),
        ),
        is_dense(index),
    )


def _scalar_accesses(statement: tirx.Stmt) -> list[_RawBufferAccess]:
    domains = _parallel_domains(statement)
    accesses: list[_RawBufferAccess] = []

    def append(buffer, indices, kind: BufferAccessKind) -> None:
        if len(indices) != len(buffer.shape):
            accesses.append(
                _RawBufferAccess(buffer, kind, _whole_buffer_ranges(buffer), False)
            )
            return
        ranges: list[ir.Range] = []
        is_exact = True
        for index in indices:
            interval, dimension_is_exact = _interval_for_index(index, domains)
            if interval is None:
                accesses.append(
                    _RawBufferAccess(
                        buffer, kind, _whole_buffer_ranges(buffer), False
                    )
                )
                return
            ranges.append(interval)
            is_exact = is_exact and dimension_is_exact
        accesses.append(_RawBufferAccess(buffer, kind, tuple(ranges), is_exact))

    def visit(node) -> None:
        if isinstance(node, tirx.BufferLoad):
            append(node.buffer, node.indices, BufferAccessKind.READ)
        elif isinstance(node, tirx.BufferStore):
            append(node.buffer, node.indices, BufferAccessKind.WRITE)

    post_order_visit(statement, visit)
    return accesses


def _tile_call_accesses(call: tirx.Call) -> list[_RawBufferAccess]:
    accesses: list[_RawBufferAccess] = []

    def visit(node) -> None:
        if not isinstance(node, tirx.Call) or _op_name(node) != "tl.tileop.region":
            return
        if len(node.args) < 2 or not isinstance(node.args[0], tirx.BufferLoad):
            return
        load = node.args[0]
        access_mask = int(node.args[1])
        extents = tuple(node.args[2:])
        is_exact = len(extents) == len(load.indices) == len(load.buffer.shape)
        ranges = (
            tuple(
                ir.Range.from_min_extent(index, extent)
                for index, extent in zip(load.indices, extents)
            )
            if is_exact
            else _whole_buffer_ranges(load.buffer)
        )
        if access_mask & 1:
            accesses.append(
                _RawBufferAccess(
                    load.buffer, BufferAccessKind.READ, ranges, is_exact
                )
            )
        if access_mask & 2:
            accesses.append(
                _RawBufferAccess(
                    load.buffer, BufferAccessKind.WRITE, ranges, is_exact
                )
            )

    post_order_visit(call, visit)
    return accesses


def _operation_info(
    name: str,
    statement: tirx.Stmt,
    reads: set[tirx.Buffer],
    writes: set[tirx.Buffer],
) -> OperationInfo:
    call_names: list[str] = []

    def visit(node) -> None:
        if isinstance(node, tirx.Call):
            call_names.append(_op_name(node))

    post_order_visit(statement, visit)
    return OperationInfo(
        name=name,
        call_names=tuple(call_names),
        read_scopes=tuple(sorted({buffer.scope() for buffer in reads})),
        write_scopes=tuple(sorted({buffer.scope() for buffer in writes})),
    )


def _is_pipeline_loop(loop: tirx.For) -> bool:
    # Keep in sync with overlap_plan::IsOverlapPipelineLoop.
    return "tl.pipelined" in loop.annotations


class _OperationCollector:
    def __init__(self, hardware: HardwareSpec) -> None:
        self.hardware = hardware
        self.operations: list[_RawOperation] = []
        self.name_counts: dict[str, int] = {}

    @staticmethod
    def _operation_name(
        base_name: str,
        reads: set[tirx.Buffer],
        writes: set[tirx.Buffer],
    ) -> str:
        inputs = "_".join(sorted(str(buffer.name) for buffer in reads))
        outputs = "_".join(sorted(str(buffer.name) for buffer in writes))
        if base_name == "copy" and inputs and outputs:
            return f"copy_{inputs}_to_{outputs}"
        if outputs:
            return f"{base_name}_{outputs}"
        if inputs:
            return f"{base_name}_{inputs}"
        return base_name

    def _append(
        self,
        base_name: str,
        statement: tirx.Stmt,
        accesses: list[_RawBufferAccess],
        pipeline_loop: tirx.For | None,
        classifier_name: str | None = None,
    ) -> None:
        reads, writes = _accessed_buffers(accesses)
        count = self.name_counts.get(base_name, 0)
        self.name_counts[base_name] = count + 1
        name = base_name if count == 0 else f"{base_name}_{count + 1}"
        instruction = self.hardware.classify(
            _operation_info(
                classifier_name or base_name, statement, reads, writes
            )
        )
        self.operations.append(
            _RawOperation(
                name, statement, tuple(accesses), pipeline_loop, instruction
            )
        )

    def visit(
        self,
        statement: tirx.Stmt,
        pipeline_loop: tirx.For | None = None,
    ) -> None:
        if isinstance(statement, tirx.SeqStmt):
            for child in statement.seq:
                self.visit(child, pipeline_loop)
            return
        if isinstance(statement, tirx.For):
            if statement.kind == tirx.ForKind.PARALLEL:
                accesses = _scalar_accesses(statement)
                reads, writes = _accessed_buffers(accesses)
                name = self._operation_name("parallel", reads, writes)
                self._append(name, statement, accesses, pipeline_loop)
                return
            next_pipeline = pipeline_loop
            if _is_pipeline_loop(statement):
                if pipeline_loop is not None:
                    raise ValueError("nested pipeline loops are not supported")
                next_pipeline = statement
            self.visit(statement.body, next_pipeline)
            return
        if isinstance(statement, tirx.SBlockRealize):
            self.visit(statement.block.body, pipeline_loop)
            return
        if isinstance(statement, tirx.SBlock):
            self.visit(statement.body, pipeline_loop)
            return
        if isinstance(statement, tirx.AttrStmt):
            self.visit(statement.body, pipeline_loop)
            return
        if isinstance(statement, tirx.IfThenElse):
            self.visit(statement.then_case, pipeline_loop)
            if statement.else_case is not None:
                self.visit(statement.else_case, pipeline_loop)
            return
        if isinstance(statement, tirx.Evaluate) and isinstance(
            statement.value, tirx.Call
        ):
            call = statement.value
            call_name = _op_name(call)
            if not call_name.startswith("tl.tileop."):
                return
            accesses = _tile_call_accesses(call)
            reads, writes = _accessed_buffers(accesses)
            name = self._operation_name(
                call_name.removeprefix("tl.tileop."), reads, writes
            )
            self._append(
                name,
                statement,
                accesses,
                pipeline_loop,
                classifier_name=call_name,
            )
            return
        if isinstance(statement, tirx.BufferStore):
            accesses = _scalar_accesses(statement)
            reads, writes = _accessed_buffers(accesses)
            name = self._operation_name("store", reads, writes)
            self._append(name, statement, accesses, pipeline_loop)


def _static_product(values) -> int | None:
    result = 1
    for value in values:
        if not isinstance(value, tirx.IntImm):
            return None
        result *= int(value)
    return result


def _buffer_nbytes(buffer: tirx.Buffer) -> int | None:
    elements = _static_product(buffer.shape)
    if elements is None:
        return None
    bits = int(buffer.dtype.bits) * int(buffer.dtype.lanes)
    return (elements * bits + 7) // 8


def _collect_buffers(
    operations: list[_RawOperation],
) -> tuple[dict[tirx.Buffer, int], tuple[BufferDescriptor, ...]]:
    buffer_ids: dict[tirx.Buffer, int] = {}
    ordered: list[tirx.Buffer] = []
    for operation in operations:
        for access in operation.accesses:
            if access.buffer not in buffer_ids:
                buffer_ids[access.buffer] = len(ordered)
                ordered.append(access.buffer)
    return buffer_ids, tuple(
        BufferDescriptor(
            buffer_id,
            str(buffer.name),
            buffer.scope(),
            _buffer_nbytes(buffer),
            buffer,
        )
        for buffer_id, buffer in enumerate(ordered)
    )


def _shift(expression, loop_var: tirx.Var, delta: int):
    return (
        expression
        if delta == 0
        else substitute(expression, {loop_var: loop_var + delta})
    )


def _may_overlap(
    earlier: BufferRangeAccess,
    later: BufferRangeAccess,
    loop_var: tirx.Var | None = None,
    iteration_delta: int = 0,
) -> bool:
    if earlier.buffer_id != later.buffer_id:
        return False
    if len(earlier.ranges) != len(later.ranges):
        return True
    analyzer = arith.Analyzer()
    for earlier_range, later_range in zip(earlier.ranges, later.ranges):
        earlier_min = earlier_range.min
        earlier_end = analyzer.simplify(earlier_min + earlier_range.extent)
        later_min = later_range.min
        later_extent = later_range.extent
        if loop_var is not None:
            later_min = _shift(later_min, loop_var, iteration_delta)
            later_extent = _shift(later_extent, loop_var, iteration_delta)
        later_end = analyzer.simplify(later_min + later_extent)
        if analyzer.can_prove(earlier_end <= later_min) or analyzer.can_prove(
            later_end <= earlier_min
        ):
            return False
    return True


def _covers(
    outer: BufferRangeAccess,
    inner: BufferRangeAccess,
    loop_var: tirx.Var | None = None,
    iteration_delta: int = 0,
) -> bool:
    if (
        outer.buffer_id != inner.buffer_id
        or len(outer.ranges) != len(inner.ranges)
        or not outer.is_exact
        or not inner.is_exact
    ):
        return False
    analyzer = arith.Analyzer()
    for outer_range, inner_range in zip(outer.ranges, inner.ranges):
        outer_min = outer_range.min
        outer_end = analyzer.simplify(outer_min + outer_range.extent)
        inner_min = inner_range.min
        inner_extent = inner_range.extent
        if loop_var is not None:
            inner_min = _shift(inner_min, loop_var, iteration_delta)
            inner_extent = _shift(inner_extent, loop_var, iteration_delta)
        inner_end = analyzer.simplify(inner_min + inner_extent)
        if not (
            analyzer.can_prove(outer_min <= inner_min)
            and analyzer.can_prove(inner_end <= outer_end)
        ):
            return False
    return True


def _resolve_prim_func(prim: tirx.PrimFunc | ir.IRModule) -> tirx.PrimFunc:
    if isinstance(prim, tirx.PrimFunc):
        return prim
    if isinstance(prim, ir.IRModule):
        functions = [
            function
            for function in prim.functions.values()
            if isinstance(function, tirx.PrimFunc)
        ]
        if len(functions) != 1:
            raise ValueError("expected exactly one PrimFunc")
        return functions[0]
    raise TypeError(f"expected PrimFunc or IRModule, got {type(prim).__name__}")


def _kernel_thread_count(function: tirx.PrimFunc) -> int | None:
    extents: dict[str, int] = {}
    dynamic = False

    def visit(node) -> None:
        nonlocal dynamic
        if isinstance(node, tirx.For) and node.thread_binding is not None:
            thread_tag = node.thread_binding.thread_tag
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

    post_order_visit(function.body, visit)
    if dynamic or not extents:
        return None
    result = 1
    for extent in extents.values():
        result *= extent
    return result


def _build_edges(
    nodes: list[DataflowNode],
    accesses: list[BufferRangeAccess],
    region_loops: list[tirx.For | None],
) -> tuple[DataflowEdge, ...]:
    edge_kinds: dict[tuple[int, int, int, int], set[DependencyKind]] = {}

    def add(producer, consumer, distance, kind, buffer_id) -> None:
        edge_kinds.setdefault(
            (producer, consumer, distance, buffer_id), set()
        ).add(kind)

    accesses_by_node = [
        tuple(access for access in accesses if access.node_id == node_id)
        for node_id in range(len(nodes))
    ]
    reaching_writes: dict[int, list[BufferRangeAccess]] = {}
    live_reads: dict[int, list[BufferRangeAccess]] = {}
    for node_id, node_accesses in enumerate(accesses_by_node):
        reads = tuple(
            access
            for access in node_accesses
            if access.kind == BufferAccessKind.READ
        )
        writes = tuple(
            access
            for access in node_accesses
            if access.kind == BufferAccessKind.WRITE
        )
        for read in reads:
            for writer in reversed(reaching_writes.get(read.buffer_id, ())):
                if _may_overlap(writer, read):
                    add(writer.node_id, node_id, 0, DependencyKind.RAW, read.buffer_id)
                if _covers(writer, read):
                    break
        for write in writes:
            for reader in live_reads.get(write.buffer_id, ()):
                if _may_overlap(reader, write):
                    add(reader.node_id, node_id, 0, DependencyKind.WAR, write.buffer_id)
            for writer in reversed(reaching_writes.get(write.buffer_id, ())):
                if _may_overlap(writer, write):
                    add(writer.node_id, node_id, 0, DependencyKind.WAW, write.buffer_id)
                if _covers(writer, write):
                    break
        for read in reads:
            live_reads.setdefault(read.buffer_id, []).append(read)
        for write in writes:
            reaching_writes[write.buffer_id] = [
                writer
                for writer in reaching_writes.get(write.buffer_id, ())
                if not _covers(write, writer)
            ]
            reaching_writes[write.buffer_id].append(write)
            live_reads[write.buffer_id] = [
                reader
                for reader in live_reads.get(write.buffer_id, ())
                if not _covers(write, reader)
            ]

    for region_id, loop in enumerate(region_loops):
        if loop is None or (
            isinstance(loop.extent, tirx.IntImm) and int(loop.extent) <= 1
        ):
            continue
        node_ids = {node.node_id for node in nodes if node.region_id == region_id}
        region_accesses = tuple(
            access for access in accesses if access.node_id in node_ids
        )
        writes_by_buffer: dict[int, list[BufferRangeAccess]] = {}
        for access in region_accesses:
            if access.kind == BufferAccessKind.WRITE:
                writes_by_buffer.setdefault(access.buffer_id, []).append(access)
        for read in region_accesses:
            if read.kind != BufferAccessKind.READ:
                continue
            writers = writes_by_buffer.get(read.buffer_id, ())
            earlier = (writer for writer in writers if writer.node_id < read.node_id)
            if any(_covers(writer, read, loop.loop_var) for writer in earlier):
                continue
            for writer in reversed(writers):
                if _may_overlap(writer, read, loop.loop_var, 1):
                    add(writer.node_id, read.node_id, 1, DependencyKind.RAW, read.buffer_id)
                if _covers(writer, read, loop.loop_var, 1):
                    break

    return tuple(
        DataflowEdge(
            producer,
            consumer,
            distance,
            frozenset(kinds),
            buffer_id,
        )
        for (producer, consumer, distance, buffer_id), kinds in sorted(
            edge_kinds.items()
        )
    )


def extract_dataflow_graph(
    prim: tirx.PrimFunc | ir.IRModule,
    hardware: HardwareSpec | None = None,
    *,
    target: Target | None = None,
) -> DataflowGraph:
    """Extract ordered operations, buffer ranges, regions, and dependencies."""

    function = _resolve_prim_func(prim)
    if hardware is None:
        if target is None:
            target = (
                function.attrs.get("target")
                if function.attrs is not None
                else None
            )
        if target is None:
            raise ValueError("hardware cannot be inferred: pass hardware or a target")
        hardware = resolve_hardware(target)

    collector = _OperationCollector(hardware)
    collector.visit(function.body)
    operations = collector.operations
    if not operations:
        raise ValueError("the PrimFunc contains no supported operations")

    region_kinds: list[RegionKind] = []
    region_loops: list[tirx.For | None] = []
    node_region_ids: list[int] = []
    previous_loop: tirx.For | None | object = object()
    for operation in operations:
        if operation.pipeline_loop is not previous_loop:
            region_kinds.append(
                RegionKind.PIPELINE
                if operation.pipeline_loop is not None
                else RegionKind.SERIAL
            )
            region_loops.append(operation.pipeline_loop)
            previous_loop = operation.pipeline_loop
        node_region_ids.append(len(region_kinds) - 1)

    buffer_ids, buffers = _collect_buffers(operations)
    nodes: list[DataflowNode] = []
    accesses: list[BufferRangeAccess] = []
    for node_id, operation in enumerate(operations):
        read_ids = tuple(
            sorted(
                {
                    buffer_ids[access.buffer]
                    for access in operation.accesses
                    if access.kind == BufferAccessKind.READ
                }
            )
        )
        write_ids = tuple(
            sorted(
                {
                    buffer_ids[access.buffer]
                    for access in operation.accesses
                    if access.kind == BufferAccessKind.WRITE
                }
            )
        )
        nodes.append(
            DataflowNode(
                node_id,
                node_region_ids[node_id],
                operation.name,
                operation.instruction,
                read_ids,
                write_ids,
                operation.statement,
            )
        )
        accesses.extend(
            BufferRangeAccess(
                node_id,
                buffer_ids[access.buffer],
                access.kind,
                access.ranges,
                access.is_exact,
            )
            for access in operation.accesses
        )

    return DataflowGraph(
        buffers=tuple(buffers),
        nodes=tuple(nodes),
        edges=_build_edges(nodes, accesses, region_loops),
        region_kinds=tuple(region_kinds),
        buffer_accesses=tuple(accesses),
        hardware=hardware,
        prim_func=function,
        kernel_threads=_kernel_thread_count(function),
    )
