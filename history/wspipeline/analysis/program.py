"""Stable program-level data model produced by TIR analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property

from tvm import ir, tirx

from .core import DataflowEdge, DataflowNode


@dataclass(frozen=True)
class BufferDescriptor:
    """Stable metadata for one exact Buffer ObjectRef in the analyzed IR."""

    buffer_id: int
    name: str
    scope: str
    nbytes: int | None
    buffer: tirx.Buffer = field(compare=False, hash=False, repr=False)


@dataclass(frozen=True)
class OperationBufferAccess:
    """Exact buffer identities read and written by one schedulable operation."""

    operation_id: int
    read_buffer_ids: tuple[int, ...]
    write_buffer_ids: tuple[int, ...]


class BufferAccessKind(str, Enum):
    """The semantic access made to a buffer region."""

    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class BufferRegionAccess:
    """One exact or conservative interval region accessed by an operation."""

    operation_id: int
    buffer_id: int
    kind: BufferAccessKind
    region: tuple[ir.Range, ...] = field(compare=False, hash=False)
    is_exact: bool = True


class ProgramRegionKind(str, Enum):
    """Whether a lexical program region executes once or is pipelined."""

    SERIAL = "serial"
    PIPELINE = "pipeline"


@dataclass(frozen=True)
class ProgramRegion:
    """One contiguous serial segment or one software-pipelined loop."""

    region_id: int
    kind: ProgramRegionKind
    operation_ids: tuple[int, ...]
    loop: tirx.For | None = field(default=None, compare=False, hash=False, repr=False)
    num_stages: int | None = None
    auto_schedule: bool = False


@dataclass(frozen=True)
class ProgramOperation:
    """The canonical program-level record for one schedulable operation."""

    operation_id: int
    node: DataflowNode
    region_id: int


@dataclass(frozen=True)
class ProgramDataflowAnalysis:
    """Whole-scope graph plus an ordered serial/pipeline region sequence."""

    prim_func: tirx.PrimFunc = field(compare=False, hash=False, repr=False)
    operations: tuple[ProgramOperation, ...]
    edges: tuple[DataflowEdge, ...]
    regions: tuple[ProgramRegion, ...]
    kernel_threads: int | None
    buffers: tuple[BufferDescriptor, ...] = ()
    region_accesses: tuple[BufferRegionAccess, ...] = ()

    def __post_init__(self) -> None:
        operation_ids = tuple(
            operation.operation_id for operation in self.operations
        )
        if operation_ids != tuple(range(len(self.operations))):
            raise ValueError("program operations must use dense ordered IDs")
        region_ids = tuple(region.region_id for region in self.regions)
        if region_ids != tuple(range(len(self.regions))):
            raise ValueError("program regions must use dense ordered IDs")
        buffer_ids = tuple(buffer.buffer_id for buffer in self.buffers)
        if buffer_ids != tuple(range(len(self.buffers))):
            raise ValueError("program buffers must use dense ordered IDs")

        expected_by_region = {
            region_id: tuple(
                operation.operation_id
                for operation in self.operations
                if operation.region_id == region_id
            )
            for region_id in range(len(self.regions))
        }
        for region in self.regions:
            if region.operation_ids != expected_by_region[region.region_id]:
                raise ValueError(
                    f"region {region.region_id} operation IDs disagree with "
                    "the canonical operation table"
                )
            if region.kind == ProgramRegionKind.SERIAL:
                if region.num_stages is not None or region.auto_schedule:
                    raise ValueError("serial region cannot carry pipeline settings")
            elif region.auto_schedule:
                if region.num_stages is not None:
                    raise ValueError("automatic pipeline depth must remain undecided")
            elif region.num_stages is None or region.num_stages < 2:
                raise ValueError("manual pipeline region needs at least two stages")
        if any(
            operation.region_id < 0 or operation.region_id >= len(self.regions)
            for operation in self.operations
        ):
            raise ValueError("program operation references an unknown region")
        if any(
            access.operation_id < 0
            or access.operation_id >= len(self.operations)
            or access.buffer_id < 0
            or access.buffer_id >= len(self.buffers)
            for access in self.region_accesses
        ):
            raise ValueError("program region access references an unknown ID")
        node_set = set(self.nodes)
        if any(
            edge.producer not in node_set
            or edge.consumer not in node_set
            or (
                edge.buffer_id is not None
                and (edge.buffer_id < 0 or edge.buffer_id >= len(self.buffers))
            )
            for edge in self.edges
        ):
            raise ValueError("program edge references an unknown operation or buffer")

    @cached_property
    def nodes(self) -> tuple[DataflowNode, ...]:
        """Return nodes in dense operation-ID order."""

        return tuple(operation.node for operation in self.operations)

    @cached_property
    def operation_accesses(self) -> tuple[OperationBufferAccess, ...]:
        """Derive buffer-level access summaries from canonical region accesses."""

        reads = [set() for _ in self.operations]
        writes = [set() for _ in self.operations]
        for access in self.region_accesses:
            target = reads if access.kind == BufferAccessKind.READ else writes
            target[access.operation_id].add(access.buffer_id)
        return tuple(
            OperationBufferAccess(
                operation_id,
                tuple(sorted(reads[operation_id])),
                tuple(sorted(writes[operation_id])),
            )
            for operation_id in range(len(self.operations))
        )

    @cached_property
    def _operations_by_node_id(self) -> dict[int, ProgramOperation]:
        """Index the canonical node objects without recursive dataclass hashing."""

        return {id(operation.node): operation for operation in self.operations}

    def operation_for(self, node: DataflowNode) -> ProgramOperation:
        operation = self._operations_by_node_id.get(id(node))
        if operation is None:
            raise ValueError(f"node {node.name} is not a canonical program operation")
        return operation

    def nodes_for_region(self, region: ProgramRegion) -> tuple[DataflowNode, ...]:
        return tuple(self.operations[index].node for index in region.operation_ids)

    def edges_for_region(self, region: ProgramRegion) -> tuple[DataflowEdge, ...]:
        """Return dependencies whose endpoints both belong to one region."""

        node_set = set(self.nodes_for_region(region))
        return tuple(
            edge
            for edge in self.edges
            if edge.producer in node_set and edge.consumer in node_set
        )

    def buffer_for_id(self, buffer_id: int) -> BufferDescriptor:
        """Return buffer metadata by its dense analysis-local identity."""

        if buffer_id < 0 or buffer_id >= len(self.buffers):
            raise KeyError(f"unknown buffer ID {buffer_id}")
        descriptor = self.buffers[buffer_id]
        if descriptor.buffer_id != buffer_id:
            raise ValueError("buffer descriptors do not use dense IDs")
        return descriptor

    def accesses_for(self, node: DataflowNode) -> OperationBufferAccess:
        """Return exact buffer accesses for ``node``."""

        operation_id = self.operation_for(node).operation_id
        return self.operation_accesses[operation_id]
