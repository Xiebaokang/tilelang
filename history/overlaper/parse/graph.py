"""Small, ID-based dataflow graph used by Overlaper."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING

from ..headware.spec import HardwareSpec, Instruction

if TYPE_CHECKING:
    from tvm import ir, tirx


class RegionKind(str, Enum):
    """Whether operations in a region may be software-pipelined."""

    SERIAL = "serial"
    PIPELINE = "pipeline"


class DependencyKind(str, Enum):
    """The buffer hazard requiring one operation to precede another."""

    RAW = "RAW"
    WAR = "WAR"
    WAW = "WAW"


class BufferAccessKind(str, Enum):
    """Whether an operation reads or writes a buffer range."""

    READ = "read"
    WRITE = "write"


@dataclass(frozen=True, slots=True)
class BufferDescriptor:
    """Stable metadata for one exact Buffer ObjectRef in the analyzed IR."""

    buffer_id: int
    name: str
    scope: str
    nbytes: int | None
    buffer: tirx.Buffer = field(compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if self.buffer_id < 0:
            raise ValueError("buffer_id must be non-negative")
        if not self.name:
            raise ValueError("buffer name cannot be empty")
        if self.nbytes is not None and self.nbytes < 0:
            raise ValueError("buffer size cannot be negative")


@dataclass(frozen=True, slots=True)
class BufferRangeAccess:
    """One exact or conservative buffer range accessed by an operation."""

    node_id: int
    buffer_id: int
    kind: BufferAccessKind
    ranges: tuple[ir.Range, ...] = field(compare=False, hash=False)
    is_exact: bool = True

    def __post_init__(self) -> None:
        if self.node_id < 0:
            raise ValueError("node_id must be non-negative")
        if self.buffer_id < 0:
            raise ValueError("buffer_id must be non-negative")


@dataclass(frozen=True, slots=True)
class DataflowNode:
    """One extracted operation; its node ID is stable within the graph."""

    node_id: int
    region_id: int
    name: str
    instruction: Instruction
    reads: tuple[int, ...] = ()
    writes: tuple[int, ...] = ()
    operation: tirx.Stmt | None = field(
        default=None, compare=False, hash=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.node_id < 0:
            raise ValueError("node_id must be non-negative")
        if self.region_id < 0:
            raise ValueError("region_id must be non-negative")
        if not self.name:
            raise ValueError("node name cannot be empty")
        if not isinstance(self.instruction, Instruction):
            raise TypeError("instruction must be an Instruction")
        if any(buffer_id < 0 for buffer_id in (*self.reads, *self.writes)):
            raise ValueError("buffer IDs must be non-negative")
        object.__setattr__(self, "reads", tuple(sorted(set(self.reads))))
        object.__setattr__(self, "writes", tuple(sorted(set(self.writes))))


@dataclass(frozen=True, slots=True)
class DataflowEdge:
    """One buffer-backed precedence relation between two operations."""

    producer_id: int
    consumer_id: int
    iteration_distance: int = 0
    dependency_kinds: frozenset[DependencyKind] = field(
        default_factory=lambda: frozenset({DependencyKind.RAW})
    )
    buffer_id: int | None = None

    def __post_init__(self) -> None:
        if self.producer_id < 0 or self.consumer_id < 0:
            raise ValueError("edge endpoint IDs must be non-negative")
        if self.buffer_id is None:
            raise ValueError("memory dependency edges require a buffer_id")
        if self.buffer_id < 0:
            raise ValueError("buffer_id must be non-negative")
        if self.iteration_distance < 0:
            raise ValueError("iteration_distance must be non-negative")
        kinds = frozenset(DependencyKind(kind) for kind in self.dependency_kinds)
        if not kinds:
            raise ValueError("dependency_kinds cannot be empty")
        if self.producer_id == self.consumer_id and self.iteration_distance == 0:
            raise ValueError("a same-iteration edge cannot depend on itself")
        object.__setattr__(self, "dependency_kinds", kinds)

    @property
    def is_loop_carried(self) -> bool:
        return self.iteration_distance > 0


@dataclass(frozen=True)
class DataflowGraph:
    """Canonical Overlaper graph with cross-table consistency checks."""

    buffers: tuple[BufferDescriptor, ...]
    nodes: tuple[DataflowNode, ...]
    edges: tuple[DataflowEdge, ...]
    region_kinds: tuple[RegionKind, ...]
    buffer_accesses: tuple[BufferRangeAccess, ...] = ()
    hardware: HardwareSpec | None = field(
        default=None, compare=False, hash=False, repr=False
    )
    prim_func: tirx.PrimFunc | None = field(
        default=None, compare=False, hash=False, repr=False
    )
    kernel_threads: int | None = None

    def __post_init__(self) -> None:
        if tuple(item.buffer_id for item in self.buffers) != tuple(
            range(len(self.buffers))
        ):
            raise ValueError("buffers must use dense ordered IDs")
        if tuple(item.node_id for item in self.nodes) != tuple(
            range(len(self.nodes))
        ):
            raise ValueError("nodes must use dense ordered IDs")
        if self.kernel_threads is not None and self.kernel_threads < 1:
            raise ValueError("kernel_threads must be positive or None")

        buffer_count = len(self.buffers)
        if any(node.region_id >= len(self.region_kinds) for node in self.nodes):
            raise ValueError("node references an unknown region")
        if self.hardware is not None and any(
            node.instruction not in self.hardware.supported_instructions
            for node in self.nodes
        ):
            raise ValueError("node uses an instruction unsupported by the hardware")
        if any(
            buffer_id >= buffer_count
            for node in self.nodes
            for buffer_id in (*node.reads, *node.writes)
        ):
            raise ValueError("node references an unknown buffer")

        for access in self.buffer_accesses:
            if access.node_id >= len(self.nodes):
                raise ValueError("buffer access references an unknown node")
            if access.buffer_id >= buffer_count:
                raise ValueError("buffer access references an unknown buffer")
            node = self.nodes[access.node_id]
            expected = (
                node.reads
                if access.kind == BufferAccessKind.READ
                else node.writes
            )
            if access.buffer_id not in expected:
                raise ValueError("buffer access disagrees with node reads/writes")
            rank = len(self.buffers[access.buffer_id].buffer.shape)
            if len(access.ranges) != rank:
                raise ValueError("buffer access rank disagrees with buffer rank")

        keys = set()
        for edge in self.edges:
            if edge.producer_id >= len(self.nodes) or edge.consumer_id >= len(
                self.nodes
            ):
                raise ValueError("edge references an unknown node")
            if edge.buffer_id is None or edge.buffer_id >= buffer_count:
                raise ValueError("edge references an unknown buffer")
            key = (
                edge.producer_id,
                edge.consumer_id,
                edge.buffer_id,
                edge.iteration_distance,
                edge.dependency_kinds,
            )
            if key in keys:
                raise ValueError("graph contains a duplicate edge")
            keys.add(key)

        self.topological_order()

    def buffer_for_id(self, buffer_id: int) -> BufferDescriptor:
        if buffer_id < 0:
            raise KeyError(f"unknown buffer ID {buffer_id}")
        try:
            return self.buffers[buffer_id]
        except IndexError as error:
            raise KeyError(f"unknown buffer ID {buffer_id}") from error

    def node_for_id(self, node_id: int) -> DataflowNode:
        if node_id < 0:
            raise KeyError(f"unknown node ID {node_id}")
        try:
            return self.nodes[node_id]
        except IndexError as error:
            raise KeyError(f"unknown node ID {node_id}") from error

    def nodes_for_region(self, region_id: int) -> tuple[DataflowNode, ...]:
        if region_id < 0 or region_id >= len(self.region_kinds):
            raise KeyError(f"unknown region ID {region_id}")
        return tuple(node for node in self.nodes if node.region_id == region_id)

    def accesses_for_node(self, node_id: int) -> tuple[BufferRangeAccess, ...]:
        self.node_for_id(node_id)
        return tuple(
            access for access in self.buffer_accesses if access.node_id == node_id
        )

    def issue_priority(self, node_id: int) -> int:
        return self.node_for_id(node_id).instruction.issue_priority

    def issue_order_key(self, node_id: int) -> tuple[int, int]:
        return (-self.issue_priority(node_id), node_id)

    @cached_property
    def incoming_edges(self) -> tuple[tuple[DataflowEdge, ...], ...]:
        result: list[list[DataflowEdge]] = [[] for _ in self.nodes]
        for edge in self.edges:
            result[edge.consumer_id].append(edge)
        return tuple(tuple(items) for items in result)

    @cached_property
    def outgoing_edges(self) -> tuple[tuple[DataflowEdge, ...], ...]:
        result: list[list[DataflowEdge]] = [[] for _ in self.nodes]
        for edge in self.edges:
            result[edge.producer_id].append(edge)
        return tuple(tuple(items) for items in result)

    def topological_order(self) -> tuple[int, ...]:
        """Return stable operation IDs, ignoring loop-carried dependencies."""

        incoming = [0] * len(self.nodes)
        successors: list[list[int]] = [[] for _ in self.nodes]
        for edge in self.edges:
            if edge.is_loop_carried:
                continue
            incoming[edge.consumer_id] += 1
            successors[edge.producer_id].append(edge.consumer_id)

        ready = [node.node_id for node in self.nodes if incoming[node.node_id] == 0]
        ready.sort(key=self.issue_order_key)
        order: list[int] = []
        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for consumer_id in successors[node_id]:
                incoming[consumer_id] -= 1
                if incoming[consumer_id] == 0:
                    ready.append(consumer_id)
            ready.sort(key=self.issue_order_key)
        if len(order) != len(self.nodes):
            raise ValueError("same-iteration dependencies contain a cycle")
        return tuple(order)
