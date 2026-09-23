"""Architecture-free IR facts used by OverlapPlan search.

This graph records TileOp structure, buffer ranges, and memory hazards.
It does not classify ISA names, engines, or resource quantities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvm import ir, tirx


class RegionKind(str, Enum):
    SERIAL = "serial"
    PIPELINE = "pipeline"


class DependencyKind(str, Enum):
    RAW = "RAW"
    WAR = "WAR"
    WAW = "WAW"


class BufferAccessKind(str, Enum):
    READ = "read"
    WRITE = "write"


class OpKind(str, Enum):
    COPY = "copy"
    GEMM = "gemm"
    FILL = "fill"
    REDUCE = "reduce"
    ELEMENTWISE = "elementwise"
    STORE = "store"
    OTHER = "other"


def is_group_visible_scope(scope: str) -> bool:
    """Return whether a buffer can communicate between warp groups."""

    return (
        scope in ("", "global")
        or scope.startswith("shared")
        or "tmem" in scope
    )


@dataclass(frozen=True, slots=True)
class GemmFact:
    """Static GEMM parameters taken from ``tl.tileop.gemm`` arguments."""

    m: int | None
    n: int | None
    k: int | None
    transpose_a: bool
    transpose_b: bool
    policy: str
    clear_accum: bool
    a_dtype: str
    b_dtype: str
    c_dtype: str
    a_scope: str
    b_scope: str
    c_scope: str


@dataclass(frozen=True, slots=True)
class ReduceFact:
    """Static reduction parameters taken from ``tl.tileop.reduce``."""

    kind: str
    dim: int | None
    clear: bool | None


@dataclass(frozen=True, slots=True)
class BufferFact:
    """Stable metadata for one exact Buffer ObjectRef in the analyzed IR."""

    buffer_id: int
    name: str
    scope: str
    dtype: str
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
class FactNode:
    """One extracted operation; its node ID is stable within the graph."""

    node_id: int
    region_id: int
    kind: OpKind
    name: str
    tileop: str
    reads: tuple[int, ...] = ()
    writes: tuple[int, ...] = ()
    gemm: GemmFact | None = None
    reduce: ReduceFact | None = None
    parallel_extents: tuple[int | None, ...] = ()
    scalar_ops: tuple[str, ...] = ()
    statement: tirx.Stmt | None = field(
        default=None, compare=False, hash=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.node_id < 0:
            raise ValueError("node_id must be non-negative")
        if self.region_id < 0:
            raise ValueError("region_id must be non-negative")
        if not self.name:
            raise ValueError("node name cannot be empty")
        if not self.tileop:
            raise ValueError("tileop cannot be empty")
        if self.kind == OpKind.GEMM and self.gemm is None:
            raise ValueError("gemm nodes require GemmFact")
        if self.kind != OpKind.GEMM and self.gemm is not None:
            raise ValueError("only gemm nodes may carry GemmFact")
        if self.kind == OpKind.REDUCE and self.reduce is None:
            raise ValueError("reduce nodes require ReduceFact")
        if self.kind != OpKind.REDUCE and self.reduce is not None:
            raise ValueError("only reduce nodes may carry ReduceFact")
        if any(buffer_id < 0 for buffer_id in (*self.reads, *self.writes)):
            raise ValueError("buffer IDs must be non-negative")
        object.__setattr__(self, "reads", tuple(sorted(set(self.reads))))
        object.__setattr__(self, "writes", tuple(sorted(set(self.writes))))


@dataclass(frozen=True, slots=True)
class FactEdge:
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


@dataclass(frozen=True, slots=True)
class RegionFact:
    """One serial or software-pipelined region in program order."""

    region_id: int
    kind: RegionKind
    static_extent: int | None = None
    loop: tirx.For | None = field(
        default=None, compare=False, hash=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.region_id < 0:
            raise ValueError("region_id must be non-negative")
        if self.static_extent is not None and self.static_extent < 1:
            raise ValueError("static_extent must be positive or None")
        if self.kind == RegionKind.SERIAL and self.loop is not None:
            raise ValueError("serial regions cannot carry a pipeline loop")
        if self.kind == RegionKind.PIPELINE and self.loop is None:
            raise ValueError("pipeline regions require the T.Pipelined loop")


@dataclass(frozen=True)
class FactGraph:
    """Canonical OverlapPlan fact graph with cross-table consistency checks."""

    buffers: tuple[BufferFact, ...]
    nodes: tuple[FactNode, ...]
    edges: tuple[FactEdge, ...]
    regions: tuple[RegionFact, ...]
    buffer_accesses: tuple[BufferRangeAccess, ...] = ()
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
        if tuple(item.region_id for item in self.regions) != tuple(
            range(len(self.regions))
        ):
            raise ValueError("regions must use dense ordered IDs")
        if self.kernel_threads is not None and self.kernel_threads < 1:
            raise ValueError("kernel_threads must be positive or None")

        buffer_count = len(self.buffers)
        if any(node.region_id >= len(self.regions) for node in self.nodes):
            raise ValueError("node references an unknown region")
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

    @property
    def region_kinds(self) -> tuple[RegionKind, ...]:
        return tuple(region.kind for region in self.regions)

    def buffer_for_id(self, buffer_id: int) -> BufferFact:
        if buffer_id < 0:
            raise KeyError(f"unknown buffer ID {buffer_id}")
        try:
            return self.buffers[buffer_id]
        except IndexError as error:
            raise KeyError(f"unknown buffer ID {buffer_id}") from error

    def node_for_id(self, node_id: int) -> FactNode:
        if node_id < 0:
            raise KeyError(f"unknown node ID {node_id}")
        try:
            return self.nodes[node_id]
        except IndexError as error:
            raise KeyError(f"unknown node ID {node_id}") from error

    def region_for_id(self, region_id: int) -> RegionFact:
        if region_id < 0:
            raise KeyError(f"unknown region ID {region_id}")
        try:
            return self.regions[region_id]
        except IndexError as error:
            raise KeyError(f"unknown region ID {region_id}") from error

    def nodes_for_region(self, region_id: int) -> tuple[FactNode, ...]:
        self.region_for_id(region_id)
        return tuple(node for node in self.nodes if node.region_id == region_id)

    def accesses_for_node(self, node_id: int) -> tuple[BufferRangeAccess, ...]:
        self.node_for_id(node_id)
        return tuple(
            access for access in self.buffer_accesses if access.node_id == node_id
        )

    def is_initializer(self, node_id: int) -> bool:
        """Return whether a node only seeds fragment state for later consumers."""

        node = self.node_for_id(node_id)
        if node.kind == OpKind.FILL:
            return True
        if node.reads:
            return False
        return bool(node.writes) and all(
            self.buffer_for_id(buffer_id).scope == "local.fragment"
            for buffer_id in node.writes
        )

    def scopes_for_node(self, node_id: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
        node = self.node_for_id(node_id)
        reads = tuple(self.buffer_for_id(buffer_id).scope for buffer_id in node.reads)
        writes = tuple(self.buffer_for_id(buffer_id).scope for buffer_id in node.writes)
        return reads, writes

    @cached_property
    def incoming_edges(self) -> tuple[tuple[FactEdge, ...], ...]:
        result: list[list[FactEdge]] = [[] for _ in self.nodes]
        for edge in self.edges:
            result[edge.consumer_id].append(edge)
        return tuple(tuple(items) for items in result)

    @cached_property
    def outgoing_edges(self) -> tuple[tuple[FactEdge, ...], ...]:
        result: list[list[FactEdge]] = [[] for _ in self.nodes]
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
        ready.sort()
        order: list[int] = []
        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for consumer_id in successors[node_id]:
                incoming[consumer_id] -= 1
                if incoming[consumer_id] == 0:
                    ready.append(consumer_id)
            ready.sort()
        if len(order) != len(self.nodes):
            raise ValueError("same-iteration dependencies contain a cycle")
        return tuple(order)
