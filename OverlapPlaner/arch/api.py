"""Architecture SPI: traits that structure search may see.

Core code classifies nothing. A plugin maps fact-graph nodes onto these
traits; L1 only compares ``kind`` / ``engine`` / async / partition flags.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from OverlapPlaner.facts import (
    DependencyKind,
    FactEdge,
    FactGraph,
    FactNode,
    is_group_visible_scope,
)

if TYPE_CHECKING:
    from OverlapPlaner.physical.model import PhysicalPlan
    from OverlapPlaner.structure.model import Structure


class ResourceKind(str, Enum):
    MEMORY = "memory"
    COMPUTE = "compute"


class EdgeAction(str, Enum):
    KEEP = "keep"
    SPLIT_STAGE = "split_stage"
    SPLIT_GROUP = "split_group"


@dataclass(frozen=True, slots=True)
class NodeTraits:
    """Architecture-visible behavior of one operation, without an ISA name."""

    kind: ResourceKind
    engine: str
    async_completion: bool = False
    occupies_cta_partition: bool = False
    issue_priority: int = 0

    def __post_init__(self) -> None:
        if not self.engine or self.engine != self.engine.lower():
            raise ValueError("engine must be a lowercase identifier")
        if not self.engine.isidentifier():
            raise ValueError("engine must be a Python identifier")
        if self.issue_priority < 0:
            raise ValueError("issue_priority cannot be negative")


@dataclass(frozen=True, slots=True)
class ClassifiedGraph:
    """A fact graph plus one trait record per node."""

    graph: FactGraph
    traits: tuple[NodeTraits, ...]

    def __post_init__(self) -> None:
        if len(self.traits) != len(self.graph.nodes):
            raise ValueError("traits must cover every node")

    def traits_for(self, node_id: int) -> NodeTraits:
        self.graph.node_for_id(node_id)
        return self.traits[node_id]


@dataclass(frozen=True, slots=True)
class DeviceResource:
    """Physical limits consumed later by realize(), not by classify()."""

    name: str
    warp_size: int
    max_threads_per_block: int
    register_file_capacity: int
    shared_memory_capacity_bytes: int
    max_registers_per_thread: int
    partition_warp_multiple: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("resource name must not be empty")
        if self.warp_size < 1:
            raise ValueError("warp_size must be positive")
        if (
            self.max_threads_per_block < self.warp_size
            or self.max_threads_per_block % self.warp_size
        ):
            raise ValueError("max_threads_per_block must contain whole warps")
        if self.partition_warp_multiple < 1:
            raise ValueError("partition_warp_multiple must be positive")
        if (
            self.register_file_capacity < 0
            or self.shared_memory_capacity_bytes < 0
            or self.max_registers_per_thread < 1
        ):
            raise ValueError("resource capacities must be non-negative")

    @property
    def max_warps_per_block(self) -> int:
        return self.max_threads_per_block // self.warp_size


def can_split_stages(producer: NodeTraits, consumer: NodeTraits) -> bool:
    """Return whether an edge may cross a software-pipeline stage.

    Memory ops may split from anyone. Two compute ops may split only when
    they use different engines.
    """

    if (
        producer.kind == ResourceKind.COMPUTE
        and consumer.kind == ResourceKind.COMPUTE
    ):
        return producer.engine != consumer.engine
    return True


def is_group_opportunity(classified: ClassifiedGraph, edge: FactEdge) -> bool:
    """Return whether ``edge`` can be cut to expose a group opportunity.

    Every edge backed by a group-visible buffer (global, shared, or tmem) is
    an opportunity, including serial-region edges and ordering hazards. This
    matches Overlaper's group constraint. A fragment RAW edge is additionally
    an opportunity when both ends are compute, the producer is not an
    initializer, and the engines differ. A register-layout copy stays with its
    consumer.
    """

    graph = classified.graph
    if edge.producer_id == edge.consumer_id:
        return False
    scope = graph.buffer_for_id(edge.buffer_id).scope
    if is_group_visible_scope(scope):
        return True
    if graph.is_initializer(edge.producer_id):
        return False
    if scope != "local.fragment":
        return False
    if DependencyKind.RAW not in edge.dependency_kinds:
        return False
    producer = classified.traits_for(edge.producer_id)
    consumer = classified.traits_for(edge.consumer_id)
    # A fragment-to-fragment copy defines the register layout consumed by its
    # users. Giving that copy a producer partition of its own only replaces a
    # register conversion with two shared-memory handoffs. Keep it with its
    # consumer; the edge feeding the copy may still cross groups, which is the
    # useful softmax -> PV boundary in attention.
    if producer.engine == "register_copy":
        return False
    return (
        producer.kind == ResourceKind.COMPUTE
        and consumer.kind == ResourceKind.COMPUTE
        and producer.engine != consumer.engine
    )


def allowed_edge_actions(
    classified: ClassifiedGraph, edge: FactEdge
) -> frozenset[EdgeAction]:
    """Return legal KEEP / SPLIT_STAGE / SPLIT_GROUP actions for one edge."""

    graph = classified.graph
    graph.node_for_id(edge.producer_id)
    graph.node_for_id(edge.consumer_id)
    actions = {EdgeAction.KEEP}
    if not graph.is_initializer(edge.producer_id) and can_split_stages(
        classified.traits_for(edge.producer_id),
        classified.traits_for(edge.consumer_id),
    ):
        actions.add(EdgeAction.SPLIT_STAGE)
    if is_group_opportunity(classified, edge):
        actions.add(EdgeAction.SPLIT_GROUP)
    return frozenset(actions)


class Architecture:
    """Plugin surface for one GPU family.

    ``classify`` fills traits for L1. ``realize`` binds warps, registers, and
    shared memory for one structure. Candidate selection belongs to the tune
    layer so architecture plugins only decide classification and feasibility.
    """

    name: str

    def resource(self) -> DeviceResource:
        raise NotImplementedError

    def classify_node(self, graph: FactGraph, node: FactNode) -> NodeTraits:
        raise NotImplementedError

    def classify(self, graph: FactGraph) -> ClassifiedGraph:
        traits = tuple(
            self.classify_node(graph, node) for node in graph.nodes
        )
        return ClassifiedGraph(graph, traits)

    def realize(
        self, classified: ClassifiedGraph, structure: Structure
    ) -> Iterator[PhysicalPlan]:
        raise NotImplementedError
