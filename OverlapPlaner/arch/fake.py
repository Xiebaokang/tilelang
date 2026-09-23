"""Minimal architecture used to test the classify/realize SPI without CUDA."""

from __future__ import annotations

from collections.abc import Iterator

from OverlapPlaner.facts import FactGraph, FactNode, OpKind
from OverlapPlaner.physical import (
    PhysicalPlan,
    WarpRequirement,
    analyze_shared_memory,
    enumerate_warp_allocations,
)
from OverlapPlaner.structure.model import Structure

from .api import Architecture, ClassifiedGraph, DeviceResource, NodeTraits, ResourceKind

FAKE_RESOURCE = DeviceResource(
    name="fake",
    warp_size=32,
    max_threads_per_block=1024,
    register_file_capacity=65536,
    shared_memory_capacity_bytes=65536,
    max_registers_per_thread=255,
    partition_warp_multiple=1,
)


class FakeArch(Architecture):
    """Two engines only: copies are memory, everything else is generic compute."""

    name = "fake"

    def resource(self) -> DeviceResource:
        return FAKE_RESOURCE

    def classify_node(self, graph: FactGraph, node: FactNode) -> NodeTraits:
        del graph
        if node.kind == OpKind.COPY:
            return NodeTraits(ResourceKind.MEMORY, "copy")
        if node.kind == OpKind.GEMM:
            return NodeTraits(
                ResourceKind.COMPUTE,
                "tensorcore",
                occupies_cta_partition=True,
            )
        return NodeTraits(ResourceKind.COMPUTE, "generic")

    def realize(
        self, classified: ClassifiedGraph, structure: Structure
    ) -> Iterator[PhysicalPlan]:
        resource = self.resource()
        granule = resource.partition_warp_multiple
        maximum = (
            resource.max_warps_per_block
            if structure.num_groups == 1
            else granule
        )
        requirements = tuple(
            WarpRequirement(granule, maximum, granule)
            for _ in range(structure.num_groups)
        )
        shared_memory = analyze_shared_memory(
            classified.graph, structure, resource
        )
        if not shared_memory.fits:
            return
        for allocation in enumerate_warp_allocations(
            classified, structure, resource, requirements
        ):
            yield PhysicalPlan(structure, allocation, shared_memory)


FAKE = FakeArch()
