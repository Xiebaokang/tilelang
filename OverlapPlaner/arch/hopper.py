"""Hopper classification and physical realization.

Hopper may reason about TMA / WGMMA internally. Copy TMA vs SIMT first
asks TileLang's warp-specialized producer classifier. Small payloads
outside an in-loop smem handoff stay SIMT. Values returned to core are
traits, warp counts, and shared-memory layouts — not ISA names.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from functools import lru_cache

from tvm.ffi import get_global_func
from tvm.target import Target

from tilelang.carver.arch.driver import get_max_dynamic_shared_size_bytes

from OverlapPlaner.facts import FactGraph, FactNode, OpKind, RegionKind
from OverlapPlaner.physical import (
    PhysicalPlan,
    SetMaxNRegPolicy,
    WarpRequirement,
    analyze_shared_memory,
    enumerate_warp_allocations,
    warp_requirements_are_feasible,
)
from OverlapPlaner.physical.shared_memory import default_buffer_alignment
from OverlapPlaner.structure.model import Structure

from .api import (
    Architecture,
    ClassifiedGraph,
    DeviceResource,
    NodeTraits,
    ResourceKind,
)

_SFU_OPS = frozenset(
    {
        "tirx.exp",
        "tirx.exp2",
        "tirx.log",
        "tirx.log2",
        "tirx.sin",
        "tirx.cos",
        "tirx.tanh",
        "tirx.sqrt",
    }
)

_HOPPER_SHARED_MEMORY_FALLBACK_BYTES = 227 * 1024


def _hopper_shared_memory_capacity_bytes() -> int:
    """Return the current device's opt-in dynamic-smem block limit."""

    queried = get_max_dynamic_shared_size_bytes()
    if queried is not None and queried > 0:
        return queried
    return _HOPPER_SHARED_MEMORY_FALLBACK_BYTES


# Compile and realize both read this one queried value. The fallback is
# used only when CUDA device attributes are unavailable (for example CPU CI).
HOPPER_RESOURCE = DeviceResource(
    name="hopper",
    warp_size=32,
    max_threads_per_block=1024,
    register_file_capacity=64512,
    shared_memory_capacity_bytes=_hopper_shared_memory_capacity_bytes(),
    max_registers_per_thread=255,
    partition_warp_multiple=4,
)

_WGMMA_TILE_ROWS = 64
_MMA_TILE_ROWS = 16
_FALLBACK_ASYNC_ALIGNMENT = 1024
# TMA launch overhead is not worth a small serial copy such as a Mamba
# prologue dt / dA vector. In-loop smem handoffs skip this cutoff.
_MIN_TMA_PAYLOAD_BYTES = 1024
_HOPPER_SETMAXNREG = SetMaxNRegPolicy(24, 240, 8)
HOPPER_CUDA_TARGET = Target(
    {
        "kind": "cuda",
        "arch": "sm_90a",
        "max_shared_memory_per_block": HOPPER_RESOURCE.shared_memory_capacity_bytes,
    }
)


def _is_shared(scope: str) -> bool:
    return scope.startswith("shared") and scope != "shared.tmem"


def _is_fragment(scope: str) -> bool:
    return scope == "local.fragment"


def _is_global(scope: str) -> bool:
    return scope in ("", "global")


def _copy_payload_bytes(graph: FactGraph, node: FactNode) -> int | None:
    """Return the static tile size of a copy, or None if it is symbolic."""

    sizes: list[int] = []
    for buffer_id in (*node.reads, *node.writes):
        nbytes = graph.buffer_for_id(buffer_id).nbytes
        if nbytes is None:
            return None
        sizes.append(nbytes)
    return min(sizes) if sizes else None


def _feeds_in_loop_smem_consumer(graph: FactGraph, node: FactNode) -> bool:
    """Return whether this copy's shared output is read in the same pipeline."""

    if graph.region_for_id(node.region_id).kind != RegionKind.PIPELINE:
        return False
    shared_writes = {
        buffer_id
        for buffer_id in node.writes
        if _is_shared(graph.buffer_for_id(buffer_id).scope)
    }
    if not shared_writes:
        return False
    return any(
        shared_writes.intersection(other.reads)
        for other in graph.nodes_for_region(node.region_id)
        if other.node_id != node.node_id
    )


def _copy_kind(graph: FactGraph, node: FactNode) -> str | None:
    """Return Hopper's TMA / cp.async / SIMT decision for one copy.

    Capability comes from ``ClassifyWarpSpecializedProducerCopy``. Small
    payloads stay SIMT unless the copy and an smem consumer both sit in the
    same ``T.Pipelined`` body.
    """

    if node.statement is None:
        return None
    classify = get_global_func("tl.cuda.ClassifyWarpSpecializedProducerCopy")
    kind = str(classify(node.statement, HOPPER_CUDA_TARGET))
    if kind in {"unknown", "unsupported"}:
        return None
    if kind in {"tma", "cp_async"}:
        payload = _copy_payload_bytes(graph, node)
        if (
            payload is not None
            and payload < _MIN_TMA_PAYLOAD_BYTES
            and not _feeds_in_loop_smem_consumer(graph, node)
        ):
            return "simt"
    return kind


@lru_cache(maxsize=32)
def optional_tma_copy_ids(classified: ClassifiedGraph) -> tuple[int, ...]:
    """Small ordinary gmem-to-smem copies where TMA is legal but not default.

    Explicit copy preferences remain the user's choice.  The scheduling
    search may compare both backends only when the shared result has a
    consumer, so the selected backend can be carried by a completion edge.
    """

    graph = classified.graph
    classify = get_global_func("tl.cuda.ClassifyWarpSpecializedProducerCopy")
    eligible = []
    for node in graph.nodes:
        if node.kind != OpKind.COPY or node.statement is None:
            continue
        call = getattr(node.statement, "value", None)
        if call is None:
            continue
        annotations = getattr(call, "annotations", {})
        if any(
            key in annotations
            for key in (
                "prefer_instruction",
                "is_tma_copy",
                "is_async_copy",
                "force_cp_async",
                "cluster_mask",
            )
        ):
            continue
        if not any(
            _is_global(graph.buffer_for_id(buffer_id).scope)
            for buffer_id in node.reads
        ) or not any(
            _is_shared(graph.buffer_for_id(buffer_id).scope)
            for buffer_id in node.writes
        ):
            continue
        payload = _copy_payload_bytes(graph, node)
        if payload is None or payload >= _MIN_TMA_PAYLOAD_BYTES:
            continue
        if classified.traits_for(node.node_id).async_completion:
            continue
        if str(classify(node.statement, HOPPER_CUDA_TARGET)) != "tma":
            continue
        shared_outputs = {
            buffer_id
            for buffer_id in node.writes
            if _is_shared(graph.buffer_for_id(buffer_id).scope)
        }
        if not any(
            edge.producer_id == node.node_id
            and edge.buffer_id in shared_outputs
            for edge in graph.edges
        ):
            continue
        eligible.append(node.node_id)
    return tuple(eligible)


def _classify_copy(
    graph: FactGraph,
    node: FactNode,
    read_scopes: tuple[str, ...],
    write_scopes: tuple[str, ...],
) -> NodeTraits:
    scopes = set(read_scopes + write_scopes)
    kind = _copy_kind(graph, node)
    if kind == "tma":
        return NodeTraits(
            ResourceKind.MEMORY,
            "copy",
            async_completion=True,
            issue_priority=6,
        )
    if kind == "cp_async":
        return NodeTraits(
            ResourceKind.MEMORY,
            "copy",
            async_completion=True,
            issue_priority=4,
        )
    if any(_is_global(scope) for scope in scopes) and any(
        _is_shared(scope) for scope in scopes
    ):
        # Scheduling priority is independent of the lowering instruction.
        # Even a small SIMT load can unlock later compute; retaining the
        # global-to-shared priority also matches Overlaper's topological order.
        return NodeTraits(ResourceKind.MEMORY, "copy", issue_priority=6)
    if any(_is_global(scope) for scope in scopes) and any(
        _is_fragment(scope) for scope in scopes
    ):
        return NodeTraits(ResourceKind.MEMORY, "copy", issue_priority=3)
    if any(_is_fragment(scope) for scope in scopes) and any(
        _is_shared(scope) for scope in scopes
    ):
        return NodeTraits(ResourceKind.MEMORY, "copy", issue_priority=2)
    if scopes and all(_is_fragment(scope) for scope in scopes):
        return NodeTraits(ResourceKind.COMPUTE, "register_copy", issue_priority=1)
    return NodeTraits(ResourceKind.COMPUTE, "generic")


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return max(value, 0)
    return (value + alignment - 1) // alignment * alignment


def _kernel_warps(graph: FactGraph) -> int | None:
    threads = graph.kernel_threads
    warp_size = HOPPER_RESOURCE.warp_size
    if threads is None or threads < 1 or threads % warp_size:
        return None
    return threads // warp_size


def _is_float8(dtype: str) -> bool:
    return dtype.startswith("float8")


def _check_wgmma_dtypes(gemm) -> bool:
    """Mirror CUDA ``CheckWgmma`` for the extracted GEMM facts."""

    if gemm.k is None or gemm.k <= 0:
        return False
    a_dtype = gemm.a_dtype
    b_dtype = gemm.b_dtype
    c_dtype = gemm.c_dtype
    if c_dtype == "float16":
        if a_dtype == b_dtype == "float16":
            return gemm.k % 16 == 0
        if _is_float8(a_dtype) and _is_float8(b_dtype):
            return (
                not gemm.transpose_a
                and gemm.transpose_b
                and gemm.k % 32 == 0
            )
        return False
    if c_dtype == "float32":
        if a_dtype == b_dtype and a_dtype in {"float16", "bfloat16"}:
            return gemm.k % 16 == 0
        if a_dtype == b_dtype == "tfloat32":
            return (
                not gemm.transpose_a
                and gemm.transpose_b
                and gemm.k % 8 == 0
            )
        if _is_float8(a_dtype) and _is_float8(b_dtype):
            return (
                not gemm.transpose_a
                and gemm.transpose_b
                and gemm.k % 32 == 0
            )
        return False
    if c_dtype == "int32" and a_dtype in {"int8", "uint8"} and b_dtype in {
        "int8",
        "uint8",
    }:
        return (
            not gemm.transpose_a
            and gemm.transpose_b
            and gemm.k % 32 == 0
        )
    return False


def _uses_warpgroup_tensorcore(gemm, kernel_warps: int | None) -> bool:
    """Return whether this GEMM issues as a Hopper warpgroup collective.

    WGMMA needs M >= 64 and a warpgroup-multiple CTA. Smaller tiles fall back
    to MMA (16-row warps). The ISA name stays inside this plugin.
    """

    if (
        gemm.m is None
        or gemm.m < _WGMMA_TILE_ROWS
        or gemm.b_scope not in {"shared", "shared.dyn"}
        or not _check_wgmma_dtypes(gemm)
    ):
        return False
    granule = HOPPER_RESOURCE.partition_warp_multiple
    return kernel_warps is not None and kernel_warps % granule == 0


def _gemm_axis_cover(gemm, granule: int, warpgroup: bool) -> int:
    """Minimum warps that cover this GEMM tile under its warp policy.

    FullRow follows M. FullCol follows the warpgroup (WGMMA) or a single warp
    (MMA); extra original-CTA warps along N come from ``kernel_warps``. Square
    takes the stricter of the two axes.
    """

    m = gemm.m
    if warpgroup:
        row = math.ceil(m / _WGMMA_TILE_ROWS) * granule if m else granule
        col = granule
    else:
        row = math.ceil(m / _MMA_TILE_ROWS) if m else 1
        col = 1
    policy = gemm.policy
    if policy == "full_col":
        return col
    if policy in ("square", "free"):
        return max(row, col)
    return row


def _gemm_required_warps(gemm, kernel_warps: int | None, granule: int) -> int:
    """Warps needed so the original CTA layout still covers this GEMM.

    OverlapPlan runs after LayoutReducer, so TileLang has already computed the
    two-dimensional ``m_warp * n_warp`` partition from the original CTA width
    and policy. Preserve that exact width. Re-deriving a lower bound from M
    would reject valid layouts where one warp or warpgroup loops over several
    instruction tiles; ignoring N would reject FullCol layouts.
    """

    if kernel_warps is not None:
        return kernel_warps
    warpgroup = _uses_warpgroup_tensorcore(gemm, kernel_warps)
    return _gemm_axis_cover(gemm, granule, warpgroup)


def _fragment_shape(buffer) -> tuple[int, ...] | None:
    shape = getattr(buffer.buffer, "shape", ())
    if not shape:
        return None
    dims = []
    for dim in shape:
        try:
            dims.append(int(dim))
        except (TypeError, ValueError):
            return None
    return tuple(dims)


def _related_gemms(graph: FactGraph, buffer_ids: set[int]):
    return tuple(
        node
        for node in graph.nodes
        if node.gemm is not None
        and buffer_ids.intersection((*node.reads, *node.writes))
    )


def _fragment_required_warps(
    graph: FactGraph,
    node: FactNode,
    granule: int,
    kernel_warps: int | None,
) -> int | None:
    """Warps needed to cover fragment layouts this compute node touches.

    LayoutReducer has already assigned fragment elements using the original
    CTA. Preserve that width for every group touching the fragment. Shape-based
    coverage remains only as a fallback for graphs without a known CTA width.
    """

    buffer_ids: set[int] = set()
    shapes: list[tuple[int, ...]] = []
    for buffer_id in (*node.reads, *node.writes):
        buffer = graph.buffer_for_id(buffer_id)
        if buffer.scope != "local.fragment":
            continue
        shape = _fragment_shape(buffer)
        if shape is None:
            continue
        buffer_ids.add(buffer_id)
        shapes.append(shape)
    if not shapes:
        return None
    if kernel_warps is not None:
        # The fragment layout is already reduced for the original CTA. Every
        # group that executes accesses to it must retain that thread domain.
        return kernel_warps
    related = _related_gemms(graph, buffer_ids)
    covers = [
        _gemm_required_warps(owner.gemm, kernel_warps, granule)
        for owner in related
        if owner.gemm is not None
    ]
    if covers:
        return max(covers)
    rows = max(shape[0] for shape in shapes)
    covers.append(math.ceil(rows / _WGMMA_TILE_ROWS) * granule)
    return max(covers)


def _partition_required_warps(
    classified: ClassifiedGraph,
    group_id: int,
    groups: Mapping[int, int],
    granule: int,
) -> int | None:
    """Warp width needed to preserve a compute partition's layout coverage.

    Fragment layouts are defined for the original CTA partition, including the
    policy-specific two-dimensional GEMM partition. Every compute group using
    one of those layouts keeps that width. Resource checks later reject
    combinations whose summed widths or register allocation do not fit.
    """

    requirements = []
    graph = classified.graph
    kernel_warps = _kernel_warps(graph)
    for node in graph.nodes:
        if groups[node.node_id] != group_id:
            continue
        if classified.traits_for(node.node_id).kind != ResourceKind.COMPUTE:
            continue
        if (
            classified.traits_for(node.node_id).occupies_cta_partition
            and node.gemm is not None
        ):
            requirements.append(
                _gemm_required_warps(node.gemm, kernel_warps, granule)
            )
        fragment = _fragment_required_warps(
            graph, node, granule, kernel_warps
        )
        if fragment is not None:
            requirements.append(fragment)
    if not requirements:
        return None
    return _align_up(max(requirements), granule)


def _group_warp_multiple(
    classified: ClassifiedGraph,
    group_id: int,
    groups: Mapping[int, int],
) -> int:
    """Return the collective width required by operations in one group.

    WGMMA and groups consuming a WGMMA-owned fragment execute in four-warp
    domains. A group whose only tensor-core collective is MMA has one-warp
    granularity. Other Hopper specialization groups retain the conservative
    warpgroup granularity used by the synchronization and register paths.
    """

    graph = classified.graph
    kernel_warps = _kernel_warps(graph)
    group_nodes = tuple(
        node for node in graph.nodes if groups[node.node_id] == group_id
    )
    gemms = tuple(node for node in group_nodes if node.gemm is not None)
    if any(
        _uses_warpgroup_tensorcore(node.gemm, kernel_warps)
        for node in gemms
        if node.gemm is not None
    ):
        return HOPPER_RESOURCE.partition_warp_multiple

    fragment_ids = {
        buffer_id
        for node in group_nodes
        for buffer_id in (*node.reads, *node.writes)
        if graph.buffer_for_id(buffer_id).scope == "local.fragment"
    }
    fragment_owners = _related_gemms(graph, fragment_ids)
    if any(
        owner.gemm is not None
        and _uses_warpgroup_tensorcore(owner.gemm, kernel_warps)
        for owner in fragment_owners
    ):
        return HOPPER_RESOURCE.partition_warp_multiple
    if gemms or fragment_ids:
        return 1
    return HOPPER_RESOURCE.partition_warp_multiple


def _warp_requirements(
    classified: ClassifiedGraph, structure: Structure
) -> tuple[WarpRequirement, ...]:
    resource = HOPPER_RESOURCE
    hardware_maximum = resource.max_warps_per_block
    groups = structure.groups
    requirements = []
    for group_id in range(structure.num_groups):
        traits = tuple(
            classified.traits_for(node.node_id)
            for node in classified.graph.nodes
            if groups[node.node_id] == group_id
        )
        memory_only = traits and all(
            item.kind == ResourceKind.MEMORY for item in traits
        )
        granule = (
            resource.partition_warp_multiple
            if memory_only
            else _group_warp_multiple(classified, group_id, groups)
        )
        required = _partition_required_warps(
            classified, group_id, groups, granule
        )
        if memory_only:
            minimum = granule
            maximum = granule
        else:
            minimum = granule
            maximum = hardware_maximum
            if required is not None:
                if required > 0:
                    minimum = max(minimum, required)
                    maximum = min(maximum, required)
                else:
                    maximum = 0
        requirements.append(WarpRequirement(minimum, maximum, granule))
    return tuple(requirements)


def _register_receivers(
    classified: ClassifiedGraph, structure: Structure
) -> frozenset[int]:
    return frozenset(
        structure.groups[node.node_id]
        for node in classified.graph.nodes
        if classified.traits_for(node.node_id).occupies_cta_partition
    )


def _buffer_alignment(classified: ClassifiedGraph, buffer_id: int) -> int:
    # MergeSharedMemoryAllocations falls back to 1024B for TMA / WGMMA
    # operands. O_shared is often a SIMT fragment store, so the old
    # async-only upgrade left it at 16B and compile rejected 32912.
    return max(
        default_buffer_alignment(classified.graph, buffer_id),
        _FALLBACK_ASYNC_ALIGNMENT,
    )


class HopperArch(Architecture):
    """Hopper plugin: classify facts, then realize warps and shared memory."""

    name = "hopper"

    def resource(self) -> DeviceResource:
        return HOPPER_RESOURCE

    def classify_node(self, graph: FactGraph, node: FactNode) -> NodeTraits:
        reads, writes = graph.scopes_for_node(node.node_id)
        if node.kind == OpKind.COPY:
            return _classify_copy(graph, node, reads, writes)
        if node.kind == OpKind.GEMM:
            return NodeTraits(
                ResourceKind.COMPUTE,
                "tensorcore",
                async_completion=node.tileop == "wgmma_gemm",
                occupies_cta_partition=True,
                issue_priority=5,
            )
        if node.kind == OpKind.ELEMENTWISE and any(
            op in _SFU_OPS for op in node.scalar_ops
        ):
            return NodeTraits(ResourceKind.COMPUTE, "sfu")
        return NodeTraits(ResourceKind.COMPUTE, "generic")

    def realize(
        self, classified: ClassifiedGraph, structure: Structure
    ) -> Iterator[PhysicalPlan]:
        resource = self.resource()
        requirements = _warp_requirements(classified, structure)
        if not warp_requirements_are_feasible(requirements):
            return
        shared_memory = analyze_shared_memory(
            classified.graph,
            structure,
            resource,
            alignment_for_buffer=lambda buffer_id: _buffer_alignment(
                classified, buffer_id
            ),
        )
        if not shared_memory.fits:
            return
        for allocation in enumerate_warp_allocations(
            classified,
            structure,
            resource,
            requirements,
            register_receivers=_register_receivers(classified, structure),
            setmaxnreg=_HOPPER_SETMAXNREG,
        ):
            yield PhysicalPlan(structure, allocation, shared_memory)


HOPPER = HopperArch()
