"""Tests for L2 physical realization."""

from dataclasses import replace

import tilelang.language as T

from OverlapPlaner.arch import FAKE, HOPPER
from OverlapPlaner.arch.hopper import _uses_warpgroup_tensorcore, _warp_requirements
from OverlapPlaner.facts import OpKind, extract_fact_graph
from OverlapPlaner.physical import analyze_shared_memory, estimate_group_registers_per_thread
from OverlapPlaner.structure import (
    SearchBudget,
    SynchronizationScope,
    enumerate_group_assignments,
    enumerate_structures,
)
from OverlapPlaner.structure.model import Structure
from OverlapPlaner.structure.order import build_program_orders
from OverlapPlaner.structure.stage import (
    effective_stage_distance,
    enumerate_program_stages,
)
from OverlapPlaner.structure.sync import build_synchronizations
from OverlapPlaner.structure.version import analyze_buffer_versions
from OverlapPlaner.tune.operators.fa3 import build as build_fa3
from OverlapPlaner.tune.operators.gemm import build as build_gemm


def _gemm_graph():
    return extract_fact_graph(
        build_gemm(
            {"gemm_m": 256, "gemm_n": 256, "gemm_k": 256},
            {"block_m": 128, "block_n": 128, "block_k": 64},
        ).prim_func
    )


def _fa3_graph(block_m: int = 128, block_n: int = 128):
    return extract_fact_graph(
        build_fa3(
            {
                "fa3_batch": 1,
                "fa3_heads": 1,
                "fa3_seq_q": 256,
                "fa3_seq_kv": 256,
                "fa3_dim": 128,
                "fa3_causal": False,
            },
            {"block_m": block_m, "block_n": block_n},
        ).prim_func
    )


def _plans(arch, graph, **budget):
    classified = arch.classify(graph)
    plans = []
    for structure in enumerate_structures(
        classified, SearchBudget(**budget), arch=arch
    ):
        plans.extend(arch.realize(classified, structure))
    return classified, plans


def _fa3_qk_and_softmax(classified):
    graph = classified.graph
    qk = next(
        node
        for node in graph.nodes
        if node.kind == OpKind.GEMM and node.gemm is not None and node.gemm.transpose_b
    )
    softmax = next(
        node
        for node in graph.nodes
        if classified.traits_for(node.node_id).engine == "sfu"
    )
    return qk, softmax


def _structure_for_groups(classified, groups):
    stages = next(
        enumerate_program_stages(
            classified, SearchBudget(max_stages=1)
        )
    )
    orders = build_program_orders(classified, stages, groups)
    versions = analyze_buffer_versions(
        classified.graph, stages, groups, orders
    )
    sync_edges = build_synchronizations(
        classified, stages, groups, orders, versions
    )
    return Structure(stages, groups, orders, versions, sync_edges)


def test_hopper_one_group_keeps_kernel_threads() -> None:
    graph = _gemm_graph()
    classified, plans = _plans(HOPPER, graph, max_structures=16)
    one_group = [item for item in plans if item.structure.num_groups == 1]
    assert one_group
    for item in one_group:
        assert item.warp_allocation.effective_threads == graph.kernel_threads
        assert item.warp_allocation.total_warps == graph.kernel_threads // 32
        assert not item.warp_allocation.setmaxnreg_enabled
        assert item.shared_memory.fits


def test_hopper_multi_group_uses_partition_granule_and_registers() -> None:
    classified, plans = _plans(HOPPER, _gemm_graph(), max_structures=64)
    multi = [item for item in plans if item.structure.num_groups >= 2]
    assert multi
    gemm = next(node for node in classified.graph.nodes if node.kind == OpKind.GEMM)
    for item in multi:
        allocation = item.warp_allocation
        assert all(group.warp_count % 4 == 0 for group in allocation.groups)
        assert allocation.groups[0].first_warp == 0
        for previous, current in zip(allocation.groups, allocation.groups[1:]):
            assert current.first_warp == previous.warp_stop
        assert allocation.setmaxnreg_enabled
        compute_group = item.structure.groups[gemm.node_id]
        assert allocation.groups[compute_group].warp_count == 8
        assert item.shared_memory.fits


def test_hopper_async_shared_buffers_use_wide_alignment() -> None:
    classified, plans = _plans(HOPPER, _gemm_graph(), max_structures=4)
    assert plans
    graph = classified.graph
    copy = next(
        node
        for node in graph.nodes
        if node.kind == OpKind.COPY
        and classified.traits_for(node.node_id).async_completion
    )
    shared_ids = [
        buffer_id
        for buffer_id in copy.writes
        if graph.buffer_for_id(buffer_id).scope.startswith("shared")
    ]
    assert shared_ids
    offsets = {
        item.buffer_id: item
        for item in plans[0].shared_memory.shared_allocations
    }
    for buffer_id in shared_ids:
        assert offsets[buffer_id].alignment == 1024
        assert offsets[buffer_id].byte_offset % 1024 == 0


def test_physical_plan_has_no_isa_names() -> None:
    _, plans = _plans(HOPPER, _gemm_graph(), max_structures=1)
    dumped = repr(plans[0])
    assert "wgmma" not in dumped
    assert "tma" not in dumped


def test_hopper_shared_capacity_uses_driver_query() -> None:
    from OverlapPlaner.arch import HOPPER_CUDA_TARGET, HOPPER_RESOURCE
    from OverlapPlaner.arch.hopper import _HOPPER_SHARED_MEMORY_FALLBACK_BYTES
    from tilelang.carver.arch.driver import get_max_dynamic_shared_size_bytes

    queried = get_max_dynamic_shared_size_bytes()
    expected = queried or _HOPPER_SHARED_MEMORY_FALLBACK_BYTES
    assert HOPPER.resource() is HOPPER_RESOURCE
    assert HOPPER.resource().shared_memory_capacity_bytes == expected
    assert int(HOPPER_CUDA_TARGET.attrs["max_shared_memory_per_block"]) == (
        HOPPER_RESOURCE.shared_memory_capacity_bytes
    )


def test_fa3_realize_fits_and_uses_contiguous_warps() -> None:
    _, plans = _plans(HOPPER, _fa3_graph(), max_structures=256)
    assert plans
    assert any(item.structure.num_groups >= 2 for item in plans)
    for item in plans:
        assert item.shared_memory.fits
        assert item.warp_allocation.total_warps <= 32
        assert all(
            placed.byte_offset % placed.alignment == 0
            for placed in item.shared_memory.shared_allocations
        )
        assert all(
            placed.byte_offset % placed.alignment == 0
            for placed in item.shared_memory.handoff_allocations
        )


def test_hopper_non_tensorcore_groups_stay_at_partition_granule() -> None:
    classified, plans = _plans(HOPPER, _fa3_graph(), max_structures=256)
    multi = [item for item in plans if item.structure.num_groups >= 2]
    assert multi
    gemm_ids = {
        node.node_id
        for node in classified.graph.nodes
        if node.kind == OpKind.GEMM
    }
    for item in multi:
        gemm_groups = {item.structure.groups[node_id] for node_id in gemm_ids}
        for group in item.warp_allocation.groups:
            if group.group_id in gemm_groups:
                assert group.warp_count == 8
            else:
                assert group.warp_count == 4
        assert item.warp_allocation.effective_threads <= 512


def _one_group_requirement(graph):
    classified = HOPPER.classify(graph)
    groups = next(enumerate_group_assignments(classified, 1))
    structure = _structure_for_groups(classified, groups)
    return classified, _warp_requirements(classified, structure)[0]


@T.prim_func(auto_overlap=True)
def _gemm_fullcol_n256(
    A: T.Tensor((64, 64), T.float16),
    B: T.Tensor((64, 256), T.float16),
    C: T.Tensor((64, 256), T.float16),
):
    with T.Kernel(1, 1, threads=256) as (_bx, _by):
        a_shared = T.alloc_shared((64, 64), T.float16)
        b_shared = T.alloc_shared((64, 256), T.float16)
        c_local = T.alloc_fragment((64, 256), T.float32)
        T.clear(c_local)
        for _k in T.Pipelined(1):
            T.copy(A, a_shared)
            T.copy(B, b_shared)
            T.gemm(
                a_shared,
                b_shared,
                c_local,
                policy=T.GemmWarpPolicy.FullCol,
            )
        T.copy(c_local, C)


@T.prim_func(auto_overlap=True)
def _gemm_fullrow_m64(
    A: T.Tensor((64, 64), T.float16),
    B: T.Tensor((64, 256), T.float16),
    C: T.Tensor((64, 256), T.float16),
):
    with T.Kernel(1, 1, threads=128) as (_bx, _by):
        a_shared = T.alloc_shared((64, 64), T.float16)
        b_shared = T.alloc_shared((64, 256), T.float16)
        c_local = T.alloc_fragment((64, 256), T.float32)
        T.clear(c_local)
        for _k in T.Pipelined(1):
            T.copy(A, a_shared)
            T.copy(B, b_shared)
            T.gemm(
                a_shared,
                b_shared,
                c_local,
                policy=T.GemmWarpPolicy.FullRow,
            )
        T.copy(c_local, C)


@T.prim_func(auto_overlap=True)
def _gemm_mma_m32(
    A: T.Tensor((32, 32), T.float16),
    B: T.Tensor((32, 64), T.float16),
    C: T.Tensor((32, 64), T.float16),
):
    with T.Kernel(1, 1, threads=128) as (_bx, _by):
        a_shared = T.alloc_shared((32, 32), T.float16)
        b_shared = T.alloc_shared((32, 64), T.float16)
        c_local = T.alloc_fragment((32, 64), T.float32)
        T.clear(c_local)
        for _k in T.Pipelined(1):
            T.copy(A, a_shared)
            T.copy(B, b_shared)
            T.gemm(
                a_shared,
                b_shared,
                c_local,
                policy=T.GemmWarpPolicy.FullRow,
            )
        T.copy(c_local, C)


@T.prim_func(auto_overlap=True)
def _gemm_mma_fullcol(
    A: T.Tensor((32, 32), T.float16),
    B: T.Tensor((32, 128), T.float16),
    C: T.Tensor((32, 128), T.float16),
):
    with T.Kernel(1, 1, threads=128) as (_bx, _by):
        a_shared = T.alloc_shared((32, 32), T.float16)
        b_shared = T.alloc_shared((32, 128), T.float16)
        c_local = T.alloc_fragment((32, 128), T.float32)
        T.clear(c_local)
        for _k in T.Pipelined(1):
            T.copy(A, a_shared)
            T.copy(B, b_shared)
            T.gemm(
                a_shared,
                b_shared,
                c_local,
                policy=T.GemmWarpPolicy.FullCol,
            )
        T.copy(c_local, C)


def test_hopper_fullcol_keeps_n_side_warpgroups() -> None:
    graph = extract_fact_graph(_gemm_fullcol_n256)
    classified, requirement = _one_group_requirement(graph)
    gemm = next(node for node in graph.nodes if node.kind == OpKind.GEMM)
    assert gemm.gemm is not None
    assert gemm.gemm.policy == "full_col"
    assert gemm.gemm.m == 64
    assert graph.kernel_threads == 256
    assert requirement.minimum == 8
    assert requirement.maximum == 8
    split = next(enumerate_group_assignments(classified, 2), None)
    assert split is not None
    structure = _structure_for_groups(classified, split)
    gemm_requirement = _warp_requirements(classified, structure)[
        structure.groups[gemm.node_id]
    ]
    assert gemm_requirement.minimum == 8
    assert gemm_requirement.maximum == 8
    _, plans = _plans(HOPPER, graph, max_structures=32)
    assert plans
    for item in plans:
        compute = item.structure.groups[gemm.node_id]
        assert item.warp_allocation.groups[compute].warp_count == 8


def test_hopper_fullrow_m64_stays_one_warpgroup() -> None:
    graph = extract_fact_graph(_gemm_fullrow_m64)
    _, requirement = _one_group_requirement(graph)
    gemm = next(node for node in graph.nodes if node.kind == OpKind.GEMM)
    assert gemm.gemm is not None
    assert gemm.gemm.policy == "full_row"
    assert requirement.minimum == 4
    assert requirement.maximum == 4


def test_hopper_mma_small_m_uses_sixteen_row_tiles() -> None:
    graph = extract_fact_graph(_gemm_mma_m32)
    classified, requirement = _one_group_requirement(graph)
    gemm = next(node for node in graph.nodes if node.kind == OpKind.GEMM)
    assert gemm.gemm is not None
    assert gemm.gemm.m == 32
    assert classified.traits_for(gemm.node_id).engine == "tensorcore"
    assert requirement.minimum == 4
    assert requirement.maximum == 4


def test_hopper_mma_fullcol_does_not_use_n_over_eight() -> None:
    graph = extract_fact_graph(_gemm_mma_fullcol)
    _, requirement = _one_group_requirement(graph)
    gemm = next(node for node in graph.nodes if node.kind == OpKind.GEMM)
    assert gemm.gemm is not None
    assert gemm.gemm.policy == "full_col"
    assert gemm.gemm.n == 128
    assert requirement.minimum == 4
    assert requirement.maximum == 4


@T.prim_func(auto_overlap=True)
def _gemm_mma_one_warp(
    A: T.Tensor((16, 16), T.float16),
    B: T.Tensor((16, 8), T.float16),
    C: T.Tensor((16, 8), T.float16),
):
    with T.Kernel(1, threads=32):
        a_shared = T.alloc_shared((16, 16), T.float16)
        b_shared = T.alloc_shared((16, 8), T.float16)
        c_local = T.alloc_fragment((16, 8), T.float32)
        T.copy(A, a_shared)
        T.copy(B, b_shared)
        T.clear(c_local)
        T.gemm(
            a_shared,
            b_shared,
            c_local,
            policy=T.GemmWarpPolicy.FullRow,
        )
        T.copy(c_local, C)


@T.prim_func(auto_overlap=True)
def _gemm_fullcol_m64_n512(
    A: T.Tensor((64, 64), T.float16),
    B: T.Tensor((64, 512), T.float16),
    C: T.Tensor((64, 512), T.float16),
):
    with T.Kernel(1, threads=256):
        a_shared = T.alloc_shared((64, 64), T.float16)
        b_shared = T.alloc_shared((64, 512), T.float16)
        c_local = T.alloc_fragment((64, 512), T.float32)
        T.copy(A, a_shared)
        T.copy(B, b_shared)
        T.clear(c_local)
        T.gemm(
            a_shared,
            b_shared,
            c_local,
            policy=T.GemmWarpPolicy.FullCol,
        )
        T.copy(c_local, C)


@T.prim_func(auto_overlap=True)
def _gemm_fullrow_m128_one_warpgroup(
    A: T.Tensor((128, 64), T.float16),
    B: T.Tensor((64, 64), T.float16),
    C: T.Tensor((128, 64), T.float32),
):
    with T.Kernel(1, threads=128):
        a_shared = T.alloc_shared((128, 64), T.float16)
        b_shared = T.alloc_shared((64, 64), T.float16)
        c_local = T.alloc_fragment((128, 64), T.float32)
        T.copy(A, a_shared)
        T.copy(B, b_shared)
        T.clear(c_local)
        T.gemm(
            a_shared,
            b_shared,
            c_local,
            policy=T.GemmWarpPolicy.FullRow,
        )
        T.copy(c_local, C)


@T.prim_func(auto_overlap=True)
def _mixed_wgmma_mma(
    A0: T.Tensor((64, 64), T.float16),
    B0: T.Tensor((64, 32), T.float16),
    A1: T.Tensor((32, 64), T.float16),
    B1: T.Tensor((64, 128), T.float16),
    C0: T.Tensor((64, 32), T.float32),
    C1: T.Tensor((32, 128), T.float32),
):
    with T.Kernel(1, threads=128):
        a0_shared = T.alloc_shared((64, 64), T.float16)
        b0_shared = T.alloc_shared((64, 32), T.float16)
        a1_shared = T.alloc_shared((32, 64), T.float16)
        b1_shared = T.alloc_shared((64, 128), T.float16)
        c0_local = T.alloc_fragment((64, 32), T.float32)
        c1_local = T.alloc_fragment((32, 128), T.float32)
        T.copy(A0, a0_shared)
        T.copy(B0, b0_shared)
        T.copy(A1, a1_shared)
        T.copy(B1, b1_shared)
        T.clear(c0_local)
        T.clear(c1_local)
        T.gemm(a0_shared, b0_shared, c0_local)
        T.gemm(a1_shared, b1_shared, c1_local)
        T.copy(c0_local, C0)
        T.copy(c1_local, C1)


def test_hopper_mma_allows_one_warp_requirement() -> None:
    graph = extract_fact_graph(_gemm_mma_one_warp)
    _, requirement = _one_group_requirement(graph)
    assert requirement.minimum == 1
    assert requirement.maximum == 1
    assert requirement.multiple == 1
    _, plans = _plans(HOPPER, graph, max_groups=1, max_structures=8)
    assert plans
    assert all(item.warp_allocation.total_warps == 1 for item in plans)


def test_hopper_fullcol_mla_width_keeps_n_side_warpgroups() -> None:
    graph = extract_fact_graph(_gemm_fullcol_m64_n512)
    _, requirement = _one_group_requirement(graph)
    assert requirement.minimum == 8
    assert requirement.maximum == 8
    _, plans = _plans(HOPPER, graph, max_groups=1, max_structures=8)
    assert plans
    assert all(item.warp_allocation.total_warps == 8 for item in plans)


def test_hopper_uses_layout_partition_instead_of_row_count() -> None:
    graph = extract_fact_graph(_gemm_fullrow_m128_one_warpgroup)
    _, requirement = _one_group_requirement(graph)
    assert requirement.minimum == 4
    assert requirement.maximum == 4
    _, plans = _plans(HOPPER, graph, max_groups=1, max_structures=8)
    assert plans


def test_hopper_mixed_wgmma_and_mma_remains_feasible() -> None:
    graph = extract_fact_graph(_mixed_wgmma_mma)
    gemms = [node for node in graph.nodes if node.kind == OpKind.GEMM]
    assert len(gemms) == 2
    wide = next(
        node for node in gemms if node.gemm is not None and node.gemm.m == 64
    )
    small = next(
        node for node in gemms if node.gemm is not None and node.gemm.m == 32
    )
    assert wide.gemm is not None and small.gemm is not None
    assert _uses_warpgroup_tensorcore(wide.gemm, 4)
    assert not _uses_warpgroup_tensorcore(small.gemm, 4)
    _, requirement = _one_group_requirement(graph)
    assert requirement.minimum == 4
    assert requirement.maximum == 4
    _, plans = _plans(HOPPER, graph, max_groups=1, max_structures=8)
    assert plans


def test_hopper_wgmma_selection_checks_scope_dtype_k_and_transpose() -> None:
    graph = extract_fact_graph(_gemm_fullcol_n256)
    gemm = next(node.gemm for node in graph.nodes if node.gemm is not None)
    assert _uses_warpgroup_tensorcore(gemm, 8)
    assert not _uses_warpgroup_tensorcore(
        replace(gemm, b_scope="local.fragment"), 8
    )
    assert not _uses_warpgroup_tensorcore(replace(gemm, k=15), 8)
    assert not _uses_warpgroup_tensorcore(replace(gemm, a_dtype="float32"), 8)
    fp8 = replace(
        gemm,
        a_dtype="float8_e4m3fn",
        b_dtype="float8_e4m3fn",
        transpose_a=False,
        transpose_b=True,
        k=32,
    )
    assert _uses_warpgroup_tensorcore(fp8, 8)
    assert not _uses_warpgroup_tensorcore(replace(fp8, transpose_b=False), 8)


def test_hopper_compute_group_preserves_128_row_fragment_coverage() -> None:
    classified = HOPPER.classify(_fa3_graph(128, 64))
    graph = classified.graph
    fragment_copy = next(
        node
        for node in graph.nodes
        if node.kind == OpKind.COPY
        and classified.traits_for(node.node_id).engine == "register_copy"
        and any(
            graph.buffer_for_id(buffer_id).name == "acc_s_cast"
            for buffer_id in node.writes
        )
    )
    groups = next(
        item
        for item in enumerate_group_assignments(classified, 2)
        if item[fragment_copy.node_id]
        != item[
            next(
                node.node_id
                for node in graph.nodes
                if node.kind == OpKind.GEMM
            )
        ]
    )
    structure = _structure_for_groups(classified, groups)
    requirement = _warp_requirements(classified, structure)[
        groups[fragment_copy.node_id]
    ]
    assert requirement.minimum == 8
    assert requirement.maximum == 8


def test_fa3_shared_offsets_are_tma_aligned() -> None:
    _, plans = _plans(HOPPER, _fa3_graph(), max_structures=32)
    assert plans
    for item in plans:
        for placed in item.shared_memory.shared_allocations:
            assert placed.alignment == 1024
            assert placed.byte_offset % 1024 == 0


def test_fa3_ws_does_not_alias_pipeline_and_epilogue_shared() -> None:
    _, plans = _plans(
        HOPPER, _fa3_graph(64, 128), max_groups=2, max_structures=128
    )
    multi = [item for item in plans if item.structure.num_groups >= 2]
    assert multi
    for item in multi:
        offsets = {
            placed.name: placed.byte_offset
            for placed in item.shared_memory.shared_allocations
        }
        assert "V_shared" in offsets and "O_shared" in offsets
        assert offsets["V_shared"] != offsets["O_shared"]


def test_fa3_cross_group_prefix_q_aliases_epilogue_o() -> None:
    classified = HOPPER.classify(_fa3_graph(128, 128))
    graph = classified.graph
    producer_names = {
        "copy_Q_to_Q_shared",
        "copy_K_to_K_shared",
        "copy_V_to_V_shared",
    }
    groups = {
        node.node_id: int(node.name in producer_names) for node in graph.nodes
    }
    structure = _structure_for_groups(classified, groups)
    smem = analyze_shared_memory(graph, structure, HOPPER.resource())
    offsets = {
        placed.name: placed.byte_offset for placed in smem.shared_allocations
    }
    assert offsets["Q_shared"] == offsets["O_shared"]
    # In-loop async writers remain conservative until an iteration-aware
    # completion proof is available.
    assert offsets["K_shared"] != offsets["O_shared"]
    assert offsets["V_shared"] != offsets["O_shared"]


def test_fa3_one_group_has_no_fragment_handoff() -> None:
    _, plans = _plans(HOPPER, _fa3_graph(), max_groups=1, max_structures=8)
    one_group = [item for item in plans if item.structure.num_groups == 1]
    assert one_group
    for item in one_group:
        assert item.shared_memory.handoff_allocations == ()


def test_fa3_cross_group_fragment_handoff_is_in_shared_plan() -> None:
    classified = HOPPER.classify(_fa3_graph())
    qk, softmax = _fa3_qk_and_softmax(classified)
    groups = next(
        item
        for item in enumerate_group_assignments(classified, 2)
        if item[qk.node_id] != item[softmax.node_id]
    )
    structure = _structure_for_groups(classified, groups)
    smem = analyze_shared_memory(
        classified.graph, structure, HOPPER.resource()
    )
    assert smem.handoff_allocations
    assert all(item.size_bytes >= 1 for item in smem.handoff_allocations)
    assert smem.shared_buffer_bytes >= sum(
        item.size_bytes for item in smem.handoff_allocations
    )
    assert smem.merged_shared_bytes >= smem.shared_buffer_bytes


def test_cross_group_fragment_stage_delay_uses_two_handoff_slots() -> None:
    classified = HOPPER.classify(_fa3_graph())
    graph = classified.graph
    acc_s = next(
        buffer
        for buffer in graph.buffers
        if buffer.name == "acc_s" and buffer.scope == "local.fragment"
    )
    found = False
    for groups in enumerate_group_assignments(classified, 2):
        delayed_edges = [
            edge
            for edge in graph.edges
            if edge.buffer_id == acc_s.buffer_id
            and groups[edge.producer_id] != groups[edge.consumer_id]
            and graph.node_for_id(edge.producer_id).region_id
            == graph.node_for_id(edge.consumer_id).region_id
        ]
        if not delayed_edges:
            continue
        for stages_by_region in enumerate_program_stages(
            classified, SearchBudget(max_stages=2)
        ):
            region_id = graph.node_for_id(delayed_edges[0].producer_id).region_id
            stages = stages_by_region[region_id]
            if not any(
                stages[edge.consumer_id] > stages[edge.producer_id]
                for edge in delayed_edges
            ):
                continue
            try:
                orders = build_program_orders(
                    classified, stages_by_region, groups
                )
            except ValueError:
                continue
            versions = analyze_buffer_versions(
                graph, stages_by_region, groups, orders
            )
            assert versions[acc_s.buffer_id] >= 2
            structure = Structure(
                stages_by_region,
                groups,
                orders,
                versions,
                build_synchronizations(
                    classified, stages_by_region, groups, orders, versions
                ),
            )
            delayed = [
                edge
                for edge in structure.sync_edges
                if edge.buffer_id == acc_s.buffer_id
                and edge.scope == SynchronizationScope.PER_ITERATION
                and stages[edge.consumer_id] > stages[edge.producer_id]
            ]
            if not delayed:
                continue
            assert all(edge.slot_count >= 2 for edge in delayed)
            smem = analyze_shared_memory(graph, structure, HOPPER.resource())
            packed = [
                item
                for item in smem.handoff_allocations
                if item.source_buffer_id == acc_s.buffer_id
            ]
            assert packed
            assert acc_s.nbytes is not None
            assert any(item.size_bytes >= acc_s.nbytes * 2 for item in packed)
            found = True
            break
        if found:
            break
    assert found


def test_loop_carried_fragment_handoff_counts_iteration_distance() -> None:
    classified = HOPPER.classify(_fa3_graph(64, 64))
    graph = classified.graph
    acc_o = next(
        buffer
        for buffer in graph.buffers
        if buffer.name == "acc_o" and buffer.scope == "local.fragment"
    )
    loop_edge = next(
        edge
        for edge in graph.edges
        if edge.buffer_id == acc_o.buffer_id and edge.is_loop_carried
    )
    groups = next(
        item
        for item in enumerate_group_assignments(classified, 2)
        if item[loop_edge.producer_id] != item[loop_edge.consumer_id]
    )
    structure = _structure_for_groups(classified, groups)
    region_id = graph.node_for_id(loop_edge.producer_id).region_id
    distance = effective_stage_distance(
        loop_edge, structure.stages_by_region[region_id]
    )
    sync = next(
        edge
        for edge in structure.sync_edges
        if edge.buffer_id == acc_o.buffer_id
        and edge.producer_id == loop_edge.producer_id
        and edge.consumer_id == loop_edge.consumer_id
    )
    assert distance == 1
    assert sync.slot_count == distance + 1
    assert structure.buffer_versions[acc_o.buffer_id] >= distance + 1

    smem = analyze_shared_memory(graph, structure, HOPPER.resource())
    handoff = next(
        item
        for item in smem.handoff_allocations
        if item.channel_id == structure.sync_edges.index(sync)
    )
    assert acc_o.nbytes is not None
    assert handoff.size_bytes == acc_o.nbytes * sync.slot_count


def test_fa3_fragment_split_counts_registers_per_group() -> None:
    classified = HOPPER.classify(_fa3_graph())
    qk, softmax = _fa3_qk_and_softmax(classified)
    groups = next(
        item
        for item in enumerate_group_assignments(classified, 2)
        if item[qk.node_id] != item[softmax.node_id]
    )
    structure = _structure_for_groups(classified, groups)
    qk_group = structure.groups[qk.node_id]
    softmax_group = structure.groups[softmax.node_id]
    resource = HOPPER.resource()
    qk_registers = estimate_group_registers_per_thread(
        classified.graph,
        qk_group,
        structure.groups,
        structure.orders,
        structure.buffer_versions,
        8,
        resource,
    )
    softmax_registers = estimate_group_registers_per_thread(
        classified.graph,
        softmax_group,
        structure.groups,
        structure.orders,
        structure.buffer_versions,
        8,
        resource,
    )
    assert qk_registers > 0
    assert softmax_registers > 0
    requirements = _warp_requirements(classified, structure)
    assert requirements[softmax_group].minimum == 8
    assert requirements[softmax_group].maximum == 8
    assert requirements[qk_group].minimum == 8
    assert requirements[qk_group].maximum == 8
    list(HOPPER.realize(classified, structure))


def test_fragment_versions_scale_register_estimate() -> None:
    classified = HOPPER.classify(_fa3_graph())
    graph = classified.graph
    acc_s = next(
        buffer
        for buffer in graph.buffers
        if buffer.name == "acc_s" and buffer.scope == "local.fragment"
    )
    qk, softmax = _fa3_qk_and_softmax(classified)
    groups = next(enumerate_group_assignments(classified, 1))
    stages_by_region = None
    orders = None
    versions = None
    for candidate in enumerate_program_stages(
        classified, SearchBudget(max_stages=2)
    ):
        stages = candidate[qk.region_id]
        if stages[qk.node_id] >= stages[softmax.node_id]:
            continue
        try:
            candidate_orders = build_program_orders(classified, candidate, groups)
        except ValueError:
            continue
        candidate_versions = analyze_buffer_versions(
            graph, candidate, groups, candidate_orders
        )
        if candidate_versions[acc_s.buffer_id] < 2:
            continue
        stages_by_region = candidate
        orders = candidate_orders
        versions = candidate_versions
        break
    assert versions is not None
    resource = HOPPER.resource()
    scaled = estimate_group_registers_per_thread(
        graph, 0, groups, orders, versions, 8, resource
    )
    unversioned = dict(versions)
    unversioned[acc_s.buffer_id] = 1
    baseline = estimate_group_registers_per_thread(
        graph, 0, groups, orders, unversioned, 8, resource
    )
    assert scaled > baseline
    assert stages_by_region is not None


def test_smem_overflow_is_not_realized() -> None:
    classified = HOPPER.classify(_gemm_graph())
    structure = next(enumerate_structures(classified, SearchBudget(max_structures=1)))
    tiny = replace(HOPPER.resource(), shared_memory_capacity_bytes=1)
    original = type(HOPPER).resource
    type(HOPPER).resource = lambda self: tiny
    try:
        assert list(HOPPER.realize(classified, structure)) == []
    finally:
        type(HOPPER).resource = original


def test_fake_arch_realizes_one_group() -> None:
    graph = _gemm_graph()
    _, plans = _plans(FAKE, graph, max_structures=8)
    one_group = [item for item in plans if item.structure.num_groups == 1]
    assert one_group
    assert one_group[0].warp_allocation.effective_threads == graph.kernel_threads
    assert one_group[0].shared_memory.fits
