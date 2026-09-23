"""Tests for architecture trait classification."""

from dataclasses import replace

import tilelang.language as T

from OverlapPlaner.arch import (
    ClassifiedGraph,
    EdgeAction,
    FAKE,
    HOPPER,
    ResourceKind,
    allowed_edge_actions,
    can_split_stages,
    is_group_opportunity,
)
from OverlapPlaner.arch.hopper import optional_tma_copy_ids
from OverlapPlaner.contract import layout_reduced_prim_func
from OverlapPlaner.facts import DependencyKind, OpKind, RegionKind, extract_fact_graph
from OverlapPlaner.tune.operators.fa3 import build as build_fa3
from OverlapPlaner.tune.operators.gemm import build as build_gemm


def _gemm_graph():
    return extract_fact_graph(
        build_gemm(
            {"gemm_m": 256, "gemm_n": 256, "gemm_k": 256},
            {"block_m": 128, "block_n": 128, "block_k": 64},
        ).prim_func
    )


def _fa3_graph():
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
            {"block_m": 128, "block_n": 128},
        ).prim_func
    )


def test_hopper_classifies_copy_and_gemm_without_isa_names() -> None:
    classified = HOPPER.classify(_gemm_graph())
    engines = {traits.engine for traits in classified.traits}
    assert "wgmma" not in engines
    assert "tma" not in engines
    assert "tensorcore" in engines
    assert "copy" in engines

    gemm = next(
        node for node in classified.graph.nodes if node.kind == OpKind.GEMM
    )
    gemm_traits = classified.traits_for(gemm.node_id)
    assert gemm_traits.kind == ResourceKind.COMPUTE
    assert gemm_traits.engine == "tensorcore"
    assert gemm_traits.occupies_cta_partition is True
    assert gemm_traits.async_completion is False

    copies = [
        classified.traits_for(node.node_id)
        for node in classified.graph.nodes
        if node.kind == OpKind.COPY
    ]
    assert any(
        traits.kind == ResourceKind.MEMORY and traits.async_completion
        for traits in copies
    )


def test_explicit_copy_backend_is_not_a_search_alternative() -> None:
    @T.prim_func(auto_overlap=True)
    def kernel(A: T.Tensor((64,), T.float16), B: T.Tensor((64,), T.float16)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64,), T.float16)
            T.copy(A, shared, prefer_instruction="tma")
            T.copy(shared, B)

    classified = HOPPER.classify(
        extract_fact_graph(layout_reduced_prim_func(kernel))
    )
    assert optional_tma_copy_ids(classified) == ()


def test_hopper_classifies_fa3_softmax_as_sfu() -> None:
    classified = HOPPER.classify(_fa3_graph())
    assert any(traits.engine == "sfu" for traits in classified.traits)
    gemms = [
        classified.traits_for(node.node_id)
        for node in classified.graph.nodes
        if node.kind == OpKind.GEMM
    ]
    assert len(gemms) == 2
    assert all(traits.engine == "tensorcore" for traits in gemms)
    assert not can_split_stages(gemms[0], gemms[1])


def test_copy_and_gemm_may_split_stage_and_group() -> None:
    classified = HOPPER.classify(_fa3_graph())
    graph = classified.graph
    qk = next(
        node
        for node in graph.nodes
        if node.kind == OpKind.GEMM and node.gemm is not None and node.gemm.transpose_b
    )
    edge = next(
        item
        for item in graph.edges
        if item.consumer_id == qk.node_id
        and graph.node_for_id(item.producer_id).kind == OpKind.COPY
    )
    producer = classified.traits_for(edge.producer_id)
    consumer = classified.traits_for(edge.consumer_id)
    assert producer.kind == ResourceKind.MEMORY
    assert producer.async_completion is True
    assert consumer.engine == "tensorcore"
    assert can_split_stages(producer, consumer)
    assert allowed_edge_actions(classified, edge) == frozenset(
        {EdgeAction.KEEP, EdgeAction.SPLIT_STAGE, EdgeAction.SPLIT_GROUP}
    )


def test_fill_consumer_edge_cannot_split() -> None:
    classified = HOPPER.classify(_gemm_graph())
    graph = classified.graph
    fill = next(node for node in graph.nodes if node.kind == OpKind.FILL)
    edge = next(
        item for item in graph.edges if item.producer_id == fill.node_id
    )
    assert allowed_edge_actions(classified, edge) == frozenset({EdgeAction.KEEP})
    assert not is_group_opportunity(classified, edge)


def test_engine_crossing_fragment_may_split_group() -> None:
    classified = HOPPER.classify(_fa3_graph())
    graph = classified.graph
    qk = next(
        node
        for node in graph.nodes
        if node.kind == OpKind.GEMM and node.gemm is not None and node.gemm.transpose_b
    )
    edge = next(
        item
        for item in graph.edges
        if item.producer_id == qk.node_id
        and DependencyKind.RAW in item.dependency_kinds
        and graph.buffer_for_id(item.buffer_id).scope == "local.fragment"
        and classified.traits_for(item.consumer_id).engine == "sfu"
    )
    producer = classified.traits_for(edge.producer_id)
    consumer = classified.traits_for(edge.consumer_id)
    assert producer.engine == "tensorcore"
    assert consumer.engine == "sfu"
    assert is_group_opportunity(classified, edge)
    assert allowed_edge_actions(classified, edge) == frozenset(
        {EdgeAction.KEEP, EdgeAction.SPLIT_STAGE, EdgeAction.SPLIT_GROUP}
    )


def test_serial_shared_output_store_can_split_group() -> None:
    classified = HOPPER.classify(_fa3_graph())
    graph = classified.graph
    serial_shared = [
        item
        for item in graph.edges
        if item.producer_id != item.consumer_id
        and graph.region_kinds[graph.node_for_id(item.producer_id).region_id]
        != RegionKind.PIPELINE
        and graph.region_kinds[graph.node_for_id(item.consumer_id).region_id]
        != RegionKind.PIPELINE
        and graph.buffer_for_id(item.buffer_id).scope.startswith("shared")
    ]
    assert serial_shared
    for edge in serial_shared:
        assert is_group_opportunity(classified, edge)
        assert EdgeAction.SPLIT_GROUP in allowed_edge_actions(classified, edge)


def test_serial_visible_edges_follow_overlaper_group_constraint() -> None:
    classified = HOPPER.classify(_fa3_graph())
    graph = classified.graph
    edge = next(
        item for item in graph.edges
        if item.producer_id != item.consumer_id
        and graph.region_kinds[graph.node_for_id(item.producer_id).region_id]
        == RegionKind.SERIAL
        and graph.region_kinds[graph.node_for_id(item.consumer_id).region_id]
        == RegionKind.SERIAL
        and graph.buffer_for_id(item.buffer_id).scope.startswith("shared")
    )
    for scope in ("", "global", "shared.dyn", "tmem"):
        buffers = list(graph.buffers)
        buffers[edge.buffer_id] = replace(buffers[edge.buffer_id], scope=scope)
        scoped = ClassifiedGraph(replace(graph, buffers=tuple(buffers)), classified.traits)
        for kind in DependencyKind:
            hazard = replace(edge, dependency_kinds=frozenset({kind}))
            assert is_group_opportunity(scoped, hazard)
            assert EdgeAction.SPLIT_GROUP in allowed_edge_actions(scoped, hazard)

    buffers[edge.buffer_id] = replace(buffers[edge.buffer_id], scope="local.fragment")
    private = ClassifiedGraph(replace(graph, buffers=tuple(buffers)), classified.traits)
    assert not is_group_opportunity(private, edge)


def test_visible_initializer_edge_can_split_group_but_not_stage() -> None:
    classified = HOPPER.classify(_fa3_graph())
    graph = classified.graph
    edge = next(
        item for item in graph.edges
        if item.producer_id != item.consumer_id
        and graph.buffer_for_id(item.buffer_id).scope.startswith("shared")
    )
    nodes = list(graph.nodes)
    nodes[edge.producer_id] = replace(nodes[edge.producer_id], kind=OpKind.FILL)
    initialized = ClassifiedGraph(replace(graph, nodes=tuple(nodes)), classified.traits)
    assert initialized.graph.is_initializer(edge.producer_id)
    assert is_group_opportunity(initialized, edge)
    assert allowed_edge_actions(initialized, edge) == frozenset(
        {EdgeAction.KEEP, EdgeAction.SPLIT_GROUP}
    )


def test_same_engine_fragment_cannot_split_group() -> None:
    classified = HOPPER.classify(_fa3_graph())
    graph = classified.graph
    edges = [
        item
        for item in graph.edges
        if item.producer_id != item.consumer_id
        and graph.buffer_for_id(item.buffer_id).scope == "local.fragment"
        and classified.traits_for(item.producer_id).engine
        == classified.traits_for(item.consumer_id).engine
        and classified.traits_for(item.producer_id).kind == ResourceKind.COMPUTE
        and classified.traits_for(item.consumer_id).kind == ResourceKind.COMPUTE
        and not graph.is_initializer(item.producer_id)
    ]
    assert edges
    for edge in edges:
        assert not is_group_opportunity(classified, edge)
        assert EdgeAction.SPLIT_GROUP not in allowed_edge_actions(classified, edge)


def test_fake_arch_uses_the_same_classify_contract() -> None:
    classified = FAKE.classify(_gemm_graph())
    gemm = next(
        node for node in classified.graph.nodes if node.kind == OpKind.GEMM
    )
    assert classified.traits_for(gemm.node_id).engine == "tensorcore"
    assert FAKE.resource().name == "fake"
    assert HOPPER.resource().partition_warp_multiple == 4


@T.prim_func
def gmem_to_smem_aligned(
    A: T.Tensor((64, 64), T.float16),
    B: T.Tensor((64, 64), T.float16),
):
    with T.Kernel(1, threads=128):
        a_shared = T.alloc_shared((64, 64), T.float16)
        T.copy(A, a_shared)
        T.copy(a_shared, B)


@T.prim_func
def gmem_to_smem_pipelined(
    A: T.Tensor((64, 64), T.float16),
    B: T.Tensor((64, 64), T.float16),
):
    with T.Kernel(1, threads=128):
        a_shared = T.alloc_shared((64, 64), T.float16)
        for _k in T.Pipelined(1):
            T.copy(A, a_shared)
        T.copy(a_shared, B)


@T.prim_func
def gmem_to_smem_small(
    A: T.Tensor((64,), T.float16),
    B: T.Tensor((64,), T.float16),
):
    with T.Kernel(1, threads=128):
        a_shared = T.alloc_shared((64,), T.float16)
        T.copy(A, a_shared)
        T.copy(a_shared, B)


@T.prim_func
def gmem_to_smem_small_pipelined(
    A: T.Tensor((64,), T.float16),
    B: T.Tensor((64,), T.float16),
):
    with T.Kernel(1, threads=128):
        a_shared = T.alloc_shared((64,), T.float16)
        for _k in T.Pipelined(1):
            T.copy(A, a_shared)
        T.copy(a_shared, B)


@T.prim_func
def gmem_to_smem_small_in_loop_consumed(
    A: T.Tensor((64,), T.float16),
    B: T.Tensor((64,), T.float16),
):
    with T.Kernel(1, threads=128):
        a_shared = T.alloc_shared((64,), T.float16)
        a_local = T.alloc_fragment((64,), T.float16)
        for _k in T.Pipelined(1):
            T.copy(A, a_shared)
            T.copy(a_shared, a_local)
        T.copy(a_local, B)


@T.prim_func
def gmem_to_smem_disable_tma(
    A: T.Tensor((64, 64), T.float16),
    B: T.Tensor((64, 64), T.float16),
):
    with T.Kernel(1, threads=128):
        a_shared = T.alloc_shared((64, 64), T.float16)
        T.copy(A, a_shared, disable_tma=True)
        T.copy(a_shared, B)


@T.prim_func
def gmem_to_smem_unaligned(
    A: T.Tensor((32, 3), T.float16),
    B: T.Tensor((32, 3), T.float16),
):
    with T.Kernel(1, threads=128):
        a_shared = T.alloc_shared((32, 3), T.float16)
        T.copy(A, a_shared)
        T.copy(a_shared, B)


def _is_gmem_smem_load(graph, node) -> bool:
    return (
        node.kind == OpKind.COPY
        and any(
            graph.buffer_for_id(buffer_id).scope in ("", "global")
            for buffer_id in node.reads
        )
        and any(
            graph.buffer_for_id(buffer_id).scope.startswith("shared")
            for buffer_id in node.writes
        )
    )


def _gmem_smem_load(graph):
    return next(node for node in graph.nodes if _is_gmem_smem_load(graph, node))


def _assert_tma_load(prim) -> None:
    classified = HOPPER.classify(extract_fact_graph(prim))
    load = _gmem_smem_load(classified.graph)
    traits = classified.traits_for(load.node_id)
    assert traits.kind == ResourceKind.MEMORY
    assert traits.engine == "copy"
    assert traits.async_completion is True
    assert traits.issue_priority == 6


def _assert_simt_load(prim) -> None:
    classified = HOPPER.classify(extract_fact_graph(prim))
    load = _gmem_smem_load(classified.graph)
    traits = classified.traits_for(load.node_id)
    assert traits.kind == ResourceKind.MEMORY
    assert traits.engine == "copy"
    assert traits.async_completion is False
    assert traits.issue_priority == 6


def test_hopper_uses_tma_for_aligned_gmem_smem_copy() -> None:
    _assert_tma_load(gmem_to_smem_aligned)
    _assert_tma_load(gmem_to_smem_pipelined)


def test_hopper_classifies_small_gmem_smem_as_simt() -> None:
    _assert_simt_load(gmem_to_smem_small)
    _assert_simt_load(gmem_to_smem_small_pipelined)


def test_hopper_keeps_tma_for_small_in_loop_smem_handoff() -> None:
    _assert_tma_load(gmem_to_smem_small_in_loop_consumed)


def test_hopper_classifies_disable_tma_gmem_smem_as_simt() -> None:
    _assert_simt_load(gmem_to_smem_disable_tma)


def test_hopper_classifies_unaligned_gmem_smem_as_simt() -> None:
    classified = HOPPER.classify(extract_fact_graph(gmem_to_smem_unaligned))
    load = _gmem_smem_load(classified.graph)
    traits = classified.traits_for(load.node_id)
    assert traits.kind == ResourceKind.MEMORY
    assert traits.engine == "copy"
    assert traits.async_completion is False


def test_hopper_classifies_fa3_gmem_smem_loads_as_tma() -> None:
    classified = HOPPER.classify(_fa3_graph())
    loads = [
        node
        for node in classified.graph.nodes
        if _is_gmem_smem_load(classified.graph, node)
    ]
    assert len(loads) == 3
    for node in loads:
        traits = classified.traits_for(node.node_id)
        assert traits.engine == "copy"
        assert traits.async_completion is True
        assert traits.issue_priority == 6
