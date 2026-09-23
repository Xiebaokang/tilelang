"""Tests for architecture-free OverlapPlan fact-graph extraction."""

import tilelang.language as T

from OverlapPlaner.facts import OpKind, RegionKind, extract_fact_graph
from OverlapPlaner.tune.operators.fa3 import build as build_fa3
from OverlapPlaner.tune.operators.gemm import build as build_gemm


@T.prim_func(auto_overlap=True)
def gemm_fullcol(
    A: T.Tensor((128, 64), T.float16),
    B: T.Tensor((64, 128), T.float16),
    C: T.Tensor((128, 128), T.float16),
):
    with T.Kernel(1, 1, threads=256) as (_bx, _by):
        a_shared = T.alloc_shared((128, 64), T.float16)
        b_shared = T.alloc_shared((64, 128), T.float16)
        c_local = T.alloc_fragment((128, 128), T.float32)
        T.copy(A, a_shared)
        T.copy(B, b_shared)
        T.clear(c_local)
        for _k in T.Pipelined(1):
            T.gemm(
                a_shared,
                b_shared,
                c_local,
                policy=T.GemmWarpPolicy.FullCol,
            )
        T.copy(c_local, C)


@T.prim_func
def pipelined_without_num_stages(
    A: T.Tensor((8,), T.float16),
    B: T.Tensor((8,), T.float16),
):
    with T.Kernel(1):
        for i in T.Pipelined(8):
            B[i] = A[i]


def test_extract_does_not_classify_isa() -> None:
    graph = extract_fact_graph(gemm_fullcol)
    assert not hasattr(graph, "hardware")
    assert all(not hasattr(node, "instruction") for node in graph.nodes)
    assert all(node.kind != "wgmma" for node in graph.nodes)
    assert all(node.tileop != "wgmma" for node in graph.nodes)


def test_extract_gemm_facts_and_copy_scopes() -> None:
    graph = extract_fact_graph(gemm_fullcol)
    gemms = [node for node in graph.nodes if node.kind == OpKind.GEMM]
    assert len(gemms) == 1
    gemm = gemms[0].gemm
    assert gemm is not None
    assert (gemm.m, gemm.n, gemm.k) == (128, 128, 64)
    assert gemm.policy == "full_col"
    assert gemm.transpose_a is False
    assert gemm.transpose_b is False
    assert gemm.c_dtype.startswith("float32")

    copies = [node for node in graph.nodes if node.kind == OpKind.COPY]
    assert copies
    assert all(node.tileop == "copy" for node in copies)
    read_scopes, write_scopes = graph.scopes_for_node(copies[0].node_id)
    assert any(scope in ("", "global") for scope in read_scopes)
    assert any(scope.startswith("shared") for scope in write_scopes)

    fills = [node for node in graph.nodes if node.kind == OpKind.FILL]
    assert fills
    assert all(graph.is_initializer(node.node_id) for node in fills)
    assert graph.kernel_threads == 256
    assert RegionKind.PIPELINE in graph.region_kinds


def test_extract_fa3_keeps_regions_edges_and_gemm_shape() -> None:
    prim_func = build_fa3(
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
    graph = extract_fact_graph(prim_func)

    assert graph.region_kinds == (
        RegionKind.SERIAL,
        RegionKind.PIPELINE,
        RegionKind.SERIAL,
    )
    assert tuple(node.node_id for node in graph.nodes) == tuple(
        range(len(graph.nodes))
    )
    assert tuple(buffer.buffer_id for buffer in graph.buffers) == tuple(
        range(len(graph.buffers))
    )
    assert graph.kernel_threads == 256

    gemms = [node for node in graph.nodes if node.kind == OpKind.GEMM]
    assert len(gemms) == 2
    qk, av = gemms
    assert qk.gemm is not None and av.gemm is not None
    assert qk.gemm.transpose_b is True
    assert qk.gemm.policy == "full_row"
    assert (qk.gemm.m, qk.gemm.n, qk.gemm.k) == (128, 128, 128)
    assert (qk.gemm.a_scope, qk.gemm.b_scope, qk.gemm.c_scope) == (
        "shared.dyn",
        "shared.dyn",
        "local.fragment",
    )
    assert (qk.gemm.a_dtype, qk.gemm.b_dtype, qk.gemm.c_dtype) == (
        "float16",
        "float16",
        "float32",
    )
    assert av.gemm.transpose_b is False
    assert (av.gemm.m, av.gemm.n, av.gemm.k) == (128, 128, 128)

    assert any(node.kind == OpKind.FILL for node in graph.nodes)
    assert any(node.kind == OpKind.REDUCE for node in graph.nodes)
    assert any("exp2" in op for node in graph.nodes for op in node.scalar_ops)
    assert any(edge.is_loop_carried for edge in graph.edges)
    assert all(node.statement is not None for node in graph.nodes)
    assert all(buffer.buffer is not None for buffer in graph.buffers)


def test_extract_operator_gemm_pipeline_region() -> None:
    prim_func = build_gemm(
        {"gemm_m": 256, "gemm_n": 256, "gemm_k": 256},
        {"block_m": 128, "block_n": 128, "block_k": 64},
    ).prim_func
    graph = extract_fact_graph(prim_func)
    gemms = [node for node in graph.nodes if node.kind == OpKind.GEMM]
    assert len(gemms) == 1
    assert gemms[0].gemm is not None
    assert (gemms[0].gemm.m, gemms[0].gemm.n, gemms[0].gemm.k) == (128, 128, 64)
    assert gemms[0].gemm.policy == "square"
    assert graph.regions[1].kind == RegionKind.PIPELINE


def test_pipelined_without_num_stages_is_a_pipeline_region() -> None:
    graph = extract_fact_graph(pipelined_without_num_stages)
    assert RegionKind.PIPELINE in graph.region_kinds
    pipeline = next(
        region for region in graph.regions if region.kind == RegionKind.PIPELINE
    )
    assert pipeline.static_extent == 8
    assert pipeline.loop is not None
    assert "num_stages" not in pipeline.loop.annotations
