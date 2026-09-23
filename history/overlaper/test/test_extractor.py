"""FA3 graph-extraction regression test for Overlaper."""

import tilelang
import tilelang.language as T
from tvm.target import Target

from history.overlaper.headware import HOPPER
from history.overlaper.parse import DependencyKind, RegionKind, extract_dataflow_graph


@tilelang.jit(
    out_idx=[3],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def flashattn(
    batch,
    heads,
    seq_q,
    seq_kv,
    dim,
    block_m=128,
    block_n=128,
    threads=256,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func(auto_overlap=True)
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(seq_q, block_m), heads, batch, threads=threads
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_m, dim], dtype)
            K_shared = T.alloc_shared([block_n, dim], dtype)
            V_shared = T.alloc_shared([block_n, dim], dtype)
            O_shared = T.alloc_shared([block_m, dim], dtype)
            acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
            acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_m], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
            scores_scale = T.alloc_fragment([block_m], accum_dtype)
            scores_sum = T.alloc_fragment([block_m], accum_dtype)
            logsum = T.alloc_fragment([block_m], accum_dtype)

            T.copy(
                Q[bz, by, bx * block_m : (bx + 1) * block_m, :],
                Q_shared,
            )
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for k in T.Pipelined(
                T.ceildiv(seq_kv, block_n), num_stages=3
            ):
                T.copy(
                    K[bz, by, k * block_n : (k + 1) * block_n, :],
                    K_shared,
                )
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.if_then_else(
                        k * block_n + j >= seq_kv,
                        -T.infinity(acc_s.dtype),
                        0,
                    )
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_m):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_m):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * scale - scores_max[i] * scale
                    )
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_m):
                    logsum[i] = (
                        logsum[i] * scores_scale[i] + scores_sum[i]
                    )
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(
                    V[bz, by, k * block_n : (k + 1) * block_n, :],
                    V_shared,
                )
                T.gemm(
                    acc_s_cast,
                    V_shared,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            for i, j in T.Parallel(block_m, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(
                O_shared,
                Output[bz, by, bx * block_m : (bx + 1) * block_m, :],
            )

    return main


def make_fa3_prim_func(
    block_m=128, block_n=128, threads=256, seq_len=256
):
    return flashattn.get_tir(
        1, 1, seq_len, seq_len, 64, block_m, block_n, threads
    )


def print_dataflow_graph(graph) -> None:
    """Print a compact, human-readable representation for ``pytest -s``."""

    print(
        f"DataflowGraph: {len(graph.nodes)} nodes, "
        f"{len(graph.buffers)} buffers, {len(graph.edges)} edges"
    )
    print("\nRegions")
    for region_id, kind in enumerate(graph.region_kinds):
        node_ids = tuple(
            node.node_id for node in graph.nodes_for_region(region_id)
        )
        print(f"  region {region_id}: kind={kind.value} nodes={node_ids}")

    print("\nBuffers")
    for buffer in graph.buffers:
        print(
            f"  buffer {buffer.buffer_id:2d}: name={buffer.name:16s} "
            f"scope={buffer.scope:14s} nbytes={buffer.nbytes}"
        )

    print("\nNodes")
    for node in graph.nodes:
        print(
            f"  node {node.node_id:2d}: region={node.region_id} "
            f"instruction={node.instruction.name:8s} "
            f"reads={node.reads} writes={node.writes} name={node.name}"
        )

    print("\nEdges")
    for edge in graph.edges:
        producer = graph.node_for_id(edge.producer_id)
        consumer = graph.node_for_id(edge.consumer_id)
        kinds = "/".join(sorted(kind.value for kind in edge.dependency_kinds))
        buffer = graph.buffer_for_id(edge.buffer_id)
        print(
            f"  {producer.node_id:2d}:{producer.name} -> "
            f"{consumer.node_id:2d}:{consumer.name} "
            f"kinds={kinds:7s} buffer={edge.buffer_id}:{buffer.name} "
            f"iteration_distance={edge.iteration_distance}"
        )


def test_extract_fa3_dataflow_graph() -> None:
    prim_func = make_fa3_prim_func()
    auto_overlap = (
        prim_func.attrs.get("tl.auto_overlap")
        if prim_func.attrs is not None
        else None
    )
    assert auto_overlap is not None and int(auto_overlap) != 0
    graph = extract_dataflow_graph(
        prim_func,
        target=Target({"kind": "cuda", "arch": "sm_90a"}),
    )
    print_dataflow_graph(graph)

    assert graph.region_kinds == (
        RegionKind.SERIAL,
        RegionKind.PIPELINE,
        RegionKind.SERIAL,
    )
    assert tuple(len(graph.nodes_for_region(i)) for i in range(3)) == (4, 15, 3)
    assert tuple(node.node_id for node in graph.nodes) == tuple(range(22))
    assert len(graph.buffers) == 16
    assert len(graph.edges) == 32
    assert graph.hardware is HOPPER
    assert graph.kernel_threads == 256

    assert graph.nodes[0].name == "copy_Q_to_Q_shared"
    assert graph.nodes[0].instruction.name == "tma"
    assert graph.nodes[6].name == "gemm_acc_s"
    assert graph.nodes[6].instruction.name == "wgmma"
    assert graph.nodes[20].name == "copy_acc_o_to_O_shared"
    assert graph.nodes[20].instruction.name == "rscp"
    assert tuple(
        node.node_id
        for node in graph.nodes
        if node.instruction.name == "function"
    ) == (11, 12)

    loop_carried = {
        (
            edge.producer_id,
            edge.consumer_id,
            graph.buffer_for_id(edge.buffer_id).name,
        )
        for edge in graph.edges
        if edge.is_loop_carried
    }
    assert loop_carried == {
        (10, 7, "scores_max"),
        (14, 14, "logsum"),
        (18, 16, "acc_o"),
    }
    assert any(
        edge.producer_id == 3
        and edge.consumer_id == 8
        and edge.dependency_kinds == frozenset({DependencyKind.WAW})
        for edge in graph.edges
    )


def _pipelined_copy_without_num_stages():
    @T.prim_func
    def main(A: T.Tensor((8,), T.float16), B: T.Tensor((8,), T.float16)):
        with T.Kernel(1):
            for i in T.Pipelined(8):
                B[i] = A[i]

    return main


def test_pipelined_without_num_stages_is_a_pipeline_region() -> None:
    from tvm import tirx
    from tvm.tirx.stmt_functor import post_order_visit

    prim_func = _pipelined_copy_without_num_stages()
    pipelined = []

    def visit(node):
        if isinstance(node, tirx.For) and "tl.pipelined" in node.annotations:
            pipelined.append(node)

    post_order_visit(prim_func.body, visit)
    assert pipelined
    assert all("num_stages" not in loop.annotations for loop in pipelined)

    graph = extract_dataflow_graph(
        prim_func,
        target=Target({"kind": "cuda", "arch": "sm_90a"}),
    )
    assert RegionKind.PIPELINE in graph.region_kinds
