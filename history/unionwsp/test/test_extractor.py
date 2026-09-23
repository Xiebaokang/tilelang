import itertools

import pytest
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tvm.target import Target

from history.unionwsp.hardware import HOPPER
from history.unionwsp.parseIR import (
    DependencyKind,
    InstructionKind,
    RegionKind,
    extract_dataflow_graph,
)


def get_configs():
    parameters = dict(block_M=[128], block_N=[128], threads=[256])
    return [
        dict(zip(parameters, values))
        for values in itertools.product(*parameters.values())
    ]


@autotune(configs=get_configs(), warmup=10, rep=10)
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
    is_causal,
    block_M=64,
    block_N=64,
    threads=128,
    auto_wsp=True,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32
    past_len = seq_kv - seq_q

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(seq_q, block_M), heads, batch, threads=threads
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(
                Q[bz, by, bx * block_M : (bx + 1) * block_M, :],
                Q_shared,
            )
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_kv, block_N),
                    T.ceildiv((bx + 1) * block_M + past_len, block_N),
                )
                if is_causal
                else T.ceildiv(seq_kv, block_N)
            )
            for k in T.Pipelined(
                loop_range,
                num_stages=2,
                auto_wsp=auto_wsp,
            ):
                T.copy(
                    K[bz, by, k * block_N : (k + 1) * block_N, :],
                    K_shared,
                )
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i + past_len
                        k_idx = k * block_N + j
                        acc_s[i, j] = T.if_then_else(
                            q_idx >= k_idx,
                            0,
                            -T.infinity(acc_s.dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * block_N + j >= seq_kv,
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
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * scale - scores_max[i] * scale
                    )
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = (
                        logsum[i] * scores_scale[i] + scores_sum[i]
                    )
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(
                    V[bz, by, k * block_N : (k + 1) * block_N, :],
                    V_shared,
                )
                T.gemm(
                    acc_s_cast,
                    V_shared,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(
                O_shared,
                Output[bz, by, bx * block_M : (bx + 1) * block_M, :],
            )

    return main


def make_fa3_prim_func():
    return flashattn.jit_impl.get_tir(
        1,
        1,
        256,
        256,
        64,
        False,
        block_M=128,
        block_N=128,
        threads=256,
        auto_wsp=True,
    )


def hopper_target():
    return Target({"kind": "cuda", "arch": "sm_90a"})


def print_dataflow_graph(graph) -> None:
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
            f"kind={node.instruction_kind.value:8s} "
            f"async={str(graph.is_async(node.node_id)):5s} "
            f"reads={node.reads} writes={node.writes} {node.name}"
        )

    print("\nBuffer range accesses")
    for access in graph.buffer_accesses:
        buffer = graph.buffer_for_id(access.buffer_id)
        ranges = ", ".join(
            f"(min={item.min}, extent={item.extent})"
            for item in access.ranges
        )
        print(
            f"  node {access.node_id:2d}: {access.kind.value:5s} "
            f"buffer={access.buffer_id}:{buffer.name} "
            f"exact={access.is_exact} ranges=[{ranges}]"
        )

    print("\nEdges")
    for edge in graph.edges:
        producer = graph.node_for_id(edge.producer_id)
        consumer = graph.node_for_id(edge.consumer_id)
        kinds = "/".join(
            sorted(kind.value for kind in edge.dependency_kinds)
        )
        buffer = (
            "-"
            if edge.buffer_id is None
            else (
                f"{edge.buffer_id}:"
                f"{graph.buffer_for_id(edge.buffer_id).name}"
            )
        )
        print(
            f"  {producer.node_id:2d}:{producer.name} -> "
            f"{consumer.node_id:2d}:{consumer.name} "
            f"kinds={kinds:11s} buffer={buffer:20s} "
            f"iteration_distance={edge.iteration_distance}"
        )


def test_extract_fa3_dataflow_graph() -> None:
    graph = extract_dataflow_graph(
        make_fa3_prim_func(), target=hopper_target()
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
    assert len(graph.buffer_accesses) > len(graph.nodes)
    assert len(graph.edges) == 32
    assert any(
        edge.producer_id == 3
        and edge.consumer_id == 8
        and edge.buffer_id == 4
        and edge.dependency_kinds == frozenset({DependencyKind.WAW})
        for edge in graph.edges
    )
    assert any(
        edge.producer_id == 9
        and edge.consumer_id == 12
        and edge.buffer_id == 7
        and edge.dependency_kinds == frozenset({DependencyKind.WAR})
        for edge in graph.edges
    )
    assert graph.hardware is HOPPER
    assert not any(
        edge.producer_id == 1
        and edge.consumer_id == 18
        and edge.iteration_distance == 0
        for edge in graph.edges
    )

    assert graph.nodes[0].name == "copy_Q_to_Q_shared"
    assert graph.nodes[0].instruction_kind == InstructionKind.TMA
    assert graph.is_async(0)
    assert graph.nodes[6].name == "gemm_acc_s"
    assert graph.nodes[6].instruction_kind == InstructionKind.WGMMA
    assert graph.is_async(6)
    assert graph.nodes[20].name == "copy_acc_o_to_O_shared"
    assert graph.nodes[20].instruction_kind == InstructionKind.RSCP
    assert tuple(
        node.node_id
        for node in graph.nodes
        if node.instruction_kind == InstructionKind.FUNCTION
    ) == (11, 12)
    assert graph.nodes[1].instruction_kind == InstructionKind.GENERIC
    assert not graph.is_async(1)
    assert graph.nodes[9].instruction_kind == InstructionKind.GENERIC

    loop_carried = {
        (edge.producer_id, edge.consumer_id, graph.buffer_for_id(edge.buffer_id).name)
        for edge in graph.edges
        if edge.is_loop_carried and edge.buffer_id is not None
    }
    assert loop_carried == {
        (10, 7, "scores_max"),
        (14, 14, "logsum"),
        (18, 16, "acc_o"),
    }


def test_infer_hardware_from_prim_func_target() -> None:
    prim_func = make_fa3_prim_func().with_attr("target", hopper_target())

    graph = extract_dataflow_graph(prim_func)

    assert graph.nodes[0].instruction_kind == InstructionKind.TMA
    assert graph.nodes[6].instruction_kind == InstructionKind.WGMMA


def test_explicit_hardware_overrides_target() -> None:
    unsupported_target = Target({"kind": "cuda", "arch": "sm_100a"})

    graph = extract_dataflow_graph(
        make_fa3_prim_func(),
        hardware=HOPPER,
        target=unsupported_target,
    )

    assert len(graph.nodes) == 22


def test_missing_hardware_information_is_rejected() -> None:
    with pytest.raises(ValueError, match="hardware cannot be inferred"):
        extract_dataflow_graph(make_fa3_prim_func())


if __name__ == "__main__":
    print_dataflow_graph(
        extract_dataflow_graph(make_fa3_prim_func(), target=hopper_target())
    )
