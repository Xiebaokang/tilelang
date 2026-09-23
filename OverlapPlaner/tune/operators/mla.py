"""MLA decode workload adapted from ``examples/deepseek_mla``.

The math and buffer layout match ``main_no_split`` (and the inlined copy in
``compile_kernels/MLA/mla_tilelang_benchmark.py``): full ``acc_o`` of shape
``[block_H, dim]``, online softmax, then ``S @ KV``. The original Hopper
kernel uses 256 threads so FullCol can partition the large output along N.
"""

from itertools import product

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig


def add_arguments(parser) -> None:
    group = parser.add_argument_group("MLA")
    group.add_argument("--mla-block-h", type=int, nargs="+", default=[16, 32, 64])
    group.add_argument("--mla-block-n", type=int, nargs="+", default=[64])
    group.add_argument("--mla-batch", type=int, default=1)
    group.add_argument("--mla-heads", type=int, default=128)
    group.add_argument("--mla-kv-heads", type=int, default=1)
    group.add_argument("--mla-seq", type=int, default=8192)
    group.add_argument("--mla-dim", type=int, default=512)
    group.add_argument("--mla-pe-dim", type=int, default=64)


def configurations(options: Options) -> list[TileConfig]:
    kv_group_num = options["mla_heads"] // options["mla_kv_heads"]
    kept: list[TileConfig] = []
    for block_h, block_n in product(
        options["mla_block_h"], options["mla_block_n"]
    ):
        if block_h > kv_group_num:
            continue
        if kv_group_num % block_h:
            continue
        kept.append({"block_h": block_h, "block_n": block_n})
    return kept


def flashattn(
    batch,
    heads,
    kv_head_num,
    seqlen_kv,
    dim,
    pe_dim,
    block_N=64,
    block_H=64,
):
    scale = ((1.0 / (dim + pe_dim)) ** 0.5) * 1.44269504
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    valid_block_h = min(block_H, kv_group_num)

    @T.prim_func(auto_overlap=True)
    def main(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(
            heads // min(block_H, kv_group_num),
            batch,
            threads=256,
        ) as (hid, bid):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            cur_kv_head = hid // (kv_group_num // block_H)
            T.copy(
                Q[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :],
                Q_shared,
            )
            T.copy(
                Q_pe[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :],
                Q_pe_shared,
            )
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for k in T.Pipelined(
                T.ceildiv(seqlen_kv, block_N),
                num_stages=2,
            ):
                T.copy(
                    KV[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :],
                    KV_shared,
                )
                T.copy(
                    K_pe[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :],
                    K_pe_shared,
                )
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(
                    Q_pe_shared,
                    K_pe_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * scale - scores_max[i] * scale
                    )
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(
                    S_shared,
                    KV_shared,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullCol,
                )
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(
                O_shared,
                Output[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :],
            )

    return main


def ref_program(q, q_pe, kv, k_pe):
    dim = q.shape[-1]
    pe_dim = q_pe.shape[-1]
    batch, heads, _ = q.shape
    kv_heads = kv.shape[2]
    groups = heads // kv_heads
    scale = (dim + pe_dim) ** 0.5
    q = q.reshape(batch, kv_heads, groups, dim).permute(0, 2, 1, 3)
    q_pe = q_pe.reshape(batch, kv_heads, groups, pe_dim).permute(0, 2, 1, 3)
    kv = kv.permute(0, 2, 1, 3)
    k_pe = k_pe.permute(0, 2, 1, 3)
    query = torch.cat([q, q_pe], dim=-1)
    key = torch.cat([kv, k_pe], dim=-1)
    scores = torch.einsum("bghd,bhsd->bghs", query, key)
    attention = F.softmax(scores / scale, dim=-1)
    output = torch.einsum("bghs,bhsd->bghd", attention, kv)
    return output.permute(0, 2, 1, 3).reshape(batch, heads, dim)


def build(options: Options, config: TileConfig) -> SearchWorkload:
    batch = options["mla_batch"]
    heads = options["mla_heads"]
    kv_heads = options["mla_kv_heads"]
    seqlen_kv = options["mla_seq"]
    dim = options["mla_dim"]
    pe_dim = options["mla_pe_dim"]
    if kv_heads != 1:
        raise ValueError("mla kv heads must be 1")
    if heads % kv_heads:
        raise ValueError("mla heads must be divisible by kv heads")
    kv_group_num = heads // kv_heads
    if config["block_h"] > kv_group_num or kv_group_num % config["block_h"]:
        raise ValueError("mla block_h must evenly divide heads per kv head")
    prim_func = flashattn(
        batch,
        heads,
        kv_heads,
        seqlen_kv,
        dim,
        pe_dim,
        block_N=config["block_n"],
        block_H=config["block_h"],
    )
    total_flops = 2.0 * batch * heads * seqlen_kv * (dim + pe_dim) + (
        2.0 * batch * heads * seqlen_kv * dim
    )
    return SearchWorkload(
        prim_func=prim_func,
        out_idx=(4,),
        total_flops=total_flops,
        reference_program=ref_program,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


OPERATOR = OperatorSpec(
    name="mla",
    description=(
        "MLA decode (no-split) adapted from "
        "examples/deepseek_mla/example_mla_decode.py"
    ),
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
