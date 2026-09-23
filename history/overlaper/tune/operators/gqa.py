"""Overlaper search kernel adapted from examples/flash_attention/example_gqa_fwd_bshd.py."""

from functools import partial
from itertools import product

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig


def add_arguments(parser) -> None:
    group = parser.add_argument_group("GQA")
    group.add_argument("--gqa-block-m", type=int, nargs="+", default=[64, 128])
    group.add_argument("--gqa-block-n", type=int, nargs="+", default=[64, 128])
    group.add_argument("--gqa-batch", type=int, default=1)
    group.add_argument("--gqa-heads", type=int, default=16)
    group.add_argument("--gqa-groups", type=int, default=8)
    group.add_argument("--gqa-seq", type=int, default=4096)
    group.add_argument("--gqa-dim", type=int, default=128)
    group.add_argument("--gqa-causal", action="store_true")


def configurations(options: Options) -> list[TileConfig]:
    return [
        {"block_m": block_m, "block_n": block_n}
        for block_m, block_n in product(
            options["gqa_block_m"], options["gqa_block_n"]
        )
    ]


def flashattn(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    groups,
    block_M=64,
    block_N=64,
    auto_wsp=True,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(seq_len, block_M),
            heads,
            batch,
            threads=block_M // 64 * 128,
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

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(
                loop_range,
                num_stages=2,
                auto_wsp=auto_wsp,
            ):
                T.copy(
                    K[
                        bz,
                        k * block_N : (k + 1) * block_N,
                        by // groups,
                        :,
                    ],
                    K_shared,
                )
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j,
                            0,
                            -T.infinity(acc_s.dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * block_N + j >= seq_len,
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
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(
                    V[
                        bz,
                        k * block_N : (k + 1) * block_N,
                        by // groups,
                        :,
                    ],
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
                Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
            )

    return main


def ref_program(Q, K, V, is_causal, groups):
    dim = Q.size(-1)
    K = K.repeat_interleave(groups, dim=2)
    V = V.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", attention_weights, V)


def build(options: Options, config: TileConfig) -> SearchWorkload:
    batch = options["gqa_batch"]
    heads = options["gqa_heads"]
    groups = options["gqa_groups"]
    seq_len = options["gqa_seq"]
    dim = options["gqa_dim"]
    causal = options["gqa_causal"]
    if heads % groups:
        raise ValueError("gqa heads must be divisible by groups")
    prim_func = flashattn(
        batch,
        heads,
        seq_len,
        dim,
        causal,
        groups,
        block_M=config["block_m"],
        block_N=config["block_n"],
        auto_wsp=True,
    )
    total_flops = 4.0 * batch * heads * seq_len * seq_len * dim
    if causal:
        total_flops *= 0.5
    return SearchWorkload(
        prim_func=prim_func,
        out_idx=(3,),
        total_flops=total_flops,
        reference_program=partial(ref_program, is_causal=causal, groups=groups),
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


OPERATOR = OperatorSpec(
    name="gqa",
    description=(
        "Grouped-query attention forward adapted from "
        "examples/flash_attention/example_gqa_fwd_bshd.py"
    ),
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
