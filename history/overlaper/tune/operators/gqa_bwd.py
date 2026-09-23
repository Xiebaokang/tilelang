"""Overlaper search kernel adapted from examples/flash_attention/example_gqa_bwd.py."""

from functools import partial
from itertools import product

import tilelang
import tilelang.language as T
import torch

from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig


def add_arguments(parser) -> None:
    group = parser.add_argument_group("GQA backward")
    group.add_argument("--gqa-bwd-block-m", type=int, nargs="+", default=[64, 128])
    group.add_argument("--gqa-bwd-block-n", type=int, nargs="+", default=[32, 64, 128])
    group.add_argument("--gqa-bwd-batch", type=int, default=1)
    group.add_argument("--gqa-bwd-heads", type=int, default=16)
    group.add_argument("--gqa-bwd-groups", type=int, default=8)
    group.add_argument("--gqa-bwd-seq", type=int, default=4096)
    group.add_argument("--gqa-bwd-dim", type=int, default=128)
    group.add_argument("--gqa-bwd-causal", action="store_true")


def configurations(options: Options) -> list[TileConfig]:
    kept: list[TileConfig] = []
    for block_m, block_n in product(
        options["gqa_bwd_block_m"], options["gqa_bwd_block_n"]
    ):
        if block_m % 64 or block_n % 64:
            continue
        if block_m // 64 != block_n // 64:
            continue
        kept.append({"block_m": block_m, "block_n": block_n})
    return kept


def make_dq_layout(dQ):
    return T.Layout(
        dQ.shape,
        lambda b, l, h, d: [
            b,
            l // 8,
            h,
            d // 8,
            d % 2,
            4 * (l % 8) + (d % 8) // 2,
        ],
    )


def flashattn_bwd(
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
    sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dk_shape = [groups, batch, seq_len, head_kv, dim]
    dv_shape = [groups, batch, seq_len, head_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(q_shape, dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dQ: T.Tensor(q_shape, accum_dtype),
        dK: T.Tensor(dk_shape, dtype),
        dV: T.Tensor(dv_shape, dtype),
    ):
        with T.Kernel(
            heads,
            T.ceildiv(seq_len, block_M),
            batch,
            threads=block_M // 64 * 128,
        ) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            q = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_M, dim], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            do = T.alloc_shared([block_N, dim], dtype)
            dv = T.alloc_fragment([block_M, dim], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)
            dv_shared = T.alloc_shared([block_M, dim], dtype)
            dk_shared = T.alloc_shared([block_M, dim], dtype)

            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                K[bz, by * block_M : (by + 1) * block_M, bx // groups, :],
                K_shared,
            )
            T.copy(
                V[bz, by * block_M : (by + 1) * block_M, bx // groups, :],
                V_shared,
            )
            T.clear(dv)
            T.clear(dk)
            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_st, loop_ed, num_stages=2, auto_wsp=auto_wsp):
                T.copy(Q[bz, k * block_N : (k + 1) * block_N, bx, :], q)
                T.clear(qkT)
                T.gemm(
                    K_shared,
                    q,
                    qkT,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(dO[bz, k * block_N : (k + 1) * block_N, bx, :], do)
                T.clear(dsT)
                T.gemm(
                    V_shared,
                    do,
                    dsT,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(lse[bz, bx, k * block_N : (k + 1) * block_N], lse_shared)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(
                            by * block_M + i <= k * block_N + j,
                            qkT[i, j],
                            0,
                        )
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)
                T.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], delta)
                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)
                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                for i, j in T.Parallel(block_N, dim):
                    T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])
            T.copy(dv, dv_shared)
            T.copy(
                dv_shared,
                dV[
                    bx % groups,
                    bz,
                    by * block_M : (by + 1) * block_M,
                    bx // groups,
                    :,
                ],
            )
            T.copy(dk, dk_shared)
            T.copy(
                dk_shared,
                dK[
                    bx % groups,
                    bz,
                    by * block_M : (by + 1) * block_M,
                    bx // groups,
                    :,
                ],
            )

    return main


def ref_program(Q, K, V, dO, lse, Delta, dQ, is_causal, groups):
    dim = Q.size(-1)
    seq_len = Q.size(1)
    heads = Q.size(2)
    head_kv = heads // groups
    sm_scale = dim**-0.5
    scale = sm_scale * 1.44269504
    K_full = K.repeat_interleave(groups, dim=2)
    V_full = V.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q.float(), K_full.float())
    weights = torch.exp2(scores * scale - lse.unsqueeze(-1))
    if is_causal:
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool)
        )
        weights = weights.masked_fill(~mask, 0)
    dV_full = torch.einsum("bhqk,bqhd->bkhd", weights, dO.float())
    dP = torch.einsum("bqhd,bkhd->bhqk", dO.float(), V_full.float())
    dS = weights * (dP - Delta.unsqueeze(-1)) * sm_scale
    dK_full = torch.einsum("bhqk,bqhd->bkhd", dS, Q.float())
    batch = Q.size(0)
    dK = (
        dK_full.reshape(batch, seq_len, head_kv, groups, dim)
        .permute(3, 0, 1, 2, 4)
        .to(K.dtype)
    )
    dV = (
        dV_full.reshape(batch, seq_len, head_kv, groups, dim)
        .permute(3, 0, 1, 2, 4)
        .to(V.dtype)
    )
    return dK, dV


def build(options: Options, config: TileConfig) -> SearchWorkload:
    batch = options["gqa_bwd_batch"]
    heads = options["gqa_bwd_heads"]
    groups = options["gqa_bwd_groups"]
    seq_len = options["gqa_bwd_seq"]
    dim = options["gqa_bwd_dim"]
    causal = options["gqa_bwd_causal"]
    if heads % groups:
        raise ValueError("gqa_bwd heads must be divisible by groups")
    prim_func = flashattn_bwd(
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
    total_flops = 10.0 * batch * heads * seq_len * seq_len * dim
    if causal:
        total_flops *= 0.5
    return SearchWorkload(
        prim_func=prim_func,
        out_idx=(7, 8),
        total_flops=total_flops,
        reference_program=partial(
            ref_program, is_causal=causal, groups=groups
        ),
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


OPERATOR = OperatorSpec(
    name="gqa_bwd",
    description=(
        "GQA backward adapted from "
        "examples/flash_attention/example_gqa_bwd.py"
    ),
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
