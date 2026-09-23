"""Grouped-query attention backward search kernel.

Adapted from ``examples/flash_attention/example_gqa_bwd.py``.  This uses the
split dK/dV form from the example so every kernel block writes a deterministic
output; the partial group dimension is intentionally retained for validation.
"""

from itertools import product

import tilelang
import tilelang.language as T
import torch

from .example_loader import NativeKernelReference
from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig


def add_arguments(parser) -> None:
    group = parser.add_argument_group("GQA backward")
    group.add_argument("--gqa-bwd-block-m", type=int, nargs="+", default=[128])
    group.add_argument("--gqa-bwd-block-n", type=int, nargs="+", default=[32])
    group.add_argument("--gqa-bwd-batch", type=int, default=1)
    group.add_argument("--gqa-bwd-heads", type=int, default=32)
    group.add_argument("--gqa-bwd-groups", type=int, default=4)
    group.add_argument("--gqa-bwd-seq", type=int, default=8192)
    group.add_argument("--gqa-bwd-dim-qk", type=int, default=128)
    group.add_argument("--gqa-bwd-dim-v", type=int, default=128)
    group.add_argument("--gqa-bwd-causal", action="store_true")


def configurations(options: Options) -> list[TileConfig]:
    return [
        {"block_m": block_m, "block_n": block_n}
        for block_m, block_n in product(
            options["gqa_bwd_block_m"], options["gqa_bwd_block_n"]
        )
    ]


def make_dq_layout(dq):
    # atomicAdd cannot be vectorized. Match the 8x8 GEMM fragment layout.
    return T.Layout(
        dq.shape,
        lambda b, l, h, d: [
            b,
            l // 8,
            h,
            d // 8,
            d % 2,
            4 * (l % 8) + (d % 8) // 2,
        ],
    )


def flashattn_bwd_split(
    batch,
    heads,
    seq_len,
    dim_qk,
    dim_v,
    is_causal,
    block_M,
    block_N,
    groups=1,
):
    sm_scale = (1.0 / dim_qk) ** 0.5
    scale = sm_scale * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    dk_shape = [groups, batch, seq_len, head_kv, dim_qk]
    dv_shape = [groups, batch, seq_len, head_kv, dim_v]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func(auto_overlap=True)
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        dO: T.Tensor([batch, seq_len, heads, dim_v], dtype),
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
            threads=256,
        ) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim_qk], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            q = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_M, dim_v], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            do = T.alloc_shared([block_N, dim_v], dtype)
            dv = T.alloc_fragment([block_M, dim_v], accum_dtype)
            dk = T.alloc_fragment([block_M, dim_qk], accum_dtype)
            dq = T.alloc_fragment([block_N, dim_qk], accum_dtype)
            dv_shared = T.alloc_shared([block_M, dim_v], dtype)
            dk_shared = T.alloc_shared([block_M, dim_qk], dtype)

            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                K[
                    bz,
                    by * block_M : (by + 1) * block_M,
                    bx // groups,
                    :,
                ],
                K_shared,
            )
            T.copy(
                V[
                    bz,
                    by * block_M : (by + 1) * block_M,
                    bx // groups,
                    :,
                ],
                V_shared,
            )
            T.clear(dv)
            T.clear(dk)
            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_st, loop_ed, num_stages=2):
                T.copy(
                    Q[bz, k * block_N : (k + 1) * block_N, bx, :],
                    q,
                )
                T.clear(qkT)
                T.gemm(
                    K_shared,
                    q,
                    qkT,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(
                    dO[bz, k * block_N : (k + 1) * block_N, bx, :],
                    do,
                )
                T.clear(dsT)
                T.gemm(
                    V_shared,
                    do,
                    dsT,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(
                    lse[bz, bx, k * block_N : (k + 1) * block_N],
                    lse_shared,
                )
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(
                        qkT[i, j] * scale - lse_shared[j]
                    )
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(
                            by * block_M + i <= k * block_N + j,
                            qkT[i, j],
                            0,
                        )
                T.copy(qkT, qkT_cast)
                T.gemm(
                    qkT_cast,
                    do,
                    dv,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(
                    Delta[bz, bx, k * block_N : (k + 1) * block_N],
                    delta,
                )
                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = (
                        qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                    )
                T.gemm(
                    dsT_cast,
                    q,
                    dk,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                for i, j in T.Parallel(block_N, dim_qk):
                    T.atomic_add(
                        dQ[bz, k * block_N + i, bx, j], dq[i, j]
                    )

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


def ref_program(Q, K, V, dO, lse, Delta, _dQ, is_causal, groups):
    dim_qk = Q.size(-1)
    sm_scale = dim_qk**-0.5
    scale = sm_scale * 1.44269504

    q = Q.float().permute(0, 2, 1, 3)
    k = K.repeat_interleave(groups, dim=2).float().permute(0, 2, 1, 3)
    v = V.repeat_interleave(groups, dim=2).float().permute(0, 2, 1, 3)
    do = dO.float().permute(0, 2, 1, 3)
    qk = torch.einsum("bhkd,bhqd->bhkq", k, q)
    p = torch.exp2(qk * scale - lse.float().unsqueeze(-2))
    if is_causal:
        seq_len = Q.size(1)
        key_index = torch.arange(seq_len, device=Q.device)[:, None]
        query_index = torch.arange(seq_len, device=Q.device)[None, :]
        p = p.masked_fill(key_index > query_index, 0)
    dp = torch.einsum("bhkd,bhqd->bhkq", v, do)
    ds = p * (dp - Delta.float().unsqueeze(-2)) * sm_scale
    dk = torch.einsum("bhkq,bhqd->bhkd", ds, q)
    dv = torch.einsum("bhkq,bhqd->bhkd", p, do)

    batch, _, seq_len, dim_k = dk.shape
    head_kv = K.size(2)
    dk = dk.permute(0, 2, 1, 3).reshape(
        batch, seq_len, head_kv, groups, dim_k
    )
    dv = dv.permute(0, 2, 1, 3).reshape(
        batch, seq_len, head_kv, groups, V.size(-1)
    )
    return (
        dk.permute(3, 0, 1, 2, 4).half(),
        dv.permute(3, 0, 1, 2, 4).half(),
    )


def build(options: Options, config: TileConfig) -> SearchWorkload:
    batch = options["gqa_bwd_batch"]
    heads = options["gqa_bwd_heads"]
    groups = options["gqa_bwd_groups"]
    seq_len = options["gqa_bwd_seq"]
    dim_qk = options["gqa_bwd_dim_qk"]
    dim_v = options["gqa_bwd_dim_v"]
    causal = options["gqa_bwd_causal"]
    block_m = config["block_m"]
    block_n = config["block_n"]
    if heads % groups:
        raise ValueError("gqa_bwd heads must be divisible by groups")
    if seq_len % block_m or seq_len % block_n:
        raise ValueError("gqa_bwd sequence length must divide both tile sizes")
    prim_func = flashattn_bwd_split(
        batch,
        heads,
        seq_len,
        dim_qk,
        dim_v,
        causal,
        block_m,
        block_n,
        groups,
    )
    total_flops = (
        2.0
        * batch
        * heads
        * seq_len
        * seq_len
        * (3 * dim_qk + 2 * dim_v)
    )
    if causal:
        total_flops *= 0.5
    return SearchWorkload(
        prim_func=prim_func,
        out_idx=(7, 8),
        total_flops=total_flops,
        reference_program=NativeKernelReference(
            prim_func, (7, 8),
            {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        ),
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


OPERATOR = OperatorSpec(
    name="gqa_bwd",
    description=(
        "Grouped-query attention backward adapted from "
        "examples/flash_attention/example_gqa_bwd.py"
    ),
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
