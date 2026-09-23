"""Multi-head attention backward search kernel.

Adapted from the BSHD implementation in
``examples/flash_attention/example_mha_bwd_bshd.py``.
"""

from functools import partial
from itertools import product

import tilelang
import tilelang.language as T
import torch

from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig


def add_arguments(parser) -> None:
    group = parser.add_argument_group("MHA backward")
    group.add_argument("--mha-bwd-block-m", type=int, nargs="+", default=[128])
    group.add_argument("--mha-bwd-block-n", type=int, nargs="+", default=[32])
    group.add_argument("--mha-bwd-batch", type=int, default=1)
    group.add_argument("--mha-bwd-heads", type=int, default=16)
    group.add_argument("--mha-bwd-seq", type=int, default=1024)
    group.add_argument("--mha-bwd-dim", type=int, default=64)
    group.add_argument("--mha-bwd-causal", action="store_true")


def configurations(options: Options) -> list[TileConfig]:
    return [
        {"block_m": block_m, "block_n": block_n}
        for block_m, block_n in product(
            options["mha_bwd_block_m"], options["mha_bwd_block_n"]
        )
    ]


def make_dq_layout(dq):
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


def flashattn_bwd(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M,
    block_N,
):
    sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * 1.44269504
    shape = [batch, seq_len, heads, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func(auto_overlap=True)
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dQ: T.Tensor(shape, accum_dtype),
        dK: T.Tensor(shape, dtype),
        dV: T.Tensor(shape, dtype),
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
            gradient_shared = T.alloc_shared([block_M, dim], dtype)

            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                K[bz, by * block_M : (by + 1) * block_M, bx, :],
                K_shared,
            )
            T.copy(
                V[bz, by * block_M : (by + 1) * block_M, bx, :],
                V_shared,
            )
            T.clear(dv)
            T.clear(dk)
            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_st, loop_ed, num_stages=2):
                T.copy(
                    Q[bz, k * block_N : (k + 1) * block_N, bx, :], q
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
                T.copy(
                    dO[bz, k * block_N : (k + 1) * block_N, bx, :], do
                )
                T.clear(dsT)
                T.gemm(
                    V_shared,
                    do,
                    dsT,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
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
                for i, j in T.Parallel(block_N, dim):
                    T.atomic_add(
                        dQ[bz, k * block_N + i, bx, j], dq[i, j]
                    )
            # dK and dV have the same tile shape. Making their native shared
            # reuse explicit also creates the dependency that prevents an
            # order from overwriting the staging tile before its TMA store.
            T.copy(dv, gradient_shared)
            T.copy(
                gradient_shared,
                dV[bz, by * block_M : (by + 1) * block_M, bx, :],
            )
            T.copy(dk, gradient_shared)
            T.copy(
                gradient_shared,
                dK[bz, by * block_M : (by + 1) * block_M, bx, :],
            )

    return main


def ref_program(Q, K, V, dO, lse, Delta, _dQ, is_causal):
    dim = Q.size(-1)
    sm_scale = dim**-0.5
    scale = sm_scale * 1.44269504
    q = Q.float().permute(0, 2, 1, 3)
    k = K.float().permute(0, 2, 1, 3)
    v = V.float().permute(0, 2, 1, 3)
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
    return dk.permute(0, 2, 1, 3).half(), dv.permute(0, 2, 1, 3).half()


def build(options: Options, config: TileConfig) -> SearchWorkload:
    batch = options["mha_bwd_batch"]
    heads = options["mha_bwd_heads"]
    seq_len = options["mha_bwd_seq"]
    dim = options["mha_bwd_dim"]
    causal = options["mha_bwd_causal"]
    block_m = config["block_m"]
    block_n = config["block_n"]
    if seq_len % block_m or seq_len % block_n:
        raise ValueError("mha_bwd sequence length must divide both tile sizes")
    prim_func = flashattn_bwd(
        batch,
        heads,
        seq_len,
        dim,
        causal,
        block_m,
        block_n,
    )
    total_flops = 10.0 * batch * heads * seq_len * seq_len * dim
    if causal:
        total_flops *= 0.5
    return SearchWorkload(
        prim_func=prim_func,
        out_idx=(7, 8),
        total_flops=total_flops,
        reference_program=partial(ref_program, is_causal=causal),
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


OPERATOR = OperatorSpec(
    name="mha_bwd",
    description=(
        "Multi-head attention backward adapted from "
        "examples/flash_attention/example_mha_bwd_bshd.py"
    ),
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
