"""Overlaper search kernel adapted from examples/linear_attention/example_mamba_chunk_scan.py."""

from itertools import product

import tilelang
import tilelang.language as T
import torch

from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig


def add_arguments(parser) -> None:
    group = parser.add_argument_group("Mamba chunk scan")
    group.add_argument("--mamba-block-m", type=int, nargs="+", default=[64, 128, 256])
    group.add_argument("--mamba-block-n", type=int, nargs="+", default=[32, 64])
    group.add_argument("--mamba-block-k", type=int, nargs="+", default=[64, 128, 256])
    group.add_argument("--mamba-batch", type=int, default=1)
    group.add_argument("--mamba-heads", type=int, default=16)
    group.add_argument("--mamba-groups", type=int, default=2)
    group.add_argument("--mamba-seq", type=int, default=2048)
    group.add_argument("--mamba-chunk", type=int, default=256)
    group.add_argument("--mamba-dim", type=int, default=64)
    group.add_argument("--mamba-dstate", type=int, default=128)


def configurations(options: Options) -> list[TileConfig]:
    return [
        {"block_m": block_m, "block_n": block_n, "block_k": block_k}
        for block_m, block_n, block_k in product(
            options["mamba_block_m"],
            options["mamba_block_n"],
            options["mamba_block_k"],
        )
    ]


def chunk_scan_fwd(
    batch,
    seqlen,
    chunk_size,
    ngroups,
    nheads,
    headdim,
    dstate,
    block_M=64,
    block_N=64,
    block_K=64,
    auto_wsp=True,
):
    dtype = T.float16
    accum_dtype = T.float32
    nchunks = seqlen // chunk_size
    p = 1.44269504

    @T.prim_func
    def main(
        cb: T.Tensor((batch, nchunks, ngroups, chunk_size, chunk_size), dtype),
        x: T.Tensor((batch, seqlen, nheads, headdim), dtype),
        dt: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),
        C: T.Tensor((batch, seqlen, ngroups, dstate), dtype),
        prev_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), dtype),
        D: T.Tensor((nheads,), dtype),
        Output: T.Tensor((batch, seqlen, nheads, headdim), dtype),
    ):
        with T.Kernel(
            nheads,
            T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N),
            batch * nchunks,
            threads=block_M // 64 * 128,
        ) as (bz, bx, by):
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
            cb_shared = T.alloc_shared((block_M, block_K), dtype)
            cb_local = T.alloc_fragment((block_M, block_K), dtype)
            dA_cs_k_shared = T.alloc_shared((block_K,), dtype)
            dA_cs_k_local = T.alloc_fragment((block_K,), accum_dtype)
            dA_cs_m_local = T.alloc_fragment((block_M,), accum_dtype)
            dt_shared = T.alloc_shared((block_K,), dtype)
            dt_local = T.alloc_fragment((block_K,), accum_dtype)
            x_shared = T.alloc_shared(
                (block_K, block_N), dtype, scope="shared.dyn"
            )
            dA_cs_m_shared = T.alloc_shared((block_M,), dtype)
            scale_m_local = T.alloc_fragment((block_M,), accum_dtype)
            C_shared = T.alloc_shared((block_M, dstate), dtype)
            prev_state_shared = T.alloc_shared((block_N, dstate), dtype)
            D_local = T.alloc_fragment((1,), accum_dtype)
            x_residual_shared = T.alloc_shared((block_M, block_N), dtype)
            x_residual_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            batch_idx = by % batch
            chunk_idx = by // batch
            m_idx = bx // T.ceildiv(headdim, block_N)
            n_idx = bx % T.ceildiv(headdim, block_N)

            T.annotate_layout(
                {
                    cb_shared: tilelang.layout.make_swizzled_layout(cb_shared),
                    x_residual_shared: tilelang.layout.make_swizzled_layout(
                        x_residual_shared
                    ),
                }
            )
            T.no_set_max_nreg()
            T.copy(
                dA_cumsum[
                    batch_idx,
                    bz,
                    chunk_idx,
                    m_idx * block_M : (m_idx + 1) * block_M,
                ],
                dA_cs_m_shared,
            )
            T.copy(dA_cs_m_shared, dA_cs_m_local)
            T.clear(acc_o)
            for i in T.Parallel(block_M):
                scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
            T.copy(
                C[
                    batch_idx,
                    chunk_idx * chunk_size
                    + m_idx * block_M : chunk_idx * chunk_size
                    + (m_idx + 1) * block_M,
                    bz // (nheads // ngroups),
                    0:dstate,
                ],
                C_shared,
            )
            T.copy(
                prev_states[
                    batch_idx,
                    chunk_idx,
                    bz,
                    n_idx * block_N : (n_idx + 1) * block_N,
                    0:dstate,
                ],
                prev_state_shared,
            )
            T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] *= scale_m_local[i]

            loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)
            for k in T.Pipelined(loop_range, num_stages=2, auto_wsp=auto_wsp):
                T.copy(
                    cb[
                        batch_idx,
                        chunk_idx,
                        bz // (nheads // ngroups),
                        m_idx * block_M : (m_idx + 1) * block_M,
                        k * block_K : (k + 1) * block_K,
                    ],
                    cb_shared,
                )
                T.copy(cb_shared, cb_local)
                T.copy(
                    dA_cumsum[
                        batch_idx,
                        bz,
                        chunk_idx,
                        k * block_K : (k + 1) * block_K,
                    ],
                    dA_cs_k_shared,
                )
                T.copy(dA_cs_k_shared, dA_cs_k_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = cb_local[i, j] * T.exp2(
                        dA_cs_m_local[i] * p - dA_cs_k_local[j] * p
                    )
                T.copy(
                    dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K],
                    dt_shared,
                )
                T.copy(dt_shared, dt_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] *= dt_local[j]
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = T.if_then_else(
                        m_idx * block_M + i >= k * block_K + j,
                        cb_local[i, j],
                        0,
                    )
                T.copy(
                    x[
                        batch_idx,
                        chunk_idx * chunk_size
                        + k * block_K : chunk_idx * chunk_size
                        + (k + 1) * block_K,
                        bz,
                        n_idx * block_N : (n_idx + 1) * block_N,
                    ],
                    x_shared,
                )
                T.gemm(cb_local, x_shared, acc_o)

            D_local[0] = D[bz]
            T.copy(
                x[
                    batch_idx,
                    chunk_idx * chunk_size
                    + m_idx * block_M : chunk_idx * chunk_size
                    + (m_idx + 1) * block_M,
                    bz,
                    n_idx * block_N : (n_idx + 1) * block_N,
                ],
                x_residual_shared,
            )
            T.copy(x_residual_shared, x_residual_local)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] += x_residual_local[i, j] * D_local[0]
            T.copy(acc_o, acc_o_shared)
            T.copy(
                acc_o_shared,
                Output[
                    batch_idx,
                    chunk_idx * chunk_size + m_idx * block_M : chunk_idx
                    * chunk_size
                    + (m_idx + 1) * block_M,
                    bz,
                    n_idx * block_N : (n_idx + 1) * block_N,
                ],
            )

    return main


def ref_program(cb, x, dt, dA_cumsum, C, prev_states, D):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    ngroups = cb.shape[2]
    dstate = C.shape[-1]
    repeat = nheads // ngroups
    C = C.repeat_interleave(repeat, dim=2)
    cb = cb.repeat_interleave(repeat, dim=2)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = cb * decay.permute(0, 2, 1, 3, 4)
    causal_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=x.device, dtype=torch.bool)
    )
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    x_chunks = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
    out = torch.einsum(
        "bchls,bhcs,bcshp->bclhp",
        scores_decay.to(x.dtype),
        dt.to(x.dtype),
        x_chunks,
    )
    state_decay_out = torch.exp(dA_cumsum.permute(0, 2, 3, 1).unsqueeze(-1))
    C_chunks = C.reshape(batch, nchunks, chunk_size, nheads, dstate)
    out_prev = (
        torch.einsum(
            "bclhn,bchpn->bclhp",
            C_chunks,
            prev_states.to(C.dtype),
        )
        * state_decay_out
    )
    out = (out + out_prev).reshape(batch, seqlen, nheads, headdim)
    return out + x * D[:, None]


def build(options: Options, config: TileConfig) -> SearchWorkload:
    batch = options["mamba_batch"]
    nheads = options["mamba_heads"]
    ngroups = options["mamba_groups"]
    seqlen = options["mamba_seq"]
    chunk_size = options["mamba_chunk"]
    headdim = options["mamba_dim"]
    dstate = options["mamba_dstate"]
    if nheads % ngroups:
        raise ValueError("mamba heads must be divisible by groups")
    if seqlen % chunk_size:
        raise ValueError("mamba seq must be divisible by chunk")
    prim_func = chunk_scan_fwd(
        batch,
        seqlen,
        chunk_size,
        ngroups,
        nheads,
        headdim,
        dstate,
        block_M=config["block_m"],
        block_N=config["block_n"],
        block_K=config["block_k"],
        auto_wsp=True,
    )
    total_flops = (
        2.0 * batch * seqlen * chunk_size * nheads * headdim * 0.5
        + 2.0 * batch * seqlen * nheads * headdim * dstate
    )
    return SearchWorkload(
        prim_func=prim_func,
        out_idx=(7,),
        total_flops=total_flops,
        reference_program=ref_program,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


OPERATOR = OperatorSpec(
    name="mamba_scan",
    description=(
        "Mamba chunk scan adapted from "
        "examples/linear_attention/example_mamba_chunk_scan.py"
    ),
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
