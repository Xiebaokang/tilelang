"""OverlapPlan workload adapted from ``example_mamba_chunk_scan.py``."""

from itertools import product

import tilelang
import tilelang.language as T
import torch

from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig


def add_arguments(parser) -> None:
    group = parser.add_argument_group("Mamba chunk scan")
    group.add_argument("--mamba-scan-block-m", type=int, nargs="+", default=[64, 128])
    group.add_argument("--mamba-scan-block-n", type=int, nargs="+", default=[32, 64])
    group.add_argument("--mamba-scan-block-k", type=int, nargs="+", default=[64, 128])
    group.add_argument("--mamba-scan-block-dstate", type=int, nargs="+", default=[128])
    group.add_argument("--mamba-scan-batch", type=int, default=8)
    group.add_argument("--mamba-scan-heads", type=int, default=80)
    group.add_argument("--mamba-scan-groups", type=int, default=1)
    group.add_argument("--mamba-scan-seq", type=int, default=4096)
    group.add_argument("--mamba-scan-chunk", type=int, default=256)
    group.add_argument("--mamba-scan-dim", type=int, default=64)
    group.add_argument("--mamba-scan-dstate", type=int, default=128)


def configurations(options: Options) -> list[TileConfig]:
    chunk = options["mamba_scan_chunk"]
    dim = options["mamba_scan_dim"]
    dstate = options["mamba_scan_dstate"]
    return [
        {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "block_dstate": block_dstate,
        }
        for block_m, block_n, block_k, block_dstate in product(
            options["mamba_scan_block_m"],
            options["mamba_scan_block_n"],
            options["mamba_scan_block_k"],
            options["mamba_scan_block_dstate"],
        )
        if chunk % block_m == 0
        and dim % block_n == 0
        and chunk % block_k == 0
        and block_dstate == dstate
    ]


def chunk_scan(
    batch,
    seq_len,
    chunk_size,
    groups,
    heads,
    dim,
    dstate,
    block_m,
    block_n,
    block_k,
    block_dstate,
    num_stages=2,
    threads=128,
):
    dtype = T.float16
    accum_dtype = T.float32
    chunks = T.ceildiv(seq_len, chunk_size)
    log2e = 1.44269504
    cb_shape = (batch, chunks, groups, chunk_size, chunk_size)
    x_shape = (batch, seq_len, heads, dim)
    scalar_shape = (batch, heads, chunks, chunk_size)
    c_shape = (batch, seq_len, groups, dstate)
    prev_shape = (batch, chunks, heads, dim, dstate)
    d_shape = (heads,)

    @T.prim_func(auto_overlap=True)
    def main(
        CB: T.Tensor(cb_shape, dtype),
        X: T.Tensor(x_shape, dtype),
        Dt: T.Tensor(scalar_shape, dtype),
        DA: T.Tensor(scalar_shape, dtype),
        C: T.Tensor(c_shape, dtype),
        Prev: T.Tensor(prev_shape, dtype),
        D: T.Tensor(d_shape, dtype),
        Output: T.Tensor(x_shape, dtype),
    ):
        with T.Kernel(
            heads,
            T.ceildiv(chunk_size, block_m) * T.ceildiv(dim, block_n),
            batch * chunks,
            threads=threads,
        ) as (bz, bx, by):
            acc = T.alloc_fragment((block_m, block_n), accum_dtype)
            acc_shared = T.alloc_shared((block_m, block_n), dtype)
            cb_shared = T.alloc_shared((block_m, block_k), dtype)
            cb_local = T.alloc_fragment((block_m, block_k), dtype)
            da_k_shared = T.alloc_shared((block_k,), dtype)
            da_k_local = T.alloc_fragment((block_k,), accum_dtype)
            da_m_local = T.alloc_fragment((block_m,), accum_dtype)
            dt_shared = T.alloc_shared((block_k,), dtype)
            dt_local = T.alloc_fragment((block_k,), accum_dtype)
            # Let the TileLang/OverlapPlan lowering choose the dynamic shared
            # representation instead of pinning it in the source workload.
            x_shared = T.alloc_shared((block_k, block_n), dtype)
            da_m_shared = T.alloc_shared((block_m,), dtype)
            scale_m = T.alloc_fragment((block_m,), accum_dtype)
            c_shared = T.alloc_shared((block_m, block_dstate), dtype)
            prev_shared = T.alloc_shared((block_n, block_dstate), dtype)
            d_local = T.alloc_fragment((1,), accum_dtype)
            residual_shared = T.alloc_shared((block_m, block_n), dtype)
            residual_local = T.alloc_fragment((block_m, block_n), accum_dtype)
            batch_idx = by % batch
            chunk_idx = by // batch
            m_idx = bx // T.ceildiv(dim, block_n)
            n_idx = bx % T.ceildiv(dim, block_n)
            T.annotate_layout(
                {
                    cb_shared: tilelang.layout.make_swizzled_layout(cb_shared),
                    residual_shared: tilelang.layout.make_swizzled_layout(
                        residual_shared
                    ),
                }
            )
            T.no_set_max_nreg()
            T.copy(
                DA[
                    batch_idx,
                    bz,
                    chunk_idx,
                    m_idx * block_m : (m_idx + 1) * block_m,
                ],
                da_m_shared,
            )
            T.copy(da_m_shared, da_m_local)
            T.clear(acc)
            for i in T.Parallel(block_m):
                scale_m[i] = T.exp2(da_m_local[i] * log2e)
            T.copy(
                    C[
                        batch_idx,
                        chunk_idx * chunk_size
                        + m_idx * block_m : chunk_idx * chunk_size
                        + (m_idx + 1) * block_m,
                        bz // (heads // groups),
                        0:block_dstate,
                ],
                c_shared,
            )
            T.copy(
                Prev[
                    batch_idx,
                    chunk_idx,
                    bz,
                    n_idx * block_n : (n_idx + 1) * block_n,
                    0:block_dstate,
                ],
                prev_shared,
            )
            T.gemm(c_shared, prev_shared, acc, transpose_B=True)
            for i, j in T.Parallel(block_m, block_n):
                acc[i, j] *= scale_m[i]
            loop_range = T.ceildiv((m_idx + 1) * block_m, block_k)
            for ik in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    CB[
                        batch_idx,
                        chunk_idx,
                        bz // (heads // groups),
                        m_idx * block_m : (m_idx + 1) * block_m,
                        ik * block_k : (ik + 1) * block_k,
                    ],
                    cb_shared,
                )
                T.copy(cb_shared, cb_local)
                T.copy(
                    DA[
                        batch_idx,
                        bz,
                        chunk_idx,
                        ik * block_k : (ik + 1) * block_k,
                    ],
                    da_k_shared,
                )
                T.copy(da_k_shared, da_k_local)
                for i, j in T.Parallel(block_m, block_k):
                    cb_local[i, j] *= T.exp2(
                        (da_m_local[i] - da_k_local[j]) * log2e
                    )
                T.copy(
                    Dt[
                        batch_idx,
                        bz,
                        chunk_idx,
                        ik * block_k : (ik + 1) * block_k,
                    ],
                    dt_shared,
                )
                T.copy(dt_shared, dt_local)
                for i, j in T.Parallel(block_m, block_k):
                    cb_local[i, j] *= dt_local[j]
                for i, j in T.Parallel(block_m, block_k):
                    cb_local[i, j] = T.if_then_else(
                        m_idx * block_m + i >= ik * block_k + j,
                        cb_local[i, j],
                        0,
                    )
                T.copy(
                    X[
                        batch_idx,
                        chunk_idx * chunk_size
                        + ik * block_k : chunk_idx * chunk_size
                        + (ik + 1) * block_k,
                        bz,
                        n_idx * block_n : (n_idx + 1) * block_n,
                    ],
                    x_shared,
                )
                T.gemm(cb_local, x_shared, acc)
            d_local[0] = D[bz]
            T.copy(
                X[
                    batch_idx,
                    chunk_idx * chunk_size
                    + m_idx * block_m : chunk_idx * chunk_size
                    + (m_idx + 1) * block_m,
                    bz,
                    n_idx * block_n : (n_idx + 1) * block_n,
                ],
                residual_shared,
            )
            T.copy(residual_shared, residual_local)
            for i, j in T.Parallel(block_m, block_n):
                acc[i, j] += residual_local[i, j] * d_local[0]
            T.copy(acc, acc_shared)
            T.copy(
                acc_shared,
                Output[
                    batch_idx,
                    chunk_idx * chunk_size
                    + m_idx * block_m : chunk_idx * chunk_size
                    + (m_idx + 1) * block_m,
                    bz,
                    n_idx * block_n : (n_idx + 1) * block_n,
                ],
            )

    return main


def reference(cb, x, dt, da, c, prev, d):
    batch, seq_len, heads, dim = x.shape
    groups = cb.shape[2]
    chunks, chunk_size = dt.shape[2:]
    c = c.repeat_interleave(heads // groups, dim=2)
    cb = cb.repeat_interleave(heads // groups, dim=2)
    delta = da[..., :, None] - da[..., None, :]
    decay = torch.exp(delta).permute(0, 2, 1, 3, 4)
    scores = cb * decay
    mask = torch.tril(
        torch.ones((chunk_size, chunk_size), dtype=torch.bool, device=x.device)
    )
    scores = scores.masked_fill(~mask, 0)
    x_chunks = x.reshape(batch, chunks, chunk_size, heads, dim)
    out = torch.einsum(
        "bchls,bhcs,bcshp->bclhp",
        scores.to(x.dtype),
        dt.to(x.dtype),
        x_chunks,
    )
    c_chunks = c.reshape(batch, chunks, chunk_size, heads, c.shape[-1])
    state_decay = torch.exp(da.permute(0, 2, 3, 1)).unsqueeze(-1)
    out += torch.einsum("bclhn,bchpn->bclhp", c_chunks, prev.to(c.dtype)) * state_decay
    out = out.reshape(batch, seq_len, heads, dim)
    return out + x * d.reshape(1, 1, heads, 1)


def build(options: Options, config: TileConfig) -> SearchWorkload:
    batch = options["mamba_scan_batch"]
    heads = options["mamba_scan_heads"]
    groups = options["mamba_scan_groups"]
    seq_len = options["mamba_scan_seq"]
    chunk_size = options["mamba_scan_chunk"]
    dim = options["mamba_scan_dim"]
    dstate = options["mamba_scan_dstate"]
    if heads % groups:
        raise ValueError("Mamba scan heads must be divisible by groups")
    if seq_len % chunk_size:
        raise ValueError("Mamba scan sequence length must be divisible by chunk size")
    if chunk_size % config["block_m"] or chunk_size % config["block_k"]:
        raise ValueError("Mamba scan M/K tiles must evenly divide chunk size")
    if dim % config["block_n"]:
        raise ValueError("Mamba scan block_n must evenly divide dim")
    if config["block_dstate"] != dstate:
        raise ValueError("Mamba scan block_dstate must equal dstate")
    prim_func = chunk_scan(
        batch,
        seq_len,
        chunk_size,
        groups,
        heads,
        dim,
        dstate,
        config["block_m"],
        config["block_n"],
        config["block_k"],
        config["block_dstate"],
    )
    total_flops = (
        batch * seq_len * chunk_size * heads * dim
        + 2.0 * batch * seq_len * heads * dim * dstate
    )
    return SearchWorkload(
        prim_func=prim_func,
        out_idx=(7,),
        total_flops=total_flops,
        reference_program=reference,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


OPERATOR = OperatorSpec(
    name="mamba_chunk_scan",
    description=(
        "Mamba chunk scan adapted from "
        "examples/linear_attention/example_mamba_chunk_scan.py"
    ),
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
