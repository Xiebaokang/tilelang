"""OverlapPlan workload adapted from ``example_mamba_chunk_state.py``."""

from itertools import product

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig


def add_arguments(parser) -> None:
    group = parser.add_argument_group("Mamba chunk state")
    group.add_argument("--mamba-state-block-m", type=int, nargs="+", default=[64, 128])
    group.add_argument("--mamba-state-block-n", type=int, nargs="+", default=[32, 64, 128])
    group.add_argument("--mamba-state-block-k", type=int, nargs="+", default=[32, 64])
    group.add_argument("--mamba-state-batch", type=int, default=8)
    group.add_argument("--mamba-state-heads", type=int, default=80)
    group.add_argument("--mamba-state-groups", type=int, default=1)
    group.add_argument("--mamba-state-seq", type=int, default=4096)
    group.add_argument("--mamba-state-chunk", type=int, default=256)
    group.add_argument("--mamba-state-dim", type=int, default=64)
    group.add_argument("--mamba-state-dstate", type=int, default=128)


def configurations(options: Options) -> list[TileConfig]:
    dim = options["mamba_state_dim"]
    dstate = options["mamba_state_dstate"]
    chunk = options["mamba_state_chunk"]
    return [
        {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
        }
        for block_m, block_n, block_k in product(
            options["mamba_state_block_m"],
            options["mamba_state_block_n"],
            options["mamba_state_block_k"],
        )
        if dim % block_m == 0
        and dstate % block_n == 0
        and chunk % block_k == 0
    ]


def chunk_state(
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
    threads=128,
):
    dtype = T.float16
    accum_dtype = T.float32
    chunks = T.ceildiv(seq_len, chunk_size)
    log2e = 1.44269504
    b_shape = (batch, seq_len, groups, dstate)
    x_shape = (batch, seq_len, heads, dim)
    scalar_shape = (batch, heads, chunks, chunk_size)
    output_shape = (batch, chunks, heads, dim, dstate)

    @T.prim_func(auto_overlap=True)
    def main(
        B: T.Tensor(b_shape, dtype),
        X: T.Tensor(x_shape, dtype),
        Dt: T.Tensor(scalar_shape, dtype),
        DA: T.Tensor(scalar_shape, dtype),
        Output: T.Tensor(output_shape, dtype),
    ):
        with T.Kernel(
            heads,
            T.ceildiv(dim, block_m) * T.ceildiv(dstate, block_n),
            batch * chunks,
            threads=threads,
        ) as (bz, bx, by):
            x_shared = T.alloc_shared((block_k, block_m), dtype)
            x_local = T.alloc_fragment((block_k, block_m), dtype)
            xt_local = T.alloc_fragment((block_m, block_k), dtype)
            b_shared = T.alloc_shared((block_k, block_n), dtype)
            dt_shared = T.alloc_shared((block_k,), dtype)
            da_shared = T.alloc_shared((block_k,), dtype)
            acc = T.alloc_fragment((block_m, block_n), accum_dtype)
            acc_shared = T.alloc_shared((block_m, block_n), dtype)
            scale = T.alloc_fragment((block_k,), accum_dtype)
            da_last = T.alloc_fragment((1,), accum_dtype)
            da_local = T.alloc_fragment((block_k,), accum_dtype)
            dt_local = T.alloc_fragment((block_k,), accum_dtype)
            batch_idx = by % batch
            chunk_idx = by // batch
            m_idx = bx // T.ceildiv(dstate, block_n)
            n_idx = bx % T.ceildiv(dstate, block_n)
            T.annotate_layout(
                {x_shared: tilelang.layout.make_swizzled_layout(x_shared)}
            )
            da_last[0] = DA[batch_idx, bz, chunk_idx, chunk_size - 1]
            T.clear(acc)
            # auto_overlap ignores this depth; it is retained only for the
            # TileLang native baseline built after tl.auto_overlap is removed.
            for ik in T.Pipelined(
                T.ceildiv(chunk_size, block_k), num_stages=2
            ):
                T.copy(
                    X[
                        batch_idx,
                        chunk_idx * chunk_size
                        + ik * block_k : chunk_idx * chunk_size
                        + (ik + 1) * block_k,
                        bz,
                        m_idx * block_m : (m_idx + 1) * block_m,
                    ],
                    x_shared,
                )
                T.copy(
                    DA[
                        batch_idx,
                        bz,
                        chunk_idx,
                        ik * block_k : (ik + 1) * block_k,
                    ],
                    da_shared,
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
                T.copy(da_shared, da_local)
                T.copy(dt_shared, dt_local)
                for i in T.Parallel(block_k):
                    scale[i] = (
                        T.exp2((da_last[0] - da_local[i]) * log2e) * dt_local[i]
                    )
                T.copy(x_shared, x_local)
                for i, j in T.Parallel(block_m, block_k):
                    xt_local[i, j] = x_local[j, i] * scale[j]
                T.copy(
                    B[
                        batch_idx,
                        chunk_idx * chunk_size
                        + ik * block_k : chunk_idx * chunk_size
                        + (ik + 1) * block_k,
                        bz // (heads // groups),
                        n_idx * block_n : (n_idx + 1) * block_n,
                    ],
                    b_shared,
                )
                T.gemm(xt_local, b_shared, acc)
            T.copy(acc, acc_shared)
            T.copy(
                acc_shared,
                Output[
                    batch_idx,
                    chunk_idx,
                    bz,
                    m_idx * block_m : (m_idx + 1) * block_m,
                    n_idx * block_n : (n_idx + 1) * block_n,
                ],
            )

    return main


def reference(b, x, dt, da):
    batch, seq_len, heads, dim = x.shape
    groups = b.shape[2]
    dstate = b.shape[-1]
    chunks, chunk_size = dt.shape[2:]
    b = b.repeat_interleave(heads // groups, dim=2)
    if seq_len < chunks * chunk_size:
        padding = chunks * chunk_size - seq_len
        x = F.pad(x, (0, 0, 0, 0, 0, padding))
        b = F.pad(b, (0, 0, 0, 0, 0, padding))
    x = x.reshape(batch, chunks, chunk_size, heads, dim)
    b = b.reshape(batch, chunks, chunk_size, heads, dstate)
    decay = torch.exp(da[..., -1:] - da)
    return torch.einsum(
        "bclhn,bhcl,bhcl,bclhp->bchpn",
        b.to(x.dtype),
        decay.to(x.dtype),
        dt.to(x.dtype),
        x,
    )


def build(options: Options, config: TileConfig) -> SearchWorkload:
    batch = options["mamba_state_batch"]
    heads = options["mamba_state_heads"]
    groups = options["mamba_state_groups"]
    seq_len = options["mamba_state_seq"]
    chunk_size = options["mamba_state_chunk"]
    dim = options["mamba_state_dim"]
    dstate = options["mamba_state_dstate"]
    if heads % groups:
        raise ValueError("Mamba state heads must be divisible by groups")
    if seq_len % chunk_size:
        raise ValueError("Mamba state sequence length must be divisible by chunk size")
    if dim % config["block_m"] or dstate % config["block_n"]:
        raise ValueError("Mamba state tile must evenly divide dim and dstate")
    if chunk_size % config["block_k"]:
        raise ValueError("Mamba state block_k must evenly divide chunk size")
    prim_func = chunk_state(
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
    )
    return SearchWorkload(
        prim_func=prim_func,
        out_idx=(4,),
        total_flops=2.0 * batch * seq_len * heads * dim * dstate,
        reference_program=reference,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


OPERATOR = OperatorSpec(
    name="mamba_chunk_state",
    description=(
        "Mamba chunk state adapted from "
        "examples/linear_attention/example_mamba_chunk_state.py"
    ),
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
