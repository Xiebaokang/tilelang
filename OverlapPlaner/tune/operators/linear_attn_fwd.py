"""OverlapPlan workload adapted from ``example_linear_attn_fwd.py``."""

from itertools import product

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig


def add_arguments(parser) -> None:
    group = parser.add_argument_group("linear attention forward")
    group.add_argument("--linear-attn-block-k", type=int, nargs="+", default=[64, 128])
    group.add_argument("--linear-attn-block-v", type=int, nargs="+", default=[64, 128])
    group.add_argument("--linear-attn-batch", type=int, default=1)
    group.add_argument("--linear-attn-seq", type=int, default=8192)
    group.add_argument("--linear-attn-heads", type=int, default=16)
    group.add_argument("--linear-attn-key-dim", type=int, default=128)
    group.add_argument("--linear-attn-value-dim", type=int, default=128)


def configurations(options: Options) -> list[TileConfig]:
    key_dim = options["linear_attn_key_dim"]
    value_dim = options["linear_attn_value_dim"]
    return [
        {"block_k": block_k, "block_v": block_v}
        for block_k, block_v in product(
            options["linear_attn_block_k"], options["linear_attn_block_v"]
        )
        if key_dim % block_k == 0 and value_dim % block_v == 0
    ]


def linear_attn(
    batch: int,
    seq_len: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    block_k: int,
    block_v: int,
):
    chunk = 64
    num_chunks = T.ceildiv(seq_len, chunk)
    num_value_tiles = T.ceildiv(value_dim, block_v)
    num_key_tiles = T.ceildiv(key_dim, block_k)
    scale = key_dim**-0.5
    dtype = T.float16
    accum_dtype = T.float32
    qk_shape = (batch, seq_len, heads, key_dim)
    v_shape = (batch, seq_len, heads, value_dim)
    state_shape = (batch, heads, key_dim, value_dim)

    @T.prim_func(auto_overlap=True)
    def main(
        Q: T.Tensor(qk_shape, dtype),
        K: T.Tensor(qk_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        O: T.Tensor(v_shape, accum_dtype),
        FinalState: T.Tensor(state_shape, accum_dtype),
    ):
        with T.Kernel(num_value_tiles, num_key_tiles, batch * heads) as (
            iv,
            ik,
            ibh,
        ):
            ib = ibh // heads
            ih = ibh % heads
            q = T.alloc_shared((chunk, block_k), dtype)
            k = T.alloc_shared((chunk, block_k), dtype)
            v = T.alloc_shared((chunk, block_v), dtype)
            h = T.alloc_fragment((block_k, block_v), accum_dtype)
            h_shared = T.alloc_shared((block_k, block_v), dtype)
            s = T.alloc_fragment((chunk, chunk), accum_dtype)
            s_shared = T.alloc_shared((chunk, chunk), dtype)
            o = T.alloc_fragment((chunk, block_v), accum_dtype)
            o_shared = T.alloc_shared((chunk, block_v), accum_dtype)

            T.use_swizzle(10)
            T.clear(h)
            for ic in T.Pipelined(0, num_chunks):
                for row, col in T.Parallel(chunk, block_k):
                    q[row, col] = (
                        Q[ib, ic * chunk + row, ih, ik * block_k + col]
                        * scale
                    )
                T.copy(
                    K[
                        ib,
                        ic * chunk : (ic + 1) * chunk,
                        ih,
                        ik * block_k : (ik + 1) * block_k,
                    ],
                    k,
                )
                T.copy(
                    V[
                        ib,
                        ic * chunk : (ic + 1) * chunk,
                        ih,
                        iv * block_v : (iv + 1) * block_v,
                    ],
                    v,
                )
                T.gemm(q, k, s, clear_accum=True, transpose_B=True)
                for row, col in T.Parallel(chunk, chunk):
                    s_shared[row, col] = T.if_then_else(
                        row >= col, s[row, col], 0
                    )
                T.gemm(s_shared, v, o, clear_accum=True)
                T.copy(h, h_shared)
                T.gemm(k, v, h, transpose_A=True)
                T.gemm(q, h_shared, o)
                T.copy(o, o_shared)
                T.atomic_add(
                    O[
                        ib,
                        ic * chunk : (ic + 1) * chunk,
                        ih,
                        iv * block_v : (iv + 1) * block_v,
                    ],
                    o_shared,
                )
            T.copy(
                h,
                FinalState[
                    ib,
                    ih,
                    ik * block_k : (ik + 1) * block_k,
                    iv * block_v : (iv + 1) * block_v,
                ],
            )

    return main


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    batch, seq_len, heads, key_dim = q.shape
    chunk = 64
    chunks = seq_len // chunk
    q = q.float().reshape(batch, chunks, chunk, heads, key_dim)
    k = k.float().reshape(batch, chunks, chunk, heads, key_dim)
    v = v.float().reshape(batch, chunks, chunk, heads, v.shape[-1])
    q = q.permute(0, 3, 1, 2, 4) * key_dim**-0.5
    k = k.permute(0, 3, 1, 2, 4)
    v = v.permute(0, 3, 1, 2, 4)
    kv = torch.matmul(k.transpose(-1, -2), v).cumsum(dim=2)
    final_state = kv[:, :, -1]
    previous = torch.cat((torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]), dim=2)
    inter = torch.matmul(q, previous)
    scores = torch.matmul(q, k.transpose(-1, -2))
    causal = torch.tril(
        torch.ones((chunk, chunk), dtype=torch.bool, device=q.device)
    )
    intra = torch.matmul(scores.masked_fill(~causal, 0), v)
    output = (inter + intra).permute(0, 2, 3, 1, 4)
    return output.reshape(batch, seq_len, heads, v.shape[-1]), final_state


def reference_final_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    _output: torch.Tensor,
):
    """Validate the returned tensor for the example's in-place output ABI."""

    return reference(q, k, v)[1]


def build(options: Options, config: TileConfig) -> SearchWorkload:
    batch = options["linear_attn_batch"]
    seq_len = options["linear_attn_seq"]
    heads = options["linear_attn_heads"]
    key_dim = options["linear_attn_key_dim"]
    value_dim = options["linear_attn_value_dim"]
    block_k = config["block_k"]
    block_v = config["block_v"]
    if seq_len % 64:
        raise ValueError("linear attention sequence length must be divisible by 64")
    if key_dim % block_k:
        raise ValueError("linear attention key dimension must be divisible by block_k")
    if value_dim % block_v:
        raise ValueError("linear attention value dimension must be divisible by block_v")
    prim_func = linear_attn(
        batch, seq_len, heads, key_dim, value_dim, block_k, block_v
    )
    total_flops = 2.0 * batch * heads * (
        seq_len * 64 * (key_dim + value_dim)
        + 2 * seq_len * key_dim * value_dim
    )
    inputs = None
    if torch.cuda.is_available():
        q = F.normalize(
            torch.randn(
                (batch, seq_len, heads, key_dim),
                device="cuda",
                dtype=torch.float16,
            ).float(),
            dim=-1,
        ).half()
        k = F.normalize(
            torch.randn(
                (batch, seq_len, heads, key_dim),
                device="cuda",
                dtype=torch.float16,
            ).float(),
            dim=-1,
        ).half()
        v = torch.randn(
            (batch, seq_len, heads, value_dim),
            device="cuda",
            dtype=torch.float16,
        )
        # Match the example: O is an in-place, caller-zeroed buffer and only
        # FinalState is returned by the compiled kernel.
        output = torch.zeros(
            (batch, seq_len, heads, value_dim),
            device="cuda",
            dtype=torch.float32,
        )
        inputs = (q, k, v, output)
    return SearchWorkload(
        prim_func=prim_func,
        out_idx=(4,),
        total_flops=total_flops,
        reference_program=reference_final_state,
        input_tensors=inputs,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


OPERATOR = OperatorSpec(
    name="linear_attn_fwd",
    description=(
        "Chunked linear attention forward adapted from "
        "examples/linear_attention/example_linear_attn_fwd.py"
    ),
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
