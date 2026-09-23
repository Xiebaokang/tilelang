"""FA3 workload adapter used by the OverlapPlan tuner."""

from itertools import product

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig
from .example_loader import NativeKernelReference


def add_arguments(parser) -> None:
    group = parser.add_argument_group("FA3")
    group.add_argument("--fa3-block-m", type=int, nargs="+", default=[64,128])
    group.add_argument(
        "--fa3-block-n", type=int, nargs="+", default=[64,128]
    )
    group.add_argument("--fa3-batch", type=int, default=1)
    group.add_argument("--fa3-heads", type=int, default=16)
    group.add_argument("--fa3-seq-q", type=int, default=8192)
    group.add_argument("--fa3-seq-kv", type=int, default=8192)
    group.add_argument("--fa3-dim", type=int, default=128)
    group.add_argument("--fa3-causal", action="store_true")


def configurations(options: Options) -> list[TileConfig]:
    return [
        {"block_m": block_m, "block_n": block_n}
        for block_m, block_n in product(
            options["fa3_block_m"], options["fa3_block_n"]
        )
    ]


def flashattn(
    batch,
    heads,
    seq_q,
    seq_kv,
    dim,
    is_causal,
    block_M=64,
    block_N=64,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

    past_len = seq_kv - seq_q
    assert past_len >= 0, "seq_kv must be greater than or equal to seq_q"

    @T.prim_func(auto_overlap=True)
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(seq_q, block_M), heads, batch, threads=block_M // 64 * 128
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

            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
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

            # Pipeline depth comes from the searched OverlapPlan when
            # auto_overlap is set.
            for k in T.Pipelined(
                loop_range,
                num_stages=2,
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
                            q_idx >= k_idx, 0, -T.infinity(acc_s.dtype)
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
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
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
            T.copy(O_shared, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return main


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_q = Q.size(2)
        seq_kv = K.size(2)
        mask = torch.tril(
            torch.ones(seq_q, seq_kv, device=scores.device),
            seq_kv - seq_q,
        )
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, V)
    return output


def build(options: Options, config: TileConfig) -> SearchWorkload:
    batch = options["fa3_batch"]
    heads = options["fa3_heads"]
    seq_q = options["fa3_seq_q"]
    seq_kv = options["fa3_seq_kv"]
    dim = options["fa3_dim"]
    causal = options["fa3_causal"]
    prim_func = flashattn(
        batch,
        heads,
        seq_q,
        seq_kv,
        dim,
        causal,
        block_M=config["block_m"],
        block_N=config["block_n"],
    )
    total_flops = 4.0 * batch * heads * seq_q * seq_kv * dim
    if causal:
        total_flops *= 0.5
    return SearchWorkload(
        prim_func=prim_func,
        out_idx=(3,),
        total_flops=total_flops,
        reference_program=NativeKernelReference(
            prim_func,
            (3,),
            {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        ),
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


OPERATOR = OperatorSpec(
    name="fa3",
    description="Hopper FlashAttention-3 forward kernel",
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
