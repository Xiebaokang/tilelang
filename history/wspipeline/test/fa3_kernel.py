import argparse
import itertools
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tvm.target import Target

import history.wspipeline as wsp


@dataclass(frozen=True)
class FA3BenchmarkResult:
    compiled: object
    latency_ms: float
    tflops: float


def get_configs():
    iter_params = dict(block_M=[128], block_N=[128], threads=[256])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn(
    batch,
    heads,
    seq_q,
    seq_kv,
    dim,
    is_causal,
    block_M=64,
    block_N=64,
    threads=128,
    auto_wsp=True,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

    past_len = seq_kv - seq_q
    assert past_len >= 0, "seq_kv must be greater than or equal to seq_q"

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
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
                T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M + past_len, block_N))
                if is_causal
                else T.ceildiv(seq_kv, block_N)
            )

            # The manual value is intentionally ignored because auto_wsp owns
            # the complete pipeline schedule, including its depth.
            for k in T.Pipelined(
                loop_range,
                num_stages=2,
                auto_wsp=auto_wsp,
            ):
                T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i + past_len
                        k_idx = k * block_N + j
                        acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_kv, -T.infinity(acc_s.dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

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
        mask = torch.tril(torch.ones(seq_q, seq_kv, device=scores.device), seq_kv - seq_q)
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, V)
    return output


def print_program_schedule(analysis, candidate):
    """Print the exact realized schedule selected during CUDA lowering."""

    logical = candidate.logical

    print("\nSelected realized program schedule")
    print(
        "  nodes (id, name, region, group):",
        [
            (
                analysis.operation_for(node).operation_id,
                node.name,
                analysis.operation_for(node).region_id,
                logical.partition.node_groups[node],
            )
            for node in analysis.nodes
        ],
    )
    for region, schedule in zip(analysis.regions, logical.region_schedules):
        print(
            f"  region {region.region_id} ({region.kind.value}):",
            {
                "operations": list(region.operation_ids),
                "num_stages": schedule.num_stages,
            },
        )
        if schedule.stages:
            print(
                "    stages:",
                {
                    analysis.operation_for(node).operation_id: stage
                    for node, stage in schedule.stages.items()
                },
            )
        print(
            "    group orders:",
            {
                group_id: [
                    analysis.operation_for(node).operation_id
                    for node in sorted(order, key=order.__getitem__)
                ]
                for group_id, order in schedule.group_orders.items()
            },
        )
    print(
        "  cross-group dependencies:",
        [
            (
                analysis.operation_for(edge.producer).operation_id,
                analysis.operation_for(edge.consumer).operation_id,
                edge.producer_group,
                edge.consumer_group,
                edge.scope.value,
                tuple(sorted(edge.dependency_kinds)),
            )
        for edge in logical.cross_group_dependencies
        ],
    )
    print(
        "  warp allocation:",
        {
            "groups": [
                (group.group_id, group.first_warp, group.warp_count)
                for group in candidate.warp_allocation.groups
            ],
            "threads": candidate.warp_allocation.effective_threads,
            "setmaxnreg": candidate.register_allocation.setmaxnreg_enabled,
            "register_domains": [
                (
                    domain.group_id,
                    domain.first_warp,
                    domain.warp_count,
                    domain.register_count,
                    "inc" if domain.is_increase else "dec",
                )
                for domain in candidate.register_allocation.domains
            ],
        },
    )
    print(
        "  buffers:",
        [
            (
                buffer.buffer_id,
                buffer.name,
                buffer.version_count,
                buffer.communication.value,
            )
            for buffer in candidate.buffers.buffers
        ],
    )
    print("  synchronization channels:", len(candidate.channels))
    print(
        "  register pressure:",
        [
            (
                group.group_id,
                group.mandatory_registers_per_thread,
                group.estimated_registers_per_thread,
                group.register_limit_per_thread,
            )
            for group in candidate.register_pressure.groups
        ],
    )


def make_program_schedule_planner(examined_limit):
    """Automatically select group count, versions, and physical MMA width."""

    selected = {}

    def planner(global_symbol, analysis, target):
        candidates = wsp.enumerate_realized_schedules(
            analysis,
            target,
            examined_limit=examined_limit,
        )

        scored_candidates = [
            (
                wsp.score_realized_schedule(analysis, candidate, target),
                candidate,
            )
            for candidate in candidates
        ]
        if not scored_candidates:
            raise RuntimeError(
                f"no complete automatic program schedule was found for "
                f"{global_symbol}"
            )
        best_by_group_count = {}
        for candidate_score, scored_candidate in scored_candidates:
            group_count = scored_candidate.logical.partition.num_groups
            previous = best_by_group_count.get(group_count)
            if previous is None or candidate_score < previous:
                best_by_group_count[group_count] = candidate_score
        print("  best score by inferred group count:", best_by_group_count)
        score, candidate = min(
            scored_candidates,
            key=lambda item: item[0],
        )
        selected[global_symbol] = candidate
        print_program_schedule(analysis, candidate)
        print("  analytical score:", score)
        return candidate

    return planner, selected


def make_fa3_prim_func(
    batch=1,
    heads=1,
    seq_q=256,
    seq_kv=256,
    dim=64,
    is_causal=False,
    auto_wsp=True,
):
    """Return the FA3-style TIR used by the automatic WSP test."""

    return flashattn.jit_impl.get_tir(
        batch,
        heads,
        seq_q,
        seq_kv,
        dim,
        is_causal,
        block_M=128,
        block_N=128,
        threads=256,
        auto_wsp=auto_wsp,
    )


def make_cuda_target(arch="sm_90a"):
    return Target(
        {
            "kind": "cuda",
            "arch": arch,
            "max_threads_per_block": 1024,
            "max_shared_memory_per_block": 232448,
        }
    )


def main(
    batch: int = 1,
    heads: int = 1,
    seq_q: int = 256,
    seq_kv: int = 256,
    dim: int = 64,
    is_causal: bool = False,
    examined_limit: int = 1000,
    arch: str = "sm_90a",
    compile_only: bool = False,
    benchmark: bool = True,
):
    flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5
    prim = make_fa3_prim_func(batch, heads, seq_q, seq_kv, dim, is_causal)
    target = make_cuda_target(arch)
    planner, selected = make_program_schedule_planner(examined_limit)
    # Include the scheduling request in the input IR so the persistent JIT
    # cache cannot reuse a kernel compiled without this planner.
    prim = prim.with_attr(
        "tl.program_schedule.request",
        (
            f"selection=automatic,arch={arch},"
            "proxy_fence=producer_release_v1,"
            "fragment_import_lifetime=consumer_order_v1"
        ),
    )
    if not compile_only and not torch.cuda.is_available():
        raise RuntimeError(
            "the correctness check requires a CUDA device; "
            "use --compile_only to validate schedule selection and lowering"
        )
    with wsp.use_schedule_planner(planner):
        if compile_only:
            with target, tilelang.transform.PassContext(
                config={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
            ):
                compiled = tilelang.lower(
                    prim,
                    target=target,
                    enable_device_compile=True,
                )
            kernel_source = compiled.kernel_source
        else:
            compiled = tilelang.compile(
                prim,
                out_idx=[3],
                target=target,
                execution_backend="cython",
                pass_configs={
                    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
                },
            )
            kernel_source = compiled.get_kernel_source()
    if not selected:
        raise RuntimeError("the CUDA compilation did not invoke the program planner")
    print(
        "Full CUDA compilation succeeded:",
        {
            "functions": tuple(selected),
            "source_lines": len(kernel_source.splitlines()),
            "has_register_reconfiguration": (
                "warpgroup_reg_alloc" in kernel_source
                and "warpgroup_reg_dealloc" in kernel_source
            ),
        },
    )
    if compile_only:
        return compiled

    ref_program_processed = partial(ref_program, is_causal=is_causal)
    profiler = compiled.get_profiler()
    profiler.assert_allclose(ref_program_processed, rtol=0.01, atol=0.01)
    print(
        "Correctness check passed:",
        {
            "shape": (batch, heads, seq_q, seq_kv, dim),
            "causal": is_causal,
            "rtol": 0.01,
            "atol": 0.01,
        },
    )
    if not benchmark:
        return compiled
    latency = profiler.do_bench(warmup=500)
    tflops = total_flops / latency * 1e-9
    print(
        "Tile-lang: {:.2f} ms  {:.2f} TFlops".format(
            latency, tflops
        )
    )
    return FA3BenchmarkResult(compiled, latency, tflops)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--heads", type=int, default=1, help="heads")
    parser.add_argument("--seq_q", type=int, default=256, help="query sequence length")
    parser.add_argument("--seq_kv", type=int, default=256, help="key/value sequence length")
    parser.add_argument("--dim", type=int, default=64, help="dim")
    parser.add_argument(
        "--is_causal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable or disable causal masking",
    )
    parser.add_argument("--examined_limit", type=int, default=1000)
    parser.add_argument("--arch", type=str, default="sm_90a")
    parser.add_argument("--compile_only", action="store_true")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="benchmark after the correctness check",
    )
    args = parser.parse_args()
    main(
        args.batch,
        args.heads,
        args.seq_q,
        args.seq_kv,
        args.dim,
        args.is_causal,
        args.examined_limit,
        args.arch,
        args.compile_only,
        args.benchmark,
    )
