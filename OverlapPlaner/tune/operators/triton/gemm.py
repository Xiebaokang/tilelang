from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from .common import benchmark_autotuned_variant, benchmark_cuda, comparison, controlled_stage_sweep, scratch_allocator, triton_config
except ImportError:  # Direct script execution avoids importing TileLang.
    from common import benchmark_autotuned_variant, benchmark_cuda, comparison, controlled_stage_sweep, scratch_allocator, triton_config


CONTROL_CONFIG = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 4}


CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": group_m},
        num_warps=warps,
        num_stages=stages,
    )
    for bm, bn, bk, group_m, warps, stages in [
        (64, 64, 32, 8, 4, 4),
        (64, 128, 32, 8, 4, 4),
        (64, 256, 32, 8, 4, 4),
        (128, 64, 32, 8, 4, 4),
        (128, 128, 32, 8, 4, 4),
        (128, 256, 32, 8, 8, 3),
        (128, 256, 64, 8, 8, 3),
        (128, 128, 64, 8, 4, 4),
        (64, 128, 64, 8, 4, 4),
        (64, 256, 64, 8, 8, 3),
    ]
]


def _prune(configs, named_args, **kwargs):
    del kwargs
    return [c for c in configs if c.num_warps == 4] if named_args["WARP_SPECIALIZE"] else configs


@triton.autotune(configs=CONFIGS, key=["M", "N", "K", "WARP_SPECIALIZE"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_group = GROUP_M * num_pid_n
    group = pid // num_pid_group
    first_m = group * GROUP_M
    group_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % num_pid_group) % group_m
    pid_n = (pid % num_pid_group) // group_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
        b = b_desc.load([k * BLOCK_K, pid_n * BLOCK_N])
        acc = tl.dot(a, b, acc)
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(tl.float16))


def launch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, *, warp_specialize: bool = False) -> None:
    triton.set_allocator(scratch_allocator)
    m, k = a.shape
    _, n = b.shape
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    gemm_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        warp_specialize,
    )


def launch_fixed(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    num_stages: int,
    warp_specialize: bool = False,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
) -> None:
    triton.set_allocator(scratch_allocator)
    m, k = a.shape
    _, n = b.shape
    block_m = block_m or CONTROL_CONFIG["BLOCK_M"]
    block_n = block_n or CONTROL_CONFIG["BLOCK_N"]
    block_k = block_k or CONTROL_CONFIG["BLOCK_K"]
    num_warps = num_warps or CONTROL_CONFIG["num_warps"]
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    gemm_kernel.fn[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        warp_specialize,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=CONTROL_CONFIG["GROUP_M"],
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run(*, warmup: int = 100, rep: int = 400, trials: int = 5) -> dict:
    m = n = k = 4096
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)
    reference = a @ b
    for ws in (False, True):
        launch_fixed(a, b, c, num_stages=2, warp_specialize=ws)
        torch.testing.assert_close(c, reference, atol=1e-1, rtol=1e-2)
    autotuned_variants = [
        benchmark_autotuned_variant(
            lambda ws=ws: launch(a, b, c, warp_specialize=ws),
            gemm_kernel,
            warp_specialize=ws,
            warmup=warmup,
            rep=rep,
            trials=trials,
        )
        for ws in (False,)
    ]
    variants = controlled_stage_sweep(
        lambda num_stages, ws: launch_fixed(
            a, b, c, num_stages=num_stages, warp_specialize=ws
        ),
        CONTROL_CONFIG,
        warmup=warmup,
        rep=rep,
        trials=trials,
    )
    best = min(
        (v for v in autotuned_variants + variants if "latency_ms" in v),
        key=lambda v: v["latency_ms"],
    )
    result = comparison(
        "gemm", best["latency_ms"], total_flops=2.0 * m * n * k,
        workload={"gemm_m": m, "gemm_n": n, "gemm_k": k},
    )
    result.update(
        shape={"m": m, "n": n, "k": k},
        variants=variants,
        autotuned_variants=autotuned_variants,
        selected_variant=best,
    )
    return result
