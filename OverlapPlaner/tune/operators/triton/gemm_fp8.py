"""E4M3 GEMM with the same transposed-B ABI as the TileLang operator."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from .common import benchmark_autotuned_variant, comparison, controlled_stage_sweep
except ImportError:
    from common import benchmark_autotuned_variant, comparison, controlled_stage_sweep

CONFIGS = [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                         num_warps=nw, num_stages=ns)
           for bm in (64, 128) for bn in (64, 128, 256)
           for bk in (64, 128, 256) for nw in (4, 8) for ns in (2, 3, 4)]
CONTROL = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4}


def _prune(configs, named_args, **kwargs):
    del kwargs
    return [c for c in configs if c.num_warps == 4] if named_args["WARP_SPECIALIZE"] else configs


@triton.autotune(configs=CONFIGS, key=["M", "N", "K", "WARP_SPECIALIZE"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def fp8_kernel(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               WARP_SPECIALIZE: tl.constexpr, BLOCK_M: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    red = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in tl.range(0, K, BLOCK_K, warp_specialize=WARP_SPECIALIZE):
        kk = start + red
        av = tl.load(a + rows[:, None] * K + kk[None, :],
                     mask=(rows[:, None] < M) & (kk[None, :] < K), other=0.0)
        # Public B is [N,K].
        bv = tl.load(b + cols[None, :] * K + kk[:, None],
                     mask=(cols[None, :] < N) & (kk[:, None] < K), other=0.0)
        acc = tl.dot(av, bv, acc)
    tl.store(c + rows[:, None] * N + cols[None, :], acc,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def launch(a, b, c, *, warp_specialize=False, fixed=None, num_stages=None):
    m, k = a.shape
    n = b.shape[0]
    if fixed is None:
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
        fp8_kernel[grid](a, b, c, m, n, k, warp_specialize)
    else:
        grid = (triton.cdiv(m, fixed["BLOCK_M"]) * triton.cdiv(n, fixed["BLOCK_N"]),)
        fp8_kernel.fn[grid](a, b, c, m, n, k, warp_specialize,
                            BLOCK_M=fixed["BLOCK_M"], BLOCK_N=fixed["BLOCK_N"],
                            BLOCK_K=fixed["BLOCK_K"], num_warps=fixed["num_warps"],
                            num_stages=num_stages)


def run(*, warmup=100, rep=400, trials=5):
    m = n = k = 4096
    torch.manual_seed(13)
    a = torch.randn((m, k), device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
    c = torch.empty((m, n), device="cuda", dtype=torch.float8_e4m3fn)
    # Small correctness case avoids materializing another 4096^2 FP32 matrix.
    # A two-dimensional slice retains the full 4096-element row stride, while
    # this kernel's public ABI is contiguous [M,K]/[N,K].  Materialize the
    # reduced correctness inputs so their strides match that ABI.
    sa = a[:128, :128].contiguous()
    sb = b[:128, :128].contiguous()
    sc = torch.empty((128, 128), device="cuda", dtype=torch.float8_e4m3fn)
    launch(sa, sb, sc, fixed=CONTROL, num_stages=2)
    torch.testing.assert_close(sc.float(), sa.float() @ sb.float().T, atol=1.0, rtol=.15)
    broad = [benchmark_autotuned_variant(
        lambda ws=ws: launch(a, b, c, warp_specialize=ws), fp8_kernel,
        warp_specialize=ws, warmup=warmup, rep=rep, trials=trials)
        for ws in (False, True)]
    controlled = controlled_stage_sweep(
        lambda ns, ws: launch(a, b, c, warp_specialize=ws, fixed=CONTROL,
                              num_stages=ns), CONTROL,
        warmup=warmup, rep=rep, trials=trials)
    best = min((x for x in broad + controlled if "latency_ms" in x),
               key=lambda x: x["latency_ms"])
    result = comparison(
        "gemm_fp8", best["latency_ms"], total_flops=2.0*m*n*k,
        workload={"gemm_fp8_m":m,"gemm_fp8_n":n,"gemm_fp8_k":k})
    result.update(shape={"m": m, "n": n, "k": k}, autotuned_variants=broad,
                  variants=controlled, selected_variant=best)
    return result
