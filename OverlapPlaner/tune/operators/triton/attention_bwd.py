"""Shared split-key attention-backward Triton implementation."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from .common import benchmark_autotuned_variant, comparison, controlled_stage_sweep
except ImportError:
    from common import benchmark_autotuned_variant, comparison, controlled_stage_sweep


CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
    for bm in (64, 128)
    for bn in (32, 64)
    for nw in (4, 8)
    for ns in (1, 2, 3, 4)
]
CONTROL = {"BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 4}


def _prune(configs, named_args, **kwargs):
    del kwargs
    if named_args["WARP_SPECIALIZE"]:
        return [config for config in configs if config.num_warps == 4]
    return configs


@triton.autotune(
    configs=CONFIGS,
    key=["SEQ", "HEADS", "GROUPS", "CAUSAL", "WARP_SPECIALIZE"],
    prune_configs_by={"early_config_prune": _prune},
)
@triton.jit
def kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dq_ptr, dk_ptr, dv_ptr,
    BATCH: tl.constexpr, SEQ: tl.constexpr, HEADS: tl.constexpr, GROUPS: tl.constexpr,
    DIM: tl.constexpr, CAUSAL: tl.constexpr, PARTIAL: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    key_block = tl.program_id(0)
    program_head = tl.program_id(1)
    batch = program_head // HEADS
    q_head = program_head % HEADS
    kv_heads: tl.constexpr = HEADS // GROUPS
    kv_head = q_head // GROUPS
    group = q_head % GROUPS
    key = key_block * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, DIM)
    key_mask = key < SEQ
    kv_base = (batch * SEQ * kv_heads + key[:, None] * kv_heads + kv_head) * DIM
    k = tl.load(k_ptr + kv_base + d[None, :], mask=key_mask[:, None], other=0.0)
    v = tl.load(v_ptr + kv_base + d[None, :], mask=key_mask[:, None], other=0.0)
    dk = tl.zeros((BLOCK_M, DIM), tl.float32)
    dv = tl.zeros((BLOCK_M, DIM), tl.float32)
    scale: tl.constexpr = DIM ** -0.5
    scale_log2: tl.constexpr = scale * 1.4426950408889634

    loop_start = key_block * BLOCK_M if CAUSAL else 0
    for query_start in tl.range(
        loop_start, SEQ, BLOCK_N, warp_specialize=WARP_SPECIALIZE
    ):
        query = query_start + tl.arange(0, BLOCK_N)
        query_mask = query < SEQ
        q_base = (batch * SEQ * HEADS + query[:, None] * HEADS + q_head) * DIM
        q = tl.load(q_ptr + q_base + d[None, :], mask=query_mask[:, None], other=0.0)
        do = tl.load(do_ptr + q_base + d[None, :], mask=query_mask[:, None], other=0.0)
        score = tl.dot(k, tl.trans(q))
        lse = tl.load(lse_ptr + (batch * HEADS + q_head) * SEQ + query,
                      mask=query_mask, other=0.0)
        p = tl.exp2(score * scale_log2 - lse[None, :])
        if CAUSAL:
            p = tl.where(key[:, None] <= query[None, :], p, 0.0)
        p = tl.where(key_mask[:, None] & query_mask[None, :], p, 0.0)
        dp = tl.dot(v, tl.trans(do))
        delta = tl.load(delta_ptr + (batch * HEADS + q_head) * SEQ + query,
                        mask=query_mask, other=0.0)
        ds = p * (dp - delta[None, :]) * scale
        dv += tl.dot(p.to(tl.float16), do)
        dk += tl.dot(ds.to(tl.float16), q)
        dq = tl.dot(tl.trans(ds.to(tl.float16)), k)
        tl.atomic_add(dq_ptr + q_base + d[None, :], dq,
                      mask=query_mask[:, None])

    if PARTIAL:
        out_base = (((group * BATCH + batch) * SEQ
                     + key[:, None]) * kv_heads + kv_head) * DIM
    else:
        out_base = (batch * SEQ * HEADS + key[:, None] * HEADS + q_head) * DIM
    mask = key_mask[:, None]
    tl.store(dk_ptr + out_base + d[None, :], dk, mask=mask)
    tl.store(dv_ptr + out_base + d[None, :], dv, mask=mask)


def launch(q, k, v, do, lse, delta, dq, dk, dv, *, groups, causal=False,
           warp_specialize=False, fixed=None, num_stages=None):
    batch, seq, heads, dim = q.shape
    grid = lambda meta: (triton.cdiv(seq, meta["BLOCK_M"]), batch * heads)
    args = (q, k, v, do, lse, delta, dq, dk, dv, batch, seq, heads, groups, dim,
            causal, groups > 1, warp_specialize)
    if fixed is None:
        kernel[grid](*args)
    else:
        kernel.fn[grid](*args, BLOCK_M=fixed["BLOCK_M"],
                        BLOCK_N=fixed["BLOCK_N"], num_warps=fixed["num_warps"],
                        num_stages=num_stages)


def reference(q, k, v, do, lse, delta, *, groups, causal):
    dim = q.shape[-1]
    qf = q.float().permute(0, 2, 1, 3)
    kf = k.repeat_interleave(groups, 2).float().permute(0, 2, 1, 3)
    vf = v.repeat_interleave(groups, 2).float().permute(0, 2, 1, 3)
    dof = do.float().permute(0, 2, 1, 3)
    p = torch.exp2(torch.einsum("bhkd,bhqd->bhkq", kf, qf)
                   * (dim ** -0.5 * 1.4426950408889634) - lse[:, :, None, :])
    if causal:
        index = torch.arange(q.shape[1], device=q.device)
        p = p.masked_fill(index[:, None] > index[None, :], 0)
    dp = torch.einsum("bhkd,bhqd->bhkq", vf, dof)
    ds = p * (dp - delta[:, :, None, :]) * dim ** -0.5
    dk = torch.einsum("bhkq,bhqd->bhkd", ds.half().float(), qf)
    dv = torch.einsum("bhkq,bhqd->bhkd", p.half().float(), dof)
    if groups == 1:
        dq = torch.einsum("bhkq,bhkd->bhqd", ds.half().float(), kf)
        return (dq.permute(0, 2, 1, 3), dk.permute(0, 2, 1, 3).half(),
                dv.permute(0, 2, 1, 3).half())
    batch, heads, seq, dim = dk.shape
    kv_heads = heads // groups
    dk = dk.permute(0, 2, 1, 3).reshape(batch, seq, kv_heads, groups, dim)
    dv = dv.permute(0, 2, 1, 3).reshape(batch, seq, kv_heads, groups, dim)
    dq = torch.einsum("bhkq,bhkd->bhqd", ds.half().float(), kf)
    return (dq.permute(0, 2, 1, 3), dk.permute(3, 0, 1, 2, 4).half(),
            dv.permute(3, 0, 1, 2, 4).half())


def run_operator(name, *, groups, warmup, rep, trials):
    batch, seq, heads, dim = 1, 4096, 32, 128
    torch.manual_seed(17)
    q = torch.randn((batch, seq, heads, dim), device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn((batch, seq, heads // groups, dim), device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn_like(k)
    do = torch.randn_like(q) * 0.1
    lse = torch.full((batch, heads, seq), 8.0, device="cuda", dtype=torch.float32)
    delta = torch.randn_like(lse) * 0.1
    output_shape = ((groups, batch, seq, heads // groups, dim) if groups > 1
                    else (batch, seq, heads, dim))
    dk = torch.empty(output_shape, device="cuda", dtype=torch.float16)
    dv = torch.empty_like(dk)
    dq = torch.zeros_like(q, dtype=torch.float32)

    # Small exact check avoids materializing a full sequence-squared reference.
    size = 128
    sq, sk, sv, sdo = q[:, :size], k[:, :size], v[:, :size], do[:, :size]
    slse, sdelta = lse[:, :, :size], delta[:, :, :size]
    small_shape = ((groups, batch, size, heads // groups, dim) if groups > 1
                   else (batch, size, heads, dim))
    sdk = torch.empty(small_shape, device="cuda", dtype=torch.float16)
    sdv = torch.empty_like(sdk)
    sdq = torch.zeros_like(sq, dtype=torch.float32)
    launch(sq, sk, sv, sdo, slse, sdelta, sdq, sdk, sdv, groups=groups,
           fixed=CONTROL, num_stages=2)
    rdq, rdk, rdv = reference(sq, sk, sv, sdo, slse, sdelta,
                              groups=groups, causal=False)
    torch.testing.assert_close(sdq, rdq, atol=0.08, rtol=0.04)
    torch.testing.assert_close(sdk, rdk, atol=0.05, rtol=0.03)
    torch.testing.assert_close(sdv, rdv, atol=0.05, rtol=0.03)

    broad = [benchmark_autotuned_variant(
        lambda ws=ws: launch(q, k, v, do, lse, delta, dq, dk, dv, groups=groups,
                             warp_specialize=ws), kernel,
        warp_specialize=ws, warmup=warmup, rep=rep, trials=trials,
    ) for ws in (False, True)]
    controlled = controlled_stage_sweep(
        lambda ns, ws: launch(q, k, v, do, lse, delta, dq, dk, dv, groups=groups,
                              warp_specialize=ws, fixed=CONTROL, num_stages=ns),
        CONTROL, warmup=warmup, rep=rep, trials=trials,
    )
    best = min((item for item in broad + controlled if "latency_ms" in item),
               key=lambda item: item["latency_ms"])
    result = comparison(name, best["latency_ms"],
                        total_flops=10.0 * batch * heads * seq * seq * dim,
                        workload={
                            f"{name}_batch": batch, f"{name}_heads": heads,
                            **({f"{name}_groups": groups} if groups > 1 else {}),
                            f"{name}_seq": seq,
                            **({f"{name}_dim_qk": dim, f"{name}_dim_v": dim}
                               if groups > 1 else {f"{name}_dim": dim}),
                            f"{name}_causal": False,
                        })
    result.update(shape={"batch": batch, "seq": seq, "heads": heads,
                         "groups": groups, "dim": dim, "causal": False},
                  autotuned_variants=broad, variants=controlled,
                  selected_variant=best)
    return result
