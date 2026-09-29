"""No-split MLA decode with latent KV used as both key and value."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from .common import benchmark_autotuned_variant, comparison, controlled_stage_sweep
except ImportError:
    from common import benchmark_autotuned_variant, comparison, controlled_stage_sweep

CONFIGS = [triton.Config({"BLOCK_H": bh, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
           for bh in (16, 32, 64) for bn in (32, 64)
           for nw in (4, 8) for ns in (2, 3, 4)]
CONTROL = {"BLOCK_H": 32, "BLOCK_N": 64, "num_warps": 4}


def _prune(configs, named_args, **kwargs):
    del kwargs
    return [c for c in configs if c.num_warps == 4] if named_args["WARP_SPECIALIZE"] else configs


@triton.autotune(configs=CONFIGS, key=["HEADS", "SEQ", "WARP_SPECIALIZE"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def mla_kernel(q, qpe, kv, kpe, out, HEADS: tl.constexpr, SEQ: tl.constexpr,
               DIM: tl.constexpr, PE_DIM: tl.constexpr,
               WARP_SPECIALIZE: tl.constexpr, BLOCK_H: tl.constexpr,
               BLOCK_N: tl.constexpr):
    batch = tl.program_id(1)
    head_start = tl.program_id(0) * BLOCK_H
    hs = head_start + tl.arange(0, BLOCK_H)
    d = tl.arange(0, DIM)
    dp = tl.arange(0, PE_DIM)
    qv = tl.load(q + (batch * HEADS + hs[:, None]) * DIM + d[None, :],
                 mask=hs[:, None] < HEADS, other=0.0)
    qpv = tl.load(qpe + (batch * HEADS + hs[:, None]) * PE_DIM + dp[None, :],
                  mask=hs[:, None] < HEADS, other=0.0)
    acc = tl.zeros((BLOCK_H, DIM), tl.float32)
    maximum = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_H,), tl.float32)
    scale: tl.constexpr = (DIM + PE_DIM) ** -0.5 * 1.4426950408889634
    for start in tl.range(0, SEQ, BLOCK_N, warp_specialize=WARP_SPECIALIZE):
        rows = start + tl.arange(0, BLOCK_N)
        kval = tl.load(kv + (batch * SEQ + rows[:, None]) * DIM + d[None, :],
                       mask=rows[:, None] < SEQ, other=0.0)
        kpval = tl.load(kpe + (batch * SEQ + rows[:, None]) * PE_DIM + dp[None, :],
                        mask=rows[:, None] < SEQ, other=0.0)
        scores = (tl.dot(qv, tl.trans(kval)) + tl.dot(qpv, tl.trans(kpval))) * scale
        scores = tl.where(rows[None, :] < SEQ, scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, axis=1))
        alpha = tl.exp2(maximum - new_max)
        probability = tl.exp2(scores - new_max[:, None])
        acc *= alpha[:, None]
        acc = tl.dot(probability.to(tl.float16), kval, acc)
        denominator = denominator * alpha + tl.sum(probability, axis=1)
        maximum = new_max
    tl.store(out + (batch * HEADS + hs[:, None]) * DIM + d[None, :],
             acc / denominator[:, None], mask=hs[:, None] < HEADS)


def launch(q, qpe, kv, kpe, out, *, warp_specialize=False, fixed=None,
           num_stages=None):
    batch, heads, dim = q.shape
    seq, pe_dim = kv.shape[1], qpe.shape[-1]
    if fixed is None:
        grid = lambda meta: (triton.cdiv(heads, meta["BLOCK_H"]), batch)
        mla_kernel[grid](q, qpe, kv, kpe, out, heads, seq, dim, pe_dim,
                         warp_specialize)
    else:
        grid = (triton.cdiv(heads, fixed["BLOCK_H"]), batch)
        mla_kernel.fn[grid](q, qpe, kv, kpe, out, heads, seq, dim, pe_dim,
                            warp_specialize, BLOCK_H=fixed["BLOCK_H"],
                            BLOCK_N=fixed["BLOCK_N"], num_warps=fixed["num_warps"],
                            num_stages=num_stages)


def run(*, warmup=100, rep=400, trials=5):
    batch, heads, seq, dim, pe_dim = 1, 128, 8192, 512, 64
    torch.manual_seed(14)
    q = torch.randn((batch, heads, dim), device="cuda", dtype=torch.float16) * .2
    qpe = torch.randn((batch, heads, pe_dim), device="cuda", dtype=torch.float16) * .2
    kv = torch.randn((batch, seq, dim), device="cuda", dtype=torch.float16) * .2
    kpe = torch.randn((batch, seq, pe_dim), device="cuda", dtype=torch.float16) * .2
    out = torch.empty_like(q)
    # Validate a smaller sequence with exactly the same feature dimensions.
    skv, skpe = kv[:, :256], kpe[:, :256]
    test = torch.empty_like(q)
    launch(q, qpe, skv, skpe, test, fixed=CONTROL, num_stages=2)
    score = (torch.einsum("bhd,bsd->bhs", q.float(), skv.float())
             + torch.einsum("bhd,bsd->bhs", qpe.float(), skpe.float()))
    ref = torch.einsum("bhs,bsd->bhd", torch.softmax(score/(dim+pe_dim)**.5, -1),
                       skv.float()).half()
    torch.testing.assert_close(test, ref, atol=.05, rtol=.02)
    broad = [benchmark_autotuned_variant(
        lambda ws=ws: launch(q, qpe, kv, kpe, out, warp_specialize=ws), mla_kernel,
        warp_specialize=ws, warmup=warmup, rep=rep, trials=trials)
        for ws in (False, True)]
    controlled = controlled_stage_sweep(
        lambda ns, ws: launch(q, qpe, kv, kpe, out, warp_specialize=ws,
                              fixed=CONTROL, num_stages=ns), CONTROL,
        warmup=warmup, rep=rep, trials=trials)
    best = min((x for x in broad + controlled if "latency_ms" in x),
               key=lambda x: x["latency_ms"])
    flops = 2.0*batch*heads*seq*(dim+pe_dim) + 2.0*batch*heads*seq*dim
    result = comparison(
        "mla", best["latency_ms"], total_flops=flops,
        workload={"mla_batch": batch, "mla_heads": heads,
                  "mla_kv_heads": 1, "mla_seq": seq, "mla_dim": dim,
                  "mla_pe_dim": pe_dim},
    )
    result.update(shape={"batch":batch,"heads":heads,"seq":seq,"dim":dim,
                         "pe_dim":pe_dim}, autotuned_variants=broad,
                  variants=controlled, selected_variant=best)
    return result
