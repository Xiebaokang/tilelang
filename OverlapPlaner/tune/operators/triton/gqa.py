"""Grouped-query attention forward with BSHD tensors."""
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
    for bm in (64, 128, 256)
    for bn in (32, 64, 128)
    for nw in (4, 8)
    for ns in (2, 3, 4)
    if not (bm * bn < 8192 and nw == 8)
]
CONTROL = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4}


def _prune(configs, named_args, **kwargs):
    del kwargs
    return [c for c in configs if c.num_warps == 4] if named_args["WARP_SPECIALIZE"] else configs


@triton.autotune(configs=CONFIGS, key=["SEQ", "HEADS", "KV_HEADS", "WARP_SPECIALIZE"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def gqa_kernel(q, k, v, out, SEQ: tl.constexpr, HEADS: tl.constexpr,
               KV_HEADS: tl.constexpr, DIM: tl.constexpr,
               WARP_SPECIALIZE: tl.constexpr, BLOCK_M: tl.constexpr,
               BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // HEADS
    head = bh % HEADS
    kv_head = head // (HEADS // KV_HEADS)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, DIM)
    q_off = ((batch * SEQ + rows[:, None]) * HEADS + head) * DIM + d[None, :]
    qv = tl.load(q + q_off, mask=rows[:, None] < SEQ, other=0.0)
    acc = tl.zeros((BLOCK_M, DIM), tl.float32)
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    for start in tl.range(0, SEQ, BLOCK_N, warp_specialize=WARP_SPECIALIZE):
        cols = start + tl.arange(0, BLOCK_N)
        kv_off = ((batch * SEQ + cols[:, None]) * KV_HEADS + kv_head) * DIM + d[None, :]
        kval = tl.load(k + kv_off, mask=cols[:, None] < SEQ, other=0.0)
        scores = tl.dot(qv, tl.trans(kval)) * (DIM ** -0.5 * 1.4426950408889634)
        scores = tl.where(cols[None, :] < SEQ, scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, axis=1))
        alpha = tl.exp2(maximum - new_max)
        probs = tl.exp2(scores - new_max[:, None])
        acc *= alpha[:, None]
        vval = tl.load(v + kv_off, mask=cols[:, None] < SEQ, other=0.0)
        acc = tl.dot(probs.to(tl.float16), vval, acc)
        denominator = denominator * alpha + tl.sum(probs, axis=1)
        maximum = new_max
    tl.store(out + q_off, (acc / denominator[:, None]).to(tl.float16),
             mask=rows[:, None] < SEQ)


def launch(q, k, v, out, *, warp_specialize=False):
    batch, seq, heads, dim = q.shape
    kv_heads = k.shape[2]
    grid = lambda meta: (triton.cdiv(seq, meta["BLOCK_M"]), batch * heads)
    gqa_kernel[grid](q, k, v, out, seq, heads, kv_heads, dim, warp_specialize)


def launch_fixed(q, k, v, out, *, num_stages, warp_specialize=False):
    batch, seq, heads, dim = q.shape
    grid = (triton.cdiv(seq, CONTROL["BLOCK_M"]), batch * heads)
    gqa_kernel.fn[grid](q, k, v, out, seq, heads, k.shape[2], dim,
                        warp_specialize, BLOCK_M=CONTROL["BLOCK_M"],
                        BLOCK_N=CONTROL["BLOCK_N"], num_warps=CONTROL["num_warps"],
                        num_stages=num_stages)


def run(*, warmup=100, rep=400, trials=5):
    batch, seq, heads, groups, dim = 1, 4096, 32, 8, 128
    kv_heads = heads // groups
    torch.manual_seed(11)
    q = torch.randn((batch, seq, heads, dim), device="cuda", dtype=torch.float16) * .5
    k = torch.randn((batch, seq, kv_heads, dim), device="cuda", dtype=torch.float16) * .5
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    small = 256
    qs, ks, vs = q[:, :small], k[:, :small], v[:, :small]
    test = torch.empty_like(qs)
    launch_fixed(qs, ks, vs, test, num_stages=2)
    ref = torch.nn.functional.scaled_dot_product_attention(
        qs.permute(0, 2, 1, 3),
        ks.repeat_interleave(groups, dim=2).permute(0, 2, 1, 3),
        vs.repeat_interleave(groups, dim=2).permute(0, 2, 1, 3),
    ).permute(0, 2, 1, 3)
    torch.testing.assert_close(test, ref, atol=2e-2, rtol=1e-2)
    broad = [benchmark_autotuned_variant(
        lambda ws=ws: launch(q, k, v, out, warp_specialize=ws), gqa_kernel,
        warp_specialize=ws, warmup=warmup, rep=rep, trials=trials)
        for ws in (False, True)]
    controlled = controlled_stage_sweep(
        lambda ns, ws: launch_fixed(q, k, v, out, num_stages=ns, warp_specialize=ws),
        CONTROL, warmup=warmup, rep=rep, trials=trials)
    best = min((x for x in broad + controlled if "latency_ms" in x),
               key=lambda x: x["latency_ms"])
    result = comparison("gqa", best["latency_ms"],
                        total_flops=4.0 * batch * heads * seq * seq * dim,
                        workload={"gqa_batch": batch, "gqa_heads": heads,
                                  "gqa_groups": groups, "gqa_seq": seq,
                                  "gqa_dim": dim, "gqa_causal": False})
    result.update(shape={"batch": batch, "seq": seq, "heads": heads,
                         "kv_heads": kv_heads, "dim": dim},
                  autotuned_variants=broad, variants=controlled,
                  selected_variant=best)
    return result
