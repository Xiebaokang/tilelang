from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from .common import benchmark_autotuned_variant, benchmark_cuda, comparison, controlled_stage_sweep, triton_config
except ImportError:  # Direct script execution avoids importing TileLang.
    from common import benchmark_autotuned_variant, benchmark_cuda, comparison, controlled_stage_sweep, triton_config


CONTROL_CONFIG = {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 4}


def _scratch_allocator(size: int, align: int, stream):
    del align, stream
    return torch.empty(size, dtype=torch.int8, device="cuda")


def _configs() -> list[triton.Config]:
    configs = []
    for bm in (64, 128):
        for bn in (32, 64, 128):
            for stages in (2, 3, 4):
                for warps in (4, 8):
                    # This is the same Hopper restriction used by Triton's
                    # official fused-attention tutorial.
                    if bm * bn < 128 * 128 and warps == 8:
                        continue
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn},
                            num_stages=stages,
                            num_warps=warps,
                        )
                    )
    return configs


def _prune_configs(configs, named_args, **kwargs):
    del kwargs
    if named_args["WARP_SPECIALIZE"]:
        # Triton's Hopper warp-specialization pass requires a four-warp input
        # kernel; eight-warps silently keeps the ordinary pipeline.
        return [config for config in configs if config.num_warps == 4]
    return configs


@triton.jit
def _descriptor(ptr, y_dim: tl.constexpr, head_dim: tl.constexpr, block_y: tl.constexpr):
    return tl.make_tensor_descriptor(
        ptr,
        shape=[y_dim, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_y, head_dim],
    )


@triton.jit
def _attention_inner(
    acc,
    l_i,
    m_i,
    q,
    desc_k,
    desc_v,
    offset_y,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    offs_n = tl.arange(0, BLOCK_N)
    offset_kv = offset_y
    for start_n in tl.range(0, N_CTX, BLOCK_N, warp_specialize=WARP_SPECIALIZE):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([offset_kv, 0]).T
        qk = tl.dot(q, k) * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        acc *= alpha[:, None]
        v = desc_v.load([offset_kv, 0])
        acc = tl.dot(p.to(tl.float16), v, acc)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij
        offset_kv += BLOCK_N
    return acc, l_i, m_i


@triton.autotune(
    configs=_configs(),
    key=["N_CTX", "HEAD_DIM", "WARP_SPECIALIZE"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    head_batch = tl.program_id(1)
    y_dim: tl.constexpr = BATCH * HEADS * N_CTX
    offset_y = head_batch * N_CTX
    q_offset = offset_y + start_m * BLOCK_M

    desc_q = _descriptor(q_ptr, y_dim, HEAD_DIM, BLOCK_M)
    desc_k = _descriptor(k_ptr, y_dim, HEAD_DIM, BLOCK_N)
    desc_v = _descriptor(v_ptr, y_dim, HEAD_DIM, BLOCK_N)
    desc_o = _descriptor(o_ptr, y_dim, HEAD_DIM, BLOCK_M)

    q = desc_q.load([q_offset, 0])
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    acc, l_i, _ = _attention_inner(
        acc,
        l_i,
        m_i,
        q,
        desc_k,
        desc_v,
        offset_y,
        start_m,
        SM_SCALE * 1.4426950408889634,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        N_CTX,
        WARP_SPECIALIZE,
    )
    desc_o.store([q_offset, 0], (acc / l_i[:, None]).to(tl.float16))


def launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    warp_specialize: bool,
) -> None:
    triton.set_allocator(_scratch_allocator)
    batch, heads, seq, dim = q.shape
    grid = lambda meta: (triton.cdiv(seq, meta["BLOCK_M"]), batch * heads, 1)
    attention_kernel[grid](
        q,
        k,
        v,
        out,
        batch,
        heads,
        seq,
        dim,
        dim**-0.5,
        warp_specialize,
    )


def launch_fixed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    num_stages: int,
    warp_specialize: bool = False,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
) -> None:
    triton.set_allocator(_scratch_allocator)
    batch, heads, seq, dim = q.shape
    block_m = block_m or CONTROL_CONFIG["BLOCK_M"]
    block_n = block_n or CONTROL_CONFIG["BLOCK_N"]
    num_warps = num_warps or CONTROL_CONFIG["num_warps"]
    grid = (triton.cdiv(seq, block_m), batch * heads, 1)
    attention_kernel.fn[grid](
        q,
        k,
        v,
        out,
        batch,
        heads,
        seq,
        dim,
        dim**-0.5,
        warp_specialize,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _validate() -> None:
    torch.manual_seed(2)
    q = torch.randn((1, 2, 256, 128), device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    for ws in (False, True):
        out = torch.empty_like(q)
        launch_fixed(q, k, v, out, num_stages=2, warp_specialize=ws)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)


def run(*, warmup: int = 100, rep: int = 400, trials: int = 5) -> dict:
    _validate()
    batch, heads, seq, dim = 1, 16, 4096, 128
    torch.manual_seed(3)
    q = torch.randn((batch, heads, seq, dim), device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty_like(q)

    autotuned_variants = [
        benchmark_autotuned_variant(
            lambda ws=ws: launch(q, k, v, out, warp_specialize=ws),
            attention_kernel,
            warp_specialize=ws,
            warmup=warmup,
            rep=rep,
            trials=trials,
        )
        # FA3's broad WS candidate set is known to compile safely.  Other
        # operators use the fixed WS-compatible sweep because one invalid
        # autotune candidate can abort the Hopper WS pass or poison CUDA.
        for ws in (False, True)
    ]

    variants = controlled_stage_sweep(
        lambda num_stages, ws: launch_fixed(
            q, k, v, out, num_stages=num_stages, warp_specialize=ws
        ),
        CONTROL_CONFIG,
        warmup=warmup,
        rep=rep,
        trials=trials,
    )
    successful = [
        item for item in autotuned_variants + variants if "latency_ms" in item
    ]
    best = min(successful, key=lambda item: item["latency_ms"])
    total_flops = 4.0 * batch * heads * seq * seq * dim
    result = comparison(
        "fa3", best["latency_ms"], total_flops=total_flops,
        workload={"fa3_batch": batch, "fa3_heads": heads,
                  "fa3_seq_q": seq, "fa3_seq_kv": seq, "fa3_dim": dim,
                  "fa3_causal": False},
    )
    result.update(
        shape={"batch": batch, "heads": heads, "seq": seq, "dim": dim, "causal": False},
        variants=variants,
        autotuned_variants=autotuned_variants,
        selected_variant=best,
    )
    return result
