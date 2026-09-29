from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from .common import benchmark_autotuned_variant, benchmark_cuda, comparison, controlled_stage_sweep, scratch_allocator, triton_config
except ImportError:
    from common import benchmark_autotuned_variant, benchmark_cuda, comparison, controlled_stage_sweep, scratch_allocator, triton_config


CONTROL_CONFIG = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4}


def _configs() -> list[triton.Config]:
    return [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
            num_warps=warps,
            num_stages=stages,
        )
        for bm in (64, 128)
        for bn in (32, 64)
        for bk in (64, 128)
        for warps in (4, 8)
        for stages in (2, 3, 4)
    ]


def _prune(configs, named_args, **kwargs):
    del kwargs
    if named_args["WARP_SPECIALIZE"]:
        return [config for config in configs if config.num_warps == 4]
    return configs


@triton.autotune(
    configs=_configs(),
    key=["CHUNK_SIZE", "DIM", "DSTATE", "WARP_SPECIALIZE"],
    prune_configs_by={"early_config_prune": _prune},
)
@triton.jit
def chunk_scan_kernel(
    cb_ptr,
    x_ptr,
    dt_ptr,
    da_ptr,
    c_ptr,
    prev_ptr,
    d_ptr,
    out_ptr,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    CHUNKS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DIM: tl.constexpr,
    DSTATE: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    head = tl.program_id(0)
    tile = tl.program_id(1)
    batch_chunk = tl.program_id(2)
    batch = batch_chunk // CHUNKS
    chunk = batch_chunk % CHUNKS
    num_n = tl.cdiv(DIM, BLOCK_N)
    m_tile = tile // num_n
    n_tile = tile % num_n
    group = head // (HEADS // GROUPS)
    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_state = tl.arange(0, DSTATE)
    m_mask = offs_m < CHUNK_SIZE
    n_mask = offs_n < DIM

    da_base = ((batch * HEADS + head) * CHUNKS + chunk) * CHUNK_SIZE
    da_m = tl.load(da_ptr + da_base + offs_m, mask=m_mask, other=0.0).to(tl.float32)
    seq_m = chunk * CHUNK_SIZE + offs_m
    c_ptrs = c_ptr + (((batch * SEQ_LEN + seq_m[:, None]) * GROUPS + group) * DSTATE + offs_state[None, :])
    prev_ptrs = prev_ptr + (((((batch * CHUNKS + chunk) * HEADS + head) * DIM + offs_n[:, None]) * DSTATE) + offs_state[None, :])
    c = tl.load(c_ptrs, mask=m_mask[:, None], other=0.0)
    prev = tl.load(prev_ptrs, mask=n_mask[:, None], other=0.0)
    acc = tl.dot(c, tl.trans(prev))
    acc *= tl.exp(da_m)[:, None]

    offs_k = tl.arange(0, BLOCK_K)
    loop_end = tl.cdiv((m_tile + 1) * BLOCK_M, BLOCK_K)
    cb_desc = tl.make_tensor_descriptor(
        cb_ptr + ((batch * CHUNKS + chunk) * GROUPS + group) * CHUNK_SIZE * CHUNK_SIZE,
        shape=[CHUNK_SIZE, CHUNK_SIZE], strides=[CHUNK_SIZE, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    x_desc = tl.make_tensor_descriptor(
        x_ptr + batch * SEQ_LEN * HEADS * DIM + head * DIM,
        shape=[SEQ_LEN, DIM], strides=[HEADS * DIM, 1],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    dt_desc = tl.make_tensor_descriptor(
        dt_ptr + da_base, shape=[1, CHUNK_SIZE], strides=[CHUNK_SIZE, 1],
        block_shape=[1, BLOCK_K],
    )
    da_desc = tl.make_tensor_descriptor(
        da_ptr + da_base, shape=[1, CHUNK_SIZE], strides=[CHUNK_SIZE, 1],
        block_shape=[1, BLOCK_K],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr + batch * SEQ_LEN * HEADS * DIM + head * DIM,
        shape=[SEQ_LEN, DIM], strides=[HEADS * DIM, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    for ko in tl.range(0, loop_end, warp_specialize=WARP_SPECIALIZE):
        k = ko * BLOCK_K + offs_k
        cb = cb_desc.load([m_tile * BLOCK_M, ko * BLOCK_K]).to(tl.float32)
        da_k = da_desc.load([0, ko * BLOCK_K]).reshape(BLOCK_K).to(tl.float32)
        dt = dt_desc.load([0, ko * BLOCK_K]).reshape(BLOCK_K).to(tl.float32)
        scores = cb * tl.exp(da_m[:, None] - da_k[None, :]) * dt[None, :]
        scores = tl.where(offs_m[:, None] >= k[None, :], scores, 0.0)
        x = x_desc.load([chunk * CHUNK_SIZE + ko * BLOCK_K, n_tile * BLOCK_N])
        acc = tl.dot(scores.to(tl.float16), x, acc)

    residual_ptrs = x_ptr + (((batch * SEQ_LEN + seq_m[:, None]) * HEADS + head) * DIM + offs_n[None, :])
    residual = tl.load(residual_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    d = tl.load(d_ptr + head).to(tl.float32)
    acc += residual * d
    out_desc.store([chunk * CHUNK_SIZE + m_tile * BLOCK_M, n_tile * BLOCK_N], acc.to(tl.float16))


def launch(cb, x, dt, da, c, prev, d, out, *, warp_specialize: bool) -> None:
    triton.set_allocator(scratch_allocator)
    batch, seq_len, heads, dim = x.shape
    groups = cb.shape[2]
    dstate = c.shape[-1]
    chunks, chunk_size = dt.shape[2:]
    grid = lambda meta: (
        heads,
        triton.cdiv(chunk_size, meta["BLOCK_M"]) * triton.cdiv(dim, meta["BLOCK_N"]),
        batch * chunks,
    )
    chunk_scan_kernel[grid](
        cb,
        x,
        dt,
        da,
        c,
        prev,
        d,
        out,
        seq_len,
        heads,
        groups,
        chunks,
        chunk_size,
        dim,
        dstate,
        warp_specialize,
    )


def launch_fixed(
    cb, x, dt, da, c, prev, d, out, *, num_stages: int, warp_specialize: bool = False,
    block_m: int | None = None, block_n: int | None = None,
    block_k: int | None = None, num_warps: int | None = None,
) -> None:
    """Launch a fixed tile/warp configuration for a controlled comparison."""
    triton.set_allocator(scratch_allocator)
    batch, seq_len, heads, dim = x.shape
    groups = cb.shape[2]
    dstate = c.shape[-1]
    chunks, chunk_size = dt.shape[2:]
    block_m = block_m or CONTROL_CONFIG["BLOCK_M"]
    block_n = block_n or CONTROL_CONFIG["BLOCK_N"]
    block_k = block_k or CONTROL_CONFIG["BLOCK_K"]
    num_warps = num_warps or CONTROL_CONFIG["num_warps"]
    grid = (
        heads,
        triton.cdiv(chunk_size, block_m) * triton.cdiv(dim, block_n),
        batch * chunks,
    )
    chunk_scan_kernel.fn[grid](
        cb,
        x,
        dt,
        da,
        c,
        prev,
        d,
        out,
        seq_len,
        heads,
        groups,
        chunks,
        chunk_size,
        dim,
        dstate,
        warp_specialize,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def reference(cb, x, dt, da, c, prev, d):
    batch, seq_len, heads, dim = x.shape
    groups = cb.shape[2]
    chunks, chunk = dt.shape[2:]
    c = c.repeat_interleave(heads // groups, dim=2)
    cb = cb.repeat_interleave(heads // groups, dim=2)
    delta = da[..., :, None] - da[..., None, :]
    scores = cb * torch.exp(delta).permute(0, 2, 1, 3, 4)
    scores = scores.masked_fill(
        ~torch.tril(torch.ones((chunk, chunk), dtype=torch.bool, device=x.device)),
        0,
    )
    x_chunks = x.reshape(batch, chunks, chunk, heads, dim)
    out = torch.einsum("bchls,bhcs,bcshp->bclhp", scores.float(), dt.float(), x_chunks.float())
    c_chunks = c.reshape(batch, chunks, chunk, heads, c.shape[-1])
    state_decay = torch.exp(da.permute(0, 2, 3, 1)).unsqueeze(-1)
    out += torch.einsum("bclhn,bchpn->bclhp", c_chunks.float(), prev.float()) * state_decay
    return (out.reshape(batch, seq_len, heads, dim) + x.float() * d.reshape(1, 1, heads, 1)).half()


def _validate() -> None:
    batch, chunks, chunk, heads, groups, dim, dstate = 1, 1, 64, 2, 1, 32, 128
    seq = chunks * chunk
    cb = torch.randn((batch, chunks, groups, chunk, chunk), device="cuda", dtype=torch.float16) * 0.1
    x = torch.randn((batch, seq, heads, dim), device="cuda", dtype=torch.float16) * 0.1
    dt = torch.rand((batch, heads, chunks, chunk), device="cuda", dtype=torch.float16)
    da = torch.cumsum(-torch.rand_like(dt).float() * 0.01, dim=-1).half()
    c = torch.randn((batch, seq, groups, dstate), device="cuda", dtype=torch.float16) * 0.1
    prev = torch.randn((batch, chunks, heads, dim, dstate), device="cuda", dtype=torch.float16) * 0.1
    d = torch.randn((heads,), device="cuda", dtype=torch.float16) * 0.1
    out = torch.empty_like(x)
    launch_fixed(cb, x, dt, da, c, prev, d, out, num_stages=2, warp_specialize=False)
    torch.testing.assert_close(out, reference(cb, x, dt, da, c, prev, d), atol=5e-2, rtol=2e-2)


def run(*, warmup: int = 100, rep: int = 400, trials: int = 5) -> dict:
    _validate()
    batch, seq, heads, groups = 2, 8192, 80, 1
    chunk, dim, dstate = 256, 64, 128
    chunks = seq // chunk
    torch.manual_seed(5)
    cb = torch.randn((batch, chunks, groups, chunk, chunk), device="cuda", dtype=torch.float16) * 0.1
    x = torch.randn((batch, seq, heads, dim), device="cuda", dtype=torch.float16) * 0.1
    dt = torch.rand((batch, heads, chunks, chunk), device="cuda", dtype=torch.float16)
    da = torch.cumsum(-torch.rand_like(dt).float() * 0.01, dim=-1).half()
    c = torch.randn((batch, seq, groups, dstate), device="cuda", dtype=torch.float16) * 0.1
    prev = torch.randn((batch, chunks, heads, dim, dstate), device="cuda", dtype=torch.float16) * 0.1
    d = torch.randn((heads,), device="cuda", dtype=torch.float16) * 0.1
    out = torch.empty_like(x)
    autotuned_variants = [
        benchmark_autotuned_variant(
            lambda ws=ws: launch(
                cb, x, dt, da, c, prev, d, out, warp_specialize=ws
            ),
            chunk_scan_kernel,
            warp_specialize=ws,
            warmup=warmup,
            rep=rep,
            trials=trials,
        )
        for ws in (False,)
    ]
    variants = controlled_stage_sweep(
        lambda num_stages, ws: launch_fixed(
            cb, x, dt, da, c, prev, d, out,
            num_stages=num_stages, warp_specialize=ws,
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
    total_flops = batch * seq * chunk * heads * dim + 2.0 * batch * seq * heads * dim * dstate
    result = comparison(
        "mamba_chunk_scan", best["latency_ms"], total_flops=total_flops,
        workload={"mamba_scan_batch": batch, "mamba_scan_heads": heads,
                  "mamba_scan_groups": groups, "mamba_scan_seq": seq,
                  "mamba_scan_chunk": chunk, "mamba_scan_dim": dim,
                  "mamba_scan_dstate": dstate},
    )
    result.update(
        shape={"batch": batch, "seq": seq, "heads": heads, "chunk": chunk, "dim": dim, "dstate": dstate},
        variants=variants,
        autotuned_variants=autotuned_variants,
        selected_variant=best,
    )
    return result
