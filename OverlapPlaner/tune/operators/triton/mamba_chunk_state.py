from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from .common import benchmark_autotuned_variant, benchmark_cuda, comparison, controlled_stage_sweep, scratch_allocator, triton_config
except ImportError:  # Direct script execution avoids importing TileLang.
    from common import benchmark_autotuned_variant, benchmark_cuda, comparison, controlled_stage_sweep, scratch_allocator, triton_config


CONTROL_CONFIG = {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4}


CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=warps,
        num_stages=stages,
    )
    for bn in (32, 64, 128)
    for bk in (32, 64)
    for warps, stages in ((4, 2), (4, 3), (8, 3))
]


def _prune(configs, named_args, **kwargs):
    del kwargs
    return [c for c in configs if c.num_warps == 4] if named_args["WARP_SPECIALIZE"] else configs


@triton.autotune(configs=CONFIGS, key=["CHUNK_SIZE", "DIM", "DSTATE", "WARP_SPECIALIZE"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def chunk_state_kernel(
    b_ptr,
    x_ptr,
    dt_ptr,
    da_ptr,
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
    pid_h = tl.program_id(0)
    pid_tile = tl.program_id(1)
    pid_bc = tl.program_id(2)
    batch = pid_bc // CHUNKS
    chunk = pid_bc % CHUNKS
    num_n = tl.cdiv(DSTATE, BLOCK_N)
    pid_m = pid_tile // num_n
    pid_n = pid_tile % num_n
    group = pid_h // (HEADS // GROUPS)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    da_base = ((batch * HEADS + pid_h) * CHUNKS + chunk) * CHUNK_SIZE
    da_last = tl.load(da_ptr + da_base + CHUNK_SIZE - 1).to(tl.float32)
    seq_base = chunk * CHUNK_SIZE
    x_desc = tl.make_tensor_descriptor(
        x_ptr + batch * SEQ_LEN * HEADS * DIM + pid_h * DIM,
        shape=[SEQ_LEN, DIM], strides=[HEADS * DIM, 1],
        block_shape=[BLOCK_K, BLOCK_M],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr + batch * SEQ_LEN * GROUPS * DSTATE + group * DSTATE,
        shape=[SEQ_LEN, DSTATE], strides=[GROUPS * DSTATE, 1],
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
        out_ptr + ((batch * CHUNKS + chunk) * HEADS + pid_h) * DIM * DSTATE,
        shape=[DIM, DSTATE], strides=[DSTATE, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    for ko in tl.range(0, tl.cdiv(CHUNK_SIZE, BLOCK_K), warp_specialize=WARP_SPECIALIZE):
        da_k = da_desc.load([0, ko * BLOCK_K]).reshape(BLOCK_K).to(tl.float32)
        dt = dt_desc.load([0, ko * BLOCK_K]).reshape(BLOCK_K).to(tl.float32)
        scale = tl.exp(da_last - da_k)
        scale *= dt

        x = x_desc.load([seq_base + ko * BLOCK_K, pid_m * BLOCK_M])
        x = tl.trans(x * scale[:, None].to(tl.float16))
        b = b_desc.load([seq_base + ko * BLOCK_K, pid_n * BLOCK_N])
        acc = tl.dot(x, b, acc)

    out_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(tl.float16))


@triton.autotune(configs=CONFIGS, key=["CHUNK_SIZE", "DIM", "DSTATE"])
@triton.jit
def chunk_state_load_kernel(
    b_ptr,
    x_ptr,
    dt_ptr,
    da_ptr,
    out_ptr,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    CHUNKS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DIM: tl.constexpr,
    DSTATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Ordinary-load baseline retained for the framework-best comparison."""

    pid_h = tl.program_id(0)
    pid_tile = tl.program_id(1)
    pid_bc = tl.program_id(2)
    batch = pid_bc // CHUNKS
    chunk = pid_bc % CHUNKS
    num_n = tl.cdiv(DSTATE, BLOCK_N)
    pid_m = pid_tile // num_n
    pid_n = pid_tile % num_n
    group = pid_h // (HEADS // GROUPS)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    da_base = ((batch * HEADS + pid_h) * CHUNKS + chunk) * CHUNK_SIZE
    da_last = tl.load(da_ptr + da_base + CHUNK_SIZE - 1).to(tl.float32)
    seq_base = chunk * CHUNK_SIZE
    for ko in tl.range(0, tl.cdiv(CHUNK_SIZE, BLOCK_K)):
        k = ko * BLOCK_K + offs_k
        da_k = tl.load(da_ptr + da_base + k).to(tl.float32)
        dt = tl.load(dt_ptr + da_base + k).to(tl.float32)
        scale = tl.exp(da_last - da_k) * dt

        seq_k = seq_base + k
        x_ptrs = (
            x_ptr
            + ((batch * SEQ_LEN + seq_k[:, None]) * HEADS + pid_h) * DIM
            + offs_m[None, :]
        )
        x = tl.load(
            x_ptrs,
            mask=(seq_k[:, None] < SEQ_LEN) & (offs_m[None, :] < DIM),
            other=0.0,
        )
        x = tl.trans(x * scale[:, None].to(tl.float16))

        b_ptrs = (
            b_ptr
            + ((batch * SEQ_LEN + seq_k[:, None]) * GROUPS + group) * DSTATE
            + offs_n[None, :]
        )
        b = tl.load(
            b_ptrs,
            mask=(seq_k[:, None] < SEQ_LEN) & (offs_n[None, :] < DSTATE),
            other=0.0,
        )
        acc = tl.dot(x, b, acc)

    out_ptrs = (
        out_ptr
        + ((batch * CHUNKS + chunk) * HEADS + pid_h) * DIM * DSTATE
        + offs_m[:, None] * DSTATE
        + offs_n[None, :]
    )
    tl.store(
        out_ptrs,
        acc.to(tl.float16),
        mask=(offs_m[:, None] < DIM) & (offs_n[None, :] < DSTATE),
    )


def launch(
    b: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    da: torch.Tensor,
    out: torch.Tensor,
    *,
    warp_specialize: bool = False,
) -> None:
    triton.set_allocator(scratch_allocator)
    batch, seq_len, heads, dim = x.shape
    groups = b.shape[2]
    dstate = b.shape[-1]
    chunks, chunk_size = dt.shape[2:]
    grid = lambda meta: (
        heads,
        triton.cdiv(dim, meta["BLOCK_M"]) * triton.cdiv(dstate, meta["BLOCK_N"]),
        batch * chunks,
    )
    chunk_state_kernel[grid](
        b,
        x,
        dt,
        da,
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


def launch_load_baseline(
    b: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    da: torch.Tensor,
    out: torch.Tensor,
) -> None:
    batch, seq_len, heads, dim = x.shape
    groups = b.shape[2]
    dstate = b.shape[-1]
    chunks, chunk_size = dt.shape[2:]
    grid = lambda meta: (
        heads,
        triton.cdiv(dim, meta["BLOCK_M"])
        * triton.cdiv(dstate, meta["BLOCK_N"]),
        batch * chunks,
    )
    chunk_state_load_kernel[grid](
        b,
        x,
        dt,
        da,
        out,
        seq_len,
        heads,
        groups,
        chunks,
        chunk_size,
        dim,
        dstate,
    )


def launch_fixed(
    b: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    da: torch.Tensor,
    out: torch.Tensor,
    *,
    num_stages: int,
    warp_specialize: bool = False,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
) -> None:
    triton.set_allocator(scratch_allocator)
    batch, seq_len, heads, dim = x.shape
    groups = b.shape[2]
    dstate = b.shape[-1]
    chunks, chunk_size = dt.shape[2:]
    block_m = block_m or CONTROL_CONFIG["BLOCK_M"]
    block_n = block_n or CONTROL_CONFIG["BLOCK_N"]
    block_k = block_k or CONTROL_CONFIG["BLOCK_K"]
    num_warps = num_warps or CONTROL_CONFIG["num_warps"]
    grid = (
        heads,
        triton.cdiv(dim, block_m) * triton.cdiv(dstate, block_n),
        batch * chunks,
    )
    chunk_state_kernel.fn[grid](
        b,
        x,
        dt,
        da,
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


def reference(b: torch.Tensor, x: torch.Tensor, dt: torch.Tensor, da: torch.Tensor) -> torch.Tensor:
    batch, seq_len, heads, dim = x.shape
    groups = b.shape[2]
    dstate = b.shape[-1]
    chunks, chunk = dt.shape[2:]
    b = b.repeat_interleave(heads // groups, dim=2).reshape(batch, chunks, chunk, heads, dstate)
    x = x.reshape(batch, chunks, chunk, heads, dim)
    decay = torch.exp(da[..., -1:] - da)
    return torch.einsum(
        "bckhn,bhck,bhck,bckhm->bchmn",
        b.float(),
        decay.float(),
        dt.float(),
        x.float(),
    ).half()


def run(*, warmup: int = 100, rep: int = 400, trials: int = 5) -> dict:
    batch, seq_len, heads, groups = 2, 8192, 80, 1
    chunk_size, dim, dstate = 256, 64, 128
    chunks = seq_len // chunk_size
    torch.manual_seed(1)
    b = torch.randn((batch, seq_len, groups, dstate), device="cuda", dtype=torch.float16) * 0.1
    x = torch.randn((batch, seq_len, heads, dim), device="cuda", dtype=torch.float16) * 0.1
    dt = torch.rand((batch, heads, chunks, chunk_size), device="cuda", dtype=torch.float16)
    da = torch.cumsum(-torch.rand_like(dt).float() * 0.01, dim=-1).half()
    # Validate layout and head/group mapping on one chunk.  The full PyTorch
    # einsum creates a large temporary in addition to the 1.6 GiB output and
    # can OOM on an otherwise runnable, shared H100.
    sb = b[:1, :chunk_size].contiguous()
    sx = x[:1, :chunk_size].contiguous()
    sdt = dt[:1, :, :1].contiguous()
    sda = da[:1, :, :1].contiguous()
    sout = torch.empty((1, 1, heads, dim, dstate), device="cuda", dtype=torch.float16)
    ref = reference(sb, sx, sdt, sda)
    launch_fixed(sb, sx, sdt, sda, sout, num_stages=2)
    torch.testing.assert_close(sout, ref, atol=3e-2, rtol=2e-2)
    launch_fixed(sb, sx, sdt, sda, sout, num_stages=2, warp_specialize=True)
    torch.testing.assert_close(sout, ref, atol=3e-2, rtol=2e-2)
    launch_load_baseline(sb, sx, sdt, sda, sout)
    torch.testing.assert_close(sout, ref, atol=3e-2, rtol=2e-2)
    del ref, sb, sx, sdt, sda, sout

    out = torch.empty((batch, chunks, heads, dim, dstate), device="cuda", dtype=torch.float16)
    autotuned_variants = [
        benchmark_autotuned_variant(
            lambda ws=ws: launch(b, x, dt, da, out, warp_specialize=ws),
            chunk_state_kernel,
            warp_specialize=ws,
            warmup=warmup,
            rep=rep,
            trials=trials,
        )
        for ws in (False,)
    ]
    autotuned_variants.append(
        benchmark_autotuned_variant(
            lambda: launch_load_baseline(b, x, dt, da, out),
            chunk_state_load_kernel,
            warp_specialize=False,
            warmup=warmup,
            rep=rep,
            trials=trials,
            descriptor_loads=False,
        )
    )
    autotuned_variants[-1]["name"] = "autotuned_ordinary_load_no_ws"
    variants = controlled_stage_sweep(
        lambda num_stages, ws: launch_fixed(
            b, x, dt, da, out, num_stages=num_stages, warp_specialize=ws
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
    total_flops = 2.0 * batch * seq_len * heads * dim * dstate
    result = comparison(
        "mamba_chunk_state", best["latency_ms"], total_flops=total_flops,
        workload={"mamba_state_batch": batch, "mamba_state_heads": heads,
                  "mamba_state_groups": groups, "mamba_state_seq": seq_len,
                  "mamba_state_chunk": chunk_size, "mamba_state_dim": dim,
                  "mamba_state_dstate": dstate},
    )
    result.update(
        shape={
            "batch": batch,
            "seq_len": seq_len,
            "heads": heads,
            "groups": groups,
            "chunk_size": chunk_size,
            "dim": dim,
            "dstate": dstate,
        },
        variants=variants,
        autotuned_variants=autotuned_variants,
        selected_variant=best,
    )
    return result
