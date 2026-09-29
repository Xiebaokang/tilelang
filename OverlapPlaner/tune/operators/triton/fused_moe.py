from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from .common import benchmark_autotuned_variant, benchmark_cuda, comparison, controlled_stage_sweep, scratch_allocator, triton_config
except ImportError:
    from common import benchmark_autotuned_variant, benchmark_cuda, comparison, controlled_stage_sweep, scratch_allocator, triton_config


CONTROL_CONFIG = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4}


def _configs() -> list[triton.Config]:
    configs = []
    for bm in (64, 128):
        for bn in (64, 128):
            for bk in (32, 64, 128):
                for warps in (4, 8):
                    for stages in (2, 3, 4):
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                                num_warps=warps,
                                num_stages=stages,
                            )
                        )
    return configs


def _prune(configs, named_args, **kwargs):
    del kwargs
    if named_args["WARP_SPECIALIZE"]:
        return [config for config in configs if config.num_warps == 4]
    return configs


@triton.autotune(
    configs=_configs(),
    key=["TOKENS", "HIDDEN", "EXPERT", "WARP_SPECIALIZE"],
    prune_configs_by={"early_config_prune": _prune},
)
@triton.jit
def fused_moe_kernel(
    x_ptr,
    gate_ptr,
    up_ptr,
    out_ptr,
    TOKENS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EXPERT: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[TOKENS, HIDDEN], strides=[HIDDEN, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    gate_desc = tl.make_tensor_descriptor(
        gate_ptr, shape=[EXPERT, HIDDEN], strides=[HIDDEN, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    up_desc = tl.make_tensor_descriptor(
        up_ptr, shape=[EXPERT, HIDDEN], strides=[HIDDEN, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[TOKENS, EXPERT], strides=[EXPERT, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for ko in tl.range(0, tl.cdiv(HIDDEN, BLOCK_K), warp_specialize=WARP_SPECIALIZE):
        x = x_desc.load([pid_m * BLOCK_M, ko * BLOCK_K])
        gate = gate_desc.load([pid_n * BLOCK_N, ko * BLOCK_K]).T
        up = up_desc.load([pid_n * BLOCK_N, ko * BLOCK_K]).T
        gate_acc = tl.dot(x, gate, gate_acc)
        up_acc = tl.dot(x, up, up_acc)
    activated = gate_acc * tl.sigmoid(gate_acc)
    out = activated * up_acc
    out_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], out.to(tl.float16))


def launch(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    out: torch.Tensor,
    *,
    warp_specialize: bool,
) -> None:
    triton.set_allocator(scratch_allocator)
    tokens, hidden = x.shape
    expert = gate.shape[0]
    grid = lambda meta: (triton.cdiv(tokens, meta["BLOCK_M"]), triton.cdiv(expert, meta["BLOCK_N"]))
    fused_moe_kernel[grid](x, gate, up, out, tokens, hidden, expert, warp_specialize)


def launch_fixed(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
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
    tokens, hidden = x.shape
    expert = gate.shape[0]
    bm = block_m or CONTROL_CONFIG["BLOCK_M"]
    bn = block_n or CONTROL_CONFIG["BLOCK_N"]
    bk = block_k or CONTROL_CONFIG["BLOCK_K"]
    num_warps = num_warps or CONTROL_CONFIG["num_warps"]
    grid = (triton.cdiv(tokens, bm), triton.cdiv(expert, bn))
    fused_moe_kernel.fn[grid](
        x,
        gate,
        up,
        out,
        tokens,
        hidden,
        expert,
        warp_specialize,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run(*, warmup: int = 100, rep: int = 400, trials: int = 5) -> dict:
    tokens, hidden, expert = 8192, 7168, 2048
    torch.manual_seed(4)
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.float16) * 0.1
    gate = torch.randn((expert, hidden), device="cuda", dtype=torch.float16) * 0.1
    up = torch.randn_like(gate)
    out = torch.empty((tokens, expert), device="cuda", dtype=torch.float16)

    launch_fixed(x, gate, up, out, num_stages=2, warp_specialize=False)
    ref = torch.nn.functional.silu(x.float() @ gate.float().T) * (x.float() @ up.float().T)
    torch.testing.assert_close(out, ref.half(), atol=1e-1, rtol=2e-2)
    launch_fixed(x, gate, up, out, num_stages=2, warp_specialize=True)
    torch.testing.assert_close(out, ref.half(), atol=1e-1, rtol=2e-2)
    del ref

    autotuned_variants = [
        benchmark_autotuned_variant(
            lambda ws=ws: launch(x, gate, up, out, warp_specialize=ws),
            fused_moe_kernel,
            warp_specialize=ws,
            warmup=warmup,
            rep=rep,
            trials=trials,
        )
        for ws in (False,)
    ]

    variants = controlled_stage_sweep(
        lambda num_stages, ws: launch_fixed(
            x, gate, up, out, num_stages=num_stages, warp_specialize=ws
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
    result = comparison(
        "fused_moe",
        best["latency_ms"],
        total_flops=4.0 * tokens * hidden * expert,
        workload={"fused_moe_tokens": tokens, "fused_moe_hidden": hidden,
                  "fused_moe_expert": expert},
    )
    result.update(
        shape={"tokens": tokens, "hidden": hidden, "expert": expert},
        variants=variants,
        autotuned_variants=autotuned_variants,
        selected_variant=best,
    )
    return result
