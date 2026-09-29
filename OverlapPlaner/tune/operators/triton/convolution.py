"""NHWC 3x3 convolution baseline matching the OverlapPlaner workload."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from .common import benchmark_autotuned_variant, comparison, controlled_stage_sweep
except ImportError:
    from common import benchmark_autotuned_variant, comparison, controlled_stage_sweep


CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                  num_warps=nw, num_stages=ns)
    for bm in (64, 128, 256)
    for bn in (64, 128, 256)
    for bk in (32, 64)
    for nw in (4, 8)
    for ns in (2, 3, 4)
    if not (bm == 256 and bn == 256)
]
CONTROL = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4}


def _prune(configs, named_args, **kwargs):
    del kwargs
    return [c for c in configs if c.num_warps == 4] if named_args["WARP_SPECIALIZE"] else configs


@triton.autotune(configs=CONFIGS, key=["HEIGHT", "WIDTH", "CHANNELS", "FILTERS", "WARP_SPECIALIZE"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def conv_kernel(data, weight, output, BATCH: tl.constexpr, HEIGHT: tl.constexpr,
                WIDTH: tl.constexpr, CHANNELS: tl.constexpr,
                FILTERS: tl.constexpr, WARP_SPECIALIZE: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    total_m: tl.constexpr = BATCH * HEIGHT * WIDTH
    reduction: tl.constexpr = 9 * CHANNELS
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    batch = rows // (HEIGHT * WIDTH)
    pixel = rows % (HEIGHT * WIDTH)
    oh = pixel // WIDTH
    ow = pixel % WIDTH
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start_k in tl.range(0, reduction, BLOCK_K, warp_specialize=WARP_SPECIALIZE):
        red = start_k + tl.arange(0, BLOCK_K)
        channel = red % CHANNELS
        kernel_pos = red // CHANNELS
        kh = kernel_pos // 3
        kw = kernel_pos % 3
        ih = oh[:, None] + kh[None, :] - 1
        iw = ow[:, None] + kw[None, :] - 1
        valid = ((rows[:, None] < total_m) & (red[None, :] < reduction)
                 & (ih >= 0) & (ih < HEIGHT) & (iw >= 0) & (iw < WIDTH))
        data_off = ((batch[:, None] * HEIGHT + ih) * WIDTH + iw) * CHANNELS + channel[None, :]
        a = tl.load(data + data_off, mask=valid, other=0.0)
        weight_off = red[:, None] * FILTERS + cols[None, :]
        b = tl.load(weight + weight_off,
                    mask=(red[:, None] < reduction) & (cols[None, :] < FILTERS), other=0.0)
        acc = tl.dot(a, b, acc)
    out_off = rows[:, None] * FILTERS + cols[None, :]
    tl.store(output + out_off, acc.to(tl.float16),
             mask=(rows[:, None] < total_m) & (cols[None, :] < FILTERS))


def launch(data, weight, output, *, warp_specialize=False):
    batch, height, width, channels = data.shape
    filters = weight.shape[-1]
    grid = lambda meta: (triton.cdiv(batch * height * width, meta["BLOCK_M"]),
                         triton.cdiv(filters, meta["BLOCK_N"]))
    conv_kernel[grid](data, weight, output, batch, height, width, channels,
                      filters, warp_specialize)


def launch_fixed(data, weight, output, *, num_stages, warp_specialize=False):
    batch, height, width, channels = data.shape
    filters = weight.shape[-1]
    grid = (triton.cdiv(batch * height * width, CONTROL["BLOCK_M"]),
            triton.cdiv(filters, CONTROL["BLOCK_N"]))
    conv_kernel.fn[grid](data, weight, output, batch, height, width, channels,
                         filters, warp_specialize, BLOCK_M=CONTROL["BLOCK_M"],
                         BLOCK_N=CONTROL["BLOCK_N"], BLOCK_K=CONTROL["BLOCK_K"],
                         num_warps=CONTROL["num_warps"], num_stages=num_stages)


def run(*, warmup=100, rep=400, trials=5):
    batch, height, width, channels, filters = 16, 56, 56, 64, 128
    torch.manual_seed(12)
    data = torch.randn((batch, height, width, channels), device="cuda", dtype=torch.float16)
    weight = torch.randn((3, 3, channels, filters), device="cuda", dtype=torch.float16)
    output = torch.empty((batch, height, width, filters), device="cuda", dtype=torch.float16)
    launch_fixed(data, weight, output, num_stages=2)
    ref = torch.nn.functional.conv2d(data.permute(0, 3, 1, 2),
                                     weight.permute(3, 2, 0, 1), padding=1)
    torch.testing.assert_close(output, ref.permute(0, 2, 3, 1), atol=.2, rtol=.02)
    broad = [benchmark_autotuned_variant(
        lambda ws=ws: launch(data, weight, output, warp_specialize=ws), conv_kernel,
        warp_specialize=ws, warmup=warmup, rep=rep, trials=trials)
        for ws in (False, True)]
    controlled = controlled_stage_sweep(
        lambda ns, ws: launch_fixed(data, weight, output, num_stages=ns,
                                    warp_specialize=ws), CONTROL,
        warmup=warmup, rep=rep, trials=trials)
    best = min((x for x in broad + controlled if "latency_ms" in x),
               key=lambda x: x["latency_ms"])
    flops = 2.0 * batch * height * width * filters * channels * 9
    result = comparison(
        "convolution", best["latency_ms"], total_flops=flops,
        workload={"conv_batch": batch, "conv_channels": channels,
                  "conv_height": height, "conv_width": width,
                  "conv_filters": filters, "conv_kernel": 3,
                  "conv_stride": 1, "conv_dilation": 1, "conv_padding": 1},
    )
    result.update(shape={"batch": batch, "height": height, "width": width,
                         "channels": channels, "filters": filters,
                         "kernel": 3, "padding": 1},
                  autotuned_variants=broad, variants=controlled,
                  selected_variant=best)
    return result
