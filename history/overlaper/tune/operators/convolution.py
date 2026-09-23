"""Overlaper search kernel adapted from examples/convolution/example_convolution.py."""

from itertools import product

import torch
import tilelang.language as T

from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig


def add_arguments(parser) -> None:
    group = parser.add_argument_group("convolution")
    group.add_argument("--conv-block-m", type=int, nargs="+", default=[64, 128])
    group.add_argument("--conv-block-n", type=int, nargs="+", default=[64, 128])
    group.add_argument("--conv-block-k", type=int, nargs="+", default=[32, 64])
    group.add_argument("--conv-batch", type=int, default=8)
    group.add_argument("--conv-channels", type=int, default=64)
    group.add_argument("--conv-height", type=int, default=32)
    group.add_argument("--conv-width", type=int, default=32)
    group.add_argument("--conv-filters", type=int, default=128)
    group.add_argument("--conv-kernel", type=int, default=3)
    group.add_argument("--conv-stride", type=int, default=1)
    group.add_argument("--conv-dilation", type=int, default=1)
    group.add_argument("--conv-padding", type=int, default=1)


def configurations(options: Options) -> list[TileConfig]:
    return [
        {"block_m": block_m, "block_n": block_n, "block_k": block_k}
        for block_m, block_n, block_k in product(
            options["conv_block_m"],
            options["conv_block_n"],
            options["conv_block_k"],
        )
    ]


def build(options: Options, config: TileConfig) -> SearchWorkload:
    batch = options["conv_batch"]
    channels = options["conv_channels"]
    height = options["conv_height"]
    width = options["conv_width"]
    filters = options["conv_filters"]
    kernel_size = options["conv_kernel"]
    stride = options["conv_stride"]
    dilation = options["conv_dilation"]
    padding = options["conv_padding"]
    block_m = config["block_m"]
    block_n = config["block_n"]
    block_k = config["block_k"]
    out_h = (
        height + 2 * padding - dilation * (kernel_size - 1) - 1
    ) // stride + 1
    out_w = (
        width + 2 * padding - dilation * (kernel_size - 1) - 1
    ) // stride + 1
    reduction = kernel_size * kernel_size * channels
    dtype = T.float16

    @T.prim_func
    def main(
        Data: T.Tensor((batch, height, width, channels), dtype),
        Kernel: T.Tensor(
            (kernel_size, kernel_size, channels, filters), dtype
        ),
        Output: T.Tensor((batch, out_h, out_w, filters), dtype),
    ):
        with T.Kernel(
            T.ceildiv(filters, block_n),
            T.ceildiv(batch * out_h * out_w, block_m),
            threads=block_m // 64 * 128,
        ) as (bx, by):
            data_shared = T.alloc_shared((block_m, block_k), dtype)
            kernel_shared = T.alloc_shared((block_k, block_n), dtype)
            output_local = T.alloc_fragment((block_m, block_n), T.float32)
            kernel_flat = T.Tensor((reduction, filters), dtype, Kernel.data)
            output_flat = T.Tensor(
                (batch * out_h * out_w, filters), dtype, Output.data
            )
            T.clear(output_local)
            for ko in T.Pipelined(T.ceildiv(reduction, block_k), num_stages=2, auto_wsp=True):
                T.im2col(
                    Data,
                    data_shared,
                    by,
                    ko,
                    kernel_size,
                    stride,
                    dilation,
                    padding,
                )
                T.copy(kernel_flat[ko * block_k, bx * block_n], kernel_shared)
                T.gemm(data_shared, kernel_shared, output_local)
            T.copy(
                output_local,
                output_flat[by * block_m, bx * block_n],
            )

    def reference(data, kernel):
        data = data.permute(0, 3, 1, 2)
        kernel = kernel.permute(3, 2, 0, 1)
        output = torch.conv2d(
            data,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        return output.permute(0, 2, 3, 1)

    total_flops = (
        2.0 * batch * out_h * out_w * filters * channels * kernel_size**2
    )
    return SearchWorkload(
        prim_func=main,
        out_idx=(2,),
        total_flops=total_flops,
        reference_program=reference,
    )


OPERATOR = OperatorSpec(
    name="convolution",
    description=(
        "NHWC FP16 convolution adapted from "
        "examples/convolution/example_convolution.py"
    ),
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
