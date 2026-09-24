"""FP4 dequantization plus GEMM from the Hopper example."""

from itertools import product

import tilelang

from .example_loader import NativeKernelReference, load_example_factory, mark_for_overlap
from .workloads import OperatorSpec, SearchWorkload, threads_from_tile_extent


def add_arguments(parser):
    group = parser.add_argument_group("FP4 dequant GEMM")
    group.add_argument("--dequant-fp4-block-m", type=int, nargs="+", default=[64, 128])
    group.add_argument("--dequant-fp4-block-n", type=int, nargs="+", default=[64, 128])
    group.add_argument("--dequant-fp4-block-k", type=int, nargs="+", default=[128, 256])
    group.add_argument("--dequant-fp4-m", type=int, default=4096)
    group.add_argument("--dequant-fp4-n", type=int, default=4096)
    group.add_argument("--dequant-fp4-k", type=int, default=4096)


def configurations(options):
    return [
        {"block_m": m, "block_n": n, "block_k": k}
        for m, n, k in product(
            options["dequant_fp4_block_m"],
            options["dequant_fp4_block_n"],
            options["dequant_fp4_block_k"],
        )
    ]


def build(options, config):
    factory = load_example_factory(
        "examples/dequantize_gemm/example_dequant_gemm_fp4_hopper.py",
        "matmul",
        ("_tir_u8_to_f4_to_f16",),
    )
    m, n, k = (
        options["dequant_fp4_m"],
        options["dequant_fp4_n"],
        options["dequant_fp4_k"],
    )
    native = factory(
        m,
        n,
        k,
        "float16",
        "float16",
        "float32",
        block_M=config["block_m"],
        block_N=config["block_n"],
        block_K=config["block_k"],
        num_stages=2,
        # This kernel computes Ct = B_dequant @ A^T, so its WGMMA M axis is
        # block_n even though the public output is [M, N].
        threads=threads_from_tile_extent(config["block_n"]),
        split=1,
    )
    passes = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    return SearchWorkload(
        prim_func=mark_for_overlap(native),
        out_idx=(2,),
        total_flops=2.0 * m * n * k,
        reference_program=NativeKernelReference(native, (2,), passes),
        pass_configs=passes,
    )


OPERATOR = OperatorSpec(
    name="dequant_gemm_fp4",
    description="Hopper FP4 dequantization fused with GEMM",
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
