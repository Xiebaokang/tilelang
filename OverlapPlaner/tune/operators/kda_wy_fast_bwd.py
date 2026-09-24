"""KDA WY representation backward kernel."""

from itertools import product

import tilelang

from .example_loader import NativeKernelReference, load_example_factory, mark_for_overlap
from .workloads import OperatorSpec, SearchWorkload, threads_from_tile_extent


def add_arguments(parser):
    group = parser.add_argument_group("KDA WY backward")
    group.add_argument("--kda-wy-bwd-block-dk", type=int, nargs="+", default=[32, 64, 128])
    group.add_argument("--kda-wy-bwd-block-dv", type=int, nargs="+", default=[32, 64, 128])
    group.add_argument("--kda-wy-bwd-batch", type=int, default=1)
    group.add_argument("--kda-wy-bwd-seq", type=int, default=8192)
    group.add_argument("--kda-wy-bwd-heads", type=int, default=8)
    group.add_argument("--kda-wy-bwd-dk", type=int, default=128)
    group.add_argument("--kda-wy-bwd-dv", type=int, default=128)
    group.add_argument("--kda-wy-bwd-chunk", type=int, default=64)


def configurations(options):
    return [
        {"block_dk": dk, "block_dv": dv}
        for dk, dv in product(
            options["kda_wy_bwd_block_dk"], options["kda_wy_bwd_block_dv"]
        )
        if dk == 32 or (dk == 64 and dv <= 64)
    ]


def build(options, config):
    factory = load_example_factory(
        "examples/kda/wy_fast_bwd.py", "tilelang_wy_fast_bwd"
    )
    b, s, h = options["kda_wy_bwd_batch"], options["kda_wy_bwd_seq"], options["kda_wy_bwd_heads"]
    dk, dv, chunk = options["kda_wy_bwd_dk"], options["kda_wy_bwd_dv"], options["kda_wy_bwd_chunk"]
    native = factory(
        b, s, h, dk, dv, "float16", "float32", "float32", "float32", "float32", chunk,
        block_DK=config["block_dk"],
        block_DV=config["block_dv"],
        threads=threads_from_tile_extent(
            max(config["block_dk"], config["block_dv"])
        ),
        num_stages=0,
    )
    passes = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    outputs = (9, 10, 11, 12, 13)
    return SearchWorkload(
        prim_func=mark_for_overlap(native), out_idx=outputs,
        total_flops=8.0 * b * h * s * chunk * (dk + dv),
        reference_program=NativeKernelReference(native, outputs, passes), pass_configs=passes,
    )


OPERATOR = OperatorSpec(
    name="kda_wy_fast_bwd", description="KDA WY representation backward",
    add_cli_arguments=add_arguments, configuration_factory=configurations, workload_factory=build,
)
