"""Gated-delta-network chunk output backward kernel."""

from itertools import product

import tilelang

from .example_loader import NativeKernelReference, load_example_factory, mark_for_overlap
from .workloads import OperatorSpec, SearchWorkload


def add_arguments(parser):
    group = parser.add_argument_group("GDN chunk output backward")
    group.add_argument("--gdn-o-bwd-block-dk", type=int, nargs="+", default=[32, 64])
    group.add_argument("--gdn-o-bwd-block-dv", type=int, nargs="+", default=[32, 64, 128])
    group.add_argument("--gdn-o-bwd-batch", type=int, default=1)
    group.add_argument("--gdn-o-bwd-seq", type=int, default=8192)
    group.add_argument("--gdn-o-bwd-heads", type=int, default=8)
    group.add_argument("--gdn-o-bwd-dk", type=int, default=128)
    group.add_argument("--gdn-o-bwd-dv", type=int, default=128)
    group.add_argument("--gdn-o-bwd-chunk", type=int, default=64)


def configurations(options):
    return [
        {"block_dk": dk, "block_dv": dv}
        for dk, dv in product(
            options["gdn_o_bwd_block_dk"], options["gdn_o_bwd_block_dv"]
        )
    ]


def build(options, config):
    factory = load_example_factory(
        "examples/gdn/example_chunk_o_bwd.py",
        "tilelang_chunk_o_bwd_dqkwg",
    )
    b, s, h = options["gdn_o_bwd_batch"], options["gdn_o_bwd_seq"], options["gdn_o_bwd_heads"]
    dk, dv, chunk = options["gdn_o_bwd_dk"], options["gdn_o_bwd_dv"], options["gdn_o_bwd_chunk"]
    native = factory(
        b, s, h, dk, dv, "float16", "float32", "float32", "float32", "float32", chunk, dk**-0.5,
        block_DK=config["block_dk"], block_DV=config["block_dv"], threads=256, num_stages=0,
    )
    passes = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    outputs = (9, 10, 11, 12)
    return SearchWorkload(
        prim_func=mark_for_overlap(native), out_idx=outputs,
        total_flops=8.0 * b * h * s * chunk * (dk + dv),
        reference_program=NativeKernelReference(native, outputs, passes), pass_configs=passes,
    )


OPERATOR = OperatorSpec(
    name="gdn_chunk_o_bwd", description="GDN chunk output backward",
    add_cli_arguments=add_arguments, configuration_factory=configurations, workload_factory=build,
)
