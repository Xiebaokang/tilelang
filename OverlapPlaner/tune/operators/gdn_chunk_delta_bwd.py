"""Gated-delta-network reverse state scan."""

import tilelang

from .example_loader import NativeKernelReference, load_example_factory, mark_for_overlap
from .workloads import OperatorSpec, SearchWorkload, threads_from_tile_extent


def add_arguments(parser):
    group = parser.add_argument_group("GDN delta backward")
    group.add_argument("--gdn-delta-bwd-block-dv", type=int, nargs="+", default=[32, 64, 128])
    group.add_argument("--gdn-delta-bwd-batch", type=int, default=1)
    group.add_argument("--gdn-delta-bwd-seq", type=int, default=8192)
    group.add_argument("--gdn-delta-bwd-heads", type=int, default=8)
    group.add_argument("--gdn-delta-bwd-dk", type=int, default=128)
    group.add_argument("--gdn-delta-bwd-dv", type=int, default=128)
    group.add_argument("--gdn-delta-bwd-chunk", type=int, default=64)


def configurations(options):
    return [{"block_dv": value} for value in options["gdn_delta_bwd_block_dv"]]


def build(options, config):
    factory = load_example_factory(
        "examples/gdn/example_chunk_delta_bwd.py",
        "tilelang_chunk_gated_delta_rule_bwd_dhu",
    )
    b, s, h = options["gdn_delta_bwd_batch"], options["gdn_delta_bwd_seq"], options["gdn_delta_bwd_heads"]
    dk, dv, chunk = options["gdn_delta_bwd_dk"], options["gdn_delta_bwd_dv"], options["gdn_delta_bwd_chunk"]
    native = factory(
        b, s, h, dk, dv, "float16", "float16", "float32", "float32", "float32", chunk, dk**-0.5,
        block_DV=config["block_dv"], threads=threads_from_tile_extent(config["block_dv"]), num_stages=0,
    )
    passes = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    outputs = (8, 9, 10)
    return SearchWorkload(
        prim_func=mark_for_overlap(native), out_idx=outputs,
        total_flops=6.0 * b * h * s * dk * dv,
        reference_program=NativeKernelReference(native, outputs, passes), pass_configs=passes,
    )


OPERATOR = OperatorSpec(
    name="gdn_chunk_delta_bwd", description="GDN reverse chunk-state backward scan",
    add_cli_arguments=add_arguments, configuration_factory=configurations, workload_factory=build,
)
