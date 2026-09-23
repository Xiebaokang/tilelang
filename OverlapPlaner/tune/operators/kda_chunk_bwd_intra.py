"""KDA intra-chunk backward kernel."""

import tilelang

from .example_loader import NativeKernelReference, load_example_factory, mark_for_overlap
from .workloads import OperatorSpec, SearchWorkload


def add_arguments(parser):
    group = parser.add_argument_group("KDA intra backward")
    group.add_argument("--kda-intra-block-dk", type=int, nargs="+", default=[32, 64, 128])
    group.add_argument("--kda-intra-batch", type=int, default=1)
    group.add_argument("--kda-intra-seq", type=int, default=8192)
    group.add_argument("--kda-intra-heads", type=int, default=8)
    group.add_argument("--kda-intra-dk", type=int, default=128)
    group.add_argument("--kda-intra-chunk", type=int, default=64)


def configurations(options):
    return [{"block_dk": value} for value in options["kda_intra_block_dk"]]


def build(options, config):
    factory = load_example_factory(
        "examples/kda/chunk_bwd_intra.py", "tilelang_chunk_bwd_intra"
    )
    b, s, h = options["kda_intra_batch"], options["kda_intra_seq"], options["kda_intra_heads"]
    dk, chunk = options["kda_intra_dk"], options["kda_intra_chunk"]
    native = factory(
        b, s, h, dk, "float16", "float32", "float32", "float32", "float32", chunk,
        config["block_dk"], block_BC=16, threads=128, num_stages=0,
    )
    passes = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    outputs = (10, 11, 12, 13)
    return SearchWorkload(
        prim_func=mark_for_overlap(native), out_idx=outputs,
        total_flops=8.0 * b * h * s * chunk * dk,
        reference_program=NativeKernelReference(native, outputs, passes), pass_configs=passes,
    )


OPERATOR = OperatorSpec(
    name="kda_chunk_bwd_intra", description="KDA intra-chunk backward",
    add_cli_arguments=add_arguments, configuration_factory=configurations, workload_factory=build,
)
