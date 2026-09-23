"""Query-parallel block-causal attention backward kernel."""

import tilelang

from .example_loader import NativeKernelReference, load_example_factory, mark_for_overlap
from .workloads import OperatorSpec, SearchWorkload


def add_arguments(parser):
    group = parser.add_argument_group("Block-causal backward")
    group.add_argument("--block-causal-bwd-block", type=int, nargs="+", default=[64])
    group.add_argument("--block-causal-bwd-batch", type=int, default=1)
    group.add_argument("--block-causal-bwd-heads", type=int, default=16)
    group.add_argument("--block-causal-bwd-seq", type=int, default=8192)
    group.add_argument("--block-causal-bwd-dim", type=int, default=128)
    group.add_argument("--block-causal-bwd-dllm-block", type=int, default=64)


def configurations(options):
    return [{"block": value} for value in options["block_causal_bwd_block"]]


def build(options, config):
    factory = load_example_factory(
        "examples/block_causal_attention/block_causal_attention.py",
        "_bwd_dq_template",
        ("_check_dllm_block", "_fwd_tile_allowed"),
    )
    # The helper uses this module-level constant in the example.
    factory.__globals__["LOG2_E"] = 1.4426950408889634
    factory.__globals__["_SUPPORTED_DLLM_BLOCKS"] = (1, 2, 4, 8, 16, 32, 64)
    b, h, s, d = (
        options["block_causal_bwd_batch"], options["block_causal_bwd_heads"],
        options["block_causal_bwd_seq"], options["block_causal_bwd_dim"],
    )
    native = factory(
        b, h, s, d, options["block_causal_bwd_dllm_block"], d**-0.5,
        block_size=config["block"], num_stages=3, threads=128, dtype="bfloat16",
    )
    passes = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    return SearchWorkload(
        prim_func=mark_for_overlap(native), out_idx=(6,),
        total_flops=6.0 * b * h * s * s * d * 0.5,
        reference_program=NativeKernelReference(native, (6,), passes), pass_configs=passes,
    )


OPERATOR = OperatorSpec(
    name="block_causal_bwd", description="Query-parallel block-causal attention backward",
    add_cli_arguments=add_arguments, configuration_factory=configurations, workload_factory=build,
)
