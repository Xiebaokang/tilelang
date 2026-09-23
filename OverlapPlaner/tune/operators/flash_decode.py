"""Split-KV MHA decoding example."""

import tilelang

from .example_loader import NativeKernelReference, load_example_factory, mark_for_overlap
from .workloads import OperatorSpec, SearchWorkload


def add_arguments(parser):
    group = parser.add_argument_group("Flash decoding")
    group.add_argument("--flash-decode-block-m", type=int, nargs="+", default=[64, 128])
    group.add_argument("--flash-decode-block-n", type=int, nargs="+", default=[64, 128])
    group.add_argument("--flash-decode-batch", type=int, default=1)
    group.add_argument("--flash-decode-heads", type=int, default=32)
    group.add_argument("--flash-decode-q-seq", type=int, default=128)
    group.add_argument("--flash-decode-kv-seq", type=int, default=8192)
    group.add_argument("--flash-decode-dim", type=int, default=128)


def configurations(options):
    return [
        {"block_m": m, "block_n": n}
        for m in options["flash_decode_block_m"]
        for n in options["flash_decode_block_n"]
        if not (m == 128 and n == 128)
    ]


def build(options, config):
    factory = load_example_factory(
        "examples/flash_decoding/example_mha_inference.py", "flashattn"
    )
    factory.__globals__["num_split"] = 4
    b, h = options["flash_decode_batch"], options["flash_decode_heads"]
    sq, sk, d = options["flash_decode_q_seq"], options["flash_decode_kv_seq"], options["flash_decode_dim"]
    native = factory(b, h, sq, sk, d, False, config["block_m"], config["block_n"])
    passes = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    return SearchWorkload(
        prim_func=mark_for_overlap(native), out_idx=(3,),
        total_flops=4.0 * b * h * sq * sk * d,
        reference_program=NativeKernelReference(native, (3,), passes), pass_configs=passes,
    )


OPERATOR = OperatorSpec(
    name="flash_decode", description="Split-KV multi-head flash decoding",
    add_cli_arguments=add_arguments, configuration_factory=configurations, workload_factory=build,
)
