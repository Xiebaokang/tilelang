"""Fused shared-expert gate/up projection from the MoE example."""

from itertools import product

import tilelang
import tilelang.language as T
import torch.nn.functional as F

from .workloads import OperatorSpec, SearchWorkload


def add_arguments(parser):
    group = parser.add_argument_group("Fused MoE gate/up")
    group.add_argument("--fused-moe-block-token", type=int, nargs="+", default=[64, 128])
    group.add_argument("--fused-moe-block-hidden", type=int, nargs="+", default=[128])
    group.add_argument("--fused-moe-block-expert", type=int, nargs="+", default=[128])
    group.add_argument("--fused-moe-tokens", type=int, default=8192)
    group.add_argument("--fused-moe-hidden", type=int, default=7168)
    group.add_argument("--fused-moe-expert", type=int, default=2048)


def configurations(options):
    return [
        {"block_token": bt, "block_hidden": bh, "block_expert": be}
        for bt, bh, be in product(
            options["fused_moe_block_token"],
            options["fused_moe_block_hidden"],
            options["fused_moe_block_expert"],
        )
    ]


def kernel(tokens, hidden, expert, block_token, block_hidden, block_expert):
    dtype = T.float16

    @T.prim_func(auto_overlap=True)
    def main(
        Input: T.Tensor([tokens, hidden], dtype),
        WGate: T.Tensor([expert, hidden], dtype),
        WUp: T.Tensor([expert, hidden], dtype),
        Output: T.Tensor([tokens, expert], dtype),
    ):
        with T.Kernel(
            T.ceildiv(tokens, block_token),
            T.ceildiv(expert, block_expert),
            threads=256,
        ) as (bx, by):
            input_shared = T.alloc_shared([block_token, block_hidden], dtype)
            gate_shared = T.alloc_shared([block_expert, block_hidden], dtype)
            up_shared = T.alloc_shared([block_expert, block_hidden], dtype)
            gate = T.alloc_fragment([block_token, block_expert], T.float32)
            up = T.alloc_fragment([block_token, block_expert], T.float32)
            T.clear(gate)
            T.clear(up)
            for k in T.Pipelined(T.ceildiv(hidden, block_hidden), num_stages=1):
                T.copy(Input[bx * block_token, k * block_hidden], input_shared)
                T.copy(WGate[by * block_expert, k * block_hidden], gate_shared)
                T.copy(WUp[by * block_expert, k * block_hidden], up_shared)
                T.gemm(input_shared, gate_shared, gate, transpose_B=True)
                T.gemm(input_shared, up_shared, up, transpose_B=True)
            for i, j in T.Parallel(block_token, block_expert):
                up[i, j] *= gate[i, j] / (1.0 + T.exp2(-gate[i, j] * 1.44269504))
            T.copy(up, Output[bx * block_token, by * block_expert])

    return main


def reference(x, gate, up):
    return (F.silu(x.float() @ gate.float().T) * (x.float() @ up.float().T)).half()


def build(options, config):
    tokens, hidden, expert = options["fused_moe_tokens"], options["fused_moe_hidden"], options["fused_moe_expert"]
    prim = kernel(tokens, hidden, expert, config["block_token"], config["block_hidden"], config["block_expert"])
    return SearchWorkload(
        prim_func=prim, out_idx=(3,),
        total_flops=4.0 * tokens * hidden * expert,
        reference_program=reference,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


OPERATOR = OperatorSpec(
    name="fused_moe", description="Fused shared-expert gate and up projections",
    add_cli_arguments=add_arguments, configuration_factory=configurations, workload_factory=build,
)
