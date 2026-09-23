"""OverlapPlan search kernel adapted from examples/gemm/example_gemm.py."""

from itertools import product

import tilelang.language as T

from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig


def add_arguments(parser) -> None:
    group = parser.add_argument_group("GEMM")
    group.add_argument("--gemm-block-m", type=int, nargs="+", default=[64, 128, 192])
    group.add_argument("--gemm-block-n", type=int, nargs="+", default=[64, 80, 96, 128, 256])
    group.add_argument("--gemm-block-k", type=int, nargs="+", default=[32, 64])
    group.add_argument("--gemm-m", type=int, default=4096)
    group.add_argument("--gemm-n", type=int, default=4096)
    group.add_argument("--gemm-k", type=int, default=4096)


def configurations(options: Options) -> list[TileConfig]:
    return [
        {"block_m": block_m, "block_n": block_n, "block_k": block_k}
        for block_m, block_n, block_k in product(
            options["gemm_block_m"],
            options["gemm_block_n"],
            options["gemm_block_k"],
        )
    ]


def build(options: Options, config: TileConfig) -> SearchWorkload:
    m, n, k = options["gemm_m"], options["gemm_n"], options["gemm_k"]
    block_m = config["block_m"]
    block_n = config["block_n"]
    block_k = config["block_k"]
    dtype = T.float16

    @T.prim_func(auto_overlap=True)
    def main(
        A: T.Tensor((m, k), dtype),
        B: T.Tensor((k, n), dtype),
        C: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(
            T.ceildiv(n, block_n),
            T.ceildiv(m, block_m),
            threads=block_m // 64 * 128,
        ) as (bx, by):
            a_shared = T.alloc_shared((block_m, block_k), dtype)
            b_shared = T.alloc_shared((block_k, block_n), dtype)
            c_local = T.alloc_fragment((block_m, block_n), T.float32)
            T.clear(c_local)
            for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=2):
                T.copy(A[by * block_m, ko * block_k], a_shared)
                T.copy(B[ko * block_k, bx * block_n], b_shared)
                T.gemm(a_shared, b_shared, c_local)
            T.copy(c_local, C[by * block_m, bx * block_n])

    return SearchWorkload(
        prim_func=main,
        out_idx=(2,),
        total_flops=2.0 * m * n * k,
        reference_program=lambda a, b: a @ b,
    )


OPERATOR = OperatorSpec(
    name="gemm",
    description="Dense FP16 GEMM adapted from examples/gemm/example_gemm.py",
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
