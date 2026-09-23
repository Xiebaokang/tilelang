"""OverlapPlan search kernel adapted from examples/gemm_fp8/example_tilelang_gemm_fp8.py."""

from itertools import product

import tilelang.language as T

from .workloads import OperatorSpec, Options, SearchWorkload, TileConfig


def add_arguments(parser) -> None:
    group = parser.add_argument_group("GEMM FP8")
    group.add_argument("--gemm-fp8-block-m", type=int, nargs="+", default=[64, 128])
    group.add_argument("--gemm-fp8-block-n", type=int, nargs="+", default=[64, 128])
    group.add_argument("--gemm-fp8-block-k", type=int, nargs="+", default=[64, 128])
    group.add_argument("--gemm-fp8-m", type=int, default=4096)
    group.add_argument("--gemm-fp8-n", type=int, default=4096)
    group.add_argument("--gemm-fp8-k", type=int, default=4096)


def configurations(options: Options) -> list[TileConfig]:
    return [
        {"block_m": block_m, "block_n": block_n, "block_k": block_k}
        for block_m, block_n, block_k in product(
            options["gemm_fp8_block_m"],
            options["gemm_fp8_block_n"],
            options["gemm_fp8_block_k"],
        )
    ]


def build(options: Options, config: TileConfig) -> SearchWorkload:
    m, n, k = options["gemm_fp8_m"], options["gemm_fp8_n"], options["gemm_fp8_k"]
    block_m = config["block_m"]
    block_n = config["block_n"]
    block_k = config["block_k"]
    dtype = T.float8_e4m3fn

    @T.prim_func(auto_overlap=True)
    def main(
        A: T.Tensor((m, k), dtype),
        B: T.Tensor((n, k), dtype),
        C: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(
            T.ceildiv(n, block_n),
            T.ceildiv(m, block_m),
            threads=block_m // 64 * 128,
        ) as (bx, by):
            a_shared = T.alloc_shared((block_m, block_k), dtype)
            b_shared = T.alloc_shared((block_n, block_k), dtype)
            c_local = T.alloc_fragment((block_m, block_n), T.float32)
            T.clear(c_local)
            for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=2):
                T.copy(A[by * block_m, ko * block_k], a_shared)
                T.copy(B[bx * block_n, ko * block_k], b_shared)
                T.gemm(a_shared, b_shared, c_local, transpose_B=True)
            T.copy(c_local, C[by * block_m, bx * block_n])

    def reference(a, b):
        return (a.float() @ b.float().T).to(a.dtype)

    return SearchWorkload(
        prim_func=main,
        out_idx=(2,),
        total_flops=2.0 * m * n * k,
        reference_program=reference,
    )


OPERATOR = OperatorSpec(
    name="gemm_fp8",
    description=(
        "Hopper FP8 GEMM adapted from "
        "examples/gemm_fp8/example_tilelang_gemm_fp8.py"
    ),
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
