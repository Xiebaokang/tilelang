"""Regression coverage for selecting wide Hopper WGMMA N instructions."""

import re

import pytest
from tvm.target import Target

import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import select_wgmma_inst_n


@pytest.mark.parametrize(
    "warp_col_tiles, expected",
    [
        (64, 64),
        (96, 96),
        (112, 112),
        (160, 160),
        (176, 176),
        (192, 192),
        (256, 256),
        (384, 192),
        (512, 256),
        # Preserve the existing N % 16 == 8 behavior.
        (24, 8),
        (40, 8),
        (88, 8),
    ],
)
def test_select_wgmma_inst_n(warp_col_tiles, expected):
    assert select_wgmma_inst_n(warp_col_tiles) == expected


def _make_matmul_nt(block_n):
    @T.prim_func
    def main(
        A: T.Tensor((128, 16), T.float16),
        B: T.Tensor((block_n, 16), T.float16),
        C: T.Tensor((128, block_n), T.float16),
    ):
        with T.Kernel(1, threads=256):
            A_shared = T.alloc_shared((128, 16), T.float16)
            B_shared = T.alloc_shared((block_n, 16), T.float16)
            C_local = T.alloc_fragment((128, block_n), T.float32)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.copy(C_local, C)

    return main


@pytest.mark.parametrize("block_n", [176, 192])
def test_gemm_lowers_to_single_wide_wgmma(block_n):
    target = Target({"kind": "cuda", "arch": "sm_90"})
    with target:
        artifact = tilelang.lower(_make_matmul_nt(block_n), target=target)
    shapes = set(
        re.findall(r"wgmma_ss<[^>]*?, (\d+, \d+, \d+),", artifact.kernel_source)
    )
    assert shapes == {f"64, {block_n}, 16"}
