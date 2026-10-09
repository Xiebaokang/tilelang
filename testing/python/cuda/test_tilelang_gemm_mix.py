"""Numerical, accumulation and RS-before-SS checks for Hopper mixed GEMM."""
import importlib.util
import inspect
import re
from pathlib import Path

import pytest
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[3])
def mixed_gemm(M, N, Ks, Kr, transpose_B=True, clear_accum=True, dtype="float16", threads=256):
    K = Ks + Kr
    b_shape = (N, K) if transpose_B else (K, N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor(b_shape, dtype),
        Initial: T.Tensor((M, N), "float32"),
        Out: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=threads):
            SA = T.alloc_shared((M, Ks), dtype)
            RA = T.alloc_fragment((M, Kr), dtype)
            SB = T.alloc_shared(b_shape, dtype)
            C = T.alloc_fragment((M, N), "float32")
            T.copy(A[:, :Ks], SA)
            T.copy(A[:, Ks:], RA)
            T.copy(B, SB)
            T.copy(Initial, C)
            T.gemm_mix(SA, RA, SB, C, transpose_B=transpose_B, clear_accum=clear_accum, policy=T.GemmWarpPolicy.FullRow)
            T.copy(C, Out)

    return main


@pytest.mark.parametrize("ks,kr,n,threads", [(32, 32, 64, 128), (80, 48, 96, 256), (112, 80, 112, 256)])
@pytest.mark.parametrize("transpose_b", [True, False])
@pytest.mark.parametrize("clear", [True, False])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_gemm_mix(ks, kr, n, threads, transpose_b, clear, dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper required")
    kernel = mixed_gemm(128, n, ks, kr, transpose_b, clear, dtype, threads)
    source = kernel.get_kernel_source()
    calls = re.findall(r"tl::wgmma_(rs|ss)<", source)
    assert calls and set(calls) == {"rs", "ss"}
    first_ss = calls.index("ss")
    assert all(op == "rs" for op in calls[:first_ss])
    assert all(op == "ss" for op in calls[first_ss:])
    assert source.count("tl::warpgroup_commit_batch()") == 1
    assert source.count("tl::warpgroup_wait<0>()") == 1
    for seed in (17, 41):
        torch.manual_seed(seed)
        a = torch.randn((128, ks + kr), device="cuda", dtype=getattr(torch, dtype)) * .2
        b = torch.randn((n, ks + kr) if transpose_b else (ks + kr, n), device="cuda", dtype=getattr(torch, dtype)) * .2
        initial = torch.randn((128, n), device="cuda", dtype=torch.float32)
        result = kernel(a, b, initial)
        expected = a.float() @ (b.float().T if transpose_b else b.float())
        if not clear:
            expected += initial
        torch.testing.assert_close(result, expected, atol=2e-4, rtol=2e-3)


def test_gemm_mix_signature():
    ordinary = list(inspect.signature(T.gemm).parameters.values())[3:]
    mixed = list(inspect.signature(T.gemm_mix).parameters.values())[4:]
    assert [(p.name, p.default) for p in ordinary] == [(p.name, p.default) for p in mixed]


@pytest.mark.parametrize("transpose_b", [True, False])
def test_gemm_mix_multiple_n_atoms(transpose_b):
    # Non-transposed B must use full K, rather than Kr/Ks, for its N-atom stride.
    test_gemm_mix(80, 48, 320, 256, transpose_b, False, "float16")


@pytest.fixture(scope="module")
def attention_example():
    path = Path(__file__).resolve().parents[3] / "examples/flash_attention/example_mha_fwd_bhsd_mix.py"
    spec = importlib.util.spec_from_file_location("gemm_mix_attention_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_attention(example, dim, causal, stages, sq=160, sk=384, p_smem_k_tiles=2, block_m=128, threads=256):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper required")
    # D192/BN128/two stages already exceeds H100 block SMEM before staging P.
    block_n = 64 if dim == 192 and stages == 2 else 128
    kernel = example.flashattn(
        1, 2, sq, sk, dim, causal, block_M=block_m, block_N=block_n,
        num_stages=stages, threads=threads, q_reg_k_tiles=2, p_smem_k_tiles=p_smem_k_tiles
    )
    source = kernel.get_kernel_source()
    assert not re.search(r"tl::ptx_stmatrix_[^(]+\([^;]*P_shared", source), "P shared handoff uses ordinary stores"
    mixed_groups = 0
    for segment in source.split("tl::warpgroup_commit_batch()"):
        calls = re.findall(r"tl::wgmma_(rs|ss)<", segment)
        if "rs" in calls and "ss" in calls:
            last_rs_pos = segment.rindex("tl::wgmma_rs<")
            first_ss_pos = segment.index("tl::wgmma_ss<")
            if "P_reg" in segment[last_rs_pos:first_ss_pos]:
                assert "__sync_thread_partial" in segment[last_rs_pos:first_ss_pos], (
                    "P_shared handoff must synchronize after PV RS and before SS"
                )
            first_ss = calls.index("ss")
            assert all(op == "rs" for op in calls[:first_ss])
            assert all(op == "ss" for op in calls[first_ss:])
            mixed_groups += 1
    assert mixed_groups >= 2, "Both QK and PV must lower to mixed RS/SS groups"
    torch.manual_seed(101)
    q = torch.randn((1, 2, sq, dim), device="cuda", dtype=torch.float16) * .2
    k = torch.randn((1, 2, sk, dim), device="cuda", dtype=torch.float16) * .2
    v = torch.randn_like(k) * .2
    result = kernel(q, k, v)
    score = (q.float() @ k.float().transpose(-1, -2)) * dim**-.5
    if causal:
        mask = torch.arange(sk, device="cuda")[None, :] <= torch.arange(sq, device="cuda")[:, None] + sk - sq
        score = score.masked_fill(~mask, float("-inf"))
    expected = score.softmax(-1) @ v.float()
    torch.testing.assert_close(result.float(), expected, atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize("dim", [64, 128, 192])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("stages", [1, 2])
def test_gemm_mix_attention(attention_example, dim, causal, stages):
    check_attention(attention_example, dim, causal, stages)


def test_gemm_mix_persistent_q_registers(attention_example):
    # Regression: CUDA 12.8 recycled the Q RS input registers for first-iteration
    # P, corrupting the second QK tile unless RA has a real non-WGMMA use.
    check_attention(attention_example, 128, True, 1, sq=128, sk=256)


@pytest.mark.parametrize("p_smem_k_tiles", [1, 3])
def test_gemm_mix_pv_split(attention_example, p_smem_k_tiles):
    check_attention(attention_example, 128, False, 2, p_smem_k_tiles=p_smem_k_tiles)


def test_gemm_mix_pv_single_warpgroup(attention_example):
    check_attention(attention_example, 128, True, 1, block_m=64, threads=128)


@tilelang.jit(out_idx=[2])
def accumulator_prefix_copy(width, offset=0, threads=256):
    @T.prim_func
    def main(A: T.Tensor((128, 32), "float16"), B: T.Tensor((128, 32), "float16"),
             Out: T.Tensor((128, width), "float16")):
        with T.Kernel(1, threads=threads):
            SA = T.alloc_shared((128, 32), "float16")
            SB = T.alloc_shared((128, 32), "float16")
            C = T.alloc_fragment((128, 128), "float32")
            Prefix = T.alloc_shared((128, width), "float16")
            T.copy(A, SA)
            T.copy(B, SB)
            T.gemm(SA, SB, C, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(C[:, offset:offset + width], Prefix)
            T.copy(Prefix, Out)
    return main


@pytest.mark.parametrize("width,offset,threads,use_stmatrix", [
    (16, 0, 256, False), (32, 0, 256, False), (48, 0, 256, False),
    (32, 16, 256, False), (32, 0, 128, False),
])
def test_accumulator_prefix_copy(width, offset, threads, use_stmatrix):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper required")
    kernel = accumulator_prefix_copy(width, offset, threads)
    source = kernel.get_kernel_source()
    assert ("tl::ptx_stmatrix_" in source) == use_stmatrix
    torch.manual_seed(29)
    a = torch.randn((128, 32), device="cuda", dtype=torch.float16) * .2
    b = torch.randn_like(a) * .2
    result = kernel(a, b)
    expected = (a.float() @ b.float().T)[:, offset:offset + width].half()
    torch.testing.assert_close(result, expected, atol=2e-4, rtol=2e-3)
