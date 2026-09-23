import math

import pytest
import tilelang
import torch
import history.wspipeline as wsp
from tvm.target import Target

from history.wspipeline.test.fa3_kernel import flashattn


def _has_hopper_gpu():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def run_fa3(auto_wsp: bool):
    """Compile and check FA3 through the selected scheduling path."""

    batch, heads, seq_q, seq_kv, dim = 1, 16, 8192, 8192, 128
    planner_called = False

    def select_schedule(_name, analysis, target):
        nonlocal planner_called
        planner_called = True
        candidates = wsp.enumerate_realized_schedules(
            analysis, target, examined_limit=1000
        )
        return min(
            candidates,
            key=lambda candidate: wsp.score_realized_schedule(
                analysis, candidate, target
            ),
        )

    prim_func = flashattn.jit_impl.get_tir(
        batch,
        heads,
        seq_q,
        seq_kv,
        dim,
        False,
        block_M=128,
        block_N=128,
        threads=256,
        auto_wsp=auto_wsp,
    )
    target = Target(
        {
            "kind": "cuda",
            "arch": "sm_90a",
            "max_threads_per_block": 1024,
            "max_shared_memory_per_block": 232448,
        }
    )
    with wsp.use_schedule_planner(select_schedule):
        kernel = tilelang.compile(
            prim_func,
            out_idx=[3],
            target=target,
            execution_backend="cython",
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
    assert planner_called is auto_wsp

    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq_q, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, heads, seq_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, heads, seq_kv, dim, device="cuda", dtype=torch.float16)

    actual = kernel(q, k, v)
    scores = q.float() @ k.float().transpose(-2, -1) / math.sqrt(dim)
    expected = (scores.softmax(dim=-1) @ v.float()).half()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)

    latency_ms = kernel.get_profiler().do_bench(
        input_tensors=[q, k, v], warmup=100, rep=100
    )
    total_flops = 4.0 * batch * heads * seq_q * seq_kv * dim
    tflops = total_flops / latency_ms * 1e-9
    print(
        f"auto_wsp={auto_wsp}: latency={latency_ms:.3f} ms, "
        f"performance={tflops:.2f} TFLOPS"
    )
    return latency_ms, tflops


@pytest.mark.parametrize("auto_wsp", [False, True])
@pytest.mark.skipif(not _has_hopper_gpu(), reason="requires a Hopper CUDA GPU")
def test_fa3_auto_wsp(auto_wsp):
    run_fa3(auto_wsp)
