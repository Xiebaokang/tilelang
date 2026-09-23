"""Replay searched OverlapPlan schedules for migrated operators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tilelang
import torch
from tvm.error import TVMError
from tvm.target import Target

from OverlapPlaner.tune.operators.gemm import build as build_gemm
from OverlapPlaner.tune.replay import infer_replay_context, replay

DATA_DIR = Path(__file__).parent / "data"
OPERATORS = ("fa3", "mla", "gemm", "gqa", "convolution", "gemm_fp8")


def _has_hopper_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.parametrize("operator_name", OPERATORS)
def test_infer_replay_context_from_test_manifest(operator_name: str) -> None:
    operator, tile, options = infer_replay_context(
        DATA_DIR / operator_name / "schedule.json"
    )
    manifest = json.loads(
        (DATA_DIR / operator_name / "manifest.json").read_text(encoding="utf-8")
    )
    assert operator.name == operator_name
    assert tile == manifest["tile"]
    assert options == manifest["options"]


def test_auto_overlap_without_plan_is_rejected() -> None:
    workload = build_gemm(
        {"gemm_m": 256, "gemm_n": 256, "gemm_k": 256},
        {"block_m": 128, "block_n": 128, "block_k": 64},
    )
    target = Target({"kind": "cuda", "arch": "sm_90a"})
    with pytest.raises(TVMError, match="requires a searched OverlapPlan"):
        with target:
            tilelang.compile(
                workload.prim_func,
                out_idx=list(workload.out_idx),
                target=target,
            )


@pytest.mark.skipif(not _has_hopper_gpu(), reason="requires a Hopper CUDA GPU")
@pytest.mark.parametrize("operator_name", OPERATORS)
def test_replay_searched_operator_schedule(operator_name: str) -> None:
    latency_ms, tflops = replay(
        DATA_DIR / operator_name / "schedule.json",
        warmup=5,
        rep=5,
    )
    assert latency_ms > 0
    assert tflops > 0
    print(
        f"\n{operator_name} OverlapPlan replay: PASS, "
        f"latency={latency_ms:.6f} ms, throughput={tflops:.3f} TFLOPS"
    )
