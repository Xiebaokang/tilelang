"""Tests for OverlapPlan schedule-folder search."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from OverlapPlaner.ir import BufferPlan, GroupPlan, OperationPlacement, OverlapPlan
from OverlapPlaner.serialization import load_plan_json, save_plan_json
from OverlapPlaner.tune.operators.gemm import OPERATOR as GEMM
from OverlapPlaner.tune.search import list_plan_files, rank_results, search

DATA_DIR = Path(__file__).parent / "data"


def _has_hopper_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def test_list_plan_files_skips_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "schedule_00001.json").write_text("{}", encoding="utf-8")
    (tmp_path / "schedule_00000.json").write_text("{}", encoding="utf-8")
    (tmp_path / "schedule_00002.error.json").write_text("{}", encoding="utf-8")
    files = list_plan_files(tmp_path)
    assert [path.name for path in files] == [
        "schedule_00000.json",
        "schedule_00001.json",
    ]


def test_rank_results_keeps_top_n_by_tflops() -> None:
    ranked = rank_results(
        [
            {"schedule_file": "a.json", "tflops": 1.0, "latency_ms": 2.0},
            {"schedule_file": "b.json", "tflops": 3.0, "latency_ms": 0.5},
            {"schedule_file": "c.json", "tflops": 2.0, "latency_ms": 1.0},
        ],
        n_rank=2,
    )
    assert [item["rank"] for item in ranked] == [1, 2]
    assert [item["schedule_file"] for item in ranked] == ["b.json", "c.json"]


def test_native_plan_json_round_trip(tmp_path: Path) -> None:
    plan = OverlapPlan(
        groups=[GroupPlan(4)],
        operations=[OperationPlacement(0, None, 0, None, 0)],
        buffers=[BufferPlan(0, None, 1, 0)],
        sync_edges=[],
        shared_arena_bytes=0,
    )
    path = tmp_path / "schedule_00000.json"
    save_plan_json(path, plan)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "schema_version" not in payload
    loaded = load_plan_json(path)
    assert int(loaded.groups[0].warp_count) == 4
    assert int(loaded.operations[0].operation_id) == 0
    assert loaded.operations[0].statement is None
    assert loaded.buffers[0].buffer is None


def test_legacy_schedule_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "schedule.json"
    path.write_text('{"groups": {"0": 0}}', encoding="utf-8")
    with pytest.raises(ValueError, match="groups must be an array"):
        load_plan_json(path)


@pytest.mark.skipif(not _has_hopper_gpu(), reason="requires a Hopper CUDA GPU")
def test_search_writes_cuda_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    payload = search(
        GEMM,
        {"block_m": 128, "block_n": 128, "block_k": 64},
        {"gemm_m": 256, "gemm_n": 256, "gemm_k": 256},
        DATA_DIR / "gemm",
        source_dir,
    )
    assert len(payload["successful"]) == 1
    assert payload["failures"] == []
    assert payload["successful"][0]["tflops"] > 0
    assert (source_dir / "schedule.cu").is_file()
    source = (source_dir / "schedule.cu").read_text(encoding="utf-8")
    assert source.strip()
    assert "__global__" in source or "cuda" in source.lower()
    assert payload["successful"][0]["source_file"].endswith("schedule.cu")
