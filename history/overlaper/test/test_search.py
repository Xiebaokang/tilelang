"""CPU-only control-flow tests for the Overlaper search driver."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import history.overlaper.tune.search as search_module
import history.overlaper.tune.run as tune_run
import pytest
import tilelang.profiler.bench as bench_module
from history.overlaper.tune.search import (
    PtxasPerformanceLoss,
    ptxas_performance_loss_warnings,
)
from history.overlaper.headware import HOPPER
from history.overlaper.integration.capture import _ACTIVE_PLANNER
from history.overlaper.parse import DataflowGraph
from history.overlaper.test.test_extractor import make_fa3_prim_func
from tvm.target import Target


class _Profiler:
    def __init__(self, latency: float, error: Exception | None = None):
        self.latency = latency
        self.error = error

    def assert_allclose(self, *_args, **_kwargs):
        if self.error is not None:
            raise self.error

    def do_bench_iterations(self, *_args, **_kwargs):
        return self.latency


class _Compiled:
    def __init__(self, profiler):
        self.profiler = profiler

    def get_profiler(self):
        return self.profiler

    def get_kernel_source(self):
        return "// generated"


def test_iteration_benchmark_uses_exact_counts(monkeypatch) -> None:
    calls = 0
    measured_repetitions = None

    def kernel() -> None:
        nonlocal calls
        calls += 1

    def measure(
        function,
        _cache,
        repetitions,
        _quantiles,
        _return_mode,
        _device_idx,
    ):
        nonlocal measured_repetitions
        measured_repetitions = repetitions
        for _ in range(repetitions):
            function()
        return 1.0

    monkeypatch.setattr(bench_module, "_make_cache", lambda *_args: object())
    monkeypatch.setattr(bench_module, "_bench_with_cuda_events", measure)

    result = bench_module.do_bench_iterations(kernel, warmup=3, rep=7)

    assert result == 1.0
    assert measured_repetitions == 7
    assert calls == 10


def test_search_records_success_and_failed_schedule(monkeypatch, tmp_path) -> None:
    graph = DataflowGraph(
        buffers=(),
        nodes=(),
        edges=(),
        region_kinds=(),
        hardware=HOPPER,
        kernel_threads=256,
    )
    schedules = [
        SimpleNamespace(
            identity=index,
            num_groups=1,
            stages_by_region={},
            warp_allocation=SimpleNamespace(effective_threads=256),
        )
        for index in range(2)
    ]
    compile_index = 0

    def fake_compile(*_args, **_kwargs):
        nonlocal compile_index
        planner = _ACTIVE_PLANNER.get()
        assert planner is not None
        planner("main", graph, Target("cuda"))
        profiler = (
            _Profiler(0.25)
            if compile_index == 0
            else _Profiler(0.5, AssertionError("wrong result"))
        )
        compile_index += 1
        return _Compiled(profiler)

    monkeypatch.setattr(
        search_module, "enumerate_schedules", lambda *_args, **_kwargs: iter(schedules)
    )
    monkeypatch.setattr(
        search_module,
        "schedule_to_dict",
        lambda _graph, schedule: {"identity": schedule.identity},
    )
    monkeypatch.setattr(search_module.tilelang, "compile", fake_compile)

    summary = search_module.search_schedules(
        make_fa3_prim_func(),
        target=Target("cuda"),
        out_idx=[3],
        total_flops=1e9,
        reference_program=lambda *_args: None,
        output_directory=tmp_path,
        warmup=0,
        rep=1,
    )

    assert summary.examined_schedules == 2
    assert summary.successful_schedules == 1
    assert summary.failed_schedules == 1
    result = json.loads((tmp_path / "results.jsonl").read_text())
    assert result["latency_ms"] == 0.25
    assert result["tflops"] == 4.0
    assert result["source_file"] == "sources/schedule_00000.cu"
    assert (tmp_path / result["source_file"]).read_text() == "// generated"
    assert (tmp_path / "sources" / "schedule_00001.cu").read_text() == (
        "// generated"
    )
    failed = json.loads(
        (tmp_path / "failures" / "schedule_00001.json").read_text()
    )
    assert failed == {"identity": 1}


def test_empty_enumeration_preserves_original_error(monkeypatch, tmp_path) -> None:
    graph = DataflowGraph(
        buffers=(),
        nodes=(),
        edges=(),
        region_kinds=(),
        hardware=HOPPER,
        kernel_threads=256,
    )

    def fake_compile(*_args, **_kwargs):
        planner = _ACTIVE_PLANNER.get()
        assert planner is not None
        planner("main", graph, Target("cuda"))

    monkeypatch.setattr(
        search_module, "enumerate_schedules", lambda *_args, **_kwargs: iter(())
    )
    monkeypatch.setattr(search_module.tilelang, "compile", fake_compile)

    with pytest.raises(
        RuntimeError, match="Overlaper did not enumerate a feasible schedule"
    ):
        search_module.search_schedules(
            make_fa3_prim_func(),
            target=Target("cuda"),
            out_idx=[3],
            total_flops=1e9,
            reference_program=lambda *_args: None,
            output_directory=tmp_path,
            warmup=0,
            rep=1,
        )

    detail = json.loads((tmp_path / "enumeration.error.json").read_text())
    assert detail["phase"] == "enumeration"
    assert detail["schedule_file"] is None


def test_ptxas_performance_loss_parser_keeps_warning_lines() -> None:
    log = "\n".join(
        (
            "2026-09-13 [TileLang] TileLang completes to compile kernel `main`",
            "ptxas info    : (C7512) Potential Performance Loss: wgmma.mma_async "
            "instructions are serialized due to insufficient register resources "
            "for the function 'main_kernel'",
            "ptxas info    : Used 240 registers",
        )
    )
    warnings = ptxas_performance_loss_warnings(log)
    assert len(warnings) == 1
    assert "C7512" in warnings[0]
    assert "Potential Performance Loss" in warnings[0]
    with pytest.raises(PtxasPerformanceLoss, match="C7512"):
        raise PtxasPerformanceLoss("skipping execution after ptxas:\n" + warnings[0])


def test_search_skips_execution_after_ptxas_performance_loss(
    monkeypatch, tmp_path
) -> None:
    graph = DataflowGraph(
        buffers=(),
        nodes=(),
        edges=(),
        region_kinds=(),
        hardware=HOPPER,
        kernel_threads=256,
    )
    schedules = [
        SimpleNamespace(
            identity=0,
            num_groups=1,
            stages_by_region={},
            warp_allocation=SimpleNamespace(effective_threads=256),
        )
    ]
    executed = {"profiler": False}

    def fake_compile(*_args, **_kwargs):
        planner = _ACTIVE_PLANNER.get()
        assert planner is not None
        planner("main", graph, Target("cuda"))
        os.write(
            2,
            (
                b"ptxas info    : (C7512) Potential Performance Loss: "
                b"wgmma.mma_async instructions are serialized due to "
                b"insufficient register resources for the function "
                b"'main_kernel'\n"
            ),
        )
        return _Compiled(_Profiler(0.25))

    original_get_profiler = _Compiled.get_profiler

    def tracked_get_profiler(self):
        executed["profiler"] = True
        return original_get_profiler(self)

    monkeypatch.setattr(_Compiled, "get_profiler", tracked_get_profiler)
    monkeypatch.setattr(
        search_module, "enumerate_schedules", lambda *_args, **_kwargs: iter(schedules)
    )
    monkeypatch.setattr(
        search_module,
        "schedule_to_dict",
        lambda _graph, schedule: {"identity": schedule.identity},
    )
    monkeypatch.setattr(search_module.tilelang, "compile", fake_compile)

    summary = search_module.search_schedules(
        make_fa3_prim_func(),
        target=Target("cuda"),
        out_idx=[3],
        total_flops=1e9,
        reference_program=lambda *_args: None,
        output_directory=tmp_path,
        warmup=0,
        rep=1,
    )

    assert summary.examined_schedules == 1
    assert summary.successful_schedules == 0
    assert summary.failed_schedules == 1
    assert executed["profiler"] is False
    assert not (tmp_path / "results.jsonl").exists()
    detail = json.loads(
        (tmp_path / "failures" / "schedule_00000.error.json").read_text()
    )
    assert detail["phase"] == "ptxas_check"
    assert detail["error_type"] == "PtxasPerformanceLoss"
    assert "C7512" in detail["error"]
    assert (tmp_path / "sources" / "schedule_00000.cu").read_text() == "// generated"


def test_timeout_record_keeps_replayable_schedule(tmp_path) -> None:
    tile = tmp_path / "m128_n128"
    segment = tile / "segments" / "run" / "start_00000"
    candidate = segment / "candidates" / "schedule_00003.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text('{"groups": {"0": 0}}', encoding="utf-8")

    tune_run._record_process_failure(
        tile,
        {
            "schedule_index": 3,
            "phase": "benchmark",
            "schedule_file": "segments/run/start_00000/candidates/schedule_00003.json",
        },
        kind="timeout",
        message="phase exceeded 5 seconds",
        extra={"timeout_seconds": 5.0},
    )

    saved = tile / "failures" / "timeout_schedule_00003.json"
    detail = tile / "failures" / "timeout_schedule_00003.error.json"
    assert json.loads(saved.read_text()) == {"groups": {"0": 0}}
    assert json.loads(detail.read_text())["error_type"] == "timeout"


def test_global_top30_is_sorted_and_records_artifact_paths(tmp_path) -> None:
    results = []
    candidates = tmp_path / "m128_n128" / "candidates"
    candidates.mkdir(parents=True)
    for index in range(35):
        schedule_file = f"m128_n128/candidates/schedule_{index:05d}.json"
        (tmp_path / schedule_file).write_text(
            json.dumps({"identity": index}), encoding="utf-8"
        )
        results.append(
            {
                "operator": "fa3",
                "tile": {"block_m": 128, "block_n": 128},
                "config_directory": "m128_n128",
                "schedule_file": schedule_file,
                "source_file": f"m128_n128/sources/schedule_{index:05d}.cu",
                "schedule_index": index,
                "block_m": 128,
                "block_n": 128,
                "latency_ms": 35 - index,
                "tflops": float(index),
            }
        )

    tune_run._write_top30(
        tmp_path,
        tune_run.FA3,
        results,
        [],
    )
    top = json.loads((tmp_path / "top30.json").read_text())["top_results"]
    assert len(top) == 30
    assert top[0]["tflops"] == 34.0
    assert top[0]["schedule_file"] == (
        "m128_n128/candidates/schedule_00034.json"
    )
    assert top[0]["source_file"] == "m128_n128/sources/schedule_00034.cu"
