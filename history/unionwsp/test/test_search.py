"""Regression tests for the measured UnionWSP search driver."""

import json
import subprocess
import sys

import pytest

import history.unionwsp.search as search_module
from history.unionwsp.hardware import HOPPER
from history.unionwsp.integration.capture import _ACTIVE_PLANNER
from history.unionwsp.parseIR import DataflowGraph
from history.unionwsp.test.test_stage import hopper_target, make_fa3_prim_func
from history.unionwsp.test import test_search_fa3_tiles as tile_search


def test_zero_schedule_search_stops_after_initialization(
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
    compile_calls = 0

    def fake_compile(*_args, **_kwargs):
        nonlocal compile_calls
        compile_calls += 1
        planner = _ACTIVE_PLANNER.get()
        assert planner is not None
        planner("main", graph, hopper_target())

    monkeypatch.setattr(
        search_module,
        "enumerate_wsp_schedules",
        lambda *_args, **_kwargs: iter(()),
    )
    monkeypatch.setattr(search_module.tilelang, "compile", fake_compile)

    with pytest.raises(
        RuntimeError, match="did not enumerate a feasible schedule"
    ):
        search_module.search_wsp_schedules(
            make_fa3_prim_func(),
            target=hopper_target(),
            out_idx=[3],
            total_flops=1.0,
            reference_program=lambda *_args: None,
            output_directory=tmp_path,
        )

    assert compile_calls == 1


def test_candidate_output_order(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        search_module,
        "schedule_to_dict",
        lambda _graph, _schedule: {"groups": {"0": 0}},
    )

    search_module._print_schedule(None, None, 0, 3)
    search_module._print_generated_code("// generated kernel", 0, 3)
    search_module._print_performance(0.125, 42.0, 0, 3)

    output = capsys.readouterr().out
    schedule_position = output.index("=== Schedule [1/3] ===")
    source_position = output.index("=== Generated code [1/3] ===")
    performance_position = output.index("=== Performance [1/3] ===")

    assert schedule_position < source_position < performance_position
    assert "latency: 0.1250 ms" in output
    assert "throughput: 42.00 TFLOPS" in output


def test_failure_artifacts_include_schedule_and_generated_source(
    tmp_path, capsys
) -> None:
    failure = {
        "schedule_index": 2,
        "phase": "correctness_validation",
        "error_type": "AssertionError",
        "error": "output mismatch",
        "traceback": "traceback text",
        "schedule": {"groups": {"0": 1}},
    }

    record = search_module._write_failure_artifacts(
        tmp_path,
        failure,
        "// generated kernel",
    )
    search_module._print_failure(record, 10)

    detail_path = tmp_path / record["detail_file"]
    source_path = tmp_path / record["source_file"]
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    jsonl_record = json.loads(
        (tmp_path / "failures.jsonl").read_text(encoding="utf-8").strip()
    )

    assert detail["schedule"] == failure["schedule"]
    assert detail["phase"] == "correctness_validation"
    assert jsonl_record == detail
    assert source_path.read_text(encoding="utf-8") == "// generated kernel"
    output = capsys.readouterr().out
    assert "=== Rejected [3/10] ===" in output
    assert "phase: correctness_validation" in output
    assert "generated_source: failures/schedule_00002.cu" in output


def test_compile_failure_does_not_claim_a_source_file(tmp_path) -> None:
    record = search_module._write_failure_artifacts(
        tmp_path,
        {
            "schedule_index": 4,
            "phase": "compilation",
            "error_type": "RuntimeError",
            "error": "compile failed",
        },
        None,
    )

    assert record["source_file"] is None
    assert not (tmp_path / "failures" / "schedule_00004.cu").exists()


def test_invalid_search_resume_range_is_rejected(tmp_path) -> None:
    with pytest.raises(
        ValueError, match="start_schedule_index must be smaller than max_schedules"
    ):
        search_module.search_wsp_schedules(
            make_fa3_prim_func(),
            target=hopper_target(),
            out_idx=[3],
            total_flops=1.0,
            reference_program=lambda *_args: None,
            output_directory=tmp_path,
            start_schedule_index=3,
            max_schedules=3,
        )


def test_watchdog_terminates_an_entire_worker_process_group() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )

    tile_search._terminate_process_group(process, grace_seconds=0.1)

    assert process.poll() is not None
