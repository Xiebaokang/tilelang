"""CPU-only tests for the multi-operator search front end."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit

import history.overlaper.tune.run as tune_run
from history.overlaper.tune.run import (
    _record_process_failure,
    _write_top30,
    make_parser,
)
from history.overlaper.tune.operators import (
    CONVOLUTION,
    GEMM,
    OPERATORS,
    OPERATOR_NAMES,
    OperatorSpec,
    SearchWorkload,
    get_operator,
)


def test_selected_operators_and_tile_grids() -> None:
    args = make_parser([GEMM, CONVOLUTION]).parse_args(
        [
            "--gemm-block-m",
            "64",
            "128",
            "--gemm-block-n",
            "64",
            "--gemm-block-k",
            "32",
        ]
    )
    assert len(get_operator("gemm").configurations(vars(args))) == 2


def test_operator_registry_has_one_uniform_interface() -> None:
    assert tuple(operator.name for operator in OPERATORS) == OPERATOR_NAMES
    assert len(set(OPERATOR_NAMES)) == len(OPERATOR_NAMES)
    assert all(isinstance(operator, OperatorSpec) for operator in OPERATORS)


def test_every_workload_builds_an_auto_scheduled_prim_func() -> None:
    args = vars(
        make_parser(list(OPERATORS)).parse_args(
            [
                "--gemm-m",
                "256",
                "--gemm-n",
                "256",
                "--gemm-k",
                "256",
                "--conv-batch",
                "1",
                "--conv-channels",
                "32",
                "--conv-height",
                "8",
                "--conv-width",
                "8",
                "--conv-filters",
                "32",
                "--fa3-heads",
                "1",
                "--fa3-seq-q",
                "256",
                "--fa3-seq-kv",
                "256",
                "--fa3-dim",
                "64",
                "--gqa-heads",
                "8",
                "--gqa-groups",
                "4",
                "--gqa-seq",
                "256",
                "--gqa-dim",
                "64",
                "--gemm-fp8-m",
                "256",
                "--gemm-fp8-n",
                "256",
                "--gemm-fp8-k",
                "256",
                "--mha-bwd-heads",
                "4",
                "--mha-bwd-seq",
                "128",
                "--mha-bwd-dim",
                "64",
                "--gqa-bwd-heads",
                "8",
                "--gqa-bwd-groups",
                "4",
                "--gqa-bwd-seq",
                "128",
                "--gqa-bwd-dim",
                "64",
                "--mamba-heads",
                "4",
                "--mamba-seq",
                "256",
                "--mamba-chunk",
                "64",
                "--mamba-dim",
                "64",
                "--mamba-dstate",
                "64",
                "--mla-heads",
                "64",
                "--mla-seq",
                "128",
                "--mla-dim",
                "64",
                "--mla-pe-dim",
                "32",
            ]
        )
    )
    for name in OPERATOR_NAMES:
        operator = get_operator(name)
        workload = operator.build(args, operator.configurations(args)[0])
        assert isinstance(workload, SearchWorkload)
        assert isinstance(workload.prim_func, tirx.PrimFunc)
        assert isinstance(workload.out_idx, tuple)
        auto_schedule = []

        def visit(node) -> None:
            if isinstance(node, tirx.For):
                value = node.annotations.get("tl.wsp.auto_schedule")
                if value is not None:
                    auto_schedule.append(int(value))

        post_order_visit(workload.prim_func.body, visit)
        assert auto_schedule == [1]
        assert workload.total_flops > 0


def test_operator_top30_is_sorted_and_contains_artifact_paths(tmp_path) -> None:
    config_dir = tmp_path / "m64_n64_k32"
    candidates = config_dir / "candidates"
    candidates.mkdir(parents=True)
    results = []
    for index in range(35):
        schedule_file = f"{config_dir.name}/candidates/schedule_{index:05d}.json"
        (tmp_path / schedule_file).write_text(
            json.dumps({"identity": index}), encoding="utf-8"
        )
        results.append(
            {
                "operator": "gemm",
                "tile": {"block_m": 64, "block_n": 64, "block_k": 32},
                "config_directory": config_dir.name,
                "schedule_file": schedule_file,
                "source_file": (
                    f"{config_dir.name}/sources/schedule_{index:05d}.cu"
                ),
                "schedule_index": index,
                "latency_ms": 35 - index,
                "tflops": float(index),
            }
        )

    _write_top30(tmp_path, GEMM, results, [])
    top = json.loads((tmp_path / "top30.json").read_text())["top_results"]
    assert len(top) == 30
    assert top[0]["tflops"] == 34.0
    assert top[0]["schedule_file"] == (
        "m64_n64_k32/candidates/schedule_00034.json"
    )
    assert top[0]["source_file"].endswith("sources/schedule_00034.cu")


def test_abnormal_worker_exit_keeps_checkpointed_schedule(tmp_path) -> None:
    config_dir = tmp_path / "m64_n64_k32"
    segment = config_dir / "segment"
    candidate = segment / "candidates" / "schedule_00007.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text('{"identity": 7}', encoding="utf-8")

    _record_process_failure(
        config_dir,
        {
            "schedule_index": 7,
            "phase": "benchmark",
            "schedule_file": "segment/candidates/schedule_00007.json",
        },
        kind="worker_exit",
        message="worker exited with status -11",
        extra={"returncode": -11},
    )

    saved = config_dir / "failures" / "worker_exit_schedule_00007.json"
    detail = config_dir / "failures" / "worker_exit_schedule_00007.error.json"
    assert json.loads(saved.read_text()) == {"identity": 7}
    assert json.loads(detail.read_text())["returncode"] == -11


def test_keyboard_interrupt_terminates_active_worker(
    tmp_path, monkeypatch
) -> None:
    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

    process = FakeProcess()
    terminated = []
    args = SimpleNamespace(
        max_schedules_per_config=1,
        warmup=0,
        rep=1,
        seed=0,
        rtol=0.01,
        atol=0.01,
        arch="sm_90a",
        execution_backend="cython",
        execution_timeout=30.0,
        compile_timeout=120.0,
        kill_grace=2.0,
    )

    monkeypatch.setattr(tune_run.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(
        tune_run,
        "_read_json",
        lambda path: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        tune_run,
        "_terminate",
        lambda current, grace: terminated.append((current, grace)),
    )

    with pytest.raises(KeyboardInterrupt):
        tune_run._supervise_config(
            args,
            GEMM,
            {"block_m": 64, "block_n": 64, "block_k": 32},
            tmp_path,
            1,
        )

    assert terminated == [(process, 2.0)]


def test_worker_converts_output_indices_to_tilelang_list(
    tmp_path, monkeypatch
) -> None:
    captured = {}
    workload = SimpleNamespace(
        prim_func=object(),
        out_idx=(3,),
        total_flops=1,
        reference_program=None,
        input_tensors=None,
        pass_configs={},
    )
    operator = SimpleNamespace(build=lambda options, config: workload)
    job = {
        "operator_module": "unused",
        "operator_name": "fa3",
        "options": {},
        "config": {"block_m": 128, "block_n": 96},
        "seed": 0,
        "arch": "sm_90a",
        "output_directory": str(tmp_path),
        "candidate_count": 1,
        "warmup": 0,
        "rep": 1,
        "start_schedule_index": 0,
        "max_schedules": 1,
        "rtol": 0.01,
        "atol": 0.01,
        "execution_backend": "cython",
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")

    monkeypatch.setattr(tune_run, "_load_operator", lambda *args: operator)
    monkeypatch.setattr(
        tune_run, "_load_schedule_candidates", lambda *args: (object(),)
    )
    monkeypatch.setattr(tune_run, "_cuda_target", lambda arch: arch)
    monkeypatch.setattr(tune_run.torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(
        tune_run,
        "search_schedules",
        lambda *args, **kwargs: captured.update(kwargs),
    )

    tune_run._run_worker(job_path)

    assert captured["out_idx"] == [3]
    assert isinstance(captured["out_idx"], list)
    assert len(captured["schedule_candidates"]) == 1
    assert captured["warmup"] == 0
    assert captured["rep"] == 1


def test_run_prints_candidate_count_before_starting_worker(
    tmp_path, monkeypatch, capsys
) -> None:
    args = make_parser([GEMM]).parse_args(
        [
            "--output",
            str(tmp_path),
            "--gemm-block-m",
            "64",
            "--gemm-block-n",
            "64",
            "--gemm-block-k",
            "32",
            "--max-schedules-per-config",
            "7",
        ]
    )
    events = []

    def count(*_args):
        events.append("count")
        return 42

    def supervise(*_args):
        events.append("supervise")
        return (
            [
                {
                    "schedule_index": 0,
                    "latency_ms": 1.25,
                    "tflops": 10.0,
                    "schedule_file": "candidates/schedule_00000.json",
                    "source_file": "sources/schedule_00000.cu",
                }
            ],
            {
                "enumerated_schedules": 42,
                "examined_schedules": 1,
                "successful_schedules": 1,
                "failed_schedules": 0,
                "timeout_schedules": 0,
                "crashed_schedules": 0,
            },
        )

    monkeypatch.setattr(tune_run, "_prepare_schedule_candidates", count)
    monkeypatch.setattr(tune_run, "_supervise_config", supervise)

    tune_run.run([GEMM], args)

    assert events == ["count", "supervise"]
    assert (
        "Generated 42 schedule candidates; searching 7."
        in capsys.readouterr().out
    )
    top = json.loads((tmp_path / "gemm" / "top30.json").read_text())
    assert top["top_results"][0]["schedule_file"] == (
        "m64_n64_k32/candidates/schedule_00000.json"
    )
    assert top["top_results"][0]["source_file"] == (
        "m64_n64_k32/sources/schedule_00000.cu"
    )


def test_run_skips_a_tile_when_every_requested_schedule_was_examined(
    tmp_path, monkeypatch, capsys
) -> None:
    args = make_parser([GEMM]).parse_args(
        [
            "--output",
            str(tmp_path),
            "--gemm-block-m",
            "64",
            "--gemm-block-n",
            "64",
            "--gemm-block-k",
            "32",
            "--max-schedules-per-config",
            "2",
        ]
    )
    config_dir = tmp_path / "gemm" / "m64_n64_k32"
    candidate_dir = config_dir / "candidates"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "manifest.json").write_text(
        json.dumps(
            {
                "operator": "gemm",
                "tile": {"block_m": 64, "block_n": 64, "block_k": 32},
                "schedule_count": 42,
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "schedule_index": 0,
                "latency_ms": 1.25,
                "tflops": 10.0,
                "schedule_file": "candidates/schedule_00000.json",
                "source_file": "sources/schedule_00000.cu",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "failures.jsonl").write_text(
        json.dumps({"schedule_index": 1, "error_type": "timeout"}) + "\n",
        encoding="utf-8",
    )

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("completed tile should have been skipped")

    monkeypatch.setattr(tune_run, "_prepare_schedule_candidates", should_not_run)
    monkeypatch.setattr(tune_run, "_supervise_config", should_not_run)

    tune_run.run([GEMM], args)

    assert "Skipping completed tile: examined 2 schedules." in capsys.readouterr().out
    top = json.loads((tmp_path / "gemm" / "top30.json").read_text())
    assert top["configurations"][0]["examined_schedules"] == 2
    assert top["configurations"][0]["timeout_schedules"] == 1
    assert top["top_results"][0]["schedule_file"] == (
        "m64_n64_k32/candidates/schedule_00000.json"
    )
