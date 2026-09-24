"""Tests for the OverlapPlan search entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

from OverlapPlaner.structure import SearchBudget
from OverlapPlaner.tune.dynamic import load_candidates
from OverlapPlaner.tune.operators.gemm import OPERATOR as GEMM
from OverlapPlaner.tune.operators.workloads import OperatorSpec
from OverlapPlaner.tune.run import (
    _evaluation_fingerprint,
    _supervise_config,
    _workload_fingerprint,
    generate_plans,
    run,
)


def test_generate_plans_writes_replay_manifest(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plans"
    generate_plans(
        GEMM,
        plan_dir,
        {"block_m": 128, "block_n": 128, "block_k": 64},
        {"gemm_m": 256, "gemm_n": 256, "gemm_k": 256},
        budget=SearchBudget(max_groups=1, max_structures=1),
    )
    manifest = json.loads(
        (plan_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["operator"] == "gemm"
    assert manifest["tile"] == {
        "block_m": 128,
        "block_n": 128,
        "block_k": 64,
    }
    assert manifest["schedule_count"] >= 1
    assert manifest["stage_beam"] == 64
    assert (plan_dir / "schedule_00000.json").is_file()
    feature_rows = [
        json.loads(line)
        for line in (plan_dir / "features.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(feature_rows) == manifest["schedule_count"]
    assert len(feature_rows[0]["fingerprint"]) == 64
    assert len(feature_rows[0]["features"]) > 16
    workload_fingerprint = _workload_fingerprint(
        GEMM,
        {"block_m": 128, "block_n": 128, "block_k": 64},
        {"gemm_m": 256, "gemm_n": 256, "gemm_k": 256},
    )
    candidate = load_candidates(
        plan_dir, fingerprint_salt=workload_fingerprint
    )[0]
    assert candidate.fingerprint == _evaluation_fingerprint(
        candidate.path, workload_fingerprint
    )


def test_workload_fingerprint_includes_problem_size() -> None:
    tile = {"block_m": 128, "block_n": 128, "block_k": 64}
    small = _workload_fingerprint(
        GEMM,
        tile,
        {"gemm_m": 1024, "gemm_n": 1024, "gemm_k": 1024},
    )
    large = _workload_fingerprint(
        GEMM,
        tile,
        {"gemm_m": 4096, "gemm_n": 4096, "gemm_k": 4096},
    )
    assert small != large


def test_run_writes_results_layout(tmp_path: Path, monkeypatch) -> None:
    tile = {"block_m": 128, "block_n": 128, "block_k": 64}

    def fake_generate(
        operator, plan_dir: Path, _tile, _options, *, budget=None
    ) -> None:
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "schedule_00000.json").write_text("{}", encoding="utf-8")

    observed = {}

    def fake_supervise(
        operator,
        search_tile,
        options,
        config_dir,
        warmup,
        rep,
        compile_timeout,
        execution_timeout,
        kill_grace,
    ):
        observed.update(
            warmup=warmup,
            rep=rep,
            compile_timeout=compile_timeout,
            execution_timeout=execution_timeout,
            kill_grace=kill_grace,
        )
        plan_dir = config_dir / "candidates"
        source_dir = config_dir / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        source = source_dir / "schedule_00000.cu"
        source.write_text("kernel", encoding="utf-8")
        result = {
            "schedule_index": 0,
            "schedule_file": str(plan_dir / "schedule_00000.json"),
            "source_file": str(source),
            "latency_ms": 1.25,
            "tflops": 8.0,
            "tile": dict(search_tile),
        }
        disk_result = {
            **result,
            "schedule_file": "candidates/schedule_00000.json",
            "source_file": "sources/schedule_00000.cu",
        }
        (config_dir / "results.jsonl").write_text(
            json.dumps(disk_result) + "\n", encoding="utf-8"
        )
        return {
            "operator": operator.name,
            "tile": dict(search_tile),
            "enumerated": 1,
            "examined": 1,
            "successful": [result],
            "failures": [],
        }

    monkeypatch.setattr("OverlapPlaner.tune.run.SEARCH_OPERATORS", [GEMM])
    monkeypatch.setattr(
        OperatorSpec,
        "configurations",
        lambda self, _options: (dict(tile),),
    )
    monkeypatch.setattr("OverlapPlaner.tune.run.generate_plans", fake_generate)

    def fake_native(
        operator,
        search_tile,
        options,
        config_dir,
        warmup,
        rep,
        compile_timeout,
        execution_timeout,
        kill_grace,
    ):
        return {
            "result": {
                "backend": "native",
                "tile": dict(search_tile),
                "source_file": str(config_dir / "native" / "source.cu"),
                "latency_ms": 2.5,
                "tflops": 4.0,
            },
            "failure": None,
        }

    monkeypatch.setattr(
        "OverlapPlaner.tune.run._supervise_native",
        fake_native,
    )
    monkeypatch.setattr(
        "OverlapPlaner.tune.run._supervise_config", fake_supervise
    )

    run(
        tmp_path,
        warmup=3,
        rep=7,
        compile_timeout=11,
        execution_timeout=13,
        kill_grace=2,
        search_mode="exhaustive",
    )
    assert observed == {
        "warmup": 3,
        "rep": 7,
        "compile_timeout": 11,
        "execution_timeout": 13,
        "kill_grace": 2,
    }

    config_dir = tmp_path / "gemm" / "m128_n128_k64"
    assert (config_dir / "candidates" / "schedule_00000.json").is_file()
    assert (config_dir / "sources" / "schedule_00000.cu").is_file()
    rows = [
        json.loads(line)
        for line in (config_dir / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert rows[0]["schedule_file"] == "candidates/schedule_00000.json"
    assert rows[0]["source_file"] == "sources/schedule_00000.cu"
    top30 = json.loads((tmp_path / "gemm" / "top30.json").read_text(encoding="utf-8"))
    assert top30["operator"] == "gemm"
    assert top30["top_results"][0]["rank"] == 1
    assert top30["top_results"][0]["schedule_file"] == (
        "m128_n128_k64/candidates/schedule_00000.json"
    )
    assert top30["top_results"][0]["speedup_vs_native"] == 2.0
    assert top30["configurations"][0]["native"]["latency_ms"] == 2.5
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary[0]["operator"] == "gemm"
    assert summary[0]["top30_file"] == "gemm/top30.json"


def test_supervisor_resumes_and_records_worker_exit(
    tmp_path: Path, monkeypatch
) -> None:
    config_dir = tmp_path / "gemm" / "m128_n128_k64"
    candidates = config_dir / "candidates"
    candidates.mkdir(parents=True)
    for index in range(2):
        (candidates / f"schedule_{index:05d}.json").write_text(
            "{}", encoding="utf-8"
        )
    tile = {"block_m": 128, "block_n": 128, "block_k": 64}
    workload_fingerprint = _workload_fingerprint(GEMM, tile, {})
    first_fingerprint = _evaluation_fingerprint(
        candidates / "schedule_00000.json", workload_fingerprint
    )
    (config_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "schedule_index": 0,
                "schedule_file": "candidates/schedule_00000.json",
                "source_file": "sources/schedule_00000.cu",
                "latency_ms": 1.0,
                "tflops": 2.0,
                "candidate_fingerprint": first_fingerprint,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_dir = config_dir / "sources"
    source_dir.mkdir()
    current_source = source_dir / "schedule_00000.cu"
    current_source.write_text("current", encoding="utf-8")
    stale_source = source_dir / "schedule_00099.cu"
    stale_source.write_text("stale", encoding="utf-8")

    launched_jobs = []

    class FailedWorker:
        returncode = 1
        pid = 123

        def __init__(self, command, **_kwargs):
            launched_jobs.append(
                json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            )

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        "OverlapPlaner.tune.run.subprocess.Popen", FailedWorker
    )
    payload = _supervise_config(
        GEMM,
        tile,
        {},
        config_dir,
        warmup=3,
        rep=7,
        compile_timeout=11,
        execution_timeout=13,
        kill_grace=2,
    )
    assert len(launched_jobs) == 1
    assert launched_jobs[0]["skip_schedule_indices"] == [0]
    assert launched_jobs[0]["schedule_indices"] == [1]
    assert launched_jobs[0]["warmup"] == 3
    assert launched_jobs[0]["rep"] == 7
    assert len(payload["successful"]) == 1
    assert len(payload["failures"]) == 1
    assert payload["failures"][0]["schedule_index"] == 1
    assert payload["failures"][0]["error_type"] == "worker_exit"
    assert current_source.read_text(encoding="utf-8") == "current"
    assert not stale_source.exists()
