"""Search entrypoint: evaluate native OverlapPlan JSON files, then rank them.

Use ``--operators`` to choose a subset, or omit it to search every registered
operator in :data:`SEARCH_OPERATORS`. All kernel results are written under a
single ``--output`` directory:

    <output>/<operator>/<tile>/candidates/
    <output>/<operator>/<tile>/sources/
    <output>/<operator>/<tile>/results.jsonl
    <output>/<operator>/<tile>/native/result.json
    <output>/<operator>/<tile>/native/source.cu
    <output>/<operator>/top30.json
    <output>/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from OverlapPlaner.arch import HOPPER
from OverlapPlaner.contract import (
    enumerate_overlap_plans,
    layout_reduced_prim_func,
    to_overlap_plan,
)
from OverlapPlaner.facts import extract_fact_graph
from OverlapPlaner.serialization import plan_from_dict, plan_to_dict, save_plan_json
from OverlapPlaner.structure import SearchBudget, Structure, connected_components
from OverlapPlaner.structure.order import build_program_orders
from OverlapPlaner.structure.stage import (
    enumerate_program_stages,
    enumerate_stage_assignments,
)
from OverlapPlaner.structure.sync import build_synchronizations
from OverlapPlaner.structure.version import analyze_buffer_versions
from OverlapPlaner.tune.dynamic import (
    Candidate,
    DynamicSearchPolicy,
    classified_plan_features,
    file_fingerprint,
    load_candidates,
    order_bucket,
    physical_plan_features,
    plan_features,
    plan_fingerprint,
    producer_copy_count,
    should_stop,
    workload_plan_fingerprint,
)
from OverlapPlaner.tune.copy_search import classified_for_plan
from OverlapPlaner.tune.operators import OPERATOR_NAMES, get_operator
from OverlapPlaner.tune.operators.convolution import OPERATOR as CONVOLUTION
from OverlapPlaner.tune.operators.dequant_gemm_fp4 import OPERATOR as DEQUANT_GEMM_FP4
from OverlapPlaner.tune.operators.fa3 import OPERATOR as FA3
from OverlapPlaner.tune.operators.fused_moe import OPERATOR as FUSED_MOE
from OverlapPlaner.tune.operators.gemm import OPERATOR as GEMM
from OverlapPlaner.tune.operators.gemm_fp8 import OPERATOR as GEMM_FP8
from OverlapPlaner.tune.operators.gqa import OPERATOR as GQA
from OverlapPlaner.tune.operators.gqa_bwd import OPERATOR as GQA_BWD
from OverlapPlaner.tune.operators.gdn_chunk_delta_bwd import OPERATOR as GDN_CHUNK_DELTA_BWD
from OverlapPlaner.tune.operators.gdn_chunk_o_bwd import OPERATOR as GDN_CHUNK_O_BWD
from OverlapPlaner.tune.operators.kda_chunk_bwd_intra import OPERATOR as KDA_CHUNK_BWD_INTRA
from OverlapPlaner.tune.operators.kda_wy_fast_bwd import OPERATOR as KDA_WY_FAST_BWD
from OverlapPlaner.tune.operators.linear_attn_fwd import (
    OPERATOR as LINEAR_ATTN_FWD,
)
from OverlapPlaner.tune.operators.mamba_chunk_scan import (
    OPERATOR as MAMBA_CHUNK_SCAN,
)
from OverlapPlaner.tune.operators.mamba_chunk_state import (
    OPERATOR as MAMBA_CHUNK_STATE,
)
from OverlapPlaner.tune.operators.mla import OPERATOR as MLA
from OverlapPlaner.tune.operators.mha_bwd import OPERATOR as MHA_BWD
from OverlapPlaner.tune.operators.workloads import OperatorSpec
from OverlapPlaner.tune.joint_search import adjacent_joint_moves, realize_joint_move
from OverlapPlaner.tune.order_search import adjacent_order_swaps, realize_order_swap
from OverlapPlaner.tune.search import (
    evaluate_native,
    list_plan_files,
    rank_results,
    search,
)


SEARCH_OPERATORS: list[OperatorSpec] = [
    FA3,
    MLA,
    DEQUANT_GEMM_FP4,
    GDN_CHUNK_O_BWD,
    GDN_CHUNK_DELTA_BWD,
    KDA_WY_FAST_BWD,
    KDA_CHUNK_BWD_INTRA,
    FUSED_MOE,
    GQA_BWD,
    MHA_BWD,
    LINEAR_ATTN_FWD,
    MAMBA_CHUNK_SCAN,
    MAMBA_CHUNK_STATE,
    GEMM,
    GQA,
    CONVOLUTION,
    GEMM_FP8,
]

# Bump whenever generated-code semantics or correctness validation changes.
# Results are measurements of a plan *and* its implementation, so a plan-only
# fingerprint must not reuse rows produced by an older lowering.
_EVALUATION_CACHE_VERSION = "overlap-plan-lowering-v6-gdn-kda-contracts"


def _workload_fingerprint(
    operator: OperatorSpec,
    tile: Mapping[str, int],
    options: Mapping[str, Any],
) -> str:
    return plan_fingerprint(
        {
            "evaluation": _EVALUATION_CACHE_VERSION,
            "operator": operator.name,
            "tile": dict(tile),
            "options": dict(options),
        }
    )


def _evaluation_fingerprint(path: Path, workload_fingerprint: str) -> str:
    return workload_plan_fingerprint(
        file_fingerprint(path), workload_fingerprint
    )


def generate_plans(
    operator: OperatorSpec,
    plan_dir: Path,
    tile: Mapping[str, int],
    options: Mapping[str, Any],
    *,
    budget: SearchBudget | None = None,
    replay_order_mutations: bool = False,
    joint_seed_budget: int = 0,
) -> None:
    """Enumerate native OverlapPlan JSON candidates for one tile."""

    budget = budget or SearchBudget()
    workload = operator.build(options, dict(tile))
    reduced = layout_reduced_prim_func(workload.prim_func)
    classified = HOPPER.classify(extract_fact_graph(reduced))
    if joint_seed_budget and not any(
        classified.traits_for(node.node_id).async_completion
        and classified.graph.region_for_id(node.region_id).kind.value == "pipeline"
        and any(
            classified.graph.buffer_for_id(buffer_id).scope.startswith("shared")
            for buffer_id in node.writes
        )
        for node in classified.graph.nodes
    ):
        budget = replace(
            budget, max_structures=budget.max_structures + joint_seed_budget
        )
        joint_seed_budget = 0
    plan_dir.mkdir(parents=True, exist_ok=True)
    for path in plan_dir.iterdir():
        if (
            path.is_file()
            and path.suffix == ".json"
            and path.name.startswith("schedule_")
            and not path.name.endswith(".error.json")
        ):
            path.unlink()
    feature_path = plan_dir / "features.jsonl"
    feature_path.unlink(missing_ok=True)
    count = 0
    known: dict[str, dict[str, Any]] = {}
    for plan in enumerate_overlap_plans(
        workload.prim_func, budget=budget
    ):
        payload = plan_to_dict(plan)
        _write_candidate(plan_dir, count, payload, classified)
        known[plan_fingerprint(payload)] = payload
        count += 1
    if joint_seed_budget:
        for payload in _independent_producer_seeds(classified, budget, joint_seed_budget):
            fingerprint = plan_fingerprint(payload)
            if fingerprint in known:
                continue
            _write_candidate(plan_dir, count, payload, classified)
            known[fingerprint] = payload
            count += 1
    base_count = count
    mutations = (
        _read_jsonl(plan_dir / "order_mutations.jsonl")
        if replay_order_mutations else []
    )
    for mutation in mutations:
        parent = known.get(mutation.get("parent_fingerprint"))
        swap = mutation.get("swap")
        if parent is None or not isinstance(swap, list) or len(swap) != 4:
            continue
        proposal = realize_order_swap(
            classified_for_plan(classified, parent), parent, tuple(swap)
        )
        if proposal is None:
            continue
        fingerprint = plan_fingerprint(proposal)
        if fingerprint != mutation.get("fingerprint") or fingerprint in known:
            continue
        _write_candidate(plan_dir, count, proposal, classified)
        known[fingerprint] = proposal
        count += 1
    if replay_order_mutations:
        for mutation in _read_jsonl(plan_dir / "joint_mutations.jsonl"):
            parent = known.get(mutation.get("parent_fingerprint"))
            move = mutation.get("move")
            if parent is None or not isinstance(move, list) or len(move) != 5:
                continue
            proposal = realize_joint_move(classified, parent, tuple(move))
            if proposal is None:
                continue
            fingerprint = plan_fingerprint(proposal)
            if fingerprint != mutation.get("fingerprint") or fingerprint in known:
                continue
            _write_candidate(plan_dir, count, proposal, classified)
            known[fingerprint] = proposal
            count += 1
    _write_json(
        plan_dir / "manifest.json",
        {
            "operator": operator.name,
            "tile": dict(tile),
            "options": dict(options),
            "schedule_count": count,
            "base_schedule_count": base_count,
            "stage_beam": budget.stage_beam,
            "candidate_features": "features.jsonl",
        },
    )


def _independent_producer_seeds(classified, budget: SearchBudget, limit: int):
    """Seed independent producers and global/shared memory groups across stages."""

    graph = classified.graph
    producer_ids = {
        node.node_id
        for node in graph.nodes
        if classified.traits_for(node.node_id).async_completion
        and graph.region_for_id(node.region_id).kind.value == "pipeline"
        and any(
            graph.buffer_for_id(buffer_id).scope.startswith("shared")
            for buffer_id in node.writes
        )
    }
    if not producer_ids or limit < 1:
        return
    memory_ids = {
        node.node_id
        for node in graph.nodes
        if node.kind.value == "copy"
        and any(
            graph.buffer_for_id(buffer_id).scope in ("", "global")
            for buffer_id in (*node.reads, *node.writes)
        )
        and any(
            graph.buffer_for_id(buffer_id).scope.startswith("shared")
            for buffer_id in (*node.reads, *node.writes)
        )
    }
    components = connected_components(classified)
    group_maps = []
    for selected_ids in (producer_ids, memory_ids):
        selected_nodes = {
            node_id
            for component in components
            if selected_ids.intersection(component)
            for node_id in component
        }
        if not selected_nodes or len(selected_nodes) == len(graph.nodes):
            continue
        groups = {
            node.node_id: int(node.node_id in selected_nodes)
            for node in graph.nodes
        }
        if groups not in group_maps:
            group_maps.append(groups)
    if not group_maps:
        return
    pipeline_regions = [
        region for region in graph.regions if region.kind.value == "pipeline"
    ]
    def stage_maps():
        if (
            len(pipeline_regions) == 1
            and len(graph.nodes_for_region(pipeline_regions[0].region_id)) <= 12
        ):
            region_id = pipeline_regions[0].region_id
            for assignment in enumerate_stage_assignments(
                classified, region_id, budget.max_stages
            ):
                yield {region_id: assignment}
        else:
            yield from enumerate_program_stages(classified, budget)

    emitted = 0
    for group_index, groups in enumerate(group_maps):
        # Reserve capacity for every template; unused capacity rolls forward.
        group_limit = limit - (len(group_maps) - group_index - 1) * (limit // len(group_maps))
        for stages in stage_maps():
            try:
                orders = build_program_orders(classified, stages, groups)
                versions = analyze_buffer_versions(graph, stages, groups, orders)
                sync = build_synchronizations(classified, stages, groups, orders, versions)
                structure = Structure(stages, groups, orders, versions, sync)
                physical = next(HOPPER.realize(classified, structure), None)
                if physical is None:
                    continue
                yield plan_to_dict(to_overlap_plan(classified, physical))
                emitted += 1
                if emitted >= group_limit:
                    break
            except ValueError:
                continue


def _write_candidate(
    plan_dir: Path,
    index: int,
    payload: dict[str, Any],
    classified,
) -> None:
    schedule_path = plan_dir / f"schedule_{index:05d}.json"
    save_plan_json(schedule_path, plan_from_dict(payload))
    candidate_classified = classified_for_plan(classified, payload)
    features = (
        *plan_features(payload),
        *classified_plan_features(payload, candidate_classified),
    )
    _append_jsonl(
        plan_dir / "features.jsonl",
        {
            "schedule_index": index,
            "schedule_file": schedule_path.name,
            "fingerprint": plan_fingerprint(payload),
            "group_count": int(features[0]),
            "stage_depth": int(features[1]),
            "producer_copies": producer_copy_count(payload, candidate_classified),
            "order_bucket": order_bucket(features),
            "features": features,
        },
    )


def _config_name(config: Mapping[str, int]) -> str:
    aliases = {"block_m": "m", "block_n": "n", "block_k": "k"}
    return "_".join(
        f"{aliases.get(key, key)}{value}" for key, value in config.items()
    )


def _default_options(operator: OperatorSpec) -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    operator.add_arguments(parser)
    return vars(parser.parse_args([]))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
        output.flush()


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Atomically replace a JSONL file with normalized rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except FileNotFoundError:
        return []


def _relative(path: str | Path | None, root: Path) -> str | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _write_top(
    operator_dir: Path,
    operator: OperatorSpec,
    results: list[Mapping[str, Any]],
    configurations: list[Mapping[str, Any]],
    rank_n: int = 30,
) -> None:
    ranked = rank_results(results, rank_n)
    _write_json(
        operator_dir / "top30.json",
        {
            "operator": operator.name,
            "description": operator.description,
            "configurations": list(configurations),
            "top_results": ranked,
        },
    )
    print(f"\n=== {operator.name} Top-{len(ranked)} ===", flush=True)
    for item in ranked:
        print(
            f"#{item['rank']:02d} tile={item['tile']} "
            f"schedule={item['schedule_index']} "
            f"latency={item['latency_ms']:.4f} ms "
            f"tflops={item['tflops']:.2f}",
            flush=True,
        )


def _candidate_index(path: Path, fallback: int) -> int:
    suffix = path.stem.removeprefix("schedule_")
    return int(suffix) if suffix.isdigit() else fallback


def _terminate(process: subprocess.Popen, grace: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_worker(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    operator = get_operator(job["operator"])
    config_dir = Path(job["config_dir"])
    state_path = Path(job["state_path"])
    mode = job.get("mode", "search")

    def write_state(state: Mapping[str, Any]) -> None:
        _write_json(state_path, state)

    if mode == "native":
        try:
            result = evaluate_native(
                operator,
                job["tile"],
                job["options"],
                job["source_path"],
                warmup=job["warmup"],
                rep=job["rep"],
                phase_callback=lambda phase: write_state({"phase": phase}),
            )
            result["source_file"] = _relative(
                result.get("source_file"), config_dir
            )
            _write_json(Path(job["result_path"]), result)
            write_state({"phase": "completed"})
        except Exception as error:
            failure = {
                "phase": (_read_json(state_path) or {}).get(
                    "phase", "unknown"
                ),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write_json(Path(job["error_path"]), failure)
            write_state({"phase": "failed"})
            raise
        return
    if mode != "search":
        raise ValueError(f"unknown worker mode: {mode}")

    results_path = Path(job["results_path"])
    failures_path = Path(job["failures_path"])

    def write_result(kind: str, row: Mapping[str, Any]) -> None:
        destination = results_path if kind == "successful" else failures_path
        normalized = dict(row)
        schedule_index = normalized.get("schedule_index")
        normalized["candidate_fingerprint"] = job[
            "candidate_fingerprints"
        ].get(str(schedule_index))
        for key in ("schedule_file", "source_file"):
            normalized[key] = _relative(normalized.get(key), config_dir)
        _append_jsonl(destination, normalized)

    search(
        operator,
        job["tile"],
        job["options"],
        job["plan_dir"],
        job["source_dir"],
        warmup=job["warmup"],
        rep=job["rep"],
        skip_schedule_indices=job["skip_schedule_indices"],
        schedule_indices=job.get("schedule_indices"),
        state_callback=write_state,
        result_callback=write_result,
    )


def _supervise_native(
    operator: OperatorSpec,
    tile: Mapping[str, int],
    options: Mapping[str, Any],
    config_dir: Path,
    warmup: int,
    rep: int,
    compile_timeout: float,
    execution_timeout: float,
    kill_grace: float,
) -> dict[str, Any]:
    """Evaluate the native baseline in an isolated worker process."""

    native_dir = config_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    state_path = native_dir / "state.json"
    job_path = native_dir / "job.json"
    result_path = native_dir / "result.json"
    error_path = native_dir / "error.json"
    source_path = native_dir / "source.cu"
    for stale in (state_path, job_path, result_path, error_path, source_path):
        stale.unlink(missing_ok=True)
    _write_json(state_path, {"phase": "startup"})
    _write_json(
        job_path,
        {
            "mode": "native",
            "operator": operator.name,
            "config_dir": str(config_dir.resolve()),
            "tile": dict(tile),
            "options": dict(options),
            "source_path": str(source_path.resolve()),
            "warmup": warmup,
            "rep": rep,
            "state_path": str(state_path.resolve()),
            "result_path": str(result_path.resolve()),
            "error_path": str(error_path.resolve()),
        },
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "OverlapPlaner.tune.run",
            "--worker-job",
            str(job_path.resolve()),
        ],
        start_new_session=True,
    )
    phase = "startup"
    phase_started = time.monotonic()
    timed_out = False
    try:
        while process.poll() is None:
            state = _read_json(state_path) or {}
            updated_phase = str(state.get("phase", phase))
            if updated_phase != phase:
                phase = updated_phase
                phase_started = time.monotonic()
            timeout = (
                execution_timeout
                if phase in {"correctness_validation", "benchmark"}
                else compile_timeout
            )
            if time.monotonic() - phase_started > timeout:
                timed_out = True
                _terminate(process, kill_grace)
                _write_json(
                    error_path,
                    {
                        "phase": phase,
                        "error_type": "timeout",
                        "error": f"phase exceeded {timeout:g} seconds",
                    },
                )
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        _terminate(process, kill_grace)
        raise

    result = _read_json(result_path)
    failure = _read_json(error_path)
    if not timed_out and process.returncode != 0 and failure is None:
        failure = {
            "phase": phase,
            "error_type": "worker_exit",
            "error": f"worker exited with status {process.returncode}",
        }
        _write_json(error_path, failure)
    if result is None and failure is None:
        failure = {
            "phase": phase,
            "error_type": "worker_exit",
            "error": "native worker exited without recording a result",
        }
        _write_json(error_path, failure)
    if result is not None and result.get("source_file") is not None:
        result["source_file"] = str(
            (config_dir / result["source_file"]).resolve()
        )
    state_path.unlink(missing_ok=True)
    job_path.unlink(missing_ok=True)
    return {"result": result, "failure": failure}


def _record_process_failure(
    config_dir: Path,
    state: Mapping[str, Any],
    error_type: str,
    message: str,
    candidate_fingerprint: str,
) -> None:
    failure = {
        "schedule_index": int(state["schedule_index"]),
        "schedule_file": _relative(state.get("schedule_file"), config_dir),
        "source_file": None,
        "phase": state.get("phase", "unknown"),
        "error_type": error_type,
        "error": message,
        "candidate_fingerprint": candidate_fingerprint,
    }
    _append_jsonl(config_dir / "failures.jsonl", failure)
    _write_json(
        config_dir
        / "failures"
        / f"{error_type}_schedule_{failure['schedule_index']:05d}.error.json",
        failure,
    )


def _supervise_config(
    operator: OperatorSpec,
    tile: Mapping[str, int],
    options: Mapping[str, Any],
    config_dir: Path,
    warmup: int,
    rep: int,
    compile_timeout: float,
    execution_timeout: float,
    kill_grace: float,
    schedule_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    plan_dir = config_dir / "candidates"
    source_dir = config_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "failures").mkdir(parents=True, exist_ok=True)
    plan_files = list_plan_files(plan_dir)
    indexed_plans = {
        _candidate_index(path, fallback): path
        for fallback, path in enumerate(plan_files)
    }
    if len(indexed_plans) != len(plan_files):
        raise ValueError("candidate schedule indices must be unique")
    workload_fingerprint = _workload_fingerprint(operator, tile, options)
    candidate_fingerprints = {
        index: _evaluation_fingerprint(path, workload_fingerprint)
        for index, path in indexed_plans.items()
    }

    def is_current(item: Mapping[str, Any]) -> bool:
        index = item.get("schedule_index")
        return (
            isinstance(index, int)
            and index in candidate_fingerprints
            and item.get("candidate_fingerprint")
            == candidate_fingerprints[index]
        )

    # Candidate indices are reused whenever plans are regenerated.  Remove
    # rows whose content fingerprint no longer matches so old correctness
    # failures and timings do not remain visible as if they described the
    # current candidate pool.
    for records_path in (
        config_dir / "results.jsonl",
        config_dir / "failures.jsonl",
    ):
        recorded = _read_jsonl(records_path)
        current_records = [row for row in recorded if is_current(row)]
        if len(current_records) != len(recorded):
            _write_jsonl(records_path, current_records)
    for error_path in (config_dir / "failures").glob("*.error.json"):
        recorded_error = _read_json(error_path)
        if recorded_error is None or not is_current(recorded_error):
            error_path.unlink(missing_ok=True)
    recorded_sources = set()
    for record_name in ("results.jsonl", "failures.jsonl"):
        for row in _read_jsonl(config_dir / record_name):
            if not is_current(row) or not row.get("source_file"):
                continue
            recorded = Path(row["source_file"])
            if not recorded.is_absolute():
                recorded = config_dir / recorded
            recorded_sources.add(recorded.resolve())
    for source_path in source_dir.glob("schedule_*.cu"):
        if source_path.resolve() not in recorded_sources:
            source_path.unlink()

    target_indices = set(indexed_plans)
    if schedule_indices is not None:
        target_indices &= set(schedule_indices)

    state_path = config_dir / "candidate_state.json"
    job_path = config_dir / "job.json"
    while True:
        successful = _read_jsonl(config_dir / "results.jsonl")
        failures = _read_jsonl(config_dir / "failures.jsonl")
        completed = {
            int(item["schedule_index"])
            for item in (*successful, *failures)
            if is_current(item)
        }
        missing = sorted(target_indices - completed)
        if not missing:
            break

        initial_index = missing[0]
        state: dict[str, Any] = {
            "schedule_index": initial_index,
            "phase": "startup",
            "schedule_file": str(indexed_plans[initial_index]),
        }
        _write_json(state_path, state)
        _write_json(
            job_path,
            {
                "operator": operator.name,
                "config_dir": str(config_dir.resolve()),
                "tile": dict(tile),
                "options": dict(options),
                "plan_dir": str(plan_dir.resolve()),
                "source_dir": str(source_dir.resolve()),
                "warmup": warmup,
                "rep": rep,
                "skip_schedule_indices": sorted(completed),
                # A fatal CUDA launch error poisons the process context.  Run
                # exactly one candidate in each worker so a bad kernel cannot
                # turn every following candidate into a spurious failure.
                "schedule_indices": [initial_index],
                "candidate_fingerprints": {
                    str(index): fingerprint
                    for index, fingerprint in candidate_fingerprints.items()
                },
                "state_path": str(state_path.resolve()),
                "results_path": str(
                    (config_dir / "results.jsonl").resolve()
                ),
                "failures_path": str(
                    (config_dir / "failures.jsonl").resolve()
                ),
            },
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "OverlapPlaner.tune.run",
                "--worker-job",
                str(job_path.resolve()),
            ],
            start_new_session=True,
        )
        active_phase: tuple[Any, Any] | None = None
        phase_started = time.monotonic()
        timed_out = False
        try:
            while process.poll() is None:
                updated = _read_json(state_path)
                if updated is not None:
                    state = updated
                    phase = (state.get("schedule_index"), state.get("phase"))
                    if phase != active_phase:
                        active_phase = phase
                        phase_started = time.monotonic()
                timeout = (
                    execution_timeout
                    if state.get("phase")
                    in {"correctness_validation", "benchmark"}
                    else compile_timeout
                )
                if time.monotonic() - phase_started > timeout:
                    timed_out = True
                    _terminate(process, kill_grace)
                    _record_process_failure(
                        config_dir,
                        state,
                        "timeout",
                        f"phase exceeded {timeout:g} seconds",
                        candidate_fingerprints[int(state["schedule_index"])],
                    )
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            _terminate(process, kill_grace)
            raise

        if not timed_out and process.returncode != 0:
            _record_process_failure(
                config_dir,
                state,
                "worker_exit",
                f"worker exited with status {process.returncode}",
                candidate_fingerprints[int(state["schedule_index"])],
            )
        elif not timed_out:
            updated_successes = _read_jsonl(config_dir / "results.jsonl")
            updated_failures = _read_jsonl(config_dir / "failures.jsonl")
            updated_completed = {
                int(item["schedule_index"])
                for item in (*updated_successes, *updated_failures)
                if is_current(item)
            }
            if updated_completed == completed:
                _record_process_failure(
                    config_dir,
                    state,
                    "worker_exit",
                    "worker exited without recording candidate progress",
                    candidate_fingerprints[int(state["schedule_index"])],
                )

    state_path.unlink(missing_ok=True)
    job_path.unlink(missing_ok=True)
    successful_by_index = {
        int(row["schedule_index"]): row
        for row in _read_jsonl(config_dir / "results.jsonl")
        if is_current(row)
        and int(row["schedule_index"]) in indexed_plans
    }
    failures_by_index = {
        int(row["schedule_index"]): row
        for row in _read_jsonl(config_dir / "failures.jsonl")
        if is_current(row)
        and int(row["schedule_index"]) in indexed_plans
        and int(row["schedule_index"]) not in successful_by_index
    }
    successful = [
        successful_by_index[index] for index in sorted(successful_by_index)
    ]
    failures = [
        failures_by_index[index] for index in sorted(failures_by_index)
    ]
    for row in (*successful, *failures):
        for key in ("schedule_file", "source_file"):
            if row.get(key) is not None:
                row[key] = str((config_dir / row[key]).resolve())
    return {
        "operator": operator.name,
        "tile": dict(tile),
        "enumerated": len(plan_files),
        "examined": len(successful) + len(failures),
        "successful": successful,
        "failures": failures,
    }


def _expand_order_candidates(
    plan_dir: Path,
    classified,
    measured_latency: Mapping[int, float],
    limit: int,
    *,
    max_extra: int,
) -> list[int]:
    """Expand good measured orders by legal adjacent swaps, within a cap."""

    manifest = _read_json(plan_dir / "manifest.json") or {}
    next_index = int(manifest.get("schedule_count", 0))
    base_count = int(manifest.get("base_schedule_count", next_index))
    remaining = min(limit, max(0, max_extra - (next_index - base_count)))
    if remaining < 1:
        return []
    candidates = load_candidates(plan_dir)
    by_index = {candidate.index: candidate for candidate in candidates}
    known = {candidate.fingerprint for candidate in candidates}
    ranked = sorted(measured_latency, key=lambda index: measured_latency[index])
    parents: list[int] = []
    covered_buckets = set()
    for index in ranked:
        candidate = by_index.get(index)
        if candidate is not None and candidate.bucket not in covered_buckets:
            parents.append(index)
            covered_buckets.add(candidate.bucket)
    parents.extend(index for index in ranked if index not in parents)
    proposals: dict[int, tuple[dict[str, Any], str, tuple[int, int, int, int]]] = {}
    proposal_candidates: list[Candidate] = []
    for parent_index in parents[:8]:
        parent = by_index[parent_index]
        payload = json.loads(parent.path.read_text(encoding="utf-8"))
        swaps = sorted(
            adjacent_order_swaps(classified, payload),
            key=lambda swap: (
                classified.graph.region_for_id(swap[0]).kind.value != "pipeline",
                swap[0],
                swap[1],
                swap[2],
            ),
        )
        for swap in swaps:
            proposal = realize_order_swap(
                classified_for_plan(classified, payload), payload, swap
            )
            if proposal is None:
                continue
            fingerprint = plan_fingerprint(proposal)
            if fingerprint in known:
                continue
            temporary_index = next_index + len(proposal_candidates)
            proposal_classified = classified_for_plan(classified, proposal)
            features = (
                *plan_features(proposal),
                *classified_plan_features(proposal, proposal_classified),
                *physical_plan_features(proposal),
            )
            proposal_candidates.append(
                Candidate(
                    index=temporary_index,
                    path=plan_dir / f"schedule_{temporary_index:05d}.json",
                    fingerprint=fingerprint,
                    bucket=(int(features[0]), int(features[1])),
                    features=features,
                    producer_copies=producer_copy_count(
                        proposal, proposal_classified
                    ),
                )
            )
            proposals[temporary_index] = (
                proposal,
                plan_fingerprint(payload),
                swap,
            )
            known.add(fingerprint)
            if len(proposal_candidates) >= remaining * 4:
                break
        if len(proposal_candidates) >= remaining * 4:
            break
    if not proposals:
        return []
    # Fit the same measurement-driven acquisition model used for the base
    # pool, but rank only newly realized order moves against one another.
    proposal_policy = DynamicSearchPolicy((*candidates, *proposal_candidates))
    existing_unmeasured = {
        item.index for item in candidates
    } - set(measured_latency)
    chosen = proposal_policy.next_batch(
        dict(measured_latency), existing_unmeasured, remaining
    )
    added: list[int] = []
    for temporary_index in chosen:
        proposal, parent_fingerprint, swap = proposals[temporary_index]
        fingerprint = plan_fingerprint(proposal)
        _write_candidate(plan_dir, next_index, proposal, classified)
        _append_jsonl(
            plan_dir / "order_mutations.jsonl",
            {
                "parent_fingerprint": parent_fingerprint,
                "swap": list(swap),
                "fingerprint": fingerprint,
            },
        )
        added.append(next_index)
        next_index += 1
    if added:
        manifest["schedule_count"] = next_index
        _write_json(plan_dir / "manifest.json", manifest)
    return added


def _expand_joint_candidates(
    plan_dir: Path,
    classified,
    measured_latency: Mapping[int, float],
    limit: int,
    *,
    max_extra: int,
) -> list[int]:
    """Explore measured schedules across placements, buffers, and copies."""

    manifest = _read_json(plan_dir / "manifest.json") or {}
    next_index = int(manifest.get("schedule_count", 0))
    base_count = int(manifest.get("base_schedule_count", next_index))
    remaining = min(limit, max(0, max_extra - (next_index - base_count)))
    if remaining < 1 or not measured_latency:
        return []
    candidates = load_candidates(plan_dir)
    by_index = {candidate.index: candidate for candidate in candidates}
    known = {candidate.fingerprint for candidate in candidates}
    ranked = sorted(measured_latency, key=lambda index: measured_latency[index])
    parents: list[int] = []
    covered = set()
    for index in ranked:
        candidate = by_index.get(index)
        if candidate is None:
            continue
        bucket = DynamicSearchPolicy._coverage_bucket(candidate)
        if bucket not in covered:
            parents.append(index)
            covered.add(bucket)
    parents.extend(
        index for index in ranked if index in by_index and index not in parents
    )

    proposals: dict[int, tuple[dict[str, Any], str, tuple]] = {}
    proposal_candidates: list[Candidate] = []
    per_dimension = {
        "stage": 0, "group": 0, "order": 0, "version": 0, "copy": 0
    }
    for parent_index in parents[:8]:
        parent = by_index[parent_index]
        payload = json.loads(parent.path.read_text(encoding="utf-8"))
        for move in adjacent_joint_moves(classified, payload):
            dimension = move[0]
            if per_dimension[dimension] >= max(remaining * 12, 24):
                continue
            proposal = realize_joint_move(classified, payload, move)
            if proposal is None:
                continue
            fingerprint = plan_fingerprint(proposal)
            if fingerprint in known:
                continue
            temporary_index = next_index + len(proposal_candidates)
            proposal_classified = classified_for_plan(classified, proposal)
            features = (
                *plan_features(proposal),
                *classified_plan_features(proposal, proposal_classified),
                *physical_plan_features(proposal),
            )
            proposal_candidates.append(
                Candidate(
                    index=temporary_index,
                    path=plan_dir / f"schedule_{temporary_index:05d}.json",
                    fingerprint=fingerprint,
                    bucket=(int(features[0]), int(features[1])),
                    features=features,
                    producer_copies=producer_copy_count(
                        proposal, proposal_classified
                    ),
                    order_bucket=order_bucket(features),
                )
            )
            proposals[temporary_index] = (proposal, plan_fingerprint(payload), move)
            per_dimension[dimension] += 1
            known.add(fingerprint)
    if not proposal_candidates:
        return []

    policy = DynamicSearchPolicy((*candidates, *proposal_candidates))
    existing_unmeasured = {item.index for item in candidates} - set(measured_latency)
    ranked_proposals = policy.next_batch(
        dict(measured_latency), existing_unmeasured, len(proposal_candidates)
    )
    # Rank all legal stage/group/order/version/copy moves together. Diversity
    # is already represented by the uncertainty and coverage terms in the
    # acquisition policy; forcing a dimension rotation can spend a scarce GPU
    # measurement on a move the feedback model predicts to be poor.
    chosen = ranked_proposals

    added: list[int] = []
    for temporary_index in chosen[:remaining]:
        proposal, parent_fingerprint, move = proposals[temporary_index]
        fingerprint = plan_fingerprint(proposal)
        _write_candidate(plan_dir, next_index, proposal, classified)
        _append_jsonl(
            plan_dir / "joint_mutations.jsonl",
            {
                "parent_fingerprint": parent_fingerprint,
                "move": list(move),
                "fingerprint": fingerprint,
            },
        )
        added.append(next_index)
        next_index += 1
    manifest["schedule_count"] = next_index
    _write_json(plan_dir / "manifest.json", manifest)
    return added


def _supervise_dynamic_config(
    operator: OperatorSpec,
    tile: Mapping[str, int],
    options: Mapping[str, Any],
    config_dir: Path,
    warmup: int,
    rep: int,
    compile_timeout: float,
    execution_timeout: float,
    kill_grace: float,
    *,
    evaluation_budget: int,
    initial_samples: int,
    batch_size: int,
    patience: int,
    minimum_improvement: float,
) -> dict[str, Any]:
    """Select candidates in batches using feedback from measured latency."""

    plan_dir = config_dir / "candidates"
    workload = operator.build(options, dict(tile))
    reduced = layout_reduced_prim_func(workload.prim_func)
    classified = HOPPER.classify(extract_fact_graph(reduced))
    workload_fingerprint = _workload_fingerprint(operator, tile, options)
    candidates = load_candidates(
        plan_dir,
        fingerprint_salt=workload_fingerprint,
        classified=classified,
    )
    policy = DynamicSearchPolicy(candidates)
    pool_fingerprint = plan_fingerprint(
        {"candidate_fingerprints": [item.fingerprint for item in candidates]}
    )
    previous_state = _read_json(config_dir / "dynamic_state.json") or {}
    if previous_state.get("candidate_pool_fingerprint") == pool_fingerprint:
        best_history = [
            float(value)
            for value in previous_state.get("best_latency_history_ms", [])
        ]
        batches = [
            [int(index) for index in batch]
            for batch in previous_state.get("batches", [])
        ][-evaluation_budget:]
        best_history = best_history[-evaluation_budget:]
        batch_diagnostics = list(
            previous_state.get("batch_diagnostics", [])
        )[-evaluation_budget:]
    else:
        best_history = []
        batches = []
        batch_diagnostics = []
    stopped_early = False
    while True:
        successful = _read_jsonl(config_dir / "results.jsonl")
        failures = _read_jsonl(config_dir / "failures.jsonl")
        fingerprints = {item.index: item.fingerprint for item in candidates}

        def current(row: Mapping[str, Any]) -> bool:
            index = row.get("schedule_index")
            return (
                isinstance(index, int)
                and fingerprints.get(index)
                == row.get("candidate_fingerprint")
            )

        measured_latency = {
            int(row["schedule_index"]): float(row["latency_ms"])
            for row in successful
            if current(row)
        }
        unavailable = {
            int(row["schedule_index"])
            for row in failures
            if current(row)
        }
        attempted = len(measured_latency) + len(unavailable)
        remaining_budget = evaluation_budget - attempted
        if remaining_budget <= 0:
            break
        fresh: list[int] = []
        if attempted >= initial_samples and measured_latency:
            fresh = _expand_joint_candidates(
                plan_dir,
                classified,
                measured_latency,
                min(1, remaining_budget),
                max_extra=evaluation_budget * 2,
            )
            if fresh:
                candidates = load_candidates(
                    plan_dir,
                    fingerprint_salt=workload_fingerprint,
                    classified=classified,
                )
                policy = DynamicSearchPolicy(candidates)
                pool_fingerprint = plan_fingerprint(
                    {"candidate_fingerprints": [item.fingerprint for item in candidates]}
                )
        if attempted < initial_samples:
            selected = policy.initial(
                set(measured_latency) | unavailable,
                min(initial_samples - attempted, remaining_budget),
            )
            selection_roles = {index: "coverage" for index in selected}
        else:
            feedback_selected = policy.next_batch(
                measured_latency,
                unavailable | set(fresh),
                min(batch_size, remaining_budget) - len(fresh),
            )
            selected = fresh + feedback_selected
            selection_roles = {
                **{index: "local_mutation" for index in fresh},
                **policy.last_selection_roles,
            }
        if not selected:
            break
        batches.append(selected)
        payload = _supervise_config(
            operator,
            tile,
            options,
            config_dir,
            warmup,
            rep,
            compile_timeout,
            execution_timeout,
            kill_grace,
            schedule_indices=tuple(selected),
        )
        batch_successes = {
            int(row["schedule_index"]): float(row["latency_ms"])
            for row in payload["successful"]
            if int(row["schedule_index"]) in selected
        }
        batch_failures = {
            int(row["schedule_index"])
            for row in payload["failures"]
            if int(row["schedule_index"]) in selected
        }
        newly_completed = (
            set(batch_successes) | batch_failures
        ) - (set(measured_latency) | unavailable)
        if not newly_completed:
            raise RuntimeError(
                "dynamic search made no progress: none of the selected "
                f"candidates {selected} produced a current result or failure"
            )
        previous = (
            best_history[-1]
            if best_history
            else min(measured_latency.values(), default=math.inf)
        )
        best_history.append(min([previous, *batch_successes.values()]))
        batch_latencies = list(batch_successes.values())
        batch_diagnostics.append(
            {
                "selected": selected,
                "selection_roles": {
                    str(index): selection_roles.get(index, "feedback")
                    for index in selected
                },
                "successful": len(batch_successes),
                "failed": len(batch_failures),
                "failure_rate": len(batch_failures) / len(selected),
                "batch_min_latency_ms": (
                    min(batch_latencies) if batch_latencies else None
                ),
                "batch_median_latency_ms": (
                    statistics.median(batch_latencies)
                    if batch_latencies
                    else None
                ),
                "cumulative_best_latency_ms": best_history[-1],
            }
        )
        _write_json(
            config_dir / "dynamic_state.json",
            {
                "evaluation_budget": evaluation_budget,
                "candidate_pool_fingerprint": pool_fingerprint,
                "attempted": attempted + len(selected),
                "batches": batches,
                "best_latency_history_ms": best_history,
                "batch_diagnostics": batch_diagnostics,
            },
        )
        if attempted + len(selected) >= initial_samples and should_stop(
            best_history,
            patience=patience,
            minimum_improvement=minimum_improvement,
        ):
            stopped_early = True
            break

    fingerprints = {item.index: item.fingerprint for item in candidates}
    final_successes = _read_jsonl(config_dir / "results.jsonl")
    final_failures = _read_jsonl(config_dir / "failures.jsonl")
    final_completed = {
        int(row["schedule_index"])
        for row in (*final_successes, *final_failures)
        if isinstance(row.get("schedule_index"), int)
        and fingerprints.get(int(row["schedule_index"]))
        == row.get("candidate_fingerprint")
    }
    _write_json(
        config_dir / "dynamic_state.json",
        {
            "evaluation_budget": evaluation_budget,
            "candidate_pool_fingerprint": pool_fingerprint,
            "attempted": len(final_completed),
            "batches": batches,
            "best_latency_history_ms": best_history,
            "batch_diagnostics": batch_diagnostics,
        },
    )

    payload = _supervise_config(
        operator,
        tile,
        options,
        config_dir,
        warmup,
        rep,
        compile_timeout,
        execution_timeout,
        kill_grace,
        schedule_indices=(),
    )
    payload["search_policy"] = "dynamic"
    payload["candidate_pool"] = len(candidates)
    payload["evaluation_budget"] = evaluation_budget
    payload["batches"] = batches
    payload["stopped_early"] = stopped_early
    return payload


def run(
    output: str | Path,
    warmup: int = 100,
    rep: int = 400,
    compile_timeout: float = 30.0,
    execution_timeout: float = 15.0,
    rank_n: int = 30,
    kill_grace: float = 2.0,
    search_mode: str = "dynamic",
    candidate_pool: int = 2048,
    evaluation_budget: int = 48,
    initial_samples: int = 12,
    batch_size: int = 4,
    patience: int = 3,
    minimum_improvement: float = 0.01,
    stage_beam: int = 64,
    operators: Sequence[OperatorSpec] | None = None,
) -> None:
    """Search selected operators into ``output``."""

    selected_operators = tuple(SEARCH_OPERATORS if operators is None else operators)
    if not selected_operators:
        raise ValueError("at least one search operator is required")
    if search_mode not in {"dynamic", "exhaustive"}:
        raise ValueError("search_mode must be dynamic or exhaustive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    search_summary: list[dict[str, Any]] = []
    for operator in selected_operators:
        operator_dir = output / operator.name
        operator_dir.mkdir(parents=True, exist_ok=True)
        options = _default_options(operator)
        results: list[dict[str, Any]] = []
        config_summaries: list[dict[str, Any]] = []
        for tile in operator.configurations(options):
            config_name = _config_name(tile)
            config_dir = operator_dir / config_name
            plan_dir = config_dir / "candidates"
            print(
                f"\n=== operator={operator.name} tile={dict(tile)} ===",
                flush=True,
            )
            print("evaluating TileLang native baseline", flush=True)
            try:
                native_payload = _supervise_native(
                    operator,
                    tile,
                    options,
                    config_dir,
                    warmup,
                    rep,
                    compile_timeout,
                    execution_timeout,
                    kill_grace,
                )
            except Exception as error:
                native_payload = {
                    "result": None,
                    "failure": {
                        "phase": "startup",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                }
            native_result = native_payload["result"]
            native_failure = native_payload["failure"]
            native_summary = None
            if native_result is not None:
                native_summary = {
                    **native_result,
                    "source_file": _relative(
                        native_result.get("source_file"), operator_dir
                    ),
                    "result_file": _relative(
                        config_dir / "native" / "result.json", operator_dir
                    ),
                }
            if native_result is not None:
                print(
                    "native "
                    f"latency={native_result['latency_ms']:.4f} ms "
                    f"tflops={native_result['tflops']:.3f}",
                    flush=True,
                )
            else:
                print(
                    "native failed: "
                    f"{native_failure['error_type']}: "
                    f"{native_failure['error']}",
                    flush=True,
                )
            base_pool = candidate_pool
            seed_pool = 0
            if search_mode == "dynamic" and candidate_pool >= 2:
                seed_pool = candidate_pool // 2
                base_pool -= seed_pool
            generation_options: dict[str, Any] = {
                "budget": SearchBudget(
                    max_structures=base_pool,
                    stage_beam=stage_beam,
                ),
            }
            if search_mode == "dynamic":
                generation_options["replay_order_mutations"] = True
                generation_options["joint_seed_budget"] = seed_pool
            generate_plans(
                operator,
                plan_dir,
                tile,
                options,
                **generation_options,
            )
            try:
                if search_mode == "dynamic":
                    payload = _supervise_dynamic_config(
                        operator,
                        tile,
                        options,
                        config_dir,
                        warmup,
                        rep,
                        compile_timeout,
                        execution_timeout,
                        kill_grace,
                        evaluation_budget=evaluation_budget,
                        initial_samples=initial_samples,
                        batch_size=batch_size,
                        patience=patience,
                        minimum_improvement=minimum_improvement,
                    )
                else:
                    payload = _supervise_config(
                        operator,
                        tile,
                        options,
                        config_dir,
                        warmup,
                        rep,
                        compile_timeout,
                        execution_timeout,
                        kill_grace,
                    )
                    payload["search_policy"] = "exhaustive"
                    payload["candidate_pool"] = payload["enumerated"]
            except Exception as error:
                config_summaries.append(
                    {
                        "tile": dict(tile),
                        "native": native_summary,
                        "native_failure": native_failure,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                print(
                    f"tile failed: {type(error).__name__}: {error}",
                    flush=True,
                )
                _write_top(operator_dir, operator, results, config_summaries, rank_n)
                continue

            found: list[dict[str, Any]] = []
            for item in payload["successful"]:
                row = {
                    "tile": dict(tile),
                    "schedule_index": item["schedule_index"],
                    "latency_ms": item["latency_ms"],
                    "tflops": item["tflops"],
                    "schedule_file": _relative(
                        item["schedule_file"], operator_dir
                    ),
                    "source_file": _relative(
                        item["source_file"], operator_dir
                    ),
                }
                if native_result is not None:
                    row["native_latency_ms"] = native_result["latency_ms"]
                    row["speedup_vs_native"] = (
                        native_result["latency_ms"] / item["latency_ms"]
                    )
                found.append(row)
            results.extend(found)
            config_summaries.append(
                {
                    "tile": dict(tile),
                    "native": native_summary,
                    "native_failure": native_failure,
                    "search_policy": payload["search_policy"],
                    "candidate_pool": payload["candidate_pool"],
                    "evaluation_budget": payload.get("evaluation_budget"),
                    "stopped_early": payload.get("stopped_early", False),
                    "batches": payload.get("batches"),
                    "enumerated_schedules": payload["enumerated"],
                    "examined_schedules": payload["examined"],
                    "successful_schedules": len(payload["successful"]),
                    "failed_schedules": len(payload["failures"]),
                }
            )
            _write_top(operator_dir, operator, results, config_summaries, rank_n)

        search_summary.append(
            {
                "operator": operator.name,
                "configurations": config_summaries,
                "top30_file": str(
                    (operator_dir / "top30.json").relative_to(output)
                ),
            }
        )
        _write_json(output / "summary.json", search_summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark TileLang native lowering and searched OverlapPlan JSON "
            "files for SEARCH_OPERATORS under --output."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="root directory for every operator's search results",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="number of warmup iterations",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=20,
        help="number of repetitions",
    )
    parser.add_argument(
        "--rank-n",
        type=int,
        default=30,
        help="number of top results to rank",
    )
    parser.add_argument("--compile-timeout", type=float, default=30.0)
    parser.add_argument("--execution-timeout", type=float, default=15.0)
    parser.add_argument("--kill-grace", type=float, default=2.0)
    parser.add_argument(
        "--search-mode",
        choices=("dynamic", "exhaustive"),
        default="dynamic",
        help="measurement-driven selection or evaluation of the whole pool",
    )
    parser.add_argument("--candidate-pool", type=int, default=4096)
    parser.add_argument("--stage-beam", type=int, default=256)
    parser.add_argument("--evaluation-budget", type=int, default=64)
    parser.add_argument("--initial-samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--minimum-improvement", type=float, default=0.01)
    parser.add_argument(
        "--operators",
        nargs="+",
        choices=OPERATOR_NAMES,
        help=(
            "operators to search; defaults to SEARCH_OPERATORS. For example: "
            "--operators gqa_bwd mha_bwd"
        ),
    )
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_job is not None:
        _run_worker(args.worker_job)
        return
    if args.warmup < 0 or args.rep < 1:
        parser.error("warmup must be non-negative and rep must be positive")
    if min(args.compile_timeout, args.execution_timeout, args.kill_grace) <= 0:
        parser.error("timeouts and kill grace must be positive")
    if min(
        args.candidate_pool,
        args.stage_beam,
        args.evaluation_budget,
        args.initial_samples,
        args.batch_size,
        args.patience,
    ) < 1:
        parser.error("dynamic search budgets must be positive")
    if args.minimum_improvement < 0:
        parser.error("minimum improvement must be non-negative")
    run(
        args.output,
        args.warmup,
        args.rep,
        args.compile_timeout,
        args.execution_timeout,
        args.rank_n,
        args.kill_grace,
        args.search_mode,
        args.candidate_pool,
        args.evaluation_budget,
        args.initial_samples,
        args.batch_size,
        args.patience,
        args.minimum_improvement,
        args.stage_beam,
        None
        if args.operators is None
        else tuple(get_operator(name) for name in args.operators),
    )


if __name__ == "__main__":
    main()
