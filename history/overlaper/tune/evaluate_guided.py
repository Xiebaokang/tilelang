"""Evaluate guided enumeration against already measured tuning results."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from history.overlaper.candidates import (
    GuidedSearchConfig,
    Schedule,
    _stage_group_order_equivalence_key,
    enumerate_guided_schedules,
    load_schedule_json,
)
from history.overlaper.parse import extract_dataflow_graph
from history.overlaper.tune.operators import get_operator
from history.overlaper.tune.run import _cuda_target


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _historical_identity(graph, schedule: Schedule) -> tuple[Any, ...]:
    """Match schedules across synchronization/shared-memory refactors."""

    allocation = schedule.warp_allocation
    return (
        _stage_group_order_equivalence_key(
            graph,
            schedule.stages_by_region,
            schedule.groups,
            schedule.orders,
        ),
        tuple(sorted(schedule.buffer_versions.items())),
        tuple(
            (item.group_id, item.warp_count) for item in allocation.groups
        ),
        allocation.register_counts,
    )


def evaluate_tile(
    operator,
    options: dict[str, Any],
    tile_dir: Path,
    policy: GuidedSearchConfig,
    arch: str,
    shared_capacity: int,
) -> dict[str, Any] | None:
    manifest_path = tile_dir / "candidates" / "manifest.json"
    results_path = tile_dir / "results.jsonl"
    if not manifest_path.exists() or not results_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tile = manifest["tile"]
    workload = operator.build(options, tile)
    graph = extract_dataflow_graph(
        workload.prim_func,
        target=_cuda_target(arch, shared_capacity),
    )
    guided = tuple(enumerate_guided_schedules(graph, config=policy))
    guided_keys = {_historical_identity(graph, item) for item in guided}

    matched_files = {
        path.name
        for path in (tile_dir / "candidates").glob("schedule_*.json")
        if _historical_identity(graph, load_schedule_json(path)) in guided_keys
    }
    measured = _read_jsonl(results_path)
    ranked = sorted(measured, key=lambda item: item["tflops"], reverse=True)
    selected = [
        item
        for item in measured
        if Path(item["schedule_file"]).name in matched_files
    ]
    if not ranked or not selected:
        return None
    selected_indices = {item["schedule_index"] for item in selected}
    top30 = {item["schedule_index"] for item in ranked[:30]}

    def precision(fraction: float) -> float:
        cutoff = ranked[min(len(ranked) - 1, int(len(ranked) * fraction))][
            "tflops"
        ]
        return sum(item["tflops"] >= cutoff for item in selected) / len(selected)

    historical_count = int(manifest.get("schedule_count", len(matched_files)))
    return {
        "operator": operator.name,
        "tile": tile,
        "historical_candidates": historical_count,
        "guided_candidates": len(guided),
        "historical_matches": len(matched_files),
        "measured_matches": len(selected),
        "compression_ratio": len(guided) / historical_count,
        "best_tflops_ratio": max(item["tflops"] for item in selected)
        / ranked[0]["tflops"],
        "top30_recall": len(top30 & selected_indices) / min(30, len(ranked)),
        "top_quartile_precision": precision(0.25),
        "top_half_precision": precision(0.5),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--operators", nargs="+", default=["fa3", "gqa"])
    parser.add_argument("--arch", default="sm_90a")
    parser.add_argument("--shared-capacity", type=int, default=253952)
    parser.add_argument("--groups-per-count", type=int, default=16)
    parser.add_argument("--structures", type=int, default=2048)
    parser.add_argument("--schedules", type=int, default=128)
    for name in ("fa3", "gqa"):
        get_operator(name).add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    policy = GuidedSearchConfig(
        groups_per_count=args.groups_per_count,
        structures=args.structures,
        schedules=args.schedules,
    )
    options = vars(args)
    reports = []
    for name in args.operators:
        operator = get_operator(name)
        operator_dir = args.results / name
        tile_dirs = sorted(
            path for path in operator_dir.iterdir() if path.is_dir()
        )
        for tile_dir in tile_dirs:
            report = evaluate_tile(
                operator,
                options,
                tile_dir,
                policy,
                args.arch,
                args.shared_capacity,
            )
            if report is not None:
                reports.append(report)
                print(json.dumps(report, sort_keys=True), flush=True)
    summary = {
        "tile_count": len(reports),
        "historical_candidates": sum(
            item["historical_candidates"] for item in reports
        ),
        "guided_candidates": sum(item["guided_candidates"] for item in reports),
        "mean_best_tflops_ratio": (
            sum(item["best_tflops_ratio"] for item in reports) / len(reports)
            if reports
            else 0.0
        ),
        "mean_top_quartile_precision": (
            sum(item["top_quartile_precision"] for item in reports)
            / len(reports)
            if reports
            else 0.0
        ),
        "mean_top_half_precision": (
            sum(item["top_half_precision"] for item in reports) / len(reports)
            if reports
            else 0.0
        ),
    }
    payload = {"policy": asdict(policy), "summary": summary, "tiles": reports}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
