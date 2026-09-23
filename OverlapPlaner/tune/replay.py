"""Compile, validate, and benchmark one searched OverlapPlan schedule."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from OverlapPlaner.tune.operators import OPERATORS, OperatorSpec, get_operator
from OverlapPlaner.tune.search import evaluate


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _manifest_from_dir(config_dir: Path) -> dict[str, Any] | None:
    for manifest_path in (
        config_dir / "manifest.json",
        config_dir / "candidates" / "manifest.json",
    ):
        manifest = _read_object(manifest_path)
        if manifest is not None:
            return manifest
    return None


def infer_replay_context(
    schedule_path: Path,
) -> tuple[OperatorSpec, dict[str, int], dict[str, Any]]:
    """Find the operator, tile, and saved options around one schedule JSON."""

    resolved = schedule_path.resolve()
    for config_dir in (resolved.parent, *resolved.parents):
        manifest = _manifest_from_dir(config_dir)
        if manifest is None:
            continue
        operator_name = manifest.get("operator")
        tile = manifest.get("tile")
        if not isinstance(operator_name, str) or not isinstance(tile, dict):
            continue
        if any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for key, value in tile.items()
        ):
            raise ValueError(f"invalid tile in {config_dir}")

        options = manifest.get("options")
        if not isinstance(options, dict):
            job = _read_object(config_dir / "job.json")
            options = None if job is None else job.get("options")
        return (
            get_operator(operator_name),
            dict(tile),
            dict(options) if isinstance(options, dict) else {},
        )
    raise ValueError(
        "cannot infer replay context: no manifest.json was found next to or "
        "above the schedule"
    )


def _parse_override(text: str) -> tuple[str, Any]:
    key, separator, raw_value = text.partition("=")
    if not separator or not key.isidentifier():
        raise argparse.ArgumentTypeError(
            "option overrides must use NAME=JSON_VALUE"
        )
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key, value


def replay(
    schedule_path: Path,
    warmup: int = 100,
    rep: int = 400,
    *,
    default_options: Mapping[str, Any] | None = None,
    option_overrides: Mapping[str, Any] | None = None,
) -> tuple[float, float]:
    """Replay one searched schedule and return ``(latency_ms, tflops)``."""

    operator, tile, saved_options = infer_replay_context(schedule_path)
    options = dict(default_options or {})
    options.update(saved_options)
    options.update(option_overrides or {})
    print(f"schedule={schedule_path}")
    print(f"operator={operator.name}")
    print(f"tile={tile}")
    print("compiling...", flush=True)
    result = evaluate(
        operator,
        tile,
        options,
        schedule_path,
        warmup=warmup,
        rep=rep,
    )
    print("correctness=PASS", flush=True)
    print(f"latency_ms={result['latency_ms']:.6f}")
    print(f"tflops={result['tflops']:.3f}")
    return result["latency_ms"], result["tflops"]


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile, validate, and benchmark one searched OverlapPlan "
            "schedule JSON. auto_overlap=True is required; there is no "
            "default plan."
        )
    )
    parser.add_argument("schedule_json", type=Path)
    parser.add_argument(
        "--warmup", type=int, default=100, help="kernel warmup iterations"
    )
    parser.add_argument(
        "--rep", type=int, default=400, help="measured kernel iterations"
    )
    parser.add_argument(
        "--set",
        dest="option_overrides",
        action="append",
        default=[],
        type=_parse_override,
        metavar="NAME=JSON_VALUE",
        help="override one saved operator option; may be repeated",
    )
    for operator in OPERATORS:
        operator.add_arguments(parser)
    return parser


def main() -> None:
    args = _make_parser().parse_args()
    replay(
        args.schedule_json,
        args.warmup,
        args.rep,
        default_options=vars(args),
        option_overrides=dict(args.option_overrides),
    )


if __name__ == "__main__":
    main()
