"""Compile, validate, and benchmark one saved Overlaper schedule."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tilelang
import torch

from history.overlaper.candidates import load_schedule_json
from history.overlaper.integration import use_schedule_planner
from history.overlaper.tune.operators import OPERATORS, OperatorSpec, get_operator
from history.overlaper.tune.run import _cuda_target
from tilelang.carver.arch.driver import get_max_dynamic_shared_size_bytes


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _infer_replay_context(
    schedule_path: Path,
) -> tuple[OperatorSpec, dict[str, int], dict[str, Any]]:
    """Find the operator, tile, and saved options around one schedule JSON."""

    resolved = schedule_path.resolve()
    for config_dir in resolved.parents:
        manifest_path = config_dir / "candidates" / "manifest.json"
        manifest = _read_object(manifest_path)
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
            raise ValueError(f"invalid tile in {manifest_path}")

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
        "cannot infer replay context: no candidates/manifest.json was found "
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
    warmup: int,
    rep: int,
    *,
    default_options: Mapping[str, Any] | None = None,
    option_overrides: Mapping[str, Any] | None = None,
    seed: int = 0,
    rtol: float = 0.01,
    atol: float = 0.01,
    arch: str = "sm_90a",
    max_shared_memory_per_block: int | None = None,
    execution_backend: str = "cython",
) -> tuple[float, float]:
    """Replay one schedule and return ``(latency_ms, tflops)``."""

    schedule_path = schedule_path.resolve()
    if not schedule_path.is_file():
        raise FileNotFoundError(f"schedule JSON does not exist: {schedule_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")

    major, minor = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError(
            f"Overlaper replay requires a Hopper GPU, found sm_{major}{minor}"
        )

    if max_shared_memory_per_block is None:
        max_shared_memory_per_block = get_max_dynamic_shared_size_bytes()
    if max_shared_memory_per_block is None or max_shared_memory_per_block <= 0:
        raise RuntimeError("failed to query a positive GPU shared-memory limit")

    operator, tile, saved_options = _infer_replay_context(schedule_path)
    options = dict(default_options or {})
    options.update(saved_options)
    options.update(option_overrides or {})
    schedule = load_schedule_json(schedule_path)
    workload = operator.build(options, tile)
    target = _cuda_target(arch, max_shared_memory_per_block)
    prim_func = workload.prim_func.with_attr(
        "tl.program_schedule.request", f"overlaper-replay-{operator.name}-v2"
    )

    print(f"schedule={schedule_path}")
    print(f"operator={operator.name}")
    print(f"tile={tile}")
    print("compiling...", flush=True)
    torch.manual_seed(seed)
    with use_schedule_planner(lambda *_: schedule), target:
        compiled = tilelang.compile(
            prim_func,
            out_idx=list(workload.out_idx),
            target=target,
            execution_backend=execution_backend,
            pass_configs=dict(workload.pass_configs),
        )

    profiler = compiled.get_profiler()
    inputs = (
        None
        if workload.input_tensors is None
        else list(workload.input_tensors)
    )
    print("validating...", flush=True)
    profiler.assert_allclose(
        workload.reference_program,
        input_tensors=inputs,
        rtol=rtol,
        atol=atol,
    )
    print("correctness=PASS", flush=True)

    latency_ms = float(
        profiler.do_bench_iterations(
            warmup=warmup, rep=rep, input_tensors=inputs
        )
    )
    tflops = workload.total_flops / latency_ms * 1e-9
    print(f"latency_ms={latency_ms:.6f}")
    print(f"tflops={tflops:.3f}")
    return latency_ms, tflops


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile, validate, and benchmark one schedule JSON from any "
            "registered Overlaper operator."
        )
    )
    parser.add_argument("schedule_json", type=Path)
    parser.add_argument(
        "--warmup", type=int, default=100, help="kernel warmup iterations"
    )
    parser.add_argument(
        "--rep", type=int, default=400, help="measured kernel iterations"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--arch", default="sm_90a")
    parser.add_argument("--max-shared-memory-per-block", type=int)
    parser.add_argument("--execution-backend", default="cython")
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
    if args.warmup < 0 or args.rep < 1:
        raise SystemExit("warmup must be non-negative and rep must be positive")
    replay(
        args.schedule_json,
        args.warmup,
        args.rep,
        default_options=vars(args),
        option_overrides=dict(args.option_overrides),
        seed=args.seed,
        rtol=args.rtol,
        atol=args.atol,
        arch=args.arch,
        max_shared_memory_per_block=args.max_shared_memory_per_block,
        execution_backend=args.execution_backend,
    )


if __name__ == "__main__":
    main()

"""
PYTHONPATH="$PWD/3rdparty/tvm/python:$PWD" \
TVM_LIBRARY_PATH="$PWD/build/lib" \
python -m overlaper.tune.replay \
results/gqa/m128_n128/candidates/schedule_00778.json
"""
