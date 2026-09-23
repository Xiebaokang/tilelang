"""Search FA3 tile sizes and UnionWSP schedules together.

This is a standalone GPU search program, not a pytest test case.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from functools import partial
from itertools import product
from pathlib import Path

import tilelang
import torch

import history.unionwsp as unionwsp
from history.wspipeline.test.fa3_kernel import flashattn, make_cuda_target, ref_program


def make_fa3_prim_func(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    dim: int,
    is_causal: bool,
    block_m: int,
    block_n: int,
):
    return flashattn.jit_impl.get_tir(
        batch,
        heads,
        seq_q,
        seq_kv,
        dim,
        is_causal,
        block_M=block_m,
        block_N=block_n,
        threads=256,
        auto_wsp=True,
    )


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _terminate_process_group(process: subprocess.Popen, grace_seconds: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _worker_command(
    args: argparse.Namespace,
    block_m: int,
    block_n: int,
    start_schedule_index: int,
    output_directory: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--block-m",
        str(block_m),
        "--block-n",
        str(block_n),
        "--seq-q",
        str(args.seq_q),
        "--seq-kv",
        str(args.seq_kv),
        "--dim",
        str(args.dim),
        "--warmup",
        str(args.warmup),
        "--rep",
        str(args.rep),
        "--start-schedule-index",
        str(start_schedule_index),
        "--output",
        str(output_directory),
    ]
    if args.max_schedules_per_tile is not None:
        command.extend(
            ["--max-schedules-per-tile", str(args.max_schedules_per_tile)]
        )
    return command


def _run_worker(args: argparse.Namespace) -> None:
    if len(args.block_m) != 1 or len(args.block_n) != 1:
        raise ValueError("a search worker accepts exactly one tile")

    block_m, block_n = args.block_m[0], args.block_n[0]
    batch, heads = 1, 16
    target = make_cuda_target()
    torch.manual_seed(0)
    q = torch.randn(
        batch, heads, args.seq_q, args.dim, device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        batch, heads, args.seq_kv, args.dim, device="cuda", dtype=torch.float16
    )
    v = torch.randn(
        batch, heads, args.seq_kv, args.dim, device="cuda", dtype=torch.float16
    )
    total_flops = 4.0 * batch * heads * args.seq_q * args.seq_kv * args.dim
    prim_func = make_fa3_prim_func(
        batch,
        heads,
        args.seq_q,
        args.seq_kv,
        args.dim,
        False,
        block_m,
        block_n,
    )
    summary = unionwsp.search_wsp_schedules(
        prim_func,
        target=target,
        out_idx=[3],
        total_flops=total_flops,
        reference_program=partial(ref_program, is_causal=False),
        input_tensors=[q, k, v],
        output_directory=args.output,
        top_k=20,
        warmup=args.warmup,
        rep=args.rep,
        start_schedule_index=args.start_schedule_index,
        max_schedules=args.max_schedules_per_tile,
        validate_each_schedule=True,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    (args.output / "worker_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _collect_segment(
    segment_directory: Path,
    tile_directory: Path,
) -> tuple[list[dict], dict[str, int]]:
    payload = _read_json(segment_directory / "top20.json")
    if payload is None:
        return [], {
            "examined_schedules": 0,
            "successful_schedules": 0,
            "failed_schedules": 0,
        }
    segment_relative = str(segment_directory.relative_to(tile_directory))
    results = []
    for item in payload.get("top_results", []):
        result = dict(item)
        result.pop("rank", None)
        result["segment_directory"] = segment_relative
        results.append(result)
    return results, {
        "examined_schedules": int(payload.get("examined_schedules", 0)),
        "successful_schedules": int(payload.get("successful_schedules", 0)),
        "failed_schedules": int(payload.get("failed_schedules", 0)),
    }


def _record_timeout(
    tile_directory: Path,
    run_directory: Path,
    segment_directory: Path,
    state: dict,
    timeout_seconds: float,
) -> dict:
    schedule_index = int(state.get("schedule_index", -1))
    schedule_file = state.get("schedule_file")
    source_file = state.get("source_file")
    schedule_path = None if schedule_file is None else segment_directory / schedule_file
    source_path = None if source_file is None else segment_directory / source_file
    record = {
        "schedule_index": schedule_index,
        "phase": state.get("phase", "unknown"),
        "error_type": "KernelTimeout",
        "error": f"candidate exceeded the {timeout_seconds:g}-second phase timeout",
        "timeout_seconds": timeout_seconds,
        "schedule_file": (
            None
            if schedule_path is None
            else str(schedule_path.relative_to(tile_directory))
        ),
        "source_file": (
            None if source_path is None else str(source_path.relative_to(tile_directory))
        ),
    }
    if schedule_path is not None:
        record["schedule"] = _read_json(schedule_path)

    timeout_directory = run_directory / "timeouts"
    timeout_directory.mkdir(parents=True, exist_ok=True)
    detail_path = timeout_directory / f"schedule_{schedule_index:05d}.json"
    record["detail_file"] = str(detail_path.relative_to(tile_directory))
    detail_path.write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (tile_directory / "timeouts.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"\n=== Timed out schedule {schedule_index} ===", flush=True)
    print(f"phase: {record['phase']}", flush=True)
    print(f"timeout: {timeout_seconds:g} seconds", flush=True)
    print(f"detail_log: {record['detail_file']}", flush=True)
    if record["source_file"] is not None:
        print(f"generated_source: {record['source_file']}", flush=True)
    return record


def _supervise_tile(
    args: argparse.Namespace,
    block_m: int,
    block_n: int,
    tile_directory: Path,
) -> tuple[list[dict], dict]:
    run_directory = tile_directory / "segments" / f"run_{time.time_ns()}"
    run_directory.mkdir(parents=True, exist_ok=True)
    start_schedule_index = 0
    enumerated_schedule_count = 0
    timeout_count = 0
    accumulated_results = []
    counters = {
        "examined_schedules": 0,
        "successful_schedules": 0,
        "failed_schedules": 0,
    }

    while True:
        segment_directory = run_directory / f"start_{start_schedule_index:05d}"
        segment_directory.mkdir(parents=True, exist_ok=True)
        initial_state = {
            "schedule_index": start_schedule_index,
            "phase": "startup",
            "enumerated_schedule_count": 0,
            "schedule_file": None,
            "source_file": None,
        }
        (segment_directory / "candidate_state.json").write_text(
            json.dumps(initial_state, indent=2, sort_keys=True), encoding="utf-8"
        )
        process = subprocess.Popen(
            _worker_command(
                args,
                block_m,
                block_n,
                start_schedule_index,
                segment_directory,
            ),
            start_new_session=True,
        )
        active_key = None
        phase_started = time.monotonic()
        timed_out = False
        timeout_state = initial_state
        timeout_limit = args.compile_timeout

        while process.poll() is None:
            state = _read_json(segment_directory / "candidate_state.json")
            if state is not None:
                timeout_state = state
                key = (state.get("schedule_index"), state.get("phase"))
                if key != active_key:
                    active_key = key
                    phase_started = time.monotonic()
                phase = state.get("phase")
                timeout_limit = (
                    args.execution_timeout
                    if phase
                    in {
                        "profiler_initialization",
                        "correctness_validation",
                        "benchmark",
                    }
                    else args.compile_timeout
                )
            if time.monotonic() - phase_started > timeout_limit:
                if process.poll() is not None:
                    break
                timed_out = True
                _terminate_process_group(process, args.kill_grace)
                break
            time.sleep(0.2)

        segment_results, segment_counters = _collect_segment(
            segment_directory, tile_directory
        )
        accumulated_results.extend(segment_results)
        for name, value in segment_counters.items():
            counters[name] += value
        enumeration = _read_json(segment_directory / "enumeration.json")
        if enumeration is not None:
            enumerated_schedule_count = int(
                enumeration.get("enumerated_schedule_count", 0)
            )

        if timed_out:
            timeout_count += 1
            counters["examined_schedules"] += 1
            counters["failed_schedules"] += 1
            _record_timeout(
                tile_directory,
                run_directory,
                segment_directory,
                timeout_state,
                timeout_limit,
            )
            stuck_index = int(timeout_state.get("schedule_index", -1))
            if timeout_state.get("schedule_file") is None or stuck_index < 0:
                return accumulated_results, {
                    **counters,
                    "enumerated_schedules": enumerated_schedule_count,
                    "timeout_schedules": timeout_count,
                    "error_type": "UnrecoverableWorkerTimeout",
                    "error": "worker timed out before a schedule checkpoint was written",
                }
            start_schedule_index = stuck_index + 1
            search_limit = args.max_schedules_per_tile or enumerated_schedule_count
            if search_limit and start_schedule_index >= search_limit:
                break
            continue

        if process.returncode != 0:
            return accumulated_results, {
                **counters,
                "enumerated_schedules": enumerated_schedule_count,
                "timeout_schedules": timeout_count,
                "error_type": "WorkerProcessError",
                "error": f"search worker exited with status {process.returncode}",
            }
        break

    return accumulated_results, {
        **counters,
        "enumerated_schedules": enumerated_schedule_count,
        "timeout_schedules": timeout_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-m", type=int, nargs="+", default=[128, 192])
    parser.add_argument("--block-n", type=int, nargs="+", default=[96, 128, 192, 256])
    parser.add_argument("--seq-q", type=int, default=8192)
    parser.add_argument("--seq-kv", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--rep", type=int, default=40)
    parser.add_argument("--compile-timeout", type=float, default=60.0)
    parser.add_argument("--execution-timeout", type=float, default=30.0)
    parser.add_argument("--kill-grace", type=float, default=2.0)
    parser.add_argument(
        "--max-schedules-per-tile",
        type=int,
        default=None,
        help="optional smoke-test limit for each tile; omit for exhaustive search",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./fa3_tile_search"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--start-schedule-index", type=int, default=0, help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("FA3 tile search requires a Hopper CUDA GPU")
    if any(value <= 0 for value in (*args.block_m, *args.block_n)):
        raise ValueError("tile sizes must be positive")
    if args.compile_timeout <= 0 or args.execution_timeout <= 0:
        raise ValueError("timeouts must be positive")
    if args.kill_grace < 0:
        raise ValueError("kill grace cannot be negative")
    if args.worker:
        _run_worker(args)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    all_results = []
    tile_summaries = []
    for block_m, block_n in product(args.block_m, args.block_n):
        tile_name = f"m{block_m}_n{block_n}"
        tile_output = args.output / tile_name
        print(f"\n=== tile block_M={block_m}, block_N={block_n} ===")
        try:
            tile_results, tile_summary = _supervise_tile(
                args, block_m, block_n, tile_output
            )
        except Exception as error:
            print(f"tile rejected: {type(error).__name__}: {error}")
            tile_summaries.append(
                {
                "block_m": block_m,
                "block_n": block_n,
                "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue

        tile_summaries.append(
            {
                "block_m": block_m,
                "block_n": block_n,
                **tile_summary,
            }
        )
        for result in tile_results:
            all_results.append(
                {
                    "block_m": block_m,
                    "block_n": block_n,
                    "tile_directory": tile_name,
                    **result,
                }
            )

    top_results = sorted(
        all_results, key=lambda result: result["tflops"], reverse=True
    )[:20]
    payload = {
        "tiles": tile_summaries,
        "top_results": [
            {"rank": rank, **result}
            for rank, result in enumerate(top_results, start=1)
        ],
    }
    summary_path = args.output / "top20_tiles.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\n=== global Top-20 ===")
    for rank, result in enumerate(top_results, start=1):
        print(
            f"#{rank:02d} tile=({result['block_m']}, {result['block_n']}) "
            f"schedule={result['schedule_index']} "
            f"latency={result['latency_ms']:.4f} ms "
            f"tflops={result['tflops']:.2f}"
        )
    print(f"Results saved to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
