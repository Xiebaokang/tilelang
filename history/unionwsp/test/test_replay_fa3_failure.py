"""Replay one FA3 UnionWSP failure JSON through the complete compile path.

This is a standalone GPU debugging program, not a pytest test case.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from functools import partial
from pathlib import Path

import tilelang
import torch

import history.unionwsp as unionwsp
from history.unionwsp.test.test_search_fa3_tiles import make_fa3_prim_func
from history.wspipeline.test.fa3_kernel import make_cuda_target, ref_program


DEFAULT_FAILURE = Path(
    "unionwsp/test/fa3_tile_search/m64_n64/failures/schedule_00001.json"
)


def _load_failure(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if "schedule" not in record:
        raise ValueError(f"{path} does not contain a serialized schedule")
    return record


def _infer_tile(path: Path) -> tuple[int, int]:
    for parent in path.resolve().parents:
        match = re.fullmatch(r"m(\d+)_n(\d+)", parent.name)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
    raise ValueError(
        "cannot infer block_M/block_N from the JSON path; pass --block-m and "
        "--block-n"
    )


def _recorded_shapes(record: dict) -> tuple[tuple[int, ...], ...]:
    shapes = tuple(
        tuple(int(value) for value in tensor["shape"])
        for tensor in record.get("input_tensors", ())
    )
    if len(shapes) != 3 or any(len(shape) != 4 for shape in shapes):
        raise ValueError("FA3 replay needs the recorded Q, K, and V shapes")
    return shapes


def replay_failure(args: argparse.Namespace) -> None:
    failure_path = args.failure.resolve()
    record = _load_failure(failure_path)
    q_shape, k_shape, v_shape = _recorded_shapes(record)
    if k_shape != v_shape:
        raise ValueError("recorded K and V shapes must match")
    if q_shape[:2] != k_shape[:2] or q_shape[3] != k_shape[3]:
        raise ValueError("recorded Q/K/V batch, head, and dimension must match")

    inferred_block_m, inferred_block_n = _infer_tile(failure_path)
    block_m = args.block_m or inferred_block_m
    block_n = args.block_n or inferred_block_n
    batch, heads, seq_q, dim = q_shape
    seq_kv = k_shape[2]
    search_config = record.get("search_configuration", {})
    warmup = (
        args.warmup
        if args.warmup is not None
        else search_config.get("warmup", 1)
    )
    rep = args.rep if args.rep is not None else search_config.get("rep", 1)
    total_flops = float(
        search_config.get(
            "total_flops",
            4.0 * batch * heads * seq_q * seq_kv * dim,
        )
    )

    output_directory = args.output or Path(
        "/tmp/unionwsp_fa3_replay_schedule_"
        f"{int(record.get('schedule_index', 0)):05d}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    result_path = output_directory / "replay_result.json"
    source_path = output_directory / "replay.cu"
    normalized_schedule_path = output_directory / "schedule.json"

    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] != 9
    ):
        raise RuntimeError("FA3 failure replay requires a Hopper CUDA GPU")

    target = make_cuda_target()
    prim_func = make_fa3_prim_func(
        batch,
        heads,
        seq_q,
        seq_kv,
        dim,
        False,
        block_m,
        block_n,
    ).with_attr(
        "tl.program_schedule.request",
        f"unionwsp-json-replay-v1={time.time_ns()}",
    )

    torch.manual_seed(args.seed)
    q = torch.randn(q_shape, device="cuda", dtype=torch.float16)
    k = torch.randn(k_shape, device="cuda", dtype=torch.float16)
    v = torch.randn(v_shape, device="cuda", dtype=torch.float16)
    loaded_schedule = None
    phase = "layout_reduction_and_schedule_application"

    def planner(_symbol, graph, _target):
        nonlocal loaded_schedule
        loaded_schedule = unionwsp.schedule_from_dict(graph, record["schedule"])
        normalized = unionwsp.schedule_to_dict(graph, loaded_schedule)
        normalized_schedule_path.write_text(
            json.dumps(normalized, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("\n=== Replayed schedule ===", flush=True)
        print(json.dumps(normalized, indent=2, sort_keys=True), flush=True)
        return loaded_schedule

    result = {
        "failure_json": str(failure_path),
        "original_phase": record.get("phase"),
        "original_error_type": record.get("error_type"),
        "block_m": block_m,
        "block_n": block_n,
    }
    try:
        phase = "compilation"
        with unionwsp.use_schedule_planner(planner), target:
            compiled = tilelang.compile(
                prim_func,
                out_idx=[3],
                target=target,
                execution_backend=search_config.get(
                    "execution_backend", "cython"
                ),
                pass_configs={
                    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
                },
            )
        if loaded_schedule is None:
            raise RuntimeError("LayoutReducer did not request a UnionWSP schedule")

        phase = "source_extraction"
        source = compiled.get_kernel_source()
        source_path.write_text(source, encoding="utf-8")
        print(f"\n=== Generated source: {source_path} ===", flush=True)
        if not args.no_print_source:
            print(source, flush=True)

        if args.compile_only:
            result.update({"status": "compiled", "source_file": str(source_path)})
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(f"\nCompilation succeeded. Result: {result_path}", flush=True)
            return

        profiler = compiled.get_profiler()
        if not args.skip_correctness:
            phase = "correctness_validation"
            profiler.assert_allclose(
                partial(ref_program, is_causal=False),
                input_tensors=[q, k, v],
                rtol=0.01,
                atol=0.01,
            )
            print("\nCorrectness: PASS", flush=True)

        phase = "benchmark"
        latency_ms = float(
            profiler.do_bench(
                warmup=warmup,
                rep=rep,
                input_tensors=[q, k, v],
            )
        )
        tflops = total_flops / latency_ms * 1e-9
        result.update(
            {
                "status": "success",
                "latency_ms": latency_ms,
                "tflops": tflops,
                "source_file": str(source_path),
            }
        )
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print("\n=== Performance ===", flush=True)
        print(f"latency: {latency_ms:.4f} ms", flush=True)
        print(f"throughput: {tflops:.2f} TFLOPS", flush=True)
        print(f"result: {result_path}", flush=True)
    except Exception as error:
        result.update(
            {
                "status": "failed",
                "phase": phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "source_file": str(source_path) if source_path.exists() else None,
            }
        )
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nReplay failed during {phase}: {type(error).__name__}: {error}")
        print(f"Detailed result: {result_path}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("failure", type=Path, nargs="?", default=DEFAULT_FAILURE)
    parser.add_argument("--block-m", type=int)
    parser.add_argument("--block-n", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--rep", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--no-print-source", action="store_true")
    args = parser.parse_args()
    if (args.block_m is None) != (args.block_n is None):
        parser.error("--block-m and --block-n must be specified together")
    if args.warmup is not None and args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.rep is not None and args.rep < 1:
        parser.error("--rep must be positive")
    replay_failure(args)


if __name__ == "__main__":
    main()
