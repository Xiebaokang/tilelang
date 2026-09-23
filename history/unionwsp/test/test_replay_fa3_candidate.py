"""Replay one saved FA3 UnionWSP candidate through the complete compile path.

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


DEFAULT_CANDIDATE = Path(
    "unionwsp/test/fa3_tile_search/m64_n64/segments/"
    "run_1788259031951020258/start_00000/candidates/schedule_00000.json"
)


def _load_candidate(path: Path) -> dict:
    schedule = json.loads(path.read_text(encoding="utf-8"))
    required_fields = {
        "buffer_versions",
        "groups",
        "operations",
        "orders",
        "stages_by_region",
        "warp_allocation",
    }
    missing = sorted(required_fields.difference(schedule))
    if missing:
        raise ValueError(f"{path} is missing schedule fields: {missing}")
    return schedule


def _infer_tile(path: Path) -> tuple[int, int]:
    for parent in path.resolve().parents:
        match = re.fullmatch(r"m(\d+)_n(\d+)", parent.name)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
    raise ValueError(
        "cannot infer block_M/block_N from the candidate path; pass "
        "--block-m and --block-n"
    )


def replay_candidate(args: argparse.Namespace) -> None:
    candidate_path = args.candidate.resolve()
    serialized_schedule = _load_candidate(candidate_path)
    inferred_block_m, inferred_block_n = _infer_tile(candidate_path)
    block_m = args.block_m or inferred_block_m
    block_n = args.block_n or inferred_block_n

    q_shape = (args.batch, args.heads, args.seq_q, args.dim)
    kv_shape = (args.batch, args.heads, args.seq_kv, args.dim)
    total_flops = float(
        4.0
        * args.batch
        * args.heads
        * args.seq_q
        * args.seq_kv
        * args.dim
    )

    output_directory = args.output or Path(
        f"/tmp/unionwsp_fa3_candidate_{candidate_path.stem}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    source_path = output_directory / "replay.cu"
    schedule_path = output_directory / "schedule.json"
    result_path = output_directory / "replay_result.json"

    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] != 9
    ):
        raise RuntimeError("FA3 candidate replay requires a Hopper CUDA GPU")

    target = make_cuda_target()
    prim_func = make_fa3_prim_func(
        args.batch,
        args.heads,
        args.seq_q,
        args.seq_kv,
        args.dim,
        False,
        block_m,
        block_n,
    ).with_attr(
        "tl.program_schedule.request",
        f"unionwsp-candidate-replay-v1={time.time_ns()}",
    )

    torch.manual_seed(args.seed)
    q = torch.randn(q_shape, device="cuda", dtype=torch.float16)
    k = torch.randn(kv_shape, device="cuda", dtype=torch.float16)
    v = torch.randn(kv_shape, device="cuda", dtype=torch.float16)
    loaded_schedule = None
    phase = "schedule_application"

    def planner(_symbol, graph, _target):
        nonlocal loaded_schedule
        loaded_schedule = unionwsp.schedule_from_dict(
            graph, serialized_schedule
        )
        normalized = unionwsp.schedule_to_dict(graph, loaded_schedule)
        schedule_path.write_text(
            json.dumps(normalized, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("\n=== Replayed schedule ===", flush=True)
        print(json.dumps(normalized, indent=2, sort_keys=True), flush=True)
        return loaded_schedule

    result = {
        "candidate_json": str(candidate_path),
        "block_m": block_m,
        "block_n": block_n,
        "q_shape": list(q_shape),
        "kv_shape": list(kv_shape),
    }
    try:
        phase = "compilation"
        with unionwsp.use_schedule_planner(planner), target:
            pass_configs = {
                tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            }
            if args.dump_ir is not None:
                args.dump_ir.mkdir(parents=True, exist_ok=True)
                pass_configs.update(
                    {
                        tilelang.PassConfigKey.TL_ENABLE_DUMP_IR: True,
                        tilelang.PassConfigKey.TL_DUMP_IR_DIR: str(args.dump_ir),
                    }
                )
            compiled = tilelang.compile(
                prim_func,
                out_idx=[3],
                target=target,
                execution_backend=args.execution_backend,
                pass_configs=pass_configs,
            )
        if loaded_schedule is None:
            raise RuntimeError(
                "LayoutReducer did not request a UnionWSP schedule"
            )

        phase = "source_extraction"
        source = compiled.get_kernel_source()
        source_path.write_text(source, encoding="utf-8")
        print(f"\n=== Generated source: {source_path} ===", flush=True)
        if not args.no_print_source:
            print(source, flush=True)

        if args.compile_only:
            result.update(
                {"status": "compiled", "source_file": str(source_path)}
            )
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"\nCompilation succeeded. Result: {result_path}")
            return

        profiler = compiled.get_profiler()
        if not args.skip_correctness:
            phase = "correctness_validation"
            profiler.assert_allclose(
                partial(ref_program, is_causal=False),
                input_tensors=[q, k, v],
                rtol=args.rtol,
                atol=args.atol,
            )
            print("\nCorrectness: PASS", flush=True)

        phase = "benchmark"
        latency_ms = float(
            profiler.do_bench(
                warmup=args.warmup,
                rep=args.rep,
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
                "source_file": (
                    str(source_path) if source_path.exists() else None
                ),
            }
        )
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"\nReplay failed during {phase}: "
            f"{type(error).__name__}: {error}"
        )
        print(f"Detailed result: {result_path}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "candidate", type=Path, nargs="?", default=DEFAULT_CANDIDATE
    )
    parser.add_argument("--block-m", type=int)
    parser.add_argument("--block-n", type=int)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq-q", type=int, default=8192)
    parser.add_argument("--seq-kv", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--rep", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--execution-backend", default="cython")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dump-ir", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--no-print-source", action="store_true")
    args = parser.parse_args()

    if (args.block_m is None) != (args.block_n is None):
        parser.error("--block-m and --block-n must be specified together")
    if min(
        args.batch,
        args.heads,
        args.seq_q,
        args.seq_kv,
        args.dim,
        args.warmup + 1,
        args.rep,
    ) <= 0:
        parser.error("shapes and repetition counts must be positive")
    replay_candidate(args)


if __name__ == "__main__":
    main()


"""

python test_replay_fa3_candidate.py fa3_tile_search/m64_n64/segments/run_1788259031951020258/start_00000/candidates/schedule_00061.json > log 2>&1

"""
