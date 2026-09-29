"""Tune standalone Triton baselines and save only each operator's winner."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
ALL_OPERATORS = (
    "fa3", "gemm", "convolution", "gqa", "gqa_bwd", "mha_bwd",
    "gdn_chunk_o_bwd", "gdn_chunk_delta_bwd", "kda_chunk_bwd_intra",
    "dequant_gemm_fp4", "kda_wy_fast_bwd", "fused_moe", "gemm_fp8",
    "mla", "linear_attn_fwd", "mamba_chunk_scan", "mamba_chunk_state",
)
IMPLEMENTED = {
    "fa3": "fa3",
    "gemm": "gemm",
    "convolution": "convolution",
    "gqa": "gqa",
    "gqa_bwd": "gqa_bwd",
    "mha_bwd": "mha_bwd",
    "gemm_fp8": "gemm_fp8",
    "mla": "mla",
    "linear_attn_fwd": "linear_attn_fwd",
    "dequant_gemm_fp4": "dequant_gemm_fp4",
    "fused_moe": "fused_moe",
    "mamba_chunk_scan": "mamba_chunk_scan",
    "mamba_chunk_state": "mamba_chunk_state",
}
UNSUPPORTED = {
    "gdn_chunk_o_bwd": "the internal four-output GDN backward ABI has no equivalent Triton kernel yet",
    "gdn_chunk_delta_bwd": "the reverse chunk-state scan has no equivalent Triton kernel yet",
    "kda_chunk_bwd_intra": "the four-output intra-chunk backward kernel has no equivalent Triton kernel yet",
    "kda_wy_fast_bwd": "the five-output WY backward kernel has no equivalent Triton kernel yet",
}


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _winner(raw: dict) -> dict:
    selected = raw["selected_variant"]
    raw_config = dict(selected.get("triton_config") or {})
    config = dict(raw_config.pop("kwargs", {}))
    config.update(raw_config)
    config["warp_specialize"] = bool(selected.get("warp_specialize", False))
    result = {
        "operator": raw["operator"],
        "status": "ok",
        "latency_ms": selected["latency_ms"],
        "tflops": raw.get("triton_tflops"),
        "shape": raw.get("shape"),
        "best_config": config,
        "samples_ms": selected.get("samples_ms"),
        "search_kind": selected.get("search_kind", "controlled_stage_sweep"),
        "details_file": None,
    }
    if "overlapplaner_reference" in raw:
        result["overlapplaner_reference"] = raw["overlapplaner_reference"]
        result["overlapplaner_speedup_vs_triton"] = raw[
            "overlapplaner_speedup_vs_triton"
        ]
    return result


def _worker(operator: str, output: Path, warmup: int, rep: int, trials: int) -> None:
    sys.path.insert(0, str(HERE))
    try:
        module = importlib.import_module(IMPLEMENTED[operator])
        raw = module.run(warmup=warmup, rep=rep, trials=trials)
        _write(output, {"status": "ok", "raw": raw})
    except BaseException as error:
        _write(output, {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })
        raise


def _environment() -> dict:
    import torch
    import triton
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "triton": triton.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", nargs="+", choices=ALL_OPERATORS,
                        default=list(ALL_OPERATORS))
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=400)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="seconds allowed for one operator, including autotune")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.warmup, args.rep, args.trials) < 1 or args.timeout <= 0:
        parser.error("timing counts and timeout must be positive")
    if args.worker:
        if len(args.operators) != 1 or args.worker_output is None:
            parser.error("worker mode requires one operator and --worker-output")
        _worker(args.operators[0], args.worker_output, args.warmup, args.rep, args.trials)
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this script in the torch conda environment")
    details = args.output.parent / f"{args.output.stem}.details"
    details.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "environment": {**_environment(), "warmup": args.warmup,
                        "rep": args.rep, "trials": args.trials},
        "search_space": ["tile sizes", "num_warps", "num_stages", "warp_specialize"],
        "results": [],
    }
    for operator in args.operators:
        if operator not in IMPLEMENTED:
            payload["results"].append({
                "operator": operator, "status": "unsupported",
                "reason": UNSUPPORTED[operator],
            })
            _write(args.output, payload)
            continue
        detail = details / f"{operator}.json"
        worker_log = details / f"{operator}.log"
        detail.unlink(missing_ok=True)
        worker_log.unlink(missing_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--operators", operator, "--worker-output", str(detail),
                   "--warmup", str(args.warmup), "--rep", str(args.rep),
                   "--trials", str(args.trials), "--timeout", str(args.timeout)]
        print(f"[triton] tuning {operator}", flush=True)
        started = time.monotonic()
        try:
            with worker_log.open("w", encoding="utf-8") as log_stream:
                process = subprocess.run(
                    command,
                    env=os.environ.copy(),
                    timeout=args.timeout,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                )
            if not detail.exists():
                raise RuntimeError(
                    f"worker exited {process.returncode} without writing {detail}"
                )
            record = json.loads(detail.read_text(encoding="utf-8"))
            if process.returncode == 0 and record["status"] == "ok":
                result = _winner(record["raw"])
                result["details_file"] = str(detail.relative_to(args.output.parent))
            else:
                result = {"operator": operator, "status": "failed",
                          "error": record.get("error", f"worker exited {process.returncode}"),
                          "details_file": str(detail.relative_to(args.output.parent))}
        except subprocess.TimeoutExpired:
            result = {"operator": operator, "status": "timeout",
                      "error": f"exceeded {args.timeout:g} seconds"}
        except Exception as error:
            result = {
                "operator": operator,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        result["worker_log"] = str(worker_log.relative_to(args.output.parent))
        result["tuning_time_s"] = time.monotonic() - started
        payload["results"].append(result)
        _write(args.output, payload)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
