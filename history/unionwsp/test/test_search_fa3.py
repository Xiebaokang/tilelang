"""Standalone exhaustive FA3 UnionWSP search; this is not a pytest case."""

import argparse
from functools import partial
from pathlib import Path

import tilelang
import torch

import history.unionwsp as unionwsp
from history.wspipeline.test.fa3_kernel import (
    make_cuda_target,
    make_fa3_prim_func,
    ref_program,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-q", type=int, default=4096)
    parser.add_argument("--seq-kv", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument(
        "--max-schedules",
        type=int,
        default=None,
        help="optional smoke-test limit; omit it to search every schedule",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("unionwsp/test/fa3_top20"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("FA3 UnionWSP search requires a Hopper CUDA GPU")

    batch, heads = 1, 16
    prim_func = make_fa3_prim_func(
        batch,
        heads,
        args.seq_q,
        args.seq_kv,
        args.dim,
        False,
    )
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
    total_flops = (
        4.0 * batch * heads * args.seq_q * args.seq_kv * args.dim
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
        max_schedules=args.max_schedules,
        validate_each_schedule=True,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    print(
        f"searched={summary.examined_schedules}, "
        f"successful={summary.successful_schedules}, "
        f"failed={summary.failed_schedules}"
    )
    for rank, result in enumerate(summary.top_results, start=1):
        print(
            f"#{rank:02d} schedule={result.schedule_index} "
            f"latency={result.latency_ms:.4f} ms "
            f"tflops={result.tflops:.2f}"
        )
    print(f"Top-20 saved to {summary.output_directory}")


if __name__ == "__main__":
    main()
