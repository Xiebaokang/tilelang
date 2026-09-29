# Triton operator baselines

This directory is intentionally standalone from TileLang/TVM. Run it with the
`torch` conda environment. Each operator is tuned in a fresh child process so
an invalid Hopper warp-specialization candidate cannot poison later runs.

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n torch \
  python OverlapPlaner/tune/operators/triton/main.py \
  --output OverlapPlaner/tune/operators/triton/results.json
```

The main result contains only the fastest correct Triton configuration for
each operator. When `OverlapPlaner/tune/results/<operator>/top30.json` exists,
it also records the saved OverlapPlaner winner and the directly comparable
speedup. `results.details/` keeps the full autotuning records and one worker
log per operator. Triton compiler diagnostics for rejected WS candidates are
written there instead of flooding the main terminal.
The search includes tile sizes, `num_warps`, `num_stages`, and both values of
`warp_specialize` where Triton's Hopper backend accepts the program.

The default command visits every operator registered by
`OverlapPlaner.tune.run`.  A shorter run can select operators explicitly:

```bash
CUDA_VISIBLE_DEVICES=1 /home/xiebaokang/miniconda3/envs/torch/bin/python \
  OverlapPlaner/tune/operators/triton/main.py \
  --operators fa3 gqa gqa_bwd mha_bwd mla gemm gemm_fp8 \
  --output OverlapPlaner/tune/operators/triton/results.json
```

The following baselines are implemented with the same default shapes and
mathematical operation as their OverlapPlaner counterparts:

- `fa3`, `gqa`, `gqa_bwd`, `mha_bwd`, and `mla`
- `gemm`, `gemm_fp8`, `dequant_gemm_fp4`, and `convolution`
- `linear_attn_fwd`, `fused_moe`, `mamba_chunk_scan`, and
  `mamba_chunk_state`

The attention-backward baselines include the dQ atomic accumulation as well as
the dK/dV writes.  The GQA result retains the partial group dimension used by
the TileLang workload.  The linear-attention output has the same caller-zeroed
additive ABI; clearing that buffer is outside both timed regions.

Four internal backward kernels remain explicitly unsupported:
`gdn_chunk_o_bwd`, `gdn_chunk_delta_bwd`, `kda_chunk_bwd_intra`, and
`kda_wy_fast_bwd`.  Their result entries contain the reason.  They require
operator-specific multi-output Triton implementations before a performance
number would be comparable.

An operator is reported as `unsupported` rather than benchmarked through a
different mathematical operation. This is essential for a fair comparison.

`smoke.py` is the fast compile and numerical check used during development. It
uses reduced shapes and is not a performance benchmark.
