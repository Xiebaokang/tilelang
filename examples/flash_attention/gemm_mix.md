# Hopper `T.gemm_mix`

```python
T.gemm_mix(SA, RA, B, C, transpose_A=False, transpose_B=False,
           policy=T.GemmWarpPolicy.Square, clear_accum=False,
           k_pack=1, mbar=None)
```

`SA[M, Ks]` is shared memory and `RA[M, Kr]` is a register fragment.
Logical A is the K concatenation `[SA, RA]`; B covers `Ks + Kr`.
The shapes determine the split. Other arguments and defaults match `T.gemm`.

The operation computes `RA @ B_tail` first, then `SA @ B_prefix`, using one
accumulator. All RS atoms precede all SS atoms, including multiple M/N atoms.
There is one arrive/commit/wait(0) for the mixed operation.
`clear_accum=True` clears only the first RS K atom of each M/N accumulator
tile. Otherwise the initial C is preserved.

The supported path is Hopper WGMMA, shared SA/B, full fragment RA/C,
`transpose_A=False`, `k_pack=1`, and `mbar=None`. Each K extent must be static,
nonempty and aligned to an MMA atom (16 for FP16/BF16). B can be transposed or
not. Unsupported targets are rejected without an ordinary MMA fallback.

## Attention example

`example_mha_fwd_bhsd_mix.py` copies Q's prefix to `Q_shared` and its suffix to
`Q_reg` before the KV loop, then uses mixed QK inside the loop. The default
`q_reg_k_tiles=2` gives Kr=32 and Ks=dim-32.

PV uses `T.gemm_mix(P_shared, P_reg, V_shared, acc_o)`. Each iteration copies
the softmax result's K prefix to `P_shared` and its suffix to `P_reg`. The
default `p_smem_k_tiles=2` gives Ks=32 and Kr=block_N-32. Both portions must
be nonempty and aligned to 16. PV preserves online accumulation.

P_shared uses ordinary shared stores. Automatic lowering inserts the shared
handoff synchronization after PV RS and before SS. The D192/two-stage test
uses BN64 because BM128/BN128 exceeds H100 block shared-memory capacity.

## Register lifetime

Mixed GEMM calls `warpgroup_fence_register_input`, a zero-delta butterfly
shuffle per packed A register. It preserves every bit and provides a real
non-WGMMA use, working around observed CUDA 12.8 register reuse that corrupted
persistent Q inputs across KV loop iterations. Ordinary operand fences are
unchanged. Accumulator fences count only the registers owned by each thread.

## OverlapPlaner

Mixed GEMM is one `OpKind.GEMM` node with the `gemm_mix` tileop name.
`GemmFact.k` is the full reduction dimension; `k_shared` and `k_register`
describe its split. `ra_scope` and `ra_dtype` describe the register input.
SA, RA and B reads and C writes participate in dependency extraction.
Hopper classifies the operation as asynchronous warpgroup tensorcore work.

## Verification

Build the native library and use this checkout on `PYTHONPATH`, then run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TILELANG_DISABLE_CACHE=1 \
  python -m pytest -q testing/python/cuda/test_tilelang_gemm_mix.py \
  testing/python/language/test_tilelang_language_wgmma_gemm.py \
  OverlapPlaner/test/test_gemm_mix.py
```

Tests cover numerical correctness, clear/accumulate, FP16/BF16, B transpose
modes, multiple M/N atoms, RS-before-SS instruction order, mixed QK/PV,
persistent Q registers, shared P handoff, and planner dependencies.
