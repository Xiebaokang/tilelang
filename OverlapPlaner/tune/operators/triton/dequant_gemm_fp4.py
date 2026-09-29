"""Packed s1e2m1 FP4 dequantization fused with GEMM."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from .common import benchmark_autotuned_variant, comparison, controlled_stage_sweep
except ImportError:
    from common import benchmark_autotuned_variant, comparison, controlled_stage_sweep

CONFIGS=[triton.Config({"BLOCK_M":bm,"BLOCK_N":bn,"BLOCK_K":bk},num_warps=nw,num_stages=ns)
         for bm in (64,128) for bn in (64,128) for bk in (128,256)
         for nw in (4,8) for ns in (1,2,3,4)]
CONTROL={"BLOCK_M":128,"BLOCK_N":128,"BLOCK_K":128,"num_warps":4}


@triton.jit
def _fp4_to_fp16(nibble):
    value = nibble.to(tl.uint16)
    sign = value >> 3
    exponent = ((value & 6) >> 1) + 14
    mantissa = value & 1
    bits = ((exponent | (sign << 5)) << 10) | (mantissa << 9)
    return tl.cast(bits, tl.float16, bitcast=True)


def _prune(configs,named_args,**kwargs):
    del kwargs
    return [c for c in configs if c.num_warps==4] if named_args["WARP_SPECIALIZE"] else configs


@triton.autotune(configs=CONFIGS,key=["M","N","K","WARP_SPECIALIZE"],
                 prune_configs_by={"early_config_prune":_prune})
@triton.jit
def kernel(a,b,ct,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
           WARP_SPECIALIZE:tl.constexpr,BLOCK_M:tl.constexpr,
           BLOCK_N:tl.constexpr,BLOCK_K:tl.constexpr):
    pm,pn=tl.program_id(0),tl.program_id(1)
    mm=pm*BLOCK_M+tl.arange(0,BLOCK_M)
    nn=pn*BLOCK_N+tl.arange(0,BLOCK_N)
    rr=tl.arange(0,BLOCK_K)
    acc=tl.zeros((BLOCK_N,BLOCK_M),tl.float32)
    for start in tl.range(0,K,BLOCK_K,warp_specialize=WARP_SPECIALIZE):
        kk=start+rr
        av=tl.load(a+mm[None,:]*K+kk[:,None],mask=(mm[None,:]<M)&(kk[:,None]<K),other=0.)
        packed=tl.load(b+nn[:,None]*(K//2)+(kk[None,:]//2),
                       mask=(nn[:,None]<N)&(kk[None,:]<K),other=0)
        nibble=(packed>>(4*(kk[None,:]&1)))&15
        bv=_fp4_to_fp16(nibble)
        acc=tl.dot(bv,av,acc)
    tl.store(ct+nn[:,None]*M+mm[None,:],acc,
             mask=(nn[:,None]<N)&(mm[None,:]<M))


def launch(a,b,c,*,warp_specialize=False,fixed=None,num_stages=None):
    m,k=a.shape;n=b.shape[0]
    if fixed is None:
        grid=lambda meta:(triton.cdiv(m,meta["BLOCK_M"]),triton.cdiv(n,meta["BLOCK_N"]))
        kernel[grid](a,b,c,m,n,k,warp_specialize)
    else:
        grid=(triton.cdiv(m,fixed["BLOCK_M"]),triton.cdiv(n,fixed["BLOCK_N"]))
        kernel.fn[grid](a,b,c,m,n,k,warp_specialize,BLOCK_M=fixed["BLOCK_M"],
                        BLOCK_N=fixed["BLOCK_N"],BLOCK_K=fixed["BLOCK_K"],
                        num_warps=fixed["num_warps"],num_stages=num_stages)


def _torch_convert(b):
    f4=torch.stack((b&15,b>>4),-1).reshape(b.shape[0],-1).to(torch.int16)
    bits=(f4>>3)*-32768+((((f4&6)>>1)+14)<<10)+((f4&1)<<9)
    return bits.view(torch.float16)


def run(*,warmup=100,rep=400,trials=5):
    m=n=k=4096
    torch.manual_seed(16)
    a=torch.randn((m,k),device="cuda",dtype=torch.float16)
    b=torch.randint(0,256,(n,k//2),device="cuda",dtype=torch.uint8)
    c=torch.empty((n,m),device="cuda",dtype=torch.float16)
    sa=a[:128,:128].contiguous();sb=b[:128,:64].contiguous();sc=torch.empty((128,128),device="cuda",dtype=torch.float16)
    launch(sa,sb,sc,fixed=CONTROL,num_stages=2)
    reference=(_torch_convert(sb).float()@sa.float().T).half()
    torch.testing.assert_close(sc,reference,atol=.3,rtol=.03)
    broad=[benchmark_autotuned_variant(lambda ws=ws:launch(a,b,c,warp_specialize=ws),kernel,
             warp_specialize=ws,warmup=warmup,rep=rep,trials=trials) for ws in (False,True)]
    controlled=controlled_stage_sweep(lambda ns,ws:launch(a,b,c,warp_specialize=ws,
                                      fixed=CONTROL,num_stages=ns),CONTROL,
                                      warmup=warmup,rep=rep,trials=trials)
    best=min((x for x in broad+controlled if "latency_ms" in x),key=lambda x:x["latency_ms"])
    result=comparison(
        "dequant_gemm_fp4",best["latency_ms"],total_flops=2.*m*n*k,
        workload={"dequant_fp4_m":m,"dequant_fp4_n":n,"dequant_fp4_k":k})
    result.update(shape={"m":m,"n":n,"k":k,"format":"s1e2m1"},
                  autotuned_variants=broad,variants=controlled,selected_variant=best)
    return result
