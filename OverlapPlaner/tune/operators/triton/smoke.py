"""Fast compile/correctness smoke tests for newly added Triton kernels."""
from __future__ import annotations

import torch

import attention_bwd
import convolution
import dequant_gemm_fp4
import gemm_fp8
import gqa
import linear_attn_fwd
import mla


def main() -> None:
    torch.manual_seed(0)
    q = torch.randn((1, 128, 8, 128), device="cuda", dtype=torch.float16) * .2
    k = torch.randn((1, 128, 2, 128), device="cuda", dtype=torch.float16) * .2
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    gqa.launch_fixed(q, k, v, out, num_stages=2)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.repeat_interleave(4, 2).permute(0, 2, 1, 3),
        v.repeat_interleave(4, 2).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    torch.testing.assert_close(out, ref, atol=.03, rtol=.02)

    bq = torch.randn((1, 128, 8, 128), device="cuda", dtype=torch.float16) * .1
    bk = torch.randn((1, 128, 2, 128), device="cuda", dtype=torch.float16) * .1
    bv = torch.randn_like(bk)
    bdo = torch.randn_like(bq) * .1
    blse = torch.full((1, 8, 128), 6., device="cuda")
    bdelta = torch.randn_like(blse) * .1
    bdk = torch.empty((4, 1, 128, 2, 128), device="cuda", dtype=torch.float16)
    bdv = torch.empty_like(bdk)
    bdq = torch.zeros_like(bq,dtype=torch.float32)
    attention_bwd.launch(bq,bk,bv,bdo,blse,bdelta,bdq,bdk,bdv,groups=4,
                         fixed=attention_bwd.CONTROL,num_stages=2)
    brdq,brdk,brdv=attention_bwd.reference(bq,bk,bv,bdo,blse,bdelta,groups=4,causal=False)
    torch.testing.assert_close(bdq,brdq,atol=.08,rtol=.04)
    torch.testing.assert_close(bdk,brdk,atol=.05,rtol=.03)
    torch.testing.assert_close(bdv,brdv,atol=.05,rtol=.03)
    mk=bk.repeat_interleave(4,dim=2)
    mv=bv.repeat_interleave(4,dim=2)
    mdk=torch.empty_like(bq)
    mdv=torch.empty_like(bq)
    mdq=torch.zeros_like(bq,dtype=torch.float32)
    attention_bwd.launch(bq,mk,mv,bdo,blse,bdelta,mdq,mdk,mdv,groups=1,
                         fixed=attention_bwd.CONTROL,num_stages=2)
    mrdq,mrdk,mrdv=attention_bwd.reference(bq,mk,mv,bdo,blse,bdelta,groups=1,causal=False)
    torch.testing.assert_close(mdq,mrdq,atol=.08,rtol=.04)
    torch.testing.assert_close(mdk,mrdk,atol=.05,rtol=.03)
    torch.testing.assert_close(mdv,mrdv,atol=.05,rtol=.03)

    data = torch.randn((1, 8, 8, 64), device="cuda", dtype=torch.float16)
    weight = torch.randn((3, 3, 64, 128), device="cuda", dtype=torch.float16)
    conv_out = torch.empty((1, 8, 8, 128), device="cuda", dtype=torch.float16)
    convolution.launch_fixed(data, weight, conv_out, num_stages=2)
    conv_ref = torch.nn.functional.conv2d(data.permute(0,3,1,2),
                                          weight.permute(3,2,0,1), padding=1)
    torch.testing.assert_close(conv_out, conv_ref.permute(0,2,3,1), atol=.2, rtol=.02)

    a = torch.randn((128,128), device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn((128,128), device="cuda").to(torch.float8_e4m3fn)
    c = torch.empty_like(a)
    gemm_fp8.launch(a,b,c,fixed=gemm_fp8.CONTROL,num_stages=2)
    torch.testing.assert_close(c.float(), a.float() @ b.float().T, atol=1., rtol=.15)

    da=torch.randn((128,128),device="cuda",dtype=torch.float16)
    db=torch.randint(0,256,(128,64),device="cuda",dtype=torch.uint8)
    dc=torch.empty((128,128),device="cuda",dtype=torch.float16)
    dequant_gemm_fp4.launch(da,db,dc,fixed=dequant_gemm_fp4.CONTROL,num_stages=2)
    dref=(dequant_gemm_fp4._torch_convert(db).float()@da.float().T).half()
    torch.testing.assert_close(dc,dref,
                               atol=.3,rtol=.03)

    mq=torch.randn((1,32,512),device="cuda",dtype=torch.float16)*.2
    mqp=torch.randn((1,32,64),device="cuda",dtype=torch.float16)*.2
    mkv=torch.randn((1,128,512),device="cuda",dtype=torch.float16)*.2
    mkp=torch.randn((1,128,64),device="cuda",dtype=torch.float16)*.2
    mout=torch.empty_like(mq)
    mla.launch(mq,mqp,mkv,mkp,mout,fixed=mla.CONTROL,num_stages=2)
    score=torch.einsum("bhd,bsd->bhs",mq.float(),mkv.float())+torch.einsum("bhd,bsd->bhs",mqp.float(),mkp.float())
    mref=torch.einsum("bhs,bsd->bhd",torch.softmax(score/576**.5,-1),mkv.float()).half()
    torch.testing.assert_close(mout,mref,atol=.06,rtol=.03)

    lq=torch.randn((1,128,2,128),device="cuda",dtype=torch.float16)
    lk=torch.randn_like(lq)
    lv=torch.randn_like(lq)
    lo=torch.zeros((1,128,2,128),device="cuda",dtype=torch.float32)
    ls=torch.empty((1,2,128,128),device="cuda",dtype=torch.float32)
    linear_attn_fwd.launch(lq,lk,lv,lo,ls,fixed=linear_attn_fwd.CONTROL,num_stages=1)
    lref=torch.einsum("bshk,bshv->bhkv",lk.float(),lv.float())
    torch.testing.assert_close(ls,lref,atol=.2,rtol=.03)
    print("triton smoke: 8 passed")


if __name__ == "__main__":
    main()
