#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* acc_o_wsp_handoff_5 = ((void*)((char*)buf_dyn_shmem + 16384));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 81920));
  void* acc_o_wsp_handoff_3 = ((void*)((char*)buf_dyn_shmem + 114688));
  void* acc_o_wsp_handoff_6 = ((void*)((char*)buf_dyn_shmem + 147456));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 180224));
  void* acc_s_wsp_handoff_2 = ((void*)((char*)buf_dyn_shmem + 196608));
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 212992));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[13];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o_v0[64];
  float logsum[2];
  float scores_max[2];
  float acc_s[32];
  float acc_o_v1[64];
  float scores_max_prev[2];
  float scores_max_clear[2];
  float scores_scale[2];
  float scores_sum[2];
  float scores_max_clear_1[2];
  half_t acc_s_cast[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(128);
    overlap_plan_mbar[3].init(128);
    overlap_plan_mbar[4].init(1);
    overlap_plan_mbar[5].init(1);
    overlap_plan_mbar[6].init(128);
    overlap_plan_mbar[7].init(128);
    overlap_plan_mbar[8].init(128);
    overlap_plan_mbar[9].init(128);
    overlap_plan_mbar[10].init(128);
    overlap_plan_mbar[11].init(128);
    overlap_plan_mbar[12].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<240>();
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(16384);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[0])), 0, (((int)blockIdx.x) * 64), ((int)blockIdx.y), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[4096])), 64, (((int)blockIdx.x) * 64), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o_v0 + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(logsum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
    float broadcast_var_2 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_2, broadcast_var_2);
    for (int k = 0; k < 64; ++k) {
      if (1 <= k) {
        overlap_plan_mbar[10].wait(1);
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[1].arrive_and_expect_tx(16384);
        tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, (k * 128), ((int)blockIdx.y), 0);
        tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, (k * 128), ((int)blockIdx.y), 0);
      }
      if (1 <= k) {
        overlap_plan_mbar[11].wait(((k + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[4].arrive_and_expect_tx(16384);
        tl::tma_load(V_desc, overlap_plan_mbar[4], (&(((half_t*)V_shared)[0])), 0, (k * 128), ((int)blockIdx.y), 0);
        tl::tma_load(V_desc, overlap_plan_mbar[4], (&(((half_t*)V_shared)[4096])), 64, (k * 128), ((int)blockIdx.y), 0);
      }
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      #pragma unroll
      for (int i_1 = 0; i_1 < 32; ++i_1) {
        acc_s[i_1] = 0x0p+0f/*0.000000e+00*/;
      }
      if (k == 0) {
        overlap_plan_mbar[0].wait(0);
      }
      overlap_plan_mbar[1].wait(0);
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)K_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      }
      overlap_plan_mbar[10].arrive();
      float broadcast_var_3 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        scores_max_clear[i_2] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 16; ++rv) {
          scores_max_clear[i_2] = max(scores_max_clear[i_2], acc_s[((((rv & 7) * 4) + (i_2 * 2)) + (rv >> 3))]);
        }
        scores_max_clear[i_2] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear[i_2]);
        scores_max[i_2] = max(scores_max[i_2], scores_max_clear[i_2]);
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        scores_max[i_3] = max(scores_max[i_3], scores_max_prev[i_3]);
      }
      #pragma unroll
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        scores_scale[i_4] = exp2f(((scores_max_prev[i_4] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_4] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 32; ++i_5) {
        acc_s[i_5] = exp2f(((acc_s[i_5] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_5 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_6 = 0; i_6 < 16; ++i_6) {
        *(float2*)(((float*)acc_s_wsp_handoff_2) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_6 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_6 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s + (i_6 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[2].arrive();
      if (1 <= k) {
        overlap_plan_mbar[7].wait(((k + 1) & 1));
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 32; ++i_7) {
        *(float2*)(acc_o_v0 + (i_7 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_5) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_7 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_7 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8192));
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 64; ++i_8) {
        acc_o_v0[i_8] = (acc_o_v0[i_8] * scores_scale[((i_8 & 3) >> 1)]);
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 32; ++i_9) {
        *(float2*)(((float*)acc_o_wsp_handoff_3) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_9 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_9 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_o_v0 + (i_9 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[3].arrive();
      #pragma unroll
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        scores_sum[i_10] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
          scores_sum[i_10] = (scores_sum[i_10] + acc_s[((((rv_1 & 7) * 4) + (i_10 * 2)) + (rv_1 >> 3))]);
        }
        scores_sum[i_10] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_10]);
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        logsum[i_11] = ((logsum[i_11] * scores_scale[i_11]) + scores_sum[i_11]);
      }
      overlap_plan_mbar[10].wait(0);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[1].arrive_and_expect_tx(16384);
        tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 128) + 64), ((int)blockIdx.y), 0);
        tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 128) + 64), ((int)blockIdx.y), 0);
      }
      if (1 <= k) {
        overlap_plan_mbar[12].wait(((k + 1) & 1));
      }
      tl::__sync_thread_partial(3, 128);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[5].arrive_and_expect_tx(16384);
        tl::tma_load(V_desc, overlap_plan_mbar[5], (&(((half_t*)V_shared)[8192])), 0, ((k * 128) + 64), ((int)blockIdx.y), 0);
        tl::tma_load(V_desc, overlap_plan_mbar[5], (&(((half_t*)V_shared)[12288])), 64, ((k * 128) + 64), ((int)blockIdx.y), 0);
      }
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      #pragma unroll
      for (int i_12 = 0; i_12 < 32; ++i_12) {
        acc_s[i_12] = 0x0p+0f/*0.000000e+00*/;
      }
      overlap_plan_mbar[1].wait(1);
      {
        tl::GmmaDescriptor desc_a_1;
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((half_t*)K_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_1 + ((((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32)) >> 4)), uint64_t(desc_b_1 + ((((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      }
      overlap_plan_mbar[10].arrive();
      float broadcast_var_4 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_4, broadcast_var_4);
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        scores_max_clear_1[i_13] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
          scores_max_clear_1[i_13] = max(scores_max_clear_1[i_13], acc_s[((((rv_2 & 7) * 4) + (i_13 * 2)) + (rv_2 >> 3))]);
        }
        scores_max_clear_1[i_13] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_1[i_13]);
        scores_max[i_13] = max(scores_max[i_13], scores_max_clear_1[i_13]);
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        scores_max[i_14] = max(scores_max[i_14], scores_max_prev[i_14]);
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        scores_scale[i_15] = exp2f(((scores_max_prev[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 32; ++i_16) {
        acc_s[i_16] = exp2f(((acc_s[i_16] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_16 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_17 = 0; i_17 < 16; ++i_17) {
        *(float2*)(((float*)acc_s_wsp_handoff_2) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_17 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_17 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s + (i_17 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[2].arrive();
      overlap_plan_mbar[6].wait((k & 1));
      #pragma unroll
      for (int i_18 = 0; i_18 < 32; ++i_18) {
        *(float2*)(acc_o_v1 + (i_18 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_5) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_18 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_18 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 64; ++i_19) {
        acc_o_v1[i_19] = (acc_o_v1[i_19] * scores_scale[((i_19 & 3) >> 1)]);
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 32; ++i_20) {
        *(float2*)(((float*)acc_o_wsp_handoff_3) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_20 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_20 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_o_v1 + (i_20 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[3].arrive();
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        scores_sum[i_21] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
          scores_sum[i_21] = (scores_sum[i_21] + acc_s[((((rv_3 & 7) * 4) + (i_21 * 2)) + (rv_3 >> 3))]);
        }
        scores_sum[i_21] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_21]);
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        logsum[i_22] = ((logsum[i_22] * scores_scale[i_22]) + scores_sum[i_22]);
      }
    }
    overlap_plan_mbar[8].wait(0);
    #pragma unroll
    for (int i_23 = 0; i_23 < 32; ++i_23) {
      *(float2*)(acc_o_v1 + (i_23 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_6) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_23 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_23 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 64; ++i_24) {
      acc_o_v1[i_24] = (acc_o_v1[i_24] / logsum[((i_24 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 8; ++i_25) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_25 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o_v1[(i_25 * 8)]), ((half_t)acc_o_v1[((i_25 * 8) + 1)])), __pack_half2(((half_t)acc_o_v1[((i_25 * 8) + 2)]), ((half_t)acc_o_v1[((i_25 * 8) + 3)])), __pack_half2(((half_t)acc_o_v1[((i_25 * 8) + 4)]), ((half_t)acc_o_v1[((i_25 * 8) + 5)])), __pack_half2(((half_t)acc_o_v1[((i_25 * 8) + 6)]), ((half_t)acc_o_v1[((i_25 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[9].arrive();
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<240>();
      for (int k_1 = 0; k_1 < 64; ++k_1) {
        overlap_plan_mbar[2].wait(0);
        #pragma unroll
        for (int i_26 = 0; i_26 < 16; ++i_26) {
          *(float2*)(acc_s + (i_26 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_2) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_26 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_26 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096));
        }
        #pragma unroll
        for (int i_27 = 0; i_27 < 8; ++i_27) {
          uint2 __1;
          float4 v_ = *(float4*)(acc_s + (i_27 * 4));
          ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
          ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
          *(uint2*)(acc_s_cast + (i_27 * 4)) = __1;
        }
        overlap_plan_mbar[3].wait(0);
        overlap_plan_mbar[4].wait((k_1 & 1));
        #pragma unroll
        for (int i_28 = 0; i_28 < 32; ++i_28) {
          *(float2*)(acc_o_v0 + (i_28 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_3) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_28 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_28 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
        }
        {
          tl::GmmaDescriptor desc_b_2;
          tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_2, (&(((half_t*)V_shared)[0])));
          tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_v0 + 0), 64);
          tl::warpgroup_arrive();
          #pragma unroll
          for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
            tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_2 * 8)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_v0 + 0), 1);
          }
          tl::warpgroup_commit_batch();
          tl::warpgroup_wait<0>();
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_v0 + 0), 64);
          tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
        }
        #pragma unroll
        for (int i_29 = 0; i_29 < 32; ++i_29) {
          *(float2*)(((float*)acc_o_wsp_handoff_5) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_29 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_29 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192)) = *(float2*)(acc_o_v0 + (i_29 * 2));
        }
        tl::fence_proxy_async();
        overlap_plan_mbar[6].arrive();
        overlap_plan_mbar[11].arrive();
        overlap_plan_mbar[2].wait(1);
        #pragma unroll
        for (int i_30 = 0; i_30 < 16; ++i_30) {
          *(float2*)(acc_s + (i_30 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_2) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_30 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_30 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096));
        }
        #pragma unroll
        for (int i_31 = 0; i_31 < 8; ++i_31) {
          uint2 __2;
          float4 v__1 = *(float4*)(acc_s + (i_31 * 4));
          ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
          ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
          *(uint2*)(acc_s_cast + (i_31 * 4)) = __2;
        }
        overlap_plan_mbar[3].wait(1);
        overlap_plan_mbar[5].wait((k_1 & 1));
        #pragma unroll
        for (int i_32 = 0; i_32 < 32; ++i_32) {
          *(float2*)(acc_o_v1 + (i_32 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_3) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_32 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_32 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
        }
        {
          tl::GmmaDescriptor desc_b_3;
          tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_3, (&(((half_t*)V_shared)[0])));
          tl::increase_descriptor_offset<int>(desc_b_3, 16384);
          tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_v1 + 0), 64);
          tl::warpgroup_arrive();
          #pragma unroll
          for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
            tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_3 * 8)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_v1 + 0), 1);
          }
          tl::warpgroup_commit_batch();
          tl::warpgroup_wait<0>();
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_v1 + 0), 64);
          tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
        }
        #pragma unroll
        for (int i_33 = 0; i_33 < 32; ++i_33) {
          *(float2*)(((float*)acc_o_wsp_handoff_5) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_33 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_33 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_o_v1 + (i_33 * 2));
        }
        if ((k_1 * 2) == 126) {
          #pragma unroll
          for (int i_34 = 0; i_34 < 32; ++i_34) {
            *(float2*)(((float*)acc_o_wsp_handoff_6) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_34 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_34 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192)) = *(float2*)(acc_o_v1 + (i_34 * 2));
          }
        }
        tl::fence_proxy_async();
        overlap_plan_mbar[7].arrive();
        overlap_plan_mbar[12].arrive();
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[8].arrive();
    } else {
      tl::warpgroup_reg_dealloc<24>();
      overlap_plan_mbar[9].wait(0);
      if (tl::tl_shuffle_elect<128>()) {
        tl::tma_store((&(Output[((((int)blockIdx.y) * 1048576) + (((int)blockIdx.x) * 8192))])), (&(((half_t*)O_shared)[0])), 16384);
        tl::tma_store_arrive();
        tl::tma_store_wait<0, true>();
      }
    }
  }
}

