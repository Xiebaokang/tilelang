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
  void* acc_o_wsp_handoff_3 = ((void*)((char*)buf_dyn_shmem + 16384));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 81920));
  void* acc_o_wsp_handoff_5 = ((void*)((char*)buf_dyn_shmem + 114688));
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
  float scores_max_prev[2];
  float scores_scale_v0[2];
  float scores_sum[2];
  float scores_scale_v1[2];
  float acc_o_v1[64];
  half_t acc_s_cast_v0[32];
  half_t acc_s_cast_v1[32];
  float scores_max_clear[2];
  float scores_max_clear_1[2];
  float scores_max_clear_2[2];
  float scores_max_clear_3[2];
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
    overlap_plan_mbar[4].init(128);
    overlap_plan_mbar[5].init(1);
    overlap_plan_mbar[6].init(1);
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
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, 0, ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, 0, ((int)blockIdx.y), 0);
      overlap_plan_mbar[5].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[5], (&(((half_t*)V_shared)[0])), 0, 0, ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[5], (&(((half_t*)V_shared)[4096])), 64, 0, ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; ++i_1) {
      acc_s[i_1] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[0].wait(0);
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
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
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
    for (int i_4 = 0; i_4 < 32; ++i_4) {
      acc_s[i_4] = exp2f(((acc_s[i_4] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_4 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      *(float2*)(((float*)acc_s_wsp_handoff_2) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_5 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_5 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s + (i_5 * 2));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[2].arrive();
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      scores_scale_v0[i_6] = exp2f(((scores_max_prev[i_6] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_6] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      scores_sum[i_7] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        scores_sum[i_7] = (scores_sum[i_7] + acc_s[((((rv_1 & 7) * 4) + (i_7 * 2)) + (rv_1 >> 3))]);
      }
      scores_sum[i_7] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      logsum[i_8] = ((logsum[i_8] * scores_scale_v0[i_8]) + scores_sum[i_8]);
    }
    for (int k = 0; k < 63; ++k) {
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
        overlap_plan_mbar[6].arrive_and_expect_tx(16384);
        tl::tma_load(V_desc, overlap_plan_mbar[6], (&(((half_t*)V_shared)[8192])), 0, ((k * 128) + 64), ((int)blockIdx.y), 0);
        tl::tma_load(V_desc, overlap_plan_mbar[6], (&(((half_t*)V_shared)[12288])), 64, ((k * 128) + 64), ((int)blockIdx.y), 0);
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 32; ++i_9) {
        acc_s[i_9] = 0x0p+0f/*0.000000e+00*/;
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
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_4 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_4, broadcast_var_4);
      #pragma unroll
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        scores_max_clear_1[i_10] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
          scores_max_clear_1[i_10] = max(scores_max_clear_1[i_10], acc_s[((((rv_2 & 7) * 4) + (i_10 * 2)) + (rv_2 >> 3))]);
        }
        scores_max_clear_1[i_10] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_1[i_10]);
        scores_max[i_10] = max(scores_max[i_10], scores_max_clear_1[i_10]);
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        scores_max[i_11] = max(scores_max[i_11], scores_max_prev[i_11]);
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 32; ++i_12) {
        acc_s[i_12] = exp2f(((acc_s[i_12] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_12 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_13 = 0; i_13 < 16; ++i_13) {
        *(float2*)(((float*)acc_s_wsp_handoff_2) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_13 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_13 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s + (i_13 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[2].arrive();
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        scores_scale_v1[i_14] = exp2f(((scores_max_prev[i_14] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_14] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        scores_sum[i_15] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
          scores_sum[i_15] = (scores_sum[i_15] + acc_s[((((rv_3 & 7) * 4) + (i_15 * 2)) + (rv_3 >> 3))]);
        }
        scores_sum[i_15] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_15]);
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 2; ++i_16) {
        logsum[i_16] = ((logsum[i_16] * scores_scale_v1[i_16]) + scores_sum[i_16]);
      }
      if (1 <= k) {
        overlap_plan_mbar[7].wait(1);
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 32; ++i_17) {
        *(float2*)(acc_o_v0 + (i_17 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_5) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_17 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_17 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 64; ++i_18) {
        acc_o_v0[i_18] = (acc_o_v0[i_18] * scores_scale_v0[((i_18 & 3) >> 1)]);
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 32; ++i_19) {
        *(float2*)(((float*)acc_o_wsp_handoff_3) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_19 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_19 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_o_v0 + (i_19 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[3].arrive();
      overlap_plan_mbar[10].wait(1);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[1].arrive_and_expect_tx(16384);
        tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 128) + 128), ((int)blockIdx.y), 0);
        tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 128) + 128), ((int)blockIdx.y), 0);
      }
      overlap_plan_mbar[11].wait((k & 1));
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[5].arrive_and_expect_tx(16384);
        tl::tma_load(V_desc, overlap_plan_mbar[5], (&(((half_t*)V_shared)[0])), 0, ((k * 128) + 128), ((int)blockIdx.y), 0);
        tl::tma_load(V_desc, overlap_plan_mbar[5], (&(((half_t*)V_shared)[4096])), 64, ((k * 128) + 128), ((int)blockIdx.y), 0);
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 32; ++i_20) {
        acc_s[i_20] = 0x0p+0f/*0.000000e+00*/;
      }
      overlap_plan_mbar[1].wait(0);
      {
        tl::GmmaDescriptor desc_a_2;
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)K_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
        tl::warpgroup_arrive();
        tl::__sync_thread_partial(3, 128);
        #pragma unroll
        for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      }
      overlap_plan_mbar[10].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_5 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_5, broadcast_var_5);
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        scores_max_clear_2[i_21] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
          scores_max_clear_2[i_21] = max(scores_max_clear_2[i_21], acc_s[((((rv_4 & 7) * 4) + (i_21 * 2)) + (rv_4 >> 3))]);
        }
        scores_max_clear_2[i_21] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_2[i_21]);
        scores_max[i_21] = max(scores_max[i_21], scores_max_clear_2[i_21]);
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        scores_max[i_22] = max(scores_max[i_22], scores_max_prev[i_22]);
      }
      #pragma unroll
      for (int i_23 = 0; i_23 < 32; ++i_23) {
        acc_s[i_23] = exp2f(((acc_s[i_23] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_23 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 16; ++i_24) {
        *(float2*)(((float*)acc_s_wsp_handoff_2) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_24 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_24 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s + (i_24 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[2].arrive();
      #pragma unroll
      for (int i_25 = 0; i_25 < 2; ++i_25) {
        scores_scale_v0[i_25] = exp2f(((scores_max_prev[i_25] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_25] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_26 = 0; i_26 < 2; ++i_26) {
        scores_sum[i_26] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_5 = 0; rv_5 < 16; ++rv_5) {
          scores_sum[i_26] = (scores_sum[i_26] + acc_s[((((rv_5 & 7) * 4) + (i_26 * 2)) + (rv_5 >> 3))]);
        }
        scores_sum[i_26] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_26]);
      }
      #pragma unroll
      for (int i_27 = 0; i_27 < 2; ++i_27) {
        logsum[i_27] = ((logsum[i_27] * scores_scale_v0[i_27]) + scores_sum[i_27]);
      }
      overlap_plan_mbar[7].wait(0);
      #pragma unroll
      for (int i_28 = 0; i_28 < 32; ++i_28) {
        *(float2*)(acc_o_v1 + (i_28 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_5) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_28 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_28 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      #pragma unroll
      for (int i_29 = 0; i_29 < 64; ++i_29) {
        acc_o_v1[i_29] = (acc_o_v1[i_29] * scores_scale_v1[((i_29 & 3) >> 1)]);
      }
      #pragma unroll
      for (int i_30 = 0; i_30 < 32; ++i_30) {
        *(float2*)(((float*)acc_o_wsp_handoff_3) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_30 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_30 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8192)) = *(float2*)(acc_o_v1 + (i_30 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[4].arrive();
    }
    overlap_plan_mbar[10].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, 8128, ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, 8128, ((int)blockIdx.y), 0);
    }
    overlap_plan_mbar[12].wait(0);
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[6].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[6], (&(((half_t*)V_shared)[8192])), 0, 8128, ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[6], (&(((half_t*)V_shared)[12288])), 64, 8128, ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 32; ++i_31) {
      acc_s[i_31] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[1].wait(1);
    {
      tl::GmmaDescriptor desc_a_3;
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_3, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_3 + ((((ki_3 >> 2) * 8192) + ((ki_3 & 3) * 32)) >> 4)), uint64_t(desc_b_3 + ((((ki_3 >> 2) * 8192) + ((ki_3 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
    }
    overlap_plan_mbar[10].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_6 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_6, broadcast_var_6);
    #pragma unroll
    for (int i_32 = 0; i_32 < 2; ++i_32) {
      scores_max_clear_3[i_32] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_6 = 0; rv_6 < 16; ++rv_6) {
        scores_max_clear_3[i_32] = max(scores_max_clear_3[i_32], acc_s[((((rv_6 & 7) * 4) + (i_32 * 2)) + (rv_6 >> 3))]);
      }
      scores_max_clear_3[i_32] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_3[i_32]);
      scores_max[i_32] = max(scores_max[i_32], scores_max_clear_3[i_32]);
    }
    #pragma unroll
    for (int i_33 = 0; i_33 < 2; ++i_33) {
      scores_max[i_33] = max(scores_max[i_33], scores_max_prev[i_33]);
    }
    #pragma unroll
    for (int i_34 = 0; i_34 < 32; ++i_34) {
      acc_s[i_34] = exp2f(((acc_s[i_34] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_34 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_35 = 0; i_35 < 16; ++i_35) {
      *(float2*)(((float*)acc_s_wsp_handoff_2) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_35 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_35 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s + (i_35 * 2));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[2].arrive();
    #pragma unroll
    for (int i_36 = 0; i_36 < 2; ++i_36) {
      scores_scale_v1[i_36] = exp2f(((scores_max_prev[i_36] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_36] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_37 = 0; i_37 < 2; ++i_37) {
      scores_sum[i_37] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_7 = 0; rv_7 < 16; ++rv_7) {
        scores_sum[i_37] = (scores_sum[i_37] + acc_s[((((rv_7 & 7) * 4) + (i_37 * 2)) + (rv_7 >> 3))]);
      }
      scores_sum[i_37] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_37]);
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 2; ++i_38) {
      logsum[i_38] = ((logsum[i_38] * scores_scale_v1[i_38]) + scores_sum[i_38]);
    }
    overlap_plan_mbar[7].wait(1);
    #pragma unroll
    for (int i_39 = 0; i_39 < 32; ++i_39) {
      *(float2*)(acc_o_v0 + (i_39 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_5) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_39 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_39 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_40 = 0; i_40 < 64; ++i_40) {
      acc_o_v0[i_40] = (acc_o_v0[i_40] * scores_scale_v0[((i_40 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_41 = 0; i_41 < 32; ++i_41) {
      *(float2*)(((float*)acc_o_wsp_handoff_3) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_41 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_41 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_o_v0 + (i_41 * 2));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[3].arrive();
    overlap_plan_mbar[7].wait(0);
    #pragma unroll
    for (int i_42 = 0; i_42 < 32; ++i_42) {
      *(float2*)(acc_o_v1 + (i_42 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_5) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_42 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_42 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_43 = 0; i_43 < 64; ++i_43) {
      acc_o_v1[i_43] = (acc_o_v1[i_43] * scores_scale_v1[((i_43 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_44 = 0; i_44 < 32; ++i_44) {
      *(float2*)(((float*)acc_o_wsp_handoff_3) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_44 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_44 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8192)) = *(float2*)(acc_o_v1 + (i_44 * 2));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[4].arrive();
    overlap_plan_mbar[8].wait(0);
    #pragma unroll
    for (int i_45 = 0; i_45 < 32; ++i_45) {
      *(float2*)(acc_o_v1 + (i_45 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_6) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_45 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_45 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_46 = 0; i_46 < 64; ++i_46) {
      acc_o_v1[i_46] = (acc_o_v1[i_46] / logsum[((i_46 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_47 = 0; i_47 < 8; ++i_47) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_47 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o_v1[(i_47 * 8)]), ((half_t)acc_o_v1[((i_47 * 8) + 1)])), __pack_half2(((half_t)acc_o_v1[((i_47 * 8) + 2)]), ((half_t)acc_o_v1[((i_47 * 8) + 3)])), __pack_half2(((half_t)acc_o_v1[((i_47 * 8) + 4)]), ((half_t)acc_o_v1[((i_47 * 8) + 5)])), __pack_half2(((half_t)acc_o_v1[((i_47 * 8) + 6)]), ((half_t)acc_o_v1[((i_47 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[9].arrive();
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_dealloc<24>();
      overlap_plan_mbar[9].wait(0);
      if (tl::tl_shuffle_elect<128>()) {
        tl::tma_store((&(Output[((((int)blockIdx.y) * 1048576) + (((int)blockIdx.x) * 8192))])), (&(((half_t*)O_shared)[0])), 16384);
        tl::tma_store_arrive();
        tl::tma_store_wait<0, true>();
      }
    } else {
      tl::warpgroup_reg_alloc<240>();
      overlap_plan_mbar[2].wait(0);
      #pragma unroll
      for (int i_48 = 0; i_48 < 16; ++i_48) {
        *(float2*)(acc_s + (i_48 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_2) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_48 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_48 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
      }
      #pragma unroll
      for (int i_49 = 0; i_49 < 8; ++i_49) {
        uint2 __1;
        float4 v_ = *(float4*)(acc_s + (i_49 * 4));
        ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
        ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
        *(uint2*)(acc_s_cast_v0 + (i_49 * 4)) = __1;
      }
      overlap_plan_mbar[2].wait(1);
      #pragma unroll
      for (int i_50 = 0; i_50 < 16; ++i_50) {
        *(float2*)(acc_s + (i_50 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_2) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_50 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_50 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
      }
      #pragma unroll
      for (int i_51 = 0; i_51 < 8; ++i_51) {
        uint2 __2;
        float4 v__1 = *(float4*)(acc_s + (i_51 * 4));
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
        ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
        *(uint2*)(acc_s_cast_v1 + (i_51 * 4)) = __2;
      }
      for (int k_1 = 0; k_1 < 63; ++k_1) {
        overlap_plan_mbar[3].wait((k_1 & 1));
        overlap_plan_mbar[5].wait((k_1 & 1));
        #pragma unroll
        for (int i_52 = 0; i_52 < 32; ++i_52) {
          *(float2*)(acc_o_v0 + (i_52 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_3) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_52 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_52 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 16384));
        }
        {
          tl::GmmaDescriptor desc_b_4;
          tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_4, (&(((half_t*)V_shared)[0])));
          tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_v0 + 0), 64);
          tl::warpgroup_arrive();
          #pragma unroll
          for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
            tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v0 + (ki_4 * 8)), uint64_t(desc_b_4 + ((ki_4 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_v0 + 0), 1);
          }
          tl::warpgroup_commit_batch();
          tl::warpgroup_wait<0>();
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_v0 + 0), 64);
          tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_53 = 0; i_53 < 32; ++i_53) {
          *(float2*)(((float*)acc_o_wsp_handoff_5) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_53 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_53 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 16384)) = *(float2*)(acc_o_v0 + (i_53 * 2));
        }
        tl::fence_proxy_async();
        overlap_plan_mbar[7].arrive();
        overlap_plan_mbar[11].arrive();
        overlap_plan_mbar[2].wait(0);
        #pragma unroll
        for (int i_54 = 0; i_54 < 16; ++i_54) {
          *(float2*)(acc_s + (i_54 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_2) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_54 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_54 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
        }
        #pragma unroll
        for (int i_55 = 0; i_55 < 8; ++i_55) {
          uint2 __3;
          float4 v__2 = *(float4*)(acc_s + (i_55 * 4));
          ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
          ((half2*)(&__3))[1] = __float22half2_rn(((float2*)(&v__2))[1]);
          *(uint2*)(acc_s_cast_v0 + (i_55 * 4)) = __3;
        }
        overlap_plan_mbar[4].wait((k_1 & 1));
        overlap_plan_mbar[6].wait((k_1 & 1));
        #pragma unroll
        for (int i_56 = 0; i_56 < 32; ++i_56) {
          *(float2*)(acc_o_v1 + (i_56 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_3) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_56 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_56 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
        }
        {
          tl::GmmaDescriptor desc_b_5;
          tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_5, (&(((half_t*)V_shared)[0])));
          tl::increase_descriptor_offset<int>(desc_b_5, 16384);
          tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_v1 + 0), 64);
          tl::warpgroup_arrive();
          #pragma unroll
          for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
            tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v1 + (ki_5 * 8)), uint64_t(desc_b_5 + ((ki_5 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_v1 + 0), 1);
          }
          tl::warpgroup_commit_batch();
          tl::warpgroup_wait<0>();
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_v1 + 0), 64);
          tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_57 = 0; i_57 < 32; ++i_57) {
          *(float2*)(((float*)acc_o_wsp_handoff_5) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_57 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_57 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 16384)) = *(float2*)(acc_o_v1 + (i_57 * 2));
        }
        tl::fence_proxy_async();
        overlap_plan_mbar[7].arrive();
        overlap_plan_mbar[12].arrive();
        overlap_plan_mbar[2].wait(1);
        #pragma unroll
        for (int i_58 = 0; i_58 < 16; ++i_58) {
          *(float2*)(acc_s + (i_58 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_2) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_58 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_58 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
        }
        #pragma unroll
        for (int i_59 = 0; i_59 < 8; ++i_59) {
          uint2 __4;
          float4 v__3 = *(float4*)(acc_s + (i_59 * 4));
          ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
          ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&v__3))[1]);
          *(uint2*)(acc_s_cast_v1 + (i_59 * 4)) = __4;
        }
      }
      overlap_plan_mbar[3].wait(1);
      overlap_plan_mbar[5].wait(1);
      #pragma unroll
      for (int i_60 = 0; i_60 < 32; ++i_60) {
        *(float2*)(acc_o_v0 + (i_60 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_3) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_60 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_60 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 16384));
      }
      {
        tl::GmmaDescriptor desc_b_6;
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_6, (&(((half_t*)V_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_v0 + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v0 + (ki_6 * 8)), uint64_t(desc_b_6 + ((ki_6 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_v0 + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_v0 + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
      }
      tl::__sync_thread_partial(4, 128);
      #pragma unroll
      for (int i_61 = 0; i_61 < 32; ++i_61) {
        *(float2*)(((float*)acc_o_wsp_handoff_5) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_61 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_61 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 16384)) = *(float2*)(acc_o_v0 + (i_61 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[7].arrive();
      overlap_plan_mbar[11].arrive();
      overlap_plan_mbar[4].wait(1);
      overlap_plan_mbar[6].wait(1);
      #pragma unroll
      for (int i_62 = 0; i_62 < 32; ++i_62) {
        *(float2*)(acc_o_v1 + (i_62 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_3) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_62 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_62 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
      }
      {
        tl::GmmaDescriptor desc_b_7;
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_7, (&(((half_t*)V_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_7, 16384);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_v1 + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v1 + (ki_7 * 8)), uint64_t(desc_b_7 + ((ki_7 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_v1 + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_v1 + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
      }
      tl::__sync_thread_partial(4, 128);
      #pragma unroll
      for (int i_63 = 0; i_63 < 32; ++i_63) {
        *(float2*)(((float*)acc_o_wsp_handoff_5) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_63 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_63 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 16384)) = *(float2*)(acc_o_v1 + (i_63 * 2));
      }
      #pragma unroll
      for (int i_64 = 0; i_64 < 32; ++i_64) {
        *(float2*)(((float*)acc_o_wsp_handoff_6) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_64 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_64 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 16384)) = *(float2*)(acc_o_v1 + (i_64 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[7].arrive();
      overlap_plan_mbar[12].arrive();
      tl::fence_proxy_async();
      overlap_plan_mbar[8].arrive();
    }
  }
}

