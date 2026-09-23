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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(__grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[7];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  float acc_s[32];
  float scores_max_prev[2];
  float scores_max_clear[2];
  float scores_scale_v0[2];
  float scores_sum[2];
  half_t acc_s_cast_v0[32];
  float scores_scale_v1[2];
  half_t acc_s_cast_v1[32];
  float scores_max_clear_1[2];
  float scores_max_clear_2[2];
  float scores_max_clear_3[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::prefetch_tma_descriptor(Output_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(1);
    overlap_plan_mbar[4].init(128);
    overlap_plan_mbar[5].init(128);
    overlap_plan_mbar[6].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[0].arrive_and_expect_tx(16384);
    tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[0])), 0, (((int)blockIdx.x) * 64), ((int)blockIdx.y), 0);
    tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[4096])), 64, (((int)blockIdx.x) * 64), ((int)blockIdx.y), 0);
  }
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(logsum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = -CUDART_INF_F;
  *(float2*)(scores_max + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[1].arrive_and_expect_tx(16384);
    tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, 0, (((int)blockIdx.y) >> 3), 0);
    tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, 0, (((int)blockIdx.y) >> 3), 0);
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
  overlap_plan_mbar[4].arrive();
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
  for (int i_4 = 0; i_4 < 2; ++i_4) {
    scores_scale_v0[i_4] = exp2f(((scores_max_prev[i_4] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_4] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 32; ++i_5) {
    acc_s[i_5] = exp2f(((acc_s[i_5] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_5 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 2; ++i_6) {
    scores_sum[i_6] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
      scores_sum[i_6] = (scores_sum[i_6] + acc_s[((((rv_1 & 7) * 4) + (i_6 * 2)) + (rv_1 >> 3))]);
    }
    scores_sum[i_6] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_6]);
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 2; ++i_7) {
    logsum[i_7] = ((logsum[i_7] * scores_scale_v0[i_7]) + scores_sum[i_7]);
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 8; ++i_8) {
    uint2 __1;
    float4 v_ = *(float4*)(acc_s + (i_8 * 4));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
    *(uint2*)(acc_s_cast_v0 + (i_8 * 4)) = __1;
  }
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[2].arrive_and_expect_tx(16384);
    tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[0])), 0, 0, (((int)blockIdx.y) >> 3), 0);
    tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[4096])), 64, 0, (((int)blockIdx.y) >> 3), 0);
  }
  for (int k = 0; k < 31; ++k) {
    overlap_plan_mbar[4].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 128) + 64), (((int)blockIdx.y) >> 3), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 128) + 64), (((int)blockIdx.y) >> 3), 0);
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
    overlap_plan_mbar[4].arrive();
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
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      scores_scale_v1[i_12] = exp2f(((scores_max_prev[i_12] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_12] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 32; ++i_13) {
      acc_s[i_13] = exp2f(((acc_s[i_13] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_13 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      scores_sum[i_14] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
        scores_sum[i_14] = (scores_sum[i_14] + acc_s[((((rv_3 & 7) * 4) + (i_14 * 2)) + (rv_3 >> 3))]);
      }
      scores_sum[i_14] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      logsum[i_15] = ((logsum[i_15] * scores_scale_v1[i_15]) + scores_sum[i_15]);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 8; ++i_16) {
      uint2 __2;
      float4 v__1 = *(float4*)(acc_s + (i_16 * 4));
      ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
      ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
      *(uint2*)(acc_s_cast_v1 + (i_16 * 4)) = __2;
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 64; ++i_17) {
      acc_o[i_17] = (acc_o[i_17] * scores_scale_v0[((i_17 & 3) >> 1)]);
    }
    if (1 <= k) {
      overlap_plan_mbar[6].wait(((k + 1) & 1));
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[8192])), 0, ((k * 128) + 64), (((int)blockIdx.y) >> 3), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[12288])), 64, ((k * 128) + 64), (((int)blockIdx.y) >> 3), 0);
    }
    overlap_plan_mbar[2].wait((k & 1));
    {
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_2, (&(((half_t*)V_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v0 + (ki_2 * 8)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
    }
    overlap_plan_mbar[5].arrive();
    overlap_plan_mbar[4].wait(1);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 128) + 128), (((int)blockIdx.y) >> 3), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 128) + 128), (((int)blockIdx.y) >> 3), 0);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 32; ++i_18) {
      acc_s[i_18] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[1].wait(0);
    {
      tl::GmmaDescriptor desc_a_2;
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_3, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_2 + ((((ki_3 >> 2) * 8192) + ((ki_3 & 3) * 32)) >> 4)), uint64_t(desc_b_3 + ((((ki_3 >> 2) * 8192) + ((ki_3 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
    }
    overlap_plan_mbar[4].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_5 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_5, broadcast_var_5);
    #pragma unroll
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      scores_max_clear_2[i_19] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
        scores_max_clear_2[i_19] = max(scores_max_clear_2[i_19], acc_s[((((rv_4 & 7) * 4) + (i_19 * 2)) + (rv_4 >> 3))]);
      }
      scores_max_clear_2[i_19] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_2[i_19]);
      scores_max[i_19] = max(scores_max[i_19], scores_max_clear_2[i_19]);
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 2; ++i_20) {
      scores_max[i_20] = max(scores_max[i_20], scores_max_prev[i_20]);
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 2; ++i_21) {
      scores_scale_v0[i_21] = exp2f(((scores_max_prev[i_21] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_21] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 32; ++i_22) {
      acc_s[i_22] = exp2f(((acc_s[i_22] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_22 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_23 = 0; i_23 < 2; ++i_23) {
      scores_sum[i_23] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_5 = 0; rv_5 < 16; ++rv_5) {
        scores_sum[i_23] = (scores_sum[i_23] + acc_s[((((rv_5 & 7) * 4) + (i_23 * 2)) + (rv_5 >> 3))]);
      }
      scores_sum[i_23] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_23]);
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 2; ++i_24) {
      logsum[i_24] = ((logsum[i_24] * scores_scale_v0[i_24]) + scores_sum[i_24]);
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 8; ++i_25) {
      uint2 __3;
      float4 v__2 = *(float4*)(acc_s + (i_25 * 4));
      ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
      ((half2*)(&__3))[1] = __float22half2_rn(((float2*)(&v__2))[1]);
      *(uint2*)(acc_s_cast_v0 + (i_25 * 4)) = __3;
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 64; ++i_26) {
      acc_o[i_26] = (acc_o[i_26] * scores_scale_v1[((i_26 & 3) >> 1)]);
    }
    overlap_plan_mbar[5].wait((k & 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[0])), 0, ((k * 128) + 128), (((int)blockIdx.y) >> 3), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[4096])), 64, ((k * 128) + 128), (((int)blockIdx.y) >> 3), 0);
    }
    overlap_plan_mbar[3].wait((k & 1));
    {
      tl::GmmaDescriptor desc_b_4;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_4, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_4, 16384);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v1 + (ki_4 * 8)), uint64_t(desc_b_4 + ((ki_4 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
    }
    overlap_plan_mbar[6].arrive();
  }
  overlap_plan_mbar[4].wait(0);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[1].arrive_and_expect_tx(16384);
    tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, 4032, (((int)blockIdx.y) >> 3), 0);
    tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, 4032, (((int)blockIdx.y) >> 3), 0);
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 32; ++i_27) {
    acc_s[i_27] = 0x0p+0f/*0.000000e+00*/;
  }
  overlap_plan_mbar[1].wait(1);
  {
    tl::GmmaDescriptor desc_a_3;
    tl::GmmaDescriptor desc_b_5;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((half_t*)Q_shared)[0])));
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_5, (&(((half_t*)K_shared)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_5 = 0; ki_5 < 8; ++ki_5) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_3 + ((((ki_5 >> 2) * 8192) + ((ki_5 & 3) * 32)) >> 4)), uint64_t(desc_b_5 + ((((ki_5 >> 2) * 8192) + ((ki_5 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
  }
  overlap_plan_mbar[4].arrive();
  *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
  float broadcast_var_6 = -CUDART_INF_F;
  *(float2*)(scores_max + 0) = make_float2(broadcast_var_6, broadcast_var_6);
  #pragma unroll
  for (int i_28 = 0; i_28 < 2; ++i_28) {
    scores_max_clear_3[i_28] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_6 = 0; rv_6 < 16; ++rv_6) {
      scores_max_clear_3[i_28] = max(scores_max_clear_3[i_28], acc_s[((((rv_6 & 7) * 4) + (i_28 * 2)) + (rv_6 >> 3))]);
    }
    scores_max_clear_3[i_28] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_3[i_28]);
    scores_max[i_28] = max(scores_max[i_28], scores_max_clear_3[i_28]);
  }
  #pragma unroll
  for (int i_29 = 0; i_29 < 2; ++i_29) {
    scores_max[i_29] = max(scores_max[i_29], scores_max_prev[i_29]);
  }
  #pragma unroll
  for (int i_30 = 0; i_30 < 2; ++i_30) {
    scores_scale_v1[i_30] = exp2f(((scores_max_prev[i_30] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_30] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 32; ++i_31) {
    acc_s[i_31] = exp2f(((acc_s[i_31] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_31 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_32 = 0; i_32 < 2; ++i_32) {
    scores_sum[i_32] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_7 = 0; rv_7 < 16; ++rv_7) {
      scores_sum[i_32] = (scores_sum[i_32] + acc_s[((((rv_7 & 7) * 4) + (i_32 * 2)) + (rv_7 >> 3))]);
    }
    scores_sum[i_32] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_32]);
  }
  #pragma unroll
  for (int i_33 = 0; i_33 < 2; ++i_33) {
    logsum[i_33] = ((logsum[i_33] * scores_scale_v1[i_33]) + scores_sum[i_33]);
  }
  #pragma unroll
  for (int i_34 = 0; i_34 < 8; ++i_34) {
    uint2 __4;
    float4 v__3 = *(float4*)(acc_s + (i_34 * 4));
    ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
    ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&v__3))[1]);
    *(uint2*)(acc_s_cast_v1 + (i_34 * 4)) = __4;
  }
  #pragma unroll
  for (int i_35 = 0; i_35 < 64; ++i_35) {
    acc_o[i_35] = (acc_o[i_35] * scores_scale_v0[((i_35 & 3) >> 1)]);
  }
  overlap_plan_mbar[6].wait(0);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[3].arrive_and_expect_tx(16384);
    tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[8192])), 0, 4032, (((int)blockIdx.y) >> 3), 0);
    tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[12288])), 64, 4032, (((int)blockIdx.y) >> 3), 0);
  }
  overlap_plan_mbar[2].wait(1);
  {
    tl::GmmaDescriptor desc_b_6;
    tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_6, (&(((half_t*)V_shared)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
      tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v0 + (ki_6 * 8)), uint64_t(desc_b_6 + ((ki_6 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
  }
  overlap_plan_mbar[5].arrive();
  #pragma unroll
  for (int i_36 = 0; i_36 < 64; ++i_36) {
    acc_o[i_36] = (acc_o[i_36] * scores_scale_v1[((i_36 & 3) >> 1)]);
  }
  overlap_plan_mbar[3].wait(1);
  {
    tl::GmmaDescriptor desc_b_7;
    tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_7, (&(((half_t*)V_shared)[0])));
    tl::increase_descriptor_offset<int>(desc_b_7, 16384);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
      tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v1 + (ki_7 * 8)), uint64_t(desc_b_7 + ((ki_7 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
  }
  overlap_plan_mbar[6].arrive();
  #pragma unroll
  for (int i_37 = 0; i_37 < 64; ++i_37) {
    acc_o[i_37] = (acc_o[i_37] / logsum[((i_37 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_38 = 0; i_38 < 8; ++i_38) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((i_38 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_38 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_38 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_o[(i_38 * 8)]), ((half_t)acc_o[((i_38 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_38 * 8) + 2)]), ((half_t)acc_o[((i_38 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_38 * 8) + 4)]), ((half_t)acc_o[((i_38 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_38 * 8) + 6)]), ((half_t)acc_o[((i_38 * 8) + 7)])));
  }
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    __syncthreads();
    tl::tma_store(Output_desc, (&(((half_t*)O_shared)[0])), 0, (((int)blockIdx.x) * 64), ((int)blockIdx.y), 0);
    tl::tma_store(Output_desc, (&(((half_t*)O_shared)[4096])), 64, (((int)blockIdx.x) * 64), ((int)blockIdx.y), 0);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

