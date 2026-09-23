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
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[9];
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
  float scores_max_clear_1[2];
  float scores_scale_v1[2];
  half_t acc_s_cast_v1[32];
  float scores_max_clear_2[2];
  half_t acc_s_cast_v2[32];
  float scores_max_clear_3[2];
  float scores_max_clear_4[2];
  float scores_max_clear_5[2];
  float scores_max_clear_6[2];
  float scores_max_clear_7[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(1);
    overlap_plan_mbar[4].init(1);
    overlap_plan_mbar[5].init(128);
    overlap_plan_mbar[6].init(128);
    overlap_plan_mbar[7].init(128);
    overlap_plan_mbar[8].init(128);
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
    tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, 0, ((int)blockIdx.y), 0);
    tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, 0, ((int)blockIdx.y), 0);
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
  overlap_plan_mbar[5].arrive();
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
  overlap_plan_mbar[5].wait(0);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[1].arrive_and_expect_tx(16384);
    tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, 64, ((int)blockIdx.y), 0);
    tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, 64, ((int)blockIdx.y), 0);
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
  overlap_plan_mbar[5].arrive();
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
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[2].arrive_and_expect_tx(16384);
    tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[0])), 0, 0, ((int)blockIdx.y), 0);
    tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[4096])), 64, 0, ((int)blockIdx.y), 0);
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 64; ++i_17) {
    acc_o[i_17] = (acc_o[i_17] * scores_scale_v0[((i_17 & 3) >> 1)]);
  }
  for (int k = 0; k < 21; ++k) {
    overlap_plan_mbar[5].wait(1);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 384) + 128), ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 384) + 128), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 32; ++i_18) {
      acc_s[i_18] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[1].wait(0);
    {
      tl::GmmaDescriptor desc_a_2;
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
    }
    overlap_plan_mbar[5].arrive();
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
      *(uint2*)(acc_s_cast_v2 + (i_25 * 4)) = __3;
    }
    if (1 <= k) {
      overlap_plan_mbar[7].wait(1);
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[8192])), 0, ((k * 384) + 64), ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[12288])), 64, ((k * 384) + 64), ((int)blockIdx.y), 0);
    }
    overlap_plan_mbar[2].wait(0);
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_3, (&(((half_t*)V_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v0 + (ki_3 * 8)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
    }
    overlap_plan_mbar[6].arrive();
    #pragma unroll
    for (int i_26 = 0; i_26 < 64; ++i_26) {
      acc_o[i_26] = (acc_o[i_26] * scores_scale_v1[((i_26 & 3) >> 1)]);
    }
    overlap_plan_mbar[5].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 384) + 192), ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 384) + 192), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 32; ++i_27) {
      acc_s[i_27] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[1].wait(1);
    {
      tl::GmmaDescriptor desc_a_3;
      tl::GmmaDescriptor desc_b_4;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_4, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_3 + ((((ki_4 >> 2) * 8192) + ((ki_4 & 3) * 32)) >> 4)), uint64_t(desc_b_4 + ((((ki_4 >> 2) * 8192) + ((ki_4 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
    }
    overlap_plan_mbar[5].arrive();
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
      *(uint2*)(acc_s_cast_v0 + (i_34 * 4)) = __4;
    }
    if (1 <= k) {
      overlap_plan_mbar[8].wait(1);
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[4], (&(((half_t*)V_shared)[16384])), 0, ((k * 384) + 128), ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[4], (&(((half_t*)V_shared)[20480])), 64, ((k * 384) + 128), ((int)blockIdx.y), 0);
    }
    overlap_plan_mbar[3].wait(0);
    {
      tl::GmmaDescriptor desc_b_5;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_5, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_5, 16384);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v1 + (ki_5 * 8)), uint64_t(desc_b_5 + ((ki_5 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
    }
    overlap_plan_mbar[7].arrive();
    #pragma unroll
    for (int i_35 = 0; i_35 < 64; ++i_35) {
      acc_o[i_35] = (acc_o[i_35] * scores_scale_v0[((i_35 & 3) >> 1)]);
    }
    overlap_plan_mbar[5].wait(1);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 384) + 256), ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 384) + 256), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_36 = 0; i_36 < 32; ++i_36) {
      acc_s[i_36] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[1].wait(0);
    {
      tl::GmmaDescriptor desc_a_4;
      tl::GmmaDescriptor desc_b_6;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_4, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_6, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_6 = 0; ki_6 < 8; ++ki_6) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_4 + ((((ki_6 >> 2) * 8192) + ((ki_6 & 3) * 32)) >> 4)), uint64_t(desc_b_6 + ((((ki_6 >> 2) * 8192) + ((ki_6 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
    }
    overlap_plan_mbar[5].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_7 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_7, broadcast_var_7);
    #pragma unroll
    for (int i_37 = 0; i_37 < 2; ++i_37) {
      scores_max_clear_4[i_37] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_8 = 0; rv_8 < 16; ++rv_8) {
        scores_max_clear_4[i_37] = max(scores_max_clear_4[i_37], acc_s[((((rv_8 & 7) * 4) + (i_37 * 2)) + (rv_8 >> 3))]);
      }
      scores_max_clear_4[i_37] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_4[i_37]);
      scores_max[i_37] = max(scores_max[i_37], scores_max_clear_4[i_37]);
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 2; ++i_38) {
      scores_max[i_38] = max(scores_max[i_38], scores_max_prev[i_38]);
    }
    #pragma unroll
    for (int i_39 = 0; i_39 < 2; ++i_39) {
      scores_scale_v0[i_39] = exp2f(((scores_max_prev[i_39] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_39] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_40 = 0; i_40 < 32; ++i_40) {
      acc_s[i_40] = exp2f(((acc_s[i_40] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_40 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_41 = 0; i_41 < 2; ++i_41) {
      scores_sum[i_41] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_9 = 0; rv_9 < 16; ++rv_9) {
        scores_sum[i_41] = (scores_sum[i_41] + acc_s[((((rv_9 & 7) * 4) + (i_41 * 2)) + (rv_9 >> 3))]);
      }
      scores_sum[i_41] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_41]);
    }
    #pragma unroll
    for (int i_42 = 0; i_42 < 2; ++i_42) {
      logsum[i_42] = ((logsum[i_42] * scores_scale_v0[i_42]) + scores_sum[i_42]);
    }
    #pragma unroll
    for (int i_43 = 0; i_43 < 8; ++i_43) {
      uint2 __5;
      float4 v__4 = *(float4*)(acc_s + (i_43 * 4));
      ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&v__4))[0]);
      ((half2*)(&__5))[1] = __float22half2_rn(((float2*)(&v__4))[1]);
      *(uint2*)(acc_s_cast_v1 + (i_43 * 4)) = __5;
    }
    overlap_plan_mbar[6].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[0])), 0, ((k * 384) + 192), ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[4096])), 64, ((k * 384) + 192), ((int)blockIdx.y), 0);
    }
    overlap_plan_mbar[4].wait(0);
    {
      tl::GmmaDescriptor desc_b_7;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_7, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_7, 32768);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v2 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v2 + (ki_7 * 8)), uint64_t(desc_b_7 + ((ki_7 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v2 + 0), 16);
    }
    overlap_plan_mbar[8].arrive();
    #pragma unroll
    for (int i_44 = 0; i_44 < 64; ++i_44) {
      acc_o[i_44] = (acc_o[i_44] * scores_scale_v1[((i_44 & 3) >> 1)]);
    }
    overlap_plan_mbar[5].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 384) + 320), ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 384) + 320), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_45 = 0; i_45 < 32; ++i_45) {
      acc_s[i_45] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[1].wait(1);
    {
      tl::GmmaDescriptor desc_a_5;
      tl::GmmaDescriptor desc_b_8;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_5, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_8, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_8 = 0; ki_8 < 8; ++ki_8) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_5 + ((((ki_8 >> 2) * 8192) + ((ki_8 & 3) * 32)) >> 4)), uint64_t(desc_b_8 + ((((ki_8 >> 2) * 8192) + ((ki_8 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
    }
    overlap_plan_mbar[5].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_8 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_8, broadcast_var_8);
    #pragma unroll
    for (int i_46 = 0; i_46 < 2; ++i_46) {
      scores_max_clear_5[i_46] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_10 = 0; rv_10 < 16; ++rv_10) {
        scores_max_clear_5[i_46] = max(scores_max_clear_5[i_46], acc_s[((((rv_10 & 7) * 4) + (i_46 * 2)) + (rv_10 >> 3))]);
      }
      scores_max_clear_5[i_46] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_5[i_46]);
      scores_max[i_46] = max(scores_max[i_46], scores_max_clear_5[i_46]);
    }
    #pragma unroll
    for (int i_47 = 0; i_47 < 2; ++i_47) {
      scores_max[i_47] = max(scores_max[i_47], scores_max_prev[i_47]);
    }
    #pragma unroll
    for (int i_48 = 0; i_48 < 2; ++i_48) {
      scores_scale_v1[i_48] = exp2f(((scores_max_prev[i_48] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_48] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_49 = 0; i_49 < 32; ++i_49) {
      acc_s[i_49] = exp2f(((acc_s[i_49] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_49 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_50 = 0; i_50 < 2; ++i_50) {
      scores_sum[i_50] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_11 = 0; rv_11 < 16; ++rv_11) {
        scores_sum[i_50] = (scores_sum[i_50] + acc_s[((((rv_11 & 7) * 4) + (i_50 * 2)) + (rv_11 >> 3))]);
      }
      scores_sum[i_50] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_50]);
    }
    #pragma unroll
    for (int i_51 = 0; i_51 < 2; ++i_51) {
      logsum[i_51] = ((logsum[i_51] * scores_scale_v1[i_51]) + scores_sum[i_51]);
    }
    #pragma unroll
    for (int i_52 = 0; i_52 < 8; ++i_52) {
      uint2 __6;
      float4 v__5 = *(float4*)(acc_s + (i_52 * 4));
      ((half2*)(&__6))[0] = __float22half2_rn(((float2*)(&v__5))[0]);
      ((half2*)(&__6))[1] = __float22half2_rn(((float2*)(&v__5))[1]);
      *(uint2*)(acc_s_cast_v2 + (i_52 * 4)) = __6;
    }
    overlap_plan_mbar[7].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[8192])), 0, ((k * 384) + 256), ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[12288])), 64, ((k * 384) + 256), ((int)blockIdx.y), 0);
    }
    overlap_plan_mbar[2].wait(1);
    {
      tl::GmmaDescriptor desc_b_9;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_9, (&(((half_t*)V_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v0 + (ki_9 * 8)), uint64_t(desc_b_9 + ((ki_9 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
    }
    overlap_plan_mbar[6].arrive();
    #pragma unroll
    for (int i_53 = 0; i_53 < 64; ++i_53) {
      acc_o[i_53] = (acc_o[i_53] * scores_scale_v0[((i_53 & 3) >> 1)]);
    }
    overlap_plan_mbar[5].wait(1);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 384) + 384), ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 384) + 384), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_54 = 0; i_54 < 32; ++i_54) {
      acc_s[i_54] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[1].wait(0);
    {
      tl::GmmaDescriptor desc_a_6;
      tl::GmmaDescriptor desc_b_10;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_6, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_10, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_10 = 0; ki_10 < 8; ++ki_10) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_6 + ((((ki_10 >> 2) * 8192) + ((ki_10 & 3) * 32)) >> 4)), uint64_t(desc_b_10 + ((((ki_10 >> 2) * 8192) + ((ki_10 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
    }
    overlap_plan_mbar[5].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_9 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_9, broadcast_var_9);
    #pragma unroll
    for (int i_55 = 0; i_55 < 2; ++i_55) {
      scores_max_clear_6[i_55] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_12 = 0; rv_12 < 16; ++rv_12) {
        scores_max_clear_6[i_55] = max(scores_max_clear_6[i_55], acc_s[((((rv_12 & 7) * 4) + (i_55 * 2)) + (rv_12 >> 3))]);
      }
      scores_max_clear_6[i_55] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_6[i_55]);
      scores_max[i_55] = max(scores_max[i_55], scores_max_clear_6[i_55]);
    }
    #pragma unroll
    for (int i_56 = 0; i_56 < 2; ++i_56) {
      scores_max[i_56] = max(scores_max[i_56], scores_max_prev[i_56]);
    }
    #pragma unroll
    for (int i_57 = 0; i_57 < 2; ++i_57) {
      scores_scale_v0[i_57] = exp2f(((scores_max_prev[i_57] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_57] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_58 = 0; i_58 < 32; ++i_58) {
      acc_s[i_58] = exp2f(((acc_s[i_58] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_58 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_59 = 0; i_59 < 2; ++i_59) {
      scores_sum[i_59] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_13 = 0; rv_13 < 16; ++rv_13) {
        scores_sum[i_59] = (scores_sum[i_59] + acc_s[((((rv_13 & 7) * 4) + (i_59 * 2)) + (rv_13 >> 3))]);
      }
      scores_sum[i_59] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_59]);
    }
    #pragma unroll
    for (int i_60 = 0; i_60 < 2; ++i_60) {
      logsum[i_60] = ((logsum[i_60] * scores_scale_v0[i_60]) + scores_sum[i_60]);
    }
    #pragma unroll
    for (int i_61 = 0; i_61 < 8; ++i_61) {
      uint2 __7;
      float4 v__6 = *(float4*)(acc_s + (i_61 * 4));
      ((half2*)(&__7))[0] = __float22half2_rn(((float2*)(&v__6))[0]);
      ((half2*)(&__7))[1] = __float22half2_rn(((float2*)(&v__6))[1]);
      *(uint2*)(acc_s_cast_v0 + (i_61 * 4)) = __7;
    }
    overlap_plan_mbar[8].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[4], (&(((half_t*)V_shared)[16384])), 0, ((k * 384) + 320), ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[4], (&(((half_t*)V_shared)[20480])), 64, ((k * 384) + 320), ((int)blockIdx.y), 0);
    }
    overlap_plan_mbar[3].wait(1);
    {
      tl::GmmaDescriptor desc_b_11;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_11, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_11, 16384);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_11 = 0; ki_11 < 4; ++ki_11) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v1 + (ki_11 * 8)), uint64_t(desc_b_11 + ((ki_11 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
    }
    overlap_plan_mbar[7].arrive();
    #pragma unroll
    for (int i_62 = 0; i_62 < 64; ++i_62) {
      acc_o[i_62] = (acc_o[i_62] * scores_scale_v1[((i_62 & 3) >> 1)]);
    }
    overlap_plan_mbar[5].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 384) + 448), ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 384) + 448), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_63 = 0; i_63 < 32; ++i_63) {
      acc_s[i_63] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[1].wait(1);
    {
      tl::GmmaDescriptor desc_a_7;
      tl::GmmaDescriptor desc_b_12;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_7, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_12, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_12 = 0; ki_12 < 8; ++ki_12) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_7 + ((((ki_12 >> 2) * 8192) + ((ki_12 & 3) * 32)) >> 4)), uint64_t(desc_b_12 + ((((ki_12 >> 2) * 8192) + ((ki_12 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
    }
    overlap_plan_mbar[5].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_10 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_10, broadcast_var_10);
    #pragma unroll
    for (int i_64 = 0; i_64 < 2; ++i_64) {
      scores_max_clear_7[i_64] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_14 = 0; rv_14 < 16; ++rv_14) {
        scores_max_clear_7[i_64] = max(scores_max_clear_7[i_64], acc_s[((((rv_14 & 7) * 4) + (i_64 * 2)) + (rv_14 >> 3))]);
      }
      scores_max_clear_7[i_64] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_7[i_64]);
      scores_max[i_64] = max(scores_max[i_64], scores_max_clear_7[i_64]);
    }
    #pragma unroll
    for (int i_65 = 0; i_65 < 2; ++i_65) {
      scores_max[i_65] = max(scores_max[i_65], scores_max_prev[i_65]);
    }
    #pragma unroll
    for (int i_66 = 0; i_66 < 2; ++i_66) {
      scores_scale_v1[i_66] = exp2f(((scores_max_prev[i_66] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_66] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_67 = 0; i_67 < 32; ++i_67) {
      acc_s[i_67] = exp2f(((acc_s[i_67] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_67 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_68 = 0; i_68 < 2; ++i_68) {
      scores_sum[i_68] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_15 = 0; rv_15 < 16; ++rv_15) {
        scores_sum[i_68] = (scores_sum[i_68] + acc_s[((((rv_15 & 7) * 4) + (i_68 * 2)) + (rv_15 >> 3))]);
      }
      scores_sum[i_68] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_68]);
    }
    #pragma unroll
    for (int i_69 = 0; i_69 < 2; ++i_69) {
      logsum[i_69] = ((logsum[i_69] * scores_scale_v1[i_69]) + scores_sum[i_69]);
    }
    #pragma unroll
    for (int i_70 = 0; i_70 < 8; ++i_70) {
      uint2 __8;
      float4 v__7 = *(float4*)(acc_s + (i_70 * 4));
      ((half2*)(&__8))[0] = __float22half2_rn(((float2*)(&v__7))[0]);
      ((half2*)(&__8))[1] = __float22half2_rn(((float2*)(&v__7))[1]);
      *(uint2*)(acc_s_cast_v1 + (i_70 * 4)) = __8;
    }
    overlap_plan_mbar[6].wait(1);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[0])), 0, ((k * 384) + 384), ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[4096])), 64, ((k * 384) + 384), ((int)blockIdx.y), 0);
    }
    overlap_plan_mbar[4].wait(1);
    {
      tl::GmmaDescriptor desc_b_13;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_13, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_13, 32768);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v2 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_13 = 0; ki_13 < 4; ++ki_13) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v2 + (ki_13 * 8)), uint64_t(desc_b_13 + ((ki_13 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v2 + 0), 16);
    }
    overlap_plan_mbar[8].arrive();
    #pragma unroll
    for (int i_71 = 0; i_71 < 64; ++i_71) {
      acc_o[i_71] = (acc_o[i_71] * scores_scale_v0[((i_71 & 3) >> 1)]);
    }
  }
  overlap_plan_mbar[7].wait(1);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[3].arrive_and_expect_tx(16384);
    tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[8192])), 0, 8128, ((int)blockIdx.y), 0);
    tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[12288])), 64, 8128, ((int)blockIdx.y), 0);
  }
  overlap_plan_mbar[2].wait(0);
  {
    tl::GmmaDescriptor desc_b_14;
    tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_14, (&(((half_t*)V_shared)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_14 = 0; ki_14 < 4; ++ki_14) {
      tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v0 + (ki_14 * 8)), uint64_t(desc_b_14 + ((ki_14 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
  }
  overlap_plan_mbar[6].arrive();
  #pragma unroll
  for (int i_72 = 0; i_72 < 64; ++i_72) {
    acc_o[i_72] = (acc_o[i_72] * scores_scale_v1[((i_72 & 3) >> 1)]);
  }
  overlap_plan_mbar[3].wait(0);
  {
    tl::GmmaDescriptor desc_b_15;
    tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_15, (&(((half_t*)V_shared)[0])));
    tl::increase_descriptor_offset<int>(desc_b_15, 16384);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_15 = 0; ki_15 < 4; ++ki_15) {
      tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v1 + (ki_15 * 8)), uint64_t(desc_b_15 + ((ki_15 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
  }
  overlap_plan_mbar[7].arrive();
  #pragma unroll
  for (int i_73 = 0; i_73 < 64; ++i_73) {
    acc_o[i_73] = (acc_o[i_73] / logsum[((i_73 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_74 = 0; i_74 < 8; ++i_74) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_74 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_74 * 8)]), ((half_t)acc_o[((i_74 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_74 * 8) + 2)]), ((half_t)acc_o[((i_74 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_74 * 8) + 4)]), ((half_t)acc_o[((i_74 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_74 * 8) + 6)]), ((half_t)acc_o[((i_74 * 8) + 7)])));
  }
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    __syncthreads();
    tl::tma_store((&(Output[((((int)blockIdx.y) * 1048576) + (((int)blockIdx.x) * 8192))])), (&(((half_t*)O_shared)[0])), 16384);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

