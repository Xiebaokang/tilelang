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
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 81920));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[9];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  float acc_s_v0[32];
  float scores_max_prev[2];
  float scores_max_clear[2];
  float scores_scale_v0[2];
  float scores_sum[2];
  float acc_s_v1[32];
  float scores_max_clear_1[2];
  float scores_scale_v1[2];
  half_t acc_s_cast_v0[32];
  half_t acc_s_cast_v1[32];
  float scores_max_clear_2[2];
  float scores_scale_v2[2];
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
    overlap_plan_mbar[5].init(256);
    overlap_plan_mbar[6].init(256);
    overlap_plan_mbar[7].init(256);
    overlap_plan_mbar[8].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[0].arrive_and_expect_tx(32768);
    tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[0])), 0, (((int)blockIdx.x) * 128), ((int)blockIdx.y), 0);
    tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[8192])), 64, (((int)blockIdx.x) * 128), ((int)blockIdx.y), 0);
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
  if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[1].arrive_and_expect_tx(16384);
    tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, 0, ((int)blockIdx.y), 0);
    tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, 0, ((int)blockIdx.y), 0);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    acc_s_v0[i_1] = 0x0p+0f/*0.000000e+00*/;
  }
  overlap_plan_mbar[0].wait(0);
  overlap_plan_mbar[1].wait(0);
  {
    tl::GmmaDescriptor desc_a;
    tl::GmmaDescriptor desc_b;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Q_shared)[0])));
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)K_shared)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 64);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki = 0; ki < 8; ++ki) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + (((((ki >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_v0 + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 64);
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
      scores_max_clear[i_2] = max(scores_max_clear[i_2], acc_s_v0[((((rv & 7) * 4) + (i_2 * 2)) + (rv >> 3))]);
    }
    scores_max_clear[i_2] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_2]);
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
    acc_s_v0[i_5] = exp2f(((acc_s_v0[i_5] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_5 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 2; ++i_6) {
    scores_sum[i_6] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
      scores_sum[i_6] = (scores_sum[i_6] + acc_s_v0[((((rv_1 & 7) * 4) + (i_6 * 2)) + (rv_1 >> 3))]);
    }
    scores_sum[i_6] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_6]);
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 2; ++i_7) {
    logsum[i_7] = ((logsum[i_7] * scores_scale_v0[i_7]) + scores_sum[i_7]);
  }
  if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[2].arrive_and_expect_tx(16384);
    tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[0])), 0, 0, ((int)blockIdx.y), 0);
    tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[4096])), 64, 0, ((int)blockIdx.y), 0);
  }
  overlap_plan_mbar[5].wait(0);
  if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[1].arrive_and_expect_tx(16384);
    tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, 64, ((int)blockIdx.y), 0);
    tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, 64, ((int)blockIdx.y), 0);
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 32; ++i_8) {
    acc_s_v1[i_8] = 0x0p+0f/*0.000000e+00*/;
  }
  overlap_plan_mbar[1].wait(1);
  {
    tl::GmmaDescriptor desc_a_1;
    tl::GmmaDescriptor desc_b_1;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)Q_shared)[0])));
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((half_t*)K_shared)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 64);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_1 + (((((ki_1 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_1 & 3) * 32)) >> 4)), uint64_t(desc_b_1 + ((((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_v1 + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 64);
  }
  overlap_plan_mbar[5].arrive();
  *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
  float broadcast_var_4 = -CUDART_INF_F;
  *(float2*)(scores_max + 0) = make_float2(broadcast_var_4, broadcast_var_4);
  #pragma unroll
  for (int i_9 = 0; i_9 < 2; ++i_9) {
    scores_max_clear_1[i_9] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
      scores_max_clear_1[i_9] = max(scores_max_clear_1[i_9], acc_s_v1[((((rv_2 & 7) * 4) + (i_9 * 2)) + (rv_2 >> 3))]);
    }
    scores_max_clear_1[i_9] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_9]);
    scores_max[i_9] = max(scores_max[i_9], scores_max_clear_1[i_9]);
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 2; ++i_10) {
    scores_max[i_10] = max(scores_max[i_10], scores_max_prev[i_10]);
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 2; ++i_11) {
    scores_scale_v1[i_11] = exp2f(((scores_max_prev[i_11] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_11] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 32; ++i_12) {
    acc_s_v1[i_12] = exp2f(((acc_s_v1[i_12] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_12 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 2; ++i_13) {
    scores_sum[i_13] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
      scores_sum[i_13] = (scores_sum[i_13] + acc_s_v1[((((rv_3 & 7) * 4) + (i_13 * 2)) + (rv_3 >> 3))]);
    }
    scores_sum[i_13] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_13]);
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 2; ++i_14) {
    logsum[i_14] = ((logsum[i_14] * scores_scale_v1[i_14]) + scores_sum[i_14]);
  }
  #pragma unroll
  for (int i_15 = 0; i_15 < 8; ++i_15) {
    uint2 __1;
    float4 v_ = *(float4*)(acc_s_v0 + (i_15 * 4));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
    *(uint2*)(acc_s_cast_v0 + (i_15 * 4)) = __1;
  }
  if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[3].arrive_and_expect_tx(16384);
    tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[8192])), 0, 64, ((int)blockIdx.y), 0);
    tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[12288])), 64, 64, ((int)blockIdx.y), 0);
  }
  for (int k = 0; k < 21; ++k) {
    overlap_plan_mbar[5].wait(1);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 384) + 128), ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 384) + 128), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 32; ++i_16) {
      acc_s_v0[i_16] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[1].wait(0);
    {
      tl::GmmaDescriptor desc_a_2;
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_2 + (((((ki_2 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_v0 + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 64);
    }
    overlap_plan_mbar[5].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_5 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_5, broadcast_var_5);
    #pragma unroll
    for (int i_17 = 0; i_17 < 2; ++i_17) {
      scores_max_clear_2[i_17] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
        scores_max_clear_2[i_17] = max(scores_max_clear_2[i_17], acc_s_v0[((((rv_4 & 7) * 4) + (i_17 * 2)) + (rv_4 >> 3))]);
      }
      scores_max_clear_2[i_17] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_17]);
      scores_max[i_17] = max(scores_max[i_17], scores_max_clear_2[i_17]);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 2; ++i_18) {
      scores_max[i_18] = max(scores_max[i_18], scores_max_prev[i_18]);
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      scores_scale_v2[i_19] = exp2f(((scores_max_prev[i_19] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_19] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 32; ++i_20) {
      acc_s_v0[i_20] = exp2f(((acc_s_v0[i_20] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_20 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 2; ++i_21) {
      scores_sum[i_21] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_5 = 0; rv_5 < 16; ++rv_5) {
        scores_sum[i_21] = (scores_sum[i_21] + acc_s_v0[((((rv_5 & 7) * 4) + (i_21 * 2)) + (rv_5 >> 3))]);
      }
      scores_sum[i_21] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_21]);
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 2; ++i_22) {
      logsum[i_22] = ((logsum[i_22] * scores_scale_v2[i_22]) + scores_sum[i_22]);
    }
    #pragma unroll
    for (int i_23 = 0; i_23 < 8; ++i_23) {
      uint2 __2;
      float4 v__1 = *(float4*)(acc_s_v1 + (i_23 * 4));
      ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
      ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
      *(uint2*)(acc_s_cast_v1 + (i_23 * 4)) = __2;
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 64; ++i_24) {
      acc_o[i_24] = (acc_o[i_24] * scores_scale_v0[((i_24 & 3) >> 1)]);
    }
    if (1 <= k) {
      overlap_plan_mbar[8].wait(1);
    }
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[4], (&(((half_t*)V_shared)[16384])), 0, ((k * 384) + 128), ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[4], (&(((half_t*)V_shared)[20480])), 64, ((k * 384) + 128), ((int)blockIdx.y), 0);
    }
    overlap_plan_mbar[2].wait(0);
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_3, (&(((half_t*)V_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v0 + (ki_3 * 8)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
    }
    overlap_plan_mbar[6].arrive();
    overlap_plan_mbar[5].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 384) + 192), ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 384) + 192), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 32; ++i_25) {
      acc_s_v1[i_25] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[1].wait(1);
    {
      tl::GmmaDescriptor desc_a_3;
      tl::GmmaDescriptor desc_b_4;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_4, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_3 + (((((ki_4 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_4 & 3) * 32)) >> 4)), uint64_t(desc_b_4 + ((((ki_4 >> 2) * 8192) + ((ki_4 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_v1 + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 64);
    }
    overlap_plan_mbar[5].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_6 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_6, broadcast_var_6);
    #pragma unroll
    for (int i_26 = 0; i_26 < 2; ++i_26) {
      scores_max_clear_3[i_26] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_6 = 0; rv_6 < 16; ++rv_6) {
        scores_max_clear_3[i_26] = max(scores_max_clear_3[i_26], acc_s_v1[((((rv_6 & 7) * 4) + (i_26 * 2)) + (rv_6 >> 3))]);
      }
      scores_max_clear_3[i_26] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_26]);
      scores_max[i_26] = max(scores_max[i_26], scores_max_clear_3[i_26]);
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 2; ++i_27) {
      scores_max[i_27] = max(scores_max[i_27], scores_max_prev[i_27]);
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 2; ++i_28) {
      scores_scale_v0[i_28] = exp2f(((scores_max_prev[i_28] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_28] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 32; ++i_29) {
      acc_s_v1[i_29] = exp2f(((acc_s_v1[i_29] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_29 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 2; ++i_30) {
      scores_sum[i_30] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_7 = 0; rv_7 < 16; ++rv_7) {
        scores_sum[i_30] = (scores_sum[i_30] + acc_s_v1[((((rv_7 & 7) * 4) + (i_30 * 2)) + (rv_7 >> 3))]);
      }
      scores_sum[i_30] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_30]);
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 2; ++i_31) {
      logsum[i_31] = ((logsum[i_31] * scores_scale_v0[i_31]) + scores_sum[i_31]);
    }
    #pragma unroll
    for (int i_32 = 0; i_32 < 8; ++i_32) {
      uint2 __3;
      float4 v__2 = *(float4*)(acc_s_v0 + (i_32 * 4));
      ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
      ((half2*)(&__3))[1] = __float22half2_rn(((float2*)(&v__2))[1]);
      *(uint2*)(acc_s_cast_v0 + (i_32 * 4)) = __3;
    }
    #pragma unroll
    for (int i_33 = 0; i_33 < 64; ++i_33) {
      acc_o[i_33] = (acc_o[i_33] * scores_scale_v1[((i_33 & 3) >> 1)]);
    }
    overlap_plan_mbar[6].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[0])), 0, ((k * 384) + 192), ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[4096])), 64, ((k * 384) + 192), ((int)blockIdx.y), 0);
    }
    overlap_plan_mbar[3].wait(0);
    {
      tl::GmmaDescriptor desc_b_5;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_5, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_5, 16384);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v1 + (ki_5 * 8)), uint64_t(desc_b_5 + ((ki_5 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
    }
    overlap_plan_mbar[7].arrive();
    overlap_plan_mbar[5].wait(1);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 384) + 256), ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 384) + 256), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_34 = 0; i_34 < 32; ++i_34) {
      acc_s_v0[i_34] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[1].wait(0);
    {
      tl::GmmaDescriptor desc_a_4;
      tl::GmmaDescriptor desc_b_6;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_4, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_6, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_6 = 0; ki_6 < 8; ++ki_6) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_4 + (((((ki_6 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_6 & 3) * 32)) >> 4)), uint64_t(desc_b_6 + ((((ki_6 >> 2) * 8192) + ((ki_6 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_v0 + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 64);
    }
    overlap_plan_mbar[5].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_7 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_7, broadcast_var_7);
    #pragma unroll
    for (int i_35 = 0; i_35 < 2; ++i_35) {
      scores_max_clear_4[i_35] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_8 = 0; rv_8 < 16; ++rv_8) {
        scores_max_clear_4[i_35] = max(scores_max_clear_4[i_35], acc_s_v0[((((rv_8 & 7) * 4) + (i_35 * 2)) + (rv_8 >> 3))]);
      }
      scores_max_clear_4[i_35] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_4[i_35]);
      scores_max[i_35] = max(scores_max[i_35], scores_max_clear_4[i_35]);
    }
    #pragma unroll
    for (int i_36 = 0; i_36 < 2; ++i_36) {
      scores_max[i_36] = max(scores_max[i_36], scores_max_prev[i_36]);
    }
    #pragma unroll
    for (int i_37 = 0; i_37 < 2; ++i_37) {
      scores_scale_v1[i_37] = exp2f(((scores_max_prev[i_37] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_37] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 32; ++i_38) {
      acc_s_v0[i_38] = exp2f(((acc_s_v0[i_38] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_38 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_39 = 0; i_39 < 2; ++i_39) {
      scores_sum[i_39] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_9 = 0; rv_9 < 16; ++rv_9) {
        scores_sum[i_39] = (scores_sum[i_39] + acc_s_v0[((((rv_9 & 7) * 4) + (i_39 * 2)) + (rv_9 >> 3))]);
      }
      scores_sum[i_39] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_39]);
    }
    #pragma unroll
    for (int i_40 = 0; i_40 < 2; ++i_40) {
      logsum[i_40] = ((logsum[i_40] * scores_scale_v1[i_40]) + scores_sum[i_40]);
    }
    #pragma unroll
    for (int i_41 = 0; i_41 < 8; ++i_41) {
      uint2 __4;
      float4 v__3 = *(float4*)(acc_s_v1 + (i_41 * 4));
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
      ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&v__3))[1]);
      *(uint2*)(acc_s_cast_v1 + (i_41 * 4)) = __4;
    }
    #pragma unroll
    for (int i_42 = 0; i_42 < 64; ++i_42) {
      acc_o[i_42] = (acc_o[i_42] * scores_scale_v2[((i_42 & 3) >> 1)]);
    }
    overlap_plan_mbar[7].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[8192])), 0, ((k * 384) + 256), ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[12288])), 64, ((k * 384) + 256), ((int)blockIdx.y), 0);
    }
    overlap_plan_mbar[4].wait(0);
    {
      tl::GmmaDescriptor desc_b_7;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_7, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_7, 32768);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v0 + (ki_7 * 8)), uint64_t(desc_b_7 + ((ki_7 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
    }
    overlap_plan_mbar[8].arrive();
    overlap_plan_mbar[5].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 384) + 320), ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 384) + 320), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_43 = 0; i_43 < 32; ++i_43) {
      acc_s_v1[i_43] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[1].wait(1);
    {
      tl::GmmaDescriptor desc_a_5;
      tl::GmmaDescriptor desc_b_8;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_5, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_8, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_8 = 0; ki_8 < 8; ++ki_8) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_5 + (((((ki_8 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_8 & 3) * 32)) >> 4)), uint64_t(desc_b_8 + ((((ki_8 >> 2) * 8192) + ((ki_8 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_v1 + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 64);
    }
    overlap_plan_mbar[5].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_8 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_8, broadcast_var_8);
    #pragma unroll
    for (int i_44 = 0; i_44 < 2; ++i_44) {
      scores_max_clear_5[i_44] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_10 = 0; rv_10 < 16; ++rv_10) {
        scores_max_clear_5[i_44] = max(scores_max_clear_5[i_44], acc_s_v1[((((rv_10 & 7) * 4) + (i_44 * 2)) + (rv_10 >> 3))]);
      }
      scores_max_clear_5[i_44] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_5[i_44]);
      scores_max[i_44] = max(scores_max[i_44], scores_max_clear_5[i_44]);
    }
    #pragma unroll
    for (int i_45 = 0; i_45 < 2; ++i_45) {
      scores_max[i_45] = max(scores_max[i_45], scores_max_prev[i_45]);
    }
    #pragma unroll
    for (int i_46 = 0; i_46 < 2; ++i_46) {
      scores_scale_v2[i_46] = exp2f(((scores_max_prev[i_46] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_46] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_47 = 0; i_47 < 32; ++i_47) {
      acc_s_v1[i_47] = exp2f(((acc_s_v1[i_47] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_47 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_48 = 0; i_48 < 2; ++i_48) {
      scores_sum[i_48] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_11 = 0; rv_11 < 16; ++rv_11) {
        scores_sum[i_48] = (scores_sum[i_48] + acc_s_v1[((((rv_11 & 7) * 4) + (i_48 * 2)) + (rv_11 >> 3))]);
      }
      scores_sum[i_48] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_48]);
    }
    #pragma unroll
    for (int i_49 = 0; i_49 < 2; ++i_49) {
      logsum[i_49] = ((logsum[i_49] * scores_scale_v2[i_49]) + scores_sum[i_49]);
    }
    #pragma unroll
    for (int i_50 = 0; i_50 < 8; ++i_50) {
      uint2 __5;
      float4 v__4 = *(float4*)(acc_s_v0 + (i_50 * 4));
      ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&v__4))[0]);
      ((half2*)(&__5))[1] = __float22half2_rn(((float2*)(&v__4))[1]);
      *(uint2*)(acc_s_cast_v0 + (i_50 * 4)) = __5;
    }
    #pragma unroll
    for (int i_51 = 0; i_51 < 64; ++i_51) {
      acc_o[i_51] = (acc_o[i_51] * scores_scale_v0[((i_51 & 3) >> 1)]);
    }
    overlap_plan_mbar[8].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[4], (&(((half_t*)V_shared)[16384])), 0, ((k * 384) + 320), ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[4], (&(((half_t*)V_shared)[20480])), 64, ((k * 384) + 320), ((int)blockIdx.y), 0);
    }
    overlap_plan_mbar[2].wait(1);
    {
      tl::GmmaDescriptor desc_b_9;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_9, (&(((half_t*)V_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v1 + (ki_9 * 8)), uint64_t(desc_b_9 + ((ki_9 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
    }
    overlap_plan_mbar[6].arrive();
    overlap_plan_mbar[5].wait(1);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 384) + 384), ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 384) + 384), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_52 = 0; i_52 < 32; ++i_52) {
      acc_s_v0[i_52] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[1].wait(0);
    {
      tl::GmmaDescriptor desc_a_6;
      tl::GmmaDescriptor desc_b_10;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_6, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_10, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_10 = 0; ki_10 < 8; ++ki_10) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_6 + (((((ki_10 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_10 & 3) * 32)) >> 4)), uint64_t(desc_b_10 + ((((ki_10 >> 2) * 8192) + ((ki_10 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_v0 + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 64);
    }
    overlap_plan_mbar[5].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_9 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_9, broadcast_var_9);
    #pragma unroll
    for (int i_53 = 0; i_53 < 2; ++i_53) {
      scores_max_clear_6[i_53] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_12 = 0; rv_12 < 16; ++rv_12) {
        scores_max_clear_6[i_53] = max(scores_max_clear_6[i_53], acc_s_v0[((((rv_12 & 7) * 4) + (i_53 * 2)) + (rv_12 >> 3))]);
      }
      scores_max_clear_6[i_53] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_6[i_53]);
      scores_max[i_53] = max(scores_max[i_53], scores_max_clear_6[i_53]);
    }
    #pragma unroll
    for (int i_54 = 0; i_54 < 2; ++i_54) {
      scores_max[i_54] = max(scores_max[i_54], scores_max_prev[i_54]);
    }
    #pragma unroll
    for (int i_55 = 0; i_55 < 2; ++i_55) {
      scores_scale_v0[i_55] = exp2f(((scores_max_prev[i_55] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_55] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_56 = 0; i_56 < 32; ++i_56) {
      acc_s_v0[i_56] = exp2f(((acc_s_v0[i_56] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_56 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_57 = 0; i_57 < 2; ++i_57) {
      scores_sum[i_57] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_13 = 0; rv_13 < 16; ++rv_13) {
        scores_sum[i_57] = (scores_sum[i_57] + acc_s_v0[((((rv_13 & 7) * 4) + (i_57 * 2)) + (rv_13 >> 3))]);
      }
      scores_sum[i_57] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_57]);
    }
    #pragma unroll
    for (int i_58 = 0; i_58 < 2; ++i_58) {
      logsum[i_58] = ((logsum[i_58] * scores_scale_v0[i_58]) + scores_sum[i_58]);
    }
    #pragma unroll
    for (int i_59 = 0; i_59 < 8; ++i_59) {
      uint2 __6;
      float4 v__5 = *(float4*)(acc_s_v1 + (i_59 * 4));
      ((half2*)(&__6))[0] = __float22half2_rn(((float2*)(&v__5))[0]);
      ((half2*)(&__6))[1] = __float22half2_rn(((float2*)(&v__5))[1]);
      *(uint2*)(acc_s_cast_v1 + (i_59 * 4)) = __6;
    }
    #pragma unroll
    for (int i_60 = 0; i_60 < 64; ++i_60) {
      acc_o[i_60] = (acc_o[i_60] * scores_scale_v1[((i_60 & 3) >> 1)]);
    }
    overlap_plan_mbar[6].wait(1);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[0])), 0, ((k * 384) + 384), ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)V_shared)[4096])), 64, ((k * 384) + 384), ((int)blockIdx.y), 0);
    }
    overlap_plan_mbar[3].wait(1);
    {
      tl::GmmaDescriptor desc_b_11;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_11, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_11, 16384);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_11 = 0; ki_11 < 4; ++ki_11) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v0 + (ki_11 * 8)), uint64_t(desc_b_11 + ((ki_11 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
    }
    overlap_plan_mbar[7].arrive();
    overlap_plan_mbar[5].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 384) + 448), ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 384) + 448), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_61 = 0; i_61 < 32; ++i_61) {
      acc_s_v1[i_61] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[1].wait(1);
    {
      tl::GmmaDescriptor desc_a_7;
      tl::GmmaDescriptor desc_b_12;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_7, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_12, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_12 = 0; ki_12 < 8; ++ki_12) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_7 + (((((ki_12 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_12 & 3) * 32)) >> 4)), uint64_t(desc_b_12 + ((((ki_12 >> 2) * 8192) + ((ki_12 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_v1 + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 64);
    }
    overlap_plan_mbar[5].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_10 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_10, broadcast_var_10);
    #pragma unroll
    for (int i_62 = 0; i_62 < 2; ++i_62) {
      scores_max_clear_7[i_62] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_14 = 0; rv_14 < 16; ++rv_14) {
        scores_max_clear_7[i_62] = max(scores_max_clear_7[i_62], acc_s_v1[((((rv_14 & 7) * 4) + (i_62 * 2)) + (rv_14 >> 3))]);
      }
      scores_max_clear_7[i_62] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_7[i_62]);
      scores_max[i_62] = max(scores_max[i_62], scores_max_clear_7[i_62]);
    }
    #pragma unroll
    for (int i_63 = 0; i_63 < 2; ++i_63) {
      scores_max[i_63] = max(scores_max[i_63], scores_max_prev[i_63]);
    }
    #pragma unroll
    for (int i_64 = 0; i_64 < 2; ++i_64) {
      scores_scale_v1[i_64] = exp2f(((scores_max_prev[i_64] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_64] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_65 = 0; i_65 < 32; ++i_65) {
      acc_s_v1[i_65] = exp2f(((acc_s_v1[i_65] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_65 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_66 = 0; i_66 < 2; ++i_66) {
      scores_sum[i_66] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_15 = 0; rv_15 < 16; ++rv_15) {
        scores_sum[i_66] = (scores_sum[i_66] + acc_s_v1[((((rv_15 & 7) * 4) + (i_66 * 2)) + (rv_15 >> 3))]);
      }
      scores_sum[i_66] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_66]);
    }
    #pragma unroll
    for (int i_67 = 0; i_67 < 2; ++i_67) {
      logsum[i_67] = ((logsum[i_67] * scores_scale_v1[i_67]) + scores_sum[i_67]);
    }
    #pragma unroll
    for (int i_68 = 0; i_68 < 8; ++i_68) {
      uint2 __7;
      float4 v__6 = *(float4*)(acc_s_v0 + (i_68 * 4));
      ((half2*)(&__7))[0] = __float22half2_rn(((float2*)(&v__6))[0]);
      ((half2*)(&__7))[1] = __float22half2_rn(((float2*)(&v__6))[1]);
      *(uint2*)(acc_s_cast_v0 + (i_68 * 4)) = __7;
    }
    #pragma unroll
    for (int i_69 = 0; i_69 < 64; ++i_69) {
      acc_o[i_69] = (acc_o[i_69] * scores_scale_v2[((i_69 & 3) >> 1)]);
    }
    overlap_plan_mbar[7].wait(1);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[8192])), 0, ((k * 384) + 448), ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[3], (&(((half_t*)V_shared)[12288])), 64, ((k * 384) + 448), ((int)blockIdx.y), 0);
    }
    overlap_plan_mbar[4].wait(1);
    {
      tl::GmmaDescriptor desc_b_13;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_13, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_13, 32768);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_13 = 0; ki_13 < 4; ++ki_13) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v1 + (ki_13 * 8)), uint64_t(desc_b_13 + ((ki_13 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v1 + 0), 16);
    }
    overlap_plan_mbar[8].arrive();
  }
  #pragma unroll
  for (int i_70 = 0; i_70 < 8; ++i_70) {
    uint2 __8;
    float4 v__7 = *(float4*)(acc_s_v1 + (i_70 * 4));
    ((half2*)(&__8))[0] = __float22half2_rn(((float2*)(&v__7))[0]);
    ((half2*)(&__8))[1] = __float22half2_rn(((float2*)(&v__7))[1]);
    *(uint2*)(acc_s_cast_v1 + (i_70 * 4)) = __8;
  }
  #pragma unroll
  for (int i_71 = 0; i_71 < 64; ++i_71) {
    acc_o[i_71] = (acc_o[i_71] * scores_scale_v0[((i_71 & 3) >> 1)]);
  }
  overlap_plan_mbar[2].wait(0);
  {
    tl::GmmaDescriptor desc_b_14;
    tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_14, (&(((half_t*)V_shared)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_v0 + 0), 16);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_14 = 0; ki_14 < 4; ++ki_14) {
      tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v0 + (ki_14 * 8)), uint64_t(desc_b_14 + ((ki_14 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
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
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_15 = 0; ki_15 < 4; ++ki_15) {
      tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_v1 + (ki_15 * 8)), uint64_t(desc_b_15 + ((ki_15 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
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
  if (tl::tl_shuffle_elect<256>()) {
    tl::fence_proxy_async();
    __syncthreads();
    tl::tma_store((&(Output[((((int)blockIdx.y) * 1048576) + (((int)blockIdx.x) * 16384))])), (&(((half_t*)O_shared)[0])), 32768);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

