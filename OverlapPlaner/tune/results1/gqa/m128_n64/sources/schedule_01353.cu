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
extern "C" __global__ void __launch_bounds__(512, 1) main_kernel(__grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 81920));
  void* acc_s_wsp_handoff_2 = ((void*)((char*)buf_dyn_shmem + 131072));
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 163840));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[15];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_s[32];
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  float scores_max_prev[2];
  float scores_scale_v0[2];
  float scores_sum[2];
  half_t acc_s_cast[32];
  float scores_scale_v1[2];
  float scores_max_clear[2];
  float scores_max_clear_1[2];
  float scores_max_clear_2[2];
  float scores_max_clear_3[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Output_desc);
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(1);
    overlap_plan_mbar[4].init(256);
    overlap_plan_mbar[5].init(1);
    overlap_plan_mbar[6].init(1);
    overlap_plan_mbar[7].init(1);
    overlap_plan_mbar[8].init(256);
    overlap_plan_mbar[9].init(256);
    overlap_plan_mbar[10].init(256);
    overlap_plan_mbar[11].init(256);
    overlap_plan_mbar[12].init(256);
    overlap_plan_mbar[13].init(256);
    overlap_plan_mbar[14].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    tl::warpgroup_reg_dealloc<32>();
    for (int k = 0; k < 64; ++k) {
      #pragma unroll
      for (int i = 0; i < 32; ++i) {
        acc_s[i] = 0x0p+0f/*0.000000e+00*/;
      }
      if (k == 0) {
        overlap_plan_mbar[0].wait(0);
      }
      overlap_plan_mbar[((k % 3) + 1)].wait(((k % 6) / 3));
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)K_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b, ((k % 3) * 16384));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
        tl::warpgroup_arrive();
        tl::__sync_thread_partial(3, 256);
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + (((((ki >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        *(float2*)(((float*)acc_s_wsp_handoff_2) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_1 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_1 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s + (i_1 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[4].arrive();
      overlap_plan_mbar[((k % 3) + 9)].arrive();
    }
    overlap_plan_mbar[8].wait(0);
    if (tl::tl_shuffle_elect<256>()) {
      tl::tma_store(Output_desc, (&(((half_t*)O_shared)[0])), 0, (((int)blockIdx.x) * 128), ((int)blockIdx.y), 0);
      tl::tma_store(Output_desc, (&(((half_t*)O_shared)[8192])), 64, (((int)blockIdx.x) * 128), ((int)blockIdx.y), 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  } else {
    tl::warpgroup_reg_alloc<216>();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(32768);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[0])), 0, (((int)blockIdx.x) * 128), ((int)blockIdx.y), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[8192])), 64, (((int)blockIdx.x) * 128), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i_2 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(logsum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
    float broadcast_var_2 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_2, broadcast_var_2);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, 0, (((int)blockIdx.y) >> 3), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, 0, (((int)blockIdx.y) >> 3), 0);
      overlap_plan_mbar[5].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[5], (&(((half_t*)V_shared)[0])), 0, 0, (((int)blockIdx.y) >> 3), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[5], (&(((half_t*)V_shared)[4096])), 64, 0, (((int)blockIdx.y) >> 3), 0);
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_3 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
    overlap_plan_mbar[4].wait(0);
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      *(float2*)(acc_s + (i_3 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_2) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_3 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_3 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      scores_max_clear[i_4] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        scores_max_clear[i_4] = max(scores_max_clear[i_4], acc_s[((((rv & 7) * 4) + (i_4 * 2)) + (rv >> 3))]);
      }
      scores_max_clear[i_4] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_4]);
      scores_max[i_4] = max(scores_max[i_4], scores_max_clear[i_4]);
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      scores_max[i_5] = max(scores_max[i_5], scores_max_prev[i_5]);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      scores_scale_v0[i_6] = exp2f(((scores_max_prev[i_6] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_6] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 32; ++i_7) {
      acc_s[i_7] = exp2f(((acc_s[i_7] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_7 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      scores_sum[i_8] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        scores_sum[i_8] = (scores_sum[i_8] + acc_s[((((rv_1 & 7) * 4) + (i_8 * 2)) + (rv_1 >> 3))]);
      }
      scores_sum[i_8] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_8]);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      logsum[i_9] = ((logsum[i_9] * scores_scale_v0[i_9]) + scores_sum[i_9]);
    }
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[2], (&(((half_t*)K_shared)[8192])), 0, 64, (((int)blockIdx.y) >> 3), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[2], (&(((half_t*)K_shared)[12288])), 64, 64, (((int)blockIdx.y) >> 3), 0);
      overlap_plan_mbar[6].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[6], (&(((half_t*)V_shared)[8192])), 0, 64, (((int)blockIdx.y) >> 3), 0);
      tl::tma_load(V_desc, overlap_plan_mbar[6], (&(((half_t*)V_shared)[12288])), 64, 64, (((int)blockIdx.y) >> 3), 0);
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      uint2 __1;
      float4 v_ = *(float4*)(acc_s + (i_10 * 4));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(acc_s_cast + (i_10 * 4)) = __1;
    }
    float broadcast_var_4 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_4, broadcast_var_4);
    overlap_plan_mbar[4].wait(1);
    #pragma unroll
    for (int i_11 = 0; i_11 < 16; ++i_11) {
      *(float2*)(acc_s + (i_11 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_2) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_11 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_11 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      scores_max_clear_1[i_12] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
        scores_max_clear_1[i_12] = max(scores_max_clear_1[i_12], acc_s[((((rv_2 & 7) * 4) + (i_12 * 2)) + (rv_2 >> 3))]);
      }
      scores_max_clear_1[i_12] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_12]);
      scores_max[i_12] = max(scores_max[i_12], scores_max_clear_1[i_12]);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      scores_max[i_13] = max(scores_max[i_13], scores_max_prev[i_13]);
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      scores_scale_v1[i_14] = exp2f(((scores_max_prev[i_14] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_14] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 32; ++i_15) {
      acc_s[i_15] = exp2f(((acc_s[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_15 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      scores_sum[i_16] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
        scores_sum[i_16] = (scores_sum[i_16] + acc_s[((((rv_3 & 7) * 4) + (i_16 * 2)) + (rv_3 >> 3))]);
      }
      scores_sum[i_16] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_16]);
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 2; ++i_17) {
      logsum[i_17] = ((logsum[i_17] * scores_scale_v1[i_17]) + scores_sum[i_17]);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 64; ++i_18) {
      acc_o[i_18] = (acc_o[i_18] * scores_scale_v0[((i_18 & 3) >> 1)]);
    }
    for (int k_1 = 0; k_1 < 31; ++k_1) {
      if (1 <= k_1) {
        overlap_plan_mbar[((((k_1 * 2) + 2) % 3) + 9)].wait(((((k_1 * 2) + 5) % 6) / 3));
      }
      tl::__sync_thread_partial(4, 256);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[((((k_1 * 2) + 2) % 3) + 1)].arrive_and_expect_tx(16384);
        tl::tma_load(K_desc, overlap_plan_mbar[((((k_1 * 2) + 2) % 3) + 1)], (&(((half_t*)K_shared)[((((k_1 * 2) + 2) % 3) * 8192)])), 0, ((k_1 * 128) + 128), (((int)blockIdx.y) >> 3), 0);
        tl::tma_load(K_desc, overlap_plan_mbar[((((k_1 * 2) + 2) % 3) + 1)], (&(((half_t*)K_shared)[(((((k_1 * 2) + 2) % 3) * 8192) + 4096)])), 64, ((k_1 * 128) + 128), (((int)blockIdx.y) >> 3), 0);
      }
      if (1 <= k_1) {
        overlap_plan_mbar[((((k_1 * 2) + 2) % 3) + 12)].wait(((((k_1 * 2) + 5) % 6) / 3));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[((((k_1 * 2) + 2) % 3) + 5)].arrive_and_expect_tx(16384);
        tl::tma_load(V_desc, overlap_plan_mbar[((((k_1 * 2) + 2) % 3) + 5)], (&(((half_t*)V_shared)[((((k_1 * 2) + 2) % 3) * 8192)])), 0, ((k_1 * 128) + 128), (((int)blockIdx.y) >> 3), 0);
        tl::tma_load(V_desc, overlap_plan_mbar[((((k_1 * 2) + 2) % 3) + 5)], (&(((half_t*)V_shared)[(((((k_1 * 2) + 2) % 3) * 8192) + 4096)])), 64, ((k_1 * 128) + 128), (((int)blockIdx.y) >> 3), 0);
      }
      overlap_plan_mbar[(((k_1 * 2) % 3) + 5)].wait((((k_1 % 3) * 2) / 3));
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_1, (&(((half_t*)V_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, (((k_1 * 2) % 3) * 16384));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
      }
      overlap_plan_mbar[(((k_1 * 2) % 3) + 12)].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      #pragma unroll
      for (int i_19 = 0; i_19 < 8; ++i_19) {
        uint2 __2;
        float4 v__1 = *(float4*)(acc_s + (i_19 * 4));
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
        ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
        *(uint2*)(acc_s_cast + (i_19 * 4)) = __2;
      }
      float broadcast_var_5 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_5, broadcast_var_5);
      overlap_plan_mbar[4].wait(0);
      tl::__sync_thread_partial(4, 256);
      #pragma unroll
      for (int i_20 = 0; i_20 < 16; ++i_20) {
        *(float2*)(acc_s + (i_20 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_2) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_20 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_20 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        scores_max_clear_2[i_21] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
          scores_max_clear_2[i_21] = max(scores_max_clear_2[i_21], acc_s[((((rv_4 & 7) * 4) + (i_21 * 2)) + (rv_4 >> 3))]);
        }
        scores_max_clear_2[i_21] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_21]);
        scores_max[i_21] = max(scores_max[i_21], scores_max_clear_2[i_21]);
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        scores_max[i_22] = max(scores_max[i_22], scores_max_prev[i_22]);
      }
      #pragma unroll
      for (int i_23 = 0; i_23 < 2; ++i_23) {
        scores_scale_v0[i_23] = exp2f(((scores_max_prev[i_23] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_23] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 32; ++i_24) {
        acc_s[i_24] = exp2f(((acc_s[i_24] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_24 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_25 = 0; i_25 < 2; ++i_25) {
        scores_sum[i_25] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_5 = 0; rv_5 < 16; ++rv_5) {
          scores_sum[i_25] = (scores_sum[i_25] + acc_s[((((rv_5 & 7) * 4) + (i_25 * 2)) + (rv_5 >> 3))]);
        }
        scores_sum[i_25] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_25]);
      }
      #pragma unroll
      for (int i_26 = 0; i_26 < 2; ++i_26) {
        logsum[i_26] = ((logsum[i_26] * scores_scale_v0[i_26]) + scores_sum[i_26]);
      }
      #pragma unroll
      for (int i_27 = 0; i_27 < 64; ++i_27) {
        acc_o[i_27] = (acc_o[i_27] * scores_scale_v1[((i_27 & 3) >> 1)]);
      }
      overlap_plan_mbar[(((k_1 * 2) % 3) + 9)].wait((((k_1 % 3) * 2) / 3));
      tl::__sync_thread_partial(4, 256);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[(((k_1 * 2) % 3) + 1)].arrive_and_expect_tx(16384);
        tl::tma_load(K_desc, overlap_plan_mbar[(((k_1 * 2) % 3) + 1)], (&(((half_t*)K_shared)[(((k_1 * 2) % 3) * 8192)])), 0, ((k_1 * 128) + 192), (((int)blockIdx.y) >> 3), 0);
        tl::tma_load(K_desc, overlap_plan_mbar[(((k_1 * 2) % 3) + 1)], (&(((half_t*)K_shared)[((((k_1 * 2) % 3) * 8192) + 4096)])), 64, ((k_1 * 128) + 192), (((int)blockIdx.y) >> 3), 0);
      }
      overlap_plan_mbar[(((k_1 * 2) % 3) + 12)].wait((((k_1 % 3) * 2) / 3));
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[(((k_1 * 2) % 3) + 5)].arrive_and_expect_tx(16384);
        tl::tma_load(V_desc, overlap_plan_mbar[(((k_1 * 2) % 3) + 5)], (&(((half_t*)V_shared)[(((k_1 * 2) % 3) * 8192)])), 0, ((k_1 * 128) + 192), (((int)blockIdx.y) >> 3), 0);
        tl::tma_load(V_desc, overlap_plan_mbar[(((k_1 * 2) % 3) + 5)], (&(((half_t*)V_shared)[((((k_1 * 2) % 3) * 8192) + 4096)])), 64, ((k_1 * 128) + 192), (((int)blockIdx.y) >> 3), 0);
      }
      overlap_plan_mbar[((((k_1 * 2) + 1) % 3) + 5)].wait(((((k_1 % 3) * 2) + 1) / 3));
      {
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_2, (&(((half_t*)V_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_2, ((((k_1 * 2) + 1) % 3) * 16384));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_2 * 8)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
      }
      overlap_plan_mbar[((((k_1 * 2) + 1) % 3) + 12)].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      #pragma unroll
      for (int i_28 = 0; i_28 < 8; ++i_28) {
        uint2 __3;
        float4 v__2 = *(float4*)(acc_s + (i_28 * 4));
        ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
        ((half2*)(&__3))[1] = __float22half2_rn(((float2*)(&v__2))[1]);
        *(uint2*)(acc_s_cast + (i_28 * 4)) = __3;
      }
      float broadcast_var_6 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_6, broadcast_var_6);
      overlap_plan_mbar[4].wait(1);
      tl::__sync_thread_partial(4, 256);
      #pragma unroll
      for (int i_29 = 0; i_29 < 16; ++i_29) {
        *(float2*)(acc_s + (i_29 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_2) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_29 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_29 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
      }
      #pragma unroll
      for (int i_30 = 0; i_30 < 2; ++i_30) {
        scores_max_clear_3[i_30] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_6 = 0; rv_6 < 16; ++rv_6) {
          scores_max_clear_3[i_30] = max(scores_max_clear_3[i_30], acc_s[((((rv_6 & 7) * 4) + (i_30 * 2)) + (rv_6 >> 3))]);
        }
        scores_max_clear_3[i_30] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_30]);
        scores_max[i_30] = max(scores_max[i_30], scores_max_clear_3[i_30]);
      }
      #pragma unroll
      for (int i_31 = 0; i_31 < 2; ++i_31) {
        scores_max[i_31] = max(scores_max[i_31], scores_max_prev[i_31]);
      }
      #pragma unroll
      for (int i_32 = 0; i_32 < 2; ++i_32) {
        scores_scale_v1[i_32] = exp2f(((scores_max_prev[i_32] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_32] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_33 = 0; i_33 < 32; ++i_33) {
        acc_s[i_33] = exp2f(((acc_s[i_33] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_33 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_34 = 0; i_34 < 2; ++i_34) {
        scores_sum[i_34] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_7 = 0; rv_7 < 16; ++rv_7) {
          scores_sum[i_34] = (scores_sum[i_34] + acc_s[((((rv_7 & 7) * 4) + (i_34 * 2)) + (rv_7 >> 3))]);
        }
        scores_sum[i_34] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_34]);
      }
      #pragma unroll
      for (int i_35 = 0; i_35 < 2; ++i_35) {
        logsum[i_35] = ((logsum[i_35] * scores_scale_v1[i_35]) + scores_sum[i_35]);
      }
      #pragma unroll
      for (int i_36 = 0; i_36 < 64; ++i_36) {
        acc_o[i_36] = (acc_o[i_36] * scores_scale_v0[((i_36 & 3) >> 1)]);
      }
    }
    overlap_plan_mbar[7].wait(0);
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_3, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_3, 32768);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_3 * 8)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
    }
    overlap_plan_mbar[14].arrive();
    #pragma unroll
    for (int i_37 = 0; i_37 < 8; ++i_37) {
      uint2 __4;
      float4 v__3 = *(float4*)(acc_s + (i_37 * 4));
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
      ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&v__3))[1]);
      *(uint2*)(acc_s_cast + (i_37 * 4)) = __4;
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 64; ++i_38) {
      acc_o[i_38] = (acc_o[i_38] * scores_scale_v1[((i_38 & 3) >> 1)]);
    }
    overlap_plan_mbar[5].wait(1);
    {
      tl::GmmaDescriptor desc_b_4;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_4, (&(((half_t*)V_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_4 * 8)), uint64_t(desc_b_4 + ((ki_4 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
    }
    overlap_plan_mbar[12].arrive();
    #pragma unroll
    for (int i_39 = 0; i_39 < 64; ++i_39) {
      acc_o[i_39] = (acc_o[i_39] / logsum[((i_39 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_40 = 0; i_40 < 8; ++i_40) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((i_40 >> 2) * 8192) + (((((int)threadIdx.x) & 255) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_40 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_40 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_o[(i_40 * 8)]), ((half_t)acc_o[((i_40 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_40 * 8) + 2)]), ((half_t)acc_o[((i_40 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_40 * 8) + 4)]), ((half_t)acc_o[((i_40 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_40 * 8) + 6)]), ((half_t)acc_o[((i_40 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[8].arrive();
  }
}

