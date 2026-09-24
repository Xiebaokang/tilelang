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
  void* acc_o_wsp_handoff_6 = ((void*)((char*)buf_dyn_shmem + 32768));
  void* acc_s_wsp_handoff_2 = ((void*)((char*)buf_dyn_shmem + 98304));
  void* acc_s_wsp_handoff_4 = ((void*)((char*)buf_dyn_shmem + 131072));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 163840));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 180224));
  void* scores_scale_wsp_handoff_3 = ((void*)((char*)buf_dyn_shmem + 196608));
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 197632));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[10];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float logsum[2];
  float scores_max[2];
  float acc_s[32];
  float scores_scale[2];
  float acc_o[64];
  float scores_max_prev[2];
  float scores_max_clear[2];
  float scores_sum[2];
  half_t acc_s_cast[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::prefetch_tma_descriptor(Output_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(256);
    overlap_plan_mbar[3].init(256);
    overlap_plan_mbar[4].init(256);
    overlap_plan_mbar[5].init(1);
    overlap_plan_mbar[6].init(256);
    overlap_plan_mbar[7].init(256);
    overlap_plan_mbar[8].init(256);
    overlap_plan_mbar[9].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    tl::warpgroup_reg_dealloc<72>();
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(logsum + 0) = make_float2(broadcast_var, broadcast_var);
    float broadcast_var_1 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_1, broadcast_var_1);
    for (int k = 0; k < 64; ++k) {
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_2 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_2, broadcast_var_2);
      overlap_plan_mbar[2].wait((k & 1));
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        *(float2*)(acc_s + (i * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_2) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        scores_max_clear[i_1] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 16; ++rv) {
          scores_max_clear[i_1] = max(scores_max_clear[i_1], acc_s[((((rv & 7) * 4) + (i_1 * 2)) + (rv >> 3))]);
        }
        scores_max_clear[i_1] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_1]);
        scores_max[i_1] = max(scores_max[i_1], scores_max_clear[i_1]);
      }
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        scores_max[i_2] = max(scores_max[i_2], scores_max_prev[i_2]);
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        scores_scale[i_3] = exp2f(((scores_max_prev[i_3] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_3] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      if ((((int)threadIdx.x) % 4) == 0) {
        #pragma unroll
        for (int i_4 = 0; i_4 < 2; ++i_4) {
          ((float*)scores_scale_wsp_handoff_3)[((((((int)threadIdx.x) >> 5) * 16) + (i_4 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = scores_scale[i_4];
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[3].arrive();
      #pragma unroll
      for (int i_5 = 0; i_5 < 32; ++i_5) {
        acc_s[i_5] = exp2f(((acc_s[i_5] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_5 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 16; ++i_6) {
        *(float2*)(((float*)acc_s_wsp_handoff_4) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_6 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_6 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s + (i_6 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[4].arrive();
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        scores_sum[i_7] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
          scores_sum[i_7] = (scores_sum[i_7] + acc_s[((((rv_1 & 7) * 4) + (i_7 * 2)) + (rv_1 >> 3))]);
        }
        scores_sum[i_7] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_7]);
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        logsum[i_8] = ((logsum[i_8] * scores_scale[i_8]) + scores_sum[i_8]);
      }
    }
    overlap_plan_mbar[6].wait(0);
    #pragma unroll
    for (int i_9 = 0; i_9 < 32; ++i_9) {
      *(float2*)(acc_o + (i_9 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_6) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_9 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_9 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 64; ++i_10) {
      acc_o[i_10] = (acc_o[i_10] / logsum[((i_10 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 8; ++i_11) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((i_11 >> 2) * 8192) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_11 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_o[(i_11 * 8)]), ((half_t)acc_o[((i_11 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_11 * 8) + 2)]), ((half_t)acc_o[((i_11 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_11 * 8) + 4)]), ((half_t)acc_o[((i_11 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_11 * 8) + 6)]), ((half_t)acc_o[((i_11 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[7].arrive();
  } else {
    tl::warpgroup_reg_alloc<176>();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(32768);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[0])), 0, (((int)blockIdx.x) * 128), ((int)blockIdx.y), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[8192])), 64, (((int)blockIdx.x) * 128), ((int)blockIdx.y), 0);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 16; ++i_12) {
      float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i_12 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    }
    for (int k_1 = 0; k_1 < 64; ++k_1) {
      if (1 <= k_1) {
        overlap_plan_mbar[8].wait(((k_1 + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[1].arrive_and_expect_tx(16384);
        tl::fence_proxy_async();
        tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, (k_1 * 64), (((int)blockIdx.y) >> 3), 0);
        tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, (k_1 * 64), (((int)blockIdx.y) >> 3), 0);
      }
      if (1 <= k_1) {
        overlap_plan_mbar[9].wait(((k_1 + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[5].arrive_and_expect_tx(16384);
        tl::fence_proxy_async();
        tl::tma_load(V_desc, overlap_plan_mbar[5], (&(((half_t*)V_shared)[0])), 0, (k_1 * 64), (((int)blockIdx.y) >> 3), 0);
        tl::tma_load(V_desc, overlap_plan_mbar[5], (&(((half_t*)V_shared)[4096])), 64, (k_1 * 64), (((int)blockIdx.y) >> 3), 0);
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 32; ++i_13) {
        acc_s[i_13] = 0x0p+0f/*0.000000e+00*/;
      }
      if (k_1 == 0) {
        overlap_plan_mbar[0].wait(0);
      }
      overlap_plan_mbar[1].wait((k_1 & 1));
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)K_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + (((((ki >> 2) * 16384) + (((((int)threadIdx.x) & 255) >> 7) * 8192)) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 16; ++i_14) {
        *(float2*)(((float*)acc_s_wsp_handoff_2) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_14 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_14 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192)) = *(float2*)(acc_s + (i_14 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[2].arrive();
      overlap_plan_mbar[8].arrive();
      overlap_plan_mbar[4].wait((k_1 & 1));
      #pragma unroll
      for (int i_15 = 0; i_15 < 16; ++i_15) {
        *(float2*)(acc_s + (i_15 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_4) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_15 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_15 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 8; ++i_16) {
        uint2 __1;
        float4 v_ = *(float4*)(acc_s + (i_16 * 4));
        ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
        ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
        *(uint2*)(acc_s_cast + (i_16 * 4)) = __1;
      }
      overlap_plan_mbar[3].wait((k_1 & 1));
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        scores_scale[i_17] = ((float*)scores_scale_wsp_handoff_3)[(((((((int)threadIdx.x) >> 5) * 16) + (i_17 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128)];
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 64; ++i_18) {
        acc_o[i_18] = (acc_o[i_18] * scores_scale[((i_18 & 3) >> 1)]);
      }
      overlap_plan_mbar[5].wait((k_1 & 1));
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_1, (&(((half_t*)V_shared)[0])));
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
      if (k_1 == 63) {
        #pragma unroll
        for (int i_19 = 0; i_19 < 32; ++i_19) {
          *(float2*)(((float*)acc_o_wsp_handoff_6) + (((((((((int)threadIdx.x) >> 5) * 2048) + ((i_19 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_19 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 16384)) = *(float2*)(acc_o + (i_19 * 2));
        }
      }
      overlap_plan_mbar[9].arrive();
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[6].arrive();
    overlap_plan_mbar[7].wait(0);
    if (tl::tl_shuffle_elect<256>()) {
      tl::tma_store(Output_desc, (&(((half_t*)O_shared)[0])), 0, (((int)blockIdx.x) * 128), ((int)blockIdx.y), 0);
      tl::tma_store(Output_desc, (&(((half_t*)O_shared)[8192])), 64, (((int)blockIdx.x) * 128), ((int)blockIdx.y), 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

