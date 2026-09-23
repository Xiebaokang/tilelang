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
extern "C" __global__ void __launch_bounds__(512, 1) main_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 81920));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[8];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  float acc_s[32];
  float scores_max_prev[2];
  half_t acc_s_cast[32];
  float scores_scale[2];
  float scores_sum[2];
  float scores_max_clear[2];
  float scores_max_clear_1[2];
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
    overlap_plan_mbar[4].init(256);
    overlap_plan_mbar[5].init(256);
    overlap_plan_mbar[6].init(256);
    overlap_plan_mbar[7].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    tl::warpgroup_reg_alloc<224>();
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
      acc_s[i_1] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[0].wait(0);
    overlap_plan_mbar[1].wait(0);
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + (((((ki >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
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
      scores_max_clear[i_2] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_2]);
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
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      uint2 __1;
      float4 v_ = *(float4*)(acc_s + (i_5 * 4));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(acc_s_cast + (i_5 * 4)) = __1;
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      scores_scale[i_6] = exp2f(((scores_max_prev[i_6] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_6] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
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
    #pragma unroll
    for (int i_9 = 0; i_9 < 64; ++i_9) {
      acc_o[i_9] = (acc_o[i_9] * scores_scale[((i_9 & 3) >> 1)]);
    }
    for (int k = 0; k < 127; ++k) {
      overlap_plan_mbar[5].wait((k & 1));
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[1].arrive_and_expect_tx(16384);
        tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, ((k * 64) + 64), ((int)blockIdx.y), 0);
        tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, ((k * 64) + 64), ((int)blockIdx.y), 0);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 32; ++i_10) {
        acc_s[i_10] = 0x0p+0f/*0.000000e+00*/;
      }
      overlap_plan_mbar[1].wait(((k + 1) & 1));
      {
        tl::GmmaDescriptor desc_a_1;
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((half_t*)K_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_1 + (((((ki_1 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_1 & 3) * 32)) >> 4)), uint64_t(desc_b_1 + ((((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
      }
      overlap_plan_mbar[5].arrive();
      overlap_plan_mbar[((k & 1) + 2)].wait(((k & 3) >> 1));
      {
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_2, (&(((half_t*)V_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_2, ((k & 1) * 16384));
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
      overlap_plan_mbar[((k & 1) + 6)].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_4 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_4, broadcast_var_4);
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        scores_max_clear_1[i_11] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
          scores_max_clear_1[i_11] = max(scores_max_clear_1[i_11], acc_s[((((rv_2 & 7) * 4) + (i_11 * 2)) + (rv_2 >> 3))]);
        }
        scores_max_clear_1[i_11] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_11]);
        scores_max[i_11] = max(scores_max[i_11], scores_max_clear_1[i_11]);
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        scores_max[i_12] = max(scores_max[i_12], scores_max_prev[i_12]);
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 32; ++i_13) {
        acc_s[i_13] = exp2f(((acc_s[i_13] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_13 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 8; ++i_14) {
        uint2 __2;
        float4 v__1 = *(float4*)(acc_s + (i_14 * 4));
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
        ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
        *(uint2*)(acc_s_cast + (i_14 * 4)) = __2;
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        scores_scale[i_15] = exp2f(((scores_max_prev[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
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
        logsum[i_17] = ((logsum[i_17] * scores_scale[i_17]) + scores_sum[i_17]);
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 64; ++i_18) {
        acc_o[i_18] = (acc_o[i_18] * scores_scale[((i_18 & 3) >> 1)]);
      }
    }
    overlap_plan_mbar[3].wait(1);
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_3, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_3, 16384);
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
    overlap_plan_mbar[7].arrive();
    #pragma unroll
    for (int i_19 = 0; i_19 < 64; ++i_19) {
      acc_o[i_19] = (acc_o[i_19] / logsum[((i_19 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 8; ++i_20) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_20 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_20 * 8)]), ((half_t)acc_o[((i_20 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_20 * 8) + 2)]), ((half_t)acc_o[((i_20 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_20 * 8) + 4)]), ((half_t)acc_o[((i_20 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_20 * 8) + 6)]), ((half_t)acc_o[((i_20 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[4].arrive();
  } else {
    if (((int)threadIdx.x) < 384) {
      tl::warpgroup_reg_dealloc<24>();
      for (int k_1 = 0; k_1 < 128; ++k_1) {
        if (2 <= k_1) {
          overlap_plan_mbar[((k_1 & 1) + 6)].wait((((k_1 >> 1) + 1) & 1));
        }
        if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
          overlap_plan_mbar[((k_1 & 1) + 2)].arrive_and_expect_tx(16384);
          tl::tma_load(V_desc, overlap_plan_mbar[((k_1 & 1) + 2)], (&(((half_t*)V_shared)[((k_1 & 1) * 8192)])), 0, (k_1 * 64), ((int)blockIdx.y), 0);
          tl::tma_load(V_desc, overlap_plan_mbar[((k_1 & 1) + 2)], (&(((half_t*)V_shared)[(((k_1 & 1) * 8192) + 4096)])), 64, (k_1 * 64), ((int)blockIdx.y), 0);
        }
      }
    } else {
      tl::warpgroup_reg_dealloc<24>();
      overlap_plan_mbar[4].wait(0);
      if (tl::tl_shuffle_elect<128>()) {
        tl::tma_store((&(Output[((((int)blockIdx.y) * 1048576) + (((int)blockIdx.x) * 16384))])), (&(((half_t*)O_shared)[0])), 32768);
        tl::tma_store_arrive();
        tl::tma_store_wait<0, true>();
      }
    }
  }
}

