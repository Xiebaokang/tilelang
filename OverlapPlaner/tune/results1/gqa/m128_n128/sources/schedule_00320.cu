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
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 98304));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[9];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  float acc_s[64];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  half_t acc_s_cast[64];
  float scores_max_clear[2];
  float scores_max_clear_1[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(Output_desc);
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
  if (((int)threadIdx.x) < 256) {
    tl::warpgroup_reg_alloc<240>();
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
      overlap_plan_mbar[1].arrive_and_expect_tx(32768);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, 0, (((int)blockIdx.y) >> 3), 0);
      tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[8192])), 64, 0, (((int)blockIdx.y) >> 3), 0);
    }
    for (int k = 0; k < 31; ++k) {
      if (1 <= k) {
        overlap_plan_mbar[(((k + 1) & 1) + 5)].wait((((k + 3) & 3) >> 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[(((k + 1) & 1) + 1)].arrive_and_expect_tx(32768);
        tl::tma_load(K_desc, overlap_plan_mbar[(((k + 1) & 1) + 1)], (&(((half_t*)K_shared)[(((k + 1) & 1) * 16384)])), 0, ((k * 128) + 128), (((int)blockIdx.y) >> 3), 0);
        tl::tma_load(K_desc, overlap_plan_mbar[(((k + 1) & 1) + 1)], (&(((half_t*)K_shared)[((((k + 1) & 1) * 16384) + 8192)])), 64, ((k * 128) + 128), (((int)blockIdx.y) >> 3), 0);
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 64; ++i_1) {
        acc_s[i_1] = 0x0p+0f/*0.000000e+00*/;
      }
      if (k == 0) {
        overlap_plan_mbar[0].wait(0);
      }
      overlap_plan_mbar[((k & 1) + 1)].wait(((k & 3) >> 1));
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)K_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b, ((k & 1) * 32768));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 128);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(uint64_t(desc_a + (((((ki >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 16384) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 128);
      }
      overlap_plan_mbar[((k & 1) + 5)].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_3 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        scores_max_clear[i_2] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 32; ++rv) {
          scores_max_clear[i_2] = max(scores_max_clear[i_2], acc_s[((((rv & 15) * 4) + (i_2 * 2)) + (rv >> 4))]);
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
        scores_scale[i_4] = exp2f(((scores_max_prev[i_4] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_4] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 64; ++i_5) {
        acc_s[i_5] = exp2f(((acc_s[i_5] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_5 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        scores_sum[i_6] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
          scores_sum[i_6] = (scores_sum[i_6] + acc_s[((((rv_1 & 15) * 4) + (i_6 * 2)) + (rv_1 >> 4))]);
        }
        scores_sum[i_6] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_6]);
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        logsum[i_7] = ((logsum[i_7] * scores_scale[i_7]) + scores_sum[i_7]);
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 16; ++i_8) {
        uint2 __1;
        float4 v_ = *(float4*)(acc_s + (i_8 * 4));
        ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
        ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
        *(uint2*)(acc_s_cast + (i_8 * 4)) = __1;
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 64; ++i_9) {
        acc_o[i_9] = (acc_o[i_9] * scores_scale[((i_9 & 3) >> 1)]);
      }
      overlap_plan_mbar[((k & 1) + 3)].wait(((k & 3) >> 1));
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_1, (&(((half_t*)V_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, ((k & 1) * 32768));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
      }
      overlap_plan_mbar[((k & 1) + 7)].arrive();
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 64; ++i_10) {
      acc_s[i_10] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[2].wait(1);
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)K_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_2, 32768);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(uint64_t(desc_a_1 + (((((ki_2 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + ((((ki_2 >> 2) * 16384) + ((ki_2 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 128);
    }
    overlap_plan_mbar[6].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_4 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_4, broadcast_var_4);
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      scores_max_clear_1[i_11] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 32; ++rv_2) {
        scores_max_clear_1[i_11] = max(scores_max_clear_1[i_11], acc_s[((((rv_2 & 15) * 4) + (i_11 * 2)) + (rv_2 >> 4))]);
      }
      scores_max_clear_1[i_11] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_11]);
      scores_max[i_11] = max(scores_max[i_11], scores_max_clear_1[i_11]);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      scores_max[i_12] = max(scores_max[i_12], scores_max_prev[i_12]);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      scores_scale[i_13] = exp2f(((scores_max_prev[i_13] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_13] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 64; ++i_14) {
      acc_s[i_14] = exp2f(((acc_s[i_14] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_14 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      scores_sum[i_15] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 32; ++rv_3) {
        scores_sum[i_15] = (scores_sum[i_15] + acc_s[((((rv_3 & 15) * 4) + (i_15 * 2)) + (rv_3 >> 4))]);
      }
      scores_sum[i_15] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_15]);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      logsum[i_16] = ((logsum[i_16] * scores_scale[i_16]) + scores_sum[i_16]);
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 16; ++i_17) {
      uint2 __2;
      float4 v__1 = *(float4*)(acc_s + (i_17 * 4));
      ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
      ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
      *(uint2*)(acc_s_cast + (i_17 * 4)) = __2;
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 64; ++i_18) {
      acc_o[i_18] = (acc_o[i_18] * scores_scale[((i_18 & 3) >> 1)]);
    }
    overlap_plan_mbar[4].wait(1);
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_3, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_3, 32768);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_3 * 8)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
    }
    overlap_plan_mbar[8].arrive();
    #pragma unroll
    for (int i_19 = 0; i_19 < 64; ++i_19) {
      acc_o[i_19] = (acc_o[i_19] / logsum[((i_19 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 8; ++i_20) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((i_20 >> 2) * 8192) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_20 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_20 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_o[(i_20 * 8)]), ((half_t)acc_o[((i_20 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_20 * 8) + 2)]), ((half_t)acc_o[((i_20 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_20 * 8) + 4)]), ((half_t)acc_o[((i_20 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_20 * 8) + 6)]), ((half_t)acc_o[((i_20 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 256);
    if (tl::tl_shuffle_elect<256>()) {
      tl::fence_proxy_async();
      tl::tma_store(Output_desc, (&(((half_t*)O_shared)[0])), 0, (((int)blockIdx.x) * 128), ((int)blockIdx.y), 0);
      tl::tma_store(Output_desc, (&(((half_t*)O_shared)[8192])), 64, (((int)blockIdx.x) * 128), ((int)blockIdx.y), 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  } else {
    tl::warpgroup_reg_dealloc<24>();
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      if (2 <= k_1) {
        overlap_plan_mbar[((k_1 & 1) + 7)].wait((((k_1 >> 1) + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[((k_1 & 1) + 3)].arrive_and_expect_tx(32768);
        tl::tma_load(V_desc, overlap_plan_mbar[((k_1 & 1) + 3)], (&(((half_t*)V_shared)[((k_1 & 1) * 16384)])), 0, (k_1 * 128), (((int)blockIdx.y) >> 3), 0);
        tl::tma_load(V_desc, overlap_plan_mbar[((k_1 & 1) + 3)], (&(((half_t*)V_shared)[(((k_1 & 1) * 16384) + 8192)])), 64, (k_1 * 128), (((int)blockIdx.y) >> 3), 0);
      }
    }
  }
}

