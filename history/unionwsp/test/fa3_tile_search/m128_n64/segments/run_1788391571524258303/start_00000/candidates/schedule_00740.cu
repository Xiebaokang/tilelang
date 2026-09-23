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

extern "C" __global__ void main_kernel(const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 81920));
  __shared__ __align__(16) uint64_t program_schedule_mbar_mem[2];
  auto program_schedule_mbar = reinterpret_cast<Barrier*>(program_schedule_mbar_mem);
  float acc_o[128];
  float logsum[4];
  float scores_max[4];
  __shared__ __align__(16) uint64_t pipeline_mbar_mem[3];
  auto pipeline_mbar = reinterpret_cast<Barrier*>(pipeline_mbar_mem);
  float scores_max_prev[4];
  float acc_s[64];
  float scores_max_clear[4];
  float scores_scale[4];
  half_t acc_s_cast[64];
  float scores_sum[4];
  float scores_max_clear_1[4];
  float scores_max_clear_2[4];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(V_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    program_schedule_mbar[0].init(128);
    program_schedule_mbar[1].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      *(uint4*)(((half_t*)Q_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + ((((((int)blockIdx.y) * 1048576) + (((int)blockIdx.x) * 16384)) + (i * 1024)) + (((int)threadIdx.x) * 8)));
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; ++i_1) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(logsum + 0) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    float broadcast_var_2 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    if (tl::tl_shuffle_elect<0>()) {
      pipeline_mbar[0].init(1);
      pipeline_mbar[1].init(1);
      pipeline_mbar[2].init(1);
    }
    tl::fence_barrier_init();
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>()) {
      pipeline_mbar[0].arrive_and_expect_tx(16384);
      tl::fence_proxy_async();
      tl::tma_load(V_desc, pipeline_mbar[0], (&(((half_t*)V_shared)[0])), 0, 0, ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, pipeline_mbar[0], (&(((half_t*)V_shared)[4096])), 64, 0, ((int)blockIdx.y), 0);
      pipeline_mbar[1].arrive_and_expect_tx(16384);
      tl::fence_proxy_async();
      tl::tma_load(V_desc, pipeline_mbar[1], (&(((half_t*)V_shared)[8192])), 0, 64, ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, pipeline_mbar[1], (&(((half_t*)V_shared)[12288])), 64, 64, ((int)blockIdx.y), 0);
    }
    *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
    #pragma unroll
    for (int i_2 = 0; i_2 < 64; ++i_2) {
      acc_s[i_2] = 0x0p+0f/*0.000000e+00*/;
    }
    program_schedule_mbar[0].wait(0);
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + (((((ki >> 2) * 16384) + (i_3 * 8192)) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + (i_3 * 32))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
    }
    program_schedule_mbar[1].arrive();
    float broadcast_var_3 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      scores_max_clear[i_4] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        scores_max_clear[i_4] = max(scores_max_clear[i_4], acc_s[(((((i_4 >> 1) * 32) + ((rv & 7) * 4)) + ((i_4 & 1) * 2)) + (rv >> 3))]);
      }
      scores_max_clear[i_4] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear[i_4]);
      scores_max[i_4] = max(scores_max[i_4], scores_max_clear[i_4]);
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      scores_max[i_5] = max(scores_max[i_5], scores_max_prev[i_5]);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      scores_scale[i_6] = exp2f(((scores_max_prev[i_6] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_6] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 64; ++i_7) {
      acc_s[i_7] = exp2f(((acc_s[i_7] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[(((i_7 >> 5) * 2) + ((i_7 & 3) >> 1))] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      uint2 __1;
      float4 v_ = *(float4*)(acc_s + (((((i_8 & 3) >> 1) * 32) + ((i_8 >> 2) * 8)) + ((i_8 & 1) * 4)));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(acc_s_cast + (i_8 * 4)) = __1;
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      scores_sum[i_9] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        scores_sum[i_9] = (scores_sum[i_9] + acc_s[(((((i_9 >> 1) * 32) + ((rv_1 & 7) * 4)) + ((i_9 & 1) * 2)) + (rv_1 >> 3))]);
      }
      scores_sum[i_9] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_9]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      logsum[i_10] = ((logsum[i_10] * scores_scale[i_10]) + scores_sum[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 128; ++i_11) {
      acc_o[i_11] = (acc_o[i_11] * scores_scale[(((i_11 >> 6) * 2) + ((i_11 & 3) >> 1))]);
    }
    tl::__sync_thread_partial(3, 128);
    for (int k = 0; k < 126; ++k) {
      if (tl::tl_shuffle_elect<128>()) {
        pipeline_mbar[((k + 2) % 3)].arrive_and_expect_tx(16384);
        tl::tma_load(V_desc, pipeline_mbar[((k + 2) % 3)], (&(((half_t*)V_shared)[(((k + 2) % 3) * 8192)])), 0, ((k * 64) + 128), ((int)blockIdx.y), 0);
        tl::tma_load(V_desc, pipeline_mbar[((k + 2) % 3)], (&(((half_t*)V_shared)[((((k + 2) % 3) * 8192) + 4096)])), 64, ((k * 64) + 128), ((int)blockIdx.y), 0);
      }
      pipeline_mbar[(k % 3)].wait(((k % 6) / 3));
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_1, (&(((half_t*)V_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, ((k % 3) * 16384));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int i_12 = 0; i_12 < 2; ++i_12) {
          #pragma unroll
          for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
            tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + ((ki_1 * 16) + (i_12 * 8))), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + (i_12 * 64)), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
      }
      *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
      #pragma unroll
      for (int i_13 = 0; i_13 < 64; ++i_13) {
        acc_s[i_13] = 0x0p+0f/*0.000000e+00*/;
      }
      program_schedule_mbar[0].wait(((k + 1) & 1));
      {
        tl::GmmaDescriptor desc_a_1;
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)K_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int i_14 = 0; i_14 < 2; ++i_14) {
          #pragma unroll
          for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
            tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_1 + (((((ki_2 >> 2) * 16384) + (i_14 * 8192)) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + (i_14 * 32))), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
      }
      program_schedule_mbar[1].arrive();
      float broadcast_var_4 = -CUDART_INF_F;
      *(float4*)(scores_max + 0) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
      #pragma unroll
      for (int i_15 = 0; i_15 < 4; ++i_15) {
        scores_max_clear_1[i_15] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
          scores_max_clear_1[i_15] = max(scores_max_clear_1[i_15], acc_s[(((((i_15 >> 1) * 32) + ((rv_2 & 7) * 4)) + ((i_15 & 1) * 2)) + (rv_2 >> 3))]);
        }
        scores_max_clear_1[i_15] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_1[i_15]);
        scores_max[i_15] = max(scores_max[i_15], scores_max_clear_1[i_15]);
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 4; ++i_16) {
        scores_max[i_16] = max(scores_max[i_16], scores_max_prev[i_16]);
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 4; ++i_17) {
        scores_scale[i_17] = exp2f(((scores_max_prev[i_17] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_17] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 64; ++i_18) {
        acc_s[i_18] = exp2f(((acc_s[i_18] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[(((i_18 >> 5) * 2) + ((i_18 & 3) >> 1))] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 16; ++i_19) {
        uint2 __2;
        float4 v__1 = *(float4*)(acc_s + (((((i_19 & 3) >> 1) * 32) + ((i_19 >> 2) * 8)) + ((i_19 & 1) * 4)));
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
        ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
        *(uint2*)(acc_s_cast + (i_19 * 4)) = __2;
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 4; ++i_20) {
        scores_sum[i_20] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
          scores_sum[i_20] = (scores_sum[i_20] + acc_s[(((((i_20 >> 1) * 32) + ((rv_3 & 7) * 4)) + ((i_20 & 1) * 2)) + (rv_3 >> 3))]);
        }
        scores_sum[i_20] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_20]);
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 4; ++i_21) {
        logsum[i_21] = ((logsum[i_21] * scores_scale[i_21]) + scores_sum[i_21]);
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 128; ++i_22) {
        acc_o[i_22] = (acc_o[i_22] * scores_scale[(((i_22 >> 6) * 2) + ((i_22 & 3) >> 1))]);
      }
    }
    pipeline_mbar[0].wait(0);
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_3, (&(((half_t*)V_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_23 = 0; i_23 < 2; ++i_23) {
        #pragma unroll
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + ((ki_3 * 16) + (i_23 * 8))), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + (i_23 * 64)), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
    }
    *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
    #pragma unroll
    for (int i_24 = 0; i_24 < 64; ++i_24) {
      acc_s[i_24] = 0x0p+0f/*0.000000e+00*/;
    }
    program_schedule_mbar[0].wait(1);
    {
      tl::GmmaDescriptor desc_a_2;
      tl::GmmaDescriptor desc_b_4;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_4, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_25 = 0; i_25 < 2; ++i_25) {
        #pragma unroll
        for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_2 + (((((ki_4 >> 2) * 16384) + (i_25 * 8192)) + ((ki_4 & 3) * 32)) >> 4)), uint64_t(desc_b_4 + ((((ki_4 >> 2) * 8192) + ((ki_4 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + (i_25 * 32))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
    }
    program_schedule_mbar[1].arrive();
    float broadcast_var_5 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
    #pragma unroll
    for (int i_26 = 0; i_26 < 4; ++i_26) {
      scores_max_clear_2[i_26] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
        scores_max_clear_2[i_26] = max(scores_max_clear_2[i_26], acc_s[(((((i_26 >> 1) * 32) + ((rv_4 & 7) * 4)) + ((i_26 & 1) * 2)) + (rv_4 >> 3))]);
      }
      scores_max_clear_2[i_26] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_2[i_26]);
      scores_max[i_26] = max(scores_max[i_26], scores_max_clear_2[i_26]);
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 4; ++i_27) {
      scores_max[i_27] = max(scores_max[i_27], scores_max_prev[i_27]);
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 4; ++i_28) {
      scores_scale[i_28] = exp2f(((scores_max_prev[i_28] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_28] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 64; ++i_29) {
      acc_s[i_29] = exp2f(((acc_s[i_29] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[(((i_29 >> 5) * 2) + ((i_29 & 3) >> 1))] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 16; ++i_30) {
      uint2 __3;
      float4 v__2 = *(float4*)(acc_s + (((((i_30 & 3) >> 1) * 32) + ((i_30 >> 2) * 8)) + ((i_30 & 1) * 4)));
      ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
      ((half2*)(&__3))[1] = __float22half2_rn(((float2*)(&v__2))[1]);
      *(uint2*)(acc_s_cast + (i_30 * 4)) = __3;
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 4; ++i_31) {
      scores_sum[i_31] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_5 = 0; rv_5 < 16; ++rv_5) {
        scores_sum[i_31] = (scores_sum[i_31] + acc_s[(((((i_31 >> 1) * 32) + ((rv_5 & 7) * 4)) + ((i_31 & 1) * 2)) + (rv_5 >> 3))]);
      }
      scores_sum[i_31] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_31]);
    }
    #pragma unroll
    for (int i_32 = 0; i_32 < 4; ++i_32) {
      logsum[i_32] = ((logsum[i_32] * scores_scale[i_32]) + scores_sum[i_32]);
    }
    #pragma unroll
    for (int i_33 = 0; i_33 < 128; ++i_33) {
      acc_o[i_33] = (acc_o[i_33] * scores_scale[(((i_33 >> 6) * 2) + ((i_33 & 3) >> 1))]);
    }
    pipeline_mbar[1].wait(0);
    {
      tl::GmmaDescriptor desc_b_5;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_5, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_5, 16384);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_34 = 0; i_34 < 2; ++i_34) {
        #pragma unroll
        for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + ((ki_5 * 16) + (i_34 * 8))), uint64_t(desc_b_5 + ((ki_5 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + (i_34 * 64)), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
    }
    #pragma unroll
    for (int i_35 = 0; i_35 < 128; ++i_35) {
      acc_o[i_35] = (acc_o[i_35] / logsum[(((i_35 >> 6) * 2) + ((i_35 & 3) >> 1))]);
    }
    #pragma unroll
    for (int i_36 = 0; i_36 < 16; ++i_36) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[((((((i_36 >> 3) * 8192) + ((((int)threadIdx.x) >> 5) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + ((i_36 & 7) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_36 * 8)]), ((half_t)acc_o[((i_36 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_36 * 8) + 2)]), ((half_t)acc_o[((i_36 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_36 * 8) + 4)]), ((half_t)acc_o[((i_36 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_36 * 8) + 6)]), ((half_t)acc_o[((i_36 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::fence_proxy_async();
      tl::tma_store((&(Output[((((int)blockIdx.y) * 1048576) + (((int)blockIdx.x) * 16384))])), (&(((half_t*)O_shared)[0])), 32768);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  } else {
    tl::warpgroup_reg_dealloc<24>();
    for (int k_1 = 0; k_1 < 128; ++k_1) {
      if (1 <= k_1) {
        program_schedule_mbar[1].wait(((k_1 + 1) & 1));
      }
      #pragma unroll
      for (int i_37 = 0; i_37 < 8; ++i_37) {
        *(uint4*)(((half_t*)K_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_37 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((((int)blockIdx.y) * 1048576) + (k_1 * 8192)) + (i_37 * 1024)) + (((int)threadIdx.x) * 8)) - 1024));
      }
      tl::fence_proxy_async();
      program_schedule_mbar[0].arrive();
    }
  }
}

