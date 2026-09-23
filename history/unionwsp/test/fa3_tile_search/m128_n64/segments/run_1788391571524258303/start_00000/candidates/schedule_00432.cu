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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  __shared__ __align__(16) uint64_t program_schedule_mbar_mem[4];
  auto program_schedule_mbar = reinterpret_cast<Barrier*>(program_schedule_mbar_mem);
  float acc_o[128];
  float logsum[4];
  float scores_max[4];
  __shared__ __align__(16) uint64_t pipeline_mbar_mem[2];
  auto pipeline_mbar = reinterpret_cast<Barrier*>(pipeline_mbar_mem);
  float scores_max_prev[4];
  float acc_s[64];
  float scores_max_clear[4];
  float scores_scale[4];
  half_t acc_s_cast[64];
  float scores_sum[4];
  float scores_max_clear_1[4];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(K_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    program_schedule_mbar[0].init(128);
    program_schedule_mbar[1].init(128);
    program_schedule_mbar[2].init(128);
    program_schedule_mbar[3].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(logsum + 0) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    float broadcast_var_2 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    if (tl::tl_shuffle_elect<0>()) {
      pipeline_mbar[0].init(1);
      pipeline_mbar[1].init(1);
    }
    tl::fence_barrier_init();
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>()) {
      pipeline_mbar[0].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, pipeline_mbar[0], (&(((half_t*)K_shared)[0])), 0, 0, ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, pipeline_mbar[0], (&(((half_t*)K_shared)[4096])), 64, 0, ((int)blockIdx.y), 0);
    }
    *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
    #pragma unroll
    for (int i_1 = 0; i_1 < 64; ++i_1) {
      acc_s[i_1] = 0x0p+0f/*0.000000e+00*/;
    }
    pipeline_mbar[0].wait(0);
    program_schedule_mbar[0].wait(0);
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + (((((ki >> 2) * 16384) + (i_2 * 8192)) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + (i_2 * 32))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
    }
    float broadcast_var_3 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      scores_max_clear[i_3] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        scores_max_clear[i_3] = max(scores_max_clear[i_3], acc_s[(((((i_3 >> 1) * 32) + ((rv & 7) * 4)) + ((i_3 & 1) * 2)) + (rv >> 3))]);
      }
      scores_max_clear[i_3] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear[i_3]);
      scores_max[i_3] = max(scores_max[i_3], scores_max_clear[i_3]);
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      scores_max[i_4] = max(scores_max[i_4], scores_max_prev[i_4]);
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      scores_scale[i_5] = exp2f(((scores_max_prev[i_5] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_5] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 64; ++i_6) {
      acc_s[i_6] = exp2f(((acc_s[i_6] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[(((i_6 >> 5) * 2) + ((i_6 & 3) >> 1))] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 16; ++i_7) {
      uint2 __1;
      float4 v_ = *(float4*)(acc_s + (((((i_7 & 3) >> 1) * 32) + ((i_7 >> 2) * 8)) + ((i_7 & 1) * 4)));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(acc_s_cast + (i_7 * 4)) = __1;
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      scores_sum[i_8] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        scores_sum[i_8] = (scores_sum[i_8] + acc_s[(((((i_8 >> 1) * 32) + ((rv_1 & 7) * 4)) + ((i_8 & 1) * 2)) + (rv_1 >> 3))]);
      }
      scores_sum[i_8] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_8]);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      logsum[i_9] = ((logsum[i_9] * scores_scale[i_9]) + scores_sum[i_9]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 128; ++i_10) {
      acc_o[i_10] = (acc_o[i_10] * scores_scale[(((i_10 >> 6) * 2) + ((i_10 & 3) >> 1))]);
    }
    for (int k = 0; k < 127; ++k) {
      if (tl::tl_shuffle_elect<128>()) {
        pipeline_mbar[((k + 1) & 1)].arrive_and_expect_tx(16384);
        tl::tma_load(K_desc, pipeline_mbar[((k + 1) & 1)], (&(((half_t*)K_shared)[0])), 0, ((k * 64) + 64), ((int)blockIdx.y), 0);
        tl::tma_load(K_desc, pipeline_mbar[((k + 1) & 1)], (&(((half_t*)K_shared)[4096])), 64, ((k * 64) + 64), ((int)blockIdx.y), 0);
      }
      program_schedule_mbar[1].wait((k & 1));
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_1, (&(((half_t*)V_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int i_11 = 0; i_11 < 2; ++i_11) {
          #pragma unroll
          for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
            tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + ((ki_1 * 16) + (i_11 * 8))), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + (i_11 * 64)), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
      }
      program_schedule_mbar[3].arrive();
      *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
      #pragma unroll
      for (int i_12 = 0; i_12 < 64; ++i_12) {
        acc_s[i_12] = 0x0p+0f/*0.000000e+00*/;
      }
      pipeline_mbar[((k + 1) & 1)].wait((((k + 1) & 3) >> 1));
      {
        tl::GmmaDescriptor desc_a_1;
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)K_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int i_13 = 0; i_13 < 2; ++i_13) {
          #pragma unroll
          for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
            tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_1 + (((((ki_2 >> 2) * 16384) + (i_13 * 8192)) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + (i_13 * 32))), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
      }
      float broadcast_var_4 = -CUDART_INF_F;
      *(float4*)(scores_max + 0) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
      #pragma unroll
      for (int i_14 = 0; i_14 < 4; ++i_14) {
        scores_max_clear_1[i_14] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
          scores_max_clear_1[i_14] = max(scores_max_clear_1[i_14], acc_s[(((((i_14 >> 1) * 32) + ((rv_2 & 7) * 4)) + ((i_14 & 1) * 2)) + (rv_2 >> 3))]);
        }
        scores_max_clear_1[i_14] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_1[i_14]);
        scores_max[i_14] = max(scores_max[i_14], scores_max_clear_1[i_14]);
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 4; ++i_15) {
        scores_max[i_15] = max(scores_max[i_15], scores_max_prev[i_15]);
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 4; ++i_16) {
        scores_scale[i_16] = exp2f(((scores_max_prev[i_16] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_16] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 64; ++i_17) {
        acc_s[i_17] = exp2f(((acc_s[i_17] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[(((i_17 >> 5) * 2) + ((i_17 & 3) >> 1))] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 16; ++i_18) {
        uint2 __2;
        float4 v__1 = *(float4*)(acc_s + (((((i_18 & 3) >> 1) * 32) + ((i_18 >> 2) * 8)) + ((i_18 & 1) * 4)));
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
        ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
        *(uint2*)(acc_s_cast + (i_18 * 4)) = __2;
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 4; ++i_19) {
        scores_sum[i_19] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
          scores_sum[i_19] = (scores_sum[i_19] + acc_s[(((((i_19 >> 1) * 32) + ((rv_3 & 7) * 4)) + ((i_19 & 1) * 2)) + (rv_3 >> 3))]);
        }
        scores_sum[i_19] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_19]);
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 4; ++i_20) {
        logsum[i_20] = ((logsum[i_20] * scores_scale[i_20]) + scores_sum[i_20]);
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 128; ++i_21) {
        acc_o[i_21] = (acc_o[i_21] * scores_scale[(((i_21 >> 6) * 2) + ((i_21 & 3) >> 1))]);
      }
    }
    program_schedule_mbar[1].wait(1);
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_3, (&(((half_t*)V_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        #pragma unroll
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + ((ki_3 * 16) + (i_22 * 8))), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + (i_22 * 64)), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 32);
    }
    program_schedule_mbar[3].arrive();
    #pragma unroll
    for (int i_23 = 0; i_23 < 128; ++i_23) {
      acc_o[i_23] = (acc_o[i_23] / logsum[(((i_23 >> 6) * 2) + ((i_23 & 3) >> 1))]);
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 16; ++i_24) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[((((((i_24 >> 3) * 8192) + ((((int)threadIdx.x) >> 5) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + ((i_24 & 7) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_24 * 8)]), ((half_t)acc_o[((i_24 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_24 * 8) + 2)]), ((half_t)acc_o[((i_24 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_24 * 8) + 4)]), ((half_t)acc_o[((i_24 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_24 * 8) + 6)]), ((half_t)acc_o[((i_24 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    program_schedule_mbar[2].arrive();
  } else {
    tl::warpgroup_reg_dealloc<24>();
    #pragma unroll
    for (int i_25 = 0; i_25 < 16; ++i_25) {
      *(uint4*)(((half_t*)Q_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_25 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((((int)blockIdx.y) * 1048576) + (((int)blockIdx.x) * 16384)) + (i_25 * 1024)) + (((int)threadIdx.x) * 8)) - 1024));
    }
    tl::fence_proxy_async();
    program_schedule_mbar[0].arrive();
    for (int k_1 = 0; k_1 < 128; ++k_1) {
      if (1 <= k_1) {
        program_schedule_mbar[3].wait(((k_1 + 1) & 1));
      }
      #pragma unroll
      for (int i_26 = 0; i_26 < 8; ++i_26) {
        *(uint4*)(((half_t*)V_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_26 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((((int)blockIdx.y) * 1048576) + (k_1 * 8192)) + (i_26 * 1024)) + (((int)threadIdx.x) * 8)) - 1024));
      }
      tl::fence_proxy_async();
      program_schedule_mbar[1].arrive();
    }
    program_schedule_mbar[2].wait(0);
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store((&(Output[((((int)blockIdx.y) * 1048576) + (((int)blockIdx.x) * 16384))])), (&(((half_t*)O_shared)[0])), 32768);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

