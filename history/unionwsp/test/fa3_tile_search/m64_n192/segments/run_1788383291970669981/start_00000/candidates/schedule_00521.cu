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
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  __shared__ __align__(16) uint64_t program_schedule_mbar_mem[5];
  auto program_schedule_mbar = reinterpret_cast<Barrier*>(program_schedule_mbar_mem);
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  __shared__ __align__(16) uint64_t pipeline_mbar_mem[3];
  auto pipeline_mbar = reinterpret_cast<Barrier*>(pipeline_mbar_mem);
  float scores_max_prev[2];
  float acc_s[96];
  float scores_max_clear[2];
  float scores_scale[2];
  half_t acc_s_cast[96];
  float scores_sum[2];
  float scores_max_clear_1[2];
  float scores_max_clear_2[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(K_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    program_schedule_mbar[0].init(128);
    program_schedule_mbar[1].init(128);
    program_schedule_mbar[2].init(128);
    program_schedule_mbar[3].init(128);
    program_schedule_mbar[4].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      *(uint4*)(((half_t*)Q_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + ((((((int)blockIdx.y) * 1048576) + (((int)blockIdx.x) * 8192)) + (i * 1024)) + (((int)threadIdx.x) * 8)));
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(logsum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
    float broadcast_var_2 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_2, broadcast_var_2);
    if (tl::tl_shuffle_elect<0>()) {
      pipeline_mbar[0].init(1);
      pipeline_mbar[1].init(1);
      pipeline_mbar[2].init(1);
    }
    tl::fence_barrier_init();
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>()) {
      pipeline_mbar[0].arrive_and_expect_tx(49152);
      tl::fence_proxy_async();
      tl::tma_load(K_desc, pipeline_mbar[0], (&(((half_t*)K_shared)[0])), 0, 0, ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, pipeline_mbar[0], (&(((half_t*)K_shared)[12288])), 64, 0, ((int)blockIdx.y), 0);
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    #pragma unroll
    for (int i_2 = 0; i_2 < 96; ++i_2) {
      acc_s[i_2] = 0x0p+0f/*0.000000e+00*/;
    }
    pipeline_mbar[0].wait(0);
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 96);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int warp_j = 0; warp_j < 3; ++warp_j) {
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + (((((ki >> 2) * 24576) + (warp_j * 8192)) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + (warp_j * 32))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 96);
    }
    float broadcast_var_3 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      scores_max_clear[i_3] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 48; ++rv) {
        scores_max_clear[i_3] = max(scores_max_clear[i_3], acc_s[((((rv % 24) * 4) + (i_3 * 2)) + (rv / 24))]);
      }
      scores_max_clear[i_3] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear[i_3]);
      scores_max[i_3] = max(scores_max[i_3], scores_max_clear[i_3]);
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      scores_max[i_4] = max(scores_max[i_4], scores_max_prev[i_4]);
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      scores_scale[i_5] = exp2f(((scores_max_prev[i_5] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_5] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 96; ++i_6) {
      acc_s[i_6] = exp2f(((acc_s[i_6] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_6 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 24; ++i_7) {
      uint2 __1;
      float4 v_ = *(float4*)(acc_s + (i_7 * 4));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(acc_s_cast + (i_7 * 4)) = __1;
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      scores_sum[i_8] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 48; ++rv_1) {
        scores_sum[i_8] = (scores_sum[i_8] + acc_s[((((rv_1 % 24) * 4) + (i_8 * 2)) + (rv_1 / 24))]);
      }
      scores_sum[i_8] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_8]);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      logsum[i_9] = ((logsum[i_9] * scores_scale[i_9]) + scores_sum[i_9]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 64; ++i_10) {
      acc_o[i_10] = (acc_o[i_10] * scores_scale[((i_10 & 3) >> 1)]);
    }
    for (int k = 0; k < 41; ++k) {
      if (tl::tl_shuffle_elect<128>()) {
        pipeline_mbar[((k + 1) % 3)].arrive_and_expect_tx(49152);
        tl::tma_load(K_desc, pipeline_mbar[((k + 1) % 3)], (&(((half_t*)K_shared)[0])), 0, ((k * 192) + 192), ((int)blockIdx.y), 0);
        tl::tma_load(K_desc, pipeline_mbar[((k + 1) % 3)], (&(((half_t*)K_shared)[12288])), 64, ((k * 192) + 192), ((int)blockIdx.y), 0);
      }
      program_schedule_mbar[(k & 1)].wait(((k & 3) >> 1));
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 1536, 64>(desc_b_1, (&(((half_t*)V_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, ((k & 1) * 49152));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 48);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 12; ++ki_1) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 48);
      }
      program_schedule_mbar[((k & 1) + 3)].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      #pragma unroll
      for (int i_11 = 0; i_11 < 96; ++i_11) {
        acc_s[i_11] = 0x0p+0f/*0.000000e+00*/;
      }
      pipeline_mbar[((k + 1) % 3)].wait((((k + 1) % 6) / 3));
      {
        tl::GmmaDescriptor desc_a_1;
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)K_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 96);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int warp_j_1 = 0; warp_j_1 < 3; ++warp_j_1) {
          #pragma unroll
          for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
            tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_1 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + (((((ki_2 >> 2) * 24576) + (warp_j_1 * 8192)) + ((ki_2 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + (warp_j_1 * 32))), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 96);
      }
      float broadcast_var_4 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_4, broadcast_var_4);
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        scores_max_clear_1[i_12] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 48; ++rv_2) {
          scores_max_clear_1[i_12] = max(scores_max_clear_1[i_12], acc_s[((((rv_2 % 24) * 4) + (i_12 * 2)) + (rv_2 / 24))]);
        }
        scores_max_clear_1[i_12] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_1[i_12]);
        scores_max[i_12] = max(scores_max[i_12], scores_max_clear_1[i_12]);
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        scores_max[i_13] = max(scores_max[i_13], scores_max_prev[i_13]);
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        scores_scale[i_14] = exp2f(((scores_max_prev[i_14] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_14] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 96; ++i_15) {
        acc_s[i_15] = exp2f(((acc_s[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_15 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 24; ++i_16) {
        uint2 __2;
        float4 v__1 = *(float4*)(acc_s + (i_16 * 4));
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
        ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
        *(uint2*)(acc_s_cast + (i_16 * 4)) = __2;
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        scores_sum[i_17] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 48; ++rv_3) {
          scores_sum[i_17] = (scores_sum[i_17] + acc_s[((((rv_3 % 24) * 4) + (i_17 * 2)) + (rv_3 / 24))]);
        }
        scores_sum[i_17] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_17]);
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 2; ++i_18) {
        logsum[i_18] = ((logsum[i_18] * scores_scale[i_18]) + scores_sum[i_18]);
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 64; ++i_19) {
        acc_o[i_19] = (acc_o[i_19] * scores_scale[((i_19 & 3) >> 1)]);
      }
    }
    if (tl::tl_shuffle_elect<128>()) {
      pipeline_mbar[0].arrive_and_expect_tx(49152);
      tl::tma_load(K_desc, pipeline_mbar[0], (&(((half_t*)K_shared)[0])), 0, 8064, ((int)blockIdx.y), 0);
      tl::tma_load(K_desc, pipeline_mbar[0], (&(((half_t*)K_shared)[12288])), 64, 8064, ((int)blockIdx.y), 0);
    }
    program_schedule_mbar[1].wait(0);
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 1536, 64>(desc_b_3, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_3, 49152);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 48);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 12; ++ki_3) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_3 * 8)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 48);
    }
    program_schedule_mbar[4].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    #pragma unroll
    for (int i_20 = 0; i_20 < 96; ++i_20) {
      float condval;
      if ((64 <= i_20)) {
        condval = -CUDART_INF_F;
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      acc_s[i_20] = condval;
    }
    pipeline_mbar[0].wait(0);
    {
      tl::GmmaDescriptor desc_a_2;
      tl::GmmaDescriptor desc_b_4;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_4, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 96);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int warp_j_2 = 0; warp_j_2 < 3; ++warp_j_2) {
        #pragma unroll
        for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_2 + ((((ki_4 >> 2) * 8192) + ((ki_4 & 3) * 32)) >> 4)), uint64_t(desc_b_4 + (((((ki_4 >> 2) * 24576) + (warp_j_2 * 8192)) + ((ki_4 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + (warp_j_2 * 32))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 96);
    }
    float broadcast_var_5 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_5, broadcast_var_5);
    #pragma unroll
    for (int i_21 = 0; i_21 < 2; ++i_21) {
      scores_max_clear_2[i_21] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_4 = 0; rv_4 < 48; ++rv_4) {
        scores_max_clear_2[i_21] = max(scores_max_clear_2[i_21], acc_s[((((rv_4 % 24) * 4) + (i_21 * 2)) + (rv_4 / 24))]);
      }
      scores_max_clear_2[i_21] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_max_clear_2[i_21]);
      scores_max[i_21] = max(scores_max[i_21], scores_max_clear_2[i_21]);
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 2; ++i_22) {
      scores_max[i_22] = max(scores_max[i_22], scores_max_prev[i_22]);
    }
    #pragma unroll
    for (int i_23 = 0; i_23 < 2; ++i_23) {
      scores_scale[i_23] = exp2f(((scores_max_prev[i_23] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_23] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 96; ++i_24) {
      acc_s[i_24] = exp2f(((acc_s[i_24] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_24 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 24; ++i_25) {
      uint2 __3;
      float4 v__2 = *(float4*)(acc_s + (i_25 * 4));
      ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
      ((half2*)(&__3))[1] = __float22half2_rn(((float2*)(&v__2))[1]);
      *(uint2*)(acc_s_cast + (i_25 * 4)) = __3;
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 2; ++i_26) {
      scores_sum[i_26] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_5 = 0; rv_5 < 48; ++rv_5) {
        scores_sum[i_26] = (scores_sum[i_26] + acc_s[((((rv_5 % 24) * 4) + (i_26 * 2)) + (rv_5 / 24))]);
      }
      scores_sum[i_26] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(scores_sum[i_26]);
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 2; ++i_27) {
      logsum[i_27] = ((logsum[i_27] * scores_scale[i_27]) + scores_sum[i_27]);
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 64; ++i_28) {
      acc_o[i_28] = (acc_o[i_28] * scores_scale[((i_28 & 3) >> 1)]);
    }
    program_schedule_mbar[0].wait(1);
    {
      tl::GmmaDescriptor desc_b_5;
      tl::initialize_wgmma_descriptor<1, 1536, 64>(desc_b_5, (&(((half_t*)V_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 48);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_5 = 0; ki_5 < 12; ++ki_5) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_5 * 8)), uint64_t(desc_b_5 + ((ki_5 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 48);
    }
    program_schedule_mbar[3].arrive();
    #pragma unroll
    for (int i_29 = 0; i_29 < 64; ++i_29) {
      acc_o[i_29] = (acc_o[i_29] / logsum[((i_29 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 8; ++i_30) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_30 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_30 * 8)]), ((half_t)acc_o[((i_30 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_30 * 8) + 2)]), ((half_t)acc_o[((i_30 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_30 * 8) + 4)]), ((half_t)acc_o[((i_30 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_30 * 8) + 6)]), ((half_t)acc_o[((i_30 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    program_schedule_mbar[2].arrive();
  } else {
    tl::warpgroup_reg_dealloc<24>();
    for (int k_1 = 0; k_1 < 43; ++k_1) {
      if (2 <= k_1) {
        program_schedule_mbar[((k_1 & 1) + 3)].wait((((k_1 >> 1) + 1) & 1));
      }
      #pragma unroll
      for (int i_31 = 0; i_31 < 24; ++i_31) {
        half_t broadcast_var_6 = half_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_1;
        if ((((k_1 * 24) + i_31) < 1024)) {
          condval_1 = *(uint4*)(V + (((((((int)blockIdx.y) * 1048576) + (k_1 * 24576)) + (i_31 * 1024)) + (((int)threadIdx.x) * 8)) - 1024));
        } else {
          condval_1 = make_uint4(__pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6));
        }
        *(uint4*)(((half_t*)V_shared) + ((((((((k_1 & 1) * 24576) + (((((int)threadIdx.x) & 15) >> 3) * 12288)) + (i_31 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_1;
      }
      tl::fence_proxy_async();
      program_schedule_mbar[(k_1 & 1)].arrive();
    }
    program_schedule_mbar[2].wait(0);
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store((&(Output[((((int)blockIdx.y) * 1048576) + (((int)blockIdx.x) * 8192))])), (&(((half_t*)O_shared)[0])), 16384);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

