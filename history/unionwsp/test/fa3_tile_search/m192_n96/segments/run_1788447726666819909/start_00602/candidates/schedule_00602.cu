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

extern "C" __global__ void main_kernel(const half_t* __restrict__ K, __grid_constant__ const CUtensorMap Output_desc, const half_t* __restrict__ Q, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(512, 1) main_kernel(const half_t* __restrict__ K, __grid_constant__ const CUtensorMap Output_desc, const half_t* __restrict__ Q, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 122880));
  __shared__ __align__(16) uint64_t program_schedule_mbar_mem[2];
  auto program_schedule_mbar = reinterpret_cast<Barrier*>(program_schedule_mbar_mem);
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  __shared__ __align__(16) uint64_t pipeline_mbar_mem[2];
  auto pipeline_mbar = reinterpret_cast<Barrier*>(pipeline_mbar_mem);
  float scores_max_prev[2];
  float acc_s[48];
  float scores_scale[2];
  half_t acc_s_cast[48];
  float scores_sum[2];
  float scores_max_clear[2];
  float scores_max_clear_1[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(V_desc);
    tl::prefetch_tma_descriptor(Output_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    program_schedule_mbar[0].init(128);
    program_schedule_mbar[1].init(384);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 384) {
    tl::warpgroup_reg_alloc<160>();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval;
      if ((((((int)blockIdx.x) * 3) + (((i * 3) + (((int)threadIdx.x) >> 7)) >> 3)) < 128)) {
        condval = *(uint4*)(Q + ((((((int)blockIdx.y) * 1048576) + (((int)blockIdx.x) * 24576)) + (i * 3072)) + (((int)threadIdx.x) * 8)));
      } else {
        condval = make_uint4(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
      }
      *(uint4*)(((half_t*)Q_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 12288) + (i * 1536)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(logsum + 0) = make_float2(broadcast_var_2, broadcast_var_2);
    float broadcast_var_3 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
    if (tl::tl_shuffle_elect<0>()) {
      pipeline_mbar[0].init(1);
      pipeline_mbar[1].init(1);
    }
    tl::fence_barrier_init();
    tl::__sync_thread_partial(3, 384);
    if (tl::tl_shuffle_elect<384>()) {
      pipeline_mbar[0].arrive_and_expect_tx(24576);
      tl::fence_proxy_async();
      tl::tma_load(V_desc, pipeline_mbar[0], (&(((half_t*)V_shared)[0])), 0, 0, ((int)blockIdx.y), 0);
      tl::tma_load(V_desc, pipeline_mbar[0], (&(((half_t*)V_shared)[6144])), 64, 0, ((int)blockIdx.y), 0);
    }
    tl::fence_proxy_async();
    for (int k = 0; k < 85; ++k) {
      if (tl::tl_shuffle_elect<384>()) {
        pipeline_mbar[((k + 1) & 1)].arrive_and_expect_tx(24576);
        tl::tma_load(V_desc, pipeline_mbar[((k + 1) & 1)], (&(((half_t*)V_shared)[(((k + 1) % 3) * 12288)])), 0, ((k * 96) + 96), ((int)blockIdx.y), 0);
        tl::tma_load(V_desc, pipeline_mbar[((k + 1) & 1)], (&(((half_t*)V_shared)[((((k + 1) % 3) * 12288) + 6144)])), 64, ((k * 96) + 96), ((int)blockIdx.y), 0);
      }
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      #pragma unroll
      for (int i_2 = 0; i_2 < 48; ++i_2) {
        acc_s[i_2] = 0x0p+0f/*0.000000e+00*/;
      }
      program_schedule_mbar[0].wait((k & 1));
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)K_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 144);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int warp_j = 0; warp_j < 3; ++warp_j) {
          #pragma unroll
          for (int ki = 0; ki < 8; ++ki) {
            tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a + (((((ki >> 2) * 24576) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + (((((ki >> 2) * 12288) + (warp_j * 4096)) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + (warp_j * 16))), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 144);
      }
      program_schedule_mbar[1].arrive();
      float broadcast_var_4 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_4, broadcast_var_4);
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        scores_max_clear[i_3] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 24; ++rv) {
          scores_max_clear[i_3] = max(scores_max_clear[i_3], acc_s[((((rv % 12) * 4) + (i_3 * 2)) + (rv / 12))]);
        }
        scores_max_clear[i_3] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<384>>::run(scores_max_clear[i_3]);
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
      for (int i_6 = 0; i_6 < 48; ++i_6) {
        acc_s[i_6] = exp2f(((acc_s[i_6] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_6 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 12; ++i_7) {
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
        for (int rv_1 = 0; rv_1 < 24; ++rv_1) {
          scores_sum[i_8] = (scores_sum[i_8] + acc_s[((((rv_1 % 12) * 4) + (i_8 * 2)) + (rv_1 / 12))]);
        }
        scores_sum[i_8] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<384>>::run(scores_sum[i_8]);
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        logsum[i_9] = ((logsum[i_9] * scores_scale[i_9]) + scores_sum[i_9]);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 64; ++i_10) {
        acc_o[i_10] = (acc_o[i_10] * scores_scale[((i_10 & 3) >> 1)]);
      }
      pipeline_mbar[(k & 1)].wait(((k & 3) >> 1));
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 768, 64>(desc_b_1, (&(((half_t*)V_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, ((k % 3) * 24576));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 24);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 192);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 6; ++ki_1) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 192);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 24);
      }
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    #pragma unroll
    for (int i_11 = 0; i_11 < 48; ++i_11) {
      float condval_1;
      if ((16 <= i_11)) {
        condval_1 = -CUDART_INF_F;
      } else {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      }
      acc_s[i_11] = condval_1;
    }
    program_schedule_mbar[0].wait(1);
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)K_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 144);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int warp_j_1 = 0; warp_j_1 < 3; ++warp_j_1) {
        #pragma unroll
        for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_1 + (((((ki_2 >> 2) * 24576) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + (((((ki_2 >> 2) * 12288) + (warp_j_1 * 4096)) + ((ki_2 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + (warp_j_1 * 16))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 144);
    }
    program_schedule_mbar[1].arrive();
    float broadcast_var_5 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_5, broadcast_var_5);
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      scores_max_clear_1[i_12] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 24; ++rv_2) {
        scores_max_clear_1[i_12] = max(scores_max_clear_1[i_12], acc_s[((((rv_2 % 12) * 4) + (i_12 * 2)) + (rv_2 / 12))]);
      }
      scores_max_clear_1[i_12] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<384>>::run(scores_max_clear_1[i_12]);
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
    for (int i_15 = 0; i_15 < 48; ++i_15) {
      acc_s[i_15] = exp2f(((acc_s[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_15 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 12; ++i_16) {
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
      for (int rv_3 = 0; rv_3 < 24; ++rv_3) {
        scores_sum[i_17] = (scores_sum[i_17] + acc_s[((((rv_3 % 12) * 4) + (i_17 * 2)) + (rv_3 / 12))]);
      }
      scores_sum[i_17] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<384>>::run(scores_sum[i_17]);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 2; ++i_18) {
      logsum[i_18] = ((logsum[i_18] * scores_scale[i_18]) + scores_sum[i_18]);
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 64; ++i_19) {
      acc_o[i_19] = (acc_o[i_19] * scores_scale[((i_19 & 3) >> 1)]);
    }
    pipeline_mbar[1].wait(0);
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 768, 64>(desc_b_3, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_3, 24576);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 24);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 192);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 6; ++ki_3) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_3 * 8)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 192);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 24);
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 64; ++i_20) {
      acc_o[i_20] = (acc_o[i_20] / logsum[((i_20 & 3) >> 1)]);
    }
    tl::__sync_thread_partial(3, 384);
    #pragma unroll
    for (int i_21 = 0; i_21 < 8; ++i_21) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((i_21 >> 2) * 12288) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_21 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_o[(i_21 * 8)]), ((half_t)acc_o[((i_21 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_21 * 8) + 2)]), ((half_t)acc_o[((i_21 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_21 * 8) + 4)]), ((half_t)acc_o[((i_21 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_21 * 8) + 6)]), ((half_t)acc_o[((i_21 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 384);
    if (tl::tl_shuffle_elect<384>()) {
      tl::fence_proxy_async();
      tl::tma_store(Output_desc, (&(((half_t*)O_shared)[0])), 0, (((int)blockIdx.x) * 192), ((int)blockIdx.y), 0);
      tl::tma_store(Output_desc, (&(((half_t*)O_shared)[12288])), 64, (((int)blockIdx.x) * 192), ((int)blockIdx.y), 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  } else {
    tl::warpgroup_reg_dealloc<24>();
    for (int k_1 = 0; k_1 < 86; ++k_1) {
      if (1 <= k_1) {
        program_schedule_mbar[1].wait(((k_1 + 1) & 1));
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 12; ++i_22) {
        half_t broadcast_var_6 = half_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_2;
        if ((((k_1 * 12) + i_22) < 1024)) {
          condval_2 = *(uint4*)(K + (((((((int)blockIdx.y) * 1048576) + (k_1 * 12288)) + (i_22 * 1024)) + (((int)threadIdx.x) * 8)) - 3072));
        } else {
          condval_2 = make_uint4(__pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6));
        }
        *(uint4*)(((half_t*)K_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 6144) + (i_22 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
      }
      tl::fence_proxy_async();
      program_schedule_mbar[0].arrive();
    }
  }
}

