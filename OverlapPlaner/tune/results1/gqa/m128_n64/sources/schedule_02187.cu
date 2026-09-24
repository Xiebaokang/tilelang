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
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* acc_s_wsp_handoff_2 = ((void*)((char*)buf_dyn_shmem + 98304));
  void* acc_s_wsp_handoff_5 = ((void*)((char*)buf_dyn_shmem + 131072));
  void* scores_scale_wsp_handoff_4 = ((void*)((char*)buf_dyn_shmem + 163840));
  void* scores_scale_wsp_handoff_3 = ((void*)((char*)buf_dyn_shmem + 164864));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[14];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[64];
  float logsum[2];
  float acc_s[32];
  half_t acc_s_cast[32];
  float scores_sum[2];
  float scores_scale_v0[2];
  float scores_scale_v1[2];
  float scores_max[2];
  float scores_max_prev[2];
  float scores_max_clear[2];
  float scores_max_clear_1[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(Output_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(256);
    overlap_plan_mbar[4].init(256);
    overlap_plan_mbar[5].init(256);
    overlap_plan_mbar[6].init(256);
    overlap_plan_mbar[7].init(256);
    overlap_plan_mbar[8].init(1);
    overlap_plan_mbar[9].init(1);
    overlap_plan_mbar[10].init(256);
    overlap_plan_mbar[11].init(256);
    overlap_plan_mbar[12].init(256);
    overlap_plan_mbar[13].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    tl::warpgroup_reg_alloc<208>();
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
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      *(float2*)(((float*)acc_s_wsp_handoff_2) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_2 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_2 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s + (i_2 * 2));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[3].arrive();
    overlap_plan_mbar[10].arrive();
    overlap_plan_mbar[7].wait(0);
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      *(float2*)(acc_s + (i_3 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_5) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_3 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_3 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      uint2 __1;
      float4 v_ = *(float4*)(acc_s + (i_4 * 4));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(acc_s_cast + (i_4 * 4)) = __1;
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      scores_sum[i_5] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        scores_sum[i_5] = (scores_sum[i_5] + acc_s[((((rv & 7) * 4) + (i_5 * 2)) + (rv >> 3))]);
      }
      scores_sum[i_5] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_5]);
    }
    overlap_plan_mbar[4].wait(0);
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      scores_scale_v0[i_6] = ((float*)scores_scale_wsp_handoff_3)[((((((int)threadIdx.x) >> 5) * 16) + (i_6 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      logsum[i_7] = ((logsum[i_7] * scores_scale_v0[i_7]) + scores_sum[i_7]);
    }
    for (int k = 0; k < 31; ++k) {
      #pragma unroll
      for (int i_8 = 0; i_8 < 32; ++i_8) {
        acc_s[i_8] = 0x0p+0f/*0.000000e+00*/;
      }
      overlap_plan_mbar[2].wait((k & 1));
      {
        tl::GmmaDescriptor desc_a_1;
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((half_t*)K_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, 16384);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
        tl::warpgroup_arrive();
        tl::__sync_thread_partial(3, 256);
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_1 + (((((ki_1 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_1 & 3) * 32)) >> 4)), uint64_t(desc_b_1 + ((((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 16; ++i_9) {
        *(float2*)(((float*)acc_s_wsp_handoff_2) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_9 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_9 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s + (i_9 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[3].arrive();
      overlap_plan_mbar[11].arrive();
      overlap_plan_mbar[5].wait((k & 1));
      #pragma unroll
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        scores_scale_v0[i_10] = ((float*)scores_scale_wsp_handoff_4)[((((((int)threadIdx.x) >> 5) * 16) + (i_10 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 64; ++i_11) {
        acc_o[i_11] = (acc_o[i_11] * scores_scale_v0[((i_11 & 3) >> 1)]);
      }
      overlap_plan_mbar[8].wait((k & 1));
      {
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_2, (&(((half_t*)V_shared)[0])));
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
      overlap_plan_mbar[12].arrive();
      overlap_plan_mbar[7].wait(1);
      #pragma unroll
      for (int i_12 = 0; i_12 < 16; ++i_12) {
        *(float2*)(acc_s + (i_12 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_5) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_12 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_12 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 8; ++i_13) {
        uint2 __2;
        float4 v__1 = *(float4*)(acc_s + (i_13 * 4));
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
        ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
        *(uint2*)(acc_s_cast + (i_13 * 4)) = __2;
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        scores_sum[i_14] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
          scores_sum[i_14] = (scores_sum[i_14] + acc_s[((((rv_1 & 7) * 4) + (i_14 * 2)) + (rv_1 >> 3))]);
        }
        scores_sum[i_14] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_14]);
      }
      overlap_plan_mbar[4].wait(1);
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        scores_scale_v1[i_15] = ((float*)scores_scale_wsp_handoff_3)[((((((int)threadIdx.x) >> 5) * 16) + (i_15 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 2; ++i_16) {
        logsum[i_16] = ((logsum[i_16] * scores_scale_v1[i_16]) + scores_sum[i_16]);
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 32; ++i_17) {
        acc_s[i_17] = 0x0p+0f/*0.000000e+00*/;
      }
      overlap_plan_mbar[1].wait(((k + 1) & 1));
      {
        tl::GmmaDescriptor desc_a_2;
        tl::GmmaDescriptor desc_b_3;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_3, (&(((half_t*)K_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_2 + (((((ki_3 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_3 & 3) * 32)) >> 4)), uint64_t(desc_b_3 + ((((ki_3 >> 2) * 8192) + ((ki_3 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
      }
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int i_18 = 0; i_18 < 16; ++i_18) {
        *(float2*)(((float*)acc_s_wsp_handoff_2) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_18 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_18 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s + (i_18 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[3].arrive();
      overlap_plan_mbar[10].arrive();
      overlap_plan_mbar[6].wait((k & 1));
      #pragma unroll
      for (int i_19 = 0; i_19 < 2; ++i_19) {
        scores_scale_v1[i_19] = ((float*)scores_scale_wsp_handoff_4)[(((((((int)threadIdx.x) >> 5) * 16) + (i_19 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 128)];
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 64; ++i_20) {
        acc_o[i_20] = (acc_o[i_20] * scores_scale_v1[((i_20 & 3) >> 1)]);
      }
      overlap_plan_mbar[9].wait((k & 1));
      {
        tl::GmmaDescriptor desc_b_4;
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_4, (&(((half_t*)V_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_4, 16384);
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
      overlap_plan_mbar[13].arrive();
      overlap_plan_mbar[7].wait(0);
      #pragma unroll
      for (int i_21 = 0; i_21 < 16; ++i_21) {
        *(float2*)(acc_s + (i_21 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_5) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_21 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_21 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 8; ++i_22) {
        uint2 __3;
        float4 v__2 = *(float4*)(acc_s + (i_22 * 4));
        ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
        ((half2*)(&__3))[1] = __float22half2_rn(((float2*)(&v__2))[1]);
        *(uint2*)(acc_s_cast + (i_22 * 4)) = __3;
      }
      #pragma unroll
      for (int i_23 = 0; i_23 < 2; ++i_23) {
        scores_sum[i_23] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
          scores_sum[i_23] = (scores_sum[i_23] + acc_s[((((rv_2 & 7) * 4) + (i_23 * 2)) + (rv_2 >> 3))]);
        }
        scores_sum[i_23] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_23]);
      }
      overlap_plan_mbar[4].wait(0);
      #pragma unroll
      for (int i_24 = 0; i_24 < 2; ++i_24) {
        scores_scale_v0[i_24] = ((float*)scores_scale_wsp_handoff_3)[((((((int)threadIdx.x) >> 5) * 16) + (i_24 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      #pragma unroll
      for (int i_25 = 0; i_25 < 2; ++i_25) {
        logsum[i_25] = ((logsum[i_25] * scores_scale_v0[i_25]) + scores_sum[i_25]);
      }
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 32; ++i_26) {
      acc_s[i_26] = 0x0p+0f/*0.000000e+00*/;
    }
    overlap_plan_mbar[2].wait(1);
    {
      tl::GmmaDescriptor desc_a_3;
      tl::GmmaDescriptor desc_b_5;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_5, (&(((half_t*)K_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_5, 16384);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
      tl::warpgroup_arrive();
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int ki_5 = 0; ki_5 < 8; ++ki_5) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_3 + (((((ki_5 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_5 & 3) * 32)) >> 4)), uint64_t(desc_b_5 + ((((ki_5 >> 2) * 8192) + ((ki_5 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 64);
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 16; ++i_27) {
      *(float2*)(((float*)acc_s_wsp_handoff_2) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_27 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_27 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s + (i_27 * 2));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[3].arrive();
    overlap_plan_mbar[11].arrive();
    overlap_plan_mbar[5].wait(1);
    #pragma unroll
    for (int i_28 = 0; i_28 < 2; ++i_28) {
      scores_scale_v0[i_28] = ((float*)scores_scale_wsp_handoff_4)[((((((int)threadIdx.x) >> 5) * 16) + (i_28 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 64; ++i_29) {
      acc_o[i_29] = (acc_o[i_29] * scores_scale_v0[((i_29 & 3) >> 1)]);
    }
    overlap_plan_mbar[8].wait(1);
    {
      tl::GmmaDescriptor desc_b_6;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_6, (&(((half_t*)V_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_6 * 8)), uint64_t(desc_b_6 + ((ki_6 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
    }
    overlap_plan_mbar[12].arrive();
    overlap_plan_mbar[7].wait(1);
    #pragma unroll
    for (int i_30 = 0; i_30 < 16; ++i_30) {
      *(float2*)(acc_s + (i_30 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_5) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_30 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_30 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 8; ++i_31) {
      uint2 __4;
      float4 v__3 = *(float4*)(acc_s + (i_31 * 4));
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
      ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&v__3))[1]);
      *(uint2*)(acc_s_cast + (i_31 * 4)) = __4;
    }
    #pragma unroll
    for (int i_32 = 0; i_32 < 2; ++i_32) {
      scores_sum[i_32] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
        scores_sum[i_32] = (scores_sum[i_32] + acc_s[((((rv_3 & 7) * 4) + (i_32 * 2)) + (rv_3 >> 3))]);
      }
      scores_sum[i_32] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_32]);
    }
    overlap_plan_mbar[4].wait(1);
    #pragma unroll
    for (int i_33 = 0; i_33 < 2; ++i_33) {
      scores_scale_v1[i_33] = ((float*)scores_scale_wsp_handoff_3)[((((((int)threadIdx.x) >> 5) * 16) + (i_33 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
    }
    #pragma unroll
    for (int i_34 = 0; i_34 < 2; ++i_34) {
      logsum[i_34] = ((logsum[i_34] * scores_scale_v1[i_34]) + scores_sum[i_34]);
    }
    overlap_plan_mbar[6].wait(1);
    #pragma unroll
    for (int i_35 = 0; i_35 < 2; ++i_35) {
      scores_scale_v1[i_35] = ((float*)scores_scale_wsp_handoff_4)[(((((((int)threadIdx.x) >> 5) * 16) + (i_35 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 128)];
    }
    #pragma unroll
    for (int i_36 = 0; i_36 < 64; ++i_36) {
      acc_o[i_36] = (acc_o[i_36] * scores_scale_v1[((i_36 & 3) >> 1)]);
    }
    overlap_plan_mbar[9].wait(1);
    {
      tl::GmmaDescriptor desc_b_7;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_7, (&(((half_t*)V_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_7, 16384);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast + (ki_7 * 8)), uint64_t(desc_b_7 + ((ki_7 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast + 0), 16);
    }
    overlap_plan_mbar[13].arrive();
    #pragma unroll
    for (int i_37 = 0; i_37 < 64; ++i_37) {
      acc_o[i_37] = (acc_o[i_37] / logsum[((i_37 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 8; ++i_38) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((i_38 >> 2) * 8192) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_38 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_38 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_o[(i_38 * 8)]), ((half_t)acc_o[((i_38 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_38 * 8) + 2)]), ((half_t)acc_o[((i_38 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_38 * 8) + 4)]), ((half_t)acc_o[((i_38 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_38 * 8) + 6)]), ((half_t)acc_o[((i_38 * 8) + 7)])));
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
    tl::warpgroup_reg_dealloc<40>();
    float broadcast_var_2 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_2, broadcast_var_2);
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      if (1 <= k_1) {
        overlap_plan_mbar[10].wait(((k_1 + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[1].arrive_and_expect_tx(16384);
        tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[0])), 0, (k_1 * 128), (((int)blockIdx.y) >> 3), 0);
        tl::tma_load(K_desc, overlap_plan_mbar[1], (&(((half_t*)K_shared)[4096])), 64, (k_1 * 128), (((int)blockIdx.y) >> 3), 0);
      }
      if (1 <= k_1) {
        overlap_plan_mbar[12].wait(((k_1 + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[8].arrive_and_expect_tx(16384);
        tl::tma_load(V_desc, overlap_plan_mbar[8], (&(((half_t*)V_shared)[0])), 0, (k_1 * 128), (((int)blockIdx.y) >> 3), 0);
        tl::tma_load(V_desc, overlap_plan_mbar[8], (&(((half_t*)V_shared)[4096])), 64, (k_1 * 128), (((int)blockIdx.y) >> 3), 0);
      }
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_3 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
      overlap_plan_mbar[3].wait(0);
      #pragma unroll
      for (int i_39 = 0; i_39 < 16; ++i_39) {
        *(float2*)(acc_s + (i_39 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_2) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_39 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_39 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
      }
      #pragma unroll
      for (int i_40 = 0; i_40 < 2; ++i_40) {
        scores_max_clear[i_40] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
          scores_max_clear[i_40] = max(scores_max_clear[i_40], acc_s[((((rv_4 & 7) * 4) + (i_40 * 2)) + (rv_4 >> 3))]);
        }
        scores_max_clear[i_40] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_40]);
        scores_max[i_40] = max(scores_max[i_40], scores_max_clear[i_40]);
      }
      #pragma unroll
      for (int i_41 = 0; i_41 < 2; ++i_41) {
        scores_max[i_41] = max(scores_max[i_41], scores_max_prev[i_41]);
      }
      #pragma unroll
      for (int i_42 = 0; i_42 < 32; ++i_42) {
        acc_s[i_42] = exp2f(((acc_s[i_42] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_42 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      tl::__sync_thread_partial(4, 256);
      #pragma unroll
      for (int i_43 = 0; i_43 < 16; ++i_43) {
        *(float2*)(((float*)acc_s_wsp_handoff_5) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_43 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_43 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192)) = *(float2*)(acc_s + (i_43 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[7].arrive();
      #pragma unroll
      for (int i_44 = 0; i_44 < 2; ++i_44) {
        scores_scale_v0[i_44] = exp2f(((scores_max_prev[i_44] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_44] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      if ((((int)threadIdx.x) % 4) == 0) {
        #pragma unroll
        for (int i_45 = 0; i_45 < 2; ++i_45) {
          ((float*)scores_scale_wsp_handoff_3)[(((((((int)threadIdx.x) >> 5) * 16) + (i_45 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128)] = scores_scale_v0[i_45];
        }
        #pragma unroll
        for (int i_46 = 0; i_46 < 2; ++i_46) {
          ((float*)scores_scale_wsp_handoff_4)[(((((((int)threadIdx.x) >> 5) * 16) + (i_46 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128)] = scores_scale_v0[i_46];
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[4].arrive();
      tl::fence_proxy_async();
      overlap_plan_mbar[5].arrive();
      if (1 <= k_1) {
        overlap_plan_mbar[11].wait(((k_1 + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(16384);
        tl::tma_load(K_desc, overlap_plan_mbar[2], (&(((half_t*)K_shared)[8192])), 0, ((k_1 * 128) + 64), (((int)blockIdx.y) >> 3), 0);
        tl::tma_load(K_desc, overlap_plan_mbar[2], (&(((half_t*)K_shared)[12288])), 64, ((k_1 * 128) + 64), (((int)blockIdx.y) >> 3), 0);
      }
      if (1 <= k_1) {
        overlap_plan_mbar[13].wait(((k_1 + 1) & 1));
      }
      tl::__sync_thread_partial(4, 256);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[9].arrive_and_expect_tx(16384);
        tl::tma_load(V_desc, overlap_plan_mbar[9], (&(((half_t*)V_shared)[8192])), 0, ((k_1 * 128) + 64), (((int)blockIdx.y) >> 3), 0);
        tl::tma_load(V_desc, overlap_plan_mbar[9], (&(((half_t*)V_shared)[12288])), 64, ((k_1 * 128) + 64), (((int)blockIdx.y) >> 3), 0);
      }
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_4 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_4, broadcast_var_4);
      overlap_plan_mbar[3].wait(1);
      #pragma unroll
      for (int i_47 = 0; i_47 < 16; ++i_47) {
        *(float2*)(acc_s + (i_47 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_2) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_47 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_47 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
      }
      #pragma unroll
      for (int i_48 = 0; i_48 < 2; ++i_48) {
        scores_max_clear_1[i_48] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_5 = 0; rv_5 < 16; ++rv_5) {
          scores_max_clear_1[i_48] = max(scores_max_clear_1[i_48], acc_s[((((rv_5 & 7) * 4) + (i_48 * 2)) + (rv_5 >> 3))]);
        }
        scores_max_clear_1[i_48] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_48]);
        scores_max[i_48] = max(scores_max[i_48], scores_max_clear_1[i_48]);
      }
      #pragma unroll
      for (int i_49 = 0; i_49 < 2; ++i_49) {
        scores_max[i_49] = max(scores_max[i_49], scores_max_prev[i_49]);
      }
      #pragma unroll
      for (int i_50 = 0; i_50 < 32; ++i_50) {
        acc_s[i_50] = exp2f(((acc_s[i_50] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_50 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      #pragma unroll
      for (int i_51 = 0; i_51 < 16; ++i_51) {
        *(float2*)(((float*)acc_s_wsp_handoff_5) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_51 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_51 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192)) = *(float2*)(acc_s + (i_51 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[7].arrive();
      #pragma unroll
      for (int i_52 = 0; i_52 < 2; ++i_52) {
        scores_scale_v1[i_52] = exp2f(((scores_max_prev[i_52] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_52] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      tl::__sync_thread_partial(4, 256);
      if ((((int)threadIdx.x) % 4) == 0) {
        #pragma unroll
        for (int i_53 = 0; i_53 < 2; ++i_53) {
          ((float*)scores_scale_wsp_handoff_3)[(((((((int)threadIdx.x) >> 5) * 16) + (i_53 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128)] = scores_scale_v1[i_53];
        }
        #pragma unroll
        for (int i_54 = 0; i_54 < 2; ++i_54) {
          ((float*)scores_scale_wsp_handoff_4)[((((((int)threadIdx.x) >> 5) * 16) + (i_54 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = scores_scale_v1[i_54];
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[4].arrive();
      tl::fence_proxy_async();
      overlap_plan_mbar[6].arrive();
    }
  }
}

