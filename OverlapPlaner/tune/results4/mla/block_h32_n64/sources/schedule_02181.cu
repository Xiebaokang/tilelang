#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap KV_desc, __grid_constant__ const CUtensorMap K_pe_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap Q_pe_desc);
extern "C" __global__ void __launch_bounds__(512, 1) main_kernel(__grid_constant__ const CUtensorMap KV_desc, __grid_constant__ const CUtensorMap K_pe_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap Q_pe_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_pe_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* KV_shared = ((void*)((char*)buf_dyn_shmem + 36864));
  void* K_pe_shared = ((void*)((char*)buf_dyn_shmem + 167936));
  void* acc_s_wsp_handoff_6 = ((void*)((char*)buf_dyn_shmem + 192512));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 208896));
  void* acc_s_wsp_handoff_5 = ((void*)((char*)buf_dyn_shmem + 217088));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 225280));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 225280));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 225280));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 225280));
  void* workspace_4 = ((void*)((char*)buf_dyn_shmem + 225280));
  void* workspace_5 = ((void*)((char*)buf_dyn_shmem + 225280));
  void* workspace_6 = ((void*)((char*)buf_dyn_shmem + 225280));
  void* workspace_7 = ((void*)((char*)buf_dyn_shmem + 225280));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[17];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[64];
  float logsum[4];
  float scores_max[4];
  float scores_max_prev[4];
  float acc_s_v0[8];
  float scores_scale_v0[4];
  float scores_sum[4];
  float acc_s_v1[8];
  float scores_scale_v1[4];
  float scores_max_clear[4];
  float scores_max_clear_1[4];
  float scores_max_clear_2[4];
  float scores_max_clear_3[4];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(Q_pe_desc);
    tl::prefetch_tma_descriptor(KV_desc);
    tl::prefetch_tma_descriptor(K_pe_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(1);
    overlap_plan_mbar[4].init(1);
    overlap_plan_mbar[5].init(1);
    overlap_plan_mbar[6].init(1);
    overlap_plan_mbar[7].init(256);
    overlap_plan_mbar[8].init(256);
    overlap_plan_mbar[9].init(256);
    overlap_plan_mbar[10].init(256);
    overlap_plan_mbar[11].init(256);
    overlap_plan_mbar[12].init(256);
    overlap_plan_mbar[13].init(256);
    overlap_plan_mbar[14].init(256);
    overlap_plan_mbar[15].init(256);
    overlap_plan_mbar[16].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    tl::warpgroup_reg_alloc<224>();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(logsum + 0) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    float broadcast_var_2 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
    float broadcast_var_3 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    overlap_plan_mbar[7].wait(0);
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      *(float2*)(acc_s_v0 + (i_1 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_5) + ((((i_1 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      scores_max_clear[i_2] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 2; ++rv) {
        scores_max_clear[i_2] = max(scores_max_clear[i_2], acc_s_v0[((i_2 * 2) + rv)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_max_clear[i_2] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_2], (&(((float*)workspace_7)[0])));
      scores_max_clear[i_2] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_2]);
      scores_max[i_2] = max(scores_max[i_2], scores_max_clear[i_2]);
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      scores_max[i_3] = max(scores_max[i_3], scores_max_prev[i_3]);
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      scores_scale_v0[i_4] = exp2f(((scores_max_prev[i_4] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_4] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      acc_s_v0[i_5] = exp2f(((acc_s_v0[i_5] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_5 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      *(float2*)(((float*)acc_s_wsp_handoff_6) + ((((i_6 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s_v0 + (i_6 * 2));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[8].arrive();
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      scores_sum[i_7] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
        scores_sum[i_7] = (scores_sum[i_7] + acc_s_v0[((i_7 * 2) + rv_1)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_sum[i_7] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_7], (&(((float*)workspace_6)[0])));
      scores_sum[i_7] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      half_t S_shared_local_cast[2];
      uint1 __1;
      float2 v_ = *(float2*)(acc_s_v0 + (i_8 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(S_shared_local_cast + 0) = __1;
      *(uint1*)(((half_t*)S_shared) + ((((((i_8 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast + 0);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      logsum[i_9] = ((logsum[i_9] * scores_scale_v0[i_9]) + scores_sum[i_9]);
    }
    for (int k = 0; k < 63; ++k) {
      *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
      float broadcast_var_4 = -CUDART_INF_F;
      *(float4*)(scores_max + 0) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
      overlap_plan_mbar[7].wait(1);
      #pragma unroll
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        *(float2*)(acc_s_v1 + (i_10 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_5) + ((((i_10 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 4; ++i_11) {
        scores_max_clear_1[i_11] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 2; ++rv_2) {
          scores_max_clear_1[i_11] = max(scores_max_clear_1[i_11], acc_s_v1[((i_11 * 2) + rv_2)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear_1[i_11] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_11], (&(((float*)workspace_2)[0])));
        scores_max_clear_1[i_11] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_11]);
        scores_max[i_11] = max(scores_max[i_11], scores_max_clear_1[i_11]);
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 4; ++i_12) {
        scores_max[i_12] = max(scores_max[i_12], scores_max_prev[i_12]);
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 4; ++i_13) {
        scores_scale_v1[i_13] = exp2f(((scores_max_prev[i_13] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_13] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 8; ++i_14) {
        acc_s_v1[i_14] = exp2f(((acc_s_v1[i_14] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_14 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 4; ++i_15) {
        *(float2*)(((float*)acc_s_wsp_handoff_6) + (((((i_15 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048)) = *(float2*)(acc_s_v1 + (i_15 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[9].arrive();
      #pragma unroll
      for (int i_16 = 0; i_16 < 4; ++i_16) {
        scores_sum[i_16] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 2; ++rv_3) {
          scores_sum[i_16] = (scores_sum[i_16] + acc_s_v1[((i_16 * 2) + rv_3)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_16] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_16], (&(((float*)workspace_4)[0])));
        scores_sum[i_16] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_16]);
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 4; ++i_17) {
        half_t S_shared_local_cast_1[2];
        uint1 __2;
        float2 v__1 = *(float2*)(acc_s_v1 + (i_17 * 2));
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
        *(uint1*)(S_shared_local_cast_1 + 0) = __2;
        *(uint1*)(((half_t*)S_shared) + (((((((i_17 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048)) = *(uint1*)(S_shared_local_cast_1 + 0);
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 4; ++i_18) {
        logsum[i_18] = ((logsum[i_18] * scores_scale_v1[i_18]) + scores_sum[i_18]);
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 64; ++i_19) {
        acc_o[i_19] = (acc_o[i_19] * scores_scale_v0[(((i_19 >> 5) * 2) + ((i_19 & 3) >> 1))]);
      }
      overlap_plan_mbar[2].wait((k & 1));
      {
        half_t A_local[16];
        half_t B_local[32];
        for (int ki = 0; ki < 4; ++ki) {
          for (int i_20 = 0; i_20 < 2; ++i_20) {
            tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((i_20 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[(i_20 * 8)])));
          }
          for (int i_21 = 0; i_21 < 4; ++i_21) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_21 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local[(i_21 * 8)])));
          }
          for (int i_22 = 0; i_22 < 2; ++i_22) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_22 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_22 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_22 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_22 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      overlap_plan_mbar[12].arrive();
      *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
      float broadcast_var_5 = -CUDART_INF_F;
      *(float4*)(scores_max + 0) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
      overlap_plan_mbar[7].wait(0);
      #pragma unroll
      for (int i_23 = 0; i_23 < 4; ++i_23) {
        *(float2*)(acc_s_v0 + (i_23 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_5) + ((((i_23 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 4; ++i_24) {
        scores_max_clear_2[i_24] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_4 = 0; rv_4 < 2; ++rv_4) {
          scores_max_clear_2[i_24] = max(scores_max_clear_2[i_24], acc_s_v0[((i_24 * 2) + rv_4)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear_2[i_24] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_24], (&(((float*)workspace_1)[0])));
        scores_max_clear_2[i_24] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_24]);
        scores_max[i_24] = max(scores_max[i_24], scores_max_clear_2[i_24]);
      }
      #pragma unroll
      for (int i_25 = 0; i_25 < 4; ++i_25) {
        scores_max[i_25] = max(scores_max[i_25], scores_max_prev[i_25]);
      }
      #pragma unroll
      for (int i_26 = 0; i_26 < 4; ++i_26) {
        scores_scale_v0[i_26] = exp2f(((scores_max_prev[i_26] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_26] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_27 = 0; i_27 < 8; ++i_27) {
        acc_s_v0[i_27] = exp2f(((acc_s_v0[i_27] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_27 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_28 = 0; i_28 < 4; ++i_28) {
        *(float2*)(((float*)acc_s_wsp_handoff_6) + ((((i_28 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s_v0 + (i_28 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[8].arrive();
      #pragma unroll
      for (int i_29 = 0; i_29 < 4; ++i_29) {
        scores_sum[i_29] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_5 = 0; rv_5 < 2; ++rv_5) {
          scores_sum[i_29] = (scores_sum[i_29] + acc_s_v0[((i_29 * 2) + rv_5)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_29] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_29], (&(((float*)workspace)[0])));
        scores_sum[i_29] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_29]);
      }
      #pragma unroll
      for (int i_30 = 0; i_30 < 4; ++i_30) {
        half_t S_shared_local_cast_2[2];
        uint1 __3;
        float2 v__2 = *(float2*)(acc_s_v0 + (i_30 * 2));
        ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
        *(uint1*)(S_shared_local_cast_2 + 0) = __3;
        *(uint1*)(((half_t*)S_shared) + ((((((i_30 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast_2 + 0);
      }
      #pragma unroll
      for (int i_31 = 0; i_31 < 4; ++i_31) {
        logsum[i_31] = ((logsum[i_31] * scores_scale_v0[i_31]) + scores_sum[i_31]);
      }
      #pragma unroll
      for (int i_32 = 0; i_32 < 64; ++i_32) {
        acc_o[i_32] = (acc_o[i_32] * scores_scale_v1[(((i_32 >> 5) * 2) + ((i_32 & 3) >> 1))]);
      }
      overlap_plan_mbar[3].wait((k & 1));
      {
        half_t A_local_1[16];
        half_t B_local_1[32];
        tl::__sync_thread_partial(3, 256);
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          for (int i_33 = 0; i_33 < 2; ++i_33) {
            tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[((((i_33 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 2048)])), (&(A_local_1[(i_33 * 8)])));
          }
          for (int i_34 = 0; i_34 < 4; ++i_34) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) >> 5) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_34 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_34 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_1[(i_34 * 8)])));
          }
          for (int i_35 = 0; i_35 < 2; ++i_35) {
            for (int j_1 = 0; j_1 < 4; ++j_1) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_35 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_35 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_35 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_35 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
            }
          }
        }
      }
      overlap_plan_mbar[13].arrive();
    }
    *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
    float broadcast_var_6 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
    overlap_plan_mbar[7].wait(1);
    #pragma unroll
    for (int i_36 = 0; i_36 < 4; ++i_36) {
      *(float2*)(acc_s_v1 + (i_36 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_5) + ((((i_36 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_37 = 0; i_37 < 4; ++i_37) {
      scores_max_clear_3[i_37] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_6 = 0; rv_6 < 2; ++rv_6) {
        scores_max_clear_3[i_37] = max(scores_max_clear_3[i_37], acc_s_v1[((i_37 * 2) + rv_6)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_max_clear_3[i_37] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_37], (&(((float*)workspace_3)[0])));
      scores_max_clear_3[i_37] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_37]);
      scores_max[i_37] = max(scores_max[i_37], scores_max_clear_3[i_37]);
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 4; ++i_38) {
      scores_max[i_38] = max(scores_max[i_38], scores_max_prev[i_38]);
    }
    #pragma unroll
    for (int i_39 = 0; i_39 < 4; ++i_39) {
      scores_scale_v1[i_39] = exp2f(((scores_max_prev[i_39] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_39] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_40 = 0; i_40 < 8; ++i_40) {
      acc_s_v1[i_40] = exp2f(((acc_s_v1[i_40] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_40 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_41 = 0; i_41 < 4; ++i_41) {
      *(float2*)(((float*)acc_s_wsp_handoff_6) + (((((i_41 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048)) = *(float2*)(acc_s_v1 + (i_41 * 2));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[9].arrive();
    #pragma unroll
    for (int i_42 = 0; i_42 < 4; ++i_42) {
      scores_sum[i_42] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_7 = 0; rv_7 < 2; ++rv_7) {
        scores_sum[i_42] = (scores_sum[i_42] + acc_s_v1[((i_42 * 2) + rv_7)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_sum[i_42] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_42], (&(((float*)workspace_5)[0])));
      scores_sum[i_42] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_42]);
    }
    #pragma unroll
    for (int i_43 = 0; i_43 < 4; ++i_43) {
      half_t S_shared_local_cast_3[2];
      uint1 __4;
      float2 v__3 = *(float2*)(acc_s_v1 + (i_43 * 2));
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
      *(uint1*)(S_shared_local_cast_3 + 0) = __4;
      *(uint1*)(((half_t*)S_shared) + (((((((i_43 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048)) = *(uint1*)(S_shared_local_cast_3 + 0);
    }
    #pragma unroll
    for (int i_44 = 0; i_44 < 4; ++i_44) {
      logsum[i_44] = ((logsum[i_44] * scores_scale_v1[i_44]) + scores_sum[i_44]);
    }
    #pragma unroll
    for (int i_45 = 0; i_45 < 64; ++i_45) {
      acc_o[i_45] = (acc_o[i_45] * scores_scale_v0[(((i_45 >> 5) * 2) + ((i_45 & 3) >> 1))]);
    }
    overlap_plan_mbar[2].wait(1);
    {
      half_t A_local_2[16];
      half_t B_local_2[32];
      tl::__sync_thread_partial(3, 256);
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        for (int i_46 = 0; i_46 < 2; ++i_46) {
          tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((i_46 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[(i_46 * 8)])));
        }
        for (int i_47 = 0; i_47 < 4; ++i_47) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_47 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_47 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_2[(i_47 * 8)])));
        }
        for (int i_48 = 0; i_48 < 2; ++i_48) {
          for (int j_2 = 0; j_2 < 4; ++j_2) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_48 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_48 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_48 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_48 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
          }
        }
      }
    }
    overlap_plan_mbar[12].arrive();
    #pragma unroll
    for (int i_49 = 0; i_49 < 64; ++i_49) {
      acc_o[i_49] = (acc_o[i_49] * scores_scale_v1[(((i_49 >> 5) * 2) + ((i_49 & 3) >> 1))]);
    }
    overlap_plan_mbar[3].wait(1);
    {
      half_t A_local_3[16];
      half_t B_local_3[32];
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        for (int i_50 = 0; i_50 < 2; ++i_50) {
          tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[((((i_50 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 2048)])), (&(A_local_3[(i_50 * 8)])));
        }
        for (int i_51 = 0; i_51 < 4; ++i_51) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) >> 5) * 4096) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_51 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_51 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_3[(i_51 * 8)])));
        }
        for (int i_52 = 0; i_52 < 2; ++i_52) {
          for (int j_3 = 0; j_3 < 4; ++j_3) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_52 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_52 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_52 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_52 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
          }
        }
      }
    }
    overlap_plan_mbar[13].arrive();
    #pragma unroll
    for (int i_53 = 0; i_53 < 64; ++i_53) {
      acc_o[i_53] = (acc_o[i_53] / logsum[(((i_53 >> 5) * 2) + ((i_53 & 3) >> 1))]);
    }
    #pragma unroll
    for (int i_54 = 0; i_54 < 8; ++i_54) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[((((((i_54 >> 2) * 8192) + ((((int)threadIdx.x) & 15) * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((i_54 & 3) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_54 * 8)]), ((half_t)acc_o[((i_54 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_54 * 8) + 2)]), ((half_t)acc_o[((i_54 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_54 * 8) + 4)]), ((half_t)acc_o[((i_54 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_54 * 8) + 6)]), ((half_t)acc_o[((i_54 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 256);
    if (tl::tl_shuffle_elect<256>()) {
      tl::fence_proxy_async();
      tl::tma_store((&(Output[(((int)blockIdx.x) * 16384)])), (&(((half_t*)O_shared)[0])), 32768);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  } else {
    tl::warpgroup_reg_dealloc<24>();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(32768);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[0])), 0, (((int)blockIdx.x) * 32), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[2048])), 64, (((int)blockIdx.x) * 32), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[4096])), 128, (((int)blockIdx.x) * 32), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[6144])), 192, (((int)blockIdx.x) * 32), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[8192])), 256, (((int)blockIdx.x) * 32), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[10240])), 320, (((int)blockIdx.x) * 32), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[12288])), 384, (((int)blockIdx.x) * 32), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[14336])), 448, (((int)blockIdx.x) * 32), 0);
      overlap_plan_mbar[1].arrive_and_expect_tx(4096);
      tl::tma_load(Q_pe_desc, overlap_plan_mbar[1], (&(((half_t*)Q_pe_shared)[0])), 0, (((int)blockIdx.x) * 32), 0);
    }
    for (int k_1 = 0; k_1 < 64; ++k_1) {
      if (1 <= k_1) {
        overlap_plan_mbar[10].wait(((k_1 + 1) & 1));
        overlap_plan_mbar[12].wait(((k_1 + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(65536);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[0])), 0, (k_1 * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[4096])), 64, (k_1 * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[8192])), 128, (k_1 * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[12288])), 192, (k_1 * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[16384])), 256, (k_1 * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[20480])), 320, (k_1 * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[24576])), 384, (k_1 * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[28672])), 448, (k_1 * 128), 0, 0);
      }
      if (2 <= k_1) {
        overlap_plan_mbar[(((k_1 * 2) % 3) + 14)].wait(((((k_1 * 2) / 3) + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[(((k_1 * 2) % 3) + 4)].arrive_and_expect_tx(8192);
        tl::tma_load(K_pe_desc, overlap_plan_mbar[(((k_1 * 2) % 3) + 4)], (&(((half_t*)K_pe_shared)[(((k_1 * 2) % 3) * 4096)])), 0, (k_1 * 128), 0, 0);
      }
      if (k_1 == 0) {
        overlap_plan_mbar[0].wait(0);
      }
      overlap_plan_mbar[2].wait((k_1 & 1));
      if (1 <= k_1) {
        overlap_plan_mbar[9].wait(((k_1 + 1) & 1));
      }
      #pragma unroll
      for (int i_55 = 0; i_55 < 4; ++i_55) {
        *(float2*)(acc_s_v0 + (i_55 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_6) + (((((i_55 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1984));
      }
      {
        half_t A_local_4[16];
        half_t B_local_4[4];
        #pragma unroll
        for (int i_56 = 0; i_56 < 2; ++i_56) {
          float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(acc_s_v0 + (i_56 * 4)) = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
        }
        tl::__sync_thread_partial(4, 256);
        for (int ki_4 = 0; ki_4 < 32; ++ki_4) {
          for (int i_57 = 0; i_57 < 2; ++i_57) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki_4 >> 2) * 2048) + (i_57 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_4[(i_57 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[(((((((((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki_4 >> 2) * 4096)) + (((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_4[0])));
          for (int i_58 = 0; i_58 < 2; ++i_58) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + (i_58 * 4)), reinterpret_cast<const unsigned*>(A_local_4 + (i_58 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + 0));
          }
        }
      }
      overlap_plan_mbar[10].arrive();
      if (k_1 == 0) {
        overlap_plan_mbar[1].wait(0);
      }
      overlap_plan_mbar[(((k_1 * 2) % 3) + 4)].wait((((k_1 % 3) * 2) / 3));
      {
        half_t A_local_5[16];
        half_t B_local_5[4];
        for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
          for (int i_59 = 0; i_59 < 2; ++i_59) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[(((i_59 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_5[(i_59 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + ((k_1 * 2) % 3)) % 3) * 4096) + (((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_5[0])));
          for (int i_60 = 0; i_60 < 2; ++i_60) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + (i_60 * 4)), reinterpret_cast<const unsigned*>(A_local_5 + (i_60 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + 0));
          }
        }
      }
      tl::__sync_thread_partial(4, 256);
      #pragma unroll
      for (int i_61 = 0; i_61 < 4; ++i_61) {
        *(float2*)(((float*)acc_s_wsp_handoff_5) + (((((i_61 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64)) = *(float2*)(acc_s_v0 + (i_61 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[7].arrive();
      overlap_plan_mbar[(((k_1 * 2) % 3) + 14)].arrive();
      if (1 <= k_1) {
        overlap_plan_mbar[11].wait(((k_1 + 1) & 1));
        overlap_plan_mbar[13].wait(((k_1 + 1) & 1));
      }
      tl::__sync_thread_partial(4, 256);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[3].arrive_and_expect_tx(65536);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[32768])), 0, ((k_1 * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[36864])), 64, ((k_1 * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[40960])), 128, ((k_1 * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[45056])), 192, ((k_1 * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[49152])), 256, ((k_1 * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[53248])), 320, ((k_1 * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[57344])), 384, ((k_1 * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[61440])), 448, ((k_1 * 128) + 64), 0, 0);
      }
      if (1 <= k_1) {
        overlap_plan_mbar[((((k_1 * 2) + 1) % 3) + 14)].wait((((((k_1 * 2) + 1) / 3) + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[((((k_1 * 2) + 1) % 3) + 4)].arrive_and_expect_tx(8192);
        tl::tma_load(K_pe_desc, overlap_plan_mbar[((((k_1 * 2) + 1) % 3) + 4)], (&(((half_t*)K_pe_shared)[((((k_1 * 2) + 1) % 3) * 4096)])), 0, ((k_1 * 128) + 64), 0, 0);
      }
      overlap_plan_mbar[3].wait((k_1 & 1));
      overlap_plan_mbar[8].wait((k_1 & 1));
      #pragma unroll
      for (int i_62 = 0; i_62 < 4; ++i_62) {
        *(float2*)(acc_s_v1 + (i_62 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_6) + (((((i_62 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
      }
      {
        half_t A_local_6[16];
        half_t B_local_6[4];
        #pragma unroll
        for (int i_63 = 0; i_63 < 2; ++i_63) {
          float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(acc_s_v1 + (i_63 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
        }
        for (int ki_6 = 0; ki_6 < 32; ++ki_6) {
          for (int i_64 = 0; i_64 < 2; ++i_64) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki_6 >> 2) * 2048) + (i_64 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_6 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_6[(i_64 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[(((((((((((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 1) & 1) * 32768) + ((ki_6 >> 2) * 4096)) + (((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_6 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_6[0])));
          for (int i_65 = 0; i_65 < 2; ++i_65) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + (i_65 * 4)), reinterpret_cast<const unsigned*>(A_local_6 + (i_65 * 8)), reinterpret_cast<const unsigned*>(B_local_6 + 0));
          }
        }
      }
      overlap_plan_mbar[11].arrive();
      overlap_plan_mbar[((((k_1 * 2) + 1) % 3) + 4)].wait(((((k_1 % 3) * 2) + 1) / 3));
      {
        half_t A_local_7[16];
        half_t B_local_7[4];
        for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
          for (int i_66 = 0; i_66 < 2; ++i_66) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[(((i_66 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_7 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_7[(i_66 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + (((k_1 * 2) + 1) % 3)) % 3) * 4096) + (((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_7 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_7[0])));
          for (int i_67 = 0; i_67 < 2; ++i_67) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + (i_67 * 4)), reinterpret_cast<const unsigned*>(A_local_7 + (i_67 * 8)), reinterpret_cast<const unsigned*>(B_local_7 + 0));
          }
        }
      }
      tl::__sync_thread_partial(4, 256);
      #pragma unroll
      for (int i_68 = 0; i_68 < 4; ++i_68) {
        *(float2*)(((float*)acc_s_wsp_handoff_5) + (((((i_68 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64)) = *(float2*)(acc_s_v1 + (i_68 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[7].arrive();
      overlap_plan_mbar[((((k_1 * 2) + 1) % 3) + 14)].arrive();
    }
  }
}

