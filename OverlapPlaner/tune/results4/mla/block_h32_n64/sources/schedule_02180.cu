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
  void* acc_s_wsp_handoff_5 = ((void*)((char*)buf_dyn_shmem + 208896));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 217088));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_4 = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_5 = ((void*)((char*)buf_dyn_shmem + 221184));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[17];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[64];
  float logsum[4];
  float scores_max[4];
  float scores_max_prev[4];
  float acc_s_v0[8];
  float scores_scale[4];
  float scores_sum[4];
  float acc_s_v1[8];
  float scores_max_clear[4];
  float scores_max_clear_1[4];
  float scores_max_clear_2[4];
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
      scores_max_clear[i_2] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_2], (&(((float*)workspace_3)[0])));
      scores_max_clear[i_2] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_2]);
      scores_max[i_2] = max(scores_max[i_2], scores_max_clear[i_2]);
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      scores_max[i_3] = max(scores_max[i_3], scores_max_prev[i_3]);
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      acc_s_v0[i_4] = exp2f(((acc_s_v0[i_4] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_4 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      *(float2*)(((float*)acc_s_wsp_handoff_6) + ((((i_5 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s_v0 + (i_5 * 2));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[8].arrive();
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)S_shared)[((((((int)threadIdx.x) & 31) >> 3) * 512) + ((((((((int)threadIdx.x) & 31) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_s_v0[0]), ((half_t)acc_s_v0[1])), __pack_half2(((half_t)acc_s_v0[2]), ((half_t)acc_s_v0[3])), __pack_half2(((half_t)acc_s_v0[4]), ((half_t)acc_s_v0[5])), __pack_half2(((half_t)acc_s_v0[6]), ((half_t)acc_s_v0[7])));
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      scores_scale[i_6] = exp2f(((scores_max_prev[i_6] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_6] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      scores_sum[i_7] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
        scores_sum[i_7] = (scores_sum[i_7] + acc_s_v0[((i_7 * 2) + rv_1)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_sum[i_7] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_7], (&(((float*)workspace_4)[0])));
      scores_sum[i_7] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      logsum[i_8] = ((logsum[i_8] * scores_scale[i_8]) + scores_sum[i_8]);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 64; ++i_9) {
      acc_o[i_9] = (acc_o[i_9] * scores_scale[(((i_9 >> 5) * 2) + ((i_9 & 3) >> 1))]);
    }
    for (int k = 0; k < 127; ++k) {
      overlap_plan_mbar[((k & 1) + 2)].wait(((k & 3) >> 1));
      {
        half_t A_local[16];
        half_t B_local[32];
        tl::__sync_thread_partial(3, 256);
        for (int ki = 0; ki < 4; ++ki) {
          for (int i_10 = 0; i_10 < 2; ++i_10) {
            tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((i_10 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[(i_10 * 8)])));
          }
          for (int i_11 = 0; i_11 < 4; ++i_11) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((k & 1) * 32768) + ((((int)threadIdx.x) >> 5) * 4096)) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local[(i_11 * 8)])));
          }
          for (int i_12 = 0; i_12 < 2; ++i_12) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_12 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_12 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      overlap_plan_mbar[((k & 1) + 12)].arrive();
      *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
      float broadcast_var_4 = -CUDART_INF_F;
      *(float4*)(scores_max + 0) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
      overlap_plan_mbar[7].wait(((k + 1) & 1));
      if (((k + 1) % 2) == 0) {
        #pragma unroll
        for (int i_13 = 0; i_13 < 4; ++i_13) {
          *(float2*)(acc_s_v0 + (i_13 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_5) + ((((i_13 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        }
        #pragma unroll
        for (int i_14 = 0; i_14 < 4; ++i_14) {
          scores_max_clear_1[i_14] = -CUDART_INF_F;
          #pragma unroll
          for (int rv_2 = 0; rv_2 < 2; ++rv_2) {
            scores_max_clear_1[i_14] = max(scores_max_clear_1[i_14], acc_s_v0[((i_14 * 2) + rv_2)]);
          }
          tl::__sync_thread_partial(3, 256);
          scores_max_clear_1[i_14] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_14], (&(((float*)workspace_1)[0])));
          scores_max_clear_1[i_14] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_14]);
          scores_max[i_14] = max(scores_max[i_14], scores_max_clear_1[i_14]);
        }
      } else {
        #pragma unroll
        for (int i_15 = 0; i_15 < 4; ++i_15) {
          *(float2*)(acc_s_v1 + (i_15 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_5) + ((((i_15 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        }
        #pragma unroll
        for (int i_16 = 0; i_16 < 4; ++i_16) {
          scores_max_clear_2[i_16] = -CUDART_INF_F;
          #pragma unroll
          for (int rv_3 = 0; rv_3 < 2; ++rv_3) {
            scores_max_clear_2[i_16] = max(scores_max_clear_2[i_16], acc_s_v1[((i_16 * 2) + rv_3)]);
          }
          tl::__sync_thread_partial(3, 256);
          scores_max_clear_2[i_16] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_16], (&(((float*)workspace_5)[0])));
          scores_max_clear_2[i_16] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_16]);
          scores_max[i_16] = max(scores_max[i_16], scores_max_clear_2[i_16]);
        }
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 4; ++i_17) {
        scores_max[i_17] = max(scores_max[i_17], scores_max_prev[i_17]);
      }
      if (((k + 1) % 2) == 0) {
        #pragma unroll
        for (int i_18 = 0; i_18 < 8; ++i_18) {
          acc_s_v0[i_18] = exp2f(((acc_s_v0[i_18] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_18 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
        }
        #pragma unroll
        for (int i_19 = 0; i_19 < 4; ++i_19) {
          *(float2*)(((float*)acc_s_wsp_handoff_6) + (((((((k + 1) & 1) * 2048) + (i_19 * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s_v0 + (i_19 * 2));
        }
      } else {
        #pragma unroll
        for (int i_20 = 0; i_20 < 8; ++i_20) {
          acc_s_v1[i_20] = exp2f(((acc_s_v1[i_20] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_20 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
        }
        #pragma unroll
        for (int i_21 = 0; i_21 < 4; ++i_21) {
          *(float2*)(((float*)acc_s_wsp_handoff_6) + (((((((k + 1) & 1) * 2048) + (i_21 * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s_v1 + (i_21 * 2));
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[(((k + 1) & 1) + 8)].arrive();
      if (((k + 1) % 2) == 0) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)S_shared)[((((((int)threadIdx.x) & 31) >> 3) * 512) + ((((((((int)threadIdx.x) & 31) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_s_v0[0]), ((half_t)acc_s_v0[1])), __pack_half2(((half_t)acc_s_v0[2]), ((half_t)acc_s_v0[3])), __pack_half2(((half_t)acc_s_v0[4]), ((half_t)acc_s_v0[5])), __pack_half2(((half_t)acc_s_v0[6]), ((half_t)acc_s_v0[7])));
      } else {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)S_shared)[((((((int)threadIdx.x) & 31) >> 3) * 512) + ((((((((int)threadIdx.x) & 31) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_s_v1[0]), ((half_t)acc_s_v1[1])), __pack_half2(((half_t)acc_s_v1[2]), ((half_t)acc_s_v1[3])), __pack_half2(((half_t)acc_s_v1[4]), ((half_t)acc_s_v1[5])), __pack_half2(((half_t)acc_s_v1[6]), ((half_t)acc_s_v1[7])));
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 4; ++i_22) {
        scores_scale[i_22] = exp2f(((scores_max_prev[i_22] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_22] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      if (((k + 1) % 2) == 0) {
        #pragma unroll
        for (int i_23 = 0; i_23 < 4; ++i_23) {
          scores_sum[i_23] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv_4 = 0; rv_4 < 2; ++rv_4) {
            scores_sum[i_23] = (scores_sum[i_23] + acc_s_v0[((i_23 * 2) + rv_4)]);
          }
          tl::__sync_thread_partial(3, 256);
          scores_sum[i_23] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_23], (&(((float*)workspace_2)[0])));
          scores_sum[i_23] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_23]);
        }
      } else {
        #pragma unroll
        for (int i_24 = 0; i_24 < 4; ++i_24) {
          scores_sum[i_24] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv_5 = 0; rv_5 < 2; ++rv_5) {
            scores_sum[i_24] = (scores_sum[i_24] + acc_s_v1[((i_24 * 2) + rv_5)]);
          }
          tl::__sync_thread_partial(3, 256);
          scores_sum[i_24] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_24], (&(((float*)workspace)[0])));
          scores_sum[i_24] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_24]);
        }
      }
      #pragma unroll
      for (int i_25 = 0; i_25 < 4; ++i_25) {
        logsum[i_25] = ((logsum[i_25] * scores_scale[i_25]) + scores_sum[i_25]);
      }
      #pragma unroll
      for (int i_26 = 0; i_26 < 64; ++i_26) {
        acc_o[i_26] = (acc_o[i_26] * scores_scale[(((i_26 >> 5) * 2) + ((i_26 & 3) >> 1))]);
      }
    }
    overlap_plan_mbar[3].wait(1);
    {
      half_t A_local_1[16];
      half_t B_local_1[32];
      tl::__sync_thread_partial(3, 256);
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        for (int i_27 = 0; i_27 < 2; ++i_27) {
          tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((i_27 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_1[(i_27 * 8)])));
        }
        for (int i_28 = 0; i_28 < 4; ++i_28) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) >> 5) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_28 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_28 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_1[(i_28 * 8)])));
        }
        for (int i_29 = 0; i_29 < 2; ++i_29) {
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_29 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_29 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_29 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_29 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
          }
        }
      }
    }
    overlap_plan_mbar[13].arrive();
    #pragma unroll
    for (int i_30 = 0; i_30 < 64; ++i_30) {
      acc_o[i_30] = (acc_o[i_30] / logsum[(((i_30 >> 5) * 2) + ((i_30 & 3) >> 1))]);
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 8; ++i_31) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[((((((i_31 >> 2) * 8192) + ((((int)threadIdx.x) & 15) * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((i_31 & 3) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_31 * 8)]), ((half_t)acc_o[((i_31 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_31 * 8) + 2)]), ((half_t)acc_o[((i_31 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_31 * 8) + 4)]), ((half_t)acc_o[((i_31 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_31 * 8) + 6)]), ((half_t)acc_o[((i_31 * 8) + 7)])));
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
      for (int i_32 = 0; i_32 < 4; ++i_32) {
        *(float2*)(acc_s_v0 + (i_32 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_6) + (((((i_32 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1984));
      }
      {
        half_t A_local_2[16];
        half_t B_local_2[4];
        #pragma unroll
        for (int i_33 = 0; i_33 < 2; ++i_33) {
          float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(acc_s_v0 + (i_33 * 4)) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
        }
        tl::__sync_thread_partial(4, 256);
        for (int ki_2 = 0; ki_2 < 32; ++ki_2) {
          for (int i_34 = 0; i_34 < 2; ++i_34) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki_2 >> 2) * 2048) + (i_34 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[(i_34 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[(((((((((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki_2 >> 2) * 4096)) + (((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[0])));
          for (int i_35 = 0; i_35 < 2; ++i_35) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + (i_35 * 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_35 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + 0));
          }
        }
      }
      overlap_plan_mbar[10].arrive();
      if (k_1 == 0) {
        overlap_plan_mbar[1].wait(0);
      }
      overlap_plan_mbar[(((k_1 * 2) % 3) + 4)].wait((((k_1 % 3) * 2) / 3));
      {
        half_t A_local_3[16];
        half_t B_local_3[4];
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          for (int i_36 = 0; i_36 < 2; ++i_36) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[(((i_36 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_3[(i_36 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + ((k_1 * 2) % 3)) % 3) * 4096) + (((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[0])));
          for (int i_37 = 0; i_37 < 2; ++i_37) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + (i_37 * 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_37 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + 0));
          }
        }
      }
      tl::__sync_thread_partial(4, 256);
      #pragma unroll
      for (int i_38 = 0; i_38 < 4; ++i_38) {
        *(float2*)(((float*)acc_s_wsp_handoff_5) + (((((i_38 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64)) = *(float2*)(acc_s_v0 + (i_38 * 2));
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
      for (int i_39 = 0; i_39 < 4; ++i_39) {
        *(float2*)(acc_s_v1 + (i_39 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_6) + (((((i_39 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
      }
      {
        half_t A_local_4[16];
        half_t B_local_4[4];
        #pragma unroll
        for (int i_40 = 0; i_40 < 2; ++i_40) {
          float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(acc_s_v1 + (i_40 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
        }
        for (int ki_4 = 0; ki_4 < 32; ++ki_4) {
          for (int i_41 = 0; i_41 < 2; ++i_41) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki_4 >> 2) * 2048) + (i_41 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_4[(i_41 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[(((((((((((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 1) & 1) * 32768) + ((ki_4 >> 2) * 4096)) + (((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_4[0])));
          for (int i_42 = 0; i_42 < 2; ++i_42) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + (i_42 * 4)), reinterpret_cast<const unsigned*>(A_local_4 + (i_42 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + 0));
          }
        }
      }
      overlap_plan_mbar[11].arrive();
      overlap_plan_mbar[((((k_1 * 2) + 1) % 3) + 4)].wait(((((k_1 % 3) * 2) + 1) / 3));
      {
        half_t A_local_5[16];
        half_t B_local_5[4];
        for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
          for (int i_43 = 0; i_43 < 2; ++i_43) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[(((i_43 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_5[(i_43 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + (((k_1 * 2) + 1) % 3)) % 3) * 4096) + (((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_5[0])));
          for (int i_44 = 0; i_44 < 2; ++i_44) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + (i_44 * 4)), reinterpret_cast<const unsigned*>(A_local_5 + (i_44 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + 0));
          }
        }
      }
      tl::__sync_thread_partial(4, 256);
      #pragma unroll
      for (int i_45 = 0; i_45 < 4; ++i_45) {
        *(float2*)(((float*)acc_s_wsp_handoff_5) + (((((i_45 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64)) = *(float2*)(acc_s_v1 + (i_45 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[7].arrive();
      overlap_plan_mbar[((((k_1 * 2) + 1) % 3) + 14)].arrive();
    }
  }
}

