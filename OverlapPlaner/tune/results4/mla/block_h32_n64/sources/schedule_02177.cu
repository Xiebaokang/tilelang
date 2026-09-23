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
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap KV_desc, __grid_constant__ const CUtensorMap K_pe_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap Q_pe_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_pe_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* KV_shared = ((void*)((char*)buf_dyn_shmem + 36864));
  void* K_pe_shared = ((void*)((char*)buf_dyn_shmem + 167936));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 192512));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 196608));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 196608));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 196608));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 196608));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[12];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[64];
  float logsum[4];
  float scores_max[4];
  float acc_s[8];
  float scores_max_prev[4];
  float scores_scale[4];
  float scores_sum[4];
  float scores_max_clear[4];
  float scores_max_clear_1[4];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(K_pe_desc);
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(Q_pe_desc);
    tl::prefetch_tma_descriptor(KV_desc);
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
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(logsum + 0) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    float broadcast_var_2 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(8192);
      tl::tma_load(K_pe_desc, overlap_plan_mbar[4], (&(((half_t*)K_pe_shared)[0])), 0, 0, 0, 0);
    }
    overlap_plan_mbar[0].wait(0);
    overlap_plan_mbar[2].wait(0);
    {
      half_t A_local[16];
      half_t B_local[4];
      #pragma unroll
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_s + (i_1 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
      }
      for (int ki = 0; ki < 32; ++ki) {
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki >> 2) * 2048) + (i_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[(i_2 * 8)])));
        }
        tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (i_3 * 4)), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + 0));
        }
      }
    }
    overlap_plan_mbar[1].wait(0);
    overlap_plan_mbar[4].wait(0);
    {
      half_t A_local_1[16];
      half_t B_local_1[4];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        for (int i_4 = 0; i_4 < 2; ++i_4) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[(((i_4 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_1[(i_4 * 8)])));
        }
        tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[(((((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[0])));
        for (int i_5 = 0; i_5 < 2; ++i_5) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (i_5 * 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + 0));
        }
      }
    }
    overlap_plan_mbar[9].arrive();
    *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
    float broadcast_var_4 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      scores_max_clear[i_6] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 2; ++rv) {
        scores_max_clear[i_6] = max(scores_max_clear[i_6], acc_s[((i_6 * 2) + rv)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_max_clear[i_6] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_6], (&(((float*)workspace)[0])));
      scores_max_clear[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_6]);
      scores_max[i_6] = max(scores_max[i_6], scores_max_clear[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      scores_max[i_7] = max(scores_max[i_7], scores_max_prev[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      acc_s[i_8] = exp2f(((acc_s[i_8] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_8 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)S_shared)[((((((int)threadIdx.x) & 31) >> 3) * 512) + ((((((((int)threadIdx.x) & 31) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_s[0]), ((half_t)acc_s[1])), __pack_half2(((half_t)acc_s[2]), ((half_t)acc_s[3])), __pack_half2(((half_t)acc_s[4]), ((half_t)acc_s[5])), __pack_half2(((half_t)acc_s[6]), ((half_t)acc_s[7])));
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      scores_scale[i_9] = exp2f(((scores_max_prev[i_9] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_9] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      scores_sum[i_10] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
        scores_sum[i_10] = (scores_sum[i_10] + acc_s[((i_10 * 2) + rv_1)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_sum[i_10] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_10], (&(((float*)workspace_1)[0])));
      scores_sum[i_10] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      logsum[i_11] = ((logsum[i_11] * scores_scale[i_11]) + scores_sum[i_11]);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 64; ++i_12) {
      acc_o[i_12] = (acc_o[i_12] * scores_scale[(((i_12 >> 5) * 2) + ((i_12 & 3) >> 1))]);
    }
    for (int k = 0; k < 127; ++k) {
      if (2 <= k) {
        overlap_plan_mbar[(((k + 1) % 3) + 9)].wait((((k + 4) % 6) / 3));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[(((k + 1) % 3) + 4)].arrive_and_expect_tx(8192);
        tl::fence_proxy_async();
        tl::tma_load(K_pe_desc, overlap_plan_mbar[(((k + 1) % 3) + 4)], (&(((half_t*)K_pe_shared)[(((k + 1) % 3) * 4096)])), 0, ((k * 64) + 64), 0, 0);
      }
      overlap_plan_mbar[(((k + 1) & 1) + 2)].wait((((k + 1) & 3) >> 1));
      {
        half_t A_local_2[16];
        half_t B_local_2[4];
        #pragma unroll
        for (int i_13 = 0; i_13 < 2; ++i_13) {
          float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(acc_s + (i_13 * 4)) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
        }
        tl::__sync_thread_partial(3, 256);
        for (int ki_2 = 0; ki_2 < 32; ++ki_2) {
          for (int i_14 = 0; i_14 < 2; ++i_14) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki_2 >> 2) * 2048) + (i_14 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[(i_14 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + ((k + 1) & 1)) & 1) * 32768) + ((ki_2 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[0])));
          for (int i_15 = 0; i_15 < 2; ++i_15) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (i_15 * 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + 0));
          }
        }
      }
      overlap_plan_mbar[(((k + 1) % 3) + 4)].wait((((k + 1) % 6) / 3));
      {
        half_t A_local_3[16];
        half_t B_local_3[4];
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          for (int i_16 = 0; i_16 < 2; ++i_16) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[(((i_16 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_3[(i_16 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[(((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + ((k + 1) % 3)) % 3) * 4096) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[0])));
          for (int i_17 = 0; i_17 < 2; ++i_17) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (i_17 * 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_17 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + 0));
          }
        }
      }
      overlap_plan_mbar[(((k + 1) % 3) + 9)].arrive();
      overlap_plan_mbar[((k & 1) + 2)].wait(((k & 3) >> 1));
      {
        half_t A_local_4[16];
        half_t B_local_4[32];
        for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
          for (int i_18 = 0; i_18 < 2; ++i_18) {
            tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((i_18 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_4[(i_18 * 8)])));
          }
          for (int i_19 = 0; i_19 < 4; ++i_19) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((k & 1) * 32768) + ((((int)threadIdx.x) >> 5) * 4096)) + (ki_4 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_19 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_4[(i_19 * 8)])));
          }
          for (int i_20 = 0; i_20 < 2; ++i_20) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_20 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local_4 + (i_20 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + (j * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_20 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + (i_20 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + ((j * 8) + 4)));
            }
          }
        }
      }
      overlap_plan_mbar[((k & 1) + 7)].arrive();
      *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
      float broadcast_var_6 = -CUDART_INF_F;
      *(float4*)(scores_max + 0) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
      #pragma unroll
      for (int i_21 = 0; i_21 < 4; ++i_21) {
        scores_max_clear_1[i_21] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 2; ++rv_2) {
          scores_max_clear_1[i_21] = max(scores_max_clear_1[i_21], acc_s[((i_21 * 2) + rv_2)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear_1[i_21] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_21], (&(((float*)workspace_3)[0])));
        scores_max_clear_1[i_21] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_21]);
        scores_max[i_21] = max(scores_max[i_21], scores_max_clear_1[i_21]);
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 4; ++i_22) {
        scores_max[i_22] = max(scores_max[i_22], scores_max_prev[i_22]);
      }
      #pragma unroll
      for (int i_23 = 0; i_23 < 8; ++i_23) {
        acc_s[i_23] = exp2f(((acc_s[i_23] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_23 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)S_shared)[((((((int)threadIdx.x) & 31) >> 3) * 512) + ((((((((int)threadIdx.x) & 31) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_s[0]), ((half_t)acc_s[1])), __pack_half2(((half_t)acc_s[2]), ((half_t)acc_s[3])), __pack_half2(((half_t)acc_s[4]), ((half_t)acc_s[5])), __pack_half2(((half_t)acc_s[6]), ((half_t)acc_s[7])));
      #pragma unroll
      for (int i_24 = 0; i_24 < 4; ++i_24) {
        scores_scale[i_24] = exp2f(((scores_max_prev[i_24] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_24] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_25 = 0; i_25 < 4; ++i_25) {
        scores_sum[i_25] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 2; ++rv_3) {
          scores_sum[i_25] = (scores_sum[i_25] + acc_s[((i_25 * 2) + rv_3)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_25] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_25], (&(((float*)workspace_2)[0])));
        scores_sum[i_25] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_25]);
      }
      #pragma unroll
      for (int i_26 = 0; i_26 < 4; ++i_26) {
        logsum[i_26] = ((logsum[i_26] * scores_scale[i_26]) + scores_sum[i_26]);
      }
      #pragma unroll
      for (int i_27 = 0; i_27 < 64; ++i_27) {
        acc_o[i_27] = (acc_o[i_27] * scores_scale[(((i_27 >> 5) * 2) + ((i_27 & 3) >> 1))]);
      }
    }
    overlap_plan_mbar[3].wait(1);
    {
      half_t A_local_5[16];
      half_t B_local_5[32];
      for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
        for (int i_28 = 0; i_28 < 2; ++i_28) {
          tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((i_28 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_5[(i_28 * 8)])));
        }
        for (int i_29 = 0; i_29 < 4; ++i_29) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) >> 5) * 4096) + (ki_5 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_29 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_29 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_5[(i_29 * 8)])));
        }
        for (int i_30 = 0; i_30 < 2; ++i_30) {
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_30 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_5 + (i_30 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + (j_1 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_30 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + (i_30 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + ((j_1 * 8) + 4)));
          }
        }
      }
    }
    overlap_plan_mbar[8].arrive();
    #pragma unroll
    for (int i_31 = 0; i_31 < 64; ++i_31) {
      acc_o[i_31] = (acc_o[i_31] / logsum[(((i_31 >> 5) * 2) + ((i_31 & 3) >> 1))]);
    }
    #pragma unroll
    for (int i_32 = 0; i_32 < 8; ++i_32) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[((((((i_32 >> 2) * 8192) + ((((int)threadIdx.x) & 15) * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((i_32 & 3) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_32 * 8)]), ((half_t)acc_o[((i_32 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_32 * 8) + 2)]), ((half_t)acc_o[((i_32 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_32 * 8) + 4)]), ((half_t)acc_o[((i_32 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_32 * 8) + 6)]), ((half_t)acc_o[((i_32 * 8) + 7)])));
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
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
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
    for (int k_1 = 0; k_1 < 128; ++k_1) {
      if (2 <= k_1) {
        overlap_plan_mbar[((k_1 & 1) + 7)].wait((((k_1 >> 1) + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[((k_1 & 1) + 2)].arrive_and_expect_tx(65536);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k_1 & 1) + 2)], (&(((half_t*)KV_shared)[((k_1 & 1) * 32768)])), 0, (k_1 * 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k_1 & 1) + 2)], (&(((half_t*)KV_shared)[(((k_1 & 1) * 32768) + 4096)])), 64, (k_1 * 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k_1 & 1) + 2)], (&(((half_t*)KV_shared)[(((k_1 & 1) * 32768) + 8192)])), 128, (k_1 * 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k_1 & 1) + 2)], (&(((half_t*)KV_shared)[(((k_1 & 1) * 32768) + 12288)])), 192, (k_1 * 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k_1 & 1) + 2)], (&(((half_t*)KV_shared)[(((k_1 & 1) * 32768) + 16384)])), 256, (k_1 * 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k_1 & 1) + 2)], (&(((half_t*)KV_shared)[(((k_1 & 1) * 32768) + 20480)])), 320, (k_1 * 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k_1 & 1) + 2)], (&(((half_t*)KV_shared)[(((k_1 & 1) * 32768) + 24576)])), 384, (k_1 * 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k_1 & 1) + 2)], (&(((half_t*)KV_shared)[(((k_1 & 1) * 32768) + 28672)])), 448, (k_1 * 64), 0, 0);
      }
    }
  }
}

