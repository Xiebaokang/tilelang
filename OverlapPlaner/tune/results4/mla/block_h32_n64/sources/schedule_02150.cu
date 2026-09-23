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
  void* workspace = ((void*)((char*)buf_dyn_shmem + 200704));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 200704));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 200704));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 200704));
  void* workspace_4 = ((void*)((char*)buf_dyn_shmem + 200704));
  void* workspace_5 = ((void*)((char*)buf_dyn_shmem + 200704));
  void* workspace_6 = ((void*)((char*)buf_dyn_shmem + 200704));
  void* workspace_7 = ((void*)((char*)buf_dyn_shmem + 200704));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[13];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[64];
  float logsum[4];
  float scores_max[4];
  float acc_s_v0[8];
  float scores_max_prev[4];
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
    overlap_plan_mbar[0].wait(0);
    overlap_plan_mbar[2].wait(0);
    {
      half_t A_local[16];
      half_t B_local[4];
      #pragma unroll
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_s_v0 + (i_1 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
      }
      for (int ki = 0; ki < 32; ++ki) {
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki >> 2) * 2048) + (i_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[(i_2 * 8)])));
        }
        tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + (i_3 * 4)), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + 0));
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
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + (i_5 * 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + 0));
        }
      }
    }
    overlap_plan_mbar[10].arrive();
    *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
    float broadcast_var_4 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      scores_max_clear[i_6] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 2; ++rv) {
        scores_max_clear[i_6] = max(scores_max_clear[i_6], acc_s_v0[((i_6 * 2) + rv)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_max_clear[i_6] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_6], (&(((float*)workspace_3)[0])));
      scores_max_clear[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_6]);
      scores_max[i_6] = max(scores_max[i_6], scores_max_clear[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      scores_max[i_7] = max(scores_max[i_7], scores_max_prev[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      scores_scale_v0[i_8] = exp2f(((scores_max_prev[i_8] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_8] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 8; ++i_9) {
      acc_s_v0[i_9] = exp2f(((acc_s_v0[i_9] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_9 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      scores_sum[i_10] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
        scores_sum[i_10] = (scores_sum[i_10] + acc_s_v0[((i_10 * 2) + rv_1)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_sum[i_10] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_10], (&(((float*)workspace_6)[0])));
      scores_sum[i_10] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      logsum[i_11] = ((logsum[i_11] * scores_scale_v0[i_11]) + scores_sum[i_11]);
    }
    for (int k = 0; k < 63; ++k) {
      overlap_plan_mbar[3].wait((k & 1));
      {
        half_t A_local_2[16];
        half_t B_local_2[4];
        #pragma unroll
        for (int i_12 = 0; i_12 < 2; ++i_12) {
          float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(acc_s_v1 + (i_12 * 4)) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
        }
        for (int ki_2 = 0; ki_2 < 32; ++ki_2) {
          for (int i_13 = 0; i_13 < 2; ++i_13) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki_2 >> 2) * 2048) + (i_13 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[(i_13 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 1) & 1) * 32768) + ((ki_2 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[0])));
          for (int i_14 = 0; i_14 < 2; ++i_14) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + (i_14 * 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_14 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + 0));
          }
        }
      }
      overlap_plan_mbar[((((k * 2) + 1) % 3) + 4)].wait(((((k % 3) * 2) + 1) / 3));
      {
        half_t A_local_3[16];
        half_t B_local_3[4];
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          for (int i_15 = 0; i_15 < 2; ++i_15) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[(((i_15 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_3[(i_15 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[(((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + (((k * 2) + 1) % 3)) % 3) * 4096) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[0])));
          for (int i_16 = 0; i_16 < 2; ++i_16) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + (i_16 * 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_16 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + 0));
          }
        }
      }
      overlap_plan_mbar[((((k * 2) + 1) % 3) + 10)].arrive();
      *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
      float broadcast_var_6 = -CUDART_INF_F;
      *(float4*)(scores_max + 0) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
      #pragma unroll
      for (int i_17 = 0; i_17 < 4; ++i_17) {
        scores_max_clear_1[i_17] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 2; ++rv_2) {
          scores_max_clear_1[i_17] = max(scores_max_clear_1[i_17], acc_s_v1[((i_17 * 2) + rv_2)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear_1[i_17] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_17], (&(((float*)workspace_4)[0])));
        scores_max_clear_1[i_17] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_17]);
        scores_max[i_17] = max(scores_max[i_17], scores_max_clear_1[i_17]);
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 4; ++i_18) {
        scores_max[i_18] = max(scores_max[i_18], scores_max_prev[i_18]);
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 4; ++i_19) {
        scores_scale_v1[i_19] = exp2f(((scores_max_prev[i_19] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_19] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 8; ++i_20) {
        acc_s_v1[i_20] = exp2f(((acc_s_v1[i_20] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_20 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 4; ++i_21) {
        scores_sum[i_21] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 2; ++rv_3) {
          scores_sum[i_21] = (scores_sum[i_21] + acc_s_v1[((i_21 * 2) + rv_3)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_21] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_21], (&(((float*)workspace_7)[0])));
        scores_sum[i_21] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_21]);
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 4; ++i_22) {
        half_t S_shared_local_cast[2];
        uint1 __1;
        float2 v_ = *(float2*)(acc_s_v0 + (i_22 * 2));
        ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
        *(uint1*)(S_shared_local_cast + 0) = __1;
        *(uint1*)(((half_t*)S_shared) + ((((((i_22 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast + 0);
      }
      #pragma unroll
      for (int i_23 = 0; i_23 < 4; ++i_23) {
        logsum[i_23] = ((logsum[i_23] * scores_scale_v1[i_23]) + scores_sum[i_23]);
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 64; ++i_24) {
        acc_o[i_24] = (acc_o[i_24] * scores_scale_v0[(((i_24 >> 5) * 2) + ((i_24 & 3) >> 1))]);
      }
      overlap_plan_mbar[2].wait((k & 1));
      {
        half_t A_local_4[16];
        half_t B_local_4[32];
        tl::__sync_thread_partial(3, 256);
        for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
          for (int i_25 = 0; i_25 < 2; ++i_25) {
            tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((i_25 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_4[(i_25 * 8)])));
          }
          for (int i_26 = 0; i_26 < 4; ++i_26) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki_4 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_26 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_26 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_4[(i_26 * 8)])));
          }
          for (int i_27 = 0; i_27 < 2; ++i_27) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_27 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local_4 + (i_27 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + (j * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_27 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + (i_27 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + ((j * 8) + 4)));
            }
          }
        }
      }
      overlap_plan_mbar[8].arrive();
      overlap_plan_mbar[2].wait(((k + 1) & 1));
      {
        half_t A_local_5[16];
        half_t B_local_5[4];
        #pragma unroll
        for (int i_28 = 0; i_28 < 2; ++i_28) {
          float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(acc_s_v0 + (i_28 * 4)) = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
        }
        for (int ki_5 = 0; ki_5 < 32; ++ki_5) {
          for (int i_29 = 0; i_29 < 2; ++i_29) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki_5 >> 2) * 2048) + (i_29 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_5 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_5[(i_29 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki_5 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_5 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_5[0])));
          for (int i_30 = 0; i_30 < 2; ++i_30) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + (i_30 * 4)), reinterpret_cast<const unsigned*>(A_local_5 + (i_30 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + 0));
          }
        }
      }
      overlap_plan_mbar[((((k * 2) + 2) % 3) + 4)].wait(((((k * 2) + 2) % 6) / 3));
      {
        half_t A_local_6[16];
        half_t B_local_6[4];
        for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
          for (int i_31 = 0; i_31 < 2; ++i_31) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[(((i_31 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_6 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_6[(i_31 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[(((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + (((k * 2) + 2) % 3)) % 3) * 4096) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_6 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_6[0])));
          for (int i_32 = 0; i_32 < 2; ++i_32) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + (i_32 * 4)), reinterpret_cast<const unsigned*>(A_local_6 + (i_32 * 8)), reinterpret_cast<const unsigned*>(B_local_6 + 0));
          }
        }
      }
      overlap_plan_mbar[((((k * 2) + 2) % 3) + 10)].arrive();
      *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
      float broadcast_var_8 = -CUDART_INF_F;
      *(float4*)(scores_max + 0) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
      #pragma unroll
      for (int i_33 = 0; i_33 < 4; ++i_33) {
        scores_max_clear_2[i_33] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_4 = 0; rv_4 < 2; ++rv_4) {
          scores_max_clear_2[i_33] = max(scores_max_clear_2[i_33], acc_s_v0[((i_33 * 2) + rv_4)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear_2[i_33] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_33], (&(((float*)workspace_5)[0])));
        scores_max_clear_2[i_33] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_33]);
        scores_max[i_33] = max(scores_max[i_33], scores_max_clear_2[i_33]);
      }
      #pragma unroll
      for (int i_34 = 0; i_34 < 4; ++i_34) {
        scores_max[i_34] = max(scores_max[i_34], scores_max_prev[i_34]);
      }
      #pragma unroll
      for (int i_35 = 0; i_35 < 4; ++i_35) {
        scores_scale_v0[i_35] = exp2f(((scores_max_prev[i_35] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_35] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_36 = 0; i_36 < 8; ++i_36) {
        acc_s_v0[i_36] = exp2f(((acc_s_v0[i_36] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_36 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_37 = 0; i_37 < 4; ++i_37) {
        scores_sum[i_37] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_5 = 0; rv_5 < 2; ++rv_5) {
          scores_sum[i_37] = (scores_sum[i_37] + acc_s_v0[((i_37 * 2) + rv_5)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_37] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_37], (&(((float*)workspace_2)[0])));
        scores_sum[i_37] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_37]);
      }
      #pragma unroll
      for (int i_38 = 0; i_38 < 4; ++i_38) {
        half_t S_shared_local_cast_1[2];
        uint1 __2;
        float2 v__1 = *(float2*)(acc_s_v1 + (i_38 * 2));
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
        *(uint1*)(S_shared_local_cast_1 + 0) = __2;
        *(uint1*)(((half_t*)S_shared) + (((((((i_38 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048)) = *(uint1*)(S_shared_local_cast_1 + 0);
      }
      #pragma unroll
      for (int i_39 = 0; i_39 < 4; ++i_39) {
        logsum[i_39] = ((logsum[i_39] * scores_scale_v0[i_39]) + scores_sum[i_39]);
      }
      #pragma unroll
      for (int i_40 = 0; i_40 < 64; ++i_40) {
        acc_o[i_40] = (acc_o[i_40] * scores_scale_v1[(((i_40 >> 5) * 2) + ((i_40 & 3) >> 1))]);
      }
      overlap_plan_mbar[3].wait((k & 1));
      {
        half_t A_local_7[16];
        half_t B_local_7[32];
        tl::__sync_thread_partial(3, 256);
        for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
          for (int i_41 = 0; i_41 < 2; ++i_41) {
            tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[((((i_41 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_7 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 2048)])), (&(A_local_7[(i_41 * 8)])));
          }
          for (int i_42 = 0; i_42 < 4; ++i_42) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) >> 5) * 4096) + (ki_7 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_42 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_42 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_7[(i_42 * 8)])));
          }
          for (int i_43 = 0; i_43 < 2; ++i_43) {
            for (int j_1 = 0; j_1 < 4; ++j_1) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_43 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_7 + (i_43 * 8)), reinterpret_cast<const unsigned*>(B_local_7 + (j_1 * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_43 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_7 + (i_43 * 8)), reinterpret_cast<const unsigned*>(B_local_7 + ((j_1 * 8) + 4)));
            }
          }
        }
      }
      overlap_plan_mbar[9].arrive();
    }
    overlap_plan_mbar[3].wait(1);
    {
      half_t A_local_8[16];
      half_t B_local_8[4];
      #pragma unroll
      for (int i_44 = 0; i_44 < 2; ++i_44) {
        float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_s_v1 + (i_44 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
      }
      for (int ki_8 = 0; ki_8 < 32; ++ki_8) {
        for (int i_45 = 0; i_45 < 2; ++i_45) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki_8 >> 2) * 2048) + (i_45 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_8[(i_45 * 8)])));
        }
        tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 1) & 1) * 32768) + ((ki_8 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_8[0])));
        for (int i_46 = 0; i_46 < 2; ++i_46) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + (i_46 * 4)), reinterpret_cast<const unsigned*>(A_local_8 + (i_46 * 8)), reinterpret_cast<const unsigned*>(B_local_8 + 0));
        }
      }
    }
    overlap_plan_mbar[5].wait(0);
    {
      half_t A_local_9[16];
      half_t B_local_9[4];
      for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
        for (int i_47 = 0; i_47 < 2; ++i_47) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[(((i_47 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_9[(i_47 * 8)])));
        }
        tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(B_local_9[0])));
        for (int i_48 = 0; i_48 < 2; ++i_48) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + (i_48 * 4)), reinterpret_cast<const unsigned*>(A_local_9 + (i_48 * 8)), reinterpret_cast<const unsigned*>(B_local_9 + 0));
        }
      }
    }
    overlap_plan_mbar[11].arrive();
    *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
    float broadcast_var_10 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_10, broadcast_var_10, broadcast_var_10, broadcast_var_10);
    #pragma unroll
    for (int i_49 = 0; i_49 < 4; ++i_49) {
      scores_max_clear_3[i_49] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_6 = 0; rv_6 < 2; ++rv_6) {
        scores_max_clear_3[i_49] = max(scores_max_clear_3[i_49], acc_s_v1[((i_49 * 2) + rv_6)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_max_clear_3[i_49] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_49], (&(((float*)workspace)[0])));
      scores_max_clear_3[i_49] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_49]);
      scores_max[i_49] = max(scores_max[i_49], scores_max_clear_3[i_49]);
    }
    #pragma unroll
    for (int i_50 = 0; i_50 < 4; ++i_50) {
      scores_max[i_50] = max(scores_max[i_50], scores_max_prev[i_50]);
    }
    #pragma unroll
    for (int i_51 = 0; i_51 < 4; ++i_51) {
      scores_scale_v1[i_51] = exp2f(((scores_max_prev[i_51] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_51] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_52 = 0; i_52 < 8; ++i_52) {
      acc_s_v1[i_52] = exp2f(((acc_s_v1[i_52] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_52 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_53 = 0; i_53 < 4; ++i_53) {
      scores_sum[i_53] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_7 = 0; rv_7 < 2; ++rv_7) {
        scores_sum[i_53] = (scores_sum[i_53] + acc_s_v1[((i_53 * 2) + rv_7)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_sum[i_53] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_53], (&(((float*)workspace_1)[0])));
      scores_sum[i_53] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_53]);
    }
    #pragma unroll
    for (int i_54 = 0; i_54 < 4; ++i_54) {
      half_t S_shared_local_cast_2[2];
      uint1 __3;
      float2 v__2 = *(float2*)(acc_s_v0 + (i_54 * 2));
      ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
      *(uint1*)(S_shared_local_cast_2 + 0) = __3;
      *(uint1*)(((half_t*)S_shared) + ((((((i_54 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast_2 + 0);
    }
    #pragma unroll
    for (int i_55 = 0; i_55 < 4; ++i_55) {
      logsum[i_55] = ((logsum[i_55] * scores_scale_v1[i_55]) + scores_sum[i_55]);
    }
    #pragma unroll
    for (int i_56 = 0; i_56 < 64; ++i_56) {
      acc_o[i_56] = (acc_o[i_56] * scores_scale_v0[(((i_56 >> 5) * 2) + ((i_56 & 3) >> 1))]);
    }
    overlap_plan_mbar[2].wait(1);
    {
      half_t A_local_10[16];
      half_t B_local_10[32];
      tl::__sync_thread_partial(3, 256);
      for (int ki_10 = 0; ki_10 < 4; ++ki_10) {
        for (int i_57 = 0; i_57 < 2; ++i_57) {
          tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((i_57 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_10 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_10[(i_57 * 8)])));
        }
        for (int i_58 = 0; i_58 < 4; ++i_58) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki_10 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_58 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_58 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_10[(i_58 * 8)])));
        }
        for (int i_59 = 0; i_59 < 2; ++i_59) {
          for (int j_2 = 0; j_2 < 4; ++j_2) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_59 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_10 + (i_59 * 8)), reinterpret_cast<const unsigned*>(B_local_10 + (j_2 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_59 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_10 + (i_59 * 8)), reinterpret_cast<const unsigned*>(B_local_10 + ((j_2 * 8) + 4)));
          }
        }
      }
    }
    overlap_plan_mbar[8].arrive();
    tl::__sync_thread_partial(3, 256);
    #pragma unroll
    for (int i_60 = 0; i_60 < 4; ++i_60) {
      half_t S_shared_local_cast_3[2];
      uint1 __4;
      float2 v__3 = *(float2*)(acc_s_v1 + (i_60 * 2));
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
      *(uint1*)(S_shared_local_cast_3 + 0) = __4;
      *(uint1*)(((half_t*)S_shared) + (((((((i_60 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048)) = *(uint1*)(S_shared_local_cast_3 + 0);
    }
    #pragma unroll
    for (int i_61 = 0; i_61 < 64; ++i_61) {
      acc_o[i_61] = (acc_o[i_61] * scores_scale_v1[(((i_61 >> 5) * 2) + ((i_61 & 3) >> 1))]);
    }
    overlap_plan_mbar[3].wait(1);
    {
      half_t A_local_11[16];
      half_t B_local_11[32];
      tl::__sync_thread_partial(3, 256);
      for (int ki_11 = 0; ki_11 < 4; ++ki_11) {
        for (int i_62 = 0; i_62 < 2; ++i_62) {
          tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[((((i_62 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 2048)])), (&(A_local_11[(i_62 * 8)])));
        }
        for (int i_63 = 0; i_63 < 4; ++i_63) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) >> 5) * 4096) + (ki_11 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_63 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_63 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_11[(i_63 * 8)])));
        }
        for (int i_64 = 0; i_64 < 2; ++i_64) {
          for (int j_3 = 0; j_3 < 4; ++j_3) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_64 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_11 + (i_64 * 8)), reinterpret_cast<const unsigned*>(B_local_11 + (j_3 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_64 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_11 + (i_64 * 8)), reinterpret_cast<const unsigned*>(B_local_11 + ((j_3 * 8) + 4)));
          }
        }
      }
    }
    overlap_plan_mbar[9].arrive();
    #pragma unroll
    for (int i_65 = 0; i_65 < 64; ++i_65) {
      acc_o[i_65] = (acc_o[i_65] / logsum[(((i_65 >> 5) * 2) + ((i_65 & 3) >> 1))]);
    }
    #pragma unroll
    for (int i_66 = 0; i_66 < 8; ++i_66) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[((((((i_66 >> 2) * 8192) + ((((int)threadIdx.x) & 15) * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((i_66 & 3) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_66 * 8)]), ((half_t)acc_o[((i_66 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_66 * 8) + 2)]), ((half_t)acc_o[((i_66 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_66 * 8) + 4)]), ((half_t)acc_o[((i_66 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_66 * 8) + 6)]), ((half_t)acc_o[((i_66 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[7].arrive();
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
        overlap_plan_mbar[((k_1 & 1) + 8)].wait((((k_1 >> 1) + 1) & 1));
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
      if (3 <= k_1) {
        overlap_plan_mbar[((k_1 % 3) + 10)].wait((((k_1 / 3) + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[((k_1 % 3) + 4)].arrive_and_expect_tx(8192);
        tl::tma_load(K_pe_desc, overlap_plan_mbar[((k_1 % 3) + 4)], (&(((half_t*)K_pe_shared)[((k_1 % 3) * 4096)])), 0, (k_1 * 64), 0, 0);
      }
    }
    overlap_plan_mbar[7].wait(0);
    tl::__sync_thread_partial(4, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store((&(Output[(((int)blockIdx.x) * 16384)])), (&(((half_t*)O_shared)[0])), 32768);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

