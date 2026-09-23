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
  void* Q_pe_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* KV_shared = ((void*)((char*)buf_dyn_shmem + 18432));
  void* K_pe_shared = ((void*)((char*)buf_dyn_shmem + 149504));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 165888));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 169984));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 169984));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 169984));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 169984));
  void* workspace_4 = ((void*)((char*)buf_dyn_shmem + 169984));
  void* workspace_5 = ((void*)((char*)buf_dyn_shmem + 169984));
  void* workspace_6 = ((void*)((char*)buf_dyn_shmem + 169984));
  void* workspace_7 = ((void*)((char*)buf_dyn_shmem + 169984));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[11];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[32];
  float logsum[2];
  float scores_max[2];
  float acc_s[4];
  float scores_max_prev[2];
  float scores_scale_v0[2];
  float scores_sum[2];
  float scores_scale_v1[2];
  float scores_max_clear[2];
  float scores_max_clear_1[2];
  float scores_max_clear_2[2];
  float scores_max_clear_3[2];
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
    overlap_plan_mbar[6].init(256);
    overlap_plan_mbar[7].init(256);
    overlap_plan_mbar[8].init(256);
    overlap_plan_mbar[9].init(256);
    overlap_plan_mbar[10].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    tl::warpgroup_reg_alloc<240>();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(16384);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[0])), 0, (((int)blockIdx.x) * 16), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[1024])), 64, (((int)blockIdx.x) * 16), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[2048])), 128, (((int)blockIdx.x) * 16), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[3072])), 192, (((int)blockIdx.x) * 16), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[4096])), 256, (((int)blockIdx.x) * 16), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[5120])), 320, (((int)blockIdx.x) * 16), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[6144])), 384, (((int)blockIdx.x) * 16), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[7168])), 448, (((int)blockIdx.x) * 16), 0);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(logsum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
    float broadcast_var_2 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_2, broadcast_var_2);
    overlap_plan_mbar[0].wait(0);
    overlap_plan_mbar[2].wait(0);
    {
      half_t A_local[8];
      half_t B_local[4];
      float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + 0) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
      for (int ki = 0; ki < 32; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
      }
    }
    overlap_plan_mbar[1].wait(0);
    overlap_plan_mbar[4].wait(0);
    {
      half_t A_local_1[8];
      half_t B_local_1[4];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_1[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[(((((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
      }
    }
    overlap_plan_mbar[9].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_4 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_4, broadcast_var_4);
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      scores_max_clear[i_1] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 2; ++rv) {
        scores_max_clear[i_1] = max(scores_max_clear[i_1], acc_s[((i_1 * 2) + rv)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_max_clear[i_1] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_1], (&(((float*)workspace_2)[0])));
      scores_max_clear[i_1] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_1]);
      scores_max[i_1] = max(scores_max[i_1], scores_max_clear[i_1]);
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      scores_max[i_2] = max(scores_max[i_2], scores_max_prev[i_2]);
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      scores_scale_v0[i_3] = exp2f(((scores_max_prev[i_3] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_3] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      acc_s[i_4] = exp2f(((acc_s[i_4] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_4 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      scores_sum[i_5] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
        scores_sum[i_5] = (scores_sum[i_5] + acc_s[((i_5 * 2) + rv_1)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_sum[i_5] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_5], (&(((float*)workspace_4)[0])));
      scores_sum[i_5] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_5]);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      half_t S_shared_local_cast[2];
      uint1 __1;
      float2 v_ = *(float2*)(acc_s + (i_6 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(S_shared_local_cast + 0) = __1;
      *(uint1*)(((half_t*)S_shared) + ((((((i_6 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast + 0);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      logsum[i_7] = ((logsum[i_7] * scores_scale_v0[i_7]) + scores_sum[i_7]);
    }
    tl::__sync_thread_partial(3, 256);
    for (int k = 0; k < 63; ++k) {
      overlap_plan_mbar[3].wait((k & 1));
      {
        half_t A_local_2[8];
        half_t B_local_2[4];
        float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_s + 0) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
        for (int ki_2 = 0; ki_2 < 32; ++ki_2) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_2 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[0])));
          tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 1) & 1) * 32768) + ((ki_2 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[0])));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 0));
        }
      }
      overlap_plan_mbar[5].wait((k & 1));
      {
        half_t A_local_3[8];
        half_t B_local_3[4];
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_3[0])));
          tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[(((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 1) & 1) * 4096) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[0])));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 0));
        }
      }
      overlap_plan_mbar[10].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_6 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_6, broadcast_var_6);
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        scores_max_clear_1[i_8] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 2; ++rv_2) {
          scores_max_clear_1[i_8] = max(scores_max_clear_1[i_8], acc_s[((i_8 * 2) + rv_2)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear_1[i_8] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_8], (&(((float*)workspace_3)[0])));
        scores_max_clear_1[i_8] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_8]);
        scores_max[i_8] = max(scores_max[i_8], scores_max_clear_1[i_8]);
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        scores_max[i_9] = max(scores_max[i_9], scores_max_prev[i_9]);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        scores_scale_v1[i_10] = exp2f(((scores_max_prev[i_10] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_10] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 4; ++i_11) {
        acc_s[i_11] = exp2f(((acc_s[i_11] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_11 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        scores_sum[i_12] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 2; ++rv_3) {
          scores_sum[i_12] = (scores_sum[i_12] + acc_s[((i_12 * 2) + rv_3)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_12] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_12], (&(((float*)workspace_6)[0])));
        scores_sum[i_12] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_12]);
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        half_t S_shared_local_cast_1[2];
        uint1 __2;
        float2 v__1 = *(float2*)(acc_s + (i_13 * 2));
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
        *(uint1*)(S_shared_local_cast_1 + 0) = __2;
        *(uint1*)(((half_t*)S_shared) + (((((((i_13 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1024)) = *(uint1*)(S_shared_local_cast_1 + 0);
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        logsum[i_14] = ((logsum[i_14] * scores_scale_v1[i_14]) + scores_sum[i_14]);
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 32; ++i_15) {
        acc_o[i_15] = (acc_o[i_15] * scores_scale_v0[((i_15 & 3) >> 1)]);
      }
      overlap_plan_mbar[2].wait((k & 1));
      {
        half_t A_local_4[8];
        half_t B_local_4[32];
        for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
          tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_4[0])));
          for (int i_16 = 0; i_16 < 4; ++i_16) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki_4 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_16 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_4[(i_16 * 8)])));
          }
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j * 8)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + ((j * 8) + 4)));
          }
        }
      }
      overlap_plan_mbar[7].arrive();
      overlap_plan_mbar[2].wait(((k + 1) & 1));
      {
        half_t A_local_5[8];
        half_t B_local_5[4];
        float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_s + 0) = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
        tl::__sync_thread_partial(3, 256);
        for (int ki_5 = 0; ki_5 < 32; ++ki_5) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_5 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_5 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_5[0])));
          tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki_5 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_5 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_5[0])));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + 0));
        }
      }
      overlap_plan_mbar[4].wait(((k + 1) & 1));
      {
        half_t A_local_6[8];
        half_t B_local_6[4];
        for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_6 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_6[0])));
          tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[(((((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_6 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_6[0])));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local_6 + 0), reinterpret_cast<const unsigned*>(B_local_6 + 0));
        }
      }
      overlap_plan_mbar[9].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_8 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_8, broadcast_var_8);
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        scores_max_clear_2[i_17] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_4 = 0; rv_4 < 2; ++rv_4) {
          scores_max_clear_2[i_17] = max(scores_max_clear_2[i_17], acc_s[((i_17 * 2) + rv_4)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear_2[i_17] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_17], (&(((float*)workspace)[0])));
        scores_max_clear_2[i_17] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_17]);
        scores_max[i_17] = max(scores_max[i_17], scores_max_clear_2[i_17]);
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 2; ++i_18) {
        scores_max[i_18] = max(scores_max[i_18], scores_max_prev[i_18]);
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 2; ++i_19) {
        scores_scale_v0[i_19] = exp2f(((scores_max_prev[i_19] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_19] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 4; ++i_20) {
        acc_s[i_20] = exp2f(((acc_s[i_20] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_20 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        scores_sum[i_21] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_5 = 0; rv_5 < 2; ++rv_5) {
          scores_sum[i_21] = (scores_sum[i_21] + acc_s[((i_21 * 2) + rv_5)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_21] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_21], (&(((float*)workspace_1)[0])));
        scores_sum[i_21] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_21]);
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        half_t S_shared_local_cast_2[2];
        uint1 __3;
        float2 v__2 = *(float2*)(acc_s + (i_22 * 2));
        ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
        *(uint1*)(S_shared_local_cast_2 + 0) = __3;
        *(uint1*)(((half_t*)S_shared) + ((((((i_22 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast_2 + 0);
      }
      #pragma unroll
      for (int i_23 = 0; i_23 < 2; ++i_23) {
        logsum[i_23] = ((logsum[i_23] * scores_scale_v0[i_23]) + scores_sum[i_23]);
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 32; ++i_24) {
        acc_o[i_24] = (acc_o[i_24] * scores_scale_v1[((i_24 & 3) >> 1)]);
      }
      overlap_plan_mbar[3].wait((k & 1));
      {
        half_t A_local_7[8];
        half_t B_local_7[32];
        tl::__sync_thread_partial(3, 256);
        for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
          tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_7 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 1024)])), (&(A_local_7[0])));
          for (int i_25 = 0; i_25 < 4; ++i_25) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) >> 5) * 4096) + (ki_7 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_25 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_25 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_7[(i_25 * 8)])));
          }
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_7 + 0), reinterpret_cast<const unsigned*>(B_local_7 + (j_1 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_7 + 0), reinterpret_cast<const unsigned*>(B_local_7 + ((j_1 * 8) + 4)));
          }
        }
      }
      overlap_plan_mbar[8].arrive();
    }
    overlap_plan_mbar[3].wait(1);
    {
      half_t A_local_8[8];
      half_t B_local_8[4];
      float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + 0) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
      for (int ki_8 = 0; ki_8 < 32; ++ki_8) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_8 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_8[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 1) & 1) * 32768) + ((ki_8 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_8[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local_8 + 0), reinterpret_cast<const unsigned*>(B_local_8 + 0));
      }
    }
    overlap_plan_mbar[5].wait(1);
    {
      half_t A_local_9[8];
      half_t B_local_9[4];
      for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_9[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[(((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 1) & 1) * 4096) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_9[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local_9 + 0), reinterpret_cast<const unsigned*>(B_local_9 + 0));
      }
    }
    overlap_plan_mbar[10].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_10 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_10, broadcast_var_10);
    #pragma unroll
    for (int i_26 = 0; i_26 < 2; ++i_26) {
      scores_max_clear_3[i_26] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_6 = 0; rv_6 < 2; ++rv_6) {
        scores_max_clear_3[i_26] = max(scores_max_clear_3[i_26], acc_s[((i_26 * 2) + rv_6)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_max_clear_3[i_26] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_26], (&(((float*)workspace_7)[0])));
      scores_max_clear_3[i_26] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_26]);
      scores_max[i_26] = max(scores_max[i_26], scores_max_clear_3[i_26]);
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 2; ++i_27) {
      scores_max[i_27] = max(scores_max[i_27], scores_max_prev[i_27]);
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 2; ++i_28) {
      scores_scale_v1[i_28] = exp2f(((scores_max_prev[i_28] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_28] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 4; ++i_29) {
      acc_s[i_29] = exp2f(((acc_s[i_29] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_29 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 2; ++i_30) {
      scores_sum[i_30] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_7 = 0; rv_7 < 2; ++rv_7) {
        scores_sum[i_30] = (scores_sum[i_30] + acc_s[((i_30 * 2) + rv_7)]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_sum[i_30] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_30], (&(((float*)workspace_5)[0])));
      scores_sum[i_30] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_30]);
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 2; ++i_31) {
      half_t S_shared_local_cast_3[2];
      uint1 __4;
      float2 v__3 = *(float2*)(acc_s + (i_31 * 2));
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
      *(uint1*)(S_shared_local_cast_3 + 0) = __4;
      *(uint1*)(((half_t*)S_shared) + (((((((i_31 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1024)) = *(uint1*)(S_shared_local_cast_3 + 0);
    }
    #pragma unroll
    for (int i_32 = 0; i_32 < 2; ++i_32) {
      logsum[i_32] = ((logsum[i_32] * scores_scale_v1[i_32]) + scores_sum[i_32]);
    }
    #pragma unroll
    for (int i_33 = 0; i_33 < 32; ++i_33) {
      acc_o[i_33] = (acc_o[i_33] * scores_scale_v0[((i_33 & 3) >> 1)]);
    }
    overlap_plan_mbar[2].wait(1);
    {
      half_t A_local_10[8];
      half_t B_local_10[32];
      for (int ki_10 = 0; ki_10 < 4; ++ki_10) {
        tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_10 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_10[0])));
        for (int i_34 = 0; i_34 < 4; ++i_34) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki_10 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_34 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_34 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_10[(i_34 * 8)])));
        }
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_10 + 0), reinterpret_cast<const unsigned*>(B_local_10 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_10 + 0), reinterpret_cast<const unsigned*>(B_local_10 + ((j_2 * 8) + 4)));
        }
      }
    }
    overlap_plan_mbar[7].arrive();
    #pragma unroll
    for (int i_35 = 0; i_35 < 32; ++i_35) {
      acc_o[i_35] = (acc_o[i_35] * scores_scale_v1[((i_35 & 3) >> 1)]);
    }
    overlap_plan_mbar[3].wait(1);
    {
      half_t A_local_11[8];
      half_t B_local_11[32];
      tl::__sync_thread_partial(3, 256);
      for (int ki_11 = 0; ki_11 < 4; ++ki_11) {
        tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 1024)])), (&(A_local_11[0])));
        for (int i_36 = 0; i_36 < 4; ++i_36) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) >> 5) * 4096) + (ki_11 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_36 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_36 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_11[(i_36 * 8)])));
        }
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_3 * 8)), reinterpret_cast<const unsigned*>(A_local_11 + 0), reinterpret_cast<const unsigned*>(B_local_11 + (j_3 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_11 + 0), reinterpret_cast<const unsigned*>(B_local_11 + ((j_3 * 8) + 4)));
        }
      }
    }
    overlap_plan_mbar[8].arrive();
    #pragma unroll
    for (int i_37 = 0; i_37 < 32; ++i_37) {
      acc_o[i_37] = (acc_o[i_37] / logsum[((i_37 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 4; ++i_38) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((int)threadIdx.x) & 15) * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (i_38 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_38 * 8)]), ((half_t)acc_o[((i_38 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_38 * 8) + 2)]), ((half_t)acc_o[((i_38 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_38 * 8) + 4)]), ((half_t)acc_o[((i_38 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_38 * 8) + 6)]), ((half_t)acc_o[((i_38 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[6].arrive();
  } else {
    tl::warpgroup_reg_dealloc<24>();
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(2048);
      tl::tma_load(Q_pe_desc, overlap_plan_mbar[1], (&(((half_t*)Q_pe_shared)[0])), 0, (((int)blockIdx.x) * 16), 0);
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
      if (2 <= k_1) {
        overlap_plan_mbar[((k_1 & 1) + 9)].wait((((k_1 >> 1) + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[((k_1 & 1) + 4)].arrive_and_expect_tx(8192);
        tl::tma_load(K_pe_desc, overlap_plan_mbar[((k_1 & 1) + 4)], (&(((half_t*)K_pe_shared)[((k_1 & 1) * 4096)])), 0, (k_1 * 64), 0, 0);
      }
    }
    overlap_plan_mbar[6].wait(0);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store((&(Output[(((int)blockIdx.x) * 8192)])), (&(((half_t*)O_shared)[0])), 16384);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

