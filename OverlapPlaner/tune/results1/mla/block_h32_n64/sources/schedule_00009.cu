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
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap KV_desc, __grid_constant__ const CUtensorMap K_pe_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap Q_pe_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_pe_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* KV_shared = ((void*)((char*)buf_dyn_shmem + 36864));
  void* K_pe_shared = ((void*)((char*)buf_dyn_shmem + 167936));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 176128));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 184320));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 184320));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 184320));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 184320));
  void* workspace_4 = ((void*)((char*)buf_dyn_shmem + 184320));
  void* workspace_5 = ((void*)((char*)buf_dyn_shmem + 184320));
  void* workspace_6 = ((void*)((char*)buf_dyn_shmem + 184320));
  void* workspace_7 = ((void*)((char*)buf_dyn_shmem + 184320));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[8];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[64];
  float logsum[4];
  float scores_max[4];
  float acc_s[8];
  float scores_max_prev[4];
  float scores_max_clear[4];
  float scores_scale_v0[4];
  float scores_sum[4];
  float scores_scale_v1[4];
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
    overlap_plan_mbar[5].init(256);
    overlap_plan_mbar[6].init(256);
    overlap_plan_mbar[7].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
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
    overlap_plan_mbar[2].arrive_and_expect_tx(65536);
    tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[0])), 0, 0, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[4096])), 64, 0, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[8192])), 128, 0, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[12288])), 192, 0, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[16384])), 256, 0, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[20480])), 320, 0, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[24576])), 384, 0, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[28672])), 448, 0, 0, 0);
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
      tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[0])));
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (i_5 * 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + 0));
      }
    }
  }
  overlap_plan_mbar[7].arrive();
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
    __syncthreads();
    scores_max_clear[i_6] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_6], (&(((float*)workspace_7)[0])));
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
    acc_s[i_9] = exp2f(((acc_s[i_9] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_9 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 4; ++i_10) {
    scores_sum[i_10] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
      scores_sum[i_10] = (scores_sum[i_10] + acc_s[((i_10 * 2) + rv_1)]);
    }
    __syncthreads();
    scores_sum[i_10] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_10], (&(((float*)workspace)[0])));
    scores_sum[i_10] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_10]);
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 4; ++i_11) {
    half_t S_shared_local_cast[2];
    uint1 __1;
    float2 v_ = *(float2*)(acc_s + (i_11 * 2));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    *(uint1*)(S_shared_local_cast + 0) = __1;
    *(uint1*)(((half_t*)S_shared) + ((((((i_11 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast + 0);
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 4; ++i_12) {
    logsum[i_12] = ((logsum[i_12] * scores_scale_v0[i_12]) + scores_sum[i_12]);
  }
  for (int k = 0; k < 63; ++k) {
    if (1 <= k) {
      overlap_plan_mbar[6].wait(((k + 1) & 1));
    }
    __syncthreads();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(65536);
      tl::fence_proxy_async();
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[32768])), 0, ((k * 128) + 64), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[36864])), 64, ((k * 128) + 64), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[40960])), 128, ((k * 128) + 64), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[45056])), 192, ((k * 128) + 64), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[49152])), 256, ((k * 128) + 64), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[53248])), 320, ((k * 128) + 64), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[57344])), 384, ((k * 128) + 64), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[61440])), 448, ((k * 128) + 64), 0, 0);
    }
    overlap_plan_mbar[7].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(8192);
      tl::fence_proxy_async();
      tl::tma_load(K_pe_desc, overlap_plan_mbar[4], (&(((half_t*)K_pe_shared)[0])), 0, ((k * 128) + 64), 0, 0);
    }
    overlap_plan_mbar[3].wait((k & 1));
    {
      half_t A_local_2[16];
      half_t B_local_2[4];
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_s + (i_13 * 4)) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
      }
      for (int ki_2 = 0; ki_2 < 32; ++ki_2) {
        for (int i_14 = 0; i_14 < 2; ++i_14) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki_2 >> 2) * 2048) + (i_14 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[(i_14 * 8)])));
        }
        tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 1) & 1) * 32768) + ((ki_2 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[0])));
        for (int i_15 = 0; i_15 < 2; ++i_15) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (i_15 * 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + 0));
        }
      }
    }
    overlap_plan_mbar[4].wait(1);
    {
      half_t A_local_3[16];
      half_t B_local_3[4];
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        for (int i_16 = 0; i_16 < 2; ++i_16) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[(((i_16 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_3[(i_16 * 8)])));
        }
        tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[0])));
        for (int i_17 = 0; i_17 < 2; ++i_17) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (i_17 * 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_17 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + 0));
        }
      }
    }
    overlap_plan_mbar[7].arrive();
    *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
    float broadcast_var_6 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
    #pragma unroll
    for (int i_18 = 0; i_18 < 4; ++i_18) {
      scores_max_clear_1[i_18] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 2; ++rv_2) {
        scores_max_clear_1[i_18] = max(scores_max_clear_1[i_18], acc_s[((i_18 * 2) + rv_2)]);
      }
      __syncthreads();
      scores_max_clear_1[i_18] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_18], (&(((float*)workspace_1)[0])));
      scores_max_clear_1[i_18] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_18]);
      scores_max[i_18] = max(scores_max[i_18], scores_max_clear_1[i_18]);
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 4; ++i_19) {
      scores_max[i_19] = max(scores_max[i_19], scores_max_prev[i_19]);
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 4; ++i_20) {
      scores_scale_v1[i_20] = exp2f(((scores_max_prev[i_20] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_20] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 8; ++i_21) {
      acc_s[i_21] = exp2f(((acc_s[i_21] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_21 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 4; ++i_22) {
      scores_sum[i_22] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 2; ++rv_3) {
        scores_sum[i_22] = (scores_sum[i_22] + acc_s[((i_22 * 2) + rv_3)]);
      }
      __syncthreads();
      scores_sum[i_22] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_22], (&(((float*)workspace_6)[0])));
      scores_sum[i_22] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_22]);
    }
    #pragma unroll
    for (int i_23 = 0; i_23 < 4; ++i_23) {
      half_t S_shared_local_cast_1[2];
      uint1 __2;
      float2 v__1 = *(float2*)(acc_s + (i_23 * 2));
      ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
      *(uint1*)(S_shared_local_cast_1 + 0) = __2;
      *(uint1*)(((half_t*)S_shared) + (((((((i_23 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048)) = *(uint1*)(S_shared_local_cast_1 + 0);
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 4; ++i_24) {
      logsum[i_24] = ((logsum[i_24] * scores_scale_v1[i_24]) + scores_sum[i_24]);
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 64; ++i_25) {
      acc_o[i_25] = (acc_o[i_25] * scores_scale_v0[(((i_25 >> 5) * 2) + ((i_25 & 3) >> 1))]);
    }
    overlap_plan_mbar[2].wait((k & 1));
    {
      half_t A_local_4[16];
      half_t B_local_4[32];
      __syncthreads();
      for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
        for (int i_26 = 0; i_26 < 2; ++i_26) {
          tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((i_26 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_4[(i_26 * 8)])));
        }
        for (int i_27 = 0; i_27 < 4; ++i_27) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki_4 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_27 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_27 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_4[(i_27 * 8)])));
        }
        for (int i_28 = 0; i_28 < 2; ++i_28) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_28 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local_4 + (i_28 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_28 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + (i_28 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + ((j * 8) + 4)));
          }
        }
      }
    }
    overlap_plan_mbar[5].arrive();
    overlap_plan_mbar[5].wait((k & 1));
    __syncthreads();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(65536);
      tl::fence_proxy_async();
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[0])), 0, ((k * 128) + 128), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[4096])), 64, ((k * 128) + 128), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[8192])), 128, ((k * 128) + 128), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[12288])), 192, ((k * 128) + 128), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[16384])), 256, ((k * 128) + 128), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[20480])), 320, ((k * 128) + 128), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[24576])), 384, ((k * 128) + 128), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[28672])), 448, ((k * 128) + 128), 0, 0);
    }
    overlap_plan_mbar[7].wait(1);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(8192);
      tl::fence_proxy_async();
      tl::tma_load(K_pe_desc, overlap_plan_mbar[4], (&(((half_t*)K_pe_shared)[0])), 0, ((k * 128) + 128), 0, 0);
    }
    overlap_plan_mbar[2].wait(((k + 1) & 1));
    {
      half_t A_local_5[16];
      half_t B_local_5[4];
      #pragma unroll
      for (int i_29 = 0; i_29 < 2; ++i_29) {
        float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_s + (i_29 * 4)) = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
      }
      for (int ki_5 = 0; ki_5 < 32; ++ki_5) {
        for (int i_30 = 0; i_30 < 2; ++i_30) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki_5 >> 2) * 2048) + (i_30 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_5 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_5[(i_30 * 8)])));
        }
        tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki_5 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_5 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_5[0])));
        for (int i_31 = 0; i_31 < 2; ++i_31) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (i_31 * 4)), reinterpret_cast<const unsigned*>(A_local_5 + (i_31 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + 0));
        }
      }
    }
    overlap_plan_mbar[4].wait(0);
    {
      half_t A_local_6[16];
      half_t B_local_6[4];
      for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
        for (int i_32 = 0; i_32 < 2; ++i_32) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[(((i_32 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_6 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_6[(i_32 * 8)])));
        }
        tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_6 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_6[0])));
        for (int i_33 = 0; i_33 < 2; ++i_33) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (i_33 * 4)), reinterpret_cast<const unsigned*>(A_local_6 + (i_33 * 8)), reinterpret_cast<const unsigned*>(B_local_6 + 0));
        }
      }
    }
    overlap_plan_mbar[7].arrive();
    *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
    float broadcast_var_8 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
    #pragma unroll
    for (int i_34 = 0; i_34 < 4; ++i_34) {
      scores_max_clear_2[i_34] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_4 = 0; rv_4 < 2; ++rv_4) {
        scores_max_clear_2[i_34] = max(scores_max_clear_2[i_34], acc_s[((i_34 * 2) + rv_4)]);
      }
      __syncthreads();
      scores_max_clear_2[i_34] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_34], (&(((float*)workspace_3)[0])));
      scores_max_clear_2[i_34] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_34]);
      scores_max[i_34] = max(scores_max[i_34], scores_max_clear_2[i_34]);
    }
    #pragma unroll
    for (int i_35 = 0; i_35 < 4; ++i_35) {
      scores_max[i_35] = max(scores_max[i_35], scores_max_prev[i_35]);
    }
    #pragma unroll
    for (int i_36 = 0; i_36 < 4; ++i_36) {
      scores_scale_v0[i_36] = exp2f(((scores_max_prev[i_36] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_36] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_37 = 0; i_37 < 8; ++i_37) {
      acc_s[i_37] = exp2f(((acc_s[i_37] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_37 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 4; ++i_38) {
      scores_sum[i_38] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_5 = 0; rv_5 < 2; ++rv_5) {
        scores_sum[i_38] = (scores_sum[i_38] + acc_s[((i_38 * 2) + rv_5)]);
      }
      __syncthreads();
      scores_sum[i_38] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_38], (&(((float*)workspace_4)[0])));
      scores_sum[i_38] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_38]);
    }
    #pragma unroll
    for (int i_39 = 0; i_39 < 4; ++i_39) {
      half_t S_shared_local_cast_2[2];
      uint1 __3;
      float2 v__2 = *(float2*)(acc_s + (i_39 * 2));
      ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
      *(uint1*)(S_shared_local_cast_2 + 0) = __3;
      *(uint1*)(((half_t*)S_shared) + ((((((i_39 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast_2 + 0);
    }
    #pragma unroll
    for (int i_40 = 0; i_40 < 4; ++i_40) {
      logsum[i_40] = ((logsum[i_40] * scores_scale_v0[i_40]) + scores_sum[i_40]);
    }
    #pragma unroll
    for (int i_41 = 0; i_41 < 64; ++i_41) {
      acc_o[i_41] = (acc_o[i_41] * scores_scale_v1[(((i_41 >> 5) * 2) + ((i_41 & 3) >> 1))]);
    }
    overlap_plan_mbar[3].wait((k & 1));
    {
      half_t A_local_7[16];
      half_t B_local_7[32];
      for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
        for (int i_42 = 0; i_42 < 2; ++i_42) {
          tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[((((i_42 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_7 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 2048)])), (&(A_local_7[(i_42 * 8)])));
        }
        for (int i_43 = 0; i_43 < 4; ++i_43) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) >> 5) * 4096) + (ki_7 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_43 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_43 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_7[(i_43 * 8)])));
        }
        for (int i_44 = 0; i_44 < 2; ++i_44) {
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_44 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_7 + (i_44 * 8)), reinterpret_cast<const unsigned*>(B_local_7 + (j_1 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_44 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_7 + (i_44 * 8)), reinterpret_cast<const unsigned*>(B_local_7 + ((j_1 * 8) + 4)));
          }
        }
      }
    }
    overlap_plan_mbar[6].arrive();
  }
  overlap_plan_mbar[6].wait(0);
  __syncthreads();
  if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[3].arrive_and_expect_tx(65536);
    tl::fence_proxy_async();
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[32768])), 0, 8128, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[36864])), 64, 8128, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[40960])), 128, 8128, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[45056])), 192, 8128, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[49152])), 256, 8128, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[53248])), 320, 8128, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[57344])), 384, 8128, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[61440])), 448, 8128, 0, 0);
  }
  overlap_plan_mbar[7].wait(0);
  if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[4].arrive_and_expect_tx(8192);
    tl::fence_proxy_async();
    tl::tma_load(K_pe_desc, overlap_plan_mbar[4], (&(((half_t*)K_pe_shared)[0])), 0, 8128, 0, 0);
  }
  overlap_plan_mbar[3].wait(1);
  {
    half_t A_local_8[16];
    half_t B_local_8[4];
    #pragma unroll
    for (int i_45 = 0; i_45 < 2; ++i_45) {
      float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_45 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
    }
    for (int ki_8 = 0; ki_8 < 32; ++ki_8) {
      for (int i_46 = 0; i_46 < 2; ++i_46) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki_8 >> 2) * 2048) + (i_46 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_8[(i_46 * 8)])));
      }
      tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 1) & 1) * 32768) + ((ki_8 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_8[0])));
      for (int i_47 = 0; i_47 < 2; ++i_47) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (i_47 * 4)), reinterpret_cast<const unsigned*>(A_local_8 + (i_47 * 8)), reinterpret_cast<const unsigned*>(B_local_8 + 0));
      }
    }
  }
  overlap_plan_mbar[4].wait(1);
  {
    half_t A_local_9[16];
    half_t B_local_9[4];
    for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
      for (int i_48 = 0; i_48 < 2; ++i_48) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[(((i_48 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_9[(i_48 * 8)])));
      }
      tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_9[0])));
      for (int i_49 = 0; i_49 < 2; ++i_49) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (i_49 * 4)), reinterpret_cast<const unsigned*>(A_local_9 + (i_49 * 8)), reinterpret_cast<const unsigned*>(B_local_9 + 0));
      }
    }
  }
  overlap_plan_mbar[7].arrive();
  *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
  float broadcast_var_10 = -CUDART_INF_F;
  *(float4*)(scores_max + 0) = make_float4(broadcast_var_10, broadcast_var_10, broadcast_var_10, broadcast_var_10);
  #pragma unroll
  for (int i_50 = 0; i_50 < 4; ++i_50) {
    scores_max_clear_3[i_50] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_6 = 0; rv_6 < 2; ++rv_6) {
      scores_max_clear_3[i_50] = max(scores_max_clear_3[i_50], acc_s[((i_50 * 2) + rv_6)]);
    }
    __syncthreads();
    scores_max_clear_3[i_50] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_50], (&(((float*)workspace_5)[0])));
    scores_max_clear_3[i_50] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_50]);
    scores_max[i_50] = max(scores_max[i_50], scores_max_clear_3[i_50]);
  }
  #pragma unroll
  for (int i_51 = 0; i_51 < 4; ++i_51) {
    scores_max[i_51] = max(scores_max[i_51], scores_max_prev[i_51]);
  }
  #pragma unroll
  for (int i_52 = 0; i_52 < 4; ++i_52) {
    scores_scale_v1[i_52] = exp2f(((scores_max_prev[i_52] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_52] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
  }
  #pragma unroll
  for (int i_53 = 0; i_53 < 8; ++i_53) {
    acc_s[i_53] = exp2f(((acc_s[i_53] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_53 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
  }
  #pragma unroll
  for (int i_54 = 0; i_54 < 4; ++i_54) {
    scores_sum[i_54] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_7 = 0; rv_7 < 2; ++rv_7) {
      scores_sum[i_54] = (scores_sum[i_54] + acc_s[((i_54 * 2) + rv_7)]);
    }
    __syncthreads();
    scores_sum[i_54] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_54], (&(((float*)workspace_2)[0])));
    scores_sum[i_54] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_54]);
  }
  #pragma unroll
  for (int i_55 = 0; i_55 < 4; ++i_55) {
    half_t S_shared_local_cast_3[2];
    uint1 __4;
    float2 v__3 = *(float2*)(acc_s + (i_55 * 2));
    ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
    *(uint1*)(S_shared_local_cast_3 + 0) = __4;
    *(uint1*)(((half_t*)S_shared) + (((((((i_55 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048)) = *(uint1*)(S_shared_local_cast_3 + 0);
  }
  #pragma unroll
  for (int i_56 = 0; i_56 < 4; ++i_56) {
    logsum[i_56] = ((logsum[i_56] * scores_scale_v1[i_56]) + scores_sum[i_56]);
  }
  #pragma unroll
  for (int i_57 = 0; i_57 < 64; ++i_57) {
    acc_o[i_57] = (acc_o[i_57] * scores_scale_v0[(((i_57 >> 5) * 2) + ((i_57 & 3) >> 1))]);
  }
  overlap_plan_mbar[2].wait(1);
  {
    half_t A_local_10[16];
    half_t B_local_10[32];
    __syncthreads();
    for (int ki_10 = 0; ki_10 < 4; ++ki_10) {
      for (int i_58 = 0; i_58 < 2; ++i_58) {
        tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((i_58 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_10 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_10[(i_58 * 8)])));
      }
      for (int i_59 = 0; i_59 < 4; ++i_59) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki_10 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_59 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_59 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_10[(i_59 * 8)])));
      }
      for (int i_60 = 0; i_60 < 2; ++i_60) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_60 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_10 + (i_60 * 8)), reinterpret_cast<const unsigned*>(B_local_10 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_60 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_10 + (i_60 * 8)), reinterpret_cast<const unsigned*>(B_local_10 + ((j_2 * 8) + 4)));
        }
      }
    }
  }
  overlap_plan_mbar[5].arrive();
  #pragma unroll
  for (int i_61 = 0; i_61 < 64; ++i_61) {
    acc_o[i_61] = (acc_o[i_61] * scores_scale_v1[(((i_61 >> 5) * 2) + ((i_61 & 3) >> 1))]);
  }
  overlap_plan_mbar[3].wait(1);
  {
    half_t A_local_11[16];
    half_t B_local_11[32];
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
  overlap_plan_mbar[6].arrive();
  #pragma unroll
  for (int i_65 = 0; i_65 < 64; ++i_65) {
    acc_o[i_65] = (acc_o[i_65] / logsum[(((i_65 >> 5) * 2) + ((i_65 & 3) >> 1))]);
  }
  #pragma unroll
  for (int i_66 = 0; i_66 < 8; ++i_66) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[((((((i_66 >> 2) * 8192) + ((((int)threadIdx.x) & 15) * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((i_66 & 3) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_66 * 8)]), ((half_t)acc_o[((i_66 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_66 * 8) + 2)]), ((half_t)acc_o[((i_66 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_66 * 8) + 4)]), ((half_t)acc_o[((i_66 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_66 * 8) + 6)]), ((half_t)acc_o[((i_66 * 8) + 7)])));
  }
  if (tl::tl_shuffle_elect<256>()) {
    tl::fence_proxy_async();
    __syncthreads();
    tl::tma_store((&(Output[(((int)blockIdx.x) * 16384)])), (&(((half_t*)O_shared)[0])), 32768);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

