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
  void* K_pe_shared = ((void*)((char*)buf_dyn_shmem + 102400));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 110592));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 114688));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 114688));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[6];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[64];
  float logsum[4];
  float scores_max[4];
  float acc_s[8];
  float scores_max_prev[4];
  float scores_max_clear[4];
  float scores_scale[4];
  float scores_sum[4];
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
    overlap_plan_mbar[4].init(256);
    overlap_plan_mbar[5].init(256);
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
  for (int k = 0; k < 128; ++k) {
    if (1 <= k) {
      overlap_plan_mbar[4].wait(((k + 1) & 1));
    }
    __syncthreads();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(65536);
      tl::fence_proxy_async();
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[0])), 0, (k * 64), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[4096])), 64, (k * 64), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[8192])), 128, (k * 64), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[12288])), 192, (k * 64), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[16384])), 256, (k * 64), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[20480])), 320, (k * 64), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[24576])), 384, (k * 64), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[28672])), 448, (k * 64), 0, 0);
    }
    if (1 <= k) {
      overlap_plan_mbar[5].wait(((k + 1) & 1));
    }
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(8192);
      tl::fence_proxy_async();
      tl::tma_load(K_pe_desc, overlap_plan_mbar[3], (&(((half_t*)K_pe_shared)[0])), 0, (k * 64), 0, 0);
    }
    if (k == 0) {
      overlap_plan_mbar[0].wait(0);
    }
    overlap_plan_mbar[2].wait((k & 1));
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
        tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[(((((((ki >> 2) * 4096) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (i_3 * 4)), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + 0));
        }
      }
    }
    if (k == 0) {
      overlap_plan_mbar[1].wait(0);
    }
    overlap_plan_mbar[3].wait((k & 1));
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
    overlap_plan_mbar[5].arrive();
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
      scores_max_clear[i_6] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_6], (&(((float*)workspace_1)[0])));
      scores_max_clear[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_6]);
      scores_max[i_6] = max(scores_max[i_6], scores_max_clear[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      scores_max[i_7] = max(scores_max[i_7], scores_max_prev[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      scores_scale[i_8] = exp2f(((scores_max_prev[i_8] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_8] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 8; ++i_9) {
      acc_s[i_9] = exp2f(((acc_s[i_9] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_9 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)S_shared)[((((((int)threadIdx.x) & 31) >> 3) * 512) + ((((((((int)threadIdx.x) & 31) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_s[0]), ((half_t)acc_s[1])), __pack_half2(((half_t)acc_s[2]), ((half_t)acc_s[3])), __pack_half2(((half_t)acc_s[4]), ((half_t)acc_s[5])), __pack_half2(((half_t)acc_s[6]), ((half_t)acc_s[7])));
    #pragma unroll
    for (int i_10 = 0; i_10 < 64; ++i_10) {
      acc_o[i_10] = (acc_o[i_10] * scores_scale[(((i_10 >> 5) * 2) + ((i_10 & 3) >> 1))]);
    }
    {
      half_t A_local_2[16];
      half_t B_local_2[32];
      __syncthreads();
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        for (int i_11 = 0; i_11 < 2; ++i_11) {
          tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((i_11 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[(i_11 * 8)])));
        }
        for (int i_12 = 0; i_12 < 4; ++i_12) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_12 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_12 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_2[(i_12 * 8)])));
        }
        for (int i_13 = 0; i_13 < 2; ++i_13) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_13 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_13 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_13 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_13 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j * 8) + 4)));
          }
        }
      }
    }
    overlap_plan_mbar[4].arrive();
    #pragma unroll
    for (int i_14 = 0; i_14 < 4; ++i_14) {
      scores_sum[i_14] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
        scores_sum[i_14] = (scores_sum[i_14] + acc_s[((i_14 * 2) + rv_1)]);
      }
      __syncthreads();
      scores_sum[i_14] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_14], (&(((float*)workspace)[0])));
      scores_sum[i_14] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      logsum[i_15] = ((logsum[i_15] * scores_scale[i_15]) + scores_sum[i_15]);
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 64; ++i_16) {
    acc_o[i_16] = (acc_o[i_16] / logsum[(((i_16 >> 5) * 2) + ((i_16 & 3) >> 1))]);
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 8; ++i_17) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[((((((i_17 >> 2) * 8192) + ((((int)threadIdx.x) & 15) * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((i_17 & 3) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_17 * 8)]), ((half_t)acc_o[((i_17 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_17 * 8) + 2)]), ((half_t)acc_o[((i_17 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_17 * 8) + 4)]), ((half_t)acc_o[((i_17 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_17 * 8) + 6)]), ((half_t)acc_o[((i_17 * 8) + 7)])));
  }
  if (tl::tl_shuffle_elect<256>()) {
    tl::fence_proxy_async();
    __syncthreads();
    tl::tma_store((&(Output[(((int)blockIdx.x) * 16384)])), (&(((half_t*)O_shared)[0])), 32768);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

