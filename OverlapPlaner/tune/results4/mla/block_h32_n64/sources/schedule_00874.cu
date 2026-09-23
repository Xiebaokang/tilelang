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
  void* acc_o_wsp_handoff_8 = ((void*)((char*)buf_dyn_shmem + 102400));
  void* K_pe_shared = ((void*)((char*)buf_dyn_shmem + 167936));
  void* acc_s_wsp_handoff_6 = ((void*)((char*)buf_dyn_shmem + 184320));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 200704));
  void* acc_s_wsp_handoff_4 = ((void*)((char*)buf_dyn_shmem + 208896));
  void* scores_scale_wsp_handoff_5 = ((void*)((char*)buf_dyn_shmem + 217088));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 217216));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 217216));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 217216));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 217216));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[17];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[64];
  float acc_s_v0[8];
  float scores_scale[4];
  float acc_s_v1[8];
  float logsum[4];
  float scores_max[4];
  float scores_max_prev[4];
  float scores_max_clear[4];
  float scores_sum[4];
  float scores_max_clear_1[4];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(KV_desc);
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(Q_pe_desc);
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
    tl::warpgroup_reg_alloc<176>();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int k = 0; k < 64; ++k) {
      if (1 <= k) {
        overlap_plan_mbar[12].wait(1);
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(65536);
        tl::fence_proxy_async();
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[0])), 0, (k * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[4096])), 64, (k * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[8192])), 128, (k * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[12288])), 192, (k * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[16384])), 256, (k * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[20480])), 320, (k * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[24576])), 384, (k * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[28672])), 448, (k * 128), 0, 0);
      }
      if (k == 0) {
        overlap_plan_mbar[0].wait(0);
      }
      overlap_plan_mbar[2].wait(0);
      if (1 <= k) {
        overlap_plan_mbar[8].wait(((k + 1) & 1));
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 4; ++i_1) {
        *(float2*)(acc_s_v0 + (i_1 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_6) + (((((i_1 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048));
      }
      {
        half_t A_local[16];
        half_t B_local[4];
        #pragma unroll
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(acc_s_v0 + (i_2 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
        }
        for (int ki = 0; ki < 32; ++ki) {
          for (int i_3 = 0; i_3 < 2; ++i_3) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki >> 2) * 2048) + (i_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[(i_3 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[(((((((ki >> 2) * 4096) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
          for (int i_4 = 0; i_4 < 2; ++i_4) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + (i_4 * 4)), reinterpret_cast<const unsigned*>(A_local + (i_4 * 8)), reinterpret_cast<const unsigned*>(B_local + 0));
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
          for (int i_5 = 0; i_5 < 2; ++i_5) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[(((i_5 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_1[(i_5 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[(((((((((int)threadIdx.x) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[0])));
          for (int i_6 = 0; i_6 < 2; ++i_6) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + (i_6 * 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_6 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + 0));
          }
        }
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        *(float2*)(((float*)acc_s_wsp_handoff_4) + ((((i_7 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s_v0 + (i_7 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[5].arrive();
      overlap_plan_mbar[13].arrive();
      overlap_plan_mbar[6].wait(0);
      #pragma unroll
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        scores_scale[i_8] = ((float*)scores_scale_wsp_handoff_5)[((i_8 * 8) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 64; ++i_9) {
        acc_o[i_9] = (acc_o[i_9] * scores_scale[(((i_9 >> 5) * 2) + ((i_9 & 3) >> 1))]);
      }
      overlap_plan_mbar[9].wait((k & 1));
      {
        half_t A_local_2[16];
        half_t B_local_2[32];
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          for (int i_10 = 0; i_10 < 2; ++i_10) {
            tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((i_10 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[(i_10 * 8)])));
          }
          for (int i_11 = 0; i_11 < 4; ++i_11) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_2[(i_11 * 8)])));
          }
          for (int i_12 = 0; i_12 < 2; ++i_12) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_12 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_12 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j * 8) + 4)));
            }
          }
        }
      }
      overlap_plan_mbar[12].arrive();
      overlap_plan_mbar[15].arrive();
      overlap_plan_mbar[12].wait(0);
      tl::__sync_thread_partial(3, 256);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(65536);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[0])), 0, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[4096])), 64, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[8192])), 128, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[12288])), 192, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[16384])), 256, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[20480])), 320, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[24576])), 384, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[28672])), 448, ((k * 128) + 64), 0, 0);
      }
      overlap_plan_mbar[2].wait(1);
      overlap_plan_mbar[7].wait((k & 1));
      #pragma unroll
      for (int i_13 = 0; i_13 < 4; ++i_13) {
        *(float2*)(acc_s_v1 + (i_13 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_6) + ((((i_13 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      {
        half_t A_local_3[16];
        half_t B_local_3[4];
        #pragma unroll
        for (int i_14 = 0; i_14 < 2; ++i_14) {
          float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(acc_s_v1 + (i_14 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
        }
        for (int ki_3 = 0; ki_3 < 32; ++ki_3) {
          for (int i_15 = 0; i_15 < 2; ++i_15) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki_3 >> 2) * 2048) + (i_15 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_3[(i_15 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[(((((((ki_3 >> 2) * 4096) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[0])));
          for (int i_16 = 0; i_16 < 2; ++i_16) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + (i_16 * 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_16 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + 0));
          }
        }
      }
      overlap_plan_mbar[4].wait((k & 1));
      {
        half_t A_local_4[16];
        half_t B_local_4[4];
        for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
          for (int i_17 = 0; i_17 < 2; ++i_17) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[(((i_17 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_4[(i_17 * 8)])));
          }
          tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[(((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 1) & 1) * 4096) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_4[0])));
          for (int i_18 = 0; i_18 < 2; ++i_18) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + (i_18 * 4)), reinterpret_cast<const unsigned*>(A_local_4 + (i_18 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + 0));
          }
        }
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 4; ++i_19) {
        *(float2*)(((float*)acc_s_wsp_handoff_4) + ((((i_19 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s_v1 + (i_19 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[5].arrive();
      overlap_plan_mbar[14].arrive();
      overlap_plan_mbar[6].wait(1);
      #pragma unroll
      for (int i_20 = 0; i_20 < 4; ++i_20) {
        scores_scale[i_20] = ((float*)scores_scale_wsp_handoff_5)[((i_20 * 8) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 64; ++i_21) {
        acc_o[i_21] = (acc_o[i_21] * scores_scale[(((i_21 >> 5) * 2) + ((i_21 & 3) >> 1))]);
      }
      overlap_plan_mbar[10].wait((k & 1));
      {
        half_t A_local_5[16];
        half_t B_local_5[32];
        for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
          for (int i_22 = 0; i_22 < 2; ++i_22) {
            tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[((((i_22 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 2048)])), (&(A_local_5[(i_22 * 8)])));
          }
          for (int i_23 = 0; i_23 < 4; ++i_23) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki_5 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_23 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_23 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_5[(i_23 * 8)])));
          }
          for (int i_24 = 0; i_24 < 2; ++i_24) {
            for (int j_1 = 0; j_1 < 4; ++j_1) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_24 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_5 + (i_24 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + (j_1 * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_24 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + (i_24 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + ((j_1 * 8) + 4)));
            }
          }
        }
      }
      tl::__sync_thread_partial(3, 256);
      if ((k * 2) == 126) {
        #pragma unroll
        for (int i_25 = 0; i_25 < 32; ++i_25) {
          *(float2*)(((float*)acc_o_wsp_handoff_8) + (((((((i_25 >> 4) * 8192) + ((i_25 & 1) * 4096)) + (((((int)threadIdx.x) & 31) >> 2) * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + (((i_25 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_o + (i_25 * 2));
        }
      }
      overlap_plan_mbar[12].arrive();
      overlap_plan_mbar[16].arrive();
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[11].arrive();
  } else {
    tl::warpgroup_reg_dealloc<72>();
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
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(logsum + 0) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    float broadcast_var_4 = -CUDART_INF_F;
    *(float4*)(scores_max + 0) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    for (int k_1 = 0; k_1 < 64; ++k_1) {
      if (1 <= k_1) {
        overlap_plan_mbar[13].wait(((k_1 + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[3].arrive_and_expect_tx(8192);
        tl::tma_load(K_pe_desc, overlap_plan_mbar[3], (&(((half_t*)K_pe_shared)[0])), 0, (k_1 * 128), 0, 0);
      }
      *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
      float broadcast_var_5 = -CUDART_INF_F;
      *(float4*)(scores_max + 0) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
      overlap_plan_mbar[5].wait(0);
      #pragma unroll
      for (int i_26 = 0; i_26 < 4; ++i_26) {
        *(float2*)(acc_s_v0 + (i_26 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_4) + (((((i_26 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
      }
      #pragma unroll
      for (int i_27 = 0; i_27 < 4; ++i_27) {
        scores_max_clear[i_27] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 2; ++rv) {
          scores_max_clear[i_27] = max(scores_max_clear[i_27], acc_s_v0[((i_27 * 2) + rv)]);
        }
        tl::__sync_thread_partial(4, 256);
        scores_max_clear[i_27] = tl::AllReduce<tl::MaxOp, 256, 32, 256, tl::NamedBarrier<256>>::run(scores_max_clear[i_27], (&(((float*)workspace_3)[0])));
        scores_max_clear[i_27] = tl::AllReduce<tl::MaxOp, 4, 1, 256, tl::NamedBarrier<256>>::run(scores_max_clear[i_27]);
        scores_max[i_27] = max(scores_max[i_27], scores_max_clear[i_27]);
      }
      #pragma unroll
      for (int i_28 = 0; i_28 < 4; ++i_28) {
        scores_max[i_28] = max(scores_max[i_28], scores_max_prev[i_28]);
      }
      #pragma unroll
      for (int i_29 = 0; i_29 < 4; ++i_29) {
        scores_scale[i_29] = exp2f(((scores_max_prev[i_29] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_29] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      if ((((((int)threadIdx.x) & 3) * 8) + (((int)threadIdx.x) >> 5)) == 8) {
        #pragma unroll
        for (int i_30 = 0; i_30 < 4; ++i_30) {
          ((float*)scores_scale_wsp_handoff_5)[((i_30 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = scores_scale[i_30];
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[6].arrive();
      #pragma unroll
      for (int i_31 = 0; i_31 < 8; ++i_31) {
        acc_s_v0[i_31] = exp2f(((acc_s_v0[i_31] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_31 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_32 = 0; i_32 < 4; ++i_32) {
        *(float2*)(((float*)acc_s_wsp_handoff_6) + (((((i_32 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64)) = *(float2*)(acc_s_v0 + (i_32 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[7].arrive();
      if (1 <= k_1) {
        overlap_plan_mbar[15].wait(((k_1 + 1) & 1));
      }
      #pragma unroll
      for (int i_33 = 0; i_33 < 4; ++i_33) {
        half_t S_shared_local_cast[2];
        uint1 __1;
        float2 v_ = *(float2*)(acc_s_v0 + (i_33 * 2));
        ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
        *(uint1*)(S_shared_local_cast + 0) = __1;
        *(uint1*)(((half_t*)S_shared) + ((((((i_33 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast + 0);
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[9].arrive();
      #pragma unroll
      for (int i_34 = 0; i_34 < 4; ++i_34) {
        scores_sum[i_34] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
          scores_sum[i_34] = (scores_sum[i_34] + acc_s_v0[((i_34 * 2) + rv_1)]);
        }
        tl::__sync_thread_partial(4, 256);
        scores_sum[i_34] = tl::AllReduce<tl::SumOp, 256, 32, 256, tl::NamedBarrier<256>>::run(scores_sum[i_34], (&(((float*)workspace_1)[0])));
        scores_sum[i_34] = tl::AllReduce<tl::SumOp, 4, 1, 256, tl::NamedBarrier<256>>::run(scores_sum[i_34]);
      }
      #pragma unroll
      for (int i_35 = 0; i_35 < 4; ++i_35) {
        logsum[i_35] = ((logsum[i_35] * scores_scale[i_35]) + scores_sum[i_35]);
      }
      if (1 <= k_1) {
        overlap_plan_mbar[14].wait(((k_1 + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[4].arrive_and_expect_tx(8192);
        tl::tma_load(K_pe_desc, overlap_plan_mbar[4], (&(((half_t*)K_pe_shared)[4096])), 0, ((k_1 * 128) + 64), 0, 0);
      }
      *(float4*)(scores_max_prev + 0) = *(float4*)(scores_max + 0);
      float broadcast_var_6 = -CUDART_INF_F;
      *(float4*)(scores_max + 0) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
      overlap_plan_mbar[5].wait(1);
      #pragma unroll
      for (int i_36 = 0; i_36 < 4; ++i_36) {
        *(float2*)(acc_s_v1 + (i_36 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_4) + (((((i_36 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
      }
      #pragma unroll
      for (int i_37 = 0; i_37 < 4; ++i_37) {
        scores_max_clear_1[i_37] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 2; ++rv_2) {
          scores_max_clear_1[i_37] = max(scores_max_clear_1[i_37], acc_s_v1[((i_37 * 2) + rv_2)]);
        }
        tl::__sync_thread_partial(4, 256);
        scores_max_clear_1[i_37] = tl::AllReduce<tl::MaxOp, 256, 32, 256, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_37], (&(((float*)workspace)[0])));
        scores_max_clear_1[i_37] = tl::AllReduce<tl::MaxOp, 4, 1, 256, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_37]);
        scores_max[i_37] = max(scores_max[i_37], scores_max_clear_1[i_37]);
      }
      #pragma unroll
      for (int i_38 = 0; i_38 < 4; ++i_38) {
        scores_max[i_38] = max(scores_max[i_38], scores_max_prev[i_38]);
      }
      #pragma unroll
      for (int i_39 = 0; i_39 < 4; ++i_39) {
        scores_scale[i_39] = exp2f(((scores_max_prev[i_39] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_39] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      if ((((((int)threadIdx.x) & 3) * 8) + (((int)threadIdx.x) >> 5)) == 8) {
        #pragma unroll
        for (int i_40 = 0; i_40 < 4; ++i_40) {
          ((float*)scores_scale_wsp_handoff_5)[((i_40 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = scores_scale[i_40];
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[6].arrive();
      #pragma unroll
      for (int i_41 = 0; i_41 < 8; ++i_41) {
        acc_s_v1[i_41] = exp2f(((acc_s_v1[i_41] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_41 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_42 = 0; i_42 < 4; ++i_42) {
        *(float2*)(((float*)acc_s_wsp_handoff_6) + (((((i_42 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1984)) = *(float2*)(acc_s_v1 + (i_42 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[8].arrive();
      if (1 <= k_1) {
        overlap_plan_mbar[16].wait(((k_1 + 1) & 1));
      }
      #pragma unroll
      for (int i_43 = 0; i_43 < 4; ++i_43) {
        half_t S_shared_local_cast_1[2];
        uint1 __2;
        float2 v__1 = *(float2*)(acc_s_v1 + (i_43 * 2));
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
        *(uint1*)(S_shared_local_cast_1 + 0) = __2;
        *(uint1*)(((half_t*)S_shared) + (((((((i_43 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048)) = *(uint1*)(S_shared_local_cast_1 + 0);
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[10].arrive();
      #pragma unroll
      for (int i_44 = 0; i_44 < 4; ++i_44) {
        scores_sum[i_44] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 2; ++rv_3) {
          scores_sum[i_44] = (scores_sum[i_44] + acc_s_v1[((i_44 * 2) + rv_3)]);
        }
        tl::__sync_thread_partial(4, 256);
        scores_sum[i_44] = tl::AllReduce<tl::SumOp, 256, 32, 256, tl::NamedBarrier<256>>::run(scores_sum[i_44], (&(((float*)workspace_2)[0])));
        scores_sum[i_44] = tl::AllReduce<tl::SumOp, 4, 1, 256, tl::NamedBarrier<256>>::run(scores_sum[i_44]);
      }
      #pragma unroll
      for (int i_45 = 0; i_45 < 4; ++i_45) {
        logsum[i_45] = ((logsum[i_45] * scores_scale[i_45]) + scores_sum[i_45]);
      }
    }
    overlap_plan_mbar[11].wait(0);
    #pragma unroll
    for (int i_46 = 0; i_46 < 32; ++i_46) {
      *(float2*)(acc_o + (i_46 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_8) + ((((((((i_46 >> 4) * 8192) + ((i_46 & 1) * 4096)) + (((((int)threadIdx.x) & 31) >> 2) * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + (((i_46 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 512));
    }
    #pragma unroll
    for (int i_47 = 0; i_47 < 64; ++i_47) {
      acc_o[i_47] = (acc_o[i_47] / logsum[(((i_47 >> 5) * 2) + ((i_47 & 3) >> 1))]);
    }
    #pragma unroll
    for (int i_48 = 0; i_48 < 8; ++i_48) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((i_48 >> 2) * 8192) + ((((int)threadIdx.x) & 15) * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((i_48 & 3) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 512)])), __pack_half2(((half_t)acc_o[(i_48 * 8)]), ((half_t)acc_o[((i_48 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_48 * 8) + 2)]), ((half_t)acc_o[((i_48 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_48 * 8) + 4)]), ((half_t)acc_o[((i_48 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_48 * 8) + 6)]), ((half_t)acc_o[((i_48 * 8) + 7)])));
    }
    tl::__sync_thread_partial(4, 256);
    if (tl::tl_shuffle_elect<256>()) {
      tl::fence_proxy_async();
      tl::tma_store((&(Output[(((int)blockIdx.x) * 16384)])), (&(((half_t*)O_shared)[0])), 32768);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

