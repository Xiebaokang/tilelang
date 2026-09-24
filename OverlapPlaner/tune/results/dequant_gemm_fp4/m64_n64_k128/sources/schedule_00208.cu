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
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap Ct_desc);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap Ct_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_dequantize_local_wsp_handoff_2 = ((void*)((char*)buf_dyn_shmem + 16384));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* Ct_shared = ((void*)((char*)buf_dyn_shmem + 36864));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[5];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  half_t B_dequantize_local[64];
  float Ct_local[32];
  uchar B_local[32];
  half_t B_dequantize_prev_local[64];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(Ct_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(128);
    overlap_plan_mbar[3].init(128);
    overlap_plan_mbar[4].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<40>();
    for (int k = 0; k < 32; ++k) {
      if (1 <= k) {
        overlap_plan_mbar[4].wait(((k + 1) & 1));
      }
      tl::__sync_thread_partial(3, 128);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[1].arrive_and_expect_tx(4096);
        tl::tma_load(B_desc, overlap_plan_mbar[1], (&(((uchar*)B_shared)[0])), (k * 64), (((int)blockIdx.x) * 64));
      }
      overlap_plan_mbar[1].wait((k & 1));
      #pragma unroll
      for (int i = 0; i < 32; ++i) {
        B_local[i] = ((uchar*)B_shared)[((((((((((int)threadIdx.x) >> 5) * 1024) + ((i >> 4) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i & 7) >> 2)) & 1) * 16)) + ((i & 3) * 4)) + (((int)threadIdx.x) & 3))];
      }
      overlap_plan_mbar[4].arrive();
      #pragma unroll
      for (int i_1 = 0; i_1 < 64; ++i_1) {
        ushort v_ = (((((((((ushort)B_local[((((i_1 & 3) >> 1) * 16) + (i_1 >> 2))]) >> (((ushort)(i_1 & 1)) * (ushort)4)) & (ushort)15) & (ushort)6) >> (ushort)1) + (ushort)14) | ((((((ushort)B_local[((((i_1 & 3) >> 1) * 16) + (i_1 >> 2))]) >> (((ushort)(i_1 & 1)) * (ushort)4)) & (ushort)15) >> (ushort)3) << (ushort)5)) << (ushort)10) | ((((((ushort)B_local[((((i_1 & 3) >> 1) * 16) + (i_1 >> 2))]) >> (((ushort)(i_1 & 1)) * (ushort)4)) & (ushort)15) & (ushort)1) << (ushort)9);
        B_dequantize_local[i_1] = (*(half_t *)(&(v_)));
      }
      #pragma unroll
      for (int i_2 = 0; i_2 < 8; ++i_2) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)B_dequantize_local_wsp_handoff_2)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_2 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(B_dequantize_local[(i_2 * 8)], B_dequantize_local[((i_2 * 8) + 1)]), __pack_half2(B_dequantize_local[((i_2 * 8) + 2)], B_dequantize_local[((i_2 * 8) + 3)]), __pack_half2(B_dequantize_local[((i_2 * 8) + 4)], B_dequantize_local[((i_2 * 8) + 5)]), __pack_half2(B_dequantize_local[((i_2 * 8) + 6)], B_dequantize_local[((i_2 * 8) + 7)]));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[2].arrive();
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(Ct_local + (i_3 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      if (1 <= k_1) {
        overlap_plan_mbar[3].wait(((k_1 + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
        overlap_plan_mbar[0].arrive_and_expect_tx(16384);
        tl::tma_load(A_desc, overlap_plan_mbar[0], (&(((half_t*)A_shared)[0])), (k_1 * 128), (((int)blockIdx.y) * 64));
        tl::tma_load(A_desc, overlap_plan_mbar[0], (&(((half_t*)A_shared)[4096])), ((k_1 * 128) + 64), (((int)blockIdx.y) * 64));
      }
      overlap_plan_mbar[2].wait((k_1 & 1));
      #pragma unroll
      for (int i_4 = 0; i_4 < 8; ++i_4) {
        tl::ptx_ldmatrix_x4((&(((half_t*)B_dequantize_local_wsp_handoff_2)[((((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_4 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 8192)])), (&(B_dequantize_local[(i_4 * 8)])));
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 8; ++i_5) {
        *(uint4*)(B_dequantize_prev_local + (i_5 * 8)) = *(uint4*)(B_dequantize_local + (i_5 * 8));
      }
      overlap_plan_mbar[0].wait((k_1 & 1));
      {
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)A_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(B_dequantize_prev_local + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(Ct_local + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(reinterpret_cast<const uint32_t*>(B_dequantize_prev_local + (ki * 8)), uint64_t(desc_b + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), reinterpret_cast<uint32_t*>(Ct_local + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(Ct_local + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(B_dequantize_prev_local + 0), 32);
      }
      overlap_plan_mbar[3].arrive();
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)Ct_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_6 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)Ct_local[(i_6 * 8)]), ((half_t)Ct_local[((i_6 * 8) + 1)])), __pack_half2(((half_t)Ct_local[((i_6 * 8) + 2)]), ((half_t)Ct_local[((i_6 * 8) + 3)])), __pack_half2(((half_t)Ct_local[((i_6 * 8) + 4)]), ((half_t)Ct_local[((i_6 * 8) + 5)])), __pack_half2(((half_t)Ct_local[((i_6 * 8) + 6)]), ((half_t)Ct_local[((i_6 * 8) + 7)])));
    }
    tl::__sync_thread_partial(4, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::fence_proxy_async();
      tl::tma_store(Ct_desc, (&(((half_t*)Ct_shared)[0])), (((int)blockIdx.y) * 64), (((int)blockIdx.x) * 64));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

