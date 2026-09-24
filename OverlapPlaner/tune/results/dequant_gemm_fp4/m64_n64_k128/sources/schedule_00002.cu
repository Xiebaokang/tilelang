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
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap Ct_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* Ct_shared = ((void*)((char*)buf_dyn_shmem + 36864));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[6];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float Ct_local[32];
  uchar B_local[32];
  half_t B_dequantize_local[64];
  half_t B_dequantize_prev_local_v0[64];
  half_t B_dequantize_prev_local_v1[64];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(Ct_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(128);
    overlap_plan_mbar[4].init(128);
    overlap_plan_mbar[5].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(Ct_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[0].arrive_and_expect_tx(16384);
    tl::tma_load(A_desc, overlap_plan_mbar[0], (&(((half_t*)A_shared)[0])), 0, (((int)blockIdx.y) * 64));
    tl::tma_load(A_desc, overlap_plan_mbar[0], (&(((half_t*)A_shared)[4096])), 64, (((int)blockIdx.y) * 64));
    overlap_plan_mbar[2].arrive_and_expect_tx(4096);
    tl::tma_load(B_desc, overlap_plan_mbar[2], (&(((uchar*)B_shared)[0])), 0, (((int)blockIdx.x) * 64));
  }
  overlap_plan_mbar[2].wait(0);
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    B_local[i_1] = ((uchar*)B_shared)[((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_1 >> 4) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_1 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_1 & 7) >> 2)) & 1) * 16)) + ((i_1 & 3) * 4)) + (((int)threadIdx.x) & 3))];
  }
  overlap_plan_mbar[5].arrive();
  #pragma unroll
  for (int i_2 = 0; i_2 < 64; ++i_2) {
    ushort v_ = (((((((((ushort)B_local[((((i_2 & 3) >> 1) * 16) + (i_2 >> 2))]) >> (((ushort)(i_2 & 1)) * (ushort)4)) & (ushort)15) & (ushort)6) >> (ushort)1) + (ushort)14) | ((((((ushort)B_local[((((i_2 & 3) >> 1) * 16) + (i_2 >> 2))]) >> (((ushort)(i_2 & 1)) * (ushort)4)) & (ushort)15) >> (ushort)3) << (ushort)5)) << (ushort)10) | ((((((ushort)B_local[((((i_2 & 3) >> 1) * 16) + (i_2 >> 2))]) >> (((ushort)(i_2 & 1)) * (ushort)4)) & (ushort)15) & (ushort)1) << (ushort)9);
    B_dequantize_local[i_2] = (*(half_t *)(&(v_)));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    *(uint4*)(B_dequantize_prev_local_v0 + (i_3 * 8)) = *(uint4*)(B_dequantize_local + (i_3 * 8));
  }
  for (int k = 0; k < 15; ++k) {
    if (1 <= k) {
      overlap_plan_mbar[4].wait(((k + 1) & 1));
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(A_desc, overlap_plan_mbar[1], (&(((half_t*)A_shared)[8192])), ((k * 256) + 128), (((int)blockIdx.y) * 64));
      tl::tma_load(A_desc, overlap_plan_mbar[1], (&(((half_t*)A_shared)[12288])), ((k * 256) + 192), (((int)blockIdx.y) * 64));
    }
    overlap_plan_mbar[5].wait(0);
    __syncthreads();
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(4096);
      tl::tma_load(B_desc, overlap_plan_mbar[2], (&(((uchar*)B_shared)[0])), ((k * 128) + 64), (((int)blockIdx.x) * 64));
    }
    overlap_plan_mbar[2].wait(1);
    #pragma unroll
    for (int i_4 = 0; i_4 < 32; ++i_4) {
      B_local[i_4] = ((uchar*)B_shared)[((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_4 >> 4) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_4 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_4 & 7) >> 2)) & 1) * 16)) + ((i_4 & 3) * 4)) + (((int)threadIdx.x) & 3))];
    }
    overlap_plan_mbar[5].arrive();
    #pragma unroll
    for (int i_5 = 0; i_5 < 64; ++i_5) {
      ushort v__1 = (((((((((ushort)B_local[((((i_5 & 3) >> 1) * 16) + (i_5 >> 2))]) >> (((ushort)(i_5 & 1)) * (ushort)4)) & (ushort)15) & (ushort)6) >> (ushort)1) + (ushort)14) | ((((((ushort)B_local[((((i_5 & 3) >> 1) * 16) + (i_5 >> 2))]) >> (((ushort)(i_5 & 1)) * (ushort)4)) & (ushort)15) >> (ushort)3) << (ushort)5)) << (ushort)10) | ((((((ushort)B_local[((((i_5 & 3) >> 1) * 16) + (i_5 >> 2))]) >> (((ushort)(i_5 & 1)) * (ushort)4)) & (ushort)15) & (ushort)1) << (ushort)9);
      B_dequantize_local[i_5] = (*(half_t *)(&(v__1)));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      *(uint4*)(B_dequantize_prev_local_v1 + (i_6 * 8)) = *(uint4*)(B_dequantize_local + (i_6 * 8));
    }
    overlap_plan_mbar[0].wait((k & 1));
    {
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)A_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(B_dequantize_prev_local_v0 + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(Ct_local + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(reinterpret_cast<const uint32_t*>(B_dequantize_prev_local_v0 + (ki * 8)), uint64_t(desc_b + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), reinterpret_cast<uint32_t*>(Ct_local + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(Ct_local + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(B_dequantize_prev_local_v0 + 0), 32);
    }
    overlap_plan_mbar[3].arrive();
    overlap_plan_mbar[3].wait((k & 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(16384);
      tl::tma_load(A_desc, overlap_plan_mbar[0], (&(((half_t*)A_shared)[0])), ((k * 256) + 256), (((int)blockIdx.y) * 64));
      tl::tma_load(A_desc, overlap_plan_mbar[0], (&(((half_t*)A_shared)[4096])), ((k * 256) + 320), (((int)blockIdx.y) * 64));
    }
    overlap_plan_mbar[5].wait(1);
    __syncthreads();
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(4096);
      tl::tma_load(B_desc, overlap_plan_mbar[2], (&(((uchar*)B_shared)[0])), ((k * 128) + 128), (((int)blockIdx.x) * 64));
    }
    overlap_plan_mbar[2].wait(0);
    #pragma unroll
    for (int i_7 = 0; i_7 < 32; ++i_7) {
      B_local[i_7] = ((uchar*)B_shared)[((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_7 >> 4) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_7 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_7 & 7) >> 2)) & 1) * 16)) + ((i_7 & 3) * 4)) + (((int)threadIdx.x) & 3))];
    }
    overlap_plan_mbar[5].arrive();
    #pragma unroll
    for (int i_8 = 0; i_8 < 64; ++i_8) {
      ushort v__2 = (((((((((ushort)B_local[((((i_8 & 3) >> 1) * 16) + (i_8 >> 2))]) >> (((ushort)(i_8 & 1)) * (ushort)4)) & (ushort)15) & (ushort)6) >> (ushort)1) + (ushort)14) | ((((((ushort)B_local[((((i_8 & 3) >> 1) * 16) + (i_8 >> 2))]) >> (((ushort)(i_8 & 1)) * (ushort)4)) & (ushort)15) >> (ushort)3) << (ushort)5)) << (ushort)10) | ((((((ushort)B_local[((((i_8 & 3) >> 1) * 16) + (i_8 >> 2))]) >> (((ushort)(i_8 & 1)) * (ushort)4)) & (ushort)15) & (ushort)1) << (ushort)9);
      B_dequantize_local[i_8] = (*(half_t *)(&(v__2)));
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 8; ++i_9) {
      *(uint4*)(B_dequantize_prev_local_v0 + (i_9 * 8)) = *(uint4*)(B_dequantize_local + (i_9 * 8));
    }
    overlap_plan_mbar[1].wait((k & 1));
    {
      tl::GmmaDescriptor desc_b_1;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((half_t*)A_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_1, 16384);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(B_dequantize_prev_local_v1 + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(Ct_local + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(reinterpret_cast<const uint32_t*>(B_dequantize_prev_local_v1 + (ki_1 * 8)), uint64_t(desc_b_1 + ((((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32)) >> 4)), reinterpret_cast<uint32_t*>(Ct_local + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(Ct_local + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(B_dequantize_prev_local_v1 + 0), 32);
    }
    overlap_plan_mbar[4].arrive();
  }
  overlap_plan_mbar[4].wait(0);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[1].arrive_and_expect_tx(16384);
    tl::tma_load(A_desc, overlap_plan_mbar[1], (&(((half_t*)A_shared)[8192])), 3968, (((int)blockIdx.y) * 64));
    tl::tma_load(A_desc, overlap_plan_mbar[1], (&(((half_t*)A_shared)[12288])), 4032, (((int)blockIdx.y) * 64));
  }
  overlap_plan_mbar[5].wait(0);
  __syncthreads();
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[2].arrive_and_expect_tx(4096);
    tl::tma_load(B_desc, overlap_plan_mbar[2], (&(((uchar*)B_shared)[0])), 1984, (((int)blockIdx.x) * 64));
  }
  overlap_plan_mbar[2].wait(1);
  #pragma unroll
  for (int i_10 = 0; i_10 < 32; ++i_10) {
    B_local[i_10] = ((uchar*)B_shared)[((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_10 >> 4) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_10 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_10 & 7) >> 2)) & 1) * 16)) + ((i_10 & 3) * 4)) + (((int)threadIdx.x) & 3))];
  }
  overlap_plan_mbar[5].arrive();
  #pragma unroll
  for (int i_11 = 0; i_11 < 64; ++i_11) {
    ushort v__3 = (((((((((ushort)B_local[((((i_11 & 3) >> 1) * 16) + (i_11 >> 2))]) >> (((ushort)(i_11 & 1)) * (ushort)4)) & (ushort)15) & (ushort)6) >> (ushort)1) + (ushort)14) | ((((((ushort)B_local[((((i_11 & 3) >> 1) * 16) + (i_11 >> 2))]) >> (((ushort)(i_11 & 1)) * (ushort)4)) & (ushort)15) >> (ushort)3) << (ushort)5)) << (ushort)10) | ((((((ushort)B_local[((((i_11 & 3) >> 1) * 16) + (i_11 >> 2))]) >> (((ushort)(i_11 & 1)) * (ushort)4)) & (ushort)15) & (ushort)1) << (ushort)9);
    B_dequantize_local[i_11] = (*(half_t *)(&(v__3)));
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 8; ++i_12) {
    *(uint4*)(B_dequantize_prev_local_v1 + (i_12 * 8)) = *(uint4*)(B_dequantize_local + (i_12 * 8));
  }
  overlap_plan_mbar[0].wait(1);
  {
    tl::GmmaDescriptor desc_b_2;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)A_shared)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(B_dequantize_prev_local_v0 + 0), 32);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(Ct_local + 0), 32);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(reinterpret_cast<const uint32_t*>(B_dequantize_prev_local_v0 + (ki_2 * 8)), uint64_t(desc_b_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), reinterpret_cast<uint32_t*>(Ct_local + 0), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(Ct_local + 0), 32);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(B_dequantize_prev_local_v0 + 0), 32);
  }
  overlap_plan_mbar[3].arrive();
  overlap_plan_mbar[1].wait(1);
  {
    tl::GmmaDescriptor desc_b_3;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_3, (&(((half_t*)A_shared)[0])));
    tl::increase_descriptor_offset<int>(desc_b_3, 16384);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(B_dequantize_prev_local_v1 + 0), 32);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(Ct_local + 0), 32);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
      tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(reinterpret_cast<const uint32_t*>(B_dequantize_prev_local_v1 + (ki_3 * 8)), uint64_t(desc_b_3 + ((((ki_3 >> 2) * 8192) + ((ki_3 & 3) * 32)) >> 4)), reinterpret_cast<uint32_t*>(Ct_local + 0), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(Ct_local + 0), 32);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(B_dequantize_prev_local_v1 + 0), 32);
  }
  overlap_plan_mbar[4].arrive();
  #pragma unroll
  for (int i_13 = 0; i_13 < 4; ++i_13) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)Ct_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_13 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)Ct_local[(i_13 * 8)]), ((half_t)Ct_local[((i_13 * 8) + 1)])), __pack_half2(((half_t)Ct_local[((i_13 * 8) + 2)]), ((half_t)Ct_local[((i_13 * 8) + 3)])), __pack_half2(((half_t)Ct_local[((i_13 * 8) + 4)]), ((half_t)Ct_local[((i_13 * 8) + 5)])), __pack_half2(((half_t)Ct_local[((i_13 * 8) + 6)]), ((half_t)Ct_local[((i_13 * 8) + 7)])));
  }
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    __syncthreads();
    tl::tma_store(Ct_desc, (&(((half_t*)Ct_shared)[0])), (((int)blockIdx.y) * 64), (((int)blockIdx.x) * 64));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

