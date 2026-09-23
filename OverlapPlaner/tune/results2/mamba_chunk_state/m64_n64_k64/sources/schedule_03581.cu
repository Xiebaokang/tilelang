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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap B_desc, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap X_desc);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap B_desc, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap X_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* da_last_wsp_handoff_0 = ((void*)((char*)buf_dyn_shmem + 0));
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 1024));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 25600));
  void* scale_wsp_handoff_4 = ((void*)((char*)buf_dyn_shmem + 33792));
  void* da_shared = ((void*)((char*)buf_dyn_shmem + 34816));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 35840));
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 36864));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[19];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float da_last[1];
  float acc[32];
  half_t x_local_v0[32];
  half_t x_local_v1[32];
  float scale_v0[16];
  half_t xt_local[32];
  float scale_v1[16];
  float da_local[16];
  float dt_local[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(X_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(Output_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(128);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(1);
    overlap_plan_mbar[4].init(1);
    overlap_plan_mbar[5].init(1);
    overlap_plan_mbar[6].init(128);
    overlap_plan_mbar[7].init(128);
    overlap_plan_mbar[8].init(1);
    overlap_plan_mbar[9].init(1);
    overlap_plan_mbar[10].init(1);
    overlap_plan_mbar[11].init(128);
    overlap_plan_mbar[12].init(128);
    overlap_plan_mbar[13].init(128);
    overlap_plan_mbar[14].init(128);
    overlap_plan_mbar[15].init(128);
    overlap_plan_mbar[16].init(128);
    overlap_plan_mbar[17].init(128);
    overlap_plan_mbar[18].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<240>();
    da_last[0] = ((float)DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 255)]);
    if (((int)threadIdx.x) == 0) {
      ((float*)da_last_wsp_handoff_0)[0] = da_last[0];
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[0].arrive();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(8192);
      tl::tma_load(X_desc, overlap_plan_mbar[1], (&(((half_t*)x_shared)[0])), 0, ((((int)blockIdx.z) >> 3) * 256), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      overlap_plan_mbar[2].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[2], 128);
      overlap_plan_mbar[4].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[4], 128);
    }
    overlap_plan_mbar[1].wait(0);
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; ++i_1) {
      x_local_v0[i_1] = ((half_t*)x_shared)[((((((((i_1 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_1 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_1 & 3) >> 1) + (i_1 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
    }
    overlap_plan_mbar[11].arrive();
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[8].arrive_and_expect_tx(8192);
      tl::tma_load(B_desc, overlap_plan_mbar[8], (&(((half_t*)b_shared)[0])), (((int)blockIdx.y) * 64), ((((int)blockIdx.z) >> 3) * 256), 0, (((int)blockIdx.z) & 7));
    }
    tl::__sync_thread_partial(3, 128);
    for (int ik = 0; ik < 1; ++ik) {
      overlap_plan_mbar[11].wait(0);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[1].arrive_and_expect_tx(8192);
        tl::tma_load(X_desc, overlap_plan_mbar[1], (&(((half_t*)x_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + 64), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
        overlap_plan_mbar[3].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)da_shared)[64])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 64)])), overlap_plan_mbar[3], 128);
        overlap_plan_mbar[5].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)dt_shared)[64])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 64)])), overlap_plan_mbar[5], 128);
      }
      overlap_plan_mbar[1].wait(1);
      #pragma unroll
      for (int i_2 = 0; i_2 < 32; ++i_2) {
        x_local_v1[i_2] = ((half_t*)x_shared)[((((((((i_2 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_2 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_2 & 3) >> 1) + (i_2 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      overlap_plan_mbar[11].arrive();
      overlap_plan_mbar[6].wait(0);
      #pragma unroll
      for (int i_3 = 0; i_3 < 8; ++i_3) {
        *(float2*)(scale_v0 + (i_3 * 2)) = *(float2*)(((float*)scale_wsp_handoff_4) + ((i_3 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      }
      #pragma unroll
      for (int i_4 = 0; i_4 < 32; ++i_4) {
        xt_local[i_4] = ((half_t)(((float)x_local_v0[((((i_4 >> 2) * 4) + ((i_4 & 1) * 2)) + ((i_4 & 3) >> 1))]) * scale_v0[(((i_4 >> 2) * 2) + (i_4 & 1))]));
      }
      tl::__sync_thread_partial(3, 128);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[9].arrive_and_expect_tx(8192);
        tl::tma_load(B_desc, overlap_plan_mbar[9], (&(((half_t*)b_shared)[4096])), (((int)blockIdx.y) * 64), (((((int)blockIdx.z) >> 3) * 256) + 64), 0, (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[8].wait(0);
      {
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b, (&(((half_t*)b_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local + (ki * 8)), uint64_t(desc_b + ((ki * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 16);
      }
      overlap_plan_mbar[16].arrive();
      overlap_plan_mbar[11].wait(1);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[1].arrive_and_expect_tx(8192);
        tl::tma_load(X_desc, overlap_plan_mbar[1], (&(((half_t*)x_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + 128), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[12].wait(0);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 128)])), overlap_plan_mbar[2], 128);
      }
      overlap_plan_mbar[14].wait(0);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[4].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 128)])), overlap_plan_mbar[4], 128);
      }
      overlap_plan_mbar[1].wait(0);
      #pragma unroll
      for (int i_5 = 0; i_5 < 32; ++i_5) {
        x_local_v0[i_5] = ((half_t*)x_shared)[((((((((i_5 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_5 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_5 & 3) >> 1) + (i_5 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      overlap_plan_mbar[11].arrive();
      overlap_plan_mbar[7].wait(0);
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_6 = 0; i_6 < 8; ++i_6) {
        *(float2*)(scale_v1 + (i_6 * 2)) = *(float2*)(((float*)scale_wsp_handoff_4) + (((i_6 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 32; ++i_7) {
        xt_local[i_7] = ((half_t)(((float)x_local_v1[((((i_7 >> 2) * 4) + ((i_7 & 1) * 2)) + ((i_7 & 3) >> 1))]) * scale_v1[(((i_7 >> 2) * 2) + (i_7 & 1))]));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[10].arrive_and_expect_tx(8192);
        tl::tma_load(B_desc, overlap_plan_mbar[10], (&(((half_t*)b_shared)[8192])), (((int)blockIdx.y) * 64), (((((int)blockIdx.z) >> 3) * 256) + 128), 0, (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[9].wait(0);
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_1, (&(((half_t*)b_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, 8192);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 16);
      }
      overlap_plan_mbar[17].arrive();
    }
    overlap_plan_mbar[11].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(8192);
      tl::tma_load(X_desc, overlap_plan_mbar[1], (&(((half_t*)x_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + 192), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[13].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)da_shared)[64])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 192)])), overlap_plan_mbar[3], 128);
    }
    overlap_plan_mbar[15].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[5].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)dt_shared)[64])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 192)])), overlap_plan_mbar[5], 128);
    }
    overlap_plan_mbar[1].wait(1);
    #pragma unroll
    for (int i_8 = 0; i_8 < 32; ++i_8) {
      x_local_v1[i_8] = ((half_t*)x_shared)[((((((((i_8 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_8 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_8 & 3) >> 1) + (i_8 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
    }
    overlap_plan_mbar[11].arrive();
    overlap_plan_mbar[6].wait(1);
    #pragma unroll
    for (int i_9 = 0; i_9 < 8; ++i_9) {
      *(float2*)(scale_v0 + (i_9 * 2)) = *(float2*)(((float*)scale_wsp_handoff_4) + ((i_9 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 32; ++i_10) {
      xt_local[i_10] = ((half_t)(((float)x_local_v0[((((i_10 >> 2) * 4) + ((i_10 & 1) * 2)) + ((i_10 & 3) >> 1))]) * scale_v0[(((i_10 >> 2) * 2) + (i_10 & 1))]));
    }
    overlap_plan_mbar[16].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[8].arrive_and_expect_tx(8192);
      tl::tma_load(B_desc, overlap_plan_mbar[8], (&(((half_t*)b_shared)[0])), (((int)blockIdx.y) * 64), (((((int)blockIdx.z) >> 3) * 256) + 192), 0, (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[10].wait(0);
    {
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_2, (&(((half_t*)b_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_2, 16384);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local + (ki_2 * 8)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 16);
    }
    overlap_plan_mbar[18].arrive();
    overlap_plan_mbar[7].wait(1);
    #pragma unroll
    for (int i_11 = 0; i_11 < 8; ++i_11) {
      *(float2*)(scale_v1 + (i_11 * 2)) = *(float2*)(((float*)scale_wsp_handoff_4) + (((i_11 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 32; ++i_12) {
      xt_local[i_12] = ((half_t)(((float)x_local_v1[((((i_12 >> 2) * 4) + ((i_12 & 1) * 2)) + ((i_12 & 3) >> 1))]) * scale_v1[(((i_12 >> 2) * 2) + (i_12 & 1))]));
    }
    overlap_plan_mbar[8].wait(1);
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_3, (&(((half_t*)b_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local + (ki_3 * 8)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 16);
    }
    overlap_plan_mbar[16].arrive();
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_13 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc[(i_13 * 8)]), ((half_t)acc[((i_13 * 8) + 1)])), __pack_half2(((half_t)acc[((i_13 * 8) + 2)]), ((half_t)acc[((i_13 * 8) + 3)])), __pack_half2(((half_t)acc[((i_13 * 8) + 4)]), ((half_t)acc[((i_13 * 8) + 5)])), __pack_half2(((half_t)acc[((i_13 * 8) + 6)]), ((half_t)acc[((i_13 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::fence_proxy_async();
      tl::tma_store(Output_desc, (&(((half_t*)acc_shared)[0])), (((int)blockIdx.y) * 64), 0, ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  } else {
    tl::warpgroup_reg_dealloc<24>();
    for (int ik_1 = 0; ik_1 < 2; ++ik_1) {
      overlap_plan_mbar[2].wait(ik_1);
      #pragma unroll
      for (int i_14 = 0; i_14 < 8; ++i_14) {
        half_t da_shared_local_cast[2];
        *(uint1*)(da_shared_local_cast + 0) = *(uint1*)(((half_t*)da_shared) + ((i_14 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __1;
        uint1 v_ = *(uint1*)(da_shared_local_cast + 0);
        ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
        *(float2*)(da_local + (i_14 * 2)) = __1;
      }
      overlap_plan_mbar[12].arrive();
      overlap_plan_mbar[4].wait(ik_1);
      #pragma unroll
      for (int i_15 = 0; i_15 < 8; ++i_15) {
        half_t dt_shared_local_cast_1[2];
        *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_15 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __2;
        uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
        ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
        *(float2*)(dt_local + (i_15 * 2)) = __2;
      }
      overlap_plan_mbar[14].arrive();
      if (ik_1 == 0) {
        overlap_plan_mbar[0].wait(0);
        da_last[0] = ((float*)da_last_wsp_handoff_0)[0];
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 16; ++i_16) {
        scale_v0[i_16] = (exp2f(((da_last[0] - da_local[i_16]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_16]);
      }
      if (((((((int)threadIdx.x) & 31) >> 2) * 4) + (((int)threadIdx.x) >> 5)) == 4) {
        #pragma unroll
        for (int i_17 = 0; i_17 < 8; ++i_17) {
          *(float2*)(((float*)scale_wsp_handoff_4) + ((i_17 * 8) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(scale_v0 + (i_17 * 2));
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[6].arrive();
      overlap_plan_mbar[3].wait(ik_1);
      #pragma unroll
      for (int i_18 = 0; i_18 < 8; ++i_18) {
        half_t da_shared_local_cast_2[2];
        *(uint1*)(da_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)da_shared) + (((i_18 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
        float2 __3;
        uint1 v__2 = *(uint1*)(da_shared_local_cast_2 + 0);
        ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__2))[0]);
        *(float2*)(da_local + (i_18 * 2)) = __3;
      }
      overlap_plan_mbar[13].arrive();
      overlap_plan_mbar[5].wait(ik_1);
      #pragma unroll
      for (int i_19 = 0; i_19 < 8; ++i_19) {
        half_t dt_shared_local_cast_3[2];
        *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + (((i_19 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
        float2 __4;
        uint1 v__3 = *(uint1*)(dt_shared_local_cast_3 + 0);
        ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__3))[0]);
        *(float2*)(dt_local + (i_19 * 2)) = __4;
      }
      overlap_plan_mbar[15].arrive();
      #pragma unroll
      for (int i_20 = 0; i_20 < 16; ++i_20) {
        scale_v1[i_20] = (exp2f(((da_last[0] - da_local[i_20]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_20]);
      }
      if (((((((int)threadIdx.x) & 31) >> 2) * 4) + (((int)threadIdx.x) >> 5)) == 4) {
        #pragma unroll
        for (int i_21 = 0; i_21 < 8; ++i_21) {
          *(float2*)(((float*)scale_wsp_handoff_4) + (((i_21 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64)) = *(float2*)(scale_v1 + (i_21 * 2));
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[7].arrive();
    }
  }
}

