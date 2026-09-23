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
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(__grid_constant__ const CUtensorMap B_desc, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap X_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 8192));
  void* da_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 17408));
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 18432));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[10];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float da_last[1];
  float acc[16];
  float da_local[16];
  float dt_local[16];
  float scale[16];
  half_t x_local[32];
  half_t xt_local[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(X_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(Output_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(1);
    overlap_plan_mbar[4].init(1);
    overlap_plan_mbar[5].init(128);
    overlap_plan_mbar[6].init(128);
    overlap_plan_mbar[7].init(128);
    overlap_plan_mbar[8].init(128);
    overlap_plan_mbar[9].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  da_last[0] = ((float)DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 255)]);
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[0].arrive_and_expect_tx(8192);
    tl::tma_load(X_desc, overlap_plan_mbar[0], (&(((half_t*)x_shared)[0])), 0, ((((int)blockIdx.z) >> 3) * 256), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    overlap_plan_mbar[1].arrive_and_expect_tx(128);
    tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[1], 128);
    overlap_plan_mbar[2].arrive_and_expect_tx(128);
    tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[2], 128);
    overlap_plan_mbar[3].arrive_and_expect_tx(4096);
    tl::tma_load(B_desc, overlap_plan_mbar[3], (&(((half_t*)b_shared)[0])), (((int)blockIdx.y) * 32), ((((int)blockIdx.z) >> 3) * 256), 0, (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[1].wait(0);
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    half_t da_shared_local_cast[2];
    *(uint1*)(da_shared_local_cast + 0) = *(uint1*)(((half_t*)da_shared) + ((i_1 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __1;
    uint1 v_ = *(uint1*)(da_shared_local_cast + 0);
    ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
    *(float2*)(da_local + (i_1 * 2)) = __1;
  }
  overlap_plan_mbar[6].arrive();
  overlap_plan_mbar[2].wait(0);
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    half_t dt_shared_local_cast_1[2];
    *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_2 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __2;
    uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
    ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
    *(float2*)(dt_local + (i_2 * 2)) = __2;
  }
  overlap_plan_mbar[7].arrive();
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    scale[i_3] = (exp2f(((da_last[0] - da_local[i_3]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_3]);
  }
  overlap_plan_mbar[0].wait(0);
  #pragma unroll
  for (int i_4 = 0; i_4 < 32; ++i_4) {
    x_local[i_4] = ((half_t*)x_shared)[((((((((i_4 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_4 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_4 & 3) >> 1) + (i_4 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
  }
  overlap_plan_mbar[5].arrive();
  #pragma unroll
  for (int i_5 = 0; i_5 < 32; ++i_5) {
    xt_local[i_5] = ((half_t)(((float)x_local[((((i_5 >> 2) * 4) + ((i_5 & 1) * 2)) + ((i_5 & 3) >> 1))]) * scale[(((i_5 >> 2) * 2) + (i_5 & 1))]));
  }
  for (int ik = 0; ik < 3; ++ik) {
    overlap_plan_mbar[5].wait((ik & 1));
    __syncthreads();
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(8192);
      tl::tma_load(X_desc, overlap_plan_mbar[0], (&(((half_t*)x_shared)[0])), 0, ((((((int)blockIdx.z) >> 3) * 256) + (ik * 64)) + 64), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[6].wait((ik & 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 64)])), overlap_plan_mbar[1], 128);
    }
    overlap_plan_mbar[7].wait((ik & 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 64)])), overlap_plan_mbar[2], 128);
    }
    if (1 <= ik) {
      overlap_plan_mbar[(ik + 7)].wait(0);
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[(((ik + 1) & 1) + 3)].arrive_and_expect_tx(4096);
      tl::tma_load(B_desc, overlap_plan_mbar[(((ik + 1) & 1) + 3)], (&(((half_t*)b_shared)[(((ik + 1) & 1) * 2048)])), (((int)blockIdx.y) * 32), ((((((int)blockIdx.z) >> 3) * 256) + (ik * 64)) + 64), 0, (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[1].wait(((ik + 1) & 1));
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      half_t da_shared_local_cast_2[2];
      *(uint1*)(da_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)da_shared) + ((i_6 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __3;
      uint1 v__2 = *(uint1*)(da_shared_local_cast_2 + 0);
      ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__2))[0]);
      *(float2*)(da_local + (i_6 * 2)) = __3;
    }
    overlap_plan_mbar[6].arrive();
    overlap_plan_mbar[((ik & 1) + 3)].wait((ik >> 1));
    {
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b, (&(((half_t*)b_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b, ((ik & 1) * 4096));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local + (ki * 8)), uint64_t(desc_b + ((ki * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 16);
    }
    overlap_plan_mbar[((ik & 1) + 8)].arrive();
    overlap_plan_mbar[2].wait(((ik + 1) & 1));
    #pragma unroll
    for (int i_7 = 0; i_7 < 8; ++i_7) {
      half_t dt_shared_local_cast_3[2];
      *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_7 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __4;
      uint1 v__3 = *(uint1*)(dt_shared_local_cast_3 + 0);
      ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__3))[0]);
      *(float2*)(dt_local + (i_7 * 2)) = __4;
    }
    overlap_plan_mbar[7].arrive();
    #pragma unroll
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      scale[i_8] = (exp2f(((da_last[0] - da_local[i_8]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_8]);
    }
    overlap_plan_mbar[0].wait(((ik + 1) & 1));
    #pragma unroll
    for (int i_9 = 0; i_9 < 32; ++i_9) {
      x_local[i_9] = ((half_t*)x_shared)[((((((((i_9 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_9 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_9 & 3) >> 1) + (i_9 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
    }
    overlap_plan_mbar[5].arrive();
    #pragma unroll
    for (int i_10 = 0; i_10 < 32; ++i_10) {
      xt_local[i_10] = ((half_t)(((float)x_local[((((i_10 >> 2) * 4) + ((i_10 & 1) * 2)) + ((i_10 & 3) >> 1))]) * scale[(((i_10 >> 2) * 2) + (i_10 & 1))]));
    }
  }
  overlap_plan_mbar[4].wait(1);
  {
    tl::GmmaDescriptor desc_b_1;
    tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_1, (&(((half_t*)b_shared)[0])));
    tl::increase_descriptor_offset<int>(desc_b_1, 4096);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 16);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
    tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 16);
  }
  overlap_plan_mbar[9].arrive();
  #pragma unroll
  for (int i_11 = 0; i_11 < 2; ++i_11) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_11) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((half_t)acc[(i_11 * 8)]), ((half_t)acc[((i_11 * 8) + 1)])), __pack_half2(((half_t)acc[((i_11 * 8) + 2)]), ((half_t)acc[((i_11 * 8) + 3)])), __pack_half2(((half_t)acc[((i_11 * 8) + 4)]), ((half_t)acc[((i_11 * 8) + 5)])), __pack_half2(((half_t)acc[((i_11 * 8) + 6)]), ((half_t)acc[((i_11 * 8) + 7)])));
  }
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    __syncthreads();
    tl::tma_store(Output_desc, (&(((half_t*)acc_shared)[0])), (((int)blockIdx.y) * 32), 0, ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

