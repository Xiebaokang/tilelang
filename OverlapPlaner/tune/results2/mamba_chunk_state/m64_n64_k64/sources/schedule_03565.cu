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
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  void* da_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 33792));
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 34816));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[12];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float da_last[1];
  float acc[32];
  float da_local[16];
  float dt_local[16];
  half_t x_local_v0[32];
  float scale_v0[16];
  half_t x_local_v1[32];
  float scale_v1[16];
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
    overlap_plan_mbar[5].init(1);
    overlap_plan_mbar[6].init(128);
    overlap_plan_mbar[7].init(128);
    overlap_plan_mbar[8].init(128);
    overlap_plan_mbar[9].init(128);
    overlap_plan_mbar[10].init(128);
    overlap_plan_mbar[11].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  da_last[0] = ((float)DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 255)]);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
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
  overlap_plan_mbar[7].arrive();
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
  overlap_plan_mbar[8].arrive();
  overlap_plan_mbar[0].wait(0);
  #pragma unroll
  for (int i_3 = 0; i_3 < 32; ++i_3) {
    x_local_v0[i_3] = ((half_t*)x_shared)[((((((((i_3 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_3 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_3 & 3) >> 1) + (i_3 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
  }
  overlap_plan_mbar[6].arrive();
  #pragma unroll
  for (int i_4 = 0; i_4 < 16; ++i_4) {
    scale_v0[i_4] = (exp2f(((da_last[0] - da_local[i_4]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_4]);
  }
  overlap_plan_mbar[6].wait(0);
  __syncthreads();
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[0].arrive_and_expect_tx(8192);
    tl::tma_load(X_desc, overlap_plan_mbar[0], (&(((half_t*)x_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + 64), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[7].wait(0);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[1].arrive_and_expect_tx(128);
    tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 64)])), overlap_plan_mbar[1], 128);
  }
  overlap_plan_mbar[8].wait(0);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[2].arrive_and_expect_tx(128);
    tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 64)])), overlap_plan_mbar[2], 128);
    overlap_plan_mbar[3].arrive_and_expect_tx(8192);
    tl::tma_load(B_desc, overlap_plan_mbar[3], (&(((half_t*)b_shared)[0])), (((int)blockIdx.y) * 64), ((((int)blockIdx.z) >> 3) * 256), 0, (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[1].wait(1);
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 8; ++i_5) {
    half_t da_shared_local_cast_2[2];
    *(uint1*)(da_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)da_shared) + ((i_5 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __3;
    uint1 v__2 = *(uint1*)(da_shared_local_cast_2 + 0);
    ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__2))[0]);
    *(float2*)(da_local + (i_5 * 2)) = __3;
  }
  overlap_plan_mbar[7].arrive();
  overlap_plan_mbar[2].wait(1);
  #pragma unroll
  for (int i_6 = 0; i_6 < 8; ++i_6) {
    half_t dt_shared_local_cast_3[2];
    *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_6 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __4;
    uint1 v__3 = *(uint1*)(dt_shared_local_cast_3 + 0);
    ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__3))[0]);
    *(float2*)(dt_local + (i_6 * 2)) = __4;
  }
  overlap_plan_mbar[8].arrive();
  overlap_plan_mbar[0].wait(1);
  #pragma unroll
  for (int i_7 = 0; i_7 < 32; ++i_7) {
    x_local_v1[i_7] = ((half_t*)x_shared)[((((((((i_7 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_7 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_7 & 3) >> 1) + (i_7 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
  }
  overlap_plan_mbar[6].arrive();
  #pragma unroll
  for (int i_8 = 0; i_8 < 16; ++i_8) {
    scale_v1[i_8] = (exp2f(((da_last[0] - da_local[i_8]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_8]);
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 32; ++i_9) {
    xt_local[i_9] = ((half_t)(((float)x_local_v0[((((i_9 >> 2) * 4) + ((i_9 & 1) * 2)) + ((i_9 & 3) >> 1))]) * scale_v0[(((i_9 >> 2) * 2) + (i_9 & 1))]));
  }
  for (int ik = 0; ik < 2; ++ik) {
    overlap_plan_mbar[6].wait(((ik + 1) & 1));
    __syncthreads();
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(8192);
      tl::tma_load(X_desc, overlap_plan_mbar[0], (&(((half_t*)x_shared)[0])), 0, ((((((int)blockIdx.z) >> 3) * 256) + (ik * 64)) + 128), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[7].wait(((ik + 1) & 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 128)])), overlap_plan_mbar[1], 128);
    }
    overlap_plan_mbar[8].wait(((ik + 1) & 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 128)])), overlap_plan_mbar[2], 128);
      overlap_plan_mbar[(ik + 4)].arrive_and_expect_tx(8192);
      tl::tma_load(B_desc, overlap_plan_mbar[(ik + 4)], (&(((half_t*)b_shared)[((ik * 4096) + 4096)])), (((int)blockIdx.y) * 64), ((((((int)blockIdx.z) >> 3) * 256) + (ik * 64)) + 64), 0, (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[(ik + 3)].wait(0);
    {
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b, (&(((half_t*)b_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b, (ik * 8192));
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
    overlap_plan_mbar[(ik + 9)].arrive();
    overlap_plan_mbar[1].wait(ik);
    __syncthreads();
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      half_t da_shared_local_cast_4[2];
      *(uint1*)(da_shared_local_cast_4 + 0) = *(uint1*)(((half_t*)da_shared) + ((i_10 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __5;
      uint1 v__4 = *(uint1*)(da_shared_local_cast_4 + 0);
      ((float2*)(&__5))[0] = __half22float2(((half2*)(&v__4))[0]);
      *(float2*)(da_local + (i_10 * 2)) = __5;
    }
    overlap_plan_mbar[7].arrive();
    overlap_plan_mbar[2].wait(ik);
    #pragma unroll
    for (int i_11 = 0; i_11 < 8; ++i_11) {
      half_t dt_shared_local_cast_5[2];
      *(uint1*)(dt_shared_local_cast_5 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_11 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __6;
      uint1 v__5 = *(uint1*)(dt_shared_local_cast_5 + 0);
      ((float2*)(&__6))[0] = __half22float2(((half2*)(&v__5))[0]);
      *(float2*)(dt_local + (i_11 * 2)) = __6;
    }
    overlap_plan_mbar[8].arrive();
    overlap_plan_mbar[0].wait(ik);
    if (ik == 0) {
      #pragma unroll
      for (int i_12 = 0; i_12 < 32; ++i_12) {
        x_local_v0[i_12] = ((half_t*)x_shared)[((((((((i_12 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_12 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_12 & 3) >> 1) + (i_12 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
    } else {
      #pragma unroll
      for (int i_13 = 0; i_13 < 32; ++i_13) {
        x_local_v1[i_13] = ((half_t*)x_shared)[((((((((i_13 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_13 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_13 & 3) >> 1) + (i_13 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
    }
    overlap_plan_mbar[6].arrive();
    if (ik == 0) {
      #pragma unroll
      for (int i_14 = 0; i_14 < 16; ++i_14) {
        scale_v0[i_14] = (exp2f(((da_last[0] - da_local[i_14]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_14]);
      }
    } else {
      #pragma unroll
      for (int i_15 = 0; i_15 < 16; ++i_15) {
        scale_v1[i_15] = (exp2f(((da_last[0] - da_local[i_15]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_15]);
      }
    }
    if (((ik + 1) % 2) == 0) {
      #pragma unroll
      for (int i_16 = 0; i_16 < 32; ++i_16) {
        xt_local[i_16] = ((half_t)(((float)x_local_v0[((((i_16 >> 2) * 4) + ((i_16 & 1) * 2)) + ((i_16 & 3) >> 1))]) * scale_v0[(((i_16 >> 2) * 2) + (i_16 & 1))]));
      }
    } else {
      #pragma unroll
      for (int i_17 = 0; i_17 < 32; ++i_17) {
        xt_local[i_17] = ((half_t)(((float)x_local_v1[((((i_17 >> 2) * 4) + ((i_17 & 1) * 2)) + ((i_17 & 3) >> 1))]) * scale_v1[(((i_17 >> 2) * 2) + (i_17 & 1))]));
      }
    }
  }
  overlap_plan_mbar[9].wait(0);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[3].arrive_and_expect_tx(8192);
    tl::tma_load(B_desc, overlap_plan_mbar[3], (&(((half_t*)b_shared)[0])), (((int)blockIdx.y) * 64), (((((int)blockIdx.z) >> 3) * 256) + 192), 0, (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[5].wait(0);
  {
    tl::GmmaDescriptor desc_b_1;
    tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_1, (&(((half_t*)b_shared)[0])));
    tl::increase_descriptor_offset<int>(desc_b_1, 16384);
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
  overlap_plan_mbar[11].arrive();
  #pragma unroll
  for (int i_18 = 0; i_18 < 32; ++i_18) {
    xt_local[i_18] = ((half_t)(((float)x_local_v1[((((i_18 >> 2) * 4) + ((i_18 & 1) * 2)) + ((i_18 & 3) >> 1))]) * scale_v1[(((i_18 >> 2) * 2) + (i_18 & 1))]));
  }
  overlap_plan_mbar[3].wait(1);
  {
    tl::GmmaDescriptor desc_b_2;
    tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_2, (&(((half_t*)b_shared)[0])));
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
  overlap_plan_mbar[9].arrive();
  #pragma unroll
  for (int i_19 = 0; i_19 < 4; ++i_19) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_19 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc[(i_19 * 8)]), ((half_t)acc[((i_19 * 8) + 1)])), __pack_half2(((half_t)acc[((i_19 * 8) + 2)]), ((half_t)acc[((i_19 * 8) + 3)])), __pack_half2(((half_t)acc[((i_19 * 8) + 4)]), ((half_t)acc[((i_19 * 8) + 5)])), __pack_half2(((half_t)acc[((i_19 * 8) + 6)]), ((half_t)acc[((i_19 * 8) + 7)])));
  }
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    __syncthreads();
    tl::tma_store(Output_desc, (&(((half_t*)acc_shared)[0])), (((int)blockIdx.y) * 64), 0, ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

