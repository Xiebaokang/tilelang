#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/atomic.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(float* __restrict__ FinalState, __grid_constant__ const CUtensorMap K_desc, float* __restrict__ O, const half_t* __restrict__ Q, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(float* __restrict__ FinalState, __grid_constant__ const CUtensorMap K_desc, float* __restrict__ O, const half_t* __restrict__ Q, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* o_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* h_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* q = ((void*)((char*)buf_dyn_shmem + 49152));
  void* s_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* v = ((void*)((char*)buf_dyn_shmem + 81920));
  void* k = ((void*)((char*)buf_dyn_shmem + 98304));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[6];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float h[32];
  float s[32];
  float o[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
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
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(h + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    half_t Q_local_cast_1[8];
    half_t q_local_cast[8];
    *(uint4*)(Q_local_cast_1 + 0) = *(uint4*)(Q + (((((((((int)blockIdx.z) >> 5) * 4194304) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    for (int vec = 0; vec < 2; ++vec) {
      float broadcast_var_1 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
      uint2 __1;
      float4 __2;
        float4 __3;
        uint2 v_ = *(uint2*)(Q_local_cast_1 + (vec * 4));
        ((float2*)(&__3))[0] = __half22float2(((half2*)(&v_))[0]);
        ((float2*)(&__3))[1] = __half22float2(((half2*)(&v_))[1]);
        float4 v__1 = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
        __2.x = (__3.x*v__1.x);
        __2.y = (__3.y*v__1.y);
        __2.z = (__3.z*v__1.z);
        __2.w = (__3.w*v__1.w);
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&__2))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&__2))[1]);
      *(uint2*)(q_local_cast + (vec * 4)) = __1;
    }
    *(uint4*)(((half_t*)q) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(q_local_cast + 0);
  }
  tl::cp_async_commit();
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[0].arrive_and_expect_tx(8192);
    tl::fence_proxy_async();
    tl::tma_load(K_desc, overlap_plan_mbar[0], (&(((half_t*)k)[0])), (((int)blockIdx.y) * 64), 0, (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
    overlap_plan_mbar[1].arrive_and_expect_tx(8192);
    tl::fence_proxy_async();
    tl::tma_load(V_desc, overlap_plan_mbar[1], (&(((half_t*)v)[0])), (((int)blockIdx.x) * 64), 0, (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  overlap_plan_mbar[0].wait(0);
  {
    tl::GmmaDescriptor desc_a;
    tl::GmmaDescriptor desc_b;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)q)[0])));
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)k)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + ((ki * 32) >> 4)), uint64_t(desc_b + ((ki * 32) >> 4)), ((uint32_t*)(s + 0)), ((0 < ki) ? 1 : 0));
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 32; ++i_2) {
    float condval;
    if ((((((i_2 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_2 & 1)) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_2 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
      condval = s[i_2];
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    ((half_t*)s_shared)[(((((((((((int)threadIdx.x) >> 5) * 1024) + (((i_2 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((i_2 >> 4) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_2 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_2 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_2 & 1))] = ((half_t)condval);
  }
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    half_t h_shared_local_cast_2[2];
    uint1 __4;
    float2 v__2 = *(float2*)(h + (i_3 * 2));
    ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
    *(uint1*)(h_shared_local_cast_2 + 0) = __4;
    *(uint1*)(((half_t*)h_shared) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_3 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_3 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_3 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_3 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(h_shared_local_cast_2 + 0);
  }
  overlap_plan_mbar[1].wait(0);
  {
    tl::GmmaDescriptor desc_a_1;
    tl::GmmaDescriptor desc_b_1;
    tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_1, (&(((half_t*)k)[0])));
    tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_1, (&(((half_t*)v)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_1 + ((ki_1 * 2048) >> 4)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), ((uint32_t*)(h + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
  }
  overlap_plan_mbar[3].arrive();
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    half_t Q_local_cast_4[8];
    half_t q_local_cast_3[8];
    *(uint4*)(Q_local_cast_4 + 0) = *(uint4*)(Q + ((((((((((int)blockIdx.z) >> 5) * 4194304) + (i_4 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 262144));
    for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
      float broadcast_var_2 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
      uint2 __5;
      float4 __6;
        float4 __7;
        uint2 v__3 = *(uint2*)(Q_local_cast_4 + (vec_1 * 4));
        ((float2*)(&__7))[0] = __half22float2(((half2*)(&v__3))[0]);
        ((float2*)(&__7))[1] = __half22float2(((half2*)(&v__3))[1]);
        float4 v__4 = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
        __6.x = (__7.x*v__4.x);
        __6.y = (__7.y*v__4.y);
        __6.z = (__7.z*v__4.z);
        __6.w = (__7.w*v__4.w);
      ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&__6))[0]);
      ((half2*)(&__5))[1] = __float22half2_rn(((float2*)(&__6))[1]);
      *(uint2*)(q_local_cast_3 + (vec_1 * 4)) = __5;
    }
    *(uint4*)(((half_t*)q) + ((((((i_4 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)) = *(uint4*)(q_local_cast_3 + 0);
  }
  tl::cp_async_commit();
  overlap_plan_mbar[3].wait(0);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[0].arrive_and_expect_tx(8192);
    tl::fence_proxy_async();
    tl::tma_load(K_desc, overlap_plan_mbar[0], (&(((half_t*)k)[0])), (((int)blockIdx.y) * 64), 64, (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
    overlap_plan_mbar[2].arrive_and_expect_tx(8192);
    tl::fence_proxy_async();
    tl::tma_load(V_desc, overlap_plan_mbar[2], (&(((half_t*)v)[4096])), (((int)blockIdx.x) * 64), 64, (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  overlap_plan_mbar[0].wait(1);
  {
    tl::GmmaDescriptor desc_a_2;
    tl::GmmaDescriptor desc_b_2;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)q)[0])));
    tl::increase_descriptor_offset<int>(desc_a_2, 8192);
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)k)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_2 + ((ki_2 * 32) >> 4)), uint64_t(desc_b_2 + ((ki_2 * 32) >> 4)), ((uint32_t*)(s + 0)), ((0 < ki_2) ? 1 : 0));
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 32; ++i_5) {
    float condval_1;
    if ((((((i_5 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_5 & 1)) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_5 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
      condval_1 = s[i_5];
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    ((half_t*)s_shared)[((((((((((((int)threadIdx.x) >> 5) * 1024) + (((i_5 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((i_5 >> 4) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_5 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_5 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_5 & 1)) + 4096)] = ((half_t)condval_1);
  }
  overlap_plan_mbar[1].wait(0);
  {
    tl::GmmaDescriptor desc_a_3;
    tl::GmmaDescriptor desc_b_3;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((half_t*)s_shared)[0])));
    tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_3, (&(((half_t*)v)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_3 + ((ki_3 * 32) >> 4)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), ((uint32_t*)(o + 0)), ((0 < ki_3) ? 1 : 0));
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
  }
  overlap_plan_mbar[4].arrive();
  #pragma unroll
  for (int i_6 = 0; i_6 < 16; ++i_6) {
    half_t h_shared_local_cast_5[2];
    uint1 __8;
    float2 v__5 = *(float2*)(h + (i_6 * 2));
    ((half2*)(&__8))[0] = __float22half2_rn(((float2*)(&v__5))[0]);
    *(uint1*)(h_shared_local_cast_5 + 0) = __8;
    *(uint1*)(((half_t*)h_shared) + (((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_6 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_6 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_6 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_6 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096)) = *(uint1*)(h_shared_local_cast_5 + 0);
  }
  overlap_plan_mbar[2].wait(0);
  {
    tl::GmmaDescriptor desc_a_4;
    tl::GmmaDescriptor desc_b_4;
    tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_4, (&(((half_t*)k)[0])));
    tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_4, (&(((half_t*)v)[0])));
    tl::increase_descriptor_offset<int>(desc_b_4, 8192);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_4 + ((ki_4 * 2048) >> 4)), uint64_t(desc_b_4 + ((ki_4 * 2048) >> 4)), ((uint32_t*)(h + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
  }
  overlap_plan_mbar[3].arrive();
  {
    tl::GmmaDescriptor desc_a_5;
    tl::GmmaDescriptor desc_b_5;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_5, (&(((half_t*)q)[0])));
    tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_5, (&(((half_t*)h_shared)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_5 + ((ki_5 * 32) >> 4)), uint64_t(desc_b_5 + ((ki_5 * 2048) >> 4)), ((uint32_t*)(o + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 16; ++i_7) {
    *(float2*)(((float*)o_shared) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_7 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_7 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(o + (i_7 * 2));
  }
  for (int ic = 0; ic < 14; ++ic) {
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      half_t Q_local_cast_7[8];
      half_t q_local_cast_6[8];
      *(uint4*)(Q_local_cast_7 + 0) = *(uint4*)(Q + (((((((((((int)blockIdx.z) >> 5) * 4194304) + (ic * 262144)) + (i_8 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 524288));
      for (int vec_2 = 0; vec_2 < 2; ++vec_2) {
        float broadcast_var_3 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
        uint2 __9;
        float4 __10;
          float4 __11;
          uint2 v__6 = *(uint2*)(Q_local_cast_7 + (vec_2 * 4));
          ((float2*)(&__11))[0] = __half22float2(((half2*)(&v__6))[0]);
          ((float2*)(&__11))[1] = __half22float2(((half2*)(&v__6))[1]);
          float4 v__7 = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
          __10.x = (__11.x*v__7.x);
          __10.y = (__11.y*v__7.y);
          __10.z = (__11.z*v__7.z);
          __10.w = (__11.w*v__7.w);
        ((half2*)(&__9))[0] = __float22half2_rn(((float2*)(&__10))[0]);
        ((half2*)(&__9))[1] = __float22half2_rn(((float2*)(&__10))[1]);
        *(uint2*)(q_local_cast_6 + (vec_2 * 4)) = __9;
      }
      *(uint4*)(((half_t*)q) + (((((((ic & 1) * 4096) + (i_8 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(q_local_cast_6 + 0);
    }
    tl::cp_async_commit();
    overlap_plan_mbar[3].wait(((ic + 1) & 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(8192);
      tl::fence_proxy_async();
      tl::tma_load(K_desc, overlap_plan_mbar[0], (&(((half_t*)k)[0])), (((int)blockIdx.y) * 64), ((ic * 64) + 128), (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
    }
    overlap_plan_mbar[((ic & 1) + 4)].wait(((ic & 3) >> 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[((ic & 1) + 1)].arrive_and_expect_tx(8192);
      tl::fence_proxy_async();
      tl::tma_load(V_desc, overlap_plan_mbar[((ic & 1) + 1)], (&(((half_t*)v)[((ic & 1) * 4096)])), (((int)blockIdx.x) * 64), ((ic * 64) + 128), (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    overlap_plan_mbar[0].wait((ic & 1));
    {
      tl::GmmaDescriptor desc_a_6;
      tl::GmmaDescriptor desc_b_6;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_6, (&(((half_t*)q)[0])));
      tl::increase_descriptor_offset<int>(desc_a_6, ((ic & 1) * 8192));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_6, (&(((half_t*)k)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_6 + ((ki_6 * 32) >> 4)), uint64_t(desc_b_6 + ((ki_6 * 32) >> 4)), ((uint32_t*)(s + 0)), ((0 < ki_6) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 32; ++i_9) {
      float condval_2;
      if ((((((i_9 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_9 & 1)) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_9 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_2 = s[i_9];
      } else {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      }
      ((half_t*)s_shared)[((((((((((ic & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((i_9 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((i_9 >> 4) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_9 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_9 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_9 & 1))] = ((half_t)condval_2);
    }
    overlap_plan_mbar[(((ic + 1) & 1) + 1)].wait((((ic + 1) & 3) >> 1));
    {
      tl::GmmaDescriptor desc_a_7;
      tl::GmmaDescriptor desc_b_7;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_7, (&(((half_t*)s_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_a_7, (((ic + 1) & 1) * 8192));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_7, (&(((half_t*)v)[0])));
      tl::increase_descriptor_offset<int>(desc_b_7, (((ic + 1) & 1) * 8192));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_7 + ((ki_7 * 32) >> 4)), uint64_t(desc_b_7 + ((ki_7 * 2048) >> 4)), ((uint32_t*)(o + 0)), ((0 < ki_7) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
    }
    overlap_plan_mbar[(((ic + 1) & 1) + 4)].arrive();
    #pragma unroll
    for (int i_10 = 0; i_10 < 16; ++i_10) {
      half_t h_shared_local_cast_8[2];
      uint1 __12;
      float2 v__8 = *(float2*)(h + (i_10 * 2));
      ((half2*)(&__12))[0] = __float22half2_rn(((float2*)(&v__8))[0]);
      *(uint1*)(h_shared_local_cast_8 + 0) = __12;
      *(uint1*)(((half_t*)h_shared) + (((((((((ic & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_10 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_10 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_10 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(h_shared_local_cast_8 + 0);
    }
    overlap_plan_mbar[((ic & 1) + 1)].wait((((ic >> 1) + 1) & 1));
    {
      tl::GmmaDescriptor desc_a_8;
      tl::GmmaDescriptor desc_b_8;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_8, (&(((half_t*)k)[0])));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_8, (&(((half_t*)v)[0])));
      tl::increase_descriptor_offset<int>(desc_b_8, ((ic & 1) * 8192));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_8 = 0; ki_8 < 4; ++ki_8) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_8 + ((ki_8 * 2048) >> 4)), uint64_t(desc_b_8 + ((ki_8 * 2048) >> 4)), ((uint32_t*)(h + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
    }
    overlap_plan_mbar[3].arrive();
    {
      tl::GmmaDescriptor desc_a_9;
      tl::GmmaDescriptor desc_b_9;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_9, (&(((half_t*)q)[0])));
      tl::increase_descriptor_offset<int>(desc_a_9, (((ic + 1) & 1) * 8192));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_9, (&(((half_t*)h_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_9, (((ic + 1) & 1) * 8192));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_9 + ((ki_9 * 32) >> 4)), uint64_t(desc_b_9 + ((ki_9 * 2048) >> 4)), ((uint32_t*)(o + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 16; ++i_11) {
      *(float2*)(((float*)o_shared) + ((((((((ic + 1) & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_11 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_11 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(o + (i_11 * 2));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 8; ++i_12) {
      AtomicAddx4((&(O[((((((((((int)blockIdx.z) >> 5) * 4194304) + (ic * 262144)) + (i_12 * 32768)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4))])), *(float4*)(((float*)o_shared) + ((((ic & 1) * 4096) + (i_12 * 512)) + (((int)threadIdx.x) * 4))));
    }
  }
  overlap_plan_mbar[2].wait(1);
  {
    tl::GmmaDescriptor desc_a_10;
    tl::GmmaDescriptor desc_b_10;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_10, (&(((half_t*)s_shared)[0])));
    tl::increase_descriptor_offset<int>(desc_a_10, 8192);
    tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_10, (&(((half_t*)v)[0])));
    tl::increase_descriptor_offset<int>(desc_b_10, 8192);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    __syncthreads();
    #pragma unroll
    for (int ki_10 = 0; ki_10 < 4; ++ki_10) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_10 + ((ki_10 * 32) >> 4)), uint64_t(desc_b_10 + ((ki_10 * 2048) >> 4)), ((uint32_t*)(o + 0)), ((0 < ki_10) ? 1 : 0));
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
  }
  overlap_plan_mbar[5].arrive();
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_a_11;
    tl::GmmaDescriptor desc_b_11;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_11, (&(((half_t*)q)[0])));
    tl::increase_descriptor_offset<int>(desc_a_11, 8192);
    tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_11, (&(((half_t*)h_shared)[0])));
    tl::increase_descriptor_offset<int>(desc_b_11, 8192);
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_11 = 0; ki_11 < 4; ++ki_11) {
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_11 + ((ki_11 * 32) >> 4)), uint64_t(desc_b_11 + ((ki_11 * 2048) >> 4)), ((uint32_t*)(o + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 16; ++i_13) {
    *(float2*)(((float*)o_shared) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_13 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_13 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096)) = *(float2*)(o + (i_13 * 2));
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 8; ++i_14) {
    AtomicAddx4((&(O[((((((((((int)blockIdx.z) >> 5) * 4194304) + (i_14 * 32768)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3670016)])), *(float4*)(((float*)o_shared) + ((i_14 * 512) + (((int)threadIdx.x) * 4))));
  }
  __syncthreads();
  #pragma unroll
  for (int i_15 = 0; i_15 < 8; ++i_15) {
    AtomicAddx4((&(O[((((((((((int)blockIdx.z) >> 5) * 4194304) + (i_15 * 32768)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3932160)])), *(float4*)(((float*)o_shared) + (((i_15 * 512) + (((int)threadIdx.x) * 4)) + 4096)));
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 16; ++i_16) {
    *(float2*)(FinalState + ((((((((((int)blockIdx.z) * 16384) + (((int)blockIdx.y) * 8192)) + ((((int)threadIdx.x) >> 5) * 2048)) + ((i_16 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + (((int)blockIdx.x) * 64)) + ((i_16 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(h + (i_16 * 2));
  }
}

