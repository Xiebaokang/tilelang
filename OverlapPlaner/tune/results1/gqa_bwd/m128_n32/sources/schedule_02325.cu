#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/atomic.h>
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

extern "C" __global__ void main_kernel(const float* __restrict__ Delta, __grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc, __grid_constant__ const CUtensorMap dK_desc, __grid_constant__ const CUtensorMap dO_desc, float* __restrict__ dQ, __grid_constant__ const CUtensorMap dV_desc, const float* __restrict__ lse);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(const float* __restrict__ Delta, __grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc, __grid_constant__ const CUtensorMap dK_desc, __grid_constant__ const CUtensorMap dO_desc, float* __restrict__ dQ, __grid_constant__ const CUtensorMap dV_desc, const float* __restrict__ lse) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* dk_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* dv_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* dsT_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* q = ((void*)((char*)buf_dyn_shmem + 40960));
  void* do_1 = ((void*)((char*)buf_dyn_shmem + 49152));
  void* delta = ((void*)((char*)buf_dyn_shmem + 53248));
  void* lse_shared = ((void*)((char*)buf_dyn_shmem + 54272));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[12];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float dv[32];
  float dk[32];
  float qkT[16];
  float dsT[16];
  half_t qkT_cast[16];
  half_t dsT_cast_v0[16];
  float dq_v0[8];
  half_t dsT_cast_v1[16];
  float dq_v1[8];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::prefetch_tma_descriptor(dO_desc);
    tl::prefetch_tma_descriptor(dV_desc);
    tl::prefetch_tma_descriptor(dK_desc);
    tl::prefetch_tma_descriptor(Q_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(1);
    overlap_plan_mbar[4].init(1);
    overlap_plan_mbar[5].init(1);
    overlap_plan_mbar[6].init(1);
    overlap_plan_mbar[7].init(256);
    overlap_plan_mbar[8].init(256);
    overlap_plan_mbar[9].init(256);
    overlap_plan_mbar[10].init(256);
    overlap_plan_mbar[11].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    tl::warpgroup_reg_alloc<240>();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[0], (&(((half_t*)K_shared)[0])), 0, (((int)blockIdx.y) * 128), (((int)blockIdx.x) >> 2), 0);
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[1], (&(((half_t*)V_shared)[0])), 0, (((int)blockIdx.y) * 128), (((int)blockIdx.x) >> 2), 0);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dv + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dk + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(qkT + (i_2 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    overlap_plan_mbar[0].wait(0);
    overlap_plan_mbar[2].wait(0);
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)K_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)q)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a + ((((((int)threadIdx.x) >> 7) * 8192) + (ki * 32)) >> 4)), uint64_t(desc_b + ((ki * 32) >> 4)), ((uint32_t*)(qkT + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT + 0), 32);
    }
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(4096);
      tl::tma_load(dO_desc, overlap_plan_mbar[4], (&(((half_t*)do_1)[0])), 0, 0, ((int)blockIdx.x), 0);
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dsT + (i_3 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    }
    overlap_plan_mbar[1].wait(0);
    overlap_plan_mbar[4].wait(0);
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_1;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)V_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((half_t*)do_1)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dsT + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_1 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_1 * 32)) >> 4)), uint64_t(desc_b_1 + ((ki_1 * 32) >> 4)), ((uint32_t*)(dsT + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dsT + 0), 32);
    }
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[5].arrive_and_expect_tx(128);
      tl::tma_load((&(((float*)lse_shared)[0])), (&(lse[(((int)blockIdx.x) * 1024)])), overlap_plan_mbar[5], 128);
    }
    overlap_plan_mbar[5].wait(0);
    tl::__sync_thread_partial(3, 256);
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      float broadcast_var_4 = 0x1.7154764ee6c2fp-3f/*1.803369e-01*/;
      float2 __1;
      float2 __2;
        float2 __3;
          float2 v_ = *(float2*)(qkT + (i_4 * 2));
          float2 v__1 = make_float2(broadcast_var_4, broadcast_var_4);
          __3.x = (v_.x*v__1.x);
          __3.y = (v_.y*v__1.y);
        float2 v__2 = *(float2*)(((float*)lse_shared) + (((i_4 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)));
        __2.x = (__3.x-v__2.x);
        __2.y = (__3.y-v__2.y);
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(qkT + (i_4 * 2)) = __1;
    }
    overlap_plan_mbar[10].arrive();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      uint2 __4;
      float4 v__3 = *(float4*)(qkT + (i_5 * 4));
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
      ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&v__3))[1]);
      *(uint2*)(qkT_cast + (i_5 * 4)) = __4;
    }
    {
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_2, (&(((half_t*)do_1)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(qkT_cast + 0), 8);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dv + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(qkT_cast + (ki_2 * 8)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), reinterpret_cast<uint32_t*>(dv + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dv + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(qkT_cast + 0), 8);
    }
    overlap_plan_mbar[9].arrive();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[6].arrive_and_expect_tx(128);
      tl::tma_load((&(((float*)delta)[0])), (&(Delta[(((int)blockIdx.x) * 1024)])), overlap_plan_mbar[6], 128);
    }
    overlap_plan_mbar[6].wait(0);
    tl::__sync_thread_partial(3, 256);
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      float delta_local_cast[2];
      *(float2*)(delta_local_cast + 0) = *(float2*)(((float*)delta) + (((i_6 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float broadcast_var_5 = 0x1p-3f/*1.250000e-01*/;
      uint1 __5;
      float2 __6;
        float2 __7;
          float2 v__4 = *(float2*)(qkT + (i_6 * 2));
          float2 __8;
            float2 v__5 = *(float2*)(dsT + (i_6 * 2));
            float2 v__6 = *(float2*)(delta_local_cast + 0);
            __8.x = (v__5.x-v__6.x);
            __8.y = (v__5.y-v__6.y);
          __7.x = (v__4.x*__8.x);
          __7.y = (v__4.y*__8.y);
        float2 v__7 = make_float2(broadcast_var_5, broadcast_var_5);
        __6.x = (__7.x*v__7.x);
        __6.y = (__7.y*v__7.y);
      ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&__6))[0]);
      *(uint1*)(dsT_cast_v0 + (i_6 * 2)) = __5;
    }
    overlap_plan_mbar[11].arrive();
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)dsT_shared)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_7) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(dsT_cast_v0[(i_7 * 8)], dsT_cast_v0[((i_7 * 8) + 1)]), __pack_half2(dsT_cast_v0[((i_7 * 8) + 2)], dsT_cast_v0[((i_7 * 8) + 3)]), __pack_half2(dsT_cast_v0[((i_7 * 8) + 4)], dsT_cast_v0[((i_7 * 8) + 5)]), __pack_half2(dsT_cast_v0[((i_7 * 8) + 6)], dsT_cast_v0[((i_7 * 8) + 7)]));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dq_v0 + (i_8 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
    }
    {
      half_t A_local[8];
      half_t B_local[8];
      tl::__sync_thread_partial(3, 256);
      for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)dsT_shared)[(((((ki_3 * 512) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)K_shared)[(((ki_3 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_v0 + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_v0 + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
    }
    for (int k = 0; k < 15; ++k) {
      #pragma unroll
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(qkT + (i_9 * 4)) = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
      }
      overlap_plan_mbar[3].wait((k & 1));
      {
        tl::GmmaDescriptor desc_a_2;
        tl::GmmaDescriptor desc_b_3;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)K_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_3, (&(((half_t*)q)[0])));
        tl::increase_descriptor_offset<int>(desc_b_3, 4096);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT + 0), 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_2 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_4 * 32)) >> 4)), uint64_t(desc_b_3 + ((ki_4 * 32) >> 4)), ((uint32_t*)(qkT + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT + 0), 32);
      }
      overlap_plan_mbar[9].wait(0);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[4].arrive_and_expect_tx(4096);
        tl::tma_load(dO_desc, overlap_plan_mbar[4], (&(((half_t*)do_1)[0])), 0, ((k * 64) + 32), ((int)blockIdx.x), 0);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(dsT + (i_10 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
      }
      overlap_plan_mbar[4].wait(1);
      {
        tl::GmmaDescriptor desc_a_3;
        tl::GmmaDescriptor desc_b_4;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((half_t*)V_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_4, (&(((half_t*)do_1)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dsT + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_3 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_5 * 32)) >> 4)), uint64_t(desc_b_4 + ((ki_5 * 32) >> 4)), ((uint32_t*)(dsT + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dsT + 0), 32);
      }
      overlap_plan_mbar[10].wait(0);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[5].arrive_and_expect_tx(128);
        tl::tma_load((&(((float*)lse_shared)[0])), (&(lse[(((((int)blockIdx.x) * 1024) + (k * 64)) + 32)])), overlap_plan_mbar[5], 128);
      }
      overlap_plan_mbar[5].wait(1);
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int i_11 = 0; i_11 < 8; ++i_11) {
        float broadcast_var_9 = 0x1.7154764ee6c2fp-3f/*1.803369e-01*/;
        float2 __9;
        float2 __10;
          float2 __11;
            float2 v__8 = *(float2*)(qkT + (i_11 * 2));
            float2 v__9 = make_float2(broadcast_var_9, broadcast_var_9);
            __11.x = (v__8.x*v__9.x);
            __11.y = (v__8.y*v__9.y);
          float2 v__10 = *(float2*)(((float*)lse_shared) + (((i_11 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)));
          __10.x = (__11.x-v__10.x);
          __10.y = (__11.y-v__10.y);
        __9.x = exp2f(__10.x);
        __9.y = exp2f(__10.y);
        *(float2*)(qkT + (i_11 * 2)) = __9;
      }
      overlap_plan_mbar[10].arrive();
      #pragma unroll
      for (int i_12 = 0; i_12 < 4; ++i_12) {
        uint2 __12;
        float4 v__11 = *(float4*)(qkT + (i_12 * 4));
        ((half2*)(&__12))[0] = __float22half2_rn(((float2*)(&v__11))[0]);
        ((half2*)(&__12))[1] = __float22half2_rn(((float2*)(&v__11))[1]);
        *(uint2*)(qkT_cast + (i_12 * 4)) = __12;
      }
      {
        tl::GmmaDescriptor desc_b_5;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_5, (&(((half_t*)do_1)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(qkT_cast + 0), 8);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dv + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_6 = 0; ki_6 < 2; ++ki_6) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(qkT_cast + (ki_6 * 8)), uint64_t(desc_b_5 + ((ki_6 * 2048) >> 4)), reinterpret_cast<uint32_t*>(dv + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dv + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(qkT_cast + 0), 8);
      }
      overlap_plan_mbar[9].arrive();
      overlap_plan_mbar[11].wait(0);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[6].arrive_and_expect_tx(128);
        tl::tma_load((&(((float*)delta)[0])), (&(Delta[(((((int)blockIdx.x) * 1024) + (k * 64)) + 32)])), overlap_plan_mbar[6], 128);
      }
      overlap_plan_mbar[6].wait(1);
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int i_13 = 0; i_13 < 8; ++i_13) {
        float delta_local_cast_1[2];
        *(float2*)(delta_local_cast_1 + 0) = *(float2*)(((float*)delta) + (((i_13 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float broadcast_var_10 = 0x1p-3f/*1.250000e-01*/;
        uint1 __13;
        float2 __14;
          float2 __15;
            float2 v__12 = *(float2*)(qkT + (i_13 * 2));
            float2 __16;
              float2 v__13 = *(float2*)(dsT + (i_13 * 2));
              float2 v__14 = *(float2*)(delta_local_cast_1 + 0);
              __16.x = (v__13.x-v__14.x);
              __16.y = (v__13.y-v__14.y);
            __15.x = (v__12.x*__16.x);
            __15.y = (v__12.y*__16.y);
          float2 v__15 = make_float2(broadcast_var_10, broadcast_var_10);
          __14.x = (__15.x*v__15.x);
          __14.y = (__15.y*v__15.y);
        ((half2*)(&__13))[0] = __float22half2_rn(((float2*)(&__14))[0]);
        *(uint1*)(dsT_cast_v1 + (i_13 * 2)) = __13;
      }
      overlap_plan_mbar[11].arrive();
      overlap_plan_mbar[2].wait((k & 1));
      {
        tl::GmmaDescriptor desc_b_6;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_6, (&(((half_t*)q)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast_v0 + 0), 8);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_7 = 0; ki_7 < 2; ++ki_7) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(dsT_cast_v0 + (ki_7 * 8)), uint64_t(desc_b_6 + ((ki_7 * 2048) >> 4)), reinterpret_cast<uint32_t*>(dk + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast_v0 + 0), 8);
      }
      overlap_plan_mbar[7].arrive();
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)dsT_shared)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(dsT_cast_v1[(i_14 * 8)], dsT_cast_v1[((i_14 * 8) + 1)]), __pack_half2(dsT_cast_v1[((i_14 * 8) + 2)], dsT_cast_v1[((i_14 * 8) + 3)]), __pack_half2(dsT_cast_v1[((i_14 * 8) + 4)], dsT_cast_v1[((i_14 * 8) + 5)]), __pack_half2(dsT_cast_v1[((i_14 * 8) + 6)], dsT_cast_v1[((i_14 * 8) + 7)]));
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        float broadcast_var_11 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(dq_v1 + (i_15 * 4)) = make_float4(broadcast_var_11, broadcast_var_11, broadcast_var_11, broadcast_var_11);
      }
      {
        half_t A_local_1[8];
        half_t B_local_1[8];
        tl::__sync_thread_partial(3, 256);
        for (int ki_8 = 0; ki_8 < 8; ++ki_8) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)dsT_shared)[(((((ki_8 * 512) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_1[0])));
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)K_shared)[(((ki_8 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_1[0])));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_v1 + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_v1 + 4), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 4));
        }
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 8; ++i_16) {
        AtomicAdd((&(dQ[((((((((k * 65536) + (((((int)threadIdx.x) & 63) >> 5) * 16384)) + (((i_16 & 3) >> 1) * 8192)) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) >> 6) * 128)) + ((i_16 >> 2) * 64)) + ((i_16 & 1) * 32)) + (((int)threadIdx.x) & 31))])), dq_v0[i_16]);
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 4; ++i_17) {
        float broadcast_var_12 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(qkT + (i_17 * 4)) = make_float4(broadcast_var_12, broadcast_var_12, broadcast_var_12, broadcast_var_12);
      }
      overlap_plan_mbar[2].wait(((k + 1) & 1));
      {
        tl::GmmaDescriptor desc_a_4;
        tl::GmmaDescriptor desc_b_7;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_4, (&(((half_t*)K_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_7, (&(((half_t*)q)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT + 0), 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_4 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_9 * 32)) >> 4)), uint64_t(desc_b_7 + ((ki_9 * 32) >> 4)), ((uint32_t*)(qkT + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT + 0), 32);
      }
      overlap_plan_mbar[9].wait(1);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[4].arrive_and_expect_tx(4096);
        tl::tma_load(dO_desc, overlap_plan_mbar[4], (&(((half_t*)do_1)[0])), 0, ((k * 64) + 64), ((int)blockIdx.x), 0);
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 4; ++i_18) {
        float broadcast_var_13 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(dsT + (i_18 * 4)) = make_float4(broadcast_var_13, broadcast_var_13, broadcast_var_13, broadcast_var_13);
      }
      overlap_plan_mbar[4].wait(0);
      {
        tl::GmmaDescriptor desc_a_5;
        tl::GmmaDescriptor desc_b_8;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_5, (&(((half_t*)V_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_8, (&(((half_t*)do_1)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dsT + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_10 = 0; ki_10 < 4; ++ki_10) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_5 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_10 * 32)) >> 4)), uint64_t(desc_b_8 + ((ki_10 * 32) >> 4)), ((uint32_t*)(dsT + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dsT + 0), 32);
      }
      overlap_plan_mbar[10].wait(1);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[5].arrive_and_expect_tx(128);
        tl::tma_load((&(((float*)lse_shared)[0])), (&(lse[(((((int)blockIdx.x) * 1024) + (k * 64)) + 64)])), overlap_plan_mbar[5], 128);
      }
      overlap_plan_mbar[5].wait(0);
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int i_19 = 0; i_19 < 8; ++i_19) {
        float broadcast_var_14 = 0x1.7154764ee6c2fp-3f/*1.803369e-01*/;
        float2 __17;
        float2 __18;
          float2 __19;
            float2 v__16 = *(float2*)(qkT + (i_19 * 2));
            float2 v__17 = make_float2(broadcast_var_14, broadcast_var_14);
            __19.x = (v__16.x*v__17.x);
            __19.y = (v__16.y*v__17.y);
          float2 v__18 = *(float2*)(((float*)lse_shared) + (((i_19 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)));
          __18.x = (__19.x-v__18.x);
          __18.y = (__19.y-v__18.y);
        __17.x = exp2f(__18.x);
        __17.y = exp2f(__18.y);
        *(float2*)(qkT + (i_19 * 2)) = __17;
      }
      overlap_plan_mbar[10].arrive();
      #pragma unroll
      for (int i_20 = 0; i_20 < 4; ++i_20) {
        uint2 __20;
        float4 v__19 = *(float4*)(qkT + (i_20 * 4));
        ((half2*)(&__20))[0] = __float22half2_rn(((float2*)(&v__19))[0]);
        ((half2*)(&__20))[1] = __float22half2_rn(((float2*)(&v__19))[1]);
        *(uint2*)(qkT_cast + (i_20 * 4)) = __20;
      }
      {
        tl::GmmaDescriptor desc_b_9;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_9, (&(((half_t*)do_1)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(qkT_cast + 0), 8);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dv + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_11 = 0; ki_11 < 2; ++ki_11) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(qkT_cast + (ki_11 * 8)), uint64_t(desc_b_9 + ((ki_11 * 2048) >> 4)), reinterpret_cast<uint32_t*>(dv + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dv + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(qkT_cast + 0), 8);
      }
      overlap_plan_mbar[9].arrive();
      overlap_plan_mbar[11].wait(1);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[6].arrive_and_expect_tx(128);
        tl::tma_load((&(((float*)delta)[0])), (&(Delta[(((((int)blockIdx.x) * 1024) + (k * 64)) + 64)])), overlap_plan_mbar[6], 128);
      }
      overlap_plan_mbar[6].wait(0);
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int i_21 = 0; i_21 < 8; ++i_21) {
        float delta_local_cast_2[2];
        *(float2*)(delta_local_cast_2 + 0) = *(float2*)(((float*)delta) + (((i_21 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float broadcast_var_15 = 0x1p-3f/*1.250000e-01*/;
        uint1 __21;
        float2 __22;
          float2 __23;
            float2 v__20 = *(float2*)(qkT + (i_21 * 2));
            float2 __24;
              float2 v__21 = *(float2*)(dsT + (i_21 * 2));
              float2 v__22 = *(float2*)(delta_local_cast_2 + 0);
              __24.x = (v__21.x-v__22.x);
              __24.y = (v__21.y-v__22.y);
            __23.x = (v__20.x*__24.x);
            __23.y = (v__20.y*__24.y);
          float2 v__23 = make_float2(broadcast_var_15, broadcast_var_15);
          __22.x = (__23.x*v__23.x);
          __22.y = (__23.y*v__23.y);
        ((half2*)(&__21))[0] = __float22half2_rn(((float2*)(&__22))[0]);
        *(uint1*)(dsT_cast_v0 + (i_21 * 2)) = __21;
      }
      overlap_plan_mbar[11].arrive();
      overlap_plan_mbar[3].wait((k & 1));
      {
        tl::GmmaDescriptor desc_b_10;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_10, (&(((half_t*)q)[0])));
        tl::increase_descriptor_offset<int>(desc_b_10, 4096);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast_v1 + 0), 8);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_12 = 0; ki_12 < 2; ++ki_12) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(dsT_cast_v1 + (ki_12 * 8)), uint64_t(desc_b_10 + ((ki_12 * 2048) >> 4)), reinterpret_cast<uint32_t*>(dk + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast_v1 + 0), 8);
      }
      overlap_plan_mbar[8].arrive();
      #pragma unroll
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)dsT_shared)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_22) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(dsT_cast_v0[(i_22 * 8)], dsT_cast_v0[((i_22 * 8) + 1)]), __pack_half2(dsT_cast_v0[((i_22 * 8) + 2)], dsT_cast_v0[((i_22 * 8) + 3)]), __pack_half2(dsT_cast_v0[((i_22 * 8) + 4)], dsT_cast_v0[((i_22 * 8) + 5)]), __pack_half2(dsT_cast_v0[((i_22 * 8) + 6)], dsT_cast_v0[((i_22 * 8) + 7)]));
      }
      #pragma unroll
      for (int i_23 = 0; i_23 < 2; ++i_23) {
        float broadcast_var_16 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(dq_v0 + (i_23 * 4)) = make_float4(broadcast_var_16, broadcast_var_16, broadcast_var_16, broadcast_var_16);
      }
      {
        half_t A_local_2[8];
        half_t B_local_2[8];
        tl::__sync_thread_partial(3, 256);
        for (int ki_13 = 0; ki_13 < 8; ++ki_13) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)dsT_shared)[(((((ki_13 * 512) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_2[0])));
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)K_shared)[(((ki_13 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_2[0])));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_v0 + 0), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 0));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_v0 + 4), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 4));
        }
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 8; ++i_24) {
        AtomicAdd((&(dQ[(((((((((k * 65536) + (((((int)threadIdx.x) & 63) >> 5) * 16384)) + (((i_24 & 3) >> 1) * 8192)) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) >> 6) * 128)) + ((i_24 >> 2) * 64)) + ((i_24 & 1) * 32)) + (((int)threadIdx.x) & 31)) + 32768)])), dq_v1[i_24]);
      }
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 4; ++i_25) {
      float broadcast_var_17 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(qkT + (i_25 * 4)) = make_float4(broadcast_var_17, broadcast_var_17, broadcast_var_17, broadcast_var_17);
    }
    overlap_plan_mbar[3].wait(1);
    {
      tl::GmmaDescriptor desc_a_6;
      tl::GmmaDescriptor desc_b_11;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_6, (&(((half_t*)K_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_11, (&(((half_t*)q)[0])));
      tl::increase_descriptor_offset<int>(desc_b_11, 4096);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_14 = 0; ki_14 < 4; ++ki_14) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_6 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_14 * 32)) >> 4)), uint64_t(desc_b_11 + ((ki_14 * 32) >> 4)), ((uint32_t*)(qkT + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT + 0), 32);
    }
    overlap_plan_mbar[9].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(4096);
      tl::tma_load(dO_desc, overlap_plan_mbar[4], (&(((half_t*)do_1)[0])), 0, 992, ((int)blockIdx.x), 0);
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 4; ++i_26) {
      float broadcast_var_18 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dsT + (i_26 * 4)) = make_float4(broadcast_var_18, broadcast_var_18, broadcast_var_18, broadcast_var_18);
    }
    overlap_plan_mbar[4].wait(1);
    {
      tl::GmmaDescriptor desc_a_7;
      tl::GmmaDescriptor desc_b_12;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_7, (&(((half_t*)V_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_12, (&(((half_t*)do_1)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dsT + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_15 = 0; ki_15 < 4; ++ki_15) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_7 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_15 * 32)) >> 4)), uint64_t(desc_b_12 + ((ki_15 * 32) >> 4)), ((uint32_t*)(dsT + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dsT + 0), 32);
    }
    overlap_plan_mbar[10].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[5].arrive_and_expect_tx(128);
      tl::tma_load((&(((float*)lse_shared)[0])), (&(lse[((((int)blockIdx.x) * 1024) + 992)])), overlap_plan_mbar[5], 128);
    }
    overlap_plan_mbar[5].wait(1);
    tl::__sync_thread_partial(3, 256);
    #pragma unroll
    for (int i_27 = 0; i_27 < 8; ++i_27) {
      float broadcast_var_19 = 0x1.7154764ee6c2fp-3f/*1.803369e-01*/;
      float2 __25;
      float2 __26;
        float2 __27;
          float2 v__24 = *(float2*)(qkT + (i_27 * 2));
          float2 v__25 = make_float2(broadcast_var_19, broadcast_var_19);
          __27.x = (v__24.x*v__25.x);
          __27.y = (v__24.y*v__25.y);
        float2 v__26 = *(float2*)(((float*)lse_shared) + (((i_27 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)));
        __26.x = (__27.x-v__26.x);
        __26.y = (__27.y-v__26.y);
      __25.x = exp2f(__26.x);
      __25.y = exp2f(__26.y);
      *(float2*)(qkT + (i_27 * 2)) = __25;
    }
    overlap_plan_mbar[10].arrive();
    #pragma unroll
    for (int i_28 = 0; i_28 < 4; ++i_28) {
      uint2 __28;
      float4 v__27 = *(float4*)(qkT + (i_28 * 4));
      ((half2*)(&__28))[0] = __float22half2_rn(((float2*)(&v__27))[0]);
      ((half2*)(&__28))[1] = __float22half2_rn(((float2*)(&v__27))[1]);
      *(uint2*)(qkT_cast + (i_28 * 4)) = __28;
    }
    {
      tl::GmmaDescriptor desc_b_13;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_13, (&(((half_t*)do_1)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(qkT_cast + 0), 8);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dv + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_16 = 0; ki_16 < 2; ++ki_16) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(qkT_cast + (ki_16 * 8)), uint64_t(desc_b_13 + ((ki_16 * 2048) >> 4)), reinterpret_cast<uint32_t*>(dv + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dv + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(qkT_cast + 0), 8);
    }
    overlap_plan_mbar[9].arrive();
    overlap_plan_mbar[11].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[6].arrive_and_expect_tx(128);
      tl::tma_load((&(((float*)delta)[0])), (&(Delta[((((int)blockIdx.x) * 1024) + 992)])), overlap_plan_mbar[6], 128);
    }
    overlap_plan_mbar[6].wait(1);
    tl::__sync_thread_partial(3, 256);
    #pragma unroll
    for (int i_29 = 0; i_29 < 8; ++i_29) {
      float delta_local_cast_3[2];
      *(float2*)(delta_local_cast_3 + 0) = *(float2*)(((float*)delta) + (((i_29 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float broadcast_var_20 = 0x1p-3f/*1.250000e-01*/;
      uint1 __29;
      float2 __30;
        float2 __31;
          float2 v__28 = *(float2*)(qkT + (i_29 * 2));
          float2 __32;
            float2 v__29 = *(float2*)(dsT + (i_29 * 2));
            float2 v__30 = *(float2*)(delta_local_cast_3 + 0);
            __32.x = (v__29.x-v__30.x);
            __32.y = (v__29.y-v__30.y);
          __31.x = (v__28.x*__32.x);
          __31.y = (v__28.y*__32.y);
        float2 v__31 = make_float2(broadcast_var_20, broadcast_var_20);
        __30.x = (__31.x*v__31.x);
        __30.y = (__31.y*v__31.y);
      ((half2*)(&__29))[0] = __float22half2_rn(((float2*)(&__30))[0]);
      *(uint1*)(dsT_cast_v1 + (i_29 * 2)) = __29;
    }
    overlap_plan_mbar[11].arrive();
    overlap_plan_mbar[2].wait(1);
    {
      tl::GmmaDescriptor desc_b_14;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_14, (&(((half_t*)q)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast_v0 + 0), 8);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_17 = 0; ki_17 < 2; ++ki_17) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(dsT_cast_v0 + (ki_17 * 8)), uint64_t(desc_b_14 + ((ki_17 * 2048) >> 4)), reinterpret_cast<uint32_t*>(dk + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast_v0 + 0), 8);
    }
    overlap_plan_mbar[7].arrive();
    #pragma unroll
    for (int i_30 = 0; i_30 < 2; ++i_30) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)dsT_shared)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_30) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(dsT_cast_v1[(i_30 * 8)], dsT_cast_v1[((i_30 * 8) + 1)]), __pack_half2(dsT_cast_v1[((i_30 * 8) + 2)], dsT_cast_v1[((i_30 * 8) + 3)]), __pack_half2(dsT_cast_v1[((i_30 * 8) + 4)], dsT_cast_v1[((i_30 * 8) + 5)]), __pack_half2(dsT_cast_v1[((i_30 * 8) + 6)], dsT_cast_v1[((i_30 * 8) + 7)]));
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 2; ++i_31) {
      float broadcast_var_21 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dq_v1 + (i_31 * 4)) = make_float4(broadcast_var_21, broadcast_var_21, broadcast_var_21, broadcast_var_21);
    }
    {
      half_t A_local_3[8];
      half_t B_local_3[8];
      tl::__sync_thread_partial(3, 256);
      for (int ki_18 = 0; ki_18 < 8; ++ki_18) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)dsT_shared)[(((((ki_18 * 512) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_3[0])));
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)K_shared)[(((ki_18 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_3[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_v1 + 0), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 0));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq_v1 + 4), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 4));
      }
    }
    #pragma unroll
    for (int i_32 = 0; i_32 < 8; ++i_32) {
      AtomicAdd((&(dQ[((((((((((((int)threadIdx.x) & 63) >> 5) * 16384) + (((i_32 & 3) >> 1) * 8192)) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) >> 6) * 128)) + ((i_32 >> 2) * 64)) + ((i_32 & 1) * 32)) + (((int)threadIdx.x) & 31)) + 983040)])), dq_v0[i_32]);
    }
    overlap_plan_mbar[3].wait(1);
    {
      tl::GmmaDescriptor desc_b_15;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_15, (&(((half_t*)q)[0])));
      tl::increase_descriptor_offset<int>(desc_b_15, 4096);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast_v1 + 0), 8);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_19 = 0; ki_19 < 2; ++ki_19) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(dsT_cast_v1 + (ki_19 * 8)), uint64_t(desc_b_15 + ((ki_19 * 2048) >> 4)), reinterpret_cast<uint32_t*>(dk + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast_v1 + 0), 8);
    }
    overlap_plan_mbar[8].arrive();
    #pragma unroll
    for (int i_33 = 0; i_33 < 8; ++i_33) {
      AtomicAdd((&(dQ[((((((((((((int)threadIdx.x) & 63) >> 5) * 16384) + (((i_33 & 3) >> 1) * 8192)) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) >> 6) * 128)) + ((i_33 >> 2) * 64)) + ((i_33 & 1) * 32)) + (((int)threadIdx.x) & 31)) + 1015808)])), dq_v1[i_33]);
    }
    tl::__sync_thread_partial(3, 256);
    #pragma unroll
    for (int i_34 = 0; i_34 < 4; ++i_34) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)dv_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_34 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_34 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)dv[(i_34 * 8)]), ((half_t)dv[((i_34 * 8) + 1)])), __pack_half2(((half_t)dv[((i_34 * 8) + 2)]), ((half_t)dv[((i_34 * 8) + 3)])), __pack_half2(((half_t)dv[((i_34 * 8) + 4)]), ((half_t)dv[((i_34 * 8) + 5)])), __pack_half2(((half_t)dv[((i_34 * 8) + 6)]), ((half_t)dv[((i_34 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 256);
    if (tl::tl_shuffle_elect<256>()) {
      tl::fence_proxy_async();
      tl::tma_store(dV_desc, (&(((half_t*)dv_shared)[0])), 0, (((int)blockIdx.y) * 128), (((int)blockIdx.x) >> 2), 0, (((int)blockIdx.x) & 3));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
    tl::__sync_thread_partial(3, 256);
    #pragma unroll
    for (int i_35 = 0; i_35 < 4; ++i_35) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)dk_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_35 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_35 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)dk[(i_35 * 8)]), ((half_t)dk[((i_35 * 8) + 1)])), __pack_half2(((half_t)dk[((i_35 * 8) + 2)]), ((half_t)dk[((i_35 * 8) + 3)])), __pack_half2(((half_t)dk[((i_35 * 8) + 4)]), ((half_t)dk[((i_35 * 8) + 5)])), __pack_half2(((half_t)dk[((i_35 * 8) + 6)]), ((half_t)dk[((i_35 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 256);
    if (tl::tl_shuffle_elect<256>()) {
      tl::fence_proxy_async();
      tl::tma_store(dK_desc, (&(((half_t*)dk_shared)[0])), 0, (((int)blockIdx.y) * 128), (((int)blockIdx.x) >> 2), 0, (((int)blockIdx.x) & 3));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  } else {
    tl::warpgroup_reg_dealloc<24>();
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      if (2 <= k_1) {
        overlap_plan_mbar[((k_1 & 1) + 7)].wait((((k_1 >> 1) + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[((k_1 & 1) + 2)].arrive_and_expect_tx(4096);
        tl::tma_load(Q_desc, overlap_plan_mbar[((k_1 & 1) + 2)], (&(((half_t*)q)[((k_1 & 1) * 2048)])), 0, (k_1 * 32), ((int)blockIdx.x), 0);
      }
    }
  }
}

