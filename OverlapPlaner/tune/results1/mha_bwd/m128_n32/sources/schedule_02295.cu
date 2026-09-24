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
  void* gradient_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* do_1 = ((void*)((char*)buf_dyn_shmem + 32768));
  void* dsT_shared = ((void*)((char*)buf_dyn_shmem + 40960));
  void* q = ((void*)((char*)buf_dyn_shmem + 49152));
  void* delta = ((void*)((char*)buf_dyn_shmem + 57344));
  void* lse_shared = ((void*)((char*)buf_dyn_shmem + 58368));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[21];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float dv[32];
  float dk[32];
  float qkT_v0[16];
  float dsT[16];
  half_t dsT_cast[16];
  float dq_v0[8];
  float qkT_v1[16];
  half_t qkT_cast[16];
  float dq_v1[8];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(dO_desc);
    tl::prefetch_tma_descriptor(dV_desc);
    tl::prefetch_tma_descriptor(dK_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(1);
    overlap_plan_mbar[4].init(1);
    overlap_plan_mbar[5].init(1);
    overlap_plan_mbar[6].init(1);
    overlap_plan_mbar[7].init(1);
    overlap_plan_mbar[8].init(1);
    overlap_plan_mbar[9].init(1);
    overlap_plan_mbar[10].init(256);
    overlap_plan_mbar[11].init(128);
    overlap_plan_mbar[12].init(256);
    overlap_plan_mbar[13].init(256);
    overlap_plan_mbar[14].init(256);
    overlap_plan_mbar[15].init(256);
    overlap_plan_mbar[16].init(256);
    overlap_plan_mbar[17].init(256);
    overlap_plan_mbar[18].init(256);
    overlap_plan_mbar[19].init(256);
    overlap_plan_mbar[20].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    tl::warpgroup_reg_alloc<240>();
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
      *(float4*)(qkT_v0 + (i_2 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    overlap_plan_mbar[0].wait(0);
    overlap_plan_mbar[2].wait(0);
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)K_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)q)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT_v0 + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a + ((((((int)threadIdx.x) >> 7) * 8192) + (ki * 32)) >> 4)), uint64_t(desc_b + ((ki * 32) >> 4)), ((uint32_t*)(qkT_v0 + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT_v0 + 0), 32);
    }
    overlap_plan_mbar[4].wait(0);
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      float broadcast_var_3 = 0x1.7154764ee6c2fp-3f/*1.803369e-01*/;
      float2 __1;
      float2 __2;
        float2 __3;
          float2 v_ = *(float2*)(qkT_v0 + (i_3 * 2));
          float2 v__1 = make_float2(broadcast_var_3, broadcast_var_3);
          __3.x = (v_.x*v__1.x);
          __3.y = (v_.y*v__1.y);
        float2 v__2 = *(float2*)(((float*)lse_shared) + (((i_3 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)));
        __2.x = (__3.x-v__2.x);
        __2.y = (__3.y-v__2.y);
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(qkT_v0 + (i_3 * 2)) = __1;
    }
    overlap_plan_mbar[15].arrive();
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dsT + (i_4 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    overlap_plan_mbar[1].wait(0);
    overlap_plan_mbar[6].wait(0);
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
    overlap_plan_mbar[8].wait(0);
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      float delta_local_cast[2];
      *(float2*)(delta_local_cast + 0) = *(float2*)(((float*)delta) + (((i_5 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float broadcast_var_5 = 0x1p-3f/*1.250000e-01*/;
      uint1 __4;
      float2 __5;
        float2 __6;
          float2 v__3 = *(float2*)(qkT_v0 + (i_5 * 2));
          float2 __7;
            float2 v__4 = *(float2*)(dsT + (i_5 * 2));
            float2 v__5 = *(float2*)(delta_local_cast + 0);
            __7.x = (v__4.x-v__5.x);
            __7.y = (v__4.y-v__5.y);
          __6.x = (v__3.x*__7.x);
          __6.y = (v__3.y*__7.y);
        float2 v__6 = make_float2(broadcast_var_5, broadcast_var_5);
        __5.x = (__6.x*v__6.x);
        __5.y = (__6.y*v__6.y);
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&__5))[0]);
      *(uint1*)(dsT_cast + (i_5 * 2)) = __4;
    }
    overlap_plan_mbar[19].arrive();
    {
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_2, (&(((half_t*)q)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast + 0), 8);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(dsT_cast + (ki_2 * 8)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), reinterpret_cast<uint32_t*>(dk + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast + 0), 8);
    }
    overlap_plan_mbar[13].arrive();
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)dsT_shared)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_6) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(dsT_cast[(i_6 * 8)], dsT_cast[((i_6 * 8) + 1)]), __pack_half2(dsT_cast[((i_6 * 8) + 2)], dsT_cast[((i_6 * 8) + 3)]), __pack_half2(dsT_cast[((i_6 * 8) + 4)], dsT_cast[((i_6 * 8) + 5)]), __pack_half2(dsT_cast[((i_6 * 8) + 6)], dsT_cast[((i_6 * 8) + 7)]));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dq_v0 + (i_7 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
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
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(qkT_v1 + (i_8 * 4)) = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
      }
      overlap_plan_mbar[3].wait((k & 1));
      {
        tl::GmmaDescriptor desc_a_2;
        tl::GmmaDescriptor desc_b_3;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)K_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_3, (&(((half_t*)q)[0])));
        tl::increase_descriptor_offset<int>(desc_b_3, 4096);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT_v1 + 0), 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_2 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_4 * 32)) >> 4)), uint64_t(desc_b_3 + ((ki_4 * 32) >> 4)), ((uint32_t*)(qkT_v1 + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT_v1 + 0), 32);
      }
      overlap_plan_mbar[5].wait((k & 1));
      #pragma unroll
      for (int i_9 = 0; i_9 < 8; ++i_9) {
        float broadcast_var_8 = 0x1.7154764ee6c2fp-3f/*1.803369e-01*/;
        float2 __8;
        float2 __9;
          float2 __10;
            float2 v__7 = *(float2*)(qkT_v1 + (i_9 * 2));
            float2 v__8 = make_float2(broadcast_var_8, broadcast_var_8);
            __10.x = (v__7.x*v__8.x);
            __10.y = (v__7.y*v__8.y);
          float2 v__9 = *(float2*)(((float*)lse_shared) + ((((i_9 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 32));
          __9.x = (__10.x-v__9.x);
          __9.y = (__10.y-v__9.y);
        __8.x = exp2f(__9.x);
        __8.y = exp2f(__9.y);
        *(float2*)(qkT_v1 + (i_9 * 2)) = __8;
      }
      overlap_plan_mbar[16].arrive();
      #pragma unroll
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(dsT + (i_10 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
      }
      overlap_plan_mbar[7].wait((k & 1));
      {
        tl::GmmaDescriptor desc_a_3;
        tl::GmmaDescriptor desc_b_4;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((half_t*)V_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_4, (&(((half_t*)do_1)[0])));
        tl::increase_descriptor_offset<int>(desc_b_4, 4096);
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
      #pragma unroll
      for (int i_11 = 0; i_11 < 4; ++i_11) {
        uint2 __11;
        float4 v__10 = *(float4*)(qkT_v0 + (i_11 * 4));
        ((half2*)(&__11))[0] = __float22half2_rn(((float2*)(&v__10))[0]);
        ((half2*)(&__11))[1] = __float22half2_rn(((float2*)(&v__10))[1]);
        *(uint2*)(qkT_cast + (i_11 * 4)) = __11;
      }
      overlap_plan_mbar[6].wait((k & 1));
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
      overlap_plan_mbar[17].arrive();
      overlap_plan_mbar[9].wait((k & 1));
      #pragma unroll
      for (int i_12 = 0; i_12 < 8; ++i_12) {
        float delta_local_cast_1[2];
        *(float2*)(delta_local_cast_1 + 0) = *(float2*)(((float*)delta) + ((((i_12 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 32));
        float broadcast_var_10 = 0x1p-3f/*1.250000e-01*/;
        uint1 __12;
        float2 __13;
          float2 __14;
            float2 v__11 = *(float2*)(qkT_v1 + (i_12 * 2));
            float2 __15;
              float2 v__12 = *(float2*)(dsT + (i_12 * 2));
              float2 v__13 = *(float2*)(delta_local_cast_1 + 0);
              __15.x = (v__12.x-v__13.x);
              __15.y = (v__12.y-v__13.y);
            __14.x = (v__11.x*__15.x);
            __14.y = (v__11.y*__15.y);
          float2 v__14 = make_float2(broadcast_var_10, broadcast_var_10);
          __13.x = (__14.x*v__14.x);
          __13.y = (__14.y*v__14.y);
        ((half2*)(&__12))[0] = __float22half2_rn(((float2*)(&__13))[0]);
        *(uint1*)(dsT_cast + (i_12 * 2)) = __12;
      }
      overlap_plan_mbar[20].arrive();
      {
        tl::GmmaDescriptor desc_b_6;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_6, (&(((half_t*)q)[0])));
        tl::increase_descriptor_offset<int>(desc_b_6, 4096);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast + 0), 8);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_7 = 0; ki_7 < 2; ++ki_7) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(dsT_cast + (ki_7 * 8)), uint64_t(desc_b_6 + ((ki_7 * 2048) >> 4)), reinterpret_cast<uint32_t*>(dk + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast + 0), 8);
      }
      overlap_plan_mbar[14].arrive();
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)dsT_shared)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_13) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(dsT_cast[(i_13 * 8)], dsT_cast[((i_13 * 8) + 1)]), __pack_half2(dsT_cast[((i_13 * 8) + 2)], dsT_cast[((i_13 * 8) + 3)]), __pack_half2(dsT_cast[((i_13 * 8) + 4)], dsT_cast[((i_13 * 8) + 5)]), __pack_half2(dsT_cast[((i_13 * 8) + 6)], dsT_cast[((i_13 * 8) + 7)]));
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        float broadcast_var_11 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(dq_v1 + (i_14 * 4)) = make_float4(broadcast_var_11, broadcast_var_11, broadcast_var_11, broadcast_var_11);
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
      for (int i_15 = 0; i_15 < 8; ++i_15) {
        AtomicAdd((&(dQ[((((((((k * 65536) + (((((int)threadIdx.x) & 63) >> 5) * 16384)) + (((i_15 & 3) >> 1) * 8192)) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) >> 6) * 128)) + ((i_15 >> 2) * 64)) + ((i_15 & 1) * 32)) + (((int)threadIdx.x) & 31))])), dq_v0[i_15]);
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 4; ++i_16) {
        float broadcast_var_12 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(qkT_v0 + (i_16 * 4)) = make_float4(broadcast_var_12, broadcast_var_12, broadcast_var_12, broadcast_var_12);
      }
      overlap_plan_mbar[2].wait(((k + 1) & 1));
      {
        tl::GmmaDescriptor desc_a_4;
        tl::GmmaDescriptor desc_b_7;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_4, (&(((half_t*)K_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_7, (&(((half_t*)q)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT_v0 + 0), 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_4 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_9 * 32)) >> 4)), uint64_t(desc_b_7 + ((ki_9 * 32) >> 4)), ((uint32_t*)(qkT_v0 + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT_v0 + 0), 32);
      }
      overlap_plan_mbar[4].wait(((k + 1) & 1));
      #pragma unroll
      for (int i_17 = 0; i_17 < 8; ++i_17) {
        float broadcast_var_13 = 0x1.7154764ee6c2fp-3f/*1.803369e-01*/;
        float2 __16;
        float2 __17;
          float2 __18;
            float2 v__15 = *(float2*)(qkT_v0 + (i_17 * 2));
            float2 v__16 = make_float2(broadcast_var_13, broadcast_var_13);
            __18.x = (v__15.x*v__16.x);
            __18.y = (v__15.y*v__16.y);
          float2 v__17 = *(float2*)(((float*)lse_shared) + (((i_17 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)));
          __17.x = (__18.x-v__17.x);
          __17.y = (__18.y-v__17.y);
        __16.x = exp2f(__17.x);
        __16.y = exp2f(__17.y);
        *(float2*)(qkT_v0 + (i_17 * 2)) = __16;
      }
      overlap_plan_mbar[15].arrive();
      #pragma unroll
      for (int i_18 = 0; i_18 < 4; ++i_18) {
        float broadcast_var_14 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(dsT + (i_18 * 4)) = make_float4(broadcast_var_14, broadcast_var_14, broadcast_var_14, broadcast_var_14);
      }
      overlap_plan_mbar[6].wait(((k + 1) & 1));
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
      #pragma unroll
      for (int i_19 = 0; i_19 < 4; ++i_19) {
        uint2 __19;
        float4 v__18 = *(float4*)(qkT_v1 + (i_19 * 4));
        ((half2*)(&__19))[0] = __float22half2_rn(((float2*)(&v__18))[0]);
        ((half2*)(&__19))[1] = __float22half2_rn(((float2*)(&v__18))[1]);
        *(uint2*)(qkT_cast + (i_19 * 4)) = __19;
      }
      overlap_plan_mbar[7].wait((k & 1));
      {
        tl::GmmaDescriptor desc_b_9;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_9, (&(((half_t*)do_1)[0])));
        tl::increase_descriptor_offset<int>(desc_b_9, 4096);
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
      overlap_plan_mbar[18].arrive();
      overlap_plan_mbar[8].wait(((k + 1) & 1));
      #pragma unroll
      for (int i_20 = 0; i_20 < 8; ++i_20) {
        float delta_local_cast_2[2];
        *(float2*)(delta_local_cast_2 + 0) = *(float2*)(((float*)delta) + (((i_20 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float broadcast_var_15 = 0x1p-3f/*1.250000e-01*/;
        uint1 __20;
        float2 __21;
          float2 __22;
            float2 v__19 = *(float2*)(qkT_v0 + (i_20 * 2));
            float2 __23;
              float2 v__20 = *(float2*)(dsT + (i_20 * 2));
              float2 v__21 = *(float2*)(delta_local_cast_2 + 0);
              __23.x = (v__20.x-v__21.x);
              __23.y = (v__20.y-v__21.y);
            __22.x = (v__19.x*__23.x);
            __22.y = (v__19.y*__23.y);
          float2 v__22 = make_float2(broadcast_var_15, broadcast_var_15);
          __21.x = (__22.x*v__22.x);
          __21.y = (__22.y*v__22.y);
        ((half2*)(&__20))[0] = __float22half2_rn(((float2*)(&__21))[0]);
        *(uint1*)(dsT_cast + (i_20 * 2)) = __20;
      }
      overlap_plan_mbar[19].arrive();
      {
        tl::GmmaDescriptor desc_b_10;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_10, (&(((half_t*)q)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast + 0), 8);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_12 = 0; ki_12 < 2; ++ki_12) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(dsT_cast + (ki_12 * 8)), uint64_t(desc_b_10 + ((ki_12 * 2048) >> 4)), reinterpret_cast<uint32_t*>(dk + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast + 0), 8);
      }
      overlap_plan_mbar[13].arrive();
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)dsT_shared)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_21) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(dsT_cast[(i_21 * 8)], dsT_cast[((i_21 * 8) + 1)]), __pack_half2(dsT_cast[((i_21 * 8) + 2)], dsT_cast[((i_21 * 8) + 3)]), __pack_half2(dsT_cast[((i_21 * 8) + 4)], dsT_cast[((i_21 * 8) + 5)]), __pack_half2(dsT_cast[((i_21 * 8) + 6)], dsT_cast[((i_21 * 8) + 7)]));
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        float broadcast_var_16 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(dq_v0 + (i_22 * 4)) = make_float4(broadcast_var_16, broadcast_var_16, broadcast_var_16, broadcast_var_16);
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
      for (int i_23 = 0; i_23 < 8; ++i_23) {
        AtomicAdd((&(dQ[(((((((((k * 65536) + (((((int)threadIdx.x) & 63) >> 5) * 16384)) + (((i_23 & 3) >> 1) * 8192)) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) >> 6) * 128)) + ((i_23 >> 2) * 64)) + ((i_23 & 1) * 32)) + (((int)threadIdx.x) & 31)) + 32768)])), dq_v1[i_23]);
      }
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 4; ++i_24) {
      float broadcast_var_17 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(qkT_v1 + (i_24 * 4)) = make_float4(broadcast_var_17, broadcast_var_17, broadcast_var_17, broadcast_var_17);
    }
    overlap_plan_mbar[3].wait(1);
    {
      tl::GmmaDescriptor desc_a_6;
      tl::GmmaDescriptor desc_b_11;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_6, (&(((half_t*)K_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_11, (&(((half_t*)q)[0])));
      tl::increase_descriptor_offset<int>(desc_b_11, 4096);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT_v1 + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_14 = 0; ki_14 < 4; ++ki_14) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_6 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_14 * 32)) >> 4)), uint64_t(desc_b_11 + ((ki_14 * 32) >> 4)), ((uint32_t*)(qkT_v1 + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT_v1 + 0), 32);
    }
    overlap_plan_mbar[5].wait(1);
    #pragma unroll
    for (int i_25 = 0; i_25 < 8; ++i_25) {
      float broadcast_var_18 = 0x1.7154764ee6c2fp-3f/*1.803369e-01*/;
      float2 __24;
      float2 __25;
        float2 __26;
          float2 v__23 = *(float2*)(qkT_v1 + (i_25 * 2));
          float2 v__24 = make_float2(broadcast_var_18, broadcast_var_18);
          __26.x = (v__23.x*v__24.x);
          __26.y = (v__23.y*v__24.y);
        float2 v__25 = *(float2*)(((float*)lse_shared) + ((((i_25 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 32));
        __25.x = (__26.x-v__25.x);
        __25.y = (__26.y-v__25.y);
      __24.x = exp2f(__25.x);
      __24.y = exp2f(__25.y);
      *(float2*)(qkT_v1 + (i_25 * 2)) = __24;
    }
    overlap_plan_mbar[16].arrive();
    #pragma unroll
    for (int i_26 = 0; i_26 < 4; ++i_26) {
      float broadcast_var_19 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dsT + (i_26 * 4)) = make_float4(broadcast_var_19, broadcast_var_19, broadcast_var_19, broadcast_var_19);
    }
    overlap_plan_mbar[7].wait(1);
    {
      tl::GmmaDescriptor desc_a_7;
      tl::GmmaDescriptor desc_b_12;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_7, (&(((half_t*)V_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_12, (&(((half_t*)do_1)[0])));
      tl::increase_descriptor_offset<int>(desc_b_12, 4096);
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
    #pragma unroll
    for (int i_27 = 0; i_27 < 4; ++i_27) {
      uint2 __27;
      float4 v__26 = *(float4*)(qkT_v0 + (i_27 * 4));
      ((half2*)(&__27))[0] = __float22half2_rn(((float2*)(&v__26))[0]);
      ((half2*)(&__27))[1] = __float22half2_rn(((float2*)(&v__26))[1]);
      *(uint2*)(qkT_cast + (i_27 * 4)) = __27;
    }
    overlap_plan_mbar[6].wait(1);
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
    overlap_plan_mbar[17].arrive();
    overlap_plan_mbar[9].wait(1);
    #pragma unroll
    for (int i_28 = 0; i_28 < 8; ++i_28) {
      float delta_local_cast_3[2];
      *(float2*)(delta_local_cast_3 + 0) = *(float2*)(((float*)delta) + ((((i_28 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 32));
      float broadcast_var_20 = 0x1p-3f/*1.250000e-01*/;
      uint1 __28;
      float2 __29;
        float2 __30;
          float2 v__27 = *(float2*)(qkT_v1 + (i_28 * 2));
          float2 __31;
            float2 v__28 = *(float2*)(dsT + (i_28 * 2));
            float2 v__29 = *(float2*)(delta_local_cast_3 + 0);
            __31.x = (v__28.x-v__29.x);
            __31.y = (v__28.y-v__29.y);
          __30.x = (v__27.x*__31.x);
          __30.y = (v__27.y*__31.y);
        float2 v__30 = make_float2(broadcast_var_20, broadcast_var_20);
        __29.x = (__30.x*v__30.x);
        __29.y = (__30.y*v__30.y);
      ((half2*)(&__28))[0] = __float22half2_rn(((float2*)(&__29))[0]);
      *(uint1*)(dsT_cast + (i_28 * 2)) = __28;
    }
    overlap_plan_mbar[20].arrive();
    {
      tl::GmmaDescriptor desc_b_14;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_14, (&(((half_t*)q)[0])));
      tl::increase_descriptor_offset<int>(desc_b_14, 4096);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast + 0), 8);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_17 = 0; ki_17 < 2; ++ki_17) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(dsT_cast + (ki_17 * 8)), uint64_t(desc_b_14 + ((ki_17 * 2048) >> 4)), reinterpret_cast<uint32_t*>(dk + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast + 0), 8);
    }
    overlap_plan_mbar[14].arrive();
    tl::__sync_thread_partial(3, 256);
    #pragma unroll
    for (int i_29 = 0; i_29 < 2; ++i_29) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)dsT_shared)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_29) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(dsT_cast[(i_29 * 8)], dsT_cast[((i_29 * 8) + 1)]), __pack_half2(dsT_cast[((i_29 * 8) + 2)], dsT_cast[((i_29 * 8) + 3)]), __pack_half2(dsT_cast[((i_29 * 8) + 4)], dsT_cast[((i_29 * 8) + 5)]), __pack_half2(dsT_cast[((i_29 * 8) + 6)], dsT_cast[((i_29 * 8) + 7)]));
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 2; ++i_30) {
      float broadcast_var_21 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dq_v1 + (i_30 * 4)) = make_float4(broadcast_var_21, broadcast_var_21, broadcast_var_21, broadcast_var_21);
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
    for (int i_31 = 0; i_31 < 8; ++i_31) {
      AtomicAdd((&(dQ[((((((((((((int)threadIdx.x) & 63) >> 5) * 16384) + (((i_31 & 3) >> 1) * 8192)) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) >> 6) * 128)) + ((i_31 >> 2) * 64)) + ((i_31 & 1) * 32)) + (((int)threadIdx.x) & 31)) + 983040)])), dq_v0[i_31]);
    }
    #pragma unroll
    for (int i_32 = 0; i_32 < 4; ++i_32) {
      uint2 __32;
      float4 v__31 = *(float4*)(qkT_v1 + (i_32 * 4));
      ((half2*)(&__32))[0] = __float22half2_rn(((float2*)(&v__31))[0]);
      ((half2*)(&__32))[1] = __float22half2_rn(((float2*)(&v__31))[1]);
      *(uint2*)(qkT_cast + (i_32 * 4)) = __32;
    }
    overlap_plan_mbar[7].wait(1);
    {
      tl::GmmaDescriptor desc_b_15;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_15, (&(((half_t*)do_1)[0])));
      tl::increase_descriptor_offset<int>(desc_b_15, 4096);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(qkT_cast + 0), 8);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dv + 0), 64);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_19 = 0; ki_19 < 2; ++ki_19) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(qkT_cast + (ki_19 * 8)), uint64_t(desc_b_15 + ((ki_19 * 2048) >> 4)), reinterpret_cast<uint32_t*>(dv + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dv + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(qkT_cast + 0), 8);
    }
    overlap_plan_mbar[18].arrive();
    #pragma unroll
    for (int i_33 = 0; i_33 < 8; ++i_33) {
      AtomicAdd((&(dQ[((((((((((((int)threadIdx.x) & 63) >> 5) * 16384) + (((i_33 & 3) >> 1) * 8192)) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) >> 6) * 128)) + ((i_33 >> 2) * 64)) + ((i_33 & 1) * 32)) + (((int)threadIdx.x) & 31)) + 1015808)])), dq_v1[i_33]);
    }
    tl::__sync_thread_partial(3, 256);
    #pragma unroll
    for (int i_34 = 0; i_34 < 4; ++i_34) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)gradient_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_34 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_34 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)dv[(i_34 * 8)]), ((half_t)dv[((i_34 * 8) + 1)])), __pack_half2(((half_t)dv[((i_34 * 8) + 2)]), ((half_t)dv[((i_34 * 8) + 3)])), __pack_half2(((half_t)dv[((i_34 * 8) + 4)]), ((half_t)dv[((i_34 * 8) + 5)])), __pack_half2(((half_t)dv[((i_34 * 8) + 6)]), ((half_t)dv[((i_34 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[10].arrive();
    overlap_plan_mbar[11].wait(0);
    tl::__sync_thread_partial(3, 256);
    #pragma unroll
    for (int i_35 = 0; i_35 < 4; ++i_35) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)gradient_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_35 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_35 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)dk[(i_35 * 8)]), ((half_t)dk[((i_35 * 8) + 1)])), __pack_half2(((half_t)dk[((i_35 * 8) + 2)]), ((half_t)dk[((i_35 * 8) + 3)])), __pack_half2(((half_t)dk[((i_35 * 8) + 4)]), ((half_t)dk[((i_35 * 8) + 5)])), __pack_half2(((half_t)dk[((i_35 * 8) + 6)]), ((half_t)dk[((i_35 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[12].arrive();
  } else {
    tl::warpgroup_reg_dealloc<24>();
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[0], (&(((half_t*)K_shared)[0])), 0, (((int)blockIdx.y) * 128), ((int)blockIdx.x), 0);
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[1], (&(((half_t*)V_shared)[0])), 0, (((int)blockIdx.y) * 128), ((int)blockIdx.x), 0);
    }
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      if (2 <= k_1) {
        overlap_plan_mbar[((k_1 & 1) + 13)].wait((((k_1 >> 1) + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[((k_1 & 1) + 2)].arrive_and_expect_tx(4096);
        tl::tma_load(Q_desc, overlap_plan_mbar[((k_1 & 1) + 2)], (&(((half_t*)q)[((k_1 & 1) * 2048)])), 0, (k_1 * 32), ((int)blockIdx.x), 0);
      }
      if (2 <= k_1) {
        overlap_plan_mbar[((k_1 & 1) + 15)].wait((((k_1 >> 1) + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[((k_1 & 1) + 4)].arrive_and_expect_tx(128);
        tl::tma_load((&(((float*)lse_shared)[((k_1 & 1) * 32)])), (&(lse[((((int)blockIdx.x) * 1024) + (k_1 * 32))])), overlap_plan_mbar[((k_1 & 1) + 4)], 128);
      }
      if (2 <= k_1) {
        overlap_plan_mbar[((k_1 & 1) + 17)].wait((((k_1 >> 1) + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[((k_1 & 1) + 6)].arrive_and_expect_tx(4096);
        tl::tma_load(dO_desc, overlap_plan_mbar[((k_1 & 1) + 6)], (&(((half_t*)do_1)[((k_1 & 1) * 2048)])), 0, (k_1 * 32), ((int)blockIdx.x), 0);
      }
      if (2 <= k_1) {
        overlap_plan_mbar[((k_1 & 1) + 19)].wait((((k_1 >> 1) + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[((k_1 & 1) + 8)].arrive_and_expect_tx(128);
        tl::tma_load((&(((float*)delta)[((k_1 & 1) * 32)])), (&(Delta[((((int)blockIdx.x) * 1024) + (k_1 * 32))])), overlap_plan_mbar[((k_1 & 1) + 8)], 128);
      }
    }
    overlap_plan_mbar[10].wait(0);
    tl::__sync_thread_partial(4, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store(dV_desc, (&(((half_t*)gradient_shared)[0])), 0, (((int)blockIdx.y) * 128), ((int)blockIdx.x), 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[11].arrive();
    overlap_plan_mbar[12].wait(0);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store(dK_desc, (&(((half_t*)gradient_shared)[0])), 0, (((int)blockIdx.y) * 128), ((int)blockIdx.x), 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

