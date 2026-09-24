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
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* do_1 = ((void*)((char*)buf_dyn_shmem + 32768));
  void* dsT_shared = ((void*)((char*)buf_dyn_shmem + 40960));
  void* q = ((void*)((char*)buf_dyn_shmem + 49152));
  void* lse_shared = ((void*)((char*)buf_dyn_shmem + 57344));
  void* delta = ((void*)((char*)buf_dyn_shmem + 58368));
  void* dv_shared = ((void*)((char*)buf_dyn_shmem + 59392));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[22];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float dv[32];
  float dk[32];
  float qkT[16];
  float dsT[16];
  half_t qkT_cast[16];
  half_t dsT_cast[16];
  float dq[8];
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
    overlap_plan_mbar[10].init(1);
    overlap_plan_mbar[11].init(256);
    overlap_plan_mbar[12].init(256);
    overlap_plan_mbar[13].init(256);
    overlap_plan_mbar[14].init(256);
    overlap_plan_mbar[15].init(256);
    overlap_plan_mbar[16].init(256);
    overlap_plan_mbar[17].init(256);
    overlap_plan_mbar[18].init(256);
    overlap_plan_mbar[19].init(256);
    overlap_plan_mbar[20].init(256);
    overlap_plan_mbar[21].init(256);
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
    for (int k = 0; k < 32; ++k) {
      #pragma unroll
      for (int i_2 = 0; i_2 < 4; ++i_2) {
        float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(qkT + (i_2 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
      }
      if (k == 0) {
        overlap_plan_mbar[0].wait(0);
      }
      overlap_plan_mbar[((k & 1) + 2)].wait(((k & 3) >> 1));
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)K_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)q)[0])));
        tl::increase_descriptor_offset<int>(desc_b, ((k & 1) * 4096));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT + 0), 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a + ((((((int)threadIdx.x) >> 7) * 8192) + (ki * 32)) >> 4)), uint64_t(desc_b + ((ki * 32) >> 4)), ((uint32_t*)(qkT + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(qkT + 0), 32);
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 4; ++i_3) {
        float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(dsT + (i_3 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
      }
      if (k == 0) {
        overlap_plan_mbar[1].wait(0);
      }
      overlap_plan_mbar[((k & 1) + 4)].wait(((k & 3) >> 1));
      {
        tl::GmmaDescriptor desc_a_1;
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)V_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((half_t*)do_1)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, ((k & 1) * 4096));
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
      overlap_plan_mbar[((k % 3) + 6)].wait(((k % 6) / 3));
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
          float2 v__2 = *(float2*)(((float*)lse_shared) + ((((k % 3) * 32) + ((i_4 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
          __2.x = (__3.x-v__2.x);
          __2.y = (__3.y-v__2.y);
        __1.x = exp2f(__2.x);
        __1.y = exp2f(__2.y);
        *(float2*)(qkT + (i_4 * 2)) = __1;
      }
      overlap_plan_mbar[((k % 3) + 17)].arrive();
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
        tl::increase_descriptor_offset<int>(desc_b_2, ((k & 1) * 4096));
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
      overlap_plan_mbar[((k & 1) + 15)].arrive();
      overlap_plan_mbar[((k & 1) + 9)].wait(((k & 3) >> 1));
      #pragma unroll
      for (int i_6 = 0; i_6 < 8; ++i_6) {
        float delta_local_cast[2];
        *(float2*)(delta_local_cast + 0) = *(float2*)(((float*)delta) + ((((k & 1) * 32) + ((i_6 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
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
        *(uint1*)(dsT_cast + (i_6 * 2)) = __5;
      }
      overlap_plan_mbar[((k & 1) + 20)].arrive();
      {
        tl::GmmaDescriptor desc_b_3;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_3, (&(((half_t*)q)[0])));
        tl::increase_descriptor_offset<int>(desc_b_3, ((k & 1) * 4096));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast + 0), 8);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(dsT_cast + (ki_3 * 8)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(dk + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(dsT_cast + 0), 8);
      }
      overlap_plan_mbar[((k & 1) + 13)].arrive();
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)dsT_shared)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_7) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(dsT_cast[(i_7 * 8)], dsT_cast[((i_7 * 8) + 1)]), __pack_half2(dsT_cast[((i_7 * 8) + 2)], dsT_cast[((i_7 * 8) + 3)]), __pack_half2(dsT_cast[((i_7 * 8) + 4)], dsT_cast[((i_7 * 8) + 5)]), __pack_half2(dsT_cast[((i_7 * 8) + 6)], dsT_cast[((i_7 * 8) + 7)]));
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(dq + (i_8 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
      }
      {
        half_t A_local[8];
        half_t B_local[8];
        tl::__sync_thread_partial(3, 256);
        for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)dsT_shared)[(((((ki_4 * 512) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[0])));
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)K_shared)[(((ki_4 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local[0])));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
        }
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 8; ++i_9) {
        AtomicAdd((&(dQ[((((((((k * 32768) + (((((int)threadIdx.x) & 63) >> 5) * 16384)) + (((i_9 & 3) >> 1) * 8192)) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) >> 6) * 128)) + ((i_9 >> 2) * 64)) + ((i_9 & 1) * 32)) + (((int)threadIdx.x) & 31))])), dq[i_9]);
      }
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)dv_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_10 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)dv[(i_10 * 8)]), ((half_t)dv[((i_10 * 8) + 1)])), __pack_half2(((half_t)dv[((i_10 * 8) + 2)]), ((half_t)dv[((i_10 * 8) + 3)])), __pack_half2(((half_t)dv[((i_10 * 8) + 4)]), ((half_t)dv[((i_10 * 8) + 5)])), __pack_half2(((half_t)dv[((i_10 * 8) + 6)]), ((half_t)dv[((i_10 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[11].arrive();
    tl::__sync_thread_partial(3, 256);
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)dk_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)dk[(i_11 * 8)]), ((half_t)dk[((i_11 * 8) + 1)])), __pack_half2(((half_t)dk[((i_11 * 8) + 2)]), ((half_t)dk[((i_11 * 8) + 3)])), __pack_half2(((half_t)dk[((i_11 * 8) + 4)]), ((half_t)dk[((i_11 * 8) + 5)])), __pack_half2(((half_t)dk[((i_11 * 8) + 6)]), ((half_t)dk[((i_11 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[12].arrive();
  } else {
    tl::warpgroup_reg_dealloc<24>();
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(16384);
      tl::tma_load(K_desc, overlap_plan_mbar[0], (&(((half_t*)K_shared)[0])), 0, (((int)blockIdx.y) * 128), (((int)blockIdx.x) >> 2), 0);
      overlap_plan_mbar[1].arrive_and_expect_tx(16384);
      tl::tma_load(V_desc, overlap_plan_mbar[1], (&(((half_t*)V_shared)[0])), 0, (((int)blockIdx.y) * 128), (((int)blockIdx.x) >> 2), 0);
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
        overlap_plan_mbar[((k_1 & 1) + 4)].arrive_and_expect_tx(4096);
        tl::tma_load(dO_desc, overlap_plan_mbar[((k_1 & 1) + 4)], (&(((half_t*)do_1)[((k_1 & 1) * 2048)])), 0, (k_1 * 32), ((int)blockIdx.x), 0);
      }
      if (3 <= k_1) {
        overlap_plan_mbar[((k_1 % 3) + 17)].wait((((k_1 / 3) + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[((k_1 % 3) + 6)].arrive_and_expect_tx(128);
        tl::tma_load((&(((float*)lse_shared)[((k_1 % 3) * 32)])), (&(lse[((((int)blockIdx.x) * 1024) + (k_1 * 32))])), overlap_plan_mbar[((k_1 % 3) + 6)], 128);
      }
      if (2 <= k_1) {
        overlap_plan_mbar[((k_1 & 1) + 20)].wait((((k_1 >> 1) + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
        overlap_plan_mbar[((k_1 & 1) + 9)].arrive_and_expect_tx(128);
        tl::tma_load((&(((float*)delta)[((k_1 & 1) * 32)])), (&(Delta[((((int)blockIdx.x) * 1024) + (k_1 * 32))])), overlap_plan_mbar[((k_1 & 1) + 9)], 128);
      }
    }
    overlap_plan_mbar[11].wait(0);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store(dV_desc, (&(((half_t*)dv_shared)[0])), 0, (((int)blockIdx.y) * 128), (((int)blockIdx.x) >> 2), 0, (((int)blockIdx.x) & 3));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
    overlap_plan_mbar[12].wait(0);
    tl::__sync_thread_partial(4, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store(dK_desc, (&(((half_t*)dk_shared)[0])), 0, (((int)blockIdx.y) * 128), (((int)blockIdx.x) >> 2), 0, (((int)blockIdx.x) & 3));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

