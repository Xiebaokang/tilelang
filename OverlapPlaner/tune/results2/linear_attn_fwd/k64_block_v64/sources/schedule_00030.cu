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
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(float* __restrict__ FinalState, __grid_constant__ const CUtensorMap K_desc, float* __restrict__ O, const half_t* __restrict__ Q, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* o_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* h_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* k = ((void*)((char*)buf_dyn_shmem + 40960));
  void* q = ((void*)((char*)buf_dyn_shmem + 49152));
  void* s_shared = ((void*)((char*)buf_dyn_shmem + 57344));
  void* v = ((void*)((char*)buf_dyn_shmem + 65536));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[8];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float h[32];
  float s[32];
  float o_v0[32];
  float o_v1[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(128);
    overlap_plan_mbar[3].init(128);
    overlap_plan_mbar[4].init(128);
    overlap_plan_mbar[5].init(128);
    overlap_plan_mbar[6].init(128);
    overlap_plan_mbar[7].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<240>();
    const dim3 blockIdx = tl::rasterization2DRow<10>();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(h + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(8192);
      tl::tma_load(K_desc, overlap_plan_mbar[0], (&(((half_t*)k)[0])), (((int)blockIdx.y) * 64), 0, (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
      overlap_plan_mbar[1].arrive_and_expect_tx(8192);
      tl::tma_load(V_desc, overlap_plan_mbar[1], (&(((half_t*)v)[0])), (((int)blockIdx.x) * 64), 0, (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)h_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)h[(i_1 * 8)]), ((half_t)h[((i_1 * 8) + 1)])), __pack_half2(((half_t)h[((i_1 * 8) + 2)]), ((half_t)h[((i_1 * 8) + 3)])), __pack_half2(((half_t)h[((i_1 * 8) + 4)]), ((half_t)h[((i_1 * 8) + 5)])), __pack_half2(((half_t)h[((i_1 * 8) + 6)]), ((half_t)h[((i_1 * 8) + 7)])));
    }
    overlap_plan_mbar[0].wait(0);
    overlap_plan_mbar[1].wait(0);
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a, (&(((half_t*)k)[0])));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b, (&(((half_t*)v)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a + ((ki * 2048) >> 4)), uint64_t(desc_b + ((ki * 2048) >> 4)), ((uint32_t*)(h + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      half_t Q_local_cast_1[8];
      half_t q_local_cast[8];
      *(uint4*)(Q_local_cast_1 + 0) = *(uint4*)(Q + (((((((((int)blockIdx.z) >> 5) * 4194304) + (i_2 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
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
      *(uint4*)(((half_t*)q) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(q_local_cast + 0);
    }
    tl::cp_async_commit();
    tl::cp_async_wait<0>();
    tl::__sync_thread_partial(3, 128);
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_1;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)q)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((half_t*)k)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_1 + ((ki_1 * 32) >> 4)), uint64_t(desc_b_1 + ((ki_1 * 32) >> 4)), ((uint32_t*)(s + 0)), ((0 < ki_1) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
    }
    overlap_plan_mbar[4].arrive();
    #pragma unroll
    for (int i_3 = 0; i_3 < 32; ++i_3) {
      float condval;
      if ((((((i_3 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_3 & 1)) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_3 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval = s[i_3];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      ((half_t*)s_shared)[(((((((((((int)threadIdx.x) >> 5) * 1024) + (((i_3 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((i_3 >> 4) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_3 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_3 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_3 & 1))] = ((half_t)condval);
    }
    {
      tl::GmmaDescriptor desc_a_2;
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)s_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_2, (&(((half_t*)v)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v0 + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_2 + ((ki_2 * 32) >> 4)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), ((uint32_t*)(o_v0 + 0)), ((0 < ki_2) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v0 + 0), 32);
    }
    overlap_plan_mbar[5].arrive();
    {
      tl::GmmaDescriptor desc_a_3;
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((half_t*)q)[0])));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_3, (&(((half_t*)h_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v0 + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_3 + ((ki_3 * 32) >> 4)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), ((uint32_t*)(o_v0 + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v0 + 0), 32);
    }
    for (int ic = 0; ic < 7; ++ic) {
      overlap_plan_mbar[4].wait(0);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[0].arrive_and_expect_tx(8192);
        tl::tma_load(K_desc, overlap_plan_mbar[0], (&(((half_t*)k)[0])), (((int)blockIdx.y) * 64), ((ic * 128) + 64), (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
      }
      overlap_plan_mbar[5].wait(0);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[1].arrive_and_expect_tx(8192);
        tl::tma_load(V_desc, overlap_plan_mbar[1], (&(((half_t*)v)[0])), (((int)blockIdx.x) * 64), ((ic * 128) + 64), (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
      }
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)h_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)h[(i_4 * 8)]), ((half_t)h[((i_4 * 8) + 1)])), __pack_half2(((half_t)h[((i_4 * 8) + 2)]), ((half_t)h[((i_4 * 8) + 3)])), __pack_half2(((half_t)h[((i_4 * 8) + 4)]), ((half_t)h[((i_4 * 8) + 5)])), __pack_half2(((half_t)h[((i_4 * 8) + 6)]), ((half_t)h[((i_4 * 8) + 7)])));
      }
      overlap_plan_mbar[0].wait(1);
      overlap_plan_mbar[1].wait(1);
      {
        tl::GmmaDescriptor desc_a_4;
        tl::GmmaDescriptor desc_b_4;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_4, (&(((half_t*)k)[0])));
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_4, (&(((half_t*)v)[0])));
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
      #pragma unroll
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        half_t Q_local_cast_3[8];
        half_t q_local_cast_2[8];
        *(uint4*)(Q_local_cast_3 + 0) = *(uint4*)(Q + (((((((((((int)blockIdx.z) >> 5) * 4194304) + (ic * 524288)) + (i_5 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 262144));
        for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
          float broadcast_var_2 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
          uint2 __4;
          float4 __5;
            float4 __6;
            uint2 v__2 = *(uint2*)(Q_local_cast_3 + (vec_1 * 4));
            ((float2*)(&__6))[0] = __half22float2(((half2*)(&v__2))[0]);
            ((float2*)(&__6))[1] = __half22float2(((half2*)(&v__2))[1]);
            float4 v__3 = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
            __5.x = (__6.x*v__3.x);
            __5.y = (__6.y*v__3.y);
            __5.z = (__6.z*v__3.z);
            __5.w = (__6.w*v__3.w);
          ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&__5))[0]);
          ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&__5))[1]);
          *(uint2*)(q_local_cast_2 + (vec_1 * 4)) = __4;
        }
        *(uint4*)(((half_t*)q) + (((((i_5 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(q_local_cast_2 + 0);
      }
      tl::cp_async_commit();
      tl::cp_async_wait<1>();
      tl::__sync_thread_partial(3, 128);
      {
        tl::GmmaDescriptor desc_a_5;
        tl::GmmaDescriptor desc_b_5;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_5, (&(((half_t*)q)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_5, (&(((half_t*)k)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_5 + ((ki_5 * 32) >> 4)), uint64_t(desc_b_5 + ((ki_5 * 32) >> 4)), ((uint32_t*)(s + 0)), ((0 < ki_5) ? 1 : 0));
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
      }
      overlap_plan_mbar[4].arrive();
      #pragma unroll
      for (int i_6 = 0; i_6 < 32; ++i_6) {
        float condval_1;
        if ((((((i_6 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_6 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval_1 = s[i_6];
        } else {
          condval_1 = 0x0p+0f/*0.000000e+00*/;
        }
        ((half_t*)s_shared)[(((((((((((int)threadIdx.x) >> 5) * 1024) + (((i_6 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((i_6 >> 4) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_6 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_6 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1))] = ((half_t)condval_1);
      }
      {
        tl::GmmaDescriptor desc_a_6;
        tl::GmmaDescriptor desc_b_6;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_6, (&(((half_t*)s_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_6, (&(((half_t*)v)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v1 + 0), 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        tl::__sync_thread_partial(3, 128);
        #pragma unroll
        for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_6 + ((ki_6 * 32) >> 4)), uint64_t(desc_b_6 + ((ki_6 * 2048) >> 4)), ((uint32_t*)(o_v1 + 0)), ((0 < ki_6) ? 1 : 0));
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v1 + 0), 32);
      }
      overlap_plan_mbar[5].arrive();
      {
        tl::GmmaDescriptor desc_a_7;
        tl::GmmaDescriptor desc_b_7;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_7, (&(((half_t*)q)[0])));
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_7, (&(((half_t*)h_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v1 + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_7 + ((ki_7 * 32) >> 4)), uint64_t(desc_b_7 + ((ki_7 * 2048) >> 4)), ((uint32_t*)(o_v1 + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v1 + 0), 32);
      }
      if (1 <= ic) {
        overlap_plan_mbar[6].wait(((ic + 1) & 1));
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 16; ++i_7) {
        *(float2*)(((float*)o_shared) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_7 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_7 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(o_v0 + (i_7 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[2].arrive();
      overlap_plan_mbar[4].wait(1);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[0].arrive_and_expect_tx(8192);
        tl::tma_load(K_desc, overlap_plan_mbar[0], (&(((half_t*)k)[0])), (((int)blockIdx.y) * 64), ((ic * 128) + 128), (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
      }
      overlap_plan_mbar[5].wait(1);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[1].arrive_and_expect_tx(8192);
        tl::tma_load(V_desc, overlap_plan_mbar[1], (&(((half_t*)v)[0])), (((int)blockIdx.x) * 64), ((ic * 128) + 128), (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)h_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_8 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)h[(i_8 * 8)]), ((half_t)h[((i_8 * 8) + 1)])), __pack_half2(((half_t)h[((i_8 * 8) + 2)]), ((half_t)h[((i_8 * 8) + 3)])), __pack_half2(((half_t)h[((i_8 * 8) + 4)]), ((half_t)h[((i_8 * 8) + 5)])), __pack_half2(((half_t)h[((i_8 * 8) + 6)]), ((half_t)h[((i_8 * 8) + 7)])));
      }
      overlap_plan_mbar[0].wait(0);
      overlap_plan_mbar[1].wait(0);
      {
        tl::GmmaDescriptor desc_a_8;
        tl::GmmaDescriptor desc_b_8;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_8, (&(((half_t*)k)[0])));
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_8, (&(((half_t*)v)[0])));
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
      #pragma unroll
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        half_t Q_local_cast_5[8];
        half_t q_local_cast_4[8];
        *(uint4*)(Q_local_cast_5 + 0) = *(uint4*)(Q + (((((((((((int)blockIdx.z) >> 5) * 4194304) + (ic * 524288)) + (i_9 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 524288));
        for (int vec_2 = 0; vec_2 < 2; ++vec_2) {
          float broadcast_var_3 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
          uint2 __7;
          float4 __8;
            float4 __9;
            uint2 v__4 = *(uint2*)(Q_local_cast_5 + (vec_2 * 4));
            ((float2*)(&__9))[0] = __half22float2(((half2*)(&v__4))[0]);
            ((float2*)(&__9))[1] = __half22float2(((half2*)(&v__4))[1]);
            float4 v__5 = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
            __8.x = (__9.x*v__5.x);
            __8.y = (__9.y*v__5.y);
            __8.z = (__9.z*v__5.z);
            __8.w = (__9.w*v__5.w);
          ((half2*)(&__7))[0] = __float22half2_rn(((float2*)(&__8))[0]);
          ((half2*)(&__7))[1] = __float22half2_rn(((float2*)(&__8))[1]);
          *(uint2*)(q_local_cast_4 + (vec_2 * 4)) = __7;
        }
        *(uint4*)(((half_t*)q) + (((((i_9 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(q_local_cast_4 + 0);
      }
      tl::cp_async_commit();
      tl::cp_async_wait<1>();
      tl::__sync_thread_partial(3, 128);
      {
        tl::GmmaDescriptor desc_a_9;
        tl::GmmaDescriptor desc_b_9;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_9, (&(((half_t*)q)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_9, (&(((half_t*)k)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_9 + ((ki_9 * 32) >> 4)), uint64_t(desc_b_9 + ((ki_9 * 32) >> 4)), ((uint32_t*)(s + 0)), ((0 < ki_9) ? 1 : 0));
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
      }
      overlap_plan_mbar[4].arrive();
      #pragma unroll
      for (int i_10 = 0; i_10 < 32; ++i_10) {
        float condval_2;
        if ((((((i_10 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_10 & 1)) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_10 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval_2 = s[i_10];
        } else {
          condval_2 = 0x0p+0f/*0.000000e+00*/;
        }
        ((half_t*)s_shared)[(((((((((((int)threadIdx.x) >> 5) * 1024) + (((i_10 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((i_10 >> 4) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_10 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_10 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_10 & 1))] = ((half_t)condval_2);
      }
      {
        tl::GmmaDescriptor desc_a_10;
        tl::GmmaDescriptor desc_b_10;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_10, (&(((half_t*)s_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_10, (&(((half_t*)v)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v0 + 0), 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        tl::__sync_thread_partial(3, 128);
        #pragma unroll
        for (int ki_10 = 0; ki_10 < 4; ++ki_10) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_10 + ((ki_10 * 32) >> 4)), uint64_t(desc_b_10 + ((ki_10 * 2048) >> 4)), ((uint32_t*)(o_v0 + 0)), ((0 < ki_10) ? 1 : 0));
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v0 + 0), 32);
      }
      overlap_plan_mbar[5].arrive();
      {
        tl::GmmaDescriptor desc_a_11;
        tl::GmmaDescriptor desc_b_11;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_11, (&(((half_t*)q)[0])));
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_11, (&(((half_t*)h_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v0 + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_11 = 0; ki_11 < 4; ++ki_11) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_11 + ((ki_11 * 32) >> 4)), uint64_t(desc_b_11 + ((ki_11 * 2048) >> 4)), ((uint32_t*)(o_v0 + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v0 + 0), 32);
      }
      if (1 <= ic) {
        overlap_plan_mbar[7].wait(((ic + 1) & 1));
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 16; ++i_11) {
        *(float2*)(((float*)o_shared) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_11 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_11 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096)) = *(float2*)(o_v1 + (i_11 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[3].arrive();
    }
    overlap_plan_mbar[4].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(8192);
      tl::tma_load(K_desc, overlap_plan_mbar[0], (&(((half_t*)k)[0])), (((int)blockIdx.y) * 64), 960, (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
    }
    overlap_plan_mbar[5].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(8192);
      tl::tma_load(V_desc, overlap_plan_mbar[1], (&(((half_t*)v)[0])), (((int)blockIdx.x) * 64), 960, (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)h_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_12 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_12 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)h[(i_12 * 8)]), ((half_t)h[((i_12 * 8) + 1)])), __pack_half2(((half_t)h[((i_12 * 8) + 2)]), ((half_t)h[((i_12 * 8) + 3)])), __pack_half2(((half_t)h[((i_12 * 8) + 4)]), ((half_t)h[((i_12 * 8) + 5)])), __pack_half2(((half_t)h[((i_12 * 8) + 6)]), ((half_t)h[((i_12 * 8) + 7)])));
    }
    overlap_plan_mbar[0].wait(1);
    overlap_plan_mbar[1].wait(1);
    {
      tl::GmmaDescriptor desc_a_12;
      tl::GmmaDescriptor desc_b_12;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_12, (&(((half_t*)k)[0])));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_12, (&(((half_t*)v)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_12 = 0; ki_12 < 4; ++ki_12) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_12 + ((ki_12 * 2048) >> 4)), uint64_t(desc_b_12 + ((ki_12 * 2048) >> 4)), ((uint32_t*)(h + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      half_t Q_local_cast_7[8];
      half_t q_local_cast_6[8];
      *(uint4*)(Q_local_cast_7 + 0) = *(uint4*)(Q + ((((((((((int)blockIdx.z) >> 5) * 4194304) + (i_13 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 3932160));
      for (int vec_3 = 0; vec_3 < 2; ++vec_3) {
        float broadcast_var_4 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
        uint2 __10;
        float4 __11;
          float4 __12;
          uint2 v__6 = *(uint2*)(Q_local_cast_7 + (vec_3 * 4));
          ((float2*)(&__12))[0] = __half22float2(((half2*)(&v__6))[0]);
          ((float2*)(&__12))[1] = __half22float2(((half2*)(&v__6))[1]);
          float4 v__7 = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
          __11.x = (__12.x*v__7.x);
          __11.y = (__12.y*v__7.y);
          __11.z = (__12.z*v__7.z);
          __11.w = (__12.w*v__7.w);
        ((half2*)(&__10))[0] = __float22half2_rn(((float2*)(&__11))[0]);
        ((half2*)(&__10))[1] = __float22half2_rn(((float2*)(&__11))[1]);
        *(uint2*)(q_local_cast_6 + (vec_3 * 4)) = __10;
      }
      *(uint4*)(((half_t*)q) + (((((i_13 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(q_local_cast_6 + 0);
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    tl::__sync_thread_partial(3, 128);
    {
      tl::GmmaDescriptor desc_a_13;
      tl::GmmaDescriptor desc_b_13;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_13, (&(((half_t*)q)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_13, (&(((half_t*)k)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_13 = 0; ki_13 < 4; ++ki_13) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_13 + ((ki_13 * 32) >> 4)), uint64_t(desc_b_13 + ((ki_13 * 32) >> 4)), ((uint32_t*)(s + 0)), ((0 < ki_13) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
    }
    overlap_plan_mbar[4].arrive();
    #pragma unroll
    for (int i_14 = 0; i_14 < 32; ++i_14) {
      float condval_3;
      if ((((((i_14 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_14 & 1)) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_14 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_3 = s[i_14];
      } else {
        condval_3 = 0x0p+0f/*0.000000e+00*/;
      }
      ((half_t*)s_shared)[(((((((((((int)threadIdx.x) >> 5) * 1024) + (((i_14 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((i_14 >> 4) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_14 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_14 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_14 & 1))] = ((half_t)condval_3);
    }
    {
      tl::GmmaDescriptor desc_a_14;
      tl::GmmaDescriptor desc_b_14;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_14, (&(((half_t*)s_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_14, (&(((half_t*)v)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v1 + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int ki_14 = 0; ki_14 < 4; ++ki_14) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_14 + ((ki_14 * 32) >> 4)), uint64_t(desc_b_14 + ((ki_14 * 2048) >> 4)), ((uint32_t*)(o_v1 + 0)), ((0 < ki_14) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v1 + 0), 32);
    }
    overlap_plan_mbar[5].arrive();
    {
      tl::GmmaDescriptor desc_a_15;
      tl::GmmaDescriptor desc_b_15;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_15, (&(((half_t*)q)[0])));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_15, (&(((half_t*)h_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v1 + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_15 = 0; ki_15 < 4; ++ki_15) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_15 + ((ki_15 * 32) >> 4)), uint64_t(desc_b_15 + ((ki_15 * 2048) >> 4)), ((uint32_t*)(o_v1 + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_v1 + 0), 32);
    }
    overlap_plan_mbar[6].wait(0);
    #pragma unroll
    for (int i_15 = 0; i_15 < 16; ++i_15) {
      *(float2*)(((float*)o_shared) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_15 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_15 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(o_v0 + (i_15 * 2));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[2].arrive();
    overlap_plan_mbar[7].wait(0);
    #pragma unroll
    for (int i_16 = 0; i_16 < 16; ++i_16) {
      *(float2*)(((float*)o_shared) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_16 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_16 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096)) = *(float2*)(o_v1 + (i_16 * 2));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[3].arrive();
    #pragma unroll
    for (int i_17 = 0; i_17 < 16; ++i_17) {
      *(float2*)(FinalState + ((((((((((int)blockIdx.z) * 16384) + (((int)blockIdx.y) * 8192)) + ((((int)threadIdx.x) >> 5) * 2048)) + ((i_17 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + (((int)blockIdx.x) * 64)) + ((i_17 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(h + (i_17 * 2));
    }
  } else {
    tl::warpgroup_reg_dealloc<24>();
    const dim3 blockIdx = tl::rasterization2DRow<10>();
    for (int ic_1 = 0; ic_1 < 16; ++ic_1) {
      overlap_plan_mbar[((ic_1 & 1) + 2)].wait(((ic_1 & 3) >> 1));
      #pragma unroll
      for (int i_18 = 0; i_18 < 8; ++i_18) {
        AtomicAddx4((&(O[(((((((((((int)blockIdx.z) >> 5) * 4194304) + (ic_1 * 262144)) + (i_18 * 32768)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4)) - 32768)])), *(float4*)(((float*)o_shared) + (((((ic_1 & 1) * 4096) + (i_18 * 512)) + (((int)threadIdx.x) * 4)) - 512)));
      }
      overlap_plan_mbar[((ic_1 & 1) + 6)].arrive();
    }
  }
}

