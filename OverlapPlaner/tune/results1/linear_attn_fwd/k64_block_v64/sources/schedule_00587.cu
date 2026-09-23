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
extern "C" __global__ void __launch_bounds__(512, 1) main_kernel(float* __restrict__ FinalState, __grid_constant__ const CUtensorMap K_desc, float* __restrict__ O, const half_t* __restrict__ Q, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* o_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* h_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* k = ((void*)((char*)buf_dyn_shmem + 49152));
  void* q = ((void*)((char*)buf_dyn_shmem + 65536));
  void* s_shared = ((void*)((char*)buf_dyn_shmem + 81920));
  void* v = ((void*)((char*)buf_dyn_shmem + 98304));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[26];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float h[32];
  float s[32];
  float o[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(128);
    overlap_plan_mbar[1].init(128);
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
    overlap_plan_mbar[12].init(128);
    overlap_plan_mbar[13].init(128);
    overlap_plan_mbar[14].init(128);
    overlap_plan_mbar[15].init(128);
    overlap_plan_mbar[16].init(128);
    overlap_plan_mbar[17].init(128);
    overlap_plan_mbar[18].init(128);
    overlap_plan_mbar[19].init(128);
    overlap_plan_mbar[20].init(128);
    overlap_plan_mbar[21].init(128);
    overlap_plan_mbar[22].init(128);
    overlap_plan_mbar[23].init(128);
    overlap_plan_mbar[24].init(256);
    overlap_plan_mbar[25].init(256);
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
      overlap_plan_mbar[2].arrive_and_expect_tx(8192);
      tl::tma_load(K_desc, overlap_plan_mbar[2], (&(((half_t*)k)[0])), (((int)blockIdx.y) * 64), 0, (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
      overlap_plan_mbar[4].arrive_and_expect_tx(8192);
      tl::tma_load(V_desc, overlap_plan_mbar[4], (&(((half_t*)v)[0])), (((int)blockIdx.x) * 64), 0, (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
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
    tl::fence_proxy_async();
    overlap_plan_mbar[0].arrive();
    overlap_plan_mbar[2].wait(0);
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)q)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)k)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
      tl::warpgroup_arrive();
      tl::__sync_thread_partial(3, 128);
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
    tl::fence_proxy_async();
    overlap_plan_mbar[6].arrive();
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      half_t h_shared_local_cast_2[2];
      uint1 __4;
      float2 v__2 = *(float2*)(h + (i_3 * 2));
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
      *(uint1*)(h_shared_local_cast_2 + 0) = __4;
      *(uint1*)(((half_t*)h_shared) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_3 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_3 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_3 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_3 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(h_shared_local_cast_2 + 0);
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[8].arrive();
    for (int ic = 0; ic < 15; ++ic) {
      if (1 <= ic) {
        overlap_plan_mbar[(((ic + 1) & 1) + 14)].wait((((ic + 3) & 3) >> 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[(((ic + 1) & 1) + 2)].arrive_and_expect_tx(8192);
        tl::tma_load(K_desc, overlap_plan_mbar[(((ic + 1) & 1) + 2)], (&(((half_t*)k)[(((ic + 1) & 1) * 4096)])), (((int)blockIdx.y) * 64), ((ic * 64) + 64), (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
      }
      if (1 <= ic) {
        overlap_plan_mbar[(((ic + 1) & 1) + 16)].wait((((ic + 3) & 3) >> 1));
        overlap_plan_mbar[(((ic + 1) & 1) + 18)].wait((((ic + 3) & 3) >> 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[(((ic + 1) & 1) + 4)].arrive_and_expect_tx(8192);
        tl::tma_load(V_desc, overlap_plan_mbar[(((ic + 1) & 1) + 4)], (&(((half_t*)v)[(((ic + 1) & 1) * 4096)])), (((int)blockIdx.x) * 64), ((ic * 64) + 64), (((int)blockIdx.z) & 31), (((int)blockIdx.z) >> 5));
      }
      if (1 <= ic) {
        overlap_plan_mbar[(((ic + 1) & 1) + 12)].wait((((ic + 3) & 3) >> 1));
      }
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        half_t Q_local_cast_4[8];
        half_t q_local_cast_3[8];
        *(uint4*)(Q_local_cast_4 + 0) = *(uint4*)(Q + (((((((((((int)blockIdx.z) >> 5) * 4194304) + (ic * 262144)) + (i_4 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 262144));
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
        *(uint4*)(((half_t*)q) + ((((((((ic + 1) & 1) * 4096) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(q_local_cast_3 + 0);
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[((ic + 1) & 1)].arrive();
      overlap_plan_mbar[(((ic + 1) & 1) + 2)].wait((((ic + 1) & 3) >> 1));
      {
        tl::GmmaDescriptor desc_a_1;
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)q)[0])));
        tl::increase_descriptor_offset<int>(desc_a_1, (((ic + 1) & 1) * 8192));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((half_t*)k)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, (((ic + 1) & 1) * 8192));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
        tl::warpgroup_arrive();
        tl::__sync_thread_partial(3, 128);
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_1 + ((ki_1 * 32) >> 4)), uint64_t(desc_b_1 + ((ki_1 * 32) >> 4)), ((uint32_t*)(s + 0)), ((0 < ki_1) ? 1 : 0));
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
      }
      if (1 <= ic) {
        overlap_plan_mbar[(((ic + 1) & 1) + 20)].wait((((ic + 3) & 3) >> 1));
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 32; ++i_5) {
        float condval_1;
        if ((((((i_5 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_5 & 1)) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_5 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval_1 = s[i_5];
        } else {
          condval_1 = 0x0p+0f/*0.000000e+00*/;
        }
        ((half_t*)s_shared)[(((((((((((ic + 1) & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((i_5 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((i_5 >> 4) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_5 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_5 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_5 & 1))] = ((half_t)condval_1);
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[(((ic + 1) & 1) + 6)].arrive();
      overlap_plan_mbar[((ic & 1) + 2)].wait(((ic & 3) >> 1));
      overlap_plan_mbar[((ic & 1) + 4)].wait(((ic & 3) >> 1));
      {
        tl::GmmaDescriptor desc_a_2;
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_2, (&(((half_t*)k)[0])));
        tl::increase_descriptor_offset<int>(desc_a_2, ((ic & 1) * 8192));
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_2, (&(((half_t*)v)[0])));
        tl::increase_descriptor_offset<int>(desc_b_2, ((ic & 1) * 8192));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_2 + ((ki_2 * 2048) >> 4)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), ((uint32_t*)(h + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
      }
      overlap_plan_mbar[((ic & 1) + 14)].arrive();
      overlap_plan_mbar[((ic & 1) + 16)].arrive();
      if (1 <= ic) {
        overlap_plan_mbar[(((ic + 1) & 1) + 22)].wait((((ic + 3) & 3) >> 1));
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 16; ++i_6) {
        half_t h_shared_local_cast_5[2];
        uint1 __8;
        float2 v__5 = *(float2*)(h + (i_6 * 2));
        ((half2*)(&__8))[0] = __float22half2_rn(((float2*)(&v__5))[0]);
        *(uint1*)(h_shared_local_cast_5 + 0) = __8;
        *(uint1*)(((half_t*)h_shared) + ((((((((((ic + 1) & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_6 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_6 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_6 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_6 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(h_shared_local_cast_5 + 0);
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[(((ic + 1) & 1) + 8)].arrive();
    }
    overlap_plan_mbar[3].wait(1);
    overlap_plan_mbar[5].wait(1);
    {
      tl::GmmaDescriptor desc_a_3;
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_3, (&(((half_t*)k)[0])));
      tl::increase_descriptor_offset<int>(desc_a_3, 8192);
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_3, (&(((half_t*)v)[0])));
      tl::increase_descriptor_offset<int>(desc_b_3, 8192);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_3 + ((ki_3 * 2048) >> 4)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), ((uint32_t*)(h + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
    }
    overlap_plan_mbar[15].arrive();
    overlap_plan_mbar[17].arrive();
    #pragma unroll
    for (int i_7 = 0; i_7 < 16; ++i_7) {
      *(float2*)(FinalState + ((((((((((int)blockIdx.z) * 16384) + (((int)blockIdx.y) * 8192)) + ((((int)threadIdx.x) >> 5) * 2048)) + ((i_7 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + (((int)blockIdx.x) * 64)) + ((i_7 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(h + (i_7 * 2));
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<216>();
      const dim3 blockIdx = tl::rasterization2DRow<10>();
      for (int ic_1 = 0; ic_1 < 16; ++ic_1) {
        overlap_plan_mbar[((ic_1 & 1) + 4)].wait(((ic_1 & 3) >> 1));
        overlap_plan_mbar[((ic_1 & 1) + 6)].wait(((ic_1 & 3) >> 1));
        {
          tl::GmmaDescriptor desc_a_4;
          tl::GmmaDescriptor desc_b_4;
          tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_4, (&(((half_t*)s_shared)[0])));
          tl::increase_descriptor_offset<int>(desc_a_4, ((ic_1 & 1) * 8192));
          tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_4, (&(((half_t*)v)[0])));
          tl::increase_descriptor_offset<int>(desc_b_4, ((ic_1 & 1) * 8192));
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
          tl::warpgroup_arrive();
          #pragma unroll
          for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
            tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_4 + ((ki_4 * 32) >> 4)), uint64_t(desc_b_4 + ((ki_4 * 2048) >> 4)), ((uint32_t*)(o + 0)), ((0 < ki_4) ? 1 : 0));
          }
          tl::warpgroup_commit_batch();
          tl::warpgroup_wait<0>();
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
        }
        overlap_plan_mbar[((ic_1 & 1) + 18)].arrive();
        overlap_plan_mbar[((ic_1 & 1) + 20)].arrive();
        overlap_plan_mbar[(ic_1 & 1)].wait(((ic_1 & 3) >> 1));
        overlap_plan_mbar[((ic_1 & 1) + 8)].wait(((ic_1 & 3) >> 1));
        {
          tl::GmmaDescriptor desc_a_5;
          tl::GmmaDescriptor desc_b_5;
          tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_5, (&(((half_t*)q)[0])));
          tl::increase_descriptor_offset<int>(desc_a_5, ((ic_1 & 1) * 8192));
          tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_5, (&(((half_t*)h_shared)[0])));
          tl::increase_descriptor_offset<int>(desc_b_5, ((ic_1 & 1) * 8192));
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
        overlap_plan_mbar[((ic_1 & 1) + 12)].arrive();
        overlap_plan_mbar[((ic_1 & 1) + 22)].arrive();
        if (2 <= ic_1) {
          overlap_plan_mbar[((ic_1 & 1) + 24)].wait((((ic_1 >> 1) + 1) & 1));
        }
        #pragma unroll
        for (int i_8 = 0; i_8 < 16; ++i_8) {
          *(float2*)(((float*)o_shared) + ((((((((ic_1 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_8 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_8 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096)) = *(float2*)(o + (i_8 * 2));
        }
        tl::fence_proxy_async();
        overlap_plan_mbar[((ic_1 & 1) + 10)].arrive();
      }
    } else {
      tl::warpgroup_reg_dealloc<24>();
      const dim3 blockIdx = tl::rasterization2DRow<10>();
      for (int ic_2 = 0; ic_2 < 16; ++ic_2) {
        overlap_plan_mbar[((ic_2 & 1) + 10)].wait(((ic_2 & 3) >> 1));
        #pragma unroll
        for (int i_9 = 0; i_9 < 4; ++i_9) {
          AtomicAddx4((&(O[(((((((((((int)blockIdx.z) >> 5) * 4194304) + (ic_2 * 262144)) + (i_9 * 65536)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4)) - 65536)])), *(float4*)(((float*)o_shared) + (((((ic_2 & 1) * 4096) + (i_9 * 1024)) + (((int)threadIdx.x) * 4)) - 1024)));
        }
        overlap_plan_mbar[((ic_2 & 1) + 24)].arrive();
      }
    }
  }
}

