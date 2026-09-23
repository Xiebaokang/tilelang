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
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(float* __restrict__ FinalState, const half_t* __restrict__ K, float* __restrict__ O, const half_t* __restrict__ Q, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(float* __restrict__ FinalState, const half_t* __restrict__ K, float* __restrict__ O, const half_t* __restrict__ Q, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* o_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* h_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* k = ((void*)((char*)buf_dyn_shmem + 24576));
  void* q = ((void*)((char*)buf_dyn_shmem + 32768));
  void* s_shared = ((void*)((char*)buf_dyn_shmem + 40960));
  void* v = ((void*)((char*)buf_dyn_shmem + 49152));
  float h[32];
  float s[32];
  float o[32];
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(h + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int ic = 0; ic < 16; ++ic) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      half_t Q_local_cast_1[8];
      half_t q_local_cast[8];
      *(uint4*)(Q_local_cast_1 + 0) = *(uint4*)(Q + ((((((((((int)blockIdx.z) >> 5) * 4194304) + (ic * 262144)) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
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
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      *(uint4*)(((half_t*)k) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((((((((int)blockIdx.z) >> 5) * 4194304) + (ic * 262144)) + (i_2 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      *(uint4*)(((half_t*)v) + (((((i_3 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + ((((((((((int)blockIdx.z) >> 5) * 4194304) + (ic * 262144)) + (i_3 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    }
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)q)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)k)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      __syncthreads();
      #pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + ((ki * 32) >> 4)), uint64_t(desc_b + ((ki * 32) >> 4)), ((uint32_t*)(s + 0)), ((0 < ki) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(s + 0), 32);
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 32; ++i_4) {
      float condval;
      if ((((((i_4 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1)) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval = s[i_4];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      ((half_t*)s_shared)[(((((((((((int)threadIdx.x) >> 5) * 1024) + (((i_4 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((i_4 >> 4) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_4 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_4 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1))] = ((half_t)condval);
    }
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_1;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)s_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_1, (&(((half_t*)v)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      __syncthreads();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_1 + ((ki_1 * 32) >> 4)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), ((uint32_t*)(o + 0)), ((0 < ki_1) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)h_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)h[(i_5 * 8)]), ((half_t)h[((i_5 * 8) + 1)])), __pack_half2(((half_t)h[((i_5 * 8) + 2)]), ((half_t)h[((i_5 * 8) + 3)])), __pack_half2(((half_t)h[((i_5 * 8) + 4)]), ((half_t)h[((i_5 * 8) + 5)])), __pack_half2(((half_t)h[((i_5 * 8) + 6)]), ((half_t)h[((i_5 * 8) + 7)])));
    }
    {
      tl::GmmaDescriptor desc_a_2;
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_2, (&(((half_t*)k)[0])));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_2, (&(((half_t*)v)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_2 + ((ki_2 * 2048) >> 4)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), ((uint32_t*)(h + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(h + 0), 32);
    }
    {
      tl::GmmaDescriptor desc_a_3;
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((half_t*)q)[0])));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_3, (&(((half_t*)h_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
      tl::warpgroup_arrive();
      __syncthreads();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_3 + ((ki_3 * 32) >> 4)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), ((uint32_t*)(o + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(o + 0), 32);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      *(float2*)(((float*)o_shared) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_6 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_6 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(o + (i_6 * 2));
    }
    __syncthreads();
    #pragma unroll
    for (int i_7 = 0; i_7 < 8; ++i_7) {
      AtomicAddx4((&(O[((((((((((int)blockIdx.z) >> 5) * 4194304) + (ic * 262144)) + (i_7 * 32768)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.z) & 31) * 128)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4))])), *(float4*)(((float*)o_shared) + ((i_7 * 512) + (((int)threadIdx.x) * 4))));
    }
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 16; ++i_8) {
    *(float2*)(FinalState + ((((((((((int)blockIdx.z) * 16384) + (((int)blockIdx.y) * 8192)) + ((((int)threadIdx.x) >> 5) * 2048)) + ((i_8 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + (((int)blockIdx.x) * 64)) + ((i_8 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(h + (i_8 * 2));
  }
}

