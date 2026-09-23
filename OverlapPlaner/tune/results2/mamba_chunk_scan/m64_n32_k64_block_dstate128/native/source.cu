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

extern "C" __global__ void main_kernel(const half_t* __restrict__ C, __grid_constant__ const CUtensorMap CB_desc, const half_t* __restrict__ D, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, __grid_constant__ const CUtensorMap Output_desc, const half_t* __restrict__ Prev, const half_t* __restrict__ X, __grid_constant__ const CUtensorMap X_desc);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const half_t* __restrict__ C, __grid_constant__ const CUtensorMap CB_desc, const half_t* __restrict__ D, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, __grid_constant__ const CUtensorMap Output_desc, const half_t* __restrict__ Prev, const half_t* __restrict__ X, __grid_constant__ const CUtensorMap X_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* da_m_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* residual_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* c_shared = ((void*)((char*)buf_dyn_shmem + 1024));
  void* prev_shared = ((void*)((char*)buf_dyn_shmem + 17408));
  void* cb_shared = ((void*)((char*)buf_dyn_shmem + 25600));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 41984));
  void* da_k_shared = ((void*)((char*)buf_dyn_shmem + 50176));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 51200));
  __shared__ __align__(16) uint64_t mbarrier_mem[16];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float da_m_local[2];
  float acc[16];
  float scale_m[2];
  float d_local[1];
  float residual_local[16];
  half_t cb_local[32];
  float da_k_local[16];
  float dt_local[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(CB_desc);
    tl::prefetch_tma_descriptor(X_desc);
    tl::prefetch_tma_descriptor(Output_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(1);
    mbarrier[4].init(1);
    mbarrier[5].init(1);
    mbarrier[6].init(1);
    mbarrier[7].init(1);
    mbarrier[8].init(128);
    mbarrier[9].init(128);
    mbarrier[10].init(128);
    mbarrier[11].init(128);
    mbarrier[12].init(128);
    mbarrier[13].init(128);
    mbarrier[14].init(128);
    mbarrier[15].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)da_m_shared)[((int)threadIdx.x)] = DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + ((((int)blockIdx.y) >> 1) * 64)) + ((int)threadIdx.x))];
  }
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(uint4*)(((half_t*)c_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(C + ((((((((int)blockIdx.z) & 7) * 524288) + ((((int)blockIdx.z) >> 3) * 32768)) + ((((int)blockIdx.y) >> 1) * 8192)) + (i * 2048)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(uint4*)(((half_t*)prev_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Prev + (((((((((int)blockIdx.z) & 7) * 10485760) + ((((int)blockIdx.z) >> 3) * 655360)) + (((int)blockIdx.x) * 8192)) + ((((int)blockIdx.y) & 1) * 4096)) + (i_1 * 2048)) + (((int)threadIdx.x) * 8)));
  }
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::fence_proxy_async();
    for (int ik = 0; ik < ((((int)blockIdx.y) >> 1) + 1); ++ik) {
      mbarrier[((ik & 1) + 8)].wait(((ik >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(ik & 1)].arrive_and_expect_tx(8192);
        tl::tma_load(CB_desc, mbarrier[(ik & 1)], (&(((half_t*)cb_shared)[((ik & 1) * 4096)])), (ik * 64), ((((int)blockIdx.y) >> 1) * 64), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
      }
      mbarrier[((ik & 1) + 10)].wait(((ik >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[((ik & 1) + 2)].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)da_k_shared)[((ik & 1) * 64)])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64))])), mbarrier[((ik & 1) + 2)], 128);
      }
      mbarrier[((ik & 1) + 12)].wait(((ik >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[((ik & 1) + 4)].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)dt_shared)[((ik & 1) * 64)])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64))])), mbarrier[((ik & 1) + 4)], 128);
      }
      mbarrier[((ik & 1) + 14)].wait(((ik >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[((ik & 1) + 6)].arrive_and_expect_tx(4096);
        tl::tma_load(X_desc, mbarrier[((ik & 1) + 6)], (&(((half_t*)x_shared)[((ik & 1) * 2048)])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + (ik * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
    }
  } else {
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      da_m_local[i_2] = ((float)((half_t*)da_m_shared)[(((((((int)threadIdx.x) >> 5) * 16) + (i_2 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)]);
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc + (i_3 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      scale_m[i_4] = exp2f((da_m_local[i_4] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
    }
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)c_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)prev_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 4096) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      acc[i_5] = (acc[i_5] * scale_m[((i_5 & 3) >> 1)]);
    }
    for (int ik_1 = 0; ik_1 < ((((int)blockIdx.y) >> 1) + 1); ++ik_1) {
      mbarrier[(ik_1 & 1)].wait((ik_1 >> 1));
      #pragma unroll
      for (int i_6 = 0; i_6 < 16; ++i_6) {
        *(uint1*)(cb_local + (i_6 * 2)) = *(uint1*)(((half_t*)cb_shared) + ((((((((((ik_1 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_6 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_6 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_6 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_6 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096));
      }
      mbarrier[((ik_1 & 1) + 8)].arrive();
      mbarrier[((ik_1 & 1) + 2)].wait((ik_1 >> 1));
      #pragma unroll
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        half_t da_k_shared_local_cast[2];
        *(uint1*)(da_k_shared_local_cast + 0) = *(uint1*)(((half_t*)da_k_shared) + ((((ik_1 & 1) * 64) + (i_7 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __1;
        uint1 v_ = *(uint1*)(da_k_shared_local_cast + 0);
        ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
        *(float2*)(da_k_local + (i_7 * 2)) = __1;
      }
      mbarrier[((ik_1 & 1) + 10)].arrive();
      #pragma unroll
      for (int i_8 = 0; i_8 < 16; ++i_8) {
        float broadcast_var_1 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
        uint1 __2;
        float2 __3;
          float2 __4;
          uint1 v__1 = *(uint1*)(cb_local + (i_8 * 2));
          ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__1))[0]);
          float2 __5;
          float2 __6;
            float2 __7;
              float2 v__2 = make_float2(da_m_local[(i_8 & 1)], da_m_local[(i_8 & 1)]);
              float2 v__3 = *(float2*)(da_k_local + ((i_8 >> 1) * 2));
              __7.x = (v__2.x-v__3.x);
              __7.y = (v__2.y-v__3.y);
            float2 v__4 = make_float2(broadcast_var_1, broadcast_var_1);
            __6.x = (__7.x*v__4.x);
            __6.y = (__7.y*v__4.y);
          __5.x = exp2f(__6.x);
          __5.y = exp2f(__6.y);
          __3.x = (__4.x*__5.x);
          __3.y = (__4.y*__5.y);
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&__3))[0]);
        *(uint1*)(cb_local + (i_8 * 2)) = __2;
      }
      mbarrier[((ik_1 & 1) + 4)].wait((ik_1 >> 1));
      #pragma unroll
      for (int i_9 = 0; i_9 < 8; ++i_9) {
        half_t dt_shared_local_cast_1[2];
        *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((((ik_1 & 1) * 64) + (i_9 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __8;
        uint1 v__5 = *(uint1*)(dt_shared_local_cast_1 + 0);
        ((float2*)(&__8))[0] = __half22float2(((half2*)(&v__5))[0]);
        *(float2*)(dt_local + (i_9 * 2)) = __8;
      }
      mbarrier[((ik_1 & 1) + 12)].arrive();
      #pragma unroll
      for (int i_10 = 0; i_10 < 16; ++i_10) {
        uint1 __9;
        float2 __10;
          float2 __11;
          uint1 v__6 = *(uint1*)(cb_local + (i_10 * 2));
          ((float2*)(&__11))[0] = __half22float2(((half2*)(&v__6))[0]);
          float2 v__7 = *(float2*)(dt_local + ((i_10 >> 1) * 2));
          __10.x = (__11.x*v__7.x);
          __10.y = (__11.y*v__7.y);
        ((half2*)(&__9))[0] = __float22half2_rn(((float2*)(&__10))[0]);
        *(uint1*)(cb_local + (i_10 * 2)) = __9;
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 32; ++i_11) {
        half_t condval;
        if (((((((ik_1 * 64) + ((i_11 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_11 & 1)) + 64) <= (((((((int)blockIdx.y) >> 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_11 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval = cb_local[i_11];
        } else {
          condval = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local[i_11] = condval;
      }
      mbarrier[((ik_1 & 1) + 6)].wait((ik_1 >> 1));
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_1, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, ((ik_1 & 1) * 4096));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local + 0), 16);
      }
      mbarrier[((ik_1 & 1) + 14)].arrive();
    }
    d_local[0] = ((float)D[((int)blockIdx.x)]);
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      *(uint4*)(((half_t*)residual_shared) + (((((i_12 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 1024)) = *(uint4*)(X + ((((((((((((int)blockIdx.z) & 7) * 20971520) + ((((int)blockIdx.z) >> 3) * 1310720)) + ((((int)blockIdx.y) >> 1) * 327680)) + (i_12 * 163840)) + ((((int)threadIdx.x) >> 2) * 5120)) + (((int)blockIdx.x) * 64)) + ((((int)blockIdx.y) & 1) * 32)) + ((((int)threadIdx.x) & 3) * 8)) - 163840));
    }
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_13 = 0; i_13 < 8; ++i_13) {
      half_t residual_shared_local_cast_2[2];
      *(uint1*)(residual_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)residual_shared) + ((((((((((int)threadIdx.x) >> 5) * 512) + ((i_13 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_13 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_13 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 2048));
      float2 __12;
      uint1 v__8 = *(uint1*)(residual_shared_local_cast_2 + 0);
      ((float2*)(&__12))[0] = __half22float2(((half2*)(&v__8))[0]);
      *(float2*)(residual_local + (i_13 * 2)) = __12;
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 16; ++i_14) {
      acc[i_14] = (acc[i_14] + (residual_local[i_14] * d_local[0]));
    }
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[((((((((int)threadIdx.x) & 127) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_15) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((half_t)acc[(i_15 * 8)]), ((half_t)acc[((i_15 * 8) + 1)])), __pack_half2(((half_t)acc[((i_15 * 8) + 2)]), ((half_t)acc[((i_15 * 8) + 3)])), __pack_half2(((half_t)acc[((i_15 * 8) + 4)]), ((half_t)acc[((i_15 * 8) + 5)])), __pack_half2(((half_t)acc[((i_15 * 8) + 6)]), ((half_t)acc[((i_15 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::fence_proxy_async();
      tl::tma_store(Output_desc, (&(((half_t*)acc_shared)[0])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

