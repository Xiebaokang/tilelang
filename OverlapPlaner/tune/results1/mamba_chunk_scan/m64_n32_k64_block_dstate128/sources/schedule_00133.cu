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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap CB_desc, __grid_constant__ const CUtensorMap C_desc, const half_t* __restrict__ D, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Prev_desc, __grid_constant__ const CUtensorMap X_desc);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(__grid_constant__ const CUtensorMap CB_desc, __grid_constant__ const CUtensorMap C_desc, const half_t* __restrict__ D, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Prev_desc, __grid_constant__ const CUtensorMap X_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* c_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* da_m_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* residual_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* prev_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  void* cb_shared = ((void*)((char*)buf_dyn_shmem + 36864));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 45056));
  void* da_k_shared = ((void*)((char*)buf_dyn_shmem + 46080));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[17];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float da_m_local[2];
  float acc[16];
  float scale_m[2];
  half_t cb_local_v0[32];
  float da_k_local[16];
  half_t cb_local_v1[32];
  float dt_local_v0[16];
  half_t cb_local_v2[32];
  float dt_local_v1[16];
  float d_local[1];
  float residual_local[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(C_desc);
    tl::prefetch_tma_descriptor(Prev_desc);
    tl::prefetch_tma_descriptor(CB_desc);
    tl::prefetch_tma_descriptor(X_desc);
    tl::prefetch_tma_descriptor(Output_desc);
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
    overlap_plan_mbar[10].init(128);
    overlap_plan_mbar[11].init(128);
    overlap_plan_mbar[12].init(128);
    overlap_plan_mbar[13].init(128);
    overlap_plan_mbar[14].init(128);
    overlap_plan_mbar[15].init(128);
    overlap_plan_mbar[16].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)da_m_shared)[((int)threadIdx.x)] = DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + ((((int)blockIdx.y) >> 1) * 64)) + ((int)threadIdx.x))];
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    da_m_local[i] = ((float)((half_t*)da_m_shared)[((((((int)threadIdx.x) >> 5) * 16) + (i * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    scale_m[i_2] = exp2f((da_m_local[i_2] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[0].arrive_and_expect_tx(16384);
    tl::fence_proxy_async();
    tl::tma_load(C_desc, overlap_plan_mbar[0], (&(((half_t*)c_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 64)), 0, (((int)blockIdx.z) & 7));
    tl::tma_load(C_desc, overlap_plan_mbar[0], (&(((half_t*)c_shared)[4096])), 64, (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 64)), 0, (((int)blockIdx.z) & 7));
    overlap_plan_mbar[1].arrive_and_expect_tx(8192);
    tl::tma_load(Prev_desc, overlap_plan_mbar[1], (&(((half_t*)prev_shared)[0])), 0, ((((int)blockIdx.y) & 1) * 32), ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
    tl::tma_load(Prev_desc, overlap_plan_mbar[1], (&(((half_t*)prev_shared)[2048])), 64, ((((int)blockIdx.y) & 1) * 32), ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[0].wait(0);
  overlap_plan_mbar[1].wait(0);
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
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    acc[i_3] = (acc[i_3] * scale_m[((i_3 & 3) >> 1)]);
  }
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[2].arrive_and_expect_tx(8192);
    tl::tma_load(CB_desc, overlap_plan_mbar[2], (&(((half_t*)cb_shared)[0])), 0, ((((int)blockIdx.y) >> 1) * 64), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[2].wait(0);
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v0[(i_4 * 8)])));
  }
  overlap_plan_mbar[10].arrive();
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[3].arrive_and_expect_tx(128);
    tl::tma_load((&(((half_t*)da_k_shared)[0])), (&(DA[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[3], 128);
  }
  overlap_plan_mbar[3].wait(0);
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 8; ++i_5) {
    half_t da_k_shared_local_cast[2];
    *(uint1*)(da_k_shared_local_cast + 0) = *(uint1*)(((half_t*)da_k_shared) + ((i_5 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __1;
    uint1 v_ = *(uint1*)(da_k_shared_local_cast + 0);
    ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
    *(float2*)(da_k_local + (i_5 * 2)) = __1;
  }
  overlap_plan_mbar[11].arrive();
  #pragma unroll
  for (int i_6 = 0; i_6 < 16; ++i_6) {
    float broadcast_var_1 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    uint1 __2;
    float2 __3;
      float2 __4;
      uint1 v__1 = *(uint1*)(cb_local_v0 + (i_6 * 2));
      ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__1))[0]);
      float2 __5;
      float2 __6;
        float2 __7;
          float2 v__2 = make_float2(da_m_local[(i_6 & 1)], da_m_local[(i_6 & 1)]);
          float2 v__3 = *(float2*)(da_k_local + ((i_6 >> 1) * 2));
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
    *(uint1*)(cb_local_v0 + (i_6 * 2)) = __2;
  }
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[4].arrive_and_expect_tx(128);
    tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[4], 128);
    overlap_plan_mbar[6].arrive_and_expect_tx(4096);
    tl::tma_load(X_desc, overlap_plan_mbar[6], (&(((half_t*)x_shared)[0])), ((((int)blockIdx.y) & 1) * 32), ((((int)blockIdx.z) >> 3) * 256), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
  }
  if (1 < ((int)blockIdx.y)) {
    overlap_plan_mbar[10].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(8192);
      tl::tma_load(CB_desc, overlap_plan_mbar[2], (&(((half_t*)cb_shared)[0])), 64, ((((int)blockIdx.y) >> 1) * 64), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[2].wait(1);
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_7 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v1[(i_7 * 8)])));
    }
    overlap_plan_mbar[10].arrive();
  }
  __syncthreads();
  if (1 < ((int)blockIdx.y)) {
    overlap_plan_mbar[11].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)da_k_shared)[0])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 64)])), overlap_plan_mbar[3], 128);
    }
  }
  __syncthreads();
  if (1 < ((int)blockIdx.y)) {
    overlap_plan_mbar[3].wait(1);
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      half_t da_k_shared_local_cast_1[2];
      *(uint1*)(da_k_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)da_k_shared) + ((i_8 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __8;
      uint1 v__5 = *(uint1*)(da_k_shared_local_cast_1 + 0);
      ((float2*)(&__8))[0] = __half22float2(((half2*)(&v__5))[0]);
      *(float2*)(da_k_local + (i_8 * 2)) = __8;
    }
    overlap_plan_mbar[11].arrive();
    #pragma unroll
    for (int i_9 = 0; i_9 < 16; ++i_9) {
      float broadcast_var_2 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
      uint1 __9;
      float2 __10;
        float2 __11;
        uint1 v__6 = *(uint1*)(cb_local_v1 + (i_9 * 2));
        ((float2*)(&__11))[0] = __half22float2(((half2*)(&v__6))[0]);
        float2 __12;
        float2 __13;
          float2 __14;
            float2 v__7 = make_float2(da_m_local[(i_9 & 1)], da_m_local[(i_9 & 1)]);
            float2 v__8 = *(float2*)(da_k_local + ((i_9 >> 1) * 2));
            __14.x = (v__7.x-v__8.x);
            __14.y = (v__7.y-v__8.y);
          float2 v__9 = make_float2(broadcast_var_2, broadcast_var_2);
          __13.x = (__14.x*v__9.x);
          __13.y = (__14.y*v__9.y);
        __12.x = exp2f(__13.x);
        __12.y = exp2f(__13.y);
        __10.x = (__11.x*__12.x);
        __10.y = (__11.y*__12.y);
      ((half2*)(&__9))[0] = __float22half2_rn(((float2*)(&__10))[0]);
      *(uint1*)(cb_local_v1 + (i_9 * 2)) = __9;
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[5].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)dt_shared)[64])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 64)])), overlap_plan_mbar[5], 128);
    }
  }
  overlap_plan_mbar[4].wait(0);
  #pragma unroll
  for (int i_10 = 0; i_10 < 8; ++i_10) {
    half_t dt_shared_local_cast_2[2];
    *(uint1*)(dt_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_10 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __15;
    uint1 v__10 = *(uint1*)(dt_shared_local_cast_2 + 0);
    ((float2*)(&__15))[0] = __half22float2(((half2*)(&v__10))[0]);
    *(float2*)(dt_local_v0 + (i_10 * 2)) = __15;
  }
  overlap_plan_mbar[12].arrive();
  if (1 < ((int)blockIdx.y)) {
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[7].arrive_and_expect_tx(4096);
      tl::tma_load(X_desc, overlap_plan_mbar[7], (&(((half_t*)x_shared)[2048])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + 64), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
  }
  for (int ik = 0; ik < ((((int)blockIdx.y) >> 1) - 1); ++ik) {
    overlap_plan_mbar[10].wait(((ik + 1) & 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(8192);
      tl::tma_load(CB_desc, overlap_plan_mbar[2], (&(((half_t*)cb_shared)[0])), ((ik * 64) + 128), ((((int)blockIdx.y) >> 1) * 64), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[2].wait(ik);
    if (((ik + 2) % 3) == 0) {
      #pragma unroll
      for (int i_11 = 0; i_11 < 4; ++i_11) {
        tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v0[(i_11 * 8)])));
      }
    } else {
      if (((ik + 2) % 3) == 1) {
        #pragma unroll
        for (int i_12 = 0; i_12 < 4; ++i_12) {
          tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_12 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_12 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v1[(i_12 * 8)])));
        }
      } else {
        #pragma unroll
        for (int i_13 = 0; i_13 < 4; ++i_13) {
          tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_13 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v2[(i_13 * 8)])));
        }
      }
    }
    overlap_plan_mbar[10].arrive();
    overlap_plan_mbar[11].wait(((ik + 1) & 1));
    __syncthreads();
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)da_k_shared)[0])), (&(DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 128)])), overlap_plan_mbar[3], 128);
    }
    overlap_plan_mbar[3].wait(ik);
    __syncthreads();
    #pragma unroll
    for (int i_14 = 0; i_14 < 8; ++i_14) {
      half_t da_k_shared_local_cast_3[2];
      *(uint1*)(da_k_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)da_k_shared) + ((i_14 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __16;
      uint1 v__11 = *(uint1*)(da_k_shared_local_cast_3 + 0);
      ((float2*)(&__16))[0] = __half22float2(((half2*)(&v__11))[0]);
      *(float2*)(da_k_local + (i_14 * 2)) = __16;
    }
    overlap_plan_mbar[11].arrive();
    if (((ik + 2) % 3) == 0) {
      #pragma unroll
      for (int i_15 = 0; i_15 < 16; ++i_15) {
        float broadcast_var_3 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
        uint1 __17;
        float2 __18;
          float2 __19;
          uint1 v__12 = *(uint1*)(cb_local_v0 + (i_15 * 2));
          ((float2*)(&__19))[0] = __half22float2(((half2*)(&v__12))[0]);
          float2 __20;
          float2 __21;
            float2 __22;
              float2 v__13 = make_float2(da_m_local[(i_15 & 1)], da_m_local[(i_15 & 1)]);
              float2 v__14 = *(float2*)(da_k_local + ((i_15 >> 1) * 2));
              __22.x = (v__13.x-v__14.x);
              __22.y = (v__13.y-v__14.y);
            float2 v__15 = make_float2(broadcast_var_3, broadcast_var_3);
            __21.x = (__22.x*v__15.x);
            __21.y = (__22.y*v__15.y);
          __20.x = exp2f(__21.x);
          __20.y = exp2f(__21.y);
          __18.x = (__19.x*__20.x);
          __18.y = (__19.y*__20.y);
        ((half2*)(&__17))[0] = __float22half2_rn(((float2*)(&__18))[0]);
        *(uint1*)(cb_local_v0 + (i_15 * 2)) = __17;
      }
    } else {
      if (((ik + 2) % 3) == 1) {
        #pragma unroll
        for (int i_16 = 0; i_16 < 32; ++i_16) {
          cb_local_v1[i_16] = ((half_t)(((float)cb_local_v1[i_16]) * exp2f(((da_m_local[((i_16 & 3) >> 1)] - da_k_local[(((i_16 >> 2) * 2) + (i_16 & 1))]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/))));
        }
      } else {
        #pragma unroll
        for (int i_17 = 0; i_17 < 16; ++i_17) {
          float broadcast_var_4 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
          uint1 __23;
          float2 __24;
            float2 __25;
            uint1 v__16 = *(uint1*)(cb_local_v2 + (i_17 * 2));
            ((float2*)(&__25))[0] = __half22float2(((half2*)(&v__16))[0]);
            float2 __26;
            float2 __27;
              float2 __28;
                float2 v__17 = make_float2(da_m_local[(i_17 & 1)], da_m_local[(i_17 & 1)]);
                float2 v__18 = *(float2*)(da_k_local + ((i_17 >> 1) * 2));
                __28.x = (v__17.x-v__18.x);
                __28.y = (v__17.y-v__18.y);
              float2 v__19 = make_float2(broadcast_var_4, broadcast_var_4);
              __27.x = (__28.x*v__19.x);
              __27.y = (__28.y*v__19.y);
            __26.x = exp2f(__27.x);
            __26.y = exp2f(__27.y);
            __24.x = (__25.x*__26.x);
            __24.y = (__25.y*__26.y);
          ((half2*)(&__23))[0] = __float22half2_rn(((float2*)(&__24))[0]);
          *(uint1*)(cb_local_v2 + (i_17 * 2)) = __23;
        }
      }
    }
    overlap_plan_mbar[(ik + 12)].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[(ik + 4)].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)dt_shared)[(ik * 64)])), (&(Dt[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 128)])), overlap_plan_mbar[(ik + 4)], 128);
    }
    overlap_plan_mbar[(((ik + 1) & 1) + 4)].wait(((ik + 1) >> 1));
    if (((ik + 1) % 2) == 0) {
      #pragma unroll
      for (int i_18 = 0; i_18 < 8; ++i_18) {
        half_t dt_shared_local_cast_4[2];
        *(uint1*)(dt_shared_local_cast_4 + 0) = *(uint1*)(((half_t*)dt_shared) + (((((ik + 1) & 1) * 64) + (i_18 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __29;
        uint1 v__20 = *(uint1*)(dt_shared_local_cast_4 + 0);
        ((float2*)(&__29))[0] = __half22float2(((half2*)(&v__20))[0]);
        *(float2*)(dt_local_v0 + (i_18 * 2)) = __29;
      }
    } else {
      #pragma unroll
      for (int i_19 = 0; i_19 < 8; ++i_19) {
        half_t dt_shared_local_cast_5[2];
        *(uint1*)(dt_shared_local_cast_5 + 0) = *(uint1*)(((half_t*)dt_shared) + (((((ik + 1) & 1) * 64) + (i_19 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __30;
        uint1 v__21 = *(uint1*)(dt_shared_local_cast_5 + 0);
        ((float2*)(&__30))[0] = __half22float2(((half2*)(&v__21))[0]);
        *(float2*)(dt_local_v1 + (i_19 * 2)) = __30;
      }
    }
    overlap_plan_mbar[(((ik + 1) & 1) + 12)].arrive();
    if (ik == 0) {
      #pragma unroll
      for (int i_20 = 0; i_20 < 16; ++i_20) {
        uint1 __31;
        float2 __32;
          float2 __33;
          uint1 v__22 = *(uint1*)(cb_local_v0 + (i_20 * 2));
          ((float2*)(&__33))[0] = __half22float2(((half2*)(&v__22))[0]);
          float2 v__23 = *(float2*)(dt_local_v0 + ((i_20 >> 1) * 2));
          __32.x = (__33.x*v__23.x);
          __32.y = (__33.y*v__23.y);
        ((half2*)(&__31))[0] = __float22half2_rn(((float2*)(&__32))[0]);
        *(uint1*)(cb_local_v0 + (i_20 * 2)) = __31;
      }
    } else {
      #pragma unroll
      for (int i_21 = 0; i_21 < 16; ++i_21) {
        uint1 __34;
        float2 __35;
          float2 __36;
          uint1 v__24 = *(uint1*)(cb_local_v1 + (i_21 * 2));
          ((float2*)(&__36))[0] = __half22float2(((half2*)(&v__24))[0]);
          float2 v__25 = *(float2*)(dt_local_v1 + ((i_21 >> 1) * 2));
          __35.x = (__36.x*v__25.x);
          __35.y = (__36.y*v__25.y);
        ((half2*)(&__34))[0] = __float22half2_rn(((float2*)(&__35))[0]);
        *(uint1*)(cb_local_v1 + (i_21 * 2)) = __34;
      }
    }
    if (ik == 0) {
      #pragma unroll
      for (int i_22 = 0; i_22 < 32; ++i_22) {
        half_t condval;
        if ((((((ik * 64) + ((i_22 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_22 & 1)) <= (((((((int)blockIdx.y) >> 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_22 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval = cb_local_v0[i_22];
        } else {
          condval = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local_v0[i_22] = condval;
      }
    } else {
      #pragma unroll
      for (int i_23 = 0; i_23 < 32; ++i_23) {
        half_t condval_1;
        if ((((((ik * 64) + ((i_23 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_23 & 1)) <= (((((((int)blockIdx.y) >> 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_23 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval_1 = cb_local_v1[i_23];
        } else {
          condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local_v1[i_23] = condval_1;
      }
    }
    if (ik == 1) {
      overlap_plan_mbar[(ik + 13)].wait(0);
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[(((ik + 2) % 3) + 6)].arrive_and_expect_tx(4096);
      tl::tma_load(X_desc, overlap_plan_mbar[(((ik + 2) % 3) + 6)], (&(((half_t*)x_shared)[(((ik + 2) % 3) * 2048)])), ((((int)blockIdx.y) & 1) * 32), ((((((int)blockIdx.z) >> 3) * 256) + (ik * 64)) + 128), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[(ik + 6)].wait(0);
    if (ik == 0) {
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_1, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, (ik * 4096));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v0 + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 16);
      }
    } else {
      {
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_2, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_2, (ik * 4096));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v1 + (ki_2 * 8)), uint64_t(desc_b_2 + ((ki_2 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 16);
      }
    }
    overlap_plan_mbar[(ik + 14)].arrive();
  }
  overlap_plan_mbar[(((((int)blockIdx.y) & 3) >> 1) + 4)].wait((((int)blockIdx.y) >> 2));
  __syncthreads();
  if (((((int)blockIdx.y) & 3) >> 1) == 0) {
    #pragma unroll
    for (int i_24 = 0; i_24 < 8; ++i_24) {
      half_t dt_shared_local_cast_6[2];
      *(uint1*)(dt_shared_local_cast_6 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_24 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __37;
      uint1 v__26 = *(uint1*)(dt_shared_local_cast_6 + 0);
      ((float2*)(&__37))[0] = __half22float2(((half2*)(&v__26))[0]);
      *(float2*)(dt_local_v0 + (i_24 * 2)) = __37;
    }
  } else {
    #pragma unroll
    for (int i_25 = 0; i_25 < 8; ++i_25) {
      half_t dt_shared_local_cast_7[2];
      *(uint1*)(dt_shared_local_cast_7 + 0) = *(uint1*)(((half_t*)dt_shared) + (((i_25 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
      float2 __38;
      uint1 v__27 = *(uint1*)(dt_shared_local_cast_7 + 0);
      ((float2*)(&__38))[0] = __half22float2(((half2*)(&v__27))[0]);
      *(float2*)(dt_local_v1 + (i_25 * 2)) = __38;
    }
  }
  overlap_plan_mbar[(((((int)blockIdx.y) & 3) >> 1) + 12)].arrive();
  if (2 <= ((int)blockIdx.y)) {
    if ((((((int)blockIdx.y) >> 1) + 5) % 6) == 0) {
      #pragma unroll
      for (int i_26 = 0; i_26 < 16; ++i_26) {
        uint1 __39;
        float2 __40;
          float2 __41;
          uint1 v__28 = *(uint1*)(cb_local_v0 + (i_26 * 2));
          ((float2*)(&__41))[0] = __half22float2(((half2*)(&v__28))[0]);
          float2 v__29 = *(float2*)(dt_local_v0 + ((i_26 >> 1) * 2));
          __40.x = (__41.x*v__29.x);
          __40.y = (__41.y*v__29.y);
        ((half2*)(&__39))[0] = __float22half2_rn(((float2*)(&__40))[0]);
        *(uint1*)(cb_local_v0 + (i_26 * 2)) = __39;
      }
    } else {
      if ((((((int)blockIdx.y) >> 1) + 5) % 6) == 1) {
        #pragma unroll
        for (int i_27 = 0; i_27 < 16; ++i_27) {
          uint1 __42;
          float2 __43;
            float2 __44;
            uint1 v__30 = *(uint1*)(cb_local_v1 + (i_27 * 2));
            ((float2*)(&__44))[0] = __half22float2(((half2*)(&v__30))[0]);
            float2 v__31 = *(float2*)(dt_local_v1 + ((i_27 >> 1) * 2));
            __43.x = (__44.x*v__31.x);
            __43.y = (__44.y*v__31.y);
          ((half2*)(&__42))[0] = __float22half2_rn(((float2*)(&__43))[0]);
          *(uint1*)(cb_local_v1 + (i_27 * 2)) = __42;
        }
      } else {
        #pragma unroll
        for (int i_28 = 0; i_28 < 16; ++i_28) {
          uint1 __45;
          float2 __46;
            float2 __47;
            uint1 v__32 = *(uint1*)(cb_local_v2 + (i_28 * 2));
            ((float2*)(&__47))[0] = __half22float2(((half2*)(&v__32))[0]);
            float2 v__33 = *(float2*)(dt_local_v0 + ((i_28 >> 1) * 2));
            __46.x = (__47.x*v__33.x);
            __46.y = (__47.y*v__33.y);
          ((half2*)(&__45))[0] = __float22half2_rn(((float2*)(&__46))[0]);
          *(uint1*)(cb_local_v2 + (i_28 * 2)) = __45;
        }
      }
    }
    overlap_plan_mbar[((((((int)blockIdx.y) >> 1) + 2) % 3) + 6)].wait(((((((int)blockIdx.y) >> 1) + 5) % 6) / 3));
    if ((((((int)blockIdx.y) >> 1) + 2) % 3) == 0) {
      {
        tl::GmmaDescriptor desc_b_3;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_3, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_3, ((((((int)blockIdx.y) >> 1) + 2) % 3) * 4096));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v0 + (ki_3 * 8)), uint64_t(desc_b_3 + ((ki_3 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 16);
      }
    } else {
      if ((((((int)blockIdx.y) >> 1) + 2) % 3) == 1) {
        {
          tl::GmmaDescriptor desc_b_4;
          tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_4, (&(((half_t*)x_shared)[0])));
          tl::increase_descriptor_offset<int>(desc_b_4, ((((((int)blockIdx.y) >> 1) + 2) % 3) * 4096));
          tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 16);
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
          tl::warpgroup_arrive();
          #pragma unroll
          for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
            tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v1 + (ki_4 * 8)), uint64_t(desc_b_4 + ((ki_4 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
          }
          tl::warpgroup_commit_batch();
          tl::warpgroup_wait<0>();
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
          tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 16);
        }
      } else {
        {
          tl::GmmaDescriptor desc_b_5;
          tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_5, (&(((half_t*)x_shared)[0])));
          tl::increase_descriptor_offset<int>(desc_b_5, ((((((int)blockIdx.y) >> 1) + 2) % 3) * 4096));
          tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v2 + 0), 16);
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
          tl::warpgroup_arrive();
          #pragma unroll
          for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
            tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v2 + (ki_5 * 8)), uint64_t(desc_b_5 + ((ki_5 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
          }
          tl::warpgroup_commit_batch();
          tl::warpgroup_wait<0>();
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
          tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v2 + 0), 16);
        }
      }
    }
    overlap_plan_mbar[((((((int)blockIdx.y) >> 1) + 2) % 3) + 14)].arrive();
  }
  if ((((int)blockIdx.y) >> 1) == 0) {
    #pragma unroll
    for (int i_29 = 0; i_29 < 16; ++i_29) {
      uint1 __48;
      float2 __49;
        float2 __50;
        uint1 v__34 = *(uint1*)(cb_local_v0 + (i_29 * 2));
        ((float2*)(&__50))[0] = __half22float2(((half2*)(&v__34))[0]);
        float2 v__35 = *(float2*)(dt_local_v0 + ((i_29 >> 1) * 2));
        __49.x = (__50.x*v__35.x);
        __49.y = (__50.y*v__35.y);
      ((half2*)(&__48))[0] = __float22half2_rn(((float2*)(&__49))[0]);
      *(uint1*)(cb_local_v0 + (i_29 * 2)) = __48;
    }
  } else {
    if ((((int)blockIdx.y) >> 1) == 1) {
      #pragma unroll
      for (int i_30 = 0; i_30 < 16; ++i_30) {
        uint1 __51;
        float2 __52;
          float2 __53;
          uint1 v__36 = *(uint1*)(cb_local_v1 + (i_30 * 2));
          ((float2*)(&__53))[0] = __half22float2(((half2*)(&v__36))[0]);
          float2 v__37 = *(float2*)(dt_local_v1 + ((i_30 >> 1) * 2));
          __52.x = (__53.x*v__37.x);
          __52.y = (__53.y*v__37.y);
        ((half2*)(&__51))[0] = __float22half2_rn(((float2*)(&__52))[0]);
        *(uint1*)(cb_local_v1 + (i_30 * 2)) = __51;
      }
    } else {
      if ((((int)blockIdx.y) >> 1) == 2) {
        #pragma unroll
        for (int i_31 = 0; i_31 < 16; ++i_31) {
          uint1 __54;
          float2 __55;
            float2 __56;
            uint1 v__38 = *(uint1*)(cb_local_v2 + (i_31 * 2));
            ((float2*)(&__56))[0] = __half22float2(((half2*)(&v__38))[0]);
            float2 v__39 = *(float2*)(dt_local_v0 + ((i_31 >> 1) * 2));
            __55.x = (__56.x*v__39.x);
            __55.y = (__56.y*v__39.y);
          ((half2*)(&__54))[0] = __float22half2_rn(((float2*)(&__55))[0]);
          *(uint1*)(cb_local_v2 + (i_31 * 2)) = __54;
        }
      } else {
        #pragma unroll
        for (int i_32 = 0; i_32 < 16; ++i_32) {
          uint1 __57;
          float2 __58;
            float2 __59;
            uint1 v__40 = *(uint1*)(cb_local_v0 + (i_32 * 2));
            ((float2*)(&__59))[0] = __half22float2(((half2*)(&v__40))[0]);
            float2 v__41 = *(float2*)(dt_local_v1 + ((i_32 >> 1) * 2));
            __58.x = (__59.x*v__41.x);
            __58.y = (__59.y*v__41.y);
          ((half2*)(&__57))[0] = __float22half2_rn(((float2*)(&__58))[0]);
          *(uint1*)(cb_local_v0 + (i_32 * 2)) = __57;
        }
      }
    }
  }
  if (((((int)blockIdx.y) % 6) >> 1) == 0) {
    #pragma unroll
    for (int i_33 = 0; i_33 < 32; ++i_33) {
      half_t condval_2;
      if (((((((((int)blockIdx.y) >> 1) * 64) + ((i_33 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_33 & 1)) <= (((((((int)blockIdx.y) >> 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_33 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_2 = cb_local_v0[i_33];
      } else {
        condval_2 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      cb_local_v0[i_33] = condval_2;
    }
  } else {
    if (((((int)blockIdx.y) % 6) >> 1) == 1) {
      #pragma unroll
      for (int i_34 = 0; i_34 < 32; ++i_34) {
        half_t condval_3;
        if (((((((((int)blockIdx.y) >> 1) * 64) + ((i_34 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_34 & 1)) <= (((((((int)blockIdx.y) >> 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_34 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval_3 = cb_local_v1[i_34];
        } else {
          condval_3 = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local_v1[i_34] = condval_3;
      }
    } else {
      #pragma unroll
      for (int i_35 = 0; i_35 < 32; ++i_35) {
        half_t condval_4;
        if (((((((((int)blockIdx.y) >> 1) * 64) + ((i_35 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_35 & 1)) <= (((((((int)blockIdx.y) >> 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_35 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval_4 = cb_local_v2[i_35];
        } else {
          condval_4 = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local_v2[i_35] = condval_4;
      }
    }
  }
  overlap_plan_mbar[(((((int)blockIdx.y) % 6) >> 1) + 6)].wait((((int)blockIdx.y) / 6));
  if (((((int)blockIdx.y) % 6) >> 1) == 0) {
    {
      tl::GmmaDescriptor desc_b_6;
      tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_6, (&(((half_t*)x_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_6, 0);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v0 + (ki_6 * 8)), uint64_t(desc_b_6 + ((ki_6 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 16);
    }
  } else {
    if (((((int)blockIdx.y) % 6) >> 1) == 1) {
      {
        tl::GmmaDescriptor desc_b_7;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_7, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_7, 4096);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v1 + (ki_7 * 8)), uint64_t(desc_b_7 + ((ki_7 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 16);
      }
    } else {
      {
        tl::GmmaDescriptor desc_b_8;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_8, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_8, 8192);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v2 + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_8 = 0; ki_8 < 4; ++ki_8) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v2 + (ki_8 * 8)), uint64_t(desc_b_8 + ((ki_8 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v2 + 0), 16);
      }
    }
  }
  overlap_plan_mbar[(((((int)blockIdx.y) % 6) >> 1) + 14)].arrive();
  d_local[0] = ((float)D[((int)blockIdx.x)]);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[9].arrive_and_expect_tx(4096);
    tl::tma_load(X_desc, overlap_plan_mbar[9], (&(((half_t*)residual_shared)[0])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[9].wait(0);
  #pragma unroll
  for (int i_36 = 0; i_36 < 8; ++i_36) {
    half_t residual_shared_local_cast_8[2];
    *(uint1*)(residual_shared_local_cast_8 + 0) = *(uint1*)(((half_t*)residual_shared) + (((((((((int)threadIdx.x) >> 5) * 512) + ((i_36 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_36 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_36 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __60;
    uint1 v__42 = *(uint1*)(residual_shared_local_cast_8 + 0);
    ((float2*)(&__60))[0] = __half22float2(((half2*)(&v__42))[0]);
    *(float2*)(residual_local + (i_36 * 2)) = __60;
  }
  #pragma unroll
  for (int i_37 = 0; i_37 < 16; ++i_37) {
    acc[i_37] = (acc[i_37] + (residual_local[i_37] * d_local[0]));
  }
  __syncthreads();
  #pragma unroll
  for (int i_38 = 0; i_38 < 2; ++i_38) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_38) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((half_t)acc[(i_38 * 8)]), ((half_t)acc[((i_38 * 8) + 1)])), __pack_half2(((half_t)acc[((i_38 * 8) + 2)]), ((half_t)acc[((i_38 * 8) + 3)])), __pack_half2(((half_t)acc[((i_38 * 8) + 4)]), ((half_t)acc[((i_38 * 8) + 5)])), __pack_half2(((half_t)acc[((i_38 * 8) + 6)]), ((half_t)acc[((i_38 * 8) + 7)])));
  }
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    __syncthreads();
    tl::tma_store(Output_desc, (&(((half_t*)acc_shared)[0])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

