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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap CB_desc, __grid_constant__ const CUtensorMap C_desc, const half_t* __restrict__ D, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Prev_desc, __grid_constant__ const CUtensorMap X_desc, __grid_constant__ const CUtensorMap X_desc_1);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(__grid_constant__ const CUtensorMap CB_desc, __grid_constant__ const CUtensorMap C_desc, const half_t* __restrict__ D, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Prev_desc, __grid_constant__ const CUtensorMap X_desc, __grid_constant__ const CUtensorMap X_desc_1) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* c_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* da_m_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* residual_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* prev_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* cb_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* da_k_shared = ((void*)((char*)buf_dyn_shmem + 81920));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 82944));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[13];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float da_m_local[4];
  float acc[64];
  float scale_m[4];
  half_t cb_local_v0[64];
  float da_k_local_v0[16];
  float dt_local_v0[16];
  half_t cb_local_v1[64];
  float da_k_local_v1[16];
  float dt_local_v1[16];
  half_t cb_local_v2[64];
  float da_k_local_v2[16];
  float dt_local_v2[16];
  float d_local[1];
  float residual_local[64];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(C_desc);
    tl::prefetch_tma_descriptor(Prev_desc);
    tl::prefetch_tma_descriptor(CB_desc);
    tl::prefetch_tma_descriptor(X_desc);
    tl::prefetch_tma_descriptor(X_desc_1);
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
    overlap_plan_mbar[8].init(128);
    overlap_plan_mbar[9].init(128);
    overlap_plan_mbar[10].init(128);
    overlap_plan_mbar[11].init(128);
    overlap_plan_mbar[12].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  ((half_t*)da_m_shared)[((int)threadIdx.x)] = DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.x))];
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    da_m_local[i] = ((float)((half_t*)da_m_shared)[(((((i >> 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    scale_m[i_2] = exp2f((da_m_local[i_2] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[0].arrive_and_expect_tx(32768);
    tl::fence_proxy_async();
    tl::tma_load(C_desc, overlap_plan_mbar[0], (&(((half_t*)c_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + (((int)blockIdx.y) * 128)), 0, (((int)blockIdx.z) & 7));
    tl::tma_load(C_desc, overlap_plan_mbar[0], (&(((half_t*)c_shared)[8192])), 64, (((((int)blockIdx.z) >> 3) * 256) + (((int)blockIdx.y) * 128)), 0, (((int)blockIdx.z) & 7));
    overlap_plan_mbar[1].arrive_and_expect_tx(16384);
    tl::tma_load(Prev_desc, overlap_plan_mbar[1], (&(((half_t*)prev_shared)[0])), 0, 0, ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
    tl::tma_load(Prev_desc, overlap_plan_mbar[1], (&(((half_t*)prev_shared)[4096])), 64, 0, ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[0].wait(0);
  overlap_plan_mbar[1].wait(0);
  {
    tl::GmmaDescriptor desc_a;
    tl::GmmaDescriptor desc_b;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)c_shared)[0])));
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)prev_shared)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + (((((ki >> 2) * 16384) + (i_3 * 8192)) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc + (i_3 * 32))), 1);
      }
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 64; ++i_4) {
    acc[i_4] = (acc[i_4] * scale_m[(((i_4 >> 5) * 2) + ((i_4 & 3) >> 1))]);
  }
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[2].arrive_and_expect_tx(16384);
    tl::tma_load(CB_desc, overlap_plan_mbar[2], (&(((half_t*)cb_shared)[0])), 0, (((int)blockIdx.y) * 128), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[2].wait(0);
  #pragma unroll
  for (int i_5 = 0; i_5 < 8; ++i_5) {
    tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[(((((i_5 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((i_5 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_5 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v0[(i_5 * 8)])));
  }
  overlap_plan_mbar[8].arrive();
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[3].arrive_and_expect_tx(128);
    tl::tma_load((&(((half_t*)da_k_shared)[0])), (&(DA[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[3], 128);
  }
  overlap_plan_mbar[3].wait(0);
  __syncthreads();
  #pragma unroll
  for (int i_6 = 0; i_6 < 8; ++i_6) {
    half_t da_k_shared_local_cast[2];
    *(uint1*)(da_k_shared_local_cast + 0) = *(uint1*)(((half_t*)da_k_shared) + ((i_6 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __1;
    uint1 v_ = *(uint1*)(da_k_shared_local_cast + 0);
    ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
    *(float2*)(da_k_local_v0 + (i_6 * 2)) = __1;
  }
  overlap_plan_mbar[9].arrive();
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[4].arrive_and_expect_tx(128);
    tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[4], 128);
  }
  overlap_plan_mbar[4].wait(0);
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 8; ++i_7) {
    half_t dt_shared_local_cast_1[2];
    *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_7 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __2;
    uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
    ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
    *(float2*)(dt_local_v0 + (i_7 * 2)) = __2;
  }
  overlap_plan_mbar[10].arrive();
  overlap_plan_mbar[8].wait(0);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[2].arrive_and_expect_tx(16384);
    tl::tma_load(CB_desc, overlap_plan_mbar[2], (&(((half_t*)cb_shared)[0])), 64, (((int)blockIdx.y) * 128), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[2].wait(1);
  #pragma unroll
  for (int i_8 = 0; i_8 < 8; ++i_8) {
    tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[(((((i_8 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((i_8 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_8 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v1[(i_8 * 8)])));
  }
  overlap_plan_mbar[8].arrive();
  overlap_plan_mbar[9].wait(0);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[3].arrive_and_expect_tx(128);
    tl::tma_load((&(((half_t*)da_k_shared)[0])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 64)])), overlap_plan_mbar[3], 128);
  }
  overlap_plan_mbar[3].wait(1);
  __syncthreads();
  #pragma unroll
  for (int i_9 = 0; i_9 < 8; ++i_9) {
    half_t da_k_shared_local_cast_2[2];
    *(uint1*)(da_k_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)da_k_shared) + ((i_9 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __3;
    uint1 v__2 = *(uint1*)(da_k_shared_local_cast_2 + 0);
    ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__2))[0]);
    *(float2*)(da_k_local_v1 + (i_9 * 2)) = __3;
  }
  overlap_plan_mbar[9].arrive();
  overlap_plan_mbar[10].wait(0);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[4].arrive_and_expect_tx(128);
    tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 64)])), overlap_plan_mbar[4], 128);
  }
  overlap_plan_mbar[4].wait(1);
  __syncthreads();
  #pragma unroll
  for (int i_10 = 0; i_10 < 8; ++i_10) {
    half_t dt_shared_local_cast_3[2];
    *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_10 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __4;
    uint1 v__3 = *(uint1*)(dt_shared_local_cast_3 + 0);
    ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__3))[0]);
    *(float2*)(dt_local_v1 + (i_10 * 2)) = __4;
  }
  overlap_plan_mbar[10].arrive();
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[5].arrive_and_expect_tx(8192);
    tl::tma_load(X_desc, overlap_plan_mbar[5], (&(((half_t*)x_shared)[0])), 0, ((((int)blockIdx.z) >> 3) * 256), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
  }
  for (int ik = 0; ik < (((int)blockIdx.y) * 2); ++ik) {
    overlap_plan_mbar[8].wait(((ik + 1) & 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(16384);
      tl::tma_load(CB_desc, overlap_plan_mbar[2], (&(((half_t*)cb_shared)[0])), ((ik * 64) + 128), (((int)blockIdx.y) * 128), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[2].wait(ik);
    if (((ik + 2) % 3) == 0) {
      #pragma unroll
      for (int i_11 = 0; i_11 < 8; ++i_11) {
        tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[(((((i_11 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((i_11 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_11 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v0[(i_11 * 8)])));
      }
    } else {
      if (((ik + 2) % 3) == 1) {
        #pragma unroll
        for (int i_12 = 0; i_12 < 8; ++i_12) {
          tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[(((((i_12 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((i_12 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_12 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v1[(i_12 * 8)])));
        }
      } else {
        #pragma unroll
        for (int i_13 = 0; i_13 < 8; ++i_13) {
          tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[(((((i_13 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((i_13 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_13 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v2[(i_13 * 8)])));
        }
      }
    }
    overlap_plan_mbar[8].arrive();
    overlap_plan_mbar[9].wait(((ik + 1) & 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)da_k_shared)[0])), (&(DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 128)])), overlap_plan_mbar[3], 128);
    }
    overlap_plan_mbar[3].wait(ik);
    __syncthreads();
    if (((ik + 2) % 3) == 0) {
      #pragma unroll
      for (int i_14 = 0; i_14 < 8; ++i_14) {
        half_t da_k_shared_local_cast_4[2];
        *(uint1*)(da_k_shared_local_cast_4 + 0) = *(uint1*)(((half_t*)da_k_shared) + ((i_14 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __5;
        uint1 v__4 = *(uint1*)(da_k_shared_local_cast_4 + 0);
        ((float2*)(&__5))[0] = __half22float2(((half2*)(&v__4))[0]);
        *(float2*)(da_k_local_v0 + (i_14 * 2)) = __5;
      }
    } else {
      if (((ik + 2) % 3) == 1) {
        #pragma unroll
        for (int i_15 = 0; i_15 < 16; ++i_15) {
          da_k_local_v1[i_15] = ((float)((half_t*)da_k_shared)[((((i_15 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_15 & 1))]);
        }
      } else {
        #pragma unroll
        for (int i_16 = 0; i_16 < 8; ++i_16) {
          half_t da_k_shared_local_cast_5[2];
          *(uint1*)(da_k_shared_local_cast_5 + 0) = *(uint1*)(((half_t*)da_k_shared) + ((i_16 * 8) + ((((int)threadIdx.x) & 3) * 2)));
          float2 __6;
          uint1 v__5 = *(uint1*)(da_k_shared_local_cast_5 + 0);
          ((float2*)(&__6))[0] = __half22float2(((half2*)(&v__5))[0]);
          *(float2*)(da_k_local_v2 + (i_16 * 2)) = __6;
        }
      }
    }
    overlap_plan_mbar[9].arrive();
    if (ik == 0) {
      #pragma unroll
      for (int i_17 = 0; i_17 < 32; ++i_17) {
        float broadcast_var_1 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
        uint1 __7;
        float2 __8;
          float2 __9;
          uint1 v__6 = *(uint1*)(cb_local_v0 + (i_17 * 2));
          ((float2*)(&__9))[0] = __half22float2(((half2*)(&v__6))[0]);
          float2 __10;
          float2 __11;
            float2 __12;
              float2 v__7 = make_float2(da_m_local[((((i_17 & 7) >> 2) * 2) + (i_17 & 1))], da_m_local[((((i_17 & 7) >> 2) * 2) + (i_17 & 1))]);
              float2 v__8 = *(float2*)(da_k_local_v0 + (((i_17 >> 3) * 4) + (((i_17 & 3) >> 1) * 2)));
              __12.x = (v__7.x-v__8.x);
              __12.y = (v__7.y-v__8.y);
            float2 v__9 = make_float2(broadcast_var_1, broadcast_var_1);
            __11.x = (__12.x*v__9.x);
            __11.y = (__12.y*v__9.y);
          __10.x = exp2f(__11.x);
          __10.y = exp2f(__11.y);
          __8.x = (__9.x*__10.x);
          __8.y = (__9.y*__10.y);
        ((half2*)(&__7))[0] = __float22half2_rn(((float2*)(&__8))[0]);
        *(uint1*)(cb_local_v0 + (i_17 * 2)) = __7;
      }
    } else {
      #pragma unroll
      for (int i_18 = 0; i_18 < 32; ++i_18) {
        float broadcast_var_2 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
        uint1 __13;
        float2 __14;
          float2 __15;
          uint1 v__10 = *(uint1*)(cb_local_v1 + (i_18 * 2));
          ((float2*)(&__15))[0] = __half22float2(((half2*)(&v__10))[0]);
          float2 __16;
          float2 __17;
            float2 __18;
              float2 v__11 = make_float2(da_m_local[((((i_18 & 7) >> 2) * 2) + (i_18 & 1))], da_m_local[((((i_18 & 7) >> 2) * 2) + (i_18 & 1))]);
              float2 v__12 = *(float2*)(da_k_local_v1 + (((i_18 >> 3) * 4) + (((i_18 & 3) >> 1) * 2)));
              __18.x = (v__11.x-v__12.x);
              __18.y = (v__11.y-v__12.y);
            float2 v__13 = make_float2(broadcast_var_2, broadcast_var_2);
            __17.x = (__18.x*v__13.x);
            __17.y = (__18.y*v__13.y);
          __16.x = exp2f(__17.x);
          __16.y = exp2f(__17.y);
          __14.x = (__15.x*__16.x);
          __14.y = (__15.y*__16.y);
        ((half2*)(&__13))[0] = __float22half2_rn(((float2*)(&__14))[0]);
        *(uint1*)(cb_local_v1 + (i_18 * 2)) = __13;
      }
    }
    overlap_plan_mbar[10].wait(((ik + 1) & 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 128)])), overlap_plan_mbar[4], 128);
    }
    overlap_plan_mbar[4].wait(ik);
    __syncthreads();
    if (((ik + 2) % 3) == 0) {
      #pragma unroll
      for (int i_19 = 0; i_19 < 8; ++i_19) {
        half_t dt_shared_local_cast_6[2];
        *(uint1*)(dt_shared_local_cast_6 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_19 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __19;
        uint1 v__14 = *(uint1*)(dt_shared_local_cast_6 + 0);
        ((float2*)(&__19))[0] = __half22float2(((half2*)(&v__14))[0]);
        *(float2*)(dt_local_v0 + (i_19 * 2)) = __19;
      }
    } else {
      if (((ik + 2) % 3) == 1) {
        #pragma unroll
        for (int i_20 = 0; i_20 < 16; ++i_20) {
          dt_local_v1[i_20] = ((float)((half_t*)dt_shared)[((((i_20 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_20 & 1))]);
        }
      } else {
        #pragma unroll
        for (int i_21 = 0; i_21 < 8; ++i_21) {
          half_t dt_shared_local_cast_7[2];
          *(uint1*)(dt_shared_local_cast_7 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_21 * 8) + ((((int)threadIdx.x) & 3) * 2)));
          float2 __20;
          uint1 v__15 = *(uint1*)(dt_shared_local_cast_7 + 0);
          ((float2*)(&__20))[0] = __half22float2(((half2*)(&v__15))[0]);
          *(float2*)(dt_local_v2 + (i_21 * 2)) = __20;
        }
      }
    }
    overlap_plan_mbar[10].arrive();
    if (ik == 0) {
      #pragma unroll
      for (int i_22 = 0; i_22 < 32; ++i_22) {
        uint1 __21;
        float2 __22;
          float2 __23;
          uint1 v__16 = *(uint1*)(cb_local_v0 + (i_22 * 2));
          ((float2*)(&__23))[0] = __half22float2(((half2*)(&v__16))[0]);
          float2 v__17 = *(float2*)(dt_local_v0 + (((i_22 >> 3) * 4) + (((i_22 & 3) >> 1) * 2)));
          __22.x = (__23.x*v__17.x);
          __22.y = (__23.y*v__17.y);
        ((half2*)(&__21))[0] = __float22half2_rn(((float2*)(&__22))[0]);
        *(uint1*)(cb_local_v0 + (i_22 * 2)) = __21;
      }
    } else {
      #pragma unroll
      for (int i_23 = 0; i_23 < 32; ++i_23) {
        uint1 __24;
        float2 __25;
          float2 __26;
          uint1 v__18 = *(uint1*)(cb_local_v1 + (i_23 * 2));
          ((float2*)(&__26))[0] = __half22float2(((half2*)(&v__18))[0]);
          float2 v__19 = *(float2*)(dt_local_v1 + (((i_23 >> 3) * 4) + (((i_23 & 3) >> 1) * 2)));
          __25.x = (__26.x*v__19.x);
          __25.y = (__26.y*v__19.y);
        ((half2*)(&__24))[0] = __float22half2_rn(((float2*)(&__25))[0]);
        *(uint1*)(cb_local_v1 + (i_23 * 2)) = __24;
      }
    }
    if (ik == 0) {
      #pragma unroll
      for (int i_24 = 0; i_24 < 64; ++i_24) {
        half_t condval;
        if (((((((ik * 64) + ((i_24 >> 4) * 16)) + (((i_24 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_24 & 1)) <= (((((((int)blockIdx.y) * 128) + (((i_24 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_24 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval = cb_local_v0[i_24];
        } else {
          condval = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local_v0[i_24] = condval;
      }
    } else {
      #pragma unroll
      for (int i_25 = 0; i_25 < 64; ++i_25) {
        half_t condval_1;
        if (((((((ik * 64) + ((i_25 >> 4) * 16)) + (((i_25 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_25 & 1)) <= (((((((int)blockIdx.y) * 128) + (((i_25 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_25 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval_1 = cb_local_v1[i_25];
        } else {
          condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local_v1[i_25] = condval_1;
      }
    }
    if (ik == 1) {
      overlap_plan_mbar[(ik + 10)].wait(0);
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[(((ik + 1) & 1) + 5)].arrive_and_expect_tx(8192);
      tl::tma_load(X_desc, overlap_plan_mbar[(((ik + 1) & 1) + 5)], (&(((half_t*)x_shared)[(((ik + 1) & 1) * 4096)])), 0, ((((((int)blockIdx.z) >> 3) * 256) + (ik * 64)) + 64), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[(ik + 5)].wait(0);
    if (ik == 0) {
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_1, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, (ik * 8192));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int i_26 = 0; i_26 < 2; ++i_26) {
          #pragma unroll
          for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
            tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v0 + ((ki_1 * 16) + (i_26 * 8))), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_26 * 32)), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 32);
      }
    } else {
      {
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_2, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_2, (ik * 8192));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int i_27 = 0; i_27 < 2; ++i_27) {
          #pragma unroll
          for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
            tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v1 + ((ki_2 * 16) + (i_27 * 8))), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_27 * 32)), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 32);
      }
    }
    overlap_plan_mbar[(ik + 11)].arrive();
  }
  if (((int)blockIdx.y) == 0) {
    #pragma unroll
    for (int i_28 = 0; i_28 < 32; ++i_28) {
      float broadcast_var_3 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
      uint1 __27;
      float2 __28;
        float2 __29;
        uint1 v__20 = *(uint1*)(cb_local_v0 + (i_28 * 2));
        ((float2*)(&__29))[0] = __half22float2(((half2*)(&v__20))[0]);
        float2 __30;
        float2 __31;
          float2 __32;
            float2 v__21 = make_float2(da_m_local[((((i_28 & 7) >> 2) * 2) + (i_28 & 1))], da_m_local[((((i_28 & 7) >> 2) * 2) + (i_28 & 1))]);
            float2 v__22 = *(float2*)(da_k_local_v0 + (((i_28 >> 3) * 4) + (((i_28 & 3) >> 1) * 2)));
            __32.x = (v__21.x-v__22.x);
            __32.y = (v__21.y-v__22.y);
          float2 v__23 = make_float2(broadcast_var_3, broadcast_var_3);
          __31.x = (__32.x*v__23.x);
          __31.y = (__32.y*v__23.y);
        __30.x = exp2f(__31.x);
        __30.y = exp2f(__31.y);
        __28.x = (__29.x*__30.x);
        __28.y = (__29.y*__30.y);
      ((half2*)(&__27))[0] = __float22half2_rn(((float2*)(&__28))[0]);
      *(uint1*)(cb_local_v0 + (i_28 * 2)) = __27;
    }
  } else {
    #pragma unroll
    for (int i_29 = 0; i_29 < 32; ++i_29) {
      float broadcast_var_4 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
      uint1 __33;
      float2 __34;
        float2 __35;
        uint1 v__24 = *(uint1*)(cb_local_v2 + (i_29 * 2));
        ((float2*)(&__35))[0] = __half22float2(((half2*)(&v__24))[0]);
        float2 __36;
        float2 __37;
          float2 __38;
            float2 v__25 = make_float2(da_m_local[((((i_29 & 7) >> 2) * 2) + (i_29 & 1))], da_m_local[((((i_29 & 7) >> 2) * 2) + (i_29 & 1))]);
            float2 v__26 = *(float2*)(da_k_local_v2 + (((i_29 >> 3) * 4) + (((i_29 & 3) >> 1) * 2)));
            __38.x = (v__25.x-v__26.x);
            __38.y = (v__25.y-v__26.y);
          float2 v__27 = make_float2(broadcast_var_4, broadcast_var_4);
          __37.x = (__38.x*v__27.x);
          __37.y = (__38.y*v__27.y);
        __36.x = exp2f(__37.x);
        __36.y = exp2f(__37.y);
        __34.x = (__35.x*__36.x);
        __34.y = (__35.y*__36.y);
      ((half2*)(&__33))[0] = __float22half2_rn(((float2*)(&__34))[0]);
      *(uint1*)(cb_local_v2 + (i_29 * 2)) = __33;
    }
  }
  if (((int)blockIdx.y) == 0) {
    #pragma unroll
    for (int i_30 = 0; i_30 < 32; ++i_30) {
      uint1 __39;
      float2 __40;
        float2 __41;
        uint1 v__28 = *(uint1*)(cb_local_v0 + (i_30 * 2));
        ((float2*)(&__41))[0] = __half22float2(((half2*)(&v__28))[0]);
        float2 v__29 = *(float2*)(dt_local_v0 + (((i_30 >> 3) * 4) + (((i_30 & 3) >> 1) * 2)));
        __40.x = (__41.x*v__29.x);
        __40.y = (__41.y*v__29.y);
      ((half2*)(&__39))[0] = __float22half2_rn(((float2*)(&__40))[0]);
      *(uint1*)(cb_local_v0 + (i_30 * 2)) = __39;
    }
  } else {
    #pragma unroll
    for (int i_31 = 0; i_31 < 32; ++i_31) {
      uint1 __42;
      float2 __43;
        float2 __44;
        uint1 v__30 = *(uint1*)(cb_local_v2 + (i_31 * 2));
        ((float2*)(&__44))[0] = __half22float2(((half2*)(&v__30))[0]);
        float2 v__31 = *(float2*)(dt_local_v2 + (((i_31 >> 3) * 4) + (((i_31 & 3) >> 1) * 2)));
        __43.x = (__44.x*v__31.x);
        __43.y = (__44.y*v__31.y);
      ((half2*)(&__42))[0] = __float22half2_rn(((float2*)(&__43))[0]);
      *(uint1*)(cb_local_v2 + (i_31 * 2)) = __42;
    }
  }
  if (((int)blockIdx.y) == 0) {
    #pragma unroll
    for (int i_32 = 0; i_32 < 64; ++i_32) {
      half_t condval_2;
      if (((((((((int)blockIdx.y) * 128) + ((i_32 >> 4) * 16)) + (((i_32 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_32 & 1)) <= (((((((int)blockIdx.y) * 128) + (((i_32 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_32 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_2 = cb_local_v0[i_32];
      } else {
        condval_2 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      cb_local_v0[i_32] = condval_2;
    }
  } else {
    #pragma unroll
    for (int i_33 = 0; i_33 < 64; ++i_33) {
      half_t condval_3;
      if (((((((((int)blockIdx.y) * 128) + ((i_33 >> 4) * 16)) + (((i_33 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_33 & 1)) <= (((((((int)blockIdx.y) * 128) + (((i_33 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_33 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_3 = cb_local_v2[i_33];
      } else {
        condval_3 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      cb_local_v2[i_33] = condval_3;
    }
  }
  if (((int)blockIdx.y) == 1) {
    overlap_plan_mbar[12].wait(((((int)blockIdx.y) + 1) & 1));
  }
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[6].arrive_and_expect_tx(8192);
    tl::tma_load(X_desc, overlap_plan_mbar[6], (&(((half_t*)x_shared)[4096])), 0, ((((((int)blockIdx.z) >> 3) * 256) + (((int)blockIdx.y) * 128)) + 64), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[5].wait(((int)blockIdx.y));
  if (((int)blockIdx.y) == 0) {
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_3, (&(((half_t*)x_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_3, 0);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_34 = 0; i_34 < 2; ++i_34) {
        #pragma unroll
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v0 + ((ki_3 * 16) + (i_34 * 8))), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_34 * 32)), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 32);
    }
  } else {
    {
      tl::GmmaDescriptor desc_b_4;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_4, (&(((half_t*)x_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_4, 0);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v2 + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_35 = 0; i_35 < 2; ++i_35) {
        #pragma unroll
        for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v2 + ((ki_4 * 16) + (i_35 * 8))), uint64_t(desc_b_4 + ((ki_4 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_35 * 32)), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v2 + 0), 32);
    }
  }
  overlap_plan_mbar[11].arrive();
  if ((((((int)blockIdx.y) * 2) + 1) % 3) == 0) {
    #pragma unroll
    for (int i_36 = 0; i_36 < 32; ++i_36) {
      float broadcast_var_5 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
      uint1 __45;
      float2 __46;
        float2 __47;
        uint1 v__32 = *(uint1*)(cb_local_v0 + (i_36 * 2));
        ((float2*)(&__47))[0] = __half22float2(((half2*)(&v__32))[0]);
        float2 __48;
        float2 __49;
          float2 __50;
            float2 v__33 = make_float2(da_m_local[((((i_36 & 7) >> 2) * 2) + (i_36 & 1))], da_m_local[((((i_36 & 7) >> 2) * 2) + (i_36 & 1))]);
            float2 v__34 = *(float2*)(da_k_local_v0 + (((i_36 >> 3) * 4) + (((i_36 & 3) >> 1) * 2)));
            __50.x = (v__33.x-v__34.x);
            __50.y = (v__33.y-v__34.y);
          float2 v__35 = make_float2(broadcast_var_5, broadcast_var_5);
          __49.x = (__50.x*v__35.x);
          __49.y = (__50.y*v__35.y);
        __48.x = exp2f(__49.x);
        __48.y = exp2f(__49.y);
        __46.x = (__47.x*__48.x);
        __46.y = (__47.y*__48.y);
      ((half2*)(&__45))[0] = __float22half2_rn(((float2*)(&__46))[0]);
      *(uint1*)(cb_local_v0 + (i_36 * 2)) = __45;
    }
  } else {
    #pragma unroll
    for (int i_37 = 0; i_37 < 32; ++i_37) {
      float broadcast_var_6 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
      uint1 __51;
      float2 __52;
        float2 __53;
        uint1 v__36 = *(uint1*)(cb_local_v1 + (i_37 * 2));
        ((float2*)(&__53))[0] = __half22float2(((half2*)(&v__36))[0]);
        float2 __54;
        float2 __55;
          float2 __56;
            float2 v__37 = make_float2(da_m_local[((((i_37 & 7) >> 2) * 2) + (i_37 & 1))], da_m_local[((((i_37 & 7) >> 2) * 2) + (i_37 & 1))]);
            float2 v__38 = *(float2*)(da_k_local_v1 + (((i_37 >> 3) * 4) + (((i_37 & 3) >> 1) * 2)));
            __56.x = (v__37.x-v__38.x);
            __56.y = (v__37.y-v__38.y);
          float2 v__39 = make_float2(broadcast_var_6, broadcast_var_6);
          __55.x = (__56.x*v__39.x);
          __55.y = (__56.y*v__39.y);
        __54.x = exp2f(__55.x);
        __54.y = exp2f(__55.y);
        __52.x = (__53.x*__54.x);
        __52.y = (__53.y*__54.y);
      ((half2*)(&__51))[0] = __float22half2_rn(((float2*)(&__52))[0]);
      *(uint1*)(cb_local_v1 + (i_37 * 2)) = __51;
    }
  }
  if ((((((int)blockIdx.y) * 2) + 1) % 3) == 0) {
    #pragma unroll
    for (int i_38 = 0; i_38 < 32; ++i_38) {
      uint1 __57;
      float2 __58;
        float2 __59;
        uint1 v__40 = *(uint1*)(cb_local_v0 + (i_38 * 2));
        ((float2*)(&__59))[0] = __half22float2(((half2*)(&v__40))[0]);
        float2 v__41 = *(float2*)(dt_local_v0 + (((i_38 >> 3) * 4) + (((i_38 & 3) >> 1) * 2)));
        __58.x = (__59.x*v__41.x);
        __58.y = (__59.y*v__41.y);
      ((half2*)(&__57))[0] = __float22half2_rn(((float2*)(&__58))[0]);
      *(uint1*)(cb_local_v0 + (i_38 * 2)) = __57;
    }
  } else {
    #pragma unroll
    for (int i_39 = 0; i_39 < 32; ++i_39) {
      uint1 __60;
      float2 __61;
        float2 __62;
        uint1 v__42 = *(uint1*)(cb_local_v1 + (i_39 * 2));
        ((float2*)(&__62))[0] = __half22float2(((half2*)(&v__42))[0]);
        float2 v__43 = *(float2*)(dt_local_v1 + (((i_39 >> 3) * 4) + (((i_39 & 3) >> 1) * 2)));
        __61.x = (__62.x*v__43.x);
        __61.y = (__62.y*v__43.y);
      ((half2*)(&__60))[0] = __float22half2_rn(((float2*)(&__61))[0]);
      *(uint1*)(cb_local_v1 + (i_39 * 2)) = __60;
    }
  }
  if ((((((int)blockIdx.y) * 2) + 1) % 3) == 0) {
    #pragma unroll
    for (int i_40 = 0; i_40 < 64; ++i_40) {
      half_t condval_4;
      if ((((((((((int)blockIdx.y) * 128) + ((i_40 >> 4) * 16)) + (((i_40 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_40 & 1)) + 64) <= (((((((int)blockIdx.y) * 128) + (((i_40 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_40 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_4 = cb_local_v0[i_40];
      } else {
        condval_4 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      cb_local_v0[i_40] = condval_4;
    }
  } else {
    #pragma unroll
    for (int i_41 = 0; i_41 < 64; ++i_41) {
      half_t condval_5;
      if ((((((((((int)blockIdx.y) * 128) + ((i_41 >> 4) * 16)) + (((i_41 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_41 & 1)) + 64) <= (((((((int)blockIdx.y) * 128) + (((i_41 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_41 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_5 = cb_local_v1[i_41];
      } else {
        condval_5 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      cb_local_v1[i_41] = condval_5;
    }
  }
  overlap_plan_mbar[6].wait(((int)blockIdx.y));
  if ((((((int)blockIdx.y) * 2) + 1) % 3) == 0) {
    {
      tl::GmmaDescriptor desc_b_5;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_5, (&(((half_t*)x_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_5, 8192);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_42 = 0; i_42 < 2; ++i_42) {
        #pragma unroll
        for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v0 + ((ki_5 * 16) + (i_42 * 8))), uint64_t(desc_b_5 + ((ki_5 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_42 * 32)), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 32);
    }
  } else {
    {
      tl::GmmaDescriptor desc_b_6;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_6, (&(((half_t*)x_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_6, 8192);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_43 = 0; i_43 < 2; ++i_43) {
        #pragma unroll
        for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v1 + ((ki_6 * 16) + (i_43 * 8))), uint64_t(desc_b_6 + ((ki_6 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_43 * 32)), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 32);
    }
  }
  overlap_plan_mbar[12].arrive();
  d_local[0] = ((float)D[((int)blockIdx.x)]);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[7].arrive_and_expect_tx(16384);
    tl::tma_load(X_desc_1, overlap_plan_mbar[7], (&(((half_t*)residual_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + (((int)blockIdx.y) * 128)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[7].wait(0);
  #pragma unroll
  for (int i_44 = 0; i_44 < 32; ++i_44) {
    half_t residual_shared_local_cast_8[2];
    *(uint1*)(residual_shared_local_cast_8 + 0) = *(uint1*)(((half_t*)residual_shared) + (((((((((i_44 >> 4) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_44 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_44 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_44 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_44 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __63;
    uint1 v__44 = *(uint1*)(residual_shared_local_cast_8 + 0);
    ((float2*)(&__63))[0] = __half22float2(((half2*)(&v__44))[0]);
    *(float2*)(residual_local + (i_44 * 2)) = __63;
  }
  #pragma unroll
  for (int i_45 = 0; i_45 < 64; ++i_45) {
    acc[i_45] = (acc[i_45] + (residual_local[i_45] * d_local[0]));
  }
  __syncthreads();
  #pragma unroll
  for (int i_46 = 0; i_46 < 8; ++i_46) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[(((((i_46 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_46 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_46 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc[(i_46 * 8)]), ((half_t)acc[((i_46 * 8) + 1)])), __pack_half2(((half_t)acc[((i_46 * 8) + 2)]), ((half_t)acc[((i_46 * 8) + 3)])), __pack_half2(((half_t)acc[((i_46 * 8) + 4)]), ((half_t)acc[((i_46 * 8) + 5)])), __pack_half2(((half_t)acc[((i_46 * 8) + 6)]), ((half_t)acc[((i_46 * 8) + 7)])));
  }
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    __syncthreads();
    tl::tma_store(Output_desc, (&(((half_t*)acc_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + (((int)blockIdx.y) * 128)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

