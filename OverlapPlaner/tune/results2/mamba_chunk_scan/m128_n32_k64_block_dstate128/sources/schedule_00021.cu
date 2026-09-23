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
  void* cb_shared = ((void*)((char*)buf_dyn_shmem + 40960));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 57344));
  void* da_k_shared = ((void*)((char*)buf_dyn_shmem + 69632));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 70656));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[15];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float da_m_local[4];
  float acc[32];
  float scale_m[4];
  half_t cb_local_v0[64];
  float da_k_local[16];
  float dt_local_v0[16];
  half_t cb_local_v1[64];
  float dt_local_v1[16];
  half_t cb_local_v2[64];
  float d_local[1];
  float residual_local[32];
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
    overlap_plan_mbar[8].init(1);
    overlap_plan_mbar[9].init(128);
    overlap_plan_mbar[10].init(128);
    overlap_plan_mbar[11].init(128);
    overlap_plan_mbar[12].init(128);
    overlap_plan_mbar[13].init(128);
    overlap_plan_mbar[14].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  ((half_t*)da_m_shared)[((int)threadIdx.x)] = DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + ((((int)blockIdx.y) >> 1) * 128)) + ((int)threadIdx.x))];
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    da_m_local[i] = ((float)((half_t*)da_m_shared)[(((((i >> 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
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
    tl::tma_load(C_desc, overlap_plan_mbar[0], (&(((half_t*)c_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 128)), 0, (((int)blockIdx.z) & 7));
    tl::tma_load(C_desc, overlap_plan_mbar[0], (&(((half_t*)c_shared)[8192])), 64, (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 128)), 0, (((int)blockIdx.z) & 7));
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
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a + (((((ki >> 2) * 16384) + (i_3 * 8192)) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 4096) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc + (i_3 * 16))), 1);
      }
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 32; ++i_4) {
    acc[i_4] = (acc[i_4] * scale_m[(((i_4 >> 4) * 2) + ((i_4 & 3) >> 1))]);
  }
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[2].arrive_and_expect_tx(16384);
    tl::tma_load(CB_desc, overlap_plan_mbar[2], (&(((half_t*)cb_shared)[0])), 0, ((((int)blockIdx.y) >> 1) * 128), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[2].wait(0);
  #pragma unroll
  for (int i_5 = 0; i_5 < 8; ++i_5) {
    tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[(((((i_5 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((i_5 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_5 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v0[(i_5 * 8)])));
  }
  overlap_plan_mbar[9].arrive();
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
    *(float2*)(da_k_local + (i_6 * 2)) = __1;
  }
  overlap_plan_mbar[10].arrive();
  #pragma unroll
  for (int i_7 = 0; i_7 < 32; ++i_7) {
    float broadcast_var_1 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    uint1 __2;
    float2 __3;
      float2 __4;
      uint1 v__1 = *(uint1*)(cb_local_v0 + (i_7 * 2));
      ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__1))[0]);
      float2 __5;
      float2 __6;
        float2 __7;
          float2 v__2 = make_float2(da_m_local[((((i_7 & 7) >> 2) * 2) + (i_7 & 1))], da_m_local[((((i_7 & 7) >> 2) * 2) + (i_7 & 1))]);
          float2 v__3 = *(float2*)(da_k_local + (((i_7 >> 3) * 4) + (((i_7 & 3) >> 1) * 2)));
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
    *(uint1*)(cb_local_v0 + (i_7 * 2)) = __2;
  }
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[4].arrive_and_expect_tx(128);
    tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[4], 128);
  }
  overlap_plan_mbar[4].wait(0);
  __syncthreads();
  #pragma unroll
  for (int i_8 = 0; i_8 < 8; ++i_8) {
    half_t dt_shared_local_cast_1[2];
    *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_8 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __8;
    uint1 v__5 = *(uint1*)(dt_shared_local_cast_1 + 0);
    ((float2*)(&__8))[0] = __half22float2(((half2*)(&v__5))[0]);
    *(float2*)(dt_local_v0 + (i_8 * 2)) = __8;
  }
  overlap_plan_mbar[11].arrive();
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[5].arrive_and_expect_tx(4096);
    tl::tma_load(X_desc, overlap_plan_mbar[5], (&(((half_t*)x_shared)[0])), ((((int)blockIdx.y) & 1) * 32), ((((int)blockIdx.z) >> 3) * 256), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[9].wait(0);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[2].arrive_and_expect_tx(16384);
    tl::tma_load(CB_desc, overlap_plan_mbar[2], (&(((half_t*)cb_shared)[0])), 64, ((((int)blockIdx.y) >> 1) * 128), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[2].wait(1);
  #pragma unroll
  for (int i_9 = 0; i_9 < 8; ++i_9) {
    tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[(((((i_9 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((i_9 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_9 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v1[(i_9 * 8)])));
  }
  overlap_plan_mbar[9].arrive();
  overlap_plan_mbar[10].wait(0);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[3].arrive_and_expect_tx(128);
    tl::tma_load((&(((half_t*)da_k_shared)[0])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 64)])), overlap_plan_mbar[3], 128);
  }
  overlap_plan_mbar[3].wait(1);
  __syncthreads();
  #pragma unroll
  for (int i_10 = 0; i_10 < 8; ++i_10) {
    half_t da_k_shared_local_cast_2[2];
    *(uint1*)(da_k_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)da_k_shared) + ((i_10 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __9;
    uint1 v__6 = *(uint1*)(da_k_shared_local_cast_2 + 0);
    ((float2*)(&__9))[0] = __half22float2(((half2*)(&v__6))[0]);
    *(float2*)(da_k_local + (i_10 * 2)) = __9;
  }
  overlap_plan_mbar[10].arrive();
  #pragma unroll
  for (int i_11 = 0; i_11 < 32; ++i_11) {
    float broadcast_var_2 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    uint1 __10;
    float2 __11;
      float2 __12;
      uint1 v__7 = *(uint1*)(cb_local_v1 + (i_11 * 2));
      ((float2*)(&__12))[0] = __half22float2(((half2*)(&v__7))[0]);
      float2 __13;
      float2 __14;
        float2 __15;
          float2 v__8 = make_float2(da_m_local[((((i_11 & 7) >> 2) * 2) + (i_11 & 1))], da_m_local[((((i_11 & 7) >> 2) * 2) + (i_11 & 1))]);
          float2 v__9 = *(float2*)(da_k_local + (((i_11 >> 3) * 4) + (((i_11 & 3) >> 1) * 2)));
          __15.x = (v__8.x-v__9.x);
          __15.y = (v__8.y-v__9.y);
        float2 v__10 = make_float2(broadcast_var_2, broadcast_var_2);
        __14.x = (__15.x*v__10.x);
        __14.y = (__15.y*v__10.y);
      __13.x = exp2f(__14.x);
      __13.y = exp2f(__14.y);
      __11.x = (__12.x*__13.x);
      __11.y = (__12.y*__13.y);
    ((half2*)(&__10))[0] = __float22half2_rn(((float2*)(&__11))[0]);
    *(uint1*)(cb_local_v1 + (i_11 * 2)) = __10;
  }
  overlap_plan_mbar[11].wait(0);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[4].arrive_and_expect_tx(128);
    tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 64)])), overlap_plan_mbar[4], 128);
  }
  overlap_plan_mbar[4].wait(1);
  __syncthreads();
  #pragma unroll
  for (int i_12 = 0; i_12 < 8; ++i_12) {
    half_t dt_shared_local_cast_3[2];
    *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_12 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __16;
    uint1 v__11 = *(uint1*)(dt_shared_local_cast_3 + 0);
    ((float2*)(&__16))[0] = __half22float2(((half2*)(&v__11))[0]);
    *(float2*)(dt_local_v1 + (i_12 * 2)) = __16;
  }
  overlap_plan_mbar[11].arrive();
  #pragma unroll
  for (int i_13 = 0; i_13 < 32; ++i_13) {
    uint1 __17;
    float2 __18;
      float2 __19;
      uint1 v__12 = *(uint1*)(cb_local_v0 + (i_13 * 2));
      ((float2*)(&__19))[0] = __half22float2(((half2*)(&v__12))[0]);
      float2 v__13 = *(float2*)(dt_local_v0 + (((i_13 >> 3) * 4) + (((i_13 & 3) >> 1) * 2)));
      __18.x = (__19.x*v__13.x);
      __18.y = (__19.y*v__13.y);
    ((half2*)(&__17))[0] = __float22half2_rn(((float2*)(&__18))[0]);
    *(uint1*)(cb_local_v0 + (i_13 * 2)) = __17;
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 64; ++i_14) {
    half_t condval;
    if (((((((i_14 >> 4) * 16) + (((i_14 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_14 & 1)) <= ((((((((int)blockIdx.y) >> 1) * 128) + (((i_14 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_14 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
      condval = cb_local_v0[i_14];
    } else {
      condval = half_t(0x0p+0f/*0.000000e+00*/);
    }
    cb_local_v0[i_14] = condval;
  }
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[6].arrive_and_expect_tx(4096);
    tl::tma_load(X_desc, overlap_plan_mbar[6], (&(((half_t*)x_shared)[2048])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + 64), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
  }
  for (int ik = 0; ik < ((((int)blockIdx.y) >> 1) * 2); ++ik) {
    overlap_plan_mbar[9].wait(((ik + 1) & 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(16384);
      tl::tma_load(CB_desc, overlap_plan_mbar[2], (&(((half_t*)cb_shared)[0])), ((ik * 64) + 128), ((((int)blockIdx.y) >> 1) * 128), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[2].wait(ik);
    if (((ik + 2) % 3) == 0) {
      #pragma unroll
      for (int i_15 = 0; i_15 < 8; ++i_15) {
        tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[(((((i_15 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((i_15 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_15 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v0[(i_15 * 8)])));
      }
    } else {
      if (((ik + 2) % 3) == 1) {
        #pragma unroll
        for (int i_16 = 0; i_16 < 8; ++i_16) {
          tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[(((((i_16 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((i_16 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_16 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v1[(i_16 * 8)])));
        }
      } else {
        #pragma unroll
        for (int i_17 = 0; i_17 < 8; ++i_17) {
          tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[(((((i_17 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((i_17 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_17 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v2[(i_17 * 8)])));
        }
      }
    }
    overlap_plan_mbar[9].arrive();
    overlap_plan_mbar[10].wait(((ik + 1) & 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)da_k_shared)[0])), (&(DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 128)])), overlap_plan_mbar[3], 128);
    }
    overlap_plan_mbar[3].wait(ik);
    __syncthreads();
    #pragma unroll
    for (int i_18 = 0; i_18 < 8; ++i_18) {
      half_t da_k_shared_local_cast_4[2];
      *(uint1*)(da_k_shared_local_cast_4 + 0) = *(uint1*)(((half_t*)da_k_shared) + ((i_18 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __20;
      uint1 v__14 = *(uint1*)(da_k_shared_local_cast_4 + 0);
      ((float2*)(&__20))[0] = __half22float2(((half2*)(&v__14))[0]);
      *(float2*)(da_k_local + (i_18 * 2)) = __20;
    }
    overlap_plan_mbar[10].arrive();
    if (((ik + 2) % 3) == 0) {
      #pragma unroll
      for (int i_19 = 0; i_19 < 32; ++i_19) {
        float broadcast_var_3 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
        uint1 __21;
        float2 __22;
          float2 __23;
          uint1 v__15 = *(uint1*)(cb_local_v0 + (i_19 * 2));
          ((float2*)(&__23))[0] = __half22float2(((half2*)(&v__15))[0]);
          float2 __24;
          float2 __25;
            float2 __26;
              float2 v__16 = make_float2(da_m_local[((((i_19 & 7) >> 2) * 2) + (i_19 & 1))], da_m_local[((((i_19 & 7) >> 2) * 2) + (i_19 & 1))]);
              float2 v__17 = *(float2*)(da_k_local + (((i_19 >> 3) * 4) + (((i_19 & 3) >> 1) * 2)));
              __26.x = (v__16.x-v__17.x);
              __26.y = (v__16.y-v__17.y);
            float2 v__18 = make_float2(broadcast_var_3, broadcast_var_3);
            __25.x = (__26.x*v__18.x);
            __25.y = (__26.y*v__18.y);
          __24.x = exp2f(__25.x);
          __24.y = exp2f(__25.y);
          __22.x = (__23.x*__24.x);
          __22.y = (__23.y*__24.y);
        ((half2*)(&__21))[0] = __float22half2_rn(((float2*)(&__22))[0]);
        *(uint1*)(cb_local_v0 + (i_19 * 2)) = __21;
      }
    } else {
      if (((ik + 2) % 3) == 1) {
        #pragma unroll
        for (int i_20 = 0; i_20 < 64; ++i_20) {
          cb_local_v1[i_20] = ((half_t)(((float)cb_local_v1[i_20]) * exp2f(((da_m_local[((((i_20 & 15) >> 3) * 2) + ((i_20 & 3) >> 1))] - da_k_local[((((i_20 >> 4) * 4) + (((i_20 & 7) >> 2) * 2)) + (i_20 & 1))]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/))));
        }
      } else {
        #pragma unroll
        for (int i_21 = 0; i_21 < 32; ++i_21) {
          float broadcast_var_4 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
          uint1 __27;
          float2 __28;
            float2 __29;
            uint1 v__19 = *(uint1*)(cb_local_v2 + (i_21 * 2));
            ((float2*)(&__29))[0] = __half22float2(((half2*)(&v__19))[0]);
            float2 __30;
            float2 __31;
              float2 __32;
                float2 v__20 = make_float2(da_m_local[((((i_21 & 7) >> 2) * 2) + (i_21 & 1))], da_m_local[((((i_21 & 7) >> 2) * 2) + (i_21 & 1))]);
                float2 v__21 = *(float2*)(da_k_local + (((i_21 >> 3) * 4) + (((i_21 & 3) >> 1) * 2)));
                __32.x = (v__20.x-v__21.x);
                __32.y = (v__20.y-v__21.y);
              float2 v__22 = make_float2(broadcast_var_4, broadcast_var_4);
              __31.x = (__32.x*v__22.x);
              __31.y = (__32.y*v__22.y);
            __30.x = exp2f(__31.x);
            __30.y = exp2f(__31.y);
            __28.x = (__29.x*__30.x);
            __28.y = (__29.y*__30.y);
          ((half2*)(&__27))[0] = __float22half2_rn(((float2*)(&__28))[0]);
          *(uint1*)(cb_local_v2 + (i_21 * 2)) = __27;
        }
      }
    }
    overlap_plan_mbar[11].wait(((ik + 1) & 1));
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 128)])), overlap_plan_mbar[4], 128);
    }
    overlap_plan_mbar[4].wait(ik);
    __syncthreads();
    if (ik == 0) {
      #pragma unroll
      for (int i_22 = 0; i_22 < 8; ++i_22) {
        half_t dt_shared_local_cast_5[2];
        *(uint1*)(dt_shared_local_cast_5 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_22 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __33;
        uint1 v__23 = *(uint1*)(dt_shared_local_cast_5 + 0);
        ((float2*)(&__33))[0] = __half22float2(((half2*)(&v__23))[0]);
        *(float2*)(dt_local_v0 + (i_22 * 2)) = __33;
      }
    } else {
      #pragma unroll
      for (int i_23 = 0; i_23 < 8; ++i_23) {
        half_t dt_shared_local_cast_6[2];
        *(uint1*)(dt_shared_local_cast_6 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_23 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __34;
        uint1 v__24 = *(uint1*)(dt_shared_local_cast_6 + 0);
        ((float2*)(&__34))[0] = __half22float2(((half2*)(&v__24))[0]);
        *(float2*)(dt_local_v1 + (i_23 * 2)) = __34;
      }
    }
    overlap_plan_mbar[11].arrive();
    if (ik == 0) {
      #pragma unroll
      for (int i_24 = 0; i_24 < 32; ++i_24) {
        uint1 __35;
        float2 __36;
          float2 __37;
          uint1 v__25 = *(uint1*)(cb_local_v1 + (i_24 * 2));
          ((float2*)(&__37))[0] = __half22float2(((half2*)(&v__25))[0]);
          float2 v__26 = *(float2*)(dt_local_v1 + (((i_24 >> 3) * 4) + (((i_24 & 3) >> 1) * 2)));
          __36.x = (__37.x*v__26.x);
          __36.y = (__37.y*v__26.y);
        ((half2*)(&__35))[0] = __float22half2_rn(((float2*)(&__36))[0]);
        *(uint1*)(cb_local_v1 + (i_24 * 2)) = __35;
      }
    } else {
      #pragma unroll
      for (int i_25 = 0; i_25 < 32; ++i_25) {
        uint1 __38;
        float2 __39;
          float2 __40;
          uint1 v__27 = *(uint1*)(cb_local_v2 + (i_25 * 2));
          ((float2*)(&__40))[0] = __half22float2(((half2*)(&v__27))[0]);
          float2 v__28 = *(float2*)(dt_local_v0 + (((i_25 >> 3) * 4) + (((i_25 & 3) >> 1) * 2)));
          __39.x = (__40.x*v__28.x);
          __39.y = (__40.y*v__28.y);
        ((half2*)(&__38))[0] = __float22half2_rn(((float2*)(&__39))[0]);
        *(uint1*)(cb_local_v2 + (i_25 * 2)) = __38;
      }
    }
    if (ik == 0) {
      #pragma unroll
      for (int i_26 = 0; i_26 < 64; ++i_26) {
        half_t condval_1;
        if ((((((((ik * 64) + ((i_26 >> 4) * 16)) + (((i_26 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_26 & 1)) + 64) <= ((((((((int)blockIdx.y) >> 1) * 128) + (((i_26 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_26 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval_1 = cb_local_v1[i_26];
        } else {
          condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local_v1[i_26] = condval_1;
      }
    } else {
      #pragma unroll
      for (int i_27 = 0; i_27 < 64; ++i_27) {
        half_t condval_2;
        if ((((((((ik * 64) + ((i_27 >> 4) * 16)) + (((i_27 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_27 & 1)) + 64) <= ((((((((int)blockIdx.y) >> 1) * 128) + (((i_27 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_27 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval_2 = cb_local_v2[i_27];
        } else {
          condval_2 = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local_v2[i_27] = condval_2;
      }
    }
    if (ik == 1) {
      overlap_plan_mbar[(ik + 11)].wait(0);
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[(((ik + 2) % 3) + 5)].arrive_and_expect_tx(4096);
      tl::tma_load(X_desc, overlap_plan_mbar[(((ik + 2) % 3) + 5)], (&(((half_t*)x_shared)[(((ik + 2) % 3) * 2048)])), ((((int)blockIdx.y) & 1) * 32), ((((((int)blockIdx.z) >> 3) * 256) + (ik * 64)) + 128), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[(ik + 5)].wait(0);
    if (ik == 0) {
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_1, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, (ik * 4096));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int i_28 = 0; i_28 < 2; ++i_28) {
          #pragma unroll
          for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
            tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v0 + ((ki_1 * 16) + (i_28 * 8))), uint64_t(desc_b_1 + ((ki_1 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_28 * 16)), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 32);
      }
    } else {
      {
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_2, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_2, (ik * 4096));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int i_29 = 0; i_29 < 2; ++i_29) {
          #pragma unroll
          for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
            tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v1 + ((ki_2 * 16) + (i_29 * 8))), uint64_t(desc_b_2 + ((ki_2 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_29 * 16)), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 32);
      }
    }
    overlap_plan_mbar[(ik + 12)].arrive();
  }
  if ((((int)blockIdx.y) >> 1) == 0) {
    #pragma unroll
    for (int i_30 = 0; i_30 < 32; ++i_30) {
      uint1 __41;
      float2 __42;
        float2 __43;
        uint1 v__29 = *(uint1*)(cb_local_v1 + (i_30 * 2));
        ((float2*)(&__43))[0] = __half22float2(((half2*)(&v__29))[0]);
        float2 v__30 = *(float2*)(dt_local_v1 + (((i_30 >> 3) * 4) + (((i_30 & 3) >> 1) * 2)));
        __42.x = (__43.x*v__30.x);
        __42.y = (__43.y*v__30.y);
      ((half2*)(&__41))[0] = __float22half2_rn(((float2*)(&__42))[0]);
      *(uint1*)(cb_local_v1 + (i_30 * 2)) = __41;
    }
  } else {
    #pragma unroll
    for (int i_31 = 0; i_31 < 32; ++i_31) {
      uint1 __44;
      float2 __45;
        float2 __46;
        uint1 v__31 = *(uint1*)(cb_local_v0 + (i_31 * 2));
        ((float2*)(&__46))[0] = __half22float2(((half2*)(&v__31))[0]);
        float2 v__32 = *(float2*)(dt_local_v1 + (((i_31 >> 3) * 4) + (((i_31 & 3) >> 1) * 2)));
        __45.x = (__46.x*v__32.x);
        __45.y = (__46.y*v__32.y);
      ((half2*)(&__44))[0] = __float22half2_rn(((float2*)(&__45))[0]);
      *(uint1*)(cb_local_v0 + (i_31 * 2)) = __44;
    }
  }
  if (((((((int)blockIdx.y) >> 1) * 2) + 1) % 3) == 0) {
    #pragma unroll
    for (int i_32 = 0; i_32 < 64; ++i_32) {
      half_t condval_3;
      if (((((((((((int)blockIdx.y) >> 1) * 128) + ((i_32 >> 4) * 16)) + (((i_32 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_32 & 1)) + 64) <= ((((((((int)blockIdx.y) >> 1) * 128) + (((i_32 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_32 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_3 = cb_local_v0[i_32];
      } else {
        condval_3 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      cb_local_v0[i_32] = condval_3;
    }
  } else {
    #pragma unroll
    for (int i_33 = 0; i_33 < 64; ++i_33) {
      half_t condval_4;
      if (((((((((((int)blockIdx.y) >> 1) * 128) + ((i_33 >> 4) * 16)) + (((i_33 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_33 & 1)) + 64) <= ((((((((int)blockIdx.y) >> 1) * 128) + (((i_33 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_33 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_4 = cb_local_v1[i_33];
      } else {
        condval_4 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      cb_local_v1[i_33] = condval_4;
    }
  }
  overlap_plan_mbar[(((((int)blockIdx.y) >> 1) * 2) + 5)].wait(0);
  __syncthreads();
  if ((((int)blockIdx.y) >> 1) == 0) {
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_3, (&(((half_t*)x_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_3, 0);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_34 = 0; i_34 < 2; ++i_34) {
        #pragma unroll
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v0 + ((ki_3 * 16) + (i_34 * 8))), uint64_t(desc_b_3 + ((ki_3 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_34 * 16)), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 32);
    }
  } else {
    {
      tl::GmmaDescriptor desc_b_4;
      tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_4, (&(((half_t*)x_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_4, 8192);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v2 + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_35 = 0; i_35 < 2; ++i_35) {
        #pragma unroll
        for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v2 + ((ki_4 * 16) + (i_35 * 8))), uint64_t(desc_b_4 + ((ki_4 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_35 * 16)), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v2 + 0), 32);
    }
  }
  overlap_plan_mbar[(((((int)blockIdx.y) >> 1) * 2) + 12)].arrive();
  overlap_plan_mbar[(((((((int)blockIdx.y) >> 1) * 2) + 1) % 3) + 5)].wait(((((((int)blockIdx.y) >> 1) * 2) + 1) / 3));
  if (((((((int)blockIdx.y) >> 1) * 2) + 1) % 3) == 0) {
    {
      tl::GmmaDescriptor desc_b_5;
      tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_5, (&(((half_t*)x_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_5, (((((((int)blockIdx.y) >> 1) * 2) + 1) % 3) * 4096));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_36 = 0; i_36 < 2; ++i_36) {
        #pragma unroll
        for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v0 + ((ki_5 * 16) + (i_36 * 8))), uint64_t(desc_b_5 + ((ki_5 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_36 * 16)), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 32);
    }
  } else {
    {
      tl::GmmaDescriptor desc_b_6;
      tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_6, (&(((half_t*)x_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_6, (((((((int)blockIdx.y) >> 1) * 2) + 1) % 3) * 4096));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_37 = 0; i_37 < 2; ++i_37) {
        #pragma unroll
        for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v1 + ((ki_6 * 16) + (i_37 * 8))), uint64_t(desc_b_6 + ((ki_6 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_37 * 16)), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 32);
    }
  }
  overlap_plan_mbar[(((((((int)blockIdx.y) >> 1) * 2) + 1) % 3) + 12)].arrive();
  d_local[0] = ((float)D[((int)blockIdx.x)]);
  if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[8].arrive_and_expect_tx(8192);
    tl::tma_load(X_desc_1, overlap_plan_mbar[8], (&(((half_t*)residual_shared)[0])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 128)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
  }
  overlap_plan_mbar[8].wait(0);
  #pragma unroll
  for (int i_38 = 0; i_38 < 16; ++i_38) {
    half_t residual_shared_local_cast_7[2];
    *(uint1*)(residual_shared_local_cast_7 + 0) = *(uint1*)(((half_t*)residual_shared) + ((((((((i_38 >> 3) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((i_38 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_38 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_38 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __47;
    uint1 v__33 = *(uint1*)(residual_shared_local_cast_7 + 0);
    ((float2*)(&__47))[0] = __half22float2(((half2*)(&v__33))[0]);
    *(float2*)(residual_local + (i_38 * 2)) = __47;
  }
  #pragma unroll
  for (int i_39 = 0; i_39 < 32; ++i_39) {
    acc[i_39] = (acc[i_39] + (residual_local[i_39] * d_local[0]));
  }
  __syncthreads();
  #pragma unroll
  for (int i_40 = 0; i_40 < 4; ++i_40) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[((((((i_40 >> 1) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_40 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((half_t)acc[(i_40 * 8)]), ((half_t)acc[((i_40 * 8) + 1)])), __pack_half2(((half_t)acc[((i_40 * 8) + 2)]), ((half_t)acc[((i_40 * 8) + 3)])), __pack_half2(((half_t)acc[((i_40 * 8) + 4)]), ((half_t)acc[((i_40 * 8) + 5)])), __pack_half2(((half_t)acc[((i_40 * 8) + 6)]), ((half_t)acc[((i_40 * 8) + 7)])));
  }
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    __syncthreads();
    tl::tma_store(Output_desc, (&(((half_t*)acc_shared)[0])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 128)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

