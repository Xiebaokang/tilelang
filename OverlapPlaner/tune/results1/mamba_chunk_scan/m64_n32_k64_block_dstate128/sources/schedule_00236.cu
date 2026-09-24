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
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap CB_desc, __grid_constant__ const CUtensorMap C_desc, const half_t* __restrict__ D, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Prev_desc, __grid_constant__ const CUtensorMap X_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* da_m_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* c_shared = ((void*)((char*)buf_dyn_shmem + 1024));
  void* residual_shared = ((void*)((char*)buf_dyn_shmem + 1024));
  void* prev_shared = ((void*)((char*)buf_dyn_shmem + 17408));
  void* cb_shared = ((void*)((char*)buf_dyn_shmem + 25600));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 33792));
  void* da_k_shared = ((void*)((char*)buf_dyn_shmem + 41984));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 43008));
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 44032));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[14];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float da_m_local[2];
  float acc[16];
  float scale_m[2];
  half_t cb_local_v0[32];
  float da_k_local[16];
  float dt_local_v0[16];
  half_t cb_local_v1[32];
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
    overlap_plan_mbar[8].init(128);
    overlap_plan_mbar[9].init(128);
    overlap_plan_mbar[10].init(128);
    overlap_plan_mbar[11].init(128);
    overlap_plan_mbar[12].init(128);
    overlap_plan_mbar[13].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<240>();
    if (((int)threadIdx.x) < 64) {
      ((half_t*)da_m_shared)[((int)threadIdx.x)] = DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + ((((int)blockIdx.y) >> 1) * 64)) + ((int)threadIdx.x))];
    }
    tl::__sync_thread_partial(3, 128);
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
      overlap_plan_mbar[3].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)da_k_shared)[0])), (&(DA[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[3], 128);
      overlap_plan_mbar[4].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[4], 128);
    }
    overlap_plan_mbar[2].wait(0);
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v0[(i_4 * 8)])));
    }
    overlap_plan_mbar[9].arrive();
    overlap_plan_mbar[3].wait(0);
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      half_t da_k_shared_local_cast[2];
      *(uint1*)(da_k_shared_local_cast + 0) = *(uint1*)(((half_t*)da_k_shared) + ((i_5 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __1;
      uint1 v_ = *(uint1*)(da_k_shared_local_cast + 0);
      ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
      *(float2*)(da_k_local + (i_5 * 2)) = __1;
    }
    overlap_plan_mbar[10].arrive();
    overlap_plan_mbar[4].wait(0);
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      half_t dt_shared_local_cast_1[2];
      *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_6 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __2;
      uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
      ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
      *(float2*)(dt_local_v0 + (i_6 * 2)) = __2;
    }
    overlap_plan_mbar[11].arrive();
    #pragma unroll
    for (int i_7 = 0; i_7 < 16; ++i_7) {
      float broadcast_var_1 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
      uint1 __3;
      float2 __4;
        float2 __5;
        uint1 v__2 = *(uint1*)(cb_local_v0 + (i_7 * 2));
        ((float2*)(&__5))[0] = __half22float2(((half2*)(&v__2))[0]);
        float2 __6;
        float2 __7;
          float2 __8;
            float2 v__3 = make_float2(da_m_local[(i_7 & 1)], da_m_local[(i_7 & 1)]);
            float2 v__4 = *(float2*)(da_k_local + ((i_7 >> 1) * 2));
            __8.x = (v__3.x-v__4.x);
            __8.y = (v__3.y-v__4.y);
          float2 v__5 = make_float2(broadcast_var_1, broadcast_var_1);
          __7.x = (__8.x*v__5.x);
          __7.y = (__8.y*v__5.y);
        __6.x = exp2f(__7.x);
        __6.y = exp2f(__7.y);
        __4.x = (__5.x*__6.x);
        __4.y = (__5.y*__6.y);
      ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&__4))[0]);
      *(uint1*)(cb_local_v0 + (i_7 * 2)) = __3;
    }
    if (1 < ((int)blockIdx.y)) {
      overlap_plan_mbar[9].wait(0);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(8192);
        tl::tma_load(CB_desc, overlap_plan_mbar[2], (&(((half_t*)cb_shared)[0])), 64, ((((int)blockIdx.y) >> 1) * 64), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
      }
    }
    tl::__sync_thread_partial(3, 128);
    if (1 < ((int)blockIdx.y)) {
      overlap_plan_mbar[10].wait(0);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[3].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)da_k_shared)[0])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 64)])), overlap_plan_mbar[3], 128);
      }
      overlap_plan_mbar[11].wait(0);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[4].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 64)])), overlap_plan_mbar[4], 128);
      }
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[5].arrive_and_expect_tx(4096);
      tl::tma_load(X_desc, overlap_plan_mbar[5], (&(((half_t*)x_shared)[0])), ((((int)blockIdx.y) & 1) * 32), ((((int)blockIdx.z) >> 3) * 256), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
    if (1 < ((int)blockIdx.y)) {
      overlap_plan_mbar[2].wait(1);
      #pragma unroll
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_8 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v1[(i_8 * 8)])));
      }
      overlap_plan_mbar[9].arrive();
    }
    tl::__sync_thread_partial(3, 128);
    if (1 < ((int)blockIdx.y)) {
      overlap_plan_mbar[3].wait(1);
      #pragma unroll
      for (int i_9 = 0; i_9 < 8; ++i_9) {
        half_t da_k_shared_local_cast_2[2];
        *(uint1*)(da_k_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)da_k_shared) + ((i_9 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __9;
        uint1 v__6 = *(uint1*)(da_k_shared_local_cast_2 + 0);
        ((float2*)(&__9))[0] = __half22float2(((half2*)(&v__6))[0]);
        *(float2*)(da_k_local + (i_9 * 2)) = __9;
      }
      overlap_plan_mbar[10].arrive();
      overlap_plan_mbar[4].wait(1);
      #pragma unroll
      for (int i_10 = 0; i_10 < 8; ++i_10) {
        half_t dt_shared_local_cast_3[2];
        *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_10 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __10;
        uint1 v__7 = *(uint1*)(dt_shared_local_cast_3 + 0);
        ((float2*)(&__10))[0] = __half22float2(((half2*)(&v__7))[0]);
        *(float2*)(dt_local_v1 + (i_10 * 2)) = __10;
      }
      overlap_plan_mbar[11].arrive();
      #pragma unroll
      for (int i_11 = 0; i_11 < 16; ++i_11) {
        float broadcast_var_2 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
        uint1 __11;
        float2 __12;
          float2 __13;
          uint1 v__8 = *(uint1*)(cb_local_v1 + (i_11 * 2));
          ((float2*)(&__13))[0] = __half22float2(((half2*)(&v__8))[0]);
          float2 __14;
          float2 __15;
            float2 __16;
              float2 v__9 = make_float2(da_m_local[(i_11 & 1)], da_m_local[(i_11 & 1)]);
              float2 v__10 = *(float2*)(da_k_local + ((i_11 >> 1) * 2));
              __16.x = (v__9.x-v__10.x);
              __16.y = (v__9.y-v__10.y);
            float2 v__11 = make_float2(broadcast_var_2, broadcast_var_2);
            __15.x = (__16.x*v__11.x);
            __15.y = (__16.y*v__11.y);
          __14.x = exp2f(__15.x);
          __14.y = exp2f(__15.y);
          __12.x = (__13.x*__14.x);
          __12.y = (__13.y*__14.y);
        ((half2*)(&__11))[0] = __float22half2_rn(((float2*)(&__12))[0]);
        *(uint1*)(cb_local_v1 + (i_11 * 2)) = __11;
      }
    }
    for (int ik = 0; ik < ((((int)blockIdx.y) >> 1) - 1); ++ik) {
      overlap_plan_mbar[9].wait(((ik + 1) & 1));
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(8192);
        tl::tma_load(CB_desc, overlap_plan_mbar[2], (&(((half_t*)cb_shared)[0])), ((ik * 64) + 128), ((((int)blockIdx.y) >> 1) * 64), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[10].wait(((ik + 1) & 1));
      tl::__sync_thread_partial(3, 128);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[3].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)da_k_shared)[0])), (&(DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 128)])), overlap_plan_mbar[3], 128);
      }
      overlap_plan_mbar[11].wait(((ik + 1) & 1));
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[4].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 128)])), overlap_plan_mbar[4], 128);
      }
      if (ik == 1) {
        overlap_plan_mbar[(ik + 11)].wait(0);
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[(((ik + 1) & 1) + 5)].arrive_and_expect_tx(4096);
        tl::tma_load(X_desc, overlap_plan_mbar[(((ik + 1) & 1) + 5)], (&(((half_t*)x_shared)[(((ik + 1) & 1) * 2048)])), ((((int)blockIdx.y) & 1) * 32), ((((((int)blockIdx.z) >> 3) * 256) + (ik * 64)) + 64), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
      if (ik == 0) {
        #pragma unroll
        for (int i_12 = 0; i_12 < 16; ++i_12) {
          uint1 __17;
          float2 __18;
            float2 __19;
            uint1 v__12 = *(uint1*)(cb_local_v0 + (i_12 * 2));
            ((float2*)(&__19))[0] = __half22float2(((half2*)(&v__12))[0]);
            float2 v__13 = *(float2*)(dt_local_v0 + ((i_12 >> 1) * 2));
            __18.x = (__19.x*v__13.x);
            __18.y = (__19.y*v__13.y);
          ((half2*)(&__17))[0] = __float22half2_rn(((float2*)(&__18))[0]);
          *(uint1*)(cb_local_v0 + (i_12 * 2)) = __17;
        }
      } else {
        #pragma unroll
        for (int i_13 = 0; i_13 < 16; ++i_13) {
          uint1 __20;
          float2 __21;
            float2 __22;
            uint1 v__14 = *(uint1*)(cb_local_v1 + (i_13 * 2));
            ((float2*)(&__22))[0] = __half22float2(((half2*)(&v__14))[0]);
            float2 v__15 = *(float2*)(dt_local_v1 + ((i_13 >> 1) * 2));
            __21.x = (__22.x*v__15.x);
            __21.y = (__22.y*v__15.y);
          ((half2*)(&__20))[0] = __float22half2_rn(((float2*)(&__21))[0]);
          *(uint1*)(cb_local_v1 + (i_13 * 2)) = __20;
        }
      }
      if (ik == 0) {
        #pragma unroll
        for (int i_14 = 0; i_14 < 32; ++i_14) {
          half_t condval;
          if ((((((ik * 64) + ((i_14 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_14 & 1)) <= (((((((int)blockIdx.y) >> 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_14 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
            condval = cb_local_v0[i_14];
          } else {
            condval = half_t(0x0p+0f/*0.000000e+00*/);
          }
          cb_local_v0[i_14] = condval;
        }
      } else {
        #pragma unroll
        for (int i_15 = 0; i_15 < 32; ++i_15) {
          half_t condval_1;
          if ((((((ik * 64) + ((i_15 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_15 & 1)) <= (((((((int)blockIdx.y) >> 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_15 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
            condval_1 = cb_local_v1[i_15];
          } else {
            condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
          }
          cb_local_v1[i_15] = condval_1;
        }
      }
      overlap_plan_mbar[(ik + 5)].wait(0);
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
      overlap_plan_mbar[(ik + 12)].arrive();
      overlap_plan_mbar[2].wait(ik);
      if (ik == 0) {
        #pragma unroll
        for (int i_16 = 0; i_16 < 4; ++i_16) {
          tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_16 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v0[(i_16 * 8)])));
        }
      } else {
        #pragma unroll
        for (int i_17 = 0; i_17 < 4; ++i_17) {
          tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_17 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(cb_local_v1[(i_17 * 8)])));
        }
      }
      overlap_plan_mbar[9].arrive();
      overlap_plan_mbar[3].wait(ik);
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_18 = 0; i_18 < 8; ++i_18) {
        half_t da_k_shared_local_cast_4[2];
        *(uint1*)(da_k_shared_local_cast_4 + 0) = *(uint1*)(((half_t*)da_k_shared) + ((i_18 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __23;
        uint1 v__16 = *(uint1*)(da_k_shared_local_cast_4 + 0);
        ((float2*)(&__23))[0] = __half22float2(((half2*)(&v__16))[0]);
        *(float2*)(da_k_local + (i_18 * 2)) = __23;
      }
      overlap_plan_mbar[10].arrive();
      overlap_plan_mbar[4].wait(ik);
      if (ik == 0) {
        #pragma unroll
        for (int i_19 = 0; i_19 < 8; ++i_19) {
          half_t dt_shared_local_cast_5[2];
          *(uint1*)(dt_shared_local_cast_5 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_19 * 8) + ((((int)threadIdx.x) & 3) * 2)));
          float2 __24;
          uint1 v__17 = *(uint1*)(dt_shared_local_cast_5 + 0);
          ((float2*)(&__24))[0] = __half22float2(((half2*)(&v__17))[0]);
          *(float2*)(dt_local_v0 + (i_19 * 2)) = __24;
        }
      } else {
        #pragma unroll
        for (int i_20 = 0; i_20 < 8; ++i_20) {
          half_t dt_shared_local_cast_6[2];
          *(uint1*)(dt_shared_local_cast_6 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_20 * 8) + ((((int)threadIdx.x) & 3) * 2)));
          float2 __25;
          uint1 v__18 = *(uint1*)(dt_shared_local_cast_6 + 0);
          ((float2*)(&__25))[0] = __half22float2(((half2*)(&v__18))[0]);
          *(float2*)(dt_local_v1 + (i_20 * 2)) = __25;
        }
      }
      overlap_plan_mbar[11].arrive();
      if (ik == 0) {
        #pragma unroll
        for (int i_21 = 0; i_21 < 16; ++i_21) {
          float broadcast_var_3 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
          uint1 __26;
          float2 __27;
            float2 __28;
            uint1 v__19 = *(uint1*)(cb_local_v0 + (i_21 * 2));
            ((float2*)(&__28))[0] = __half22float2(((half2*)(&v__19))[0]);
            float2 __29;
            float2 __30;
              float2 __31;
                float2 v__20 = make_float2(da_m_local[(i_21 & 1)], da_m_local[(i_21 & 1)]);
                float2 v__21 = *(float2*)(da_k_local + ((i_21 >> 1) * 2));
                __31.x = (v__20.x-v__21.x);
                __31.y = (v__20.y-v__21.y);
              float2 v__22 = make_float2(broadcast_var_3, broadcast_var_3);
              __30.x = (__31.x*v__22.x);
              __30.y = (__31.y*v__22.y);
            __29.x = exp2f(__30.x);
            __29.y = exp2f(__30.y);
            __27.x = (__28.x*__29.x);
            __27.y = (__28.y*__29.y);
          ((half2*)(&__26))[0] = __float22half2_rn(((float2*)(&__27))[0]);
          *(uint1*)(cb_local_v0 + (i_21 * 2)) = __26;
        }
      } else {
        #pragma unroll
        for (int i_22 = 0; i_22 < 16; ++i_22) {
          float broadcast_var_4 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
          uint1 __32;
          float2 __33;
            float2 __34;
            uint1 v__23 = *(uint1*)(cb_local_v1 + (i_22 * 2));
            ((float2*)(&__34))[0] = __half22float2(((half2*)(&v__23))[0]);
            float2 __35;
            float2 __36;
              float2 __37;
                float2 v__24 = make_float2(da_m_local[(i_22 & 1)], da_m_local[(i_22 & 1)]);
                float2 v__25 = *(float2*)(da_k_local + ((i_22 >> 1) * 2));
                __37.x = (v__24.x-v__25.x);
                __37.y = (v__24.y-v__25.y);
              float2 v__26 = make_float2(broadcast_var_4, broadcast_var_4);
              __36.x = (__37.x*v__26.x);
              __36.y = (__37.y*v__26.y);
            __35.x = exp2f(__36.x);
            __35.y = exp2f(__36.y);
            __33.x = (__34.x*__35.x);
            __33.y = (__34.y*__35.y);
          ((half2*)(&__32))[0] = __float22half2_rn(((float2*)(&__33))[0]);
          *(uint1*)(cb_local_v1 + (i_22 * 2)) = __32;
        }
      }
    }
    if (4 <= ((int)blockIdx.y)) {
      overlap_plan_mbar[(((((int)blockIdx.y) & 3) >> 1) + 12)].wait(0);
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[(((((int)blockIdx.y) & 3) >> 1) + 5)].arrive_and_expect_tx(4096);
      tl::tma_load(X_desc, overlap_plan_mbar[(((((int)blockIdx.y) & 3) >> 1) + 5)], (&(((half_t*)x_shared)[(((((int)blockIdx.y) & 3) >> 1) * 2048)])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
    if (2 <= ((int)blockIdx.y)) {
      if ((((((int)blockIdx.y) >> 1) + 1) % 2) == 0) {
        #pragma unroll
        for (int i_23 = 0; i_23 < 16; ++i_23) {
          uint1 __38;
          float2 __39;
            float2 __40;
            uint1 v__27 = *(uint1*)(cb_local_v0 + (i_23 * 2));
            ((float2*)(&__40))[0] = __half22float2(((half2*)(&v__27))[0]);
            float2 v__28 = *(float2*)(dt_local_v0 + ((i_23 >> 1) * 2));
            __39.x = (__40.x*v__28.x);
            __39.y = (__40.y*v__28.y);
          ((half2*)(&__38))[0] = __float22half2_rn(((float2*)(&__39))[0]);
          *(uint1*)(cb_local_v0 + (i_23 * 2)) = __38;
        }
      } else {
        #pragma unroll
        for (int i_24 = 0; i_24 < 16; ++i_24) {
          uint1 __41;
          float2 __42;
            float2 __43;
            uint1 v__29 = *(uint1*)(cb_local_v1 + (i_24 * 2));
            ((float2*)(&__43))[0] = __half22float2(((half2*)(&v__29))[0]);
            float2 v__30 = *(float2*)(dt_local_v1 + ((i_24 >> 1) * 2));
            __42.x = (__43.x*v__30.x);
            __42.y = (__43.y*v__30.y);
          ((half2*)(&__41))[0] = __float22half2_rn(((float2*)(&__42))[0]);
          *(uint1*)(cb_local_v1 + (i_24 * 2)) = __41;
        }
      }
      overlap_plan_mbar[((((((int)blockIdx.y) >> 1) + 1) & 1) + 5)].wait(((((((int)blockIdx.y) >> 1) + 3) & 3) >> 1));
      if ((((((int)blockIdx.y) >> 1) + 1) % 2) == 0) {
        {
          tl::GmmaDescriptor desc_b_3;
          tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_3, (&(((half_t*)x_shared)[0])));
          tl::increase_descriptor_offset<int>(desc_b_3, ((((((int)blockIdx.y) >> 1) + 1) & 1) * 4096));
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
        {
          tl::GmmaDescriptor desc_b_4;
          tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_4, (&(((half_t*)x_shared)[0])));
          tl::increase_descriptor_offset<int>(desc_b_4, ((((((int)blockIdx.y) >> 1) + 1) & 1) * 4096));
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
      }
      overlap_plan_mbar[((((((int)blockIdx.y) >> 1) + 1) & 1) + 12)].arrive();
    }
    if (((((int)blockIdx.y) & 3) >> 1) == 0) {
      #pragma unroll
      for (int i_25 = 0; i_25 < 16; ++i_25) {
        uint1 __44;
        float2 __45;
          float2 __46;
          uint1 v__31 = *(uint1*)(cb_local_v0 + (i_25 * 2));
          ((float2*)(&__46))[0] = __half22float2(((half2*)(&v__31))[0]);
          float2 v__32 = *(float2*)(dt_local_v0 + ((i_25 >> 1) * 2));
          __45.x = (__46.x*v__32.x);
          __45.y = (__46.y*v__32.y);
        ((half2*)(&__44))[0] = __float22half2_rn(((float2*)(&__45))[0]);
        *(uint1*)(cb_local_v0 + (i_25 * 2)) = __44;
      }
    } else {
      #pragma unroll
      for (int i_26 = 0; i_26 < 16; ++i_26) {
        uint1 __47;
        float2 __48;
          float2 __49;
          uint1 v__33 = *(uint1*)(cb_local_v1 + (i_26 * 2));
          ((float2*)(&__49))[0] = __half22float2(((half2*)(&v__33))[0]);
          float2 v__34 = *(float2*)(dt_local_v1 + ((i_26 >> 1) * 2));
          __48.x = (__49.x*v__34.x);
          __48.y = (__49.y*v__34.y);
        ((half2*)(&__47))[0] = __float22half2_rn(((float2*)(&__48))[0]);
        *(uint1*)(cb_local_v1 + (i_26 * 2)) = __47;
      }
    }
    if (((((int)blockIdx.y) & 3) >> 1) == 0) {
      #pragma unroll
      for (int i_27 = 0; i_27 < 32; ++i_27) {
        half_t condval_2;
        if (((((((((int)blockIdx.y) >> 1) * 64) + ((i_27 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_27 & 1)) <= (((((((int)blockIdx.y) >> 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_27 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval_2 = cb_local_v0[i_27];
        } else {
          condval_2 = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local_v0[i_27] = condval_2;
      }
    } else {
      #pragma unroll
      for (int i_28 = 0; i_28 < 32; ++i_28) {
        half_t condval_3;
        if (((((((((int)blockIdx.y) >> 1) * 64) + ((i_28 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_28 & 1)) <= (((((((int)blockIdx.y) >> 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_28 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval_3 = cb_local_v1[i_28];
        } else {
          condval_3 = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local_v1[i_28] = condval_3;
      }
    }
    overlap_plan_mbar[(((((int)blockIdx.y) & 3) >> 1) + 5)].wait((((int)blockIdx.y) >> 2));
    if (((((int)blockIdx.y) & 3) >> 1) == 0) {
      {
        tl::GmmaDescriptor desc_b_5;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_5, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_5, 0);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v0 + (ki_5 * 8)), uint64_t(desc_b_5 + ((ki_5 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v0 + 0), 16);
      }
    } else {
      {
        tl::GmmaDescriptor desc_b_6;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_6, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_6, 4096);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local_v1 + (ki_6 * 8)), uint64_t(desc_b_6 + ((ki_6 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local_v1 + 0), 16);
      }
    }
    overlap_plan_mbar[(((((int)blockIdx.y) & 3) >> 1) + 12)].arrive();
    d_local[0] = ((float)D[((int)blockIdx.x)]);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[7].arrive_and_expect_tx(4096);
      tl::tma_load(X_desc, overlap_plan_mbar[7], (&(((half_t*)residual_shared)[0])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[7].wait(0);
    #pragma unroll
    for (int i_29 = 0; i_29 < 8; ++i_29) {
      half_t residual_shared_local_cast_7[2];
      *(uint1*)(residual_shared_local_cast_7 + 0) = *(uint1*)(((half_t*)residual_shared) + (((((((((int)threadIdx.x) >> 5) * 512) + ((i_29 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_29 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_29 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __50;
      uint1 v__35 = *(uint1*)(residual_shared_local_cast_7 + 0);
      ((float2*)(&__50))[0] = __half22float2(((half2*)(&v__35))[0]);
      *(float2*)(residual_local + (i_29 * 2)) = __50;
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 16; ++i_30) {
      acc[i_30] = (acc[i_30] + (residual_local[i_30] * d_local[0]));
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 2; ++i_31) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_31) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((half_t)acc[(i_31 * 8)]), ((half_t)acc[((i_31 * 8) + 1)])), __pack_half2(((half_t)acc[((i_31 * 8) + 2)]), ((half_t)acc[((i_31 * 8) + 3)])), __pack_half2(((half_t)acc[((i_31 * 8) + 4)]), ((half_t)acc[((i_31 * 8) + 5)])), __pack_half2(((half_t)acc[((i_31 * 8) + 6)]), ((half_t)acc[((i_31 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[8].arrive();
  } else {
    tl::warpgroup_reg_dealloc<24>();
    overlap_plan_mbar[8].wait(0);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store(Output_desc, (&(((half_t*)acc_shared)[0])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

