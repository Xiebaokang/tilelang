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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap B_desc, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap X_desc);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap B_desc, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap X_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  void* xt_local_wsp_handoff_4 = ((void*)((char*)buf_dyn_shmem + 73728));
  void* scale_wsp_handoff_3 = ((void*)((char*)buf_dyn_shmem + 90112));
  void* da_shared = ((void*)((char*)buf_dyn_shmem + 91136));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 92160));
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 93184));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[21];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  half_t x_local_v0[32];
  half_t x_local_v1[32];
  float scale_v0[16];
  half_t xt_local_v0[32];
  float scale_v1[16];
  half_t xt_local_v1[32];
  float da_last[1];
  float acc[64];
  float da_local[16];
  float dt_local[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(X_desc);
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(1);
    overlap_plan_mbar[4].init(1);
    overlap_plan_mbar[5].init(128);
    overlap_plan_mbar[6].init(128);
    overlap_plan_mbar[7].init(128);
    overlap_plan_mbar[8].init(128);
    overlap_plan_mbar[9].init(1);
    overlap_plan_mbar[10].init(1);
    overlap_plan_mbar[11].init(1);
    overlap_plan_mbar[12].init(128);
    overlap_plan_mbar[13].init(128);
    overlap_plan_mbar[14].init(128);
    overlap_plan_mbar[15].init(128);
    overlap_plan_mbar[16].init(128);
    overlap_plan_mbar[17].init(128);
    overlap_plan_mbar[18].init(128);
    overlap_plan_mbar[19].init(128);
    overlap_plan_mbar[20].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<72>();
    overlap_plan_mbar[0].wait(0);
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      x_local_v0[i] = ((half_t*)x_shared)[((((((((i >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i & 3) >> 1) + (i & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
    }
    overlap_plan_mbar[13].arrive();
    for (int ik = 0; ik < 1; ++ik) {
      overlap_plan_mbar[1].wait(0);
      #pragma unroll
      for (int i_1 = 0; i_1 < 32; ++i_1) {
        x_local_v1[i_1] = ((half_t*)x_shared)[(((((((((i_1 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_1 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_1 & 3) >> 1) + (i_1 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 4096)];
      }
      overlap_plan_mbar[14].arrive();
      overlap_plan_mbar[5].wait(0);
      #pragma unroll
      for (int i_2 = 0; i_2 < 8; ++i_2) {
        *(float2*)(scale_v0 + (i_2 * 2)) = *(float2*)(((float*)scale_wsp_handoff_3) + ((i_2 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 32; ++i_3) {
        xt_local_v0[i_3] = ((half_t)(((float)x_local_v0[((((i_3 >> 2) * 4) + ((i_3 & 1) * 2)) + ((i_3 & 3) >> 1))]) * scale_v0[(((i_3 >> 2) * 2) + (i_3 & 1))]));
      }
      #pragma unroll
      for (int i_4 = 0; i_4 < 16; ++i_4) {
        *(uint1*)(((half_t*)xt_local_wsp_handoff_4) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_4 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_4 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(xt_local_v0 + (i_4 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[7].arrive();
      overlap_plan_mbar[2].wait(0);
      #pragma unroll
      for (int i_5 = 0; i_5 < 32; ++i_5) {
        x_local_v0[i_5] = ((half_t*)x_shared)[(((((((((i_5 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_5 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_5 & 3) >> 1) + (i_5 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 8192)];
      }
      overlap_plan_mbar[15].arrive();
      overlap_plan_mbar[6].wait(0);
      #pragma unroll
      for (int i_6 = 0; i_6 < 8; ++i_6) {
        *(float2*)(scale_v1 + (i_6 * 2)) = *(float2*)(((float*)scale_wsp_handoff_3) + (((i_6 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 32; ++i_7) {
        xt_local_v1[i_7] = ((half_t)(((float)x_local_v1[((((i_7 >> 2) * 4) + ((i_7 & 1) * 2)) + ((i_7 & 3) >> 1))]) * scale_v1[(((i_7 >> 2) * 2) + (i_7 & 1))]));
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 16; ++i_8) {
        *(uint1*)(((half_t*)xt_local_wsp_handoff_4) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_8 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_8 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096)) = *(uint1*)(xt_local_v1 + (i_8 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[8].arrive();
    }
    overlap_plan_mbar[0].wait(1);
    #pragma unroll
    for (int i_9 = 0; i_9 < 32; ++i_9) {
      x_local_v1[i_9] = ((half_t*)x_shared)[((((((((i_9 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_9 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_9 & 3) >> 1) + (i_9 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
    }
    overlap_plan_mbar[13].arrive();
    overlap_plan_mbar[5].wait(1);
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      *(float2*)(scale_v0 + (i_10 * 2)) = *(float2*)(((float*)scale_wsp_handoff_3) + ((i_10 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 32; ++i_11) {
      xt_local_v0[i_11] = ((half_t)(((float)x_local_v0[((((i_11 >> 2) * 4) + ((i_11 & 1) * 2)) + ((i_11 & 3) >> 1))]) * scale_v0[(((i_11 >> 2) * 2) + (i_11 & 1))]));
    }
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_12 = 0; i_12 < 16; ++i_12) {
      *(uint1*)(((half_t*)xt_local_wsp_handoff_4) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_12 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_12 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(xt_local_v0 + (i_12 * 2));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[7].arrive();
    overlap_plan_mbar[6].wait(1);
    #pragma unroll
    for (int i_13 = 0; i_13 < 8; ++i_13) {
      *(float2*)(scale_v1 + (i_13 * 2)) = *(float2*)(((float*)scale_wsp_handoff_3) + (((i_13 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 32; ++i_14) {
      xt_local_v1[i_14] = ((half_t)(((float)x_local_v1[((((i_14 >> 2) * 4) + ((i_14 & 1) * 2)) + ((i_14 & 3) >> 1))]) * scale_v1[(((i_14 >> 2) * 2) + (i_14 & 1))]));
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 16; ++i_15) {
      *(uint1*)(((half_t*)xt_local_wsp_handoff_4) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_15 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_15 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096)) = *(uint1*)(xt_local_v1 + (i_15 * 2));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[8].arrive();
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<240>();
      da_last[0] = ((float)DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 255)]);
      #pragma unroll
      for (int i_16 = 0; i_16 < 16; ++i_16) {
        float broadcast_var = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc + (i_16 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
        overlap_plan_mbar[0].arrive_and_expect_tx(8192);
        tl::tma_load(X_desc, overlap_plan_mbar[0], (&(((half_t*)x_shared)[0])), 0, ((((int)blockIdx.z) >> 3) * 256), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
        overlap_plan_mbar[3].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[3], 128);
        overlap_plan_mbar[4].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[4], 128);
      }
      overlap_plan_mbar[3].wait(0);
      tl::__sync_thread_partial(4, 128);
      #pragma unroll
      for (int i_17 = 0; i_17 < 8; ++i_17) {
        half_t da_shared_local_cast[2];
        *(uint1*)(da_shared_local_cast + 0) = *(uint1*)(((half_t*)da_shared) + ((i_17 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __1;
        uint1 v_ = *(uint1*)(da_shared_local_cast + 0);
        ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
        *(float2*)(da_local + (i_17 * 2)) = __1;
      }
      overlap_plan_mbar[16].arrive();
      overlap_plan_mbar[4].wait(0);
      #pragma unroll
      for (int i_18 = 0; i_18 < 8; ++i_18) {
        half_t dt_shared_local_cast_1[2];
        *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_18 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __2;
        uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
        ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
        *(float2*)(dt_local + (i_18 * 2)) = __2;
      }
      overlap_plan_mbar[17].arrive();
      #pragma unroll
      for (int i_19 = 0; i_19 < 16; ++i_19) {
        scale_v0[i_19] = (exp2f(((da_last[0] - da_local[i_19]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_19]);
      }
      if (((((((int)threadIdx.x) & 31) >> 2) * 4) + (((int)threadIdx.x) >> 5)) == 4) {
        #pragma unroll
        for (int i_20 = 0; i_20 < 8; ++i_20) {
          *(float2*)(((float*)scale_wsp_handoff_3) + ((i_20 * 8) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(scale_v0 + (i_20 * 2));
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[5].arrive();
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
        overlap_plan_mbar[9].arrive_and_expect_tx(16384);
        tl::tma_load(B_desc, overlap_plan_mbar[9], (&(((half_t*)b_shared)[0])), 0, ((((int)blockIdx.z) >> 3) * 256), 0, (((int)blockIdx.z) & 7));
        tl::tma_load(B_desc, overlap_plan_mbar[9], (&(((half_t*)b_shared)[4096])), 64, ((((int)blockIdx.z) >> 3) * 256), 0, (((int)blockIdx.z) & 7));
        overlap_plan_mbar[1].arrive_and_expect_tx(8192);
        tl::tma_load(X_desc, overlap_plan_mbar[1], (&(((half_t*)x_shared)[4096])), 0, (((((int)blockIdx.z) >> 3) * 256) + 64), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[16].wait(0);
      tl::__sync_thread_partial(4, 128);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
        overlap_plan_mbar[3].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 64)])), overlap_plan_mbar[3], 128);
      }
      overlap_plan_mbar[17].wait(0);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
        overlap_plan_mbar[4].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 64)])), overlap_plan_mbar[4], 128);
      }
      overlap_plan_mbar[3].wait(1);
      tl::__sync_thread_partial(4, 128);
      #pragma unroll
      for (int i_21 = 0; i_21 < 8; ++i_21) {
        half_t da_shared_local_cast_2[2];
        *(uint1*)(da_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)da_shared) + ((i_21 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __3;
        uint1 v__2 = *(uint1*)(da_shared_local_cast_2 + 0);
        ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__2))[0]);
        *(float2*)(da_local + (i_21 * 2)) = __3;
      }
      overlap_plan_mbar[16].arrive();
      overlap_plan_mbar[4].wait(1);
      #pragma unroll
      for (int i_22 = 0; i_22 < 8; ++i_22) {
        half_t dt_shared_local_cast_3[2];
        *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_22 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __4;
        uint1 v__3 = *(uint1*)(dt_shared_local_cast_3 + 0);
        ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__3))[0]);
        *(float2*)(dt_local + (i_22 * 2)) = __4;
      }
      overlap_plan_mbar[17].arrive();
      #pragma unroll
      for (int i_23 = 0; i_23 < 16; ++i_23) {
        scale_v1[i_23] = (exp2f(((da_last[0] - da_local[i_23]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_23]);
      }
      if (((((((int)threadIdx.x) & 31) >> 2) * 4) + (((int)threadIdx.x) >> 5)) == 4) {
        #pragma unroll
        for (int i_24 = 0; i_24 < 8; ++i_24) {
          *(float2*)(((float*)scale_wsp_handoff_3) + (((i_24 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64)) = *(float2*)(scale_v1 + (i_24 * 2));
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[6].arrive();
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
        overlap_plan_mbar[10].arrive_and_expect_tx(16384);
        tl::tma_load(B_desc, overlap_plan_mbar[10], (&(((half_t*)b_shared)[8192])), 0, (((((int)blockIdx.z) >> 3) * 256) + 64), 0, (((int)blockIdx.z) & 7));
        tl::tma_load(B_desc, overlap_plan_mbar[10], (&(((half_t*)b_shared)[12288])), 64, (((((int)blockIdx.z) >> 3) * 256) + 64), 0, (((int)blockIdx.z) & 7));
      }
      tl::__sync_thread_partial(4, 128);
      for (int ik_1 = 0; ik_1 < 2; ++ik_1) {
        if (ik_1 == 1) {
          overlap_plan_mbar[(ik_1 + 12)].wait(0);
        }
        if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
          overlap_plan_mbar[((ik_1 + 2) % 3)].arrive_and_expect_tx(8192);
          tl::tma_load(X_desc, overlap_plan_mbar[((ik_1 + 2) % 3)], (&(((half_t*)x_shared)[(((ik_1 + 2) % 3) * 4096)])), 0, ((((((int)blockIdx.z) >> 3) * 256) + (ik_1 * 64)) + 128), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
        }
        overlap_plan_mbar[16].wait(((ik_1 + 1) & 1));
        if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
          overlap_plan_mbar[3].arrive_and_expect_tx(128);
          tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik_1 * 64)) + 128)])), overlap_plan_mbar[3], 128);
        }
        overlap_plan_mbar[17].wait(((ik_1 + 1) & 1));
        if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
          overlap_plan_mbar[4].arrive_and_expect_tx(128);
          tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik_1 * 64)) + 128)])), overlap_plan_mbar[4], 128);
        }
        overlap_plan_mbar[3].wait(ik_1);
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_25 = 0; i_25 < 8; ++i_25) {
          half_t da_shared_local_cast_4[2];
          *(uint1*)(da_shared_local_cast_4 + 0) = *(uint1*)(((half_t*)da_shared) + ((i_25 * 8) + ((((int)threadIdx.x) & 3) * 2)));
          float2 __5;
          uint1 v__4 = *(uint1*)(da_shared_local_cast_4 + 0);
          ((float2*)(&__5))[0] = __half22float2(((half2*)(&v__4))[0]);
          *(float2*)(da_local + (i_25 * 2)) = __5;
        }
        overlap_plan_mbar[16].arrive();
        overlap_plan_mbar[4].wait(ik_1);
        #pragma unroll
        for (int i_26 = 0; i_26 < 8; ++i_26) {
          half_t dt_shared_local_cast_5[2];
          *(uint1*)(dt_shared_local_cast_5 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_26 * 8) + ((((int)threadIdx.x) & 3) * 2)));
          float2 __6;
          uint1 v__5 = *(uint1*)(dt_shared_local_cast_5 + 0);
          ((float2*)(&__6))[0] = __half22float2(((half2*)(&v__5))[0]);
          *(float2*)(dt_local + (i_26 * 2)) = __6;
        }
        overlap_plan_mbar[17].arrive();
        if (ik_1 == 0) {
          #pragma unroll
          for (int i_27 = 0; i_27 < 16; ++i_27) {
            scale_v0[i_27] = (exp2f(((da_last[0] - da_local[i_27]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_27]);
          }
          if (((((((int)threadIdx.x) & 31) >> 2) * 4) + (((int)threadIdx.x) >> 5)) == 4) {
            #pragma unroll
            for (int i_28 = 0; i_28 < 8; ++i_28) {
              *(float2*)(((float*)scale_wsp_handoff_3) + (((ik_1 * 64) + (i_28 * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(scale_v0 + (i_28 * 2));
            }
          }
        } else {
          #pragma unroll
          for (int i_29 = 0; i_29 < 16; ++i_29) {
            scale_v1[i_29] = (exp2f(((da_last[0] - da_local[i_29]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_29]);
          }
          if (((((((int)threadIdx.x) & 31) >> 2) * 4) + (((int)threadIdx.x) >> 5)) == 4) {
            #pragma unroll
            for (int i_30 = 0; i_30 < 8; ++i_30) {
              *(float2*)(((float*)scale_wsp_handoff_3) + (((ik_1 * 64) + (i_30 * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(scale_v1 + (i_30 * 2));
            }
          }
        }
        tl::fence_proxy_async();
        overlap_plan_mbar[(ik_1 + 5)].arrive();
        if (ik_1 == 1) {
          overlap_plan_mbar[(ik_1 + 17)].wait(0);
        }
        tl::__sync_thread_partial(4, 128);
        if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
          overlap_plan_mbar[(((ik_1 + 2) % 3) + 9)].arrive_and_expect_tx(16384);
          tl::tma_load(B_desc, overlap_plan_mbar[(((ik_1 + 2) % 3) + 9)], (&(((half_t*)b_shared)[(((ik_1 + 2) % 3) * 8192)])), 0, ((((((int)blockIdx.z) >> 3) * 256) + (ik_1 * 64)) + 128), 0, (((int)blockIdx.z) & 7));
          tl::tma_load(B_desc, overlap_plan_mbar[(((ik_1 + 2) % 3) + 9)], (&(((half_t*)b_shared)[((((ik_1 + 2) % 3) * 8192) + 4096)])), 64, ((((((int)blockIdx.z) >> 3) * 256) + (ik_1 * 64)) + 128), 0, (((int)blockIdx.z) & 7));
        }
        overlap_plan_mbar[(ik_1 + 7)].wait(0);
        overlap_plan_mbar[(ik_1 + 9)].wait(0);
        if (ik_1 == 0) {
          #pragma unroll
          for (int i_31 = 0; i_31 < 16; ++i_31) {
            *(uint1*)(xt_local_v0 + (i_31 * 2)) = *(uint1*)(((half_t*)xt_local_wsp_handoff_4) + (((((((ik_1 * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_31 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_31 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096));
          }
          {
            tl::GmmaDescriptor desc_b;
            tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b, (&(((half_t*)b_shared)[0])));
            tl::increase_descriptor_offset<int>(desc_b, (ik_1 * 16384));
            tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v0 + 0), 16);
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
            tl::warpgroup_arrive();
            #pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
              tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local_v0 + (ki * 8)), uint64_t(desc_b + ((ki * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
            }
            tl::warpgroup_commit_batch();
            tl::warpgroup_wait<0>();
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
            tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v0 + 0), 16);
          }
        } else {
          #pragma unroll
          for (int i_32 = 0; i_32 < 16; ++i_32) {
            *(uint1*)(xt_local_v1 + (i_32 * 2)) = *(uint1*)(((half_t*)xt_local_wsp_handoff_4) + (((((((ik_1 * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_32 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_32 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096));
          }
          {
            tl::GmmaDescriptor desc_b_1;
            tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_1, (&(((half_t*)b_shared)[0])));
            tl::increase_descriptor_offset<int>(desc_b_1, (ik_1 * 16384));
            tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v1 + 0), 16);
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
            tl::warpgroup_arrive();
            #pragma unroll
            for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
              tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local_v1 + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
            }
            tl::warpgroup_commit_batch();
            tl::warpgroup_wait<0>();
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
            tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v1 + 0), 16);
          }
        }
        overlap_plan_mbar[(ik_1 + 18)].arrive();
      }
      overlap_plan_mbar[7].wait(1);
      overlap_plan_mbar[11].wait(0);
      #pragma unroll
      for (int i_33 = 0; i_33 < 16; ++i_33) {
        *(uint1*)(xt_local_v0 + (i_33 * 2)) = *(uint1*)(((half_t*)xt_local_wsp_handoff_4) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_33 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_33 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096));
      }
      {
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_2, (&(((half_t*)b_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_2, 32768);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v0 + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_arrive();
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local_v0 + (ki_2 * 8)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v0 + 0), 16);
      }
      overlap_plan_mbar[20].arrive();
      overlap_plan_mbar[8].wait(1);
      overlap_plan_mbar[9].wait(1);
      #pragma unroll
      for (int i_34 = 0; i_34 < 16; ++i_34) {
        *(uint1*)(xt_local_v1 + (i_34 * 2)) = *(uint1*)(((half_t*)xt_local_wsp_handoff_4) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_34 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_34 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      {
        tl::GmmaDescriptor desc_b_3;
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_3, (&(((half_t*)b_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v1 + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local_v1 + (ki_3 * 8)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v1 + 0), 16);
      }
      overlap_plan_mbar[18].arrive();
      #pragma unroll
      for (int i_35 = 0; i_35 < 8; ++i_35) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[((((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_35 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 8192)])), __pack_half2(((half_t)acc[(i_35 * 8)]), ((half_t)acc[((i_35 * 8) + 1)])), __pack_half2(((half_t)acc[((i_35 * 8) + 2)]), ((half_t)acc[((i_35 * 8) + 3)])), __pack_half2(((half_t)acc[((i_35 * 8) + 4)]), ((half_t)acc[((i_35 * 8) + 5)])), __pack_half2(((half_t)acc[((i_35 * 8) + 6)]), ((half_t)acc[((i_35 * 8) + 7)])));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[12].arrive();
    } else {
      tl::warpgroup_reg_dealloc<24>();
      overlap_plan_mbar[12].wait(0);
      if (tl::tl_shuffle_elect<128>()) {
        tl::tma_store((&(Output[((((((int)blockIdx.z) & 7) * 10485760) + ((((int)blockIdx.z) >> 3) * 655360)) + (((int)blockIdx.x) * 8192))])), (&(((half_t*)acc_shared)[0])), 16384);
        tl::tma_store_arrive();
        tl::tma_store_wait<0, true>();
      }
    }
  }
}

