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
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap CB_desc, __grid_constant__ const CUtensorMap C_desc, const half_t* __restrict__ D, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Prev_desc, __grid_constant__ const CUtensorMap X_desc, __grid_constant__ const CUtensorMap X_desc_1) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* da_m_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* c_shared = ((void*)((char*)buf_dyn_shmem + 1024));
  void* prev_shared = ((void*)((char*)buf_dyn_shmem + 33792));
  void* acc_wsp_handoff_8 = ((void*)((char*)buf_dyn_shmem + 41984));
  void* cb_local_wsp_handoff_4 = ((void*)((char*)buf_dyn_shmem + 58368));
  void* cb_local_wsp_handoff_6 = ((void*)((char*)buf_dyn_shmem + 74752));
  void* cb_shared = ((void*)((char*)buf_dyn_shmem + 91136));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 107520));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 119808));
  void* da_k_shared = ((void*)((char*)buf_dyn_shmem + 120832));
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 121856));
  void* residual_shared = ((void*)((char*)buf_dyn_shmem + 130048));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[23];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  half_t cb_local[64];
  float d_local[1];
  float residual_local[32];
  float acc[32];
  float da_m_local[4];
  float scale_m[4];
  float dt_local[16];
  float da_k_local[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(X_desc);
    tl::prefetch_tma_descriptor(X_desc_1);
    tl::prefetch_tma_descriptor(C_desc);
    tl::prefetch_tma_descriptor(Prev_desc);
    tl::prefetch_tma_descriptor(CB_desc);
    tl::prefetch_tma_descriptor(Output_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(1);
    overlap_plan_mbar[4].init(128);
    overlap_plan_mbar[5].init(1);
    overlap_plan_mbar[6].init(1);
    overlap_plan_mbar[7].init(1);
    overlap_plan_mbar[8].init(128);
    overlap_plan_mbar[9].init(1);
    overlap_plan_mbar[10].init(1);
    overlap_plan_mbar[11].init(1);
    overlap_plan_mbar[12].init(128);
    overlap_plan_mbar[13].init(1);
    overlap_plan_mbar[14].init(128);
    overlap_plan_mbar[15].init(128);
    overlap_plan_mbar[16].init(128);
    overlap_plan_mbar[17].init(128);
    overlap_plan_mbar[18].init(128);
    overlap_plan_mbar[19].init(128);
    overlap_plan_mbar[20].init(128);
    overlap_plan_mbar[21].init(128);
    overlap_plan_mbar[22].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<72>();
    for (int ik = 0; ik < (((((int)blockIdx.y) >> 1) * 2) + 2); ++ik) {
      if (ik == 3) {
        overlap_plan_mbar[((ik % 3) + 20)].wait(0);
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[((ik % 3) + 9)].arrive_and_expect_tx(4096);
        tl::tma_load(X_desc, overlap_plan_mbar[((ik % 3) + 9)], (&(((half_t*)x_shared)[((ik % 3) * 2048)])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + (ik * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[((ik % 3) + 5)].wait((ik / 3));
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        half_t dt_shared_local_cast[2];
        *(uint1*)(dt_shared_local_cast + 0) = *(uint1*)(((half_t*)dt_shared) + ((((ik % 3) * 64) + (i * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __1;
        uint1 v_ = *(uint1*)(dt_shared_local_cast + 0);
        ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
        *(float2*)(dt_local + (i * 2)) = __1;
      }
      overlap_plan_mbar[((ik % 3) + 17)].arrive();
      overlap_plan_mbar[4].wait((ik & 1));
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        tl::ptx_ldmatrix_x4((&(((half_t*)cb_local_wsp_handoff_4)[((((((i_1 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((i_1 >> 1) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), (&(cb_local[(i_1 * 8)])));
      }
      #pragma unroll
      for (int i_2 = 0; i_2 < 32; ++i_2) {
        uint1 __2;
        float2 __3;
          float2 __4;
          uint1 v__1 = *(uint1*)(cb_local + (i_2 * 2));
          ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__1))[0]);
          float2 v__2 = *(float2*)(dt_local + (((i_2 >> 3) * 4) + (((i_2 & 3) >> 1) * 2)));
          __3.x = (__4.x*v__2.x);
          __3.y = (__4.y*v__2.y);
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&__3))[0]);
        *(uint1*)(cb_local + (i_2 * 2)) = __2;
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 64; ++i_3) {
        half_t condval;
        if (((((((ik * 64) + ((i_3 >> 4) * 16)) + (((i_3 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_3 & 1)) <= ((((((((int)blockIdx.y) >> 1) * 128) + (((i_3 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_3 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval = cb_local[i_3];
        } else {
          condval = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local[i_3] = condval;
      }
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_4 = 0; i_4 < 8; ++i_4) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)cb_local_wsp_handoff_6)[((((((i_4 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((i_4 >> 1) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(cb_local[(i_4 * 8)], cb_local[((i_4 * 8) + 1)]), __pack_half2(cb_local[((i_4 * 8) + 2)], cb_local[((i_4 * 8) + 3)]), __pack_half2(cb_local[((i_4 * 8) + 4)], cb_local[((i_4 * 8) + 5)]), __pack_half2(cb_local[((i_4 * 8) + 6)], cb_local[((i_4 * 8) + 7)]));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[8].arrive();
    }
    d_local[0] = ((float)D[((int)blockIdx.x)]);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[13].arrive_and_expect_tx(8192);
      tl::tma_load(X_desc_1, overlap_plan_mbar[13], (&(((half_t*)residual_shared)[0])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 128)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[13].wait(0);
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      half_t residual_shared_local_cast_1[2];
      *(uint1*)(residual_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)residual_shared) + ((((((((i_5 >> 3) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((i_5 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_5 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_5 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __5;
      uint1 v__3 = *(uint1*)(residual_shared_local_cast_1 + 0);
      ((float2*)(&__5))[0] = __half22float2(((half2*)(&v__3))[0]);
      *(float2*)(residual_local + (i_5 * 2)) = __5;
    }
    overlap_plan_mbar[12].wait(0);
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      *(float2*)(acc + (i_6 * 2)) = *(float2*)(((float*)acc_wsp_handoff_8) + (((((((i_6 >> 3) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((i_6 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((i_6 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 32; ++i_7) {
      acc[i_7] = (acc[i_7] + (residual_local[i_7] * d_local[0]));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[((((((i_8 >> 1) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((half_t)acc[(i_8 * 8)]), ((half_t)acc[((i_8 * 8) + 1)])), __pack_half2(((half_t)acc[((i_8 * 8) + 2)]), ((half_t)acc[((i_8 * 8) + 3)])), __pack_half2(((half_t)acc[((i_8 * 8) + 4)]), ((half_t)acc[((i_8 * 8) + 5)])), __pack_half2(((half_t)acc[((i_8 * 8) + 6)]), ((half_t)acc[((i_8 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[14].arrive();
  } else {
    tl::warpgroup_reg_alloc<240>();
    ((half_t*)da_m_shared)[(((int)threadIdx.x) - 128)] = DA[(((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + ((((int)blockIdx.y) >> 1) * 128)) + ((int)threadIdx.x)) - 128)];
    tl::__sync_thread_partial(4, 128);
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      da_m_local[i_9] = ((float)((half_t*)da_m_shared)[((((((i_9 >> 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc + (i_10 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      scale_m[i_11] = exp2f((da_m_local[i_11] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
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
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a + (((((ki >> 2) * 16384) + (i_12 * 8192)) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 4096) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc + (i_12 * 16))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 32; ++i_13) {
      acc[i_13] = (acc[i_13] * scale_m[(((i_13 >> 4) * 2) + ((i_13 & 3) >> 1))]);
    }
    for (int ik_1 = 0; ik_1 < (((((int)blockIdx.y) >> 1) * 2) + 2); ++ik_1) {
      if (1 <= ik_1) {
        overlap_plan_mbar[15].wait(((ik_1 + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(16384);
        tl::fence_proxy_async();
        tl::tma_load(CB_desc, overlap_plan_mbar[2], (&(((half_t*)cb_shared)[0])), (ik_1 * 64), ((((int)blockIdx.y) >> 1) * 128), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
      }
      if (1 <= ik_1) {
        overlap_plan_mbar[16].wait(((ik_1 + 1) & 1));
      }
      tl::__sync_thread_partial(4, 128);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
        overlap_plan_mbar[3].arrive_and_expect_tx(128);
        tl::fence_proxy_async();
        tl::tma_load((&(((half_t*)da_k_shared)[0])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik_1 * 64))])), overlap_plan_mbar[3], 128);
      }
      if (ik_1 == 3) {
        overlap_plan_mbar[((ik_1 % 3) + 17)].wait(0);
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
        overlap_plan_mbar[((ik_1 % 3) + 5)].arrive_and_expect_tx(128);
        tl::fence_proxy_async();
        tl::tma_load((&(((half_t*)dt_shared)[((ik_1 % 3) * 64)])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik_1 * 64))])), overlap_plan_mbar[((ik_1 % 3) + 5)], 128);
      }
      overlap_plan_mbar[2].wait((ik_1 & 1));
      #pragma unroll
      for (int i_14 = 0; i_14 < 8; ++i_14) {
        tl::ptx_ldmatrix_x4((&(((half_t*)cb_shared)[((((((i_14 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((i_14 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_14 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) - 4096)])), (&(cb_local[(i_14 * 8)])));
      }
      overlap_plan_mbar[15].arrive();
      overlap_plan_mbar[3].wait((ik_1 & 1));
      tl::__sync_thread_partial(4, 128);
      #pragma unroll
      for (int i_15 = 0; i_15 < 8; ++i_15) {
        half_t da_k_shared_local_cast_2[2];
        *(uint1*)(da_k_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)da_k_shared) + ((i_15 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __6;
        uint1 v__4 = *(uint1*)(da_k_shared_local_cast_2 + 0);
        ((float2*)(&__6))[0] = __half22float2(((half2*)(&v__4))[0]);
        *(float2*)(da_k_local + (i_15 * 2)) = __6;
      }
      overlap_plan_mbar[16].arrive();
      #pragma unroll
      for (int i_16 = 0; i_16 < 32; ++i_16) {
        float broadcast_var_1 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
        uint1 __7;
        float2 __8;
          float2 __9;
          uint1 v__5 = *(uint1*)(cb_local + (i_16 * 2));
          ((float2*)(&__9))[0] = __half22float2(((half2*)(&v__5))[0]);
          float2 __10;
          float2 __11;
            float2 __12;
              float2 v__6 = make_float2(da_m_local[((((i_16 & 7) >> 2) * 2) + (i_16 & 1))], da_m_local[((((i_16 & 7) >> 2) * 2) + (i_16 & 1))]);
              float2 v__7 = *(float2*)(da_k_local + (((i_16 >> 3) * 4) + (((i_16 & 3) >> 1) * 2)));
              __12.x = (v__6.x-v__7.x);
              __12.y = (v__6.y-v__7.y);
            float2 v__8 = make_float2(broadcast_var_1, broadcast_var_1);
            __11.x = (__12.x*v__8.x);
            __11.y = (__12.y*v__8.y);
          __10.x = exp2f(__11.x);
          __10.y = exp2f(__11.y);
          __8.x = (__9.x*__10.x);
          __8.y = (__9.y*__10.y);
        ((half2*)(&__7))[0] = __float22half2_rn(((float2*)(&__8))[0]);
        *(uint1*)(cb_local + (i_16 * 2)) = __7;
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 8; ++i_17) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)cb_local_wsp_handoff_4)[(((((((i_17 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((i_17 >> 1) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 4096)])), __pack_half2(cb_local[(i_17 * 8)], cb_local[((i_17 * 8) + 1)]), __pack_half2(cb_local[((i_17 * 8) + 2)], cb_local[((i_17 * 8) + 3)]), __pack_half2(cb_local[((i_17 * 8) + 4)], cb_local[((i_17 * 8) + 5)]), __pack_half2(cb_local[((i_17 * 8) + 6)], cb_local[((i_17 * 8) + 7)]));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[4].arrive();
      overlap_plan_mbar[8].wait((ik_1 & 1));
      overlap_plan_mbar[((ik_1 % 3) + 9)].wait((ik_1 / 3));
      #pragma unroll
      for (int i_18 = 0; i_18 < 8; ++i_18) {
        tl::ptx_ldmatrix_x4((&(((half_t*)cb_local_wsp_handoff_6)[(((((((i_18 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((i_18 >> 1) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 4096)])), (&(cb_local[(i_18 * 8)])));
      }
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_1, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, ((ik_1 % 3) * 4096));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int i_19 = 0; i_19 < 2; ++i_19) {
          #pragma unroll
          for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
            tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local + ((ki_1 * 16) + (i_19 * 8))), uint64_t(desc_b_1 + ((ki_1 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_19 * 16)), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local + 0), 32);
      }
      if (ik_1 == (((((int)blockIdx.y) >> 1) * 2) + 1)) {
        #pragma unroll
        for (int i_20 = 0; i_20 < 16; ++i_20) {
          *(float2*)(((float*)acc_wsp_handoff_8) + ((((((((i_20 >> 3) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((i_20 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((i_20 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 2048)) = *(float2*)(acc + (i_20 * 2));
        }
      }
      overlap_plan_mbar[((ik_1 % 3) + 20)].arrive();
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[12].arrive();
    overlap_plan_mbar[14].wait(0);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store(Output_desc, (&(((half_t*)acc_shared)[0])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 128)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

