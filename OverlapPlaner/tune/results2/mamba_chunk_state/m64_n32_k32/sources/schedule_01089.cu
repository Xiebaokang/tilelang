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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap B_desc, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap X_desc);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap B_desc, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap X_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 12288));
  void* xt_local_wsp_handoff_4 = ((void*)((char*)buf_dyn_shmem + 16384));
  void* scale_wsp_handoff_3 = ((void*)((char*)buf_dyn_shmem + 20480));
  void* da_shared = ((void*)((char*)buf_dyn_shmem + 21504));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 22528));
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 23552));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[18];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float da_last[1];
  float acc[16];
  float da_local[8];
  float dt_local[8];
  float scale_v0[8];
  float scale_v1[8];
  half_t xt_local[16];
  half_t x_local_v0[16];
  half_t x_local_v1[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(X_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(Output_desc);
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
    overlap_plan_mbar[8].init(1);
    overlap_plan_mbar[9].init(1);
    overlap_plan_mbar[10].init(128);
    overlap_plan_mbar[11].init(128);
    overlap_plan_mbar[12].init(128);
    overlap_plan_mbar[13].init(128);
    overlap_plan_mbar[14].init(128);
    overlap_plan_mbar[15].init(128);
    overlap_plan_mbar[16].init(128);
    overlap_plan_mbar[17].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<240>();
    da_last[0] = ((float)DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 255)]);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(4096);
      tl::tma_load(X_desc, overlap_plan_mbar[0], (&(((half_t*)x_shared)[0])), 0, ((((int)blockIdx.z) >> 3) * 256), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      overlap_plan_mbar[3].arrive_and_expect_tx(64);
      tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[3], 64);
      overlap_plan_mbar[4].arrive_and_expect_tx(64);
      tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[4], 64);
    }
    overlap_plan_mbar[3].wait(0);
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      half_t da_shared_local_cast[2];
      *(uint1*)(da_shared_local_cast + 0) = *(uint1*)(((half_t*)da_shared) + ((i_1 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __1;
      uint1 v_ = *(uint1*)(da_shared_local_cast + 0);
      ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
      *(float2*)(da_local + (i_1 * 2)) = __1;
    }
    overlap_plan_mbar[14].arrive();
    overlap_plan_mbar[4].wait(0);
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      half_t dt_shared_local_cast_1[2];
      *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_2 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __2;
      uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
      ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
      *(float2*)(dt_local + (i_2 * 2)) = __2;
    }
    overlap_plan_mbar[15].arrive();
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      scale_v0[i_3] = (exp2f(((da_last[0] - da_local[i_3]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_3]);
    }
    if (((((((int)threadIdx.x) & 31) >> 2) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        *(float2*)(((float*)scale_wsp_handoff_3) + ((i_4 * 8) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(scale_v0 + (i_4 * 2));
      }
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[5].arrive();
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[8].arrive_and_expect_tx(2048);
      tl::tma_load(B_desc, overlap_plan_mbar[8], (&(((half_t*)b_shared)[0])), (((int)blockIdx.y) * 32), ((((int)blockIdx.z) >> 3) * 256), 0, (((int)blockIdx.z) & 7));
    }
    for (int ik = 0; ik < 7; ++ik) {
      if (2 <= ik) {
        overlap_plan_mbar[(((ik + 1) % 3) + 11)].wait(((ik - 2) / 3));
      }
      tl::__sync_thread_partial(3, 128);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[((ik + 1) % 3)].arrive_and_expect_tx(4096);
        tl::tma_load(X_desc, overlap_plan_mbar[((ik + 1) % 3)], (&(((half_t*)x_shared)[(((ik + 1) % 3) * 2048)])), 0, ((((((int)blockIdx.z) >> 3) * 256) + (ik * 32)) + 32), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[14].wait((ik & 1));
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[3].arrive_and_expect_tx(64);
        tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 32)) + 32)])), overlap_plan_mbar[3], 64);
      }
      overlap_plan_mbar[15].wait((ik & 1));
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[4].arrive_and_expect_tx(64);
        tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 32)) + 32)])), overlap_plan_mbar[4], 64);
      }
      overlap_plan_mbar[3].wait(((ik + 1) & 1));
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        half_t da_shared_local_cast_2[2];
        *(uint1*)(da_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)da_shared) + ((i_5 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __3;
        uint1 v__2 = *(uint1*)(da_shared_local_cast_2 + 0);
        ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__2))[0]);
        *(float2*)(da_local + (i_5 * 2)) = __3;
      }
      overlap_plan_mbar[14].arrive();
      overlap_plan_mbar[4].wait(((ik + 1) & 1));
      #pragma unroll
      for (int i_6 = 0; i_6 < 4; ++i_6) {
        half_t dt_shared_local_cast_3[2];
        *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_6 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __4;
        uint1 v__3 = *(uint1*)(dt_shared_local_cast_3 + 0);
        ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__3))[0]);
        *(float2*)(dt_local + (i_6 * 2)) = __4;
      }
      overlap_plan_mbar[15].arrive();
      if (((ik + 1) % 2) == 0) {
        #pragma unroll
        for (int i_7 = 0; i_7 < 8; ++i_7) {
          scale_v0[i_7] = (exp2f(((da_last[0] - da_local[i_7]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_7]);
        }
        if (((((((int)threadIdx.x) & 31) >> 2) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_8 = 0; i_8 < 4; ++i_8) {
            *(float2*)(((float*)scale_wsp_handoff_3) + (((((ik + 1) & 1) * 32) + (i_8 * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(scale_v0 + (i_8 * 2));
          }
        }
      } else {
        #pragma unroll
        for (int i_9 = 0; i_9 < 8; ++i_9) {
          scale_v1[i_9] = (exp2f(((da_last[0] - da_local[i_9]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_9]);
        }
        if (((((((int)threadIdx.x) & 31) >> 2) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_10 = 0; i_10 < 4; ++i_10) {
            *(float2*)(((float*)scale_wsp_handoff_3) + (((((ik + 1) & 1) * 32) + (i_10 * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(scale_v1 + (i_10 * 2));
          }
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[(((ik + 1) & 1) + 5)].arrive();
      if (1 <= ik) {
        overlap_plan_mbar[(((ik + 1) & 1) + 16)].wait((((ik + 3) & 3) >> 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[(((ik + 1) & 1) + 8)].arrive_and_expect_tx(2048);
        tl::tma_load(B_desc, overlap_plan_mbar[(((ik + 1) & 1) + 8)], (&(((half_t*)b_shared)[(((ik + 1) & 1) * 1024)])), (((int)blockIdx.y) * 32), ((((((int)blockIdx.z) >> 3) * 256) + (ik * 32)) + 32), 0, (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[7].wait((ik & 1));
      overlap_plan_mbar[((ik & 1) + 8)].wait(((ik & 3) >> 1));
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        tl::ptx_ldmatrix_x4((&(((half_t*)xt_local_wsp_handoff_4)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (i_11 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), (&(xt_local[(i_11 * 8)])));
      }
      {
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b, (&(((half_t*)b_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b, ((ik & 1) * 2048));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 8);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local + (ki * 8)), uint64_t(desc_b + ((ki * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 8);
      }
      overlap_plan_mbar[((ik & 1) + 16)].arrive();
    }
    overlap_plan_mbar[7].wait(1);
    overlap_plan_mbar[9].wait(1);
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      tl::ptx_ldmatrix_x4((&(((half_t*)xt_local_wsp_handoff_4)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (i_12 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), (&(xt_local[(i_12 * 8)])));
    }
    {
      tl::GmmaDescriptor desc_b_1;
      tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_1, (&(((half_t*)b_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_1, 2048);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 8);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 8);
    }
    overlap_plan_mbar[17].arrive();
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[(((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_13) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((half_t)acc[(i_13 * 8)]), ((half_t)acc[((i_13 * 8) + 1)])), __pack_half2(((half_t)acc[((i_13 * 8) + 2)]), ((half_t)acc[((i_13 * 8) + 3)])), __pack_half2(((half_t)acc[((i_13 * 8) + 4)]), ((half_t)acc[((i_13 * 8) + 5)])), __pack_half2(((half_t)acc[((i_13 * 8) + 6)]), ((half_t)acc[((i_13 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[10].arrive();
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_dealloc<32>();
      overlap_plan_mbar[0].wait(0);
      #pragma unroll
      for (int i_14 = 0; i_14 < 16; ++i_14) {
        x_local_v0[i_14] = ((half_t*)x_shared)[((((((((i_14 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_14 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_14 & 3) >> 1) + (i_14 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      overlap_plan_mbar[11].arrive();
      for (int ik_1 = 0; ik_1 < 3; ++ik_1) {
        overlap_plan_mbar[(((ik_1 * 2) + 1) % 3)].wait((((ik_1 * 2) + 1) / 3));
        #pragma unroll
        for (int i_15 = 0; i_15 < 16; ++i_15) {
          x_local_v1[i_15] = ((half_t*)x_shared)[(((((((((((ik_1 * 2) + 1) % 3) * 2048) + ((i_15 >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (((i_15 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_15 & 3) >> 1) + (i_15 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
        }
        overlap_plan_mbar[((((ik_1 * 2) + 1) % 3) + 11)].arrive();
        overlap_plan_mbar[5].wait((ik_1 & 1));
        #pragma unroll
        for (int i_16 = 0; i_16 < 4; ++i_16) {
          *(float2*)(scale_v0 + (i_16 * 2)) = *(float2*)(((float*)scale_wsp_handoff_3) + ((i_16 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        }
        #pragma unroll
        for (int i_17 = 0; i_17 < 16; ++i_17) {
          xt_local[i_17] = ((half_t)(((float)x_local_v0[((((i_17 >> 2) * 4) + ((i_17 & 1) * 2)) + ((i_17 & 3) >> 1))]) * scale_v0[(((i_17 >> 2) * 2) + (i_17 & 1))]));
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_18 = 0; i_18 < 2; ++i_18) {
          tl::ptx_stmatrix_m8n8_x4((&(((half_t*)xt_local_wsp_handoff_4)[((((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (i_18 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 2048)])), __pack_half2(xt_local[(i_18 * 8)], xt_local[((i_18 * 8) + 1)]), __pack_half2(xt_local[((i_18 * 8) + 2)], xt_local[((i_18 * 8) + 3)]), __pack_half2(xt_local[((i_18 * 8) + 4)], xt_local[((i_18 * 8) + 5)]), __pack_half2(xt_local[((i_18 * 8) + 6)], xt_local[((i_18 * 8) + 7)]));
        }
        tl::fence_proxy_async();
        overlap_plan_mbar[7].arrive();
        overlap_plan_mbar[(((ik_1 * 2) + 2) % 3)].wait(((((ik_1 * 2) + 2) % 6) / 3));
        #pragma unroll
        for (int i_19 = 0; i_19 < 16; ++i_19) {
          x_local_v0[i_19] = ((half_t*)x_shared)[(((((((((((ik_1 * 2) + 2) % 3) * 2048) + ((i_19 >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (((i_19 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_19 & 3) >> 1) + (i_19 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
        }
        overlap_plan_mbar[((((ik_1 * 2) + 2) % 3) + 11)].arrive();
        overlap_plan_mbar[6].wait((ik_1 & 1));
        #pragma unroll
        for (int i_20 = 0; i_20 < 4; ++i_20) {
          *(float2*)(scale_v1 + (i_20 * 2)) = *(float2*)(((float*)scale_wsp_handoff_3) + (((i_20 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 32));
        }
        #pragma unroll
        for (int i_21 = 0; i_21 < 16; ++i_21) {
          xt_local[i_21] = ((half_t)(((float)x_local_v1[((((i_21 >> 2) * 4) + ((i_21 & 1) * 2)) + ((i_21 & 3) >> 1))]) * scale_v1[(((i_21 >> 2) * 2) + (i_21 & 1))]));
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_22 = 0; i_22 < 2; ++i_22) {
          tl::ptx_stmatrix_m8n8_x4((&(((half_t*)xt_local_wsp_handoff_4)[((((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (i_22 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 2048)])), __pack_half2(xt_local[(i_22 * 8)], xt_local[((i_22 * 8) + 1)]), __pack_half2(xt_local[((i_22 * 8) + 2)], xt_local[((i_22 * 8) + 3)]), __pack_half2(xt_local[((i_22 * 8) + 4)], xt_local[((i_22 * 8) + 5)]), __pack_half2(xt_local[((i_22 * 8) + 6)], xt_local[((i_22 * 8) + 7)]));
        }
        tl::fence_proxy_async();
        overlap_plan_mbar[7].arrive();
      }
      overlap_plan_mbar[1].wait(0);
      #pragma unroll
      for (int i_23 = 0; i_23 < 16; ++i_23) {
        x_local_v1[i_23] = ((half_t*)x_shared)[(((((((((i_23 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_23 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_23 & 3) >> 1) + (i_23 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 2048)];
      }
      overlap_plan_mbar[12].arrive();
      overlap_plan_mbar[5].wait(1);
      #pragma unroll
      for (int i_24 = 0; i_24 < 4; ++i_24) {
        *(float2*)(scale_v0 + (i_24 * 2)) = *(float2*)(((float*)scale_wsp_handoff_3) + ((i_24 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      }
      #pragma unroll
      for (int i_25 = 0; i_25 < 16; ++i_25) {
        xt_local[i_25] = ((half_t)(((float)x_local_v0[((((i_25 >> 2) * 4) + ((i_25 & 1) * 2)) + ((i_25 & 3) >> 1))]) * scale_v0[(((i_25 >> 2) * 2) + (i_25 & 1))]));
      }
      tl::__sync_thread_partial(4, 128);
      #pragma unroll
      for (int i_26 = 0; i_26 < 2; ++i_26) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)xt_local_wsp_handoff_4)[((((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (i_26 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 2048)])), __pack_half2(xt_local[(i_26 * 8)], xt_local[((i_26 * 8) + 1)]), __pack_half2(xt_local[((i_26 * 8) + 2)], xt_local[((i_26 * 8) + 3)]), __pack_half2(xt_local[((i_26 * 8) + 4)], xt_local[((i_26 * 8) + 5)]), __pack_half2(xt_local[((i_26 * 8) + 6)], xt_local[((i_26 * 8) + 7)]));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[7].arrive();
      overlap_plan_mbar[6].wait(1);
      #pragma unroll
      for (int i_27 = 0; i_27 < 4; ++i_27) {
        *(float2*)(scale_v1 + (i_27 * 2)) = *(float2*)(((float*)scale_wsp_handoff_3) + (((i_27 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 32));
      }
      #pragma unroll
      for (int i_28 = 0; i_28 < 16; ++i_28) {
        xt_local[i_28] = ((half_t)(((float)x_local_v1[((((i_28 >> 2) * 4) + ((i_28 & 1) * 2)) + ((i_28 & 3) >> 1))]) * scale_v1[(((i_28 >> 2) * 2) + (i_28 & 1))]));
      }
      tl::__sync_thread_partial(4, 128);
      #pragma unroll
      for (int i_29 = 0; i_29 < 2; ++i_29) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)xt_local_wsp_handoff_4)[((((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (i_29 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 2048)])), __pack_half2(xt_local[(i_29 * 8)], xt_local[((i_29 * 8) + 1)]), __pack_half2(xt_local[((i_29 * 8) + 2)], xt_local[((i_29 * 8) + 3)]), __pack_half2(xt_local[((i_29 * 8) + 4)], xt_local[((i_29 * 8) + 5)]), __pack_half2(xt_local[((i_29 * 8) + 6)], xt_local[((i_29 * 8) + 7)]));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[7].arrive();
    } else {
      tl::warpgroup_reg_dealloc<24>();
      overlap_plan_mbar[10].wait(0);
      if (tl::tl_shuffle_elect<128>()) {
        tl::tma_store(Output_desc, (&(((half_t*)acc_shared)[0])), (((int)blockIdx.y) * 32), 0, ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
        tl::tma_store_arrive();
        tl::tma_store_wait<0, true>();
      }
    }
  }
}

