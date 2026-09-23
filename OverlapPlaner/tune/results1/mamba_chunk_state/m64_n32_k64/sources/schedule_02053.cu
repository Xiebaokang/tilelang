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
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap B_desc, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap X_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* xt_local_wsp_handoff_3 = ((void*)((char*)buf_dyn_shmem + 0));
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* da_shared = ((void*)((char*)buf_dyn_shmem + 40960));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 41984));
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 43008));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[14];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float da_last[1];
  float da_local[16];
  float dt_local[16];
  half_t x_local[32];
  float scale[16];
  half_t xt_local_v0[32];
  half_t xt_local_v1[32];
  half_t xt_local_v2[32];
  float acc[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(X_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(Output_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(128);
    overlap_plan_mbar[4].init(128);
    overlap_plan_mbar[5].init(128);
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
    tl::warpgroup_reg_dealloc<72>();
    da_last[0] = ((float)DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 255)]);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(8192);
      tl::tma_load(X_desc, overlap_plan_mbar[0], (&(((half_t*)x_shared)[0])), 0, ((((int)blockIdx.z) >> 3) * 256), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      overlap_plan_mbar[1].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[1], 128);
      overlap_plan_mbar[2].arrive_and_expect_tx(128);
      tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[2], 128);
    }
    overlap_plan_mbar[1].wait(0);
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      half_t da_shared_local_cast[2];
      *(uint1*)(da_shared_local_cast + 0) = *(uint1*)(((half_t*)da_shared) + ((i * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __1;
      uint1 v_ = *(uint1*)(da_shared_local_cast + 0);
      ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
      *(float2*)(da_local + (i * 2)) = __1;
    }
    overlap_plan_mbar[10].arrive();
    overlap_plan_mbar[2].wait(0);
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      half_t dt_shared_local_cast_1[2];
      *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_1 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __2;
      uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
      ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
      *(float2*)(dt_local + (i_1 * 2)) = __2;
    }
    overlap_plan_mbar[11].arrive();
    overlap_plan_mbar[0].wait(0);
    #pragma unroll
    for (int i_2 = 0; i_2 < 32; ++i_2) {
      x_local[i_2] = ((half_t*)x_shared)[((((((((i_2 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_2 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_2 & 3) >> 1) + (i_2 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
    }
    overlap_plan_mbar[9].arrive();
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      scale[i_3] = (exp2f(((da_last[0] - da_local[i_3]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_3]);
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 32; ++i_4) {
      xt_local_v0[i_4] = ((half_t)(((float)x_local[((((i_4 >> 2) * 4) + ((i_4 & 1) * 2)) + ((i_4 & 3) >> 1))]) * scale[(((i_4 >> 2) * 2) + (i_4 & 1))]));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      *(uint1*)(((half_t*)xt_local_wsp_handoff_3) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_5 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_5 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(xt_local_v0 + (i_5 * 2));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[3].arrive();
    for (int ik = 0; ik < 3; ++ik) {
      overlap_plan_mbar[9].wait((ik & 1));
      tl::__sync_thread_partial(3, 128);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[0].arrive_and_expect_tx(8192);
        tl::tma_load(X_desc, overlap_plan_mbar[0], (&(((half_t*)x_shared)[0])), 0, ((((((int)blockIdx.z) >> 3) * 256) + (ik * 64)) + 64), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[10].wait((ik & 1));
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[1].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 64)])), overlap_plan_mbar[1], 128);
      }
      overlap_plan_mbar[11].wait((ik & 1));
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 64)])), overlap_plan_mbar[2], 128);
      }
      if (ik == 2) {
        overlap_plan_mbar[((ik & 1) + 12)].wait(0);
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[((ik & 1) + 6)].arrive_and_expect_tx(4096);
        tl::tma_load(B_desc, overlap_plan_mbar[((ik & 1) + 6)], (&(((half_t*)b_shared)[((ik & 1) * 2048)])), (((int)blockIdx.y) * 32), (((((int)blockIdx.z) >> 3) * 256) + (ik * 64)), 0, (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[1].wait(((ik + 1) & 1));
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_6 = 0; i_6 < 8; ++i_6) {
        half_t da_shared_local_cast_2[2];
        *(uint1*)(da_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)da_shared) + ((i_6 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __3;
        uint1 v__2 = *(uint1*)(da_shared_local_cast_2 + 0);
        ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__2))[0]);
        *(float2*)(da_local + (i_6 * 2)) = __3;
      }
      overlap_plan_mbar[10].arrive();
      overlap_plan_mbar[2].wait(((ik + 1) & 1));
      #pragma unroll
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        half_t dt_shared_local_cast_3[2];
        *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_7 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __4;
        uint1 v__3 = *(uint1*)(dt_shared_local_cast_3 + 0);
        ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__3))[0]);
        *(float2*)(dt_local + (i_7 * 2)) = __4;
      }
      overlap_plan_mbar[11].arrive();
      overlap_plan_mbar[0].wait(((ik + 1) & 1));
      #pragma unroll
      for (int i_8 = 0; i_8 < 32; ++i_8) {
        x_local[i_8] = ((half_t*)x_shared)[((((((((i_8 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_8 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_8 & 3) >> 1) + (i_8 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      overlap_plan_mbar[9].arrive();
      #pragma unroll
      for (int i_9 = 0; i_9 < 16; ++i_9) {
        scale[i_9] = (exp2f(((da_last[0] - da_local[i_9]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_9]);
      }
      if (((ik + 1) % 3) == 0) {
        #pragma unroll
        for (int i_10 = 0; i_10 < 32; ++i_10) {
          xt_local_v0[i_10] = ((half_t)(((float)x_local[((((i_10 >> 2) * 4) + ((i_10 & 1) * 2)) + ((i_10 & 3) >> 1))]) * scale[(((i_10 >> 2) * 2) + (i_10 & 1))]));
        }
        #pragma unroll
        for (int i_11 = 0; i_11 < 16; ++i_11) {
          *(uint1*)(((half_t*)xt_local_wsp_handoff_3) + ((((((((ik + 1) % 3) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_11 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_11 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(xt_local_v0 + (i_11 * 2));
        }
      } else {
        if (((ik + 1) % 3) == 1) {
          #pragma unroll
          for (int i_12 = 0; i_12 < 32; ++i_12) {
            xt_local_v1[i_12] = ((half_t)(((float)x_local[((((i_12 >> 2) * 4) + ((i_12 & 1) * 2)) + ((i_12 & 3) >> 1))]) * scale[(((i_12 >> 2) * 2) + (i_12 & 1))]));
          }
          #pragma unroll
          for (int i_13 = 0; i_13 < 16; ++i_13) {
            *(uint1*)(((half_t*)xt_local_wsp_handoff_3) + ((((((((ik + 1) % 3) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_13 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_13 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(xt_local_v1 + (i_13 * 2));
          }
        } else {
          #pragma unroll
          for (int i_14 = 0; i_14 < 32; ++i_14) {
            xt_local_v2[i_14] = ((half_t)(((float)x_local[((((i_14 >> 2) * 4) + ((i_14 & 1) * 2)) + ((i_14 & 3) >> 1))]) * scale[(((i_14 >> 2) * 2) + (i_14 & 1))]));
          }
          #pragma unroll
          for (int i_15 = 0; i_15 < 16; ++i_15) {
            *(uint1*)(((half_t*)xt_local_wsp_handoff_3) + ((((((((ik + 1) % 3) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_15 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_15 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(xt_local_v2 + (i_15 * 2));
          }
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[(((ik + 1) % 3) + 3)].arrive();
    }
    overlap_plan_mbar[13].wait(0);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[7].arrive_and_expect_tx(4096);
      tl::tma_load(B_desc, overlap_plan_mbar[7], (&(((half_t*)b_shared)[2048])), (((int)blockIdx.y) * 32), (((((int)blockIdx.z) >> 3) * 256) + 192), 0, (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[8].wait(0);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store(Output_desc, (&(((half_t*)acc_shared)[0])), (((int)blockIdx.y) * 32), 0, ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i_16 = 0; i_16 < 4; ++i_16) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc + (i_16 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int ik_1 = 0; ik_1 < 1; ++ik_1) {
      overlap_plan_mbar[3].wait(0);
      overlap_plan_mbar[6].wait(0);
      #pragma unroll
      for (int i_17 = 0; i_17 < 16; ++i_17) {
        *(uint1*)(xt_local_v0 + (i_17 * 2)) = *(uint1*)(((half_t*)xt_local_wsp_handoff_3) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_17 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_17 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096));
      }
      {
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b, (&(((half_t*)b_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v0 + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local_v0 + (ki * 8)), uint64_t(desc_b + ((ki * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v0 + 0), 16);
      }
      overlap_plan_mbar[12].arrive();
      overlap_plan_mbar[4].wait(0);
      overlap_plan_mbar[7].wait(0);
      #pragma unroll
      for (int i_18 = 0; i_18 < 16; ++i_18) {
        *(uint1*)(xt_local_v1 + (i_18 * 2)) = *(uint1*)(((half_t*)xt_local_wsp_handoff_3) + ((((((((int)threadIdx.x) >> 5) * 1024) + ((i_18 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_18 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_1, (&(((half_t*)b_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, 4096);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v1 + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local_v1 + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v1 + 0), 16);
      }
      overlap_plan_mbar[13].arrive();
      overlap_plan_mbar[5].wait(0);
      overlap_plan_mbar[6].wait(1);
      #pragma unroll
      for (int i_19 = 0; i_19 < 16; ++i_19) {
        *(uint1*)(xt_local_v2 + (i_19 * 2)) = *(uint1*)(((half_t*)xt_local_wsp_handoff_3) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_19 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_19 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096));
      }
      {
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_2, (&(((half_t*)b_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v2 + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local_v2 + (ki_2 * 8)), uint64_t(desc_b_2 + ((ki_2 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v2 + 0), 16);
      }
      overlap_plan_mbar[12].arrive();
    }
    overlap_plan_mbar[3].wait(1);
    overlap_plan_mbar[7].wait(1);
    #pragma unroll
    for (int i_20 = 0; i_20 < 16; ++i_20) {
      *(uint1*)(xt_local_v0 + (i_20 * 2)) = *(uint1*)(((half_t*)xt_local_wsp_handoff_3) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_20 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_20 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096));
    }
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_3, (&(((half_t*)b_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_3, 4096);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v0 + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local_v0 + (ki_3 * 8)), uint64_t(desc_b_3 + ((ki_3 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v0 + 0), 16);
    }
    overlap_plan_mbar[13].arrive();
    #pragma unroll
    for (int i_21 = 0; i_21 < 2; ++i_21) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[((((((((int)threadIdx.x) & 127) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_21) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((half_t)acc[(i_21 * 8)]), ((half_t)acc[((i_21 * 8) + 1)])), __pack_half2(((half_t)acc[((i_21 * 8) + 2)]), ((half_t)acc[((i_21 * 8) + 3)])), __pack_half2(((half_t)acc[((i_21 * 8) + 4)]), ((half_t)acc[((i_21 * 8) + 5)])), __pack_half2(((half_t)acc[((i_21 * 8) + 6)]), ((half_t)acc[((i_21 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[8].arrive();
  }
}

