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
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* xt_local_wsp_handoff_4 = ((void*)((char*)buf_dyn_shmem + 32768));
  void* scale_wsp_handoff_3 = ((void*)((char*)buf_dyn_shmem + 40960));
  void* da_shared = ((void*)((char*)buf_dyn_shmem + 41984));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 43008));
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 44032));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[13];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float da_last[1];
  float acc[64];
  float scale[16];
  half_t xt_local[32];
  float da_local[16];
  float dt_local[16];
  half_t x_local[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(X_desc);
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(1);
    overlap_plan_mbar[4].init(128);
    overlap_plan_mbar[5].init(128);
    overlap_plan_mbar[6].init(1);
    overlap_plan_mbar[7].init(128);
    overlap_plan_mbar[8].init(128);
    overlap_plan_mbar[9].init(128);
    overlap_plan_mbar[10].init(128);
    overlap_plan_mbar[11].init(128);
    overlap_plan_mbar[12].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<240>();
    da_last[0] = ((float)DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 255)]);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int ik = 0; ik < 4; ++ik) {
      if (2 <= ik) {
        overlap_plan_mbar[(ik + 6)].wait(0);
      }
      tl::__sync_thread_partial(3, 128);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[(ik & 1)].arrive_and_expect_tx(8192);
        tl::tma_load(X_desc, overlap_plan_mbar[(ik & 1)], (&(((half_t*)x_shared)[((ik & 1) * 4096)])), 0, (((((int)blockIdx.z) >> 3) * 256) + (ik * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
      if (1 <= ik) {
        overlap_plan_mbar[10].wait(((ik + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64))])), overlap_plan_mbar[2], 128);
      }
      if (1 <= ik) {
        overlap_plan_mbar[11].wait(((ik + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[3].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64))])), overlap_plan_mbar[3], 128);
      }
      overlap_plan_mbar[2].wait((ik & 1));
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        half_t da_shared_local_cast[2];
        *(uint1*)(da_shared_local_cast + 0) = *(uint1*)(((half_t*)da_shared) + ((i_1 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __1;
        uint1 v_ = *(uint1*)(da_shared_local_cast + 0);
        ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
        *(float2*)(da_local + (i_1 * 2)) = __1;
      }
      overlap_plan_mbar[10].arrive();
      overlap_plan_mbar[3].wait((ik & 1));
      #pragma unroll
      for (int i_2 = 0; i_2 < 8; ++i_2) {
        half_t dt_shared_local_cast_1[2];
        *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_2 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __2;
        uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
        ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
        *(float2*)(dt_local + (i_2 * 2)) = __2;
      }
      overlap_plan_mbar[11].arrive();
      #pragma unroll
      for (int i_3 = 0; i_3 < 16; ++i_3) {
        scale[i_3] = (exp2f(((da_last[0] - da_local[i_3]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_3]);
      }
      if (((((((int)threadIdx.x) & 31) >> 2) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
        #pragma unroll
        for (int i_4 = 0; i_4 < 8; ++i_4) {
          *(float2*)(((float*)scale_wsp_handoff_3) + ((i_4 * 8) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(scale + (i_4 * 2));
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[4].arrive();
      if (1 <= ik) {
        overlap_plan_mbar[12].wait(((ik + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[6].arrive_and_expect_tx(16384);
        tl::tma_load(B_desc, overlap_plan_mbar[6], (&(((half_t*)b_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + (ik * 64)), 0, (((int)blockIdx.z) & 7));
        tl::tma_load(B_desc, overlap_plan_mbar[6], (&(((half_t*)b_shared)[4096])), 64, (((((int)blockIdx.z) >> 3) * 256) + (ik * 64)), 0, (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[5].wait((ik & 1));
      overlap_plan_mbar[6].wait((ik & 1));
      #pragma unroll
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        tl::ptx_ldmatrix_x4((&(((half_t*)xt_local_wsp_handoff_4)[(((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (i_5 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), (&(xt_local[(i_5 * 8)])));
      }
      {
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b, (&(((half_t*)b_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local + (ki * 8)), uint64_t(desc_b + ((ki * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 16);
      }
      overlap_plan_mbar[12].arrive();
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_6 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc[(i_6 * 8)]), ((half_t)acc[((i_6 * 8) + 1)])), __pack_half2(((half_t)acc[((i_6 * 8) + 2)]), ((half_t)acc[((i_6 * 8) + 3)])), __pack_half2(((half_t)acc[((i_6 * 8) + 4)]), ((half_t)acc[((i_6 * 8) + 5)])), __pack_half2(((half_t)acc[((i_6 * 8) + 6)]), ((half_t)acc[((i_6 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[7].arrive();
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_dealloc<40>();
      for (int ik_1 = 0; ik_1 < 4; ++ik_1) {
        overlap_plan_mbar[(ik_1 & 1)].wait((ik_1 >> 1));
        #pragma unroll
        for (int i_7 = 0; i_7 < 32; ++i_7) {
          x_local[i_7] = ((half_t*)x_shared)[(((((((((ik_1 & 1) * 4096) + ((i_7 >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (((i_7 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_7 & 3) >> 1) + (i_7 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
        }
        overlap_plan_mbar[((ik_1 & 1) + 8)].arrive();
        overlap_plan_mbar[4].wait((ik_1 & 1));
        #pragma unroll
        for (int i_8 = 0; i_8 < 8; ++i_8) {
          *(float2*)(scale + (i_8 * 2)) = *(float2*)(((float*)scale_wsp_handoff_3) + ((i_8 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        }
        #pragma unroll
        for (int i_9 = 0; i_9 < 32; ++i_9) {
          xt_local[i_9] = ((half_t)(((float)x_local[((((i_9 >> 2) * 4) + ((i_9 & 1) * 2)) + ((i_9 & 3) >> 1))]) * scale[(((i_9 >> 2) * 2) + (i_9 & 1))]));
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_10 = 0; i_10 < 4; ++i_10) {
          tl::ptx_stmatrix_m8n8_x4((&(((half_t*)xt_local_wsp_handoff_4)[((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (i_10 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 4096)])), __pack_half2(xt_local[(i_10 * 8)], xt_local[((i_10 * 8) + 1)]), __pack_half2(xt_local[((i_10 * 8) + 2)], xt_local[((i_10 * 8) + 3)]), __pack_half2(xt_local[((i_10 * 8) + 4)], xt_local[((i_10 * 8) + 5)]), __pack_half2(xt_local[((i_10 * 8) + 6)], xt_local[((i_10 * 8) + 7)]));
        }
        tl::fence_proxy_async();
        overlap_plan_mbar[5].arrive();
      }
    } else {
      tl::warpgroup_reg_dealloc<24>();
      overlap_plan_mbar[7].wait(0);
      if (tl::tl_shuffle_elect<128>()) {
        tl::tma_store((&(Output[((((((int)blockIdx.z) & 7) * 10485760) + ((((int)blockIdx.z) >> 3) * 655360)) + (((int)blockIdx.x) * 8192))])), (&(((half_t*)acc_shared)[0])), 16384);
        tl::tma_store_arrive();
        tl::tma_store_wait<0, true>();
      }
    }
  }
}

