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
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap B_desc, const half_t* __restrict__ DA, const half_t* __restrict__ Dt, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap X_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* xt_local_wsp_handoff_3 = ((void*)((char*)buf_dyn_shmem + 16384));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  void* da_shared = ((void*)((char*)buf_dyn_shmem + 28672));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 29696));
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 30720));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[13];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float da_last[1];
  half_t xt_local_v0[16];
  half_t xt_local_v1[16];
  float acc[64];
  float da_local[8];
  float dt_local[8];
  half_t x_local[16];
  float scale[8];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(X_desc);
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(128);
    overlap_plan_mbar[4].init(128);
    overlap_plan_mbar[5].init(1);
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
    tl::warpgroup_reg_dealloc<32>();
    da_last[0] = ((float)DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 255)]);
    for (int ik = 0; ik < 4; ++ik) {
      if (1 <= ik) {
        overlap_plan_mbar[8].wait(1);
      }
      tl::__sync_thread_partial(3, 128);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[0].arrive_and_expect_tx(4096);
        tl::tma_load(X_desc, overlap_plan_mbar[0], (&(((half_t*)x_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + (ik * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
      if (1 <= ik) {
        overlap_plan_mbar[9].wait(1);
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[1].arrive_and_expect_tx(64);
        tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64))])), overlap_plan_mbar[1], 64);
      }
      if (1 <= ik) {
        overlap_plan_mbar[10].wait(1);
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(64);
        tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64))])), overlap_plan_mbar[2], 64);
      }
      if (1 <= ik) {
        overlap_plan_mbar[11].wait(((ik + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[5].arrive_and_expect_tx(8192);
        tl::tma_load(B_desc, overlap_plan_mbar[5], (&(((half_t*)b_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + (ik * 64)), 0, (((int)blockIdx.z) & 7));
        tl::tma_load(B_desc, overlap_plan_mbar[5], (&(((half_t*)b_shared)[2048])), 64, (((((int)blockIdx.z) >> 3) * 256) + (ik * 64)), 0, (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[1].wait(0);
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        half_t da_shared_local_cast[2];
        *(uint1*)(da_shared_local_cast + 0) = *(uint1*)(((half_t*)da_shared) + ((i * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __1;
        uint1 v_ = *(uint1*)(da_shared_local_cast + 0);
        ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
        *(float2*)(da_local + (i * 2)) = __1;
      }
      overlap_plan_mbar[9].arrive();
      overlap_plan_mbar[2].wait(0);
      #pragma unroll
      for (int i_1 = 0; i_1 < 4; ++i_1) {
        half_t dt_shared_local_cast_1[2];
        *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_1 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __2;
        uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
        ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
        *(float2*)(dt_local + (i_1 * 2)) = __2;
      }
      overlap_plan_mbar[10].arrive();
      overlap_plan_mbar[0].wait(0);
      #pragma unroll
      for (int i_2 = 0; i_2 < 16; ++i_2) {
        x_local[i_2] = ((half_t*)x_shared)[((((((((i_2 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_2 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_2 & 3) >> 1) + (i_2 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      overlap_plan_mbar[8].arrive();
      #pragma unroll
      for (int i_3 = 0; i_3 < 8; ++i_3) {
        scale[i_3] = (exp2f(((da_last[0] - da_local[i_3]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_3]);
      }
      #pragma unroll
      for (int i_4 = 0; i_4 < 16; ++i_4) {
        xt_local_v0[i_4] = ((half_t)(((float)x_local[((((i_4 >> 2) * 4) + ((i_4 & 1) * 2)) + ((i_4 & 3) >> 1))]) * scale[(((i_4 >> 2) * 2) + (i_4 & 1))]));
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 8; ++i_5) {
        *(uint1*)(((half_t*)xt_local_wsp_handoff_3) + ((((((((int)threadIdx.x) >> 5) * 512) + ((i_5 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((i_5 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(xt_local_v0 + (i_5 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[3].arrive();
      overlap_plan_mbar[8].wait(0);
      tl::__sync_thread_partial(3, 128);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[0].arrive_and_expect_tx(4096);
        tl::tma_load(X_desc, overlap_plan_mbar[0], (&(((half_t*)x_shared)[0])), 0, ((((((int)blockIdx.z) >> 3) * 256) + (ik * 64)) + 32), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[9].wait(0);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[1].arrive_and_expect_tx(64);
        tl::tma_load((&(((half_t*)da_shared)[0])), (&(DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 32)])), overlap_plan_mbar[1], 64);
      }
      overlap_plan_mbar[10].wait(0);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(64);
        tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 64)) + 32)])), overlap_plan_mbar[2], 64);
      }
      if (1 <= ik) {
        overlap_plan_mbar[12].wait(((ik + 1) & 1));
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[6].arrive_and_expect_tx(8192);
        tl::tma_load(B_desc, overlap_plan_mbar[6], (&(((half_t*)b_shared)[4096])), 0, ((((((int)blockIdx.z) >> 3) * 256) + (ik * 64)) + 32), 0, (((int)blockIdx.z) & 7));
        tl::tma_load(B_desc, overlap_plan_mbar[6], (&(((half_t*)b_shared)[6144])), 64, ((((((int)blockIdx.z) >> 3) * 256) + (ik * 64)) + 32), 0, (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[1].wait(1);
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_6 = 0; i_6 < 4; ++i_6) {
        half_t da_shared_local_cast_2[2];
        *(uint1*)(da_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)da_shared) + ((i_6 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __3;
        uint1 v__2 = *(uint1*)(da_shared_local_cast_2 + 0);
        ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__2))[0]);
        *(float2*)(da_local + (i_6 * 2)) = __3;
      }
      overlap_plan_mbar[9].arrive();
      overlap_plan_mbar[2].wait(1);
      #pragma unroll
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        half_t dt_shared_local_cast_3[2];
        *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_7 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __4;
        uint1 v__3 = *(uint1*)(dt_shared_local_cast_3 + 0);
        ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__3))[0]);
        *(float2*)(dt_local + (i_7 * 2)) = __4;
      }
      overlap_plan_mbar[10].arrive();
      overlap_plan_mbar[0].wait(1);
      #pragma unroll
      for (int i_8 = 0; i_8 < 16; ++i_8) {
        x_local[i_8] = ((half_t*)x_shared)[((((((((i_8 >> 2) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_8 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_8 & 3) >> 1) + (i_8 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      overlap_plan_mbar[8].arrive();
      #pragma unroll
      for (int i_9 = 0; i_9 < 8; ++i_9) {
        scale[i_9] = (exp2f(((da_last[0] - da_local[i_9]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_9]);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 16; ++i_10) {
        xt_local_v1[i_10] = ((half_t)(((float)x_local[((((i_10 >> 2) * 4) + ((i_10 & 1) * 2)) + ((i_10 & 3) >> 1))]) * scale[(((i_10 >> 2) * 2) + (i_10 & 1))]));
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 8; ++i_11) {
        *(uint1*)(((half_t*)xt_local_wsp_handoff_3) + (((((((((int)threadIdx.x) >> 5) * 512) + ((i_11 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((i_11 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048)) = *(uint1*)(xt_local_v1 + (i_11 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[4].arrive();
    }
    overlap_plan_mbar[7].wait(0);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store((&(Output[((((((int)blockIdx.z) & 7) * 10485760) + ((((int)blockIdx.z) >> 3) * 655360)) + (((int)blockIdx.x) * 8192))])), (&(((half_t*)acc_shared)[0])), 16384);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i_12 = 0; i_12 < 16; ++i_12) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc + (i_12 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int ik_1 = 0; ik_1 < 4; ++ik_1) {
      overlap_plan_mbar[3].wait((ik_1 & 1));
      overlap_plan_mbar[5].wait((ik_1 & 1));
      #pragma unroll
      for (int i_13 = 0; i_13 < 8; ++i_13) {
        *(uint1*)(xt_local_v0 + (i_13 * 2)) = *(uint1*)(((half_t*)xt_local_wsp_handoff_3) + (((((((((int)threadIdx.x) >> 5) * 512) + ((i_13 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((i_13 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 2048));
      }
      {
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 256, 64>(desc_b, (&(((half_t*)b_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v0 + 0), 8);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local_v0 + (ki * 8)), uint64_t(desc_b + ((ki * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v0 + 0), 8);
      }
      overlap_plan_mbar[11].arrive();
      overlap_plan_mbar[4].wait((ik_1 & 1));
      overlap_plan_mbar[6].wait((ik_1 & 1));
      #pragma unroll
      for (int i_14 = 0; i_14 < 8; ++i_14) {
        *(uint1*)(xt_local_v1 + (i_14 * 2)) = *(uint1*)(((half_t*)xt_local_wsp_handoff_3) + ((((((((int)threadIdx.x) >> 5) * 512) + ((i_14 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((i_14 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 256, 64>(desc_b_1, (&(((half_t*)b_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, 8192);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v1 + 0), 8);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local_v1 + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local_v1 + 0), 8);
      }
      overlap_plan_mbar[12].arrive();
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 8; ++i_15) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[((((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_15 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 8192)])), __pack_half2(((half_t)acc[(i_15 * 8)]), ((half_t)acc[((i_15 * 8) + 1)])), __pack_half2(((half_t)acc[((i_15 * 8) + 2)]), ((half_t)acc[((i_15 * 8) + 3)])), __pack_half2(((half_t)acc[((i_15 * 8) + 4)]), ((half_t)acc[((i_15 * 8) + 5)])), __pack_half2(((half_t)acc[((i_15 * 8) + 6)]), ((half_t)acc[((i_15 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[7].arrive();
  }
}

