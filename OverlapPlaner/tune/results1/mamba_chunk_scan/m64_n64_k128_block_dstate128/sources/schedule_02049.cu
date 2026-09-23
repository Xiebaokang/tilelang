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
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* c_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* residual_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* prev_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* scale_m_wsp_handoff_1 = ((void*)((char*)buf_dyn_shmem + 32768));
  void* da_m_shared = ((void*)((char*)buf_dyn_shmem + 33792));
  void* cb_shared = ((void*)((char*)buf_dyn_shmem + 34816));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 67584));
  void* cb_local_wsp_handoff_6 = ((void*)((char*)buf_dyn_shmem + 100352));
  void* da_k_shared = ((void*)((char*)buf_dyn_shmem + 116736));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 117760));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[20];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc[32];
  float scale_m[2];
  float dt_local[32];
  half_t cb_local[64];
  float d_local[1];
  float residual_local[32];
  float da_m_local[2];
  float da_k_local[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(C_desc);
    tl::prefetch_tma_descriptor(Prev_desc);
    tl::prefetch_tma_descriptor(CB_desc);
    tl::prefetch_tma_descriptor(X_desc);
    tl::prefetch_tma_descriptor(X_desc_1);
    tl::prefetch_tma_descriptor(Output_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(128);
    overlap_plan_mbar[1].init(128);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(1);
    overlap_plan_mbar[4].init(1);
    overlap_plan_mbar[5].init(1);
    overlap_plan_mbar[6].init(1);
    overlap_plan_mbar[7].init(1);
    overlap_plan_mbar[8].init(128);
    overlap_plan_mbar[9].init(1);
    overlap_plan_mbar[10].init(1);
    overlap_plan_mbar[11].init(1);
    overlap_plan_mbar[12].init(1);
    overlap_plan_mbar[13].init(128);
    overlap_plan_mbar[14].init(128);
    overlap_plan_mbar[15].init(128);
    overlap_plan_mbar[16].init(128);
    overlap_plan_mbar[17].init(128);
    overlap_plan_mbar[18].init(128);
    overlap_plan_mbar[19].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<240>();
    if (((int)threadIdx.x) < 64) {
      ((half_t*)da_m_shared)[((int)threadIdx.x)] = DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x))];
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[0].arrive();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(16384);
      tl::tma_load(C_desc, overlap_plan_mbar[2], (&(((half_t*)c_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + (((int)blockIdx.y) * 64)), 0, (((int)blockIdx.z) & 7));
      tl::tma_load(C_desc, overlap_plan_mbar[2], (&(((half_t*)c_shared)[4096])), 64, (((((int)blockIdx.z) >> 3) * 256) + (((int)blockIdx.y) * 64)), 0, (((int)blockIdx.z) & 7));
      overlap_plan_mbar[3].arrive_and_expect_tx(16384);
      tl::tma_load(Prev_desc, overlap_plan_mbar[3], (&(((half_t*)prev_shared)[0])), 0, 0, ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
      tl::tma_load(Prev_desc, overlap_plan_mbar[3], (&(((half_t*)prev_shared)[4096])), 64, 0, ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[2].wait(0);
    overlap_plan_mbar[3].wait(0);
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)c_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)prev_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    }
    overlap_plan_mbar[1].wait(0);
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      scale_m[i_1] = ((float*)scale_m_wsp_handoff_1)[((((((int)threadIdx.x) >> 5) * 16) + (i_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 32; ++i_2) {
      acc[i_2] = (acc[i_2] * scale_m[((i_2 & 3) >> 1)]);
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(16384);
      tl::tma_load(CB_desc, overlap_plan_mbar[4], (&(((half_t*)cb_shared)[0])), 0, (((int)blockIdx.y) * 64), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
      tl::tma_load(CB_desc, overlap_plan_mbar[4], (&(((half_t*)cb_shared)[4096])), 64, (((int)blockIdx.y) * 64), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
      overlap_plan_mbar[6].arrive_and_expect_tx(256);
      tl::tma_load((&(((half_t*)da_k_shared)[0])), (&(DA[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[6], 256);
      overlap_plan_mbar[9].arrive_and_expect_tx(256);
      tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256))])), overlap_plan_mbar[9], 256);
      overlap_plan_mbar[10].arrive_and_expect_tx(16384);
      tl::tma_load(X_desc, overlap_plan_mbar[10], (&(((half_t*)x_shared)[0])), 0, ((((int)blockIdx.z) >> 3) * 256), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[9].wait(0);
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      half_t dt_shared_local_cast[2];
      *(uint1*)(dt_shared_local_cast + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_3 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __1;
      uint1 v_ = *(uint1*)(dt_shared_local_cast + 0);
      ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
      *(float2*)(dt_local + (i_3 * 2)) = __1;
    }
    overlap_plan_mbar[17].arrive();
    overlap_plan_mbar[8].wait(0);
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      tl::ptx_ldmatrix_x4((&(((half_t*)cb_local_wsp_handoff_6)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_4 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), (&(cb_local[(i_4 * 8)])));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 32; ++i_5) {
      uint1 __2;
      float2 __3;
        float2 __4;
        uint1 v__1 = *(uint1*)(cb_local + (i_5 * 2));
        ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__1))[0]);
        float2 v__2 = *(float2*)(dt_local + ((i_5 >> 1) * 2));
        __3.x = (__4.x*v__2.x);
        __3.y = (__4.y*v__2.y);
      ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&__3))[0]);
      *(uint1*)(cb_local + (i_5 * 2)) = __2;
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 64; ++i_6) {
      half_t condval;
      if ((((((i_6 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) <= ((((((int)blockIdx.y) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_6 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval = cb_local[i_6];
      } else {
        condval = half_t(0x0p+0f/*0.000000e+00*/);
      }
      cb_local[i_6] = condval;
    }
    tl::__sync_thread_partial(3, 128);
    for (int ik = 0; ik < (((int)blockIdx.y) >> 1); ++ik) {
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[(ik + 5)].arrive_and_expect_tx(16384);
        tl::tma_load(CB_desc, overlap_plan_mbar[(ik + 5)], (&(((half_t*)cb_shared)[((ik * 8192) + 8192)])), ((ik * 128) + 128), (((int)blockIdx.y) * 64), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
        tl::tma_load(CB_desc, overlap_plan_mbar[(ik + 5)], (&(((half_t*)cb_shared)[((ik * 8192) + 12288)])), ((ik * 128) + 192), (((int)blockIdx.y) * 64), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
        overlap_plan_mbar[(ik + 7)].arrive_and_expect_tx(256);
        tl::tma_load((&(((half_t*)da_k_shared)[((ik * 128) + 128)])), (&(DA[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 128)) + 128)])), overlap_plan_mbar[(ik + 7)], 256);
      }
      overlap_plan_mbar[17].wait(ik);
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[9].arrive_and_expect_tx(256);
        tl::tma_load((&(((half_t*)dt_shared)[0])), (&(Dt[((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 128)) + 128)])), overlap_plan_mbar[9], 256);
        overlap_plan_mbar[(ik + 11)].arrive_and_expect_tx(16384);
        tl::tma_load(X_desc, overlap_plan_mbar[(ik + 11)], (&(((half_t*)x_shared)[((ik * 8192) + 8192)])), 0, ((((((int)blockIdx.z) >> 3) * 256) + (ik * 128)) + 128), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
      overlap_plan_mbar[(ik + 10)].wait(0);
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_1, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, (ik * 16384));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local + (ki_1 * 8)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local + 0), 32);
      }
      overlap_plan_mbar[(ik + 18)].arrive();
      overlap_plan_mbar[9].wait((ik + 1));
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_7 = 0; i_7 < 16; ++i_7) {
        half_t dt_shared_local_cast_1[2];
        *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_7 * 8) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __5;
        uint1 v__3 = *(uint1*)(dt_shared_local_cast_1 + 0);
        ((float2*)(&__5))[0] = __half22float2(((half2*)(&v__3))[0]);
        *(float2*)(dt_local + (i_7 * 2)) = __5;
      }
      overlap_plan_mbar[17].arrive();
      overlap_plan_mbar[8].wait((ik + 1));
      #pragma unroll
      for (int i_8 = 0; i_8 < 8; ++i_8) {
        tl::ptx_ldmatrix_x4((&(((half_t*)cb_local_wsp_handoff_6)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_8 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), (&(cb_local[(i_8 * 8)])));
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 32; ++i_9) {
        uint1 __6;
        float2 __7;
          float2 __8;
          uint1 v__4 = *(uint1*)(cb_local + (i_9 * 2));
          ((float2*)(&__8))[0] = __half22float2(((half2*)(&v__4))[0]);
          float2 v__5 = *(float2*)(dt_local + ((i_9 >> 1) * 2));
          __7.x = (__8.x*v__5.x);
          __7.y = (__8.y*v__5.y);
        ((half2*)(&__6))[0] = __float22half2_rn(((float2*)(&__7))[0]);
        *(uint1*)(cb_local + (i_9 * 2)) = __6;
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 64; ++i_10) {
        half_t condval_1;
        if (((((((ik * 128) + ((i_10 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_10 & 1)) + 128) <= ((((((int)blockIdx.y) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_10 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval_1 = cb_local[i_10];
        } else {
          condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local[i_10] = condval_1;
      }
    }
    overlap_plan_mbar[((((int)blockIdx.y) >> 1) + 10)].wait(0);
    {
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_2, (&(((half_t*)x_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_2, ((((int)blockIdx.y) >> 1) * 16384));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
        tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local + (ki_2 * 8)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local + 0), 32);
    }
    overlap_plan_mbar[((((int)blockIdx.y) >> 1) + 18)].arrive();
    d_local[0] = ((float)D[((int)blockIdx.x)]);
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[12].arrive_and_expect_tx(8192);
      tl::tma_load(X_desc_1, overlap_plan_mbar[12], (&(((half_t*)residual_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + (((int)blockIdx.y) * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[12].wait(0);
    #pragma unroll
    for (int i_11 = 0; i_11 < 16; ++i_11) {
      half_t residual_shared_local_cast_2[2];
      *(uint1*)(residual_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)residual_shared) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_11 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_11 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_11 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_11 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __9;
      uint1 v__6 = *(uint1*)(residual_shared_local_cast_2 + 0);
      ((float2*)(&__9))[0] = __half22float2(((half2*)(&v__6))[0]);
      *(float2*)(residual_local + (i_11 * 2)) = __9;
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 32; ++i_12) {
      acc[i_12] = (acc[i_12] + (residual_local[i_12] * d_local[0]));
    }
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_13 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc[(i_13 * 8)]), ((half_t)acc[((i_13 * 8) + 1)])), __pack_half2(((half_t)acc[((i_13 * 8) + 2)]), ((half_t)acc[((i_13 * 8) + 3)])), __pack_half2(((half_t)acc[((i_13 * 8) + 4)]), ((half_t)acc[((i_13 * 8) + 5)])), __pack_half2(((half_t)acc[((i_13 * 8) + 6)]), ((half_t)acc[((i_13 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::fence_proxy_async();
      tl::tma_store(Output_desc, (&(((half_t*)acc_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + (((int)blockIdx.y) * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  } else {
    tl::warpgroup_reg_dealloc<40>();
    overlap_plan_mbar[0].wait(0);
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      da_m_local[i_14] = ((float)((half_t*)da_m_shared)[(((((((int)threadIdx.x) >> 5) * 16) + (i_14 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      scale_m[i_15] = exp2f((da_m_local[i_15] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
    }
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_16 = 0; i_16 < 2; ++i_16) {
        ((float*)scale_m_wsp_handoff_1)[(((((((int)threadIdx.x) >> 5) * 16) + (i_16 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)] = scale_m[i_16];
      }
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[1].arrive();
    for (int ik_1 = 0; ik_1 < ((((int)blockIdx.y) >> 1) + 1); ++ik_1) {
      overlap_plan_mbar[(ik_1 + 4)].wait(0);
      #pragma unroll
      for (int i_17 = 0; i_17 < 32; ++i_17) {
        *(uint1*)(cb_local + (i_17 * 2)) = *(uint1*)(((half_t*)cb_shared) + ((((((((((ik_1 * 8192) + ((i_17 >> 4) * 4096)) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_17 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_17 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_17 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_17 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096));
      }
      overlap_plan_mbar[(ik_1 + 13)].arrive();
      overlap_plan_mbar[(ik_1 + 6)].wait(0);
      #pragma unroll
      for (int i_18 = 0; i_18 < 16; ++i_18) {
        half_t da_k_shared_local_cast_3[2];
        *(uint1*)(da_k_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)da_k_shared) + (((ik_1 * 128) + (i_18 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __10;
        uint1 v__7 = *(uint1*)(da_k_shared_local_cast_3 + 0);
        ((float2*)(&__10))[0] = __half22float2(((half2*)(&v__7))[0]);
        *(float2*)(da_k_local + (i_18 * 2)) = __10;
      }
      overlap_plan_mbar[(ik_1 + 15)].arrive();
      #pragma unroll
      for (int i_19 = 0; i_19 < 32; ++i_19) {
        float broadcast_var_1 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
        uint1 __11;
        float2 __12;
          float2 __13;
          uint1 v__8 = *(uint1*)(cb_local + (i_19 * 2));
          ((float2*)(&__13))[0] = __half22float2(((half2*)(&v__8))[0]);
          float2 __14;
          float2 __15;
            float2 __16;
              float2 v__9 = make_float2(da_m_local[(i_19 & 1)], da_m_local[(i_19 & 1)]);
              float2 v__10 = *(float2*)(da_k_local + ((i_19 >> 1) * 2));
              __16.x = (v__9.x-v__10.x);
              __16.y = (v__9.y-v__10.y);
            float2 v__11 = make_float2(broadcast_var_1, broadcast_var_1);
            __15.x = (__16.x*v__11.x);
            __15.y = (__16.y*v__11.y);
          __14.x = exp2f(__15.x);
          __14.y = exp2f(__15.y);
          __12.x = (__13.x*__14.x);
          __12.y = (__13.y*__14.y);
        ((half2*)(&__11))[0] = __float22half2_rn(((float2*)(&__12))[0]);
        *(uint1*)(cb_local + (i_19 * 2)) = __11;
      }
      tl::__sync_thread_partial(4, 128);
      #pragma unroll
      for (int i_20 = 0; i_20 < 8; ++i_20) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)cb_local_wsp_handoff_6)[((((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_20 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 8192)])), __pack_half2(cb_local[(i_20 * 8)], cb_local[((i_20 * 8) + 1)]), __pack_half2(cb_local[((i_20 * 8) + 2)], cb_local[((i_20 * 8) + 3)]), __pack_half2(cb_local[((i_20 * 8) + 4)], cb_local[((i_20 * 8) + 5)]), __pack_half2(cb_local[((i_20 * 8) + 6)], cb_local[((i_20 * 8) + 7)]));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[8].arrive();
    }
  }
}

