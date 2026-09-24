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
  void* c_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* prev_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* da_m_shared = ((void*)((char*)buf_dyn_shmem + 40960));
  void* cb_shared = ((void*)((char*)buf_dyn_shmem + 41984));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 74752));
  void* da_k_shared = ((void*)((char*)buf_dyn_shmem + 82944));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 83968));
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 84992));
  void* residual_shared = ((void*)((char*)buf_dyn_shmem + 93184));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[21];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float da_m_local[4];
  float acc[32];
  float scale_m[4];
  float da_k_local_v0[16];
  float dt_local_v0[16];
  float da_k_local_v1[16];
  float dt_local_v1[16];
  half_t cb_local[64];
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
    overlap_plan_mbar[0].init(128);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(1);
    overlap_plan_mbar[4].init(1);
    overlap_plan_mbar[5].init(1);
    overlap_plan_mbar[6].init(1);
    overlap_plan_mbar[7].init(1);
    overlap_plan_mbar[8].init(1);
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
    tl::warpgroup_reg_alloc<240>();
    overlap_plan_mbar[0].wait(0);
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
    overlap_plan_mbar[1].wait(0);
    overlap_plan_mbar[2].wait(0);
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)c_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)prev_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
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
    overlap_plan_mbar[5].wait(0);
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      half_t da_k_shared_local_cast[2];
      *(uint1*)(da_k_shared_local_cast + 0) = *(uint1*)(((half_t*)da_k_shared) + ((i_5 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __1;
      uint1 v_ = *(uint1*)(da_k_shared_local_cast + 0);
      ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
      *(float2*)(da_k_local_v0 + (i_5 * 2)) = __1;
    }
    overlap_plan_mbar[15].arrive();
    overlap_plan_mbar[7].wait(0);
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      half_t dt_shared_local_cast_1[2];
      *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_6 * 8) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __2;
      uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
      ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
      *(float2*)(dt_local_v0 + (i_6 * 2)) = __2;
    }
    overlap_plan_mbar[17].arrive();
    overlap_plan_mbar[6].wait(0);
    #pragma unroll
    for (int i_7 = 0; i_7 < 8; ++i_7) {
      half_t da_k_shared_local_cast_2[2];
      *(uint1*)(da_k_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)da_k_shared) + (((i_7 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
      float2 __3;
      uint1 v__2 = *(uint1*)(da_k_shared_local_cast_2 + 0);
      ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__2))[0]);
      *(float2*)(da_k_local_v1 + (i_7 * 2)) = __3;
    }
    overlap_plan_mbar[16].arrive();
    overlap_plan_mbar[8].wait(0);
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      half_t dt_shared_local_cast_3[2];
      *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + (((i_8 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
      float2 __4;
      uint1 v__3 = *(uint1*)(dt_shared_local_cast_3 + 0);
      ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__3))[0]);
      *(float2*)(dt_local_v1 + (i_8 * 2)) = __4;
    }
    overlap_plan_mbar[18].arrive();
    overlap_plan_mbar[3].wait(0);
    #pragma unroll
    for (int i_9 = 0; i_9 < 32; ++i_9) {
      *(uint1*)(cb_local + (i_9 * 2)) = *(uint1*)(((half_t*)cb_shared) + ((((((((((i_9 & 7) >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_9 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((i_9 >> 3) * 16) + (((i_9 & 3) >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_9 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_9 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    overlap_plan_mbar[13].arrive();
    for (int ik = 0; ik < ((((int)blockIdx.y) >> 1) * 2); ++ik) {
      if (ik == 0) {
        #pragma unroll
        for (int i_10 = 0; i_10 < 32; ++i_10) {
          float broadcast_var_1 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
          uint1 __5;
          float2 __6;
            float2 __7;
            uint1 v__4 = *(uint1*)(cb_local + (i_10 * 2));
            ((float2*)(&__7))[0] = __half22float2(((half2*)(&v__4))[0]);
            float2 __8;
            float2 __9;
              float2 __10;
                float2 v__5 = make_float2(da_m_local[((((i_10 & 7) >> 2) * 2) + (i_10 & 1))], da_m_local[((((i_10 & 7) >> 2) * 2) + (i_10 & 1))]);
                float2 v__6 = *(float2*)(da_k_local_v0 + (((i_10 >> 3) * 4) + (((i_10 & 3) >> 1) * 2)));
                __10.x = (v__5.x-v__6.x);
                __10.y = (v__5.y-v__6.y);
              float2 v__7 = make_float2(broadcast_var_1, broadcast_var_1);
              __9.x = (__10.x*v__7.x);
              __9.y = (__10.y*v__7.y);
            __8.x = exp2f(__9.x);
            __8.y = exp2f(__9.y);
            __6.x = (__7.x*__8.x);
            __6.y = (__7.y*__8.y);
          ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&__6))[0]);
          *(uint1*)(cb_local + (i_10 * 2)) = __5;
        }
      } else {
        #pragma unroll
        for (int i_11 = 0; i_11 < 32; ++i_11) {
          float broadcast_var_2 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
          uint1 __11;
          float2 __12;
            float2 __13;
            uint1 v__8 = *(uint1*)(cb_local + (i_11 * 2));
            ((float2*)(&__13))[0] = __half22float2(((half2*)(&v__8))[0]);
            float2 __14;
            float2 __15;
              float2 __16;
                float2 v__9 = make_float2(da_m_local[((((i_11 & 7) >> 2) * 2) + (i_11 & 1))], da_m_local[((((i_11 & 7) >> 2) * 2) + (i_11 & 1))]);
                float2 v__10 = *(float2*)(da_k_local_v1 + (((i_11 >> 3) * 4) + (((i_11 & 3) >> 1) * 2)));
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
          *(uint1*)(cb_local + (i_11 * 2)) = __11;
        }
      }
      if (ik == 0) {
        #pragma unroll
        for (int i_12 = 0; i_12 < 32; ++i_12) {
          uint1 __17;
          float2 __18;
            float2 __19;
            uint1 v__12 = *(uint1*)(cb_local + (i_12 * 2));
            ((float2*)(&__19))[0] = __half22float2(((half2*)(&v__12))[0]);
            float2 v__13 = *(float2*)(dt_local_v0 + (((i_12 >> 3) * 4) + (((i_12 & 3) >> 1) * 2)));
            __18.x = (__19.x*v__13.x);
            __18.y = (__19.y*v__13.y);
          ((half2*)(&__17))[0] = __float22half2_rn(((float2*)(&__18))[0]);
          *(uint1*)(cb_local + (i_12 * 2)) = __17;
        }
      } else {
        #pragma unroll
        for (int i_13 = 0; i_13 < 32; ++i_13) {
          uint1 __20;
          float2 __21;
            float2 __22;
            uint1 v__14 = *(uint1*)(cb_local + (i_13 * 2));
            ((float2*)(&__22))[0] = __half22float2(((half2*)(&v__14))[0]);
            float2 v__15 = *(float2*)(dt_local_v1 + (((i_13 >> 3) * 4) + (((i_13 & 3) >> 1) * 2)));
            __21.x = (__22.x*v__15.x);
            __21.y = (__22.y*v__15.y);
          ((half2*)(&__20))[0] = __float22half2_rn(((float2*)(&__21))[0]);
          *(uint1*)(cb_local + (i_13 * 2)) = __20;
        }
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 64; ++i_14) {
        half_t condval;
        if (((((((ik * 64) + ((i_14 >> 4) * 16)) + (((i_14 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_14 & 1)) <= ((((((((int)blockIdx.y) >> 1) * 128) + (((i_14 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_14 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval = cb_local[i_14];
        } else {
          condval = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local[i_14] = condval;
      }
      overlap_plan_mbar[(ik + 9)].wait(0);
      {
        tl::GmmaDescriptor desc_b_1;
        tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_1, (&(((half_t*)x_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, (ik * 4096));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int i_15 = 0; i_15 < 2; ++i_15) {
          #pragma unroll
          for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
            tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local + ((ki_1 * 16) + (i_15 * 8))), uint64_t(desc_b_1 + ((ki_1 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_15 * 16)), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local + 0), 32);
      }
      overlap_plan_mbar[(ik + 19)].arrive();
      overlap_plan_mbar[(ik + 5)].wait(1);
      if (ik == 0) {
        #pragma unroll
        for (int i_16 = 0; i_16 < 8; ++i_16) {
          half_t da_k_shared_local_cast_4[2];
          *(uint1*)(da_k_shared_local_cast_4 + 0) = *(uint1*)(((half_t*)da_k_shared) + (((ik * 64) + (i_16 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
          float2 __23;
          uint1 v__16 = *(uint1*)(da_k_shared_local_cast_4 + 0);
          ((float2*)(&__23))[0] = __half22float2(((half2*)(&v__16))[0]);
          *(float2*)(da_k_local_v0 + (i_16 * 2)) = __23;
        }
      } else {
        #pragma unroll
        for (int i_17 = 0; i_17 < 8; ++i_17) {
          half_t da_k_shared_local_cast_5[2];
          *(uint1*)(da_k_shared_local_cast_5 + 0) = *(uint1*)(((half_t*)da_k_shared) + (((ik * 64) + (i_17 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
          float2 __24;
          uint1 v__17 = *(uint1*)(da_k_shared_local_cast_5 + 0);
          ((float2*)(&__24))[0] = __half22float2(((half2*)(&v__17))[0]);
          *(float2*)(da_k_local_v1 + (i_17 * 2)) = __24;
        }
      }
      overlap_plan_mbar[(ik + 15)].arrive();
      overlap_plan_mbar[(ik + 7)].wait(1);
      if (ik == 0) {
        #pragma unroll
        for (int i_18 = 0; i_18 < 8; ++i_18) {
          half_t dt_shared_local_cast_6[2];
          *(uint1*)(dt_shared_local_cast_6 + 0) = *(uint1*)(((half_t*)dt_shared) + (((ik * 64) + (i_18 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
          float2 __25;
          uint1 v__18 = *(uint1*)(dt_shared_local_cast_6 + 0);
          ((float2*)(&__25))[0] = __half22float2(((half2*)(&v__18))[0]);
          *(float2*)(dt_local_v0 + (i_18 * 2)) = __25;
        }
      } else {
        #pragma unroll
        for (int i_19 = 0; i_19 < 8; ++i_19) {
          half_t dt_shared_local_cast_7[2];
          *(uint1*)(dt_shared_local_cast_7 + 0) = *(uint1*)(((half_t*)dt_shared) + (((ik * 64) + (i_19 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
          float2 __26;
          uint1 v__19 = *(uint1*)(dt_shared_local_cast_7 + 0);
          ((float2*)(&__26))[0] = __half22float2(((half2*)(&v__19))[0]);
          *(float2*)(dt_local_v1 + (i_19 * 2)) = __26;
        }
      }
      overlap_plan_mbar[(ik + 17)].arrive();
      overlap_plan_mbar[(((ik + 1) & 1) + 3)].wait(((ik + 1) >> 1));
      #pragma unroll
      for (int i_20 = 0; i_20 < 32; ++i_20) {
        *(uint1*)(cb_local + (i_20 * 2)) = *(uint1*)(((half_t*)cb_shared) + (((((((((((ik + 1) & 1) * 8192) + (((i_20 & 7) >> 2) * 4096)) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_20 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((i_20 >> 3) * 16) + (((i_20 & 3) >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_20 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_20 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      overlap_plan_mbar[(((ik + 1) & 1) + 13)].arrive();
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 32; ++i_21) {
      float broadcast_var_3 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
      uint1 __27;
      float2 __28;
        float2 __29;
        uint1 v__20 = *(uint1*)(cb_local + (i_21 * 2));
        ((float2*)(&__29))[0] = __half22float2(((half2*)(&v__20))[0]);
        float2 __30;
        float2 __31;
          float2 __32;
            float2 v__21 = make_float2(da_m_local[((((i_21 & 7) >> 2) * 2) + (i_21 & 1))], da_m_local[((((i_21 & 7) >> 2) * 2) + (i_21 & 1))]);
            float2 v__22 = *(float2*)(da_k_local_v0 + (((i_21 >> 3) * 4) + (((i_21 & 3) >> 1) * 2)));
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
      *(uint1*)(cb_local + (i_21 * 2)) = __27;
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 32; ++i_22) {
      uint1 __33;
      float2 __34;
        float2 __35;
        uint1 v__24 = *(uint1*)(cb_local + (i_22 * 2));
        ((float2*)(&__35))[0] = __half22float2(((half2*)(&v__24))[0]);
        float2 v__25 = *(float2*)(dt_local_v0 + (((i_22 >> 3) * 4) + (((i_22 & 3) >> 1) * 2)));
        __34.x = (__35.x*v__25.x);
        __34.y = (__35.y*v__25.y);
      ((half2*)(&__33))[0] = __float22half2_rn(((float2*)(&__34))[0]);
      *(uint1*)(cb_local + (i_22 * 2)) = __33;
    }
    #pragma unroll
    for (int i_23 = 0; i_23 < 64; ++i_23) {
      half_t condval_1;
      if ((((((((((int)blockIdx.y) >> 1) * 128) + ((i_23 >> 4) * 16)) + (((i_23 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_23 & 1)) <= ((((((((int)blockIdx.y) >> 1) * 128) + (((i_23 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_23 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_1 = cb_local[i_23];
      } else {
        condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      cb_local[i_23] = condval_1;
    }
    overlap_plan_mbar[9].wait((((int)blockIdx.y) >> 1));
    {
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_2, (&(((half_t*)x_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_2, 0);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_24 = 0; i_24 < 2; ++i_24) {
        #pragma unroll
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local + ((ki_2 * 16) + (i_24 * 8))), uint64_t(desc_b_2 + ((ki_2 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_24 * 16)), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local + 0), 32);
    }
    overlap_plan_mbar[19].arrive();
    overlap_plan_mbar[4].wait((((int)blockIdx.y) >> 1));
    #pragma unroll
    for (int i_25 = 0; i_25 < 32; ++i_25) {
      *(uint1*)(cb_local + (i_25 * 2)) = *(uint1*)(((half_t*)cb_shared) + (((((((((((i_25 & 7) >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_25 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((i_25 >> 3) * 16) + (((i_25 & 3) >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_25 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_25 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8192));
    }
    overlap_plan_mbar[14].arrive();
    #pragma unroll
    for (int i_26 = 0; i_26 < 32; ++i_26) {
      float broadcast_var_4 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
      uint1 __36;
      float2 __37;
        float2 __38;
        uint1 v__26 = *(uint1*)(cb_local + (i_26 * 2));
        ((float2*)(&__38))[0] = __half22float2(((half2*)(&v__26))[0]);
        float2 __39;
        float2 __40;
          float2 __41;
            float2 v__27 = make_float2(da_m_local[((((i_26 & 7) >> 2) * 2) + (i_26 & 1))], da_m_local[((((i_26 & 7) >> 2) * 2) + (i_26 & 1))]);
            float2 v__28 = *(float2*)(da_k_local_v1 + (((i_26 >> 3) * 4) + (((i_26 & 3) >> 1) * 2)));
            __41.x = (v__27.x-v__28.x);
            __41.y = (v__27.y-v__28.y);
          float2 v__29 = make_float2(broadcast_var_4, broadcast_var_4);
          __40.x = (__41.x*v__29.x);
          __40.y = (__41.y*v__29.y);
        __39.x = exp2f(__40.x);
        __39.y = exp2f(__40.y);
        __37.x = (__38.x*__39.x);
        __37.y = (__38.y*__39.y);
      ((half2*)(&__36))[0] = __float22half2_rn(((float2*)(&__37))[0]);
      *(uint1*)(cb_local + (i_26 * 2)) = __36;
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 32; ++i_27) {
      uint1 __42;
      float2 __43;
        float2 __44;
        uint1 v__30 = *(uint1*)(cb_local + (i_27 * 2));
        ((float2*)(&__44))[0] = __half22float2(((half2*)(&v__30))[0]);
        float2 v__31 = *(float2*)(dt_local_v1 + (((i_27 >> 3) * 4) + (((i_27 & 3) >> 1) * 2)));
        __43.x = (__44.x*v__31.x);
        __43.y = (__44.y*v__31.y);
      ((half2*)(&__42))[0] = __float22half2_rn(((float2*)(&__43))[0]);
      *(uint1*)(cb_local + (i_27 * 2)) = __42;
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 64; ++i_28) {
      half_t condval_2;
      if (((((((((((int)blockIdx.y) >> 1) * 128) + ((i_28 >> 4) * 16)) + (((i_28 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_28 & 1)) + 64) <= ((((((((int)blockIdx.y) >> 1) * 128) + (((i_28 & 15) >> 3) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_28 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_2 = cb_local[i_28];
      } else {
        condval_2 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      cb_local[i_28] = condval_2;
    }
    overlap_plan_mbar[10].wait((((int)blockIdx.y) >> 1));
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b_3, (&(((half_t*)x_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_3, 4096);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_29 = 0; i_29 < 2; ++i_29) {
        #pragma unroll
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(cb_local + ((ki_3 * 16) + (i_29 * 8))), uint64_t(desc_b_3 + ((ki_3 * 1024) >> 4)), reinterpret_cast<uint32_t*>(acc + (i_29 * 16)), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(cb_local + 0), 32);
    }
    overlap_plan_mbar[20].arrive();
    d_local[0] = ((float)D[((int)blockIdx.x)]);
    overlap_plan_mbar[11].wait(0);
    #pragma unroll
    for (int i_30 = 0; i_30 < 16; ++i_30) {
      half_t residual_shared_local_cast_8[2];
      *(uint1*)(residual_shared_local_cast_8 + 0) = *(uint1*)(((half_t*)residual_shared) + ((((((((i_30 >> 3) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((i_30 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_30 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_30 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __45;
      uint1 v__32 = *(uint1*)(residual_shared_local_cast_8 + 0);
      ((float2*)(&__45))[0] = __half22float2(((half2*)(&v__32))[0]);
      *(float2*)(residual_local + (i_30 * 2)) = __45;
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 32; ++i_31) {
      acc[i_31] = (acc[i_31] + (residual_local[i_31] * d_local[0]));
    }
    #pragma unroll
    for (int i_32 = 0; i_32 < 4; ++i_32) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[((((((i_32 >> 1) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_32 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((half_t)acc[(i_32 * 8)]), ((half_t)acc[((i_32 * 8) + 1)])), __pack_half2(((half_t)acc[((i_32 * 8) + 2)]), ((half_t)acc[((i_32 * 8) + 3)])), __pack_half2(((half_t)acc[((i_32 * 8) + 4)]), ((half_t)acc[((i_32 * 8) + 5)])), __pack_half2(((half_t)acc[((i_32 * 8) + 6)]), ((half_t)acc[((i_32 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[12].arrive();
  } else {
    tl::warpgroup_reg_dealloc<24>();
    ((half_t*)da_m_shared)[(((int)threadIdx.x) - 128)] = DA[(((((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + ((((int)blockIdx.y) >> 1) * 128)) + ((int)threadIdx.x)) - 128)];
    tl::fence_proxy_async();
    overlap_plan_mbar[0].arrive();
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(32768);
      tl::tma_load(C_desc, overlap_plan_mbar[1], (&(((half_t*)c_shared)[0])), 0, (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 128)), 0, (((int)blockIdx.z) & 7));
      tl::tma_load(C_desc, overlap_plan_mbar[1], (&(((half_t*)c_shared)[8192])), 64, (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 128)), 0, (((int)blockIdx.z) & 7));
      overlap_plan_mbar[2].arrive_and_expect_tx(8192);
      tl::tma_load(Prev_desc, overlap_plan_mbar[2], (&(((half_t*)prev_shared)[0])), 0, ((((int)blockIdx.y) & 1) * 32), ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
      tl::tma_load(Prev_desc, overlap_plan_mbar[2], (&(((half_t*)prev_shared)[2048])), 64, ((((int)blockIdx.y) & 1) * 32), ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
    }
    for (int ik_1 = 0; ik_1 < (((((int)blockIdx.y) >> 1) * 2) + 2); ++ik_1) {
      if (2 <= ik_1) {
        overlap_plan_mbar[((ik_1 & 1) + 13)].wait(0);
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
        overlap_plan_mbar[((ik_1 & 1) + 3)].arrive_and_expect_tx(16384);
        tl::tma_load(CB_desc, overlap_plan_mbar[((ik_1 & 1) + 3)], (&(((half_t*)cb_shared)[((ik_1 & 1) * 8192)])), (ik_1 * 64), ((((int)blockIdx.y) >> 1) * 128), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
      }
      if (2 <= ik_1) {
        overlap_plan_mbar[((ik_1 & 1) + 15)].wait(0);
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
        overlap_plan_mbar[((ik_1 & 1) + 5)].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)da_k_shared)[((ik_1 & 1) * 64)])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik_1 * 64))])), overlap_plan_mbar[((ik_1 & 1) + 5)], 128);
      }
      if (2 <= ik_1) {
        overlap_plan_mbar[((ik_1 & 1) + 17)].wait(0);
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
        overlap_plan_mbar[((ik_1 & 1) + 7)].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)dt_shared)[((ik_1 & 1) * 64)])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik_1 * 64))])), overlap_plan_mbar[((ik_1 & 1) + 7)], 128);
      }
      if (2 <= ik_1) {
        overlap_plan_mbar[((ik_1 & 1) + 19)].wait(0);
      }
      if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
        overlap_plan_mbar[((ik_1 & 1) + 9)].arrive_and_expect_tx(4096);
        tl::tma_load(X_desc, overlap_plan_mbar[((ik_1 & 1) + 9)], (&(((half_t*)x_shared)[((ik_1 & 1) * 2048)])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + (ik_1 * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
    }
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 4)) {
      overlap_plan_mbar[11].arrive_and_expect_tx(8192);
      tl::tma_load(X_desc_1, overlap_plan_mbar[11], (&(((half_t*)residual_shared)[0])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 128)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
    overlap_plan_mbar[12].wait(0);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store(Output_desc, (&(((half_t*)acc_shared)[0])), ((((int)blockIdx.y) & 1) * 32), (((((int)blockIdx.z) >> 3) * 256) + ((((int)blockIdx.y) >> 1) * 128)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

