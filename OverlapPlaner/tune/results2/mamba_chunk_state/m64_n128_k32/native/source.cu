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
  void* acc_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* da_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 25600));
  __shared__ __align__(16) uint64_t mbarrier_mem[16];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float da_last[1];
  float acc[64];
  float da_local[8];
  float dt_local[8];
  float scale[8];
  half_t x_local[16];
  half_t xt_local[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(X_desc);
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(1);
    mbarrier[4].init(1);
    mbarrier[5].init(1);
    mbarrier[6].init(1);
    mbarrier[7].init(1);
    mbarrier[8].init(128);
    mbarrier[9].init(128);
    mbarrier[10].init(128);
    mbarrier[11].init(128);
    mbarrier[12].init(128);
    mbarrier[13].init(128);
    mbarrier[14].init(128);
    mbarrier[15].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    for (int ik = 0; ik < 8; ++ik) {
      mbarrier[((ik & 1) + 8)].wait((((ik & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(ik & 1)].arrive_and_expect_tx(4096);
        tl::tma_load(X_desc, mbarrier[(ik & 1)], (&(((half_t*)x_shared)[((ik & 1) * 2048)])), 0, (((((int)blockIdx.z) >> 3) * 256) + (ik * 32)), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
      mbarrier[((ik & 1) + 10)].wait((((ik & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[((ik & 1) + 2)].arrive_and_expect_tx(64);
        tl::tma_load((&(((half_t*)da_shared)[((ik & 1) * 32)])), (&(DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 32))])), mbarrier[((ik & 1) + 2)], 64);
      }
      mbarrier[((ik & 1) + 12)].wait((((ik & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[((ik & 1) + 4)].arrive_and_expect_tx(64);
        tl::tma_load((&(((half_t*)dt_shared)[((ik & 1) * 32)])), (&(Dt[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + (ik * 32))])), mbarrier[((ik & 1) + 4)], 64);
      }
      mbarrier[((ik & 1) + 14)].wait((((ik & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[((ik & 1) + 6)].arrive_and_expect_tx(8192);
        tl::tma_load(B_desc, mbarrier[((ik & 1) + 6)], (&(((half_t*)b_shared)[((ik & 1) * 4096)])), 0, (((((int)blockIdx.z) >> 3) * 256) + (ik * 32)), 0, (((int)blockIdx.z) & 7));
        tl::tma_load(B_desc, mbarrier[((ik & 1) + 6)], (&(((half_t*)b_shared)[(((ik & 1) * 4096) + 2048)])), 64, (((((int)blockIdx.z) >> 3) * 256) + (ik * 32)), 0, (((int)blockIdx.z) & 7));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    da_last[0] = ((float)DA[(((((((int)blockIdx.z) & 7) * 327680) + (((int)blockIdx.x) * 4096)) + ((((int)blockIdx.z) >> 3) * 256)) + 255)]);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int ik_1 = 0; ik_1 < 8; ++ik_1) {
      mbarrier[((ik_1 & 1) + 2)].wait(((ik_1 & 3) >> 1));
      #pragma unroll
      for (int i_1 = 0; i_1 < 4; ++i_1) {
        half_t da_shared_local_cast[2];
        *(uint1*)(da_shared_local_cast + 0) = *(uint1*)(((half_t*)da_shared) + ((((ik_1 & 1) * 32) + (i_1 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __1;
        uint1 v_ = *(uint1*)(da_shared_local_cast + 0);
        ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
        *(float2*)(da_local + (i_1 * 2)) = __1;
      }
      mbarrier[((ik_1 & 1) + 10)].arrive();
      mbarrier[((ik_1 & 1) + 4)].wait(((ik_1 & 3) >> 1));
      #pragma unroll
      for (int i_2 = 0; i_2 < 4; ++i_2) {
        half_t dt_shared_local_cast_1[2];
        *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((((ik_1 & 1) * 32) + (i_2 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __2;
        uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
        ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
        *(float2*)(dt_local + (i_2 * 2)) = __2;
      }
      mbarrier[((ik_1 & 1) + 12)].arrive();
      #pragma unroll
      for (int i_3 = 0; i_3 < 8; ++i_3) {
        scale[i_3] = (exp2f(((da_last[0] - da_local[i_3]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)) * dt_local[i_3]);
      }
      mbarrier[(ik_1 & 1)].wait(((ik_1 & 3) >> 1));
      #pragma unroll
      for (int i_4 = 0; i_4 < 16; ++i_4) {
        x_local[i_4] = ((half_t*)x_shared)[(((((((((ik_1 & 1) * 2048) + ((i_4 >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (((i_4 & 3) >> 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_4 & 3) >> 1) + (i_4 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      mbarrier[((ik_1 & 1) + 8)].arrive();
      #pragma unroll
      for (int i_5 = 0; i_5 < 16; ++i_5) {
        xt_local[i_5] = ((half_t)(((float)x_local[((((i_5 >> 2) * 4) + ((i_5 & 1) * 2)) + ((i_5 & 3) >> 1))]) * scale[(((i_5 >> 2) * 2) + (i_5 & 1))]));
      }
      mbarrier[((ik_1 & 1) + 6)].wait(((ik_1 & 3) >> 1));
      {
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 256, 64>(desc_b, (&(((half_t*)b_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b, ((ik_1 & 1) * 8192));
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 8);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
          tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(xt_local + (ki * 8)), uint64_t(desc_b + ((ki * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc + 0), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
        tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(xt_local + 0), 8);
      }
      mbarrier[((ik_1 & 1) + 14)].arrive();
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_shared)[((((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_6 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 8192)])), __pack_half2(((half_t)acc[(i_6 * 8)]), ((half_t)acc[((i_6 * 8) + 1)])), __pack_half2(((half_t)acc[((i_6 * 8) + 2)]), ((half_t)acc[((i_6 * 8) + 3)])), __pack_half2(((half_t)acc[((i_6 * 8) + 4)]), ((half_t)acc[((i_6 * 8) + 5)])), __pack_half2(((half_t)acc[((i_6 * 8) + 6)]), ((half_t)acc[((i_6 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::fence_proxy_async();
      tl::tma_store((&(Output[((((((int)blockIdx.z) & 7) * 10485760) + ((((int)blockIdx.z) >> 3) * 655360)) + (((int)blockIdx.x) * 8192))])), (&(((half_t*)acc_shared)[0])), 16384);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

