#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap KV_desc, __grid_constant__ const CUtensorMap K_pe_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap Q_pe_desc);
extern "C" __global__ void __launch_bounds__(512, 1) main_kernel(__grid_constant__ const CUtensorMap KV_desc, __grid_constant__ const CUtensorMap K_pe_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap Q_pe_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_pe_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* KV_shared = ((void*)((char*)buf_dyn_shmem + 18432));
  void* acc_o_wsp_handoff_7 = ((void*)((char*)buf_dyn_shmem + 149504));
  void* K_pe_shared = ((void*)((char*)buf_dyn_shmem + 182272));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 190464));
  void* scores_scale_wsp_handoff_5 = ((void*)((char*)buf_dyn_shmem + 194560));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 194624));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 194624));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[16];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float logsum[2];
  float scores_max[2];
  float scores_scale[2];
  float acc_o[32];
  float acc_s[4];
  float scores_max_prev[2];
  float scores_max_clear[2];
  float scores_sum[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(Q_pe_desc);
    tl::prefetch_tma_descriptor(KV_desc);
    tl::prefetch_tma_descriptor(K_pe_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    overlap_plan_mbar[0].init(1);
    overlap_plan_mbar[1].init(1);
    overlap_plan_mbar[2].init(1);
    overlap_plan_mbar[3].init(1);
    overlap_plan_mbar[4].init(1);
    overlap_plan_mbar[5].init(256);
    overlap_plan_mbar[6].init(256);
    overlap_plan_mbar[7].init(256);
    overlap_plan_mbar[8].init(256);
    overlap_plan_mbar[9].init(256);
    overlap_plan_mbar[10].init(256);
    overlap_plan_mbar[11].init(256);
    overlap_plan_mbar[12].init(256);
    overlap_plan_mbar[13].init(256);
    overlap_plan_mbar[14].init(256);
    overlap_plan_mbar[15].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    tl::warpgroup_reg_dealloc<40>();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(16384);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[0])), 0, (((int)blockIdx.x) * 16), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[1024])), 64, (((int)blockIdx.x) * 16), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[2048])), 128, (((int)blockIdx.x) * 16), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[3072])), 192, (((int)blockIdx.x) * 16), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[4096])), 256, (((int)blockIdx.x) * 16), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[5120])), 320, (((int)blockIdx.x) * 16), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[6144])), 384, (((int)blockIdx.x) * 16), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[7168])), 448, (((int)blockIdx.x) * 16), 0);
      overlap_plan_mbar[1].arrive_and_expect_tx(2048);
      tl::tma_load(Q_pe_desc, overlap_plan_mbar[1], (&(((half_t*)Q_pe_shared)[0])), 0, (((int)blockIdx.x) * 16), 0);
    }
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(logsum + 0) = make_float2(broadcast_var, broadcast_var);
    float broadcast_var_1 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_1, broadcast_var_1);
    for (int k = 0; k < 128; ++k) {
      if (2 <= k) {
        overlap_plan_mbar[((k & 1) + 9)].wait((((k >> 1) + 1) & 1));
        overlap_plan_mbar[((k & 1) + 11)].wait((((k >> 1) + 1) & 1));
      }
      tl::__sync_thread_partial(3, 256);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[((k & 1) + 2)].arrive_and_expect_tx(65536);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k & 1) + 2)], (&(((half_t*)KV_shared)[((k & 1) * 32768)])), 0, (k * 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k & 1) + 2)], (&(((half_t*)KV_shared)[(((k & 1) * 32768) + 4096)])), 64, (k * 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k & 1) + 2)], (&(((half_t*)KV_shared)[(((k & 1) * 32768) + 8192)])), 128, (k * 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k & 1) + 2)], (&(((half_t*)KV_shared)[(((k & 1) * 32768) + 12288)])), 192, (k * 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k & 1) + 2)], (&(((half_t*)KV_shared)[(((k & 1) * 32768) + 16384)])), 256, (k * 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k & 1) + 2)], (&(((half_t*)KV_shared)[(((k & 1) * 32768) + 20480)])), 320, (k * 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k & 1) + 2)], (&(((half_t*)KV_shared)[(((k & 1) * 32768) + 24576)])), 384, (k * 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[((k & 1) + 2)], (&(((half_t*)KV_shared)[(((k & 1) * 32768) + 28672)])), 448, (k * 64), 0, 0);
      }
      if (1 <= k) {
        overlap_plan_mbar[13].wait(((k + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[4].arrive_and_expect_tx(8192);
        tl::tma_load(K_pe_desc, overlap_plan_mbar[4], (&(((half_t*)K_pe_shared)[0])), 0, (k * 64), 0, 0);
      }
      if (k == 0) {
        overlap_plan_mbar[0].wait(0);
      }
      overlap_plan_mbar[((k & 1) + 2)].wait(((k & 3) >> 1));
      {
        half_t A_local[8];
        half_t B_local[4];
        float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_s + 0) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
        for (int ki = 0; ki < 32; ++ki) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
          tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + (k & 1)) & 1) * 32768) + ((ki >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        }
      }
      overlap_plan_mbar[((k & 1) + 9)].arrive();
      if (k == 0) {
        overlap_plan_mbar[1].wait(0);
      }
      overlap_plan_mbar[4].wait((k & 1));
      {
        half_t A_local_1[8];
        half_t B_local_1[4];
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_1[0])));
          tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[0])));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
        }
      }
      overlap_plan_mbar[13].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_3 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        scores_max_clear[i] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 2; ++rv) {
          scores_max_clear[i] = max(scores_max_clear[i], acc_s[((i * 2) + rv)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear[i] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i], (&(((float*)workspace)[0])));
        scores_max_clear[i] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i]);
        scores_max[i] = max(scores_max[i], scores_max_clear[i]);
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        scores_max[i_1] = max(scores_max[i_1], scores_max_prev[i_1]);
      }
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        scores_scale[i_2] = exp2f(((scores_max_prev[i_2] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_2] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      if ((((((int)threadIdx.x) & 3) * 8) + (((int)threadIdx.x) >> 5)) == 0) {
        #pragma unroll
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          ((float*)scores_scale_wsp_handoff_5)[((i_3 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = scores_scale[i_3];
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[5].arrive();
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        acc_s[i_4] = exp2f(((acc_s[i_4] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_4 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      if (2 <= k) {
        overlap_plan_mbar[((k & 1) + 14)].wait((((k >> 1) + 1) & 1));
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        half_t S_shared_local_cast[2];
        uint1 __1;
        float2 v_ = *(float2*)(acc_s + (i_5 * 2));
        ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
        *(uint1*)(S_shared_local_cast + 0) = __1;
        *(uint1*)(((half_t*)S_shared) + ((((((((k & 1) * 1024) + (i_5 * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast + 0);
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[((k & 1) + 6)].arrive();
      #pragma unroll
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        scores_sum[i_6] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
          scores_sum[i_6] = (scores_sum[i_6] + acc_s[((i_6 * 2) + rv_1)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_6] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_6], (&(((float*)workspace_1)[0])));
        scores_sum[i_6] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_6]);
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        logsum[i_7] = ((logsum[i_7] * scores_scale[i_7]) + scores_sum[i_7]);
      }
    }
    overlap_plan_mbar[8].wait(0);
    #pragma unroll
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      *(float2*)(acc_o + (i_8 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_7) + ((((((i_8 & 1) * 4096) + (((((int)threadIdx.x) & 31) >> 2) * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((i_8 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 32; ++i_9) {
      acc_o[i_9] = (acc_o[i_9] / logsum[((i_9 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((int)threadIdx.x) & 15) * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (i_10 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_10 * 8)]), ((half_t)acc_o[((i_10 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_10 * 8) + 2)]), ((half_t)acc_o[((i_10 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_10 * 8) + 4)]), ((half_t)acc_o[((i_10 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_10 * 8) + 6)]), ((half_t)acc_o[((i_10 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 256);
    if (tl::tl_shuffle_elect<256>()) {
      tl::fence_proxy_async();
      tl::tma_store((&(Output[(((int)blockIdx.x) * 8192)])), (&(((half_t*)O_shared)[0])), 16384);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  } else {
    tl::warpgroup_reg_alloc<208>();
    #pragma unroll
    for (int i_11 = 0; i_11 < 8; ++i_11) {
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i_11 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    for (int k_1 = 0; k_1 < 128; ++k_1) {
      overlap_plan_mbar[5].wait((k_1 & 1));
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        scores_scale[i_12] = ((float*)scores_scale_wsp_handoff_5)[((i_12 * 8) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 32; ++i_13) {
        acc_o[i_13] = (acc_o[i_13] * scores_scale[((i_13 & 3) >> 1)]);
      }
      overlap_plan_mbar[((k_1 & 1) + 2)].wait(((k_1 & 3) >> 1));
      overlap_plan_mbar[((k_1 & 1) + 6)].wait(((k_1 & 3) >> 1));
      {
        half_t A_local_2[8];
        half_t B_local_2[32];
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[((((k_1 & 1) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[0])));
          for (int i_14 = 0; i_14 < 4; ++i_14) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((k_1 & 1) * 32768) + (((((int)threadIdx.x) & 255) >> 5) * 4096)) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_14 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_14 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_2[(i_14 * 8)])));
          }
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j * 8) + 4)));
          }
        }
      }
      tl::__sync_thread_partial(4, 256);
      if (k_1 == 127) {
        #pragma unroll
        for (int i_15 = 0; i_15 < 16; ++i_15) {
          *(float2*)(((float*)acc_o_wsp_handoff_7) + (((((((i_15 & 1) * 4096) + (((((int)threadIdx.x) & 31) >> 2) * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((i_15 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 512)) = *(float2*)(acc_o + (i_15 * 2));
        }
      }
      overlap_plan_mbar[((k_1 & 1) + 11)].arrive();
      overlap_plan_mbar[((k_1 & 1) + 14)].arrive();
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[8].arrive();
  }
}

