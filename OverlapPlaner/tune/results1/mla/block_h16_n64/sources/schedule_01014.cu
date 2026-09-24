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
  void* acc_o_wsp_handoff_8 = ((void*)((char*)buf_dyn_shmem + 149504));
  void* K_pe_shared = ((void*)((char*)buf_dyn_shmem + 182272));
  void* acc_s_wsp_handoff_6 = ((void*)((char*)buf_dyn_shmem + 198656));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 206848));
  void* acc_s_wsp_handoff_4 = ((void*)((char*)buf_dyn_shmem + 210944));
  void* scores_scale_wsp_handoff_5 = ((void*)((char*)buf_dyn_shmem + 215040));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 215104));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 215104));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 215104));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 215104));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[19];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float logsum[2];
  float scores_max[2];
  float acc_s_v0[4];
  float scores_scale[2];
  float acc_s_v1[4];
  float acc_o[32];
  float scores_max_prev[2];
  float scores_max_clear[2];
  float scores_sum[2];
  float scores_max_clear_1[2];
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
    overlap_plan_mbar[5].init(1);
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
    overlap_plan_mbar[16].init(256);
    overlap_plan_mbar[17].init(256);
    overlap_plan_mbar[18].init(256);
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
    for (int k = 0; k < 64; ++k) {
      if (1 <= k) {
        overlap_plan_mbar[13].wait(((k + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(65536);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[0])), 0, (k * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[4096])), 64, (k * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[8192])), 128, (k * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[12288])), 192, (k * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[16384])), 256, (k * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[20480])), 320, (k * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[24576])), 384, (k * 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[28672])), 448, (k * 128), 0, 0);
      }
      if (1 <= k) {
        overlap_plan_mbar[15].wait(((k + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[4].arrive_and_expect_tx(8192);
        tl::tma_load(K_pe_desc, overlap_plan_mbar[4], (&(((half_t*)K_pe_shared)[0])), 0, (k * 128), 0, 0);
      }
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_2 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_2, broadcast_var_2);
      overlap_plan_mbar[6].wait(0);
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        *(float2*)(acc_s_v0 + (i * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_4) + ((((i * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        scores_max_clear[i_1] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 2; ++rv) {
          scores_max_clear[i_1] = max(scores_max_clear[i_1], acc_s_v0[((i_1 * 2) + rv)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear[i_1] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_1], (&(((float*)workspace_3)[0])));
        scores_max_clear[i_1] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_1]);
        scores_max[i_1] = max(scores_max[i_1], scores_max_clear[i_1]);
      }
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        scores_max[i_2] = max(scores_max[i_2], scores_max_prev[i_2]);
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        scores_scale[i_3] = exp2f(((scores_max_prev[i_3] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_3] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      if ((((((int)threadIdx.x) & 3) * 8) + (((int)threadIdx.x) >> 5)) == 0) {
        #pragma unroll
        for (int i_4 = 0; i_4 < 2; ++i_4) {
          ((float*)scores_scale_wsp_handoff_5)[((i_4 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = scores_scale[i_4];
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[7].arrive();
      #pragma unroll
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        acc_s_v0[i_5] = exp2f(((acc_s_v0[i_5] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_5 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        *(float2*)(((float*)acc_s_wsp_handoff_6) + ((((i_6 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s_v0 + (i_6 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[8].arrive();
      if (1 <= k) {
        overlap_plan_mbar[17].wait(((k + 1) & 1));
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        half_t S_shared_local_cast[2];
        uint1 __1;
        float2 v_ = *(float2*)(acc_s_v0 + (i_7 * 2));
        ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
        *(uint1*)(S_shared_local_cast + 0) = __1;
        *(uint1*)(((half_t*)S_shared) + ((((((i_7 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast + 0);
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[10].arrive();
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        scores_sum[i_8] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
          scores_sum[i_8] = (scores_sum[i_8] + acc_s_v0[((i_8 * 2) + rv_1)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_8] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_8], (&(((float*)workspace_1)[0])));
        scores_sum[i_8] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_8]);
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        logsum[i_9] = ((logsum[i_9] * scores_scale[i_9]) + scores_sum[i_9]);
      }
      if (1 <= k) {
        overlap_plan_mbar[14].wait(((k + 1) & 1));
      }
      tl::__sync_thread_partial(3, 256);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[3].arrive_and_expect_tx(65536);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[32768])), 0, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[36864])), 64, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[40960])), 128, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[45056])), 192, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[49152])), 256, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[53248])), 320, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[57344])), 384, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[61440])), 448, ((k * 128) + 64), 0, 0);
      }
      if (1 <= k) {
        overlap_plan_mbar[16].wait(((k + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[5].arrive_and_expect_tx(8192);
        tl::tma_load(K_pe_desc, overlap_plan_mbar[5], (&(((half_t*)K_pe_shared)[4096])), 0, ((k * 128) + 64), 0, 0);
      }
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_3 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
      overlap_plan_mbar[6].wait(1);
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        *(float2*)(acc_s_v1 + (i_10 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_4) + ((((i_10 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        scores_max_clear_1[i_11] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 2; ++rv_2) {
          scores_max_clear_1[i_11] = max(scores_max_clear_1[i_11], acc_s_v1[((i_11 * 2) + rv_2)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear_1[i_11] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_11], (&(((float*)workspace)[0])));
        scores_max_clear_1[i_11] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_11]);
        scores_max[i_11] = max(scores_max[i_11], scores_max_clear_1[i_11]);
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        scores_max[i_12] = max(scores_max[i_12], scores_max_prev[i_12]);
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        scores_scale[i_13] = exp2f(((scores_max_prev[i_13] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_13] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      if ((((((int)threadIdx.x) & 3) * 8) + (((int)threadIdx.x) >> 5)) == 0) {
        #pragma unroll
        for (int i_14 = 0; i_14 < 2; ++i_14) {
          ((float*)scores_scale_wsp_handoff_5)[((i_14 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = scores_scale[i_14];
        }
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[7].arrive();
      #pragma unroll
      for (int i_15 = 0; i_15 < 4; ++i_15) {
        acc_s_v1[i_15] = exp2f(((acc_s_v1[i_15] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_15 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 2; ++i_16) {
        *(float2*)(((float*)acc_s_wsp_handoff_6) + (((((i_16 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1024)) = *(float2*)(acc_s_v1 + (i_16 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[9].arrive();
      if (1 <= k) {
        overlap_plan_mbar[18].wait(((k + 1) & 1));
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        half_t S_shared_local_cast_1[2];
        uint1 __2;
        float2 v__1 = *(float2*)(acc_s_v1 + (i_17 * 2));
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
        *(uint1*)(S_shared_local_cast_1 + 0) = __2;
        *(uint1*)(((half_t*)S_shared) + (((((((i_17 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1024)) = *(uint1*)(S_shared_local_cast_1 + 0);
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[11].arrive();
      #pragma unroll
      for (int i_18 = 0; i_18 < 2; ++i_18) {
        scores_sum[i_18] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 2; ++rv_3) {
          scores_sum[i_18] = (scores_sum[i_18] + acc_s_v1[((i_18 * 2) + rv_3)]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_18] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_18], (&(((float*)workspace_2)[0])));
        scores_sum[i_18] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_18]);
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 2; ++i_19) {
        logsum[i_19] = ((logsum[i_19] * scores_scale[i_19]) + scores_sum[i_19]);
      }
    }
    overlap_plan_mbar[12].wait(0);
    #pragma unroll
    for (int i_20 = 0; i_20 < 16; ++i_20) {
      *(float2*)(acc_o + (i_20 * 2)) = *(float2*)(((float*)acc_o_wsp_handoff_8) + ((((((i_20 & 1) * 4096) + (((((int)threadIdx.x) & 31) >> 2) * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((i_20 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 32; ++i_21) {
      acc_o[i_21] = (acc_o[i_21] / logsum[((i_21 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 4; ++i_22) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((int)threadIdx.x) & 15) * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (i_22 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_22 * 8)]), ((half_t)acc_o[((i_22 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_22 * 8) + 2)]), ((half_t)acc_o[((i_22 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_22 * 8) + 4)]), ((half_t)acc_o[((i_22 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_22 * 8) + 6)]), ((half_t)acc_o[((i_22 * 8) + 7)])));
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
    for (int i_23 = 0; i_23 < 8; ++i_23) {
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i_23 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    for (int k_1 = 0; k_1 < 64; ++k_1) {
      if (k_1 == 0) {
        overlap_plan_mbar[0].wait(0);
      }
      overlap_plan_mbar[2].wait((k_1 & 1));
      if (1 <= k_1) {
        overlap_plan_mbar[9].wait(((k_1 + 1) & 1));
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 2; ++i_24) {
        *(float2*)(acc_s_v0 + (i_24 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_6) + (((((i_24 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 960));
      }
      {
        half_t A_local[8];
        half_t B_local[4];
        float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_s_v0 + 0) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
        for (int ki = 0; ki < 32; ++ki) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
          tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[(((((((((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki >> 2) * 4096)) + (((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        }
      }
      if (k_1 == 0) {
        overlap_plan_mbar[1].wait(0);
      }
      overlap_plan_mbar[4].wait((k_1 & 1));
      {
        half_t A_local_1[8];
        half_t B_local_1[4];
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_1[0])));
          tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) & 255) >> 5) * 512) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[0])));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
        }
      }
      tl::__sync_thread_partial(4, 256);
      #pragma unroll
      for (int i_25 = 0; i_25 < 2; ++i_25) {
        *(float2*)(((float*)acc_s_wsp_handoff_4) + (((((i_25 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64)) = *(float2*)(acc_s_v0 + (i_25 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[6].arrive();
      overlap_plan_mbar[15].arrive();
      overlap_plan_mbar[7].wait(0);
      #pragma unroll
      for (int i_26 = 0; i_26 < 2; ++i_26) {
        scores_scale[i_26] = ((float*)scores_scale_wsp_handoff_5)[((i_26 * 8) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      #pragma unroll
      for (int i_27 = 0; i_27 < 32; ++i_27) {
        acc_o[i_27] = (acc_o[i_27] * scores_scale[((i_27 & 3) >> 1)]);
      }
      overlap_plan_mbar[10].wait((k_1 & 1));
      {
        half_t A_local_2[8];
        half_t B_local_2[32];
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[0])));
          for (int i_28 = 0; i_28 < 4; ++i_28) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) & 255) >> 5) * 4096) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_28 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_28 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_2[(i_28 * 8)])));
          }
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j * 8) + 4)));
          }
        }
      }
      overlap_plan_mbar[13].arrive();
      overlap_plan_mbar[17].arrive();
      overlap_plan_mbar[3].wait((k_1 & 1));
      overlap_plan_mbar[8].wait((k_1 & 1));
      #pragma unroll
      for (int i_29 = 0; i_29 < 2; ++i_29) {
        *(float2*)(acc_s_v1 + (i_29 * 2)) = *(float2*)(((float*)acc_s_wsp_handoff_6) + (((((i_29 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
      }
      {
        half_t A_local_3[8];
        half_t B_local_3[4];
        float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_s_v1 + 0) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
        tl::__sync_thread_partial(4, 256);
        for (int ki_3 = 0; ki_3 < 32; ++ki_3) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_3 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_3[0])));
          tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[(((((((((((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 1) & 1) * 32768) + ((ki_3 >> 2) * 4096)) + (((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[0])));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + 0), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 0));
        }
      }
      overlap_plan_mbar[5].wait((k_1 & 1));
      {
        half_t A_local_4[8];
        half_t B_local_4[4];
        for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_4[0])));
          tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 1) & 1) * 4096) + (((((((int)threadIdx.x) & 255) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_4[0])));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + 0), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + 0));
        }
      }
      tl::__sync_thread_partial(4, 256);
      #pragma unroll
      for (int i_30 = 0; i_30 < 2; ++i_30) {
        *(float2*)(((float*)acc_s_wsp_handoff_4) + (((((i_30 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64)) = *(float2*)(acc_s_v1 + (i_30 * 2));
      }
      tl::fence_proxy_async();
      overlap_plan_mbar[6].arrive();
      overlap_plan_mbar[16].arrive();
      overlap_plan_mbar[7].wait(1);
      #pragma unroll
      for (int i_31 = 0; i_31 < 2; ++i_31) {
        scores_scale[i_31] = ((float*)scores_scale_wsp_handoff_5)[((i_31 * 8) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      #pragma unroll
      for (int i_32 = 0; i_32 < 32; ++i_32) {
        acc_o[i_32] = (acc_o[i_32] * scores_scale[((i_32 & 3) >> 1)]);
      }
      overlap_plan_mbar[11].wait((k_1 & 1));
      {
        half_t A_local_5[8];
        half_t B_local_5[32];
        tl::__sync_thread_partial(4, 256);
        for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
          tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 1024)])), (&(A_local_5[0])));
          for (int i_33 = 0; i_33 < 4; ++i_33) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((((int)threadIdx.x) & 255) >> 5) * 4096) + (ki_5 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_33 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_33 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_5[(i_33 * 8)])));
          }
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + (j_1 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + ((j_1 * 8) + 4)));
          }
        }
      }
      tl::__sync_thread_partial(4, 256);
      if ((k_1 * 2) == 126) {
        #pragma unroll
        for (int i_34 = 0; i_34 < 16; ++i_34) {
          *(float2*)(((float*)acc_o_wsp_handoff_8) + (((((((i_34 & 1) * 4096) + (((((int)threadIdx.x) & 31) >> 2) * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((i_34 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 512)) = *(float2*)(acc_o + (i_34 * 2));
        }
      }
      overlap_plan_mbar[14].arrive();
      overlap_plan_mbar[18].arrive();
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[12].arrive();
  }
}

