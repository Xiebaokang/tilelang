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
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap KV_desc, __grid_constant__ const CUtensorMap K_pe_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap Q_pe_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_pe_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* KV_shared = ((void*)((char*)buf_dyn_shmem + 73728));
  void* K_pe_shared = ((void*)((char*)buf_dyn_shmem + 139264));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 147456));
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 155648));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_4 = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_5 = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_6 = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_7 = ((void*)((char*)buf_dyn_shmem + 221184));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[7];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[128];
  float logsum[2];
  float scores_max[2];
  float acc_s_v0[16];
  float scores_max_prev[2];
  float scores_scale_v0[2];
  float acc_s_v1[16];
  float scores_scale_v1[2];
  float scores_sum[2];
  float scores_max_clear[2];
  float scores_max_clear_1[2];
  float scores_max_clear_2[2];
  float scores_max_clear_3[2];
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
    overlap_plan_mbar[4].init(256);
    overlap_plan_mbar[5].init(256);
    overlap_plan_mbar[6].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    tl::warpgroup_reg_alloc<240>();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[0].arrive_and_expect_tx(65536);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[0])), 0, (((int)blockIdx.x) * 64), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[4096])), 64, (((int)blockIdx.x) * 64), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[8192])), 128, (((int)blockIdx.x) * 64), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[12288])), 192, (((int)blockIdx.x) * 64), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[16384])), 256, (((int)blockIdx.x) * 64), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[20480])), 320, (((int)blockIdx.x) * 64), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[24576])), 384, (((int)blockIdx.x) * 64), 0);
      tl::tma_load(Q_desc, overlap_plan_mbar[0], (&(((half_t*)Q_shared)[28672])), 448, (((int)blockIdx.x) * 64), 0);
      overlap_plan_mbar[1].arrive_and_expect_tx(8192);
      tl::tma_load(Q_pe_desc, overlap_plan_mbar[1], (&(((half_t*)Q_pe_shared)[0])), 0, (((int)blockIdx.x) * 64), 0);
    }
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(logsum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
    float broadcast_var_2 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_2, broadcast_var_2);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(65536);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[0])), 0, 0, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[4096])), 64, 0, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[8192])), 128, 0, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[12288])), 192, 0, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[16384])), 256, 0, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[20480])), 320, 0, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[24576])), 384, 0, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[28672])), 448, 0, 0, 0);
      overlap_plan_mbar[3].arrive_and_expect_tx(8192);
      tl::tma_load(K_pe_desc, overlap_plan_mbar[3], (&(((half_t*)K_pe_shared)[0])), 0, 0, 0, 0);
    }
    overlap_plan_mbar[0].wait(0);
    overlap_plan_mbar[2].wait(0);
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)KV_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 16);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki = 0; ki < 32; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + (((((ki >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_v0 + 0)), ((0 < ki) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 16);
    }
    overlap_plan_mbar[1].wait(0);
    overlap_plan_mbar[3].wait(0);
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_1;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)Q_pe_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((half_t*)K_pe_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 16);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_1 + ((ki_1 * 32) >> 4)), uint64_t(desc_b_1 + ((((((int)threadIdx.x) >> 7) * 4096) + (ki_1 * 32)) >> 4)), ((uint32_t*)(acc_s_v0 + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 16);
    }
    overlap_plan_mbar[6].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_3 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      scores_max_clear[i_1] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        scores_max_clear[i_1] = max(scores_max_clear[i_1], acc_s_v0[((((rv & 3) * 4) + (i_1 * 2)) + (rv >> 2))]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_max_clear[i_1] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_1], (&(((float*)workspace)[0])));
      scores_max_clear[i_1] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_1]);
      scores_max[i_1] = max(scores_max[i_1], scores_max_clear[i_1]);
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      scores_max[i_2] = max(scores_max[i_2], scores_max_prev[i_2]);
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      scores_scale_v0[i_3] = exp2f(((scores_max_prev[i_3] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_3] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      acc_s_v0[i_4] = exp2f(((acc_s_v0[i_4] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[((i_4 & 3) >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)S_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_5) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_s_v0[(i_5 * 8)]), ((half_t)acc_s_v0[((i_5 * 8) + 1)])), __pack_half2(((half_t)acc_s_v0[((i_5 * 8) + 2)]), ((half_t)acc_s_v0[((i_5 * 8) + 3)])), __pack_half2(((half_t)acc_s_v0[((i_5 * 8) + 4)]), ((half_t)acc_s_v0[((i_5 * 8) + 5)])), __pack_half2(((half_t)acc_s_v0[((i_5 * 8) + 6)]), ((half_t)acc_s_v0[((i_5 * 8) + 7)])));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 128; ++i_6) {
      acc_o[i_6] = (acc_o[i_6] * scores_scale_v0[((i_6 & 3) >> 1)]);
    }
    {
      tl::GmmaDescriptor desc_a_2;
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)S_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_2, (&(((half_t*)KV_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 256, 16, false, true, 1, 1>(uint64_t(desc_a_2 + ((ki_2 * 32) >> 4)), uint64_t(desc_b_2 + ((((((int)threadIdx.x) >> 7) * 32768) + (ki_2 * 2048)) >> 4)), ((uint32_t*)(acc_o + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
    }
    overlap_plan_mbar[5].arrive();
    for (int k = 0; k < 63; ++k) {
      overlap_plan_mbar[5].wait(0);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(65536);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[0])), 0, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[4096])), 64, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[8192])), 128, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[12288])), 192, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[16384])), 256, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[20480])), 320, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[24576])), 384, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[28672])), 448, ((k * 128) + 64), 0, 0);
      }
      overlap_plan_mbar[6].wait(0);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[3].arrive_and_expect_tx(8192);
        tl::tma_load(K_pe_desc, overlap_plan_mbar[3], (&(((half_t*)K_pe_shared)[0])), 0, ((k * 128) + 64), 0, 0);
      }
      overlap_plan_mbar[2].wait(1);
      {
        tl::GmmaDescriptor desc_a_3;
        tl::GmmaDescriptor desc_b_3;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_3, (&(((half_t*)KV_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_3 = 0; ki_3 < 32; ++ki_3) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_3 + ((((ki_3 >> 2) * 8192) + ((ki_3 & 3) * 32)) >> 4)), uint64_t(desc_b_3 + (((((ki_3 >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki_3 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_v1 + 0)), ((0 < ki_3) ? 1 : 0));
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 16);
      }
      overlap_plan_mbar[3].wait(1);
      {
        tl::GmmaDescriptor desc_a_4;
        tl::GmmaDescriptor desc_b_4;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_4, (&(((half_t*)Q_pe_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_4, (&(((half_t*)K_pe_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_4 + ((ki_4 * 32) >> 4)), uint64_t(desc_b_4 + ((((((int)threadIdx.x) >> 7) * 4096) + (ki_4 * 32)) >> 4)), ((uint32_t*)(acc_s_v1 + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 16);
      }
      overlap_plan_mbar[6].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_4 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_4, broadcast_var_4);
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        scores_max_clear_1[i_7] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
          scores_max_clear_1[i_7] = max(scores_max_clear_1[i_7], acc_s_v1[((((rv_1 & 3) * 4) + (i_7 * 2)) + (rv_1 >> 2))]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear_1[i_7] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_7], (&(((float*)workspace_2)[0])));
        scores_max_clear_1[i_7] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_7]);
        scores_max[i_7] = max(scores_max[i_7], scores_max_clear_1[i_7]);
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        scores_max[i_8] = max(scores_max[i_8], scores_max_prev[i_8]);
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        scores_scale_v1[i_9] = exp2f(((scores_max_prev[i_9] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_9] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 16; ++i_10) {
        acc_s_v1[i_10] = exp2f(((acc_s_v1[i_10] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[((i_10 & 3) >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        scores_sum[i_11] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
          scores_sum[i_11] = (scores_sum[i_11] + acc_s_v0[((((rv_2 & 3) * 4) + (i_11 * 2)) + (rv_2 >> 2))]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_11] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_11], (&(((float*)workspace_5)[0])));
        scores_sum[i_11] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_11]);
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)S_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_12) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_s_v1[(i_12 * 8)]), ((half_t)acc_s_v1[((i_12 * 8) + 1)])), __pack_half2(((half_t)acc_s_v1[((i_12 * 8) + 2)]), ((half_t)acc_s_v1[((i_12 * 8) + 3)])), __pack_half2(((half_t)acc_s_v1[((i_12 * 8) + 4)]), ((half_t)acc_s_v1[((i_12 * 8) + 5)])), __pack_half2(((half_t)acc_s_v1[((i_12 * 8) + 6)]), ((half_t)acc_s_v1[((i_12 * 8) + 7)])));
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        logsum[i_13] = ((logsum[i_13] * scores_scale_v0[i_13]) + scores_sum[i_13]);
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 128; ++i_14) {
        acc_o[i_14] = (acc_o[i_14] * scores_scale_v1[((i_14 & 3) >> 1)]);
      }
      {
        tl::GmmaDescriptor desc_a_5;
        tl::GmmaDescriptor desc_b_5;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_5, (&(((half_t*)S_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_5, (&(((half_t*)KV_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        tl::__sync_thread_partial(3, 256);
        #pragma unroll
        for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 256, 16, false, true, 1, 1>(uint64_t(desc_a_5 + ((ki_5 * 32) >> 4)), uint64_t(desc_b_5 + ((((((int)threadIdx.x) >> 7) * 32768) + (ki_5 * 2048)) >> 4)), ((uint32_t*)(acc_o + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      }
      overlap_plan_mbar[5].arrive();
      overlap_plan_mbar[5].wait(1);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[2].arrive_and_expect_tx(65536);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[0])), 0, ((k * 128) + 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[4096])), 64, ((k * 128) + 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[8192])), 128, ((k * 128) + 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[12288])), 192, ((k * 128) + 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[16384])), 256, ((k * 128) + 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[20480])), 320, ((k * 128) + 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[24576])), 384, ((k * 128) + 128), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[28672])), 448, ((k * 128) + 128), 0, 0);
      }
      overlap_plan_mbar[6].wait(1);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[3].arrive_and_expect_tx(8192);
        tl::tma_load(K_pe_desc, overlap_plan_mbar[3], (&(((half_t*)K_pe_shared)[0])), 0, ((k * 128) + 128), 0, 0);
      }
      overlap_plan_mbar[2].wait(0);
      {
        tl::GmmaDescriptor desc_a_6;
        tl::GmmaDescriptor desc_b_6;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_6, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_6, (&(((half_t*)KV_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_6 = 0; ki_6 < 32; ++ki_6) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_6 + ((((ki_6 >> 2) * 8192) + ((ki_6 & 3) * 32)) >> 4)), uint64_t(desc_b_6 + (((((ki_6 >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki_6 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_v0 + 0)), ((0 < ki_6) ? 1 : 0));
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 16);
      }
      overlap_plan_mbar[3].wait(0);
      {
        tl::GmmaDescriptor desc_a_7;
        tl::GmmaDescriptor desc_b_7;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_7, (&(((half_t*)Q_pe_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_7, (&(((half_t*)K_pe_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_7 + ((ki_7 * 32) >> 4)), uint64_t(desc_b_7 + ((((((int)threadIdx.x) >> 7) * 4096) + (ki_7 * 32)) >> 4)), ((uint32_t*)(acc_s_v0 + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v0 + 0), 16);
      }
      overlap_plan_mbar[6].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_5 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_5, broadcast_var_5);
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        scores_max_clear_2[i_15] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
          scores_max_clear_2[i_15] = max(scores_max_clear_2[i_15], acc_s_v0[((((rv_3 & 3) * 4) + (i_15 * 2)) + (rv_3 >> 2))]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear_2[i_15] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_15], (&(((float*)workspace_4)[0])));
        scores_max_clear_2[i_15] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_15]);
        scores_max[i_15] = max(scores_max[i_15], scores_max_clear_2[i_15]);
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 2; ++i_16) {
        scores_max[i_16] = max(scores_max[i_16], scores_max_prev[i_16]);
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        scores_scale_v0[i_17] = exp2f(((scores_max_prev[i_17] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_17] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 16; ++i_18) {
        acc_s_v0[i_18] = exp2f(((acc_s_v0[i_18] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[((i_18 & 3) >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 2; ++i_19) {
        scores_sum[i_19] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_4 = 0; rv_4 < 8; ++rv_4) {
          scores_sum[i_19] = (scores_sum[i_19] + acc_s_v1[((((rv_4 & 3) * 4) + (i_19 * 2)) + (rv_4 >> 2))]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_19] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_19], (&(((float*)workspace_7)[0])));
        scores_sum[i_19] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_19]);
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 2; ++i_20) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)S_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_20) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_s_v0[(i_20 * 8)]), ((half_t)acc_s_v0[((i_20 * 8) + 1)])), __pack_half2(((half_t)acc_s_v0[((i_20 * 8) + 2)]), ((half_t)acc_s_v0[((i_20 * 8) + 3)])), __pack_half2(((half_t)acc_s_v0[((i_20 * 8) + 4)]), ((half_t)acc_s_v0[((i_20 * 8) + 5)])), __pack_half2(((half_t)acc_s_v0[((i_20 * 8) + 6)]), ((half_t)acc_s_v0[((i_20 * 8) + 7)])));
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        logsum[i_21] = ((logsum[i_21] * scores_scale_v1[i_21]) + scores_sum[i_21]);
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 128; ++i_22) {
        acc_o[i_22] = (acc_o[i_22] * scores_scale_v0[((i_22 & 3) >> 1)]);
      }
      {
        tl::GmmaDescriptor desc_a_8;
        tl::GmmaDescriptor desc_b_8;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_8, (&(((half_t*)S_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_8, (&(((half_t*)KV_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        tl::__sync_thread_partial(3, 256);
        #pragma unroll
        for (int ki_8 = 0; ki_8 < 4; ++ki_8) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 256, 16, false, true, 1, 1>(uint64_t(desc_a_8 + ((ki_8 * 32) >> 4)), uint64_t(desc_b_8 + ((((((int)threadIdx.x) >> 7) * 32768) + (ki_8 * 2048)) >> 4)), ((uint32_t*)(acc_o + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      }
      overlap_plan_mbar[5].arrive();
    }
    overlap_plan_mbar[5].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(65536);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[0])), 0, 8128, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[4096])), 64, 8128, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[8192])), 128, 8128, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[12288])), 192, 8128, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[16384])), 256, 8128, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[20480])), 320, 8128, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[24576])), 384, 8128, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[28672])), 448, 8128, 0, 0);
    }
    overlap_plan_mbar[6].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(8192);
      tl::tma_load(K_pe_desc, overlap_plan_mbar[3], (&(((half_t*)K_pe_shared)[0])), 0, 8128, 0, 0);
    }
    overlap_plan_mbar[2].wait(1);
    {
      tl::GmmaDescriptor desc_a_9;
      tl::GmmaDescriptor desc_b_9;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_9, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_9, (&(((half_t*)KV_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 16);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_9 = 0; ki_9 < 32; ++ki_9) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_9 + ((((ki_9 >> 2) * 8192) + ((ki_9 & 3) * 32)) >> 4)), uint64_t(desc_b_9 + (((((ki_9 >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki_9 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_v1 + 0)), ((0 < ki_9) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 16);
    }
    overlap_plan_mbar[3].wait(1);
    {
      tl::GmmaDescriptor desc_a_10;
      tl::GmmaDescriptor desc_b_10;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_10, (&(((half_t*)Q_pe_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_10, (&(((half_t*)K_pe_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 16);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_10 = 0; ki_10 < 4; ++ki_10) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_10 + ((ki_10 * 32) >> 4)), uint64_t(desc_b_10 + ((((((int)threadIdx.x) >> 7) * 4096) + (ki_10 * 32)) >> 4)), ((uint32_t*)(acc_s_v1 + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_v1 + 0), 16);
    }
    overlap_plan_mbar[6].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_6 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_6, broadcast_var_6);
    #pragma unroll
    for (int i_23 = 0; i_23 < 2; ++i_23) {
      scores_max_clear_3[i_23] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_5 = 0; rv_5 < 8; ++rv_5) {
        scores_max_clear_3[i_23] = max(scores_max_clear_3[i_23], acc_s_v1[((((rv_5 & 3) * 4) + (i_23 * 2)) + (rv_5 >> 2))]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_max_clear_3[i_23] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_23], (&(((float*)workspace_1)[0])));
      scores_max_clear_3[i_23] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_23]);
      scores_max[i_23] = max(scores_max[i_23], scores_max_clear_3[i_23]);
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 2; ++i_24) {
      scores_max[i_24] = max(scores_max[i_24], scores_max_prev[i_24]);
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 2; ++i_25) {
      scores_scale_v1[i_25] = exp2f(((scores_max_prev[i_25] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_25] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 16; ++i_26) {
      acc_s_v1[i_26] = exp2f(((acc_s_v1[i_26] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[((i_26 & 3) >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 2; ++i_27) {
      scores_sum[i_27] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_6 = 0; rv_6 < 8; ++rv_6) {
        scores_sum[i_27] = (scores_sum[i_27] + acc_s_v0[((((rv_6 & 3) * 4) + (i_27 * 2)) + (rv_6 >> 2))]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_sum[i_27] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_27], (&(((float*)workspace_6)[0])));
      scores_sum[i_27] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_27]);
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 2; ++i_28) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)S_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_28) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_s_v1[(i_28 * 8)]), ((half_t)acc_s_v1[((i_28 * 8) + 1)])), __pack_half2(((half_t)acc_s_v1[((i_28 * 8) + 2)]), ((half_t)acc_s_v1[((i_28 * 8) + 3)])), __pack_half2(((half_t)acc_s_v1[((i_28 * 8) + 4)]), ((half_t)acc_s_v1[((i_28 * 8) + 5)])), __pack_half2(((half_t)acc_s_v1[((i_28 * 8) + 6)]), ((half_t)acc_s_v1[((i_28 * 8) + 7)])));
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 2; ++i_29) {
      logsum[i_29] = ((logsum[i_29] * scores_scale_v0[i_29]) + scores_sum[i_29]);
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 128; ++i_30) {
      acc_o[i_30] = (acc_o[i_30] * scores_scale_v1[((i_30 & 3) >> 1)]);
    }
    {
      tl::GmmaDescriptor desc_a_11;
      tl::GmmaDescriptor desc_b_11;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_11, (&(((half_t*)S_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_11, (&(((half_t*)KV_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int ki_11 = 0; ki_11 < 4; ++ki_11) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 256, 16, false, true, 1, 1>(uint64_t(desc_a_11 + ((ki_11 * 32) >> 4)), uint64_t(desc_b_11 + ((((((int)threadIdx.x) >> 7) * 32768) + (ki_11 * 2048)) >> 4)), ((uint32_t*)(acc_o + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
    }
    overlap_plan_mbar[5].arrive();
    #pragma unroll
    for (int i_31 = 0; i_31 < 2; ++i_31) {
      scores_sum[i_31] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_7 = 0; rv_7 < 8; ++rv_7) {
        scores_sum[i_31] = (scores_sum[i_31] + acc_s_v1[((((rv_7 & 3) * 4) + (i_31 * 2)) + (rv_7 >> 2))]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_sum[i_31] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_31], (&(((float*)workspace_3)[0])));
      scores_sum[i_31] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_31]);
    }
    #pragma unroll
    for (int i_32 = 0; i_32 < 2; ++i_32) {
      logsum[i_32] = ((logsum[i_32] * scores_scale_v1[i_32]) + scores_sum[i_32]);
    }
    #pragma unroll
    for (int i_33 = 0; i_33 < 128; ++i_33) {
      acc_o[i_33] = (acc_o[i_33] / logsum[((i_33 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_34 = 0; i_34 < 16; ++i_34) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((((int)threadIdx.x) & 127) >> 5) * 8192) + ((((int)threadIdx.x) & 15) * 512)) + ((((int)threadIdx.x) >> 7) * 256)) + (i_34 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_34 * 8)]), ((half_t)acc_o[((i_34 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_34 * 8) + 2)]), ((half_t)acc_o[((i_34 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_34 * 8) + 4)]), ((half_t)acc_o[((i_34 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_34 * 8) + 6)]), ((half_t)acc_o[((i_34 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[4].arrive();
  } else {
    tl::warpgroup_reg_dealloc<24>();
    overlap_plan_mbar[4].wait(0);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store((&(Output[(((int)blockIdx.x) * 32768)])), (&(((half_t*)O_shared)[0])), 65536);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

