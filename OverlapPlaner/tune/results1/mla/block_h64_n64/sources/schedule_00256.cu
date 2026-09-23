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
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_pe_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* KV_shared = ((void*)((char*)buf_dyn_shmem + 73728));
  void* K_pe_shared = ((void*)((char*)buf_dyn_shmem + 204800));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 212992));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 221184));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[8];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[128];
  float logsum[2];
  float scores_max[2];
  float acc_s[16];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  float scores_max_clear[2];
  float scores_max_clear_1[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(KV_desc);
    tl::prefetch_tma_descriptor(K_pe_desc);
    tl::prefetch_tma_descriptor(Q_pe_desc);
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
      overlap_plan_mbar[4].arrive_and_expect_tx(8192);
      tl::tma_load(K_pe_desc, overlap_plan_mbar[4], (&(((half_t*)K_pe_shared)[0])), 0, 0, 0, 0);
    }
    overlap_plan_mbar[0].wait(0);
    overlap_plan_mbar[2].wait(0);
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)KV_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki = 0; ki < 32; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + (((((ki >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), ((0 < ki) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
    }
    overlap_plan_mbar[1].wait(0);
    overlap_plan_mbar[4].wait(0);
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_1;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)Q_pe_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((half_t*)K_pe_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_1 + ((ki_1 * 32) >> 4)), uint64_t(desc_b_1 + ((((((int)threadIdx.x) >> 7) * 4096) + (ki_1 * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
    }
    overlap_plan_mbar[7].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_3 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      scores_max_clear[i_1] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        scores_max_clear[i_1] = max(scores_max_clear[i_1], acc_s[((((rv & 3) * 4) + (i_1 * 2)) + (rv >> 2))]);
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
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      acc_s[i_3] = exp2f(((acc_s[i_3] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[((i_3 & 3) >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)S_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_4) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_s[(i_4 * 8)]), ((half_t)acc_s[((i_4 * 8) + 1)])), __pack_half2(((half_t)acc_s[((i_4 * 8) + 2)]), ((half_t)acc_s[((i_4 * 8) + 3)])), __pack_half2(((half_t)acc_s[((i_4 * 8) + 4)]), ((half_t)acc_s[((i_4 * 8) + 5)])), __pack_half2(((half_t)acc_s[((i_4 * 8) + 6)]), ((half_t)acc_s[((i_4 * 8) + 7)])));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      scores_scale[i_5] = exp2f(((scores_max_prev[i_5] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_5] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      scores_sum[i_6] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        scores_sum[i_6] = (scores_sum[i_6] + acc_s[((((rv_1 & 3) * 4) + (i_6 * 2)) + (rv_1 >> 2))]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_sum[i_6] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_6], (&(((float*)workspace_2)[0])));
      scores_sum[i_6] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      logsum[i_7] = ((logsum[i_7] * scores_scale[i_7]) + scores_sum[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 128; ++i_8) {
      acc_o[i_8] = (acc_o[i_8] * scores_scale[((i_8 & 3) >> 1)]);
    }
    for (int k = 0; k < 127; ++k) {
      if (1 <= k) {
        overlap_plan_mbar[(((k + 1) & 1) + 5)].wait((((k + 3) & 3) >> 1));
      }
      tl::__sync_thread_partial(3, 256);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[(((k + 1) & 1) + 2)].arrive_and_expect_tx(65536);
        tl::fence_proxy_async();
        tl::tma_load(KV_desc, overlap_plan_mbar[(((k + 1) & 1) + 2)], (&(((half_t*)KV_shared)[(((k + 1) & 1) * 32768)])), 0, ((k * 64) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[(((k + 1) & 1) + 2)], (&(((half_t*)KV_shared)[((((k + 1) & 1) * 32768) + 4096)])), 64, ((k * 64) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[(((k + 1) & 1) + 2)], (&(((half_t*)KV_shared)[((((k + 1) & 1) * 32768) + 8192)])), 128, ((k * 64) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[(((k + 1) & 1) + 2)], (&(((half_t*)KV_shared)[((((k + 1) & 1) * 32768) + 12288)])), 192, ((k * 64) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[(((k + 1) & 1) + 2)], (&(((half_t*)KV_shared)[((((k + 1) & 1) * 32768) + 16384)])), 256, ((k * 64) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[(((k + 1) & 1) + 2)], (&(((half_t*)KV_shared)[((((k + 1) & 1) * 32768) + 20480)])), 320, ((k * 64) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[(((k + 1) & 1) + 2)], (&(((half_t*)KV_shared)[((((k + 1) & 1) * 32768) + 24576)])), 384, ((k * 64) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[(((k + 1) & 1) + 2)], (&(((half_t*)KV_shared)[((((k + 1) & 1) * 32768) + 28672)])), 448, ((k * 64) + 64), 0, 0);
      }
      overlap_plan_mbar[7].wait((k & 1));
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[4].arrive_and_expect_tx(8192);
        tl::fence_proxy_async();
        tl::tma_load(K_pe_desc, overlap_plan_mbar[4], (&(((half_t*)K_pe_shared)[0])), 0, ((k * 64) + 64), 0, 0);
      }
      overlap_plan_mbar[(((k + 1) & 1) + 2)].wait((((k + 1) & 3) >> 1));
      {
        tl::GmmaDescriptor desc_a_2;
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)KV_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_2, (((k + 1) & 1) * 65536));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki_2 = 0; ki_2 < 32; ++ki_2) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + (((((ki_2 >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki_2 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), ((0 < ki_2) ? 1 : 0));
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
      }
      overlap_plan_mbar[4].wait(((k + 1) & 1));
      {
        tl::GmmaDescriptor desc_a_3;
        tl::GmmaDescriptor desc_b_3;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((half_t*)Q_pe_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_3, (&(((half_t*)K_pe_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_3 + ((ki_3 * 32) >> 4)), uint64_t(desc_b_3 + ((((((int)threadIdx.x) >> 7) * 4096) + (ki_3 * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
      }
      overlap_plan_mbar[7].arrive();
      overlap_plan_mbar[((k & 1) + 2)].wait(((k & 3) >> 1));
      {
        tl::GmmaDescriptor desc_a_4;
        tl::GmmaDescriptor desc_b_4;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_4, (&(((half_t*)S_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_4, (&(((half_t*)KV_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_4, ((k & 1) * 65536));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 256, 16, false, true, 1, 1>(uint64_t(desc_a_4 + ((ki_4 * 32) >> 4)), uint64_t(desc_b_4 + ((((((int)threadIdx.x) >> 7) * 32768) + (ki_4 * 2048)) >> 4)), ((uint32_t*)(acc_o + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      }
      overlap_plan_mbar[((k & 1) + 5)].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_4 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_4, broadcast_var_4);
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        scores_max_clear_1[i_9] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
          scores_max_clear_1[i_9] = max(scores_max_clear_1[i_9], acc_s[((((rv_2 & 3) * 4) + (i_9 * 2)) + (rv_2 >> 2))]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear_1[i_9] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_9], (&(((float*)workspace_3)[0])));
        scores_max_clear_1[i_9] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_9]);
        scores_max[i_9] = max(scores_max[i_9], scores_max_clear_1[i_9]);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        scores_max[i_10] = max(scores_max[i_10], scores_max_prev[i_10]);
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 16; ++i_11) {
        acc_s[i_11] = exp2f(((acc_s[i_11] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[((i_11 & 3) >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)S_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_12) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((half_t)acc_s[(i_12 * 8)]), ((half_t)acc_s[((i_12 * 8) + 1)])), __pack_half2(((half_t)acc_s[((i_12 * 8) + 2)]), ((half_t)acc_s[((i_12 * 8) + 3)])), __pack_half2(((half_t)acc_s[((i_12 * 8) + 4)]), ((half_t)acc_s[((i_12 * 8) + 5)])), __pack_half2(((half_t)acc_s[((i_12 * 8) + 6)]), ((half_t)acc_s[((i_12 * 8) + 7)])));
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        scores_scale[i_13] = exp2f(((scores_max_prev[i_13] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_13] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        scores_sum[i_14] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
          scores_sum[i_14] = (scores_sum[i_14] + acc_s[((((rv_3 & 3) * 4) + (i_14 * 2)) + (rv_3 >> 2))]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_14] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_14], (&(((float*)workspace_1)[0])));
        scores_sum[i_14] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_14]);
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        logsum[i_15] = ((logsum[i_15] * scores_scale[i_15]) + scores_sum[i_15]);
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 128; ++i_16) {
        acc_o[i_16] = (acc_o[i_16] * scores_scale[((i_16 & 3) >> 1)]);
      }
    }
    overlap_plan_mbar[3].wait(1);
    {
      tl::GmmaDescriptor desc_a_5;
      tl::GmmaDescriptor desc_b_5;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_5, (&(((half_t*)S_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_5, (&(((half_t*)KV_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_5, 65536);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 256, 16, false, true, 1, 1>(uint64_t(desc_a_5 + ((ki_5 * 32) >> 4)), uint64_t(desc_b_5 + ((((((int)threadIdx.x) >> 7) * 32768) + (ki_5 * 2048)) >> 4)), ((uint32_t*)(acc_o + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
    }
    overlap_plan_mbar[6].arrive();
    #pragma unroll
    for (int i_17 = 0; i_17 < 128; ++i_17) {
      acc_o[i_17] = (acc_o[i_17] / logsum[((i_17 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 16; ++i_18) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((((int)threadIdx.x) & 127) >> 5) * 8192) + ((((int)threadIdx.x) & 15) * 512)) + ((((int)threadIdx.x) >> 7) * 256)) + (i_18 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_18 * 8)]), ((half_t)acc_o[((i_18 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_18 * 8) + 2)]), ((half_t)acc_o[((i_18 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_18 * 8) + 4)]), ((half_t)acc_o[((i_18 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_18 * 8) + 6)]), ((half_t)acc_o[((i_18 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 256);
    if (tl::tl_shuffle_elect<256>()) {
      tl::fence_proxy_async();
      tl::tma_store((&(Output[(((int)blockIdx.x) * 32768)])), (&(((half_t*)O_shared)[0])), 65536);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  } else {
    tl::warpgroup_reg_dealloc<24>();
    if (tl::tl_shuffle_elect<128>() && ((((int)threadIdx.x) >> 5) == 8)) {
      overlap_plan_mbar[1].arrive_and_expect_tx(8192);
      tl::tma_load(Q_pe_desc, overlap_plan_mbar[1], (&(((half_t*)Q_pe_shared)[0])), 0, (((int)blockIdx.x) * 64), 0);
    }
  }
}

