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
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 204800));
  void* K_pe_shared = ((void*)((char*)buf_dyn_shmem + 221184));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 229376));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 229376));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 229376));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 229376));
  void* workspace_4 = ((void*)((char*)buf_dyn_shmem + 229376));
  void* workspace_5 = ((void*)((char*)buf_dyn_shmem + 229376));
  void* workspace_6 = ((void*)((char*)buf_dyn_shmem + 229376));
  void* workspace_7 = ((void*)((char*)buf_dyn_shmem + 229376));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[9];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[128];
  float logsum[2];
  float scores_max[2];
  float acc_s[16];
  float scores_max_prev[2];
  float scores_scale_v0[2];
  float scores_sum[2];
  float scores_scale_v1[2];
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
    overlap_plan_mbar[4].init(1);
    overlap_plan_mbar[5].init(256);
    overlap_plan_mbar[6].init(256);
    overlap_plan_mbar[7].init(256);
    overlap_plan_mbar[8].init(256);
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
    overlap_plan_mbar[8].arrive();
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
      scores_max_clear[i_1] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_1], (&(((float*)workspace_4)[0])));
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
      acc_s[i_4] = exp2f(((acc_s[i_4] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[((i_4 & 3) >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      scores_sum[i_5] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        scores_sum[i_5] = (scores_sum[i_5] + acc_s[((((rv_1 & 3) * 4) + (i_5 * 2)) + (rv_1 >> 2))]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_sum[i_5] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_5], (&(((float*)workspace_6)[0])));
      scores_sum[i_5] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_5]);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      half_t S_shared_local_cast[2];
      uint1 __1;
      float2 v_ = *(float2*)(acc_s + (i_6 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(S_shared_local_cast + 0) = __1;
      *(uint1*)(((half_t*)S_shared) + (((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + ((i_6 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_6 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_6 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast + 0);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      logsum[i_7] = ((logsum[i_7] * scores_scale_v0[i_7]) + scores_sum[i_7]);
    }
    for (int k = 0; k < 63; ++k) {
      if (1 <= k) {
        overlap_plan_mbar[7].wait(((k + 1) & 1));
      }
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[3].arrive_and_expect_tx(65536);
        tl::fence_proxy_async();
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[32768])), 0, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[36864])), 64, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[40960])), 128, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[45056])), 192, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[49152])), 256, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[53248])), 320, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[57344])), 384, ((k * 128) + 64), 0, 0);
        tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[61440])), 448, ((k * 128) + 64), 0, 0);
      }
      overlap_plan_mbar[8].wait(0);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[4].arrive_and_expect_tx(8192);
        tl::fence_proxy_async();
        tl::tma_load(K_pe_desc, overlap_plan_mbar[4], (&(((half_t*)K_pe_shared)[0])), 0, ((k * 128) + 64), 0, 0);
      }
      overlap_plan_mbar[3].wait((k & 1));
      {
        tl::GmmaDescriptor desc_a_2;
        tl::GmmaDescriptor desc_b_2;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)KV_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_2, 65536);
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
      overlap_plan_mbar[4].wait(1);
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
      overlap_plan_mbar[8].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_4 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_4, broadcast_var_4);
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        scores_max_clear_1[i_8] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
          scores_max_clear_1[i_8] = max(scores_max_clear_1[i_8], acc_s[((((rv_2 & 3) * 4) + (i_8 * 2)) + (rv_2 >> 2))]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear_1[i_8] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_8], (&(((float*)workspace_3)[0])));
        scores_max_clear_1[i_8] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_8]);
        scores_max[i_8] = max(scores_max[i_8], scores_max_clear_1[i_8]);
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        scores_max[i_9] = max(scores_max[i_9], scores_max_prev[i_9]);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        scores_scale_v1[i_10] = exp2f(((scores_max_prev[i_10] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_10] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 16; ++i_11) {
        acc_s[i_11] = exp2f(((acc_s[i_11] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[((i_11 & 3) >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        scores_sum[i_12] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
          scores_sum[i_12] = (scores_sum[i_12] + acc_s[((((rv_3 & 3) * 4) + (i_12 * 2)) + (rv_3 >> 2))]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_12] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_12], (&(((float*)workspace_7)[0])));
        scores_sum[i_12] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_12]);
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 8; ++i_13) {
        half_t S_shared_local_cast_1[2];
        uint1 __2;
        float2 v__1 = *(float2*)(acc_s + (i_13 * 2));
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
        *(uint1*)(S_shared_local_cast_1 + 0) = __2;
        *(uint1*)(((half_t*)S_shared) + ((((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + ((i_13 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_13 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096)) = *(uint1*)(S_shared_local_cast_1 + 0);
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        logsum[i_14] = ((logsum[i_14] * scores_scale_v1[i_14]) + scores_sum[i_14]);
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 128; ++i_15) {
        acc_o[i_15] = (acc_o[i_15] * scores_scale_v0[((i_15 & 3) >> 1)]);
      }
      overlap_plan_mbar[2].wait((k & 1));
      {
        tl::GmmaDescriptor desc_a_4;
        tl::GmmaDescriptor desc_b_4;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_4, (&(((half_t*)S_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_4, (&(((half_t*)KV_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        tl::__sync_thread_partial(3, 256);
        #pragma unroll
        for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 256, 16, false, true, 1, 1>(uint64_t(desc_a_4 + ((ki_4 * 32) >> 4)), uint64_t(desc_b_4 + ((((((int)threadIdx.x) >> 7) * 32768) + (ki_4 * 2048)) >> 4)), ((uint32_t*)(acc_o + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      }
      overlap_plan_mbar[6].arrive();
      overlap_plan_mbar[6].wait((k & 1));
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
      overlap_plan_mbar[8].wait(1);
      if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
        overlap_plan_mbar[4].arrive_and_expect_tx(8192);
        tl::tma_load(K_pe_desc, overlap_plan_mbar[4], (&(((half_t*)K_pe_shared)[0])), 0, ((k * 128) + 128), 0, 0);
      }
      overlap_plan_mbar[2].wait(((k + 1) & 1));
      {
        tl::GmmaDescriptor desc_a_5;
        tl::GmmaDescriptor desc_b_5;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_5, (&(((half_t*)Q_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_5, (&(((half_t*)KV_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_5 = 0; ki_5 < 32; ++ki_5) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_5 + ((((ki_5 >> 2) * 8192) + ((ki_5 & 3) * 32)) >> 4)), uint64_t(desc_b_5 + (((((ki_5 >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki_5 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), ((0 < ki_5) ? 1 : 0));
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
      }
      overlap_plan_mbar[4].wait(0);
      {
        tl::GmmaDescriptor desc_a_6;
        tl::GmmaDescriptor desc_b_6;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_6, (&(((half_t*)Q_pe_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_6, (&(((half_t*)K_pe_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_6 + ((ki_6 * 32) >> 4)), uint64_t(desc_b_6 + ((((((int)threadIdx.x) >> 7) * 4096) + (ki_6 * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
      }
      overlap_plan_mbar[8].arrive();
      *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
      float broadcast_var_5 = -CUDART_INF_F;
      *(float2*)(scores_max + 0) = make_float2(broadcast_var_5, broadcast_var_5);
      #pragma unroll
      for (int i_16 = 0; i_16 < 2; ++i_16) {
        scores_max_clear_2[i_16] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_4 = 0; rv_4 < 8; ++rv_4) {
          scores_max_clear_2[i_16] = max(scores_max_clear_2[i_16], acc_s[((((rv_4 & 3) * 4) + (i_16 * 2)) + (rv_4 >> 2))]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_max_clear_2[i_16] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_16], (&(((float*)workspace)[0])));
        scores_max_clear_2[i_16] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_16]);
        scores_max[i_16] = max(scores_max[i_16], scores_max_clear_2[i_16]);
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        scores_max[i_17] = max(scores_max[i_17], scores_max_prev[i_17]);
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 2; ++i_18) {
        scores_scale_v0[i_18] = exp2f(((scores_max_prev[i_18] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_18] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 16; ++i_19) {
        acc_s[i_19] = exp2f(((acc_s[i_19] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[((i_19 & 3) >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 2; ++i_20) {
        scores_sum[i_20] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_5 = 0; rv_5 < 8; ++rv_5) {
          scores_sum[i_20] = (scores_sum[i_20] + acc_s[((((rv_5 & 3) * 4) + (i_20 * 2)) + (rv_5 >> 2))]);
        }
        tl::__sync_thread_partial(3, 256);
        scores_sum[i_20] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_20], (&(((float*)workspace_1)[0])));
        scores_sum[i_20] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_20]);
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 8; ++i_21) {
        half_t S_shared_local_cast_2[2];
        uint1 __3;
        float2 v__2 = *(float2*)(acc_s + (i_21 * 2));
        ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
        *(uint1*)(S_shared_local_cast_2 + 0) = __3;
        *(uint1*)(((half_t*)S_shared) + (((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + ((i_21 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_21 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_21 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast_2 + 0);
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        logsum[i_22] = ((logsum[i_22] * scores_scale_v0[i_22]) + scores_sum[i_22]);
      }
      #pragma unroll
      for (int i_23 = 0; i_23 < 128; ++i_23) {
        acc_o[i_23] = (acc_o[i_23] * scores_scale_v1[((i_23 & 3) >> 1)]);
      }
      overlap_plan_mbar[3].wait((k & 1));
      {
        tl::GmmaDescriptor desc_a_7;
        tl::GmmaDescriptor desc_b_7;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_7, (&(((half_t*)S_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_a_7, 8192);
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_7, (&(((half_t*)KV_shared)[0])));
        tl::increase_descriptor_offset<int>(desc_b_7, 65536);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
          tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 256, 16, false, true, 1, 1>(uint64_t(desc_a_7 + ((ki_7 * 32) >> 4)), uint64_t(desc_b_7 + ((((((int)threadIdx.x) >> 7) * 32768) + (ki_7 * 2048)) >> 4)), ((uint32_t*)(acc_o + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      }
      overlap_plan_mbar[7].arrive();
    }
    overlap_plan_mbar[7].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(65536);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[32768])), 0, 8128, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[36864])), 64, 8128, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[40960])), 128, 8128, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[45056])), 192, 8128, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[49152])), 256, 8128, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[53248])), 320, 8128, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[57344])), 384, 8128, 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[61440])), 448, 8128, 0, 0);
    }
    overlap_plan_mbar[8].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(8192);
      tl::tma_load(K_pe_desc, overlap_plan_mbar[4], (&(((half_t*)K_pe_shared)[0])), 0, 8128, 0, 0);
    }
    overlap_plan_mbar[3].wait(1);
    {
      tl::GmmaDescriptor desc_a_8;
      tl::GmmaDescriptor desc_b_8;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_8, (&(((half_t*)Q_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_8, (&(((half_t*)KV_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_8, 65536);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_8 = 0; ki_8 < 32; ++ki_8) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_8 + ((((ki_8 >> 2) * 8192) + ((ki_8 & 3) * 32)) >> 4)), uint64_t(desc_b_8 + (((((ki_8 >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki_8 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), ((0 < ki_8) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
    }
    overlap_plan_mbar[4].wait(1);
    {
      tl::GmmaDescriptor desc_a_9;
      tl::GmmaDescriptor desc_b_9;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_9, (&(((half_t*)Q_pe_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_9, (&(((half_t*)K_pe_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_9 + ((ki_9 * 32) >> 4)), uint64_t(desc_b_9 + ((((((int)threadIdx.x) >> 7) * 4096) + (ki_9 * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 16);
    }
    overlap_plan_mbar[8].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_6 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_6, broadcast_var_6);
    #pragma unroll
    for (int i_24 = 0; i_24 < 2; ++i_24) {
      scores_max_clear_3[i_24] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_6 = 0; rv_6 < 8; ++rv_6) {
        scores_max_clear_3[i_24] = max(scores_max_clear_3[i_24], acc_s[((((rv_6 & 3) * 4) + (i_24 * 2)) + (rv_6 >> 2))]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_max_clear_3[i_24] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_24], (&(((float*)workspace_5)[0])));
      scores_max_clear_3[i_24] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_24]);
      scores_max[i_24] = max(scores_max[i_24], scores_max_clear_3[i_24]);
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 2; ++i_25) {
      scores_max[i_25] = max(scores_max[i_25], scores_max_prev[i_25]);
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 2; ++i_26) {
      scores_scale_v1[i_26] = exp2f(((scores_max_prev[i_26] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_26] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 16; ++i_27) {
      acc_s[i_27] = exp2f(((acc_s[i_27] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[((i_27 & 3) >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 2; ++i_28) {
      scores_sum[i_28] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_7 = 0; rv_7 < 8; ++rv_7) {
        scores_sum[i_28] = (scores_sum[i_28] + acc_s[((((rv_7 & 3) * 4) + (i_28 * 2)) + (rv_7 >> 2))]);
      }
      tl::__sync_thread_partial(3, 256);
      scores_sum[i_28] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[i_28], (&(((float*)workspace_2)[0])));
      scores_sum[i_28] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_28]);
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 8; ++i_29) {
      half_t S_shared_local_cast_3[2];
      uint1 __4;
      float2 v__3 = *(float2*)(acc_s + (i_29 * 2));
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
      *(uint1*)(S_shared_local_cast_3 + 0) = __4;
      *(uint1*)(((half_t*)S_shared) + ((((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + ((i_29 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_29 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_29 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096)) = *(uint1*)(S_shared_local_cast_3 + 0);
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 2; ++i_30) {
      logsum[i_30] = ((logsum[i_30] * scores_scale_v1[i_30]) + scores_sum[i_30]);
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 128; ++i_31) {
      acc_o[i_31] = (acc_o[i_31] * scores_scale_v0[((i_31 & 3) >> 1)]);
    }
    overlap_plan_mbar[2].wait(1);
    {
      tl::GmmaDescriptor desc_a_10;
      tl::GmmaDescriptor desc_b_10;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_10, (&(((half_t*)S_shared)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_10, (&(((half_t*)KV_shared)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int ki_10 = 0; ki_10 < 4; ++ki_10) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 256, 16, false, true, 1, 1>(uint64_t(desc_a_10 + ((ki_10 * 32) >> 4)), uint64_t(desc_b_10 + ((((((int)threadIdx.x) >> 7) * 32768) + (ki_10 * 2048)) >> 4)), ((uint32_t*)(acc_o + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
    }
    overlap_plan_mbar[6].arrive();
    #pragma unroll
    for (int i_32 = 0; i_32 < 128; ++i_32) {
      acc_o[i_32] = (acc_o[i_32] * scores_scale_v1[((i_32 & 3) >> 1)]);
    }
    overlap_plan_mbar[3].wait(1);
    {
      tl::GmmaDescriptor desc_a_11;
      tl::GmmaDescriptor desc_b_11;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_11, (&(((half_t*)S_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_a_11, 8192);
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_11, (&(((half_t*)KV_shared)[0])));
      tl::increase_descriptor_offset<int>(desc_b_11, 65536);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_11 = 0; ki_11 < 4; ++ki_11) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 256, 16, false, true, 1, 1>(uint64_t(desc_a_11 + ((ki_11 * 32) >> 4)), uint64_t(desc_b_11 + ((((((int)threadIdx.x) >> 7) * 32768) + (ki_11 * 2048)) >> 4)), ((uint32_t*)(acc_o + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o + 0), 128);
    }
    overlap_plan_mbar[7].arrive();
    #pragma unroll
    for (int i_33 = 0; i_33 < 128; ++i_33) {
      acc_o[i_33] = (acc_o[i_33] / logsum[((i_33 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_34 = 0; i_34 < 16; ++i_34) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((((int)threadIdx.x) & 127) >> 5) * 8192) + ((((int)threadIdx.x) & 15) * 512)) + ((((int)threadIdx.x) >> 7) * 256)) + (i_34 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_34 * 8)]), ((half_t)acc_o[((i_34 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_34 * 8) + 2)]), ((half_t)acc_o[((i_34 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_34 * 8) + 4)]), ((half_t)acc_o[((i_34 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_34 * 8) + 6)]), ((half_t)acc_o[((i_34 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    overlap_plan_mbar[5].arrive();
  } else {
    tl::warpgroup_reg_dealloc<24>();
    overlap_plan_mbar[5].wait(0);
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store((&(Output[(((int)blockIdx.x) * 32768)])), (&(((half_t*)O_shared)[0])), 65536);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

