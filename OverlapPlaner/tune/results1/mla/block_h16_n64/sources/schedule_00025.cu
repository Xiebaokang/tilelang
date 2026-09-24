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
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap KV_desc, __grid_constant__ const CUtensorMap K_pe_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap Q_pe_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_pe_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* KV_shared = ((void*)((char*)buf_dyn_shmem + 18432));
  void* K_pe_shared = ((void*)((char*)buf_dyn_shmem + 215040));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 223232));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_4 = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_5 = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_6 = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_7 = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_8 = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_9 = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_10 = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_11 = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_12 = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_13 = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_14 = ((void*)((char*)buf_dyn_shmem + 227328));
  void* workspace_15 = ((void*)((char*)buf_dyn_shmem + 227328));
  __shared__ __align__(16) uint64_t overlap_plan_mbar_mem[10];
  auto overlap_plan_mbar = reinterpret_cast<Barrier*>(overlap_plan_mbar_mem);
  float acc_o[32];
  float logsum[2];
  float scores_max[2];
  float acc_s_v0[4];
  float scores_max_prev[2];
  float scores_max_clear[2];
  float scores_scale_v0[2];
  float scores_sum[2];
  float acc_s_v1[4];
  float scores_max_clear_1[2];
  float scores_scale_v1[2];
  float scores_max_clear_2[2];
  float scores_scale_v2[2];
  float scores_max_clear_3[2];
  float scores_max_clear_4[2];
  float scores_max_clear_5[2];
  float scores_max_clear_6[2];
  float scores_max_clear_7[2];
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
  }
  tl::fence_barrier_init();
  __syncthreads();
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
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
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
    overlap_plan_mbar[5].arrive_and_expect_tx(8192);
    tl::tma_load(K_pe_desc, overlap_plan_mbar[5], (&(((half_t*)K_pe_shared)[0])), 0, 0, 0, 0);
  }
  overlap_plan_mbar[0].wait(0);
  overlap_plan_mbar[2].wait(0);
  {
    half_t A_local[8];
    half_t B_local[4];
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s_v0 + 0) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    for (int ki = 0; ki < 32; ++ki) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
      tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
      tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
    }
  }
  overlap_plan_mbar[1].wait(0);
  overlap_plan_mbar[5].wait(0);
  {
    half_t A_local_1[8];
    half_t B_local_1[4];
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_1[0])));
      tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[0])));
      tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
    }
  }
  overlap_plan_mbar[9].arrive();
  *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
  float broadcast_var_4 = -CUDART_INF_F;
  *(float2*)(scores_max + 0) = make_float2(broadcast_var_4, broadcast_var_4);
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    scores_max_clear[i_1] = -CUDART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 2; ++rv) {
      scores_max_clear[i_1] = max(scores_max_clear[i_1], acc_s_v0[((i_1 * 2) + rv)]);
    }
    __syncthreads();
    scores_max_clear[i_1] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_1], (&(((float*)workspace_9)[0])));
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
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    acc_s_v0[i_4] = exp2f(((acc_s_v0[i_4] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_4 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 2; ++i_5) {
    scores_sum[i_5] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
      scores_sum[i_5] = (scores_sum[i_5] + acc_s_v0[((i_5 * 2) + rv_1)]);
    }
    __syncthreads();
    scores_sum[i_5] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_5], (&(((float*)workspace_2)[0])));
    scores_sum[i_5] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_5]);
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 2; ++i_6) {
    logsum[i_6] = ((logsum[i_6] * scores_scale_v0[i_6]) + scores_sum[i_6]);
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[3].arrive_and_expect_tx(65536);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[32768])), 0, 64, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[36864])), 64, 64, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[40960])), 128, 64, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[45056])), 192, 64, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[49152])), 256, 64, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[53248])), 320, 64, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[57344])), 384, 64, 0, 0);
    tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[61440])), 448, 64, 0, 0);
  }
  overlap_plan_mbar[9].wait(0);
  if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
    overlap_plan_mbar[5].arrive_and_expect_tx(8192);
    tl::tma_load(K_pe_desc, overlap_plan_mbar[5], (&(((half_t*)K_pe_shared)[0])), 0, 64, 0, 0);
  }
  overlap_plan_mbar[3].wait(0);
  {
    half_t A_local_2[8];
    half_t B_local_2[4];
    float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s_v1 + 0) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
    for (int ki_2 = 0; ki_2 < 32; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_2 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[0])));
      tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[(((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki_2 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 32768)])), (&(B_local_2[0])));
      tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + 0), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 0));
    }
  }
  overlap_plan_mbar[5].wait(1);
  {
    half_t A_local_3[8];
    half_t B_local_3[4];
    for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_3[0])));
      tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[0])));
      tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + 0), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 0));
    }
  }
  overlap_plan_mbar[9].arrive();
  *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
  float broadcast_var_6 = -CUDART_INF_F;
  *(float2*)(scores_max + 0) = make_float2(broadcast_var_6, broadcast_var_6);
  #pragma unroll
  for (int i_7 = 0; i_7 < 2; ++i_7) {
    scores_max_clear_1[i_7] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 2; ++rv_2) {
      scores_max_clear_1[i_7] = max(scores_max_clear_1[i_7], acc_s_v1[((i_7 * 2) + rv_2)]);
    }
    __syncthreads();
    scores_max_clear_1[i_7] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_1[i_7], (&(((float*)workspace_8)[0])));
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
  for (int i_10 = 0; i_10 < 4; ++i_10) {
    acc_s_v1[i_10] = exp2f(((acc_s_v1[i_10] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_10 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 2; ++i_11) {
    scores_sum[i_11] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 2; ++rv_3) {
      scores_sum[i_11] = (scores_sum[i_11] + acc_s_v1[((i_11 * 2) + rv_3)]);
    }
    __syncthreads();
    scores_sum[i_11] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_11], (&(((float*)workspace_3)[0])));
    scores_sum[i_11] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_11]);
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 2; ++i_12) {
    half_t S_shared_local_cast[2];
    uint1 __1;
    float2 v_ = *(float2*)(acc_s_v0 + (i_12 * 2));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    *(uint1*)(S_shared_local_cast + 0) = __1;
    *(uint1*)(((half_t*)S_shared) + ((((((i_12 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast + 0);
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 2; ++i_13) {
    logsum[i_13] = ((logsum[i_13] * scores_scale_v1[i_13]) + scores_sum[i_13]);
  }
  for (int k = 0; k < 21; ++k) {
    if (1 <= k) {
      overlap_plan_mbar[8].wait(1);
    }
    __syncthreads();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(65536);
      tl::fence_proxy_async();
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[65536])), 0, ((k * 384) + 128), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[69632])), 64, ((k * 384) + 128), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[73728])), 128, ((k * 384) + 128), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[77824])), 192, ((k * 384) + 128), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[81920])), 256, ((k * 384) + 128), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[86016])), 320, ((k * 384) + 128), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[90112])), 384, ((k * 384) + 128), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[94208])), 448, ((k * 384) + 128), 0, 0);
    }
    overlap_plan_mbar[9].wait(1);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[5].arrive_and_expect_tx(8192);
      tl::fence_proxy_async();
      tl::tma_load(K_pe_desc, overlap_plan_mbar[5], (&(((half_t*)K_pe_shared)[0])), 0, ((k * 384) + 128), 0, 0);
    }
    overlap_plan_mbar[4].wait(0);
    {
      half_t A_local_4[8];
      half_t B_local_4[4];
      float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s_v0 + 0) = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
      for (int ki_4 = 0; ki_4 < 32; ++ki_4) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_4 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_4[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 2) % 3) * 32768) + ((ki_4 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_4[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + 0), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + 0));
      }
    }
    overlap_plan_mbar[5].wait(0);
    {
      half_t A_local_5[8];
      half_t B_local_5[4];
      for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_5[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_5[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + 0), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + 0));
      }
    }
    overlap_plan_mbar[9].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_8 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_8, broadcast_var_8);
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      scores_max_clear_2[i_14] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_4 = 0; rv_4 < 2; ++rv_4) {
        scores_max_clear_2[i_14] = max(scores_max_clear_2[i_14], acc_s_v0[((i_14 * 2) + rv_4)]);
      }
      __syncthreads();
      scores_max_clear_2[i_14] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_14], (&(((float*)workspace_4)[0])));
      scores_max_clear_2[i_14] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_2[i_14]);
      scores_max[i_14] = max(scores_max[i_14], scores_max_clear_2[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      scores_max[i_15] = max(scores_max[i_15], scores_max_prev[i_15]);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      scores_scale_v2[i_16] = exp2f(((scores_max_prev[i_16] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_16] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 4; ++i_17) {
      acc_s_v0[i_17] = exp2f(((acc_s_v0[i_17] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_17 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 2; ++i_18) {
      scores_sum[i_18] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_5 = 0; rv_5 < 2; ++rv_5) {
        scores_sum[i_18] = (scores_sum[i_18] + acc_s_v0[((i_18 * 2) + rv_5)]);
      }
      __syncthreads();
      scores_sum[i_18] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_18], (&(((float*)workspace_5)[0])));
      scores_sum[i_18] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_18]);
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      half_t S_shared_local_cast_1[2];
      uint1 __2;
      float2 v__1 = *(float2*)(acc_s_v1 + (i_19 * 2));
      ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
      *(uint1*)(S_shared_local_cast_1 + 0) = __2;
      *(uint1*)(((half_t*)S_shared) + (((((((i_19 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1024)) = *(uint1*)(S_shared_local_cast_1 + 0);
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 2; ++i_20) {
      logsum[i_20] = ((logsum[i_20] * scores_scale_v2[i_20]) + scores_sum[i_20]);
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 32; ++i_21) {
      acc_o[i_21] = (acc_o[i_21] * scores_scale_v0[((i_21 & 3) >> 1)]);
    }
    overlap_plan_mbar[2].wait(0);
    {
      half_t A_local_6[8];
      half_t B_local_6[32];
      for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
        tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_6 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_6[0])));
        for (int i_22 = 0; i_22 < 4; ++i_22) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki_6 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_22 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_22 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_6[(i_22 * 8)])));
        }
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j * 8)), reinterpret_cast<const unsigned*>(A_local_6 + 0), reinterpret_cast<const unsigned*>(B_local_6 + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_6 + 0), reinterpret_cast<const unsigned*>(B_local_6 + ((j * 8) + 4)));
        }
      }
    }
    overlap_plan_mbar[6].arrive();
    overlap_plan_mbar[6].wait(0);
    __syncthreads();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(65536);
      tl::fence_proxy_async();
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[0])), 0, ((k * 384) + 192), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[4096])), 64, ((k * 384) + 192), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[8192])), 128, ((k * 384) + 192), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[12288])), 192, ((k * 384) + 192), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[16384])), 256, ((k * 384) + 192), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[20480])), 320, ((k * 384) + 192), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[24576])), 384, ((k * 384) + 192), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[28672])), 448, ((k * 384) + 192), 0, 0);
    }
    overlap_plan_mbar[9].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[5].arrive_and_expect_tx(8192);
      tl::fence_proxy_async();
      tl::tma_load(K_pe_desc, overlap_plan_mbar[5], (&(((half_t*)K_pe_shared)[0])), 0, ((k * 384) + 192), 0, 0);
    }
    overlap_plan_mbar[2].wait(1);
    {
      half_t A_local_7[8];
      half_t B_local_7[4];
      float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s_v1 + 0) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
      for (int ki_7 = 0; ki_7 < 32; ++ki_7) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_7 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_7 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_7[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki_7 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_7 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_7[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + 0), reinterpret_cast<const unsigned*>(A_local_7 + 0), reinterpret_cast<const unsigned*>(B_local_7 + 0));
      }
    }
    overlap_plan_mbar[5].wait(1);
    {
      half_t A_local_8[8];
      half_t B_local_8[4];
      for (int ki_8 = 0; ki_8 < 4; ++ki_8) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_8 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_8[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_8 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_8[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + 0), reinterpret_cast<const unsigned*>(A_local_8 + 0), reinterpret_cast<const unsigned*>(B_local_8 + 0));
      }
    }
    overlap_plan_mbar[9].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_10 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_10, broadcast_var_10);
    #pragma unroll
    for (int i_23 = 0; i_23 < 2; ++i_23) {
      scores_max_clear_3[i_23] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_6 = 0; rv_6 < 2; ++rv_6) {
        scores_max_clear_3[i_23] = max(scores_max_clear_3[i_23], acc_s_v1[((i_23 * 2) + rv_6)]);
      }
      __syncthreads();
      scores_max_clear_3[i_23] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_23], (&(((float*)workspace_6)[0])));
      scores_max_clear_3[i_23] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_3[i_23]);
      scores_max[i_23] = max(scores_max[i_23], scores_max_clear_3[i_23]);
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 2; ++i_24) {
      scores_max[i_24] = max(scores_max[i_24], scores_max_prev[i_24]);
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 2; ++i_25) {
      scores_scale_v0[i_25] = exp2f(((scores_max_prev[i_25] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_25] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 4; ++i_26) {
      acc_s_v1[i_26] = exp2f(((acc_s_v1[i_26] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_26 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 2; ++i_27) {
      scores_sum[i_27] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_7 = 0; rv_7 < 2; ++rv_7) {
        scores_sum[i_27] = (scores_sum[i_27] + acc_s_v1[((i_27 * 2) + rv_7)]);
      }
      __syncthreads();
      scores_sum[i_27] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_27], (&(((float*)workspace_7)[0])));
      scores_sum[i_27] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_27]);
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 2; ++i_28) {
      half_t S_shared_local_cast_2[2];
      uint1 __3;
      float2 v__2 = *(float2*)(acc_s_v0 + (i_28 * 2));
      ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
      *(uint1*)(S_shared_local_cast_2 + 0) = __3;
      *(uint1*)(((half_t*)S_shared) + ((((((i_28 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast_2 + 0);
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 2; ++i_29) {
      logsum[i_29] = ((logsum[i_29] * scores_scale_v0[i_29]) + scores_sum[i_29]);
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 32; ++i_30) {
      acc_o[i_30] = (acc_o[i_30] * scores_scale_v1[((i_30 & 3) >> 1)]);
    }
    overlap_plan_mbar[3].wait(0);
    {
      half_t A_local_9[8];
      half_t B_local_9[32];
      __syncthreads();
      for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
        tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 1024)])), (&(A_local_9[0])));
        for (int i_31 = 0; i_31 < 4; ++i_31) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) >> 5) * 4096) + (ki_9 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_31 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_31 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_9[(i_31 * 8)])));
        }
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_9 + 0), reinterpret_cast<const unsigned*>(B_local_9 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_9 + 0), reinterpret_cast<const unsigned*>(B_local_9 + ((j_1 * 8) + 4)));
        }
      }
    }
    overlap_plan_mbar[7].arrive();
    overlap_plan_mbar[7].wait(0);
    __syncthreads();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(65536);
      tl::fence_proxy_async();
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[32768])), 0, ((k * 384) + 256), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[36864])), 64, ((k * 384) + 256), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[40960])), 128, ((k * 384) + 256), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[45056])), 192, ((k * 384) + 256), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[49152])), 256, ((k * 384) + 256), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[53248])), 320, ((k * 384) + 256), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[57344])), 384, ((k * 384) + 256), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[61440])), 448, ((k * 384) + 256), 0, 0);
    }
    overlap_plan_mbar[9].wait(1);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[5].arrive_and_expect_tx(8192);
      tl::fence_proxy_async();
      tl::tma_load(K_pe_desc, overlap_plan_mbar[5], (&(((half_t*)K_pe_shared)[0])), 0, ((k * 384) + 256), 0, 0);
    }
    overlap_plan_mbar[3].wait(1);
    {
      half_t A_local_10[8];
      half_t B_local_10[4];
      float broadcast_var_11 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s_v0 + 0) = make_float4(broadcast_var_11, broadcast_var_11, broadcast_var_11, broadcast_var_11);
      for (int ki_10 = 0; ki_10 < 32; ++ki_10) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_10 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_10 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_10[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[(((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki_10 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_10 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 32768)])), (&(B_local_10[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + 0), reinterpret_cast<const unsigned*>(A_local_10 + 0), reinterpret_cast<const unsigned*>(B_local_10 + 0));
      }
    }
    overlap_plan_mbar[5].wait(0);
    {
      half_t A_local_11[8];
      half_t B_local_11[4];
      for (int ki_11 = 0; ki_11 < 4; ++ki_11) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_11[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_11[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + 0), reinterpret_cast<const unsigned*>(A_local_11 + 0), reinterpret_cast<const unsigned*>(B_local_11 + 0));
      }
    }
    overlap_plan_mbar[9].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_12 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_12, broadcast_var_12);
    #pragma unroll
    for (int i_32 = 0; i_32 < 2; ++i_32) {
      scores_max_clear_4[i_32] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_8 = 0; rv_8 < 2; ++rv_8) {
        scores_max_clear_4[i_32] = max(scores_max_clear_4[i_32], acc_s_v0[((i_32 * 2) + rv_8)]);
      }
      __syncthreads();
      scores_max_clear_4[i_32] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_4[i_32], (&(((float*)workspace_10)[0])));
      scores_max_clear_4[i_32] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_4[i_32]);
      scores_max[i_32] = max(scores_max[i_32], scores_max_clear_4[i_32]);
    }
    #pragma unroll
    for (int i_33 = 0; i_33 < 2; ++i_33) {
      scores_max[i_33] = max(scores_max[i_33], scores_max_prev[i_33]);
    }
    #pragma unroll
    for (int i_34 = 0; i_34 < 2; ++i_34) {
      scores_scale_v1[i_34] = exp2f(((scores_max_prev[i_34] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_34] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_35 = 0; i_35 < 4; ++i_35) {
      acc_s_v0[i_35] = exp2f(((acc_s_v0[i_35] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_35 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_36 = 0; i_36 < 2; ++i_36) {
      scores_sum[i_36] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_9 = 0; rv_9 < 2; ++rv_9) {
        scores_sum[i_36] = (scores_sum[i_36] + acc_s_v0[((i_36 * 2) + rv_9)]);
      }
      __syncthreads();
      scores_sum[i_36] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_36], (&(((float*)workspace_11)[0])));
      scores_sum[i_36] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_36]);
    }
    #pragma unroll
    for (int i_37 = 0; i_37 < 2; ++i_37) {
      half_t S_shared_local_cast_3[2];
      uint1 __4;
      float2 v__3 = *(float2*)(acc_s_v1 + (i_37 * 2));
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
      *(uint1*)(S_shared_local_cast_3 + 0) = __4;
      *(uint1*)(((half_t*)S_shared) + (((((((i_37 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1024)) = *(uint1*)(S_shared_local_cast_3 + 0);
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 2; ++i_38) {
      logsum[i_38] = ((logsum[i_38] * scores_scale_v1[i_38]) + scores_sum[i_38]);
    }
    #pragma unroll
    for (int i_39 = 0; i_39 < 32; ++i_39) {
      acc_o[i_39] = (acc_o[i_39] * scores_scale_v2[((i_39 & 3) >> 1)]);
    }
    overlap_plan_mbar[4].wait(0);
    {
      half_t A_local_12[8];
      half_t B_local_12[32];
      for (int ki_12 = 0; ki_12 < 4; ++ki_12) {
        tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_12 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_12 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_12[0])));
        for (int i_40 = 0; i_40 < 4; ++i_40) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) >> 5) * 4096) + (ki_12 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_40 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_40 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 65536)])), (&(B_local_12[(i_40 * 8)])));
        }
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_12 + 0), reinterpret_cast<const unsigned*>(B_local_12 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_12 + 0), reinterpret_cast<const unsigned*>(B_local_12 + ((j_2 * 8) + 4)));
        }
      }
    }
    overlap_plan_mbar[8].arrive();
    overlap_plan_mbar[8].wait(0);
    __syncthreads();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[4].arrive_and_expect_tx(65536);
      tl::fence_proxy_async();
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[65536])), 0, ((k * 384) + 320), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[69632])), 64, ((k * 384) + 320), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[73728])), 128, ((k * 384) + 320), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[77824])), 192, ((k * 384) + 320), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[81920])), 256, ((k * 384) + 320), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[86016])), 320, ((k * 384) + 320), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[90112])), 384, ((k * 384) + 320), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[4], (&(((half_t*)KV_shared)[94208])), 448, ((k * 384) + 320), 0, 0);
    }
    overlap_plan_mbar[9].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[5].arrive_and_expect_tx(8192);
      tl::fence_proxy_async();
      tl::tma_load(K_pe_desc, overlap_plan_mbar[5], (&(((half_t*)K_pe_shared)[0])), 0, ((k * 384) + 320), 0, 0);
    }
    overlap_plan_mbar[4].wait(1);
    {
      half_t A_local_13[8];
      half_t B_local_13[4];
      float broadcast_var_13 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s_v1 + 0) = make_float4(broadcast_var_13, broadcast_var_13, broadcast_var_13, broadcast_var_13);
      for (int ki_13 = 0; ki_13 < 32; ++ki_13) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_13 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_13 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_13[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) + 2) % 3) * 32768) + ((ki_13 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_13 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_13[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + 0), reinterpret_cast<const unsigned*>(A_local_13 + 0), reinterpret_cast<const unsigned*>(B_local_13 + 0));
      }
    }
    overlap_plan_mbar[5].wait(1);
    {
      half_t A_local_14[8];
      half_t B_local_14[4];
      for (int ki_14 = 0; ki_14 < 4; ++ki_14) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_14 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_14 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_14[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_14 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_14 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_14[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + 0), reinterpret_cast<const unsigned*>(A_local_14 + 0), reinterpret_cast<const unsigned*>(B_local_14 + 0));
      }
    }
    overlap_plan_mbar[9].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_14 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_14, broadcast_var_14);
    #pragma unroll
    for (int i_41 = 0; i_41 < 2; ++i_41) {
      scores_max_clear_5[i_41] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_10 = 0; rv_10 < 2; ++rv_10) {
        scores_max_clear_5[i_41] = max(scores_max_clear_5[i_41], acc_s_v1[((i_41 * 2) + rv_10)]);
      }
      __syncthreads();
      scores_max_clear_5[i_41] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_5[i_41], (&(((float*)workspace_13)[0])));
      scores_max_clear_5[i_41] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_5[i_41]);
      scores_max[i_41] = max(scores_max[i_41], scores_max_clear_5[i_41]);
    }
    #pragma unroll
    for (int i_42 = 0; i_42 < 2; ++i_42) {
      scores_max[i_42] = max(scores_max[i_42], scores_max_prev[i_42]);
    }
    #pragma unroll
    for (int i_43 = 0; i_43 < 2; ++i_43) {
      scores_scale_v2[i_43] = exp2f(((scores_max_prev[i_43] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_43] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_44 = 0; i_44 < 4; ++i_44) {
      acc_s_v1[i_44] = exp2f(((acc_s_v1[i_44] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_44 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_45 = 0; i_45 < 2; ++i_45) {
      scores_sum[i_45] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_11 = 0; rv_11 < 2; ++rv_11) {
        scores_sum[i_45] = (scores_sum[i_45] + acc_s_v1[((i_45 * 2) + rv_11)]);
      }
      __syncthreads();
      scores_sum[i_45] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_45], (&(((float*)workspace)[0])));
      scores_sum[i_45] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_45]);
    }
    #pragma unroll
    for (int i_46 = 0; i_46 < 2; ++i_46) {
      half_t S_shared_local_cast_4[2];
      uint1 __5;
      float2 v__4 = *(float2*)(acc_s_v0 + (i_46 * 2));
      ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&v__4))[0]);
      *(uint1*)(S_shared_local_cast_4 + 0) = __5;
      *(uint1*)(((half_t*)S_shared) + ((((((i_46 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast_4 + 0);
    }
    #pragma unroll
    for (int i_47 = 0; i_47 < 2; ++i_47) {
      logsum[i_47] = ((logsum[i_47] * scores_scale_v2[i_47]) + scores_sum[i_47]);
    }
    #pragma unroll
    for (int i_48 = 0; i_48 < 32; ++i_48) {
      acc_o[i_48] = (acc_o[i_48] * scores_scale_v0[((i_48 & 3) >> 1)]);
    }
    overlap_plan_mbar[2].wait(1);
    {
      half_t A_local_15[8];
      half_t B_local_15[32];
      for (int ki_15 = 0; ki_15 < 4; ++ki_15) {
        tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_15 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_15 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 1024)])), (&(A_local_15[0])));
        for (int i_49 = 0; i_49 < 4; ++i_49) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki_15 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_49 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_49 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_15[(i_49 * 8)])));
        }
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_3 * 8)), reinterpret_cast<const unsigned*>(A_local_15 + 0), reinterpret_cast<const unsigned*>(B_local_15 + (j_3 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_15 + 0), reinterpret_cast<const unsigned*>(B_local_15 + ((j_3 * 8) + 4)));
        }
      }
    }
    overlap_plan_mbar[6].arrive();
    overlap_plan_mbar[6].wait(1);
    __syncthreads();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[2].arrive_and_expect_tx(65536);
      tl::fence_proxy_async();
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[0])), 0, ((k * 384) + 384), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[4096])), 64, ((k * 384) + 384), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[8192])), 128, ((k * 384) + 384), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[12288])), 192, ((k * 384) + 384), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[16384])), 256, ((k * 384) + 384), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[20480])), 320, ((k * 384) + 384), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[24576])), 384, ((k * 384) + 384), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[2], (&(((half_t*)KV_shared)[28672])), 448, ((k * 384) + 384), 0, 0);
    }
    overlap_plan_mbar[9].wait(1);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[5].arrive_and_expect_tx(8192);
      tl::fence_proxy_async();
      tl::tma_load(K_pe_desc, overlap_plan_mbar[5], (&(((half_t*)K_pe_shared)[0])), 0, ((k * 384) + 384), 0, 0);
    }
    overlap_plan_mbar[2].wait(0);
    {
      half_t A_local_16[8];
      half_t B_local_16[4];
      float broadcast_var_15 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s_v0 + 0) = make_float4(broadcast_var_15, broadcast_var_15, broadcast_var_15, broadcast_var_15);
      for (int ki_16 = 0; ki_16 < 32; ++ki_16) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_16 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_16 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_16[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki_16 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_16 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_16[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + 0), reinterpret_cast<const unsigned*>(A_local_16 + 0), reinterpret_cast<const unsigned*>(B_local_16 + 0));
      }
    }
    overlap_plan_mbar[5].wait(0);
    {
      half_t A_local_17[8];
      half_t B_local_17[4];
      for (int ki_17 = 0; ki_17 < 4; ++ki_17) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_17 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_17[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_17 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_17[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v0 + 0), reinterpret_cast<const unsigned*>(A_local_17 + 0), reinterpret_cast<const unsigned*>(B_local_17 + 0));
      }
    }
    overlap_plan_mbar[9].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_16 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_16, broadcast_var_16);
    #pragma unroll
    for (int i_50 = 0; i_50 < 2; ++i_50) {
      scores_max_clear_6[i_50] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_12 = 0; rv_12 < 2; ++rv_12) {
        scores_max_clear_6[i_50] = max(scores_max_clear_6[i_50], acc_s_v0[((i_50 * 2) + rv_12)]);
      }
      __syncthreads();
      scores_max_clear_6[i_50] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_6[i_50], (&(((float*)workspace_14)[0])));
      scores_max_clear_6[i_50] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_6[i_50]);
      scores_max[i_50] = max(scores_max[i_50], scores_max_clear_6[i_50]);
    }
    #pragma unroll
    for (int i_51 = 0; i_51 < 2; ++i_51) {
      scores_max[i_51] = max(scores_max[i_51], scores_max_prev[i_51]);
    }
    #pragma unroll
    for (int i_52 = 0; i_52 < 2; ++i_52) {
      scores_scale_v0[i_52] = exp2f(((scores_max_prev[i_52] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_52] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_53 = 0; i_53 < 4; ++i_53) {
      acc_s_v0[i_53] = exp2f(((acc_s_v0[i_53] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_53 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_54 = 0; i_54 < 2; ++i_54) {
      scores_sum[i_54] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_13 = 0; rv_13 < 2; ++rv_13) {
        scores_sum[i_54] = (scores_sum[i_54] + acc_s_v0[((i_54 * 2) + rv_13)]);
      }
      __syncthreads();
      scores_sum[i_54] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_54], (&(((float*)workspace_1)[0])));
      scores_sum[i_54] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_54]);
    }
    #pragma unroll
    for (int i_55 = 0; i_55 < 2; ++i_55) {
      half_t S_shared_local_cast_5[2];
      uint1 __6;
      float2 v__5 = *(float2*)(acc_s_v1 + (i_55 * 2));
      ((half2*)(&__6))[0] = __float22half2_rn(((float2*)(&v__5))[0]);
      *(uint1*)(S_shared_local_cast_5 + 0) = __6;
      *(uint1*)(((half_t*)S_shared) + (((((((i_55 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1024)) = *(uint1*)(S_shared_local_cast_5 + 0);
    }
    #pragma unroll
    for (int i_56 = 0; i_56 < 2; ++i_56) {
      logsum[i_56] = ((logsum[i_56] * scores_scale_v0[i_56]) + scores_sum[i_56]);
    }
    #pragma unroll
    for (int i_57 = 0; i_57 < 32; ++i_57) {
      acc_o[i_57] = (acc_o[i_57] * scores_scale_v1[((i_57 & 3) >> 1)]);
    }
    overlap_plan_mbar[3].wait(1);
    {
      half_t A_local_18[8];
      half_t B_local_18[32];
      __syncthreads();
      for (int ki_18 = 0; ki_18 < 4; ++ki_18) {
        tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_18 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_18 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_18[0])));
        for (int i_58 = 0; i_58 < 4; ++i_58) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) >> 5) * 4096) + (ki_18 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_58 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_58 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_18[(i_58 * 8)])));
        }
        for (int j_4 = 0; j_4 < 4; ++j_4) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local_18 + 0), reinterpret_cast<const unsigned*>(B_local_18 + (j_4 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_18 + 0), reinterpret_cast<const unsigned*>(B_local_18 + ((j_4 * 8) + 4)));
        }
      }
    }
    overlap_plan_mbar[7].arrive();
    overlap_plan_mbar[7].wait(1);
    __syncthreads();
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[3].arrive_and_expect_tx(65536);
      tl::fence_proxy_async();
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[32768])), 0, ((k * 384) + 448), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[36864])), 64, ((k * 384) + 448), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[40960])), 128, ((k * 384) + 448), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[45056])), 192, ((k * 384) + 448), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[49152])), 256, ((k * 384) + 448), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[53248])), 320, ((k * 384) + 448), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[57344])), 384, ((k * 384) + 448), 0, 0);
      tl::tma_load(KV_desc, overlap_plan_mbar[3], (&(((half_t*)KV_shared)[61440])), 448, ((k * 384) + 448), 0, 0);
    }
    overlap_plan_mbar[9].wait(0);
    if (tl::tl_shuffle_elect<256>() && ((((int)threadIdx.x) >> 5) == 0)) {
      overlap_plan_mbar[5].arrive_and_expect_tx(8192);
      tl::fence_proxy_async();
      tl::tma_load(K_pe_desc, overlap_plan_mbar[5], (&(((half_t*)K_pe_shared)[0])), 0, ((k * 384) + 448), 0, 0);
    }
    overlap_plan_mbar[3].wait(0);
    {
      half_t A_local_19[8];
      half_t B_local_19[4];
      float broadcast_var_17 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s_v1 + 0) = make_float4(broadcast_var_17, broadcast_var_17, broadcast_var_17, broadcast_var_17);
      for (int ki_19 = 0; ki_19 < 32; ++ki_19) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_19 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_19 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_19[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)KV_shared)[(((((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) >> 3) * 32768) + ((ki_19 >> 2) * 4096)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_19 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 32768)])), (&(B_local_19[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + 0), reinterpret_cast<const unsigned*>(A_local_19 + 0), reinterpret_cast<const unsigned*>(B_local_19 + 0));
      }
    }
    overlap_plan_mbar[5].wait(1);
    {
      half_t A_local_20[8];
      half_t B_local_20[4];
      for (int ki_20 = 0; ki_20 < 4; ++ki_20) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_pe_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_20 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_20 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_20[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)K_pe_shared)[((((((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_20 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_20 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_20[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_v1 + 0), reinterpret_cast<const unsigned*>(A_local_20 + 0), reinterpret_cast<const unsigned*>(B_local_20 + 0));
      }
    }
    overlap_plan_mbar[9].arrive();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_18 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_18, broadcast_var_18);
    #pragma unroll
    for (int i_59 = 0; i_59 < 2; ++i_59) {
      scores_max_clear_7[i_59] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_14 = 0; rv_14 < 2; ++rv_14) {
        scores_max_clear_7[i_59] = max(scores_max_clear_7[i_59], acc_s_v1[((i_59 * 2) + rv_14)]);
      }
      __syncthreads();
      scores_max_clear_7[i_59] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear_7[i_59], (&(((float*)workspace_12)[0])));
      scores_max_clear_7[i_59] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear_7[i_59]);
      scores_max[i_59] = max(scores_max[i_59], scores_max_clear_7[i_59]);
    }
    #pragma unroll
    for (int i_60 = 0; i_60 < 2; ++i_60) {
      scores_max[i_60] = max(scores_max[i_60], scores_max_prev[i_60]);
    }
    #pragma unroll
    for (int i_61 = 0; i_61 < 2; ++i_61) {
      scores_scale_v1[i_61] = exp2f(((scores_max_prev[i_61] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[i_61] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_62 = 0; i_62 < 4; ++i_62) {
      acc_s_v1[i_62] = exp2f(((acc_s_v1[i_62] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[(i_62 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    #pragma unroll
    for (int i_63 = 0; i_63 < 2; ++i_63) {
      scores_sum[i_63] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_15 = 0; rv_15 < 2; ++rv_15) {
        scores_sum[i_63] = (scores_sum[i_63] + acc_s_v1[((i_63 * 2) + rv_15)]);
      }
      __syncthreads();
      scores_sum[i_63] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_63], (&(((float*)workspace_15)[0])));
      scores_sum[i_63] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_63]);
    }
    #pragma unroll
    for (int i_64 = 0; i_64 < 2; ++i_64) {
      half_t S_shared_local_cast_6[2];
      uint1 __7;
      float2 v__6 = *(float2*)(acc_s_v0 + (i_64 * 2));
      ((half2*)(&__7))[0] = __float22half2_rn(((float2*)(&v__6))[0]);
      *(uint1*)(S_shared_local_cast_6 + 0) = __7;
      *(uint1*)(((half_t*)S_shared) + ((((((i_64 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast_6 + 0);
    }
    #pragma unroll
    for (int i_65 = 0; i_65 < 2; ++i_65) {
      logsum[i_65] = ((logsum[i_65] * scores_scale_v1[i_65]) + scores_sum[i_65]);
    }
    #pragma unroll
    for (int i_66 = 0; i_66 < 32; ++i_66) {
      acc_o[i_66] = (acc_o[i_66] * scores_scale_v2[((i_66 & 3) >> 1)]);
    }
    overlap_plan_mbar[4].wait(1);
    {
      half_t A_local_21[8];
      half_t B_local_21[32];
      for (int ki_21 = 0; ki_21 < 4; ++ki_21) {
        tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_21 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 1024)])), (&(A_local_21[0])));
        for (int i_67 = 0; i_67 < 4; ++i_67) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) >> 5) * 4096) + (ki_21 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_67 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_67 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 65536)])), (&(B_local_21[(i_67 * 8)])));
        }
        for (int j_5 = 0; j_5 < 4; ++j_5) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_5 * 8)), reinterpret_cast<const unsigned*>(A_local_21 + 0), reinterpret_cast<const unsigned*>(B_local_21 + (j_5 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_21 + 0), reinterpret_cast<const unsigned*>(B_local_21 + ((j_5 * 8) + 4)));
        }
      }
    }
    overlap_plan_mbar[8].arrive();
  }
  __syncthreads();
  #pragma unroll
  for (int i_68 = 0; i_68 < 2; ++i_68) {
    half_t S_shared_local_cast_7[2];
    uint1 __8;
    float2 v__7 = *(float2*)(acc_s_v1 + (i_68 * 2));
    ((half2*)(&__8))[0] = __float22half2_rn(((float2*)(&v__7))[0]);
    *(uint1*)(S_shared_local_cast_7 + 0) = __8;
    *(uint1*)(((half_t*)S_shared) + (((((((i_68 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1024)) = *(uint1*)(S_shared_local_cast_7 + 0);
  }
  #pragma unroll
  for (int i_69 = 0; i_69 < 32; ++i_69) {
    acc_o[i_69] = (acc_o[i_69] * scores_scale_v0[((i_69 & 3) >> 1)]);
  }
  overlap_plan_mbar[2].wait(0);
  {
    half_t A_local_22[8];
    half_t B_local_22[32];
    for (int ki_22 = 0; ki_22 < 4; ++ki_22) {
      tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_22 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_22 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_22[0])));
      for (int i_70 = 0; i_70 < 4; ++i_70) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki_22 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_70 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_70 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_22[(i_70 * 8)])));
      }
      for (int j_6 = 0; j_6 < 4; ++j_6) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_6 * 8)), reinterpret_cast<const unsigned*>(A_local_22 + 0), reinterpret_cast<const unsigned*>(B_local_22 + (j_6 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_6 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_22 + 0), reinterpret_cast<const unsigned*>(B_local_22 + ((j_6 * 8) + 4)));
      }
    }
  }
  overlap_plan_mbar[6].arrive();
  #pragma unroll
  for (int i_71 = 0; i_71 < 32; ++i_71) {
    acc_o[i_71] = (acc_o[i_71] * scores_scale_v1[((i_71 & 3) >> 1)]);
  }
  overlap_plan_mbar[3].wait(0);
  {
    half_t A_local_23[8];
    half_t B_local_23[32];
    __syncthreads();
    for (int ki_23 = 0; ki_23 < 4; ++ki_23) {
      tl::ptx_ldmatrix_x4((&(((half_t*)S_shared)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_23 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_23 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 1024)])), (&(A_local_23[0])));
      for (int i_72 = 0; i_72 < 4; ++i_72) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)KV_shared)[((((((((int)threadIdx.x) >> 5) * 4096) + (ki_23 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_72 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_72 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_23[(i_72 * 8)])));
      }
      for (int j_7 = 0; j_7 < 4; ++j_7) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_7 * 8)), reinterpret_cast<const unsigned*>(A_local_23 + 0), reinterpret_cast<const unsigned*>(B_local_23 + (j_7 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_7 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_23 + 0), reinterpret_cast<const unsigned*>(B_local_23 + ((j_7 * 8) + 4)));
      }
    }
  }
  overlap_plan_mbar[7].arrive();
  #pragma unroll
  for (int i_73 = 0; i_73 < 32; ++i_73) {
    acc_o[i_73] = (acc_o[i_73] / logsum[((i_73 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_74 = 0; i_74 < 4; ++i_74) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((int)threadIdx.x) & 15) * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (i_74 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_74 * 8)]), ((half_t)acc_o[((i_74 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_74 * 8) + 2)]), ((half_t)acc_o[((i_74 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_74 * 8) + 4)]), ((half_t)acc_o[((i_74 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_74 * 8) + 6)]), ((half_t)acc_o[((i_74 * 8) + 7)])));
  }
  if (tl::tl_shuffle_elect<256>()) {
    tl::fence_proxy_async();
    __syncthreads();
    tl::tma_store((&(Output[(((int)blockIdx.x) * 8192)])), (&(((half_t*)O_shared)[0])), 16384);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

