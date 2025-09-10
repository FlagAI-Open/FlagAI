#pragma once
#include <cuda_fp16.h>
#include "core/scalar_type.hpp"

namespace marlin {

int determine_reduce_max_m(int prob_m, int max_par);

}

template <typename T>
void gptq_marlin_gemm(T* a, int32_t* b_q_weight,
                               T* b_scales, int32_t* b_zeros,
                               int32_t* g_idx, int32_t* perm,
                               int32_t* workspace,
                               vllm::ScalarType const& b_q_type, // init in linear
                               int64_t size_m, int64_t size_n, int64_t size_k,
                               bool is_k_full, bool has_zp,
                               bool use_fp32_reduce, 
                               T* c,
                               int num_groups, int group_size,
                               int b_q_weight_size1,
                               bool has_act_order,
                               cudaStream_t stream,
                               T* a_tmp,
                               float* c_tmp
                               );