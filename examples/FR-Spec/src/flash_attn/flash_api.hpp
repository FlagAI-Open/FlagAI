/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// We modified flash-attention based on https://github.com/Dao-AILab/flash-attention/blob/v2.4.2/csrc/flash_attn/flash_api.cpp

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"
#include "../model/mask.cuh"

void set_params_fprop(Flash_fwd_params &params,
                      bool is_bf16,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      void* q,
                      void* k,
                      void* v,
                      void* out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false) {

    // Reset the parameters
    params = {};

    params.is_bf16 = is_bf16;

    // Set the pointers and strides.
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v;
    // All stride are in elements, not bytes.
    params.q_row_stride = h * d;
    params.k_row_stride = h_k * d;
    params.v_row_stride = h_k * d;
    params.q_head_stride = d;
    params.k_head_stride = d;
    params.v_head_stride = d;
    params.o_ptr = out;
    params.o_row_stride = h * d;
    params.o_head_stride = d;

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = seqlen_q * h * d;
        params.k_batch_stride = seqlen_k * h_k * d;
        params.v_batch_stride = seqlen_k * h_k * d;
        params.o_batch_stride = seqlen_q * h * d;
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

int set_params_splitkv(const int batch_size, const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q) {
    cudaDeviceProp dprops;
    cudaGetDeviceProperties(&dprops, 0);

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;

    // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
    int num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, dprops.multiProcessorCount * 2, num_n_blocks, 128);

    return num_splits;
}


void mha_fwd_kvcache(
    bool is_bf16,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int seqlen_knew,
    int num_heads,
    int num_heads_k,
    int head_size,
    void* q,                    // batch_size x seqlen_q x num_heads x head_size
    void* kcache,               // batch_size x seqlen_k x num_heads_k x head_size
    void* vcache,               // batch_size x seqlen_k x num_heads_k x head_size
    int* seqlens_k,             // batch_size
    const Mask& mask,           // batch_size x seqlen_q x seqlen_k_range
    void* out,                  // batch_size x seqlen_q x num_heads x head_size
    float* softmax_lse,         // batch_size x num_heads x seqlen_q
    float* softmax_lse_accum,   // num_splits x batch_size x num_heads x seqlen_q
    float* oaccum,              // num_splits x batch_size x num_heads x seqlen_q x head_size
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    cudaStream_t stream
) {
    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // TODO ignore this for now
    // // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // // H/t Daniel Haziza
    // const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
    // if (seqlenq_ngroups_swapped) {
    //     const int ngroups = num_heads / num_heads_k;
    //     q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
    //     seqlen_q = ngroups;
    //     num_heads = num_heads_k;
    // }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = head_size <= 192 ? round_multiple(head_size, 32) : 256;
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    Flash_fwd_params params;
    set_params_fprop(params,
                     is_bf16,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, kcache, vcache, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     /*p_ptr=*/nullptr,
                     (void*)softmax_lse,
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap
                     );

    params.mask_2d = mask.ptr;
    params.mask_q_range = mask.mask_q_range;
    params.mask_k_range = mask.mask_k_range;

    params.rotary_dim = 0;

    params.softmax_lseaccum_ptr = (void*)softmax_lse_accum;
    params.oaccum_ptr = (void*)oaccum;

    params.page_block_size = 1;

    params.alibi_slopes_ptr = nullptr;

    if (seqlens_k != nullptr) {
        params.cu_seqlens_k = seqlens_k;
        params.is_seqlens_k_cumulative = false;
        params.num_splits = 4; // TODO 4 for decode
    } else {
        params.num_splits = 1;
    }

    run_mha_fwd(params, stream);

    // TODO ignore this for now
    // if (seqlenq_ngroups_swapped) {
    //     out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
    //     softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    // }
}
