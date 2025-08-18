#pragma once
#include <cuda_runtime.h>

namespace {
__global__ void verify_kernel(int num_tokens, int32_t* pred, const int32_t* gt, const int32_t* position_ids, const int32_t* cache_length, const uint64_t* attn_mask, const int32_t* tree_parent, int32_t* d_best) {
    int i = threadIdx.x;

    __shared__ uint64_t s_correct_mask[2];
    uint64_t correct_mask = 1;
    if (0 < i && i < num_tokens && pred[i] == gt[tree_parent[i]]) correct_mask |= 1ULL << i;
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 16);
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 8);
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 4);
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 2);
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 1);
    if (i % 32 == 0) s_correct_mask[i / 32] = correct_mask;
    __syncthreads();
    if (i == 0) s_correct_mask[0] |= s_correct_mask[1];
    __syncthreads();
    correct_mask = s_correct_mask[0];

    __shared__ int32_t mx[64], mx_idx[64];
    int prefix_length = cache_length[0];
    if (i < num_tokens && ((correct_mask & attn_mask[i]) == attn_mask[i])) {
        mx[i] = position_ids[i] - prefix_length + 1; mx_idx[i] = i;
    } else {
        mx[i] = 1; mx_idx[i] = 0;
    }
    __syncthreads();
    for (int offset = 32; offset > 0; offset >>= 1) {
        if (i < offset && mx[i + offset] > mx[i]) {
            mx[i] = mx[i + offset];
            mx_idx[i] = mx_idx[i + offset];
        }
        __syncthreads();
    }
    if (i == 0) {
        d_best[0] = mx[0]; d_best[1] = mx_idx[0];
    }
    __syncthreads();

    int p = mx_idx[0];
    if (i < num_tokens && (attn_mask[p] >> i & 1)) {
        pred[position_ids[i] - prefix_length] = i;
    }
}

template<typename T>
__global__ void fix_kvcache_kernel_1(int num_caches, int dim, int32_t* pred, const int32_t* gt, const int32_t* cache_length, const T* const* flat_caches, float4* tmp_kvcache) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k = blockIdx.y;
    int prefix_length = cache_length[0];
    int real_i = pred[i] + prefix_length;
    float4* tmp = tmp_kvcache + i * num_caches * dim;
    const float4* flat = (const float4*)flat_caches[k];
    for (int d = j; d < dim; d += blockDim.x) {
        tmp[k * dim + d] = flat[real_i * dim + d];
    }
}

template<typename T>
__global__ void fix_kvcache_kernel_2(int num_caches, int dim, int32_t* pred, const int32_t* gt, const int32_t* cache_length, T** flat_caches, const float4* tmp_kvcache) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k = blockIdx.y;
    int prefix_length = cache_length[0];
    int real_i = i + prefix_length;
    const float4* tmp = tmp_kvcache + i * num_caches * dim;
    float4* flat = (float4*)flat_caches[k];
    for (int d = j; d < dim; d += blockDim.x) {
        flat[real_i * dim + d] = tmp[k * dim + d];
    }
    if (j == 0 && k == 0) {
        pred[i] = gt[pred[i]];
    }
}

template<typename T>
__global__ void remap_copy_kernel(int32_t dim, const T* src, T* dst, const int32_t* token_id_remap) {
    int row = blockIdx.x;
    int real_row = token_id_remap[row];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[row * dim + i] = src[real_row * dim + i];
    }
}

__global__ void remap_kernel(int32_t num_tokens, const int32_t* input, int32_t* output, const int32_t* token_id_remap) {
    output[threadIdx.x] = token_id_remap[input[threadIdx.x]];
}
}

void verify_draft(const Stream& stream, int num_tokens, int32_t* pred, const int32_t* gt, const int32_t* position_ids, const int32_t* cache_length, const uint64_t* attn_mask, const int32_t* tree_parent, int32_t* best) {
    verify_kernel<<<1, 64, 0, stream.stream>>>(num_tokens, pred, gt, position_ids, cache_length, attn_mask, tree_parent, best);
}

template<typename T>
void fix_kv_cache(const Stream& stream, int accept_length, int num_caches, int dim, int32_t* pred, const int32_t* gt, const int32_t* cache_length, T** flat_caches, T* tmp_kvcache) {
    fix_kvcache_kernel_1<T><<<dim3(accept_length, num_caches, 1), 256, 0, stream.stream>>>(num_caches, dim/(16/sizeof(T)), pred, gt, cache_length, flat_caches, (float4*)tmp_kvcache);
    fix_kvcache_kernel_2<T><<<dim3(accept_length, num_caches, 1), 256, 0, stream.stream>>>(num_caches, dim/(16/sizeof(T)), pred, gt, cache_length, flat_caches, (float4*)tmp_kvcache);
}

template<typename T>
void remap_copy(const Stream& stream, const T* src, T* dst, int32_t dim, int32_t num_tokens, const int32_t* token_id_remap) {
    dim = dim / (16 / sizeof(T));
    remap_copy_kernel<<<num_tokens, 512, 0, stream.stream>>>(dim, (float4*)src, (float4*)dst, token_id_remap);
}

void remap(const Stream& stream, int32_t num_tokens, const int32_t* input, int32_t* output, const int32_t* token_id_remap) {
    remap_kernel<<<1, num_tokens, 0, stream.stream>>>(num_tokens, input, output, token_id_remap);
}