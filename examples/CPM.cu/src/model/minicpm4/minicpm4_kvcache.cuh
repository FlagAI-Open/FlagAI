#pragma once
#include "../kvcache.cuh"
#include "../topk.cuh"

namespace {
template <typename T>
__global__ void meanpooling_16_kernel(int left, int dim, T* compressed, const T* k_cache) {
    __shared__ T s[32][33];

    int idx = blockIdx.x + left;
    int orig_left = idx * 16;
    T* c = compressed + idx * dim;
    const T* k = k_cache + orig_left * dim;
    int i = threadIdx.x / 32;
    int j = threadIdx.x % 32;

    for (int offset = 0; offset < dim; offset += 32) {
        s[i][j] = k[i * dim + offset + j];
        __syncthreads();
        float v = s[j][i];
        v += __shfl_down_sync(0xffffffff, v, 16);
        v += __shfl_down_sync(0xffffffff, v, 8);
        v += __shfl_down_sync(0xffffffff, v, 4);
        v += __shfl_down_sync(0xffffffff, v, 2);
        v += __shfl_down_sync(0xffffffff, v, 1);
        if (j == 0) {
            c[offset + i] = T(v / 32.0f);
        }
    }
}

template <typename T>
__global__ void meanpooling_64_kernel(int left, int dim, T* compressed, const T* k_cache) {
    __shared__ T s[32][33];

    int idx = blockIdx.x + left;
    int orig_left = idx * 64;
    T* c = compressed + idx * dim;
    const T* k = k_cache + orig_left * dim;
    int i = threadIdx.x / 32;
    int j = threadIdx.x % 32;

    for (int offset = 0; offset < dim; offset += 32) {
        float v_sum[32] = {0};
        for (int offset_row = 0; offset_row < 128; offset_row += 32) {
            s[i][j] = k[(i + offset_row) * dim + offset + j];
            __syncthreads();
            float v = s[j][i];
            v += __shfl_down_sync(0xffffffff, v, 16);
            v += __shfl_down_sync(0xffffffff, v, 8);
            v += __shfl_down_sync(0xffffffff, v, 4);
            v += __shfl_down_sync(0xffffffff, v, 2);
            v += __shfl_down_sync(0xffffffff, v, 1);
            if (j == 0) {
                v_sum[i] += v;
            }
        }
        if (j == 0) {
            c[offset + i] = T(v_sum[i] / 128.0f);
        }
    }
}

template <typename T>
__global__ void maxpooling_kernel(
    const T* input,
    T* output,
    int num_heads,
    int q_len,
    int q_round,
    int k_len,
    int out_len,
    int cache_len,
    int init_blocks,
    int local_blocks,
    int kernel_size,
    int stride,
    int padding,
    int block_size
) {
    int bidh = blockIdx.y;
    int bidq = blockIdx.x;
    const T* in = input + bidh * q_round * k_len + bidq * k_len;
    T* out = output + bidh * q_len * out_len + bidq * out_len;
    int q_block = (bidq + cache_len) / block_size;

    for (int k = threadIdx.x; k < out_len; k += blockDim.x) {
        int start = k * stride - padding;
        int end = start + kernel_size;
        start = max(start, 0);
        end = min(end, k_len);
        
        T max_val;
        if (k < init_blocks) {
            max_val = TypeTraits<T>::inf();
        } else if (q_block - local_blocks < k) {
            max_val = -TypeTraits<T>::inf();
        } else {
            max_val = in[start];
            for (int i = start + 1; i < end; i++) {
                if (in[i] > max_val) {
                    max_val = in[i];
                }
            }
        }
        out[k] = max_val;
    }
}

__global__ void kernel_topk_to_uint64(
    const int* topk_idx,
    uint64_t* result,
    int batch_size,
    int k,
    int k_blocks,
    int n_uint64_per_row
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    
    if (row >= batch_size || col >= n_uint64_per_row) return;
    
    int out_idx = row * n_uint64_per_row + col;
    
    int bit_start = col * 64;
    
    uint64_t packed_value = 0;
    
    for (int i = 0; i < k; i++) {
        int idx_offset = row * k + i;
        int idx = topk_idx[idx_offset];
        
        if (idx == -1) continue;
        
        if (idx >= bit_start && idx < bit_start + 64) {
            int local_bit = idx - bit_start;
            packed_value |= (1ULL << local_bit);
        }
    }
    
    result[out_idx] = packed_value;
}

template <typename T>
void meanpooling(const Stream& stream, int left, int right, int dim, T* compressed, const T* k_cache, int stride) {
    if (left == right) return;
    if (stride == 16) {
        meanpooling_16_kernel<<<right-left, 1024, 0, stream.stream>>>(left, dim, compressed, k_cache);
    } else if (stride == 64) {
        meanpooling_64_kernel<<<right-left, 1024, 0, stream.stream>>>(left, dim, compressed, k_cache);
    } else {
        throw std::runtime_error("Unsupported meanpooling stride: " + std::to_string(stride));
    }
}

template <typename T>
void maxpooling_func(
    cudaStream_t stream,
    const T* input,
    T* output,
    int num_heads,
    int q_len,
    int q_round,
    int k_len,
    int cache_len,
    int init_blocks,
    int local_blocks,
    int &out_len,
    int kernel_size=5,
    int stride=4,
    int padding=1,
    int block_size=64
) {
    out_len = (cache_len + block_size - 1) / block_size;
    maxpooling_kernel<<<dim3(q_len, num_heads), 256, 0, stream>>>(
        input, output, num_heads, q_len, q_round, k_len, out_len, cache_len, init_blocks, local_blocks, kernel_size, stride, padding, block_size
    );
} 

void topk_to_uint64_func( // TODO not necessary now, since topk is small
    cudaStream_t stream,
    const int* topk_idx,          // Input topk indices
    uint64_t* result,             // Output uint64 array
    int batch_size,               // num_heads x q_len
    int topk,                     // Number of topk values per row
    int k_len,                    // k_len
    int block_size = 64
) {
    int k_blocks = (k_len + block_size - 1) / block_size;
    int n_uint64_per_row = (k_blocks + block_size - 1) / block_size;

    const int threads_per_block = 256;
    const int blocks_per_row = (batch_size + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_per_row, n_uint64_per_row);
    dim3 block(threads_per_block, 1);
    
    kernel_topk_to_uint64<<<grid, block, 0, stream>>>(
        topk_idx, result, batch_size, topk, k_blocks, n_uint64_per_row
    );
} 
}

template <typename T>
struct MiniCPM4KVCache : KVCache<T> {
    uint64_t *blockmask;
    T *stage1_score, *pool_score;
    functions::TopK<T> *topk_func;
    T *c1_cache, *c2_cache;
    int c1_stride, c2_stride;
    int c1_len, c2_len;
    int prev_kv_length;
    int next_kv_length;
    bool apply_compress_lse;

    MiniCPM4KVCache(int dim, RotaryEmbedding<T> *rotary_embedding, uint64_t *blockmask, T* stage1_score, T* pool_score, functions::TopK<T> *topk_func, bool apply_compress_lse) : KVCache<T>(dim, rotary_embedding) {
        this->blockmask = blockmask;
        this->stage1_score = stage1_score;
        this->pool_score = pool_score;
        this->topk_func = topk_func;
        c1_stride = 16;
        c2_stride = 64;
        assert(this->dim % 32 == 0);
        this->apply_compress_lse = apply_compress_lse;
    }

    void init() {
        this->prev_kv_length = 0;
        this->next_kv_length = 0;
        this->c1_len = 0;
        this->c2_len = 0;
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int32_t num_c1, int32_t num_c2, int64_t offset) {
        offset = KVCache<T>::init_output_ptr(memory, num_tokens, offset);
        offset = memory->allocate((void**)&this->c1_cache, offset, num_c1 * this->dim * sizeof(T));
        if (apply_compress_lse) {
            offset = memory->allocate((void**)&this->c2_cache, offset, num_c2 * this->dim * sizeof(T));
        }
        return offset;
    }

    void compress(const Stream& stream) {
        int prev_pos;
        prev_pos = c1_len;
        c1_len = max((this->next_kv_length - c1_stride) / c1_stride, 0);
        meanpooling(stream, prev_pos, c1_len, this->dim, this->c1_cache, this->k_cache, c1_stride);
        if (apply_compress_lse) {
            prev_pos = c2_len;
            c2_len = max((this->next_kv_length - c2_stride) / c2_stride, 0);
            meanpooling(stream, prev_pos, c2_len, this->dim, this->c2_cache, this->k_cache, c2_stride);
        }
        this->prev_kv_length = this->next_kv_length;
    }
};

template <typename T>
struct MiniCPM4KVCacheManager {
    int num_hidden_layers;
    int dim;
    int budget;
    int budget_c1, budget_c2;
    std::vector<MiniCPM4KVCache<T>*> caches;
    T **h_flat_caches, **d_flat_caches;
    RotaryEmbedding<T> *rotary_embedding;
    uint64_t *blockmask;
    T* stage1_score, *pool_score;
    functions::TopK<T> *topk_func;
    bool apply_compress_lse;

    MiniCPM4KVCacheManager(int num_hidden_layers, int num_key_value_heads, int head_dim, int sparse_topk_k, bool apply_compress_lse) {
        this->num_hidden_layers = num_hidden_layers;
        this->dim = num_key_value_heads * head_dim;
        this->rotary_embedding = new RotaryEmbedding<T>(head_dim);
        this->topk_func = new functions::TopK<T>(4096, sparse_topk_k); // 256k/64
        this->apply_compress_lse = apply_compress_lse;
    }

    void init_weight_ptr(Memory* memory) {
        this->rotary_embedding->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int num_tokens, int64_t offset, float ratio=1.0) {
        // 2 = num_heads
        offset = memory->allocate((void**)&this->blockmask, offset, 2 * num_tokens * 64 * sizeof(uint64_t)); // 256k/64/64
        offset = memory->allocate((void**)&this->stage1_score, offset, 2 * num_tokens * 16384 * sizeof(T)); // 256k/16
        offset = memory->allocate((void**)&this->pool_score, offset, 2 * num_tokens * 4096 * sizeof(T)); // 256k/64
        offset = topk_func->init_output_ptr(memory, 2 * num_tokens, offset);

        offset = memory->allocate((void**)&this->d_flat_caches, offset, num_hidden_layers * 2 * sizeof(T*));

        budget = int64_t(memory->get_remaining_memory(offset) * ratio * 0.999) / (this->num_hidden_layers * 2 * this->dim * sizeof(T)) - 1;
        for (int i = 0; i < this->num_hidden_layers; i++) {
            caches.push_back(new MiniCPM4KVCache<T>(this->dim, this->rotary_embedding, this->blockmask, stage1_score, pool_score, topk_func, apply_compress_lse));
        }
        budget_c2 = (int)(budget / 69.0); // 1 + 4 + 64
        budget_c1 = budget_c2 * 4;
        budget = budget_c1 * 16;
        for (int i = 0; i < this->num_hidden_layers; i++) {
            offset = caches[i]->init_output_ptr(memory, budget, budget_c1, budget_c2, offset);
        }
        this->h_flat_caches = new T*[num_hidden_layers * 2];
        for (int i = 0; i < num_hidden_layers; i++) {
            this->h_flat_caches[i * 2] = caches[i]->k_cache;
            this->h_flat_caches[i * 2 + 1] = caches[i]->v_cache;
        }
        cudaMemcpy(this->d_flat_caches, this->h_flat_caches, num_hidden_layers * 2 * sizeof(T*), cudaMemcpyHostToDevice);
        return offset;
    }

    void add_length(int length) {
        for (int i = 0; i < this->num_hidden_layers; i++) {
            caches[i]->next_kv_length += length;
        }
    }
};
