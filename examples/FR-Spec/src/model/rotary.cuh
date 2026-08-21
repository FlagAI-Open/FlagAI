#pragma once
#include <cuda_runtime.h>
#include "../utils.cuh"

namespace {
template<typename T>
__global__ void rotary_embedding_kernel(int num_heads, int num_heads_kv, int half_dim, const float *inv_freq, const int* pos, T* q, T* k) {
    int tid = threadIdx.x;

    int p = pos[blockIdx.x];

    for (int i = tid; i < num_heads * half_dim; i += blockDim.x) {
        int row = i / half_dim;
        int col = i % half_dim;
        int offset = blockIdx.x * num_heads * half_dim * 2 + row * half_dim * 2;
        float freq = p * inv_freq[col];
        float cos_freq = cos(freq), sin_freq = sin(freq);
        float a = float(q[offset + col]);
        float b = float(q[offset + col + half_dim]);
        q[offset + col] = T(a * cos_freq - b * sin_freq);
        q[offset + col + half_dim] = T(a * sin_freq + b * cos_freq);
    }
    for (int i = tid; i < num_heads_kv * half_dim; i += blockDim.x) {
        int row = i / half_dim;
        int col = i % half_dim;
        int offset = blockIdx.x * num_heads_kv * half_dim * 2 + row * half_dim * 2;
        float freq = p * inv_freq[col];
        float cos_freq = cos(freq), sin_freq = sin(freq);
        float a = float(k[offset + col]);
        float b = float(k[offset + col + half_dim]);
        k[offset + col] = T(a * cos_freq - b * sin_freq);
        k[offset + col + half_dim] = T(a * sin_freq + b * cos_freq);
    }
}
}

template<typename T>
void rotary_embedding(const Stream& stream, int num_tokens, int num_heads, int num_heads_kv, int half_dim, const float *inv_freq, const int* pos, T* q, T* k) {
    rotary_embedding_kernel<T><<<num_tokens, 512, 0, stream.stream>>>(num_heads, num_heads_kv, half_dim, inv_freq, pos, q, k);
}

template <typename T>
struct RotaryEmbedding {
    int half_dim;

    float *inv_freq;
    // float attention_scaling;

    RotaryEmbedding(int head_dim) {
        this->half_dim = head_dim / 2;
    }

    void init_weight_ptr(Memory* memory) {
        this->inv_freq = (float*)memory->allocate_for_model(half_dim * sizeof(float));
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("inv_freq") != std::string::npos) {
            cudaMemcpy((void*)inv_freq, ptr, half_dim * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            throw std::runtime_error("Unsupported rotary embedding weight name: " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, int num_heads, int num_heads_kv, T* q, T* k, int32_t* position_ids) {
        rotary_embedding(stream, num_tokens, num_heads, num_heads_kv, this->half_dim, this->inv_freq, position_ids, q, k);
    }
};