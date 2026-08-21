#pragma once
#include <cuda_runtime.h>
#include "../utils.cuh"

namespace {
template <typename T>
__global__ void embedding_kernel(int32_t num_cols, const int32_t* input, const float4* weight, float4* output) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    int offset_output = row * num_cols;
    int offset_weight = input[row] * num_cols;
    for (int i = col; i < num_cols; i += blockDim.x) {
        output[offset_output + i] = weight[offset_weight + i];
    }
}

template <typename T>
void embedding(const Stream& stream, int32_t num_tokens, int32_t hidden_size, const int32_t* input, const T* weight, T* output) {
    embedding_kernel<T><<<num_tokens, 256, 0, stream.stream>>>(hidden_size/(16/sizeof(T)), input, (float4*)weight, (float4*)output);
}
} // namespace

template <typename T>
struct Embedding {
    int vocab_size;
    int hidden_size;
    T* weight;
    T* output;

    Embedding(int vocab_size, int hidden_size) {
        this->vocab_size = vocab_size;
        this->hidden_size = hidden_size;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(vocab_size * hidden_size * sizeof(T));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * hidden_size * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {
        cudaMemcpy((void*)weight, ptr, vocab_size * hidden_size * sizeof(T), cudaMemcpyHostToDevice);
    }

    void prefill(const Stream& stream, int32_t num_tokens, int32_t* input) {
        embedding(stream, num_tokens, this->hidden_size, input, this->weight, this->output);
    }
};