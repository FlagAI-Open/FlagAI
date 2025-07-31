#pragma once
#include <cuda_runtime.h>
#include "../trait.cuh"
#include "../utils.cuh"

namespace {
template <typename T, typename T2>
__global__ void rms_norm_kernel(int dim, const T2* input, const T2* weight, T2* output, float eps) {
    // __shared__ T2 s_input[2048];
    __shared__ float shared_sum;
    __shared__ float warp_sum[16];
    int row = blockIdx.x;
    int col = threadIdx.x;
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 val = input[row * dim + i];
        // s_input[i] = val;
        float val1 = float(val.x);
        float val2 = float(val.y);
        sum1 += val1 * val1;
        sum2 += val2 * val2;
    }
    float sum = sum1 + sum2;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (col % 32 == 0) warp_sum[col / 32] = sum;
    __syncthreads();
    if (col < 16) {
        sum = warp_sum[col];
        sum += __shfl_down_sync(0x0000ffff, sum, 8);
        sum += __shfl_down_sync(0x0000ffff, sum, 4);
        sum += __shfl_down_sync(0x0000ffff, sum, 2);
        sum += __shfl_down_sync(0x0000ffff, sum, 1);
    }
    if (col == 0) {
        shared_sum = rsqrtf(sum / (2*dim) + eps);
    }
    __syncthreads();
    sum = shared_sum;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 inp = input[row * dim + i];
        T2 w = weight[i];
        output[row * dim + i] = T2(
            T(sum * float(inp.x) * float(w.x)),
            T(sum * float(inp.y) * float(w.y))
        );
    }
}

template <typename T, typename T2>
__global__ void add_and_rms_norm_kernel(int dim, T2* input, const T2* prev_output, const T2* weight, T2* output, float eps) {
    // __shared__ T2 s_input[2048];
    __shared__ float shared_sum;
    __shared__ float warp_sum[16];
    int row = blockIdx.x;
    int col = threadIdx.x;
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 val = input[row * dim + i];
        T2 prev = prev_output[row * dim + i];
        val = val + prev;
        input[row * dim + i] = val;
        float val1 = float(val.x);
        float val2 = float(val.y);
        sum1 += val1 * val1;
        sum2 += val2 * val2;
    }
    float sum = sum1 + sum2;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (col % 32 == 0) warp_sum[col / 32] = sum;
    __syncthreads();
    if (col < 16) {
        sum = warp_sum[col];
        sum += __shfl_down_sync(0x0000ffff, sum, 8);
        sum += __shfl_down_sync(0x0000ffff, sum, 4);
        sum += __shfl_down_sync(0x0000ffff, sum, 2);
        sum += __shfl_down_sync(0x0000ffff, sum, 1);
    }
    if (col == 0) {
        shared_sum = rsqrtf(sum / (2*dim) + eps);
    }
    __syncthreads();
    sum = shared_sum;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 inp = input[row * dim + i];
        T2 w = weight[i];
        output[row * dim + i] = T2(
            T(sum * float(inp.x) * float(w.x)),
            T(sum * float(inp.y) * float(w.y))
        );
    }
}

template <typename T>
void rms_norm(const Stream& stream, int num_tokens, int dim, const T* input, const T* weight, T* output, float eps) {
    using T2 = typename TypeTraits<T>::half2;
    rms_norm_kernel<T, T2><<<num_tokens, 512, 0, stream.stream>>>(dim/2, (T2*)input, (T2*)weight, (T2*)output, eps);
}

template <typename T>
void add_and_rms_norm(const Stream& stream, int num_tokens, int dim, T* input, const T* prev_output, const T* weight, T* output, float eps) {
    using T2 = typename TypeTraits<T>::half2;
    add_and_rms_norm_kernel<T, T2><<<num_tokens, 512, 0, stream.stream>>>(dim/2, (T2*)input, (T2*)prev_output, (T2*)weight, (T2*)output, eps);
}
}

template <typename T>
struct Norm {
    T* output;
    virtual void init_weight_ptr(Memory* memory) = 0;
    virtual int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) = 0;
    virtual void load_to_storage(std::string name, void* ptr) = 0;
    virtual void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output, T* tgt=nullptr) = 0;
};

template <typename T>
struct RMSNorm : Norm<T> {
    int dim;
    float eps;
    T* weight;

    RMSNorm(int dim, float eps) {
        this->dim = dim;
        this->eps = eps;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(dim * sizeof(T));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * dim * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {
        cudaMemcpy((void*)weight, ptr, dim * sizeof(T), cudaMemcpyHostToDevice);
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output, T* tgt=nullptr) {
        if (tgt == nullptr) tgt = this->output;
        if (prev_output == nullptr) {
            rms_norm(stream, num_tokens, this->dim, input, this->weight, tgt, this->eps);
        } else {
            add_and_rms_norm(stream, num_tokens, this->dim, input, prev_output, this->weight, tgt, this->eps);
        }
    }
};