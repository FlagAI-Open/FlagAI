#pragma once
#include "../trait.cuh"
#include <cuda_runtime.h>

namespace {
template <typename T>
__global__ void gated_silu_interleaved_kernel(int dim, const T* src, T* tgt) {
    int row_offset = blockIdx.x * dim;
    int row_offset_2 = row_offset * 2;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    int col2 = col + dim;
    if (col < dim) {
        float g = float(src[row_offset_2 + col]);
        float u = float(src[row_offset_2 + col2]);
        float s = 1.0f / (1.0f + expf(-g));
        tgt[row_offset + col] = T(g * s * u);
    }
}

template<typename T>
__global__ void gated_silu_kernel(int dim, const T* src, T* tgt) {
    int row_offset = blockIdx.x * dim;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col < dim) {
        float g = float(src[row_offset + col]);
        float u = float(tgt[row_offset + col]);
        float s = 1.0f / (1.0f + expf(-g));
        tgt[row_offset + col] = T(g * s * u);
    }
}

template<typename T>
__global__ void silu_kernel(int dim, const T* src, T* tgt) {
    int row_offset = blockIdx.x * dim;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col < dim) {
        float g = float(src[row_offset + col]);
        float s = 1.0f / (1.0f + expf(-g));
        tgt[row_offset + col] = T(g * s);
    }
}

template<typename T>
__global__ void relu_kernel(int dim, const T* src, T* tgt) {
    int row_offset = blockIdx.x * dim;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col < dim) {
        T v = src[row_offset + col];
        tgt[row_offset + col] = v > T(0) ? v : T(0);
    }
}
}

template <typename T>
void gated_silu_interleaved(const Stream& stream, int num_tokens, int dim, const T* src, T* tgt) {
    gated_silu_interleaved_kernel<T><<<dim3(num_tokens, CEIL_DIV(dim, 256)), 256, 0, stream.stream>>>(dim, src, tgt);
}

template <typename T>
void gated_silu(const Stream& stream, int num_tokens, int dim, const T* src, T* tgt) {
    gated_silu_kernel<T><<<dim3(num_tokens, CEIL_DIV(dim, 256)), 256, 0, stream.stream>>>(dim, src, tgt);
}

template<typename T>
void silu(const Stream& stream, int num_tokens, int dim, const T* src, T* tgt) {
    silu_kernel<T><<<dim3(num_tokens, CEIL_DIV(dim, 256)), 256, 0, stream.stream>>>(dim, src, tgt);
}

template <typename T>
void silu_inplace(const Stream& stream, int num_tokens, int dim, T* x) {
    silu(stream, num_tokens, dim, x, x);
}


template<typename T>
void relu(const Stream& stream, int num_tokens, int dim, const T* src, T* tgt) {
    relu_kernel<T><<<dim3(num_tokens, CEIL_DIV(dim, 256)), 256, 0, stream.stream>>>(dim, src, tgt);
}

template <typename T>
void relu_inplace(const Stream& stream, int num_tokens, int dim, T* x) {
    relu(stream, num_tokens, dim, x, x);
}