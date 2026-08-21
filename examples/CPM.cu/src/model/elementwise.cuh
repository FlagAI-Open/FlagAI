#pragma once
#include <cuda_runtime.h>
#include "../trait.cuh"
#include "../utils.cuh"

namespace {
template <typename T2>
__global__ void batched_add_kernel(int dim, const T2* a, const T2* b, T2* c) {
    int row = blockIdx.x * dim;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col < dim) {
        c[row + col] = a[row + col] + b[col];
    }
}

template <typename T2>
__global__ void elementwise_add_kernel(int dim, const T2* a, const T2* b, T2* c) {
    int row = blockIdx.x * dim;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col < dim) {
        c[row + col] = a[row + col] + b[row + col];
    }
}

template <typename T2>
__global__ void elementwise_add3_kernel(int dim, const T2* a, const T2* b, const T2*c,  T2* d) {
    int row = blockIdx.x * dim;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col < dim) {
        d[row + col] = a[row + col] + b[row + col] + c[row+col];
    }
}

template <typename T, typename T2>
__global__ void elementwise_scale_kernel(int dim, const T2* a, float v, T2* b) {
    int row = blockIdx.x * dim;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col < dim) {
        b[row + col] = a[row + col] * T2(T(v), T(v));
    }
}

template <typename T>
__global__ void batched_mul_kernel(int dim, const T* a, const T* b, T* c) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    T bv = b[row];
    for (int i = col; i < dim; i += blockDim.x) {
        c[row * dim + i] = a[row * dim + i] * bv;
    }
}
} // namespace

template <typename T>
void batched_add(const Stream& stream, int num_tokens, int dim, const T* a, const T* b, T* c) {
    using T2 = typename TypeTraits<T>::half2;
    dim = dim / 2;
    batched_add_kernel<T2><<<dim3(num_tokens, CEIL_DIV(dim, 512)), 512, 0, stream.stream>>>(dim, (T2*)a, (T2*)b, (T2*)c);
}

template <typename T>
void elementwise_add(const Stream& stream, int num_tokens, int dim, const T* a, const T* b, T* c) {
    using T2 = typename TypeTraits<T>::half2;
    dim = dim / 2;
    elementwise_add_kernel<T2><<<dim3(num_tokens, CEIL_DIV(dim, 512)), 512, 0, stream.stream>>>(dim, (T2*)a, (T2*)b, (T2*)c);
}

template <typename T>
void elementwise_add3(const Stream& stream, int num_tokens, int dim, const T* a, const T* b, const T* c, T* d) {
    using T2 = typename TypeTraits<T>::half2;
    dim = dim / 2;
    elementwise_add3_kernel<T2><<<dim3(num_tokens, CEIL_DIV(dim, 512)), 512, 0, stream.stream>>>(dim, (T2*)a, (T2*)b, (T2*)c, (T2*)d);
}

template <typename T>
void elementwise_scale(const Stream& stream, int num_tokens, int dim, T* a, float v, T* b = nullptr) {
    if (v == 1.0 && b == nullptr) return;
    if (b == nullptr) b = a;
    using T2 = typename TypeTraits<T>::half2;
    dim = dim / 2;
    elementwise_scale_kernel<T, T2><<<dim3(num_tokens, CEIL_DIV(dim, 512)), 512, 0, stream.stream>>>(dim, (T2*)a, v, (T2*)b);
}

template <typename T>
void batched_mul(const Stream& stream, int num_tokens, int dim, const T* a, const T* b, T* c) {
    batched_mul_kernel<<<num_tokens, 128, 0, stream.stream>>>(dim, (T*)a, (T*)b, (T*)c);
}