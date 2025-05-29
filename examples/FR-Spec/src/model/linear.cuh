#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../trait.cuh"
#include "../utils.cuh"
#include "elementwise.cuh"

template <typename T, bool transposed=true>
void linear(const Stream& stream, int num_tokens, int dim_in, int dim_out, const T* input, const T* weight, T* output, bool inplace=false) {
    float alpha = 1.0f;
    float beta = inplace ? 1.0f : 0.0f;
    if constexpr (transposed) {
        cublasCheck(cublasGemmEx(stream.cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_out, num_tokens, dim_in,
            &alpha,
            weight, TypeTraits<T>::cublas_type(), dim_in,
            input, TypeTraits<T>::cublas_type(), dim_in,
            &beta,
            output, TypeTraits<T>::cublas_type(), dim_out,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ));
    } else {
        cublasCheck(cublasGemmEx(stream.cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_out, num_tokens, dim_in,
            &alpha,
            weight, TypeTraits<T>::cublas_type(), dim_out,
            input, TypeTraits<T>::cublas_type(), dim_in,
            &beta,
            output, TypeTraits<T>::cublas_type(), dim_out,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ));
    }
}

template <typename T, bool transposed=true, bool has_bias=false>
struct Linear {
    int dim_in;
    int dim_out;
    T* output;
    T* weight;
    T* bias;

    Linear(int dim_in, int dim_out) {
        this->dim_in = dim_in;
        this->dim_out = dim_out;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(dim_in * dim_out * sizeof(T));
        if constexpr (has_bias) {
            bias = (T*)memory->allocate_for_model(dim_out * sizeof(T));
        }
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * dim_out * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("weight") != std::string::npos) {
            cudaMemcpy((void*)weight, ptr, dim_in * dim_out * sizeof(T), cudaMemcpyHostToDevice);
        } else if (name.find("bias") != std::string::npos) {
            cudaMemcpy((void*)bias, ptr, dim_out * sizeof(T), cudaMemcpyHostToDevice);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* tgt=nullptr, bool inplace=false) {
        if (tgt == nullptr) tgt = this->output;
        linear<T, transposed>(stream, num_tokens, dim_in, dim_out, input, weight, tgt, inplace);
        if constexpr (has_bias) {
            batched_add<T>(stream, num_tokens, dim_out, tgt, bias, tgt);
        }
    }
};