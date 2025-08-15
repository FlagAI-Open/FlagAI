#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<__half> {
    using half2 = __half2;

    static __inline__ cudaDataType_t cublas_type() {
        return CUDA_R_16F;
    }

    static __inline__ int type_code() {
        return 0;
    }

    static __host__ __device__ __inline__ constexpr __half inf() { const short v = 0x7c00; return *(reinterpret_cast<const __half *>(&(v))); }
};

template <>
struct TypeTraits<__nv_bfloat16> {
    using half2 = __nv_bfloat162;

    static __inline__ cudaDataType_t cublas_type() {
        return CUDA_R_16BF;
    }

    static __inline__ int type_code() {
        return 1;
    }

    static __host__ __device__ __inline__ constexpr __nv_bfloat16 inf() { const short v = 0x7f80; return *(reinterpret_cast<const __nv_bfloat16 *>(&(v))); }
};
