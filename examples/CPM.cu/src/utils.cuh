#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "signal_handler.cuh"

extern bool initialized;

struct Stream {
  cudaStream_t stream;
  cublasHandle_t cublas_handle;
};

extern Stream calc_stream;

extern int graphCreated_padding_length;
extern int graphCreated_input_length;
extern cudaGraph_t graph;
extern cudaGraphExec_t graphExec;

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define ROUND_UP(M, N) (((M) + (N) - 1) / (N) * (N))

inline const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "Unknown cuBLAS error";
    }
}

#define cudaCheck(err) \
  if (err != cudaSuccess) { \
    std::cerr << "cuda error at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::cerr << cudaGetErrorString(err) << std::endl; \
    print_stack_trace(); \
    exit(EXIT_FAILURE); \
  }

#define cublasCheck(err) \
  if (err != CUBLAS_STATUS_SUCCESS) { \
    std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::cerr << "Error code: " << err << " (" << cublasGetErrorString(err) << ")" << std::endl; \
    print_stack_trace(); \
    exit(EXIT_FAILURE); \
  }

void init_resources();
