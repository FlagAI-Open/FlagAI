#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

#define cudaCheck(err) \
  if (err != cudaSuccess) { \
    std::cerr << "cuda error at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::cerr << cudaGetErrorString(err) << std::endl; \
    exit(EXIT_FAILURE); \
  }

#define cublasCheck(err) \
  if (err != CUBLAS_STATUS_SUCCESS) { \
    std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::cerr << err << std::endl; \
    exit(EXIT_FAILURE); \
  }

void init_resources();
