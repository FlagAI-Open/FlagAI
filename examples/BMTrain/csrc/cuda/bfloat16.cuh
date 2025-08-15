#include <cuda.h>
#if defined(__CUDACC__) && CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#define BF16_SUPPORT
#endif