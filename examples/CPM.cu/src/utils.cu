#include "perf.cuh"
#include "utils.cuh"
#include "signal_handler.cuh"

bool initialized = false;

Stream calc_stream;

int graphCreated_padding_length = -1;
int graphCreated_input_length = -1;
cudaGraph_t graph;
cudaGraphExec_t graphExec;

void init_resources() {
  if (initialized) return;
  
  // 初始化signal处理器
  init_signal_handlers();
  perf_init();
 
  cudaCheck(cudaStreamCreate(&calc_stream.stream));
  cublasCheck(cublasCreate(&calc_stream.cublas_handle));
  cublasCheck(cublasSetStream(calc_stream.cublas_handle, calc_stream.stream));
  initialized = true;
}
