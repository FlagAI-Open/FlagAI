#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {
// blocks <n // 1024>,      threads<min(n, 1024)>
__global__ void adam_fp32_accum(
    int32_t n,
    const half *g,        // (n)
    half *m,        // (n)
    float *v,        // (n)
    float* param,   // (n)
    half* param_h,  // (n)
    float beta1,
    float beta2,
    float eps,
    float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        float local_g = __half2float(g[col]);                                       // real_g * scale
        float local_m = beta1 * __half2float(m[col]) + (1 - beta1) * local_g;       // real_m * scale
        float local_v = beta2 * v[col] + (1 - beta2) * local_g * local_g / scale;   // real_v * scale
        float local_p = param[col];
        local_p = local_p - lr * local_m / bias_correction1 / (sqrtf(local_v * scale / bias_correction2) + eps * scale) - lr * weight_decay * local_p;

        param_h[col] = __float2half(local_p);
        param[col] = local_p;
        v[col] = local_v;
        m[col] = __float2half(local_m);
    }
}
}

void adam_launcher(
    const torch::Tensor &param_fp32,
    const torch::Tensor &param_fp16,
    const torch::Tensor &g_fp16,
    const torch::Tensor &m_fp16,
    const torch::Tensor &v_fp32,
    float beta1, float beta2,
    float eps, float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    int32_t n = param_fp32.numel();
    if (n <= 0) return;
    auto g_ptr = reinterpret_cast<half*>(g_fp16.data_ptr<at::Half>());
    auto m_ptr = reinterpret_cast<half*>(m_fp16.data_ptr<at::Half>());
    auto v_ptr = v_fp32.data_ptr<float>();
    auto param_ptr = param_fp32.data_ptr<float>();
    auto param_h_ptr = reinterpret_cast<half*>(param_fp16.data_ptr<at::Half>());
    int32_t threads = 1024;
    dim3 block_size = dim3(threads, 1, 1);
    dim3 grid_size = dim3((n + threads - 1) / threads, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();
    adam_fp32_accum<<<grid_size, block_size, 0, stream.stream()>>>(n, g_ptr, m_ptr, v_ptr, param_ptr, param_h_ptr, beta1, beta2, eps, lr, scale, weight_decay, bias_correction1, bias_correction2);
}