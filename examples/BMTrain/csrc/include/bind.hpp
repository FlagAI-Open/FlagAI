#include <pybind11/pybind11.h>
#include "nccl.hpp"
#include "adam_cpu.hpp"

int is_bf16_supported();

void has_nan_inf_fp16_launcher(int32_t n, std::uintptr_t g_fp16, std::uintptr_t mid, std::uintptr_t out, std::uintptr_t stream);
void has_nan_inf_bf16_launcher(int32_t n, std::uintptr_t g_bf16, std::uintptr_t mid, std::uintptr_t out, std::uintptr_t stream);

void fp16_from_fp32_value_launcher(
    int64_t n,
    std::uintptr_t param_fp32,
    std::uintptr_t param_fp16
);
void bf16_from_fp32_value_launcher(
    int64_t n,
    std::uintptr_t param_fp32,
    std::uintptr_t param_bf16
);
void cross_entropy_forward_fp16_launcher(
    int32_t m, int32_t n,
    std::uintptr_t input,
    std::uintptr_t target,
    std::uintptr_t softmax,
    std::uintptr_t output,
    int32_t ignore_index,
    std::uintptr_t stream
);
void cross_entropy_backward_inplace_fp16_launcher(
    int32_t m, int32_t n,
    std::uintptr_t grad_output,
    std::uintptr_t target,
    std::uintptr_t x,
    int32_t ignore_index,
    std::uintptr_t stream
);
void cross_entropy_forward_bf16_launcher(
    int32_t m, int32_t n,
    std::uintptr_t input,
    std::uintptr_t target,
    std::uintptr_t softmax,
    std::uintptr_t output,
    int32_t ignore_index,
    std::uintptr_t stream
);
void cross_entropy_backward_inplace_bf16_launcher(
    int32_t m, int32_t n,
    std::uintptr_t grad_output,
    std::uintptr_t target,
    std::uintptr_t x,
    int32_t ignore_index,
    std::uintptr_t stream
);
void fused_sumexp_fp16_launcher(
    int32_t m, int32_t n,
    std::uintptr_t logits,
    std::uintptr_t max_logits,
    std::uintptr_t sum_exp_logits,
    std::uintptr_t stream
);
void fused_sumexp_bf16_launcher(
    int32_t m, int32_t n,
    std::uintptr_t logits,
    std::uintptr_t max_logits,
    std::uintptr_t sum_exp_logits,
    std::uintptr_t stream
);
void fused_softmax_inplace_fp16_launcher(
    int32_t m, int32_t n,
    std::uintptr_t logits,
    std::uintptr_t max_logits,
    std::uintptr_t sum_exp_logits,
    std::uintptr_t stream
);
void fused_softmax_inplace_bf16_launcher(
    int32_t m, int32_t n,
    std::uintptr_t logits,
    std::uintptr_t max_logits,
    std::uintptr_t sum_exp_logits,
    std::uintptr_t stream
);
void adam_fp16_launcher(
    int n,
    std::uintptr_t param_fp32,
    std::uintptr_t param_fp16,
    std::uintptr_t g_fp16,
    std::uintptr_t m_fp16,
    std::uintptr_t v_fp32,
    float beta1, float beta2,
    float eps, float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    uintptr_t stream
);
void adam_bf16_launcher(
    int n,
    std::uintptr_t param_fp32,
    std::uintptr_t param_bf16,
    std::uintptr_t g_bf16,
    std::uintptr_t m_fp32,
    std::uintptr_t v_fp32,
    float beta1, float beta2,
    float eps, float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    uintptr_t stream
);
