#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <immintrin.h>
#include <emmintrin.h>

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

#if defined(__AVX512F__)

#pragma message "Using AVX512"
#define __AVX512__ 1

#elif defined(__AVX__) and defined(__FMA__) and defined(__F16C__)

#pragma message "Using AVX256"
#define __AVX256__ 1

#endif

void adam_cpu_launcher(
    int n,
    float* param_fp32,
    at::Half* param_fp16,
    at::Half* g_fp16,
    float* m_fp32,
    float* v_fp32,
    float beta1, float beta2, 
    float eps, float lr, 
    float scale, 
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
#if defined(__AVX512__)
    auto avx_beta1 = _mm512_set1_ps(beta1);
    auto avx_beta2 = _mm512_set1_ps(beta2);
    auto avx_beta1_1 = _mm512_set1_ps(1 - beta1);
    auto avx_beta2_1 = _mm512_set1_ps(1 - beta2);
    auto avx_eps = _mm512_set1_ps(eps);
    auto avx_neg_lr = _mm512_set1_ps(-lr);
    auto avx_scale = _mm512_set1_ps(scale);
    auto avx_weight_decay = _mm512_set1_ps(weight_decay);
    auto avx_bias_correction1 = _mm512_set1_ps(bias_correction1);
    auto avx_bias_correction2 = _mm512_set1_ps(bias_correction2);
    int64_t span = 16;
#elif defined(__AVX256__)
    auto avx_beta1 = _mm256_set1_ps(beta1);
    auto avx_beta2 = _mm256_set1_ps(beta2);
    auto avx_beta1_1 = _mm256_set1_ps(1 - beta1);
    auto avx_beta2_1 = _mm256_set1_ps(1 - beta2);
    auto avx_eps = _mm256_set1_ps(eps);
    auto avx_neg_lr = _mm256_set1_ps(-lr);
    auto avx_scale = _mm256_set1_ps(scale);
    auto avx_weight_decay = _mm256_set1_ps(weight_decay);
    auto avx_bias_correction1 = _mm256_set1_ps(bias_correction1);
    auto avx_bias_correction2 = _mm256_set1_ps(bias_correction2);
    int64_t span = 8;
#else
    int64_t span = 1;
#endif

    at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
        for (int64_t j = start; j < end; j += span) {
#if defined(__AVX256__) or defined(__AVX512__)
            if (j + span > end) {
#else
            if (true) {
#endif
                // No AVX or n is not alinged
                for (int64_t i = j; i < end; i++) {
                    float g = c10::detail::fp16_ieee_to_fp32_value(g_fp16[i].x) / scale;
                    float m = m_fp32[i];
                    float v = v_fp32[i];
                    float p = param_fp32[i];
                    m = beta1 * m + (1 - beta1) * g;
                    v = beta2 * v + (1 - beta2) * g * g;
                    p = p - lr * m  / bias_correction1 / (sqrtf(v / bias_correction2) + eps) - lr * weight_decay * p;
                    param_fp32[i] = p;
                    param_fp16[i] = at::Half(p);
                    m_fp32[i] = m;
                    v_fp32[i] = v;
                }
                break; // must break here
            } else {
                // use AVX here
#if defined(__AVX512__)
                auto g = _mm512_div_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)&g_fp16[j])), avx_scale);
                auto m = _mm512_loadu_ps(&m_fp32[j]);
                auto v = _mm512_loadu_ps(&v_fp32[j]);
                auto p = _mm512_loadu_ps(&param_fp32[j]);
                m = _mm512_fmadd_ps(avx_beta1, m, _mm512_mul_ps(avx_beta1_1, g));
                v = _mm512_fmadd_ps(avx_beta2, v, _mm512_mul_ps(avx_beta2_1, _mm512_mul_ps(g, g)));
                p = _mm512_fmadd_ps(avx_neg_lr, _mm512_mul_ps(avx_weight_decay, p), p); // p = p - lr * weight_decay * p
                p = _mm512_fmadd_ps(
                    avx_neg_lr,
                    _mm512_div_ps(
                        _mm512_div_ps(m, avx_bias_correction1), // m / bias_correction1
                        _mm512_add_ps(
                            _mm512_sqrt_ps(_mm512_div_ps(v, avx_bias_correction2)),
                            avx_eps
                        )   // sqrt(v / bias_correction2) + eps
                    ),
                    p
                );  // p = p - lr * m / bias_correction1 / (sqrtf(v / bias_correction2) + eps)
                _mm512_storeu_ps(&param_fp32[j], p);
                _mm256_storeu_si256((__m256i*)&param_fp16[j], _mm512_cvtps_ph(p, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                _mm512_storeu_ps(&m_fp32[j], m);
                _mm512_storeu_ps(&v_fp32[j], v);
#elif defined(__AVX256__)
                auto g = _mm256_div_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)&g_fp16[j])), avx_scale);
                auto m = _mm256_loadu_ps(&m_fp32[j]);
                auto v = _mm256_loadu_ps(&v_fp32[j]);
                auto p = _mm256_loadu_ps(&param_fp32[j]);
                m = _mm256_fmadd_ps(avx_beta1, m, _mm256_mul_ps(avx_beta1_1, g));
                v = _mm256_fmadd_ps(avx_beta2, v, _mm256_mul_ps(avx_beta2_1, _mm256_mul_ps(g, g)));
                p = _mm256_fmadd_ps(avx_neg_lr, _mm256_mul_ps(avx_weight_decay, p), p); // p = p - lr * weight_decay * p
                p = _mm256_fmadd_ps(
                    avx_neg_lr,
                    _mm256_div_ps(
                        _mm256_div_ps(m, avx_bias_correction1), // m / bias_correction1
                        _mm256_add_ps(_mm256_sqrt_ps(_mm256_div_ps(v, avx_bias_correction2)), avx_eps)  // sqrt(v / bias_correction2) + eps
                    ),  // m / bias_correction1 / (sqrt(v / bias_correction2) + eps)
                    p
                );  // p = p - lr * m / bias_correction1 / (sqrt(v / bias_correction2) + eps)
                _mm256_storeu_ps(&param_fp32[j], p);
                _mm_storeu_si128((__m128i*)&param_fp16[j], _mm256_cvtps_ph(p, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                _mm256_storeu_ps(&m_fp32[j], m);
                _mm256_storeu_ps(&v_fp32[j], v);
#endif
            }
        }

    });
}

void F_adam_cpu(
    const torch::Tensor &param_fp32, 
    const torch::Tensor &param_fp16, 
    const torch::Tensor &g_fp16, 
    const torch::Tensor &m_fp32, 
    const torch::Tensor &v_fp32, 
    float beta1, float beta2, 
    float eps, float lr, 
    float scale, 
    float weight_decay,
    int64_t step
) {
    CHECK_CONTIGUOUS(param_fp32);
    CHECK_CONTIGUOUS(param_fp16);
    CHECK_CONTIGUOUS(g_fp16);
    CHECK_CONTIGUOUS(m_fp32);
    CHECK_CONTIGUOUS(v_fp32);
    AT_ASSERTM(param_fp32.dtype() == torch::kFloat, "param_fp32 must be a float tensor");
    AT_ASSERTM(param_fp16.dtype() == torch::kHalf, "param_fp16 must be a half tensor");
    AT_ASSERTM(g_fp16.dtype() == torch::kHalf, "g_fp16 must be a half tensor");
    AT_ASSERTM(m_fp32.dtype() == torch::kFloat, "m_fp32 must be a float tensor");
    AT_ASSERTM(v_fp32.dtype() == torch::kFloat, "v_fp32 must be a float tensor");
    AT_ASSERTM(param_fp32.is_cpu(), "param_fp32 must be a cpu tensor");
    AT_ASSERTM(param_fp16.is_cpu(), "param_fp16 must be a cpu tensor");
    AT_ASSERTM(g_fp16.is_cpu(), "g_fp16 must be a cpu tensor");
    AT_ASSERTM(m_fp32.is_cpu(), "m_fp32 must be a cpu tensor");
    AT_ASSERTM(v_fp32.is_cpu(), "v_fp32 must be a cpu tensor");
    AT_ASSERTM(param_fp32.numel() == param_fp16.numel(), "param_fp32 and param_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == g_fp16.numel(), "param_fp32 and g_fp16 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == m_fp32.numel(), "param_fp32 and m_fp32 must have the same number of elements");
    AT_ASSERTM(param_fp32.numel() == v_fp32.numel(), "param_fp32 and v_fp32 must have the same number of elements");

    float bias_correction1 = 1 - powf(beta1, step);
    float bias_correction2 = 1 - powf(beta2, step);

    adam_cpu_launcher(
        param_fp32.numel(),
        param_fp32.data_ptr<float>(),
        param_fp16.data_ptr<at::Half>(),
        g_fp16.data_ptr<at::Half>(),
        m_fp32.data_ptr<float>(),
        v_fp32.data_ptr<float>(),
        beta1, beta2, 
        eps, lr, 
        scale, 
        weight_decay,
        bias_correction1,
        bias_correction2
    );
} 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_adam_cpu", &F_adam_cpu, "adam function cpu");
}
