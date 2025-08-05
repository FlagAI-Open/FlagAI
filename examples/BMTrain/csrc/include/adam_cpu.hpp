#include <emmintrin.h>
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <sched.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <mutex>
#include <vector>
#include <thread>
#include <algorithm>
#include "cpu_info.h"
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32 = {w};
    return fp32.as_value;
}

inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32 = {f};
    return fp32.as_bits;
}

template <class F>
inline void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const F& f) {
    // Number of iterations
    int64_t numiter = end - begin;

    // Number of threads to use
    int64_t num_threads = 1;  // Default to serial execution

    if (grain_size > 0) {
        num_threads = std::max((numiter+grain_size-1) / grain_size, static_cast<int64_t>(1));
    }
    else{
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
        num_threads = CPU_COUNT(&cpu_set);
        grain_size = std::max((numiter+num_threads-1) / num_threads, static_cast<int64_t>(1));

    }

    // Check if parallel execution is feasible
    if (num_threads > 1) {
        py::gil_scoped_release release;  // Release the GIL
        std::vector<std::thread> threads(num_threads);
        for (int64_t t = 0; t < num_threads; ++t) {
            threads[t] = std::thread([&, t]() {
                int64_t left = std::min(begin + t * grain_size, end);
                int64_t right = std::min(begin + (t + 1) * grain_size, end);
                f(left, right);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // If not feasible or grain_size is 0, perform the operation serially
        f(begin, end);
    }
}

// fp32 -> fp16
inline uint16_t fp16_ieee_from_fp32_value(float f) {
    // const float scale_to_inf = 0x1.0p+112f;
    // const float scale_to_zero = 0x1.0p-110f;
    uint32_t scale_to_inf_bits = (uint32_t) 239 << 23;
    uint32_t scale_to_zero_bits = (uint32_t) 17 << 23;
    float scale_to_inf_val, scale_to_zero_val;
    std::memcpy(&scale_to_inf_val, &scale_to_inf_bits, sizeof(scale_to_inf_val));
    std::memcpy(&scale_to_zero_val, &scale_to_zero_bits, sizeof(scale_to_zero_val));
    const float scale_to_inf = scale_to_inf_val;
    const float scale_to_zero = scale_to_zero_val;

    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = (uint32_t)fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = (uint32_t)fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return static_cast<uint16_t>(
        (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign)
    );
}

// fp16 -> fp32 
inline float fp16_ieee_to_fp32_value(uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;

  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
  const float exp_scale = 0x1.0p-112f;
  const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
        sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                            : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

inline uint16_t bf16_from_fp32_value(float f){
  return *reinterpret_cast<uint32_t*>(&f) >> 16;
} 
// fp32 -> bf16
void bf16_from_fp32_value_launcher(
    int64_t n,
    std::uintptr_t param_fp32,
    std::uintptr_t param_bf16
){
    int span = 1;
    auto param_fp32_ptr = reinterpret_cast<float*>(param_fp32);
    auto param_bf16_ptr = reinterpret_cast<uint16_t*>(param_bf16);
    parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
        for (int64_t j = start; j < end; j += span) {
            for (int64_t i = j; i < end; i++) {
                float p = param_fp32_ptr[i];
                param_bf16_ptr[i] = bf16_from_fp32_value(p);
            }
            break; // must break here
        }
    });
}

void fp16_from_fp32_value_launcher(
    int64_t n,
    std::uintptr_t param_fp32,
    std::uintptr_t param_fp16 
){
    int span = 1;
    auto param_fp32_ptr = reinterpret_cast<float*>(param_fp32);
    auto param_fp16_ptr = reinterpret_cast<uint16_t*>(param_fp16);
    parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
        for (int64_t j = start; j < end; j += span) {
            for (int64_t i = j; i < end; i++) {
                float p = param_fp32_ptr[i];
                param_fp16_ptr[i] = fp16_ieee_from_fp32_value(p);
            }
            break; // must break here
        }
    });
}
// bf16 -> fp32
inline float bf16_to_fp32_value(uint16_t h){
    uint32_t src = h;
    src <<= 16;
    return *reinterpret_cast<float*>(&src);
}

void adam_cpu_0(
    int64_t n,
    float* param_fp32_ptr,
    uint16_t* param_fp16_ptr,
    float* delta_info_ptr,
    uint16_t* g_fp16_ptr,
    float* m_fp32_ptr,
    float* v_fp32_ptr,
    float beta1, float beta2, 
    float eps, float lr, 
    float scale, 
    float weight_decay,
    float bias_correction1,
    float bias_correction2
){
    int64_t span = 1;
    float sum_sq_delta = 0;
    float sum_delta = 0;
    std::mutex delta_mutex;
    parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
        float sum_sq_delta_i = 0;
        float sum_delta_i = 0;
        for (int64_t j = start; j < end; j += span) {
            for (int64_t i = j; i < end; i++) {
                float g = fp16_ieee_to_fp32_value(g_fp16_ptr[i]) / scale;
                float m = m_fp32_ptr[i];
                float v = v_fp32_ptr[i];
                float p = param_fp32_ptr[i];
                m = beta1 * m + (1 - beta1) * g;
                v = beta2 * v + (1 - beta2) * g * g;
                if (delta_info_ptr != NULL){
                    float delta = m  / bias_correction1 / (sqrtf(v / bias_correction2) + eps) + weight_decay * p;
                    sum_delta_i += delta;
                    sum_sq_delta_i += delta * delta; 
                }
                p = p - lr * m  / bias_correction1 / (sqrtf(v / bias_correction2) + eps) - lr * weight_decay * p;
                param_fp32_ptr[i] = p;
                param_fp16_ptr[i] = fp16_ieee_from_fp32_value(p);
                m_fp32_ptr[i] = m;
                v_fp32_ptr[i] = v;
            }
            break; // must break here
        }
        if (delta_info_ptr != NULL){
            delta_mutex.lock();
            sum_delta += sum_delta_i;
            sum_sq_delta += sum_sq_delta_i;
            delta_mutex.unlock();
        }
    });
    if (delta_info_ptr != NULL){
        delta_info_ptr[0] = sum_delta / n;
        delta_info_ptr[1] = sum_sq_delta / n - sum_delta * sum_delta / (n * n);// var = E(x^2) - E(x)^2
        delta_info_ptr[2] = sum_delta;
        delta_info_ptr[3] = sum_sq_delta;
    }
}

void adam_cpu_bf16_0(
    int64_t n,
    float* param_fp32_ptr,
    uint16_t* param_bf16_ptr,
    float* delta_info_ptr,
    uint16_t* g_bf16_ptr,
    float* m_fp32_ptr,
    float* v_fp32_ptr,
    float beta1, float beta2,
    float eps, float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
){
    int64_t span = 1;
    float sum_sq_delta = 0;
    float sum_delta = 0;
    std::mutex delta_mutex;
    parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
        float sum_sq_delta_i = 0;
        float sum_delta_i = 0;
        for (int64_t j = start; j < end; j += span) {
            for (int64_t i = j; i < end; i++) {
                float g = bf16_to_fp32_value(g_bf16_ptr[i]) / scale;
                float m = m_fp32_ptr[i];
                float v = v_fp32_ptr[i];
                float p = param_fp32_ptr[i];
                m = beta1 * m + (1 - beta1) * g;
                v = beta2 * v + (1 - beta2) * g * g;
                if (delta_info_ptr != NULL){
                    float delta = m  / bias_correction1 / (sqrtf(v / bias_correction2) + eps) + weight_decay * p;
                    sum_delta_i += delta;
                    sum_sq_delta_i += delta * delta;
                }
                p = p - lr * m  / bias_correction1 / (sqrtf(v / bias_correction2) + eps) - lr * weight_decay * p;
                param_fp32_ptr[i] = p;
                param_bf16_ptr[i] = bf16_from_fp32_value(p);
                m_fp32_ptr[i] = m;
                v_fp32_ptr[i] = v;
            }
            break; // must break here
        }
        if (delta_info_ptr != NULL){
            delta_mutex.lock();
            sum_delta += sum_delta_i;
            sum_sq_delta += sum_sq_delta_i;
            delta_mutex.unlock();
        }
    });
    if (delta_info_ptr != NULL){
        delta_info_ptr[0] = sum_delta / n;
        delta_info_ptr[1] = sum_sq_delta / n - sum_delta * sum_delta / (n * n);// var = E(x^2) - E(x)^2
        delta_info_ptr[2] = sum_delta;
        delta_info_ptr[3] = sum_sq_delta;
    }
}

static void __attribute__ ((__target__ ("avx,fma,f16c"))) adam_cpu_1(
    int64_t n,
    float* param_fp32_ptr,
    uint16_t* param_fp16_ptr,
    float* delta_info_ptr,
    uint16_t* g_fp16_ptr,
    float* m_fp32_ptr,
    float* v_fp32_ptr,
    float beta1, float beta2, 
    float eps, float lr, 
    float scale, 
    float weight_decay,
    float bias_correction1,
    float bias_correction2
){
    float sum_sq_delta = 0;
    float sum_delta = 0;
    std::mutex delta_mutex;
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
    parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
        float sum_sq_delta_i = 0;
        float sum_delta_i = 0;
        for (int64_t j = start; j < end; j += span) {
            if (j + span > end) {
                for (int64_t i = j; i < end; i++) {
                    float g = fp16_ieee_to_fp32_value(g_fp16_ptr[i]) / scale;
                    float m = m_fp32_ptr[i];
                    float v = v_fp32_ptr[i];
                    float p = param_fp32_ptr[i];
                    m = beta1 * m + (1 - beta1) * g;
                    v = beta2 * v + (1 - beta2) * g * g;
                    if (delta_info_ptr != NULL){
                        float delta =  m  / bias_correction1 / (sqrtf(v / bias_correction2) + eps) + weight_decay * p;
                        sum_delta_i += delta;
                        sum_sq_delta_i += delta * delta;
                    }
                    p = p - lr * m  / bias_correction1 / (sqrtf(v / bias_correction2) + eps) - lr * weight_decay * p;
                    param_fp32_ptr[i] = p;
                    param_fp16_ptr[i] = fp16_ieee_from_fp32_value(p);
                    m_fp32_ptr[i] = m;
                    v_fp32_ptr[i] = v;
                }
                break; // must break here
            } else {
                auto g = _mm256_div_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)&g_fp16_ptr[j])), avx_scale);
                auto m = _mm256_loadu_ps(&m_fp32_ptr[j]);
                auto v = _mm256_loadu_ps(&v_fp32_ptr[j]);
                auto p = _mm256_loadu_ps(&param_fp32_ptr[j]);
                m = _mm256_fmadd_ps(avx_beta1, m, _mm256_mul_ps(avx_beta1_1, g));
                v = _mm256_fmadd_ps(avx_beta2, v, _mm256_mul_ps(avx_beta2_1, _mm256_mul_ps(g, g)));
                if (delta_info_ptr != NULL){
                    auto delta_256 = _mm256_add_ps(
                        _mm256_div_ps(
                            _mm256_div_ps(m, avx_bias_correction1), // m / bias_correction1
                            _mm256_add_ps(_mm256_sqrt_ps(_mm256_div_ps(v, avx_bias_correction2)), avx_eps)  // sqrt(v / bias_correction2) + eps
                        ),  // m / bias_correction1 / (sqrt(v / bias_correction2) + eps)
                        _mm256_mul_ps(avx_weight_decay, p) // weight_decay * p
                    ); // delta = m / bias_correction1 / (sqrt(v / bias_correction2) + eps) + weight_decay * p 
                    sum_delta_i += _mm256_reduce_add_ps(delta_256);
                    sum_sq_delta_i += _mm256_reduce_add_ps(_mm256_mul_ps(delta_256, delta_256));
                }
                p = _mm256_fmadd_ps(avx_neg_lr, _mm256_mul_ps(avx_weight_decay, p), p); // p = p - lr * weight_decay * p
                p = _mm256_fmadd_ps(
                    avx_neg_lr,
                    _mm256_div_ps(
                        _mm256_div_ps(m, avx_bias_correction1), // m / bias_correction1
                        _mm256_add_ps(_mm256_sqrt_ps(_mm256_div_ps(v, avx_bias_correction2)), avx_eps)  // sqrt(v / bias_correction2) + eps
                    ),  // m / bias_correction1 / (sqrt(v / bias_correction2) + eps)
                    p
                );  // p = p - lr * m / bias_correction1 / (sqrt(v / bias_correction2) + eps)
                _mm256_storeu_ps(&param_fp32_ptr[j], p);
                _mm_storeu_si128((__m128i*)&param_fp16_ptr[j], _mm256_cvtps_ph(p, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                _mm256_storeu_ps(&m_fp32_ptr[j], m);
                _mm256_storeu_ps(&v_fp32_ptr[j], v);
            }
        }
        if (delta_info_ptr != NULL){
            delta_mutex.lock();
            sum_delta += sum_delta_i;
            sum_sq_delta += sum_sq_delta_i;
            delta_mutex.unlock();
        }
    });
    if (delta_info_ptr != NULL){
        delta_info_ptr[0] = sum_delta / n;
        delta_info_ptr[1] = sum_sq_delta / n - sum_delta * sum_delta / (n * n);// var = E(x^2) - E(x)^2
        delta_info_ptr[2] = sum_delta;
        delta_info_ptr[3] = sum_sq_delta;
    }
}

static void __attribute__ ((__target__ ("avx512f"))) adam_cpu_2(
    int64_t n,
    float* param_fp32_ptr,
    uint16_t* param_fp16_ptr,
    float* delta_info_ptr,
    uint16_t* g_fp16_ptr,
    float* m_fp32_ptr,
    float* v_fp32_ptr,
    float beta1, float beta2, 
    float eps, float lr, 
    float scale, 
    float weight_decay,
    float bias_correction1,
    float bias_correction2
){
    float sum_sq_delta = 0;
    float sum_delta = 0;
    std::mutex delta_mutex;
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
    parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
        float sum_sq_delta_i = 0;
        float sum_delta_i = 0;
        for (int64_t j = start; j < end; j += span) {
            if (j + span > end) {
                for (int64_t i = j; i < end; i++) {
                    float g = fp16_ieee_to_fp32_value(g_fp16_ptr[i]) / scale;
                    float m = m_fp32_ptr[i];
                    float v = v_fp32_ptr[i];
                    float p = param_fp32_ptr[i];
                    m = beta1 * m + (1 - beta1) * g;
                    v = beta2 * v + (1 - beta2) * g * g;
                    if (delta_info_ptr != NULL){
                        float delta =  m  / bias_correction1 / (sqrtf(v / bias_correction2) + eps) + weight_decay * p;
                        sum_delta_i += delta;
                        sum_sq_delta_i += delta * delta;
                    }
                    p = p - lr * m  / bias_correction1 / (sqrtf(v / bias_correction2) + eps) - lr * weight_decay * p;
                    param_fp32_ptr[i] = p;
                    param_fp16_ptr[i] = fp16_ieee_from_fp32_value(p);
                    m_fp32_ptr[i] = m;
                    v_fp32_ptr[i] = v;
                }
                break; // must break here
            }else{
                auto g = _mm512_div_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)&g_fp16_ptr[j])), avx_scale);
                auto m = _mm512_loadu_ps(&m_fp32_ptr[j]);
                auto v = _mm512_loadu_ps(&v_fp32_ptr[j]);
                auto p = _mm512_loadu_ps(&param_fp32_ptr[j]);
                m = _mm512_fmadd_ps(avx_beta1, m, _mm512_mul_ps(avx_beta1_1, g));
                v = _mm512_fmadd_ps(avx_beta2, v, _mm512_mul_ps(avx_beta2_1, _mm512_mul_ps(g, g)));
                if (delta_info_ptr != NULL){
                    auto delta_512 = _mm512_add_ps(
                        _mm512_div_ps(
                            _mm512_div_ps(m, avx_bias_correction1), // m / bias_correction1
                            _mm512_add_ps(_mm512_sqrt_ps(_mm512_div_ps(v, avx_bias_correction2)), avx_eps)  // sqrt(v / bias_correction2) + eps
                        ),  // m / bias_correction1 / (sqrt(v / bias_correction2) + eps)
                        _mm512_mul_ps(avx_weight_decay, p) // weight_decay * p
                    ); // delta = m / bias_correction1 / (sqrt(v / bias_correction2) + eps) + weight_decay * p
                    sum_delta_i += _mm512_reduce_add_ps(delta_512); 
                    sum_sq_delta_i += _mm512_reduce_add_ps(_mm512_mul_ps(delta_512, delta_512));
                }
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
                _mm512_storeu_ps(&param_fp32_ptr[j], p);
                _mm256_storeu_si256((__m256i*)&param_fp16_ptr[j], _mm512_cvtps_ph(p, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                _mm512_storeu_ps(&m_fp32_ptr[j], m);
                _mm512_storeu_ps(&v_fp32_ptr[j], v);
            }
        }
        if (delta_info_ptr != NULL){
            delta_mutex.lock();
            sum_delta += sum_delta_i;
            sum_sq_delta += sum_sq_delta_i;
            delta_mutex.unlock();
        }
    });
    if (delta_info_ptr != NULL){
        delta_info_ptr[0] = sum_delta / n;
        delta_info_ptr[1] = sum_sq_delta / n - sum_delta * sum_delta / (n * n);// var = E(x^2) - E(x)^2
        delta_info_ptr[2] = sum_delta;
        delta_info_ptr[3] = sum_sq_delta;
    }
}

void adam_cpu_fp16_launcher(
    int64_t n,
    std::uintptr_t param_fp32,
    std::uintptr_t param_fp16,
    std::uintptr_t delta_info,
    std::uintptr_t g_fp16,
    std::uintptr_t m_fp32,
    std::uintptr_t v_fp32,
    float beta1, float beta2, 
    float eps, float lr, 
    float scale, 
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    auto delta_info_ptr = reinterpret_cast<float*>(delta_info);
    auto param_fp32_ptr = reinterpret_cast<float*>(param_fp32);
    auto m_fp32_ptr = reinterpret_cast<float*>(m_fp32);
    auto v_fp32_ptr = reinterpret_cast<float*>(v_fp32);
    auto param_fp16_ptr = reinterpret_cast<uint16_t*>(param_fp16);
    auto g_fp16_ptr  = reinterpret_cast<uint16_t*>(g_fp16);
    int cpu_level = get_cpu_level();
    if (cpu_level == 0 ){
        adam_cpu_0(n, param_fp32_ptr, param_fp16_ptr, delta_info_ptr, g_fp16_ptr, m_fp32_ptr, v_fp32_ptr, beta1, beta2, eps, lr, scale, weight_decay, bias_correction1, bias_correction2);
    }else if(cpu_level == 1){
        adam_cpu_1(n, param_fp32_ptr, param_fp16_ptr, delta_info_ptr, g_fp16_ptr, m_fp32_ptr, v_fp32_ptr, beta1, beta2, eps, lr, scale, weight_decay, bias_correction1, bias_correction2);
    }else{
        adam_cpu_2(n, param_fp32_ptr, param_fp16_ptr, delta_info_ptr, g_fp16_ptr, m_fp32_ptr, v_fp32_ptr, beta1, beta2, eps, lr, scale, weight_decay, bias_correction1, bias_correction2);
    }
}

void adam_cpu_bf16_launcher(
    int64_t n,
    std::uintptr_t param_fp32,
    std::uintptr_t param_bf16,
    std::uintptr_t delta_info,
    std::uintptr_t g_bf16,
    std::uintptr_t m_fp32,
    std::uintptr_t v_fp32,
    float beta1, float beta2,
    float eps, float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    auto delta_info_ptr = reinterpret_cast<float*>(delta_info);
    auto m_fp32_ptr = reinterpret_cast<float*>(m_fp32);
    auto v_fp32_ptr = reinterpret_cast<float*>(v_fp32);
    auto param_fp32_ptr = reinterpret_cast<float*>(param_fp32);
    auto param_bf16_ptr = reinterpret_cast<uint16_t*>(param_bf16);
    auto g_bf16_ptr  = reinterpret_cast<uint16_t*>(g_bf16);
    adam_cpu_bf16_0(n, param_fp32_ptr, param_bf16_ptr, delta_info_ptr, g_bf16_ptr, m_fp32_ptr, v_fp32_ptr, beta1, beta2, eps, lr, scale, weight_decay, bias_correction1, bias_correction2);
}
