// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

#include <stdexcept>

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_ALIBI
  #define ALIBI_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

#if defined(ENABLE_DTYPE_FP16) && defined(ENABLE_DTYPE_BF16)
#define FP16_SWITCH(COND, ...) \
  [&] { \
    if (COND) { \
      using elem_type = cutlass::half_t; \
      return __VA_ARGS__(); \
    } else { \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__(); \
    } \
  }()
#elif defined(ENABLE_DTYPE_FP16)
#define FP16_SWITCH(COND, ...) \
  [&] { \
    if (!(COND)) { \
      throw std::runtime_error("BF16 support not compiled. Please recompile with CPMCU_DTYPE=bf16 or CPMCU_DTYPE=fp16,bf16"); \
    } \
    using elem_type = cutlass::half_t; \
    return __VA_ARGS__(); \
  }()
#elif defined(ENABLE_DTYPE_BF16)
#define FP16_SWITCH(COND, ...) \
  [&] { \
    if (COND) { \
      throw std::runtime_error("FP16 support not compiled. Please recompile with CPMCU_DTYPE=fp16 or CPMCU_DTYPE=fp16,bf16"); \
    } \
    using elem_type = cutlass::bfloat16_t; \
    return __VA_ARGS__(); \
  }()
#else
#error "At least one of ENABLE_DTYPE_FP16 or ENABLE_DTYPE_BF16 must be defined"
#endif

// TODO only compile 64 for debug
#define HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
      constexpr static int kHeadDim = 128; \
      return __VA_ARGS__();              \
  }()
  //   if (HEADDIM <= 32) {                   \
  //     constexpr static int kHeadDim = 32;  \
  //     return __VA_ARGS__();                \
  //   } else if (HEADDIM <= 64) {            \
  //     constexpr static int kHeadDim = 64;  \
  //     return __VA_ARGS__();                \
  //   } else if (HEADDIM <= 96) {            \
  //     constexpr static int kHeadDim = 96;  \
  //     return __VA_ARGS__();                \
  //   } else if (HEADDIM <= 128) {           \
  //     constexpr static int kHeadDim = 128; \
  //     return __VA_ARGS__();                \
  //   } else if (HEADDIM <= 160) {           \
  //     constexpr static int kHeadDim = 160; \
  //     return __VA_ARGS__();                \
  //   } else if (HEADDIM <= 192) {           \
  //     constexpr static int kHeadDim = 192; \
  //     return __VA_ARGS__();                \
  //   } else if (HEADDIM <= 256) {           \
  //     constexpr static int kHeadDim = 256; \
  //     return __VA_ARGS__();                \
  //   }                                      \
  // }()
