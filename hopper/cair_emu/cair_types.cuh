/*
 * CAIR FP8 CoFDA POD types: Operand / Product accumulation operands and
 * OutputTraits for FP32 -> BF16/FP16 rounding.
 * Ported from vllm-cair/csrc/quantization/cair/cair_types.cuh.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace vllm {
namespace cair {

/**
 * @brief Decoded operand for CoFDA accumulation: sign-magnitude, with the
 *        significand carrying F fractional bits below the radix.
 */
struct Operand {
  int sign;             // +1 / -1
  int exponent;         // unbiased
  int64_t significand;  // F fractional bits, signed magnitude
  bool is_zero;
  bool is_nan;
  bool is_inf;
};

/**
 * @brief Unnormalized product of two FP8 E4M3 values, pre-CoFDA-accumulation.
 *        Significand carries F fractional bits; exponent is exp_a + exp_b.
 */
struct Product {
  int sign;              // +1 / -1
  int exponent;          // exp_a + exp_b
  uint64_t significand;  // F fractional bits
  bool is_zero;          // either operand was zero
  bool is_nan;           // either operand was NaN
  bool is_inf;
  bool is_subnormal;
};

/// @brief Per-output-dtype round-to-nearest-even conversion from FP32.
template <typename T>
struct OutputTraits;

template <>
struct OutputTraits<__nv_bfloat16> {
  using Type = __nv_bfloat16;

  [[nodiscard]] __device__ __forceinline__ static Type from_float(float val) {
    return __float2bfloat16_rn(val);
  }
};

template <>
struct OutputTraits<__half> {
  using Type = __half;

  [[nodiscard]] __device__ __forceinline__ static Type from_float(float val) {
    return __float2half_rn(val);
  }
};

/**
 * @brief Convert FP32 to output type with round-to-nearest.
 */
template <typename OutDtype>
[[nodiscard]] __device__ __forceinline__ OutDtype
float_to_output_rn(float val) {
  return OutputTraits<OutDtype>::from_float(val);
}

}  // namespace cair
}  // namespace vllm
