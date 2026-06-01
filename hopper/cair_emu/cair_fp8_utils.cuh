/*
 * CAIR FP8 E4M3 Utilities (decode-on-load path)
 *
 * FP8-specific helpers for the CoFDA emulation kernel:
 * - DecodedOperand: a decoded + subnormal-resolved E4M3 operand
 * - decode_operand: FP8 byte -> DecodedOperand (once, at SMEM load)
 * - pack_decoded / unpack_decoded: DecodedOperand <-> packed uint32 (the
 *   bank-conflict-free SMEM word) + DecodedFrag accessor
 * - fp8_multiply_predecoded<F>: unnormalized Product from two DecodedOperands
 */

#pragma once

#include <cuda_fp8.h>
#include <cstdint>
#include "cair_types.cuh"

namespace vllm {
namespace cair {

// Pre-resolved FP8 operand: decoded + subnormal-resolved once at SMEM-load
// time (decode-on-load). `significand` already folds the implicit one (normal)
// or equals the raw mantissa (subnormal); `exp` is debiased. The per-element
// multiply (fp8_multiply_predecoded) consumes this directly, hoisting
// decode/subnormal resolution out of the inner loop. See
// docs/superpowers/specs/2026-05-22-cair-fp8-decode-on-load-design.md.
struct DecodedOperand {
  int8_t sign;          ///< +1 / -1
  uint8_t significand;  ///< normal: 8+mantissa (8..15); subnormal: mantissa (1..7)
  int8_t exp;           ///< debiased exponent (-6..8)
  uint8_t flags;        ///< bit0 = is_nan, bit1 = is_zero
};

namespace decoded_flag {
constexpr uint8_t NAN_BIT = 0x1;
constexpr uint8_t ZERO_BIT = 0x2;
}  // namespace decoded_flag

// Pack/unpack DecodedOperand <-> one uint32 word so SMEM stores one 4-byte
// element (a single LDS.32) instead of four 1-byte SoA loads. The signed
// fields round-trip exactly: int8 -> (uint8_t) raw byte -> (int8_t). Used by
// the bank-conflict-free packed+padded SMEM layout. See
// docs/superpowers/specs/2026-05-22-cair-fp8-smem-bank-conflict-design.md.
[[nodiscard]] __device__ __forceinline__ uint32_t
pack_decoded(DecodedOperand d) {
  return (static_cast<uint32_t>(static_cast<uint8_t>(d.sign))) |
         (static_cast<uint32_t>(d.significand) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(d.exp)) << 16) |
         (static_cast<uint32_t>(d.flags) << 24);
}

[[nodiscard]] __device__ __forceinline__ DecodedOperand
unpack_decoded(uint32_t w) {
  DecodedOperand d;
  d.sign = static_cast<int8_t>(w & 0xFF);
  d.significand = static_cast<uint8_t>((w >> 8) & 0xFF);
  d.exp = static_cast<int8_t>((w >> 16) & 0xFF);
  d.flags = static_cast<uint8_t>((w >> 24) & 0xFF);
  return d;
}

// Accessor over one operand row stored as packed uint32 words: one LDS.32 +
// unpack per element. fp8_cofda_mma still indexes `frag[i]` -> DecodedOperand.
struct DecodedFrag {
  const uint32_t* __restrict__ words;
  [[nodiscard]] __device__ __forceinline__ DecodedOperand operator[](
      int i) const {
    return unpack_decoded(words[i]);
  }
};

namespace fp8 {

constexpr int EXPONENT_BITS = 4;
constexpr int MANTISSA_BITS = 3;
constexpr int EXPONENT_BIAS = 7;
constexpr int IMPLICIT_ONE = 8;       ///< 1 << MANTISSA_BITS
constexpr int PRODUCT_RADIX_BIT = 6;  ///< 2 * MANTISSA_BITS
constexpr uint8_t NAN_PATTERN = 0x7F;

}  // namespace fp8

/**
 * @brief Decode + resolve one FP8 E4M3 byte into a DecodedOperand.
 *
 * E4M3: bit7 sign, bits6-3 biased exponent, bits2-0 mantissa; zero (exp=0,
 * mant=0), subnormal (exp=0, mant!=0), NaN = 0x7F/0xFF (no infinity).
 * Resolves the implicit one and debiases the exponent here so the inner-loop
 * multiply (fp8_multiply_predecoded) has no per-element decode/subnormal
 * branch. Called once per unique operand byte at SMEM-load time.
 */
[[nodiscard]] __device__ __forceinline__ DecodedOperand
decode_operand(uint8_t bits) {
  DecodedOperand d;
  d.sign = (bits & 0x80) ? -1 : 1;
  const int exponent = (bits >> 3) & 0x0F;  // biased field
  const uint32_t mantissa = bits & 0x07;

  if ((bits & 0x7F) == fp8::NAN_PATTERN) {
    d.flags = decoded_flag::NAN_BIT;
    d.significand = 0;
    d.exp = 0;
    return d;
  }
  if (exponent == 0 && mantissa == 0) {  // zero
    d.flags = decoded_flag::ZERO_BIT;
    d.significand = 0;
    d.exp = 0;
    return d;
  }
  d.flags = 0;
  if (exponent == 0) {  // subnormal
    d.significand = static_cast<uint8_t>(mantissa);
    d.exp = static_cast<int8_t>(1 - fp8::EXPONENT_BIAS);
  } else {  // normal
    d.significand = static_cast<uint8_t>(fp8::IMPLICIT_ONE + mantissa);
    d.exp = static_cast<int8_t>(exponent - fp8::EXPONENT_BIAS);
  }
  return d;
}

/**
 * @brief Multiply two pre-decoded FP8 operands -> unnormalized Product with F
 *        fractional bits. Operands arrive already decoded + resolved (no
 *        decompose, no subnormal branch): computes the sign, the F-scaled
 *        significand product (Q2.6 raw product shifted by F-6), and the summed
 *        exponent, with NaN-then-zero precedence.
 */
template <int F>
[[nodiscard]] __device__ __forceinline__ Product
fp8_multiply_predecoded(DecodedOperand a, DecodedOperand b) {
  Product result;
  result.is_nan = false;
  result.is_inf = false;

  if ((a.flags & decoded_flag::NAN_BIT) || (b.flags & decoded_flag::NAN_BIT)) {
    result.is_nan = true;
    result.is_zero = false;
    result.sign = 1;
    result.exponent = 0;
    result.significand = 0;
    return result;
  }

  if ((a.flags & decoded_flag::ZERO_BIT) ||
      (b.flags & decoded_flag::ZERO_BIT)) {
    result.is_zero = true;
    result.sign = 1;
    result.exponent = 0;
    result.significand = 0;
    return result;
  }

  result.is_zero = false;
  result.sign = a.sign * b.sign;

  const uint64_t raw_product = static_cast<uint64_t>(a.significand) *
                               static_cast<uint64_t>(b.significand);
  constexpr int SCALE_SHIFT = F - fp8::PRODUCT_RADIX_BIT;
  if constexpr (SCALE_SHIFT >= 0) {
    result.significand = raw_product << SCALE_SHIFT;
  } else {
    result.significand = raw_product >> (-SCALE_SHIFT);
  }
  result.exponent = a.exp + b.exp;

  return result;
}

}  // namespace cair
}  // namespace vllm
