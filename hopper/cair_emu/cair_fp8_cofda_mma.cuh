/*
 * CAIR FP8 CoFDA "Emulated MMA" Primitive
 *
 * Software emulation of the warp-level FP8 tensor-core MMA
 * (mma.sync...e4m3.e4m3.f32) from vllm-cair's cair_scaled_fp8_mm_tc.cuh, but
 * per-thread: one output cell per call. Operands come from decode-on-load SMEM
 * fragments (DecodedFrag).
 */

#pragma once

#include <cuda_fp8.h>

#include <cstdint>

#include "cair_types.cuh"
#include "cair_fp32_utils.cuh"
#include "cair_fp8_utils.cuh"

namespace vllm {
namespace cair {

/**
 * @brief CoFDA-emulated dot-accumulate for one output cell:
 *        c += dot(A_row_chunk, B_col_chunk) over CHUNK_SIZE FP8 (E4M3) values.
 *
 * Two passes over the (decode-on-load) fragments: Pass 1 scans for NaN/Inf and
 * the max exponent; Pass 2 aligns each product to max_exp and accumulates an
 * int64 mantissa sum, normalized back to FP32 via fixed_to_fp32<F>. Products
 * are streamed through registers (no Product[]/Operand[] arrays) and recomputed
 * in Pass 2 from the pre-decoded operands.
 *
 * Caller contract: assumes CHUNK_SIZE valid FP8 elements per fragment; the emu
 * kernel zero-pads out-of-range bytes during the SMEM load.
 */
template <int F, int CHUNK_SIZE>
[[nodiscard]] __device__ __forceinline__ float fp8_cofda_mma(
    DecodedFrag a_frag, DecodedFrag b_frag, float c) {
  Operand c_operand = fp32_to_operand<F>(c);

  if (c_operand.is_nan) {
    return bits_to_fp32(fp32::QNAN_BITS);
  }

  // ---- Pass 1: NaN/Inf scan + max_exp (Product streamed in registers) ----
  bool any_nan = false;
  bool any_inf = false;
  int pos_inf_count = 0;
  int neg_inf_count = 0;
  int max_exp = -9999;
  int non_zero_count = 0;

  if (c_operand.is_inf) {
    any_inf = true;
    if (c_operand.sign > 0)
      pos_inf_count++;
    else
      neg_inf_count++;
  }
  if (!c_operand.is_zero && !c_operand.is_nan && !c_operand.is_inf) {
    max_exp = c_operand.exponent;
    non_zero_count++;
  }

#pragma unroll
  for (int i = 0; i < CHUNK_SIZE; i++) {
    Product p = fp8_multiply_predecoded<F>(a_frag[i], b_frag[i]);
    if (p.is_nan) {
      any_nan = true;
    }
    if (p.is_inf) {
      any_inf = true;
      if (p.sign > 0)
        pos_inf_count++;
      else
        neg_inf_count++;
    }
    if (!p.is_zero && !p.is_nan && !p.is_inf) {
      max_exp = max(max_exp, p.exponent);
      non_zero_count++;
    }
  }

  if (any_nan) {
    return bits_to_fp32(fp32::QNAN_BITS);
  }

  if (any_inf) {
    if (pos_inf_count > 0 && neg_inf_count > 0) {
      return bits_to_fp32(fp32::QNAN_BITS);
    }
    return bits_to_fp32(pos_inf_count > 0 ? fp32::POS_INF_BITS
                                          : fp32::NEG_INF_BITS);
  }

  if (non_zero_count == 0) {
    return c;
  }

  // ---- Pass 2: align + sum (recompute products in-place) ----
  int64_t mantissa_sum = 0;

  if (!c_operand.is_zero && !c_operand.is_inf) {
    int exp_diff = max_exp - c_operand.exponent;
    int64_t aligned =
        (exp_diff >= 64) ? 0 : (c_operand.significand >> exp_diff);
    mantissa_sum += c_operand.sign * aligned;
  }

#pragma unroll
  for (int i = 0; i < CHUNK_SIZE; i++) {
    Product p = fp8_multiply_predecoded<F>(a_frag[i], b_frag[i]);
    if (p.is_zero || p.is_nan || p.is_inf) {
      continue;
    }
    int exp_diff = max_exp - p.exponent;
    int64_t aligned =
        (exp_diff >= 64) ? 0
                         : (static_cast<int64_t>(p.significand) >> exp_diff);
    mantissa_sum += p.sign * aligned;
  }

  return fixed_to_fp32<F>(mantissa_sum, max_exp);
}

}  // namespace cair
}  // namespace vllm
