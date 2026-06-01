// hopper/gemm_qk_cofda_emu.h
//
// Software CoFDA FP8 emulation of S = Q·Kᵀ for flash-attention's SM90 forward
// kernel.  Two parts:
//   (a)  cofda_dot<F>(fa, fb, D)  — numeric core, decoupled from operand LAYOUT.
//   (b)  flash::gemm_qk_cofda_emu — kernel entry that reads raw FP8 Q/K from
//        SWIZZLED smem tensors.  Requires CuTe.
//
// CAIR's fp8_cofda_mma expects DECODED operands (DecodedFrag over packed uint32)
// because in CAIR's own kernel decode happens once at smem load.  In
// flash-attention the smem holds RAW, SWIZZLED FP8 — so we decode on the fly and
// fetch each FP8 byte through a caller-supplied accessor fa(k)/fb(k) that returns
// the raw uint8 at column k.  This keeps the CoFDA math identical while letting
// the caller own the (swizzled tensor vs flat array) addressing.

#pragma once
#include <cassert>
#include <cstdint>
#include <cute/tensor.hpp>
#include "cair_emu/cair_fp8_cofda_mma.cuh"  // vllm::cair::{fp8_cofda_mma<F,CHUNK_SIZE>, decode_operand, pack_decoded, DecodedFrag}

// --- helper: reinterpret one FP8 element as its raw byte --------------------
// Works for both cutlass float_e4m3_t and __nv_fp8_e4m3.
template <class E>
__device__ __forceinline__ uint8_t fp8_bits(E const& e) {
  return *reinterpret_cast<const uint8_t*>(&e);
}

// --- (a) Numeric core, decoupled from operand LAYOUT ------------------------
//
// fa(k) / fb(k) are caller-supplied device lambdas that return the raw uint8_t
// FP8 byte at column k.  They must be __device__-callable.
//
// D must be a multiple of CHUNK=32 (CoFDA slice-1 constraint).
// kHeadDim ∈ {64, 128, 192, 256} — all divisible by 32.
//
// F controls the number of fractional bits in the CoFDA accumulator:
//   F=25 ≈ FP32 precision (tight tolerance vs FP32 reference)
//   F=13 uses a restricted accumulator that visibly diverges
template <int F, class FetchA, class FetchB>
__device__ float cofda_dot(FetchA fa, FetchB fb, int D) {
  constexpr int CHUNK = 32;
  assert(D % 32 == 0);
  // Decoded operands in registers (NOT smem — one chunk at a time)
  uint32_t a_dec[CHUNK], b_dec[CHUNK];
  float acc = 0.f;
  for (int base = 0; base < D; base += CHUNK) {
    #pragma unroll
    for (int j = 0; j < CHUNK; ++j) {
      a_dec[j] = vllm::cair::pack_decoded(vllm::cair::decode_operand(fa(base + j)));
      b_dec[j] = vllm::cair::pack_decoded(vllm::cair::decode_operand(fb(base + j)));
    }
    vllm::cair::DecodedFrag a_frag{a_dec};
    vllm::cair::DecodedFrag b_frag{b_dec};
    // fp8_cofda_mma<F, CHUNK_SIZE>(a_frag, b_frag, acc) — chunked accumulate
    // with intermediate rounding; CHUNK_SIZE must be 32 (cair slice-1 only).
    acc = vllm::cair::fp8_cofda_mma<F, CHUNK>(a_frag, b_frag, acc);
  }
  return acc;
}

// --- (b) Kernel entry: replace the QK WGMMA. Reads raw FP8 Q/K from SWIZZLED smem.
//
// sQ_pi / sK_pi are position-independent swizzle views; cute/tensor.hpp is
// included unconditionally above so this section is always parsed.
namespace flash {

// sQ_pi / sK_pi are position-independent swizzle views shaped (M, D) / (N, D).
// We index them as tensors — sQ_pi(m, k) — so the swizzle is applied correctly;
// we do NOT take a raw pointer and stride by hand.
template <int F, class TiledMma, class TensorSQ, class TensorSK, class TensorC>
CUTLASS_DEVICE void gemm_qk_cofda_emu(TiledMma const& tiled_mma,
                                      TensorSQ const& sQ_pi,
                                      TensorSK const& sK_pi,
                                      TensorC& tSrS,
                                      int thread_idx) {
  using namespace cute;
  // Logical (m,n) coordinate for each accumulator element this thread owns.
  // Mirrors the masking code at mainloop_fwd_sm90_tma_gmma_ws.hpp:1151-1154.
  auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
  Tensor cS = make_identity_tensor(make_shape(size<0>(sQ_pi), size<0>(sK_pi)));  // (M,N)->(m,n)
  Tensor tScS = thr_mma.partition_C(cS);   // same layout/size as tSrS
  CUTE_STATIC_ASSERT_V(size(tScS) == size(tSrS));
  const int D = size<1>(sQ_pi);            // kHeadDim
  #pragma unroll
  for (int i = 0; i < size(tSrS); ++i) {
    auto coord = tScS(i);                  // (m, n)
    int m = get<0>(coord), n = get<1>(coord);
    // Swizzle-correct, decode-on-the-fly fetchers.
    auto fa = [&] __device__ (int k) { return fp8_bits(sQ_pi(m, k)); };
    auto fb = [&] __device__ (int k) { return fp8_bits(sK_pi(n, k)); };
    tSrS(i) = cofda_dot<F>(fa, fb, D);
  }
}

}  // namespace flash
