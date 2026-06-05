// hopper/gemm_pv_cofda_emu.h
//
// Software CoFDA FP8 emulation of O = P·V for flash-attention's SM90 forward
// kernel. Mirrors gemm_qk_cofda_emu.h but for the SECOND attention GEMM.
//
// P is the FP8 softmax output, staged to smem by the caller (the register
// fragment is warpgroup-distributed, so a thread cannot read P[m, all k] from
// registers). V is the FP8 value tile in smem. The emu disables the FP8 STSM
// transpose (Transpose_V), so V keeps LOGICAL (headdim, seqlen) order in smem
// (SmemLayoutVt); the position-independent swizzle view gives sV_pi(n,k) == V[k,n]
// directly, with no seqlen permutation to misalign P[m,k] with V[k,n].
// The reduction is over kBlockN (this KV block's keys); the CoFDA accumulator is
// SEEDED with the existing tOrO so the across-block reduction models the
// hardware's continuous FP32 accumulator.

#pragma once
#include <cute/tensor.hpp>
#include "gemm_qk_cofda_emu.h"  // cofda_dot_acc<F>, fp8_bits

namespace flash {

// sP_pi : position-independent swizzle view of P, shape (kBlockM, kBlockN)   -> sP_pi(m, k)
// sV_pi : position-independent swizzle view of V, shape (kHeadDimV, kBlockN)  -> sV_pi(n, k) == V[k,n]
// tOrO  : FP32 output accumulator (in/out). ZeroInit==true on the first KV block.
template <int F, bool ZeroInit, class TiledMmaPV,
          class TensorSP, class TensorSV, class TensorO>
CUTLASS_DEVICE void gemm_pv_cofda_emu(TiledMmaPV const& tiled_mma,
                                      TensorSP const& sP_pi,
                                      TensorSV const& sV_pi,
                                      TensorO& tOrO,
                                      int thread_idx) {
  using namespace cute;
  auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
  // (M, N) -> (m, n) for the output O; M = kBlockM = size<0>(sP_pi),
  // N = kHeadDimV = size<0>(sV_pi). Same identity-partition trick as taccOcO.
  Tensor cO = make_identity_tensor(make_shape(size<0>(sP_pi), size<0>(sV_pi)));
  Tensor tOcO = thr_mma.partition_C(cO);
  CUTE_STATIC_ASSERT_V(size(tOcO) == size(tOrO));
  CUTE_STATIC_ASSERT_V(size<1>(sP_pi) == size<1>(sV_pi));  // shared reduction extent (kBlockN)
  const int K = size<1>(sP_pi);  // kBlockN
  static_assert(decltype(size<1>(sP_pi))::value % 32 == 0,
                "kBlockN must be a multiple of CHUNK=32 for CoFDA emulation");
  #pragma unroll
  for (int i = 0; i < size(tOrO); ++i) {
    auto coord = tOcO(i);
    int m = get<0>(coord), n = get<1>(coord);
    auto fa = [&] __device__ (int k) { return fp8_bits(sP_pi(m, k)); };
    auto fb = [&] __device__ (int k) { return fp8_bits(sV_pi(n, k)); };
    float acc_init = ZeroInit ? 0.f : tOrO(i);
    tOrO(i) = cofda_dot_acc<F>(fa, fb, K, acc_init);
  }
}

}  // namespace flash
