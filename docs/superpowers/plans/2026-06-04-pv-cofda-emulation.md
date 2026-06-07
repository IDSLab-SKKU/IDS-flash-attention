# PV CoFDA Emulation Variant — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runtime-selectable variant of the SM90 FP8 forward attention kernel that replaces the hardware `O = P·V` WGMMA with a software CoFDA emulation, independent of the existing QK emulation, so FP8 accumulator precision (`F ∈ {13,25}`) can be studied for the second attention GEMM inside the real kernel.

**Architecture:** Mirror the existing `UseQKEmu`/`QKEmuFbits` compile-time-axis threading with a parallel, independent `UsePVEmu`/`PVEmuFbits` axis, threaded python → `mha_fwd` → `Flash_fwd_params` → `BOOL_SWITCH` → `CollectiveMainloopFwdSm90` → an `if constexpr (UsePVEmu)` branch at the PV gemm site that calls a new `flash::gemm_pv_cofda_emu<F, ZeroInit>` device function. Because the PV `P` operand is a warpgroup-distributed register fragment, `P` is staged to smem first (new emu-only smem layout), then `P` and `V` are read from smem per-element. The CoFDA accumulator is seeded with the carried `tOrO` across KV blocks to model the hardware's continuous FP32 accumulator. The QK long-context race lesson (per-WG `NamedBarrier` before the operand `consumer_release`) is applied to `V` up front.

**Tech Stack:** CUDA 13.0 / nvcc, CUTLASS CuTe (SM90 GMMA), PyTorch C++ extension (pybind11), Python (flash_attn interface + vLLM backend). Target: H100 sm_90a.

**Spec:** `docs/superpowers/specs/2026-06-04-pv-cofda-emulation-design.md`
**Companion (QK) plan/spec:** `docs/superpowers/plans/2026-06-01-qk-cofda-emulation.md`, `docs/superpowers/specs/2026-06-01-qk-cofda-emulation-design.md`
**Reference notes:** `AI/GEMM_WGMMA_ANALYSIS.md`, `AI/SASS_PTX_MMA_ANALYSIS.md`

---

## Key facts established from the code (do not re-derive)

- For FP8 head_dim==128: `Is_FP8=true`, `Transpose_V=true` ⇒ `TensorStorage = TensorStorageTransposeV` (`mainloop_fwd_sm90_tma_gmma_ws.hpp:371`), which has **no `smem_p` field** (P is `MmaPV_is_RS`, register-source). Staging P needs an emu-only storage variant.
- `MmaPV_use_RS_WG1` and `LargeHeadDimV` are both **false** at head_dim==128/headdim_v==128 ⇒ the separate `mma_pv` function (`:1494`) is unused; the only live PV gemm is in the main `mma` non-overlap path at **`:1428`**.
- The emu forces the non-overlap path (`IntraWGOverlap` is gated off for emu), so the IntraWGOverlap PV gemm (`:1277`) is **not** reached when `UsePVEmu` is on.
- Smem tensors at the PV site: `sV = make_tensor(smem_v.data(), SmemLayoutVtMma{})` shaped **(kHeadDimV, kBlockN, kStages)** ⇒ `sV(n, k, stage)` is `V[k,n]`. `sP` shaped **(kBlockM, kBlockN)** ⇒ `sP(m, k)` is `P[m,k]`. Output coord `(m,n)` per `tOrO` element comes from `partition_C(make_identity_tensor(select<0,1>(TileShape_MNK_PV{})))` (same trick as `taccOcO`, `:1084`).
- `kBlockN == 128` for the FP8 d=128 tile (multiple of CHUNK=32 ⇒ exact chunking).
- PV gemm at `:1428` uses `zero_init = Is_first_iter`; non-first iters rescale `tOrO` (FP32) at `:1423` before the gemm ⇒ the emu seeds its accumulator with the already-rescaled `tOrO`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `hopper/gemm_qk_cofda_emu.h` | Add `cofda_dot_acc<F>(fa, fb, D, acc_init)` (seeded numeric core); make `cofda_dot` call it with `0` | Modify |
| `hopper/gemm_pv_cofda_emu.h` | `flash::gemm_pv_cofda_emu<F, ZeroInit>` — per-element seeded CoFDA over the PV accumulator, reading P/V from smem | Create |
| `hopper/test_pv_cofda_emu.cu` | Standalone nvcc unit test: seeded `cofda_dot_acc` + cross-block accumulation vs FP32 reference | Create |
| `hopper/flash.h` | `pv_emu_enabled`, `pv_emu_fbits` on `Flash_fwd_params` | Modify |
| `hopper/flash_api.cpp` | `mha_fwd` params + set into `params` + validation | Modify |
| `hopper/flash_api_torch_lib.cpp` | register `pv_emu_enabled`/`pv_emu_fbits` in fwd schema | Modify |
| `hopper/flash_fwd_launch_template.h` | `run_flash_fwd` axes; `BOOL_SWITCH(UsePVEmu)`+fbits; gate `IntraWGOverlap`/`UsePersistentScheduler` off for emu; pass axes to collective | Modify |
| `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` | `UsePVEmu_`/`PVEmuFbits_` template axes; emu-only `TensorStorageTransposeVWithP`; `sP` wiring; PV emu branch at `:1428`; P-write warpgroup sync + V per-WG race barrier | Modify |
| `hopper/flash_attn_interface.py` | thread `pv_emu_enabled`/`pv_emu_fbits` kwargs to the fwd op | Modify |
| `vllm-mma/vllm/envs.py` | declare `VLLM_USE_CAIR_PV_EMU`, `VLLM_CAIR_PV_EMU_FBITS` | Modify |
| `vllm-mma/vllm/ids_flash_attn/flash_attn_interface.py` | thread kwargs to `torch.ops._ids_fa3_C.fwd` | Modify |
| `vllm-mma/vllm/v1/attention/backends/ids_flash_attn.py` | read env vars, pass kwargs | Modify |
| `vllm-mma/tests/kernels/attention/test_pv_cofda_emu.py` | bit-exact + determinism + signal tests | Create |
| `vllm-mma/cair/experiments/configs/ids_fa_pv_emu_wikitext.yaml` | WikiText PPL rows (ref / ref-1 / pv-emu-f13 / pv-emu-f25) | Create |

**Conventions:** Mirror the `UseQKEmu` pattern exactly. C++17, `--extended-lambda` for device lambdas, 2-space indent in CUDA files. Do not restructure existing files.

---

## Task 1: Seeded CoFDA core + PV emu device function (standalone TDD)

This is the numerical core. Build and test it in **isolation** with nvcc (fast loop), independent of the full flash build.

**Files:**
- Modify: `hopper/gemm_qk_cofda_emu.h`
- Create: `hopper/gemm_pv_cofda_emu.h`
- Test: `hopper/test_pv_cofda_emu.cu`

- [ ] **Step 1: Add the seeded numeric core to `hopper/gemm_qk_cofda_emu.h`**

Replace the existing `cofda_dot` (lines ~40-60) with a seeded version plus a zero-seeded wrapper. The seeded variant is what PV needs (continuous accumulator); QK keeps calling the zero-seeded form unchanged.

```cpp
// Seeded numeric core: accumulate fa·fb over D into a CoFDA-F accumulator that
// STARTS at acc_init (used by PV to model the hardware's continuous FP32
// accumulator carried across KV blocks). D must be a multiple of CHUNK=32.
template <int F, class FetchA, class FetchB>
__device__ float cofda_dot_acc(FetchA fa, FetchB fb, int D, float acc_init) {
  constexpr int CHUNK = 32;
  assert(D % 32 == 0);
  uint32_t a_dec[CHUNK], b_dec[CHUNK];
  float acc = acc_init;
  for (int base = 0; base < D; base += CHUNK) {
    #pragma unroll
    for (int j = 0; j < CHUNK; ++j) {
      a_dec[j] = vllm::cair::pack_decoded(vllm::cair::decode_operand(fa(base + j)));
      b_dec[j] = vllm::cair::pack_decoded(vllm::cair::decode_operand(fb(base + j)));
    }
    vllm::cair::DecodedFrag a_frag{a_dec};
    vllm::cair::DecodedFrag b_frag{b_dec};
    acc = vllm::cair::fp8_cofda_mma<F, CHUNK>(a_frag, b_frag, acc);
  }
  return acc;
}

// Zero-seeded form (QK uses this; behavior unchanged).
template <int F, class FetchA, class FetchB>
__device__ float cofda_dot(FetchA fa, FetchB fb, int D) {
  return cofda_dot_acc<F>(fa, fb, D, 0.f);
}
```

- [ ] **Step 2: Create `hopper/gemm_pv_cofda_emu.h`**

```cpp
// hopper/gemm_pv_cofda_emu.h
//
// Software CoFDA FP8 emulation of O = P·V for flash-attention's SM90 forward
// kernel. Mirrors gemm_qk_cofda_emu.h but for the SECOND attention GEMM.
//
// P is the FP8 softmax output, staged to smem by the caller (the register
// fragment is warpgroup-distributed, so a thread cannot read P[m, all k] from
// registers). V is the FP8 value tile in smem (transposed: sV(n,k) == V[k,n]).
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
```

- [ ] **Step 3: Write the failing standalone unit test `hopper/test_pv_cofda_emu.cu`**

Tests the seeded numeric core and the cross-block (two-block) accumulation semantics against an FP32 reference. Does not exercise the CuTe partition (covered by the kernel pytest in Task 8).

```cpp
// hopper/test_pv_cofda_emu.cu
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_fp8.h>
#include "cair_emu/cair_fp8_cofda_mma.cuh"
#include "gemm_qk_cofda_emu.h"  // cofda_dot_acc<F>, fp8_bits

// One output cell O[m][n] = Σ_k P[m][k]·V[k][n], computed in TWO blocks of
// width B each, seeding the second block's accumulator with the first block's
// result — the cross-block continuous-accumulator path the kernel uses.
template <int F>
__global__ void run_pv(const __nv_fp8_e4m3* P, const __nv_fp8_e4m3* V,
                       float* O, int M, int N, int Kfull, int B) {
  int m = blockIdx.x, n = threadIdx.x;
  if (m < M && n < N) {
    float acc = 0.f;
    for (int base = 0; base < Kfull; base += B) {
      auto fa = [&] __device__ (int k) { return fp8_bits(P[m * Kfull + (base + k)]); };
      auto fb = [&] __device__ (int k) { return fp8_bits(V[n * Kfull + (base + k)]); };
      acc = cofda_dot_acc<F>(fa, fb, B, acc);  // seed with running acc
    }
    O[m * N + n] = acc;
  }
}

int main() {
  const int M = 4, N = 4, Kfull = 64, B = 32;  // two CHUNK=32 blocks
  std::vector<__nv_fp8_e4m3> hP(M * Kfull), hV(N * Kfull);
  std::vector<float> O_ref(M * N, 0.f);
  for (int i = 0; i < M * Kfull; ++i) hP[i] = __nv_fp8_e4m3(0.5f + 0.1f * (i % 7));
  for (int i = 0; i < N * Kfull; ++i) hV[i] = __nv_fp8_e4m3(2.0f * (-0.3f + 0.05f * (i % 5)));
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < Kfull; ++k)
        O_ref[m * N + n] += float(hP[m * Kfull + k]) * float(hV[n * Kfull + k]);

  __nv_fp8_e4m3 *dP, *dV; float* dO;
  cudaMalloc(&dP, hP.size() * sizeof(__nv_fp8_e4m3));
  cudaMalloc(&dV, hV.size() * sizeof(__nv_fp8_e4m3));
  cudaMalloc(&dO, M * N * sizeof(float));
  cudaMemcpy(dP, hP.data(), hP.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
  cudaMemcpy(dV, hV.data(), hV.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);

  // F=25 ≈ FP32: must agree closely with the FP32 reference.
  run_pv<25><<<M, N>>>(dP, dV, dO, M, N, Kfull, B);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { printf("CUDA error (F25): %s\n", cudaGetErrorString(err)); return 1; }
  cudaDeviceSynchronize();
  std::vector<float> O25(M * N);
  cudaMemcpy(O25.data(), dO, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  double max_abs = 0.0;
  for (int i = 0; i < M * N; ++i) max_abs = std::fmax(max_abs, std::fabs(O25[i] - O_ref[i]));
  printf("F25 vs FP32 ref: max|diff| = %.6f\n", max_abs);
  if (max_abs > 1e-2) { printf("FAIL: F25 should track FP32 reference\n"); return 1; }

  // F=13 restricted accumulator should produce a measurable, bounded divergence.
  run_pv<13><<<M, N>>>(dP, dV, dO, M, N, Kfull, B);
  err = cudaGetLastError();
  if (err != cudaSuccess) { printf("CUDA error (F13): %s\n", cudaGetErrorString(err)); return 1; }
  cudaDeviceSynchronize();
  std::vector<float> O13(M * N);
  cudaMemcpy(O13.data(), dO, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  double max_div = 0.0;
  for (int i = 0; i < M * N; ++i) max_div = std::fmax(max_div, std::fabs(O13[i] - O25[i]));
  printf("F13 vs F25: max|diff| = %.6f\n", max_div);
  if (max_div <= 0.0) { printf("FAIL: F13 should diverge from F25\n"); return 1; }

  cudaFree(dP); cudaFree(dV); cudaFree(dO);
  printf("PASS\n");
  return 0;
}
```

- [ ] **Step 4: Compile and run; verify it passes**

Run (from `hopper/`):
```bash
cd /workspace/IDS-flash-attention/hopper
nvcc -std=c++17 --extended-lambda -arch=sm_90a -I. -I../csrc/cutlass/include \
  -o /tmp/test_pv_cofda test_pv_cofda_emu.cu && /tmp/test_pv_cofda
```
Expected: prints `F25 vs FP32 ref: max|diff| = …` (< 1e-2), `F13 vs F25: max|diff| = …` (> 0), then `PASS`, exit 0.

- [ ] **Step 5: Commit**

```bash
cd /workspace/IDS-flash-attention
git add hopper/gemm_qk_cofda_emu.h hopper/gemm_pv_cofda_emu.h hopper/test_pv_cofda_emu.cu
git commit -m "feat(pv-emu): seeded CoFDA core + gemm_pv_cofda_emu device fn + unit test"
```

---

## Task 2: Params field + runtime validation + torch schema

**Files:**
- Modify: `hopper/flash.h:145-147`
- Modify: `hopper/flash_api.cpp:750-751`, `:1046-1058`
- Modify: `hopper/flash_api_torch_lib.cpp:59-61`, `:133-135`

- [ ] **Step 1: Add fields to `Flash_fwd_params` (`hopper/flash.h`)**

After the `qk_emu_fbits` line (`:147`), add:
```cpp
    bool pv_emu_enabled;   // if true, replace the PV WGMMA with software CoFDA emulation (default false)
    int  pv_emu_fbits;     // CoFDA fractional accumulator bits, 13 or 25 (only used when pv_emu_enabled)
```

- [ ] **Step 2: Add `mha_fwd` parameters (`hopper/flash_api.cpp`)**

After `int64_t qk_emu_fbits` (`:751`), extend the signature:
```cpp
        int64_t qk_emu_fbits,
        bool pv_emu_enabled,
        int64_t pv_emu_fbits
```

- [ ] **Step 3: Set + validate in `mha_fwd` (`hopper/flash_api.cpp`)**

After the QK validation block (after `:1056`), add:
```cpp
    params.pv_emu_enabled = pv_emu_enabled;
    params.pv_emu_fbits = static_cast<int>(pv_emu_fbits);
    if (pv_emu_enabled) {
        TORCH_CHECK(q_type == at::ScalarType::Float8_e4m3fn,
                    "pv_emu_enabled requires e4m3 (Float8_e4m3fn) query/key dtype");
        TORCH_CHECK(get_current_arch() == 90,
                    "pv_emu_enabled requires SM90 (Hopper)");
        TORCH_CHECK(pv_emu_fbits == 13 || pv_emu_fbits == 25,
                    "pv_emu_fbits must be 13 or 25, got ", pv_emu_fbits);
        TORCH_CHECK(params.d == 128,
                    "pv_emu_enabled is only supported for head_dim == 128 (the only "
                    "validated config)");
    }
```
> Use the **same** `q_type` / arch accessor the adjacent QK block uses (copy its exact expressions — verify the QK block's variable names at `:1048-1056` and match them).

- [ ] **Step 4: Register in the torch-library schema (`hopper/flash_api_torch_lib.cpp`)**

In the C++ `mha_fwd(...)` declaration (after `:61`):
```cpp
        int64_t qk_emu_fbits = 25,
        bool pv_emu_enabled = false,
        int64_t pv_emu_fbits = 25
```
In the `ops.def("fwd(...)` string schema (after `:135`, the `qk_emu_fbits=25` line):
```cpp
            "    int      qk_emu_fbits=25,"
            "    bool     pv_emu_enabled=False,"
            "    int      pv_emu_fbits=25) -> Tensor[]");
```
> The `-> Tensor[]` terminator must move to the new last line. Keep the previous line's trailing comma.

- [ ] **Step 5: Commit**

```bash
cd /workspace/IDS-flash-attention
git add hopper/flash.h hopper/flash_api.cpp hopper/flash_api_torch_lib.cpp
git commit -m "feat(pv-emu): add pv_emu_enabled/pv_emu_fbits param, validation, torch schema"
```

---

## Task 3: Thread template axes through the launch template

**Files:**
- Modify: `hopper/flash_fwd_launch_template.h:31`, `:56`, `:66`, `:96`, `:257-279`

- [ ] **Step 1: Add `UsePVEmu`/`PVEmuFbits` to `run_flash_fwd` (`:31`)**

Extend the template parameter list (currently ends `..., bool UseQKEmu=false, int QKEmuFbits=25>`):
```cpp
          , bool UseQKEmu=false, int QKEmuFbits=25, bool UsePVEmu=false, int PVEmuFbits=25>
```

- [ ] **Step 2: Gate `IntraWGOverlap` and `UsePersistentScheduler` off for PV emu (`:56`, `:96`)**

`:56` becomes:
```cpp
    static constexpr bool IntraWGOverlap = !UseQKEmu && !UsePVEmu && std::get<3>(kBlockMN_RS_IntraWGOverlap);
```
`:96` becomes:
```cpp
    static constexpr bool UsePersistentScheduler = !UseQKEmu && !UsePVEmu && (Arch >= 90 ? !(Split && !Varlen) : ((Is_causal && !Varlen) || (Varlen && Split)));
```
> Routes the PV emu through the synchronous non-overlap path (the only PV gemm site we instrument is `:1428`), exactly as the QK emu did.

- [ ] **Step 3: Pass the axes into `CollectiveMainloopFwdSm90` (`:66`)**

Append `, UsePVEmu, PVEmuFbits` to the `CollectiveMainloopFwdSm90<...>` template argument list (after `QKEmuFbits`):
```cpp
        flash::CollectiveMainloopFwdSm90<kStages, ClusterShape, TileShape_MNK, kHeadDimV, Element, float, cutlass::arch::Sm90, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV, HasQv, MmaPV_is_RS, IntraWGOverlap, PackGQA, Split, V_colmajor, ElementS, kBlockH, DisableFP8TwoLevel, UseQKEmu, QKEmuFbits, UsePVEmu, PVEmuFbits>,
```

- [ ] **Step 4: Select the PV axes at the call site (`:257-279`)**

Inside the `BOOL_SWITCH(params.fp8_no_two_level_accum, DisableFP8TwoLevel, …)` block, replace the QK-only dispatch (`:262-273`) with a nested QK×PV dispatch. Both emus are independent booleans, each with its own fbits, so the cross product is 3×3 minus invalid f-bits (only 13/25 reachable, validated in `mha_fwd`). Use this exact structure:

```cpp
                                            // QK and PV emulation are independent axes; head_dim==128/e4m3/sm90
                                            // is enforced in mha_fwd, so only these branches are reachable here.
                                            auto dispatch_pv = [&](auto UseQKEmu_c, auto QKFbits_c) {
                                                constexpr bool UseQKEmu = decltype(UseQKEmu_c)::value;
                                                constexpr int  QKEmuFbits = decltype(QKFbits_c)::value;
                                                if (params.pv_emu_enabled && params.pv_emu_fbits == 13) {
                                                    run_flash_fwd<Arch, kHeadDim, kHeadDimV, ClusterM, T, T_out, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV && Varlen, HasQv, PackGQA, Split, V_colmajor, Use_one_mma_wg, kBlockH, DisableFP8TwoLevel, UseQKEmu, QKEmuFbits, /*UsePVEmu=*/true, /*PVEmuFbits=*/13>(params, stream);
                                                } else if (params.pv_emu_enabled) {  // pv f_bits == 25
                                                    run_flash_fwd<Arch, kHeadDim, kHeadDimV, ClusterM, T, T_out, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV && Varlen, HasQv, PackGQA, Split, V_colmajor, Use_one_mma_wg, kBlockH, DisableFP8TwoLevel, UseQKEmu, QKEmuFbits, /*UsePVEmu=*/true, /*PVEmuFbits=*/25>(params, stream);
                                                } else {
                                                    run_flash_fwd<Arch, kHeadDim, kHeadDimV, ClusterM, T, T_out, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV && Varlen, HasQv, PackGQA, Split, V_colmajor, Use_one_mma_wg, kBlockH, DisableFP8TwoLevel, UseQKEmu, QKEmuFbits, /*UsePVEmu=*/false, /*PVEmuFbits=*/25>(params, stream);
                                                }
                                            };
                                            if (params.qk_emu_enabled && params.qk_emu_fbits == 13) {
                                                dispatch_pv(std::true_type{}, std::integral_constant<int, 13>{});
                                            } else if (params.qk_emu_enabled) {  // qk f_bits == 25
                                                dispatch_pv(std::true_type{}, std::integral_constant<int, 25>{});
                                            } else {
                                                dispatch_pv(std::false_type{}, std::integral_constant<int, 25>{});
                                            }
```
> This replaces the body between the `BOOL_SWITCH(params.fp8_no_two_level_accum, …, [&] {` open and its matching close that currently holds `:265-273`. The non-FP8 `else` branch at `:279` (which forces all emu axes false) gains `, /*UsePVEmu=*/false, /*PVEmuFbits=*/25` appended to its `run_flash_fwd<…>` argument list.

- [ ] **Step 5: Commit (compiles only after Task 4 adds the collective template params — commit together if needed)**

```bash
cd /workspace/IDS-flash-attention
git add hopper/flash_fwd_launch_template.h
git commit -m "feat(pv-emu): thread UsePVEmu/PVEmuFbits through run_flash_fwd dispatch"
```

---

## Task 4: Mainloop template axes + emu-only P smem staging

**Files:**
- Modify: `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:33-35`, `:49-55`, `:320`, `:361-371`, `:1033-1040`

- [ ] **Step 1: Add the template parameters (`:33-35`)**

Extend the `CollectiveMainloopFwdSm90` template parameter list (currently ends `..., bool DisableFP8TwoLevel_=false, bool UseQKEmu_=false, int QKEmuFbits_=25>`):
```cpp
        bool DisableFP8TwoLevel_=false, bool UseQKEmu_=false, int QKEmuFbits_=25,
        bool UsePVEmu_=false, int PVEmuFbits_=25>
```

- [ ] **Step 2: Bind the constexpr members + static_assert (after `:49-51`, the `UseQKEmu` block)**

```cpp
    static constexpr bool UsePVEmu = UsePVEmu_;
    static constexpr int  PVEmuFbits = PVEmuFbits_;
    static_assert(!UsePVEmu || cute::is_same_v<Element, cutlass::float_e4m3_t>,
                  "PV CoFDA emulation is e4m3-only");
```

- [ ] **Step 3: Make `SmemP_t` non-empty when `UsePVEmu` (`:320`)**

```cpp
    using SmemP_t = std::conditional_t<MmaPV_is_RS && !UsePVEmu, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>, SmemAlignmentP>>;
```

- [ ] **Step 4: Add an emu-only TransposeV storage variant with `smem_p` (`:361-371`)**

The non-emu FP8 path must stay byte-identical (the existing `TensorStorageTransposeV` deliberately omits `smem_p` — adding it unconditionally can bump smem and break the kernel). Add a separate struct and select it only when `UsePVEmu`. After `TensorStorageTransposeV` (`:369`), add:
```cpp
    struct TensorStorageTransposeVWithP : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentV, SmemAlignmentP), _0> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVtMma>, SmemAlignmentV> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVt> smem_vt;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
        SmemQv_t smem_qv;
        SmemScale_t smem_scale;
        SmemP_t smem_p;
        cute::array_aligned<ElementSAux, cute::cosize_v<SmemLayoutSAux>, 128> smem_s_aux;
    };
```
Change the `TensorStorage` alias (`:371`):
```cpp
    using TensorStorage = std::conditional_t<!Transpose_V, TensorStorageNoTranspose,
        std::conditional_t<UsePVEmu, TensorStorageTransposeVWithP, TensorStorageTransposeV>>;
```

- [ ] **Step 5: Wire `sP` to real smem when `UsePVEmu` (`:1033-1040`)**

```cpp
        Tensor sP = [&] {
            if constexpr (MmaPV_is_RS && !UsePVEmu) {
                // placeholder: smem_q is unused as P storage on the pure-RS path
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutP{});
            } else {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutP{});
            }
        }();
```

- [ ] **Step 6: Verify it compiles (full extension build — see Task 8 for the command). Commit.**

```bash
cd /workspace/IDS-flash-attention
git add hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
git commit -m "feat(pv-emu): add UsePVEmu/PVEmuFbits axes + emu-only P smem staging"
```

---

## Task 5: PV emu branch at the gemm site + synchronization

**Files:**
- Modify: `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:26` (include), `:1422-1435` (the non-overlap PV gemm + release)

- [ ] **Step 1: Include the PV emu header (`:26`, next to the QK include)**

```cpp
#include "gemm_pv_cofda_emu.h"
```

- [ ] **Step 2: Stage P + branch the PV gemm + sync V before release (`:1422-1435`)**

In the non-overlap `fwd_step`, the relevant region currently is:
```cpp
                if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
                if constexpr (!Is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }
                if constexpr (!MmaPV_is_RS && !MmaPV_use_RS_WG1) { arrive_on_P_write_barrier(); }
                if constexpr (!HasQv) { consumer_wait(pipeline_v, smem_pipe_read); }
                warp_scheduler_barrier_sync();
                if constexpr (!MmaPV_use_RS_WG1) {
                    flash::gemm</*zero_init=*/Is_first_iter, /*wg_wait=*/-1, /*SwapAB=*/false, /*M_slice=*/-1, /*DisableFP8TwoLevel=*/DisableFP8TwoLevel>(tiled_mma_pv, cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP), tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                } else {
                    TiledMmaPV_RS tiled_mma_pv_rs;
                    flash::gemm</*zero_init=*/Is_first_iter, /*wg_wait=*/-1, /*SwapAB=*/false, /*M_slice=*/-1, /*DisableFP8TwoLevel=*/DisableFP8TwoLevel>(tiled_mma_pv_rs, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                }
                if constexpr (!MmaPV_is_RS && MmaPV_use_RS_WG1) { arrive_on_P_write_barrier(); }
                warpgroup_wait<0>();
                pipeline_v.consumer_release(smem_pipe_read);  // release V
```
Replace it with (adds: force-stage P for emu, warpgroup sync after P write, emu branch, per-WG V race barrier):
```cpp
                if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
                if constexpr (UsePVEmu) {
                    // P is register-source on this config; the emu needs arbitrary P[m,k]
                    // from any thread, so stage P to smem regardless of MmaPV_is_RS.
                    write_P_to_smem(tOrP);
                    // Full-warpgroup ordering: the emu reads sP(m,k) written by OTHER threads
                    // in this warpgroup (the P-fragment owner differs from the O-fragment
                    // owner), so __syncwarp is insufficient — use a per-WG NamedBarrier.
                    cutlass::arch::fence_view_async_shared();
                    cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup,
                        static_cast<uint32_t>(FwdNamedBarriers::AppendKV) - 1 + flash::canonical_warp_group_idx_nosync());
                }
                if constexpr (!Is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }
                if constexpr (!MmaPV_is_RS && !MmaPV_use_RS_WG1) { arrive_on_P_write_barrier(); }
                if constexpr (!HasQv) { consumer_wait(pipeline_v, smem_pipe_read); }
                warp_scheduler_barrier_sync();
                if constexpr (UsePVEmu) {
                    // Software CoFDA emulation of O=P·V (synchronous; replaces the PV WGMMA).
                    // tOrO is seeded inside the emu (ZeroInit==Is_first_iter); rescale above
                    // already applied for non-first iters.
                    Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
                    Tensor sV_pi = cute::as_position_independent_swizzle_tensor(sV);
                    flash::gemm_pv_cofda_emu<PVEmuFbits, /*ZeroInit=*/Is_first_iter>(
                        tiled_mma_pv, sP_pi, sV_pi(_, _, smem_pipe_read.index()), tOrO, thread_idx);
                } else if constexpr (!MmaPV_use_RS_WG1) {
                    flash::gemm</*zero_init=*/Is_first_iter, /*wg_wait=*/-1, /*SwapAB=*/false, /*M_slice=*/-1, /*DisableFP8TwoLevel=*/DisableFP8TwoLevel>(tiled_mma_pv, cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP), tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                } else {
                    TiledMmaPV_RS tiled_mma_pv_rs;
                    flash::gemm</*zero_init=*/Is_first_iter, /*wg_wait=*/-1, /*SwapAB=*/false, /*M_slice=*/-1, /*DisableFP8TwoLevel=*/DisableFP8TwoLevel>(tiled_mma_pv_rs, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                }
                if constexpr (!MmaPV_is_RS && MmaPV_use_RS_WG1) { arrive_on_P_write_barrier(); }
                if constexpr (!UsePVEmu) {
                    warpgroup_wait<0>();
                } else {
                    // The emu reads V via synchronous LDS; consumer_release is COLLECTIVE,
                    // so a straggler thread can still be mid-read when the V producer reuses
                    // the stage (the QK long-context race, applied to V). Per-WG NamedBarrier
                    // so every thread finished reading V before this WG releases the stage.
                    cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup,
                        static_cast<uint32_t>(FwdNamedBarriers::AppendKV) - 1 + flash::canonical_warp_group_idx_nosync());
                }
                pipeline_v.consumer_release(smem_pipe_read);  // release V
```
> Notes: (1) When `MmaPV_is_RS` (our config) `write_P_to_smem` is normally skipped; the new `if constexpr (UsePVEmu)` block force-stages it. The two `write_P_to_smem(tOrP)` calls cannot both fire — the first is `!MmaPV_is_RS`, the second is `UsePVEmu` (which only matters when `MmaPV_is_RS`). (2) The `AppendKV` barrier id is idle in the main KV loop (established by the QK fix) and is reused sequentially (P-sync → V-sync) within one iteration, which is safe. When BOTH QK and PV emu are on, the QK K-release barrier (`:1397`) and these PV barriers are at distinct points in the iteration and never live simultaneously.

- [ ] **Step 3: Build + run the kernel pytest (Task 8) to verify. Commit.**

```bash
cd /workspace/IDS-flash-attention
git add hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
git commit -m "feat(pv-emu): gemm-site emu branch + P-write/V-release warpgroup barriers"
```

---

## Task 6: Python interface kwargs (IDS flash-attention)

**Files:**
- Modify: `hopper/flash_attn_interface.py` (mirror every `qk_emu_enabled`/`qk_emu_fbits` occurrence: `:57-58`, `:109-110`, `:278-279`, `:308-309`, `:379-380`, `:413-414`, `:535-536`, `:601-602`, `:630-631`, `:657-658`)

- [ ] **Step 1: Add `pv_emu_enabled`/`pv_emu_fbits` next to every QK pair**

For each location where `qk_emu_enabled` / `qk_emu_fbits` appear (function signatures with defaults, and call-site forwarding), add the parallel `pv_emu_enabled` / `pv_emu_fbits` immediately after. Signature defaults:
```python
        qk_emu_enabled=False,
        qk_emu_fbits=25,
        pv_emu_enabled=False,
        pv_emu_fbits=25,
```
Positional forwarding to the C++ op (e.g. `:109-110`, `:601-602`):
```python
        qk_emu_enabled,
        qk_emu_fbits,
        pv_emu_enabled,
        pv_emu_fbits,
```
Keyword forwarding (e.g. `:308-309`, `:413-414`):
```python
            qk_emu_enabled=qk_emu_enabled,
            qk_emu_fbits=qk_emu_fbits,
            pv_emu_enabled=pv_emu_enabled,
            pv_emu_fbits=pv_emu_fbits,
```
> The C++ `mha_fwd` argument ORDER is `…, qk_emu_enabled, qk_emu_fbits, pv_emu_enabled, pv_emu_fbits` (Task 2). Keep positional forwarding in that order.

- [ ] **Step 2: Commit**

```bash
cd /workspace/IDS-flash-attention
git add hopper/flash_attn_interface.py
git commit -m "feat(pv-emu): thread pv_emu kwargs through the python fwd interface"
```

---

## Task 7: vLLM env vars + backend + interface threading

**Files:**
- Modify: `vllm-mma/vllm/envs.py:176-177`, `:1302-1303`
- Modify: `vllm-mma/vllm/ids_flash_attn/flash_attn_interface.py:210-211`, `:370-371`
- Modify: `vllm-mma/vllm/v1/attention/backends/ids_flash_attn.py:666-667`

- [ ] **Step 1: Declare env vars (`vllm-mma/vllm/envs.py`)**

After the QK declarations (`:177`):
```python
    VLLM_USE_CAIR_PV_EMU: bool = False
    VLLM_CAIR_PV_EMU_FBITS: int = 25
```
After the QK lambdas (`:1303`):
```python
    "VLLM_USE_CAIR_PV_EMU": lambda: bool(int(os.getenv("VLLM_USE_CAIR_PV_EMU", "0"))),
    "VLLM_CAIR_PV_EMU_FBITS": lambda: int(os.getenv("VLLM_CAIR_PV_EMU_FBITS", "25")),
```

- [ ] **Step 2: Add kwargs to the IDS interface (`vllm-mma/vllm/ids_flash_attn/flash_attn_interface.py`)**

After the signature QK pair (`:211`):
```python
    pv_emu_enabled: bool = False,
    pv_emu_fbits: int = 25,
```
After the forwarding QK pair (`:371`):
```python
            qk_emu_enabled,
            qk_emu_fbits,
            pv_emu_enabled,
            pv_emu_fbits,
```
> Match the forwarding style already present at `:370-371` (positional vs keyword) and keep the QK→PV order.

- [ ] **Step 3: Read env vars in the backend (`vllm-mma/vllm/v1/attention/backends/ids_flash_attn.py`)**

After the QK pass-through (`:667`):
```python
                    qk_emu_enabled=envs.VLLM_USE_CAIR_QK_EMU,
                    qk_emu_fbits=envs.VLLM_CAIR_QK_EMU_FBITS,
                    pv_emu_enabled=envs.VLLM_USE_CAIR_PV_EMU,
                    pv_emu_fbits=envs.VLLM_CAIR_PV_EMU_FBITS,
```

- [ ] **Step 4: Commit**

```bash
cd /workspace/vllm-mma
git add vllm/envs.py vllm/ids_flash_attn/flash_attn_interface.py vllm/v1/attention/backends/ids_flash_attn.py
git commit -m "feat(pv-emu): VLLM_USE_CAIR_PV_EMU env var + backend/interface threading"
```

---

## Task 8: Build, bit-exact/determinism pytest, WikiText config

**Files:**
- Create: `vllm-mma/tests/kernels/attention/test_pv_cofda_emu.py`
- Create: `vllm-mma/cair/experiments/configs/ids_fa_pv_emu_wikitext.yaml`

- [ ] **Step 1: Rebuild the extension (after all kernel changes)**

```bash
cd /workspace/vllm-mma
MAX_JOBS=32 TORCH_CUDA_ARCH_LIST="9.0" uv pip install -e . -v --torch-backend cu130 2>&1 | tee build.log
```
Expected: build succeeds. If it fails with `invalid argument` at launch, the P-staging smem pushed the kernel over budget — reduce `kStages` for the emu config or report back (this is the tracked smem risk).
> Build note: if a `VLLM_IDS_FLASH_ATTN_SRC_DIR` build is used and the FA2 target fails to compile due to a missing submodule, run `git -C /workspace/IDS-flash-attention submodule update --init csrc/cutlass`.

- [ ] **Step 2: Write the bit-exact + determinism pytest**

Create `vllm-mma/tests/kernels/attention/test_pv_cofda_emu.py`. Validation: with QK fixed to hardware (`qk_emu_enabled=False`) and PV matched (`fp8_no_two_level_accum=True` on both runs), `pv_emu(F=13)` must equal the hardware `no_two_level` PV MMA bit-exactly AND be deterministic across runs, including long context.

```python
import pytest
import torch
from vllm.ids_flash_attn.flash_attn_interface import flash_attn_varlen_func

HEAD_SIZES = [128]
SEQLENS = [512, 2048, 4096]

def _run(q, k, v, dq, s, causal, **kw):
    cu = torch.tensor([0, s], dtype=torch.int32, device="cuda")
    o = flash_attn_varlen_func(
        q, k, v, max_seqlen_q=s, cu_seqlens_q=cu, max_seqlen_k=s, cu_seqlens_k=cu,
        softmax_scale=q.shape[-1] ** -0.5, causal=causal, fa_version=3,
        q_descale=dq, k_descale=dq, v_descale=dq, **kw)
    return (o[0] if isinstance(o, (tuple, list)) else o).float()

@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("seqlen", SEQLENS)
@pytest.mark.parametrize("causal", [True, False])
def test_pv_emu_f13_bit_exact_with_no_two_level(head_size, seqlen, causal):
    dev, nq, nkv, d, s = "cuda", 8, 2, head_size, seqlen
    torch.manual_seed(0)
    mk = lambda nh: torch.randn(s, nh, d, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    q, k, v = mk(nq), mk(nkv), mk(nkv)
    dq = torch.ones(1, nkv, device=dev)
    # Reference: hardware no_two_level PV (QK also hardware no_two_level).
    ref = _run(q, k, v, dq, s, causal, fp8_no_two_level_accum=True)
    # PV emu F=13, QK left on hardware, PV path matched (no two-level).
    emu1 = _run(q, k, v, dq, s, causal, fp8_no_two_level_accum=True,
                pv_emu_enabled=True, pv_emu_fbits=13)
    emu2 = _run(q, k, v, dq, s, causal, fp8_no_two_level_accum=True,
                pv_emu_enabled=True, pv_emu_fbits=13)
    assert torch.equal(emu1, emu2), f"non-deterministic at s={s}, causal={causal}"
    assert torch.equal(emu1, ref), (
        f"pv_emu(F=13) not bit-exact with hardware no_two_level at s={s}, causal={causal}; "
        f"mismatches={(emu1 != ref).sum().item()}")

@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("causal", [True, False])
def test_pv_emu_f13_vs_f25_diverges(head_size, causal):
    dev, nq, nkv, d, s = "cuda", 8, 2, head_size, 2048
    torch.manual_seed(0)
    mk = lambda nh: torch.randn(s, nh, d, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    q, k, v = mk(nq), mk(nkv), mk(nkv)
    dq = torch.ones(1, nkv, device=dev)
    f13 = _run(q, k, v, dq, s, causal, fp8_no_two_level_accum=True, pv_emu_enabled=True, pv_emu_fbits=13)
    f25 = _run(q, k, v, dq, s, causal, fp8_no_two_level_accum=True, pv_emu_enabled=True, pv_emu_fbits=25)
    assert (f13 != f25).any(), "F=13 and F=25 should differ (the precision signal)"
```

- [ ] **Step 3: Run the pytest; verify it passes**

```bash
cd /workspace/vllm-mma
.venv/bin/python -m pytest tests/kernels/attention/test_pv_cofda_emu.py -v
```
Expected: all parametrizations PASS. If `test_pv_emu_f13_bit_exact_with_no_two_level` fails only at long seqlen (s≥2048) and is non-deterministic, the V race barrier (Task 5) is not covering the hazard — revisit the barrier id / granularity (see the QK race doc `vllm-mma/cair/experiments/docs/qk-emu-longcontext-race-debug.md`). If it fails deterministically at all seqlens, the cross-block seeded-accumulator model (§5 of the spec) does not match the hardware accumulator — capture max|diff| and the differing (m,n) pattern before changing the model.

- [ ] **Step 4: Add the WikiText PPL config**

Create `vllm-mma/cair/experiments/configs/ids_fa_pv_emu_wikitext.yaml`, copying the QK config (`ids_fa_qk_emu_wikitext.yaml`) and swapping the emu rows to PV. Four rows: `ref-two-level`, `ref1-no-two-level`, `cair-pv-emu-f13`, `cair-pv-emu-f25`. The emu rows set `cair_pv_emu: true` / `cair_pv_emu_fbits: {13,25}` (→ `VLLM_USE_CAIR_PV_EMU=1` / `VLLM_CAIR_PV_EMU_FBITS`), `fp8_two_level_accum: false`, `cair_qk_emu: false` (QK stays hardware). Keep `model: nvidia/Llama-3.1-8B-Instruct-FP8`, `kv_cache_dtype: fp8_e4m3`, `attention_backend: IDS_FLASH_ATTN`, `force_fp8_kernel: cutlass`, `max_model_len: 16384`, `enforce_eager: true`, `tasks: wikitext / word_perplexity / limit: 1` exactly as the QK config.
> Verify the runner maps `cair_pv_emu`/`cair_pv_emu_fbits` to the new env vars; if the runner has an explicit config→env mapping (as the QK keys do), add `pv` entries mirroring the `qk` ones.

- [ ] **Step 5: Smoke-run the PPL sweep (optional but recommended)**

```bash
cd /workspace/vllm-mma/cair/experiments
./.venv/bin/python -m runner.sweep configs/ids_fa_pv_emu_wikitext.yaml --baseline
```
Expected: four rows complete; `cair-pv-emu-f13` word-perplexity tracks `ref1-no-two-level` (PV emu reproduces the hardware no-two-level PV).

- [ ] **Step 6: Commit**

```bash
cd /workspace/vllm-mma
git add tests/kernels/attention/test_pv_cofda_emu.py cair/experiments/configs/ids_fa_pv_emu_wikitext.yaml
git commit -m "test(pv-emu): bit-exact + determinism pytest and WikiText PPL config"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** §4 architecture → Tasks 2-7; §2/§4 P-to-smem staging → Task 4; §5 cross-block seeded accumulator → Task 1 (`cofda_dot_acc`) + Task 5 (`ZeroInit`); §6 validation criterion → Task 8 pytest; §7 race/sync → Task 5; §8 tests → Tasks 1 & 8; §3 toggle independence → Tasks 2/3/7 (separate axis); build gating §4(e) → Task 3 (`UsePVEmu` only reachable for e4m3/sm90/d128). All covered.
- **Type consistency:** device fn `gemm_pv_cofda_emu<F, ZeroInit>` (Task 1) is called with matching `<PVEmuFbits, Is_first_iter>` (Task 5); `cofda_dot_acc<F>(fa, fb, D, acc_init)` signature consistent across Tasks 1 and 5; param names `pv_emu_enabled`/`pv_emu_fbits` consistent across flash.h → flash_api.cpp → torch schema → python → vLLM; template axis names `UsePVEmu`/`PVEmuFbits` consistent across launch template and mainloop; `sV(n,k)`/`sP(m,k)` indexing consistent with the established smem layouts.
- **Placeholder scan:** no TBD/TODO; every code step shows full code; build/test commands are concrete with expected output.
