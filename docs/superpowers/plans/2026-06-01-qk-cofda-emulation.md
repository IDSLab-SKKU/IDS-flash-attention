# QK CoFDA Emulation Variant — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runtime-selectable variant of the SM90 FP8 forward attention kernel that replaces the hardware `S = Q·Kᵀ` WGMMA with a software CoFDA emulation, so FP8 accumulator precision (`F ∈ {13,25}`) can be studied inside the real attention kernel.

**Architecture:** Mirror the existing `fp8_no_two_level_accum` → `DisableFP8TwoLevel` compile-time-axis threading. Add a `UseQKEmu` boolean axis + a compile-time `QKEmuFbits` selected at runtime, threaded python → `mha_fwd` → `Flash_fwd_params` → `BOOL_SWITCH` → `CollectiveMainloopFwdSm90` → an `if constexpr` branch at the QK gemm site that calls a new `flash::gemm_qk_cofda_emu<F>` device function. CoFDA headers are vendored from vLLM's `cair/` into `hopper/cair_emu/`. vLLM selects the variant via an env var, mirroring `VLLM_USE_CAIR_GEMM_FP8`.

**Tech Stack:** CUDA 13.0 / nvcc, CUTLASS CuTe (SM90 GMMA), PyTorch C++ extension (pybind11), Python (flash_attn interface + vLLM backend). Target: H100 sm_90a.

**Spec:** `docs/superpowers/specs/2026-06-01-qk-cofda-emulation-design.md`
**Reference notes:** `AI/GEMM_WGMMA_ANALYSIS.md`, `AI/SASS_PTX_MMA_ANALYSIS.md`

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `hopper/cair_emu/cair_types.cuh`, `cair_fp32_utils.cuh`, `cair_fp8_utils.cuh`, `cair_fp8_cofda_mma.cuh` | Vendored CoFDA device emulation (header-only, no torch dep) | Create (copy) |
| `hopper/cair_emu/PROVENANCE.md` | Record source + sync obligation | Create |
| `hopper/gemm_qk_cofda_emu.h` | `flash::gemm_qk_cofda_emu<F>` — per-element CoFDA over the QK accumulator | Create |
| `hopper/test_qk_cofda_emu.cu` | Standalone nvcc unit test (emu vs FP32 reference), fast TDD loop | Create |
| `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` | Add `UseQKEmu_`,`QKEmuFbits_` template axes; branch at QK gemm sites `:1217`,`:1356` | Modify |
| `hopper/flash_fwd_launch_template.h` | `BOOL_SWITCH(UseQKEmu)` + f_bits switch; pass axes to collective | Modify |
| `hopper/flash.h` | `qk_emu_enabled`, `qk_emu_fbits` fields on `Flash_fwd_params` | Modify |
| `hopper/flash_api.cpp` | `mha_fwd` params + set into `params` + validation | Modify |
| `hopper/flash_attn_interface.py` | thread kwargs to `flash_attn_3_cuda.fwd(...)` | Modify |
| `vllm-mma/vllm/v1/attention/backends/flash_attn.py` | read env vars, pass kwargs | Modify |
| `vllm-mma/vllm/envs.py` | declare `VLLM_USE_CAIR_QK_EMU`, `VLLM_CAIR_QK_EMU_FBITS` | Modify |

**Conventions to follow:** This repo uses large kernel headers and a single compile-time-axis threading pattern (`DisableFP8TwoLevel`). Do **not** restructure; mirror that pattern exactly. C++17 (`hopper/CMakeLists`/setup uses `-std=c++17`). 2-space indent in CUDA files.

---

## Task 1: Vendor CoFDA headers

**Files:**
- Create: `hopper/cair_emu/cair_types.cuh`, `hopper/cair_emu/cair_fp32_utils.cuh`, `hopper/cair_emu/cair_fp8_utils.cuh`, `hopper/cair_emu/cair_fp8_cofda_mma.cuh`
- Create: `hopper/cair_emu/PROVENANCE.md`

- [ ] **Step 1: Copy the four header-only CoFDA files verbatim**

```bash
cd /workspace/Project/IDS-flash-attention
mkdir -p hopper/cair_emu
SRC=/workspace/Project/vllm-mma/csrc/libtorch_stable/quantization/cair
cp "$SRC/cair_types.cuh" "$SRC/cair_fp32_utils.cuh" "$SRC/cair_fp8_utils.cuh" "$SRC/cair_fp8_cofda_mma.cuh" hopper/cair_emu/
```

- [ ] **Step 2: Verify the copied headers self-include correctly (no torch dependency)**

Run:
```bash
cd /workspace/Project/IDS-flash-attention/hopper/cair_emu
grep -l "torch\|ATen\|c10" *.cuh || echo "NO_TORCH_DEPS_OK"
```
Expected: `NO_TORCH_DEPS_OK` (the CoFDA device headers must not pull in torch). If any file references torch, it is the wrong file — re-check the source; only the four listed device headers are needed (NOT `cair_fp8_mm.cu`).

- [ ] **Step 3: Write provenance note**

Create `hopper/cair_emu/PROVENANCE.md`:
```markdown
# Vendored CoFDA headers

Source: `vllm-mma/csrc/libtorch_stable/quantization/cair/` (vLLM, ported from the
vllm-cair research fork). Copied 2026-06-01.

These are the device-side, header-only CoFDA emulation utilities used by
`hopper/gemm_qk_cofda_emu.h`. They are a VERBATIM copy — if the upstream CAIR
headers change, re-sync these. Do not edit them here except to fix include paths.

Public entry point used by flash-attention:
`fp8_cofda_mma<int F, int CHUNK=32>(DecodedFrag a, DecodedFrag b, float c) -> float`
(in `cair_fp8_cofda_mma.cuh`).
```

- [ ] **Step 4: Commit**

```bash
cd /workspace/Project/IDS-flash-attention
git add hopper/cair_emu
git commit -m "feat(qk-emu): vendor CoFDA device headers from vLLM cair"
```

---

## Task 2: `gemm_qk_cofda_emu` device function + standalone unit test (TDD)

This is the numerical core. It is built and tested in **isolation** with nvcc (fast loop), independent of the full flash build. The unit test compiles the emulation against a CPU FP32 reference for a single Q·Kᵀ tile.

**Files:**
- Create: `hopper/gemm_qk_cofda_emu.h`
- Test: `hopper/test_qk_cofda_emu.cu`

- [ ] **Step 1: Write the failing standalone test**

Create `hopper/test_qk_cofda_emu.cu`. It fills small Q `[M=4, D=64]` and K `[N=4, D=64]` with FP8 e4m3 values, computes a reference `S_ref[m][n] = Σ_k float(Q[m][k]) * float(K[n][k])` on host in FP32, runs a device kernel that calls `cofda_dot<F>` (the per-cell helper that `gemm_qk_cofda_emu` will use) for each `(m,n)`, and asserts agreement at `F=25`.

```cpp
// hopper/test_qk_cofda_emu.cu
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_fp8.h>
#include "cair_emu/cair_fp8_cofda_mma.cuh"  // provides fp8_cofda_mma / DecodedFrag / decode

// The header provides the fetch-callable core: cofda_dot<F>(fa, fb, D), where
// fa(k)/fb(k) return the raw uint8 FP8 byte at column k.
#include "gemm_qk_cofda_emu.h"

template <int F>
__global__ void run_dot(const __nv_fp8_e4m3* Q, const __nv_fp8_e4m3* K,
                        float* S, int M, int N, int D) {
  int m = blockIdx.x, n = threadIdx.x;
  if (m < M && n < N) {
    const __nv_fp8_e4m3* q_row = &Q[m * D];
    const __nv_fp8_e4m3* k_col = &K[n * D];
    auto fa = [&] __device__ (int k) { return fp8_bits(q_row[k]); };
    auto fb = [&] __device__ (int k) { return fp8_bits(k_col[k]); };
    S[m * N + n] = cofda_dot<F>(fa, fb, D);
  }
}

int main() {
  const int M = 4, N = 4, D = 64;
  std::vector<__nv_fp8_e4m3> hQ(M * D), hK(N * D);
  std::vector<float> S_ref(M * N, 0.f);
  for (int i = 0; i < M * D; ++i) hQ[i] = __nv_fp8_e4m3(0.5f + 0.1f * (i % 7));
  for (int i = 0; i < N * D; ++i) hK[i] = __nv_fp8_e4m3(-0.3f + 0.05f * (i % 5));
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < D; ++k)
        S_ref[m * N + n] += float(hQ[m * D + k]) * float(hK[n * D + k]);

  __nv_fp8_e4m3 *dQ, *dK; float* dS;
  cudaMalloc(&dQ, hQ.size()); cudaMalloc(&dK, hK.size()); cudaMalloc(&dS, M * N * sizeof(float));
  cudaMemcpy(dQ, hQ.data(), hQ.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK.data(), hK.size(), cudaMemcpyHostToDevice);
  run_dot<25><<<M, N>>>(dQ, dK, dS, M, N, D);
  cudaDeviceSynchronize();
  std::vector<float> S(M * N);
  cudaMemcpy(S.data(), dS, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  int fails = 0;
  for (int i = 0; i < M * N; ++i) {
    float a = S[i], b = S_ref[i];
    float tol = 1e-2f * (std::fabs(b) + 1.f);   // F=25 ≈ FP32: tight tolerance
    if (std::fabs(a - b) > tol) { printf("MISMATCH[%d] emu=%f ref=%f\n", i, a, b); ++fails; }
  }
  printf(fails ? "FAIL: %d mismatches\n" : "PASS\n", fails);
  return fails ? 1 : 0;
}
```

- [ ] **Step 2: Run the test to verify it fails to compile (helper not defined)**

Run:
```bash
cd /workspace/Project/IDS-flash-attention/hopper
nvcc -std=c++17 -arch=sm_90a -I. -o /tmp/test_qk_cofda /tmp/_unused 2>/dev/null; \
nvcc -std=c++17 --extended-lambda -arch=sm_90a -I. -o /tmp/test_qk_cofda test_qk_cofda_emu.cu
```
Expected: FAIL — link/compile error `cofda_dot` undefined (it is only forward-declared). This confirms the test exercises the not-yet-written helper.

- [ ] **Step 3: Implement `gemm_qk_cofda_emu.h` with the `cofda_dot` helper**

Create `hopper/gemm_qk_cofda_emu.h`. It provides (a) `cofda_dot<F>` — a per-cell decode-and-CoFDA over D, used by the unit test, and (b) `flash::gemm_qk_cofda_emu<F>` — the CuTe-fragment entry used by the kernel. `cofda_dot` decodes each FP8 operand and feeds CHUNK=32 windows to the vendored `fp8_cofda_mma<F,32>`.

```cpp
// hopper/gemm_qk_cofda_emu.h
#pragma once
#include <cstdint>
#include <cute/tensor.hpp>
#include "cair_emu/cair_fp8_cofda_mma.cuh"   // vllm::cair::{fp8_cofda_mma<F,CHUNK>, decode_operand, pack_decoded, DecodedFrag}

// --- (a) Numeric core, decoupled from operand LAYOUT. ---------------------------
// CAIR's fp8_cofda_mma expects DECODED operands (DecodedFrag over packed uint32),
// because in CAIR's own kernel decode happens once at smem load. In flash-attention
// the smem holds RAW, SWIZZLED FP8 — so we decode on the fly and, crucially, fetch
// each FP8 byte through a caller-supplied accessor `fa(k)`/`fb(k)` that returns the
// raw uint8 at column k. This keeps the CoFDA math identical while letting the caller
// own the (swizzled tensor vs flat array) addressing. fa/fb MUST return uint8_t.
template <int F, class FetchA, class FetchB>
__device__ float cofda_dot(FetchA fa, FetchB fb, int D) {
  constexpr int CHUNK = 32;
  assert(D % CHUNK == 0);  // CoFDA chunk size is fixed at 32; kHeadDim ∈ {64,128,192,256} all divisible.
  float acc = 0.f;
  uint32_t a_dec[CHUNK], b_dec[CHUNK];   // decoded operands in registers (NOT smem)
  for (int base = 0; base < D; base += CHUNK) {
    #pragma unroll
    for (int j = 0; j < CHUNK; ++j) {
      a_dec[j] = vllm::cair::pack_decoded(vllm::cair::decode_operand(fa(base + j)));
      b_dec[j] = vllm::cair::pack_decoded(vllm::cair::decode_operand(fb(base + j)));
    }
    vllm::cair::DecodedFrag a_frag{a_dec};
    vllm::cair::DecodedFrag b_frag{b_dec};
    acc = vllm::cair::fp8_cofda_mma<F, CHUNK>(a_frag, b_frag, acc);   // chunked accumulate w/ intermediate rounding
  }
  return acc;
}

// Reinterpret one FP8 element (cutlass float_e4m3_t or __nv_fp8_e4m3) as its raw byte.
template <class E> __device__ __forceinline__ uint8_t fp8_bits(E const& e) {
  return *reinterpret_cast<const uint8_t*>(&e);
}

namespace flash {
// --- (b) Kernel entry: replace the QK WGMMA. Reads raw FP8 Q/K from SWIZZLED smem. ---
// sQ_pi / sK_pi are position-independent swizzle views (cute::as_position_independent_
// swizzle_tensor) shaped (M, D) / (N, D). We index them as tensors — sQ_pi(m, k) —
// so the swizzle is applied correctly; we do NOT take a raw pointer and stride by hand.
template <int F, class TiledMma, class TensorSQ, class TensorSK, class TensorC>
CUTLASS_DEVICE void gemm_qk_cofda_emu(TiledMma const& tiled_mma,
                                      TensorSQ const& sQ_pi, TensorSK const& sK_pi,
                                      TensorC& tSrS, int thread_idx) {
  using namespace cute;
  // Logical (m,n) coordinate for each accumulator element this thread owns.
  // Mirrors the masking code at mainloop_fwd_sm90_tma_gmma_ws.hpp:1151-1154.
  auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
  Tensor cS = make_identity_tensor(make_shape(size<0>(sQ_pi), size<0>(sK_pi)));  // (M,N)->(m,n)
  Tensor tScS = thr_mma.partition_C(cS);                  // same layout/size as tSrS
  CUTE_STATIC_ASSERT_V(size(tScS) == size(tSrS));
  const int D = size<1>(sQ_pi);                            // kHeadDim
  #pragma unroll
  for (int i = 0; i < size(tSrS); ++i) {
    auto coord = tScS(i);                                  // (m, n)
    int m = get<0>(coord), n = get<1>(coord);
    // Swizzle-correct, decode-on-the-fly fetchers — this is the adaptation the
    // CAIR "expects decoded params" coupling requires for flash's raw smem.
    auto fa = [&] __device__ (int k) { return fp8_bits(sQ_pi(m, k)); };
    auto fb = [&] __device__ (int k) { return fp8_bits(sK_pi(n, k)); };
    tSrS(i) = cofda_dot<F>(fa, fb, D);
  }
}
}  // namespace flash
```

> **Note on symbol names:** the vendored header uses namespace `vllm::cair` with
> `decode_operand(uint8_t)->DecodedOperand`, `pack_decoded(DecodedOperand)->uint32_t`,
> `struct DecodedFrag { const uint32_t* words; }`, and `fp8_cofda_mma<int F,int CHUNK>`
> (verified in `cair_fp8_utils.cuh` / `cair_fp8_cofda_mma.cuh`). Use them verbatim.
> Compile lambdas with `--extended-lambda` (add to the test nvcc command and to the
> extension's nvcc flags if not already present).

- [ ] **Step 4: Run the unit test to verify it passes**

Run:
```bash
cd /workspace/Project/IDS-flash-attention/hopper
nvcc -std=c++17 --extended-lambda -arch=sm_90a -I. -o /tmp/test_qk_cofda test_qk_cofda_emu.cu && /tmp/test_qk_cofda
```
Expected: compiles, prints `PASS`. If `MISMATCH` at `F=25`, the decode/chunk path is wrong (NOT a precision artifact — F=25 should match FP32 closely). Debug `cofda_dot` against the vendored `fp8_cofda_mma` contract before proceeding.

- [ ] **Step 5: Add an F=13 divergence assertion (research signal is bounded, not zero)**

Append to `test_qk_cofda_emu.cu` `main()` before `return`: run `run_dot<13>` into a second buffer `S13`, and assert each `|S13[i] - S_ref[i]|` is **larger** than the F=25 error but still bounded (`< 0.5f * (|ref| + 1)`); print `F13_SIGNAL_OK` when so. Re-run Step 4's command; expect `PASS` and `F13_SIGNAL_OK`.

- [ ] **Step 6: Commit**

```bash
cd /workspace/Project/IDS-flash-attention
git add hopper/gemm_qk_cofda_emu.h hopper/test_qk_cofda_emu.cu
git commit -m "feat(qk-emu): add gemm_qk_cofda_emu device fn + standalone unit test"
```

---

## Task 3: Add `qk_emu` fields to `Flash_fwd_params`

**Files:**
- Modify: `hopper/flash.h:145` (next to `fp8_no_two_level_accum`)

- [ ] **Step 1: Add the fields**

In `hopper/flash.h`, immediately after the `bool fp8_no_two_level_accum;` line (`:145`), add:
```cpp
    bool qk_emu_enabled;   // if true, replace the QK WGMMA with software CoFDA emulation (default false)
    int  qk_emu_fbits;     // CoFDA fractional accumulator bits, 13 or 25 (only used when qk_emu_enabled)
```

- [ ] **Step 2: Verify it parses (header-only syntax check)**

Run:
```bash
cd /workspace/Project/IDS-flash-attention/hopper
nvcc -std=c++17 -arch=sm_90a -I. -I../csrc/cutlass/include -fsyntax-only flash.h 2>&1 | head -5 || echo "see errors"
```
Expected: no new errors referencing the two added lines. (Pre-existing unrelated include errors from a partial tree are acceptable; the two new lines must not be among them.)

- [ ] **Step 3: Commit**

```bash
git add hopper/flash.h
git commit -m "feat(qk-emu): add qk_emu_enabled/qk_emu_fbits to Flash_fwd_params"
```

---

## Task 4: Thread axes through `CollectiveMainloopFwdSm90` + branch at gemm sites

**Files:**
- Modify: `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:31-47` (template params + constexpr), `:1217`, `:1356` (gemm sites), and the `#include` block

- [ ] **Step 1: Add the include**

Near the other `#include`s at the top of `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`, add:
```cpp
#include "gemm_qk_cofda_emu.h"
```

- [ ] **Step 2: Add the two template axes**

Change the template header (`:31-34`) ending `..., int kBlockH_=1, bool DisableFP8TwoLevel_=false>` to also accept the new axes:
```cpp
        bool DisableFP8TwoLevel_=false, bool UseQKEmu_=false, int QKEmuFbits_=25>
```
And after `static constexpr bool DisableFP8TwoLevel = DisableFP8TwoLevel_;` (`:47`) add:
```cpp
    static constexpr bool UseQKEmu = UseQKEmu_;
    static constexpr int  QKEmuFbits = QKEmuFbits_;
    static_assert(!UseQKEmu || Is_FP8, "QK CoFDA emulation requires FP8 (e4m3) inputs");
```

- [ ] **Step 3: Branch at the IntraWGOverlap QK gemm site (`:1217`)**

Replace the single `flash::gemm<...>(tiled_mma_qk, tSrQ, tSrK(...), tSrS);` at `:1217` with:
```cpp
            if constexpr (UseQKEmu) {
                Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(sQ);
                Tensor sK_pi = cute::as_position_independent_swizzle_tensor(sK(_, _, smem_pipe_read.index()));
                flash::gemm_qk_cofda_emu<QKEmuFbits>(tiled_mma_qk, sQ_pi, sK_pi, tSrS, thread_idx);
            } else {
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/false, /*M_slice=*/-1, /*DisableFP8TwoLevel=*/DisableFP8TwoLevel>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
            }
```
> `sQ` / `sK` are defined at `:653-654`. Confirm `thread_idx` is in scope at this site (it is used for `partition_C` nearby at `:1152`); if the local is named differently here, use that name.

- [ ] **Step 4: Branch at the non-overlap QK gemm site (`:1356`)**

Apply the identical `if constexpr (UseQKEmu) { … } else { … }` wrapper around the `flash::gemm<...>` call at `:1356`, using `smem_pipe_read.index()` exactly as that site already does.

- [ ] **Step 5: Syntax-check the header compiles in isolation**

Run:
```bash
cd /workspace/Project/IDS-flash-attention/hopper
nvcc -std=c++17 -arch=sm_90a -I. -I../csrc/cutlass/include -fsyntax-only mainloop_fwd_sm90_tma_gmma_ws.hpp 2>&1 | grep -iE "UseQKEmu|gemm_qk_cofda|qk_emu" | head
```
Expected: no errors mentioning the new symbols. (Unrelated include-path errors from the partial cutlass tree are acceptable; new-symbol errors are not.)

- [ ] **Step 6: Commit**

```bash
git add hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
git commit -m "feat(qk-emu): add UseQKEmu/QKEmuFbits axes + gemm-site branch in fwd mainloop"
```

---

## Task 5: Thread axes through `run_flash_fwd` / `BOOL_SWITCH`

**Files:**
- Modify: `hopper/flash_fwd_launch_template.h:30` (template), `:56` (collective instantiation), `:238-244` (switch)

- [ ] **Step 1: Add template params to `run_flash_fwd`**

At `:30`, extend the template parameter list ending `..., int kBlockH=1, bool DisableFP8TwoLevel=false>` to:
```cpp
          ..., int kBlockH=1, bool DisableFP8TwoLevel=false, bool UseQKEmu=false, int QKEmuFbits=25>
```

- [ ] **Step 2: Pass them into the collective instantiation**

At `:56`, the `flash::CollectiveMainloopFwdSm90<...., DisableFP8TwoLevel>` instantiation: append the two new axes at the end:
```cpp
        flash::CollectiveMainloopFwdSm90<kStages, ClusterShape, TileShape_MNK, kHeadDimV, Element, float, cutlass::arch::Sm90, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV, HasQv, MmaPV_is_RS, IntraWGOverlap, PackGQA, Split, V_colmajor, ElementS, kBlockH, DisableFP8TwoLevel, UseQKEmu, QKEmuFbits>,
```

- [ ] **Step 3: Add the runtime→compile-time switch**

Replace the `BOOL_SWITCH(params.fp8_no_two_level_accum, DisableFP8TwoLevel, [&] { run_flash_fwd<...>(...); });` block (`:238-240`) with a nested switch that also selects `UseQKEmu` and the F-bits value. Only FP8 e4m3 + Arch 90 can enable emu (guarded by the collective's `static_assert`, so the `true` arm is only reachable for FP8 instantiations — for non-FP8 `Element`, keep `UseQKEmu=false`):
```cpp
                                        BOOL_SWITCH(params.fp8_no_two_level_accum, DisableFP8TwoLevel, [&] {
                                            if (params.qk_emu_enabled && params.qk_emu_fbits == 13) {
                                                run_flash_fwd<Arch, kHeadDim, kHeadDimV, ClusterM, T, T_out, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV && Varlen, HasQv, PackGQA, Split, V_colmajor, Use_one_mma_wg, kBlockH, DisableFP8TwoLevel, /*UseQKEmu=*/true, /*QKEmuFbits=*/13>(params, stream);
                                            } else if (params.qk_emu_enabled) {  // f_bits == 25 (validated upstream)
                                                run_flash_fwd<Arch, kHeadDim, kHeadDimV, ClusterM, T, T_out, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV && Varlen, HasQv, PackGQA, Split, V_colmajor, Use_one_mma_wg, kBlockH, DisableFP8TwoLevel, /*UseQKEmu=*/true, /*QKEmuFbits=*/25>(params, stream);
                                            } else {
                                                run_flash_fwd<Arch, kHeadDim, kHeadDimV, ClusterM, T, T_out, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV && Varlen, HasQv, PackGQA, Split, V_colmajor, Use_one_mma_wg, kBlockH, DisableFP8TwoLevel, /*UseQKEmu=*/false, /*QKEmuFbits=*/25>(params, stream);
                                            }
                                        });
```
Also update the `/*DisableFP8TwoLevel=*/false` direct call at `:244` (the non-`use_one_mma_wg` arm) to append `, /*UseQKEmu=*/false, /*QKEmuFbits=*/25` so its template arity matches the new `run_flash_fwd` signature.

> **Build-cost guard:** the `UseQKEmu=true` arms only instantiate for the `Element=float_e4m3_t` specializations that reach this code; the `static_assert` in Task 4 makes a non-FP8 `UseQKEmu=true` instantiation a compile error, so ensure those arms are only compiled where `T` is FP8. If the surrounding dispatch instantiates this block for BF16 `T`, gate the `qk_emu_enabled` arms behind `if constexpr (std::is_same_v<T, cutlass::float_e4m3_t>)`; otherwise no guard is needed.

- [ ] **Step 4: Commit**

```bash
git add hopper/flash_fwd_launch_template.h
git commit -m "feat(qk-emu): thread UseQKEmu/QKEmuFbits through run_flash_fwd dispatch"
```

---

## Task 6: `mha_fwd` params, validation, pybind, and python kwargs

**Files:**
- Modify: `hopper/flash_api.cpp:749` (signature), after `:1043` (set + validate), `:1749` (pybind)
- Modify: `hopper/flash_attn_interface.py` (`_flash_attn_forward` `.fwd(...)` call + `flash_attn_func`/`flash_attn_varlen_func` kwargs)

- [ ] **Step 1: Add the two params to `mha_fwd`**

In `hopper/flash_api.cpp`, change the last param of `mha_fwd` (`:749`, currently `bool fp8_no_two_level_accum`) to:
```cpp
        bool fp8_no_two_level_accum,
        bool qk_emu_enabled,
        int64_t qk_emu_fbits
```

- [ ] **Step 2: Set into params + validate (after `:1043`)**

After `params.fp8_no_two_level_accum = fp8_no_two_level_accum;` (`:1043`) add:
```cpp
    params.qk_emu_enabled = qk_emu_enabled;
    params.qk_emu_fbits = static_cast<int>(qk_emu_fbits);
    if (qk_emu_enabled) {
        TORCH_CHECK(q_type == at::ScalarType::Float8_e4m3fn,
                    "qk_emu_enabled requires e4m3 (Float8_e4m3fn) query/key dtype");
        TORCH_CHECK(dprops->major == 9,
                    "qk_emu_enabled requires SM90 (Hopper)");
        TORCH_CHECK(qk_emu_fbits == 13 || qk_emu_fbits == 25,
                    "qk_emu_fbits must be 13 or 25, got ", qk_emu_fbits);
    }
```
> Use the existing dtype variable name for `q_type` (grep near `:760` for how the FP8 dtype is detected — match the existing `q_type`/`is_e4m3` local; do not invent a new one).

- [ ] **Step 3: Expose defaults at the pybind boundary**

At `:1749`, change `m.def("fwd", &mha_fwd, "Forward pass");` so the two new params have defaults (so existing python callers that omit them keep working). pybind requires explicit `py::arg` for defaults; add them for just the trailing optional args by giving the whole call `py::arg(...)` would be large — instead give only the new trailing args defaults via a lambda wrapper is overkill. Simplest correct change: keep positional binding and require callers to pass them (we update the only caller in Step 4). Replace with:
```cpp
    m.def("fwd", &mha_fwd, "Forward pass");  // qk_emu_enabled/qk_emu_fbits are trailing positional args
```
(No functional change here; the contract is that every `.fwd(...)` caller passes the two new trailing args — enforced by Step 4 and Task 7.)

- [ ] **Step 4: Update the python `.fwd(...)` call**

In `hopper/flash_attn_interface.py`, the `flash_attn_3_cuda.fwd(...)` call inside `_flash_attn_forward` currently ends with `cp_tot_seqused_k,`. Append the two new args (threaded from new function kwargs):
```python
        cp_tot_seqused_k,
        fp8_no_two_level_accum,
        qk_emu_enabled,
        qk_emu_fbits,
    )
```
Add `fp8_no_two_level_accum=False, qk_emu_enabled=False, qk_emu_fbits=25` to the `_flash_attn_forward` parameter list (keyword args with defaults), and surface `qk_emu_enabled`/`qk_emu_fbits` as kwargs on `flash_attn_func` (`:503`) and `flash_attn_varlen_func` (`:588`), passing them through to `_flash_attn_forward`.
> If `fp8_no_two_level_accum` is not currently a parameter of `_flash_attn_forward`, add it here too (the C++ requires it positionally before our two args). Grep the file first: `grep -n "fp8_no_two_level_accum" hopper/flash_attn_interface.py`.

- [ ] **Step 5: Build the extension (heavy — minutes) and smoke-import**

Run:
```bash
cd /workspace/Project/IDS-flash-attention/hopper
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90 \
  /workspace/Project/vllm-mma/.venv/bin/python -m pip install -e . 2>&1 | tail -20
/workspace/Project/vllm-mma/.venv/bin/python -c "import flash_attn_3_cuda; print('import OK')"
```
Expected: build completes, prints `import OK`. (This is the first full kernel build; expect several minutes. If build time is prohibitive, restrict instantiations via the existing `FLASHATTENTION_DISABLE_*` env flags to FP8-only for a faster iteration.)

- [ ] **Step 6: Commit**

```bash
git add hopper/flash_api.cpp hopper/flash_attn_interface.py
git commit -m "feat(qk-emu): thread qk_emu kwargs python->mha_fwd with validation"
```

---

## Task 7: End-to-end parity + signal test (real kernel)

**Files:**
- Modify: `hopper/test_flash_attn.py` (add an FP8 emu parity test)

- [ ] **Step 1: Write the failing parity test**

Add to `hopper/test_flash_attn.py`:
```python
import torch, pytest
from flash_attn_interface import flash_attn_func

@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
                    reason="QK CoFDA emu requires SM90")
@pytest.mark.parametrize("d", [64, 128])
def test_qk_emu_parity_and_signal(d):
    torch.manual_seed(0)
    b, s, h = 1, 256, 4
    def mk(): return (torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
                      .to(torch.float8_e4m3fn))
    q, k, v = mk(), mk(), mk()
    descale = torch.ones(1, device="cuda", dtype=torch.float32)
    common = dict(softmax_scale=d**-0.5, causal=True,
                  q_descale=descale, k_descale=descale, v_descale=descale)
    out_hw,  _ = flash_attn_func(q, k, v, **common, qk_emu_enabled=False)
    out_f25, _ = flash_attn_func(q, k, v, **common, qk_emu_enabled=True, qk_emu_fbits=25)
    out_f13, _ = flash_attn_func(q, k, v, **common, qk_emu_enabled=True, qk_emu_fbits=13)
    # F=25 ≈ hardware FP32 QK: close parity.
    err25 = (out_f25.float() - out_hw.float()).abs().mean().item()
    # F=13 ≈ restricted accumulator: measurably different, but bounded.
    err13 = (out_f13.float() - out_hw.float()).abs().mean().item()
    assert err25 < 5e-2, f"F25 parity too loose: {err25}"
    assert err13 > err25, f"F13 should diverge more than F25: {err13} vs {err25}"
    assert err13 < 1.0, f"F13 divergence unbounded: {err13}"
```

- [ ] **Step 2: Run the test to verify it fails before the build of Task 6 / passes after**

Run:
```bash
cd /workspace/Project/IDS-flash-attention/hopper
/workspace/Project/vllm-mma/.venv/bin/python -m pytest test_flash_attn.py::test_qk_emu_parity_and_signal -v
```
Expected (after Task 6 build): PASS for `d=64` and `d=128`. If `qk_emu_enabled` is rejected, re-check Task 6 validation (dtype/arch). If `err25` is large, the coordinate-mapping in `gemm_qk_cofda_emu` (Task 2 Step 3b) is wrong — debug against Task 2's unit test first, then the `(m,n)` recovery.

- [ ] **Step 3: Commit**

```bash
git add hopper/test_flash_attn.py
git commit -m "test(qk-emu): end-to-end parity (F25) + signal (F13) for QK CoFDA emu"
```

---

## Task 8: vLLM env-var selection surface

**Files:**
- Modify: `vllm-mma/vllm/envs.py` (declare the two env vars)
- Modify: `vllm-mma/vllm/v1/attention/backends/flash_attn.py:809-829` (pass kwargs)

- [ ] **Step 1: Declare the env vars**

In `vllm-mma/vllm/envs.py`, mirroring `VLLM_USE_CAIR_GEMM_FP8`, add:
```python
    "VLLM_USE_CAIR_QK_EMU": lambda: bool(int(os.getenv("VLLM_USE_CAIR_QK_EMU", "0"))),
    "VLLM_CAIR_QK_EMU_FBITS": lambda: int(os.getenv("VLLM_CAIR_QK_EMU_FBITS", "25")),
```
> Match the file's existing declaration style (grep `VLLM_USE_CAIR_GEMM_FP8` in `envs.py` and copy its exact form — some versions use an `environment_variables` dict, others a typed accessor).

- [ ] **Step 2: Pass kwargs from the flash_attn backend**

In `vllm-mma/vllm/v1/attention/backends/flash_attn.py`, the `flash_attn_varlen_func(...)` call (`:809-829`) — add, alongside `q_descale=...`:
```python
                    qk_emu_enabled=envs.VLLM_USE_CAIR_QK_EMU,
                    qk_emu_fbits=envs.VLLM_CAIR_QK_EMU_FBITS,
```
Add a one-time warning if the env is set but the path isn't FP8 e4m3 (guard near where `is_quantized_kv_cache` is checked, `:763`):
```python
        if envs.VLLM_USE_CAIR_QK_EMU and not is_quantized_kv_cache(self.kv_cache_dtype):
            logger.warning_once(
                "VLLM_USE_CAIR_QK_EMU is set but kv_cache_dtype is not fp8 (e4m3); "
                "QK CoFDA emulation is inert on this path.")
```
> `vllm/vllm_flash_attn/flash_attn_interface.py` (the copy vLLM imports) must also accept and forward `qk_emu_enabled`/`qk_emu_fbits` to its `.fwd(...)` call — apply the same change as Task 6 Step 4 to that file. Grep: `grep -rn "def flash_attn_varlen_func\|_cuda.fwd(" vllm/vllm_flash_attn/`.

- [ ] **Step 3: Smoke test the wiring (no model required)**

Run:
```bash
cd /workspace/Project/vllm-mma
VLLM_USE_CAIR_QK_EMU=1 VLLM_CAIR_QK_EMU_FBITS=13 \
  .venv/bin/python -c "import vllm.envs as e; print(e.VLLM_USE_CAIR_QK_EMU, e.VLLM_CAIR_QK_EMU_FBITS)"
```
Expected: prints `True 13`.

- [ ] **Step 4: Commit**

```bash
cd /workspace/Project/vllm-mma
git add vllm/envs.py vllm/v1/attention/backends/flash_attn.py vllm/vllm_flash_attn/flash_attn_interface.py
git commit -m "feat(qk-emu): VLLM_USE_CAIR_QK_EMU env selection for QK CoFDA emulation"
```

---

## Task 9: Bump IDS-flash-attention pin in vLLM cmake + rebuild

vLLM builds IDS-flash-attention as a `FetchContent` external project pinned to a git commit
(`GIT_TAG`) of `https://github.com/IDSLab-SKKU/IDS-flash-attention.git`. For the kernel changes
(Tasks 1–7) to be compiled into vLLM, the pin must advance to the new flash-attn commit.

Two modes — pick per environment:
- **Local-dev (no push needed):** set `VLLM_IDS_FLASH_ATTN_SRC_DIR=/workspace/Project/IDS-flash-attention`
  so cmake builds the local working tree directly. Use this for iteration.
- **Pinned (TAG bump):** push the flash-attn branch, then set `GIT_TAG` to the new commit SHA.
  Use this for a reproducible build. **Pushing to the shared org repo is an outward action —
  confirm with the maintainer before pushing.**

**Files:**
- Modify: `vllm-mma/cmake/external_projects/ids_flash_attn.cmake` (the `GIT_TAG` line)

- [ ] **Step 1 (Local-dev path): build vLLM against the local flash-attn tree**

Run:
```bash
cd /workspace/Project/vllm-mma
VLLM_IDS_FLASH_ATTN_SRC_DIR=/workspace/Project/IDS-flash-attention \
  .venv/bin/python -m pip install -e . --no-build-isolation 2>&1 | tail -25
```
Expected: configure picks `ids-flash-attention is available at /workspace/Project/IDS-flash-attention`; build completes. This compiles the new QK-emu kernel into `_ids_fa3_C` without any push.

- [ ] **Step 2 (Pinned path): bump GIT_TAG to the new flash-attn commit**

After the flash-attn branch is pushed (confirm first), get its SHA and update the pin:
```bash
cd /workspace/Project/IDS-flash-attention && git rev-parse HEAD   # NEW_SHA
```
In `vllm-mma/cmake/external_projects/ids_flash_attn.cmake`, replace the line
`GIT_TAG c8d10ee8036c05c9187fdd2e4b2cc7ff7c6b96cd` with `GIT_TAG <NEW_SHA>`.
> The cmake has configure-time defensive guards that grep the fetched `CMakeLists.txt` for
> `VLLM_FA_INSTALL_DIR` and `VLLM_FA2_LIB_NAME`. Our branch descends from the current pin
> (`c8d10ee`), which already has both, so the bump is safe. Do NOT bump to a commit lacking them.

- [ ] **Step 3 (Pinned path): rebuild vLLM against the bumped pin**

Run (unset the SRC_DIR override so the pin is used):
```bash
cd /workspace/Project/vllm-mma
unset VLLM_IDS_FLASH_ATTN_SRC_DIR
.venv/bin/python -m pip install -e . --no-build-isolation 2>&1 | tail -25
```
Expected: FetchContent clones the new SHA; build completes.

- [ ] **Step 4: Commit the cmake bump**

```bash
cd /workspace/Project/vllm-mma
git add cmake/external_projects/ids_flash_attn.cmake
git commit -m "build(qk-emu): bump IDS-flash-attention pin to include QK CoFDA emulation"
```

---

## Final Verification

- [ ] Standalone unit test passes: `/tmp/test_qk_cofda` prints `PASS` and `F13_SIGNAL_OK` (Task 2).
- [ ] Extension builds and imports (Task 6).
- [ ] Parity/signal test passes for `d ∈ {64,128}` (Task 7).
- [ ] vLLM env vars resolve (Task 8).
- [ ] Baseline unchanged: run an existing FP8 forward test with `qk_emu_enabled=False` (default) and confirm it still passes — proves the variant is additive and the hardware path is untouched.

```bash
cd /workspace/Project/IDS-flash-attention/hopper
/workspace/Project/vllm-mma/.venv/bin/python -m pytest test_flash_attn.py -k "fp8 or e4m3" -q
```
Expected: pre-existing FP8 forward tests still PASS.
