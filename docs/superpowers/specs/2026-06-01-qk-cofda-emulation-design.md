# Design: Runtime-Selectable FP8 Forward-Attention CoFDA Emulation Variant

**Date:** 2026-06-01
**Status:** Approved design — ready for implementation plan
**Scope:** Replace the FP8 `S = Q·Kᵀ` hardware WGMMA in the SM90 forward attention kernel
with a software CoFDA (Chunked Accumulation with Intermediate Rounding) emulation,
selectable at runtime, so FP8 accumulator precision can be studied **inside the real
attention kernel**.

---

## 1. Background & Motivation

The flash-attention SM90 forward kernel computes `S = Q·Kᵀ` via a hardware warpgroup
MMA (`cute::gemm` → `wgmma.mma_async...f32.e4m3.e4m3`, SASS `QGMMA.64xNx32.F32.E4M3.E4M3`,
**FP32 accumulator**, K=32 per instruction). See `AI/GEMM_WGMMA_ANALYSIS.md` for the call
site (`hopper/utils.h:235-328`, invoked at `mainloop_fwd_sm90_tma_gmma_ws.hpp:1217`/`:1356`)
and `AI/SASS_PTX_MMA_ANALYSIS.md` for the SASS realization.

vLLM ships a separate FP8 GEMM emulation, **CAIR** (`vllm-mma/.../quantization/cair/`),
which models low-precision accumulator behavior on CUDA cores (the `fp8_cofda_mma<F,32>`
device function, F = fractional accumulator bits ∈ {13, 25}). CAIR currently only covers
*linear-layer* GEMMs, not attention. The goal here is to bring that same CoFDA emulation
into the attention `Q·Kᵀ` so the precision effect can be measured where it matters for
long-context inference.

**Non-goal:** production performance. This is a research instrument; correctness of the
emulated precision is the priority, not speed.

---

## 2. Requirements (decided)

| Decision | Choice |
|---|---|
| Replacement target | The flash-attn `flash::gemm` PTX/WGMMA call (not anything inside `cair/`, which has no PTX) |
| Scope | **Forward `S = Q·Kᵀ` only**, first. P·V and backward untouched. |
| Mechanism | A **new runtime-selectable kernel variant** (like `cutlass_scaled_mm` vs `cair_scaled_mm`), not a hard replacement and not compile-time-only. Baseline WGMMA path preserved. |
| Integration fidelity | **Correctness-first**: per-element decode of FP8 operands from smem, faithful CoFDA. No perf tiling. |
| vLLM selection surface | **Environment variable**, mirroring `VLLM_USE_CAIR_GEMM_FP8`. |
| F-bits range | Runtime `{13, 25}`, mirroring CAIR slice-1. Compile-time-specialized per value. |
| Datatype | **e4m3 only** (CAIR slice-1 and flash FP8 fwd are both e4m3; e5m2 out of scope). |

---

## 3. Activation Condition (replaces the earlier "constraint" concern)

The emulated `Q·Kᵀ` path is exercised exactly when vLLM runs **true FP8 e4m3 attention**
through FA3 on Hopper. Verified in code:

| Condition | Required value | Evidence |
|---|---|---|
| `kv_cache_dtype` | `fp8` or `fp8_e4m3` (**e5m2 rejected**) | `flash_attn.py:175-178` (`get_fp8_dtype_for_flashattn`) |
| K/V passed as e4m3 | cache `.view(float8_e4m3fn)` | `flash_attn.py:763-769` |
| Query quantized to e4m3 | `supports_quant_query_input` → `not is_xpu()` (always true on CUDA) | `fa_utils.py:203-204`; `flash_attn.py:764,781-785` |
| FP8 supported | Hopper + FA3/FA4 | `fa_utils.py:194-203` (`flash_attn_supports_fp8`) |
| Hardware | sm_90 (H100) | — |

When all hold, `q`, `k`, `v` are all e4m3 and `flash_attn_varlen_func` dispatches the
`run_mha_fwd_<90, float_e4m3_t, …>` kernel, whose `S = Q·Kᵀ` is the e4m3 WGMMA this design
replaces. Turning on the env var routes that one MMA through CoFDA; everything downstream
(softmax, descale via `q_descale`/`k_descale`, P·V) is unchanged.

> Earlier exploration incorrectly claimed vLLM never runs FP8 attention at the kernel level.
> The code above shows it does, via FA3 with a quantized query. This design depends on that path.

---

## 4. Architecture

Add a compile-time template axis `UseQKEmu` (+ a compile-time `QKEmuFbits ∈ {13,25}`)
to the forward kernel, threaded from python through to the QK gemm site, exactly mirroring
the existing `fp8_no_two_level_accum` → `DisableFP8TwoLevel` machinery
(`flash_fwd_launch_template.h:238-244`, the `BOOL_SWITCH` pattern). At the gemm site:

```cpp
if constexpr (UseQKEmu) {
    flash::gemm_qk_cofda_emu<QKEmuFbits>(tiled_mma_qk, sQ_pi, sK_pi, tSrS, thread_idx);
} else {
    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, …>(tiled_mma_qk, tSrQ, tSrK(…), tSrS);
}
```

### Components (each single-responsibility)

**(a) Vendored CoFDA headers — `hopper/cair_emu/`**
Copies of CAIR's device-side, header-only files: `cair_types.cuh`, `cair_fp32_utils.cuh`,
`cair_fp8_utils.cuh`, `cair_fp8_cofda_mma.cuh`. No torch dependency. A provenance header
records the source (`vllm-mma/csrc/libtorch_stable/quantization/cair/`, vllm-cair fork) and
that they must stay in sync. Public interface used:
`fp8_cofda_mma<int F, int CHUNK=32>(DecodedFrag a, DecodedFrag b, float c) -> float`.

**(b) Emulation device function — `hopper/gemm_qk_cofda_emu.h` (new)**
```cpp
template <int F, class TiledMma, class TensorSQ, class TensorSK, class TensorC>
CUTLASS_DEVICE void gemm_qk_cofda_emu(TiledMma const& mma,
                                      TensorSQ const& sQ_pi,   // raw FP8 e4m3, swizzle-free view
                                      TensorSK const& sK_pi,   // raw FP8 e4m3, swizzle-free view
                                      TensorC& tSrS,           // FP32 accumulator (in/out)
                                      int thread_idx);
```
Algorithm (correctness-first):
1. Recover each accumulator element's logical `(m,n)`:
   `cS = make_identity_tensor(select<0,1>(TileShape_MNK{}))`,
   `tScS = mma.get_thread_slice(thread_idx).partition_C(cS)` — the exact mechanism the
   kernel already uses for masking at `mainloop_fwd_sm90_tma_gmma_ws.hpp:1151-1154`.
2. For each owned element `i` with coord `(m,n)`: loop `k` over `kHeadDim` in `CHUNK=32`
   chunks; read `sQ_pi(m, k…)` and `sK_pi(n, k…)` as `float_e4m3_t`; decode via CAIR's
   `decode_operand`; call `fp8_cofda_mma<F,32>`; accumulate into `tSrS(i)`.
3. Output is the **raw, pre-descale** `S` in FP32 — identical semantics to the WGMMA result
   (modulo the intentional F-bit accumulator restriction), so no downstream change.

Smem access uses `cute::as_position_independent_swizzle_tensor(sQ/sK)` so logical `(row, k)`
indexing is valid despite the swizzled smem layout (pattern at `:871`).

**(c) Template-axis threading** (mirror `DisableFP8TwoLevel`)
- `flash.h::Flash_fwd_params`: add `bool qk_emu_enabled = false; int qk_emu_fbits = 25;`.
- `flash_api.cpp::mha_fwd`: add params `bool qk_emu_enabled, int64_t qk_emu_fbits`; set into
  `params`; validate (`is_e4m3 && arch==90`, `qk_emu_fbits ∈ {13,25}`) via `TORCH_CHECK`.
- `flash_fwd_launch_template.h`: wrap with `BOOL_SWITCH(params.qk_emu_enabled, UseQKEmu, …)`
  and an inner `switch (params.qk_emu_fbits)` selecting compile-time `QKEmuFbits` (13/25),
  passing both into `run_flash_fwd<…, UseQKEmu, QKEmuFbits>`.
- `CollectiveMainloopFwdSm90`: add `bool UseQKEmu_ = false, int QKEmuFbits_ = 25` template
  params; store as `constexpr`; branch at the QK gemm sites (`:1217`, `:1356`).

**(d) Build gating**
The emu axis is only instantiated for `Element == float_e4m3_t && Arch == 90`. All other
configs compile `UseQKEmu = false` only, keeping the kernel-instantiation/build-time blowup
contained. A `static_assert` rejects `UseQKEmu && !Is_FP8`. Guard analogous to CAIR's
`ENABLE_CAIR_FP8` in `setup.py`.

**(e) Python API**
`flash_attn_func` / `flash_attn_varlen_func` / the `_flash_attn_forward` custom op gain
optional kwargs `qk_emu_enabled: bool = False, qk_emu_fbits: int = 25`, threaded to
`flash_attn_gpu.fwd(...)` → `mha_fwd`. Default off ⇒ zero behavior change.

**(f) vLLM surface**
In `vllm/v1/attention/backends/flash_attn.py`, read env vars `VLLM_USE_CAIR_QK_EMU` (bool)
and `VLLM_CAIR_QK_EMU_FBITS` (int, default 25); pass them as the new kwargs to
`flash_attn_varlen_func`. Mirrors `VLLM_USE_CAIR_GEMM_FP8`. Only effective when the FP8 e4m3
FA3 path of §3 is active.

---

## 5. Data Flow

```
env var / kwarg
  → mha_fwd (validate)                              flash_api.cpp
  → Flash_fwd_params{qk_emu_enabled, qk_emu_fbits}  flash.h
  → BOOL_SWITCH + fbits switch                      flash_fwd_launch_template.h
  → CollectiveMainloopFwdSm90<…, UseQKEmu, QKEmuFbits>
  → QK gemm site (if constexpr UseQKEmu)            mainloop_fwd_sm90_tma_gmma_ws.hpp:1217/1356
      ├─ true : gemm_qk_cofda_emu<F>(sQ,sK,mma,tSrS)   hopper/gemm_qk_cofda_emu.h
      └─ false: flash::gemm<…>(…)                       hopper/utils.h  (baseline WGMMA)
  → tSrS (raw FP32 scores)
  → existing softmax / descale / P·V                (unchanged)
```

---

## 6. Numerical Semantics

- The accumulator the emulation models is the hardware **FP32** accumulator (confirmed:
  `flash_fwd_launch_template.h:56` binds `ElementAccum = float`; e4m3 WGMMA can use F16 or
  F32, flash uses F32). CoFDA's `F` restricts the *fractional bits retained during
  accumulation*, modeling a narrower-than-F32 accumulator. `F = 25` ≈ full FP32 fidelity
  (baseline); `F = 13` ≈ Ada-style restricted accumulation.
- The FP8 **two-level accumulation** in `flash::gemm` (§4 of `GEMM_WGMMA_ANALYSIS.md`) is a
  *different* precision mechanism that sits on top of the F32 hardware accumulator. In the
  emu path CoFDA performs the entire accumulation, so two-level is **N/A for QK** and is
  bypassed. Documented in code.
- `kHeadDim ∈ {64,128,192,256}` are all multiples of `CHUNK=32`, so chunking is exact;
  `static_assert(kHeadDim % 32 == 0)`.
- Descale (`q_descale`, `k_descale`) is applied downstream exactly as today; the emulation
  produces the same *raw* pre-descale `S`.

---

## 7. Error Handling & Validation

- Compile: `UseQKEmu` only instantiated for e4m3 + sm_90; `static_assert` otherwise.
- Runtime (`mha_fwd`): `TORCH_CHECK` that `qk_emu_enabled` implies e4m3 inputs, arch 90, and
  `qk_emu_fbits ∈ {13, 25}`; else a clear error (CAIR-style `STD_TORCH_CHECK` messages).
- vLLM: if the env var is set but the active attention path is not FP8 e4m3 (§3 unmet), the
  flag is silently inert — log a one-time warning so users aren't misled.

---

## 8. Testing Strategy

1. **Unit** — a standalone test of `gemm_qk_cofda_emu` for one tile vs a torch FP32 reference
   `Q·Kᵀ` (FP8-decoded inputs), asserting bit-reasonable agreement at `F = 25`.
2. **Parity** — existing FP8 forward attention test (`tests/`) with emu-off (WGMMA) vs
   emu-on `F = 25`: expect close agreement within tolerance (F25 ≈ FP32).
3. **Signal** — `F = 13` vs `F = 25`: a measurable, documented divergence. This is the
   research signal, **not** a correctness failure; the test asserts the divergence is bounded
   and monotone, not zero.
4. **Smoke** — vLLM end-to-end with `VLLM_USE_CAIR_QK_EMU=1` on an FP8-e4m3-KV-cache model,
   confirming the path activates (and a short generation runs).

---

## 9. Scope Boundaries (YAGNI)

- Forward `Q·Kᵀ` only. **No** P·V, **no** backward.
- e4m3 only. `F ∈ {13, 25}`. `CHUNK = 32`.
- No performance optimization; document expected slowdown (CUDA-core per-element CoFDA over
  a path the hardware does in one WGMMA batch).
- No new torch op is registered for attention; selection is via existing
  `flash_attn_varlen_func` kwargs + env, not a parallel op (attention isn't a standalone GEMM
  op the way `cair_scaled_mm` is).

---

## 10. Open Items / Risks

- **Vendored-header drift**: `hopper/cair_emu/` copies CAIR headers; if CAIR changes, they
  must be re-synced. Provenance note + a follow-up to consider a shared submodule.
- **Register pressure / occupancy**: per-element CoFDA holds decoded operands and int64
  fixed-point sums in registers; on the attention kernel's already-tight register budget this
  may spill or reduce occupancy. Acceptable for a research path; measure.
- **Coordinate mapping correctness**: the `(m,n)` recovery via `partition_C(identity)` must
  match the WGMMA accumulator fragment exactly; the unit test (8.1) is the guard.
- **Two MMA-WG configs**: some FP8 tiles split M across warpgroups; the emu must cover all
  accumulator elements each thread owns (the identity-tensor partition handles this, but it
  is the primary correctness risk to verify).
