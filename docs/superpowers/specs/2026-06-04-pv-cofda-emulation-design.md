# Design: Runtime-Selectable FP8 Forward-Attention PV CoFDA Emulation Variant

**Date:** 2026-06-04
**Status:** Approved design — ready for implementation plan
**Scope:** Replace the FP8 `O = P·V` hardware WGMMA in the SM90 forward attention kernel
with a software CoFDA (Chunked Accumulation with Intermediate Rounding) emulation,
selectable at runtime and **independent of** the existing QK emulation, so FP8 accumulator
precision can be studied for the second attention GEMM **inside the real attention kernel**.

This mirrors the QK CoFDA emulation
(`docs/superpowers/specs/2026-06-01-qk-cofda-emulation-design.md`); read that first. This
document records only what is **different** for PV, and restates the shared machinery
briefly.

---

## 1. Background & Motivation

The flash-attention SM90 forward kernel computes attention in two FP8 GEMMs:
`S = Q·Kᵀ` (QK), then — after softmax — `O = P·V` (PV), where `P` is the FP8-quantized
softmax probabilities and `V` is the FP8 value cache. The QK emulation already replaces the
first GEMM's hardware WGMMA with a software CoFDA emulation
(`hopper/gemm_qk_cofda_emu.h`, `UseQKEmu`/`QKEmuFbits`). This design brings the same CoFDA
treatment to the **second** GEMM so the FP8 accumulator precision of `P·V` can be measured
independently of QK.

**Non-goal:** production performance. Research instrument; correctness of the emulated
precision is the priority, not speed.

---

## 2. What is structurally different from QK (the crux)

| Aspect | QK (`S = Q·Kᵀ`) | PV (`O = P·V`) |
|---|---|---|
| A operand | `Q`: raw FP8 in smem (swizzled tensor) | **`P`: register fragment** — softmax output converted to e4m3 (`tOrP`), distributed across the warpgroup (`MmaPV_is_RS` is forced `true` for FP8). |
| B operand | `K`: raw FP8 in smem | `V`: FP8 in smem, **transposed** (`Transpose_V = Is_FP8 && !V_colmajor`). |
| Reduction dim | `headdim` (=128) — entirely within one GEMM call | `seqlen_k` — **split across KV blocks** (`kBlockN` per call), with FP32 online-softmax rescale of `O` between blocks. |
| Output | raw pre-descale `S` → softmax downstream | `tOrO` (FP32) **accumulated across KV blocks** (`zero_init` only on the first block). |

Two consequences drive the design:

1. **P must be staged to smem.** A thread computing output `(m,n)` needs `P[m, all k]`, but
   the register fragment `tOrP` is partitioned across the warpgroup — a thread does not hold
   all of row `m`. So the emu cannot read `P` from registers; it must read from smem. The
   kernel already has the machinery (`SmemLayoutP`, `write_P_to_smem`, and a smem-source PV
   GEMM path at the `!MmaPV_is_RS` / no-overlap sites), so we reuse it.

2. **Cross-block accumulation.** Unlike QK, the PV reduction is split across KV blocks and
   the FP32 accumulator `tOrO` is carried (and online-softmax-rescaled) between them. The
   emulation must model the hardware's **continuous FP32 accumulator** across blocks (see §5).

Everything else — descale, online-softmax rescale, the FP32 nature of the accumulator — is
unchanged. The emu produces exactly the raw `P·V` contribution the WGMMA would write into
`tOrO`; descale (`v_descale`) and rescale stay downstream, identical to today.

---

## 3. Requirements (decided)

| Decision | Choice |
|---|---|
| Replacement target | The flash-attn `flash::gemm` PTX/WGMMA call at the PV GEMM sites (`tiled_mma_pv`). |
| Scope | **Forward `O = P·V` only.** QK untouched by this work (it has its own axis); backward untouched. |
| Mechanism | A **new runtime-selectable kernel variant**, independent of `UseQKEmu`. Baseline WGMMA path preserved. |
| Toggle granularity | **Independent toggle** — `UsePVEmu` separate from `UseQKEmu`; QK-only / PV-only / both are all selectable, each with its own f-bits. |
| Integration fidelity | **Correctness-first**, bit-exact target (§6). Per-element CoFDA over smem-staged operands. No perf tiling. |
| vLLM selection surface | Env vars `VLLM_USE_CAIR_PV_EMU` / `VLLM_CAIR_PV_EMU_FBITS`, mirroring the QK env vars. |
| F-bits range | Runtime `{13, 25}`, compile-time-specialized per value. |
| Datatype | **e4m3 only**. |
| Config gate | **head_dim == 128 only** (the validated Llama-3.1-8B config), e4m3, sm_90 — same gate as QK. |

---

## 4. Architecture & Template-Axis Threading

Add a compile-time `UsePVEmu` (+ compile-time `PVEmuFbits ∈ {13,25}`) axis, threaded exactly
like `UseQKEmu`/`QKEmuFbits` and `DisableFP8TwoLevel`:

```
env VLLM_USE_CAIR_PV_EMU / VLLM_CAIR_PV_EMU_FBITS
  → python kwargs (pv_emu_enabled: bool = False, pv_emu_fbits: int = 25)
  → mha_fwd (validate: e4m3 && arch==90 && head_dim==128 && pv_emu_fbits ∈ {13,25})
  → Flash_fwd_params{ pv_emu_enabled, pv_emu_fbits }
  → flash_fwd_launch_template.h: BOOL_SWITCH(params.pv_emu_enabled, UsePVEmu, …)
        + inner switch(params.pv_emu_fbits) → compile-time PVEmuFbits
  → CollectiveMainloopFwdSm90<…, UseQKEmu, QKEmuFbits, UsePVEmu, PVEmuFbits>
  → if constexpr (UsePVEmu) branch at the PV gemm sites
```

`UsePVEmu` and `UseQKEmu` are orthogonal axes — the kernel is instantiated for the cross
product only in the gated (e4m3 + sm90 + head_dim==128) config; all other configs compile
`UsePVEmu = false` only, containing the instantiation/build-time blowup.

### Components (each single-responsibility)

**(a) Vendored CoFDA headers — `hopper/cair_emu/` (reuse, no change)**
The PV emu uses the *same* `cair_fp8_cofda_mma.cuh` and the layout-agnostic
`cofda_dot<F>(fa, fb, D)` numeric core already in `hopper/gemm_qk_cofda_emu.h`. `cofda_dot`
takes caller-supplied `fa(k)`/`fb(k)` fetchers, so it is reused verbatim — only the operand
addressing differs.

**(b) Emulation device function — `hopper/gemm_pv_cofda_emu.h` (new)**
```cpp
template <int F, bool ZeroInit, class TiledMmaPV,
          class TensorSP, class TensorSV, class TensorO>
CUTLASS_DEVICE void gemm_pv_cofda_emu(TiledMmaPV const& mma,
                                      TensorSP const& sP_pi,   // raw FP8 e4m3 P, swizzle-free view, (M, kBlockN)
                                      TensorSV const& sV_pi,   // raw FP8 e4m3 V, swizzle-free view, (kBlockN, kHeadDimV)
                                      TensorO& tOrO,           // FP32 accumulator (in/out)
                                      int thread_idx);
```
Algorithm (correctness-first):
1. Recover each `tOrO` element's logical `(m,n)`:
   `cO = make_identity_tensor(select<0,1>(TileShape_MNK_PV{}))`,
   `tOcO = mma.get_thread_slice(thread_idx).partition_C(cO)` — same identity-partition trick
   the kernel uses for the output coordinate (`taccOcO`, ~line 1084).
2. For each owned element `i` with coord `(m,n)`: seed the CoFDA accumulator —
   `acc = ZeroInit ? 0.f : tOrO(i)` — then loop `k` over `kBlockN` in `CHUNK=32` chunks,
   reading `sP_pi(m, k)` and `sV_pi(k, n)` as `float_e4m3_t`, accumulating via
   `fp8_cofda_mma<F,32>`; write `tOrO(i) = acc`.
3. Output is the raw `P·V` contribution in FP32 — identical semantics to the WGMMA result
   (modulo the F-bit accumulator restriction). Descale/rescale unchanged downstream.

`ZeroInit` mirrors the `zero_init` template argument the WGMMA call already carries per site
(`true` on the first KV block, `false` thereafter). Seeding `acc` with the existing
(already-rescaled) `tOrO(i)` is what makes the emulation model the hardware's **continuous**
FP32 accumulator across blocks (§5).

Smem access uses `cute::as_position_independent_swizzle_tensor` for `sP`/`sV` so logical
indexing is valid despite swizzle (same pattern as the QK emu).

**(c) Operand staging**
- **P → smem:** reuse `write_P_to_smem(tOrP)` (writes the e4m3 `tOrP` into `sP`). For FP8,
  `MmaPV_is_RS == true` so `SmemP_t` is normally a zero-size array (`smem_p` empty). In the
  **emu build only**, allocate the real `SmemP_t` so `sP` exists, gated by `UsePVEmu`.
- **V from smem:** read the transposed value tile as a swizzle-aware tensor and index it as
  `V[k,n]` (the transpose means `sV`/`sVt` indexing is swapped vs storage — the exact tensor
  and index order are pinned by the unit test, §8).

**(d) Template-axis threading** (mirror `UseQKEmu`)
- `flash.h::Flash_fwd_params`: add `bool pv_emu_enabled = false; int pv_emu_fbits = 25;`.
- `flash_api.cpp::mha_fwd`: add params `bool pv_emu_enabled, int64_t pv_emu_fbits`; set into
  `params`; validate (`is_e4m3 && arch==90 && head_dim==128`, `pv_emu_fbits ∈ {13,25}`).
- `flash_api_torch_lib.cpp`: register `pv_emu_enabled` / `pv_emu_fbits` in the fwd schema.
- `flash_fwd_launch_template.h`: nested `BOOL_SWITCH(UsePVEmu)` + fbits switch, passing
  `UsePVEmu, PVEmuFbits` into `run_flash_fwd` / `CollectiveMainloopFwdSm90`.
- `CollectiveMainloopFwdSm90`: add `bool UsePVEmu_ = false, int PVEmuFbits_ = 25` template
  params; store as `constexpr`; branch at every PV gemm site.

**(e) Build gating**
`UsePVEmu` only instantiated for `Element == float_e4m3_t && Arch == 90 && head_dim == 128`.
`static_assert(!UsePVEmu || Is_FP8)`. Guard analogous to QK / `ENABLE_CAIR_FP8`.

**(f) Python API**
`flash_attn_func` / `flash_attn_varlen_func` / the fwd custom op gain optional kwargs
`pv_emu_enabled: bool = False, pv_emu_fbits: int = 25`, threaded to the fwd op. Default off ⇒
zero behavior change.

**(g) vLLM surface**
In `vllm/v1/attention/backends/ids_flash_attn.py`, read env vars `VLLM_USE_CAIR_PV_EMU`
(bool) and `VLLM_CAIR_PV_EMU_FBITS` (int, default 25); declare both in `vllm/envs.py`; pass
as the new kwargs through `vllm/ids_flash_attn/flash_attn_interface.py` →
`torch.ops._ids_fa3_C.fwd`. Only effective on the FP8 e4m3 FA3 head_dim==128 path. QK and PV
env vars are independent and may both be set.

---

## 5. Cross-Block Accumulation Semantics (the key PV-specific decision)

The PV reduction over `seqlen_k` is split across KV blocks. The hardware WGMMA accumulates
`P·V` into the FP32 register accumulator `tOrO`, carrying it across KV iterations; between
iterations the online softmax **rescales** `tOrO` in FP32 (`RescaleOBeforeGemm` /
`softmax.rescale_o`). To model the hardware's **continuous FP32 accumulator faithfully**:

- The FP32 rescale of `tOrO` between blocks stays exactly as today (full FP32, unchanged).
- Each PV emu call **seeds** its CoFDA accumulator with the existing (already-rescaled)
  `tOrO(i)` when `ZeroInit == false`, and with `0` on the first block. It then chunk-
  accumulates this block's `k ∈ [0, kBlockN)` with intermediate rounding at `F` bits.

This makes the entire `seqlen_k` reduction behave as one continuous CoFDA-`F` accumulator
(seeded across blocks), which is the faithful analog of the hardware's single carried FP32
accumulator under `no_two_level`. `kBlockN` (=128 for the d=128 tile) is a multiple of
`CHUNK=32`, so chunking is exact; `static_assert(kBlockN % 32 == 0)`.

> **Risk (primary):** whether this seeded-continuous-accumulator model is literally bit-exact
> with the hardware `no_two_level` PV WGMMA can only be confirmed by test (§8.2), exactly as
> the QK bit-exactness was. The rounding order of "rescale → seed → chunk-accumulate" must
> match the hardware accumulator; if it does not, this is where it will show.

The FP8 two-level accumulation is bypassed in the emu path (CoFDA performs the whole
accumulation), same as QK; `no_two_level` is the comparison baseline.

---

## 6. Numerical Semantics & Validation Criterion

- `F = 13` ≈ hardware `no_two_level` PV accumulator; `F = 25` ≈ full FP32.
- **Validation criterion (mirrors QK):** with the PV path matched (`fp8_no_two_level_accum =
  True` on both runs) and **QK fixed to hardware** (so only PV varies), `emu(F=13)` for PV
  **must be bit-exact** with the hardware `no_two_level` PV MMA for short context, and must
  **stay bit-exact AND deterministic for long context** (s = 512–4096). `F = 13` vs `F = 25`
  yields a bounded, documented divergence — the research signal, not a failure.
- `v_descale` is applied downstream exactly as today; the emulation produces the same raw
  pre-descale `P·V` contribution.

---

## 7. Synchronization / Race (apply the QK lesson up front)

The QK emu hit a long-context stale-operand race: the synchronous emu reads K via LDS where
the hardware WGMMA read async, and `PipelineTmaAsync::consumer_release` assumes a warpgroup
synchronization the emu lacks — a straggler thread read K bytes the producer had already
overwritten once the pipeline wrapped (s > 512). Fixed with a per-WG `NamedBarrier` before
`consumer_release(K)`.

The PV emu reads **V** synchronously from smem, so it has the **same hazard** against the V
producer (`pipeline_v` / `pipeline_vt`). Design in the fix from the start:

- Insert a **per-warpgroup `NamedBarrier::sync`** (128 threads, distinct id per WG) before
  the PV emu's `consumer_release(V/Vt)`, restoring the synchronization the WGMMA got for free.
  The barrier id must be one that is idle at the PV site and does not cross-pair with the QK
  emu's barrier when **both** emus are enabled (id selection is a tracked risk).
- **P is WG-local** (this warpgroup just wrote `sP`), so it has no producer race, but the
  emu's read of `sP` must be ordered after `write_P_to_smem` — verify a warpgroup-level sync
  exists between the write and the emu read (the existing `__syncwarp` at ~line 1136 may be
  insufficient for a warpgroup-wide read pattern).

---

## 8. Testing Strategy

1. **Unit** — `hopper/test_pv_cofda_emu.cu`: one tile `P·V` (FP8-decoded P, V) vs a torch FP32
   reference, asserting bit-reasonable agreement at `F = 25`; pins the `V[k,n]` index mapping
   and the `(m,n)` accumulator-coordinate recovery.
2. **Parity / bit-exact** — `vllm-mma/tests/kernels/attention/test_pv_cofda_emu.py`:
   `emu(F=13)` vs hardware `no_two_level` PV, **parametrized over seqlen {512, 2048, 4096}
   plus a determinism check** (include long context from the start — that is where the QK race
   surfaced). Asserts bit-exact AND `torch.equal` across two runs.
3. **Signal** — `F = 13` vs `F = 25`: bounded, monotone divergence.
4. **WikiText PPL** — add `cair-pv-emu-f13` / `cair-pv-emu-f25` rows to the experiment config
   (`vllm-mma/cair/experiments/configs/`), QK fixed to hardware, PV = CoFDA emulation.

---

## 9. Scope Boundaries (YAGNI)

- Forward `O = P·V` only. **No** QK changes (separate axis), **no** backward.
- e4m3 only. head_dim == 128 only. `F ∈ {13, 25}`. `CHUNK = 32`.
- No performance optimization; document the expected slowdown (per-element CoFDA over a path
  the hardware does in one WGMMA batch, plus the forced P-to-smem stage).
- No new torch op; selection via existing `flash_attn_varlen_func` kwargs + env, like QK.

---

## 10. Open Items / Risks

- **Cross-block bit-exactness (primary):** the seeded-continuous-accumulator model (§5) must
  match the hardware carried FP32 accumulator bit-for-bit; confirmable only by §8.2.
- **PV race barrier id:** a per-WG `NamedBarrier` id that is idle at the PV site and does not
  cross-pair with the QK emu's barrier when both emus are on, under the scheduler ping-pong.
- **P-to-smem staging:** allocating `SmemP_t` in the FP8+emu path increases smem on an
  already-tight budget; gated to emu builds. Measure occupancy.
- **V index mapping:** reading `V[k,n]` from the transposed `sV`/`sVt` correctly; unit test
  (§8.1) is the guard.
- **Coordinate mapping correctness:** the `(m,n)` recovery via `partition_C(identity)` on
  `tiled_mma_pv` must match the PV accumulator fragment exactly.
- **Vendored-header drift:** shared with QK — `hopper/cair_emu/` must stay synced with CAIR.
