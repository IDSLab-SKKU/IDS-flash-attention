# Design: Embed the PV CoFDA Emulation Inside the Two-Level Accumulation Path

**Date:** 2026-06-06
**Status:** Approved design — ready for implementation plan
**Scope:** Restructure the SM90 FP8 forward-attention **PV** (`O = P·V`) CoFDA emulation so the
per-KV-block emulation runs **inside the existing FP8 two-level accumulation structure**
(backup → clear → accumulate-from-zero → FP32 merge), instead of threading the carried FP32
accumulator through the CoFDA primitive as a seed. The goal is a **bit-exact drop-in for the
production two-level reference at all context lengths**.

Companion / supersedes the cross-block-seeding model in
`docs/superpowers/specs/2026-06-04-pv-cofda-emulation-design.md` (§5). The CoFDA numeric core
(`hopper/cair_emu/cair_fp8_cofda_mma.cuh`) is **unchanged** and remains vendored verbatim.

---

## 1. Background & Motivation

The current PV emulation replaces the hardware `O = P·V` WGMMA with `gemm_pv_cofda_emu`, which
per KV block calls `cofda_dot_acc<F>(P, V, kBlockN, acc_init = carried tOrO)` — i.e. it threads
the carried FP32 accumulator through the CoFDA primitive as the **seed** to model the hardware's
continuous FP32 accumulator across blocks (the previous spec §5).

**Validated behavior (2026-06-06, current build):** with the comparison matched to
`fp8_no_two_level_accum=True` on both runs, `pv_emu(F=13)` is bit-exact **and** deterministic
with the hardware `no_two_level` PV for `s ≤ 512` (≤4 KV blocks), but at `s ≥ 2048` it is
**deterministic yet NOT bit-exact**: 5–9 mismatches per multi-million-element output,
`max|diff| ≈ 2.4e-4 … 4.9e-4` (~1 ULP), scattered across query rows, both causal modes. The
error grows with the number of cross-block seams (bit-exact at ≤3 seams, diverges at ≥15).

**Root cause (established).** The QK emulation succeeds at all lengths because the QK WGMMA is
always issued with `ScaleOut::Zero` (`zero_init=true` → `D = A·B`): it never reads an external
accumulator `C`, so the emulation's **zero-seeded** path matches it exactly. The PV WGMMA, for
non-first blocks, is issued with `ScaleOut::One` (`D = C + A·B`) and reads the carried FP32
accumulator `C`. CUTLASS holds `C` as a plain FP32 register
(`MMA_64x8x32_F32E4M3E4M3_SS_TN::CRegisters = float[4]`, `"+f"` in/out — **no promotion**); the
accumulate-vs-overwrite choice is solely the hardware `scale_D` predicate. The emulation models
the `ScaleOut::One` case by truncating `C` to `F` bits via `fp32_to_operand<F>` at the cross-block
seed; this re-quantization diverges from the hardware once the **`softmax.rescale_o` FP32 multiply
at a block seam** spreads `C` to more than `F` fractional bits. Only PV exercises `ScaleOut::One`,
and only at cross-block seams — exactly matching the observed `s`-dependence.

**Key structural insight.** The FP8 **two-level accumulation** already present in `flash::gemm`
(`hopper/utils.h:238-327`) *is* the clean cross-block-accumulation structure:

```cpp
tCrC_original(i) = tCrC(i);          // backup the carried (rescaled) accumulator
cute::clear(tCrC);                   // start this block from 0
// ... WGMMA accumulates THIS block's P·V into the cleared tCrC (from 0) ...
tCrC(i) = tCrC_original(i) + tCrC(i);// element-wise FP32 merge
```

Each block is accumulated **from zero** (the `ScaleOut::Zero`-equivalent case the emulation
already reproduces bit-exactly), and the cross-block combination is a **plain FP32 add** outside
the reduced-precision datapath. Embedding the emulation in this structure eliminates the
problematic `>F`-bit cross-block re-seed entirely.

**Non-goal:** production performance. Research instrument; correctness/bit-exactness is the
priority.

---

## 2. The Decided Change (Approach A)

Run `gemm_pv_cofda_emu` **always zero-seeded** (per-block `O = P·V` from 0), and reproduce the
two-level backup/clear/merge **in the mainloop PV emulation branch**, mirroring `flash::gemm`'s
two-level structure line-for-line with the inner WGMMA replaced by the emulation call.

| Aspect | Decision |
|---|---|
| Per-block emulation | `gemm_pv_cofda_emu<F, /*ZeroInit=*/true>` — accumulate this block's `P·V` from 0. |
| Cross-block accumulation | Explicit two-level **backup → clear → emu(from 0) → FP32 merge** in the mainloop emu branch (non-first blocks). |
| First block | `ZeroInit=true`, **no merge** (reference also skips two-level when `zero_init=true`) — unchanged from today. |
| `rescale_o` | Stays at the existing site (before the gemm); the backup captures the **rescaled** accumulator, identical to the reference. |
| CoFDA primitive | **Unchanged.** `cair_fp8_cofda_mma.cuh` / `cair_fp32_utils.cuh` vendored verbatim; truncation semantics kept. |
| `flash::gemm` | **Unchanged.** Not routed through; the two-level merge is reproduced locally in the emu branch. |
| Comparison / reference | **Two-level** hardware path (`fp8_no_two_level_accum=False`, the production default). The previous `no_two_level` comparison is retired. |
| Toggle / axes | `UsePVEmu` / `PVEmuFbits` and the env/python surface unchanged. |
| Cross-block seed path | The `ZeroInit=false` seed-carry path in `gemm_pv_cofda_emu` becomes **unused** at the call site (kept in the function for generality, or removed — see §6). |

---

## 3. Architecture & Code Changes

### 3.1 `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` — the PV emu branch

The current emu branch (≈`:1488-1524`) stages V/P then calls
`gemm_pv_cofda_emu<PVEmuFbits, /*ZeroInit=*/Is_first_iter>(…, tOrO, …)`, which seeds with the
carried `tOrO`. Replace the **call** (P/V staging and the per-WG sync barriers are unchanged)
with the two-level structure:

```cpp
Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
if constexpr (Is_first_iter) {
    // No two-level on the first block (reference uses zero_init=true here).
    flash::gemm_pv_cofda_emu<PVEmuFbits, /*ZeroInit=*/true>(
        tiled_mma_pv, sP_pi, sVl, tOrO, thread_idx);
} else {
    // Two-level: backup the (already rescaled) accumulator, accumulate this
    // block's P·V from 0, then FP32-merge — mirrors flash::gemm utils.h:243-326.
    Tensor tOrO_original = cute::make_fragment_like(tOrO);
    #pragma unroll
    for (int i = 0; i < cute::size(tOrO); ++i) { tOrO_original(i) = tOrO(i); }
    flash::gemm_pv_cofda_emu<PVEmuFbits, /*ZeroInit=*/true>(
        tiled_mma_pv, sP_pi, sVl, tOrO, thread_idx);   // overwrites tOrO with this block's sum
    #pragma unroll
    for (int i = 0; i < cute::size(tOrO); ++i) { tOrO(i) = tOrO_original(i) + tOrO(i); }
}
```

Notes:
- `gemm_pv_cofda_emu<…, ZeroInit=true>` already does `acc_init = 0` and **writes** `tOrO(i) =
  cofda_dot_acc<F>(…, 0)` (overwrite), so an explicit `clear(tOrO)` is unnecessary; the backup +
  overwrite + merge sequence is exactly the two-level semantics.
- The `rescale_o(tOrO, scores_scale)` at the existing non-first site stays **before** this block,
  so `tOrO_original` holds the rescaled accumulator — identical to the reference, where
  `flash::gemm` backs up the rescaled `tCrC`.
- All P-staging, V-from-gmem staging, `fence_view_async_shared`, and the per-WG `NamedBarrier`
  race guards are **unchanged** (they protect the smem operands the synchronous emu reads).
- The `permute_output_fp8` gating (`… && !UsePVEmu`) is unchanged — the emu still produces
  logical-ordered `O`.

### 3.2 `hopper/gemm_pv_cofda_emu.h`

No required change to the function body; it is now always invoked with `ZeroInit=true`. The
header comment should be updated to state that cross-block accumulation is handled by the caller's
two-level merge (not by the seed), and that the function computes a **single block's** `O = P·V`
from zero. The `ZeroInit=false` seed-carry branch is no longer reached from the kernel (§6).

### 3.3 Validation surface (vLLM)

No kernel-API or env/python changes. The **test reference** changes from `no_two_level` to the
default **two-level** path (§5).

---

## 4. Why This Reproduces the Two-Level Accumulator Bit-for-Bit

Element-by-element, for a non-first block:

| Reference (`flash::gemm`, `Use_Two_Level`) | Approach A (emu) | Equivalence |
|---|---|---|
| `tCrC_original(i) = tCrC(i)` (rescaled `C`) | `tOrO_original(i) = tOrO(i)` (rescaled `C`) | same fragment copy |
| `clear(tCrC)` then WGMMA accumulates block from 0 | `gemm_pv_cofda_emu<F,true>` writes block sum from 0 | **per-block zero-seed equivalence** ↓ |
| `tCrC(i) = tCrC_original(i) + tCrC(i)` | `tOrO(i) = tOrO_original(i) + tOrO(i)` | identical element-wise FP32 add |

The whole argument rests on one already-validated property:

> **per-block, zero-seed emulation == per-block, zero-init WGMMA (bit-exact).**

Evidence it already holds:
- **QK** is a zero-init single-batch WGMMA (headdim 128, 4 K-steps); the emulation's zero-seed
  `cofda_dot` (128, 4 chunks) is **bit-exact** with it at all lengths.
- **PV first block** (`zero_init=true`) is **bit-exact** today.
- The two-level inner batch is `clear(C)` then `D = 0 + A·B`, which equals `ScaleOut::Zero`
  bit-for-bit (`0 + x = x` in FP32) — so it is the same zero-init case.

Everything the reference does *around* the inner accumulation (backup, clear, FP32 merge, the
prior `rescale_o`) is reproduced with the identical operations. Therefore the two-level
accumulator logic is reproduced exactly; the only substituted component (the inner per-block
accumulation) is the case proven bit-exact.

`F = 13` reproduces the hardware per-block accumulator; `F = 25 ≈ FP32` is a higher-precision
research variant that intentionally diverges from the hardware (the F13-vs-F25 signal).

---

## 5. Validation Criterion & Testing

**Reference baseline is now the production two-level path** (`fp8_no_two_level_accum=False`).

1. **Bit-exact + determinism** —
   `vllm-mma/tests/kernels/attention/test_pv_cofda_emu.py`, parametrized over
   `seqlen ∈ {512, 2048, 4096}` and both causal modes:
   - `ref = flash_attn_varlen_func(…)` with the **default two-level** PV (no emu).
   - `emu = flash_attn_varlen_func(…, pv_emu_enabled=True, pv_emu_fbits=13)`.
   - Assert `torch.equal(emu, ref)` (bit-exact, **including `s ≥ 2048`** — the case that fails
     today) AND `torch.equal(emu_run1, emu_run2)` (determinism).
   > QK stays on hardware (`qk_emu_enabled=False`) so only the PV path is under test.
2. **Signal** — `pv_emu(F=13)` vs `pv_emu(F=25)`: bounded, deterministic divergence (the
   per-block accumulator-precision research signal).
3. **WikiText PPL** (deferred / optional) — if added, the emu rows compare against the
   **two-level** baseline, not `no_two_level`.

If `s ≥ 2048` is still not bit-exact after the change, the residual divergence is in the
per-block zero-seed path (which QK already exercises) or the merge ordering — capture
`max|diff|` and the differing `(seq, head, dim)` pattern before changing the model.

---

## 6. Scope Boundaries (YAGNI)

- **Forward `O = P·V` only.** No QK changes, no backward.
- **Primitive untouched.** `hopper/cair_emu/*` (incl. `cair_fp8_cofda_mma.cuh`,
  `cair_fp32_utils.cuh`) stays verbatim; truncation semantics kept.
- **`flash::gemm` untouched** — the two-level merge is reproduced locally in the emu branch
  (Approach A), not by routing the emu through the generic helper.
- The `ZeroInit=false` seed-carry branch of `gemm_pv_cofda_emu` is no longer reached. Prefer to
  **remove** it (and the `acc_init`/seed machinery) for clarity once the new path is validated;
  keeping it dead is acceptable only if removal risks churn. Decide in the plan.
- e4m3 only, head_dim == 128 only, `F ∈ {13, 25}`, `CHUNK = 32` — unchanged gates.
- No performance optimization; the extra backup fragment + merge loop are negligible vs the
  per-element CoFDA cost already present.

---

## 7. Build & Test Commands

**Build (rebuild the extension after kernel changes), explicit:**

```bash
cd /workspace/vllm-mma
MAX_JOBS=32 TORCH_CUDA_ARCH_LIST="9.0" uv pip install -e . -v --torch-backend cu130 2>&1 | tee build.log
```
Use `VLLM_IDS_FLASH_ATTN_SRC_DIR=/workspace/IDS-flash-attention` to build against the local
IDS-flash-attention checkout instead of re-cloning the pinned tag.

**Test:**
```bash
cd /workspace/vllm-mma
.venv/bin/python -m pytest tests/kernels/attention/test_pv_cofda_emu.py -v
```

---

## 8. Risks / Open Items

- **Per-block bit-exactness at long context:** the design assumes the per-block zero-seed emu
  matches the per-block zero-init WGMMA at every block of a long sequence. QK validates this for
  headdim-128 single batches; confirm it holds for the PV tile across `s = 2048/4096` (§5 test).
- **Merge fragment layout:** `tOrO_original` must share `tOrO`'s accumulator fragment layout so
  the element-wise copy/merge indices line up (`make_fragment_like` guarantees this).
- **Register pressure:** one extra accumulator-sized fragment (`tOrO_original`) live across the
  emu call, on an already-tight emu build. Measure occupancy; it mirrors `flash::gemm`'s
  `tCrC_original`, so the cost is the same as the reference two-level path.
- **Reference change in tests:** any existing PV-emu test or config that compared against
  `no_two_level` must switch to the two-level default, or it will (correctly) report a mismatch.
