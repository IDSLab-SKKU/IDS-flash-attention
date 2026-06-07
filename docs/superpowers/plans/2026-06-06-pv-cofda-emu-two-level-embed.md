# PV CoFDA Emulation — Two-Level Embed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SM90 FP8 forward PV CoFDA emulation a bit-exact drop-in for the production FP8 two-level accumulation by running the per-KV-block emulation from zero and reproducing `flash::gemm`'s two-level backup/clear/FP32-merge in the mainloop, eliminating the cross-block seed that diverges at `s ≥ 2048`.

**Architecture:** Replace the single `gemm_pv_cofda_emu<…, ZeroInit=Is_first_iter>(…, tOrO)` call (which seeds the CoFDA accumulator with the carried `tOrO`) with: first block → emulate from zero into `tOrO`; non-first block → back up the already-rescaled `tOrO`, emulate this block's `P·V` from zero (overwriting `tOrO`), then element-wise FP32 add the backup. This mirrors `hopper/utils.h:243-326` line-for-line with the inner WGMMA replaced by the (already bit-exact) zero-seed emulation. The CoFDA numeric core and `flash::gemm` are untouched.

**Tech Stack:** CUDA 13.0 / nvcc, CUTLASS CuTe (SM90 GMMA), PyTorch C++ extension, Python (vLLM FA3 backend + tests). Target: H100 sm_90a.

**Spec:** `docs/superpowers/specs/2026-06-06-pv-cofda-emu-two-level-embed-design.md`

---

## Key facts established from the code (do not re-derive)

- The PV emu branch is the **non-overlap** `fwd_step` path in `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`. The emu call is at **`:1521-1524`**; P-staging (`:1452-1477`), V-from-gmem staging (`:1488-1517`), the `fence_view_async_shared` + per-WG `NamedBarrier` race guards (`:1518-1520`, `:1534-1541`), and `pipeline_v.consumer_release` (`:1542`) are correct and **must not change**.
- `softmax.rescale_o(tOrO, scores_scale)` for non-first blocks is at **`:1484`**, BEFORE the emu — so the backup captures the rescaled accumulator, identical to the reference where `flash::gemm` backs up the rescaled `tCrC`.
- `flash::gemm`'s two-level (`hopper/utils.h:243-251, :321-326`): `tCrC_original(i)=tCrC(i)` → `clear(tCrC)` → WGMMA accumulates block from 0 → `tCrC(i)=tCrC_original(i)+tCrC(i)`. The element-wise loops use `cute::size(tCrC)` and `cute::make_fragment_like(tCrC)`.
- `gemm_pv_cofda_emu<F, ZeroInit>` (`hopper/gemm_pv_cofda_emu.h`) already supports `ZeroInit`: with `ZeroInit=true` it sets `acc_init=0` and **overwrites** `tOrO(i) = cofda_dot_acc<F>(…, 0)` (the per-block `O=P·V` from zero). No body change is needed; it will now always be called with `ZeroInit=true`. The `ZeroInit=false` branch becomes dead at the call site (kept for interface generality — do NOT remove, to minimize kernel churn).
- Per-block zero-seed equivalence (`emu(seed=0) == zero-init WGMMA`, bit-exact) is already validated: QK is bit-exact at all lengths and PV first block is bit-exact today.
- The build picks up local IDS-flash-attention changes only when `VLLM_IDS_FLASH_ATTN_SRC_DIR` points at the checkout; otherwise it clones the pinned tag.
- Scratch diagnostic scripts `vllm-mma/tests/kernels/attention/_pv_emu_driver.py` and `_pv_emu_mismatch.py` exist from root-causing; they compare against `no_two_level` and are now obsolete.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` | PV emu branch: replace the seed-carry call with the two-level backup/emit-from-zero/merge structure | Modify (`:1521-1524`) |
| `hopper/gemm_pv_cofda_emu.h` | Header doc: state per-block-from-zero + caller-owned two-level merge | Modify (comment only, `:13-15`) |
| `vllm-mma/tests/kernels/attention/test_pv_cofda_emu.py` | Bit-exact + determinism vs **two-level** reference; F13-vs-F25 signal | Modify (rewrite to two-level baseline) |
| `vllm-mma/tests/kernels/attention/_pv_emu_driver.py` | obsolete scratch diagnostic | Delete |
| `vllm-mma/tests/kernels/attention/_pv_emu_mismatch.py` | obsolete scratch diagnostic | Delete |

**Conventions:** 2-space indent in CUDA files; match the surrounding `if constexpr` style. Two git repos: kernel/header/plan in `/workspace/IDS-flash-attention`, tests in `/workspace/vllm-mma`.

---

## Task 1: Switch the test to the two-level reference (red)

Establishes the failing baseline: with the *current* (seed-carry) build, `pv_emu(F=13)` is not bit-exact with the two-level reference (it models `no_two_level`), so the updated test fails — especially at `s ≥ 2048`.

**Files:**
- Modify: `vllm-mma/tests/kernels/attention/test_pv_cofda_emu.py`
- Delete: `vllm-mma/tests/kernels/attention/_pv_emu_driver.py`, `vllm-mma/tests/kernels/attention/_pv_emu_mismatch.py`

- [ ] **Step 1: Rewrite the test to compare against the default two-level path**

Overwrite `vllm-mma/tests/kernels/attention/test_pv_cofda_emu.py` with exactly:
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
def test_pv_emu_f13_bit_exact_with_two_level(head_size, seqlen, causal):
    # Embedded-in-two-level PV emu must be bit-exact with the DEFAULT (two-level)
    # hardware PV at all context lengths, including s>=2048 where the old
    # seed-carry model diverged. QK stays on hardware (only PV is under test).
    dev, nq, nkv, d, s = "cuda", 8, 2, head_size, seqlen
    torch.manual_seed(0)
    mk = lambda nh: torch.randn(s, nh, d, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    q, k, v = mk(nq), mk(nkv), mk(nkv)
    dq = torch.ones(1, nkv, device=dev)
    ref = _run(q, k, v, dq, s, causal)  # default two-level PV, no emu
    emu1 = _run(q, k, v, dq, s, causal, pv_emu_enabled=True, pv_emu_fbits=13)
    emu2 = _run(q, k, v, dq, s, causal, pv_emu_enabled=True, pv_emu_fbits=13)
    assert torch.equal(emu1, emu2), f"non-deterministic at s={s}, causal={causal}"
    assert torch.equal(emu1, ref), (
        f"pv_emu(F=13) not bit-exact with hardware two-level at s={s}, causal={causal}; "
        f"mismatches={(emu1 != ref).sum().item()} max|diff|={(emu1 - ref).abs().max().item():.3e}")


@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("causal", [True, False])
def test_pv_emu_f13_vs_f25_diverges(head_size, causal):
    # F controls the per-block accumulator precision; F=13 (hardware) vs F=25
    # (~FP32) must produce a bounded, deterministic divergence.
    dev, nq, nkv, d, s = "cuda", 8, 2, head_size, 2048
    torch.manual_seed(0)
    mk = lambda nh: torch.randn(s, nh, d, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    q, k, v = mk(nq), mk(nkv), mk(nkv)
    dq = torch.ones(1, nkv, device=dev)
    f13 = _run(q, k, v, dq, s, causal, pv_emu_enabled=True, pv_emu_fbits=13)
    f25 = _run(q, k, v, dq, s, causal, pv_emu_enabled=True, pv_emu_fbits=25)
    assert (f13 != f25).any(), "F=13 and F=25 should differ (the precision signal)"
```

- [ ] **Step 2: Delete the obsolete scratch diagnostics**

```bash
rm -f /workspace/vllm-mma/tests/kernels/attention/_pv_emu_driver.py \
      /workspace/vllm-mma/tests/kernels/attention/_pv_emu_mismatch.py
```

- [ ] **Step 3: Run the bit-exact test against the CURRENT build; confirm it FAILS**

```bash
cd /workspace/vllm-mma
.venv/bin/python -m pytest tests/kernels/attention/test_pv_cofda_emu.py::test_pv_emu_f13_bit_exact_with_two_level -k "2048 and not True" -v 2>&1 | tail -20
```
Expected: **FAIL** at `s=2048` (the current seed-carry emu models `no_two_level`, not two-level), reported as a `mismatches=…` assertion. This is the red state. (If the import fails because `.venv/bin/python` is missing, recreate the symlink: `ln -s /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 /workspace/vllm-mma/.venv/bin/python`.)

- [ ] **Step 4: Commit the test change**

```bash
cd /workspace/vllm-mma
git add tests/kernels/attention/test_pv_cofda_emu.py
git rm -q --ignore-unmatch tests/kernels/attention/_pv_emu_driver.py tests/kernels/attention/_pv_emu_mismatch.py
git commit -m "test(pv-emu): compare against the two-level reference; drop no_two_level scratch"
```

---

## Task 2: Restructure the mainloop PV emu branch into the two-level path

**Files:**
- Modify: `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1521-1524`
- Modify: `hopper/gemm_pv_cofda_emu.h:13-15`

- [ ] **Step 1: Replace the seed-carry emu call with the two-level structure**

In `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`, replace exactly these lines (`:1521-1524`):
```cpp
                    // Software CoFDA emulation of O=P.V (synchronous; replaces the PV WGMMA).
                    Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
                    flash::gemm_pv_cofda_emu<PVEmuFbits, /*ZeroInit=*/Is_first_iter>(
                        tiled_mma_pv, sP_pi, sVl, tOrO, thread_idx);
```
with:
```cpp
                    // Software CoFDA emulation of O=P.V (synchronous; replaces the PV WGMMA),
                    // embedded in the FP8 two-level accumulation structure (mirrors
                    // flash::gemm, utils.h:243-326): each KV block is accumulated FROM ZERO by
                    // the emu (the zero-seed case that is bit-exact with a zero-init WGMMA), and
                    // the carried accumulator is recombined by an FP32 merge OUTSIDE the
                    // reduced-precision datapath -- reproducing the hardware two-level accumulator.
                    Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
                    if constexpr (Is_first_iter) {
                        // First block: no two-level (reference uses zero_init=true here).
                        flash::gemm_pv_cofda_emu<PVEmuFbits, /*ZeroInit=*/true>(
                            tiled_mma_pv, sP_pi, sVl, tOrO, thread_idx);
                    } else {
                        // tOrO was already rescaled (softmax.rescale_o, above). Back it up,
                        // accumulate THIS block's P.V from 0 (the emu overwrites tOrO), then
                        // element-wise FP32 merge -- identical ops to utils.h:247-248,324.
                        Tensor tOrO_original = cute::make_fragment_like(tOrO);
                        #pragma unroll
                        for (int i = 0; i < cute::size(tOrO); ++i) { tOrO_original(i) = tOrO(i); }
                        flash::gemm_pv_cofda_emu<PVEmuFbits, /*ZeroInit=*/true>(
                            tiled_mma_pv, sP_pi, sVl, tOrO, thread_idx);
                        #pragma unroll
                        for (int i = 0; i < cute::size(tOrO); ++i) { tOrO(i) = tOrO_original(i) + tOrO(i); }
                    }
```

- [ ] **Step 2: Update the `gemm_pv_cofda_emu.h` header doc**

In `hopper/gemm_pv_cofda_emu.h`, replace exactly these lines (`:13-15`):
```cpp
// The reduction is over kBlockN (this KV block's keys); the CoFDA accumulator is
// SEEDED with the existing tOrO so the across-block reduction models the
// hardware's continuous FP32 accumulator.
```
with:
```cpp
// The reduction is over kBlockN (this KV block's keys). This computes a SINGLE
// block's O = P.V from zero (always called with ZeroInit=true); cross-block
// accumulation is handled by the caller's FP8 two-level merge (backup + clear +
// FP32 add), so the emu never re-seeds a >F-bit carried accumulator. See
// docs/superpowers/specs/2026-06-06-pv-cofda-emu-two-level-embed-design.md.
```

- [ ] **Step 3: Commit the kernel change**

```bash
cd /workspace/IDS-flash-attention
git add hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp hopper/gemm_pv_cofda_emu.h
git commit -m "feat(pv-emu): embed the PV emu in the FP8 two-level accumulation path"
```

---

## Task 3: Rebuild the extension against the local IDS checkout

**Files:** none (build only)

- [ ] **Step 1: Build**

```bash
cd /workspace/vllm-mma
VLLM_IDS_FLASH_ATTN_SRC_DIR=/workspace/IDS-flash-attention \
MAX_JOBS=32 TORCH_CUDA_ARCH_LIST="9.0" uv pip install -e . -v --torch-backend cu130 2>&1 | tee build.log
```
Expected: build succeeds; `tail -5 build.log` shows a successful install (no `error:` / `ptxas` failure). If a FA2 target fails on a missing submodule, run `git -C /workspace/IDS-flash-attention submodule update --init csrc/cutlass` and rebuild. If the build reports the kernel over the smem budget (`invalid argument` at launch later), the extra `tOrO_original` fragment pushed occupancy — report back (tracked risk §8 of the spec).

- [ ] **Step 2: Verify the new symbol path imports**

```bash
cd /workspace/vllm-mma
.venv/bin/python -c "from vllm.ids_flash_attn.flash_attn_interface import flash_attn_varlen_func; print('import OK')"
```
Expected: `import OK`. (If `.venv/bin/python` is missing, recreate the symlink as in Task 1 Step 3.)

---

## Task 4: Verify bit-exact + determinism + signal (green)

**Files:** none (validation); commit `build.log` ignore if untracked.

- [ ] **Step 1: Run the full bit-exact + determinism test**

```bash
cd /workspace/vllm-mma
.venv/bin/python -m pytest tests/kernels/attention/test_pv_cofda_emu.py::test_pv_emu_f13_bit_exact_with_two_level -v 2>&1 | tail -30
```
Expected: **all 6 parametrizations PASS** (`seqlen ∈ {512, 2048, 4096}` × `causal ∈ {True, False}`), including `s = 2048` and `s = 4096` — the cases that failed in Task 1 Step 3.
> If `s ≥ 2048` still fails (mismatch) but `s = 512` passes, the per-block zero-seed path is not matching the per-block WGMMA at long context — capture `max|diff|` and the differing `(seq, head, dim)` from the assertion message before changing the model (spec §5/§8). If it fails at ALL seqlens, re-check the merge ordering / that `rescale_o` still precedes the backup.

- [ ] **Step 2: Run the F13-vs-F25 signal test**

```bash
cd /workspace/vllm-mma
.venv/bin/python -m pytest tests/kernels/attention/test_pv_cofda_emu.py::test_pv_emu_f13_vs_f25_diverges -v 2>&1 | tail -10
```
Expected: PASS (F=13 and F=25 differ on at least one element). The signal is per-block precision only (two-level removes cross-block swamping), so it is expected to be weaker than the old `no_two_level` signal; a PASS confirms the F axis is still live.

- [ ] **Step 3: Sanity-check a short context is unaffected**

```bash
cd /workspace/vllm-mma
.venv/bin/python -m pytest "tests/kernels/attention/test_pv_cofda_emu.py::test_pv_emu_f13_bit_exact_with_two_level[False-512-128]" -v 2>&1 | tail -5
```
Expected: PASS (regression guard for the short-context path that already worked).

- [ ] **Step 4: Commit any remaining test/config artifacts**

```bash
cd /workspace/vllm-mma
git add -A tests/kernels/attention/test_pv_cofda_emu.py
git commit -m "test(pv-emu): bit-exact + determinism vs two-level confirmed at s=512/2048/4096" --allow-empty
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** §2 decision (zero-seed per block + two-level merge in mainloop) → Task 2 Step 1; §3.2 header doc → Task 2 Step 2; §4 bit-exact argument → Task 4 Step 1; §5 validation against two-level reference → Task 1 + Task 4; §7 build command (explicit, with `VLLM_IDS_FLASH_ATTN_SRC_DIR`) → Task 3; §6 keep `ZeroInit=false` branch (no removal) → noted in Key facts + Task 2 leaves the function body unchanged. All covered.
- **Placeholder scan:** no TBD/TODO; every code step shows the full replacement text; build/test commands are concrete with expected output.
- **Type/identifier consistency:** `gemm_pv_cofda_emu<PVEmuFbits, ZeroInit>` signature and arg order (`tiled_mma_pv, sP_pi, sVl, tOrO, thread_idx`) match the existing call; `cute::make_fragment_like` / `cute::size` mirror `flash::gemm` (utils.h); test fn names (`test_pv_emu_f13_bit_exact_with_two_level`, `test_pv_emu_f13_vs_f25_diverges`) are consistent across Tasks 1 and 4; `pv_emu_enabled`/`pv_emu_fbits` kwargs unchanged from the existing interface.
