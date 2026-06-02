# SASS Analysis — PTX MMA (WGMMA) Path, SM90 FP8 Forward Attention

**Target:** ONLY the Tensor-Core MMA instructions (`wgmma` PTX → `QGMMA`/`HGMMA` SASS) that
the forward kernel issues via `cute::gemm`. Focus: the FP8 `S = Q·Kᵀ` MMA and its accumulator
type. Hardware: H100 sm_90, CUDA 13.0 (nvcc 13.0).

**Relationship to existing notes:**
- `AI/SASS_MMA_ANALYSIS.md` — covers **BF16 backward** `HGMMA.*.F32.BF16` (different dtype,
  different direction). This note covers **FP8 forward** `QGMMA.*.F32.E4M3.E4M3`.
- `AI/GEMM_WGMMA_ANALYSIS.md` — the PTX *call site* (`flash::gemm`, `hopper/utils.h:235-328`).
  This note is the *SASS realization* of that call.

All SASS/PTX below is **real**, extracted from a minimal single-atom probe TU compiled with
`nvcc -arch=sm_90a` against the CUTLASS headers used by this project (not inferred). Probe:
two kernels issuing one `MMA_64x64x32_{F32,F16}E4M3E4M3_SS_TN` each.

---

## 1. Accumulator Type — the headline result

**The FP8 forward `Q·Kᵀ` MMA uses an FP32 accumulator.** "bf16 accumulator" does not exist
for WGMMA — the hardware offers only **F16 (half) or F32**; bf16 is an *input* type only.
Flash binds `ElementAccum = float` (`flash_fwd_launch_template.h:56`,
`ss_op_selector<e4m3, e4m3, float, …>` at `mainloop_fwd_sm90_tma_gmma_ws.hpp:98`), selecting
the F32 variant.

The accumulator type is an explicit, visible field at **both** PTX and SASS level:

| | Accumulator | PTX | SASS | Dest registers |
|---|---|---|---|---|
| **Flash QK MMA** | **F32** | `wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3` | `QGMMA.64x64x32.F32.E4M3.E4M3` | 32 × `.f32` (`%f1..%f32`) |
| Contrast (not used) | F16 | `wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3` | `QGMMA.64x64x32.F16.E4M3.E4M3` | 16 × `.b32` (`%r1..%r16`) |

The F32 path emits **twice** the destination registers (32 vs 16) — the accumulator footprint
is directly observable. CUTLASS exposes 48 `F32E4M3E4M3` and 48 `F16E4M3E4M3` atom variants;
flash uses F32 exclusively in the forward kernel (no fp16/bf16-accumulator option is wired in).

---

## 2. PTX WGMMA Forms Emitted

The forward kernel's two MMAs (both via `cute::gemm` → unrolled `wgmma.mma_async`):

**QK gemm — `S = Q·Kᵀ`**, e4m3 × e4m3 → f32:
```
wgmma.mma_async.sync.aligned.m64nNk32.f32.e4m3.e4m3  {d0..d_{2N/... }}, descA, descB, scaleD, scaleA, scaleB ;
```
- Real (N=64 probe): `...m64n64k32.f32.e4m3.e4m3 {%f1...%f32}, %rd1, %rd2, p, 1, 1;`
- N ∈ {96, 128, 192} in production tiles (`tile_size.h`); M is always 64 per warpgroup.
- **K = 32 per instruction** (8-bit operands pack 32 along K), vs K=16 for 16-bit operands.
- Both operands are 64-bit **smem descriptors** (`%rd1`, `%rd2`) → the `_SS` (smem–smem) form,
  matching `MmaQK_is_RS = false` (`mainloop_fwd…:86`).

**PV gemm — `O = P·V`**, bf16 (P, from softmax) × e4m3 (V) → f32:
```
wgmma.mma_async.sync.aligned.m64nNk32.f32.bf16.e4m3  … ;
```
- A operand (P) is register-sourced (`MmaPV_is_RS = true` for FP8), B (V) is an smem descriptor.
- Still **f32 accumulator**; K=32.

(For comparison, the BF16 path in `SASS_MMA_ANALYSIS.md` is `...m64nNk16.f32.bf16.bf16` →
`HGMMA`, K=16.)

### Notation
`m64` = M per warpgroup (fixed, 128 threads). `nN` = N output width. `k32` = K elements per
instruction (FP8). `f32` = accumulator/output. `e4m3`/`bf16` = operand dtype.
`.sync.aligned` = synchronous, aligned descriptor form. Trailing `1, 1` = `scaleA, scaleB`
(scaleD = `p`, the accumulate predicate → `ScaleOut::One` after the first K-step).

---

## 3. SASS GMMA Mapping (real disassembly)

```
WARPGROUP.ARRIVE ;
QGMMA.64x64x32.F32.E4M3.E4M3  R24, gdesc[UR8], R24, gsb0 ;      ; QK, F32 accumulator
WARPGROUP.DEPBAR.LE  gsb0, 0x0 ;
```
- `QGMMA` = the SASS family for 8-bit ("quarter"/quad-byte) MMA inputs (e4m3/e5m2, incl. mixed
  bf16×e4m3). 16-bit inputs (bf16/fp16) map to `HGMMA`; tf32 → `OGMMA`; int8 → `IGMMA`.
- `gdesc[UR8]` = the shared-memory **matrix descriptor** operand (SS form). When an operand is
  register-sourced (RS, e.g. PV's P) it appears as a plain `R<n>` instead — the same RS/SS
  decoding shown for HGMMA in `SASS_MMA_ANALYSIS.md`.
- `R24` (dest = source) realizes accumulate-in-place (`D = C + A·B`); the first K-step of a
  batch instead uses `ScaleOut::Zero` (`D = A·B`).
- `gsb0` = the GMMA **scoreboard**; completion is awaited via `WARPGROUP.DEPBAR.LE gsb0, 0x0`.

### Warpgroup async sequence (SASS ↔ `flash::gemm`)
The `flash::gemm` PTX sequence (`utils.h:265-318`) lowers as:

| `flash::gemm` step | PTX | SASS |
|---|---|---|
| `warpgroup_fence_operand` | (fence) | scheduling fence around the GMMA region |
| `warpgroup_arrive()` | `wgmma.fence` + arrive | `WARPGROUP.ARRIVE` |
| `cute::gemm` ×K (unrolled) | `wgmma.mma_async…` | `QGMMA.64xNx32.F32.E4M3.E4M3` ×K |
| `warpgroup_commit_batch()` | `wgmma.commit_group` | (scoreboard commit) |
| `warpgroup_wait<N>()` | `wgmma.wait_group N` | `WARPGROUP.DEPBAR.LE gsb0, N` |

ptxas auto-injects the arrive/wait around the GMMA region; the wait is realized as
`WARPGROUP.DEPBAR.LE` on the scoreboard.

---

## 4. FP8 (QGMMA) vs BF16 (HGMMA) — summary

| Aspect | FP8 e4m3 fwd (this note) | BF16 bwd (`SASS_MMA_ANALYSIS.md`) |
|---|---|---|
| SASS family | `QGMMA` | `HGMMA` |
| Operand dtype | `.E4M3.E4M3` (QK), `.BF16.E4M3` (PV) | `.F32.BF16` |
| **Accumulator** | **F32** (`.F32`, 32 `%f` regs) | F32 |
| K per instruction | **32** (`64xNx32`) | 16 (`64xNx16`) |
| Operand source | QK: SS/SS; PV: RS/SS | RS or SS per gemm |
| Extra precision logic | two-level FP8 accumulation in F32 regs (`utils.h:238-251,320-327`) | none |

The F32 accumulator is what makes the FP8 **two-level accumulation** sensible: the hardware
already accumulates in F32, and the software backup/clear/add (in F32 registers) recovers the
small terms that FP8→F32 conversion swamping would otherwise drop. It is *not* a narrower
accumulator — that distinction matters for the CoFDA emulation work (see
`docs/superpowers/specs/2026-06-01-qk-cofda-emulation-design.md` §6), where CoFDA's `F`
parameter deliberately models a *narrower-than-F32* accumulator.

---

## 5. Accumulator / Register Footprint (observed)

- One `QGMMA.64x64x32.F32` writes **32 `.f32` registers** per issuing thread-lane slice; the
  F16 variant writes 16 `.b32`. Production N (128/192) scales the dest register count with N.
- Two-level FP8 adds a second `tCrC_original` fragment of equal size in registers — a real
  occupancy cost the design's §10 risk list calls out.

---

## 6. Method / Reproduction

```bash
# minimal probe (one warpgroup GMMA atom, F32 and F16 accumulators)
nvcc -std=c++17 -arch=sm_90a -I<cutlass>/include -cubin -o k.cubin qk_fp8_wgmma.cu
nvcc -std=c++17 -arch=sm_90a -I<cutlass>/include -ptx   -o k.ptx   qk_fp8_wgmma.cu
cuobjdump -sass k.cubin | grep -iE 'GMMA|WARPGROUP'
grep wgmma k.ptx
```
The probe instantiates `SM90::GMMA::MMA_64x64x32_{F32,F16}E4M3E4M3_SS_TN<>::fma(...)`
(note: `ScaleOut::One` needs the `SM90::GMMA::` qualifier). Disassembly above is verbatim.

> Production tiles use N ∈ {96,128,192} and unroll K; the *instruction form, accumulator
> field, and K=32* are identical to the probe — only N and the unroll count differ.
