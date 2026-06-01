# `flash::gemm` WGMMA Helper — Detailed Analysis

**Source:** `hopper/utils.h:235-328`
**Role:** The single warp-group-collective matrix-multiply primitive used by every SM90 mainloop (fwd & bwd). Every Tensor-Core MMA in the Hopper kernels funnels through this one helper.

> **Why this note exists:** This helper is the canonical "PTX MMA call site" in the codebase — `cute::gemm(tiled_mma, ...)` lowers to Hopper `wgmma.mma_async` instructions. Understanding it precisely is the prerequisite for the downstream goal of substituting a CAIR-style software MMA emulation in place of the hardware WGMMA.

---

## 1. Signature & Template Parameters

```cpp
template <bool zero_init=false, int wg_wait=0, bool SwapAB=false,
          int M_slice=-1, bool DisableFP8TwoLevel=false,
          typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTLASS_DEVICE void gemm(TiledMma& tiled_mma,
                         Tensor0 const& tCrA,   // A operand fragment (regs or smem-desc)
                         Tensor1 const& tCrB,   // B operand fragment (smem-desc)
                         Tensor2& tCrC)         // C accumulator fragment (regs, in/out)
```

Computes, in place: **`tCrC += A · B`** (or `tCrC = A · B` when `zero_init`), over the K mode of the operands. `A`/`B`/`C` are CuTe register/smem-descriptor *fragments* already partitioned for the warpgroup by `tiled_mma`.

| Param | Default | Meaning |
|---|---|---|
| `zero_init` | `false` | First WGMMA of the batch uses `ScaleOut::Zero` → `D = A·B` (ignore prior C). Otherwise `D = C + A·B`. |
| `wg_wait` | `0` | Arg to `warpgroup_wait<N>()`. `0` = drain fully before returning. `-1` = **skip the wait** (caller overlaps the async MMA with other work — software pipelining). `1` = leave one batch in flight. |
| `SwapAB` | `false` | Compute `Bᵀ·Aᵀ` instead of `A·B` by swapping operand order into `cute::gemm`. Lets a kernel reuse one accumulator layout for transposed problems (heavily used in bwd: `SdP_swapAB`, `dKV_swapAB`). |
| `M_slice` | `-1` | `-1` = whole accumulator. `≥0` = operate on just M-row slice `M_slice` of C (and the matching operand). Enables two warpgroups to split the M dimension of one CTA accumulator. |
| `DisableFP8TwoLevel` | `false` | Force-off the FP8 two-level accumulation even for FP8 inputs. |

`Tensor0/1/2`, `TiledMma` are deduced from the arguments.

---

## 2. Control-Flow Map

```
gemm()
├─ [FP8 prologue]  (#ifndef FLASHATTENTION_DISABLE_FP8_TWO_LEVEL_ACCUMULATION)
│     Use_Two_Level = Is_FP8 && !zero_init && !DisableFP8TwoLevel
│     if Use_Two_Level: backup tCrC → tCrC_original; clear(tCrC)
│
├─ if constexpr (M_slice >= 0):                      ── slicing branch
│     slice C (and A or B per SwapAB) by logical_divide, recurse with M_slice=-1
│
└─ else:                                              ── WGMMA branch
│     Is_RS = A comes from registers (not smem descriptor)
│     [fence operands]  warpgroup_fence_operand(A if RS) ; warpgroup_fence_operand(C)
│     warpgroup_arrive()
│     if zero_init: tiled_mma.accumulate_ = ScaleOut::Zero
│     for k_block in 0 .. min(kNumKIters,16):  cute::gemm(...); accumulate_=One   ── wgmma.mma_async ×N
│     if kNumKIters>16: second loop with k_offset USEL trick (anti-spill)
│     warpgroup_commit_batch()
│     if wg_wait>=0: warpgroup_wait<wg_wait>()
│     [fence operands]  warpgroup_fence_operand(C) ; warpgroup_fence_operand(A if RS)
│
└─ [FP8 epilogue]  if Use_Two_Level: tCrC = tCrC_original + tCrC
```

---

## 3. The WGMMA Issue Sequence (the core)

The `else` branch (`utils.h:265-318`) is a textbook Hopper warpgroup-MMA dispatch. Order is **mandatory** — the hardware async-proxy contract requires it.

1. **`warpgroup_fence_operand(...)`** (`:270-275`) — register fence. Tells the compiler/HW that registers feeding (A if register-sourced) and the accumulator C must be *settled* before the async WGMMA reads/writes them. Without it, the scheduler could reorder a register write past the async MMA and corrupt operands. `Is_RS` (`:266`) distinguishes the **RS** path (A in registers, needs an operand fence) from the **SS** path (A as an smem descriptor, no register fence needed). Note `const_cast` (`:270`) — the fence API is non-const but the operation is logically read-only here.

2. **`warpgroup_arrive()`** (`:276`) — issues the named-barrier arrive marking this warpgroup ready to issue WGMMA in the async proxy.

3. **K-unrolled `cute::gemm`** (`:284-291`) — each iteration emits one `wgmma.mma_async.sync.aligned.m64nNk16...` (shape/types per `tiled_mma`). The accumulate flag is the trick:
   - iteration 0 (only if `zero_init`): `accumulate_ = ScaleOut::Zero` → `D = A·B`
   - every iteration sets `accumulate_ = ScaleOut::One` afterward → all later K-blocks do `D += A·B`.

   Manual unroll (rather than letting `cute::gemm` walk the K mode) is what lets the code flip Zero→One exactly once at the boundary.

4. **`warpgroup_commit_batch()`** (`:308`) — closes the batch; the group of `mma_async` issued since `arrive` becomes one committed batch the HW will track.

5. **`warpgroup_wait<wg_wait>()`** (`:309`, gated on `wg_wait>=0`) — blocks until all but `wg_wait` committed batches retire. `wg_wait=0` → results in C are ready on return. `wg_wait=-1` → **don't wait**; the caller is responsible for a later `warpgroup_wait`, enabling MMA/copy overlap. This is why many call sites pass `-1`.

6. **Closing fences** (`:310-316`) — re-fence C (results now live) and A, so subsequent register reads of C see committed values.

**`ScaleOut::Zero`/`One`** (`GMMA` enum) is the hardware accumulate-or-overwrite bit baked into the WGMMA descriptor via `tiled_mma.accumulate_`.

---

## 4. FP8 Two-Level Accumulation (`:238-251`, `:320-327`)

Active only when `Is_FP8 && !zero_init && !DisableFP8TwoLevel`.

```
tCrC_original ← tCrC      // save running accumulator
clear(tCrC)               // start this batch from 0
... WGMMA accumulates this K-batch into the cleared tCrC ...
tCrC ← tCrC_original + tCrC   // re-merge, done in FP32 registers
```

**Why:** Hopper FP8 WGMMA accumulates in FP32, but the *input* path that feeds large running sums back through low-precision FP8 operands loses small contributions (swamping: a big running C makes each new `A·B` term round away). By zeroing C for each K-batch, every batch's partial sum stays in a comparable magnitude band, and the **batch-to-batch merge is a clean FP32 add** in registers — preserving the small terms that a single monotonic FP8 accumulation would drop. This is the well-known FlashAttention-3 FP8 accuracy fix. `zero_init` skips it (there is no prior C to preserve); `DisableFP8TwoLevel` skips it when a kernel knows a single batch suffices.

Cost: one extra register fragment (`tCrC_original`) and an add loop — both `constexpr`-gated to zero overhead for non-FP8.

---

## 5. `M_slice` Branch (`:253-264`)

When `M_slice >= 0`, the helper operates on **one M-row partition** of the accumulator rather than the whole thing:

```cpp
static constexpr int MMA_M = decltype(size<1>(tCrC))::value;   // # M-tiles in C
// C layout after partition is ((2,2,V),(MMA_M,1),MMA_N)
Tensor tCrC_slice = logical_divide(tCrC, Shape<_,Int<MMA_M>>{})(_, make_coord(Int<M_slice>{}, _), _);
```

`logical_divide` re-tiles the M mode so a single `make_coord(M_slice, _)` index selects that slice with **no data movement** (pure layout view). The matching operand is sliced too — A under `!SwapAB`, B under `SwapAB` — while the other operand stays whole. It then **recurses with `M_slice=-1`** so the actual WGMMA runs on the slice.

**Use:** the bwd kernel issues `gemm<...,M_slice=0>` then `gemm<...,M_slice=1>` (e.g. `mainloop_bwd...:955,962`) so the two warpgroups of a CTA each drive half of the dV/dK accumulator, with independent `wg_wait` scheduling per slice (slice 0 uses `-1`, slice 1 uses `1` to stagger the waits).

---

## 6. The `kMaxKIters = 16` Register-Spill Workaround (`:280-307`)

```cpp
static constexpr int kNumKIters = CUTE_STATIC_V(size<2>(tCrA));
static constexpr int kMaxKIters = 16;
for (k_block 0 .. min(kNumKIters,16)) cute::gemm(...);          // straight unroll
if constexpr (kNumKIters > kMaxKIters) {
    int const k_offset = cutlass::canonical_warp_group_idx() < 128 ? 0 : 1;  // ALWAYS 0
    for (k_block 16 .. kNumKIters) cute::gemm(..., tCrA(_,_,k_block + k_offset), ...);
}
```

**Problem:** with `kNumKIters > 16`, ptxas computes all smem operand descriptor addresses up front and **keeps them live in registers** across the long unroll → register pressure → spills to local memory → throughput loss.

**Trick:** `k_offset` is provably `0` (`canonical_warp_group_idx()` is always `< 128`), but ptxas *cannot* prove it, so it must recompute `tCrA(_,_,k_block + k_offset)`'s descriptor each iteration from a runtime value. This emits `USEL` (uniform-select) address math instead of caching addresses in registers — trading a couple of cheap ALU ops for eliminating the spill. The code's own comment flags it as a hack ("There's probably a better way to do this").

---

## 7. Call-Site Catalogue (representative)

| Site | Template args | Notes |
|---|---|---|
| `mainloop_fwd...:1217` (Q·K) | `<true,-1,false,-1,DisableFP8TwoLevel>` | `zero_init` (S starts fresh), no wait (overlap), FP8 flag threaded from kernel |
| `mainloop_bwd...:820` (S·dP) | `<true,-1,SdP_swapAB>` | swapAB, zero_init, no wait |
| `mainloop_bwd...:834` (dO·V) | `<true,-1,SdP_swapAB>` | second dP gemm |
| `mainloop_bwd...:955` (dV, slice0) | `<false,-1,dKV_swapAB,0>` | accumulate, M_slice=0, no wait |
| `mainloop_bwd...:962` (dV, slice1) | `<false,1,dKV_swapAB,1>` | accumulate, M_slice=1, `wg_wait=1` staggers retirement |

Pattern: forward S=Q·Kᵀ uses `zero_init`; accumulating gemms (dK/dV over the K loop) use `zero_init=false` and rely on two-level FP8; `wg_wait=-1` is the norm so the warpgroup overlaps MMA with TMA copies, with an explicit wait inserted by the pipeline elsewhere.

---

## 8. Helper Reference

| Symbol | Origin | Lowering / effect |
|---|---|---|
| `cute::gemm(tiled_mma,a,b,c)` | CuTe | One `wgmma.mma_async` per K-block per `tiled_mma` atom |
| `warpgroup_fence_operand` | `cutlass/arch/mma_sm90.../warpgroup` | Register/operand fence around async MMA |
| `warpgroup_arrive` | CUTLASS arch | Async-proxy arrive before issue |
| `warpgroup_commit_batch` | CUTLASS arch | Commit the issued `mma_async` group |
| `warpgroup_wait<N>` | CUTLASS arch | Retire all but N committed batches |
| `GMMA::ScaleOut::{Zero,One}` | CUTLASS GMMA | Overwrite vs accumulate bit in WGMMA |
| `cute::logical_divide` | CuTe | Zero-copy re-tiling for `M_slice` |
| `cute::make_fragment_like`,`cute::clear` | CuTe | FP8 backup buffer + zeroing |
| `cutlass::canonical_warp_group_idx` | CUTLASS | Always `<128`; used only to defeat ptxas address caching |

---

## 9. Relevance to the CAIR Emulation Swap

The downstream goal is to replace the hardware WGMMA — the `cute::gemm(tiled_mma, ...)` calls at `utils.h:286,288,301,303` — with a **software CoFDA emulation** (cf. `vllm-mma/.../cair/cair_fp8_cofda_mma.cuh`), so FP8 accumulator precision can be studied inside a real attention kernel rather than only in a standalone GEMM. The seams that matter for that substitution:

- **Operand source:** WGMMA consumes smem descriptors / register fragments laid out for the Tensor Core. A CoFDA emulation consumes *decoded* FP8 operands from smem (`DecodedOperand`, packed uint32). A shim must materialize CAIR's decoded layout from CuTe fragments.
- **Accumulator semantics:** the `ScaleOut::Zero/One` + two-level FP8 logic here is exactly the precision behavior CAIR exists to model. An emulation swap should *replace* the two-level block, not nest under it.
- **Async vs synchronous:** WGMMA is async (`arrive/commit/wait`, `wg_wait=-1` overlap). A CUDA-core emulation is synchronous; the `warpgroup_*` scaffolding and `wg_wait` plumbing would be bypassed for emulated paths, which changes the kernel's pipeline overlap and must be accounted for in scheduling.
- **Tile shape mismatch:** CAIR slice-1 is fixed at BM×BN×BK = 32×8×32, CHUNK_SIZE=32; the flash MMA atom is m64nNk16. Any swap needs a tiling adapter or a generalized CoFDA over the WGMMA atom shape.

These four seams are the substance of the replacement design (tracked separately).
