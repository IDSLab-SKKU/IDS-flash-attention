# Vendored CoFDA headers

Source: `vllm-mma/csrc/libtorch_stable/quantization/cair/` (vLLM, ported from the
vllm-cair research fork). Copied 2026-06-01.

These are the device-side, header-only CoFDA emulation utilities used by
`hopper/gemm_qk_cofda_emu.h`. They are a VERBATIM copy — if the upstream CAIR
headers change, re-sync these. Do not edit them here except to fix include paths.

Public entry point used by flash-attention:
`fp8_cofda_mma<int F, int CHUNK=32>(DecodedFrag a, DecodedFrag b, float c) -> float`
(in `cair_fp8_cofda_mma.cuh`).
