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
  // Scale inputs by 2.0 so F=13 restricted accumulator visibly diverges from F=25
  for (int i = 0; i < M * D; ++i) hQ[i] = __nv_fp8_e4m3(2.0f * (0.5f + 0.1f * (i % 7)));
  for (int i = 0; i < N * D; ++i) hK[i] = __nv_fp8_e4m3(2.0f * (-0.3f + 0.05f * (i % 5)));
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < D; ++k)
        S_ref[m * N + n] += float(hQ[m * D + k]) * float(hK[n * D + k]);

  __nv_fp8_e4m3 *dQ, *dK; float* dS;
  cudaMalloc(&dQ, hQ.size() * sizeof(__nv_fp8_e4m3));
  cudaMalloc(&dK, hK.size() * sizeof(__nv_fp8_e4m3));
  cudaMalloc(&dS, M * N * sizeof(float));
  cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK.data(), hK.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
  run_dot<25><<<M, N>>>(dQ, dK, dS, M, N, D);
  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }
  }
  std::vector<float> S(M * N);
  cudaMemcpy(S.data(), dS, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  int fails = 0;
  for (int i = 0; i < M * N; ++i) {
    float a = S[i], b = S_ref[i];
    float tol = 1e-2f * (std::fabs(b) + 1.f);   // F=25 ≈ FP32: tight tolerance
    if (std::fabs(a - b) > tol) { printf("MISMATCH[%d] emu=%f ref=%f\n", i, a, b); ++fails; }
  }
  if (fails) {
    printf("FAIL: %d mismatches\n", fails);
    return 1;
  }
  printf("PASS\n");

  // Step 5: F=13 divergence assertion
  float* dS13;
  cudaMalloc(&dS13, M * N * sizeof(float));
  run_dot<13><<<M, N>>>(dQ, dK, dS13, M, N, D);
  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }
  }
  std::vector<float> S13(M * N);
  cudaMemcpy(S13.data(), dS13, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Compute F=25 errors for comparison
  std::vector<float> err25(M * N), err13(M * N);
  for (int i = 0; i < M * N; ++i) {
    err25[i] = std::fabs(S[i] - S_ref[i]);
    err13[i] = std::fabs(S13[i] - S_ref[i]);
  }

  int f13_bound_fail = 0;
  float total_err25 = 0.f, total_err13 = 0.f;
  int f13_strictly_worse = 0;  // elements where F13 is strictly worse than F25
  for (int i = 0; i < M * N; ++i) {
    float ref = S_ref[i];
    float bound = 0.5f * (std::fabs(ref) + 1.f);
    if (err13[i] > bound) { ++f13_bound_fail; }
    total_err25 += err25[i];
    total_err13 += err13[i];
    if (err13[i] > err25[i]) ++f13_strictly_worse;
  }

  if (f13_bound_fail > 0) {
    printf("F13_BOUND_FAIL: %d elements exceeded 0.5*(|ref|+1)\n", f13_bound_fail);
    return 1;
  }
  // F=13 restricted accumulator must produce STRICTLY MORE total error than F=25
  // (F=25 ≈ FP32; F=13 truncates mantissa bits causing measurable drift).
  // Also require at least one element where F=13 is strictly worse.
  if (total_err13 <= total_err25 || f13_strictly_worse == 0) {
    printf("F13_SIGNAL_WEAK: total_err25=%.6f total_err13=%.6f strictly_worse=%d/%d\n",
           total_err25, total_err13, f13_strictly_worse, M * N);
    return 1;
  }
  printf("F13_SIGNAL_OK (total_err13=%.6f > total_err25=%.6f, "
         "F13 strictly worse on %d/%d elements, all bounded)\n",
         total_err13, total_err25, f13_strictly_worse, M * N);

  cudaFree(dQ); cudaFree(dK); cudaFree(dS); cudaFree(dS13);
  return 0;
}
