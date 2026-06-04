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
  if (max_div <= 1e-4) { printf("FAIL: F13 should measurably diverge from F25\n"); return 1; }

  cudaFree(dP); cudaFree(dV); cudaFree(dO);
  printf("PASS\n");
  return 0;
}
