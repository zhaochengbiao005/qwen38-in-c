#ifndef Q35_MM_H
#define Q35_MM_H

#include <stdint.h>
#include <stddef.h>

/* matvec kernels. y[m] = sum_n W[m,n] * x[n]; W row-major [rows, cols].
   FP8 path: W is e4m3fn bytes, scale is BF16 [ceil(rows/128), ceil(cols/128)],
   dequant fused per 128x128 block, no BF16 materialization (ADR-0001).

   Determinism contract: scalar and AVX2 paths use identical lane-wise
   accumulation (acc[n & 7], fused multiply-add), so outputs are bit-identical
   across paths ? hardware-lockable via FNV-1a. */

/* e4m3fn byte -> f32 conversion table (256 entries), built once at init. */
extern float q35_mm_fp8_lut[256];
void q35_mm_init(void);

float q35_mm_e4m3_to_f32(uint8_t v);

/* BF16 (storage bits) -> f32, bit-exact. Shared by loader and kernels. */
float q35_mm_bf16_to_f32(uint16_t v);

void q35_mm_fp8(const uint8_t *W, const uint16_t *scale_bf16,
                uint32_t rows, uint32_t cols,
                const float *x, float *y);

/* Batch matmul: Y[rows, n] = W[rows, cols] @ X[n, cols]^T
 * X is [n, cols] row-major (n tokens, each cols elements).
 * Y is [rows, n] row-major (one output per row).
 * Same FP8 block-dequant, same lane order as q35_mm_fp8. */
void q35_mm_fp8_batch(const uint8_t *W, const uint16_t *scale_bf16,
                      uint32_t rows, uint32_t cols, uint32_t n,
                      const float *X, float *Y);

/* Multi-weight matvec: up to 4 weight matrices sharing the same input x
 * and cols, different row counts, computed in one fork/join. total_rows =
 * sum of all rows. Each segment's output goes to y[n][0..rows[n]-1].
 * Pass rows[n]=0 to skip a segment. */
void q35_mm_fp8_multi4(const uint8_t *W[4], const uint16_t *sc[4],
                      uint32_t rows[4], uint32_t cols,
                      const float *x, float *y[4]);

void q35_mm_bf16(const uint16_t *W, uint32_t rows, uint32_t cols,
                 const float *x, float *y);

/* which implementation ran (for tests) */
const char *q35_mm_impl_fp8(void);

#endif
