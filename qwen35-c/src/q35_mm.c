/* FP8/BF16 matmul kernels. FP contraction disabled and a fixed sequential
 * reduction order are used so AVX2 and scalar paths produce bit-identical
 * outputs — matches the FNV-anchored reproducibility contract (map #15/#16). */
#pragma STDC FP_CONTRACT OFF

#include "q35/q35_mm.h"

#include <math.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

float q35_mm_fp8_lut[256];
static int lut_ready = 0;

float q35_mm_e4m3_to_f32(uint8_t v)
{
    /* e4m3fn: S EEEE MMM, bias 7, no inf, NaN codes 0x7F/0xFF */
    int sign = (v >> 7) & 1;
    int exp = (v >> 3) & 0xF;
    int man = v & 0x7;
    float f;
    if (exp == 0) {
        f = (float)man * 0x1.0p-9f; /* 2^-9 subnormal step */
    } else if (exp == 15 && man == 7) {
        f = NAN;
    } else {
        f = ldexpf(1.0f + (float)man * 0.125f, exp - 7);
    }
    return sign ? -f : f;
}

void q35_mm_init(void)
{
    if (lut_ready) return;
    for (int i = 0; i < 256; i++) q35_mm_fp8_lut[i] = q35_mm_e4m3_to_f32((uint8_t)i);
    lut_ready = 1;
}

float q35_mm_bf16_to_f32(uint16_t v)
{
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* ---------------- FP8 ---------------- */

#if defined(__AVX2__)
/* Convert 8 FP8 e4m3 bytes to 8 F32 values via ALU bit manipulation,
 * avoiding the slow _mm256_i32gather_ps. For normal values (exp > 0,
 * ~94% of byte values) the result is bit-identical to the LUT. For
 * subnormals and zeros (exp == 0) we fall back to the LUT gather —
 * rare in trained weights, so the branch is well-predicted. */
static inline __m256 fp8x8_to_f32_avx2(__m128i w8)
{
    __m256i v = _mm256_cvtepu8_epi32(w8);
    __m256i exp = _mm256_and_si256(_mm256_srli_epi32(v, 3),
                                   _mm256_set1_epi32(0xF));
    __m256i is_sub = _mm256_cmpeq_epi32(exp, _mm256_setzero_si256());
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(is_sub))) {
        /* all normal: (sign<<31)|((exp+120)<<23)|(man<<20) */
        __m256i sign = _mm256_slli_epi32(_mm256_srli_epi32(v, 7), 31);
        __m256i fexp = _mm256_slli_epi32(
            _mm256_add_epi32(exp, _mm256_set1_epi32(120)), 23);
        __m256i fman = _mm256_slli_epi32(
            _mm256_and_si256(v, _mm256_set1_epi32(7)), 20);
        return _mm256_castsi256_ps(
            _mm256_or_si256(sign, _mm256_or_si256(fexp, fman)));
    }
    /* slow path: has subnormals/zeros — gather from LUT */
    return _mm256_i32gather_ps(q35_mm_fp8_lut, v, 4);
}
#endif

/* Compute one row of W[sc] @ x. The single source of truth for the FP8
 * dot-product: AVX2 path uses 16-wide ALU dequant + fmadd with 8-lane
 * reduction; scalar path mirrors the same 8-lane n&7 pattern. Both produce
 * bit-identical sums (FNV-anchored contract, map #15/#16). All higher-level
 * matmul entry points call this. */
static inline float mm_fp8_row(const uint8_t *wr, const uint16_t *sc,
                               uint32_t ri, uint32_t cols, uint32_t sc_cols,
                               const float *x)
{
#if defined(__AVX2__)
    __m256 vacc = _mm256_setzero_ps();
    _Alignas(32) float lane[8];
    memset(lane, 0, sizeof(lane));
    uint32_t n = 0;
    while (n < cols) {
        uint32_t bend = n + 128 < cols ? n + 128 : cols;
        float s = q35_mm_bf16_to_f32(sc[(size_t)(ri / 128) * sc_cols + n / 128]);
        __m256 vs = _mm256_set1_ps(s);
        while (n + 16 <= bend) {
            __m128i w16 = _mm_loadu_si128((const __m128i *)(wr + n));
            /* prefetch next weight + input cache lines (2 ahead = 128 bytes) */
            if (n + 128 <= bend) {
                _mm_prefetch((const char *)(wr + n + 128), _MM_HINT_T0);
                _mm_prefetch((const char *)(x + n + 128), _MM_HINT_T0);
            }
            __m256 wf0 = fp8x8_to_f32_avx2(w16);
            __m256 wf1 = fp8x8_to_f32_avx2(_mm_srli_si128(w16, 8));
            vacc = _mm256_fmadd_ps(_mm256_mul_ps(wf0, vs), _mm256_loadu_ps(x + n), vacc);
            vacc = _mm256_fmadd_ps(_mm256_mul_ps(wf1, vs), _mm256_loadu_ps(x + n + 8), vacc);
            n += 16;
        }
        if (n < bend) {
            _Alignas(32) float spill[8];
            _mm256_store_ps(spill, vacc);
            for (int j = 0; j < 8; j++) lane[j] = spill[j];
            vacc = _mm256_setzero_ps();
            for (; n < bend; n++)
                lane[n & 7] = fmaf(q35_mm_fp8_lut[wr[n]] * s, x[n], lane[n & 7]);
        }
        n = bend;
    }
    _Alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vacc);
    float sum = 0.f;
    for (int j = 0; j < 8; j++) sum += tmp[j] + lane[j];
    return sum;
#else
    float acc[8] = { 0 };
    uint32_t n = 0;
    while (n < cols) {
        uint32_t bend = n + 128 < cols ? n + 128 : cols;
        float s = q35_mm_bf16_to_f32(sc[(size_t)(ri / 128) * sc_cols + n / 128]);
        for (; n < bend; n++)
            acc[n & 7] = fmaf(q35_mm_fp8_lut[wr[n]] * s, x[n], acc[n & 7]);
    }
    float sum = 0.f;
    for (int j = 0; j < 8; j++) sum += acc[j];
    return sum;
#endif
}

void q35_mm_fp8(const uint8_t *W, const uint16_t *scale_bf16,
                uint32_t rows, uint32_t cols,
                const float *x, float *y)
{
    if (!lut_ready) q35_mm_init();
    uint32_t sc_cols = (cols + 127) / 128;
    #pragma omp parallel for schedule(static)
    for (uint32_t r = 0; r < rows; r++)
        y[r] = mm_fp8_row(W + (size_t)r * cols, scale_bf16, r, cols, sc_cols, x);
}

/* ---- batch matmul: Y[rows,n] = W[rows,cols] @ X[n,cols]^T ----
 * For each weight row r, compute dot products against all n input rows.
 * Results are bit-identical to calling q35_mm_fp8 n times. */
void q35_mm_fp8_batch(const uint8_t *W, const uint16_t *scale_bf16,
                      uint32_t rows, uint32_t cols, uint32_t n,
                      const float *X, float *Y)
{
    if (!lut_ready) q35_mm_init();
    uint32_t sc_cols = (cols + 127) / 128;
    #pragma omp parallel for schedule(static)
    for (uint32_t r = 0; r < rows; r++) {
        const uint8_t *wr = W + (size_t)r * cols;
        for (uint32_t t = 0; t < n; t++)
            Y[(size_t)t * rows + r] =
                mm_fp8_row(wr, scale_bf16, r, cols, sc_cols,
                           X + (size_t)t * cols);
    }
}

/* ---- multi4: up to 4 weight matrices in one fork/join ---- */

void q35_mm_fp8_multi4(const uint8_t *W[4], const uint16_t *sc[4],
                      uint32_t rows[4], uint32_t cols,
                      const float *x, float *y[4])
{
    if (!lut_ready) q35_mm_init();
    uint32_t sc_cols = (cols + 127) / 128;
    /* compute segment offsets for the linear global row index */
    uint32_t off[5];
    off[0] = 0;
    for (int s = 0; s < 4; s++) off[s + 1] = off[s] + rows[s];
    uint32_t total = off[4];
    if (total == 0) return;

    #pragma omp parallel for schedule(static)
    for (uint32_t gr = 0; gr < total; gr++) {
        /* find which segment this global row belongs to */
        int s;
        for (s = 0; s < 4; s++) if (gr < off[s + 1]) break;
        uint32_t ri = gr - off[s];
        const uint8_t *wr = W[s] + (size_t)ri * cols;
        y[s][ri] = mm_fp8_row(wr, sc[s], ri, cols, sc_cols, x);
    }
}

const char *q35_mm_impl_fp8(void)
{
#if defined(__AVX2__)
    return "avx2";
#else
    return "scalar";
#endif
}

/* ---------------- BF16 ---------------- */

void q35_mm_bf16(const uint16_t *W, uint32_t rows, uint32_t cols,
                 const float *x, float *y)
{
    #pragma omp parallel for schedule(static)
    for (uint32_t r = 0; r < rows; r++) {
        const uint16_t *wr = W + (size_t)r * cols;
        float acc[8] = { 0 };
#if defined(__AVX2__)
        uint32_t n = 0;
        __m256 acc8 = _mm256_setzero_ps();
        for (; n + 16 <= cols; n += 16) {
            __m128i b0 = _mm_loadu_si128((const __m128i *)(wr + n));
            __m128i b1 = _mm_loadu_si128((const __m128i *)(wr + n + 8));
            __m256i w0 = _mm256_cvtepu16_epi32(b0);
            __m256i w1 = _mm256_cvtepu16_epi32(b1);
            w0 = _mm256_slli_epi32(w0, 16);
            w1 = _mm256_slli_epi32(w1, 16);
            __m256 f0 = _mm256_castsi256_ps(w0);
            __m256 f1 = _mm256_castsi256_ps(w1);
            acc8 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x + n), acc8);
            acc8 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x + n + 8), acc8);
        }
        _Alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc8);
        for (int j = 0; j < 8; j++) acc[j] = tmp[j];
        for (; n < cols; n++)
            acc[n & 7] = fmaf(q35_mm_bf16_to_f32(wr[n]), x[n], acc[n & 7]);
#else
        for (uint32_t n = 0; n < cols; n++)
            acc[n & 7] = fmaf(q35_mm_bf16_to_f32(wr[n]), x[n], acc[n & 7]);
#endif
        float sum = 0.f;
        for (int j = 0; j < 8; j++) sum += acc[j];
        y[r] = sum;
    }
}
