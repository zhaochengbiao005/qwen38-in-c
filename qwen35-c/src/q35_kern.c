/* Ticket-16: elementwise kernel pack. See q35_kern.h for the contract.
 *
 * Numerical discipline: FP contraction is turned off for this translation
 * unit so that the scalar path and the AVX2 path execute the exact same
 * sequence of FP operations; AVX2 is used purely as an 8-wide data mover.
 * This guarantees bit-identical outputs (FNV-1a equal) between paths.
 */
#pragma STDC FP_CONTRACT OFF

#include "q35/q35_kern.h"

#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- fixed-order reduction helpers ----------------------------------- */

/* sum of x^2 in fixed 8-accumulator order: element i goes to acc[i%8],
 * lanes folded 0..7 sequentially at the end. */
static float sumsq_fixed(const float *x, size_t n)
{
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    size_t i = 0;
#if defined(__AVX2__)
    __m256 v = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 t = _mm256_loadu_ps(x + i);
        v = _mm256_add_ps(v, _mm256_mul_ps(t, t));
    }
    _mm256_storeu_ps(acc, v);
    for (; i < n; i++) { float e = x[i]; acc[i & 7] = acc[i & 7] + e * e; }
#else
    for (; i + 8 <= n; i += 8) {
        for (int j = 0; j < 8; j++) { float e = x[i + j]; acc[j] = acc[j] + e * e; }
    }
    for (int j = 0; i < n; i++, j++) { float e = x[i]; acc[j] = acc[j] + e * e; }
#endif
    float s = acc[0];
    for (int j = 1; j < 8; j++) s = s + acc[j];
    return s;
}

static float sum_fixed(const float *x, size_t n)
{
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    size_t i = 0;
#if defined(__AVX2__)
    __m256 v = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
        v = _mm256_add_ps(v, _mm256_loadu_ps(x + i));
    _mm256_storeu_ps(acc, v);
    for (; i < n; i++) acc[i & 7] = acc[i & 7] + x[i];
#else
    for (; i + 8 <= n; i += 8)
        for (int j = 0; j < 8; j++) acc[j] = acc[j] + x[i + j];
    for (int j = 0; i < n; i++, j++) acc[j] = acc[j] + x[i];
#endif
    float s = acc[0];
    for (int j = 1; j < 8; j++) s = s + acc[j];
    return s;
}

static float sigmoidf_(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/* ---- rmsnorm (zero-centered weight) ---------------------------------- */

void q35_kern_rmsnorm_scalar(const float *x, const float *w, float *y,
                             size_t n, float eps)
{
    float mean = sumsq_fixed(x, n) / (float)n;
    float inv = 1.0f / sqrtf(mean + eps);
    for (size_t i = 0; i < n; i++) {
        float t = x[i] * inv;
        y[i] = t * (1.0f + w[i]);
    }
}

void q35_kern_rmsnorm(const float *x, const float *w, float *y,
                      size_t n, float eps)
{
#if defined(__AVX2__)
    float mean = sumsq_fixed(x, n) / (float)n;
    float inv = 1.0f / sqrtf(mean + eps);
    __m256 vs = _mm256_set1_ps(inv);
    __m256 one = _mm256_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        __m256 t = _mm256_mul_ps(vx, vs);
        __m256 g = _mm256_add_ps(one, vw);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(t, g));
    }
    for (; i < n; i++) y[i] = (x[i] * inv) * (1.0f + w[i]);
#else
    q35_kern_rmsnorm_scalar(x, w, y, n, eps);
#endif
}

/* ---- silu ------------------------------------------------------------ */

void q35_kern_silu_scalar(const float *x, float *y, size_t n)
{
    for (size_t i = 0; i < n; i++)
        y[i] = x[i] * sigmoidf_(x[i]);
}

void q35_kern_silu(const float *x, float *y, size_t n)
{
    /* per-element transcendental: identical op sequence in both paths */
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float tmp[8];
        for (int j = 0; j < 8; j++) tmp[j] = sigmoidf_(x[i + j]);
        _mm256_storeu_ps(y + i,
            _mm256_mul_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(tmp)));
    }
    for (; i < n; i++) y[i] = x[i] * sigmoidf_(x[i]);
#else
    q35_kern_silu_scalar(x, y, n);
#endif
}

/* ---- swiglu ---------------------------------------------------------- */

void q35_kern_swiglu_scalar(const float *gate, const float *up, float *y,
                            size_t n)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        float t = gate[i] * sigmoidf_(gate[i]);
        y[i] = t * up[i];
    }
}

void q35_kern_swiglu(const float *gate, const float *up, float *y, size_t n)
{
#if defined(__AVX2__)
    #pragma omp parallel for schedule(static)
    for (size_t blk = 0; blk < n; blk += 8) {
        size_t end = blk + 8 <= n ? blk + 8 : n;
        float tmp[8] = {0};
        for (size_t j = blk; j < end; j++)
            tmp[j - blk] = gate[j] * sigmoidf_(gate[j]);
        __m256 t = _mm256_loadu_ps(tmp);
        __m256 u = _mm256_loadu_ps(up + blk);
        if (end == blk + 8)
            _mm256_storeu_ps(y + blk, _mm256_mul_ps(t, u));
        else
            for (size_t j = blk; j < end; j++) y[j] = tmp[j - blk] * up[j];
    }
#else
    q35_kern_swiglu_scalar(gate, up, y, n);
#endif
}

/* ---- per-head q/k rmsnorm -------------------------------------------- */

void q35_kern_qk_norm_scalar(float *x, const float *w, size_t heads,
                             size_t head_dim, float eps)
{
    for (size_t h = 0; h < heads; h++)
        q35_kern_rmsnorm_scalar(x + h * head_dim, w, x + h * head_dim, head_dim, eps);
}

void q35_kern_qk_norm(float *x, const float *w, size_t heads, size_t head_dim,
                      float eps)
{
    for (size_t h = 0; h < heads; h++)
        q35_kern_rmsnorm(x + h * head_dim, w, x + h * head_dim, head_dim, eps);
}

/* ---- partial rope (rotate_half) --------------------------------------- */

/* rotate one head in place; rot = rotary_dim (even), half = rot/2 */
static void rope_head(float *x, size_t head_dim, size_t rot,
                      const float *cosv, const float *sinv)
{
    size_t half = rot / 2;
    size_t i = 0;
#if defined(__AVX2__)
    float y0[8], y1[8];
    for (; i + 8 <= half; i += 8) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + half);
        __m256 c = _mm256_loadu_ps(cosv + i);
        __m256 s = _mm256_loadu_ps(sinv + i);
        /* same op sequence as scalar: mul then sub, mul then add */
        _mm256_storeu_ps(y0, _mm256_sub_ps(_mm256_mul_ps(x0, c),
                                           _mm256_mul_ps(x1, s)));
        _mm256_storeu_ps(y1, _mm256_add_ps(_mm256_mul_ps(x1, c),
                                           _mm256_mul_ps(x0, s)));
        memcpy(x + i, y0, sizeof(y0));
        memcpy(x + i + half, y1, sizeof(y1));
    }
#endif
    for (; i < half; i++) {
        float x0 = x[i], x1 = x[i + half];
        float c = cosv[i], s = sinv[i];
        x[i]        = x0 * c - x1 * s;
        x[i + half] = x1 * c + x0 * s;
    }
}

static int rope_apply(float *x, size_t heads, size_t head_dim,
                      size_t rotary_dim, int position, float theta)
{
    if (head_dim == 0 || rotary_dim == 0 || rotary_dim > head_dim ||
        (rotary_dim & 1u))
        return Q35_KERN_ERR_ARG;
    size_t half = rotary_dim / 2;
    float cosv[128], sinv[128]; /* half <= head_dim/2, cap 128 pairs */
    if (half > 128) return Q35_KERN_ERR_ARG;
    for (size_t i = 0; i < half; i++) {
        /* HF: 1.0 / (theta ** (arange(0,d,2,float32)/d)), freed in f32 */
        float invf = 1.0f / powf(theta, (2.0f * (float)i) / (float)rotary_dim);
        float f = (float)position * invf;
        cosv[i] = cosf(f);
        sinv[i] = sinf(f);
    }

    for (size_t h = 0; h < heads; h++)
        rope_head(x + h * head_dim, head_dim, rotary_dim, cosv, sinv);
    return Q35_KERN_OK;
}

int q35_kern_rope(float *x, size_t heads, size_t head_dim, size_t rotary_dim,
                  int position, float theta)
{
    return rope_apply(x, heads, head_dim, rotary_dim, position, theta);
}

int q35_kern_rope_scalar(float *x, size_t heads, size_t head_dim,
                         size_t rotary_dim, int position, float theta)
{
    if (head_dim == 0 || rotary_dim == 0 || rotary_dim > head_dim ||
        (rotary_dim & 1u))
        return Q35_KERN_ERR_ARG;
    size_t half = rotary_dim / 2;
    if (half > 128) return Q35_KERN_ERR_ARG;
    float cosv[128], sinv[128];
    for (size_t i = 0; i < half; i++) {
        float invf = 1.0f / powf(theta, (2.0f * (float)i) / (float)rotary_dim);
        float f = (float)position * invf;
        cosv[i] = cosf(f);
        sinv[i] = sinf(f);
    }
    for (size_t h = 0; h < heads; h++) {
        float *px = x + h * head_dim;
        for (size_t i = 0; i < half; i++) {
            float x0 = px[i], x1 = px[i + half];
            px[i]        = x0 * cosv[i] - x1 * sinv[i];
            px[i + half] = x1 * cosv[i] + x0 * sinv[i];
        }
    }
    return Q35_KERN_OK;
}

/* ---- attention output gate (plain sigmoid) ---------------------------- */

void q35_kern_attn_gate_scalar(const float *attn, const float *gate,
                               float *y, size_t n)
{
    for (size_t i = 0; i < n; i++)
        y[i] = attn[i] * sigmoidf_(gate[i]);
}

void q35_kern_attn_gate(const float *attn, const float *gate, float *y,
                        size_t n)
{
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float tmp[8];
        for (int j = 0; j < 8; j++) tmp[j] = sigmoidf_(gate[i + j]);
        _mm256_storeu_ps(y + i,
            _mm256_mul_ps(_mm256_loadu_ps(attn + i), _mm256_loadu_ps(tmp)));
    }
    for (; i < n; i++) y[i] = attn[i] * sigmoidf_(gate[i]);
#else
    q35_kern_attn_gate_scalar(attn, gate, y, n);
#endif
}

/* ---- softmax (stable, fixed reduction order) --------------------------- */

void q35_kern_softmax_scalar(const float *x, float *y, size_t n)
{
    if (n == 0) return;
    float m = x[0];
    for (size_t i = 1; i < n; i++) if (x[i] > m) m = x[i];
    for (size_t i = 0; i < n; i++) y[i] = expf(x[i] - m);
    float s = sum_fixed(y, n);
    float inv = 1.0f / s;
    for (size_t i = 0; i < n; i++) y[i] = y[i] * inv;
}

void q35_kern_softmax(const float *x, float *y, size_t n)
{
#if defined(__AVX2__)
    if (n == 0) return;
    float m = x[0];
    for (size_t i = 1; i < n; i++) if (x[i] > m) m = x[i];
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float tmp[8];
        for (int j = 0; j < 8; j++) tmp[j] = expf(x[i + j] - m);
        _mm256_storeu_ps(y + i, _mm256_loadu_ps(tmp));
    }
    for (; i < n; i++) y[i] = expf(x[i] - m);
    float inv = 1.0f / sum_fixed(y, n);
    __m256 vs = _mm256_set1_ps(inv);
    i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), vs));
    for (; i < n; i++) y[i] = y[i] * inv;
#else
    q35_kern_softmax_scalar(x, y, n);
#endif
}

/* ---- fnv-1a ------------------------------------------------------------ */

unsigned q35_kern_fnv1a(const void *data, size_t nbytes)
{
    const unsigned char *p = (const unsigned char *)data;
    unsigned h = 2166136261u;
    for (size_t i = 0; i < nbytes; i++) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}