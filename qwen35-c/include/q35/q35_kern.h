#ifndef Q35_KERN_H
#define Q35_KERN_H

/* Ticket-16: elementwise kernel pack for Qwen3.8-27B (Qwen3_5 arch).
 *
 * All kernels: pure functions, f32 working precision, HF-equivalent math.
 * Every kernel ships two entry points:
 *   q35_kern_<name>         default build; uses AVX2 SIMD when compiled with
 *                           __AVX2__, scalar otherwise.
 *   q35_kern_<name>_scalar  force the scalar path.
 * Both paths use an identical, fixed reduction order (8 fixed accumulators,
 * folded lane 0..7) so FNV-1a of the output bytes is identical across paths.
 * FP contraction is disabled in q35_kern.c to guarantee bit-identical output
 * between the two paths (offered FMA fusion traded for reproducibility).
 */

#include <stddef.h>

#ifdef _WIN32
#  define Q35_API __declspec(dllexport)
#else
#  define Q35_API
#endif

#define Q35_KERN_RMS_EPS 1e-6f

enum q35_kern_err {
    Q35_KERN_OK = 0,
    Q35_KERN_ERR_ARG = -1
};

/* Zero-centered RMSNorm: y = x/sqrt(mean(x^2)+eps) * (1+w).
 * x[n] in, w[n] zero-centered weights, y[n] out. mean/rsqrt in f32. */
Q35_API void q35_kern_rmsnorm(const float *x, const float *w, float *y,
                              size_t n, float eps);
Q35_API void q35_kern_rmsnorm_scalar(const float *x, const float *w, float *y,
                                     size_t n, float eps);

/* SiLU: y = x * sigmoid(x). */
Q35_API void q35_kern_silu(const float *x, float *y, size_t n);
Q35_API void q35_kern_silu_scalar(const float *x, float *y, size_t n);

/* SwiGLU: y = silu(gate) * up. */
Q35_API void q35_kern_swiglu(const float *gate, const float *up, float *y,
                             size_t n);
Q35_API void q35_kern_swiglu_scalar(const float *gate, const float *up,
                                    float *y, size_t n);

/* Per-head q/k RMSNorm, zero-centered, applied in place to x[heads*head_dim].
 * Each head vector normalized independently. */
Q35_API void q35_kern_qk_norm(float *x, const float *w, size_t heads,
                              size_t head_dim, float eps);
Q35_API void q35_kern_qk_norm_scalar(float *x, const float *w, size_t heads,
                                     size_t head_dim, float eps);

/* Partial RoPE in place on x[heads*head_dim].
 * Only the first rotary_dim dims of each head are rotated (rotate_half style:
 * the rotary_dim block is split into two rotary_dim/2 halves); the remaining
 * head_dim - rotary_dim dims pass through untouched.
 * inv_freq[i] = theta^(-2i/rotary_dim), i in [0, rotary_dim/2), f32.
 * Returns Q35_KERN_ERR_ARG if rotary_dim is odd, zero, or > head_dim. */
Q35_API int q35_kern_rope(float *x, size_t heads, size_t head_dim,
                          size_t rotary_dim, int position, float theta);
Q35_API int q35_kern_rope_scalar(float *x, size_t heads, size_t head_dim,
                                 size_t rotary_dim, int position, float theta);

/* Attention output gate: y = attn * sigmoid(gate)  (plain sigmoid, NOT silu). */
Q35_API void q35_kern_attn_gate(const float *attn, const float *gate,
                                float *y, size_t n);
Q35_API void q35_kern_attn_gate_scalar(const float *attn, const float *gate,
                                       float *y, size_t n);

/* Numerically stable softmax (max-subtracted), f32, fixed reduction order. */
Q35_API void q35_kern_softmax(const float *x, float *y, size_t n);
Q35_API void q35_kern_softmax_scalar(const float *x, float *y, size_t n);

/* FNV-1a 32-bit over raw bytes; used to assert scalar/SIMD path equality. */
Q35_API unsigned q35_kern_fnv1a(const void *data, size_t nbytes);

#endif