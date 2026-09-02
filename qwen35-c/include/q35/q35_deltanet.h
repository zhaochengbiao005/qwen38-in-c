#ifndef Q35_DELTANET_H
#define Q35_DELTANET_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#  define Q35_API __declspec(dllexport)
#else
#  define Q35_API
#endif

/* Ticket-17: Gated DeltaNet linear-attention layer, per-token forward.
 *
 * Implements the HF Qwen3_5 GatedDeltaNet recurrence (numspec sec 1):
 *   projections (in_proj_qkv/z/a/b, all FP8 block-dequant via q35_mm_fp8)
 *   -> depthwise causal conv1d (kernel 4) + SiLU
 *   -> per-head L2 norm on q,k; q scaled by 1/sqrt(head_k_dim)
 *   -> g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b), f32
 *   -> per-token state recurrence, S f32, caller-owned:
 *        S <- exp(g) * S
 *        kv  = S^T k
 *        S  += k outer (beta * (v - kv))
 *        o   = S^T q
 *   -> per-v-head RMSNormGated: o = rms(o) * w; o *= silu(z)
 *   -> out_proj (FP8)
 *
 * k_heads broadcast to v_heads by group index h / (v_heads/k_heads).
 *
 * L2 norm convention: x / sqrt(sum(x^2) + l2_eps).
 * RMSNormGated convention: x / sqrt(mean(x^2) + norm_eps) * w.
 *
 * Determinism: FP_CONTRACT OFF, fixed sequential reduction order in the
 * state recurrence; projection matvec determinism inherited from q35_mm.
 *
 * The same entry point serves decode (seq_len 1, incremental state update)
 * and prefill (serial loop over seq_len tokens).
 */

enum q35_dn_err {
    Q35_DN_OK = 0,
    Q35_DN_ERR_ARG = -1,
    Q35_DN_ERR_NOMEM = -2
};

#define Q35_DN_L2_EPS   1e-6f
#define Q35_DN_NORM_EPS 1e-6f

typedef struct {
    uint32_t hidden;       /* H */
    uint32_t k_heads;      /* Hk */
    uint32_t v_heads;      /* Hv, must be multiple of k_heads */
    uint32_t head_k_dim;   /* dk */
    uint32_t head_v_dim;   /* dv */

    /* FP8 weights, row-major [rows, hidden] (or [hidden, V] for w_out),
     * with BF16 per-128x128-block scales, laid out exactly like q35_mm. */
    const uint8_t  *w_qkv;  const uint16_t *sc_qkv;  /* [2*K+V, H] */
    const uint8_t  *w_z;    const uint16_t *sc_z;    /* [V, H] */
    const uint8_t  *w_a;    const uint16_t *sc_a;    /* [Hv, H] */
    const uint8_t  *w_b;    const uint16_t *sc_b;    /* [Hv, H] */
    const uint8_t  *w_out;  const uint16_t *sc_out;  /* [H, V] */

    const float *conv_w;      /* [2*K+V, 4] row-major, f32, no bias */
    const float *gate_norm_w; /* [head_v_dim], plain (not zero-centered) */
    const float *A_log;       /* [Hv] f32 */
    const float *dt_bias;     /* [Hv] f32 */

    float l2_eps;    /* pass 0 for Q35_DN_L2_EPS */
    float norm_eps;  /* pass 0 for Q35_DN_NORM_EPS */
} q35_deltanet_t;

/* Caller-owned persistent state. S must hold v_heads*head_k_dim*head_v_dim
 * floats, conv must hold (2*K+V)*4 floats. Zero-initialized = fresh layer. */
typedef struct {
    float *S;     /* [v_heads][head_k_dim][head_v_dim] row-major */
    float *conv;  /* [2*K+V][4], right-aligned last-4 conv inputs */
} q35_dn_state_t;

Q35_API void q35_dn_state_init(q35_dn_state_t *st, float *S, float *conv,
                               size_t qkv_dim);

/* Forward over seq_len tokens. x is [seq_len, hidden] row-major input,
 * y is [seq_len, hidden] row-major output. State is updated in place.
 * seq_len 1 = decode step; seq_len > 1 = serial prefill recurrence. */
Q35_API int q35_dn_forward(const q35_deltanet_t *layer, q35_dn_state_t *st,
                           const float *x, float *y, size_t seq_len);

/* forward with externally-provided scratch (6 buffers, see q35_dn_forward).
 * Avoids per-call malloc/free — caller allocates once, reuses across layers. */
Q35_API int q35_dn_forward_s(const q35_deltanet_t *layer, q35_dn_state_t *st,
                             const float *x, float *y, size_t seq_len,
                             float *scratch[6]);

/* Forward with externally-provided decode scratch (6) AND chunked-prefill
 * scratch (13). chunk_scratch is only used when seq_len > 1; pass NULL for
 * decode (seq_len==1). Sizes: see Q35_DN_MAX_CHUNK (64).
 * chunk_scratch[0..12] = q_g,k_g,v_g,o_g,g_vec,beta_vec,attn,v_beta,
 *                        k_beta,v_new,k_cd,v_adj,qk_attn */
Q35_API int q35_dn_forward_s2(const q35_deltanet_t *layer, q35_dn_state_t *st,
                              const float *x, float *y, size_t seq_len,
                              float *scratch[6], float *chunk_scratch[13]);

#endif
