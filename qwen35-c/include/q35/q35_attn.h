#ifndef Q35_ATTN_H
#define Q35_ATTN_H

/* Ticket-18: GQA full-attention layer + KV cache.
 *
 * One full-attention layer forward, per numspec §2:
 *   q_proj (rows = q_heads * 2 * head_dim, per-head [query|gate] split)
 *   k_proj / v_proj
 *   -> per-head zero-centered RMSNorm on q/k (q35_kern_qk_norm)
 *   -> partial RoPE on q/k, first rotary_dim dims, rotate_half, theta=1e7
 *      (q35_kern_rope)
 *   -> GQA attention q_heads : kv_heads over the KV cache (f32 softmax,
 *      q35_kern_softmax), scale = 1/sqrt(head_dim)
 *   -> sigmoid output gate: attn * sigmoid(gate)  (q35_kern_attn_gate)
 *   -> o_proj
 *
 * Causal masking is structural: each token attends to cache[0 .. its own
 * position] only, because the cache holds nothing further-ahead. Prefill
 * calls this function with nt>1; decode with nt==1. Both paths run the same
 * per-token sequence and therefore produce bit-identical outputs.
 *
 * Projections are f32 matvecs over FP8 e4m3 weights with BF16 128x128 block
 * scales via q35_mm_fp8; q/k norm weights are f32 (zero-centered: y uses
 * (1+w)).
 */

#include <stdint.h>
#include <stddef.h>

#include "q35/q35_kern.h"

typedef struct {
    uint32_t hidden;      /* H (5120) */
    uint32_t q_heads;     /* 24 */
    uint32_t kv_heads;    /* 4 */
    uint32_t head_dim;    /* 256 */
    uint32_t rotary_dim;  /* 64 (partial_rotary_factor 0.25) */
    float    theta;       /* 1e7 */
    float    eps;         /* 1e-6, qk norm eps */
    int      dbg_layer;   /* debug-only: layer index for Q35_DUMP records */
} q35_attn_cfg;

typedef struct {
    /* FP8 e4m3 weights + bf16 128x128 block scales (q35_mm_fp8 layout) */
    const uint8_t  *wq;      /* [q_heads*2*head_dim, hidden] */
    const uint16_t *wq_sc;
    const uint8_t  *wk;      /* [kv_heads*head_dim, hidden] */
    const uint16_t *wk_sc;
    const uint8_t  *wv;      /* [kv_heads*head_dim, hidden] */
    const uint16_t *wv_sc;
    const uint8_t  *wo;      /* [hidden, q_heads*head_dim] */
    const uint16_t *wo_sc;
    /* zero-centered per-head rmsnorm weights, f32 [head_dim] */
    const float *q_nw;
    const float *k_nw;
} q35_attn_weights;

/* KV cache, layout: two f32 buffers
 *   k/v: [kv_heads][cap][head_dim]  (head-major, seq-contiguous).
 * len = number of tokens currently stored.
 * cap can grow up to max_cap (dynamic growth via q35_kvcache_grow).
 * Buffers come from large pages when the OS offers them (k_large/v_large),
 * else calloc — the free path must match the allocating one. */
typedef struct {
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t cap;
    uint32_t max_cap;  /* max capacity cap can grow to (0 or cap = no growth) */
    uint32_t len;
    int      k_large, v_large;  /* allocation kind of k/v (see above) */
    float   *k;
    float   *v;
} q35_kvcache;

enum q35_attn_err {
    Q35_ATTN_OK = 0,
    Q35_ATTN_ERR_ARG = -1,
    Q35_ATTN_ERR_FULL = -2
};

/* Allocates k/v buffers. cap is the initial capacity; max_cap is set to
 * cap (no growth by default). Use q35_kvcache_grow to expand. Returns 0 or
 * Q35_ATTN_ERR_ARG. */
Q35_API int q35_kvcache_init(q35_kvcache *c, uint32_t kv_heads,
                             uint32_t head_dim, uint32_t cap);
Q35_API void q35_kvcache_free(q35_kvcache *c);

/* Grow capacity to new_cap (must be > current cap). Existing data is
 * preserved per-head. Returns 0 or Q35_ATTN_ERR_ARG. */
Q35_API int q35_kvcache_grow(q35_kvcache *c, uint32_t new_cap);

/* Append one token's k/v (each [kv_heads * head_dim], kv-head-major).
 * Q35_ATTN_ERR_FULL when len == cap. */
Q35_API int q35_kvcache_append(q35_kvcache *c, const float *k, const float *v);

/* Full-attention forward for nt tokens (prefill nt>1 or decode nt==1).
 * x:   [nt * hidden] f32
 * out: [nt * hidden] f32
 * The nt new k/v entries are appended to cache; cache->len grows by nt.
 * Positions used for RoPE are cache->len_before .. cache->len_before+nt-1.
 */
Q35_API int q35_attn_forward(const q35_attn_cfg *cfg,
                             const q35_attn_weights *w,
                             q35_kvcache *cache,
                             const float *x, uint32_t nt, float *out);

/* Forward with caller-provided scratch (avoids per-call malloc).
 * scratch[0]=sq[qrows] [1]=sk[krows] [2]=sv[krows]
 * [3]=sa[qh*hd] [4]=sg[qh*hd] [5]=sc[max_cap*nthr] */
Q35_API int q35_attn_forward_s(const q35_attn_cfg *cfg,
                               const q35_attn_weights *w,
                               q35_kvcache *cache,
                               const float *x, uint32_t nt, float *out,
                               float *scratch[6]);

#endif
