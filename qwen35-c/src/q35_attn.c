/* Ticket-18: GQA full-attention layer + KV cache. See q35_attn.h.
 *
 * Numerical discipline mirrors q35_kern.c: FP contraction off, f32 working
 * precision, fixed reduction order for dot products and the softmax-weighted
 * v accumulation, so prefill and single-token decode run the exact same op
 * sequence for the same cache state and are bit-identical.
 */
#pragma STDC FP_CONTRACT OFF

#include "q35/q35_attn.h"
#include "q35/q35_mm.h"
#include "q35_plat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* Ticket-20 debug: sub-layer dump into the shared Q35_DUMP stream.
 * tags: 20 q (post-norm/rope), 21 k (post-norm/rope), 22 softmax p (per
 * head, kind=120+h), 23 attn out post-gate. */
extern FILE *q35_dbg_file(void);
static void adbg(int pos, int layer, int kind, const float *v, int n)
{
    FILE *f = q35_dbg_file();
    if (!f) return;
    int hdr[4] = { pos, layer, kind, n };
    fwrite(hdr, 4, 4, f); fwrite(v, 4, (size_t)n, f); fflush(f);
}

/* f32 dot of a,b (n elements), fixed 8-accumulator order (q35_kern style).
 * AVX2 path uses separate mul+add (not FMA) to stay bit-identical with the
 * scalar path under FP_CONTRACT OFF. Each lane j accumulates the same values
 * in the same order as acc[j] in the scalar version. */
static float dot_fixed(const float *a, const float *b, size_t n)
{
#if defined(__AVX2__)
    __m256 vacc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vacc = _mm256_add_ps(vacc, _mm256_mul_ps(va, vb));
    }
    _Alignas(32) float acc[8];
    _mm256_store_ps(acc, vacc);
    for (size_t j = 0; i < n; i++, j++) acc[j] = acc[j] + a[i] * b[i];
    float s = acc[0];
    for (int j = 1; j < 8; j++) s = s + acc[j];
    return s;
#else
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (size_t j = 0; j < 8; j++) acc[j] = acc[j] + a[i + j] * b[i + j];
    for (size_t j = 0; i < n; i++, j++) acc[j] = acc[j] + a[i] * b[i];
    float s = acc[0];
    for (int j = 1; j < 8; j++) s = s + acc[j];
    return s;
#endif
}

/* Allocate a zeroed kv buffer: large pages when the OS offers them, calloc
 * otherwise (ticket #31: the never-substituted VirtualAlloc fallback is
 * slower than calloc). *is_large records which free path applies. */
static float *kv_alloc(size_t n_floats, int *is_large)
{
    float *p = (float *)q35_plat_large_alloc(n_floats * sizeof(float));
    if (p) { *is_large = 1; return p; }
    *is_large = 0;
    return (float *)calloc(n_floats, sizeof(float));
}

static void kv_free(float *p, int is_large)
{
    if (!p) return;
    if (is_large) q35_plat_large_free(p);
    else free(p);
}

int q35_kvcache_init(q35_kvcache *c, uint32_t kv_heads, uint32_t head_dim,
                     uint32_t cap)
{
    if (!c || !kv_heads || !head_dim || !cap) return Q35_ATTN_ERR_ARG;
    size_t n = (size_t)kv_heads * (size_t)cap * head_dim;
    c->k = kv_alloc(n, &c->k_large);
    c->v = kv_alloc(n, &c->v_large);
    if (!c->k || !c->v) {
        kv_free(c->k, c->k_large); kv_free(c->v, c->v_large);
        c->k = c->v = 0;
        return Q35_ATTN_ERR_ARG;
    }
    c->kv_heads = kv_heads;
    c->head_dim = head_dim;
    c->cap = cap;
    c->max_cap = cap;
    c->len = 0;
    return Q35_ATTN_OK;
}

void q35_kvcache_free(q35_kvcache *c)
{
    if (!c) return;
    kv_free(c->k, c->k_large); kv_free(c->v, c->v_large);
    c->k = c->v = 0;
    c->k_large = c->v_large = 0;
    c->len = c->cap = c->max_cap = 0;
}

int q35_kvcache_grow(q35_kvcache *c, uint32_t new_cap)
{
    if (!c || !c->k || new_cap <= c->cap) return Q35_ATTN_ERR_ARG;
    size_t hd = c->head_dim;
    size_t n_new = (size_t)c->kv_heads * new_cap * hd;
    int nk_large, nv_large;
    float *new_k = kv_alloc(n_new, &nk_large);
    float *new_v = kv_alloc(n_new, &nv_large);
    if (!new_k || !new_v) {
        kv_free(new_k, nk_large); kv_free(new_v, nv_large);
        return Q35_ATTN_ERR_ARG;
    }
    /* copy existing live data per-head (layout [vh][cap][hd] -> [vh][new_cap][hd]) */
    for (uint32_t h = 0; h < c->kv_heads; h++) {
        memcpy(new_k + (size_t)h * new_cap * hd,
               c->k + (size_t)h * c->cap * hd,
               (size_t)c->len * hd * sizeof(float));
        memcpy(new_v + (size_t)h * new_cap * hd,
               c->v + (size_t)h * c->cap * hd,
               (size_t)c->len * hd * sizeof(float));
    }
    kv_free(c->k, c->k_large);
    kv_free(c->v, c->v_large);
    c->k = new_k;
    c->v = new_v;
    c->k_large = nk_large;
    c->v_large = nv_large;
    c->cap = new_cap;
    return Q35_ATTN_OK;
}

int q35_kvcache_append(q35_kvcache *c, const float *k, const float *v)
{
    if (!c || !k || !v || !c->k || !c->v) return Q35_ATTN_ERR_ARG;
    if (c->len >= c->cap) {
        if (c->max_cap > c->cap) {
            uint32_t new_cap = c->cap * 2;
            if (new_cap > c->max_cap) new_cap = c->max_cap;
            if (q35_kvcache_grow(c, new_cap) != Q35_ATTN_OK)
                return Q35_ATTN_ERR_FULL;
        } else {
            return Q35_ATTN_ERR_FULL;
        }
    }
    size_t hd = c->head_dim;
    for (uint32_t h = 0; h < c->kv_heads; h++) {
        float *dk = c->k + ((size_t)h * c->cap + c->len) * hd;
        float *dv = c->v + ((size_t)h * c->cap + c->len) * hd;
        memcpy(dk, k + (size_t)h * hd, hd * sizeof(float));
        memcpy(dv, v + (size_t)h * hd, hd * sizeof(float));
    }
    c->len++;
    return Q35_ATTN_OK;
}

/* post-projection core: takes pre-projected q/k/v and does norm, rope,
 * append, attend, gate, o_proj. sq/sv are const (read-only); sk is mutable
 * (modified by qk_norm and rope in place). */
static int attn_token_core(const q35_attn_cfg *cfg, const q35_attn_weights *w,
                           q35_kvcache *cache,
                           const float *sq, float *sk, const float *sv,
                           float *out, float *sa, float *sg, float *sc)
{
    const uint32_t qh = cfg->q_heads, kh = cfg->kv_heads, hd = cfg->head_dim;
    const uint32_t gsz = qh / kh;               /* query heads per kv head */
    const float scale = 1.0f / sqrtf((float)hd);
    float *q = sa;
    float *gate = sg;

    adbg((int)cache->len, cfg->dbg_layer, 26, sq, (int)(qh * 2 * hd));

    /* per-head [query|gate] split of the q_proj output */
    for (uint32_t h = 0; h < qh; h++) {
        memcpy(q + (size_t)h * hd, sq + (size_t)h * 2 * hd, hd * sizeof(float));
        memcpy(gate + (size_t)h * hd, sq + (size_t)h * 2 * hd + hd,
               hd * sizeof(float));
    }
    adbg((int)cache->len, cfg->dbg_layer, 28, q, (int)(qh * hd));

    q35_kern_qk_norm(q, w->q_nw, qh, hd, cfg->eps);
    q35_kern_qk_norm(sk, w->k_nw, kh, hd, cfg->eps);
    adbg((int)cache->len, cfg->dbg_layer, 29, q, (int)(qh * hd));

    int pos = (int)cache->len;            /* RoPE position = append index */
    if (q35_kern_rope(q, qh, hd, cfg->rotary_dim, pos, cfg->theta))
        return Q35_ATTN_ERR_ARG;
    if (q35_kern_rope(sk, kh, hd, cfg->rotary_dim, pos, cfg->theta))
        return Q35_ATTN_ERR_ARG;
    adbg(pos, cfg->dbg_layer, 20, q, (int)(qh * hd));
    adbg(pos, cfg->dbg_layer, 21, sk, (int)(kh * hd));

    if (q35_kvcache_append(cache, sk, sv)) return Q35_ATTN_ERR_FULL;

    /* GQA: query head h reads kv head h/gsz. Causality is structural: the
     * cache only ever contains the past (and the current token). The 24 query
     * heads are independent — parallelize across them, each thread gets its
     * own sc slice from the shared buffer (cap floats each, laid out
     * [thread][cap]). */
    uint32_t L = cache->len;
    int _nthr = 1;
#ifdef _OPENMP
    _nthr = omp_get_max_threads();
#endif
    #pragma omp parallel for schedule(static) num_threads(_nthr > (int)qh ? (int)qh : _nthr)
    for (uint32_t h = 0; h < qh; h++) {
        const float *kh_k = cache->k + ((size_t)(h / gsz) * cache->cap) * hd;
        const float *kh_v = cache->v + ((size_t)(h / gsz) * cache->cap) * hd;
        const float *qh_vec = q + (size_t)h * hd;
        float *o = sa + (size_t)h * hd;
        float *my_sc;
#ifdef _OPENMP
        my_sc = sc + (size_t)omp_get_thread_num() * cache->cap;
#else
        my_sc = sc;
#endif
        for (uint32_t t = 0; t < L; t++)
            my_sc[t] = dot_fixed(qh_vec, kh_k + (size_t)t * hd, hd) * scale;
        q35_kern_softmax(my_sc, my_sc, L);
        adbg(pos, cfg->dbg_layer, 120 + (int)h, sc, (int)L);
        /* o[i] = sum_t p_t * v[t][i], fixed cache-index order.
         * Vectorized with separate mul+add (not FMA) for bit-identical
         * results with the scalar path. */
        memset(o, 0, hd * sizeof(float));
        for (uint32_t t = 0; t < L; t++) {
            const float p = my_sc[t];
            const float *vrow = kh_v + (size_t)t * hd;
#if defined(__AVX2__)
            __m256 vp = _mm256_set1_ps(p);
            uint32_t i = 0;
            for (; i + 8 <= hd; i += 8) {
                __m256 vo = _mm256_loadu_ps(o + i);
                __m256 vv = _mm256_loadu_ps(vrow + i);
                _mm256_storeu_ps(o + i, _mm256_add_ps(vo, _mm256_mul_ps(vp, vv)));
            }
            for (; i < hd; i++) o[i] = o[i] + p * vrow[i];
#else
            for (uint32_t i = 0; i < hd; i++) o[i] = o[i] + p * vrow[i];
#endif
        }
    }

    /* sigmoid output gate on o_proj input, then o_proj */
    q35_kern_attn_gate(sa, gate, sa, (size_t)qh * hd);
    adbg(pos, cfg->dbg_layer, 23, sa, (int)(qh * hd));
    q35_mm_fp8(w->wo, w->wo_sc, cfg->hidden, qh * hd, sa, out);
    return Q35_ATTN_OK;
}

/* full per-token: project q/k/v then call core. */
static int attn_token(const q35_attn_cfg *cfg, const q35_attn_weights *w,
                      q35_kvcache *cache, const float *x, float *out,
                      float *sq, float *sk, float *sv, float *sa, float *sg,
                      float *sc)
{
    const uint32_t qh = cfg->q_heads, kh = cfg->kv_heads, hd = cfg->head_dim;
    q35_mm_fp8(w->wq, w->wq_sc, qh * 2 * hd, cfg->hidden, x, sq);
    q35_mm_fp8(w->wk, w->wk_sc, kh * hd, cfg->hidden, x, sk);
    q35_mm_fp8(w->wv, w->wv_sc, kh * hd, cfg->hidden, x, sv);
    return attn_token_core(cfg, w, cache, sq, sk, sv, out, sa, sg, sc);
}

int q35_attn_forward_s(const q35_attn_cfg *cfg, const q35_attn_weights *w,
                       q35_kvcache *cache, const float *x, uint32_t nt,
                       float *out, float *scratch[6])
{
    if (!cfg || !w || !cache || !x || !out || !nt || !scratch) return Q35_ATTN_ERR_ARG;
    if (!cache->k || !cfg->q_heads || !cfg->kv_heads || !cfg->head_dim ||
        cfg->q_heads % cfg->kv_heads)
        return Q35_ATTN_ERR_ARG;
    if (cache->kv_heads != cfg->kv_heads || cache->head_dim != cfg->head_dim)
        return Q35_ATTN_ERR_ARG;

    const uint32_t qh = cfg->q_heads, kh = cfg->kv_heads, hd = cfg->head_dim;
    const uint32_t qrows = qh * 2 * hd, krows = kh * hd;
    float *sq = scratch[0], *sk = scratch[1], *sv = scratch[2];
    float *sa = scratch[3], *sg = scratch[4], *sc = scratch[5];

    int rc = Q35_ATTN_OK;
    if (nt > 1) {
        /* batch project q/k/v for all nt tokens at once */
        float *all_sq = (float *)malloc((size_t)nt * qrows * sizeof(float));
        float *all_sk = (float *)malloc((size_t)nt * krows * sizeof(float));
        float *all_sv = (float *)malloc((size_t)nt * krows * sizeof(float));
        if (!all_sq || !all_sk || !all_sv) {
            free(all_sq); free(all_sk); free(all_sv);
            return Q35_ATTN_ERR_ARG;
        }
        q35_mm_fp8_batch(w->wq, w->wq_sc, qrows, cfg->hidden, nt, x, all_sq);
        q35_mm_fp8_batch(w->wk, w->wk_sc, krows, cfg->hidden, nt, x, all_sk);
        q35_mm_fp8_batch(w->wv, w->wv_sc, krows, cfg->hidden, nt, x, all_sv);
        for (uint32_t t = 0; t < nt; t++) {
            /* sk is modified by qk_norm/rope in core, so copy to mutable buf */
            memcpy(sk, all_sk + (size_t)t * krows, (size_t)krows * sizeof(float));
            rc = attn_token_core(cfg, w, cache,
                                 all_sq + (size_t)t * qrows,
                                 sk,
                                 all_sv + (size_t)t * krows,
                                 out + (size_t)t * cfg->hidden,
                                 sa, sg, sc);
            if (rc) break;
        }
        free(all_sq); free(all_sk); free(all_sv);
    } else {
        rc = attn_token(cfg, w, cache, x, out, sq, sk, sv, sa, sg, sc);
    }
    return rc;
}

int q35_attn_forward(const q35_attn_cfg *cfg, const q35_attn_weights *w,
                     q35_kvcache *cache, const float *x, uint32_t nt,
                     float *out)
{
    const uint32_t qh = cfg->q_heads, kh = cfg->kv_heads, hd = cfg->head_dim;
    const uint32_t qrows = qh * 2 * hd, krows = kh * hd;
    int _nthr = 1;
#ifdef _OPENMP
    _nthr = omp_get_max_threads();
#endif
    uint32_t sc_cap = cache->max_cap > cache->cap ? cache->max_cap : cache->cap;
    float *scratch[6] = {
        malloc((size_t)qrows * sizeof(float)), malloc((size_t)krows * sizeof(float)),
        malloc((size_t)krows * sizeof(float)), malloc((size_t)qh * hd * sizeof(float)),
        malloc((size_t)qh * hd * sizeof(float)),
        malloc((size_t)sc_cap * (size_t)_nthr * sizeof(float)),
    };
    if (!scratch[0] || !scratch[1] || !scratch[2] || !scratch[3] ||
        !scratch[4] || !scratch[5]) {
        for (int i = 0; i < 6; i++) free(scratch[i]);
        return Q35_ATTN_ERR_ARG;
    }
    int rc = q35_attn_forward_s(cfg, w, cache, x, nt, out, scratch);
    for (int i = 0; i < 6; i++) free(scratch[i]);
    return rc;
}
