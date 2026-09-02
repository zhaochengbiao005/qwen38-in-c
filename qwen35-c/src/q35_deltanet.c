/* Ticket-17: Gated DeltaNet linear-attention layer. See q35_deltanet.h.
 *
 * Numerical discipline: FP contraction disabled; the state recurrence uses a
 * fixed sequential reduction order so outputs are reproducible bit for bit.
 * All discretization and recurrence math runs in f32 per numspec sec 5.
 */
#pragma STDC FP_CONTRACT OFF

#include "q35/q35_deltanet.h"
#include "q35/q35_mm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static float dn_sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static float dn_softplusf(float x)
{
    /* log1p(exp(x)), stable for large positive x */
    return x > 20.0f ? x : log1pf(expf(x));
}

/* per-head L2 norm: y = x / sqrt(sum(x^2) + eps), sequential sum order */
static void dn_l2norm_head(float *x, size_t n, float eps)
{
    float ss = 0.0f;
    size_t i;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss + eps);
    for (i = 0; i < n; i++) x[i] = x[i] * inv;
}

/* causal depthwise conv1d step, kernel 4:
 * shift conv state right-append cur input, dot with taps, SiLU in place. */
static void dn_conv_step(float *conv, const float *tap, float *cur,
                         size_t channels)
{
    size_t c;
    for (c = 0; c < channels; c++) {
        float *s = conv + c * 4;
        float acc;
        s[0] = s[1];
        s[1] = s[2];
        s[2] = s[3];
        s[3] = cur[c];
        acc  = tap[c * 4 + 0] * s[0];
        acc += tap[c * 4 + 1] * s[1];
        acc += tap[c * 4 + 2] * s[2];
        acc += tap[c * 4 + 3] * s[3];
        cur[c] = acc * dn_sigmoidf(acc);
    }
}

/* one-token delta-rule update for a single value head.
 * S is [dk][dv]; kv and o are scratch of size dv. */
static void dn_delta_step(float *S, size_t dk, size_t dv,
                          const float *q, const float *k, const float *v,
                          float alpha, float beta, float *kv, float *o)
{
    size_t i, j;
    for (i = 0; i < dk * dv; i++) S[i] = S[i] * alpha;
    for (j = 0; j < dv; j++) {
        float acc = 0.0f;
        for (i = 0; i < dk; i++) acc += S[i * dv + j] * k[i];
        kv[j] = (v[j] - acc) * beta;
    }
    for (i = 0; i < dk; i++)
        for (j = 0; j < dv; j++)
            S[i * dv + j] += k[i] * kv[j];
    for (j = 0; j < dv; j++) {
        float acc = 0.0f;
        for (i = 0; i < dk; i++) acc += S[i * dv + j] * q[i];
        o[j] = acc;
    }
}

/* RMSNormGated: h = rms(h) * w; h *= silu(z). Plain weight parameterization
 * (no +1), norm first then gate, per numspec sec 1.5. */
static void dn_rmsnorm_gated(float *h, const float *z, const float *w,
                             size_t n, float eps)
{
    float ss = 0.0f;
    size_t i;
    for (i = 0; i < n; i++) ss += h[i] * h[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (i = 0; i < n; i++)
        h[i] = (h[i] * inv * w[i]) * (z[i] * dn_sigmoidf(z[i]));
}

/* ---- chunked gated delta rule (mathematically equivalent to recurrent,
 * different reduction order — matches HF torch_chunk_gated_delta_rule) ----
 *
 * Processes one chunk of C tokens for a single v-head. The chunk is fully
 * self-contained: all C tokens use the SAME incoming state S (snapshot before
 * this chunk), and S is updated once at the end.
 *
 * Math (C = chunk_size, dk = key_dim, dv = value_dim):
 *   g_cum = cumsum(g_vec)                    # [C]
 *   decay_mask[i][j] = exp(g_cum[i] - g_cum[j])  for j < i, else 0
 *   attn_raw[i][j] = -(k_beta[i]·k[j]) * decay_mask[i][j]  for j < i
 *   attn = (I + strict_lower_tri)^{-1} expansion of attn_raw  # in-chunk inverse
 *   v_new = attn @ (v * beta)                # [C, dv]
 *   k_cd = attn @ (k_beta * exp(g_cum))       # [C, dk] — cross-chunk decay key
 *   v_prime = k_cd @ S                       # [C, dv] — read from old state
 *   v_adj = v_new - v_prime                  # [C, dv]
 *   qk_attn = (q @ k^T) * decay_mask         # [C, C]
 *   out = (q * exp(g_cum)) @ S + qk_attn @ v_adj   # [C, dv]
 *   S = S * exp(g_cum[-1]) + sum_i (k[i] * exp(g_cum[-1] - g_cum[i]))^T @ v_adj[i]
 */
#define Q35_DN_MAX_CHUNK 64

/* per-v-head chunked delta-rule buffers: inputs, output and scratch, all
 * [C, dk] / [C, dv] / [C, C] slices owned by one calling thread. */
typedef struct {
    const float *q;         /* [C, dk] input */
    const float *k;         /* [C, dk] input */
    const float *v;         /* [C, dv] input */
    float *o;               /* [C, dv] output */
    const float *g_vec;     /* [C] */
    const float *beta_vec;  /* [C] */
    float *attn;            /* [C, C] scratch */
    float *v_beta;          /* [C, dv] scratch */
    float *k_beta;          /* [C, dk] scratch */
    float *v_new;           /* [C, dv] scratch */
    float *k_cumdecay;      /* [C, dk] scratch */
    float *v_adj;           /* [C, dv] scratch */
    float *qk_attn;         /* [C, C] scratch (intra-chunk attention) */
} Q35DnChunk;

static void dn_chunk_head(float *S, size_t dk, size_t dv, uint32_t C,
                          const Q35DnChunk *b)
{
    uint32_t i, j, r;

    /* g_cumsum */
    float g_cum[Q35_DN_MAX_CHUNK];
    g_cum[0] = b->g_vec[0];
    for (i = 1; i < C; i++) g_cum[i] = g_cum[i-1] + b->g_vec[i];

    /* v_beta = v * beta, k_beta = k * beta */
    for (i = 0; i < C; i++) {
        float beta = b->beta_vec[i];
        for (j = 0; j < dv; j++) b->v_beta[i * dv + j] = b->v[i * dv + j] * beta;
        for (j = 0; j < dk; j++) b->k_beta[i * dk + j] = b->k[i * dk + j] * beta;
    }

    /* attn[i][j] = -(k_beta[i] · k[j]) * exp(g_cum[j] - g_cum[i])  for j < i */
    memset(b->attn, 0, (size_t)C * C * sizeof(float));
    for (i = 0; i < C; i++)
        for (j = 0; j < i; j++) {
            float dot = 0.0f;
            for (r = 0; r < dk; r++)
                dot += b->k_beta[i * dk + r] * b->k[j * dk + r];
            b->attn[i * C + j] = -dot * expf(g_cum[i] - g_cum[j]);
        }

    /* in-chunk recurrence: (I + strict-lower-tri) expansion.
     * HF: row = attn[i,:i].clone(); sub = attn[:i,:i].clone();
     *     attn[i,:i] = row + (row.unsqueeze(-1) * sub).sum(-2)
     * Must use ORIGINAL row (not in-place modified values). */
    for (i = 1; i < C; i++) {
        float row[Q35_DN_MAX_CHUNK];
        for (j = 0; j < i; j++) row[j] = b->attn[i * C + j];  /* save original */
        for (j = 0; j < i; j++) {
            float corr = 0.0f;
            for (r = 0; r < i; r++)  /* sum over all r < i (sub is [:i,:i]) */
                corr += row[r] * b->attn[r * C + j];
            b->attn[i * C + j] = row[j] + corr;
        }
    }
    for (i = 0; i < C; i++) b->attn[i * C + i] += 1.0f;

    /* v_new = attn @ v_beta  (lower-tri, r <= i) */
    for (i = 0; i < C; i++)
        for (j = 0; j < dv; j++) {
            float acc = 0.0f;
            for (r = 0; r <= i; r++) acc += b->attn[i * C + r] * b->v_beta[r * dv + j];
            b->v_new[i * dv + j] = acc;
        }

    /* k_cumdecay = attn @ (k_beta * exp(g_cum)) */
    for (i = 0; i < C; i++)
        for (j = 0; j < dk; j++) {
            float acc = 0.0f;
            for (r = 0; r <= i; r++)
                acc += b->attn[i * C + r] * (b->k_beta[r * dk + j] * expf(g_cum[r]));
            b->k_cumdecay[i * dk + j] = acc;
        }

    /* v_adj = v_new - k_cumdecay @ S */
    for (i = 0; i < C; i++)
        for (j = 0; j < dv; j++) {
            float vp = 0.0f;
            for (r = 0; r < dk; r++) vp += b->k_cumdecay[i * dk + r] * S[r * dv + j];
            b->v_adj[i * dv + j] = b->v_new[i * dv + j] - vp;
        }

    /* qk_attn[i][j] = (q[i] · k[j]) * exp(g_cum[j] - g_cum[i])  for j <= i */
    memset(b->qk_attn, 0, (size_t)C * C * sizeof(float));
    for (i = 0; i < C; i++)
        for (j = 0; j <= i; j++) {
            float dot = 0.0f;
            for (r = 0; r < dk; r++) dot += b->q[i * dk + r] * b->k[j * dk + r];
            b->qk_attn[i * C + j] = dot * expf(g_cum[i] - g_cum[j]);
        }

    /* out[i] = (q[i] * exp(g_cum[i])) @ S + qk_attn[i] @ v_adj */
    for (i = 0; i < C; i++) {
        float eg_i = expf(g_cum[i]);
        for (j = 0; j < dv; j++) {
            float inter = 0.0f;
            for (r = 0; r < dk; r++) inter += b->q[i * dk + r] * eg_i * S[r * dv + j];
            float intra = 0.0f;
            for (r = 0; r <= i; r++) intra += b->qk_attn[i * C + r] * b->v_adj[r * dv + j];
            b->o[i * dv + j] = inter + intra;
        }
    }

    /* state update: S = S * exp(g_last) + sum_i (k[i] * exp(g_last - g_cum[i]))^T ⊗ v_adj[i] */
    float g_last = g_cum[C - 1];
    float exp_g_last = expf(g_last);
    for (r = 0; r < dk * dv; r++) S[r] *= exp_g_last;
    for (i = 0; i < C; i++) {
        float decay_i = expf(g_last - g_cum[i]);
        for (r = 0; r < dk; r++) {
            float kr = b->k[i * dk + r] * decay_i;
            for (j = 0; j < dv; j++)
                S[r * dv + j] += kr * b->v_adj[i * dv + j];
        }
    }
}

void q35_dn_state_init(q35_dn_state_t *st, float *S, float *conv,
                       size_t qkv_dim)
{
    st->S = S;
    st->conv = conv;
    memset(conv, 0, qkv_dim * 4 * sizeof(float));
}

int q35_dn_forward(const q35_deltanet_t *layer, q35_dn_state_t *st,
                   const float *x, float *y, size_t seq_len)
{
    /* legacy entry: allocate scratch internally */
    const uint32_t H = layer->hidden;
    const uint32_t Hv = layer->v_heads;
    const uint32_t dk = layer->head_k_dim;
    const uint32_t dv = layer->head_v_dim;
    const uint32_t Hk = layer->k_heads;
    const uint32_t K = Hk * dk;
    const uint32_t V = Hv * dv;
    const uint32_t QKV = 2 * K + V;
    int nthr = 1;
#ifdef _OPENMP
    nthr = omp_get_max_threads();
#endif
    float *scratch[6] = {
        malloc((size_t)QKV * sizeof(float)), malloc((size_t)V * sizeof(float)),
        malloc((size_t)Hv * sizeof(float)), malloc((size_t)Hv * sizeof(float)),
        malloc((size_t)V * sizeof(float)), malloc((size_t)nthr * dv * sizeof(float)),
    };
    if (!scratch[0] || !scratch[1] || !scratch[2] ||
        !scratch[3] || !scratch[4] || !scratch[5]) {
        for (int i = 0; i < 6; i++) free(scratch[i]);
        return Q35_DN_ERR_NOMEM;
    }
    int rc = q35_dn_forward_s(layer, st, x, y, seq_len, scratch);
    for (int i = 0; i < 6; i++) free(scratch[i]);
    return rc;
}

/* forward with externally-provided scratch (avoids per-call malloc/free).
 * scratch layout: [0]=qkv[QKV] [1]=zq[V] [2]=ao[Hv] [3]=bo[Hv] [4]=o[V]
 * [5]=head_in[nthr*dv] */
static int q35_dn_forward_s_impl(const q35_deltanet_t *layer, q35_dn_state_t *st,
                     const float *x, float *y, size_t seq_len,
                     float *scratch[6], float *chunk_scratch[13]);

int q35_dn_forward_s(const q35_deltanet_t *layer, q35_dn_state_t *st,
                     const float *x, float *y, size_t seq_len,
                     float *scratch[6])
{
    return q35_dn_forward_s_impl(layer, st, x, y, seq_len, scratch, NULL);
}

static int q35_dn_forward_s_impl(const q35_deltanet_t *layer, q35_dn_state_t *st,
                     const float *x, float *y, size_t seq_len,
                     float *scratch[6], float *chunk_scratch[13])
{
    const uint32_t H = layer->hidden;
    const uint32_t Hk = layer->k_heads;
    const uint32_t Hv = layer->v_heads;
    const uint32_t dk = layer->head_k_dim;
    const uint32_t dv = layer->head_v_dim;
    const uint32_t K = Hk * dk;
    const uint32_t V = Hv * dv;
    const uint32_t QKV = 2 * K + V;
    const uint32_t group = Hv / Hk;
    float l2_eps = layer->l2_eps > 0.0f ? layer->l2_eps : Q35_DN_L2_EPS;
    float norm_eps = layer->norm_eps > 0.0f ? layer->norm_eps : Q35_DN_NORM_EPS;
    const float q_scale = 1.0f / sqrtf((float)dk);
    float *qkv, *zq, *ao, *bo, *q, *k, *v, *o, *head_in;
    uint32_t h, i;

    if (!layer || !st || !st->S || !st->conv || !x || !y || !scratch) return Q35_DN_ERR_ARG;
    if (!H || !Hk || !Hv || !dk || !dv || (Hv % Hk)) return Q35_DN_ERR_ARG;

    qkv = scratch[0]; zq = scratch[1]; ao = scratch[2];
    bo = scratch[3]; o = scratch[4]; head_in = scratch[5];
    q = qkv;
    k = qkv + K;
    v = qkv + 2 * K;

    /* ===== single-token decode path (unchanged) ===== */
    if (seq_len <= 1) {
        const uint8_t *Ws[4]   = { layer->w_qkv, layer->w_z, layer->w_a, layer->w_b };
        const uint16_t *Ss[4]  = { layer->sc_qkv, layer->sc_z, layer->sc_a, layer->sc_b };
        uint32_t Rs[4] = { QKV, V, Hv, Hv };
        float *Ys[4] = { qkv, zq, ao, bo };
        q35_mm_fp8_multi4(Ws, Ss, Rs, H, x, Ys);

        dn_conv_step(st->conv, layer->conv_w, qkv, (size_t)QKV);

        for (h = 0; h < Hk; h++) {
            dn_l2norm_head(q + (size_t)h * dk, dk, l2_eps);
            dn_l2norm_head(k + (size_t)h * dk, dk, l2_eps);
            for (i = 0; i < dk; i++) q[(size_t)h * dk + i] *= q_scale;
        }

        #pragma omp parallel for schedule(static)
        for (h = 0; h < Hv; h++) {
            float g = -expf(layer->A_log[h]) * dn_softplusf(ao[h] + layer->dt_bias[h]);
            float alpha = expf(g);
            float beta = dn_sigmoidf(bo[h]);
            uint32_t kh = h / group;
            float *Sh = st->S + (size_t)h * dk * dv;
            float *my_kv;
#ifdef _OPENMP
            my_kv = head_in + (size_t)omp_get_thread_num() * dv;
#else
            my_kv = head_in;
#endif
            dn_delta_step(Sh, dk, dv,
                          q + (size_t)kh * dk, k + (size_t)kh * dk,
                          v + (size_t)h * dv, alpha, beta,
                          my_kv, o + (size_t)h * dv);
            dn_rmsnorm_gated(o + (size_t)h * dv, zq + (size_t)h * dv,
                             layer->gate_norm_w, dv, norm_eps);
        }

        q35_mm_fp8(layer->w_out, layer->sc_out, H, V, o, y);
        return Q35_DN_OK;
    }

    /* ===== chunked prefill path (seq_len > 1) =====
     * Batch all projections, then per-token conv1d+l2norm, then per v-head
     * dn_chunk_head processes the whole chunk at once (mathematically
     * equivalent to per-token dn_delta_step, different fp32 reduction order),
     * then per-token rmsnorm_gated + out_proj. */
    {
        uint32_t L = (uint32_t)seq_len;

        float *all_qkv = (float *)malloc((size_t)QKV * L * sizeof(float));
        float *all_zq  = (float *)malloc((size_t)V   * L * sizeof(float));
        float *all_ao  = (float *)malloc((size_t)Hv  * L * sizeof(float));
        float *all_bo  = (float *)malloc((size_t)Hv  * L * sizeof(float));
        float *all_q  = (float *)malloc((size_t)K * L * sizeof(float));
        float *all_k  = (float *)malloc((size_t)K * L * sizeof(float));
        float *all_v  = (float *)malloc((size_t)V * L * sizeof(float));
        float *all_o  = (float *)malloc((size_t)V * L * sizeof(float));
        if (!all_qkv || !all_zq || !all_ao || !all_bo ||
            !all_q || !all_k || !all_v || !all_o) {
            free(all_qkv); free(all_zq); free(all_ao); free(all_bo);
            free(all_q); free(all_k); free(all_v); free(all_o);
            return Q35_DN_ERR_NOMEM;
        }

        q35_mm_fp8_batch(layer->w_qkv, layer->sc_qkv, QKV, H, L, x, all_qkv);
        q35_mm_fp8_batch(layer->w_z,   layer->sc_z,   V,   H, L, x, all_zq);
        q35_mm_fp8_batch(layer->w_a,   layer->sc_a,   Hv,  H, L, x, all_ao);
        q35_mm_fp8_batch(layer->w_b,   layer->sc_b,   Hv,  H, L, x, all_bo);

        /* per-token conv1d + l2norm → fill all_q/all_k/all_v */
        for (size_t t = 0; t < L; t++) {
            memcpy(qkv, all_qkv + t * QKV, QKV * sizeof(float));
            dn_conv_step(st->conv, layer->conv_w, qkv, (size_t)QKV);
            for (h = 0; h < Hk; h++) {
                dn_l2norm_head(q + (size_t)h * dk, dk, l2_eps);
                dn_l2norm_head(k + (size_t)h * dk, dk, l2_eps);
                for (i = 0; i < dk; i++) q[(size_t)h * dk + i] *= q_scale;
            }
            memcpy(all_q + t * K, q, K * sizeof(float));
            memcpy(all_k + t * K, k, K * sizeof(float));
            memcpy(all_v + t * V, v, V * sizeof(float));
        }

        /* per v-head chunked delta rule.
         * Process tokens in chunks of Q35_DN_MAX_CHUNK; dn_chunk_head reads
         * the incoming state S, updates it in place, so sequential chunks
         * carry state forward exactly like the recurrent path. */
        {
            uint32_t C = Q35_DN_MAX_CHUNK;
            /* use external chunk_scratch if provided, else allocate */
            float *q_g,*k_g,*v_g,*o_g,*g_vec,*beta_vec;
            float *attn,*v_beta,*k_beta,*v_new,*k_cd,*v_adj,*qk_attn;
            float *cs[13];
            int allocated_ch = 0;
            if (chunk_scratch) {
                for (int i = 0; i < 13; i++) cs[i] = chunk_scratch[i];
            } else {
                int nthr_ch = 1;
#ifdef _OPENMP
                nthr_ch = omp_get_max_threads();
#endif
                cs[0]=malloc((size_t)C*dk*nthr_ch*sizeof(float)); cs[1]=malloc((size_t)C*dk*nthr_ch*sizeof(float));
                cs[2]=malloc((size_t)C*dv*nthr_ch*sizeof(float)); cs[3]=malloc((size_t)C*dv*nthr_ch*sizeof(float));
                cs[4]=malloc((size_t)C*nthr_ch*sizeof(float));   cs[5]=malloc((size_t)C*nthr_ch*sizeof(float));
                cs[6]=malloc((size_t)C*C*nthr_ch*sizeof(float));  cs[7]=malloc((size_t)C*dv*nthr_ch*sizeof(float));
                cs[8]=malloc((size_t)C*dk*nthr_ch*sizeof(float)); cs[9]=malloc((size_t)C*dv*nthr_ch*sizeof(float));
                cs[10]=malloc((size_t)C*dk*nthr_ch*sizeof(float)); cs[11]=malloc((size_t)C*dv*nthr_ch*sizeof(float));
                cs[12]=malloc((size_t)C*C*nthr_ch*sizeof(float));
                allocated_ch = 1;
                int ok = 1;
                for (int i = 0; i < 13; i++) if (!cs[i]) ok = 0;
                if (!ok) {
                    for (int i = 0; i < 13; i++) free(cs[i]);
                    free(all_qkv);free(all_zq);free(all_ao);free(all_bo);
                    free(all_q);free(all_k);free(all_v);free(all_o);
                    return Q35_DN_ERR_NOMEM;
                }
            }
            q_g=cs[0]; k_g=cs[1]; v_g=cs[2]; o_g=cs[3];
            g_vec=cs[4]; beta_vec=cs[5]; attn=cs[6]; v_beta=cs[7];
            k_beta=cs[8]; v_new=cs[9]; k_cd=cs[10]; v_adj=cs[11]; qk_attn=cs[12];

            /* per-thread offsets into the nthr-sized scratch arrays */
            #pragma omp parallel for schedule(static)
            for (h = 0; h < Hv; h++) {
                uint32_t kh = h / group;
                float *Sh = st->S + (size_t)h * dk * dv;
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                /* thread-local slice of each scratch buffer */
                float *my_q = q_g + (size_t)tid * C * dk;
                float *my_k = k_g + (size_t)tid * C * dk;
                float *my_v = v_g + (size_t)tid * C * dv;
                float *my_o = o_g + (size_t)tid * C * dv;
                float *my_gv = g_vec + (size_t)tid * C;
                float *my_bv = beta_vec + (size_t)tid * C;
                float *my_attn = attn + (size_t)tid * C * C;
                float *my_vb = v_beta + (size_t)tid * C * dv;
                float *my_kb = k_beta + (size_t)tid * C * dk;
                float *my_vn = v_new + (size_t)tid * C * dv;
                float *my_kcd = k_cd + (size_t)tid * C * dk;
                float *my_va = v_adj + (size_t)tid * C * dv;
                float *my_qka = qk_attn + (size_t)tid * C * C;

                for (uint32_t t0 = 0; t0 < L; t0 += C) {
                    uint32_t cur = L - t0 < C ? L - t0 : C;
                    for (size_t t = 0; t < cur; t++) {
                        memcpy(my_q + t*dk, all_q + (size_t)(t0+t)*K + kh*dk, dk*sizeof(float));
                        memcpy(my_k + t*dk, all_k + (size_t)(t0+t)*K + kh*dk, dk*sizeof(float));
                        memcpy(my_v + t*dv, all_v + (size_t)(t0+t)*V + h*dv, dv*sizeof(float));
                        my_gv[t] = -expf(layer->A_log[h]) *
                                   dn_softplusf(all_ao[(size_t)(t0+t)*Hv + h] + layer->dt_bias[h]);
                        my_bv[t] = dn_sigmoidf(all_bo[(size_t)(t0+t)*Hv + h]);
                    }
                    Q35DnChunk ck = { my_q, my_k, my_v, my_o, my_gv, my_bv,
                                      my_attn, my_vb, my_kb, my_vn, my_kcd,
                                      my_va, my_qka };
                    dn_chunk_head(Sh, dk, dv, cur, &ck);
                    for (size_t t = 0; t < cur; t++)
                        memcpy(all_o + (size_t)(t0+t)*V + h*dv, my_o + t*dv, dv*sizeof(float));
                }
            }
            if (allocated_ch)
                for (int i = 0; i < 13; i++) free(cs[i]);
        }

        /* per-token rmsnorm_gated + out_proj */
        for (size_t t = 0; t < L; t++) {
            memcpy(o, all_o + t*V, V*sizeof(float));
            memcpy(zq, all_zq + t*V, V*sizeof(float));
            for (h = 0; h < Hv; h++)
                dn_rmsnorm_gated(o + h*dv, zq + h*dv,
                                 layer->gate_norm_w, dv, norm_eps);
            q35_mm_fp8(layer->w_out, layer->sc_out, H, V, o, y + t*(size_t)H);
        }

        free(all_qkv); free(all_zq); free(all_ao); free(all_bo);
        free(all_q); free(all_k); free(all_v); free(all_o);
        return Q35_DN_OK;
    }
}

int q35_dn_forward_s2(const q35_deltanet_t *layer, q35_dn_state_t *st,
                      const float *x, float *y, size_t seq_len,
                      float *scratch[6], float *chunk_scratch[13])
{
    return q35_dn_forward_s_impl(layer, st, x, y, seq_len, scratch, chunk_scratch);
}
