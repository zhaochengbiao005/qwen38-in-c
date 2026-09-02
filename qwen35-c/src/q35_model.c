/* Ticket-19: full-model assembly. See include/q35/q35_model.h. */

#pragma STDC FP_CONTRACT OFF
#ifdef __clang__
#pragma clang fp contract(off)
#endif

#include "q35/q35_model.h"
#include "q35/q35_st.h"
#include "q35/q35_cfg.h"
#include "q35/q35_mm.h"
#include "q35/q35_kern.h"
#include "q35/q35_deltanet.h"
#include "q35/q35_attn.h"
#include "q35_plat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define Q35_PREFIX "model.language_model."
#define Q35_KV_CAP_DEFAULT 65536u

/* ---------- dtype helpers ---------- */

static uint16_t f32_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u); /* round to nearest even */
    return (uint16_t)(u >> 16);
}

/* nearest e4m3fn byte by brute force over the exact dequant LUT;
   ties break toward the even byte index (round-half-even). */
static uint8_t f32_e4m3(float v)
{
    float a = fabsf(v);
    float best_d = 1e30f;
    int best = 0;
    int i;
    if (a != a) return 0x7Fu;
    for (i = 0; i < 128; i++) {
        float lv = q35_mm_fp8_lut[i];
        float d;
        if (lv != lv) continue;
        d = fabsf(a - lv);
        if (d < best_d) { best_d = d; best = i; }
        else if (d == best_d && (i & 1) == 0 && (best & 1)) { best = i; }
    }
    return (uint8_t)((v < 0.0f ? 0x80u : 0u) | (uint8_t)best);
}

/* ---------- read-only shard mapping (mmap via PAL; indices match q35_st) -- */

typedef struct { Q35PlatMmap *mm; const uint8_t *base; } Q35Map;

/* Enumerate with the exact same predicate as q35_st_open_dir so that the
   file ordering (and hence Q35Tensor.file_idx) matches the index built by
   q35_st: q35_st_is_shard_name(). */
static int map_files(const char *dir, Q35Map **out, int *nout)
{
    Q35Map *maps = NULL;
    int n = 0;
    Q35PlatDir *pd = q35_plat_dir_open(dir);
    if (!pd) return -1;
    const char *nm;
    while ((nm = q35_plat_dir_next(pd)) != NULL) {
        if (!q35_st_is_shard_name(nm)) continue;
        char full[4096];
        q35_plat_path_join(full, sizeof(full), dir, nm);
        fprintf(stderr, "model: mapping shard %d: %s\n", n, nm);
        const uint8_t *base = NULL;
        Q35PlatMmap *mm = q35_plat_mmap_ro(full, &base);
        if (!mm) goto fail;
        Q35Map *nm2 = (Q35Map *)realloc(maps, (size_t)(n + 1) * sizeof(Q35Map));
        if (!nm2) { q35_plat_munmap(mm); goto fail; }
        maps = nm2;
        maps[n].mm = mm;
        maps[n].base = base;
        n++;
        continue;
    fail:
        q35_plat_dir_close(pd);
        while (n > 0) { n--; q35_plat_munmap(maps[n].mm); }
        free(maps);
        return -1;
    }
    q35_plat_dir_close(pd);
    if (n == 0) { free(maps); return -1; }
    *out = maps;
    *nout = n;
    return 0;
}

static void unmap_files(Q35Map *maps, int n)
{
    int i;
    if (!maps) return;
    for (i = 0; i < n; i++)
        q35_plat_munmap(maps[i].mm);
    free(maps);
}

/* ---------- model object ---------- */

typedef struct {
    int type; /* Q35_LAYER_LINEAR / Q35_LAYER_FULL */
    float *w_in_ln;   /* [H] f32, zero-centered RMSNorm weight */
    float *w_post_ln; /* [H] f32, zero-centered RMSNorm weight */
    /* dense SwiGLU MLP, FP8 + BF16 128x128 block scales */
    const uint8_t *w_gate;  const uint16_t *sc_gate;  /* [I, H] */
    const uint8_t *w_up;    const uint16_t *sc_up;    /* [I, H] */
    const uint8_t *w_down;  const uint16_t *sc_down;  /* [H, I] */
    union {
        struct {
            q35_deltanet_t layer;
            q35_dn_state_t state;
            float *S;      /* owned */
            float *conv;   /* owned state buffer */
            float *conv_w, *gnorm, *alog, *dt; /* owned f32 copies */
            uint8_t *ab_q;              /* owned quantized a/b (nullable) */
            uint16_t *ab_sc;
        } dn;
        struct {
            q35_attn_cfg cfg;
            q35_attn_weights w;
            q35_kvcache kv;
            float *q_nw, *k_nw; /* owned f32 copies */
        } fa;
    } u;
} Q35ModelLayer;

struct Q35Model {
    Q35Cfg cfg;
    Q35St *st;
    Q35Map *maps;
    int nmaps;

    const uint16_t *w_embed;  /* BF16 [vocab,H] mapped */
    const uint16_t *w_lmhead; /* BF16 [vocab,H] mapped */
    float *w_norm;            /* final norm, f32 [H], owned */

    Q35ModelLayer *layers;
    int nlayers;

    float *x, *xn, *y, *gate, *up, *mid; /* scratch */
    float *dn_scratch[6];  /* DeltaNet decode scratch (qkv/zq/ao/bo/o/head_in), allocated once */
    float *attn_scratch[6]; /* attention scratch (sq/sk/sv/sa/sg/sc), allocated once */
    /* DeltaNet chunked-prefill scratch (13 buffers × nthr copies, allocated
     * once, reused across all 48 DeltaNet layers per prefill call). Each
     * thread uses its own set for parallel v-head processing. */
    float *dn_ch[13];
    int dn_ch_nthr;

    uint32_t pos;
    uint32_t kv_cap;
    void **owned;
    size_t nowned, cap_owned;
};

const char *q35_model_err_str(int e)
{
    switch (e) {
    case Q35_MODEL_OK: return "ok";
    case Q35_MODEL_ERR_ARG: return "bad argument";
    case Q35_MODEL_ERR_IO: return "io error";
    case Q35_MODEL_ERR_CFG: return "config error";
    case Q35_MODEL_ERR_ST: return "safetensors error";
    case Q35_MODEL_ERR_MISSING: return "missing tensor";
    case Q35_MODEL_ERR_SHAPE: return "shape/dtype mismatch";
    case Q35_MODEL_ERR_NOMEM: return "out of memory";
    case Q35_MODEL_ERR_STATE: return "kv cache exhausted";
    }
    return "unknown";
}

static int own(Q35Model *m, void *p)
{
    if (!p) return 0;
    if (m->nowned == m->cap_owned) {
        size_t nc = m->cap_owned ? m->cap_owned * 2 : 64;
        void **np = (void **)realloc(m->owned, nc * sizeof(void *));
        if (!np) return 0;
        m->owned = np;
        m->cap_owned = nc;
    }
    m->owned[m->nowned++] = p;
    return 1;
}

static void *ownz(Q35Model *m, size_t bytes)
{
    void *p = calloc(1, bytes ? bytes : 1);
    if (p && !own(m, p)) { free(p); return NULL; }
    return p;
}

static const uint8_t *tensor_ptr(const Q35Model *m, const Q35Tensor *t)
{
    if (!t || t->file_idx < 0 || t->file_idx >= m->nmaps) return NULL;
    return m->maps[t->file_idx].base + t->data_off;
}

/* ---------- loader error slot ---------- */

/* Loader helpers share one error slot (message buffer + status code)
 * instead of threading (errbuf, errcap, &er) through every signature. */
typedef struct {
    char *buf;   /* nullable caller buffer */
    int   cap;
    int   code;  /* Q35_MODEL_* */
} Q35Err;

#define QERR(e, code_, ...) \
    do { if ((e)->buf) snprintf((e)->buf, (size_t)(e)->cap, __VA_ARGS__); \
         (e)->code = (code_); } while (0)

/* locate + strict-validate a tensor. On error: fills *e, returns NULL */
static const Q35Tensor *need_tensor(const Q35Model *m, const char *name,
                                    Q35Dtype dt, int ndim,
                                    const uint32_t *shape, Q35Err *e)
{
    const Q35Tensor *t = q35_st_find(m->st, name);
    int i;
    if (!t) { QERR(e, Q35_MODEL_ERR_MISSING, "missing tensor: %s", name); return NULL; }
    if (t->dtype != dt) {
        QERR(e, Q35_MODEL_ERR_SHAPE, "tensor %s: dtype %s, want %s",
             name, q35_dtype_name(t->dtype), q35_dtype_name(dt));
        return NULL;
    }
    if ((int)t->ndim != ndim) {
        QERR(e, Q35_MODEL_ERR_SHAPE, "tensor %s: ndim %u, want %d",
             name, t->ndim, ndim);
        return NULL;
    }
    for (i = 0; i < ndim; i++) {
        if (t->shape[i] != shape[i]) {
            QERR(e, Q35_MODEL_ERR_SHAPE, "tensor %s: shape[%d]=%u, want %u",
                 name, i, t->shape[i], shape[i]);
            return NULL;
        }
    }
    return t;
}

/* bind FP8 weight + BF16 128x128 block-scale pointers straight into mapping */
static int bind_fp8(const Q35Model *m, const char *wname,
                    uint32_t rows, uint32_t cols,
                    const uint8_t **w, const uint16_t **sc, Q35Err *e)
{
    uint32_t sc_shape[2] = { (rows + 127u) / 128u, (cols + 127u) / 128u };
    uint32_t w_shape[2] = { rows, cols };
    char sname[256];
    const Q35Tensor *tw, *ts;
    tw = need_tensor(m, wname, Q35_DTYPE_F8_E4M3, 2, w_shape, e);
    if (!tw) return e->code;
    snprintf(sname, sizeof(sname), "%s_scale_inv", wname);
    ts = need_tensor(m, sname, Q35_DTYPE_BF16, 2, sc_shape, e);
    if (!ts) return e->code;
    *w = tensor_ptr(m, tw);
    *sc = (const uint16_t *)tensor_ptr(m, ts);
    if (!*w || !*sc) {
        QERR(e, Q35_MODEL_ERR_IO, "tensor %s not mapped", wname);
        return Q35_MODEL_ERR_IO;
    }
    return Q35_MODEL_OK;
}

/* load BF16 small tensor -> owned f32 buffer */
static float *load_f32(Q35Model *m, const char *name, int ndim,
                       const uint32_t *shape, size_t nelem, Q35Err *e)
{
    const Q35Tensor *t = need_tensor(m, name, Q35_DTYPE_BF16, ndim, shape, e);
    const uint16_t *src;
    float *dst;
    size_t i;
    if (!t) return NULL;
    src = (const uint16_t *)tensor_ptr(m, t);
    if (!src) {
        QERR(e, Q35_MODEL_ERR_IO, "tensor %s not mapped", name);
        return NULL;
    }
    dst = (float *)ownz(m, nelem * sizeof(float));
    if (!dst) {
        QERR(e, Q35_MODEL_ERR_NOMEM, "oom: %s", name);
        return NULL;
    }
    for (i = 0; i < nelem; i++) dst[i] = q35_mm_bf16_to_f32(src[i]);
    return dst;
}

/* linear_attn.in_proj_a/b: BF16 [v_heads,H] in the real repo (no scale);
   quantize to FP8 e4m3 + BF16 block scales at load so q35_deltanet's
   FP8-only matvec contract is satisfied. If the tensor already is
   F8_E4M3 (e.g. fixtures), bind directly instead. */
static int bind_ab(Q35Model *m, const char *wname, uint32_t rows, uint32_t cols,
                   const uint8_t **w, const uint16_t **sc, Q35Err *e)
{
    uint32_t w_shape[2] = { rows, cols };
    const Q35Tensor *tw = q35_st_find(m->st, wname);
    if (!tw) { QERR(e, Q35_MODEL_ERR_MISSING, "missing tensor: %s", wname); return Q35_MODEL_ERR_MISSING; }
    if (tw->dtype == Q35_DTYPE_F8_E4M3)
        return bind_fp8(m, wname, rows, cols, w, sc, e);
    if (tw->dtype != Q35_DTYPE_BF16) {
        QERR(e, Q35_MODEL_ERR_SHAPE, "tensor %s: dtype %s, want BF16 or F8_E4M3",
             wname, q35_dtype_name(tw->dtype));
        return Q35_MODEL_ERR_SHAPE;
    }
    tw = need_tensor(m, wname, Q35_DTYPE_BF16, 2, w_shape, e);
    if (!tw) return e->code;
    {
        uint32_t sbr = (rows + 127u) / 128u, sbc = (cols + 127u) / 128u;
        const uint16_t *src = (const uint16_t *)tensor_ptr(m, tw);
        uint8_t *q = (uint8_t *)ownz(m, (size_t)rows * cols);
        uint16_t *s = (uint16_t *)ownz(m, (size_t)sbr * sbc * sizeof(uint16_t));
        uint32_t r, c, r0, c0;
        if (!src) return Q35_MODEL_ERR_IO;
        if (!q || !s) return Q35_MODEL_ERR_NOMEM;
        for (r0 = 0; r0 < rows; r0 += 128) {
            for (c0 = 0; c0 < cols; c0 += 128) {
                uint32_t rend = r0 + 128 < rows ? r0 + 128 : rows;
                uint32_t cend = c0 + 128 < cols ? c0 + 128 : cols;
                float mx = 0.0f, sc;
                for (r = r0; r < rend; r++)
                    for (c = c0; c < cend; c++) {
                        float v = fabsf(q35_mm_bf16_to_f32(src[(size_t)r * cols + c]));
                        if (v > mx) mx = v;
                    }
                sc = mx / 448.0f;
                sc = q35_mm_bf16_to_f32(f32_bf16(sc));
                s[(size_t)(r0 / 128) * sbc + c0 / 128] = f32_bf16(sc);
                if (sc == 0.0f) continue;
                for (r = r0; r < rend; r++)
                    for (c = c0; c < cend; c++) {
                        float v = q35_mm_bf16_to_f32(src[(size_t)r * cols + c]);
                        q[(size_t)r * cols + c] = f32_e4m3(v / sc);
                    }
            }
        }
        *w = q;
        *sc = s;
    }
    return Q35_MODEL_OK;
}

/* ---------- load ---------- */

Q35Model *q35_model_load(const char *dir, uint32_t kv_cap,
                         char *errbuf, int errcap, int *err)
{
    Q35Model *m = NULL;
    char cfgpath[4096];
    Q35StErr ste;
    int i;
    Q35Err qe = { errbuf, errcap, Q35_MODEL_OK };

#define MFAIL(c) do { qe.code = (c); goto fail; } while (0)
#define CHK(expr) do { int rc_ = (expr); if (rc_ != Q35_MODEL_OK) MFAIL(rc_); } while (0)

    if (!dir) { qe.code = Q35_MODEL_ERR_ARG; goto fail; }
    q35_mm_init();

    m = (Q35Model *)calloc(1, sizeof(Q35Model));
    if (!m) MFAIL(Q35_MODEL_ERR_NOMEM);

    q35_plat_path_join(cfgpath, sizeof(cfgpath), dir, "config.json");
    if (q35_cfg_load(cfgpath, &m->cfg, errbuf, errcap) != 0)
        MFAIL(Q35_MODEL_ERR_CFG);

    m->st = q35_st_open_dir(dir, &ste);
    if (!m->st) {
        QERR(&qe, Q35_MODEL_ERR_ST, "q35_st_open_dir: %s", q35_st_err_str(ste));
        MFAIL(Q35_MODEL_ERR_ST);
    }

    if (map_files(dir, &m->maps, &m->nmaps) != 0) {
        QERR(&qe, Q35_MODEL_ERR_IO, "cannot map shards in %s", dir);
        MFAIL(Q35_MODEL_ERR_IO);
    }

    /* architecture validation (config-driven; 64 layers / 3:1 mix only
       checked for consistency, not hardcoded) */
    {
        const Q35Cfg *c = &m->cfg;
        int vh = c->linear_num_value_heads * c->linear_value_head_dim;
        int kh = c->linear_num_key_heads * c->linear_key_head_dim;
        int qkv = 2 * kh + vh;
        if (c->hidden_size <= 0 || c->intermediate_size <= 0 ||
            c->vocab_size <= 0 || c->num_hidden_layers <= 0) {
            QERR(&qe, Q35_MODEL_ERR_CFG, "bad base dims");
            MFAIL(Q35_MODEL_ERR_CFG);
        }
        if (vh <= 0 || kh <= 0 ||
            c->linear_num_value_heads % c->linear_num_key_heads != 0) {
            QERR(&qe, Q35_MODEL_ERR_CFG, "linear head dims invalid");
            MFAIL(Q35_MODEL_ERR_CFG);
        }
        if (qkv <= 0) MFAIL(Q35_MODEL_ERR_CFG);
        if (c->num_attention_heads <= 0 || c->num_key_value_heads <= 0 ||
            c->num_attention_heads % c->num_key_value_heads != 0 ||
            c->head_dim <= 0) {
            QERR(&qe, Q35_MODEL_ERR_CFG, "attn head dims invalid");
            MFAIL(Q35_MODEL_ERR_CFG);
        }
        {
            double rd = c->head_dim * c->partial_rotary_factor;
            int rd_i = (int)(rd + 0.5);
            if (rd_i <= 0 || rd_i > c->head_dim || (rd_i & 1) != 0) {
                QERR(&qe, Q35_MODEL_ERR_CFG, "rotary_dim %d invalid", rd_i);
                MFAIL(Q35_MODEL_ERR_CFG);
            }
        }
        m->nlayers = c->num_hidden_layers;
        m->kv_cap = kv_cap ? kv_cap
                           : (c->max_position_embeddings > 0 &&
                              (uint32_t)c->max_position_embeddings < Q35_KV_CAP_DEFAULT
                                  ? (uint32_t)c->max_position_embeddings
                                  : Q35_KV_CAP_DEFAULT);
        if (m->kv_cap == 0) MFAIL(Q35_MODEL_ERR_CFG);

        /* OOM guard: estimate KV cache memory (16 full-attn layers × 2 ×
         * KVH × HD × cap × 4 bytes) and auto-reduce cap if it would push
         * total RSS past 80% of available physical memory. */
        {
            uint32_t n_full = 0;
            for (int i = 0; i < m->nlayers; i++)
                if (c->layer_type[i] == Q35_LAYER_FULL) n_full++;
            /* weights ~25 GB for the 27B FP8 model; KV cache is additional */
            double weight_gb = 26.0;  /* conservative estimate */
            double avail_gb = 0;
            uint64_t avail_bytes = q35_plat_avail_phys();
            if (avail_bytes > 0)
                avail_gb = (double)avail_bytes / 1073741824.0;
            if (avail_gb > 0) {
                double kv_per_cap_gb = (double)n_full * 2 *
                    c->num_key_value_heads * c->head_dim * 4.0 / 1073741824.0;
                double budget_gb = avail_gb * 0.8 - weight_gb;
                if (budget_gb < 1.0) budget_gb = 1.0;
                uint32_t max_cap = (uint32_t)(budget_gb / (kv_per_cap_gb > 0 ? kv_per_cap_gb : 1));
                if (max_cap < m->kv_cap) {
                    /* informational only: not a failure */
                    if (qe.buf)
                        snprintf(qe.buf, (size_t)qe.cap,
                                 "KV cache cap reduced %u -> %u (avail %.0f GB, weight ~%.0f GB)",
                                 m->kv_cap, max_cap, avail_gb, weight_gb);
                    m->kv_cap = max_cap > 0 ? max_cap : 1;
                }
            }
        }
    }

    m->layers = (Q35ModelLayer *)ownz(m, sizeof(Q35ModelLayer) *
                                      (size_t)m->nlayers);
    if (!m->layers) MFAIL(Q35_MODEL_ERR_NOMEM);

    {
        const Q35Cfg *c = &m->cfg;
        uint32_t H = (uint32_t)c->hidden_size;
        uint32_t II = (uint32_t)c->intermediate_size;
        uint32_t V = (uint32_t)c->linear_num_value_heads *
                     (uint32_t)c->linear_value_head_dim;
        uint32_t K = (uint32_t)c->linear_num_key_heads *
                     (uint32_t)c->linear_key_head_dim;
        uint32_t QKV = 2 * K + V;
        uint32_t Hv = (uint32_t)c->linear_num_value_heads;
        uint32_t QH = (uint32_t)c->num_attention_heads;
        uint32_t KVH = (uint32_t)c->num_key_value_heads;
        uint32_t HD = (uint32_t)c->head_dim;
        uint32_t RD = (uint32_t)(c->head_dim * c->partial_rotary_factor + 0.5);

        for (i = 0; i < m->nlayers; i++) {
            Q35ModelLayer *L = &m->layers[i];
            char nb[256];
            uint32_t sh1[1];

            L->type = c->layer_type[i];

            snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.input_layernorm.weight", i);
            sh1[0] = H;
            L->w_in_ln = load_f32(m, nb, 1, sh1, H, &qe);
            if (!L->w_in_ln) goto fail;
            snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.post_attention_layernorm.weight", i);
            L->w_post_ln = load_f32(m, nb, 1, sh1, H, &qe);
            if (!L->w_post_ln) goto fail;

            if (L->type == Q35_LAYER_LINEAR) {
                q35_deltanet_t *dn = &L->u.dn.layer;
                dn->hidden = H;
                dn->k_heads = (uint32_t)c->linear_num_key_heads;
                dn->v_heads = Hv;
                dn->head_k_dim = (uint32_t)c->linear_key_head_dim;
                dn->head_v_dim = (uint32_t)c->linear_value_head_dim;
                dn->l2_eps = 0.0f;   /* Q35_DN_L2_EPS */
                dn->norm_eps = 0.0f; /* Q35_DN_NORM_EPS */

                snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.linear_attn.in_proj_qkv.weight", i);
                CHK(bind_fp8(m, nb, QKV, H, &dn->w_qkv, &dn->sc_qkv, &qe));
                snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.linear_attn.in_proj_z.weight", i);
                CHK(bind_fp8(m, nb, V, H, &dn->w_z, &dn->sc_z, &qe));
                snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.linear_attn.in_proj_a.weight", i);
                CHK(bind_ab(m, nb, Hv, H, &dn->w_a, &dn->sc_a, &qe));
                snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.linear_attn.in_proj_b.weight", i);
                CHK(bind_ab(m, nb, Hv, H, &dn->w_b, &dn->sc_b, &qe));
                snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.linear_attn.out_proj.weight", i);
                CHK(bind_fp8(m, nb, H, V, &dn->w_out, &dn->sc_out, &qe));

                {
                    uint32_t shc[3] = { QKV, 1, (uint32_t)c->linear_conv_kernel_dim };
                    snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.linear_attn.conv1d.weight", i);
                    dn->conv_w = load_f32(m, nb, 3, shc,
                                          (size_t)QKV * (size_t)c->linear_conv_kernel_dim,
                                          &qe);
                    if (!dn->conv_w) goto fail;
                }
                sh1[0] = (uint32_t)c->linear_value_head_dim;
                snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.linear_attn.norm.weight", i);
                dn->gate_norm_w = load_f32(m, nb, 1, sh1, sh1[0], &qe);
                if (!dn->gate_norm_w) goto fail;
                sh1[0] = Hv;
                snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.linear_attn.A_log", i);
                dn->A_log = load_f32(m, nb, 1, sh1, Hv, &qe);
                if (!dn->A_log) goto fail;
                snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.linear_attn.dt_bias", i);
                dn->dt_bias = load_f32(m, nb, 1, sh1, Hv, &qe);
                if (!dn->dt_bias) goto fail;

                L->u.dn.S = (float *)ownz(m, (size_t)Hv * c->linear_key_head_dim *
                                            c->linear_value_head_dim * sizeof(float));
                L->u.dn.conv = (float *)ownz(m, (size_t)QKV * 4 * sizeof(float));
                if (!L->u.dn.S || !L->u.dn.conv) MFAIL(Q35_MODEL_ERR_NOMEM);
                q35_dn_state_init(&L->u.dn.state, L->u.dn.S, L->u.dn.conv, QKV);
            } else if (L->type == Q35_LAYER_FULL) {
                q35_attn_cfg *ac = &L->u.fa.cfg;
                q35_attn_weights *aw = &L->u.fa.w;
                ac->hidden = H;
                ac->q_heads = QH;
                ac->kv_heads = KVH;
                ac->head_dim = HD;
                ac->rotary_dim = RD;
                ac->theta = (float)c->rope_theta;
                ac->eps = (float)c->rms_norm_eps;
                ac->dbg_layer = i;

                snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.self_attn.q_proj.weight", i);
                CHK(bind_fp8(m, nb, QH * 2 * HD, H, &aw->wq, &aw->wq_sc, &qe));
                snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.self_attn.k_proj.weight", i);
                CHK(bind_fp8(m, nb, KVH * HD, H, &aw->wk, &aw->wk_sc, &qe));
                snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.self_attn.v_proj.weight", i);
                CHK(bind_fp8(m, nb, KVH * HD, H, &aw->wv, &aw->wv_sc, &qe));
                snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.self_attn.o_proj.weight", i);
                CHK(bind_fp8(m, nb, H, QH * HD, &aw->wo, &aw->wo_sc, &qe));

                sh1[0] = HD;
                snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.self_attn.q_norm.weight", i);
                aw->q_nw = load_f32(m, nb, 1, sh1, HD, &qe);
                if (!aw->q_nw) goto fail;
                snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.self_attn.k_norm.weight", i);
                aw->k_nw = load_f32(m, nb, 1, sh1, HD, &qe);
                if (!aw->k_nw) goto fail;

                if (q35_kvcache_init(&L->u.fa.kv, KVH, HD,
                                     m->kv_cap > 4096 ? 4096 : m->kv_cap) != Q35_ATTN_OK)
                    MFAIL(Q35_MODEL_ERR_NOMEM);
                L->u.fa.kv.max_cap = m->kv_cap;
            } else {
                QERR(&qe, Q35_MODEL_ERR_CFG, "layer %d: unknown type %d", i, L->type);
                MFAIL(Q35_MODEL_ERR_CFG);
            }

            /* dense SwiGLU MLP on both layer types */
            snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.mlp.gate_proj.weight", i);
            CHK(bind_fp8(m, nb, II, H, &L->w_gate, &L->sc_gate, &qe));
            snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.mlp.up_proj.weight", i);
            CHK(bind_fp8(m, nb, II, H, &L->w_up, &L->sc_up, &qe));
            snprintf(nb, sizeof(nb), Q35_PREFIX "layers.%d.mlp.down_proj.weight", i);
            CHK(bind_fp8(m, nb, H, II, &L->w_down, &L->sc_down, &qe));
        }

        /* embedding, final norm, lm_head (all BF16, stay in mapping) */
        {
            uint32_t sh2[2] = { (uint32_t)c->vocab_size, H };
            uint32_t shn[1] = { H };
            const Q35Tensor *te = need_tensor(m, Q35_PREFIX "embed_tokens.weight",
                                              Q35_DTYPE_BF16, 2, sh2, &qe);
            if (!te) MFAIL(qe.code);
            m->w_embed = (const uint16_t *)tensor_ptr(m, te);
            const Q35Tensor *th = need_tensor(m, "lm_head.weight",
                                              Q35_DTYPE_BF16, 2, sh2, &qe);
            if (!th) MFAIL(qe.code);
            m->w_lmhead = (const uint16_t *)tensor_ptr(m, th);
            m->w_norm = load_f32(m, Q35_PREFIX "norm.weight", 1, shn, H, &qe);
            if (!m->w_norm) goto fail;
            if (!m->w_embed || !m->w_lmhead) MFAIL(Q35_MODEL_ERR_IO);
        }

        /* scratch */
        m->x  = (float *)ownz(m, sizeof(float) * H);
        m->xn = (float *)ownz(m, sizeof(float) * H);
        m->y  = (float *)ownz(m, sizeof(float) * H);
        m->gate = (float *)ownz(m, sizeof(float) * II);
        m->up   = (float *)ownz(m, sizeof(float) * II);
        m->mid  = (float *)ownz(m, sizeof(float) * II);
        if (!m->x || !m->xn || !m->y || !m->gate || !m->up || !m->mid)
            MFAIL(Q35_MODEL_ERR_NOMEM);

        /* DeltaNet scratch: max dims across all linear layers */
        {
            uint32_t maxQKV = 2 * K + V;
            int nthr = 1;
#ifdef _OPENMP
            nthr = omp_get_max_threads();
#endif
            m->dn_scratch[0] = (float *)ownz(m, sizeof(float) * maxQKV);       /* qkv */
            m->dn_scratch[1] = (float *)ownz(m, sizeof(float) * V);             /* zq */
            m->dn_scratch[2] = (float *)ownz(m, sizeof(float) * Hv);           /* ao */
            m->dn_scratch[3] = (float *)ownz(m, sizeof(float) * Hv);           /* bo */
            m->dn_scratch[4] = (float *)ownz(m, sizeof(float) * V);             /* o */
            m->dn_scratch[5] = (float *)ownz(m, sizeof(float) * (size_t)nthr * c->linear_value_head_dim); /* head_in */
            if (!m->dn_scratch[0] || !m->dn_scratch[5])
                MFAIL(Q35_MODEL_ERR_NOMEM);
        }

        /* attention scratch: sq/sk/sv/sa/sg/sc, sized for full-attn dims */
        {
            uint32_t QH = (uint32_t)c->num_attention_heads;
            uint32_t KVH = (uint32_t)c->num_key_value_heads;
            uint32_t HD = (uint32_t)c->head_dim;
            uint32_t qrows = QH * 2 * HD, krows = KVH * HD;
            int nthr = 1;
#ifdef _OPENMP
            nthr = omp_get_max_threads();
#endif
            uint32_t sc_cap = m->kv_cap > 0 ? m->kv_cap : 4096;
            m->attn_scratch[0] = (float *)ownz(m, sizeof(float) * qrows);
            m->attn_scratch[1] = (float *)ownz(m, sizeof(float) * krows);
            m->attn_scratch[2] = (float *)ownz(m, sizeof(float) * krows);
            m->attn_scratch[3] = (float *)ownz(m, sizeof(float) * QH * HD);
            m->attn_scratch[4] = (float *)ownz(m, sizeof(float) * QH * HD);
            m->attn_scratch[5] = (float *)ownz(m, sizeof(float) * (size_t)sc_cap * nthr);
            if (!m->attn_scratch[0] || !m->attn_scratch[5])
                MFAIL(Q35_MODEL_ERR_NOMEM);
        }

        /* DeltaNet chunked-prefill scratch: 13 buffers × nthr copies, sized
         * to Q35_DN_MAX_CHUNK (64) × max dims, allocated once and reused
         * across all 48 DeltaNet layers per prefill call. Each thread uses
         * its own set for parallel v-head processing. */
        {
            uint32_t C = 64; /* Q35_DN_MAX_CHUNK */
            uint32_t dk = (uint32_t)c->linear_key_head_dim;
            uint32_t dv = (uint32_t)c->linear_value_head_dim;
            int nthr2 = 1;
#ifdef _OPENMP
            nthr2 = omp_get_max_threads();
#endif
            m->dn_ch_nthr = nthr2;
            m->dn_ch[0]  = ownz(m, sizeof(float) * C * dk * nthr2);  /* q_g */
            m->dn_ch[1]  = ownz(m, sizeof(float) * C * dk * nthr2);  /* k_g */
            m->dn_ch[2]  = ownz(m, sizeof(float) * C * dv * nthr2);  /* v_g */
            m->dn_ch[3]  = ownz(m, sizeof(float) * C * dv * nthr2);  /* o_g */
            m->dn_ch[4]  = ownz(m, sizeof(float) * C * nthr2);       /* g_vec */
            m->dn_ch[5]  = ownz(m, sizeof(float) * C * nthr2);       /* beta_vec */
            m->dn_ch[6]  = ownz(m, sizeof(float) * C * C * nthr2);   /* attn */
            m->dn_ch[7]  = ownz(m, sizeof(float) * C * dv * nthr2);  /* v_beta */
            m->dn_ch[8]  = ownz(m, sizeof(float) * C * dk * nthr2);  /* k_beta */
            m->dn_ch[9]  = ownz(m, sizeof(float) * C * dv * nthr2);  /* v_new */
            m->dn_ch[10] = ownz(m, sizeof(float) * C * dk * nthr2);  /* k_cd */
            m->dn_ch[11] = ownz(m, sizeof(float) * C * dv * nthr2);  /* v_adj */
            m->dn_ch[12] = ownz(m, sizeof(float) * C * C * nthr2);   /* qk_attn */
            if (!m->dn_ch[0] || !m->dn_ch[12])
                MFAIL(Q35_MODEL_ERR_NOMEM);
        }
    }

    /* Prefetch all weight shards into RAM: MapViewOfFile is lazy, so without
     * this the first forward pass demand-faults 26 GB interleaved with
     * compute. PrefetchVirtualMemory (or sequential touch fallback) pages
     * everything in with sequential I/O before inference begins. US-10:
     * per-shard progress so the ~26 GB page-in is not a blank wait. */
    {
        double pf0 = q35_plat_now();
        for (int i = 0; i < m->nmaps; i++) {
            fprintf(stderr, "model: prefetching shard %d/%d\r", i + 1, m->nmaps);
            q35_plat_mmap_prefetch(m->maps[i].mm);
        }
        double pf1 = q35_plat_now();
        fprintf(stderr, "model: prefetched %d shards in %.1f s\n", m->nmaps, pf1 - pf0);
    }

    m->pos = 0;
    goto ret;

fail:
    if (err) *err = qe.code;
    q35_model_free(m);
    return NULL;

ret:
    if (err) *err = Q35_MODEL_OK;
    return m;

#undef MFAIL
#undef CHK
}

/* ---------- lifecycle ---------- */

void q35_model_free(Q35Model *m)
{
    int i;
    if (!m) return;
    if (m->layers) {
        for (i = 0; i < m->nlayers; i++)
            if (m->layers[i].type == Q35_LAYER_FULL &&
                (m->layers[i].u.fa.kv.k || m->layers[i].u.fa.kv.v))
                q35_kvcache_free(&m->layers[i].u.fa.kv);
    }
    for (i = 0; (size_t)i < m->nowned; i++) free(m->owned[i]);
    free(m->owned);
    unmap_files(m->maps, m->nmaps);
    if (m->st) q35_st_close(m->st);
    free(m);
}

void q35_model_reset(Q35Model *m)
{
    int i;
    if (!m) return;
    for (i = 0; i < m->nlayers; i++) {
        Q35ModelLayer *L = &m->layers[i];
        if (L->type == Q35_LAYER_LINEAR) {
            const Q35Cfg *c = &m->cfg;
            size_t qkv = (size_t)2 * c->linear_num_key_heads *
                         c->linear_key_head_dim +
                         (size_t)c->linear_num_value_heads *
                         c->linear_value_head_dim;
            memset(L->u.dn.S, 0, (size_t)c->linear_num_value_heads *
                                 c->linear_key_head_dim *
                                 c->linear_value_head_dim * sizeof(float));
            memset(L->u.dn.conv, 0, qkv * 4 * sizeof(float));
        } else {
            L->u.fa.kv.len = 0;
        }
    }
    m->pos = 0;
}

/* ---------- save / load state ---------- */

#define Q35_STATE_MAGIC 0x53533551u  /* 'Q35S' little-endian */
#define Q35_STATE_VERSION 1

int q35_model_save_state(const Q35Model *m, const char *path)
{
    FILE *f;
    int i;
    uint32_t magic = Q35_STATE_MAGIC, ver = Q35_STATE_VERSION;
    const Q35Cfg *c;

    if (!m || !path) return Q35_MODEL_ERR_ARG;
    c = &m->cfg;
    f = fopen(path, "wb");
    if (!f) return Q35_MODEL_ERR_IO;
    fwrite(&magic, 4, 1, f);
    fwrite(&ver, 4, 1, f);
    fwrite(&m->pos, 4, 1, f);
    fwrite(&m->nlayers, 4, 1, f);

    for (i = 0; i < m->nlayers; i++) {
        Q35ModelLayer *L = &m->layers[i];
        uint32_t ty = (uint32_t)L->type;
        fwrite(&ty, 4, 1, f);
        if (L->type == Q35_LAYER_LINEAR) {
            size_t sn = (size_t)c->linear_num_value_heads *
                        c->linear_key_head_dim *
                        c->linear_value_head_dim;
            size_t qkv = (size_t)2 * c->linear_num_key_heads *
                         c->linear_key_head_dim +
                         (size_t)c->linear_num_value_heads *
                         c->linear_value_head_dim;
            fwrite(L->u.dn.S, sizeof(float), sn, f);
            fwrite(L->u.dn.conv, sizeof(float), qkv * 4, f);
        } else {
            q35_kvcache *kv = &L->u.fa.kv;
            fwrite(&kv->len, 4, 1, f);
            /* save only live portion: [kv_heads][len][head_dim] */
            for (uint32_t h = 0; h < kv->kv_heads; h++) {
                fwrite(kv->k + (size_t)h * kv->cap * kv->head_dim,
                       sizeof(float), (size_t)kv->len * kv->head_dim, f);
                fwrite(kv->v + (size_t)h * kv->cap * kv->head_dim,
                       sizeof(float), (size_t)kv->len * kv->head_dim, f);
            }
        }
    }
    fclose(f);
    return Q35_MODEL_OK;
}

int q35_model_load_state(Q35Model *m, const char *path)
{
    FILE *f;
    uint32_t magic, ver, nlayers;
    int i;
    const Q35Cfg *c;

    if (!m || !path) return Q35_MODEL_ERR_ARG;
    c = &m->cfg;
    f = fopen(path, "rb");
    if (!f) return Q35_MODEL_ERR_IO;
    if (fread(&magic, 4, 1, f) != 1 || magic != Q35_STATE_MAGIC) { fclose(f); return Q35_MODEL_ERR_IO; }
    if (fread(&ver, 4, 1, f) != 1 || ver != Q35_STATE_VERSION) { fclose(f); return Q35_MODEL_ERR_IO; }
    if (fread(&m->pos, 4, 1, f) != 1) { fclose(f); return Q35_MODEL_ERR_IO; }
    if (fread(&nlayers, 4, 1, f) != 1 || (int)nlayers != m->nlayers) { fclose(f); return Q35_MODEL_ERR_SHAPE; }

    for (i = 0; i < m->nlayers; i++) {
        Q35ModelLayer *L = &m->layers[i];
        uint32_t ty;
        if (fread(&ty, 4, 1, f) != 1 || (int)ty != L->type) { fclose(f); return Q35_MODEL_ERR_SHAPE; }
        if (L->type == Q35_LAYER_LINEAR) {
            size_t sn = (size_t)c->linear_num_value_heads *
                        c->linear_key_head_dim *
                        c->linear_value_head_dim;
            size_t qkv = (size_t)2 * c->linear_num_key_heads *
                         c->linear_key_head_dim +
                         (size_t)c->linear_num_value_heads *
                         c->linear_value_head_dim;
            if (fread(L->u.dn.S, sizeof(float), sn, f) != sn) { fclose(f); return Q35_MODEL_ERR_IO; }
            if (fread(L->u.dn.conv, sizeof(float), qkv * 4, f) != qkv * 4) { fclose(f); return Q35_MODEL_ERR_IO; }
        } else {
            q35_kvcache *kv = &L->u.fa.kv;
            uint32_t len;
            if (fread(&len, 4, 1, f) != 1 || len > kv->max_cap) { fclose(f); return Q35_MODEL_ERR_SHAPE; }
            if (len > kv->cap) {
                uint32_t new_cap = kv->cap;
                while (new_cap < len) new_cap *= 2;
                if (new_cap > kv->max_cap) new_cap = kv->max_cap;
                if (q35_kvcache_grow(kv, new_cap) != Q35_ATTN_OK) { fclose(f); return Q35_MODEL_ERR_NOMEM; }
            }
            kv->len = len;
            for (uint32_t h = 0; h < kv->kv_heads; h++) {
                if (fread(kv->k + (size_t)h * kv->cap * kv->head_dim,
                          sizeof(float), (size_t)len * kv->head_dim, f) != (size_t)len * kv->head_dim) { fclose(f); return Q35_MODEL_ERR_IO; }
                if (fread(kv->v + (size_t)h * kv->cap * kv->head_dim,
                          sizeof(float), (size_t)len * kv->head_dim, f) != (size_t)len * kv->head_dim) { fclose(f); return Q35_MODEL_ERR_IO; }
            }
        }
    }
    fclose(f);
    return Q35_MODEL_OK;
}

/* ---------- forward ---------- */

/* debug dump hook: Q35_DUMP=<file> -> records (pos,layer,kind,n)+f32 vec */
static FILE *df = NULL;
static int df_init = 0;
static void q35_dump(int pos, int layer, int kind, const float *v, uint32_t n)
{
    if (!df_init) { df_init = 1; if (getenv("Q35_DUMP")) df = fopen(getenv("Q35_DUMP"), "wb"); }
    if (!df) return;
    { int hdr[4] = { pos, layer, kind, (int)n }; fwrite(hdr, 4, 4, df); fwrite(v, 4, n, df); fflush(df); }
}

/* shared handle so q35_attn can append sub-layer records to the same dump */
FILE *q35_dbg_file(void)
{
    if (!df_init) { df_init = 1; if (getenv("Q35_DUMP")) df = fopen(getenv("Q35_DUMP"), "wb"); }
    return df;
}

static int forward_one(Q35Model *m, int32_t tok, float *logits /* nullable */)
{
    const Q35Cfg *c = &m->cfg;
    const uint32_t H = (uint32_t)c->hidden_size;
    const uint32_t II = (uint32_t)c->intermediate_size;
    const float eps = (float)c->rms_norm_eps;
    uint32_t j;
    int i, rc;

    /* per-phase profiler (Q35_PROF=1) — zero cost when off */
    static int prof_on = -1;
    static double p_attn, p_dn, p_mlp, p_head;
    if (prof_on < 0) prof_on = getenv("Q35_PROF") ? 1 : 0;
    double _ps;
    #define PT0() do { if (prof_on) _ps = q35_plat_now(); } while (0)
    #define PT1(var) do { if (prof_on) var += q35_plat_now() - _ps; } while (0)

    if (tok < 0 || (uint32_t)tok >= (uint32_t)c->vocab_size)
        return Q35_MODEL_ERR_ARG;
    if (m->pos >= m->kv_cap)
        return Q35_MODEL_ERR_STATE;

    {
        const uint16_t *row = m->w_embed + (size_t)tok * H;
        for (j = 0; j < H; j++) m->x[j] = q35_mm_bf16_to_f32(row[j]);
    }
    q35_dump((int)m->pos, -1, 0, m->x, H);

    for (i = 0; i < m->nlayers; i++) {
        Q35ModelLayer *L = &m->layers[i];

        q35_kern_rmsnorm(m->x, L->w_in_ln, m->xn, H, eps);
        if (L->type == Q35_LAYER_LINEAR) {
            PT0();
            rc = q35_dn_forward_s(&L->u.dn.layer, &L->u.dn.state, m->xn, m->y, 1, m->dn_scratch);
            PT1(p_dn);
            if (rc != Q35_DN_OK) { fprintf(stderr, "dn layer %d rc=%d pos=%u\n", i, rc, m->pos); return Q35_MODEL_ERR_ARG; }
        } else {
            PT0();
            rc = q35_attn_forward_s(&L->u.fa.cfg, &L->u.fa.w, &L->u.fa.kv,
                                    m->xn, 1, m->y, m->attn_scratch);
            PT1(p_attn);
            if (rc == Q35_ATTN_ERR_FULL) return Q35_MODEL_ERR_STATE;
            if (rc != Q35_ATTN_OK) { fprintf(stderr, "attn layer %d rc=%d pos=%u\n", i, rc, m->pos); return Q35_MODEL_ERR_ARG; }

        }
        q35_dump((int)m->pos, i, 1, m->y, H);
        for (j = 0; j < H; j++) m->x[j] += m->y[j];
        q35_dump((int)m->pos, i, 2, m->x, H);

        q35_kern_rmsnorm(m->x, L->w_post_ln, m->xn, H, eps);
        PT0();
        {
            const uint8_t *Ws[4] = { L->w_gate, L->w_up, NULL, NULL };
            const uint16_t *Ss[4] = { L->sc_gate, L->sc_up, NULL, NULL };
            uint32_t Rs[4] = { II, II, 0, 0 };
            float *Ys[4] = { m->gate, m->up, NULL, NULL };
            q35_mm_fp8_multi4(Ws, Ss, Rs, H, m->xn, Ys);
        }
        q35_kern_swiglu(m->gate, m->up, m->mid, II);
        q35_mm_fp8(L->w_down, L->sc_down, H, II, m->mid, m->y);
        PT1(p_mlp);
        for (j = 0; j < H; j++) m->x[j] += m->y[j];
        q35_dump((int)m->pos, i, 3, m->x, H);
    }

    if (logits) {
        q35_kern_rmsnorm(m->x, m->w_norm, m->xn, H, eps);
        q35_dump((int)m->pos, -2, 8, m->xn, H);
        PT0();
        q35_mm_bf16(m->w_lmhead, (uint32_t)c->vocab_size, H, m->xn, logits);
        PT1(p_head);
        q35_dump((int)m->pos, -3, 9, logits, (uint32_t)c->vocab_size);
    }
    m->pos++;

    if (prof_on && m->pos % 5 == 0) {
        fprintf(stderr, "[prof] pos=%u  dn=%.3f attn=%.3f mlp=%.3f head=%.3f  tot=%.3f\n",
                m->pos, p_dn, p_attn, p_mlp, p_head,
                p_dn + p_attn + p_mlp + p_head);
    }
    return Q35_MODEL_OK;
}

/* batch MLP for all n tokens, shared by both prefill branches:
 * post-attention norm -> gate/up batch matmul -> swiglu -> down batch
 * matmul -> residual add. */
static void mlp_batch(const Q35ModelLayer *L, size_t n, uint32_t H, uint32_t II,
                      float eps, float *x_batch, float *xn_batch,
                      float *gate_batch, float *up_batch, float *mid_batch,
                      float *y_batch)
{
    for (size_t t = 0; t < n; t++)
        q35_kern_rmsnorm(x_batch + t * H, L->w_post_ln,
                         xn_batch + t * H, H, eps);
    q35_mm_fp8_batch(L->w_gate, L->sc_gate, II, H, (uint32_t)n, xn_batch, gate_batch);
    q35_mm_fp8_batch(L->w_up, L->sc_up, II, H, (uint32_t)n, xn_batch, up_batch);
    q35_kern_swiglu(gate_batch, up_batch, mid_batch, n * II);
    q35_mm_fp8_batch(L->w_down, L->sc_down, H, II, (uint32_t)n, mid_batch, y_batch);
    size_t total = n * H;
    for (size_t k = 0; k < total; k++) x_batch[k] += y_batch[k];
}

int q35_forward_prefill(Q35Model *m, const int32_t *tokens, size_t n,
                        float *logits)
{
    size_t t;
    int rc = Q35_MODEL_OK;
    if (!m || (!tokens && n > 0)) return Q35_MODEL_ERR_ARG;

    /* For multi-token prefill, batch DeltaNet layer projections.
     * Strategy: run layer-by-layer; for each layer, batch-embed +
     * batch-rmsnorm all n tokens, then:
     *   - DeltaNet: q35_dn_forward_s(seq_len=n) — batch projection
     *   - Attention/MLP: still per-token (sequential KV cache / matmul)
     * This cuts 4×n fork/join calls on the 48 DeltaNet layers. */
    if (n > 1) {
        const Q35Cfg *c = &m->cfg;
        const uint32_t H = (uint32_t)c->hidden_size;
        const uint32_t II = (uint32_t)c->intermediate_size;
        const float eps = (float)c->rms_norm_eps;
        uint32_t j;
        /* batch prefill profiler */
        static int pf_prof = -1;
        if (pf_prof < 0) pf_prof = getenv("Q35_PROF") ? 1 : 0;
        double pf_attn = 0, pf_mlp = 0, pf_dn = 0, pf_s = 0;

        /* batch input: x_batch[n][H] = residual chain, xn_batch[n][H] = normed */
        float *x_batch = (float *)malloc(n * H * sizeof(float));
        float *xn_batch = (float *)malloc(n * H * sizeof(float));
        float *y_batch = (float *)malloc(n * H * sizeof(float));
        /* MLP batch buffers: gate/up/down for n tokens */
        float *gate_batch = (float *)malloc(n * II * sizeof(float));
        float *up_batch = (float *)malloc(n * II * sizeof(float));
        float *mid_batch = (float *)malloc(n * II * sizeof(float));
        if (!x_batch || !xn_batch || !y_batch || !gate_batch || !up_batch || !mid_batch) {
            free(x_batch); free(xn_batch); free(y_batch);
            free(gate_batch); free(up_batch); free(mid_batch);
            return Q35_MODEL_ERR_NOMEM;
        }
        int pf_rc = Q35_MODEL_OK;

        /* embed all tokens into x_batch (residual chain) */
        for (t = 0; t < n; t++) {
            const uint16_t *row = m->w_embed + (size_t)tokens[t] * H;
            for (j = 0; j < H; j++) x_batch[t * H + j] = q35_mm_bf16_to_f32(row[j]);
        }

        for (int i = 0; i < m->nlayers; i++) {
            Q35ModelLayer *L = &m->layers[i];

            /* batch input_layernorm: xn_batch = norm(x_batch) */
            for (t = 0; t < n; t++)
                q35_kern_rmsnorm(x_batch + t * H, L->w_in_ln,
                                 xn_batch + t * H, H, eps);

            if (L->type == Q35_LAYER_LINEAR) {
                if (pf_prof) pf_s = q35_plat_now();
                rc = q35_dn_forward_s2(&L->u.dn.layer, &L->u.dn.state,
                                      xn_batch, y_batch, n, m->dn_scratch,
                                      m->dn_ch);
                if (pf_prof) pf_dn += q35_plat_now() - pf_s;
                if (rc != Q35_DN_OK) {
                    fprintf(stderr, "dn layer %d rc=%d (batch)\n", i, rc);
                    pf_rc = Q35_MODEL_ERR_ARG;
                    goto pfdone;
                }
                /* residual: x_batch += y_batch */
                for (t = 0; t < n; t++)
                    for (j = 0; j < H; j++)
                        x_batch[t * H + j] += y_batch[t * H + j];

                /* post-attention norm + MLP — batch all n tokens */
                if (pf_prof) pf_s = q35_plat_now();
                mlp_batch(L, n, H, II, eps, x_batch, xn_batch,
                          gate_batch, up_batch, mid_batch, y_batch);
                if (pf_prof) pf_mlp += q35_plat_now() - pf_s;
            } else {
                /* attention: batch q/k/v projection + per-token attend
                 * (KV cache append/attend is still sequential) */
                if (pf_prof) pf_s = q35_plat_now();
                rc = q35_attn_forward_s(&L->u.fa.cfg, &L->u.fa.w, &L->u.fa.kv,
                                        xn_batch, (uint32_t)n, y_batch,
                                        m->attn_scratch);
                if (rc != Q35_ATTN_OK) {
                    fprintf(stderr, "attn layer %d rc=%d (batch)\n", i, rc);
                    pf_rc = Q35_MODEL_ERR_ARG;
                    goto pfdone;
                }
                { size_t total = (size_t)n * H;
                  for (size_t k = 0; k < total; k++) x_batch[k] += y_batch[k];
                }
                if (pf_prof) pf_attn += q35_plat_now() - pf_s;

                /* batch MLP for all n tokens (same as DeltaNet branch) */
                if (pf_prof) pf_s = q35_plat_now();
                mlp_batch(L, n, H, II, eps, x_batch, xn_batch,
                          gate_batch, up_batch, mid_batch, y_batch);
                if (pf_prof) pf_mlp += q35_plat_now() - pf_s;
            }
        }
        m->pos += (uint32_t)n;
        if (pf_prof)
            fprintf(stderr, "[prof-pf] n=%zu dn=%.2f attn=%.2f mlp=%.2f tot=%.2f\n",
                    n, pf_dn, pf_attn, pf_mlp, pf_dn + pf_attn + pf_mlp);

        /* copy last token's residual to m->x for final norm + lm_head */
        memcpy(m->x, x_batch + (n - 1) * H, H * sizeof(float));

        if (logits) {
            q35_kern_rmsnorm(m->x, m->w_norm, m->xn, H, eps);
            q35_mm_bf16(m->w_lmhead, (uint32_t)c->vocab_size, H, m->xn, logits);
        }

pfdone:
        free(x_batch); free(xn_batch); free(y_batch);
        free(gate_batch); free(up_batch); free(mid_batch);
        return pf_rc;
    }

    /* single-token path (original) */
    for (t = 0; t < n; t++) {
        float *lo = (t == n - 1) ? logits : NULL;
        rc = forward_one(m, tokens[t], lo);
        if (rc != Q35_MODEL_OK) return rc;
    }
    return rc;
}

int q35_forward_decode(Q35Model *m, int32_t token, float *logits)
{
    if (!m) return Q35_MODEL_ERR_ARG;
    return forward_one(m, token, logits);
}

uint32_t q35_model_vocab(const Q35Model *m)  { return m ? (uint32_t)m->cfg.vocab_size : 0; }
uint32_t q35_model_hidden(const Q35Model *m) { return m ? (uint32_t)m->cfg.hidden_size : 0; }
uint32_t q35_model_pos(const Q35Model *m)    { return m ? m->pos : 0; }
uint32_t q35_model_layers(const Q35Model *m) { return m ? (uint32_t)m->nlayers : 0; }
