/* Ticket-18: GQA full-attention fixture harness.
 *
 * Loads tests/fixtures/attn/manifest.json (+ binaries), then:
 *   1. prefill: q35_attn_forward with all nt tokens at once
 *   2. decode:  q35_attn_forward one token at a time (fresh cache)
 *   3. both outputs within manifest atol+rtol of the f64 reference
 *   4. prefill and decode outputs bit-identical (FNV-1a over raw bytes)
 * Missing manifest/fixtures: prints NOT RUN and exits 77.
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "q35/q35_attn.h"
#include "q35/q35_mm.h"

#define SKIP_RETURN_CODE 77

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); failures++; } \
} while (0)

static unsigned char *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *b = (unsigned char *)malloc(*n ? *n : 1);
    if (*n && fread(b, 1, *n, f) != *n) { fclose(f); free(b); return NULL; }
    fclose(f);
    return b;
}

static uint64_t fnv1a64(const void *p, size_t n)
{
    const unsigned char *c = (const unsigned char *)p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ULL; }
    return h;
}

static unsigned get_u(const char *json, const char *key, unsigned def)
{
    char pat[64];
    sprintf(pat, "\"%s\": ", key);
    const char *p = strstr(json, pat);
    unsigned v = def;
    if (p) sscanf(p + strlen(pat), "%u", &v);
    return v;
}

static float get_f(const char *json, const char *key, float def)
{
    char pat[64];
    sprintf(pat, "\"%s\": ", key);
    const char *p = strstr(json, pat);
    double v = def;
    if (p) sscanf(p + strlen(pat), "%lf", &v);
    return (float)v;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "tests/fixtures/attn";
    char path[512];
    size_t n;

    sprintf(path, "%s/manifest.json", dir);
    char *mj = (char *)slurp(path, &n);
    if (!mj) { printf("NOT RUN: %s missing\n", path); return SKIP_RETURN_CODE; }

    q35_attn_cfg cfg;
    cfg.hidden = get_u(mj, "hidden", 0);
    cfg.q_heads = get_u(mj, "q_heads", 0);
    cfg.kv_heads = get_u(mj, "kv_heads", 0);
    cfg.head_dim = get_u(mj, "head_dim", 0);
    cfg.rotary_dim = get_u(mj, "rotary_dim", 0);
    cfg.theta = (float)get_f(mj, "theta", 1e7f);
    cfg.eps = get_f(mj, "eps", 1e-6f);
    unsigned nt = get_u(mj, "nt", 0);
    double atol = get_f(mj, "tol_atol", 1e-4f);
    double rtol = get_f(mj, "tol_rtol", 1e-3f);
    free(mj);
    if (!cfg.hidden || !cfg.q_heads || !cfg.kv_heads || !cfg.head_dim ||
        !cfg.rotary_dim || !nt) {
        printf("NOT RUN: bad manifest\n");
        return SKIP_RETURN_CODE;
    }
    printf("cfg hidden=%u q=%u kv=%u hd=%u rot=%u nt=%u theta=%g eps=%g\n",
           cfg.hidden, cfg.q_heads, cfg.kv_heads, cfg.head_dim,
           cfg.rotary_dim, nt, cfg.theta, cfg.eps);

    struct { const char *fn; unsigned char *buf; size_t sz; } blobs[10];
    static const char *names[] = {
        "wq.w8", "wq.sc", "wk.w8", "wk.sc", "wv.w8", "wv.sc",
        "wo.w8", "wo.sc", "x.bin", "yref.bin"
    };
    for (int i = 0; i < 10; i++) {
        sprintf(path, "%s/%s", dir, names[i]);
        blobs[i].fn = names[i];
        blobs[i].buf = slurp(path, &blobs[i].sz);
        if (!blobs[i].buf) {
            printf("NOT RUN: %s missing\n", path);
            return SKIP_RETURN_CODE;
        }
    }
    sprintf(path, "%s/q_nw.bin", dir);
    unsigned char *qnw = slurp(path, &n);
    sprintf(path, "%s/k_nw.bin", dir);
    unsigned char *knw = slurp(path, &n);
    if (!qnw || !knw) { printf("NOT RUN: norm weights missing\n"); return SKIP_RETURN_CODE; }

    q35_attn_weights w;
    w.wq = blobs[0].buf; w.wq_sc = (const uint16_t *)blobs[1].buf;
    w.wk = blobs[2].buf; w.wk_sc = (const uint16_t *)blobs[3].buf;
    w.wv = blobs[4].buf; w.wv_sc = (const uint16_t *)blobs[5].buf;
    w.wo = blobs[6].buf; w.wo_sc = (const uint16_t *)blobs[7].buf;
    w.q_nw = (const float *)qnw;
    w.k_nw = (const float *)knw;
    const float *x = (const float *)blobs[8].buf;
    const double *yref = (const double *)blobs[9].buf;

    size_t outn = (size_t)nt * cfg.hidden;
    float *y_pre = (float *)malloc(outn * sizeof(float));
    float *y_dec = (float *)malloc(outn * sizeof(float));

    q35_kvcache c_pre, c_dec;
    CHECK(q35_kvcache_init(&c_pre, cfg.kv_heads, cfg.head_dim, nt) == 0,
          "kv cache prefill init");
    CHECK(q35_kvcache_init(&c_dec, cfg.kv_heads, cfg.head_dim, nt) == 0,
          "kv cache decode init");

    q35_mm_init();
    int rc = q35_attn_forward(&cfg, &w, &c_pre, x, nt, y_pre);
    CHECK(rc == 0, "prefill forward rc=%d", rc);
    CHECK(c_pre.len == nt, "prefill cache len %u != %u", c_pre.len, nt);
    for (unsigned t = 0; t < nt; t++) {
        rc = q35_attn_forward(&cfg, &w, &c_dec,
                              x + (size_t)t * cfg.hidden, 1,
                              y_dec + (size_t)t * cfg.hidden);
        CHECK(rc == 0, "decode token %u rc=%d", t, rc);
    }
    CHECK(c_dec.len == nt, "decode cache len %u != %u", c_dec.len, nt);

    double maxabs = 0, maxrel = 0;
    for (size_t i = 0; i < outn; i++) {
        double d = fabs((double)y_pre[i] - yref[i]);
        double r = d / (fabs(yref[i]) > 1e-9 ? fabs(yref[i]) : 1e-9);
        if (d > maxabs) maxabs = d;
        if (r > maxrel) maxrel = r;
        CHECK(d <= atol + rtol * fabs(yref[i]),
              "prefill[%zu] y=%.9g ref=%.9g abs=%.3g", i, y_pre[i], yref[i], d);
    }
    printf("prefill vs ref: maxabs=%.3g maxrel=%.3g impl=%s\n",
           maxabs, maxrel, q35_mm_impl_fp8());

    double maxabs_d = 0;
    for (size_t i = 0; i < outn; i++) {
        double d = fabs((double)y_dec[i] - yref[i]);
        if (d > maxabs_d) maxabs_d = d;
    }
    for (size_t i = 0; i < outn; i++)
        CHECK(fabs((double)y_dec[i] - yref[i]) <= atol + rtol * fabs(yref[i]),
              "decode[%zu] y=%.9g ref=%.9g", i, y_dec[i], yref[i]);
    printf("decode  vs ref: maxabs=%.3g\n", maxabs_d);

    /* prefill vs decode must be bit-identical (same op sequence) */
    uint64_t h_pre = fnv1a64(y_pre, outn * sizeof(float));
    uint64_t h_dec = fnv1a64(y_dec, outn * sizeof(float));
    CHECK(h_pre == h_dec, "prefill fnv=%016llx decode fnv=%016llx",
          (unsigned long long)h_pre, (unsigned long long)h_dec);
    printf("prefill/decode fnv=%016llx %s\n", (unsigned long long)h_pre,
           h_pre == h_dec ? "bit-identical" : "MISMATCH");

    /* cache exhaustion must be reported */
    CHECK(q35_attn_forward(&cfg, &w, &c_pre, x, 1, y_pre) == Q35_ATTN_ERR_FULL,
          "full cache not signalled");

    printf("%s: %d failures\n", failures ? "FAIL" : "ok", failures);
    return failures ? 1 : 0;
}
