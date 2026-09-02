/* Ticket-19: whole-model fixture test. */
#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "q35/q35_model.h"

static unsigned char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n ? *n : 1);
    if (*n && fread(b, 1, *n, f) != *n) { fclose(f); free(b); return NULL; }
    fclose(f); return b;
}

static unsigned fnv1a(const void *p, size_t n) {
    const unsigned char *c = p; unsigned h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 16777619u; }
    return h;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/fixtures/model";
    const char *realdir = argc > 2 ? argv[2] : NULL;
    char path[512], errbuf[512];
    int err, fails = 0;

    /* tokens.json: crude scan */
    int tokens[64]; size_t ntok = 0; double atol = 1e-2, rtol = 1e-2;
    {
        snprintf(path, sizeof path, "%s/tokens.json", dir);
        size_t n; char *j = (char *)slurp(path, &n);
        if (!j) { printf("SKIP: no tokens.json (exit 77)\n"); return 77; }
        char *t = strstr(j, "\"tokens\"");
        if (t) { char *p = strchr(t, '[');
            char *pend = p ? strchr(p, ']') : NULL;
            if (pend) {
                p++;
                while (p < pend && ntok < 64) {
                    if (*p == ',' || *p == ' ' || *p == '\n') { p++; continue; }
                    tokens[ntok++] = (int)strtol(p, &p, 10);
                }
            } }
        char *a = strstr(j, "\"atol\""); if (a) sscanf(a, "\"atol\": %lf", &atol);
        char *r = strstr(j, "\"rtol\""); if (r) sscanf(r, "\"rtol\": %lf", &rtol);
        free(j);
    }

    snprintf(path, sizeof path, "%s/logits_ref.bin", dir);
    size_t lrn; double *lref = (double *)slurp(path, &lrn);
    if (!lref) { printf("SKIP: no logits_ref.bin (exit 77)\n"); return 77; }

    Q35Model *m = q35_model_load(dir, 0, errbuf, sizeof errbuf, &err);
    if (!m) { printf("FAIL load: %s (err=%d)\n", errbuf, err); return 1; }
    uint32_t vocab = q35_model_vocab(m);
    if ((size_t)vocab * 8 != lrn) { printf("FAIL: ref size %zu vs vocab*8 %zu\n", lrn, (size_t)vocab*8); return 1; }
    printf("loaded %s: layers=%u vocab=%u hidden=%u\n", dir,
           q35_model_layers(m), vocab, q35_model_hidden(m));

    float *lg_a = malloc(vocab * 4), *lg_b = malloc(vocab * 4);
    /* path A: one prefill over all tokens */
    if (q35_forward_prefill(m, tokens, ntok, lg_a) != Q35_MODEL_OK) { printf("FAIL prefill\n"); return 1; }
    double maxabs = 0, maxrel = 0;
    for (uint32_t i = 0; i < vocab; i++) {
        double d = fabs((double)lg_a[i] - lref[i]);
        if (d > maxabs) maxabs = d;
        double r = d / (fabs(lref[i]) > 1e-12 ? fabs(lref[i]) : 1e-12);
        if (r > maxrel) maxrel = r;
        if (d > atol + rtol * fabs(lref[i])) {
            printf("FAIL logits[%u]: %g vs ref %g\n", i, lg_a[i], lref[i]);
            fails++; break;
        }
    }
    printf("prefill-all: maxabs=%.3g maxrel=%.3g (atol=%g rtol=%g)\n", maxabs, maxrel, atol, rtol);
    for (int pp = 0; pp < 4; pp++) printf("lg[%d]=%.8g ref=%.8g\n", pp, lg_a[pp], lref[pp]);

    /* path B: prefill n-1 without logits, then decode last */
    q35_model_reset(m);
    if (q35_forward_prefill(m, tokens, ntok - 1, NULL) != Q35_MODEL_OK) { printf("FAIL prefill(n-1)\n"); return 1; }
    if (q35_forward_decode(m, tokens[ntok - 1], lg_b) != Q35_MODEL_OK) { printf("FAIL decode\n"); return 1; }
    /* batch prefill (chunked delta rule) vs per-token decode use different fp32
     * reduction orders, so check tolerance, not bitwise FNV. */
    unsigned ha = fnv1a(lg_a, vocab * 4), hb = fnv1a(lg_b, vocab * 4);
    double pd_maxabs = 0;
    for (uint32_t i = 0; i < vocab; i++) {
        double d = fabs((double)lg_a[i] - lg_b[i]);
        if (d > pd_maxabs) pd_maxabs = d;
    }
    if (pd_maxabs > 0.05) { printf("FAIL: prefill-vs-decode maxabs=%.3g (fnv %08x != %08x)\n", pd_maxabs, ha, hb); fails++; }
    else printf("prefill-vs-decode: maxabs=%.3g (fnv %08x vs %08x) OK\n", pd_maxabs, ha, hb);

    /* argmax sanity */
    uint32_t best = 0; float bv = lg_a[0];
    for (uint32_t i = 1; i < vocab; i++) if (lg_a[i] > bv) { bv = lg_a[i]; best = i; }
    printf("argmax=%u val=%g\n", best, bv);

    q35_model_free(m);

    if (realdir) {
        printf("--- real model smoke: %s ---\n", realdir);
        clock_t t0 = clock();
        Q35Model *rm = q35_model_load(realdir, 64, errbuf, sizeof errbuf, &err);
        if (!rm) { printf("FAIL real load: %s (err=%d)\n", errbuf, err); return 1; }
        printf("real load ok: layers=%u vocab=%u hidden=%u (%.0f ms)\n",
               q35_model_layers(rm), q35_model_vocab(rm), q35_model_hidden(rm),
               1000.0 * (clock() - t0) / CLOCKS_PER_SEC);
        float *lg = malloc((size_t)q35_model_vocab(rm) * 4);
        int32_t toks[2] = { 9707, 11 };
        t0 = clock();
        int rc = q35_forward_prefill(rm, toks, 2, lg);
        printf("real prefill(2 tok): rc=%d (%.0f ms)\n", rc,
               1000.0 * (clock() - t0) / CLOCKS_PER_SEC);
        if (rc == Q35_MODEL_OK) {
            uint32_t vb = q35_model_vocab(rm), bi = 0; float bf = lg[0];
            for (uint32_t i = 1; i < vb; i++) if (lg[i] > bf) { bf = lg[i]; bi = i; }
            printf("real argmax=%u val=%g fnv=%08x\n", bi, bf, fnv1a(lg, (size_t)vb * 4));
        } else fails++;
        q35_model_free(rm);
    }

    printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
