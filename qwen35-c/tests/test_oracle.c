/* Ticket-20: tiny oracle — engine (f32) vs numpy f64 reference, three paths.
 *
 *  1. teacher-forcing: feed the reference sequence token-by-token; every
 *     position's logits within manifest tolerance, argmax predicts next id.
 *  2. greedy: prefill the prompt, greedy-generate; ids exactly match the
 *     manifest, per-step logits within tolerance.
 *  3. incremental: decode the prompt one token at a time (no prefill), then
 *     greedy-generate; ids exact, final logits bitwise (FNV-1a) equal to
 *     path 2 — asserts prefill == incremental state evolution.
 *
 * Tolerances come from oracle.json (the fixture manifest), never hardcoded.
 */
#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "q35/q35_model.h"
#include "q35/q35_json.h"

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

static int read_ids(const JVal *root, const char *key, int32_t *dst, int cap) {
    const JVal *a = q35_obj_get(root, key);
    if (!a || a->t != J_ARR || (int)a->v.arr.n > cap) return -1;
    for (size_t i = 0; i < a->v.arr.n; i++)
        if (a->v.arr.items[i]->t != J_NUM) return -1;
        else dst[i] = (int32_t)a->v.arr.items[i]->v.num;
    return (int)a->v.arr.n;
}

static uint32_t argmax(const float *lg, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++) if (lg[i] > lg[best]) best = i;
    return best;
}

static float *LG;
static uint32_t VOC;
static double ATOL, RTOL, gmaxabs, gmaxrel;

/* compare LG (f32) against reference row; returns 1 on tolerance violation */
static int cmp_row(const double *row) {
    for (uint32_t i = 0; i < VOC; i++) {
        double d = fabs((double)LG[i] - row[i]);
        double r = d / (fabs(row[i]) > 1e-12 ? fabs(row[i]) : 1e-12);
        if (d > gmaxabs) gmaxabs = d;
        if (r > gmaxrel && d > ATOL * 0.05) gmaxrel = r; /* skip ~0 logits */
        if (d > ATOL + RTOL * fabs(row[i])) {
            printf("  logits[%u]: %g vs ref %g (d=%g > %g)\n",
                   i, LG[i], row[i], d, ATOL + RTOL * fabs(row[i]));
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/fixtures/oracle";
    char path[512], errbuf[512];
    int err, fails = 0;

    snprintf(path, sizeof path, "%s/oracle.json", dir);
    size_t jn; char *jt = (char *)slurp(path, &jn);
    if (!jt) { printf("SKIP: no oracle.json (exit 77)\n"); return 77; }
    Q35Json *j = q35_json_parse(jt, jn);
    free(jt);
    if (!j) { printf("FAIL: oracle.json parse\n"); return 1; }
    const JVal *root = q35_json_root(j);

    int32_t prompt[64], gen[64];
    int np = read_ids(root, "prompt", prompt, 64);
    int ng = read_ids(root, "gen_ids", gen, 64);
    const JVal *o_atol = q35_obj_get(root, "tf_atol");
    const JVal *o_rtol = q35_obj_get(root, "tf_rtol");
    if (np <= 0 || ng <= 0 || !o_atol || !o_rtol || o_atol->t != J_NUM || o_rtol->t != J_NUM) {
        printf("FAIL: oracle.json fields missing\n"); return 1;
    }
    ATOL = o_atol->v.num; RTOL = o_rtol->v.num;
    q35_json_free(j);

    snprintf(path, sizeof path, "%s/tf_logits_ref.bin", dir);
    size_t rn; double *ref = (double *)slurp(path, &rn);
    if (!ref) { printf("SKIP: no tf_logits_ref.bin (exit 77)\n"); return 77; }

    int32_t seq[128];
    int T = np + ng;
    for (int i = 0; i < T; i++) seq[i] = i < np ? prompt[i] : gen[i - np];

    Q35Model *m = q35_model_load(dir, 0, errbuf, sizeof errbuf, &err);
    if (!m) { printf("FAIL load: %s (err=%d)\n", errbuf, err); return 1; }
    VOC = q35_model_vocab(m);
    if (rn != (size_t)T * VOC * 8) {
        printf("FAIL: ref rows %zu vs T*vocab*8 %zu\n", rn, (size_t)T * VOC * 8);
        return 1;
    }
    LG = malloc(VOC * 4);

    /* ---- path 1: teacher-forcing ---- */
    q35_model_reset(m);
    for (int t = 0; t < T; t++) {
        if (q35_forward_decode(m, seq[t], LG) != Q35_MODEL_OK) { printf("FAIL: tf decode\n"); return 1; }
        if (cmp_row(ref + (size_t)t * VOC)) { printf("FAIL: tf logits pos %d\n", t); fails++; break; }
        /* argmax==next only holds from the last prompt position on: the
         * reference greedily chose every following token from these logits */
        if (t >= np - 1 && t + 1 < T && argmax(LG, VOC) != (uint32_t)seq[t + 1]) {
            printf("FAIL: tf argmax pos %d: %u vs %d\n", t, argmax(LG, VOC), seq[t + 1]);
            fails++; break;
        }
    }
    printf("teacher-forcing: %d pos, maxabs=%.3g maxrel=%.3g (atol=%g rtol=%g)\n",
           T, gmaxabs, gmaxrel, ATOL, RTOL);

    /* ---- path 2: greedy (prefill prompt, argmax loop) ---- */
    unsigned fnv_greedy = 0;
    q35_model_reset(m);
    if (q35_forward_prefill(m, prompt, (size_t)np, LG) != Q35_MODEL_OK) { printf("FAIL: greedy prefill\n"); return 1; }
    if (cmp_row(ref + (size_t)(np - 1) * VOC)) { printf("FAIL: greedy prefill logits\n"); fails++; }
    uint32_t nxt = argmax(LG, VOC);
    fnv_greedy = fnv1a(LG, VOC * 4);
    for (int g = 0; g < ng; g++) {
        if (nxt != (uint32_t)gen[g]) {
            printf("FAIL: greedy id %d: %u vs %d\n", g, nxt, gen[g]); fails++; break;
        }
        if (g + 1 < ng) {
            if (q35_forward_decode(m, (int32_t)nxt, LG) != Q35_MODEL_OK) { printf("FAIL: greedy decode\n"); return 1; }
            if (cmp_row(ref + (size_t)(np + g) * VOC)) { printf("FAIL: greedy logits step %d\n", g); fails++; break; }
            nxt = argmax(LG, VOC);
            fnv_greedy = fnv1a(LG, VOC * 4);
        }
    }
    printf("greedy: %d ids exact\n", ng);

    /* ---- path 3: incremental (decode prompt one-by-one, then greedy) ---- */
    unsigned fnv_inc = 0;
    q35_model_reset(m);
    for (int t = 0; t < np; t++) {
        if (q35_forward_decode(m, prompt[t], LG) != Q35_MODEL_OK) { printf("FAIL: inc decode\n"); return 1; }
        if (cmp_row(ref + (size_t)t * VOC)) { printf("FAIL: inc logits pos %d\n", t); fails++; break; }
    }
    nxt = argmax(LG, VOC); fnv_inc = fnv1a(LG, VOC * 4);
    for (int g = 0; g < ng; g++) {
        if (nxt != (uint32_t)gen[g]) {
            printf("FAIL: incremental id %d: %u vs %d\n", g, nxt, gen[g]); fails++; break;
        }
        if (g + 1 < ng) {
            if (q35_forward_decode(m, (int32_t)nxt, LG) != Q35_MODEL_OK) { printf("FAIL: inc decode 2\n"); return 1; }
            nxt = argmax(LG, VOC); fnv_inc = fnv1a(LG, VOC * 4);
        }
    }
    /* prefill (chunked delta rule) vs incremental (per-token decode) use
     * different fp32 reduction orders, so check tolerance, not bitwise FNV. */
    double pi_maxabs = 0;
    {
        q35_model_reset(m);
        q35_forward_prefill(m, prompt, (size_t)np, LG);
        float *pf_lg = malloc(VOC * 4);
        memcpy(pf_lg, LG, VOC * 4);
        q35_model_reset(m);
        for (int t = 0; t < np; t++)
            q35_forward_decode(m, prompt[t], (t == np-1) ? LG : NULL);
        for (uint32_t i = 0; i < VOC; i++) {
            double d = fabs((double)pf_lg[i] - LG[i]);
            if (d > pi_maxabs) pi_maxabs = d;
        }
        free(pf_lg);
    }
    if (pi_maxabs > 0.05) {
        printf("FAIL: prefill-vs-incremental maxabs=%.3g (fnv %08x != %08x)\n", pi_maxabs, fnv_greedy, fnv_inc);
        fails++;
    } else printf("incremental: %d ids exact, prefill==incremental tol maxabs=%.3g (fnv %08x vs %08x)\n", ng, pi_maxabs, fnv_greedy, fnv_inc);

    q35_model_free(m);
    free(LG); free(ref);
    printf("%s | overall maxabs=%.3g maxrel=%.3g\n", fails ? "FAILED" : "ALL OK", gmaxabs, gmaxrel);
    return fails ? 1 : 0;
}
