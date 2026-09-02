/* Ticket-20 debug: does q35_model_reset fully restore fresh-load state?
 *
 * A: decode prompt[0..5] from fresh load, capture logits each step
 * reset
 * B: decode same tokens again, capture
 * Compare per-step bitwise (FNV) + maxabs. Also re-load the model fresh (C)
 * to separate "reset incomplete" from "load nondeterminism".
 */
#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "q35/q35_model.h"

static unsigned fnv1a(const void *p, size_t n) {
    const unsigned char *c = p; unsigned h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 16777619u; }
    return h;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/fixtures/oracle";
    int prompt[] = {5, 42, 7, 99, 13, 201};
    int np = 6;
    char errbuf[512]; int err;

    Q35Model *m = q35_model_load(dir, 0, errbuf, sizeof errbuf, &err);
    if (!m) { printf("load fail: %s\n", errbuf); return 1; }
    uint32_t vocab = q35_model_vocab(m);
    float *lg = malloc(vocab * 4);
    unsigned ha[16], hb[16];
    double maxd = 0;

    for (int t = 0; t < np; t++) q35_forward_decode(m, prompt[t], lg), ha[t] = fnv1a(lg, vocab * 4);
    /* stash last logits of run A */
    float *la = malloc(vocab * 4); memcpy(la, lg, vocab * 4);

    q35_model_reset(m);
    for (int t = 0; t < np; t++) {
        q35_forward_decode(m, prompt[t], lg);
        hb[t] = fnv1a(lg, vocab * 4);
        if (t == np - 1) for (uint32_t i = 0; i < vocab; i++) {
            double d = fabs((double)lg[i] - la[i]);
            if (d > maxd) maxd = d;
        }
        printf("pos %d: A=%08x B=%08x %s\n", t, ha[t], hb[t], ha[t] == hb[t] ? "" : "  <-- DIFF");
    }
    printf("A-vs-B last-logits maxabs = %.3g\n", maxd);

    /* fresh reload for control */
    q35_model_free(m);
    m = q35_model_load(dir, 0, errbuf, sizeof errbuf, &err);
    for (int t = 0; t < np; t++) q35_forward_decode(m, prompt[t], lg);
    printf("fresh-reload vs A: %s (fnv=%08x vs %08x)\n",
           fnv1a(lg, vocab * 4) == ha[np - 1] ? "same" : "DIFF",
           fnv1a(lg, vocab * 4), ha[np - 1]);
    q35_model_free(m);
    return 0;
}
