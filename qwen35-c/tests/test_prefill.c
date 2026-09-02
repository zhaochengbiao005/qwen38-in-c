#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "q35/q35_model.h"
#include "q35/q35_kern.h"

#define fnv1a q35_kern_fnv1a

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/fixtures/model";
    char errbuf[512]; int err;
    int tokens[4] = {5, 42, 7, 99};
    int n = 4;

    Q35Model *m = q35_model_load(dir, 0, errbuf, sizeof errbuf, &err);
    if (!m) { printf("load fail\n"); return 1; }
    uint32_t vocab = q35_model_vocab(m);
    float *lg_a = malloc(vocab * 4);
    q35_forward_prefill(m, tokens, n, lg_a);
    unsigned ha = fnv1a(lg_a, vocab * 4);
    q35_model_free(m);

    m = q35_model_load(dir, 0, errbuf, sizeof errbuf, &err);
    float *lg_b = malloc(vocab * 4);
    for (int t = 0; t < n; t++)
        q35_forward_decode(m, tokens[t], (t == n-1) ? lg_b : NULL);
    unsigned hb = fnv1a(lg_b, vocab * 4);
    q35_model_free(m);

    /* batch path (chunked delta rule) vs per-token path use different fp32
     * reduction orders, so check maxabs tolerance, not FNV bitwise. */
    double maxabs = 0;
    for (uint32_t i = 0; i < vocab; i++) {
        double d = fabs(lg_a[i] - lg_b[i]);
        if (d > maxabs) maxabs = d;
    }
    printf("batch fnv=%08x  single fnv=%08x  maxabs=%.6g  %s\n",
           ha, hb, maxabs, maxabs < 1.0 ? "OK" : "MISMATCH");
    free(lg_a); free(lg_b);
    return maxabs < 1.0 ? 0 : 1;
}
