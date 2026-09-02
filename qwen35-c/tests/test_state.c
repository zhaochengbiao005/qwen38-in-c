/* save/load state roundtrip: run prefill, save state, reload model, load
 * state, decode one more token — must match decoding without save/load. */
#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "q35/q35_model.h"

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/fixtures/model";
    char errbuf[512]; int err;
    int tokens[4] = {5, 42, 7, 99};
    int n = 4;

    /* run A: prefill all tokens, decode one more */
    Q35Model *m = q35_model_load(dir, 0, errbuf, sizeof errbuf, &err);
    if (!m) { printf("load fail\n"); return 1; }
    uint32_t vocab = q35_model_vocab(m);
    float *lg_a = malloc(vocab * 4);
    q35_forward_prefill(m, tokens, n, NULL);
    q35_forward_decode(m, 3, lg_a);  /* one more token */
    q35_model_free(m);

    /* run B: prefill, save state, reload, load state, decode one more */
    m = q35_model_load(dir, 0, errbuf, sizeof errbuf, &err);
    float *lg_b = malloc(vocab * 4);
    q35_forward_prefill(m, tokens, n, NULL);
    int rc = q35_model_save_state(m, "test_state.bin");
    if (rc != Q35_MODEL_OK) { printf("save fail rc=%d\n", rc); return 1; }
    q35_model_free(m);

    m = q35_model_load(dir, 0, errbuf, sizeof errbuf, &err);
    rc = q35_model_load_state(m, "test_state.bin");
    if (rc != Q35_MODEL_OK) { printf("load fail rc=%d\n", rc); return 1; }
    q35_forward_decode(m, 3, lg_b);
    q35_model_free(m);
    remove("test_state.bin");

    /* compare: save/load roundtrip must be bitwise identical */
    double maxabs = 0;
    for (uint32_t i = 0; i < vocab; i++) {
        double d = fabs((double)lg_a[i] - lg_b[i]);
        if (d > maxabs) maxabs = d;
    }
    printf("save/load roundtrip maxabs=%.6g  %s\n", maxabs, maxabs < 1e-5 ? "OK" : "MISMATCH");
    free(lg_a); free(lg_b);
    return maxabs < 1e-5 ? 0 : 1;
}
