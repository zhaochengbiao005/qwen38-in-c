/* Ticket-22: engine-side real-model gate runner.
 *
 * Loads the real FP8 checkpoint, decodes the fixed prompt token-by-token
 * (bitwise-identical to prefill per #20), greedy-generates, and dumps:
 *   eng_logits.bin  f32 [rows, vocab] (same row layout as real_ref.py)
 *   eng_ids.json    ids + timings + peak RSS
 *
 * Manual gate (needs the 28.7 GiB checkpoint), not part of ctest.
 * Usage: dump_real <model_dir> [prompt_ids_csv]
 */
#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "q35/q35_model.h"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
static size_t peak_rss(void) {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc))
        return (size_t)pmc.PeakWorkingSetSize;
    return 0;
}
#else
static size_t peak_rss(void) { return 0; }
#endif

static double now_s(void) {
    static int init; static clock_t c0;
    if (!init) { init = 1; c0 = clock(); }
    return (double)(clock() - c0) / CLOCKS_PER_SEC;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "../Qwen3.8-27B-FP8";
    int prompt[32] = {760, 6511, 314, 5440, 369};
    int np = 5;
    const int GEN = 8;
    if (argc > 2) {
        np = 0;
        for (char *p = strtok(argv[2], ","); p && np < 32; p = strtok(NULL, ","))
            prompt[np++] = (int)strtol(p, NULL, 10);
    }
    char errbuf[512]; int err;

    double t0 = now_s();
    Q35Model *m = q35_model_load(dir, 0, errbuf, sizeof errbuf, &err);
    if (!m) { fprintf(stderr, "load fail: %s (err=%d)\n", errbuf, err); return 1; }
    double load_s = now_s() - t0;
    uint32_t vocab = q35_model_vocab(m);
    fprintf(stderr, "[eng] loaded in %.1f s, peak RSS %.1f GB\n",
            load_s, (double)peak_rss() / 1073741824.0);

    float *lg = malloc((size_t)vocab * 4);
    FILE *out = fopen("eng_logits.bin", "wb");
    FILE *js = fopen("eng_ids.json", "w");
    int ids[64], nids = 0;
    double decode_s[64]; int ndec = 0;
    for (int t = 0; t < np; t++) ids[nids++] = prompt[t];

    for (int t = 0; t < np; t++) {   /* per-position logits via decode path */
        t0 = now_s();
        if (q35_forward_decode(m, prompt[t], lg) != Q35_MODEL_OK) {
            fprintf(stderr, "decode fail at %d\n", t); return 1;
        }
        decode_s[ndec++] = now_s() - t0;
        fwrite(lg, 4, vocab, out);
    }
    for (int g = 0; g < GEN; g++) {
        uint32_t best = 0;
        for (uint32_t i = 1; i < vocab; i++) if (lg[i] > lg[best]) best = i;
        ids[nids++] = (int)best;
        if (g + 1 < GEN) {
            t0 = now_s();
            if (q35_forward_decode(m, (int32_t)best, lg) != Q35_MODEL_OK) {
                fprintf(stderr, "decode fail at gen %d\n", g); return 1;
            }
            decode_s[ndec++] = now_s() - t0;
            fwrite(lg, 4, vocab, out);
        }
    }
    fclose(out);

    double dec_sum = 0; for (int i = 0; i < ndec; i++) dec_sum += decode_s[i];
    fprintf(stderr, "[eng] %d steps in %.1f s (avg %.2f s/tok), peak RSS %.1f GB\n",
            ndec, dec_sum, dec_sum / ndec, (double)peak_rss() / 1073741824.0);

    fprintf(js, "{\"prompt\": [");
    for (int t = 0; t < np; t++) fprintf(js, "%d%s", prompt[t], t + 1 < np ? ", " : "");
    fprintf(js, "], \"ids\": [");
    for (int t = 0; t < nids; t++) fprintf(js, "%d%s", ids[t], t + 1 < nids ? ", " : "");
    fprintf(js, "], \"gen\": [");
    for (int t = np; t < nids; t++) fprintf(js, "%d%s", ids[t], t + 1 < nids ? ", " : "");
    fprintf(js, "], \"vocab\": %u, \"rows\": %d, \"load_s\": %.2f, "
            "\"decode_avg_s\": %.3f, \"peak_rss_gb\": %.2f}\n",
            vocab, ndec, load_s, dec_sum / ndec, (double)peak_rss() / 1073741824.0);
    fclose(js);
    printf("ids: ");
    for (int t = 0; t < nids; t++) printf("%d ", ids[t]);
    printf("\n");
    q35_model_free(m);
    return 0;
}
