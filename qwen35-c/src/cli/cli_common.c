#include "cli_common.h"
#include "q35_plat.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* xorshift32 for sampling */
static uint32_t rng_state;
static uint32_t rng_next(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x ? x : 0x9E3779B9u;
    return rng_state;
}
static double rng_f01(void) { return (rng_next() >> 8) / 16777216.0; }

void cli_srand(uint32_t seed) {
    rng_state = seed;
    if (!rng_state) rng_state = 0x12345678u;
}

typedef struct { uint32_t id; float prob; } TokenProb;

static int cmp_tp(const void *a, const void *b) {
    float pa = ((const TokenProb *)a)->prob, pb = ((const TokenProb *)b)->prob;
    return (pa < pb) - (pa > pb);  /* descending */
}

uint32_t cli_pick_token(const float *logits, uint32_t vocab,
                        double temp, int top_k, double top_p)
{
    /* greedy */
    if (temp <= 0.0) {
        uint32_t bi = 0; float bv = logits[0];
        for (uint32_t i = 1; i < vocab; i++) if (logits[i] > bv) { bv = logits[i]; bi = i; }
        return bi;
    }

    /* softmax with temperature */
    TokenProb *tp = (TokenProb *)malloc((size_t)vocab * sizeof(TokenProb));
    if (!tp) return 0;
    double mx = logits[0];
    for (uint32_t i = 1; i < vocab; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (uint32_t i = 0; i < vocab; i++) {
        double d = (logits[i] - mx) / temp;
        float p = d > -80.0 ? (float)exp(d) : 0.0f;
        tp[i].id = i; tp[i].prob = p; sum += p;
    }
    if (sum <= 0.0) { free(tp); return 0; }
    for (uint32_t i = 0; i < vocab; i++) tp[i].prob = (float)(tp[i].prob / sum);

    /* sort by probability descending */
    qsort(tp, vocab, sizeof(TokenProb), cmp_tp);

    /* apply top-k */
    uint32_t k = vocab;
    if (top_k > 0 && (uint32_t)top_k < k) k = (uint32_t)top_k;

    /* apply top-p (nucleus): keep smallest prefix with cumsum >= top_p) */
    if (top_p > 0.0 && top_p < 1.0) {
        double cum = 0.0; uint32_t kp = k;
        for (uint32_t i = 0; i < k; i++) {
            cum += tp[i].prob;
            if (cum >= top_p) { kp = i + 1; break; }
        }
        k = kp;
    }
    if (k == 0) k = 1;

    /* renormalize over the kept set and sample */
    double ksum = 0.0;
    for (uint32_t i = 0; i < k; i++) ksum += tp[i].prob;
    if (ksum <= 0.0) { k = 1, ksum = tp[0].prob; }
    double r = rng_f01() * ksum, acc = 0.0;
    uint32_t pick = tp[0].id;
    for (uint32_t i = 0; i < k; i++) {
        acc += tp[i].prob;
        if (r <= acc) { pick = tp[i].id; break; }
    }
    free(tp);
    return pick;
}

void cli_omp_tune(int threads, const char *who)
{
#ifdef _OPENMP
    if (threads > 0) {
        omp_set_num_threads(threads);
    } else {
        char *env = getenv("OMP_NUM_THREADS");
        if (!env || !*env) {
            int phys = q35_plat_num_phys_cores();
            if (phys < 1) phys = omp_get_max_threads();
            int auto_thr = phys * 5 / 3;
            int max_thr = omp_get_max_threads();
            if (auto_thr > max_thr) auto_thr = max_thr;
            if (auto_thr < 1) auto_thr = max_thr;
            omp_set_num_threads(auto_thr);
        }
    }
    int nthr = omp_get_max_threads();
    fprintf(stderr, "%s: using %d thread%s\n", who, nthr, nthr == 1 ? "" : "s");
#else
    (void)threads; (void)who;
    fprintf(stderr, "%s: using 1 thread\n", who);
#endif
}
