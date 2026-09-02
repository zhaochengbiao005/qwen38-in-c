/* Verify q35_mm_fp8_batch == q35_mm_fp8 called n times (bit-identical). */
#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "q35/q35_mm.h"
#include "q35/q35_kern.h"

#define fnv1a q35_kern_fnv1a

int main(void) {
    /* dims similar to real model: W[10240, 5120], X[5, 5120] */
    uint32_t rows = 10240, cols = 5120, n = 5;
    uint8_t *W = malloc((size_t)rows * cols);
    uint16_t *sc = calloc((size_t)((rows/128+1) * (cols/128+1)), 2);
    float *X = malloc((size_t)n * cols * 4);
    float *Y_batch = malloc((size_t)rows * n * 4);
    float *Y_single = malloc((size_t)rows * 4);
    for (size_t i = 0; i < (size_t)rows * cols; i++) W[i] = (uint8_t)(i * 7 + 13);
    for (size_t i = 0; i < (size_t)n * cols; i++) X[i] = (float)((i % 200) - 100) * 0.001f;
    q35_mm_init();

    /* batch */
    q35_mm_fp8_batch(W, sc, rows, cols, n, X, Y_batch);

    /* single n times */
    unsigned h_batch = 0, h_single = 0;
    for (uint32_t t = 0; t < n; t++) {
        q35_mm_fp8(W, sc, rows, cols, X + (size_t)t * cols, Y_single);
        h_single ^= fnv1a(Y_single, rows * 4);
        h_batch ^= fnv1a(Y_batch + (size_t)t * rows, rows * 4);
    }

    /* also check element-wise maxabs — output is [n, rows] (token-major) */
    double maxabs = 0;
    for (uint32_t t = 0; t < n; t++)
        for (uint32_t r = 0; r < rows; r++) {
            double d = fabs(Y_batch[(size_t)t * rows + r] - Y_single[r]);
            if (d > maxabs) maxabs = d;
        }

    printf("batch vs single: fnv batch=%08x single=%08x  maxabs=%.3g\n",
           h_batch, h_single, maxabs);
    /* also verify batch vs multi4 (different code path, same per-row logic) */
    {
        const uint8_t *Ws[4] = { W, NULL, NULL, NULL };
        const uint16_t *Ss[4] = { sc, NULL, NULL, NULL };
        uint32_t Rs[4] = { rows, 0, 0, 0 };
        float *Y_m4 = malloc((size_t)rows * 4);
        float *Y_b1 = malloc(4);
        float *Ys[4] = { Y_m4, Y_b1, Y_b1, Y_b1 };
        q35_mm_fp8_multi4(Ws, Ss, Rs, cols, X, Ys);  /* first token only */
        double m4_maxabs = 0;
        unsigned h_m4 = fnv1a(Y_m4, rows * 4);
        unsigned h_b0 = fnv1a(Y_batch, rows * 4);  /* first token column */
        /* multi4 output is row-major [rows], batch is [n, rows] — compare first token */
        for (uint32_t r = 0; r < rows; r++) {
            double d = fabs(Y_m4[r] - Y_batch[r]);  /* first token: Y_batch[0*rows + r] */
            if (d > m4_maxabs) m4_maxabs = d;
        }
        printf("batch vs multi4: fnv batch=%08x multi4=%08x maxabs=%.3g\n",
               h_b0, h_m4, m4_maxabs);
        if (m4_maxabs != 0.0) printf("  *** MISMATCH — batch != multi4!\n");
        free(Y_m4); free(Y_b1);
    }

    printf("%s\n", (h_batch == h_single && maxabs == 0.0) ? "ALL OK" : "FAILED");
    free(W); free(sc); free(X); free(Y_batch); free(Y_single);
    return (h_batch == h_single && maxabs == 0.0) ? 0 : 1;
}
