#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "q35/q35_mm.h"

/* minimal manifest parse: cases are fixed-name arrays; we hand-roll tiny reader */
typedef struct { char name[64]; unsigned rows, cols; int fp8; unsigned sr, sc; float tol; } Case;

static unsigned char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n ? *n : 1);
    if (*n && fread(b, 1, *n, f) != *n) { fprintf(stderr, "read %s\n", path); exit(1); }
    fclose(f); return b;
}

static uint64_t fnv1a(const void *p, size_t n) {
    const unsigned char *c = p; uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ULL; }
    return h;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/fixtures/mm";
    size_t mn; char path[512];
    sprintf(path, "%s/manifest.json", dir);
    char *mj = (char *)slurp(path, &mn);
    /* crude JSON scan: pull fields per object */
    int fails = 0, count = 0;
    char *cur = mj;
    while ((cur = strstr(cur, "\"name\"")) != NULL) {
        Case c; memset(&c, 0, sizeof c);
        sscanf(cur, "\"name\": \"%63[^\"]\"", c.name);
        char *p;
        if ((p = strstr(cur, "\"rows\""))) sscanf(p, "\"rows\": %u", &c.rows);
        if ((p = strstr(cur, "\"cols\""))) sscanf(p, "\"cols\": %u", &c.cols);
        c.fp8 = strstr(cur, "\"fp8\": true") && strstr(cur, "\"fp8\": true") < strstr(cur, "\"sr\"") ? 1 : 0;
        if ((p = strstr(cur, "\"sr\""))) sscanf(p, "\"sr\": %u", &c.sr);
        if ((p = strstr(cur, "\"sc\""))) sscanf(p, "\"sc\": %u", &c.sc);
        if ((p = strstr(cur, "\"tol\""))) { double t; sscanf(p, "\"tol\": %lf", &t); c.tol = (float)t; }
        count++;

        size_t wn, xn, yn;
        sprintf(path, "%s/%s.%s", dir, c.name, c.fp8 ? "w8" : "w16");
        unsigned char *w = slurp(path, &wn);
        unsigned char *sc = NULL;
        if (c.fp8) { sprintf(path, "%s/%s.sc", dir, c.name); sc = slurp(path, &(size_t){0}); }
        sprintf(path, "%s/%s.x", dir, c.name);
        float *x = (float *)slurp(path, &xn);
        sprintf(path, "%s/%s.yref", dir, c.name);
        double *yref = (double *)slurp(path, &yn);

        size_t want_w = (size_t)c.rows * c.cols * (c.fp8 ? 1 : 2);
        if (wn != want_w) { printf("FAIL %s: w size %zu != %zu\n", c.name, wn, want_w); fails++; goto next; }
        if (xn != (size_t)c.cols * 4 || yn != (size_t)c.rows * 8) { printf("FAIL %s: x/yref size\n", c.name); fails++; goto next; }

        float *y = malloc((size_t)c.rows * 4);
        if (c.fp8) q35_mm_fp8(w, (const uint16_t *)sc, c.rows, c.cols, x, y);
        else       q35_mm_bf16((const uint16_t *)w, c.rows, c.cols, x, y);

        double maxabs = 0, maxrel = 0;
        for (unsigned i = 0; i < c.rows; i++) {
            double d = fabs((double)y[i] - yref[i]);
            if (d > maxabs) maxabs = d;
            double r = d / (fabs(yref[i]) > 1e-9 ? fabs(yref[i]) : 1e-9);
            if (r > maxrel) maxrel = r;
            if (d > (double)c.tol + (double)c.tol * fabs(yref[i])) {
                printf("FAIL %s row %u: y=%.9g ref=%.9g\n", c.name, i, y[i], yref[i]);
                fails++;
                maxabs = -1.0;
                break;
            }
        }
        if (maxabs < 0) { free(y); goto next; }
        /* determinism anchor: FNV-1a over raw y bytes */
        uint64_t h = fnv1a(y, (size_t)c.rows * 4);
        printf("ok %s (%ux%u %s) impl=%s fnv=%016llx maxabs=%.3g maxrel=%.3g\n",
               c.name, c.rows, c.cols, c.fp8 ? "fp8" : "bf16",
               q35_mm_impl_fp8(), (unsigned long long)h, maxabs, maxrel);
        free(y);
next:
        free(w); free(sc); free(x); free(yref);
        cur += 8;
    }
    free(mj);
    printf("%d cases, %d failed\n", count, fails);
    return fails ? 1 : 0;
}
