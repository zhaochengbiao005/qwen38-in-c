#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "q35/q35_deltanet.h"
#include "q35/q35_mm.h"

#define SKIP_RETURN_CODE 77

typedef struct {
    char name[64];
    unsigned L, H, Hk, Hv, dk, dv;
    double atol, rtol;
} Case;

static unsigned char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n ? *n : 1);
    if (*n && fread(b, 1, *n, f) != *n) { free(b); fclose(f); return NULL; }
    fclose(f); return b;
}

static int check(const char *tag, const char *name,
                 const float *got, const double *ref, size_t n,
                 double atol, double rtol, int *fails) {
    double maxabs = 0, maxrel = 0;
    size_t i;
    for (i = 0; i < n; i++) {
        double d = fabs((double)got[i] - ref[i]);
        if (d > maxabs) maxabs = d;
        double r = d / (fabs(ref[i]) > 1e-9 ? fabs(ref[i]) : 1e-9);
        if (d / (atol + rtol * fabs(ref[i])) > maxrel) maxrel = r;
        if (d > atol + rtol * fabs(ref[i])) {
            printf("FAIL %s.%s idx %zu: got %.9g ref %.9g\n", name, tag, i,
                   (double)got[i], ref[i]);
            (*fails)++;
            return -1;
        }
    }
    printf("  ok %s.%s maxabs=%.3g maxrel=%.3g\n", name, tag, maxabs, maxrel);
    return 0;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/fixtures/deltanet";
    char path[512];
    size_t mn;
    sprintf(path, "%s/manifest.json", dir);
    char *mj = (char *)slurp(path, &mn);
    if (!mj) {
        printf("SKIP: %s missing, run tools/gen_deltanet_fixture.py\n", path);
        return SKIP_RETURN_CODE;
    }
    q35_mm_init();
    int fails = 0, count = 0;
    char *cur = mj;
    while ((cur = strstr(cur, "\"name\"")) != NULL) {
        Case c; memset(&c, 0, sizeof c);
        sscanf(cur, "\"name\": \"%63[^\"]\"", c.name);
        char *pp;
        if ((pp = strstr(cur, "\"L\"")))  sscanf(pp, "\"L\": %u", &c.L);
        if ((pp = strstr(cur, "\"H\"")))  sscanf(pp, "\"H\": %u", &c.H);
        if ((pp = strstr(cur, "\"Hk\""))) sscanf(pp, "\"Hk\": %u", &c.Hk);
        if ((pp = strstr(cur, "\"Hv\""))) sscanf(pp, "\"Hv\": %u", &c.Hv);
        if ((pp = strstr(cur, "\"dk\""))) sscanf(pp, "\"dk\": %u", &c.dk);
        if ((pp = strstr(cur, "\"dv\""))) sscanf(pp, "\"dv\": %u", &c.dv);
        if ((pp = strstr(cur, "\"atol\""))) sscanf(pp, "\"atol\": %lf", &c.atol);
        if ((pp = strstr(cur, "\"rtol\""))) sscanf(pp, "\"rtol\": %lf", &c.rtol);
        count++;
        printf("case %s L=%u H=%u Hk=%u Hv=%u dk=%u dv=%u\n",
               c.name, c.L, c.H, c.Hk, c.Hv, c.dk, c.dv);

        unsigned K = c.Hk * c.dk, V = c.Hv * c.dv, QKV = 2 * K + V;
        size_t n;
        q35_deltanet_t lay;
        memset(&lay, 0, sizeof lay);
        lay.hidden = c.H; lay.k_heads = c.Hk; lay.v_heads = c.Hv;
        lay.head_k_dim = c.dk; lay.head_v_dim = c.dv;
        lay.l2_eps = 1e-6f; lay.norm_eps = 1e-6f;

        #define LOADP(field, suffix) \
            sprintf(path, "%s/%s.%s", dir, c.name, suffix); \
            lay.field = (void *)slurp(path, &n); \
            if (!lay.field) { printf("FAIL %s: cannot load %s\n", c.name, suffix); fails++; goto next; }
        LOADP(w_qkv, "w_qkv.w8") LOADP(sc_qkv, "w_qkv.sc")
        LOADP(w_z,   "w_z.w8")   LOADP(sc_z,   "w_z.sc")
        LOADP(w_a,   "w_a.w8")   LOADP(sc_a,   "w_a.sc")
        LOADP(w_b,   "w_b.w8")   LOADP(sc_b,   "w_b.sc")
        LOADP(w_out, "w_out.w8") LOADP(sc_out, "w_out.sc")
        LOADP(conv_w, "conv") LOADP(gate_norm_w, "normw")
        LOADP(A_log, "alog") LOADP(dt_bias, "dtbias")

        float *x, *s0, *yref64tmp; double *yref, *sref;
        #define LOADV(var, suffix) \
            sprintf(path, "%s/%s.%s", dir, c.name, suffix); \
            var = (void *)slurp(path, &n); \
            if (!var) { printf("FAIL %s: cannot load %s\n", c.name, suffix); fails++; goto next; }
        LOADV(x, "x") LOADV(s0, "s0") LOADV(yref, "yref") LOADV(sref, "sref")
        (void)yref64tmp;

        {
            float *S = malloc((size_t)c.Hv * c.dk * c.dv * sizeof(float));
            float *conv = malloc((size_t)QKV * 4 * sizeof(float));
            float *y = malloc((size_t)c.L * c.H * sizeof(float));
            memcpy(S, s0, (size_t)c.Hv * c.dk * c.dv * sizeof(float));
            q35_dn_state_t st;
            q35_dn_state_init(&st, S, conv, QKV);
            int rc = q35_dn_forward(&lay, &st, x, y, c.L);
            if (rc != Q35_DN_OK) { printf("FAIL %s: forward rc=%d\n", c.name, rc); fails++; }
            else {
                check("y", c.name, y, yref, (size_t)c.L * c.H, c.atol, c.rtol, &fails);
                check("S", c.name, S, sref, (size_t)c.Hv * c.dk * c.dv, c.atol, c.rtol, &fails);
            }

            /* decode-vs-prefill equivalence: chunk path (seq_len>1) vs
             * per-token path (seq_len=1) use different fp32 reduction orders,
             * so we check tolerance, not bitwise. */
            if (c.L > 1 && rc == Q35_DN_OK) {
                float *S2 = malloc((size_t)c.Hv * c.dk * c.dv * sizeof(float));
                float *conv2 = malloc((size_t)QKV * 4 * sizeof(float));
                float *y2 = malloc((size_t)c.L * c.H * sizeof(float));
                memcpy(S2, s0, (size_t)c.Hv * c.dk * c.dv * sizeof(float));
                q35_dn_state_t st2;
                q35_dn_state_init(&st2, S2, conv2, QKV);
                for (unsigned t = 0; t < c.L; t++)
                    q35_dn_forward(&lay, &st2, x + (size_t)t * c.H,
                                   y2 + (size_t)t * c.H, 1);
                double ymax = 0, smax = 0;
                size_t k;
                int ok = 1;
                for (k = 0; k < (size_t)c.L * c.H; k++) {
                    double d = fabs((double)y[k] - y2[k]);
                    if (d > ymax) ymax = d;
                    if (d > c.atol + c.rtol * fabs(y2[k])) ok = 0;
                }
                for (k = 0; k < (size_t)c.Hv * c.dk * c.dv; k++) {
                    double d = fabs((double)S[k] - S2[k]);
                    if (d > smax) smax = d;
                    if (d > c.atol + c.rtol * fabs(S2[k])) ok = 0;
                }
                if (!ok) {
                    printf("FAIL %s: decode vs prefill tol (ymax=%.3g smax=%.3g)\n",
                           c.name, ymax, smax);
                    fails++;
                } else {
                    printf("  ok %s.decode==prefill tol (ymax=%.3g smax=%.3g)\n",
                           c.name, ymax, smax);
                }
                free(S2); free(conv2); free(y2);
            }
            free(S); free(conv); free(y);
        }
        free(x); free(s0); free(yref); free(sref);
next:
        free((void *)lay.w_qkv); free((void *)lay.sc_qkv);
        free((void *)lay.w_z); free((void *)lay.sc_z);
        free((void *)lay.w_a); free((void *)lay.sc_a);
        free((void *)lay.w_b); free((void *)lay.sc_b);
        free((void *)lay.w_out); free((void *)lay.sc_out);
        free((void *)lay.conv_w); free((void *)lay.gate_norm_w);
        free((void *)lay.A_log); free((void *)lay.dt_bias);
        cur += 8;
    }
    free(mj);
    printf("%d cases, %d failed\n", count, fails);
    return fails ? 1 : 0;
}
