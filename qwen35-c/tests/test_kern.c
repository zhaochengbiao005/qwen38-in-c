/* Ticket-16: kernel fixture harness.
 * Reads tests/fixtures/kern/manifest.txt, runs each case through both the
 * scalar and default (AVX2) kernel paths, asserts:
 *   1. FNV-1a(scalar out) == FNV-1a(default out)  (bit-identical paths)
 *   2. both outputs within manifest atol/rtol of the NumPy reference
 *   3. fixture output bytes match manifest crc32 (fixture integrity)
 * Missing manifest/fixtures: prints NOT RUN and exits 77.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "q35/q35_kern.h"
#include "q35_plat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MAX_IN 8
#define MAXF  1048576

static int failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); failures++; } \
} while (0)

struct kv { char key[32], val[256]; };

struct case_ {
    char name[128];
    char kernel[32];
    double atol, rtol;
    uint32_t crc32;
    char in[MAX_IN][128]; int nin;
    char out[128];
    long n, heads, head_dim, rotary_dim, pos;
    double eps, theta;
};

static uint32_t crc32_of(const void *data, size_t len)
{
    static uint32_t tab[256];
    static int init = 0;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            tab[i] = c;
        }
        init = 1;
    }
    uint32_t c = 0xFFFFFFFFu;
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < len; i++) c = tab[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static float *load_f32(const char *dir, const char *file, size_t *count)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    size_t len = 0;
    void *buf = q35_plat_read_file(path, &len);
    if (!buf) return NULL;
    if (len % 4) { q35_plat_free(buf); return NULL; }
    *count = len / 4;
    return (float *)buf;
}

static int parse_line(char *line, struct case_ *c)
{
    memset(c, 0, sizeof(*c));
    c->eps = 1e-6; c->theta = 1e7; c->heads = 1;
    char *tok = strtok(line, " \t\r\n");
    if (!tok || tok[0] == '#') return 0;
    strncpy(c->name, tok, sizeof(c->name) - 1);
    while ((tok = strtok(NULL, " \t\r\n")) != NULL) {
        char *eq = strchr(tok, '=');
        if (!eq) continue;
        *eq = 0;
        char *k = tok, *v = eq + 1;
        if (!strcmp(k, "kernel")) strncpy(c->kernel, v, sizeof(c->kernel) - 1);
        else if (!strcmp(k, "atol")) c->atol = atof(v);
        else if (!strcmp(k, "rtol")) c->rtol = atof(v);
        else if (!strcmp(k, "crc32")) c->crc32 = (uint32_t)strtoul(v, NULL, 16);
        else if (!strcmp(k, "out")) strncpy(c->out, v, sizeof(c->out) - 1);
        else if (!strcmp(k, "n")) c->n = atol(v);
        else if (!strcmp(k, "heads")) c->heads = atol(v);
        else if (!strcmp(k, "head_dim")) c->head_dim = atol(v);
        else if (!strcmp(k, "rotary_dim")) c->rotary_dim = atol(v);
        else if (!strcmp(k, "pos")) c->pos = atol(v);
        else if (!strcmp(k, "eps")) c->eps = atof(v);
        else if (!strcmp(k, "theta")) c->theta = atof(v);
        else if (!strcmp(k, "in")) {
            /* manual comma split: strtok here would clobber the outer
             * strtok state and eat every manifest field after "in=" */
            char *p = v;
            while (*p && c->nin < MAX_IN) {
                char *comma = strchr(p, ',');
                if (comma) *comma = 0;
                strncpy(c->in[c->nin++], p, 127);
                if (!comma) break;
                p = comma + 1;
            }
        }
    }
    return c->kernel[0] != 0;
}

static double compare(const float *got, const float *ref, size_t n,
                      double atol, double rtol, double *maxabs, double *maxrel)
{
    *maxabs = 0; *maxrel = 0;
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        double da = fabs((double)got[i] - (double)ref[i]);
        double dr = da / (fabs((double)ref[i]) + 1e-30);
        if (da > *maxabs) *maxabs = da;
        if (dr > *maxrel) *maxrel = dr;
        if (da > atol + rtol * fabs((double)ref[i])) bad++;
    }
    return bad == 0;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "tests/fixtures/kern";
    char mpath[512];
    snprintf(mpath, sizeof(mpath), "%s/manifest.txt", dir);
    FILE *mf = fopen(mpath, "rb");
    if (!mf) { fprintf(stderr, "NOT RUN: no fixtures at %s\n", mpath); return 77; }

    char line[2048];
    int ncase = 0, notrun = 0;
    while (fgets(line, sizeof(line), mf)) {
        struct case_ c;
        if (!parse_line(line, &c)) continue;
        ncase++;

        float *ins[MAX_IN] = {0};
        size_t icnt[MAX_IN] = {0};
        int missing = 0;
        for (int i = 0; i < c.nin; i++) {
            ins[i] = load_f32(dir, c.in[i], &icnt[i]);
            if (!ins[i]) missing = 1;
        }
        size_t ocnt = 0;
        float *ref = load_f32(dir, c.out, &ocnt);
        if (missing || !ref) {
            printf("NOT RUN: %s (fixture file missing)\n", c.name);
            notrun++;
            for (int i = 0; i < c.nin; i++) if (ins[i]) q35_plat_free(ins[i]);
            if (ref) q35_plat_free(ref);
            continue;
        }
        if (crc32_of(ref, ocnt * 4) != c.crc32) {
            failures++;
            printf("FAIL: %s fixture crc32 mismatch\n", c.name);
        }

        size_t n = ocnt;
        float *out_d = (float *)malloc(n * 4);
        float *out_s = (float *)malloc(n * 4);
        memcpy(out_d, ins[0], n * 4);
        memcpy(out_s, ins[0], n * 4);

        int rc_d = 0, rc_s = 0;
        if (!strcmp(c.kernel, "rmsnorm")) {
            q35_kern_rmsnorm(ins[0], ins[1], out_d, (size_t)c.n, (float)c.eps);
            q35_kern_rmsnorm_scalar(ins[0], ins[1], out_s, (size_t)c.n, (float)c.eps);
        } else if (!strcmp(c.kernel, "silu")) {
            q35_kern_silu(ins[0], out_d, (size_t)c.n);
            q35_kern_silu_scalar(ins[0], out_s, (size_t)c.n);
        } else if (!strcmp(c.kernel, "swiglu")) {
            q35_kern_swiglu(ins[0], ins[1], out_d, (size_t)c.n);
            q35_kern_swiglu_scalar(ins[0], ins[1], out_s, (size_t)c.n);
        } else if (!strcmp(c.kernel, "qknorm")) {
            q35_kern_qk_norm(out_d, ins[1], (size_t)c.heads, (size_t)c.head_dim, (float)c.eps);
            q35_kern_qk_norm_scalar(out_s, ins[1], (size_t)c.heads, (size_t)c.head_dim, (float)c.eps);
        } else if (!strcmp(c.kernel, "rope")) {
            rc_d = q35_kern_rope(out_d, (size_t)c.heads, (size_t)c.head_dim,
                                 (size_t)c.rotary_dim, (int)c.pos, (float)c.theta);
            rc_s = q35_kern_rope_scalar(out_s, (size_t)c.heads, (size_t)c.head_dim,
                                        (size_t)c.rotary_dim, (int)c.pos, (float)c.theta);
        } else if (!strcmp(c.kernel, "attn_gate")) {
            q35_kern_attn_gate(ins[0], ins[1], out_d, (size_t)c.n);
            q35_kern_attn_gate_scalar(ins[0], ins[1], out_s, (size_t)c.n);
        } else if (!strcmp(c.kernel, "softmax")) {
            q35_kern_softmax(ins[0], out_d, (size_t)c.n);
            q35_kern_softmax_scalar(ins[0], out_s, (size_t)c.n);
        } else {
            CHECK(0, "%s unknown kernel %s", c.name, c.kernel);
        }
        CHECK(rc_d == Q35_KERN_OK && rc_s == Q35_KERN_OK, "%s kernel rc", c.name);

        unsigned hd = q35_kern_fnv1a(out_d, n * 4);
        unsigned hs = q35_kern_fnv1a(out_s, n * 4);
        CHECK(hd == hs, "%s fnv1a default=%08x scalar=%08x", c.name, hd, hs);

        double ma_d, mr_d, ma_s, mr_s;
        int ok_d = (int)compare(out_d, ref, n, c.atol, c.rtol, &ma_d, &mr_d);
        int ok_s = (int)compare(out_s, ref, n, c.atol, c.rtol, &ma_s, &mr_s);
        CHECK(ok_d, "%s default vs fixture maxabs=%.3g maxrel=%.3g", c.name, ma_d, mr_d);
        CHECK(ok_s, "%s scalar vs fixture maxabs=%.3g maxrel=%.3g", c.name, ma_s, mr_s);
        printf("PASS %-18s maxabs=%.3g maxrel=%.3g fnv1a=%08x%s\n",
               c.name, ma_d, mr_d, hd, hd == hs ? "" : " HASH-MISMATCH");

        for (int i = 0; i < c.nin; i++) q35_plat_free(ins[i]);
        q35_plat_free(ref);
        free(out_d); free(out_s);
    }
    fclose(mf);

    if (ncase == 0) { printf("NOT RUN: empty manifest\n"); return 77; }
    printf("kern: %d cases, %d not-run, %d failures\n", ncase, notrun, failures);
    return failures ? 1 : 0;
}