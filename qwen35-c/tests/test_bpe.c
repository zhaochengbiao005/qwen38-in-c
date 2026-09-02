#include "q35/q35_bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

static size_t unhex(const char *h, uint8_t *out)
{
    size_t n = 0;
    while (h[0] && h[1] && h[0] != '\t') {
        unsigned v;
        sscanf(h, "%2x", &v);
        out[n++] = (uint8_t)v;
        h += 2;
    }
    return n;
}

int main(int argc, char **argv)
{
    const char *tok_path = argc > 1 ? argv[1] : "tokenizer/tokenizer.bin";
    const char *fx_path = argc > 2 ? argv[2] : "tests/fixtures/encode_fixtures.txt";

    Q35TokErr err;
    Q35Tok *tok = q35_tok_load(tok_path, &err);
    if (!tok) { fprintf(stderr, "NOT RUN: blob %s\n", q35_tok_err_str(err)); return 77; }
    FILE *f = fopen(fx_path, "rb");
    if (!f) { fprintf(stderr, "NOT RUN: fixtures missing\n"); q35_tok_free(tok); return 77; }

    Q35Bpe *b = q35_bpe_new(tok);
    CHECK(b != NULL);

    char line[1 << 16];
    int ncase = 0, npass = 0;
    int failed_cases[64]; int nfailed = 0;
    while (fgets(line, sizeof(line), f)) {
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        uint8_t text[8192];
        size_t tn = unhex(line, text);
        text[tn] = 0;
        /* expected ids */
        uint32_t want[8192]; size_t nw = 0;
        for (char *p = tab + 1; *p && *p != '\n' && *p != '\r';) {
            want[nw++] = (uint32_t)strtoul(p, &p, 10);
            if (*p == ',') p++;
        }
        uint32_t got[8192];
        int ng = q35_bpe_encode(b, (const char *)text, got, 8192);
        int same = (ng == (int)nw) && (nw == 0 || memcmp(want, got, nw * 4) == 0);
        if (!same) {
            if (nfailed < 64) failed_cases[nfailed++] = ncase;
            if (failures < 3) {
                printf("CASE %d mismatch: want %zu ids, got %d\n", ncase, nw, ng);
                size_t m = nw < (size_t)ng ? nw : (size_t)ng;
                for (size_t i = 0; i < m; i++)
                    if (want[i] != got[i]) {
                        printf("  first diff at %zu: want %u got %u\n", i, want[i], got[i]);
                        break;
                    }
            }
            failures++;
        } else {
            npass++;
        }
        ncase++;
    }
    fclose(f);

    /* decode roundtrip on first case */
    printf("cases: %d pass, %d fail (total %d)\n", npass, ncase - npass, ncase);
    CHECK(nfailed == 0);

    q35_bpe_free(b);
    q35_tok_free(tok);
    if (failures) return 1;
    printf("PASS\n");
    return 0;
}
