#include "q35/q35_tok.h"
#include "fixtures/tok_lookup.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

static size_t hex_to_bytes(const char *hex, uint8_t *out)
{
    size_t n = 0;
    while (hex[0] && hex[1]) {
        unsigned v;
        sscanf(hex, "%2x", &v);
        out[n++] = (uint8_t)v;
        hex += 2;
    }
    return n;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "tokenizer/tokenizer.bin";

    Q35TokErr err = Q35_TOK_OK;
    Q35Tok *t = q35_tok_load(path, &err);
    if (!t) {
        fprintf(stderr, "NOT RUN: %s (%s)\n", path, q35_tok_err_str(err));
        return 77; /* convention: fixture/resource missing, not a pass */
    }

    CHECK(q35_tok_declared_vocab(t) == 248320u);
    CHECK(q35_tok_defined_vocab(t) == 248077u);
    CHECK(q35_tok_merge_count(t) == 247587u);
    CHECK(q35_tok_bos(t) == 248044u);
    CHECK(q35_tok_eos(t) == 248044u);
    CHECK(q35_tok_pad(t) == 248044u);
    CHECK(q35_tok_chat_eos(t) == 248046u);
    CHECK(q35_tok_flags(t) & Q35_TOK_FLAG_ISOLATED_SPLIT);
    CHECK(q35_tok_flags(t) & Q35_TOK_FLAG_NFC_NORMALIZE);
    CHECK(q35_tok_pretok_pattern(t) != NULL);
    CHECK(q35_tok_added_count(t) == 33u);

    /* bidirectional lookup against fixtures */
    uint8_t buf[128];
    for (int i = 0; i < TOK_FIXTURES_N; i++) {
        size_t blen = hex_to_bytes(TOK_FIXTURES[i].hex, buf);
        int32_t id = q35_tok_lookup(t, buf, (uint32_t)blen);
        CHECK(id == (int32_t)TOK_FIXTURES[i].id);
        uint32_t outlen = 0;
        const uint8_t *bytes = q35_tok_bytes(t, TOK_FIXTURES[i].id, &outlen);
        CHECK(bytes != NULL && outlen == blen &&
              (blen == 0 || memcmp(bytes, buf, blen) == 0));
    }

    /* decode roundtrip: "Hello world" style concat */
    {
        static const uint32_t ids[] = { 9419u, 1814u }; /* "Hello" + " world" */
        uint8_t text[64]; size_t n = 0;
        for (int i = 0; i < 2; i++) {
            uint32_t l; const uint8_t *b = q35_tok_bytes(t, ids[i], &l);
            CHECK(b != NULL);
            memcpy(text + n, b, l); n += l;
        }
        CHECK(n == 11 && memcmp(text, "Hello world", 11) == 0);
    }

    /* special token query */
    {
        uint32_t id, len; const uint8_t *bytes; int special;
        CHECK(q35_tok_added_get(t, 2, &id, &bytes, &len, &special));
        CHECK(id == 248046u && special == 1);
        CHECK(len == 10 && memcmp(bytes, "<|im_end|>", 10) == 0);
    }

    /* negative: out-of-range id, unknown bytes */
    CHECK(q35_tok_bytes(t, 248077u, NULL) == NULL);
    CHECK(q35_tok_lookup(t, (const uint8_t *)"\xff\xfe\x00\x01\x7f", 5) == -1);

    printf("loaded: defined=%u declared=%u merges=%u added=%u mem=%.1f MB\n",
           q35_tok_defined_vocab(t), q35_tok_declared_vocab(t),
           q35_tok_merge_count(t), q35_tok_added_count(t),
           q35_tok_memory_bytes(t) / 1048576.0);

    q35_tok_free(t);

    if (failures) { printf("FAILURES: %d\n", failures); return 1; }
    printf("PASS\n");
    return 0;
}
