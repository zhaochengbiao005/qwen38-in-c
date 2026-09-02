#include "q35/q35_tok.h"
#include "q35/q35_kern.h"
#include "../src/q35_plat.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- little-endian readers (blob is LE; x86-64 is LE but stay explicit) ---- */
static uint32_t rd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}
static uint64_t rd_u64(const uint8_t *p)
{
    uint64_t lo = rd_u32(p), hi = rd_u32(p + 4);
    return lo | (hi << 32);
}

struct Q35Tok {
    /* META */
    uint32_t flags;
    uint32_t declared_vocab, defined_vocab;
    uint32_t bos, eos, pad, chat_eos;
    /* PRET */
    char *pattern; /* heap, NUL-terminated utf-8 */
    /* VOCB */
    uint32_t vocab_count;
    const uint8_t **bytes;  /* [vocab_count] into blob copy */
    uint32_t *lens;
    /* MRGS */
    uint32_t merge_count;
    uint32_t *merge_a, *merge_b; /* [merge_count] */
    /* ATOK */
    uint32_t added_count;
    uint32_t *added_id, *added_len;
    uint8_t *added_special;
    const uint8_t **added_bytes;
    /* hash table bytes->id over VOCB */
    uint32_t hash_size_mask;
    int32_t *hash_ids;    /* -1 empty */
    uint8_t *blob;        /* owned copy of file */
    size_t blob_len;
};

const char *q35_tok_err_str(Q35TokErr e)
{
    switch (e) {
    case Q35_TOK_OK: return "ok";
    case Q35_TOK_ERR_OPEN: return "cannot open file";
    case Q35_TOK_ERR_MAGIC: return "bad magic";
    case Q35_TOK_ERR_VERSION: return "unsupported version";
    case Q35_TOK_ERR_TRUNCATED: return "truncated blob";
    case Q35_TOK_ERR_SECTION: return "malformed section";
    case Q35_TOK_ERR_INTEGRITY: return "integrity check failed";
    case Q35_TOK_ERR_NOMEM: return "out of memory";
    }
    return "unknown";
}

void q35_tok_free(Q35Tok *t)
{
    if (!t) return;
    free(t->pattern);
    free(t->bytes);
    free(t->lens);
    free(t->merge_a);
    free(t->merge_b);
    free(t->added_id);
    free(t->added_len);
    free(t->added_special);
    free(t->added_bytes);
    free(t->hash_ids);
    q35_plat_free(t->blob);
    free(t);
}

static Q35TokErr parse_blob(Q35Tok *t)
{
    const uint8_t *p = t->blob;
    size_t n = t->blob_len;
    if (n < 16) return Q35_TOK_ERR_TRUNCATED;
    if (memcmp(p, Q35_TOK_MAGIC, 8) != 0) return Q35_TOK_ERR_MAGIC;
    if (rd_u32(p + 8) != Q35_TOK_VERSION) return Q35_TOK_ERR_VERSION;
    uint32_t nsec = rd_u32(p + 12);
    size_t off = 16;

    int got_meta = 0, got_pret = 0, got_vocb = 0, got_mrgs = 0, got_atok = 0;
    for (uint32_t s = 0; s < nsec; s++) {
        if (off + 12 > n) return Q35_TOK_ERR_TRUNCATED;
        char tag[5];
        memcpy(tag, p + off, 4); tag[4] = 0;
        uint64_t plen = rd_u64(p + off + 4);
        off += 12;
        if (plen > n - off) return Q35_TOK_ERR_SECTION;
        const uint8_t *q = p + off;
        const uint8_t *qend = q + plen;

        if (memcmp(tag, "META", 4) == 0) {
            if (plen < 28) return Q35_TOK_ERR_SECTION;
            t->flags = rd_u32(q);
            t->declared_vocab = rd_u32(q + 4);
            t->defined_vocab = rd_u32(q + 8);
            t->bos = rd_u32(q + 12);
            t->eos = rd_u32(q + 16);
            t->pad = rd_u32(q + 20);
            t->chat_eos = rd_u32(q + 24);
            got_meta = 1;
        } else if (memcmp(tag, "PRET", 4) == 0) {
            if (plen < 8) return Q35_TOK_ERR_SECTION;
            t->flags |= rd_u32(q); /* section flags (isolated split) */
            uint32_t rlen = rd_u32(q + 4);
            if (q + 8 + rlen > qend) return Q35_TOK_ERR_SECTION;
            t->pattern = malloc((size_t)rlen + 1);
            if (!t->pattern) return Q35_TOK_ERR_NOMEM;
            memcpy(t->pattern, q + 8, rlen);
            t->pattern[rlen] = 0;
            got_pret = 1;
        } else if (memcmp(tag, "VOCB", 4) == 0) {
            if (plen < 4) return Q35_TOK_ERR_SECTION;
            uint32_t count = rd_u32(q);
            q += 4;
            t->vocab_count = count;
            t->bytes = calloc(count, sizeof(uint8_t *));
            t->lens = calloc(count, sizeof(uint32_t));
            if (!t->bytes || !t->lens) return Q35_TOK_ERR_NOMEM;
            for (uint32_t i = 0; i < count; i++) {
                if (q + 8 > qend) return Q35_TOK_ERR_SECTION;
                uint32_t id = rd_u32(q), blen = rd_u32(q + 4);
                q += 8;
                if (blen > (uint32_t)(qend - q)) return Q35_TOK_ERR_SECTION;
                if (id >= count) return Q35_TOK_ERR_INTEGRITY;
                if (t->lens[id] != 0 || t->bytes[id] != NULL)
                    return Q35_TOK_ERR_INTEGRITY; /* duplicate id (or id 0 twice) */
                t->lens[id] = blen;
                t->bytes[id] = q;
                q += blen;
            }
            if (q != qend) return Q35_TOK_ERR_SECTION;
            got_vocb = 1;
        } else if (memcmp(tag, "MRGS", 4) == 0) {
            if (plen < 4) return Q35_TOK_ERR_SECTION;
            uint32_t count = rd_u32(q);
            q += 4;
            if ((uint64_t)count * 8 > (uint64_t)(qend - q)) return Q35_TOK_ERR_SECTION;
            t->merge_count = count;
            t->merge_a = malloc((size_t)count * 4);
            t->merge_b = malloc((size_t)count * 4);
            if (!t->merge_a || !t->merge_b) return Q35_TOK_ERR_NOMEM;
            for (uint32_t i = 0; i < count; i++) {
                t->merge_a[i] = rd_u32(q + (size_t)i * 8);
                t->merge_b[i] = rd_u32(q + (size_t)i * 8 + 4);
            }
            got_mrgs = 1;
        } else if (memcmp(tag, "ATOK", 4) == 0) {
            if (plen < 4) return Q35_TOK_ERR_SECTION;
            uint32_t count = rd_u32(q);
            q += 4;
            t->added_count = count;
            t->added_id = calloc(count, sizeof(uint32_t));
            t->added_len = calloc(count, sizeof(uint32_t));
            t->added_special = calloc(count, sizeof(uint8_t));
            t->added_bytes = calloc(count, sizeof(uint8_t *));
            if (!t->added_id || !t->added_len || !t->added_special || !t->added_bytes)
                return Q35_TOK_ERR_NOMEM;
            for (uint32_t i = 0; i < count; i++) {
                if (q + 12 > qend) return Q35_TOK_ERR_SECTION;
                t->added_id[i] = rd_u32(q);
                t->added_special[i] = (uint8_t)rd_u32(q + 4);
                uint32_t alen = rd_u32(q + 8);
                q += 12;
                if (alen > (uint32_t)(qend - q)) return Q35_TOK_ERR_SECTION;
                t->added_len[i] = alen;
                t->added_bytes[i] = q;
                q += alen;
            }
            if (q != qend) return Q35_TOK_ERR_SECTION;
            got_atok = 1;
        }
        off += (size_t)plen;
    }
    if (off != n) return Q35_TOK_ERR_SECTION;
    if (!got_meta || !got_pret || !got_vocb || !got_mrgs || !got_atok)
        return Q35_TOK_ERR_INTEGRITY;
    return Q35_TOK_OK;
}

/* cross-section integrity: ids inside ranges, added ids continue base vocab */
static Q35TokErr verify_integrity(Q35Tok *t)
{
    if (t->defined_vocab == 0 || t->defined_vocab != t->vocab_count)
        return Q35_TOK_ERR_INTEGRITY;
    if (t->declared_vocab < t->defined_vocab)
        return Q35_TOK_ERR_INTEGRITY;
    for (uint32_t i = 0; i < t->merge_count; i++)
        if (t->merge_a[i] >= t->defined_vocab || t->merge_b[i] >= t->defined_vocab)
            return Q35_TOK_ERR_INTEGRITY;
    for (uint32_t i = 0; i < t->added_count; i++)
        if (t->added_id[i] >= t->defined_vocab)
            return Q35_TOK_ERR_INTEGRITY;
    if (t->bos >= t->defined_vocab || t->eos >= t->defined_vocab ||
        t->pad >= t->defined_vocab || t->chat_eos >= t->defined_vocab)
        return Q35_TOK_ERR_INTEGRITY;
    return Q35_TOK_OK;
}

static Q35TokErr build_hash(Q35Tok *t)
{
    uint32_t cap = 1;
    while (cap < t->vocab_count * 2) cap <<= 1;
    t->hash_size_mask = cap - 1;
    t->hash_ids = malloc((size_t)cap * sizeof(int32_t));
    if (!t->hash_ids) return Q35_TOK_ERR_NOMEM;
    for (uint32_t i = 0; i < cap; i++) t->hash_ids[i] = -1;
    for (uint32_t id = 0; id < t->vocab_count; id++) {
        uint32_t len = t->lens[id];
        /* id 0 with len==0 is impossible in practice; guard anyway */
        uint32_t h = q35_kern_fnv1a(t->bytes[id], len) & t->hash_size_mask;
        while (t->hash_ids[h] >= 0) h = (h + 1) & t->hash_size_mask;
        t->hash_ids[h] = (int32_t)id;
    }
    return Q35_TOK_OK;
}

Q35Tok *q35_tok_load(const char *path, Q35TokErr *err)
{
    size_t len = 0;
    uint8_t *blob = q35_plat_read_file(path, &len);
    if (!blob) { if (err) *err = Q35_TOK_ERR_OPEN; return NULL; }
    Q35Tok *t = calloc(1, sizeof(Q35Tok));
    if (!t) { q35_plat_free(blob); if (err) *err = Q35_TOK_ERR_NOMEM; return NULL; }
    t->blob = blob;
    t->blob_len = len;
    Q35TokErr e = parse_blob(t);
    if (e == Q35_TOK_OK) e = verify_integrity(t);
    if (e == Q35_TOK_OK) e = build_hash(t);
    if (e != Q35_TOK_OK) {
        q35_tok_free(t);
        t = NULL;
    }
    if (err) *err = e;
    return t;
}

uint32_t q35_tok_declared_vocab(const Q35Tok *t) { return t->declared_vocab; }
uint32_t q35_tok_defined_vocab(const Q35Tok *t) { return t->defined_vocab; }
uint32_t q35_tok_merge_count(const Q35Tok *t) { return t->merge_count; }
uint32_t q35_tok_flags(const Q35Tok *t) { return t->flags; }
uint32_t q35_tok_bos(const Q35Tok *t) { return t->bos; }
uint32_t q35_tok_eos(const Q35Tok *t) { return t->eos; }
uint32_t q35_tok_pad(const Q35Tok *t) { return t->pad; }
uint32_t q35_tok_chat_eos(const Q35Tok *t) { return t->chat_eos; }
const char *q35_tok_pretok_pattern(const Q35Tok *t) { return t->pattern; }

const uint8_t *q35_tok_bytes(const Q35Tok *t, uint32_t id, uint32_t *len)
{
    if (!t || id >= t->vocab_count) return NULL;
    if (len) *len = t->lens[id];
    return t->bytes[id];
}

int32_t q35_tok_lookup(const Q35Tok *t, const uint8_t *bytes, uint32_t len)
{
    uint32_t h = q35_kern_fnv1a(bytes, len) & t->hash_size_mask;
    for (;;) {
        int32_t id = t->hash_ids[h];
        if (id < 0) return -1;
        if (t->lens[id] == len &&
            (len == 0 || memcmp(t->bytes[id], bytes, len) == 0))
            return id;
        h = (h + 1) & t->hash_size_mask;
    }
}

uint32_t q35_tok_added_count(const Q35Tok *t) { return t->added_count; }

int q35_tok_added_get(const Q35Tok *t, uint32_t index,
                      uint32_t *id, const uint8_t **bytes, uint32_t *len,
                      int *is_special)
{
    if (!t || index >= t->added_count) return 0;
    if (id) *id = t->added_id[index];
    if (bytes) *bytes = t->added_bytes[index];
    if (len) *len = t->added_len[index];
    if (is_special) *is_special = t->added_special[index] ? 1 : 0;
    return 1;
}

int q35_tok_merge(const Q35Tok *t, uint32_t index, uint32_t *a, uint32_t *b)
{
    if (!t || index >= t->merge_count) return 0;
    if (a) *a = t->merge_a[index];
    if (b) *b = t->merge_b[index];
    return 1;
}

size_t q35_tok_memory_bytes(const Q35Tok *t)
{
    size_t m = sizeof(Q35Tok) + t->blob_len;
    m += (size_t)t->vocab_count * (sizeof(uint8_t *) + sizeof(uint32_t));
    m += (size_t)t->merge_count * 2 * sizeof(uint32_t);
    m += (size_t)t->added_count * (3 * sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint8_t *));
    m += ((size_t)t->hash_size_mask + 1) * sizeof(int32_t);
    m += t->pattern ? strlen(t->pattern) + 1 : 0;
    return m;
}


