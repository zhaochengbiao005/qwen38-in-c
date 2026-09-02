#include "q35/q35_bpe.h"
#include "q35_unicode_tables.h"

#include <stdlib.h>
#include <string.h>

/* ================= UTF-8 ================= */

static int utf8_decode(const uint8_t *s, size_t n, size_t *i, uint32_t *cp)
{
    uint8_t c = s[*i];
    if (c < 0x80) { *cp = c; (*i)++; return 1; }
    int len; uint32_t v;
    if ((c & 0xE0) == 0xC0) { len = 2; v = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { len = 3; v = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { len = 4; v = c & 0x07; }
    else return 0;
    if (*i + (size_t)len > n) return 0;
    for (int k = 1; k < len; k++) {
        uint8_t cc = s[*i + (size_t)k];
        if ((cc & 0xC0) != 0x80) return 0;
        v = (v << 6) | (cc & 0x3F);
    }
    if ((len == 2 && v < 0x80) || (len == 3 && v < 0x800) ||
        (len == 4 && v < 0x10000) || v > 0x10FFFF ||
        (v >= 0xD800 && v <= 0xDFFF)) return 0;
    *cp = v; *i += (size_t)len; return 1;
}

static int utf8_encode(uint32_t cp, uint8_t out[4])
{
    if (cp < 0x80) { out[0] = (uint8_t)cp; return 1; }
    if (cp < 0x800) { out[0] = 0xC0 | (uint8_t)(cp >> 6); out[1] = 0x80 | (uint8_t)(cp & 0x3F); return 2; }
    if (cp < 0x10000) {
        out[0] = 0xE0 | (uint8_t)(cp >> 12); out[1] = 0x80 | (uint8_t)((cp >> 6) & 0x3F);
        out[2] = 0x80 | (uint8_t)(cp & 0x3F); return 3;
    }
    out[0] = 0xF0 | (uint8_t)(cp >> 18); out[1] = 0x80 | (uint8_t)((cp >> 12) & 0x3F);
    out[2] = 0x80 | (uint8_t)((cp >> 6) & 0x3F); out[3] = 0x80 | (uint8_t)(cp & 0x3F);
    return 4;
}

/* ================= Unicode property lookups ================= */

static int in_ranges(const uint32_t (*r)[2], int n, uint32_t cp)
{
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (cp < r[mid][0]) hi = mid - 1;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}

static int cat_L(uint32_t cp) { return in_ranges(q35_cat_CL, Q35_CL_N, cp); }
static int cat_M(uint32_t cp) { return in_ranges(q35_cat_CM, Q35_CM_N, cp); }
static int cat_N(uint32_t cp) { return in_ranges(q35_cat_CN, Q35_CN_N, cp); }

static uint8_t ccc_of(uint32_t cp)
{
    int lo = 0, hi = Q35_CCC_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (cp < q35_ccc[mid][0]) hi = mid - 1;
        else if (cp > q35_ccc[mid][1]) lo = mid + 1;
        else return (uint8_t)q35_ccc[mid][2];
    }
    return 0;
}

/* \s = Rust regex unicode White_Space */
static int is_ws(uint32_t cp)
{
    switch (cp) {
    case 0x20: case 0x85: case 0xA0: case 0x1680:
    case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
        return 1;
    }
    if (cp >= 0x09 && cp <= 0x0D) return 1;
    if (cp >= 0x2000 && cp <= 0x200A) return 1;
    return 0;
}

/* ================= NFC ================= */

#define SB 0xAC00u
#define LB 0x1100u
#define VB 0x1161u
#define TB 0x11A7u
#define LCNT 19u
#define VCNT 21u
#define TCNT 28u
#define NCNT (VCNT * TCNT)
#define SCNT (LCNT * NCNT)

static const uint8_t *decomp_of(uint32_t cp, uint32_t *len)
{
    int lo = 0, hi = Q35_DECOMP_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (cp < q35_decomp[mid][0]) hi = mid - 1;
        else if (cp > q35_decomp[mid][0]) lo = mid + 1;
        else { *len = q35_decomp[mid][2]; return q35_decomp_bytes + q35_decomp[mid][1]; }
    }
    return NULL;
}

static uint32_t compose_pair(uint32_t c1, uint32_t c2)
{
    if (c1 >= LB && c1 < LB + LCNT && c2 >= VB && c2 < VB + VCNT)
        return SB + ((c1 - LB) * VCNT + (c2 - VB)) * TCNT;
    if (c1 >= SB && c1 < SB + SCNT && ((c1 - SB) % TCNT) == 0 &&
        c2 > TB && c2 < TB + TCNT)
        return c1 + (c2 - TB);
    int lo = 0, hi = Q35_COMPOSE_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        uint32_t a = q35_compose[mid][0], b = q35_compose[mid][1];
        if (c1 < a || (c1 == a && c2 < b)) hi = mid - 1;
        else if (c1 == a && c2 == b) return q35_compose[mid][2];
        else lo = mid + 1;
    }
    return 0;
}

/* NFC: normalize src[0..n) into dst (capacity cap); returns new count */
static int nfc_cps(const uint32_t *src, int n, uint32_t *dst, int cap)
{
    int m = 0;
    for (int i = 0; i < n && m < cap; i++) {
        uint32_t cp = src[i];
        if (cp >= SB && cp < SB + SCNT) { /* Hangul decompose */
            uint32_t s = cp - SB;
            if (m + 2 > cap) break;
            dst[m++] = LB + s / NCNT;
            dst[m++] = VB + (s % NCNT) / TCNT;
            uint32_t t = s % TCNT;
            if (t && m < cap) dst[m++] = TB + t;
            continue;
        }
        uint32_t len; const uint8_t *b = decomp_of(cp, &len);
        if (!b) { dst[m++] = cp; continue; }
        size_t k = 0;
        while (k < len && m < cap) { uint32_t q; if (!utf8_decode(b, len, &k, &q)) break; dst[m++] = q; }
    }
    n = m;
    uint32_t *cps = dst;
    /* canonical ordering (stable) */
    for (int i = 1; i < n; i++) {
        uint8_t c = ccc_of(cps[i]);
        if (!c) continue;
        int j = i;
        while (j > 0) {
            uint8_t p = ccc_of(cps[j - 1]);
            if (p == 0 || p <= c) break;
            uint32_t t = cps[j - 1]; cps[j - 1] = cps[j]; cps[j] = t; j--;
        }
    }
    /* composition */
    if (!n) return 0;
    int out = 1, starter = 0; uint8_t prev_ccc = 0;
    for (int i = 1; i < n; i++) {
        uint8_t c = ccc_of(cps[i]);
        uint32_t comp = compose_pair(cps[starter], cps[i]);
        if (comp && (starter == i - 1 || c == 0 || prev_ccc < c)) {
            cps[starter] = comp;
        } else {
            if (c == 0) { starter = out; prev_ccc = 0; }
            else prev_ccc = c;
            cps[out++] = cps[i];
        }
    }
    return out;
}

/* ================= pretokenizer (Qwen2 pattern, hand-coded) ================= */

/* pattern: (?i:'s|'t|'re|'ve|'m|'ll|'d)
   | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N}
   | \ ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+ */

static int is_crlf(uint32_t cp) { return cp == '\r' || cp == '\n'; }

/* try contraction at cps[i..]; returns 0 or match length (codepoints) */
static int try_contraction(const uint32_t *cps, int n, int i)
{
    static const struct { const char *s; int len; } ct[] = {
        {"'re", 3}, {"'ve", 3}, {"'ll", 3}, {"'s", 2}, {"'t", 2},
        {"'m", 2}, {"'d", 2}
    };
    for (size_t k = 0; k < sizeof(ct) / sizeof(ct[0]); k++) {
        if (i + ct[k].len > n) continue;
        int ok = 1;
        for (int j = 0; j < ct[k].len; j++) {
            uint32_t c = cps[i + (size_t)j];
            char w = ct[k].s[j];
            uint32_t lo = (c >= 'A' && c <= 'Z') ? c + 32 : c;
            uint32_t wl = (w >= 'A' && w <= 'Z') ? (uint32_t)w + 32 : (uint32_t)w;
            if (lo != wl) { ok = 0; break; }
        }
        if (ok) return ct[k].len;
    }
    return 0;
}

/* returns length in codepoints of the piece starting at i (always >= 1) */
static int split_at(const uint32_t *cps, int n, int i)
{
    /* 1. contractions */
    int cl = try_contraction(cps, n, i);
    if (cl) return cl;

    uint32_t c0 = cps[i];
    int L0 = cat_L(c0), M0 = cat_M(c0), N0 = cat_N(c0), W0 = is_ws(c0);

    /* 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ */
    {
        int j = i;
        if (!L0 && !N0 && !is_crlf(c0)) j++;  /* note: M allowed in the optional class */
        int cnt = 0;
        while (j + cnt < n) {
            uint32_t c = cps[j + cnt];
            if (!cat_L(c) && !cat_M(c)) break;
            cnt++;
        }
        if (cnt > 0) return (j - i) + cnt;
        /* fall through to other alternatives if no letter/run here */
    }

    /* 3. \p{N} */
    if (N0) return 1;

    /* 4. \ ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* */
    {
        int j = i;
        if (c0 == ' ') j++;
        int cnt = 0;
        while (j + cnt < n) {
            uint32_t c = cps[j + cnt];
            if (is_ws(c) || cat_L(c) || cat_M(c) || cat_N(c)) break;
            cnt++;
        }
        if (cnt > 0) {
            int t = 0;
            while (j + cnt + t < n && is_crlf(cps[j + cnt + t])) t++;
            return (j - i) + cnt + t;
        }
    }

    /* whitespace alternatives */
    if (W0) {
        int j = i;
        while (j < n && is_ws(cps[j])) j++;
        int run = j - i; /* total ws run length */
        /* 5. \s*[\r\n]+ : trailing part of the run must be crlf */
        {
            int e = j, r = e;
            while (r > i && is_crlf(cps[r - 1])) r--;
            if (e > r) return run; /* \s* keeps prefix, [\r\n]+ takes trailing run */
        }
        /* 6. \s+(?!\S): run followed by end, or shorten to leave ws next */
        if (j == n) return run;
        if (run >= 2) return run - 1;
        /* 7. \s+ */
        return run;
    }

    /* nothing matched: can't happen for valid classes; take 1 */
    return 1;
}

/* ================= pair rank table ================= */

struct Q35Bpe {
    const Q35Tok *tok;
    uint32_t mask;
    uint64_t *pkey;
    uint32_t *prank;
    uint8_t *pused;
};

static uint32_t pair_hash(uint64_t key)
{
    key ^= key >> 33; key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33; key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (uint32_t)key;
}

Q35Bpe *q35_bpe_new(const Q35Tok *tok)
{
    if (!tok) return NULL;
    Q35Bpe *b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->tok = tok;
    uint32_t n = q35_tok_merge_count(tok);
    uint32_t cap = 1;
    while (cap < n * 2) cap <<= 1;
    b->mask = cap - 1;
    b->pkey = malloc((size_t)cap * sizeof(uint64_t));
    b->prank = malloc((size_t)cap * sizeof(uint32_t));
    b->pused = calloc(cap, 1);
    if (!b->pkey || !b->prank || !b->pused) { q35_bpe_free(b); return NULL; }
    for (uint32_t r = 0; r < n; r++) {
        uint32_t a, c;
        if (!q35_tok_merge(tok, r, &a, &c)) { q35_bpe_free(b); return NULL; }
        uint64_t key = ((uint64_t)a << 32) | c;
        uint32_t h = pair_hash(key) & b->mask;
        while (b->pused[h]) {
            if (b->pkey[h] == key) { q35_bpe_free(b); return NULL; } /* dup */
            h = (h + 1) & b->mask;
        }
        b->pused[h] = 1; b->pkey[h] = key; b->prank[h] = r;
    }
    return b;
}

void q35_bpe_free(Q35Bpe *b)
{
    if (!b) return;
    free(b->pkey); free(b->prank); free(b->pused); free(b);
}

static int prank_of(const Q35Bpe *b, uint32_t a, uint32_t c)
{
    uint64_t key = ((uint64_t)a << 32) | c;
    uint32_t h = pair_hash(key) & b->mask;
    while (b->pused[h]) {
        if (b->pkey[h] == key) return (int)b->prank[h];
        h = (h + 1) & b->mask;
    }
    return -1;
}

/* BPE-merge one byte string (one pretoken) into ids.
   Returns count or -1. */
static int bpe_piece(Q35Bpe *b, const uint8_t *bytes, size_t n,
                     uint32_t *out, size_t cap)
{
    if (n == 0) return 0;
    if (n > cap) return -1;
    /* initial: one byte per token, id via single-byte vocab */
    int cnt = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t id = q35_tok_lookup(b->tok, bytes + i, 1);
        if (id < 0) return -1;
        out[cnt++] = (uint32_t)id;
    }
    /* byte sequence per slot for concat-lookup */
    uint8_t **seq = malloc((size_t)cnt * sizeof(uint8_t *));
    uint32_t *slen = malloc((size_t)cnt * sizeof(uint32_t));
    if (!seq || !slen) { free(seq); free(slen); return -1; }
    for (int i = 0; i < cnt; i++) { seq[i] = (uint8_t *)(bytes + i); slen[i] = 1; }

    while (cnt > 1) {
        int best_rank = -1, best_i = -1;
        for (int i = 0; i + 1 < cnt; i++) {
            int r = prank_of(b, out[i], out[i + 1]);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) {
                best_rank = r; best_i = i;
            }
        }
        if (best_rank < 0) break;
        /* merged bytes = concat; find its id via vocab hash */
        uint32_t nl = slen[best_i] + slen[best_i + 1];
        uint8_t *cat = malloc(nl);
        if (!cat) { free(seq); free(slen); return -1; }
        memcpy(cat, seq[best_i], slen[best_i]);
        memcpy(cat + slen[best_i], seq[best_i + 1], slen[best_i + 1]);
        int32_t mid = q35_tok_lookup(b->tok, cat, nl);
        /* free owned buffers (seq slots own memory only after the first merge) */
        if (slen[best_i] > 1) free(seq[best_i]);
        if (slen[best_i + 1] > 1) free(seq[best_i + 1]);
        if (mid < 0) { free(cat); free(seq); free(slen); return -1; }
        out[best_i] = (uint32_t)mid;
        seq[best_i] = cat; slen[best_i] = nl;
        memmove(out + best_i + 1, out + best_i + 2, (size_t)(cnt - best_i - 2) * sizeof(uint32_t));
        memmove(seq + best_i + 1, seq + best_i + 2, (size_t)(cnt - best_i - 2) * sizeof(void *));
        memmove(slen + best_i + 1, slen + best_i + 2, (size_t)(cnt - best_i - 2) * sizeof(uint32_t));
        cnt--;
    }
    for (int i = 0; i < cnt; i++)
        if (slen[i] > 1) free(seq[i]);
    free(seq); free(slen);
    return cnt;
}

/* ============ specials ============ */

/* longest-match special at text[i..]; returns added-index or -1 */
static int match_special(const Q35Tok *t, const char *text, size_t n, size_t i,
                         size_t *mlen, uint32_t *mid)
{
    uint32_t na = q35_tok_added_count(t);
    int best = -1; size_t bl = 0;
    for (uint32_t k = 0; k < na; k++) {
        uint32_t id, len; const uint8_t *bytes; int sp;
        if (!q35_tok_added_get(t, k, &id, &bytes, &len, &sp)) continue;
        (void)sp;
        if (len == 0) continue;
        if (i + len <= n && memcmp(text + i, bytes, len) == 0 && len > bl) {
            best = (int)k; bl = len; *mid = id;
        }
    }
    if (best >= 0) *mlen = bl;
    return best;
}

/* next position >= i where any special starts (or n) */
static size_t next_special(const Q35Tok *t, const char *text, size_t n, size_t i)
{
    uint32_t na = q35_tok_added_count(t);
    size_t b = n;
    for (uint32_t k = 0; k < na; k++) {
        uint32_t id, len; const uint8_t *bytes; int sp;
        if (!q35_tok_added_get(t, k, &id, &bytes, &len, &sp)) continue;
        if (len == 0 || len > n - i) continue;
        for (size_t j = i; j + len <= n; j++)
            if (memcmp(text + j, bytes, len) == 0) { if (j < b) b = j; break; }
    }
    return b;
}

/* decode UTF-8 to codepoints, recording the byte offset of each; returns the
 * codepoint count (stopping at cap), or -1 on invalid UTF-8. coff[count] = n. */
static int decode_to_cps(const uint8_t *s, size_t n, uint32_t *cps,
                         size_t *coff, int cap)
{
    int ncp = 0;
    size_t i = 0;
    while (i < n) {
        uint32_t cp;
        if (!utf8_decode(s + i, n - i, &(size_t){0}, &cp)) return -1;
        coff[ncp] = i;
        i += (cp < 0x80) ? 1 : (cp < 0x800) ? 2 : (cp < 0x10000) ? 3 : 4;
        cps[ncp++] = cp;
        if (ncp >= cap) break;
    }
    coff[ncp] = n;
    return ncp;
}

/* encode a plain-text span (no specials) */
static int encode_span(Q35Bpe *b, const char *text, size_t n,
                       uint32_t *out, size_t cap)
{
    /* to codepoints */
    int capc = (int)n + 8;
    uint32_t *cps = malloc((size_t)capc * sizeof(uint32_t));
    size_t *coff = malloc(((size_t)capc + 1) * sizeof(size_t)); /* byte offsets */
    if (!cps || !coff) { free(cps); free(coff); return -1; }
    int ncp = decode_to_cps((const uint8_t *)text, n, cps, coff, capc);
    if (ncp < 0) { free(cps); free(coff); return -1; }
    if (q35_tok_flags(b->tok) & Q35_TOK_FLAG_NFC_NORMALIZE) {
        int cap2 = ncp * 4 + 16;
        uint32_t *tmp2 = malloc((size_t)cap2 * sizeof(uint32_t));
        if (!tmp2) { free(cps); free(coff); return -1; }
        int n2 = nfc_cps(cps, ncp, tmp2, cap2);
        /* re-encode normalized codepoints to bytes */
        uint8_t *bytes = malloc((size_t)n2 * 4 + 4);
        if (!bytes) { free(tmp2); free(cps); free(coff); return -1; }
        size_t bl = 0;
        for (int k = 0; k < n2; k++) bl += (size_t)utf8_encode(tmp2[k], bytes + bl);
        free(tmp2); free(cps); free(coff);
        /* split+cop the normalized byte string the same way */
        int cap3 = (int)bl + 8;
        uint32_t *c2 = malloc((size_t)cap3 * sizeof(uint32_t));
        size_t *o2 = malloc(((size_t)cap3 + 1) * sizeof(size_t));
        if (!c2 || !o2) { free(bytes); free(c2); free(o2); return -1; }
        int m2 = decode_to_cps(bytes, bl, c2, o2, cap3);
        if (m2 < 0) { free(bytes); free(c2); free(o2); return -1; }
        int total = 0;
        int pos = 0;
        while (pos < m2) {
            int len = split_at(c2, m2, pos);
            size_t sb = o2[pos], eb = o2[pos + len];
            int r = bpe_piece(b, bytes + sb, eb - sb, out + total, cap - (size_t)total);
            if (r < 0) { free(bytes); free(c2); free(o2); return -1; }
            total += r;
            pos += len;
        }
        free(bytes); free(c2); free(o2);
        return total;
    }
    /* no NFC */
    int total = 0, pos = 0;
    while (pos < ncp) {
        int len = split_at(cps, ncp, pos);
        size_t sb = coff[pos], eb = coff[pos + len];
        int r = bpe_piece(b, (const uint8_t *)text + sb, eb - sb,
                          out + total, cap - (size_t)total);
        if (r < 0) { free(cps); free(coff); return -1; }
        total += r;
        pos += len;
    }
    free(cps); free(coff);
    return total;
}

int q35_bpe_encode(Q35Bpe *b, const char *text, uint32_t *out, size_t cap)
{
    if (!b || !text || !out) return -1;
    size_t n = strlen(text);
    if (n == 0) return 0;
    int total = 0;
    size_t i = 0;
    while (i < n) {
        size_t mlen; uint32_t mid;
        if (match_special(b->tok, text, n, i, &mlen, &mid) >= 0) {
            if ((size_t)total >= cap) return -1;
            out[total++] = mid;
            i += mlen;
            continue;
        }
        size_t nxt = next_special(b->tok, text, n, i);
        int r = encode_span(b, text + i, nxt - i, out + total, cap - (size_t)total);
        if (r < 0) return -1;
        total += r;
        i = nxt;
    }
    return total;
}

size_t q35_bpe_decode(const Q35Tok *tok, const uint32_t *ids, size_t n,
                      uint8_t *out, size_t cap)
{
    size_t total = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t len;
        const uint8_t *bytes = q35_tok_bytes(tok, ids[i], &len);
        if (!bytes) return (size_t)-1;
        if (total + len > cap) return (size_t)-1;
        memcpy(out + total, bytes, len);
        total += len;
    }
    return total;
}
