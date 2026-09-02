#include "q35/q35_json.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct {
    char **ptrs; size_t n, cap;
} Arena;

struct Q35Json {
    Arena arena;
    JVal *root;
};

static void arena_push(Q35Json *j, void *p)
{
    if (j->arena.n == j->arena.cap) {
        j->arena.cap = j->arena.cap ? j->arena.cap * 2 : 64;
        j->arena.ptrs = realloc(j->arena.ptrs, j->arena.cap * sizeof(void *));
    }
    j->arena.ptrs[j->arena.n++] = p;
}

static void *amalloc(Q35Json *j, size_t n)
{
    void *p = calloc(1, n);
    if (p) arena_push(j, p);
    return p;
}

typedef struct {
    const char *s, *end;
    Q35Json *j;
    int err;
} P;

static void ws(P *p) { while (p->s < p->end && isspace((unsigned char)*p->s)) p->s++; }
static int peek(P *p) { return p->s < p->end ? *p->s : -1; }

static JVal *parse_val(P *p);

static char *parse_str(P *p)
{
    if (peek(p) != '"') { p->err = 1; return NULL; }
    p->s++;
    size_t cap = 32, n = 0;
    char *buf = malloc(cap);
    while (p->s < p->end) {
        int c = *p->s++;
        if (c == '"') { buf[n] = 0; /* adopt into arena */ char *r = amalloc(p->j, n + 1); memcpy(r, buf, n + 1); free(buf); return r; }
        if (n + 5 > cap) { cap *= 2; buf = realloc(buf, cap); }
        if (c == '\\') {
            if (p->s >= p->end) break;
            int e = *p->s++;
            switch (e) {
            case '"': c = '"'; break;
            case '\\': c = '\\'; break;
            case '/': c = '/'; break;
            case 'b': c = 8; break;
            case 'f': c = 12; break;
            case 'n': c = 10; break;
            case 'r': c = 13; break;
            case 't': c = 9; break;
            case 'u': {
                if (p->end - p->s < 4) { p->err = 1; free(buf); return NULL; }
                unsigned cp = 0;
                for (int i = 0; i < 4; i++) {
                    int h = *p->s++;
                    cp <<= 4;
                    cp |= (h >= '0' && h <= '9') ? h - '0' : (h >= 'a' && h <= 'f') ? h - 'a' + 10 : (h >= 'A' && h <= 'F') ? h - 'A' + 10 : (p->err = 1, 0);
                }
                /* encode cp as utf-8 (no surrogate pairing needed for our data) */
                if (cp < 0x80) buf[n++] = (char)cp;
                else if (cp < 0x800) { buf[n++] = (char)(0xC0 | (cp >> 6)); buf[n++] = (char)(0x80 | (cp & 63)); }
                else { buf[n++] = (char)(0xE0 | (cp >> 12)); buf[n++] = (char)(0x80 | ((cp >> 6) & 63)); buf[n++] = (char)(0x80 | (cp & 63)); }
                continue;
            }
            default: p->err = 1; free(buf); return NULL;
            }
        }
        buf[n++] = (char)c;
    }
    p->err = 1; free(buf); return NULL;
}

static JVal *new_val(P *p, JType t)
{
    JVal *v = amalloc(p->j, sizeof(JVal));
    if (!v) { p->err = 1; return NULL; }
    v->t = t;
    return v;
}

static JVal *parse_arr(P *p)
{
    JVal *v = new_val(p, J_ARR);
    p->s++; /* [ */
    size_t cap = 8;
    v->v.arr.items = amalloc(p->j, cap * sizeof(JVal *));
    v->v.arr.n = 0;
    ws(p);
    if (peek(p) == ']') { p->s++; return v; }
    for (;;) {
        ws(p);
        JVal *item = parse_val(p);
        if (p->err) return NULL;
        if (v->v.arr.n == cap) {
            cap *= 2;
            JVal **ni = amalloc(p->j, cap * sizeof(JVal *));
            memcpy(ni, v->v.arr.items, v->v.arr.n * sizeof(JVal *));
            v->v.arr.items = ni;
        }
        v->v.arr.items[v->v.arr.n++] = item;
        ws(p);
        int c = peek(p);
        if (c == ',') { p->s++; continue; }
        if (c == ']') { p->s++; return v; }
        p->err = 1; return NULL;
    }
}

static JVal *parse_obj(P *p)
{
    JVal *v = new_val(p, J_OBJ);
    p->s++; /* { */
    size_t cap = 8;
    v->v.obj.pairs = amalloc(p->j, cap * sizeof(JPair));
    v->v.obj.n = 0;
    ws(p);
    if (peek(p) == '}') { p->s++; return v; }
    for (;;) {
        ws(p);
        char *key = parse_str(p);
        if (p->err) return NULL;
        ws(p);
        if (peek(p) != ':') { p->err = 1; return NULL; }
        p->s++;
        ws(p);
        JVal *val = parse_val(p);
        if (p->err) return NULL;
        if (v->v.obj.n == cap) {
            cap *= 2;
            JPair *np = amalloc(p->j, cap * sizeof(JPair));
            memcpy(np, v->v.obj.pairs, v->v.obj.n * sizeof(JPair));
            v->v.obj.pairs = np;
        }
        v->v.obj.pairs[v->v.obj.n].key = key;
        v->v.obj.pairs[v->v.obj.n].val = val;
        v->v.obj.n++;
        ws(p);
        int c = peek(p);
        if (c == ',') { p->s++; continue; }
        if (c == '}') { p->s++; return v; }
        p->err = 1; return NULL;
    }
}

static JVal *parse_val(P *p)
{
    ws(p);
    int c = peek(p);
    if (c == '{') return parse_obj(p);
    if (c == '[') return parse_arr(p);
    if (c == '"') {
        char *s = parse_str(p);
        if (p->err) return NULL;
        JVal *v = new_val(p, J_STR); v->v.str = s; return v;
    }
    if (c == 't' && p->end - p->s >= 4 && !memcmp(p->s, "true", 4)) {
        p->s += 4; JVal *v = new_val(p, J_BOOL); v->v.b = 1; return v;
    }
    if (c == 'f' && p->end - p->s >= 5 && !memcmp(p->s, "false", 5)) {
        p->s += 5; JVal *v = new_val(p, J_BOOL); return v;
    }
    if (c == 'n' && p->end - p->s >= 4 && !memcmp(p->s, "null", 4)) {
        p->s += 4; return new_val(p, J_NULL);
    }
    if (c == '-' || isdigit(c)) {
        char *e;
        double d = strtod(p->s, &e);
        if (e == p->s) { p->err = 1; return NULL; }
        p->s = e;
        JVal *v = new_val(p, J_NUM); v->v.num = d; return v;
    }
    p->err = 1;
    return NULL;
}

Q35Json *q35_json_parse(const char *text, size_t len)
{
    Q35Json *j = calloc(1, sizeof(Q35Json));
    P p = { text, text + len, j, 0 };
    j->root = parse_val(&p);
    ws(&p);
    if (p.err || p.s != p.end || !j->root) {
        q35_json_free(j);
        return NULL;
    }
    return j;
}

void q35_json_free(Q35Json *j)
{
    if (!j) return;
    for (size_t i = 0; i < j->arena.n; i++) free(j->arena.ptrs[i]);
    free(j->arena.ptrs);
    free(j);
}

JVal *q35_json_root(Q35Json *j) { return j->root; }

const JVal *q35_obj_get(const JVal *obj, const char *key)
{
    if (!obj || obj->t != J_OBJ) return NULL;
    for (size_t i = 0; i < obj->v.obj.n; i++)
        if (!strcmp(obj->v.obj.pairs[i].key, key)) return obj->v.obj.pairs[i].val;
    return NULL;
}

double q35_jnum(const JVal *obj, const char *key, int *present)
{
    const JVal *v = q35_obj_get(obj, key);
    if (!v || v->t != J_NUM) { if (present) *present = 0; return 0; }
    if (present) *present = 1;
    return v->v.num;
}

const char *q35_jstr(const JVal *obj, const char *key)
{
    const JVal *v = q35_obj_get(obj, key);
    return (v && v->t == J_STR) ? v->v.str : NULL;
}

int q35_jint_present(const JVal *obj, const char *key, long *out)
{
    int present;
    double d = q35_jnum(obj, key, &present);
    if (!present) return 0;
    *out = (long)d;
    return 1;
}
