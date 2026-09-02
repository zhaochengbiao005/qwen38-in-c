#include "q35/q35_st.h"
#include "q35/q35_json.h"
#include "q35_plat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Q35St {
    char **paths;
    FILE **fps;
    int nfiles;
    Q35Tensor *tensors;
    uint32_t ntensors, cap;
    int sorted;
};


size_t q35_dtype_size(Q35Dtype t)
{
    switch (t) {
    case Q35_DTYPE_F8_E4M3: return 1;
    case Q35_DTYPE_BF16: return 2;
    case Q35_DTYPE_F16: return 2;
    case Q35_DTYPE_F32: return 4;
    case Q35_DTYPE_I64: return 8;
    default: return 0;
    }
}

const char *q35_dtype_name(Q35Dtype t)
{
    switch (t) {
    case Q35_DTYPE_F8_E4M3: return "F8_E4M3";
    case Q35_DTYPE_BF16: return "BF16";
    case Q35_DTYPE_F16: return "F16";
    case Q35_DTYPE_F32: return "F32";
    case Q35_DTYPE_I64: return "I64";
    default: return "UNKNOWN";
    }
}

const char *q35_st_err_str(Q35StErr e)
{
    switch (e) {
    case Q35_ST_OK: return "ok";
    case Q35_ST_ERR_OPEN: return "cannot open";
    case Q35_ST_ERR_HEADER: return "bad safetensors header";
    case Q35_ST_ERR_DTYPE: return "unsupported dtype";
    case Q35_ST_ERR_RANGE: return "tensor range out of file";
    case Q35_ST_ERR_IO: return "io error";
    }
    return "unknown";
}

static Q35Dtype dtype_of(const char *s)
{
    if (!strcmp(s, "F8_E4M3")) return Q35_DTYPE_F8_E4M3;
    if (!strcmp(s, "BF16")) return Q35_DTYPE_BF16;
    if (!strcmp(s, "F16")) return Q35_DTYPE_F16;
    if (!strcmp(s, "F32")) return Q35_DTYPE_F32;
    if (!strcmp(s, "I64")) return Q35_DTYPE_I64;
    return Q35_DTYPE_UNKNOWN;
}

int q35_st_is_shard_name(const char *name)
{
    return strstr(name, "safetensors") != NULL &&
           strstr(name, "json") == NULL &&
           strstr(name, ".safetensors") != NULL;
}

static int push_tensor(Q35St *st)
{
    if (st->ntensors == st->cap) {
        uint32_t newcap = st->cap ? st->cap * 2 : 256;
        Q35Tensor *nt = realloc(st->tensors, (size_t)newcap * sizeof(Q35Tensor));
        if (!nt) return 0;
        st->tensors = nt;
        st->cap = newcap;
    }
    st->ntensors++;
    return 1;
}

static Q35StErr index_file(Q35St *st, const char *path, int file_idx)
{
    FILE *f = fopen(path, "rb");
    if (!f) return Q35_ST_ERR_OPEN;
    uint8_t hbuf[8];
    if (fread(hbuf, 1, 8, f) != 8) { fclose(f); return Q35_ST_ERR_HEADER; }
    uint64_t hlen = 0;
    for (int i = 0; i < 8; i++) hlen |= (uint64_t)hbuf[i] << (8 * i);
    if (hlen == 0 || hlen > (1u << 24)) { fclose(f); return Q35_ST_ERR_HEADER; }

    /* file size */
    if (q35_plat_fseek64(f, 0, SEEK_END) != 0) { fclose(f); return Q35_ST_ERR_IO; }
    int64_t fsize = q35_plat_ftell64(f);
    if (fsize < 0) { fclose(f); return Q35_ST_ERR_IO; }
    if (q35_plat_fseek64(f, 8, SEEK_SET) != 0) { fclose(f); return Q35_ST_ERR_IO; }

    char *hjson = malloc((size_t)hlen + 1);
    if (!hjson) { fclose(f); return Q35_ST_ERR_IO; }
    if (fread(hjson, 1, (size_t)hlen, f) != hlen) { free(hjson); fclose(f); return Q35_ST_ERR_HEADER; }
    hjson[hlen] = 0;
    fclose(f);

    Q35Json *j = q35_json_parse(hjson, (size_t)hlen);
    free(hjson);
    if (!j) return Q35_ST_ERR_HEADER;

    JVal *root = q35_json_root(j);
    uint64_t data_base = 8 + hlen;
    Q35StErr out = Q35_ST_OK;
    for (size_t pi = 0; pi < root->v.obj.n; pi++) {
        const char *name = root->v.obj.pairs[pi].key;
        const JVal *tv = root->v.obj.pairs[pi].val;
        if (!strcmp(name, "__metadata__") || tv->t != J_OBJ) continue;

        const char *dt = q35_jstr(tv, "dtype");
        Q35Dtype d = dtype_of(dt);
        if (d == Q35_DTYPE_UNKNOWN) { out = Q35_ST_ERR_DTYPE; break; }

        const JVal *shape = q35_obj_get(tv, "shape");
        const JVal *offs = q35_obj_get(tv, "data_offsets");
        if (!shape || shape->t != J_ARR || !offs || offs->t != J_ARR ||
            offs->v.arr.n != 2) { out = Q35_ST_ERR_HEADER; break; }
        if (shape->v.arr.n > 8) { out = Q35_ST_ERR_HEADER; break; }
        uint64_t o0 = (uint64_t)offs->v.arr.items[0]->v.num;
        uint64_t o1 = (uint64_t)offs->v.arr.items[1]->v.num;
        if (o1 < o0 || data_base + o1 > (uint64_t)fsize) { out = Q35_ST_ERR_RANGE; break; }

        uint64_t elems = 1;
        for (size_t k = 0; k < shape->v.arr.n; k++)
            elems *= (uint64_t)(shape->v.arr.items[k]->v.num);
        if (elems * q35_dtype_size(d) != o1 - o0) { out = Q35_ST_ERR_RANGE; break; }

        if (!push_tensor(st)) { out = Q35_ST_ERR_IO; break; }
        Q35Tensor *t = &st->tensors[st->ntensors - 1];
        t->name = q35_plat_strdup(name);
        t->dtype = d;
        t->ndim = (uint32_t)shape->v.arr.n;
        for (uint32_t k = 0; k < 8; k++)
            t->shape[k] = k < shape->v.arr.n ? (uint32_t)shape->v.arr.items[k]->v.num : 0;
        t->file_idx = file_idx;
        t->data_off = data_base + o0;
        t->data_len = o1 - o0;
    }
    q35_json_free(j);
    return out;
}

static Q35St *st_alloc(void)
{
    Q35St *st = calloc(1, sizeof(Q35St));
    return st;
}

Q35St *q35_st_open_file(const char *path, Q35StErr *err)
{
    Q35St *st = st_alloc();
    if (!st) { if (err) *err = Q35_ST_ERR_IO; return NULL; }
    st->paths = malloc(sizeof(char *));
    st->fps = calloc(1, sizeof(FILE *));
    if (!st->paths || !st->fps) { q35_st_close(st); if (err) *err = Q35_ST_ERR_IO; return NULL; }
    st->paths[0] = q35_plat_strdup(path);
    st->nfiles = 1;
    Q35StErr e = index_file(st, path, 0);
    if (e != Q35_ST_OK) { q35_st_close(st); if (err) *err = e; return NULL; }
    if (err) *err = Q35_ST_OK;
    return st;
}


Q35St *q35_st_open_dir(const char *dir, Q35StErr *err)
{
    Q35St *st = st_alloc();
    if (!st) { if (err) *err = Q35_ST_ERR_IO; return NULL; }

    Q35PlatDir *pd = q35_plat_dir_open(dir);
    if (!pd) { q35_st_close(st); if (err) *err = Q35_ST_ERR_OPEN; return NULL; }
    const char *n;
    while ((n = q35_plat_dir_next(pd)) != NULL) {
        if (q35_st_is_shard_name(n)) {
            st->paths = realloc(st->paths, (size_t)(st->nfiles + 1) * sizeof(char *));
            st->fps = realloc(st->fps, (size_t)(st->nfiles + 1) * sizeof(FILE *));
            if (!st->paths || !st->fps) { q35_plat_dir_close(pd); q35_st_close(st); if (err) *err = Q35_ST_ERR_IO; return NULL; }
            st->fps[st->nfiles] = NULL;
            char full[4096];
            q35_plat_path_join(full, sizeof(full), dir, n);
            st->paths[st->nfiles] = q35_plat_strdup(full);
            Q35StErr e = index_file(st, full, st->nfiles);
            if (e != Q35_ST_OK) {
                q35_plat_dir_close(pd);
                q35_st_close(st);
                if (err) *err = e;
                return NULL;
            }
            st->nfiles++;
        }
    }
    q35_plat_dir_close(pd);
    if (st->nfiles == 0) { q35_st_close(st); if (err) *err = Q35_ST_ERR_OPEN; return NULL; }
    if (err) *err = Q35_ST_OK;
    return st;
}

void q35_st_close(Q35St *s)
{
    if (!s) return;
    for (int i = 0; i < s->nfiles; i++) {
        if (s->paths) free(s->paths[i]);
        if (s->fps && s->fps[i]) fclose(s->fps[i]);
    }
    free(s->paths);
    free(s->fps);
    for (uint32_t i = 0; i < s->ntensors; i++) free(s->tensors[i].name);
    free(s->tensors);
    free(s);
}

static int cmp_tensor(const void *a, const void *b)
{
    return strcmp(((const Q35Tensor *)a)->name, ((const Q35Tensor *)b)->name);
}

static void sort_index(Q35St *s)
{
    qsort(s->tensors, s->ntensors, sizeof(Q35Tensor), cmp_tensor);
}

const Q35Tensor *q35_st_find(const Q35St *s, const char *name)
{
    Q35St *mut = (Q35St *)s;
    if (!mut->sorted) { sort_index(mut); mut->sorted = 1; }
    Q35Tensor key = { 0 };
    char nkbuf[512];
    snprintf(nkbuf, sizeof(nkbuf), "%s", name);
    key.name = nkbuf;
    return bsearch(&key, s->tensors, s->ntensors, sizeof(Q35Tensor), cmp_tensor);
}

uint32_t q35_st_tensor_count(const Q35St *s) { return s->ntensors; }
const Q35Tensor *q35_st_tensor_at(const Q35St *s, uint32_t i) { return &s->tensors[i]; }

int q35_st_read(const Q35St *s, const Q35Tensor *t, void *dst)
{
    Q35St *mut = (Q35St *)s;
    if (t->file_idx < 0 || t->file_idx >= mut->nfiles) return 0;
    FILE *f = mut->fps[t->file_idx];
    if (!f) {
        f = fopen(mut->paths[t->file_idx], "rb");
        if (!f) return 0;
        mut->fps[t->file_idx] = f;
    }
    if (q35_plat_fseek64(f, (int64_t)t->data_off, SEEK_SET) != 0) return 0;
    return fread(dst, 1, (size_t)t->data_len, f) == t->data_len;
}





