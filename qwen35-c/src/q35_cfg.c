#include "q35/q35_cfg.h"
#include "q35/q35_json.h"
#include "q35_plat.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int need_int(const JVal *o, const char *key, long *out, char *e, int ec)
{
    if (!q35_jint_present(o, key, out)) {
        snprintf(e, (size_t)ec, "config missing/invalid: %s", key);
        return 0;
    }
    return 1;
}

static int need_num(const JVal *o, const char *key, double *out, char *e, int ec)
{
    int p;
    double d = q35_jnum(o, key, &p);
    if (!p) { snprintf(e, (size_t)ec, "config missing/invalid: %s", key); return 0; }
    *out = d;
    return 1;
}

int q35_cfg_load(const char *path, Q35Cfg *out, char *errbuf, int errcap)
{
    char eb[256] = { 0 };
    char *e = errbuf ? errbuf : eb;
    int ec = errbuf ? errcap : (int)sizeof(eb);

    size_t len;
    char *text = q35_plat_read_file(path, &len);
    if (!text) { snprintf(e, (size_t)ec, "cannot read %s", path); return -1; }
    Q35Json *j = q35_json_parse(text, len);
    free(text);
    if (!j) { snprintf(e, (size_t)ec, "config.json parse error"); return -1; }

    JVal *root = q35_json_root(j);
    const JVal *tc = q35_obj_get(root, "text_config");
    if (!tc || tc->t != J_OBJ) tc = root;

#define REQ_INT(field, key) \
    { long v; if (!need_int(tc, key, &v, e, ec)) { q35_json_free(j); return -1; } out->field = (int)v; }
#define REQ_NUM(field, key) \
    { double v; if (!need_num(tc, key, &v, e, ec)) { q35_json_free(j); return -1; } out->field = v; }

    memset(out, 0, sizeof(*out));
    REQ_INT(num_hidden_layers, "num_hidden_layers");
    REQ_INT(hidden_size, "hidden_size");
    REQ_INT(intermediate_size, "intermediate_size");
    REQ_INT(num_attention_heads, "num_attention_heads");
    REQ_INT(num_key_value_heads, "num_key_value_heads");
    REQ_INT(head_dim, "head_dim");
    REQ_INT(vocab_size, "vocab_size");
    REQ_INT(linear_num_key_heads, "linear_num_key_heads");
    REQ_INT(linear_key_head_dim, "linear_key_head_dim");
    REQ_INT(linear_num_value_heads, "linear_num_value_heads");
    REQ_INT(linear_value_head_dim, "linear_value_head_dim");
    REQ_INT(linear_conv_kernel_dim, "linear_conv_kernel_dim");
    REQ_INT(bos_token_id, "bos_token_id");
    REQ_INT(eos_token_id, "eos_token_id");
    REQ_INT(max_position_embeddings, "max_position_embeddings");
    if (!need_num(tc, "rope_theta", &out->rope_theta, e, ec)) {
        const JVal *rp = q35_obj_get(tc, "rope_parameters");
        const JVal *rv = (rp && rp->t == J_OBJ) ? q35_obj_get(rp, "rope_theta") : NULL;
        if (!rv || rv->t != J_NUM) { snprintf(e, (size_t)ec, "config missing/invalid: rope_theta"); q35_json_free(j); return -1; }
        out->rope_theta = rv->v.num;
    }
    REQ_NUM(partial_rotary_factor, "partial_rotary_factor");
    REQ_NUM(rms_norm_eps, "rms_norm_eps");
#undef REQ_INT
#undef REQ_NUM

    if (out->num_hidden_layers <= 0 || out->num_hidden_layers > Q35_MAX_LAYERS) {
        snprintf(e, (size_t)ec, "num_hidden_layers out of range");
        q35_json_free(j); return -1;
    }

    /* layer_types: "linear_attention" | "full_attention" */
    const JVal *lt = q35_obj_get(tc, "layer_types");
    if (!lt || lt->t != J_ARR || (int)lt->v.arr.n != out->num_hidden_layers) {
        snprintf(e, (size_t)ec, "layer_types missing or wrong length");
        q35_json_free(j); return -1;
    }
    for (int i = 0; i < out->num_hidden_layers; i++) {
        const JVal *it = lt->v.arr.items[i];
        if (it->t != J_STR) { snprintf(e, (size_t)ec, "layer_types[%d] not string", i); q35_json_free(j); return -1; }
        if (!strcmp(it->v.str, "linear_attention")) out->layer_type[i] = Q35_LAYER_LINEAR;
        else if (!strcmp(it->v.str, "full_attention")) out->layer_type[i] = Q35_LAYER_FULL;
        else { snprintf(e, (size_t)ec, "unknown layer_types[%d]=%s", i, it->v.str); q35_json_free(j); return -1; }
    }

    /* sanity relations — refuse architecture drift early */
    if (out->head_dim * out->num_attention_heads <= 0 ||
        out->linear_key_head_dim * out->linear_num_key_heads <= 0 ||
        out->linear_value_head_dim * out->linear_num_value_heads <= 0) {
        snprintf(e, (size_t)ec, "head dims invalid");
        q35_json_free(j); return -1;
    }

    q35_json_free(j);
    return 0;
}
