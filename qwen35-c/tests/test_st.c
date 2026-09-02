#include "q35/q35_st.h"
#include "q35/q35_cfg.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

static uint32_t crc32_buf(const uint8_t *p, size_t n)
{
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++)
            c = (c >> 1) ^ (0xEDB88320u & (uint32_t)-(int)(c & 1));
    }
    return ~c;
}

int main(int argc, char **argv)
{
    const char *fx = argc > 1 ? argv[1] : "tests/fixtures/st";
    const char *real = argc > 2 ? argv[2] : NULL;

    /* --- synthetic file --- */
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/mini.safetensors", fx);
        Q35StErr err;
        Q35St *st = q35_st_open_file(path, &err);
        if (!st) { printf("NOT RUN: synthetic shard missing\n"); return 77; }
        CHECK(q35_st_tensor_count(st) == 4);

        const Q35Tensor *t = q35_st_find(st, "model.language_model.layers.0.mlp.gate_proj.weight");
        CHECK(t != NULL);
        CHECK(t && t->dtype == Q35_DTYPE_F8_E4M3);
        CHECK(t && t->ndim == 2 && t->shape[0] == 256 && t->shape[1] == 128);
        if (t) {
            uint8_t *buf = malloc((size_t)t->data_len);
            CHECK(q35_st_read(st, t, buf));
            CHECK(crc32_buf(buf, (size_t)t->data_len) == 0x1b43eabdu);
            free(buf);
        }
        t = q35_st_find(st, "model.language_model.norm.weight");
        CHECK(t && t->dtype == Q35_DTYPE_F32 && t->ndim == 1 && t->shape[0] == 16);
        if (t) {
            uint8_t *buf = malloc((size_t)t->data_len);
            CHECK(q35_st_read(st, t, buf));
            CHECK(crc32_buf(buf, (size_t)t->data_len) == 0x9855c0d4u);
            free(buf);
        }
        CHECK(q35_st_find(st, "does.not.exist") == NULL);
        q35_st_close(st);
    }

    /* --- synthetic config --- */
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/config.json", fx);
        Q35Cfg cfg;
        char err[256];
        CHECK(q35_cfg_load(path, &cfg, err, sizeof(err)) == 0);
        CHECK(cfg.num_hidden_layers == 4);
        CHECK(cfg.hidden_size == 128);
        CHECK(cfg.rope_theta > 9.0e6 && cfg.rope_theta < 1.1e7); /* nested read works */
        CHECK(cfg.partial_rotary_factor > 0.24 && cfg.partial_rotary_factor < 0.26);
        CHECK(cfg.layer_type[0] == Q35_LAYER_LINEAR);
        CHECK(cfg.layer_type[3] == Q35_LAYER_FULL);
        CHECK(cfg.vocab_size == 512);
    }

    /* --- negative: truncated header --- */
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/mini.safetensors", fx);
        FILE *rd = fopen(path, "rb");
        uint8_t head[64];
        size_t got = fread(head, 1, 64, rd);
        fclose(rd);
        snprintf(path, sizeof(path), "%s/trunc.safetensors", fx);
        FILE *wr = fopen(path, "wb");
        fwrite(head, 1, got, wr);
        fclose(wr);
        Q35StErr err;
        Q35St *st = q35_st_open_file(path, &err);
        CHECK(st == NULL);
    }

    /* --- real checkpoint dir (skip if absent) --- */
    if (real) {
        Q35StErr err;
        Q35St *st = q35_st_open_dir(real, &err);
        if (!st) {
            printf("NOT RUN: real dir absent (%s)\n", q35_st_err_str(err));
        } else {
            char path[1024];
            snprintf(path, sizeof(path), "%s/config.json", real);
            Q35Cfg cfg;
            char cErr[256];
            int okc = q35_cfg_load(path, &cfg, cErr, sizeof(cErr));
            CHECK(okc == 0);
            if (okc == 0) {
                CHECK(cfg.num_hidden_layers == 64);
                CHECK(cfg.hidden_size == 5120);
                CHECK(cfg.vocab_size == 248320);
                int full = 0, lin = 0;
                for (int i = 0; i < cfg.num_hidden_layers; i++)
                    cfg.layer_type[i] == Q35_LAYER_FULL ? full++ : lin++;
                CHECK(full == 16 && lin == 48);
            }
            const Q35Tensor *w = q35_st_find(st, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight");
            CHECK(w && w->dtype == Q35_DTYPE_F8_E4M3);
            if (w) {
                /* shape [rows, hidden] FP8: rows = qkv+kv dims; check scale exists */
                char sc[512];
                snprintf(sc, sizeof(sc), "%s%s",
                    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv", "");
                const Q35Tensor *s = q35_st_find(st, sc);
                CHECK(s && s->dtype == Q35_DTYPE_BF16);
                if (s) {
                    CHECK((uint32_t)((w->shape[0] + 127) / 128) == s->shape[0]);
                    CHECK((uint32_t)((w->shape[1] + 127) / 128) == s->shape[1]);
                }
            }
            printf("real: %u tensors, cfg 64 layers ok\n", q35_st_tensor_count(st));
            q35_st_close(st);
        }
    }

    if (failures) { printf("FAILURES: %d\n", failures); return 1; }
    printf("PASS\n");
    return 0;
}
