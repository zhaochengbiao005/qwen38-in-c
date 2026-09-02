#ifndef Q35_MODEL_H
#define Q35_MODEL_H

/* Ticket-19: full-model assembly (embed -> N mixed layers -> final RMSNorm
 * -> lm_head -> f32 logits) for Qwen3_5 text subgraph.
 *
 * Weight dtype/binding convention (matches q35_deltanet/q35_attn/q35_mm):
 *   - FP8 e4m3 matmul weights + BF16 128x128 block scales:
 *     bound as zero-copy pointers into the read-only memory mapping of the
 *     shard files (mapping only, pages are not touched at load).
 *   - BF16 small tensors (all RMSNorm weights, conv1d, A_log, dt_bias):
 *     converted once to f32 at load (q35_deltanet/q35_kern take f32).
 *   - linear_attn.in_proj_a / in_proj_b are BF16 without scales in the real
 *     repo -> quantized to FP8 e4m3 + BF16 block scales at load time
 *     (q35_deltanet only accepts FP8 projections).
 *   - embed_tokens / lm_head stay BF16 in the mapping; embedding rows are
 *     expanded to f32 on lookup, lm_head is consumed by q35_mm_bf16.
 *
 * Layer pattern comes from config layer_types (linear_attention /
 * full_attention); real 27B: 64 layers, full attention at 3,7,...,63.
 */

#include <stdint.h>
#include <stddef.h>

typedef struct Q35Model Q35Model;

enum q35_model_err {
    Q35_MODEL_OK = 0,
    Q35_MODEL_ERR_ARG = 1,     /* bad arguments */
    Q35_MODEL_ERR_IO = 2,      /* open/read/mapping failure */
    Q35_MODEL_ERR_CFG = 3,     /* config invalid / arch validation failed */
    Q35_MODEL_ERR_ST = 4,      /* safetensors index failure */
    Q35_MODEL_ERR_MISSING = 5, /* required tensor missing */
    Q35_MODEL_ERR_SHAPE = 6,   /* tensor shape/dtype mismatch */
    Q35_MODEL_ERR_NOMEM = 7,   /* allocation failure */
    Q35_MODEL_ERR_STATE = 8    /* KV cache exhausted / state overflow */
};

/* Load model from a directory containing config.json + *.safetensors[?]
 * shards. kv_cap: max sequence length held in each full-attention KV cache
 * (0 -> min(max_position_embeddings, 65536)). Returns NULL on error, *err set
 * and errbuf filled with a human-readable message. */
Q35Model *q35_model_load(const char *dir, uint32_t kv_cap,
                         char *errbuf, int errcap, int *err);

void q35_model_free(Q35Model *m);

/* Reset all recurrent state (DeltaNet S/conv, KV caches, position). */
void q35_model_reset(Q35Model *m);

/* Save/load recurrent state (DeltaNet S/conv, KV cache contents, position)
 * to/from a binary file. Enables multi-turn conversation persistence across
 * process restarts. Returns Q35_MODEL_OK or an error code. */
int q35_model_save_state(const Q35Model *m, const char *path);
int q35_model_load_state(Q35Model *m, const char *path);

/* Run n tokens as prefill serially (state updated); logits for the LAST
 * token written to logits[vocab] f32. q35_forward_decode is one token step;
 * both return Q35_MODEL_OK or an error code. */
int q35_forward_prefill(Q35Model *m, const int32_t *tokens, size_t n,
                        float *logits);
int q35_forward_decode(Q35Model *m, int32_t token, float *logits);

uint32_t q35_model_vocab(const Q35Model *m);
uint32_t q35_model_hidden(const Q35Model *m);
uint32_t q35_model_pos(const Q35Model *m);     /* tokens forwarded so far */
uint32_t q35_model_layers(const Q35Model *m);

const char *q35_model_err_str(int e);

#endif
