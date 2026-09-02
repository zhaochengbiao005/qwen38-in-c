#ifndef Q35_CFG_H
#define Q35_CFG_H

#include <stdint.h>

/* Strictly parsed model config (text subgraph of Qwen3_5). Every field is
   required; unknown/missing/wrongly-typed values are fatal. */

#define Q35_MAX_LAYERS 256
#define Q35_LAYER_LINEAR 0
#define Q35_LAYER_FULL   1

typedef struct {
    int num_hidden_layers;
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    int vocab_size;
    /* linear attention */
    int linear_num_key_heads;
    int linear_key_head_dim;
    int linear_num_value_heads;
    int linear_value_head_dim;
    int linear_conv_kernel_dim;
    /* rope */
    double rope_theta;
    double partial_rotary_factor;
    /* misc */
    double rms_norm_eps;
    int bos_token_id;
    int eos_token_id;
    int max_position_embeddings;
    /* per-layer type map (0=linear, 1=full) */
    int layer_type[Q35_MAX_LAYERS];
} Q35Cfg;

/* 0 ok; on failure sets *errbuf (if given) and returns -1 */
int q35_cfg_load(const char *config_path, Q35Cfg *out, char *errbuf, int errcap);

#endif
