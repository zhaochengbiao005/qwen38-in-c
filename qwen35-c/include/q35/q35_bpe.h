#ifndef Q35_BPE_H
#define Q35_BPE_H

#include "q35_tok.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque encoder: pretokenizer splitter (Qwen pattern) + NFC + merge tables. */
typedef struct Q35Bpe Q35Bpe;

Q35Bpe *q35_bpe_new(const Q35Tok *tok);            /* NULL on OOM */
void    q35_bpe_free(Q35Bpe *b);

/* Encode UTF-8 text to token ids. Special tokens in text are recognized
   (longest-match). Returns token count (>0), 0 for empty input, -1 on error
   (e.g. invalid UTF-8). out capacity must be >= cap. */
int q35_bpe_encode(Q35Bpe *b, const char *text, uint32_t *out, size_t cap);

/* Decode ids back to bytes (caller-provided buffer). Returns byte count. */
size_t q35_bpe_decode(const Q35Tok *tok, const uint32_t *ids, size_t n,
                      uint8_t *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif
