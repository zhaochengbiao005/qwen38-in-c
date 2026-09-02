#ifndef Q35_TOK_H
#define Q35_TOK_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  define Q35_API __declspec(dllexport)
#else
#  define Q35_API
#endif

#define Q35_TOK_MAGIC "Q35TOK10"
#define Q35_TOK_VERSION 1u
#define Q35_TOK_FLAG_ISOLATED_SPLIT 1u
#define Q35_TOK_FLAG_NFC_NORMALIZE  2u

typedef struct Q35Tok Q35Tok;

typedef enum {
    Q35_TOK_OK = 0,
    Q35_TOK_ERR_OPEN,        /* cannot open/read file */
    Q35_TOK_ERR_MAGIC,       /* bad magic bytes */
    Q35_TOK_ERR_VERSION,     /* unsupported blob version */
    Q35_TOK_ERR_TRUNCATED,   /* ran past end of blob */
    Q35_TOK_ERR_SECTION,     /* malformed section / bad length */
    Q35_TOK_ERR_INTEGRITY,   /* content failed self-consistency checks */
    Q35_TOK_ERR_NOMEM        /* allocation failed */
} Q35TokErr;

/* Load tokenizer.bin; returns NULL on error (err filled). Not thread-safe to free twice. */
Q35_API Q35Tok *q35_tok_load(const char *path, Q35TokErr *err);
Q35_API void    q35_tok_free(Q35Tok *t);

/* Integrity-checked metadata */
Q35_API uint32_t q35_tok_declared_vocab(const Q35Tok *t); /* 248320 */
Q35_API uint32_t q35_tok_defined_vocab(const Q35Tok *t);  /* 248077 */
Q35_API uint32_t q35_tok_merge_count(const Q35Tok *t);    /* 247587 */
Q35_API uint32_t q35_tok_flags(const Q35Tok *t);
Q35_API uint32_t q35_tok_bos(const Q35Tok *t);        /* 248044 */
Q35_API uint32_t q35_tok_eos(const Q35Tok *t);        /* 248044 */
Q35_API uint32_t q35_tok_pad(const Q35Tok *t);        /* 248044 */
Q35_API uint32_t q35_tok_chat_eos(const Q35Tok *t);   /* 248046 <|im_end|> */
Q35_API const char *q35_tok_pretok_pattern(const Q35Tok *t); /* utf-8, NUL-terminated */

/* id -> bytes: NULL if id >= defined vocab. */
Q35_API const uint8_t *q35_tok_bytes(const Q35Tok *t, uint32_t id, uint32_t *len);
/* bytes -> id: -1 if not a token. */
Q35_API int32_t q35_tok_lookup(const Q35Tok *t, const uint8_t *bytes, uint32_t len);

/* Special tokens: index 0..count-1 (added-token table order). */
Q35_API uint32_t q35_tok_added_count(const Q35Tok *t);
Q35_API int q35_tok_added_get(const Q35Tok *t, uint32_t index,
                              uint32_t *id, const uint8_t **bytes, uint32_t *len,
                              int *is_special);

/* merge table access (encode path): index in [0, merge_count) -> pair ids. */
Q35_API int q35_tok_merge(const Q35Tok *t, uint32_t index, uint32_t *a, uint32_t *b);

/* Approximate resident memory of the loaded tokenizer, bytes. */
Q35_API size_t q35_tok_memory_bytes(const Q35Tok *t);

Q35_API const char *q35_tok_err_str(Q35TokErr e);

#endif


