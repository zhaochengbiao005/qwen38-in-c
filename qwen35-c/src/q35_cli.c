/* q35 CLI 入口。Subcommands:
 *   q35 tokenize --tokenizer <bin> [--prompt <text> | --stdin] [--ids-only]
 *
 * Exit codes: 0 ok; 2 usage/arg error; 3 tokenizer load failure; 4 encode error.
 */
#include "q35/q35_bpe.h"
#include "../src/q35_plat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_stdin_all(char **out, size_t *out_len)
{
    size_t cap = 1 << 16, n = 0;
    char *buf = malloc(cap);
    if (!buf) return 0;
    for (;;) {
        if (n + 4096 + 1 > cap) { cap *= 2; buf = realloc(buf, cap); if (!buf) return 0; }
        size_t r = fread(buf + n, 1, 4096, stdin);
        n += r;
        if (r < 4096) break;
    }
    buf[n] = 0;
    *out = buf; *out_len = n;
    return 1;
}

static int cmd_tokenize(int argc, char **argv)
{
    const char *tok_path = NULL, *prompt = NULL;
    int use_stdin = 0, ids_only = 0;
    const char *prompt_file = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--stdin")) use_stdin = 1;
        else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) prompt_file = argv[++i];
        else if (!strcmp(argv[i], "--ids-only")) ids_only = 1;
        else {
            fprintf(stderr, "tokenize: unknown/incomplete arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (!tok_path || (!prompt && !use_stdin && !prompt_file)) {
        fprintf(stderr, "usage: q35 tokenize --tokenizer <bin> (--prompt <text> | --prompt-file <path> | --stdin) [--ids-only]\n");
        return 2;
    }

    Q35TokErr err;
    Q35Tok *tok = q35_tok_load(tok_path, &err);
    if (!tok) {
        fprintf(stderr, "tokenize: failed to load %s: %s\n", tok_path, q35_tok_err_str(err));
        return 3;
    }

    char *stdin_buf = NULL;
    if (prompt_file) {
        size_t n;
        stdin_buf = q35_plat_read_file(prompt_file, &n);
        if (!stdin_buf) { fprintf(stderr, "tokenize: cannot read %s\n", prompt_file); return 4; }
        prompt = stdin_buf;
    }
    if (use_stdin) {
        size_t n;
        if (!read_stdin_all(&stdin_buf, &n)) { fprintf(stderr, "tokenize: stdin read failed\n"); return 4; }
        prompt = stdin_buf;
    }

    Q35Bpe *b = q35_bpe_new(tok);
    if (!b) { fprintf(stderr, "tokenize: encoder init failed\n"); return 4; }

    size_t cap = strlen(prompt) + 64;
    uint32_t *ids = malloc(cap * sizeof(uint32_t));
    if (!ids) { fprintf(stderr, "tokenize: oom\n"); return 4; }
    int n = q35_bpe_encode(b, prompt, ids, cap);
    if (n < 0) { fprintf(stderr, "tokenize: encode failed (invalid UTF-8?)\n"); return 4; }

    for (int i = 0; i < n; i++) printf("%u%s", ids[i], i + 1 == n ? "\n" : " ");

    if (!ids_only) {
        size_t dlen = 0;
        for (int i = 0; i < n; i++) {
            uint32_t l; q35_tok_bytes(tok, ids[i], &l); dlen += l;
        }
        uint8_t *text = malloc(dlen + 1);
        if (text) {
            size_t got = q35_bpe_decode(tok, ids, (size_t)n, text, dlen);
            if (got != (size_t)-1) {
                text[got] = 0;
                printf("%s\n", (const char *)text);
            }
            free(text);
        }
    }
    free(ids);
    q35_bpe_free(b);
    q35_tok_free(tok);
    free(stdin_buf);
    return 0;
}

int q35_run_cmd(int argc, char **argv);
int q35_serve_cmd(int argc, char **argv);
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
            "q35 - Qwen3.5 C engine\n"
            "subcommands:\n"
            "  tokenize --tokenizer <bin> (--prompt <text>|--stdin) [--ids-only]\n"
            "  run --model <dir> --prompt <text> [--temperature T] [--top-k K] [--top-p P]\n"
            "  serve --model <dir> [--port N] [--tokenizer <bin>]\n");
        return 2;
    }
    if (!strcmp(argv[1], "tokenize")) return cmd_tokenize(argc, argv);
    if (!strcmp(argv[1], "run")) return q35_run_cmd(argc, argv);
    if (!strcmp(argv[1], "serve")) return q35_serve_cmd(argc, argv);
    fprintf(stderr, "unknown subcommand: %s\n", argv[1]);
    return 2;
}

