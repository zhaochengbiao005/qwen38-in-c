/* q35 serve: minimal HTTP/1.1 API server around the inference engine.
 *
 * Endpoints:
 *   GET  /health           -> {"status":"ok","model":...,"pos":N}
 *   POST /v1/completions   -> {"prompt":"...", "max_tokens":N, "temperature":T,
 *                               "top_k":K, "top_p":P, "stream":bool,
 *                               "reset":bool, "chat":bool}
 *
 * Non-stream: full JSON response {"text":"...","tokens":N,"elapsed":S}
 * Stream:    chunked transfer, one JSON line per token:
 *            {"text":"..."}\n , terminated by {"done":true}\n
 *
 * One request at a time (model state is a single conversation). The server
 * is single-threaded: accept -> read request -> run inference -> respond.
 * Streaming decode is mutually exclusive with other requests by design.
 *
 * Exit codes: 0 ok; 2 usage; 3 tokenizer; 5 config; 6 model load; 8 socket.
 */
#include "q35/q35_tok.h"
#include "q35/q35_bpe.h"
#include "q35/q35_model.h"
#include "q35/q35_cfg.h"
#include "q35/q35_json.h"
#include "q35_plat.h"
#include "cli_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- minimal HTTP helpers ---------- */

typedef struct {
    int fd;
    char method[8];
    char path[256];
    char *body;     /* malloc'd request body, NULL if none */
    size_t body_len;
} HttpReq;

static void http_close(HttpReq *r)
{
    q35_plat_sock_close(r->fd);
    r->fd = -1;
}

static int send_all(int fd, const char *buf, size_t len)
{
    size_t off = 0;
    while (off < len) {
        int n = q35_plat_sock_send(fd, buf + off, (int)(len - off));
        if (n <= 0) return 0;
        off += (size_t)n;
    }
    return 1;
}

static void http_respond(int fd, int code, const char *ctype,
                         const char *body, size_t body_len)
{
    const char *status = code == 200 ? "OK"
                       : code == 404 ? "Not Found"
                       : code == 400 ? "Bad Request" : "Error";
    char hdr[256];
    int hn = snprintf(hdr, sizeof hdr,
                      "HTTP/1.1 %d %s\r\n"
                      "Content-Type: %s\r\n"
                      "Content-Length: %zu\r\n"
                      "Connection: close\r\n"
                      "\r\n", code, status, ctype, body_len);
    if (hn > 0) send_all(fd, hdr, (size_t)hn);
    if (body && body_len) send_all(fd, body, body_len);
}

/* read request until headers end + Content-Length body; 0=ok, -1=err */
static int http_read_request(int fd, HttpReq *r)
{
    size_t cap = 1 << 16, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return -1;
    /* read until \r\n\r\n */
    char *hdr_end = NULL;
    while (!hdr_end) {
        if (len + 4097 > cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); return -1; }
            buf = nb;
        }
        int n = q35_plat_sock_recv(fd, buf + len, 4096);
        if (n <= 0) { free(buf); return -1; }
        len += (size_t)n;
        buf[len] = 0;
        hdr_end = strstr(buf, "\r\n\r\n");
        if (hdr_end) {
            /* headers complete: keep reading until the Content-Length body
             * has fully arrived (also correct for the zero-byte-body case) */
            char *cl = strstr(buf, "Content-Length:");
            if (!cl) cl = strstr(buf, "content-length:");
            if (cl) {
                size_t want = (size_t)strtoul(cl + 15, NULL, 10);
                size_t hdr_len = (size_t)(hdr_end - buf) + 4;
                if (len < hdr_len + want) hdr_end = NULL;  /* keep reading */
            }
        }
    }
    /* parse method/path */
    if (sscanf(buf, "%7s %255s", r->method, r->path) != 2) { free(buf); return -1; }
    size_t hdr_len = (size_t)(hdr_end - buf) + 4;
    r->body_len = len - hdr_len;
    r->body = (char *)malloc(r->body_len + 1);
    if (!r->body) { free(buf); return -1; }
    memcpy(r->body, buf + hdr_len, r->body_len);
    r->body[r->body_len] = 0;
    free(buf);
    return 0;
}

/* ---------- JSON escaping for response ---------- */

static size_t json_escape(const char *in, char *out, size_t out_cap)
{
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)in; *p && o + 7 < out_cap; p++) {
        if (*p == '"' || *p == '\\') { out[o++] = '\\'; out[o++] = (char)*p; }
        else if (*p == '\n') { out[o++] = '\\'; out[o++] = 'n'; }
        else if (*p == '\r') { out[o++] = '\\'; out[o++] = 'r'; }
        else if (*p == '\t') { out[o++] = '\\'; out[o++] = 't'; }
        else if (*p < 0x20) { o += (size_t)snprintf(out + o, out_cap - o, "\\u%04x", *p); }
        else out[o++] = (char)*p;
    }
    out[o] = 0;
    return o;
}

/* append piece to a growable accumulation buffer; 0 = ok, -1 = oom */
static int acc_push(char **acc, size_t *len, size_t *cap, const char *piece)
{
    size_t ol = strlen(piece);
    if (*len + ol + 1 > *cap) {
        size_t nc = *cap ? *cap * 2 : 8192;
        while (nc < *len + ol + 1) nc *= 2;
        char *nb = (char *)realloc(*acc, nc);
        if (!nb) return -1;
        *acc = nb;
        *cap = nc;
    }
    memcpy(*acc + *len, piece, ol);
    *len += ol;
    (*acc)[*len] = 0;
    return 0;
}

/* ---------- serve command ---------- */

int q35_serve_cmd(int argc, char **argv)
{
    const char *model_dir = NULL, *tok_path = NULL;
    int port = 8080, threads = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else { fprintf(stderr, "serve: unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!model_dir) { fprintf(stderr, "serve: --model required\n"); return 2; }
    if (!tok_path) tok_path = "tokenizer/tokenizer.bin";

    cli_omp_tune(threads, "serve");
    cli_srand(1);

    /* load model + tokenizer */
    char errbuf[512] = "";
    int merr = 0;
    fprintf(stderr, "serve: loading model %s ...\n", model_dir);
    Q35Model *m = q35_model_load(model_dir, 0, errbuf, sizeof errbuf, &merr);
    if (!m) { fprintf(stderr, "serve: model load failed: %s\n", errbuf); return 6; }
    Q35TokErr terr;
    Q35Tok *tok = q35_tok_load(tok_path, &terr);
    if (!tok) { fprintf(stderr, "serve: tokenizer load failed: %s\n", q35_tok_err_str(terr)); return 3; }
    Q35Bpe *bpe = q35_bpe_new(tok);
    if (!bpe) { fprintf(stderr, "serve: bpe init failed\n"); return 4; }
    uint32_t vocab = q35_model_vocab(m);
    float *logits = (float *)malloc((size_t)vocab * sizeof(float));
    if (!logits) return 4;
    fprintf(stderr, "serve: model ready (vocab=%u pos=%u)\n", vocab, q35_model_pos(m));

    /* socket init (Winsock on Windows, no-op elsewhere) */
    if (q35_plat_sock_init() != 0) {
        fprintf(stderr, "serve: socket init failed\n");
        return 8;
    }

    int lfd = q35_plat_sock_listen(port);
    if (lfd < 0) { fprintf(stderr, "serve: listen 127.0.0.1:%d failed\n", port); return 8; }
    fprintf(stderr, "serve: listening on http://127.0.0.1:%d\n", port);
    fprintf(stderr, "serve: POST /v1/completions {\"prompt\":\"...\",\"max_tokens\":N,\"stream\":bool}\n");

    for (;;) {
        int cfd = q35_plat_sock_accept(lfd);
        if (cfd < 0) continue;
        HttpReq req;
        memset(&req, 0, sizeof req);
        req.fd = cfd;
        if (http_read_request(cfd, &req) != 0) { http_close(&req); continue; }
        fprintf(stderr, "serve: %s %s (%zu bytes)\n", req.method, req.path, req.body_len);

        if (!strcmp(req.path, "/health") && !strcmp(req.method, "GET")) {
            char out[256];
            int n = snprintf(out, sizeof out,
                "{\"status\":\"ok\",\"pos\":%u,\"layers\":%u}",
                q35_model_pos(m), q35_model_layers(m));
            http_respond(cfd, 200, "application/json", out, (size_t)n);
        }
        else if (!strcmp(req.path, "/v1/completions") && !strcmp(req.method, "POST")) {
            /* parse request JSON */
            Q35Json *j = q35_json_parse(req.body, req.body_len);
            if (!j) {
                const char *e = "{\"error\":\"invalid json\"}";
                http_respond(cfd, 400, "application/json", e, strlen(e));
                http_close(&req);
                continue;
            }
            JVal *root = q35_json_root(j);
            const char *prompt = q35_jstr(root, "prompt");
            if (!prompt || !*prompt) {
                q35_json_free(j);
                const char *e = "{\"error\":\"missing prompt\"}";
                http_respond(cfd, 400, "application/json", e, strlen(e));
                http_close(&req);
                continue;
            }
            const JVal *v;
            int max_tokens = 32, top_k = 0, do_reset = 0, do_chat = 0;
            double temperature = 0.0, top_p = 1.0;
            int do_stream = 0;
            if ((v = q35_obj_get(root, "max_tokens")) && v->t == J_NUM)
                max_tokens = (int)v->v.num;
            if ((v = q35_obj_get(root, "temperature")) && v->t == J_NUM)
                temperature = v->v.num;
            if ((v = q35_obj_get(root, "top_k")) && v->t == J_NUM)
                top_k = (int)v->v.num;
            if ((v = q35_obj_get(root, "top_p")) && v->t == J_NUM)
                top_p = v->v.num;
            if ((v = q35_obj_get(root, "stream")) && v->t == J_BOOL)
                do_stream = v->v.b;
            if ((v = q35_obj_get(root, "reset")) && v->t == J_BOOL)
                do_reset = v->v.b;
            if ((v = q35_obj_get(root, "chat")) && v->t == J_BOOL)
                do_chat = v->v.b;
            if (max_tokens < 1) max_tokens = 1;
            if (max_tokens > 4096) max_tokens = 4096;

            if (do_reset) q35_model_reset(m);

            /* chat wrap */
            char *owned = NULL;
            const char *text = prompt;
            if (do_chat) {
                const char *pre = "<|im_start|>user\n";
                const char *post = "<|im_end|>\n<|im_start|>assistant\n";
                owned = (char *)malloc(strlen(pre) + strlen(prompt) + strlen(post) + 1);
                if (owned) {
                    strcpy(owned, pre); strcat(owned, prompt); strcat(owned, post);
                    text = owned;
                }
            }

            /* encode + prefill */
            size_t tcap = strlen(text) + 64;
            uint32_t *ids = (uint32_t *)malloc(tcap * sizeof(uint32_t));
            if (!ids) {
                q35_json_free(j); free(owned);
                const char *e = "{\"error\":\"oom\"}";
                http_respond(cfd, 500, "application/json", e, strlen(e));
                http_close(&req);
                continue;
            }
            int n_prompt = q35_bpe_encode(bpe, (char *)text, ids, tcap);
            free(owned);
            if (n_prompt <= 0) {
                free(ids); q35_json_free(j);
                const char *e = "{\"error\":\"encode failed\"}";
                http_respond(cfd, 400, "application/json", e, strlen(e));
                http_close(&req);
                continue;
            }
            double t0 = q35_plat_now();
            int rc = q35_forward_prefill(m, (const int32_t *)ids, (size_t)n_prompt, logits);
            free(ids);
            if (rc != Q35_MODEL_OK) {
                q35_json_free(j);
                const char *e = "{\"error\":\"prefill failed\"}";
                http_respond(cfd, 500, "application/json", e, strlen(e));
                http_close(&req);
                continue;
            }

            if (do_stream) {
                /* chunked streaming: header, then one JSON line per token */
                char hdr[128];
                int hn = snprintf(hdr, sizeof hdr,
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: application/x-ndjson\r\n"
                    "Transfer-Encoding: chunked\r\n"
                    "Connection: close\r\n\r\n");
                send_all(cfd, hdr, (size_t)hn);
            }

            /* decode loop; non-stream accumulates into a growable buffer */
            char *acc = NULL;
            size_t acc_len = 0, acc_cap = 0;
            char esc[8192];
            for (int t = 0; t < max_tokens; t++) {
                uint32_t tid = cli_pick_token(logits, vocab, temperature, top_k, top_p);
                uint32_t blen;
                const uint8_t *bts = q35_tok_bytes(tok, tid, &blen);
                char piece[1024];
                size_t pl = blen < sizeof piece - 1 ? blen : sizeof piece - 1;
                memcpy(piece, bts, pl);
                piece[pl] = 0;
                if (do_stream) {
                    json_escape(piece, esc, sizeof esc);
                    char line[8500];
                    int ln = snprintf(line, sizeof line, "{\"text\":\"%s\"}\n", esc);
                    /* chunked encoding: %x\r\n<data>\r\n */
                    char chunk[8600];
                    int cn = snprintf(chunk, sizeof chunk, "%x\r\n%s\r\n", ln, line);
                    if (cn > 0 && !send_all(cfd, chunk, (size_t)cn)) break;
                } else {
                    if (acc_push(&acc, &acc_len, &acc_cap, piece) != 0) break; /* oom: send what we have */
                }
                rc = q35_forward_decode(m, (int32_t)tid, logits);
                if (rc != Q35_MODEL_OK) break;
            }

            double elapsed = q35_plat_now() - t0;
            if (do_stream) {
                const char *fin = "{\"done\":true}\n";
                char chunk[128];
                int cn = snprintf(chunk, sizeof chunk, "%x\r\n%s\r\n0\r\n\r\n",
                                  (int)strlen(fin), fin);
                if (cn > 0) send_all(cfd, chunk, (size_t)cn);
            } else {
                /* full response with escaped text */
                char *full = (char *)malloc(acc_len * 6 + 256);
                if (full) {
                    char *ep = (char *)malloc(acc_len * 6 + 8);
                    if (ep) {
                        json_escape(acc ? acc : "", ep, acc_len * 6 + 8);
                        int n = snprintf(full, acc_len * 6 + 256,
                            "{\"text\":\"%s\",\"elapsed\":%.2f}",
                            ep, elapsed);
                        http_respond(cfd, 200, "application/json", full, (size_t)n);
                        free(ep);
                    } else {
                        const char *e = "{\"error\":\"oom\"}";
                        http_respond(cfd, 200, "application/json", e, strlen(e));
                    }
                    free(full);
                } else {
                    const char *e = "{\"error\":\"oom\"}";
                    http_respond(cfd, 500, "application/json", e, strlen(e));
                }
            }
            free(acc);
            q35_json_free(j);
        }
        else {
            const char *e = "{\"error\":\"not found\"}";
            http_respond(cfd, 404, "application/json", e, strlen(e));
        }
        http_close(&req);
    }
    /* unreachable */
    return 0;
}
