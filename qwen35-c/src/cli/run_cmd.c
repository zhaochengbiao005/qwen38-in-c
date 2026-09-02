/* Ticket-21: `qwen35 run` subcommand.
 * Load full model, BPE-encode prompt, prefill, greedy/sample decode loop,
 * stream decoded bytes to stdout. Exit codes: 0 ok, 2 usage, 3 tokenizer,
 * 4 io/encode, 5 config, 6 model load, 7 forward. */
#include "q35/q35_tok.h"
#include "q35/q35_bpe.h"
#include "q35/q35_model.h"
#include "q35/q35_cfg.h"
#include "q35_plat.h"
#include "cli_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int file_exists(const char *p) {
    FILE *f = fopen(p, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

int q35_run_cmd(int argc, char **argv) {
    const char *model_dir = NULL, *tok_path = NULL, *prompt_file = NULL, *prompt = NULL;
    const char *save_state_path = NULL, *load_state_path = NULL;
    int max_tokens = 32, dump_ids = 0, chat = 0, top_k = 0, threads = 0, interactive = 0;
    double temperature = 0.0, top_p = 1.0;
    uint32_t seed = 0;
    int have_seed = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) prompt_file = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if ((!strcmp(argv[i], "--max-tokens") || !strcmp(argv[i], "--tokens")) && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temperature") && i + 1 < argc) temperature = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) top_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) { seed = (uint32_t)strtoul(argv[++i], NULL, 10); have_seed = 1; }
        else if (!strcmp(argv[i], "--dump-ids")) dump_ids = 1;
        else if (!strcmp(argv[i], "--chat")) chat = 1;
        else if (!strcmp(argv[i], "--interactive") || !strcmp(argv[i], "-i") || !strcmp(argv[i], "--incremental")) { interactive = 1; chat = 1; }
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save-state") && i + 1 < argc) save_state_path = argv[++i];
        else if (!strcmp(argv[i], "--load-state") && i + 1 < argc) load_state_path = argv[++i];
        else { fprintf(stderr, "run: unknown/incomplete arg: %s\n", argv[i]); return 2; }
    }
    if (!model_dir || (!prompt && !prompt_file && !interactive)) {
        fputs("usage: qwen35 run --model <dir> (--prompt <text> | --prompt-file <path> | --interactive)\n"
              "             [--tokenizer <bin>] [--max-tokens N] [--temperature T]\n"
              "             [--top-k K] [--top-p P] [--seed N] [--chat] [--dump-ids] [--threads T]\n"
              "             [--interactive|-i|--incremental]  multi-turn chat (model state persists across turns)\n"
              "             [--save-state <path>] [--load-state <path>]\n"
              "  temperature 0=greedy, >0 samples from softmax(logits/T)\n"
              "  top-k       restrict to K highest-prob tokens (0=off)\n"
              "  top-p       nucleus: smallest set with cumprob >= P (1.0=off)\n"
              "  save-state  serialize DeltaNet state + KV cache + pos to file after run\n"
              "  load-state  restore state from file before run (resume conversation)\n", stderr);
        return 2;
    }
    if (max_tokens < 0) max_tokens = 0;

    /* set OpenMP thread count: --threads > OMP_NUM_THREADS > auto-tuned
     * (SMT-aware policy in cli_omp_tune) */
    cli_omp_tune(threads, "run");

    if (!tok_path) {
        if (file_exists("tokenizer/tokenizer.bin")) tok_path = "tokenizer/tokenizer.bin";
        else { fputs("run: --tokenizer not given and tokenizer/tokenizer.bin not found\n", stderr); return 3; }
    }

    char cfg_path[1024];
    q35_plat_path_join(cfg_path, sizeof cfg_path, model_dir, "config.json");
    Q35Cfg cfg; char errbuf[512] = "";
    if (q35_cfg_load(cfg_path, &cfg, errbuf, sizeof errbuf) != 0) {
        fprintf(stderr, "run: config load failed: %s\n", errbuf);
        return 5;
    }

    Q35TokErr terr;
    Q35Tok *tok = q35_tok_load(tok_path, &terr);
    if (!tok) { fprintf(stderr, "run: tokenizer load %s: %s\n", tok_path, q35_tok_err_str(terr)); return 3; }

    char *owned = NULL;
    const char *text = prompt;
    if (prompt_file) {
        size_t n;
        char *raw = (char *)q35_plat_read_file(prompt_file, &n);
        if (!raw) { fprintf(stderr, "run: cannot read %s\n", prompt_file); return 4; }
        owned = (char *)malloc(n + 1);
        if (!owned) { fputs("run: oom\n", stderr); q35_plat_free(raw); return 4; }
        memcpy(owned, raw, n); owned[n] = 0;
        q35_plat_free(raw);
        text = owned;
    }
    if (chat && text) {  /* skip chat-wrap for interactive (NULL text) — it wraps per-turn */
        size_t pn = strlen(text);
        const char *pre = "<|im_start|>user\n";
        const char *post = "<|im_end|>\n<|im_start|>assistant\n";
        char *t2 = (char *)malloc(strlen(pre) + pn + strlen(post) + 1);
        if (!t2) { fputs("run: oom\n", stderr); return 4; }
        strcpy(t2, pre); strcat(t2, text); strcat(t2, post);
        free(owned); owned = t2; text = t2;
    }

    Q35Bpe *bpe = q35_bpe_new(tok);
    if (!bpe) { fputs("run: bpe init failed\n", stderr); return 4; }
    uint32_t *ids = NULL;
    int n_prompt = 0;
    if (text) {
        size_t cap = strlen(text) + 64;
        ids = (uint32_t *)malloc(cap * sizeof(uint32_t));
        if (!ids) { fputs("run: oom\n", stderr); return 4; }
        n_prompt = q35_bpe_encode(bpe, text, ids, cap);
        if (n_prompt <= 0) { fputs("run: encode failed (empty or invalid UTF-8?)\n", stderr); return 4; }
        fprintf(stderr, "run: prompt = %d tokens\n", n_prompt);
    }

    double t0 = q35_plat_now();
    int merr;
    Q35Model *m = q35_model_load(model_dir, 0, errbuf, sizeof errbuf, &merr);
    if (!m) { fprintf(stderr, "run: model load: %s (err=%d)\n", errbuf, merr); return 6; }
    double t_load = q35_plat_now() - t0;
    uint32_t vocab = q35_model_vocab(m);
    fprintf(stderr, "run: loaded %s (layers=%u vocab=%u hidden=%u) in %.1f s, RSS %.1f MB\n",
            model_dir, q35_model_layers(m), vocab, q35_model_hidden(m),
            t_load, (double)q35_plat_peak_rss() / 1048576.0);

    if (load_state_path) {
        int src = q35_model_load_state(m, load_state_path);
        if (src != Q35_MODEL_OK) {
            fprintf(stderr, "run: load-state %s failed (err=%d)\n", load_state_path, src);
            return 6;
        }
        fprintf(stderr, "run: restored state from %s (pos=%u)\n", load_state_path, q35_model_pos(m));
    }

    float *logits = (float *)malloc((size_t)vocab * sizeof(float));
    if (!logits) { fputs("run: oom\n", stderr); return 4; }

    cli_srand(have_seed ? seed : (uint32_t)time(NULL));

    uint32_t eos_ids[4]; int n_eos = 0;
    if (cfg.eos_token_id > 0) eos_ids[n_eos++] = (uint32_t)cfg.eos_token_id;
    { uint32_t ce = q35_tok_chat_eos(tok); if (ce && ce != (uint32_t)cfg.eos_token_id) eos_ids[n_eos++] = ce; }

    /* ---- interactive multi-turn chat ---- */
    if (interactive) {
        int rc;
        const char *pre = "<|im_start|>user\n";
        const char *mid = "<|im_end|>\n<|im_start|>assistant\n";
        const char *end = "<|im_end|>";
        char line[8192];
        int turn = 0;
        fprintf(stderr, "run: interactive chat (type your message, Ctrl-Z+Enter to exit)\n");
        for (;;) {
            fprintf(stderr, "\n[user] ");
            fflush(stderr);
            if (!fgets(line, sizeof line, stdin)) break;
            /* strip trailing newline */
            size_t ln = strlen(line);
            while (ln > 0 && (line[ln-1] == '\n' || line[ln-1] == '\r')) line[--ln] = 0;
            if (ln == 0) continue;

            /* wrap: <im_start>user\n{input}<im_end>\n<im_start>assistant\n */
            size_t buflen = strlen(pre) + ln + strlen(mid) + 1;
            char *buf = (char *)malloc(buflen);
            if (!buf) break;
            strcpy(buf, pre); strcat(buf, line); strcat(buf, mid);

            /* BPE can expand bytes (byte-level fallback), give generous cap */
            size_t cap2 = buflen * 4 + 256;
            uint32_t *ids2 = (uint32_t *)malloc(cap2 * 4);
            if (!ids2) { free(buf); break; }
            int n2 = q35_bpe_encode(bpe, buf, ids2, cap2);
            free(buf);
            if (n2 <= 0) { fprintf(stderr, "run: encode failed\n"); free(ids2); continue; }

            fprintf(stderr, "[assistant] ");
            fflush(stderr);
            t0 = q35_plat_now();
            rc = q35_forward_prefill(m, (const int32_t *)ids2, (size_t)n2, logits);
            if (rc != Q35_MODEL_OK) { fprintf(stderr, "\nrun: prefill rc=%d\n", rc); free(ids2); continue; }
            double t_pf = q35_plat_now() - t0;
            fprintf(stderr, "(prefill %d tok %.1fs) ", n2, t_pf);
            fflush(stderr);

            t0 = q35_plat_now();
            int n_gen = 0;
            for (int step = 0; step < max_tokens; step++) {
                uint32_t t = cli_pick_token(logits, vocab, temperature, top_k, top_p);
                int is_eos = 0;
                for (int k = 0; k < n_eos; k++) if (t == eos_ids[k]) { is_eos = 1; break; }
                if (is_eos) break;
                uint32_t blen = 0;
                const uint8_t *bts = q35_tok_bytes(tok, t, &blen);
                if (bts && blen) fwrite(bts, 1, blen, stdout);
                fflush(stdout);
                n_gen++;
                rc = q35_forward_decode(m, (int32_t)t, logits);
                if (rc != Q35_MODEL_OK) break;
            }
            double t_gen = q35_plat_now() - t0;
            /* append <im_end>\n token sequence so next turn's prefill sees it */
            {
                char *eb = (char *)malloc(strlen(end) + 2);
                if (eb) {
                    strcpy(eb, end); eb[strlen(end)] = '\n'; eb[strlen(end)+1] = 0;
                    uint32_t eid[32]; int en = q35_bpe_encode(bpe, eb, eid, 32);
                    if (en > 0) q35_forward_prefill(m, (const int32_t *)eid, (size_t)en, NULL);
                    free(eb);
                }
            }
            fprintf(stderr, "\n(%d tok, %.1fs, %.2f tok/s)\n", n_gen, t_gen,
                    t_gen > 0 ? n_gen / t_gen : 0.0);
            free(ids2);
            turn++;
        }
        fprintf(stderr, "\nrun: %d turn%s, exiting.\n", turn, turn == 1 ? "" : "s");
        if (save_state_path) {
            int src = q35_model_save_state(m, save_state_path);
            if (src != Q35_MODEL_OK)
                fprintf(stderr, "run: save-state %s failed (err=%d)\n", save_state_path, src);
            else
                fprintf(stderr, "run: saved state to %s (pos=%u)\n", save_state_path, q35_model_pos(m));
        }
        free(logits); free(ids); free(owned);
        q35_bpe_free(bpe); q35_tok_free(tok); q35_model_free(m);
        return 0;
    }

    t0 = q35_plat_now();
    int rc = q35_forward_prefill(m, (const int32_t *)ids, (size_t)n_prompt, logits);
    if (rc != Q35_MODEL_OK) { fprintf(stderr, "run: prefill failed rc=%d\n", rc); return 7; }
    double t_prefill = q35_plat_now() - t0;
    fprintf(stderr, "run: prefill %d tokens in %.1f s\n", n_prompt, t_prefill);

    if (dump_ids) fprintf(stderr, "ids:");
    int n_gen = 0, hit_eos = 0;
    t0 = q35_plat_now();
    for (int step = 0; step < max_tokens; step++) {
        uint32_t t = cli_pick_token(logits, vocab, temperature, top_k, top_p);
        int is_eos = 0;
        for (int k = 0; k < n_eos; k++) if (t == eos_ids[k]) { is_eos = 1; break; }
        if (is_eos) { hit_eos = 1; break; }
        if (dump_ids) fprintf(stderr, " %u", t);
        uint32_t blen = 0;
        const uint8_t *bts = q35_tok_bytes(tok, t, &blen);
        if (bts && blen) fwrite(bts, 1, blen, stdout);
        fflush(stdout);
        n_gen++;
        rc = q35_forward_decode(m, (int32_t)t, logits);
        if (rc != Q35_MODEL_OK) { fprintf(stderr, "\nrun: decode failed rc=%d\n", rc); return 7; }
    }
    double t_gen = q35_plat_now() - t0;
    if (dump_ids) fputc(10, stderr);
    fputc(10, stdout);
    fprintf(stderr, "run: generated %d tokens in %.1f s (%.2f tok/s)%s | peak RSS %.1f MB\n",
            n_gen, t_gen, t_gen > 0 ? n_gen / t_gen : 0.0, hit_eos ? " [EOS]" : " [max-tokens]",
            (double)q35_plat_peak_rss() / 1048576.0);

    if (save_state_path) {
        int src = q35_model_save_state(m, save_state_path);
        if (src != Q35_MODEL_OK)
            fprintf(stderr, "run: save-state %s failed (err=%d)\n", save_state_path, src);
        else
            fprintf(stderr, "run: saved state to %s (pos=%u)\n", save_state_path, q35_model_pos(m));
    }

    free(logits); free(ids); free(owned);
    q35_bpe_free(bpe); q35_tok_free(tok); q35_model_free(m);
    return 0;
}
