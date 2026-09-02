/* Shared CLI helpers for run/serve subcommands: sampling (greedy /
 * temperature + top-k + top-p) and OpenMP thread auto-tune. Extracted from
 * run_cmd.c / serve_cmd.c where it was duplicated verbatim (ticket #25). */
#ifndef Q35_CLI_COMMON_H
#define Q35_CLI_COMMON_H

#include <stdint.h>

/* Seed the shared xorshift32 sampler (0 -> time(NULL) fallback handled by
 * caller). Safe to call again between conversations. */
void cli_srand(uint32_t seed);

/* Sampling: temperature, top-k, top-p (nucleus). When temp<=0, greedy.
 * When top_k>0, restrict to k highest-prob tokens. When top_p<1.0, restrict
 * to the smallest set whose cumulative prob >= top_p. Both can combine. */
uint32_t cli_pick_token(const float *logits, uint32_t vocab,
                        double temp, int top_k, double top_p);

/* Set OpenMP thread count: threads > OMP_NUM_THREADS > auto-tuned.
 * Auto-tune: on SMT CPUs, running one thread per logical core over-subscribes
 * the shared FP units. The sweet spot is ~5/3 x physical cores (e.g. 20 on a
 * 12C/24T Zen3), balancing memory-latency hiding against FMA contention.
 * Prints "<who>: using N threads" to stderr. */
void cli_omp_tune(int threads, const char *who);

#endif
