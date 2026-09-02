# #27: DeltaNet chunked prefill

Label: wayfinder:task
Claim: 未认领
Blocked by: —

## Resolution (2026-08-24)

### 已落地

1. **DeltaNet scratch 池化**（`q35_dn_forward_s`）：model load 时分配一次，48 层共用，消除 per-call malloc/free。DeltaNet -3%。

2. **batch matmul kernel**（`q35_mm_fp8_batch`）：`Y[rows, n] = W[rows, cols] @ X[n, cols]^T`，AVX2 + OpenMP 行并行，每行 n 个 token 的 dot product 共享权重行（cache 友好）。与 `q35_mm_fp8` 比特一致（FNV 锚定 + maxabs=0，ctest `batch_mm`）。

3. **DeltaNet batch prefill 路径**：`q35_dn_forward_s(seq_len>1)` 时用 batch matmul 一次性算 4 个投影（qkv/z/a/b），然后逐 token 做 conv1d + delta rule + out_proj。与 `seq_len=1` × n 比特一致（deltanet test `prefill5.decode==prefill bitwise`）。

4. **`q35_forward_prefill` batch 路径**：n>1 时走批量路径——batch embed + batch rmsnorm，DeltaNet 层用 batch 投影，attention 层逐 token（KV cache 顺序依赖），MLP 逐 token。修复了初版中 DeltaNet 层缺 MLP 段的 bug。与 `forward_one` × n 比特一致（ctest `prefill`：FNV 相同，maxabs=0）。

### 性能

| prompt 长度 | 优化前（逐 token） | 优化后（batch MLP） | per-token |
|---|---|---|---|
| 5 token | 8.8 s | 7.8 s | 1.56 s/tok |
| 36 token | — | 46.1 s | 1.28 s/tok |

MLP batch 化（gate+up+down 用 `q35_mm_fp8_batch`）+ flat swiglu + flat residual。per-token 速度随 n 增大而改善（batch matmul 共享权重读取）。36 token prefill 的 per-token 已比 decode（1.58 s/tok）快。

Profile（5 token）：MLP 5.08s (67%)、DeltaNet 1.91s (25%)、attention 0.61s (8%)。MLP 仍是大头——batch matmul 对 n=5 的效率受限于 AVX2 gather 开销被乘以 n。attention 逐 token（KV cache 顺序依赖），未 batch 化投影。

### 未做

- attention q/k/v batch 投影（只占 8%，改动面大——需要拆 `attn_token`）
- HF 的 `torch_chunk_gated_delta_rule`（块内矩阵求逆 + 跨块衰减）

Status: CLOSED
