# qwen35-c — Windows 纯 C99 推理引擎 for Qwen3.8-27B-FP8

**[中文](#中文) | [English](#english)**

---

## 中文

零深度学习框架依赖、可审计的本地推理引擎：在 Windows x86-64 上用可移植 C99 直接加载官方 `Qwen/Qwen3.8-27B-FP8` safetensors，输出与 HuggingFace transformers 参考逐 token 一致的推理结果。

以 [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) 的设计思想为蓝本（可移植 C99、手写 safetensors reader、FNV-1a 多路径哈希验证、tiny oracle 三路径对齐），适配 Qwen3.5 新架构。

### 核心特性

- **零框架依赖**：纯 C99 + OpenMP，不依赖 Python/PyTorch/GGUF 转换链——直接读官方 FP8-E4M3 safetensors（block-wise 128×128 量化），不存在中间格式数值漂移盲点
- **Qwen3.5 混合架构**：64 层文本子图 = 48 层 Gated DeltaNet 线性注意力（chunked delta rule prefill）+ 16 层 GQA full attention（partial RoPE + sigmoid output gate）
- **数值可验证**：AVX2 与标量路径 FNV-1a 比特一致；greedy ids 与 HF 参考逐位一致（12/12），logits cos ≥ 0.99996
- **严格 config 校验**：config.json 缺字段或非预期值直接拒绝运行，不会静默用错架构参数
- **平台抽象层（PAL）**：算法文件与 CLI 均零 `#ifdef _WIN32`，Win32 调用全部收进 `q35_plat_win.c`
- **CLI 三件套**：`tokenize`（BPE encode/decode）、`run`（推理 + 采样 + 交互多轮对话 + 状态持久化）、`serve`（localhost HTTP API，`/health` + `/v1/completions`，支持流式）

### 性能（Ryzen 9 5900X，96 GB RAM，Windows 11）

| 指标 | 引擎 | HF bf16 torch |
|---|---|---|
| 加载（含预取） | 6.1 s | 95.7 s |
| prefill | 0.56 s/tok | 13.7 s/tok |
| decode | 1.07 tok/s | 0.022 tok/s |
| 峰值 RSS | 25.9 GB | ~58 GB |

OpenMP 20 线程（SMT 自动调优），端到端比 HF bf16 快 ~48×。

### 快速开始

```cmd
cd qwen35-c
scripts\build.cmd          :: clang-cl + CMake + Ninja，构建 + 12 项 ctest

:: 推理（模型目录放仓库同级，含官方 FP8 safetensors）
build\qwen35.exe run --model ..\Qwen3.8-27B-FP8 --prompt "The capital of China is" --max-tokens 32

:: 采样 / 多轮对话 / HTTP API
build\qwen35.exe run --model ..\Qwen3.8-27B-FP8 --interactive --temperature 0.8 --top-p 0.9
build\qwen35.exe serve --model ..\Qwen3.8-27B-FP8 --port 8080
```

> 模型权重（29 GB）不入库——从 HuggingFace 下载 `Qwen/Qwen3.8-27B-FP8` 放到仓库同级目录即可。

### 测试

- **12 项 ctest**（秒级）：tokenizer / BPE（440 fixture × HF byte-exact）/ safetensors reader / FP8 matmul / elementwise kernels / DeltaNet / attention / 模型前向 / tiny oracle 三路径 / batch matmul / prefill 一致性 / save-load state bitwise roundtrip
- **20 条多 prompt 回归集**：QA/代码/翻译/数学/创意 × HF 参考，greedy ids 精确一致 + logits cos ≥ 0.999
- 详细文档见 [`qwen35-c/README.md`](qwen35-c/README.md)

### 目录结构

```
qwen35-c/     引擎本体（src / include / tests / tools / scripts）
docs/         ADR（FP8 in-memory dequant 决策）+ agent 文档
.wayfinder/   工程地图 + 35 张已闭环票据 + 调研产物（权重 schema、数值规格）
```

### 许可证

[MIT](LICENSE)。蓝本 [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) 的思路致谢见 [技术报告](kimi-k3-in-c-技术报告.md)。

---

## English

# qwen35-c — Pure C99 Inference Engine for Qwen3.8-27B-FP8 on Windows

A zero-framework, auditable local inference engine: portable C99 on Windows x86-64 that loads the official `Qwen/Qwen3.8-27B-FP8` safetensors directly and produces token-for-token identical output against the HuggingFace transformers reference.

Modeled on the design philosophy of [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) (portable C99, hand-written safetensors reader, FNV-1a multi-path hash verification, tiny-oracle three-path alignment), adapted to the Qwen3.5 architecture.

### Highlights

- **Zero framework deps**: pure C99 + OpenMP — no Python/PyTorch, no GGUF conversion chain. Reads the official FP8-E4M3 safetensors (block-wise 128×128 quantization) directly, so there is no intermediate-format numerical drift.
- **Qwen3.5 hybrid architecture**: 64-layer text stack = 48 Gated DeltaNet linear-attention layers (chunked delta-rule prefill) + 16 GQA full-attention layers (partial RoPE + sigmoid output gate).
- **Numerically verifiable**: AVX2 and scalar paths are FNV-1a bit-identical; greedy ids match the HF reference exactly (12/12), logits cos ≥ 0.99996.
- **Strict config validation**: missing or unexpected fields in config.json abort the run instead of silently using wrong architecture parameters.
- **Platform abstraction layer (PAL)**: zero `#ifdef _WIN32` in algorithm files and the CLI — all Win32 calls live in `q35_plat_win.c`.
- **Three CLI subcommands**: `tokenize` (BPE encode/decode), `run` (inference + sampling + interactive multi-turn chat + state persistence), `serve` (localhost HTTP API, `/health` + `/v1/completions`, streaming supported).

### Performance (Ryzen 9 5900X, 96 GB RAM, Windows 11)

| Metric | Engine | HF bf16 torch |
|---|---|---|
| Load (incl. prefetch) | 6.1 s | 95.7 s |
| Prefill | 0.56 s/tok | 13.7 s/tok |
| Decode | 1.07 tok/s | 0.022 tok/s |
| Peak RSS | 25.9 GB | ~58 GB |

20 OpenMP threads (SMT auto-tuned); ~48× faster end-to-end than HF bf16.

### Quick Start

```cmd
cd qwen35-c
scripts\build.cmd          :: clang-cl + CMake + Ninja: build + 12 ctest suite

:: inference (place the model dir next to this repo, containing the official FP8 safetensors)
build\qwen35.exe run --model ..\Qwen3.8-27B-FP8 --prompt "The capital of China is" --max-tokens 32

:: sampling / interactive chat / HTTP API
build\qwen35.exe run --model ..\Qwen3.8-27B-FP8 --interactive --temperature 0.8 --top-p 0.9
build\qwen35.exe serve --model ..\Qwen3.8-27B-FP8 --port 8080
```

> Model weights (29 GB) are NOT committed — download `Qwen/Qwen3.8-27B-FP8` from HuggingFace and place it next to this repo.

### Testing

- **12 ctest cases** (seconds): tokenizer / BPE (440 fixtures × HF byte-exact) / safetensors reader / FP8 matmul / elementwise kernels / DeltaNet / attention / model forward / tiny-oracle three-path / batch matmul / prefill consistency / save-load state bitwise roundtrip.
- **20-prompt regression set**: QA/code/translation/math/creative × HF reference — exact greedy-id match + logits cos ≥ 0.999.
- Full details in [`qwen35-c/README.md`](qwen35-c/README.md).

### Repository Layout

```
qwen35-c/     the engine (src / include / tests / tools / scripts)
docs/         ADRs (FP8 in-memory dequant) + agent docs
.wayfinder/   engineering map + 35 closed tickets + research artifacts (weight schema, numerical spec)
```

### License

[MIT](LICENSE). Credit to the blueprint [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c); see the [technical report](kimi-k3-in-c-技术报告.md) (Chinese).
