# qwen35-c — Windows 纯 C 推理引擎 for Qwen3.8-27B-FP8

零深度学习框架依赖、可审计的本地推理引擎，在 Windows x86-64 上直接加载官方 FP8 safetensors 跑出与 HuggingFace transformers 参考逐 token 一致的输出。

## 概述

基于 [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) 的设计思想（可移植 C99、手写 safetensors reader、FNV-1a 多路径哈希验证、tiny oracle 三路径对齐），适配 Qwen3.5 架构（Gated DeltaNet 线性注意力 + GQA 混合，48:16 交替层）。本仓库为干净重建，借鉴其存储驱动推理、防御式 config、四层测试金字塔等思想；不复制其完整代码。

- **模型**：`Qwen/Qwen3.8-27B-FP8`（dense，64 层，hidden 5120，FP8-E4M3 block-wise 128×128 量化）
- **权重**：直接读官方 safetensors，无中间格式转换（ADR-0001）
- **内存**：全驻留设计，FP8+scale 常驻 ~25 GB，内存地板 32 GB
- **SIMD**：AVX2 + FMA only（基准机 Ryzen 9 5900X，无 AVX-512）

## 构建

### 依赖

- Visual Studio 2026 Build Tools（VC++ + Windows SDK 19041）
- LLVM clang-cl 22（`C:\Program Files\LLVM`）
- CMake 3.20+、Ninja
- Python 3.10+（仅生成 fixture 和跑参考侧对比，非运行时依赖）
  - `torch`、`transformers`、`safetensors`、`ml_dtypes`、`numpy`（仅 HF 参考对比用）

### 一键构建

```cmd
cd qwen35-c
scripts\build.cmd
```

构建链：`devcmd.cmd`（VS 环境）→ CMake + Ninja + clang-cl → 编译 + 部署 libomp.dll → ctest 12/12。

## 用法

### tokenize

```cmd
build\qwen35.exe tokenize --tokenizer tokenizer\tokenizer.bin --prompt "Hello world" --ids-only
```

### run（推理）

```cmd
:: greedy
build\qwen35.exe run --model ..\Qwen3.8-27B-FP8 --prompt "The capital of China is" --max-tokens 32

:: 采样（temperature + top-p + top-k）
build\qwen35.exe run --model ..\Qwen3.8-27B-FP8 --prompt "Write a poem about autumn" ^
    --temperature 0.8 --top-p 0.9 --top-k 40 --seed 42 --max-tokens 64

:: chat 模式（自动包装 im_start/im_end）
build\qwen35.exe run --model ..\Qwen3.8-27B-FP8 --prompt "What is 2+2?" --chat --max-tokens 32

:: 从文件读 prompt + 指定线程数
build\qwen35.exe run --model ..\Qwen3.8-27B-FP8 --prompt-file input.txt --threads 8 --max-tokens 64
```

### serve（HTTP API server）

```cmd
build\qwen35.exe serve --model ..\Qwen3.8-27B-FP8 --port 8080
```

启动后监听 `http://127.0.0.1:8080`（仅 localhost），一次处理一个请求（模型状态是单一会话）：

**GET /health** — 健康检查
```
curl http://127.0.0.1:8080/health
→ {"status":"ok","pos":8,"layers":64}
```

**POST /v1/completions** — 推理（非流式）
```cmd
curl -X POST http://127.0.0.1:8080/v1/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"The capital of China is\",\"max_tokens\":8}"
→ {"text":" Beijing.\nThe capital of China is","elapsed":9.33}
```

**POST /v1/completions** — 推理（流式，NDJSON 逐 token）
```cmd
curl -X POST http://127.0.0.1:8080/v1/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"What is 2+2?\",\"max_tokens\":16,\"chat\":true,\"stream\":true}"
→ {"text":"The"}\n{"text":" answer"}\n...{"done":true}\n
```

请求体参数：

| 参数 | 说明 |
|---|---|
| `prompt` | 输入文本（必填） |
| `max_tokens` | 最多生成 token 数（默认 32，上限 4096） |
| `temperature` | 0=greedy，>0 采样（默认 0） |
| `top_k` / `top_p` | 采样截断（默认 off / 1.0） |
| `stream` | true 时 NDJSON 逐 token 流式返回 |
| `chat` | true 时自动包装 Qwen chat 格式（`<\|im_start\|>user...`） |
| `reset` | true 时先清空会话状态再推理（多轮对话不传则延续上文） |

多轮对话：不传 `reset` 时，多次请求共享同一会话状态（DeltaNet state + KV cache 延续），相当于同一个对话的连续轮次。

### save/load state（跨进程对话持久化）

```cmd
:: 第一轮对话，结束后保存状态
build\qwen35.exe run --model ..\Qwen3.8-27B-FP8 --prompt "Hello" --max-tokens 32 --save-state chat.bin

:: 第二轮，加载之前的状态继续对话
build\qwen35.exe run --model ..\Qwen3.8-27B-FP8 --prompt "What did I say?" --max-tokens 32 --load-state chat.bin
```

| 参数 | 说明 |
|---|---|
| `--model <dir>` | 模型目录（含 config.json + *.safetensors） |
| `--tokenizer <bin>` | tokenizer.bin（默认 `tokenizer/tokenizer.bin`） |
| `--prompt <text>` | 输入文本 |
| `--prompt-file <path>` | 从文件读输入（UTF-8） |
| `--max-tokens N` | 最多生成 N 个 token（默认 32） |
| `--temperature T` | 0=greedy，>0 从 softmax(logits/T) 采样 |
| `--top-k K` | 只从概率最高的 K 个 token 里选（0=off） |
| `--top-p P` | nucleus：累积概率 ≥ P 的最小集合（1.0=off） |
| `--seed N` | 随机种子 |
| `--threads N` | OpenMP 线程数（默认=全部核心） |
| `--chat` | 包装为 Qwen chat 格式 |
| `--interactive` / `-i` / `--incremental` | 多轮对话模式（模型状态跨轮累积，Ctrl-Z+Enter 退出） |
| `--dump-ids` | 输出 token id 序列到 stderr |
| `--save-state <path>` | 运行结束后序列化 DeltaNet 状态 + KV cache + 位置到文件 |
| `--load-state <path>` | 运行前从文件恢复状态（恢复对话） |

## 测试

### ctest（12 项，秒级）

```cmd
ctest --test-dir build --output-on-failure
```

| 测试 | 覆盖 |
|---|---|
| tok | tokenizer loader |
| bpe | BPE encode（440 fixture × HF byte-exact） |
| st | safetensors reader（真实 checkpoint 索引） |
| mm | FP8 dequant matmul + BF16 matmul |
| kern | elementwise（rmsnorm/silu/swiglu/rope/softmax） |
| deltanet | DeltaNet 线性注意力层 |
| attn | GQA full attention + KV cache |
| model | 8 层混合 mini 模型前向 |
| oracle | tiny oracle 三路径（teacher-forcing/greedy/incremental） |
| batch_mm | batch FP8 matmul kernel（Y[rows,n]=W@X^T，AVX2+OpenMP） |
| prefill | batch prefill 路径 vs 逐 token decode 一致性 |
| state | save/load state roundtrip（bitwise 一致） |

### 多 prompt 回归集（手动，~30 秒）

```cmd
:: 先生成 HF 参考（首次，~1.5 小时）
python tools\gen_regression_ref.py ..\Qwen3.8-27B-FP8

:: 引擎侧回归
build\run_regression.exe ..\Qwen3.8-27B-FP8
```

20 条 diverse prompt（QA/代码/翻译/数学/化学/创意）× HF 参考，每条验 greedy ids 精确一致 + 末 token logits cos ≥ 0.999。

## 性能

Ryzen 9 5900X（12C/24T），96 GB RAM，Windows 11：

| 指标 | 引擎 | HF bf16 torch |
|---|---|---|
| 加载（含预取） | 6.1 s | 95.7 s |
| prefill 5 tok | 4.3 s | 73.9 s |
| prefill 75 tok | 42.2 s（0.56 s/tok） | — |
| decode | 1.07 tok/s | 44.7 s/tok |
| 峰值 RSS | 25.9 GB | ~58 GB |

OpenMP 20 线程（自动调优，SMT 5/3 × 物理核），比 HF bf16 快 48 倍。

Per-phase profile（`Q35_PROF=1`）：MLP 64%，DeltaNet 24%，attention 7%，lm_head 6%。

### 优化历程

- **FP8 ALU dequant**：FP8→F32 不再用 `_mm256_i32gather_ps` LUT 查表（~11 周期/gather），改用纯位移拼接 `(sign<<31)|((exp+120)<<23)|(man<<20)`（3-4 条指令）。对正常值 bit-identical，subnormal 回退 LUT。
- **Attention SIMD**：`dot_fixed` 和输出累加 `o[i]+=p*v[i]` 从标量改为 AVX2 `mul+add`（不用 FMA，保持 bit-identical）。
- **DeltaNet prefill 并行化**：48 层 DeltaNet 的 chunked prefill v-head 循环从串行改为 `#pragma omp parallel for`，每线程独立 scratch。
- **malloc 消除**：attention（6 buffer × 64 层 = 384 次/token）和 DeltaNet chunked（13 buffer × 48 层 = 624 次/prefill）的 scratch 全部预分配到 `Q35Model`，模型加载时一次分配，所有层复用。
- **大页权重映射**：25.9 GB 权重 mmap 加 `SEC_LARGE_PAGES`，TLB 条目从 ~6.3M 降到 ~12.4K（无权限自动回退 4KB 页）。
- **LTO**：`-flto` 跨翻译单元内联。
- **权重预取**：加载后用 `PrefetchVirtualMemory`（Win8+，无则回退顺序触摸）将 26 GB 权重 page-in 到 RAM，把 I/O 与首次推理计算分离。总时间不变但 prefill 计算阶段不再被 page-fault 打断，profiling 数据更干净。
- **线程数自动调优**：SMT CPU 上满逻辑核会 FMA 单元竞争。自动检测物理核数，默认用 `5/3 × 物理核`（5900X 12C → 20 线程），比满 24 线程快 5-10%。可用 `--threads` 或 `OMP_NUM_THREADS` 覆盖。
- **软件预取**：matmul 内层循环用 `_mm_prefetch` 提前 128 字节预取下一段权重和输入向量到 L1，减少 cache miss 停顿。

## 已知限制

- DeltaNet chunked delta rule 已集成到 prefill 主路径（`dn_chunk_head`），多 chunk 循环支持任意长度 prompt。chunk 与逐 token 递推的 fp32 归约序不同，输出在容差内一致而非 bitwise 一致。
- DeltaNet chunked prefill 的 v-head 循环已 OpenMP 并行化，每线程独立 scratch。
- KV cache 有动态增长（初始 4096，翻倍扩容到 max_cap）和 OOM 防护，但无 sliding window / ring buffer 淘汰
- greedy + temperature/top-k/top-p 采样 only，无 beam search / speculative decoding
- 单累加器 matmul（FNV 契约限制）：多累加器可 3-5x 提速但需放宽 bit-identical 契约，未实现

## 架构

```
qwen35-c/
├── src/
│   ├── q35_model.c      64 层混合前向总装
│   ├── q35_deltanet.c    Gated DeltaNet 线性注意力
│   ├── q35_attn.c        GQA full attention + KV cache
│   ├── q35_mm.c          FP8/BF16 matvec kernel（AVX2 + OpenMP + multi4）
│   ├── q35_kern.c        elementwise kernels
│   ├── q35_st.c          手写 safetensors reader
│   ├── q35_cfg.c         config.json 严格解析
│   ├── q35_tok.c         tokenizer blob loader
│   ├── q35_bpe.c         byte-level BPE encoder
│   ├── q35_json.c        极简 JSON DOM 解析器
│   ├── q35_plat_win.c    Windows PAL
│   └── cli/
│       ├── q35_cli.c     入口（tokenize / run / serve）
│       ├── cli_common.c  采样器 + 线程调优（run/serve 共享）
│       ├── run_cmd.c     run 子命令
│       └── serve_cmd.c   serve 子命令（HTTP API）
├── include/q35/          公共头文件
├── tests/                12 个测试 + fixture
├── tools/                Python 生成器 + C 工具
├── tokenizer/            tokenizer.bin
├── scripts/              build.cmd / devcmd.cmd
└── CMakeLists.txt
```

## 设计决策

- **ADR-0001**：FP8 权重常驻内存，matmul 内层逐 128×128 块 on-the-fly dequant，不物化 BF16 副本。
- **OpenMP**：clang-cl 的 `/openmp:llvm` 被静默忽略，用 `-openmp` + 显式链接 `libomp.lib`。
- **数值验证**：FNV-1a 比特哈希验证 AVX2/标量一致性；tiny oracle 三路径精确对齐；真实 checkpoint vs HF 参考 logits 逐元素对比（cos ≥ 0.999）。

## 致谢

蓝本：[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)（v1.0.0）。技术调研报告见仓库根目录 `kimi-k3-in-c-技术报告.md`。

## 构建（Windows）

前置：LLVM (clang-cl)、VS 2026 Build Tools（VC++ + Windows SDK）、CMake、Ninja。

```cmd
scripts\build.cmd
```

## 目录

- `tools/convert_tokenizer.py` — 官方 tokenizer.json → `tokenizer/tokenizer.bin`（离线，一次性）
- `tokenizer/src/` — 官方 tokenizer 源文件（tokenizer.json 需走 LFS 下载）
- `include/q35/q35_tok.h` / `src/q35_tok.c` — tokenizer blob 加载器与查找 API
- `src/q35_plat.h` / `src/q35_plat_win.c` — 平台抽象层（算法文件零 `#ifdef _WIN32`）
- `tests/` — 单元测试（`test_tok` 等；fixture 缺失时按 77 退出码显式 NOT RUN）

## 状态

端到端已跑通（2026-08-25）。12 ctest + 20 prompt 回归集全绿。

- [x] Tokenizer 转换工具 + blob（248077 定义 / 248320 声明词表，247587 merges，9 fixture 与 HF byte-exact）
- [x] Tokenizer C 加载器（完整性校验 + 双向查找 + 特殊 token API）
- [x] BPE encode（#12）：440/440 fixture byte-exact（含 NFC、Hangul、emoji fuzz）
- [x] model 主体：64 层混合前向 + safetensors reader + FP8 kernel
- [x] 真实 27B 门禁（#22）：greedy ids 12/12 逐位一致，logits cos≥0.99996
- [x] OpenMP 并行 matmul + 深层并行化（7.2× 单线程加速）
- [x] 采样（temperature + top-k + top-p）
- [x] `--interactive` 多轮对话模式
- [x] KV cache OOM 防护
- [x] batch prefill 路径（`q35_forward_prefill` n>1）
- [x] DeltaNet chunked delta rule 集成（`dn_chunk_head`，多 chunk 循环，decay_mask bug 已修）
- [x] DeltaNet chunked prefill v-head 并行化（OpenMP，每线程独立 scratch）
- [x] FP8 ALU dequant（gather→位移拼接，bit-identical）
- [x] Attention dot product + 输出累加 AVX2 向量化
- [x] malloc 消除（attention + DeltaNet chunked scratch 预分配到 Q35Model）
- [x] 权重映射大页（`SEC_LARGE_PAGES`，无权限自动回退）
- [x] LTO（`-flto` 跨 TU 内联）
- [x] 权重预取（`PrefetchVirtualMemory`，I/O 与计算分离）
- [x] 线程数自动调优（`5/3 × 物理核`，SMT FMA 竞争优化）
- [x] 软件预取（`_mm_prefetch` matmul 内层循环提前 128B 预取）
- [x] PAL 隔离（算法文件零 `#ifdef _WIN32`，Win32 调用全收进 `q35_plat_win.c`）
- [x] mm 内核去重（4 份 AVX2 FP8 行内核合并为单一 `mm_fp8_row`）
- [x] HTTP API server（`serve` 子命令：`/health` + `/v1/completions`，非流式 + NDJSON 流式，多轮会话状态，chat 模式）
- [x] 加载进度可见（逐 shard 映射 + prefetch 进度输出，US-10）
- [x] `--incremental` 长对话 flag（`--interactive` 的规格别名，US-12）


