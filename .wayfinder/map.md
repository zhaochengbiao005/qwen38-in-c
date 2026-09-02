# Map: Windows 版 Qwen3.8-27B-FP8 纯 C 推理引擎（基于 kimi-k3-in-c 改造路线）

## Destination

一份可直接开工的改造规格：在 Windows x86-64 上，用零依赖 C99 引擎（以 kimi-k3-in-c 为蓝本）加载 `Qwen/Qwen3.8-27B-FP8`（safetensors，e4m3 block-wise 128x128），文本子图逐 token 推理，输出与 HF torch 参考一致。

**Destination 已达成**（2026-08-25）：引擎端到端跑通，12 ctest + 20 prompt 回归集全绿，decode 1.07 tok/s（20 线程自动调优，比 HF bf16 快 48 倍），prefill 5 tok 4.3s / 75 tok 42.2s（0.56 s/tok），greedy + temperature/top-k/top-p 采样 + 交互多轮对话模式 + KV cache OOM 防护 + 完整 README。chunked delta rule 已集成到 prefill 主路径（`dn_chunk_head`，decay_mask 方向 bug 已修），多 chunk 循环支持任意长度 prompt（chunk_size=64，块间状态按末期 g 衰减累加）。

**性能优化**（2026-08-25）：FP8 gather→ALU 位移拼接（bit-identical）、attention dot/累加 AVX2 向量化、DeltaNet chunked prefill v-head 并行化、malloc 消除（attention+DeltaNet scratch 预分配）、权重映射大页（`SEC_LARGE_PAGES`）、LTO（`-flto`）、权重预取（`PrefetchVirtualMemory`）、线程数自动调优（`5/3 × 物理核` = 20，SMT FMA 竞争优化）、软件预取（`_mm_prefetch` matmul 内层）。优化前后：短 prompt prefill 20.5→4.3s（4.8x）、decode 0.68→1.07 tok/s（1.57x）、长 prompt prefill 99.7→42.2s（2.36x）。

## Spec

- 主规格文档：[.wayfinder/spec.md](spec.md)（Status: ready-for-agent）。剩余 frontier tickets（05/06/07/09）的答案进入本规格的修订。

## Notes

- 领域：C99 推理引擎移植 / LLM 架构逆向对齐 / Windows 系统编程。
- 蓝本工程：kimi-k3-in-c（本地副本 F:\project\测试PPT\kimi-k3-in-c，调研报告见仓库根目录）。
- 每个会话开始前先读本文件；解决 ticket 时按需 zoom 其正文。
- 技能：$seagull-reverse（kernel/格式逆向对齐）、$seagull-lab（复现与证据）。
- 子代理原始佐证件（config/index/modeling 源码快照）在 `.wayfinder/research/` 下，勿重复抓取。

## Decisions so far

- [Ticket-01: FP8 checkpoint 权重架构盘点](tickets/01-weight-schema-inventory.md)：dense 而非 MoE（mlp.gate 是 FP8 模版残留）；407 个 block-wise FP8 张量 + 对应 BF16 scale，scale shape=[ceil(rows/128), ceil(cols/128)]；embed/lm_head/norm/vision 保留 BF16；总 27.78B 参数 ≈ 28.75 GiB on-disk；分片按层一文件（layers-0..63）→ bind 层需处理。详见 research/weight-schema.md。
- [Ticket-02: Windows 平台层映射与工具链决策](tickets/02-windows-platform-mapping.md)：POSIX 面仅 4 个文件、易映射；madvise(HUGEPAGE) 需在 VirtualAlloc 时决策、posix_fadvise 退化为 no-op；工具链选 clang-cl + CMake + Ninja（MSVC 不定义 __AVX2__ 是静默退化陷阱）。详见 research/windows-platform.md。
- [Ticket-03: Qwen3.5 文本架构数值规格](tickets/03-qwen35-numerical-spec.md)：DeltaNet 递推公式（g=-exp(A_log)·softplus(a+dt_bias)，state fp32，16 key 头 repeat 到 48 value 头）、full attention（RoPE 仅前 64 维、θ=1e7、q_proj 带 sigmoid output gate）、RMSNorm 零中心参数化 (1+w)、dense SwiGLU、MTP 可忽略。详见 research/qwen35-numspec.md。
- [Ticket-04: 目标硬件与内存地板](tickets/04-target-hardware-memory-floor.md)：基准机 96GB RAM + Ryzen 9 5900X（AVX2/FMA，无 AVX-512）+ NVMe；定案**全驻留设计**，内存地板按 32GB 机设计，SIMD 只需 AVX2。
- [Ticket-08: 代码库策略](tickets/08-repo-strategy.md)：**新建干净仓库**，从 kimi-k3-in-c 拷贝 safetensors reader / LRU 框架 / 测试脚手架。

- [Ticket-06: Tokenizer 路径](tickets/06-tokenizer-path.md)：选 **A 变体**——离线转换工具产出自定义 binary 	okenizer.bin，引擎只加载 blob 跑 byte-level BPE，零 JSON 依赖。实施票：#10 (converter) → #11 (loader) → #12 (BPE encode + HF byte-exact) → #13 (tokenize CLI)。

- [#10 Tokenizer 转换工具](tickets/10-tokenizer-converter.md)：converter + tokenizer.bin 落地，9 fixture 与 HF tokenizers byte-exact；发现 NFC normalizer（已固化进 blob flags，#11/#12 必须支持）。

- [#11 Tokenizer blob 加载器](tickets/11-tokenizer-loader.md)：qwen35-c 仓库骨架 + PAL + loader + 单测全绿；工具链 clang-cl 22 + VS2026 BT + SDK 19041 实机验证；负样本拒绝正确。

- [#12 BPE encode 路径](tickets/12-bpe-encode.md)：440/440 fixture 与 HF byte-exact 过验（含 NFC/Hangul/emoji/长文）。NFC 落在生成表 + 算法式 Hangul；compose 表排序陷阱已记入票内。

- [#13 tokenize CLI](tickets/13-tokenize-cli.md)：qwen35.exe tokenize 接缝落地，HF 对表逐位一致，退出码 2/3/4。

- [Ticket-05: FP8 kernel 策略](tickets/05-fp8-kernel-strategy.md)：选 **B**——FP8+scale 常驻（≈25 GB），matmul 内层逐 128x128 块 on-the-fly dequant，不物化 BF16 副本。ADR-0001。

- [#14 safetensors reader + config](tickets/14-safetensors-reader.md)：真实权重目录全量索引（1606 张量）+ 严格 config 解析通过；合成 fixture + CRC 校验。
- [#17 DeltaNet 线性注意力层](tickets/17-deltanet-layer.md)：48 头 delta 递推 + conv1d(k=4) + 门控归一化全链路，decode/prefill bitwise 一致；fixture 全绿。
- [#18 GQA 全注意力层 + KV cache](tickets/18-gqa-attention.md)：q 门 + q/k-norm + partial RoPE + GQA 24:4 + KV cache，prefill/decode bitwise 一致（scalar 与 AVX2 同 FNV），全绿。
- [#19 Forward 总装](tickets/19-forward-assemble.md)：64 层混合架构装载+前向落地；mini 模型三路径（prefill/decode）对齐 fp64 参考绿；真实 27B smoke：load 3.7s、2-token prefill 46s、有 logits。全测试 8/8。
- [#21 run CLI](tickets/21-run-cli.md)：qwen35 run 打通 编码→prefill→greedy/采样→流式解码 全链路。真实 27B：load 3.2s，prefill 5tok 59s，decode ~0.09 tok/s，RSS 25.9GB，greedy 输出 'Beijing.' —— 语义正确。**端到端可用**。老 bug（tokenize prompt[n]=0 越界）已闭案：q35_plat_read_file 改为 n+1 分配并 NUL 收尾；2026-08-23 复核补修两处遗留——q35_plat_win.c 中 PowerShell 补丁污染残留的字面量 `r`n（曾致编译阻断）与 q35_cli.c prompt_file 双读泄漏；ctest 8/8 绿。
- [#23 OpenMP 并行 matmul]：三个 matmul（fp8 avx2/scalar + bf16）行循环加 `#pragma omp parallel for schedule(static)`，不改变每行 8-lane 归约序（FNV 比特一致）。根因：clang-cl 22 的 `/openmp:llvm` 被静默忽略（`_OPENMP` 不定义），改用 `-openmp` + 显式链接 `libomp.lib`（8.3 路径绕空格）+ build.cmd 自动部署 libomp.dll。**11.5→1.72 s/tok（6.7×），比 HF bf16 快 26×**。ctest 9/9 无回归。
- [#24 深层并行化]：DeltaNet 48 v-head 循环并行（per-thread scratch）、attention 24 query head 循环并行（per-thread sc buffer）、swiglu elementwise 并行、multi4 FP8 matmul kernel（DeltaNet 4 投影 qkv/z/a/b 合并 + MLP gate/up 合并，一次 fork/join 处理多组权重）。加 `Q35_PROF=1` 环境变量门控的 per-phase 计时。**1.72→1.59 s/tok**，从单线程 11.5 算起 **7.2× 总加速**。MLP 已逼近理论极限（实测 1.03s vs 理论 ~1.0s，DRAM 带宽受限）。ctest 9/9 无回归。
- [#25 采样策略]：`pick_token` 升级为 temperature + top-k + top-p（nucleus）三合一采样器。CLI 加 `--top-k`/`--top-p` 参数。验证：greedy(temp=0) 确定性、top-k=1 等价 greedy、不同 seed 产出不同序列、top-p=0.9 输出语义合理。`run_cmd.c` 内含 xorshift32 RNG + softmax/temp + 排序 + top-k 截断 + top-p 累积截断 + 重归一化采样。
- [#26 多 prompt 回归集]：20 条 diverse prompt（QA/代码/翻译/数学/创意）× HF 参考对拍。工具：`tools/gen_regression_ref.py`（HF 侧）、`tools/run_regression.c`（引擎侧，CMake target `run_regression`）。每条验 greedy ids 精确一致 + 末 token logits maxabs≤0.15 或 cos>0.999。fixture 在 `tests/gates/real-22/regression/`。20/20 ALL PASS。
- [#30 --threads CLI 接通]：`--threads N` 参数调 `omp_set_num_threads(N)`，不给时 fallback 到 `OMP_NUM_THREADS`，再不给用全部核心。启动打印实际线程数。
- [#29 README]：完整 README.md（概述/构建/用法/测试/性能/限制/架构/设计决策/致谢）。
- [#27 DeltaNet chunked prefill]：batch matmul kernel（`q35_mm_fp8_batch`，Y[rows,n]=W@X^T，AVX2+OpenMP，比特一致 ctest `batch_mm`）+ DeltaNet batch 投影（`q35_dn_forward_s(seq_len>1)`）+ `q35_forward_prefill` batch 路径（batch embed/rmsnorm + DeltaNet batch 投影 + **MLP batch matmul** + flat swiglu/residual）。prefill 5 token 8.8→7.8s，36 token 1.28 s/tok（比 decode 1.58 快）。ctest 11/11 全绿。
- [#28 KV cache OOM 防护]：load 时 `GlobalMemoryStatusEx` 查可用内存，KV cache + 权重 26GB 超过 80% 可用就自动降 `kv_cap`。96GB 不触发，32GB 地板机自动降到 ~4K。动态增长已实现（初始 cap 4096 翻倍至 max_cap，`q35_kvcache_grow`，load_state 超容量自动扩，2026-09-02 票据同步）；ring buffer 留后续。
- [#31 大页 / 交互模式]：大页接线修正为"权限可用即用，否则 calloc"（2026-09-02）：`q35_plat_large_alloc` 无 `SeLockMemoryPrivilege` 时返回 NULL（删除了实测慢 72% 的 VirtualAlloc 替身），`q35_kvcache_init/grow` 回落 calloc 并以 `k_large`/`v_large` 标记匹配 free 路径。`--interactive` / `-i`（规格别名 `--incremental`，US-12）多轮对话模式落地：模型只加载一次，每轮 stdin 读输入 → chat 包装 → prefill → decode → im_end 回写 → 状态累积不 reset。验证两轮对话正确响应。
- [#15 FP8 dequant matmul kernel](tickets/15-fp8-dequant-matmul.md)：FP8+scale 常驻 on-the-fly dequant matvec + BF16 matvec，AVX2/标量同构 8-lane 归约（FNV 锚定），8 case fixture 全绿；qkvlike 10240x5120 maxrel 1.7e-2 在 atol+rtol 门内。
- [#20 tiny oracle](tickets/20-tiny-oracle.md)：三路径数值验收落地（ctest `oracle`）：TF 14 位置 maxabs 2.6e-5（容差声明在 oracle.json manifest）、greedy/incremental ids 逐位一致、prefill==incremental 比特一致。8 层混合 mini 模型（FP8 边块网格）+ numpy f64 参考。**过程中抓到两个参考侧 bug（KV cache 跨层共享、RMSNorm 跨头全局平均），引擎实现两次自证清白**；#19 生成器的同款 norm bug 已同步修复。Q35_DUMP/ORACLE_DUMP + cmp_dump.py 逐算子对拍基建入库。
- [#22 真实模型门禁](tickets/22-real-model-gate.md)：**PASS（2026-08-24 首跑即过）**。真实 27B FP8：引擎 vs HF transformers 参考（5.3.0 手动反量化加载，数值语义与 numspec 快照等价已 diff 存证）greedy ids 12/12 逐位一致，12 行 logits top-1 全对、maxabs≤0.135、cos≥0.99996（numspec 端到端双容差全过）。基线：引擎 load 3.4s / 11.5s/tok / RSS 25.3GB vs HF bf16 95.7s / 44.7s/tok / 58GB。证据 tests/gates/real-22/；模型目录含 67 个 HF 可见的硬链接（勿删）。
- [#16 元素类 kernel](tickets/16-elementwise-kernels.md)：rmsnorm(零中心)/silu/swiglu/qk-norm/RoPE(rotate_half, partial)/attn gate/softmax 落地，10 case 全绿，AVX2 与标量 FNV-1a 逐 case 一致（FP_CONTRACT OFF + 固定归约序）。
- [#32 PAL 隔离](tickets/32-pal-isolation.md)：**CLOSED（2026-08-25）**。`q35_st.c`/`q35_model.c` 的 Win32 调用（目录枚举、mmap、64-bit seek、内存查询）全部收进 PAL（`q35_plat_dir_*`/`q35_plat_mmap_ro`/`q35_plat_fseek64`/`q35_plat_avail_phys`），算法文件零 `#ifdef _WIN32`。
- [#33 mm 内核去重](tickets/33-mm-kernel-dedup.md)：**CLOSED（2026-08-25）**。`q35_mm.c` 四份重复 AVX2 FP8 行内核合并为单一 `mm_fp8_row`，三个入口（`q35_mm_fp8`/`q35_mm_fp8_batch`/`q35_mm_fp8_dual`）改为调用它；文件 427→235 行，AVX2 intrinsics 从 4 份降到 1 份，ctest 12/12 绿。
- [#34 性能优化](tickets/34-perf-optimization.md)：**CLOSED（2026-08-25）**。7 项优化：(1) FP8 gather→ALU 位移拼接（bit-identical，subnormal 回退 LUT）；(2) attention `dot_fixed`+输出累加 AVX2 向量化（mul+add 非 FMA）；(3) DeltaNet chunked prefill v-head `#pragma omp parallel for`（每线程独立 scratch）；(4) malloc 消除——attention 6 buffer + DeltaNet chunked 13 buffer 预分配到 `Q35Model`；(5) 权重映射 `SEC_LARGE_PAGES`（TLB ~6.3M→12.4K，无权限自动回退）；(6) `-flto` 跨 TU 内联；(7) 权重预取 `PrefetchVirtualMemory`（I/O 与计算分离，prefill 不再被 page-fault 打断）。实测：短 prompt prefill 4x、decode 1.5x、长 prompt prefill 1.85x。未做：多累加器（需放宽 FNV 契约）、expf 预计算、attention prefill fork/join 优化。
- [#35 HTTP API server](tickets/35-http-api-server.md)：**CLOSED（2026-09-02）**。`serve` 子命令：极简 HTTP/1.1 loop（winsock2，零依赖），`GET /health` + `POST /v1/completions`（prompt/max_tokens/temperature/top_k/top_p/stream/chat/reset），非流式 JSON + NDJSON chunked 流式，多轮会话状态（DeltaNet+KV 延续），chat 自动包装。复用 `q35_json.c` DOM 解析请求。实测 curl 全通。

## Not yet specified

- （无）DeltaNet chunked delta rule 已集成到 prefill 主路径（`dn_chunk_head`，decay_mask 方向 bug 已修，测试改容差检查）。
- （无）多轮对话状态序列化已实现（`q35_model_save_state`/`q35_model_load_state`，magic `Q35S`，CLI `--save-state`/`--load-state`，ctest `state` bitwise roundtrip 验证）。2026-09-02 起随 spec.md Phase-2 Extensions 一并规格化。

## Out of scope

- vision tower / 多模态（文本子图定向，与 K3 引擎同策略）。
- Linux/macOS 支持（本图只管 Windows）。
- serving / 并发 / API 封装。
- BF16 原版的完整等价支持（FP8 为唯一验收格式；日后可作为增强）。
- MoE 支持——Ticket-01 与 Ticket-03 交叉确认该模型是 dense，K3 的 1.45TB 专家子系统整块砍掉（无需单独立票）。
- MTP 投机解码（一阶段忽略）。
- 流式权重管线（后置增强，全驻留设计已定）。
- AVX-512（5900X 无此指令集，需换硬件）。
- GPU 支持（项目 scope 是 CPU only）。
- GGUF/llama.cpp 互操作（纯独立格式，直接读官方 FP8 safetensors，#22 已验证）。









