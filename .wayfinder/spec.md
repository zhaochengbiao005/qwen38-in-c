# Spec: Windows 版 Qwen3.8-27B-FP8 纯 C 推理引擎

Status: ready-for-agent

来源：由本会话上下文综合生成（to-spec），决策依据见 `.wayfinder/tickets/01-09` 与 `.wayfinder/research/`。

## Problem Statement

Qwen3.8-27B 是 2026 年新架构（Qwen3.5：Gated DeltaNet 线性注意力 + GQA 混合，48:16 交替层），用户想要一个零深度学习框架依赖、可审计、可控的本地推理引擎，在 Windows x86-64 上直接加载官方 FP8 safetensors 跑出与 HF 参考逐 token 一致的输出。现有工具要么依赖 Python/PyTorch 全家桶，要么（llama.cpp）是 GGUF 转换链黑盒，无法满足研究/审计诉求。

## Solution

新建干净仓库（不含蓝本 git 历史），以 kimi-k3-in-c 的设计思想为蓝本，用可移植 C99 实现：
- 手写 safetensors reader + FP8-E4M3 block-wise (128x128) 反量化
- 64 层文本子图前向（48 层 Gated DeltaNet + 16 层 full attention GQA）
- 全驻留权重加载（模型 28.75 GiB，基准机 96 GB RAM），内存地板按 32 GB 机器设计
- 贪心解码 CLI，token 序列可作为与 HF 参考的验收接缝
- 可选 tokenizer 处理 Qwen 词表（248320，特殊 token：bos=eos=248044 等）

## User Stories

1. As a 研究人员, I want 在 Windows 上运行 `qwen35.exe run --model <目录> --prompt "..."`，so that 无需任何 Python 环境即可得到推理输出。
2. As a 研究人员, I want CLI 输出逐 token 的 token id 序列（stderr 或 --dump-ids），so that 能与 HF torch 参考逐位比对。
3. As a 审计者, I want 引擎读的是官方 FP8 safetensors 原始文件，so that 不存在中间格式转换引入的数值漂移盲点。
4. As a 审计者, I want config.json 缺字段或非预期值时引擎直接拒绝运行，so that 不会出现静默用错架构参数。
5. As a 开发者, I want kernel 级单元测试（FP8 dequant、DeltaNet 递推、RoPE、q/k norm、out-gate、SwiGLU、RMSNorm），so that 每个算子可独立验证。
6. As a 开发者, I want AVX2 与标量路径输出做 FNV-1a 比特哈希比对，so that SIMD 偷偷改变归约顺序能被抓住。
7. As a 验证者, I want 一个 Qwen3.5 拓扑的 tiny oracle 模型 + 三路径（teacher-forcing / greedy / incremental）逐 token 对齐测试，so that 小于 1 分钟内能验证架构实现无歧义。
8. As a 验证者, I want logits 级对比脚本（真实 checkpoint vs torch 参考逐元素），so that 端到端数值偏差有量化证据。
9. As a 用户, I want tokenizer 与 HF tokenizer byte-exact roundtrip 对齐，so that prompt 边界不影响横向评估。
10. As a 用户, I want 模型加载耗时进度可见，so that 28.7 GiB 加载不是黑屏等待。
11. As a 用户, I want 在 32 GB RAM 的机器上能跑（全驻留），so that 不需要 96 GB 的基准机。
12. As a 用户, I want `--incremental` 长对话下 DeltaNet 状态与 KV cache 正确演进，so that 多轮会话不退化。
13. As a 开发者, I want 平台层（k3_plat 风格的 PAL）把 Win32/Clang-cl 差异封装在独立模块，so that 算法文件零 `#ifdef _WIN32`。
14. As a 开发者, I want CMake + Ninja + clang-cl 一键构建脚本，so that Windows 上 `build.cmd` 即可出二进制+测试。
15. As a 维护者, I want 蓝本来源在 README 中注明并可追溯，so that 版权与思路归属清晰。

## Implementation Decisions

全部决策有据可查（括号内为 ticket 指针）：

1. **仓库策略**：新建干净仓库，从 kimi-k3-in-c 抄用以下可复用件——safetensors reader（Windows 化改造）、LRU 缓存框架（全驻留下裁剪为 arena/分配器）、测试脚手架（fixture 容器、FNV-1a 哈希门、tiny oracle 生成器思路）。k3 的 1.45TB 专家流式子系统整体不抄（目标模型是 dense）。（Ticket-08）
2. **工具链**：clang-cl + CMake + Ninja，OpenMP 用 `/openmp:llvm`。MSVC 备选但需补 `K3_ENABLE_AVX2` 探测宏——MSVC `/arch:AVX2` 不定义 `__AVX2__` 是静默退化陷阱。MinGW 排除。（Ticket-02）
3. **SIMD 上限**：AVX2 + FMA only（基准机 Ryzen 9 5900X 无 AVX-512），不追 AVX-512。（Ticket-04）
4. **内存形态**：全驻留默认。on-disk 28.75 GiB（FP8 权重 23.00 GiB + scale + BF16 保留区 5.74 GiB）；内存地板按 32 GB 机器设计；流式管线为后置增强，不进一阶段验收。（Ticket-04）
5. **数值规格**（ground truth = HF `modeling_qwen3_5.py` 快照，`research/qwen35-numspec.md`）：
   - DeltaNet：conv1d k=4 causal + SiLU；`g=-exp(A_log)·softplus(a+dt_bias)`；`beta=sigmoid(b)`；状态 S fp32，`S←exp(g)S + k⊗(β(v-Sᵀk))`；q/k per-head L2 norm；16 key 头 repeat→48 value 头；输出 per-head RMSNormGated × SiLU(z)。
   - Full attention：head_dim 256，24:4 GQA；per-head q/k RMSNorm 在 RoPE 前；RoPE 只转前 64 维（32/32 rotate_half），θ=1e7；q_proj 附带每头 sigmoid 输出门，乘在 o_proj 之前；softmax fp32。
   - RMSNorm 零中心参数化 `y=(x/rms)·(1+w)`，eps 1e-6。
   - MLP dense SwiGLU（5120→17408→5120）。
   - MTP 层一阶段忽略。
6. **权重 schema**：407 个 FP8 张量各配 BF16 `weight_scale_inv`（shape=[ceil(rows/128), ceil(cols/128)]）；embed/lm_head/norm/vision 保留 BF16；分片为「每层一文件」（layers-0..63.safetensors）→ bind 层按层索引文件，不做跨层连续映射。（Ticket-01）
7. **FP8 kernel 策略**（Ticket-05 待定，候选：A 加载时整块 dequant→BF16 常驻；B in-memory 保 FP8 逐块 dequant；C 常驻 dequant+流式保 FP8 的混合。决策输入：全驻留后纯省 RAM 的诉求 + dequant 代码复杂度。倾向 C→简化版 A/B 二选一，开工 session 定）。
8. **Tokenizer 路径**（Ticket-06 待定，候选：A 一次性转换 tokenizer.json→tiktoken 格式复用蓝本 BPE 引擎；B 引擎内写 JSON BPE 解析；C vendor 极简 BPE）。
9. **测试金字塔**（Ticket-07，四层）：kernel fixture（容差声明式 manifest）；真实层单测（离线手动门禁，CI 跑动）；tiny oracle 三路径精确对齐；真实 checkpoint logits 逐元素对比（本地手动，torch 参考 dump 环境待定）。
10. **平台抽象层**：POSIX 面仅 4 个文件（open/pread/O_DIRECT/posix_memalign/madvise/clock_gettime/getrusage/proc/meminfo/opendir），全部收敛进 PAL；madvise(HUGEPAGE)→`VirtualAlloc(MEM_LARGE_PAGES)` 分配期决策；`posix_fadvise` 退化为 no-op。（Ticket-02）

## Testing Decisions

- 好测试的定义：只测外部行为（kernel 输出向量 / CLI token 序列 / logits），不测内部状态；所有容差写在 manifest/fixture 里而不是硬编码在断言中（沿用蓝本风格）。
- 接缝：最高接缝 = CLI 的 token id 序列输出；中间 = 单层前向函数（纯函数式签名）；最低 = 单 kernel。
- 被测模块：FP8 dequant→matmul 链、DeltaNet 递推、full attention、SwiGLU、RMSNorm、RoPE、tokenizer、CLI 集成。
- Prior art：蓝本 kimi-k3-in-c 的 `tests/fixtures/`、`tools/make_k3_oracle.py`、FNV-1a 多路径哈希比对——直接平移方法论，fixture 需为 Qwen3.5 拓扑重新生成。
- tokenizer 测试语义：fixture launch 时若无模型参考资源必须显式 NOT RUN 而非静默通过（沿用蓝本原则）。

## Out of Scope

- vision tower / 多模态
- Linux/macOS
- serving、并发、API 封装（→ 已由 Phase-2 扩展修订，见下节 #35）
- BF16 原版完整支持（FP8 唯一验收格式）
- MoE（模型为 dense，已证）
- MTP 投机解码（一阶段忽略）
- 流式权重管线（后置增强）
- 采样策略扩展（一阶段 greedy only）（→ 已由 Phase-2 扩展修订，见下节 #25）

## Phase-2 Extensions（2026-09-02 修订）

一阶段验收全绿后，以下能力经 wayfinder 票据逐项立项落地。它们是对上节对应 Out of Scope 行的显式修订（有票可查），不是静默越界：

- **采样策略**（#25）：temperature + top-k + top-p（nucleus）采样器，CLI `--temperature`/`--top-k`/`--top-p`/`--seed`。greedy（temp=0）保持默认且与参考逐位一致。
- **serving**（#35）：`serve` 子命令，localhost 单并发 HTTP/1.1（`/health` + `/v1/completions`，非流式 JSON + NDJSON chunked 流式）。
- **多轮对话状态序列化**：`--save-state`/`--load-state`（magic `Q35S`，ctest `state` bitwise roundtrip 验证）。
- **`--threads` 线程控制**（#30）与**多 prompt 回归集**（#26，20 条 × HF 参考对拍）。
- **大页**（#31 最终决策）：KV cache 优先 `MEM_LARGE_PAGES`，无 `SeLockMemoryPrivilege` 时回落 calloc——PAL 不做 VirtualAlloc 替身（demand-zero 实测比 calloc 慢 72%）。
- **KV cache 动态增长**（#28 修订）：初始 cap 4096，按需翻倍至 max_cap；load_state 超容量自动扩容。

其余 Out of Scope 条目（vision/多模态、Linux/macOS、BF16 完整支持、MoE、MTP、流式权重管线）不变。

## Further Notes

- 风险首位：DeltaNet 递推的 fp32 状态对齐 + clamp/离散化细节，是 "看起来对但数值漂" 的高发区；tiny oracle 是防线的关键。
- 风险次位：tokenizer 248320 词表的 BPE 合并规则边界（Qwen 的特殊 token 与 byte fallback）。
- 蓝本版权声明与致谢进 README；不复制蓝本完整代码，只抄可复用件并改写。
