# kimi-k3-in-c 技术调研报告

> 调研对象：<https://github.com/FareedKhan-dev/kimi-k3-in-c>（v1.0.0）
> 目标模型：moonshotai/Kimi-K3（HF 官方 checkpoint，2026-06-13 上传，96 个 safetensors 分片）
> 本地副本：F:\project\测试PPT\kimi-k3-in-c ｜ 调研日期：2026-08-21

## 0. 摘要（TL;DR）

kimi-k3-in-c 是一个用**可移植 C99**（无 BLAS、无深度学习框架、无 GPU，二进制约 176 KB）实现的 Kimi K3 纯文本推理引擎。Kimi K3 是一个 2.78T 总参数、约 104B 激活、896 路由专家 MoE 的模型，磁盘 checkpoint 1.56 TB。该引擎的核心贡献不是"算得快"，而是一套完整的**异构流式管线（streaming pipeline）**：把权重分成三类驻留/流式区域，使最低 8 GB RAM 的普通机器也能逐 token 跑出与 224 GB 机器完全一致（逐 token 相同）的输出——内存只买速度，不买能力。

一句话管线：文本 → 手写 BPE tokenizer → 逐层前向（KDA 线性注意力 / Gated MLA / LatentMoE）→ 专家权重按需从磁盘 pread + LRU 缓存 → trunk 固定序流式 + pinned 前缀 → argmax → 下一个 token。

## 1. 背景与问题陈述

### 1.1 目标模型规模（已用 HF API 交叉验证）

| 指标 | 数值 | 证据 |
|---|---|---|
| 总参数 | 2,779,931,837,184（≈2.78T） | HF safetensors.parameters.total |
| U8（MXFP4 packed 专家权重，按字节计） | 2,722,740,830,208 | HF API |
| BF16（trunk/注意力/共享专家） | 57,179,884,544 | HF API |
| F32 | 11,122,432 | HF API |
| 磁盘用量 | 1,561 GB（1.56 TB） | HF usedStorage，与 README 完全一致 |
| 分片 | 96 个 safetensors | HF siblings |
| 下载 | 2.45 M | HF API，2026-08-20 |

模型配置要点（config.json，引擎 include/k3/k3_cfg.h 启动时读取，不允许任何缺省）：

- 93 层 = 69 层 KDA（Kimi Delta Attention，线性注意力）+ 24 层 Gated MLA（one-based 位置 4,8,...,92,93，顶部两层连续 MLA）
- hidden 7168，96 heads，词表 163,840
- MoE：896 路由专家/层，top-16，另加 2 个全宽共享专家；LatentMoE latent 宽 3584；专家中间维 3072
- 激活：SiTU-GLU（β1=4, β2=25）
- 专家权重原生 MXFP4（compressed-tensors mxfp4-pack-quantized），注意力/共享专家/lm_head 被量化 ignore（保持 BF16）
- 官方为多模态（image-text-to-text，含 vision_tower）；本引擎仅实现文本子图

### 1.2 问题定义

在 2.78T 模型上做 CPU 推理，关键矛盾是容量 vs 带宽：全部权重 1.56 TB 装不进 RAM；MoE 每 token 只激活 92 层 × top-16 = 1472 个专家；测量表明 I/O 占 wall clock 40.9–60.6%——这不是算力问题，是存储流水线问题。README 原话：8GB 配置下单 token ≈ 36s trunk 读 + 11s 专家读 + 10s 计算，"80% of a token is waiting for a disk"。

## 2. 总体架构与内存分区

### 2.1 权重三区域划分（全设计的总纲）

| 组件 | 体量 | 驻留策略 |
|---|---:|---|
| 路由专家（92 层 × 896 个 × 17.55 MB） | 1.45 TB | 永不驻留：MXFP4 packed，按需磁盘读取 + LRU 缓存 |
| 稠密 trunk（注意力、norm、共享专家等） | 108.81 GB | 可驻留可流式：pinned 前缀 + 单 ring slot，内存旋钮 |
| embed + final norm + lm_head | 4.70 GB | 恒驻留 |
| KDA recurrent state（93 层） | 626 MB | 恒驻留 |
| MLA KV cache | 2.37 MB/位置 | 仅 --incremental 时 |

内存地板 ≈ 4.7 GB + 0.63 GB + 少量 buffer + 至少 1 个专家 slot + 至少 1 层 trunk ring slot ≈ 8 GB。在此之上多给的每一 GB 都变成 trunk pin 数量或专家缓存容量，单调提速但不改输出。

### 2.2 源码模块图

```
include/k3/k3.h        public 类型、config、三条必须满足的不变量
include/k3/k3_cfg.h    config 解析：缺字段即拒绝运行
src/core/k3_ops.c      全部 kernel：RMSNorm、SiTU-GLU、ShortConv、KDA 递推、
                       Gated MLA、AttnRes、MoE、MXFP4/bf16/int8 matmul（含 AVX2）
src/io/k3_st.c         手写 safetensors reader（零依赖）
src/io/k3_load.c       每专家一次合并 pread
src/io/k3_trunk.c      trunk 流式：pinned prefix + ring buffer，O_DIRECT
src/cache/k3_cache.c   专家 LRU 缓存 + batch prefetch
src/model/k3_bind.c    按张量名把 checkpoint 绑定到层结构
src/tokenizer/k3_tok.h 手写 byte-level BPE，直接加载 tiktoken.model
src/cli/k3_run.c       主程序：内存计划、decode loop、运行报告
tools/                 Python：pack_trunk、sim_cache、make_k3_oracle、k3_ref、verify_mla、tok_parity
benchmarks/            cgroup memory ladder、split sweep
tests/                 fixtures、tiny oracle、单元测试（ops/cache/st/tok/cfg/expert/real_layer）
```

## 3. 启动管线（per-run setup）

1. **机器体检**：scripts/k3-doctor.sh 检查盘容量、内存、CPU。
2. **构建**：CMake/Makefile，C99，make test 无需权重即可跑 oracle 门禁。
3. **safetensors 索引**：k3_st.c 只读各分片 header，96 分片共 497,220 个张量在 0.74 s 内建索引；bf16→f32 的 dtype widening 用字节模式校验而非容差。
4. **config 严防死守**：k3_cfg.h 缺一字段直接 abort——"半懂的 config 会跑出架构错误的模型，比 crash 更糟"。
5. **trunk 打包**：tools/pack_trunk.py 验证每层 trunk 张量在分片内连续后，93 次顺序 range copy 合成单文件；之后加载某层 = 一次已知偏移的 pread。
6. **内存计划**：k3_run 先打印预测 plan，运行后用实测 peak RSS 回填（quote this, not the plan）。

## 4. 推理管线（per-token forward pass）

### 4.1 层间骨架

93 层逐层执行；kernel 内用 OpenMP (schedule static) 并行，保持线程数无关的归约顺序（fixtures 用 1 线程 vs N 线程同一性把关）。注意力残差（Attention Residuals）：层按 12 块切分，每层对前面各 block 的输出快照做注意力而非单一全局残差；block 边界快照并清零，token embedding 恒为第一源（layer 0 自身是边界）。

### 4.2 KDA 层（69 层，线性注意力）

逐语句复现官方 modeling 的递推：

```
q/k/v 投影 → ShortConv(融合 SiLU) → 仅 q,k 做 L2Norm
→ 每头 β = sigmoid(·)
→ decay: g = g_min · sigmoid(exp(A) · z)，g_min = -5，A 按 head 索引
→ 状态递推（顺序不可换）：1.decay 状态 S  2.从 S 读  3.写入 delta  4.从已更新的 S 读输出
→ head-wise RMSNorm → 全秩输出门
```

状态是 O(hidden²) 常量大小（不随上下文增长），93 层共 626 MB——Reduction two："attention with a memory that never grows"。

### 4.3 Gated MLA 层（24 层）

- KV 压缩进低秩 latent（512 + 64 / token / 层），缓存 latent 而非展开的 96×320 头 KV，内存省 53×，用时再展开；
- 采用 NoPE（不加位置旋转），但 64 维 rope slot 仍保留——删掉会把 head 宽从 192 变 128，静默跑出个"别的模型"；
- tools/verify_mla.py 用官方 KimiMLAAttention 类直接验证自家 torch 参考，防止"引擎对参考、参考对错了"的循环自证。

### 4.4 Stable LatentMoE（92 层路由；layer 0 为 dense FFN）

```
7168 → 3584 latent → router: 独立 sigmoid 打分（不归一化！）
→ frozen per-expert bias 只参与选 top-16；组合权重用 unbiased 分
→ 16 个专家各自计算 → 对聚合结果（而非单 expert 输出）做 RMSNorm
→ 投回 7168 + 2 个全宽共享专家（不加权相加）
```

两个陷阱被显式文档化：(1) bias 若混入组合权重，选出来的专家不变、只是分布被扰动，输出错得很难发现；(2) 可流式的只有 routed experts，2 个共享专家不流式——手算内存账就错在这。

### 4.5 MXFP4 专家 kernel（本项目的 I/O 命脉）

专家权重全程不解量化。每个专家 33,030,144 参数 = 17,547,264 字节（0.53125 B/weight）。k3_matmul_mxfp4 直接消费 packed nibble：

```
value = E2M1[nibble] · 2^(E8M0_scale − 127)     // 32 元素共享一个 8-bit scale
```

算术理由：解量化后每专家 17.5 MB → 132 MB，每 token 1472 个专家 = 194 GB 无效膨胀流量，整个流式设计就崩了。nibble 顺序（low nibble = 偶数元素）是约定，反了会"每个统计量都对、模型是错的"——专门有 fixture 防它（测试输出：PASS mxfp4 64 rows × 3584 elems, EXACT on released checkpoint bytes）。

### 4.6 采样与生成循环

采样器只有一行 greedy argmax；--incremental：step 0 喂整个 prompt，其后每步只喂 1 个 token（KDA 增量状态 + MLA latent cache）；forward 返回码检查：forward 没写 logits 就 abort，绝不在脏 buffer 上 argmax。

## 5. 两条 I/O 管线（本项目核心技术贡献）

### 5.1 Trunk 流式：pinned prefix + 单 ring slot

关键洞察：引擎每 token 以固定顺序遍历 0→92 层，这是循环扫描（cyclic scan）——LRU 的最坏情形：90 个 slot 的 LRU 在 93 层循环上命中率恰好为 0；把前 N 层直接 pin 住则得到确定性的 N/93 命中率（N=90 时 96.8%）。原话："显而易见的数据结构不只是次优，是最坏方向的错。"

实现细节：ring 只有 1 个 slot（约 2.37 GB），未 pin 层轮转复用；pinned 数与 slot 大小互相依赖，迭代 4 轮求不动点；读用 O_DIRECT 绕过 page cache；固定遍历序 ⇒ prefetch 完美，reader 线程在计算期间预读下一层，I/O 与计算重叠。

### 5.2 专家缓存：LRU + batch prefetch + 离线回放论证

- k3_load.c：一个专家一次合并 pread（不逐张量读）；
- k3_cache.c：LRU 槽（8GB 配置约 56 slot；32GB 约 341 slot）；batch prefetch 按本层 top-16 批量发起；
- 为什么不用更聪明的替换策略：tools/sim_cache.py 用 k3_cache 在真实推理中记录的 (layer, expert) trace 离线回放 LRU vs Belady（理论最优天花板）vs Pinned hot+LRU。实测 trace 上任何策略命中率天花板 90%（compulsory misses），LRU 与 Belady 差距小 ⇒ 聪明替换不值得做，工程应花在 prefetch 和 pin 上——用证据否定一个工程方向。
- 分配定律（allocation beats capacity）：同预算下大头给 trunk、小头给专家缓存（如 --trunk-gb 110 --cache-gb 13）比反过来分更优——trunk 每 token 必用 93 次，专家命中是统计性的。

## 6. 数值正确性管线（四层验证门禁）

1. **Kernel 级**（test_ops，22 个 kernel 全过）：每个 kernel 对照 fixture 参考值，容差在 manifest 声明而非硬编码；路由 fixture 的 top-2 在 6 行里换序 5 次，专抓"忽略路由 bias"的实现；SiTU-GLU fixture 把激活推到解析上界。AVX2 与标量路径的一致性不用容差验证，而是对输出比特做 FNV-1a 哈希再比对（容差会放过悄悄改变归约顺序的 kernel）。最差 kernel 只用了 8% 的允许容差，其中 mxfp4 与 bf16 两条路径是 EXACT。
2. **Layer 级**：真实 checkpoint 93 层逐层单测（test_real_layer）。
3. **Model 级（tiny oracle）**：tools/make_k3_oracle.py 生成张量图与原架构完全一致的迷你模型（13 层——作者论证这是"能区分正确实现与似是而非实现的最小深度"），C 引擎必须在 teacher-forcing / greedy / incremental 三条路径上与参考逐 token 精确一致；fixture 不含权重，make test 秒级完成。
4. **数值级**：真实 93 层 checkpoint 的 logits 与纯 torch 参考逐元素对比（错一个绑定不会差 1e-6，会差 ~1）。

tokenizer 专项：CI 里当 tiktoken.model 存在时跑 parity + byte-exact roundtrip，不存在就显式 NOT RUN 而非静默通过——"静默退化的检查比没有检查更糟，因为绿勾还是会被信"。

## 7. 实测性能数据（均来自仓库 docs/data/ 可复现数据）

### 7.1 内存阶梯（cgroup 硬上限，MemorySwapMax=0 防 swap 作弊；12 档）

| RAM (GB) | s/token | peak RSS (GB) | 专家读取 (GB) | 缓存命中率 |
|---:|---:|---:|---:|---:|
| 8 | 32.69 | 8.24 | 25.83 | 0% |
| 32 | 31.44 | 31.90 | 25.83 | 0% |
| 64 | 28.60 | 63.71 | 25.83 | 0% |
| 128 | 29.40 | 128.18 | 17.51 | 32.2% |
| 224 | 19.21 | 223.82 | 14.53 | 43.75% |

要点：8–64GB 几乎同速（trunk 流式是瓶颈，专家缓存塞不下收效甚微）；拐点出现在能 pin 下大半 trunk 之后；作者披露跑间 33% 方差带，低于此的单样本差异不算效果。

### 7.2 真实模型门禁数据（tests/fixtures/gates/gates.txt 节选）

- --trunk-gb 16 --cache-gb 6 --incremental：index 497,220 张量 0.74 s；trunk pin 10/93；2 tokens / 339.5 s；peak RSS 实测 25.83 GB；trunk 读 2883 MB/s；专家 LRU 命中 39.31%（341 slots）。
- 8GB 地板档：1 token / 134.6 s（含冷启动），peak RSS 8.73 GB。

### 7.3 同一性主张

12 档内存预算下输出 token id 序列逐位相同（memory-ladder.tsv 的 ids 列），即"内存只买速度"；examples/02-memory-budgets.sh 三次运行可复现。

## 8. 工程质量观察

- 文档刀法：README 即技术报告本体（Part IV 是测量数据），每个性能主张附可复现原始数据文件；docs/README.md 列出"三条承重主张 + 各自证据"。
- 防御式实现：config 拒绝猜测、forward 返回码检查、O_DIRECT、cgroup 关 swap 防测出 swap 带宽。
- 反驳自己的工程方向：docs/notes/int8-draft-container.md 记录了一个被实测否定的混合 int8 draft 方案，失败实验也写复盘。
- 零外部依赖：JSON、safetensors、tiktoken BPE 全部手写/vendored；CI 覆盖 C 测试、Python lint、tokenizer parity。
- 局限：采样只有 greedy；仅 Linux x86-64；KDA 递推仍是标量 C（ROADMAP 列为最大未向量化非 I/O kernel）；无 vision 子图；8GB 档 ~30 s/token 仅适合研究/验证，不具 serving 意义。

## 9. 结论与可借鉴经验

**结论**：kimi-k3-in-c 是工程完成度和证据链都高于平均水平的纯 C 推理引擎。核心解法归纳为四条 Reduction：专家原生 4bit 不解量化；KDA 定量状态替代增长 KV；MLA latent 压缩 KV；trunk 从地板变旋钮。共同思想是把模型体积问题转化为"什么必须驻留、什么可以流式、流式如何不被缓存策略坑"的存储调度问题，且每个决策都用实测 trace 或对照实验论证。

值得借鉴的做法：把 I/O 带宽当一等设计参数（固定遍历序、合并 pread、不解量化）；用 Belady 界限论证"不做某个优化"；AV2/标量多路径按比特哈希而非容差验证；tiny oracle 三路径精确对齐能暴露"看起来对"的架构误读（bias 混入权重、nibble 反序等）；性能汇报附方差带，低于阈值不算效果。

**一句话评价**：模型是真的（moonshotai/Kimi-K3，HF 已核验 2.78T），数字可复核，设计思想（MoE 时代 CPU 推理 = 存储流水线优化）对端侧/低资源大模型推理有直接参考价值；定位是概念验证与研究基线，不是生产 serving 工具。

## 附录 A：关键证据文件清单

- 架构：docs/ARCHITECTURE.md
- 测试：docs/TESTING.md、tests/fixtures/gates/gates.txt
- 性能：docs/PERFORMANCE.md、docs/data/memory-ladder.tsv、docs/data/replication.tsv
- 主文档：README.md（自含 Part I–IV 完整论证）
- 官方模型卡：moonshotai/Kimi-K3（HF API 已核验总参数 2,779,931,837,184）
- 失败实验复盘：docs/notes/int8-draft-container.md
