# Ticket-05: FP8 kernel 策略（task, AFK）

Label: wayfinder:task
Claim: 未认领
Blocked by: Ticket-01, Ticket-02, Ticket-04

## Question

FP8 block-wise 权重的 matmul 怎么做：A) 加载时整块 dequant 成 BF16 再进现有 bf16 matmul（牺牲内存收益换实现简单）；B) 磁盘上保持 FP8，逐块 on-the-fly dequant 进 cache line 级计算（继承 K3「不解量化」思想，省一半 I/O 带宽与内存）；C) 折中：常驻层 dequant、流式层保持 FP8。取决于 Ticket-01 的 scale 布局、Ticket-02 的 SIMD 能力结论、Ticket-04 的内存目标。

Claim: 已认领（2026-08-22）

## Resolution
（2026-08-22 定案，CLOSED）

选 **B：in-memory 保持 FP8，matmul 内层逐 128x128 块 on-the-fly dequant**。

算账过程：
- **A（加载时 dequant 成 BF16）**：常驻 ≈ 56 GB 权重 + KV + 激活，直接捅穿 Ticket-04 定下的 32 GB 地板（只能在 ≥64 GB 机器上跑）。地板是硬约束，A 出局。
- **B（FP8 常驻 23 GiB + scale）**：常驻 ≈ 25 GB（含 BF16 norm/embed/lm_head 区）+ KV cache + 激活 buffer，32 GB 机器可跑长上下文直到 KV 吃满。基准机 96 GB 宽裕。
- **C（混合）**：全驻留下「常驻层 vs 流式层」的区分已不存在（Ticket-04 砍掉了流式），C 失去意义。

B 在 CPU 上的正确成本模型：CPU 没有 FP8 张量指令，dequant 不省算力，省的是 **DRAM 带宽与容量**——而稠密 27B 每 token 都要读完全部权重，内存带宽正是 Zen3 上的瓶颈之一（23 GB vs 56 GB 每 token 读取量直接差 2.4 倍）。dequant 融进 matmul 内层（加载 8×int8 → 转 f32 → 乘 block scale → FMA），避免物化 BF16 副本。AVX2 上 e4m3→f32 用 byte→u16 shift 后位运算摊牌成 fp32（符号/指数/尾数重排 + lut 或移位构造），每 128x128 块结束时乘 scale。

约束与后续：
- e4m3→f32 的转换表/公式必须符合 MX 规范的 round-to-nearest-even 语义；验收走 kernel fixture（Ticket-07 的范围）。
- scale 是 BF16，乘法前转 f32 免费（<<16）。
- KV cache dtype 不在本票范围（另立；默认 BF16，256k 上下文约 16 GB，或后续量化）。
- 加载峰值内存 ≈ 一次性 read + 解码重排，要控制 < 常驻 + 1 层大小。
