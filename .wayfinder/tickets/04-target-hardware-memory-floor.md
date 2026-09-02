# Ticket-04: 目标硬件与内存地板（grilling, HITL）

Label: wayfinder:grilling
Claim: 未认领（等待用户）
Blocked by: 无（frontier）

## Question

给这个 Windows 版本定目标机器规格：设计目标的内存地板是多少 GB（16 GB？32 GB？只考虑开发者机？）、是否要支持无 AVX-512 的纯 AVX2、磁盘类型（NVMe vs SATA，决定流式读取的 floor）。这些答案直接决定 trunk streaming/expert 缓存设计是否存在（26B 级别的模型在 Windows 上全驻留是否默认成立——全驻留就没必要复刻 K3 的流式管线，engineering 形态完全不同）。

## Resolution
（2026-08-21 用户直接用本机配置定案，CLOSED）

基准机实测：
- RAM：96 GB（当前空闲 56 GB）
- CPU：AMD Ryzen 9 5900X（Zen 3，12C24T；**有 AVX2+FMA，无 AVX-512**）→ SIMD 上限锁定 AVX2
- 存储：多块 NVMe/SATA SSD；模型拟放 NVMe（F: 盘 2TB Kioxia，空闲 344 GB），28.75 GiB 权重轻松装下

结论：
1. **全驻留默认成立**：28.75 GiB 权重对 96 GB 机器不构成压力，K3 式 trunk 流式管线**不是**本工程的第一形态；改为「全量常驻 + 可选内存旋钮」。
2. 内存地板按 **32 GB RAM 机器**设计（权重 28.75 GiB + KV/状态/buffer ≈ 2-3 GiB），流式降级路径后置为增强项，不进一阶段验收。
3. SIMD 只需 AVX2+FMA，不追 AVX-512。
4. I/O 假设为 NVMe（顺序读带宽充裕），加载耗时不是主要矛盾——预测模型加载 < 60s 即可接受。
