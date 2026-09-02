# ADR-0001: FP8 权重常驻内存，matmul 内层逐块 dequant

状态：Accepted（2026-08-22）

背景：Qwen3.8-27B-FP8 权重为 block-wise FP8-E4M3（128x128 块 + BF16 scale），on-disk 23 GiB。基准机 96 GB RAM（Zen3，AVX2，无 AVX-512），内存地板目标 32 GB。

决策：不把权重 dequant 成 BF16 常驻（56 GB，违反地板），而是常驻 FP8+scale 原格式，matmul 内层把 e4m3 → f32 转换和 block scale 乘法融进计算内循环，从不物化 BF16 完整副本。

理由：CPU 上 FP8 不省算力只省带宽/容量；稠密模型每 token 读完全部权重，内存带宽恰是瓶颈；32 GB 地板是硬约束。

后果：matmul kernel 复杂度增加（两种权重路径：FP8 块路径 + BF16 直通路径——embed/norm/lm_head 保持 BF16）；e4m3 转换语义由 kernel fixture 与 HF 参考锁住；KV cache dtype 另议。
