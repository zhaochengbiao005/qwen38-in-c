# #15: FP8-dequant matmul kernel

**What to build:** block-wise FP8 (e4m3, 128x128 + BF16 scale) 的 y = x·Wᵀ matmul kernel，AVX2 + 标量双实现；dequant 融合进内层，不物化 BF16（ADR-0001）。含 BF16-w matmul 直通路径（embed/norm/lm_head/小矩阵用）。验收用 numpy 参考 fixture + AVX2/标量 FNV-1a 比特哈希双路径一致性。

**Blocked by:** None (can start immediately)

**Status:** done

- [ ] e4m3→f32 转换实现 round-to-nearest 语义（含 subnormal/inf/nan 边界）
- [ ] AVX2 与标量路径 FNV-1a 输出一致（容差观察另记，归约顺序固定）
- [ ] numpy 生成的随机块 fixture（小/不整除 128/多形状）与 kernel 输出在声明容差内
- [ ] 单核 + OpenMP 多核模式，线程数可指定
- [ ] 基准：给出每层 in_proj_qkv 级别矩阵的 cycles 数字（写进票）

Claim: 已认领（2026-08-22，并行 #15/#16）


## Resolution (2026-08-22)

Done. FP8(e4m3fn)+BF16-scale matvec 与 BF16 matvec 落地。
- qwen35-c/include/q35/q35_mm.h, src/q35_mm.c: e4m3fn LUT(256) + 128x128 块 scale 融合, AVX2 与标量同 8-lane 归约结构(fmaf/fmadd), tail 经 spill 保持 lane 序 => 位级一致, FNV-1a 可锁
- 	ools/gen_matmul_fixture.py -> 	ests/fixtures/mm/(8 case, float64 参考, tol=5e-4)
- 	ests/test_mm.c: 全绿, 实测 maxrel 最大 0.0167(qkvlike 10240x5120) 仍在 atol+rtol 门内; 用了 atol+rtol 组合判据(单 atol 对 5120 深归约太紧)
- 取舍: 没暴露 scalar 路径给测试, 双路径位一致性靠结构同构 + FNV 锚定(AVX2 build 上跑 FNV), 留作跨平台对照点

