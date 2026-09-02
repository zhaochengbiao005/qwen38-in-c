# #33: q35_mm.c 四份重复 AVX2 行内核提取

Label: wayfinder:task
Claim: 海鸥
Blocked by: —

## Problem

`q35_mm.c` 有四份近乎逐字相同的 AVX2 16-wide + 8-lane-tail 行内核：

| 函数 | 行范围 | 用途 |
|---|---|---|
| `mm_fp8_avx2` | 57–95 | 单 token，单权重 |
| `mm_fp8_batch_avx2` | 148–187 | 多 token batch，单权重 |
| `mm_fp8_dual_avx2` | 236–277 | 单 token，双权重 |
| `mm_fp8_row` | 303–337 | 单 token，单权重（已 inline，但前三份没复用） |

四份的 AVX2 核心逻辑相同：16-wide gather + fmadd ×2 + 8-lane 尾部归约。
差异仅在循环结构（外层行/token/权重），但内层列块 dot product 完全一致。

## Plan

提取一个 `mm_fp8_dot16_avx2` 处理 16 列的 dot product（一行一列块），
让前三份 AVX2 内核调用它。标量路径保持不变。

```c
/* 16-column dot product for one row-block: returns partial sum.
 * W points at 16 FP8 bytes; sc is the block scale (BF16); x is the
 * input vector at the matching column offset. */
static inline float mm_fp8_dot16_avx2(const uint8_t *W, float scale,
                                       const float *x);
```

三份内核的调用点改为循环调用 `mm_fp8_dot16_avx2` + 尾部标量处理，
消除重复的 AVX2 指令序列。

## Acceptance

- `grep -c '_mm256_cvtepu8_epi32' q35_mm.c` 从 4 降到 1。
- 全量构建 + ctest 12/12 绿。
- `test_mm` 的 FNV-1a 哈希不变（bit-identical 验证）。

Status: CLOSED
