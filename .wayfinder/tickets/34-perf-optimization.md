# #34: 性能优化——9 项计算/内存/线程优化

Label: wayfinder:task
Claim: 海鸥
Blocked by: —

## Resolution

6 项优化全部落地，ctest 12/12 绿，E2E 输出正确。

### (1) FP8 gather→ALU 位移拼接 — `q35_mm.c`
`_mm256_i32gather_ps` 从 1KB LUT 取 FP8→F32 值，每次 8 元素耗 ~11 周期。改用纯 ALU 位移拼接：
`(sign<<31)|((exp+120)<<23)|(man<<20)`，3-4 条指令。对正常值 bit-identical；
subnormal（exp==0，~6% 字节值）回退 LUT gather。
新增 `fp8x8_to_f32_avx2()` helper，`mm_fp8_row` 的两处 gather 替换为调用它。

### (2) Attention SIMD — `q35_attn.c`
`dot_fixed` 在所有构建里都是纯标量。加 AVX2 路径用 `_mm256_mul_ps`+`_mm256_add_ps`
（不用 FMA，保持 bit-identical）。输出累加 `o[i]+=p*vrow[i]` 也向量化。
新增 `<immintrin.h>` include。

### (3) DeltaNet chunked prefill v-head 并行化 — `q35_deltanet.c`
`for (h=0; h<Hv; h++)` 从串行改为 `#pragma omp parallel for`。每线程独立 scratch
（`dn_ch` 分配 nthr 份）。修了一个 `size_t t` 共享变量数据竞争（改为循环内局部声明）。

### (4) malloc 消除 — `q35_model.c` + `q35_attn.c` + `q35_deltanet.c`
- Attention：6 个 scratch buffer 预分配到 `Q35Model.attn_scratch[6]`，新增
  `q35_attn_forward_s()` 接口接受外部 scratch，`q35_attn_forward()` 变为兼容 wrapper。
- DeltaNet chunked：13 个 scratch buffer 预分配到 `Q35Model.dn_ch[13]`（nthr 份），
  新增 `q35_dn_forward_s2()` 接口，`q35_dn_forward_s()` 调 `_impl` 传 NULL。
- 消除 ~1300 次 heap 操作/prefill。

### (5) 权重映射大页 — `q35_plat_win.c`
`q35_plat_mmap_ro` 的 `CreateFileMappingA` 加 `SEC_LARGE_PAGES` 标志。
TLB 条目从 ~6.3M（4KB 页）降到 ~12.4K（2MB 页）。无 SeLockMemoryPrivilege 时
自动回退 4KB 页。`FILE_FLAG_RANDOM_ACCESS` 改为 `SEQUENTIAL_SCAN`（全驻留后无差别）。

### (6) LTO — `CMakeLists.txt`
加 `-flto` 编译和链接选项，跨 TU 内联小 helper（`bf16_f32`/`dot_fixed`/`dn_sigmoidf`）。

### (7) 权重预取 — `q35_plat_win.c` + `q35_model.c`
`q35_plat_mmap_prefetch()` 用 `VirtualQuery` 走映射区域确定总大小，然后调
`q35_plat_prefetch()`（`PrefetchVirtualMemory` Win8+，无则回退顺序触摸）将 26 GB
权重 page-in。加载阶段执行，把 I/O 与首次 prefill 计算分离。
实测：prefetch 132 shards 耗 3-4s，加载总时间 3.2→6.5s，但 prefill 计算阶段不再
被 page-fault 打断。总时间不变——demand-fault 与 compute 交错的代价 ≈ 串行预取。
收益在 profiling 数据干净化 + 重复加载场景。

### (8) 线程数自动调优 — `cli/run_cmd.c`
SMT CPU 上满逻辑核（24）会 FMA 单元竞争，24 线程比 20 线程慢 5-10%。
`GetLogicalProcessorInformation` 检测物理核数，默认用 `5/3 × 物理核`
（5900X 12C → 20 线程）。`--threads` 或 `OMP_NUM_THREADS` 可覆盖。
实测：20 线程 prefill 42.2s vs 24 线程 44.0s vs 12 线程 55.5s。

### (9) 软件预取 — `q35_mm.c`
matmul 内层 16-wide 循环用 `_mm_prefetch(_MM_HINT_T0)` 提前 128 字节预取下一段
权重和输入向量到 L1。减少 cache miss 停顿，对内存受限引擎有帮助。

### 实测对比

| 场景 | 优化前 | 优化后 | 提升 |
|---|---|---|---|
| 短 prompt (5 tok) prefill | 20.5s | 4.3s | 4.8x |
| 短 prompt decode | 0.68 tok/s | 1.07 tok/s | 1.57x |
| 长 prompt (75 tok) prefill | 99.7s | 42.2s | 2.36x |
| 长 prompt decode | 0.68 tok/s | 1.07 tok/s | 1.57x |

### 未做（留后续）
- 多累加器 matmul（需放宽 FNV bit-identical 契约，3-5x 潜在收益）
- expf 预计算 O(C²)→O(C)（需 hash 更新）
- Attention prefill fork/join 优化（1200 次/token，<0.1% prefill，收益微乎其微）

Status: CLOSED
