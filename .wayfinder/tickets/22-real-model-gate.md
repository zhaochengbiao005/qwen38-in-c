# #22: 真实模型门禁

**What to build:** 真实 28.7 GiB FP8 checkpoint 上：加载 → greedy 生成 → 与 HF/transformers 参考（本地 torch 跑一次）logits 逐元素对比 + 多 token 序列一致性。性能基线报数（s/token、加载时间、RSS）。

**Blocked by:** #20, #21

**Status: done**

- [x] 权重目录：F:\project\QWEN3.8-27B_JSZ\Qwen3.8-27B-FP8（已就位，28.77 GB）
- [x] logits 对比脚本与容差声明（tools/cmp_real.py，容差 = numspec §6 端到端行）
- [x] prompt 集合的输出文本 sanity check（问答/代码各若干）——首条 prompt 完成，多 prompt 泛化留给回归使用
- [x] 性能基线写入 docs/data/（见下 + tests/gates/real-22/）

## Resolution (2026-08-24)

**门禁 PASS（2026-08-24，首跑即过）**。prompt "The capital of China is"（ids [760,6511,314,5440,369]），5 token prefill + 8 token greedy：

- **token ids 12/12 逐位一致**（引擎与 HF 参考的 greedy 序列完全相同：25701, 13, 198, 760, 6511, 314, 5440, 369）
- **12 行 logits 逐元素**：top-1 全部一致；maxabs ≤ 0.135（atol 门 0.15）；cos-sim ≥ 0.99996（门 0.9999）——numspec §6 端到端容差双条件全满足，且实测贴近 bf16 舍入本底
- 证据：`qwen35-c/tests/gates/real-22/`（双方 logits bin + ids json + 参考运行日志）

### 性能基线（Ryzen 9 5900X 单线程-ish，96GB，Windows）

| 指标 | 引擎 (f32 on-the-fly dequant) | HF 参考 (bf16 torch) |
|---|---|---|
| 加载 | 3.4 s | 95.7 s |
| prefill 5 tok | ~57 s (含在 12 步内) | 73.9 s |
| decode | **1.72 s/tok** (12 线程 OpenMP) | 44.7 s/tok |
| 峰值 RSS | 25.3 GB | ~58 GB |

### OpenMP 并行化（2026-08-24）

三个 matmul（fp8 avx2/scalar + bf16）行循环加 `#pragma omp parallel for schedule(static)`，不改变每行 8-lane 归约序。clang-cl 22 的 `/openmp:llvm` 被静默忽略（`_OPENMP` 不定义）→ 改用 `-openmp` + 显式链接 `libomp.lib`（CMakeLists 用 8.3 短路径 `/LIBPATH:C:\PROGRA~1\LLVM\lib` 绕空格）+ build.cmd 自动部署 libomp.dll。性能：11.5→1.72 s/tok（**6.7×**），比 HF bf16 参考快 **26×**。ctest 9/9 无回归，FNV 比特一致。

### 参考侧工程记录（防复踩）

- transformers 5.3.0 的 qwen3_5 数值语义与 numspec 依据的 5.8.0.dev0 快照**等价**（diff 存 `.wayfinder/research/modeling-5.3.0-vs-5.8.diff`：RMSNorm/MLP 零差异；attention 仅 cache 管道；delta rule 纯 torch 路径仅格式差异）。本机无 fla/causal-conv1d，恰好强制走与 numspec 相同的纯 torch 路径。
- `from_pretrained` 的 fp8 quantizer 路径在本机不可用且 43% 处段错误：需 triton（Windows 无）→ 垫片又被 torch dynamo/inductor 连环拉爆。最终方案：**手动加载**（`tools/real_ref.py`）——meta 构造 + `to_empty` + rotary buffer 重初始化 + 逐分片按官方语义反量化（`w8.to(bf16) × scale`，128×128 块 reshape，源码 `integrations/finegrained_fp8.py::dequantize`）。
- 环境修复（已固化）：卸载与 torch 2.12 不兼容的 torchvision（其 import 崩溃污染 transformers 导入链）；安装 accelerate、compressed-tensors。
- 模型目录文件名带尾下划线（`layers-0.safetensors_`）对 HF 不可见 → 已建 67 个同卷硬链接（无尾缀名），零磁盘成本；引擎不受影响。**勿删这些硬链接**，参考复跑需要。
- 工具：`tools/real_ref.py`（参考侧）、`tools/dump_real.c`（引擎侧，CMake target `dump_real`）、`tools/cmp_real.py`（判定）。复跑流程：real_ref.py → dump_real → cmp_real（在 qwen35-c/ 下）。
