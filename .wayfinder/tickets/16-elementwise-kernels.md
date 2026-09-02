# #16: 元素类 kernel 包

**What to build:** RMSNorm（零中心 (1+w)、eps=1e-6、f32 内部计算）、SiLU、SwiGLU、per-head q/k RMSNorm、partial RoPE（head_dim 256 只旋转前 64 维，rotate_half 32/32，θ=1e7）、sigmoid attention output gate。全部与 Python fixture 对照。

**Blocked by:** None (can start immediately)

**Status:** done

- [ ] 每个 kernel 的 fixture 对齐（容差在 fixture manifest 中声明）
- [ ] RoPE 的 mrope_interleaved / section (11,11,10) 参数化进代码但文本路径先按 plain position id 验证（mrope 文本 ≡ 三轴同位 id）
- [ ] AVX2/标量 FNV-1a 一致性

Claim: 已认领（子代理，2026-08-22）


## Resolution

已完成（2026-08-23，实现子代理）。全部 7 个 kernel 交付，AVX2/标量 FNV-1a 全部一致，test_kern.exe 10/10 通过（AVX2 构建与无 `-mavx2` 纯标量构建均通过）。

### 交付文件
- `qwen35-c/include/q35/q35_kern.h` — API：每个 kernel 双入口（默认含 AVX2 分支 + `_scalar` 强制标量），RoPE 参数坏值返回 `Q35_KERN_ERR_ARG`
- `qwen35-c/src/q35_kern.c` — 实现。`#pragma STDC FP_CONTRACT OFF`：关 FP 合同、AVX2 只做 8-lane 数据搬运（不用 `_mm256_fmadd`），保证两条路径执行完全相同的 FP 操作序列 → bit 级一致。归约统一 8 固定累加器、lane 0..7 顺序折叠（rmsnorm 的 sumsq、softmax 的 sum 均同序）
- `qwen35-c/tests/test_kern.c` — manifest 驱动：校验 fixture CRC32、跑双路径、FNV-1a 断言、atol/rtol 逐元素比对；fixture 缺失输出 NOT RUN 返回 77。修了一个 manifest 解析 bug（嵌套 strtok 共享状态吃掉 `in=` 之后的字段，改为手动逗号切分）
- `qwen35-c/tools/gen_kern_fixtures.py` + `tests/fixtures/kern/`（11 个 case 文件 + manifest.txt）

### 数值细节
- RMSNorm/qk-norm：零中心 `(1+w)`，eps=1e-6，f32；fixture 用与 C 同序的 8 累加器 NumPy 参考 → **bit-exact（误差=0）**
- RoPE：partial（head_dim=256，rotary_dim=64），32/32 rotate_half 半旋转（以 modeling_qwen3_5.py 为准，非 interleave），θ=1e7，`inv_freq = 1/(θ^(2i/64))` f32，后 192 维直通；pos=7/2025 两案
- attention gate：纯 sigmoid（无 SiLU），`attn * sigmoid(gate)`，逐元素 f32
- softmax：减 max 稳定版，固定序求和

### 容差与实测最大误差
| case | atol | rtol | 实测 maxabs | 实测 maxrel |
|---|---|---|---|---|
| rmsnorm_5120 | 2e-6 | 2e-6 | 0 | 0 |
| silu_17408 | 1e-6 | 1e-5 | 4.77e-07 | 3.22e-07 |
| swiglu_17408 | 1e-6 | 1e-5 | 9.54e-07 | 3.30e-07 |
| qknorm_24x256 | 2e-6 | 2e-6 | 0 | 0 |
| rope_pos7 | 2e-6 | 2e-5 | 4.77e-07 | 7.09e-07 |
| rope_pos2025 | 2e-6 | 2e-5 | 4.77e-07 | 8.87e-07 |
| attn_gate_6144 | 1e-6 | 1e-5 | 4.77e-07 | 2.69e-07 |
| softmax_1 / 128 / 1023 | 1e-6 | 1e-5 | ≤5.96e-08 | ≤3.55e-07 |

误差全部来自 UCRT `expf/powf/cosf/sinf` 与 NumPy 的 ulp 级差异，远低于容差。

### AVX2/标量 FNV-1a 一致性
10/10 case 哈希一致（rmsnorm=33d21de0, silu=f22b0770, swiglu=43e87b1d, qknorm=6c51b0e7, rope_pos7=4b9972df, rope_pos2025=fe3a098d, attn_gate=13e4a3c5, softmax_1=1b587698, softmax_128=05a38314, softmax_1023=c3e669ba），test 内逐 case 断言。

### 验证命令
```
cmd /c "call scripts\devcmd.cmd && cd /d qwen35-c && clang-cl /MD /O2 -mavx2 -mfma -I include -I src -I tests tests/test_kern.c src/q35_kern.c src/q35_plat_win.c /Fetest_kern.exe && test_kern.exe"
```
（注：test_kern 未接入 CMakeLists，按要求手动编译。）
