# Ticket-01 权重架构盘点：Qwen/Qwen3.8-27B-FP8

来源：hf-mirror.com（镜像 huggingface.co），2026-08-21 抓取。
原始数据：同目录 `config.fp8.json` / `index.fp8.json` / `index.bf16.json` / `headers.fp8.json`（全部 67 个 safetensors 分片的完整 header，经 HTTP Range 读取）。

## 0. 结论先行（颠覆性发现）

- **这不是 Qwen3 文本 dense 模型**：`architectures = Qwen3_5ForConditionalGeneration`，`model_type = qwen3_5`（Qwen3.5 系多模态）。组成：64 层语言模型 + 27 层视觉塔（ViT）+ 1 层 MTP（多 token 预测）头。
- **语言部分是 hybrid linear attention**：64 层中 48 层 `linear_attn`（linear_key_head_dim=128 / linear_num_key_heads=16 / linear_num_value_heads=48 / value_head_dim=128 / conv kernel=4，Gated-DeltaNet 风格），16 层 `self_attn`（full attention，每 4 层一个，`full_attention_interval=4`，层号 3,7,...,63）。
- **没有 MoE**。`quantization_config.modules_to_not_convert` 里的 `mlp.gate` / `mlp.shared_expert_gate` 是 Qwen FP8 模版的防御性条目；**checkpoint 中不存在任何 `experts*` / `mlp.gate.weight` / `shared_expert*` 张量**（1606 个张量逐一 grep，0 命中）。MLP 是纯 dense 三件套 `gate_proj / up_proj / down_proj`。
- 因此 ticket 问题 5 的答案：**BF16 与 FP8 结构完全一致，两边都是 dense hybrid + vision + MTP**（diff 见第 4 节）。
- MTP 头（`mtp.*`，1 层 full attention + dense MLP + fc 融合层）存在且其 Linear 多数被 FP8 量化。

## 1. 仓库文件与分片布局

- 分片：`layers-0.safetensors` … `layers-63.safetensors`（每个恰好放一层语言模型），外加：
  - `outside.safetensors`：embed_tokens、lm_head、final norm、整个视觉塔；
  - `mtp.safetensors`：整个 MTP 头。
- 索引就是标准名 `model.safetensors.index.json`（137,335 字节，weight_map 共 1606 个张量）。

## 2. 张量模式全表（dtype + shape）

`N` = layer index。hidden_size=5120，intermediate_size=17408。

### 2.1 linear_attention 层（48 层，每层 20 个张量）

| 张量名模式（前缀 `model.language_model.layers.N.`） | dtype | shape | 备注 |
|---|---|---|---|
| `input_layernorm.weight` | BF16 | 5120 | RMSNorm |
| `linear_attn.in_proj_qkv.weight` | F8_E4M3 | 10240×5120 | FP8 |
| `linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | 80×40 | 块逆缩放 |
| `linear_attn.in_proj_z.weight` | F8_E4M3 | 6144×5120 | output gate |
| `linear_attn.in_proj_z.weight_scale_inv` | BF16 | 48×40 | |
| `linear_attn.in_proj_a.weight` | BF16 | 48×5120 | 低秩小矩阵，未量化 |
| `linear_attn.in_proj_b.weight` | BF16 | 48×5120 | 未量化 |
| `linear_attn.conv1d.weight` | BF16 | 10240×1×4 | 深度可分离 conv |
| `linear_attn.A_log` | BF16 | 48 | |
| `linear_attn.dt_bias` | BF16 | 48 | |
| `linear_attn.norm.weight` | BF16 | 128 | gated norm（head_dim） |
| `linear_attn.out_proj.weight` | F8_E4M3 | 5120×6144 | FP8 |
| `linear_attn.out_proj.weight_scale_inv` | BF16 | 40×48 | |
| `post_attention_layernorm.weight` | BF16 | 5120 | |
| `mlp.gate_proj.weight` | F8_E4M3 | 17408×5120 | FP8 |
| `mlp.gate_proj.weight_scale_inv` | BF16 | 136×40 | |
| `mlp.up_proj.weight` | F8_E4M3 | 17408×5120 | FP8 |
| `mlp.up_proj.weight_scale_inv` | BF16 | 136×40 | |
| `mlp.down_proj.weight` | F8_E4M3 | 5120×17408 | FP8 |
| `mlp.down_proj.weight_scale_inv` | BF16 | 40×136 | |

### 2.2 full_attention 层（16 层，N∈{3,7,...,63}，每层 18 个张量）

layernorm 与 MLP 部分同 2.1。注意力部分：

| 张量名模式（前缀 `model.language_model.layers.N.self_attn.`） | dtype | shape |
|---|---|---|
| `q_proj.weight` / `q_proj.weight_scale_inv` | F8_E4M3 / BF16 | 12288×5120 / 96×40 |
| `k_proj.weight` / `k_proj.weight_scale_inv` | F8_E4M3 / BF16 | 1024×5120 / 8×40 |
| `v_proj.weight` / `v_proj.weight_scale_inv` | F8_E4M3 / BF16 | 1024×5120 / 8×40 |
| `o_proj.weight` / `o_proj.weight_scale_inv` | F8_E4M3 / BF16 | 5120×6144 / 40×48 |
| `q_norm.weight` | BF16 | 256 |
| `k_norm.weight` | BF16 | 256 |

q_proj 12288 = 24 heads × head_dim 256 × 2（`attn_output_gate=true`，Q 与 gate 融合）；k/v = 4 kv heads × 256 = 1024；o_proj 输入 6144 = 24×256。

### 2.3 顶层 / 视觉塔（`outside.safetensors`，全部 BF16，无 FP8）

| 张量 | dtype | shape |
|---|---|---|
| `model.language_model.embed_tokens.weight` | BF16 | 248320×5120 |
| `lm_head.weight` | BF16 | 248320×5120 |
| `model.language_model.norm.weight` | BF16 | 5120 |
| `model.visual.patch_embed.proj.weight` / `.bias` | BF16 | 1152×3×2×16×16 / 1152 |
| `model.visual.pos_embed.weight` | BF16 | 2304×1152 |
| `model.visual.blocks.{0..26}.attn.qkv.weight` / `.bias` | BF16 | 3456×1152 / 3456 |
| `model.visual.blocks.{0..26}.attn.proj.weight` / `.bias` | BF16 | 1152×1152 / 1152 |
| `model.visual.blocks.{0..26}.mlp.linear_fc1.weight` / `.bias` | BF16 | 4304×1152 / 4304 |
| `model.visual.blocks.{0..26}.mlp.linear_fc2.weight` / `.bias` | BF16 | 1152×4304 / 1152 |
| `model.visual.blocks.{0..26}.norm1/norm2.weight` / `.bias` | BF16 | 1152 |
| `model.visual.merger.linear_fc1.weight` / `.bias` | BF16 | 4608×4608 / 4608 |
| `model.visual.merger.linear_fc2.weight` / `.bias` | BF16 | 5120×4608 / 5120 |
| `model.visual.merger.norm.weight` / `.bias` | BF16 | 1152 |

**embed_tokens / lm_head / 所有 norm / 整个视觉塔全部 BF16**。这就是 `modules_to_not_convert` 巨长的原因：逐 block 列出全部视觉子模块 + `in_proj_a/b` + `A_log/conv1d/dt_bias` + 所有 norm。

### 2.4 MTP 头（`mtp.safetensors`）

`mtp.layers.0.*` = 一层 full attention + dense MLP：q/k/v/o_proj 与 gate/up/down_proj 全部 FP8（各带 weight_scale_inv，shape 同 2.2）；`mtp.fc.weight`（BF16，5120×10240）、`mtp.norm`、`mtp.pre_fc_norm_embedding`、`mtp.pre_fc_norm_hidden`、两个 layernorm、q/k_norm 均为 BF16。

## 3. FP8 scale 张量命名与布局（128×128 块核实）

- 量化配置：`quant_method=fp8`、`fmt=e4m3`、`activation_scheme=dynamic`、`weight_block_size=[128,128]`——deepseek 风格 **block-wise weight-only FP8**，激活动态量化。
- 命名规则：`X.weight`（F8_E4M3）与 `X.weight_scale_inv`（BF16）一一配对、同分片相邻。全 checkpoint **407 个 FP8 张量 + 407 个 scale 张量**。
- **shape 验证：`scale.shape == [ceil(rows/128), ceil(cols/128)]`，全量 407 对逐一核对成立**。样例：
  - 10240×5120 → 80×40 ✓
  - 12288×5120 → 96×40 ✓
  - 5120×17408 → 40×136 ✓
  - 17408×5120 → 136×40 ✓
- 即 scale 第 0 维对应权重行（out_features）块、第 1 维对应列（in_features）块（row-major `[out,in]`）。反量化：`W = W_fp8 ⊙ scale_inv`（128×128 块 broadcast）后转 BF16。

## 4. 与 BF16 版（Qwen/Qwen3.8-27B）的 tensor-set diff

对两边 index.json 的 weight_map 做模式级 diff（层号归一化）：

- **BF16 有而 FP8 没有：0 个。**
- **FP8 有而 BF16 没有：17 个模式，全部是 `weight_scale_inv`**（量化副产物）。
- 其余名字集合完全一致（含 vision、mtp）。**不存在 "BF16 是 dense 而 FP8 是 MoE" 或反向的结构差异——两版同为 dense hybrid + vision + MTP**。
- BF16 版 `metadata.total_size = 55,562,855,904` 字节（≈51.8 GiB），FP8 压缩比 ≈ 1.79×。

## 5. 参数量与字节拆分（由全部 67 个 header 精确求和）

| 部分 | 数量 | 字节 |
|---|---|---|
| 总参数量 | 27.783 B（FP8 元素按 1 参数计，含 vision/mtp） | — |
| FP8 权重（F8_E4M3） | 407 张量 | **23.003 GiB** |
| scale（weight_scale_inv，BF16） | 407 张量 | **2.88 MiB**（可忽略，但 broadcast 逻辑必须有） |
| BF16 保留部分 | 792 张量 | **5.741 GiB**（embed 1.27B + lm_head 1.27B + vision ~0.63B + norm/小参数） |
| payload 合计 | 1606 张量 | **≈ 28.75 GiB** |

## 6. 对 bind 层 / FP8 kernel 的工程提示

1. 加载器必须处理"每层一个分片"的映射（`layers-N.safetensors` ↔ `layers.N`），不是常见的连续区间切分。
2. MTP 在 `mtp.*` 命名空间（不在 `model.*` 下）；`language_model_only=false`。纯文本推理可整包丢弃 `mtp` + `model.visual`，此时实际加载约 26.4B 参数。
3. 文本侧所有 Linear 均无 bias；视觉 ViT 全带 bias。
4. 反量化目标 dtype BF16；norm 只有 `weight` 无 bias。
5. 层类型判定用 `text_config.layer_types`（64 项字符串数组）；full attention 层号 = 3+4k（k=0..15）。
6. `modules_to_not_convert` 里同时带 `model.visual...` 与裸 `visual...` 双前缀条目，做白名单匹配时要兼容。
