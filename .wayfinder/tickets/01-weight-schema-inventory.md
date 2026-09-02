# Ticket-01: FP8 checkpoint 权重架构盘点（research, AFK）

Label: wayfinder:research
Claim: 已自动声明（charting 会话分发）
Blocked by: 无（frontier）

## Question

完整列出 Qwen/Qwen3.8-27B-FP8 的权重 schema：全部张量名、shape、dtype（FP8_E4M3 vs BF16）、每 128x128 块对应的 scale 张量命名与布局、有多少真正是 MoE 专家结构（config 的 modules_to_not_convert 里出现 mlp.gate / shared_expert_gate，说明可能是 MoE 而非纯 dense——必须核实 experts 权重在不在、什么命名）、embed/lm_head/norm 的 dtype，以及与 BF16 版的张量 diff。产出：一份 schema 表格（.wayfinder/research/weight-schema.md），作为 bind 层与 FP8 kernel 的输入。

## Resolution

✅ Question 已解决（研究子代理，2026-08-21，数据经 hf-mirror.com 抓取，全文见 `.wayfinder/research/weight-schema.md`）。

结论摘要：

1. **架构不是你想象的那个**：这是 `qwen3_5` 多模态 hybrid 模型（`Qwen3_5ForConditionalGeneration`）——64 层语言模型（48 层 linear_attention + 16 层 full attention，每 4 层一个）+ 27 层视觉塔 + 1 层 MTP 头。
2. **不是 MoE**：1606 个张量中没有任何 `experts*` / `mlp.gate` / `shared_expert*`；config 里那两个条目是 FP8 模版防御性残留。MLP 为 dense `gate_proj/up_proj/down_proj`（intermediate=17408）。
3. **量化方案**：block-wise FP8-E4M3，`weight_block_size=[128,128]`，激活动态；407 个 FP8 张量（F8_E4M3）各配一个 `X.weight_scale_inv`（BF16），`scale.shape = [ceil(rows/128), ceil(cols/128)]` 全量核对成立。
4. **embed_tokens / lm_head / 所有 norm / 整个视觉塔 = BF16**（无 FP8）。分片：`layers-0..63`（每层一文件）+ `outside.safetensors` + `mtp.safetensors`。
5. **与 BF16 版 diff**：结构完全一致（同 dense hybrid + vision + MTP）；FP8 仅多 17 类 `weight_scale_inv` 张量，BF16 无独有张量。BF16 total_size ≈51.8 GiB，压缩比 ≈1.79×。
6. **字节拆分**：总参数 27.783B；FP8 权重 23.003 GiB + scale 2.88 MiB + BF16 保留 5.741 GiB ≈ 28.75 GiB。

工程影响：bind 层需按 `text_config.layer_types` 区分两种层型；纯文本路径可丢弃 `mtp.*` 与 `model.visual.*`（约省 1.4B 参数）；每层一分片的映射要专门处理。
