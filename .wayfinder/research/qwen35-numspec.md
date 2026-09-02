# Ticket-03: Qwen3.8-27B（Qwen3_5 架构）文本子图数值规格

来源：
- config: https://hf-mirror.com/Qwen/Qwen3.8-27B/raw/main/config.json（已存 `.wayfinder/research/qwen3.8-config.json`）
- 权重索引/张量形状: safetensors index + 分片 header 实测（BF16 权重）
- 实现源码: transformers main `modeling_qwen3_5.py`（已存 `.wayfinder/research/modeling_qwen3_5.py`，对应 transformers 5.8.0.dev0）
- 仓库内存在 `mtp.*` 权重，HF transformers 用 `_keys_to_ignore_on_load_unexpected = ["^mtp.*"]` 忽略（仅投机解码用，vLLM 等推理框架自行加载）

全局常量（text_config）：
- hidden_size H=5120，num_hidden_layers=64，每 4 层一个 full attention（`full_attention_interval=4`），即 48 层 linear_attention + 16 层 full_attention
- rms_norm_eps=1e-6，dtype=bfloat16，mamba_ssm_dtype=float32
- vocab=248320，tie_word_embeddings=false

**特别注意 RMSNorm 参数化**：`Qwen3_5RMSNorm` 权重 w 以零为中心存储，前向为
`y = (x / rms(x)) * (1 + w)`，全部在 float32 下计算后转回输入 dtype。合并/对照权重时不可直接当作普通 RMSNorm 的 gamma；等效 gamma = 1 + w。`input_layernorm`/`post_attention_layernorm` 及 Q/K norm 均为此形式；linear_attn.norm（Qwen3_5RMSNormGated）为普通直接乘 weight 的参数化（见 §1.5）。
## 1. Gated DeltaNet（linear_attn.*，48 层）

维度：num_v_heads Hv=48 × head_v_dim dv=128 → value_dim V=6144；num_k_heads Hk=16 × head_k_dim dk=128 → key_dim K=2048；`Hv/Hk=3`（GQA 广播）。

### 1.1 投影（全部无 bias，BF16）
- in_proj_qkv: [K*2+V=10240, 5120] → q/k/v 拼接
- in_proj_z:   [V=6144, 5120]     → z（输出门）
- in_proj_a:   [Hv=48, 5120]      → a（每 v-head 一个标量/token）
- in_proj_b:   [Hv=48, 5120]      → b
- out_proj:    [5120, V=6144]
- norm.weight: [128]（Qwen3_5RMSNormGated，per-v-head）
- A_log: [48]，dt_bias: [48]

### 1.2 conv1d
- 输入 mixed_qkv 转置为 [B, 10240, L]，深度可分离 Conv1d：weight [10240,1,4]，groups=10240，kernel=4，无 bias，padding=3（左补零）后截断到长度 L（causal），随后逐通道 SiLU 激活。
- 单 token 解码走 `causal_conv1d_update`：conv_state [B, 10240, 4]（缓存最近 4 个输入，右对齐），逐 token shift+卷积+SiLU 原地更新。

### 1.3 离散化（beta / g）
每 token、每 v-head：
- beta_t = sigmoid(b_t)          ∈ (0,1)
- g_t    = -exp(A_log.float()) * softplus(a_t.float() + dt_bias)   ≤ 0
- 衰减系数 α_t = exp(g_t) ∈ (0,1)
`A_log`、`dt_bias` 与 a 的计算强制 float32（源码注释：fp16 下 exp(A_log) 会 -inf）。

### 1.4 Delta rule 递推（fp32）
conv 输出 split 成 q/k [B,L,16,128]、v [B,L,48,128]；q,k 按头 `repeat_interleave(3)` 广播到 48 头；进 kernel 前 q,k 做 per-head L2 norm（eps=1e-6，`use_qk_l2norm_in_kernel=True`），q 乘 scale = 1/sqrt(128)。

状态 S: [B, 48, dk=128, dv=128]，float32。逐 token（torch_recurrent_gated_delta_rule）：
```
S <- S * exp(g_t)                        # 全状态 scalar 衰减（每头一个 α）
kv_mem = (S * k_t[...,None]).sum(-2)     # S^T k_t → [B,48,128]
delta  = (v_t - kv_mem) * beta_t
S <- S + k_t[...,None] * delta[...,None,:]   # k_t ⊗ delta
o_t = (S * q_t[...,None]).sum(-2)        # S^T q_t → [B,48,128]
```
prefill 多 token 走 chunked 实现（torch_chunk_gated_delta_rule，chunk_size=64，数学等价：块内用 (I+严格下三角) 递推求局部逆、g cumsum 得块内/跨块衰减掩码、块间状态按末期 g 衰减并累加 k⊗v_new）。chunk 版全程 float32，q/k/v/beta/g 进入 kernel 即 `.to(torch.float32)`，输出 cast 回输入 dtype。

### 1.5 输出门（output_gate_type=swish）
kernel 输出 reshape 成 [B*L, 128]/头，z 由 in_proj_z reshape 成 [B*L, 128]：
```
h = RMSNorm_fp32(h) * weight             # Rsq norm 先算（norm before gate），eps=1e-6，普通 weight 参数化（无 +1）
h = h * SiLU(z.float())                  # 门乘在归一化之后
```
再 reshape [B,L,6144] 过 out_proj → [B,L,5120]。注意 norm 与门都在 out_proj 之前。

## 2. Full Attention（self_attn，16 层：层号 3,7,…,63；mtp.layers.0 亦同构）

head_dim D=256，num_attention_heads=24，num_key_value_heads=4（GQA 24:4 = 6 组广播），scaling=1/sqrt(256)=1/16，全部投影无 bias。

- q_proj: [12288, 5120] = 24 × (256 query + 256 gate)；每头 512 维内对半 chunk（`torch.chunk(..., 2, dim=-1)` on [B,L,24,512]），gate reshape 成 [B,L,6144]。
- k_proj / v_proj: [1024, 5120]（4×256）；o_proj: [5120, 6144]。
- **q_norm / k_norm**：per-head RMSNorm，dim=256，eps=1e-6，零中心权重（等效 1+w）；在 reshape 成 per-head 后、RoPE 前应用。
- **RoPE**：partial_rotary_factor=0.25 → rotary_dim = 64（每头前 64 维旋转，后 192 维直通）；rotate_half 在 64 维内按 32/32 对半切分（GPT-NeoX 式半旋转，非交错）。rope_theta=1e7，inv_freq 共 32 个频率。`rope_type="default"`；多模态时启用 interleaved MRoPE（mrope_section [11,11,10]，3 路 position_ids），纯文本三路相等退化为默认 RoPE。cos/sin 在 float32 下计算（attention_scaling=1.0），用后 cast 回 bf16。
- GQA：eager 路径 `repeat_kv`（expand+reshape 等价 repeat_interleave）将 k/v 4→24 头。
- softmax 强制 float32（`softmax(..., dim=-1, dtype=torch.float32)`）后转回。
- **输出门**（attn_output_gate=true）：attn_output reshape [B,L,6144] 后 `attn = attn * sigmoid(gate)`，再过 o_proj。是 **sigmoid 门**（非 swish），恰好乘在 o_proj 输入上。

## 3. MLP（dense SwiGLU，非 MoE）

本模型（27B）**没有专家**：safetensors index 中无 `mlp.experts`、`shared_expert`、`mlp.gate`(router) 张量，config 无 num_experts 字段。每层 `Qwen3_5MLP`：
```
mlp(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
gate_proj/up_proj: [17408, 5120]；down_proj: [5120, 17408]；无 bias，BF16。
```
（给 FP8 index 子任务的对照结论：该 repo 文本侧为纯 dense SwiGLU；`mlp.gate_proj` 是 SwiGLU 的 gate 分支，不是 router。）

## 4. MTP 层（投机解码，可选）

权重组 `mtp.*`（transformers 加载时忽略，`^mtp.*` unexpected）：mtp.fc.weight [5120, 10240]、`mtp.pre_fc_norm_embedding`/`mtp.pre_fc_norm_hidden`（各 [5120]，零中心 RMSNorm）、`mtp.layers.0.*`（一个 full-attention decoder 层，q_proj [12288,5120] 同样带输出门，含 q/k norm 256）、`mtp.norm` [5120]。接入主干接口（Qwen3-Next 式 MTP）：
```
h = fc( cat( pre_fc_norm_hidden(h_L), pre_fc_norm_embedding(Emb(t+1)) ) )   # 10240 → 5120
h = MTPDecoderLayer(h); h = mtp.norm(h); logits = lm_head(h)
```
`mtp_use_dedicated_embeddings=false` → 共享主干 embed_tokens 与 lm_head。

## 5. Decoder 层骨架与 float32 节点汇总

每层 pre/post 结构（decoder layer forward）：
```
x <- x + (linear_attn 或 self_attn)(input_layernorm(x))
x <- x + mlp(post_attention_layernorm(x))
```
主干末尾 final_layernorm + lm_head（lm_head 未 tie）。

必须 float32 的中间量（对数值复现敏感）：
1. 所有 RMSNorm 的平方均值 / rsqrt / 缩放（含零中心 1+w 叠加），进出转回 bf16。
2. GDN：A_log/a/dt_bias → g 的计算与 beta；q/k L2 norm；delta-rule（chunked/recurrent）全程 fp32（含状态 S 常驻 fp32，`mamba_ssm_dtype=float32`）。
3. RMSNormGated 的门支路：`SiLU(z.float())`。
4. full attention softmax（fp32）。
5. RoPE 的 inv_freq/freqs/cos/sin 计算（fp32，用完 cast）。
其余一切 matmul、残差相加为 BF16。

## 6. 建议浮点容差（被测实现 vs HF BF16 eager 参考）

| 算子/中间量 | 容差 |
|---|---|
| 线性投影（in_proj_*, out_proj, q/k/v/o, MLP 三个 proj） | rtol 2e-2, atol 2e-2（BF16 matmul，5120 阶累加） |
| causal conv1d（含 SiLU） | rtol 1e-2, atol 1e-2 |
| RMSNorm / RMSNormGated 输出 | rtol 1e-2, atol 5e-3（对纯 fp32 参考可收紧到 1e-3） |
| g / beta（fp32 标量） | rtol 1e-5, atol 1e-6 |
| delta-rule 每步 o_t 与状态 S（fp32） | rtol 1e-3, atol 1e-3；chunked vs recurrent 相互差 ~1e-4 |
| full attention 单层输出 | rtol 2e-2, atol 2e-2 |
| 整个 decoder 层输出（含残差） | rtol 2e-2, atol 1e-2（bf16 残差主导） |
| 端到端 64 层最终 logits | top-1 token 必须一致；logits atol ~0.15 或 cos-sim > 0.9999 |

注：上述以 HF BF16 参考为基准；若参考是 fp32（同一算子序列），容差的下界即 BF16 自身舍入（特征维 5120 累加典型相对误差 5e-3~1e-2）。推荐两步对拍：先 fp32 参考对拍 fp32 强制段（GDN 递推、Norm、softmax），再整链 bf16 对拍。

