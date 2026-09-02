# Ticket-03: Qwen3.5 文本架构数值规格（research, AFK）

Label: wayfinder:research
Claim: 已自动声明（charting 会话分发）
Blocked by: 无（frontier）

## Question

锁定 Qwen3.8-27B 文本子图每个算子的精确数学定义（以 HF transformers 的 Qwen3_5 实现为 ground truth）：Gated DeltaNet 递推公式（四路投影 in_proj_qkv/a/b/z、conv1d kernel=4、A_log/dt_bias 的离散化、48 value heads/16 key heads、output_gate_type=swish、状态 dtype float32 的位置）；GQA full attention（RoPE partial_rotary 0.25、theta 1e7、q_norm/k_norm、attn_output_gate）；MLP 结构（是否 MoE 取决于 Ticket-01 结论）；RMSNorm eps；MTP 层忽略路径。产出：逐算子公式表 + 对齐验收容差建议（.wayfinder/research/qwen35-numspec.md）。

## Resolution
（待研究代理回填）
- 已核实源码（transformers main `modeling_qwen3_5.py`）+ hf-mirror config + safetensors index/分片 header 实测，完成逐算子数值规格，详见 `.wayfinder/research/qwen35-numspec.md`。要点：
  - **Gated DeltaNet**：K=2048(16×128)、V=6144(48×128)，in_proj_qkv/z/a/b 分别为 [10240,5120]/[6144,5120]/[48,5120]/[48,5120]；depthwise causal conv1d k=4 + SiLU；`g=-exp(A_log)*softplus(a+dt_bias)`、`beta=sigmoid(b)`（fp32 计算）；状态 S=[B,48,128,128] fp32，递推 `S←αS+k⊗(β(v-S k))`，输出 `S q`（q/k 先 per-head L2 norm，q×1/√128，k 头 16→48 repeat_interleave）；输出门为 per-head RMSNormGated(norm 在前) × SiLU(z)，再 out_proj。
  - **Full attention**：head_dim 256，24:4 GQA，per-head q/k RMSNorm（RoPE 前）；RoPE 只作用于前 64 维（partial 0.25），32/32 半旋转，θ=1e7；q_proj 输出含每头 256 维 gate，attention 输出乘 sigmoid(gate) 后过 o_proj；softmax fp32。
  - **MLP 为 dense SwiGLU**（5120→17408），无 experts/router/shared_expert_gate——对照 FP8 index 子任务确认。
  - **RMSNorm 零中心权重**：y=(x/rms)·(1+w)，eps 1e-6，fp32 计算；MTP 为单 full-attn 层 + fc[5120,10240] + 双 pre_fc_norm，HF 加载时忽略（投机解码专用）。
  - **fp32 节点**：Norm/g-beta/delta-rule/softmax/RoPE 计算，其余 bf16。容差表见 numspec §6：投影 rtol~2e-2，fp32 递推段 rtol 1e-3，端到端 top-1 一致、logits cos-sim>0.9999。
