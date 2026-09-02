# #17: DeltaNet 线性注意力层

**What to build:** 48 层线性注意力的完整逐 token 前向：四路投影（in_proj_qkv/a/b/z）→ conv1d(depthwise k=4 causal) + SiLU → q/k per-head L2 norm → g=-exp(A_log)·softplus(a+dt_bias)、β=sigmoid(b)（f32）→ 状态递推 S←exp(g)S+k⊗(β(v−Sᵀk))（S 为 [heads, 128, 128] f32 常驻）→ 16 key 头 broadcast 到 48 value 头 → per-head RMSNormGated × SiLU(z) → out_proj。

**Blocked by:** #15, #16

**Status:** done

- [ ] 合成权重的单层前向与 Python 参考（numpy 按 modeling_qwen3_5.py 公式）对齐
- [ ] state 常驻布局与增量更新 API（供 incremental decode）
- [ ] prefill 多 token 路径（串行递推即可，chunked 优化后置）

## Resolution (2026-08-23)

落地：合成 fixture（fp64 numpy 参考）全绿，prefill 与逐 token decode 两路径 bitwise 一致（FNV-1a）；实测 maxrel ≤2.2e-5，远低于容差。详见 qwen35-c/src/q35_deltanet.c、q35_attn.c 与对应 tests。

