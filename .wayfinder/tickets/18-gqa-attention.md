# #18: GQA 全注意力层 + KV cache

**What to build:** 16 层 full attention 的逐 token/prefill 前向：q_proj（含每头 256 维 gate）/k_proj/v_proj → q/k per-head RMSNorm → partial RoPE → GQA 24:4 attention（softmax f32）→ sigmoid 输出门（o_proj 输入侧）→ o_proj；KV cache 追加与滑窗无关（全量）。

**Blocked by:** #15, #16

**Status:** done

- [ ] 合成权重单层对齐 Python 参考
- [ ] KV cache 布局固定（16 层 × 4 头 × 256 维 × 2 × BF16），append API
- [ ] prefill 与逐 token 两路径一致

## Resolution (2026-08-23)

落地：合成 fixture（fp64 numpy 参考）全绿，prefill 与逐 token decode 两路径 bitwise 一致（FNV-1a）；实测 maxrel ≤2.2e-5，远低于容差。详见 qwen35-c/src/q35_deltanet.c、q35_attn.c 与对应 tests。

