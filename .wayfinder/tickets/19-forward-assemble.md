# #19: forward 总装（64 层混合图）

**What to build:** embed → 64 层（layer_types 按 3:1 交替）→ final RMSNorm → lm_head → logits + argmax。合成随机模型（小 config：如 4 层，3 linear +1 full）跑通单 token 前向，结构图正确（数值由 #20 保证）。

**Blocked by:** #14, #15, #17, #18

**Status:** done

- [ ] 合成 mini-config 驱动：hidden/层数/heads 从 config 读，拒绝缺省
- [ ] KDA state + KV cache 生命周期管理（init/increment/reset）
- [ ] logits 输出 fp32

## Resolution (2026-08-23)

落地（Peirce 写 model 层后断线，主线收尾）：q35_model.h/c（mmap 只读绑定 FP8+scale 零拷贝、BF16 小张量转 f32、in_proj_a/b 装载时量化）、tools/gen_model_fixture.py（3 层 mini 模型 safetensors + fp64 numpy 参考 + dump 对拍机制）、tests/test_model.c。合成模型 8/8 测试全绿（logits maxabs 1.9e-2 vs f64 参考，容差内）；prefill vs decode logits bitwise 一致。**真实 27B 模型 smoke 通过**：装载 3.7s（纯索引+映射），2 token prefill 46s（首次触页），logits 输出正常。排障记录：test 侧 tokens.json 解析 bug（尾 ']' 前步进）导致 prefill 64 token 的假象；定位方法 = Q35_DUMP 环境变量逐层 dump + numpy 重算对拍（dump hook 保留在 q35_model.c 供 #20/#22 用）。

