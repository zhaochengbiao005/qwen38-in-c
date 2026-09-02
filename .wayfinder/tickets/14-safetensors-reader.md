# #14: safetensors reader（Windows 版）+ config 解析 + 绑定框架

**What to build:** 引擎能打开真实 FP8 权重目录（layers-N.safetensors_ 式命名，容忍下划线后缀），按张量名读出 header（dtype/shape/offset）与原始字节；解析 config.json（text_config 全部字段，缺字段即拒绝）；提供按层索引的 bind 表。验证用「真实 layer-0 shard + 合成小 shard」双数据源。

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] 从零手写 safetensors header JSON 解析器（头部 <100KB，无需全 JSON 库）
- [x] 张量名→(文件, offset, len, dtype, shape) 索引，查找 O(log n)
- [x] 容忍 `.safetensors_` 后缀命名（下载工具产物）
- [x] config.json 解析并断言全部必需字段（qwen3_5_text schema）
- [x] 用真实 layers-0 shard 验证各张量 shape 与 Ticket-01 schema 一致
- [x] 负样本：截断 header、坏 dtype、offset 越界，全部明确报错


Claim: 已认领（2026-08-22）

## Resolution
（2026-08-22 解决，CLOSED）

落地：`q35_json.c`（手写 DOM JSON parser，arena 分配）、`q35_st.c`（目录/单文件索引 + 惰性 FILE* + 64 位偏移读取）、`q35_cfg.c`（严格 config 解析，缺字段/层类型未知即拒绝；rope_theta 支持 `rope_parameters` 嵌套）。

实测：
- 合成 fixture（python + numpy 生成 mini.safetensors + manifest CRC32）全绿，负样本截断 header 正确拒绝
- **真实 checkpoint 全量索引**：1606 个张量（与 Ticket-01 盘点一致），config 读出 64 层 / 48 linear + 16 full / hidden 5120 / vocab 248320
- 真实 FP8+scale 形状校验通过（in_proj_qkv 的 scale shape = ceil(dim/128) 两边成立）

踩坑：vision 的 `patch_embed.proj.weight` 是 5 维张量（[1152,3,2,16,16]）——reader 维度上限提到 8（视觉塔仅索引不加载，不越权实现视觉）。

文件：include/q35/q35_st.h、q35_json.h、q35_cfg.h；src/q35_st.c、q35_json.c、q35_cfg.c；tools/gen_st_fixture.py
