# Ticket-06: Tokenizer 路径（task, AFK）

Label: wayfinder:task
Claim: 未认领
Blocked by: Ticket-01

## Question

K3 是手写 byte-level BPE 直接吃 tiktoken.model；Qwen3.8 词表 248320，官方发 tokenizer.json（HF BPE）。选路：A) 写一次性转换工具 tokenizer.json → tiktoken-compatible 格式，复用 K3 的 BPE 引擎；B) 引擎内新增 tokenizer.json 解析器（零转换链但更重的 JSON 工作）；C) vendored 一个极简 BPE。验收：与 HF tokenizer 的 byte-exact roundtrip + 特殊 token（image/video/eos=248044）处理对齐。

## Resolution
（2026-08-21 定案，CLOSED）

选 **A 变体**：离线一次性转换工具（Python）把官方 `tokenizer.json`/`tokenizer_config.json` 转成自定义紧凑二进制 `tokenizer.bin`，C 引擎只加载 blob 跑 byte-level BPE。

理由：
- 引擎保持零 JSON 依赖（与蓝本 K3 的「JSON 手写/vendor」哲学一致，审计友好）
- 转换链是离线工具：schema 漂移在 build 期爆炸而不是运行时静默错词
- Qwen 的 pre-tokenizer split 模式（及其 248320 词表 + 特殊 token bos=eos=248044）在转换期固化进 blob 并校验
- 为什么不要 B/C：B 要在 C 里写通用 JSON 解析，范囩大于收益；C vendor 别人的 BPE 违反零依赖原则且口径难对齐

实施拆分为实现票据 #10–#13（见各票）。
