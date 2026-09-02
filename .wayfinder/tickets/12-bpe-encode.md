# #12: BPE encode 路径（splitter + merge 循环）

**What to build:** 完整 byte-level BPE encode：pre-tokenizer split（按 blob 固化的模式）→ merge 循环 → token id 序列。在中文、英文、emoji、代码、混合长文本的 fixture 语料上与 HF tokenizer 输出 byte-exact 一致。

**Blocked by:** #11 Tokenizer blob 加载器

**Status:** resolved

- [x] fixture 语料逐条与 HF tokenizer 期望 id 序列完全一致
- [x] byte fallback / 未知字节处理与 HF 行为一致
- [x] 空白/换行/多字节 UTF-8 边界用例覆盖
- [x] fixture 资源缺失时测试显式 NOT RUN，不静默通过
- [x] fixture 生成脚本（Python 调 HF tokenizer）入库，可重生成


Claim: 已认领（2026-08-22）

## Resolution
（2026-08-22 解决，CLOSED）

落地：`qwen35-c/src/q35_bpe.c` + `include/q35/q35_bpe.h` + `src/q35_unicode_tables.h`（生成物，工具见 `tools/gen_unicode_tables.py`）。

组成：
- **NFC**：规范化全路径 C 实现——canonical 分解表（2061 条，含递归展开）+ ccc 稳定重排 + 组合表（941 对；**关键坑：表按 cp 生成顺序并非 (c1,c2) 有序，二分前必须 sort**）+ Hangul 算法式分解/组合（SBase 公式）。
- **pretokenizer**：手写 Qwen2 Split 正则（7 条 alternative、含 `(?!\S)` lookahead 的回溯语义、Rust `White_Space` 表），作用于 NFC 后的 codepoint 序列。
- **merge 循环**：FNV 风格 64-bit hash 的 pair→rank 开地址表（加载期自排双冲突），贪心最小-rank 合并，merged 字节回查 vocab hash 取结果 id。
- **特殊 token**：最长匹配先行切分（`q35_tok_added_*`），与 HF Isolated 行为一致。

验收：`tests/fixtures/encode_fixtures.txt` 440 条（20 条人工边缘 + 400 条跨平面随机 fuzz + 20 条长文应激），**440/440 与 HF tokenizers byte-exact**（31,426 个 id 全部一致）。NFC 正反例（e+U+0301 vs U+00E9）、Hangeul jamo 组合、emoji ZWJ/区旗覆盖。

工具：`tools/gen_encode_fixtures.py` 可重生成 fixture；`tools/gen_unicode_tables.py` 重生成 Unicode 表（Python unicodedata 快照，注意与目标 Unicode 版本绑定）。
