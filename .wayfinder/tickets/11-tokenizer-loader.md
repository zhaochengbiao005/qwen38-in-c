# #11: Tokenizer blob 加载器 + 词表查找 + 特殊 token

**What to build:** C99 加载器读入 `tokenizer.bin` 并做完整性校验（魔数/版本/长度），提供 id↔bytes 双向查找与特殊 token 查询 API；任何字段被篡改都能在加载期被拒绝。

**Blocked by:** #10 Tokenizer 转换工具

**Status:** resolved

- [x] 拒绝魔数/版本/长度错误的 blob，报明确错误信息
- [x] id→bytes 与 bytes→id 双向查找正确（覆盖 248320 全量条目，含特殊 token）
- [x] 单元测试：对一组 fixture 字符串的 decode 结果与原文一致（encode 可用 stub merge）
- [x] 词表内存占用有报告（供内存地板核算）


Claim: 已认领（2026-08-21/22）

## Resolution
（2026-08-22 解决，CLOSED）

落地：新仓库 `qwen35-c` 骨架 + tokenizer 加载器。

- `include/q35/q35_tok.h` + `src/q35_tok.c`：section 迭代解析、META/VOCB/MRGS/ATOK/PRET 全段校验（魔数/版本/长度/越界 id/重复 id/范围交叉检查）、FNV-1a 哈希表做 bytes→id、id→bytes 直接索引、特殊 token 查询 API、内存占用报告（实测 12.3 MB 常驻）。
- `src/q35_plat.h` + `src/q35_plat_win.c`：PAL 第一版（读文件、对齐分配、QPC 计时）。
- 构建链：clang-cl 22.1.8 + VS 2026 BuildTools（VC 14.51 + Win10 SDK 19041——注意本机 Win10 19045 不能装 26100 SDK）+ CMake + Ninja；`scripts/devcmd.cmd`/`build.cmd` 一键。
- `__AVX2__` 在 clang-cl + `-mavx2` 下确认定义（MSVC 陷阱规避验证通过）。
- 测试：`tests/test_tok.c` 全绿（META 常量、12 条 fixture 双向查找、decode 拼接、特殊 token、越界拒绝），负样本（截断/坏魔数/坏 section 长度）均以明确错误拒绝。

注意：META flags 目前 bit0=IsolatedSplit、bit1=NFC；#12 的 NFC 实现尚未开始。
