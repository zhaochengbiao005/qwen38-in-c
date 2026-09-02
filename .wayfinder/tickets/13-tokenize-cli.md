# #13: `--tokenize` CLI 接缝

**What to build:** 引擎 CLI 提供不加载模型权重即可运行的子命令：`qwen35.exe tokenize --tokenizer tokenizer.bin --prompt "..."`，输出 token id 序列（一行一个或以分隔符）与回译文本。成为后续模型级验收的最高独立接缝。

**Blocked by:** #12 BPE encode 路径

**Status:** resolved

- [x] 无需 --model 即可运行
- [x] 输出 id 序列与回译文本到 stdout；prompt 从参数或 stdin 读取
- [x] 与 HF tokenizer 对同一提示词的 id 逐位一致（集成测试）
- [x] 退出码与错误信息规范（blob 缺失/损坏/编码失败）

Claim: 已认领（2026-08-22）


## Resolution
（2026-08-22 解决，CLOSED）

落地：`qwen35-c/src/q35_cli.c`，产物 `build/qwen35.exe`。

- `qwen35.exe tokenize --tokenizer <bin> (--prompt <text> | --prompt-file <path> | --stdin) [--ids-only]`，输出 id 序列（空格分隔）+ 回译文本。
- 与 HF tokenizers 对表：`Hello, world!` → [9419 11 1814 0] 一致；UTF-8 中文带特殊 token → [109266 3709 96748 6115 248046] 一致。
- 退出码：2=参数错，3=tokenizer 加载失败，4=编码失败。
- Windows 注意：控制台 codepage 非 UTF-8 时 `--prompt` 里的非 ASCII 会以 ANSI 传入而被正确拒绝；中文/emoji 用 `--prompt-file` 或 `--stdin`。
- 构建坑记录：`OUTPUT_NAME q35` 的 exe 会把 import library 也命名为 q35.lib，覆盖同名的静态库——最终二进制定名 `qwen35.exe`。
