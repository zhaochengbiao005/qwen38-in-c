# #21: run CLI

**What to build:** `qwen35 run --model <目录> --tokenizer <bin> --prompt ...` [--tokens N] [--dump-ids] [--threads T]，加载全量权重常住，greedy decode；加载进度可见；合成模型下先跑通。

**Blocked by:** #19

**Status:** done

- [x] 加载进度条/时间报告
- [x] greedy 循环 + EOS(248046 im_end / 248044) 停止
- [x] --dump-ids 输出 token id 序列（验收接缝）
- [x] 峰值 RSS 报告

## Resolution (2026-08-23)

qwen35 run 全链路打通（真实 27B greedy 输出 'Beijing.'，性能 ~0.09 tok/s 单线程 CPU）。另顺手根治 q35_plat_read_file 无 NUL 收尾导致的 1 字节堆越界。

- 2026-08-23 复核：上述修复落地时留下两处问题，已补修——(1) q35_plat_win.c 里 PowerShell 补丁污染写入的字面量 ` `r`n ` 夹在两条语句中间，该文件此前无法编译（build/ 里的旧 exe 是污染前产物）；(2) q35_cli.c `--prompt-file` 路径把文件读两次（首块 buffer 泄漏）且保留冗余 `prompt[n]=0`。现改为单次读取。验证：clang-cl rebuild 通过，ctest 8/8 绿，`--prompt-file` 与 `--prompt` 输出 ids 逐位一致，缺文件退出码 4。

