# #10: Tokenizer 转换工具

**What to build:** 从命令行运行离线转换工具，输入官方 `tokenizer.json` + `tokenizer_config.json`，输出自定义紧凑二进制 `tokenizer.bin`（byte-level BPE 词表 + 合并表 + pre-tokenizer split 模式 + 特殊 token 表），并在转换结束时自校验（词表/merges/特殊 token 计数与源逐项一致，不一致即以非零退出码失败）。

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] 给定本地或镜像下载的官方 tokenizer 文件，产出 `tokenizer.bin`
- [ ] 词表条目数 = 248320（含特殊 token），merges 数与源一致
- [ ] 特殊 token（bos=eos=248044, image/video 等）在 blob 中带 id 与文本表示
- [ ] blob 每个字段有魔数/版本/长度前缀，格式文档写在工具注释里
- [ ] 自校验失败退出码非零

Claim: 已认领（charting 后续会话，2026-08-21）


## Resolution
（2026-08-21 解决，CLOSED）

落地：`qwen35-c/tools/convert_tokenizer.py` + 产物 `qwen35-c/tokenizer/tokenizer.bin`（5,809,877 bytes）。

关键事实修正：词表定义条目 248077（base 248044 + added 33），config 声明 248320（尾部 padding）；merge 数 247587；特殊 token 33 个中 21 个 special；blob 记录 bos=eos=248044，chat_eos(<|im_end|>)=248046。

研究发现（写进后续票）：源 tokenizer 带 **NFC normalizer**——input 先做 NFC 归一化再 pretokenize；已在 blob PRET/META flags 中固化（bit1），C 侧 #11/#12 必须实现 NFC（或文档化降级路径）。

验收证据：转换器对每个 fixture 用 blob 内置表做参考重编码，与 HF `tokenizers` 库输出 9/9 byte-exact；自校验失败路径以退出码 1 关闭。
