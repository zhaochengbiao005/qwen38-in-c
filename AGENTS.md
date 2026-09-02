# QWEN3.8-27B_JSZ — Agent Instructions

本仓库目标：把 kimi-k3-in-c 的纯 C99 推理引擎思路移植到 Windows，适配 `Qwen/Qwen3.8-27B-FP8`。

关键背景文档：
- 调研报告：`kimi-k3-in-c-技术报告.md`
- Wayfinder 工程地图：`.wayfinder/map.md`（干活前先读；tickets 在 `.wayfinder/tickets/`，研究产物在 `.wayfinder/research/`）

## Agent skills

### Issue tracker

Issues 以本地 markdown 文件形式存放在 `.wayfinder/`（map + tickets + research）。See `docs/agents/issue-tracker.md`.

### Triage labels

使用默认五个规范角色标签（`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`），记录在票据文件的 `Status:` 行。See `docs/agents/triage-labels.md`.

### Domain docs

单 CONTEXT 布局：根目录 `CONTEXT.md` + `docs/adr/`，按需懒创建。See `docs/agents/domain.md`.
