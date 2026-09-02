# #31: 大页 + 交互模式

Label: wayfinder:task
Claim: 未认领
Blocked by: —

## Question

两项完善：大页（MEM_LARGE_PAGES）减少 TLB miss；`--interactive` 多轮对话交互模式。

## Resolution (2026-08-24)

### 大页（已实现 PAL，未接线）

`q35_plat_large_alloc` / `q35_plat_large_free` 在 `q35_plat_win.c` / `q35_plat.h` 中实现。试 `VirtualAlloc(MEM_LARGE_PAGES)`，需要 `SeLockMemoryPrivilege`。

**未接线原因**：当前进程无 `SeLockMemoryPrivilege`，`MEM_LARGE_PAGES` 失败后 fallback 到 `VirtualAlloc(MEM_COMMIT)`——但 `VirtualAlloc` 的 demand-zero paging 比 `calloc` 慢（首次访问触发 page fault 两次），实测 KV cache 17GB 从 calloc 改 VirtualAlloc 后 1.58→2.72 s/tok（退化 72%）。

加了 LP 可用性探测（第一次试分配大页，成功则后续都用大页，失败则永久 fallback 到 calloc）。但最终决定**不接线**——无权限时帮倒忙，有权限时收益不确定（KV cache 是顺序读，TLB miss 不是主要瓶颈）。PAL 留着，有权限时可以一行接通。

### Update (2026-09-02)

接线方式修正为"权限可用即用，否则 calloc"：`q35_plat_large_alloc` 无 `SeLockMemoryPrivilege` 时返回 NULL（删除了 PAL 内部的裸 VirtualAlloc 替身——那正是本票实测否决的 demand-zero 慢路径，且曾与注释声称的 calloc fallback 相反）；`q35_kvcache_init/grow` 收到 NULL 后用 calloc，`k_large`/`v_large` 标记保证 free 路径与分配方式匹配。注释腐烂（"calloc permanently" vs 实际 VirtualAlloc）已修。

### 交互模式（已落地）

`--interactive` / `-i` 多轮对话：
- 模型只加载一次
- 每轮：stdin 读 user 输入 → `<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n` 包装 → BPE encode → prefill → greedy/采样 decode → stdout 输出 → `<|im_end|>\n` 回写模型
- 状态（DeltaNet S/conv + KV cache + position）在轮间累积，不调 `q35_model_reset`
- Ctrl-Z+Enter（EOF）退出

验证：两轮对话（"What is 2+2?" + "What is the capital of France?"）正确响应，第二轮看到第一轮上下文。ctest 9/9 无回归。

用法：
```cmd
build\qwen35.exe run --model ..\Qwen3.8-27B-FP8 --interactive --temperature 0.8 --top-p 0.9 --max-tokens 64 --threads 12
```

Status: CLOSED
