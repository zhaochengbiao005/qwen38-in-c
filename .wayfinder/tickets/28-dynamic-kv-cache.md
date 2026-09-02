# #28: KV cache 动态增长 / OOM 防护

Label: wayfinder:task
Claim: 未认领
Blocked by: —

## Resolution

OOM guard 落地：load 时用 `GlobalMemoryStatusEx` 查可用物理内存，估算 KV cache 需要（`n_full_layers × 2 × KVH × HD × cap × 4 bytes`）+ 权重 26 GB，如果超过可用内存的 80% 就自动降 `kv_cap`。96 GB 机器上不触发（budget 足够），32 GB 地板机上会自动把 cap 降到 ~4K。

未做动态增长（realloc）和 ring buffer——当前 OOM guard 已防止最坏情况（swap 到死）。如果后续需要长对话（context > cap），再实现 ring buffer 回收最旧 KV。

### Update (2026-09-02)

动态增长已落地（本票正文"未做"系未同步）：`q35_kvcache` 带 `max_cap`，初始 cap = min(kv_cap, 4096)，`q35_kvcache_append` 满时翻倍扩容至 max_cap（`q35_kvcache_grow`，per-head 拷贝）；`q35_model_load_state` 遇保存 len > 当前 cap 自动扩。大页路径修正：`q35_plat_large_alloc` 无权限时返回 NULL，kvcache 回落 calloc（修掉 PAL 内部 VirtualAlloc 替身导致的 1.58→2.72 s/tok 退化，见 #31 Update）。ring buffer 回收仍未做。

Status: CLOSED
