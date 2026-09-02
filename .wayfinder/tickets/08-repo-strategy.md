# Ticket-08: 代码库策略（task, HITL）

Label: wayfinder:task
Claim: 未认领（等待用户）
Blocked by: Ticket-02

## Question

A) fork kimi-k3-in-c 原地改（历史连续、对比容易、结构熟悉）；B) 新建仓库 rwr-clean-room，把 K3 的可复用件（safetensors reader、LRU 框架、测试脚手架）拷贝过来（干净、无残姚，但丢 git 历史）；C) 同 repo 加抽象层并存双模型（维护成本高，但 Windows 改动可以上upstream）。用户偏好直接决定后续所有文件的归属。

## Resolution
（2026-08-21 用户拍板，CLOSED）

选 **B：新建干净仓库**。从 kimi-k3-in-c 拷贝可复用件：safetensors reader（k3_st.c 改造 Windows 化）、LRU 缓存框架（若全驻留则退化为通用 arena/分配器，按需裁剪）、测试脚手架（fixture 容器、FNV-1a 比特哈希比对门、tiny oracle 生成思路）。放弃 git 历史连续性，README 里注明蓝本出处与致谢。新仓库建议名 `qwen35-c`（最终命名开工时定）。
