# #20: tiny oracle 数值验收（四层测试金字塔核心）

**What to build:** Python 参考前向（按 modeling_qwen3_5.py 抽出的迷你 Qwen3.5，纯 numpy/torch 均可）+ 一个手工构造的小 checkpoint（含 FP8 量化路径走通：参考侧按 BF16 读等价 dequant 权重），engine 与参考在 teacher-forcing / greedy / incremental 三路径逐 token 一致；容差声明在 fixture manifest。

**Blocked by:** #19

**Status: done**

- [x] tiny 模型权重由工具生成并入库（几 MB 内）
- [x] 三路径 token id 逐位一致
- [x] logits 相对误差在声明容差内（f32）

## Resolution (2026-08-23)

`tools/gen_oracle_fixture.py` + `tests/test_oracle.c`（ctest 名 `oracle`），fixture 在 `tests/fixtures/oracle/`（1.9 MB，171 张量）。模型 8 层 [lin×3, full]×2（interval 4 同真实）、H=160/II=192/GQA 6:1/Hv:Hk=3，FP8 块网格带不完整边块（qkv 320×160 → scale [3,2]）。numpy f64 参考跑在存储量化值上（FP8 经 block scale 反量化），与引擎零共享代码。

最终门禁数据：teacher-forcing 14 位置全序列 maxabs **2.6e-5**（声明容差 atol=1e-3/rtol=1e-2，写在 oracle.json manifest，测试不硬编码）；greedy 8 token ids 逐位一致；incremental 逐 token decode 推进后 ids 一致且最终 logits 与 prefill 路径 **FNV 比特一致**。ctest 9/9 绿。

### 过程中抓到的 bug（都在参考侧，引擎自证清白）

1. **参考的 KV cache 全局共享**：两个 full attention 层往同一个 numpy 数组 append，第二层 attend 到第一层的 k/v。单 attention 层模型踩不到——#19 因此从未暴露。层结构二分（单 attn 干净/双 attn 崩）定位。改成每层独立 cache。
2. **参考的 RMSNorm 全局平均**：`np.mean(x*x)` 在 2-D q [QH,HD] 上跨头平均，numspec 要求 per-head。KVH=1 时 k 不受影响（单头全局=本地），q 全错。引擎 per-head 实现正确。已同步修复 `gen_model_fixture.py`（#19 同款潜伏 bug）并重新生成 fixture，test_model 仍绿。
3. test_oracle 自身的 argmax 断言错误：prompt 是任意选的 token，pos<np-1 处 argmax≠next 是预期，断言只对 greedy 生成段（t≥np-1）有效。

### 侦探工具（保留复用）

- `Q35_DUMP=<file>` 环境变量：引擎逐层中间量 dump（model 层 + attention 内部 q/k/softmax 概率/门控，tag 见 q35_attn.c 头注释）；生成器 `ORACLE_DUMP=<file>` 输出同格式参考 dump；`tools/cmp_dump.py` 逐记录对比。这次靠它在 20 分钟内把"0.6 的偏差"钉死到"q post-norm 与 pre-norm 之间"。
- `tools/dbg_reset.c`（CMake target `dbg_reset`）：验证 reset 后 decode 与新鲜加载逐位一致。
- 生成器支持 `ORACLE_NLAY/ORACLE_ATTN_AT/ORACLE_ALL_LINEAR/ORACLE_FULL_ONLY/ORACLE_H/QH/HD/KVH/THETA` 环境变量做层结构/维度探针。

### 方法论教训（对应蓝本 verify_mla.py 的忠告）

"引擎对参考、参考对错了"的循环自证真实发生：两次都是参考错、引擎对。防线是 (a) 独立于引擎的参考 + (b) 层内逐算子 dump 对拍——只看端到端 logits 会把参考的锅扣到引擎头上。另：pos 0 的单条目 softmax 恒为 1，天然掩盖 q/k 路径错误；单 attention 层模型掩盖共享状态错误——tiny oracle 的拓扑必须包含 ≥2 个同类型层。
