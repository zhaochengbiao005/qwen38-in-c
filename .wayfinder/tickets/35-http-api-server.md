# #35: HTTP API server（serve 子命令）

Label: wayfinder:task
Claim: 海鸥
Blocked by: —

## Resolution

零依赖嵌入式 HTTP/1.1 API server，`qwen35.exe serve --model <dir> --port N` 启动。

### 架构

- `src/cli/serve_cmd.c`：单文件实现，winsock2 TCP loop，仅监听 `127.0.0.1`（localhost）。
- 单线程 accept loop：一次处理一个请求（模型是单一会话状态，串行处理保证多轮上下文正确）。
- 请求解析：手写 HTTP 头解析（method/path/Content-Length），body 传给 `q35_json.c` DOM 解析器。
- 采样逻辑：从 `run_cmd.c` 镜像 `pick_token`（temperature/top-k/top-p，xorshift RNG）。

### 端点

**GET /health** → `{"status":"ok","pos":N,"layers":64}`

**POST /v1/completions**，body：
```json
{"prompt":"...", "max_tokens":N, "temperature":T, "top_k":K, "top_p":P,
 "stream":bool, "chat":bool, "reset":bool}
```

- 非流式：`{"text":"...","elapsed":S}`
- 流式：chunked transfer，NDJSON 逐 token `{"text":"..."}\n`，终帧 `{"done":true}\n`
- `chat:true` 自动包装 `<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n`
- `reset:true` 先调 `q35_model_reset`；不传则延续会话（多轮对话共享 DeltaNet state + KV cache）

### 设计取舍

- **仅 localhost**：推理引擎不暴露公网，安全面最小。
- **单请求串行**：模型状态是全局单会话，并发请求会交错污染上下文；串行是最正确的语义。生产场景需要并发时应在外面排队。
- **Connection: close**：每请求短连接，无 keep-alive（简化实现，客户端库自动重连）。
- **复用 q35_json**：零新增依赖，JSON DOM 解析已有且测试覆盖。

### 实测（2026-09-02）

- `GET /health` → 200 `{"status":"ok","pos":8,"layers":64}`
- 非流式 8 token → `{"text":" Beijing.\nThe capital of China is","elapsed":9.33}`
- 流式 4 token → 每 chunk ~800ms 到达（= decode 1.07 tok/s），NDJSON 帧格式正确
- chat + 采样 → `{"text":"\nThe user is asking a simple arithmetic question: 2+2.","elapsed":24.27}`

### 构建

CMakeLists `q35cli` target 加 `src/cli/serve_cmd.c` + `ws2_32` 链接。

Status: CLOSED
