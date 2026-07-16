# openpi 控制方式与频率飞书专题设计

## 目标

在现有飞书报告第 5 章“在线推理请求时序”末尾增加一个 openpi 控制方式与频率专题，解释 openpi 如何通过 Action Chunk 解耦大模型推理与机器人动作执行，并用源码数据比较 ALOHA、DROID、UR5e/Franka 和 LIBERO 的控制频率、重规划步数与实际推理请求频率。

## 插入位置

专题放在第 5 章现有在线推理时序、图解、消息语义和安全提醒之后，第 6 章“生产运行状态机”之前。这样读者先理解通用边缘—云请求，再看到 openpi 的具体实现，最后进入生产状态管理。

## 内容结构

### 1. 控制机制流程图

使用一张 Mermaid 流程图表达：

```text
Observation → WebSocket Client → π₀/π₀.₅ Policy Server
            → Action Chunk → Edge Action Buffer
            → 15/20/50 Hz 逐步执行
            → 执行 N 步后重新请求
```

图中区分 Cloud Policy 与 Robot Edge，明确当前官方示例采用同步请求：Action Chunk 用尽时客户端阻塞等待下一次推理。图中同时标出名义请求频率公式 `f_request = f_control / N_execute`。

### 2. 平台频率对照表

表格至少包含：

- ALOHA 真机：50 Hz，执行 25 步后请求，名义 2 Hz。
- ALOHA 仿真：50 Hz，执行 10 步后请求，名义 5 Hz。
- DROID：15 Hz，执行 8 步后请求，名义 1.875 Hz。
- UR5e / 普通 Franka：官方动作空间说明为 20 Hz，重规划步数由客户端决定。
- LIBERO：默认每 5 个环境 step 重规划，不绑定真实墙钟频率；按 10 Hz 数据时间尺度理解时约为 2 Hz。

名义频率是忽略推理阻塞的上限。对于同步客户端，近似实际频率为 `1 / (N_execute / f_control + T_infer)`。

### 3. Action Horizon 对照

列出当前源码配置：π₀/π₀.₅ 默认 50、π₀-FAST 默认 32、`pi0_droid` 为 10、`pi05_droid` 为 15、`pi05_libero` 为 10。强调模型返回长度、客户端实际执行步数和重规划频率是三个不同参数。

### 4. 推理延迟与生产含义

说明 DROID 官方 README 把每个 chunk 的 0.5–1 秒延迟视为正常，因此 15 Hz / 8 步的名义 1.875 Hz 在同步等待下通常降低到约 0.65–1.0 Hz。openpi 服务端与 Policy 会分别返回 `server_timing` 和 `policy_timing`，生产系统应统计 P50、P95、P99、deadline miss 和 stale response，而不是只看平均延迟。

### 5. 源码一致性风险

当前 checkout 中，默认 DROID 服务使用 `pi05_droid`，其 action horizon 为 15，但 DROID 客户端仍硬编码断言返回 shape 为 `(10, 8)`。专题将把它标为需要部署前统一的源码一致性风险，不把任一值描述为普适标准。

## 来源

只引用 Physical Intelligence openpi 官方仓库中的以下文件：

- `docs/norm_stats.md`
- `docs/remote_inference.md`
- `examples/aloha_real/main.py`
- `examples/aloha_sim/main.py`
- `examples/droid/main.py`
- `examples/droid/README.md`
- `examples/libero/main.py`
- `packages/openpi-client/src/openpi_client/action_chunk_broker.py`
- `src/openpi/models/pi0_config.py`
- `src/openpi/models/pi0_fast.py`
- `src/openpi/training/config.py`
- `src/openpi/serving/websocket_policy_server.py`
- `src/openpi/policies/policy.py`

## 验收标准

- 专题位于第 5 章末尾、第 6 章之前。
- 新增一张控制机制画板，整篇画板总数从 10 变为 11。
- ALOHA、DROID、UR5e/Franka、LIBERO 的频率和重规划语义均有说明。
- 区分控制频率、Action Horizon、执行步数、名义请求频率和实际请求频率。
- 包含同步阻塞、0.5–1 秒 DROID 延迟、timing 指标和 DROID horizon 不一致风险。
- 原有 11 个一级章节、10 个图解说明、关键术语与来源保持存在。

