# 机器人公司 Day 1 飞书调研报告 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建并核验一篇以 10 张框图、流程图和状态图为核心的飞书调研报告，完整呈现通用机器人公司的 Day 1 基础设施设计。

**Architecture:** 先在飞书“我的空间”创建包含固定章节标题的文档骨架，获取各章节 block ID；再用 XML `block_insert_after` 向每个章节插入 Mermaid 画板和关键词解释。最后回读文档目录与全文，核验章节、画板数量、关键术语、来源链接和标题。

**Tech Stack:** `lark-cli docs`、飞书 Docx XML、Mermaid whiteboard、Python/JSON 只读校验、openpi 本地源码与文档

---

### Task 1: 飞书能力、身份与输入预检

**Files:**
- Read: `docs/superpowers/specs/2026-07-16-robotics-company-day1-infrastructure-design.md`
- Read: `/home/zky-miakho/.agents/skills/lark-shared/SKILL.md`
- Read: `/home/zky-miakho/.agents/skills/lark-doc/SKILL.md`
- Read: `/home/zky-miakho/.agents/skills/lark-doc/references/lark-doc-xml.md`
- Read: `/home/zky-miakho/.agents/skills/lark-doc/references/style/lark-doc-style.md`
- Read: `/home/zky-miakho/.agents/skills/lark-doc/references/style/lark-doc-create-workflow.md`
- Read: `/home/zky-miakho/.agents/skills/lark-doc/references/lark-doc-create.md`
- Read: `/home/zky-miakho/.agents/skills/lark-doc/references/lark-doc-fetch.md`
- Read: `/home/zky-miakho/.agents/skills/lark-doc/references/lark-doc-update.md`
- Read: `/home/zky-miakho/.agents/skills/lark-doc/references/style/lark-doc-update-workflow.md`

- [ ] **Step 1: 核验书面设计完整性**

Run: `rg -n '^## ' docs/superpowers/specs/2026-07-16-robotics-company-day1-infrastructure-design.md`

Expected: 包含目标、约束、方案、原则、模块、四平面、契约、状态机、闭环、Monorepo、CI、安全、路线、报告设计、依据和验收标准。

- [ ] **Step 2: 检查飞书用户身份**

Run: `LARKSUITE_CLI_NO_UPDATE_NOTIFIER=1 LARKSUITE_CLI_NO_SKILLS_NOTIFIER=1 lark-cli auth status --json --verify`

Expected: 当前用户身份 verified，Docs 创建、更新和读取权限可用；若缺 scope，按错误返回的最小 scope 走 split-flow 授权。

### Task 2: 创建飞书报告骨架

**Files:**
- Create: `/tmp/robotics-company-day1-feishu/report-manifest.json`

- [ ] **Step 1: 使用 XML 创建文档骨架**

创建标题为 `机器人公司 Day 1 基础设施：从 openpi 到可生产通用平台` 的文档，正文骨架固定为：

```xml
<title>机器人公司 Day 1 基础设施：从 openpi 到可生产通用平台</title>
<callout emoji="🎯" background-color="light-blue">
  <p>适用范围：3–10 人团队、1–5 台机器人、机器人形态未定；边缘安全持续运行，云端提供低频策略推理。</p>
</callout>
<h1>执行摘要</h1>
<h1>1. 约束、方案与架构决策</h1>
<h1>2. 边缘—云总体模块架构</h1>
<h1>3. Day 1 最薄垂直切片</h1>
<h1>4. 四平面数据传输</h1>
<h1>5. 在线推理请求时序</h1>
<h1>6. 生产运行状态机</h1>
<h1>7. 数据—模型—生产闭环</h1>
<h1>8. CI、测试与可观测性</h1>
<h1>9. 0–90 天路线图</h1>
<h1>10. 风险、术语与来源</h1>
```

Run: `lark-cli docs +create --as user --content -`

通过 stdin 传入 XML。Expected: 返回唯一 `document_id` 和 URL，warnings 为空或仅包含不影响正文的降级说明。

- [ ] **Step 2: 保存报告清单**

把 `document_id`、URL、标题和预期画板数 `10` 写入 `/tmp/robotics-company-day1-feishu/report-manifest.json`，不得保存 access token。

### Task 3: 获取章节 block ID

**Files:**
- Modify: `/tmp/robotics-company-day1-feishu/report-manifest.json`

- [ ] **Step 1: 获取目录与 block ID**

Run:

```bash
lark-cli docs +fetch --as user \
  --doc "$document_id" \
  --scope outline \
  --max-depth 1 \
  --detail with-ids
```

Expected: 返回 11 个一级标题及各自 block ID。

- [ ] **Step 2: 将标题映射写入清单**

保存以下稳定键到 block ID 的映射：

```text
summary
decision
architecture
day1
transport
inference
state_machine
ml_loop
quality
roadmap
risks_terms_sources
```

Expected: 11 个键均有唯一非空 block ID。

### Task 4: 写入执行摘要

**Files:**
- Read: `/tmp/robotics-company-day1-feishu/report-manifest.json`

- [ ] **Step 1: 在“执行摘要”后插入结论**

内容必须包含：

- 采用边缘—云分层；
- Safety Kernel、实时控制和最终动作裁决留在边缘；
- 云端只返回 ActionCandidate；
- Day 1 验收是仿真 + Policy Mock + Safety + CI 故障闭环；
- 不在 Day 1 建 Kubernetes、完整数据湖、真实 OTA 或真实 VLA serving；
- 生产化的判断标准是可验证、可诊断、可回滚，而不是“机器人能动”。

Run:

```bash
lark-cli docs +update --as user --doc "$document_id" \
  --command block_insert_after --block-id "$summary_block_id" --content -
```

Expected: 更新结果 success。

### Task 5: 写入方案比较与决策图

**Files:**
- Read: `docs/superpowers/specs/2026-07-16-robotics-company-day1-infrastructure-design.md`

- [ ] **Step 1: 插入三种方案比较 Mermaid 画板**

```mermaid
flowchart LR
  A[云端中心\n机器人端极薄] --> A1[优点：原型快]
  A --> A2[风险：网络进入控制关键路径]
  B[边缘—云分层\n采用方案] --> B1[边缘：安全/实时/降级]
  B --> B2[云端：推理/数据/Fleet]
  C[边缘优先\n本地运行 VLA] --> C1[优点：离线最强]
  C --> C2[代价：边缘 GPU 与运维复杂]
  style B fill:#dcfce7,stroke:#16a34a,stroke-width:3px
  style A2 fill:#fee2e2,stroke:#dc2626
  style C2 fill:#ffedd5,stroke:#f97316
```

画板后写入已确认约束和采用 B 的理由，并解释 Edge、Cloud、Safety、ActionCandidate。

Expected: 更新 result=success，返回一个 whiteboard new block。

### Task 6: 写入总体模块架构图

- [ ] **Step 1: 插入边缘—云模块框图**

```mermaid
flowchart LR
  subgraph EDGE[机器人 / Edge]
    S[01 Safety Kernel]
    H[02 Hardware HAL]
    C[03 Real-time Control]
    R[04 Robot Runtime]
    G[05 Edge Gateway]
    H --> C --> S
    R --> S
    R <--> G
  end
  subgraph CLOUD[Cloud]
    I[06 Inference Plane]
    D[07 Data Platform]
    T[08 Training & Evaluation]
    M[09 Model Registry & Release]
    F[10 Fleet Control Plane]
    O[11 Observability & Audit]
    D --> T --> M --> I
    F --> M
  end
  X[12 Shared Contracts & Developer Platform]
  G <--> I
  G --> D
  G <--> F
  EDGE -.依赖契约.-> X
  CLOUD -.依赖契约.-> X
  I -.只能返回动作候选.-> R
  I -.-x S
  style S fill:#fee2e2,stroke:#dc2626,stroke-width:3px
  style X fill:#faf5ff,stroke:#9333ea
```

图后按 12 个模块分别列出职责、输入输出、禁止依赖和早期 owner 原则。

Expected: 图中明确表示 Cloud 不能绕过 Safety Kernel。

### Task 7: 写入 Day 1 闭环与 Monorepo 图

- [ ] **Step 1: 插入 Day 1 最薄垂直切片**

```mermaid
flowchart LR
  T[Task v1] --> SIM[Deterministic Simulator + HAL]
  SIM --> RT[Robot Runtime\nObservation v1]
  RT <--> PM[Cloud Policy Mock\nActionCandidate v1]
  RT --> SS[Safety Supervisor\naccept / clamp / reject]
  SS --> ACT[Apply Action]
  ACT --> REC[Episode Record + Replay]
  REC -.CI 验证.-> T
  FI[Fault Injection\n断网/超时/坏 schema/越界动作] --> RT
  FI --> SS
```

- [ ] **Step 2: 插入 Monorepo 模块树**

```mermaid
mindmap
  root((robot-platform))
    contracts
      schemas
      compatibility tests
    edge
      hal
      control
      safety
      runtime
      gateway
    cloud
      inference-api
      policy-mock
    simulation
    observability
    deploy
    tests
    docs/adr
```

图后写入 Day 1 必须完成和明确不做清单。

Expected: 本章节新增 2 个 whiteboard。

### Task 8: 写入四平面传输图

- [ ] **Step 1: 插入四平面架构图**

```mermaid
flowchart TB
  subgraph E[机器人现场]
    L[边缘实时执行\n传感器→估计→Safety→控制器→驱动]
    P[在线推理客户端\nObservationEnvelope / ActionCandidate]
    B[批量数据代理\nepisode / manifest / chunks]
    A[Edge Agent\n设备身份 / artifact cache]
  end
  subgraph C[云端]
    IG[Inference Gateway + GPU Policy]
    OS[对象存储 + Metadata Catalog]
    FC[Fleet API + Release Controller]
  end
  L -.-x IG
  P <-->|mTLS + deadline| IG
  B -->|异步、断点续传、checksum| OS
  A <-->|pull desired state + report| FC
  style L fill:#fee2e2,stroke:#dc2626
  style IG fill:#eff6ff,stroke:#2563eb
```

图后解释四个“平面”、mTLS、deadline、manifest、checksum、对象存储和 desired/reported state。

Expected: 正文明确纠正“三平面”为“四平面”。

### Task 9: 写入在线推理时序图

- [ ] **Step 1: 插入 sequence diagram**

```mermaid
sequenceDiagram
  participant RT as Robot Runtime
  participant EG as Edge Gateway
  participant IG as Inference Gateway
  participant GPU as Policy Replica
  participant SS as Safety Supervisor
  RT->>EG: ObservationEnvelope + deadline
  EG->>IG: mTLS request
  IG->>GPU: fixed model_release_id
  GPU-->>IG: ActionCandidate + valid_until
  IG-->>EG: response + server timing
  EG-->>RT: candidate or timeout
  alt deadline 内且版本兼容
    RT->>SS: candidate
    SS-->>RT: accept / clamp / reject
  else 迟到、断网或坏版本
    RT->>RT: discard response
    RT->>SS: enter DEGRADED_SAFE
  end
```

图后解释 monotonic clock、request_id、episode_id、valid_until、action chunk、clamp/reject 和迟到响应。

Expected: 云端响应不会直接到驱动。

### Task 10: 写入生产状态机图

- [ ] **Step 1: 插入 state diagram**

```mermaid
stateDiagram-v2
  [*] --> BOOT
  BOOT --> SELF_CHECK: artifact/config 校验通过
  SELF_CHECK --> READY: 硬件/标定/Safety 正常
  READY --> RUNNING: 接受任务
  RUNNING --> PAUSED: 人工暂停
  PAUSED --> RUNNING: 受控恢复
  RUNNING --> DEGRADED: 偶发超时
  DEGRADED --> RUNNING: 网络恢复且健康
  DEGRADED --> DEGRADED_SAFE: 连续超时/断网
  DEGRADED_SAFE --> READY: 受控恢复
  BOOT --> FAULT_LATCHED: 校验失败
  SELF_CHECK --> FAULT_LATCHED: 硬件/Safety 异常
  RUNNING --> FAULT_LATCHED: 严重安全或驱动异常
  FAULT_LATCHED --> SELF_CHECK: 明确权限的受控复位
```

图后解释 watchdog、safe stop、故障锁存和为什么网络恢复不能自动解除 FAULT_LATCHED。

### Task 11: 写入数据—模型—生产闭环图

- [ ] **Step 1: 插入发布闭环**

```mermaid
flowchart LR
  P[Production Episodes] --> I[Ingest & Catalog]
  I --> C[Curate & Label]
  C --> DS[Immutable Dataset Snapshot]
  DS --> T[Train]
  T --> OE[Offline Eval]
  OE --> SE[Simulation / Replay / Fault Injection]
  SE --> SH[Shadow\n只预测不控制]
  SH --> CA[Canary\n1 台围栏机器人、低速]
  CA --> GR[Gradual Rollout]
  GR --> P
  X{任一门禁失败?} -->|是| RB[停止扩量 / 回滚]
  OE --> X
  SE --> X
  SH --> X
  CA --> X
```

图后定义 ModelRelease 的完整字段，并解释 Shadow、Canary、dataset snapshot、data lineage、P95/P99 和四类门禁指标。

### Task 12: 写入 CI、测试与可观测性图

- [ ] **Step 1: 插入测试金字塔**

```mermaid
flowchart TB
  R[围栏真实机器人\n少量关键验收 / Canary]
  H[Hardware-in-the-Loop]
  S[仿真 / 回放 / 故障注入]
  U[单元 / 属性 / 契约 / 静态检查]
  U --> S --> H --> R
  style U fill:#dcfce7,stroke:#16a34a
  style R fill:#fee2e2,stroke:#dc2626
```

图后写入 CI 自动步骤，并用表格解释 Logs、Metrics、Traces、Episode Record、Safety Event 和共同关联 ID。

Expected: 强调真实机器人测试不能替代下层自动化测试。

### Task 13: 写入 0–90 天路线图

- [ ] **Step 1: 插入时间线**

```mermaid
timeline
  title 通用机器人平台 0–90 天路线
  Day 1 : Monorepo 与 contracts : 仿真 HAL : Runtime + Safety : Policy Mock : CI 故障闭环
  Day 2–14 : 接第一台真实机器人 : 围栏低速任务 : episode 记录 : 远程推理 MVP
  Day 15–45 : 数据目录与快照 : 训练/评测 : ModelRelease : Shadow/Canary : 基础 Fleet
  Day 46–90 : 3–5 台机器人 : SLO/告警 : 灰度回滚演练 : 值班与故障手册 : 第二种 robot profile
```

图后写入每阶段验收证据、owner 建议和升级条件。

### Task 14: 写入风险、术语和来源

- [ ] **Step 1: 写入风险清单**

必须包含：

- 把云端推理误放进实时/安全关键路径；
- Observation/Action 单位或坐标系漂移；
- 权重与 config/transforms/norm stats 混用；
- 日志上传阻塞控制；
- 机器人自动解除锁存故障；
- 过早引入 Kubernetes、多云和完整平台；
- Safety owner 与模型发布权限未分离；
- 只看平均延迟而忽略 P95/P99。

- [ ] **Step 2: 写入术语表**

至少解释：CI、Edge、HAL、Safety Supervisor、watchdog、safe stop、ObservationEnvelope、ActionCandidate、action chunk、deadline、monotonic clock、mTLS、WebSocket、gRPC、QUIC、manifest、checksum、object storage、Fleet、Edge Agent、desired/reported state、OTA、rollout、rollback、Shadow、Canary、ModelRelease、P95/P99、Logs/Metrics/Traces、FAULT_LATCHED。

- [ ] **Step 3: 写入来源**

写入并链接：

- `https://github.com/Physical-Intelligence/openpi`
- openpi `docs/remote_inference.md` 和 WebSocket server/client 源码链接；
- `https://docs.ros.org/en/rolling/Tutorials/Demos/Real-Time-Programming.html`
- `https://www.iso.org/standard/73933.html`
- `https://www.iso.org/standard/62996.html`

将 OpenTelemetry、SLSA、TUF/Uptane、MCAP 标记为“后续技术选型参考”，不描述为本次已完整核实的事实依据。

### Task 15: 远端回读与图表核验

**Files:**
- Create: `/tmp/robotics-company-day1-feishu/fetched.xml`
- Create: `/tmp/robotics-company-day1-feishu/verification.json`

- [ ] **Step 1: 回读目录**

Run:

```bash
lark-cli docs +fetch --as user --doc "$document_id" \
  --scope outline --max-depth 2 --detail with-ids
```

Expected: 11 个一级章节均存在且顺序正确。

- [ ] **Step 2: 回读全文**

Run:

```bash
lark-cli docs +fetch --as user --doc "$document_id" --detail full
```

Expected: XML 中有 10 个 `<whiteboard token=`；正文包含 `Safety Kernel`、`ObservationEnvelope`、`ActionCandidate`、`ModelRelease`、`FAULT_LATCHED`、`CI` 和 `四平面`。

- [ ] **Step 3: 校验来源和降级**

Expected: openpi、ROS 2、ISO 10218 和 ISO/TS 15066 链接均存在；写入结果没有 failed/partial_success；若 Mermaid 出现降级 warning，定位对应章节并重新写入简化 Mermaid。

- [ ] **Step 4: 返回交付结果**

向用户提供文档 URL、画板数量 `10`、回读结果和任何已知表达降级。不得只凭写入返回值宣称完成。
