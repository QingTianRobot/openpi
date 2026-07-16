# 通用机器人公司 Day 1 基础设施设计

## 1. 目标

为一个 3–10 人、初期只有 1–5 台机器人、机器人形态尚未确定的创业团队设计第一天就应建立的软件基础设施。最终目标是让机器人系统的各个模块能够独立演进，并沿着可验证、可诊断、可回滚的路径进入围栏生产环境。

Day 1 的验收不是接入真实机器人或训练真实 VLA，而是完成以下最薄垂直切片：

```text
Task v1
  → 确定性仿真环境和 HAL
  → Robot Runtime 生成 Observation v1
  → 云端 Policy Mock 返回 ActionCandidate v1
  → 边缘 Safety Supervisor accept / clamp / reject
  → 仿真执行动作
  → 记录可回放 episode
```

同一条链路必须在 CI 中自动注入断网、超时、坏协议、旧版本和越界动作，并产生确定的状态转换与审计事件。

## 2. 已确认的产品约束

- 首款机器人形态暂未确定，基础设施必须通过硬件抽象、机器人 profile 和版本化契约保持通用。
- 安全、状态估计、轨迹和电机级控制必须留在边缘端；云端只提供低频策略推理、数据、训练和 Fleet 管理。
- 云端断网或推理超时时，边缘安全控制继续运行，并进入可观测的降级状态。
- 首批部署在围栏隔离区域，后续才扩展到人机协作；协作场景需要独立安全评估和更严格的发布门禁。
- Day 1 面向 3–10 人团队和 1–5 台机器人，不为多云、多区域、数百台设备提前建设复杂平台。
- Day 1 要交付仓库骨架、接口契约、CI、仿真闭环和云端推理 Mock，不要求接通真实机器人。
- 飞书调研报告以框图、流程图、状态图和路线图为主要表达形式，并为关键词提供中文解释。

## 3. 方案比较与决策

### 3.1 方案 A：云端中心

机器人端只采集传感器并执行云端命令，任务编排、策略推理和主要决策集中在云端。

优点是原型开发快、集中管理简单。缺点是网络进入控制关键路径，云端超时和断网直接影响机器人行为，也难以证明安全回路的确定性。

该方案不满足已确认的离线安全要求，不采用。

### 3.2 方案 B：边缘—云分层

边缘承担安全、硬件抽象、实时控制、任务运行时、动作队列和离线降级；云端承担低频策略推理、数据平台、训练评测、模型注册、Fleet 管理和聚合可观测性。两侧只通过版本化契约交互。

该方案既复用 openpi 的轻量客户端、远程策略服务和 action chunk 模式，又补上生产需要的安全域、设备身份、失败状态、发布门禁和审计链路。

这是已确认的采用方案。

### 3.3 方案 C：边缘优先

安全、控制、任务和 VLA 推理全部运行在机器人计算机上，云端只负责训练、归档和 OTA。

该方案离线能力最好，但 Day 1 就要承担模型压缩、边缘 GPU、硬件矩阵和设备运维，对小团队范围过重。保留为未来网络条件或产品约束要求时的部署 profile，不作为默认路线。

## 4. 总体设计原则

1. **安全域不依赖云端。** Safety Kernel 不依赖模型、网络、云端或 ROS graph。
2. **云端只能返回动作候选。** ActionCandidate 经过 deadline、协议和 Safety Supervisor 检查后才能成为最终动作。
3. **数据与控制分离。** 电机和轨迹控制不经过公网；视频、回放和遥测异步上传。
4. **契约优先。** Robot Runtime 依赖 Observation、Action、Task 和 ArtifactManifest 契约，不直接依赖具体模型、云厂商或硬件驱动。
5. **模型发布单元不可拆分。** 权重、模型配置、transforms、normalization statistics、协议版本和评测报告共同组成 ModelRelease。
6. **失败是显式状态。** 超时、断网、旧版本、传感器异常和 Safety 拒绝都必须触发确定状态和错误码。
7. **默认拒绝未知语义。** 未知硬件 profile、动作单位、坐标系或 major schema version 不得自动执行。
8. **先仿真和回放，再真实机器人。** 真实机器测试位于测试金字塔顶端，只承担少量关键验收。
9. **期望状态由机器人主动拉取。** Fleet 平面不依赖公网入站 SSH 或直接远程 shell。
10. **每个模块只有一个明确 owner。** 早期可以一人拥有多个模块，但 Safety 的批准权限与模型团队分离。

## 5. 模块边界

### 5.1 Edge：机器人与现场域

#### 01 Safety Kernel / Safety Supervisor

职责：

- E-stop、安全门、硬限位和安全 PLC 接口；
- 速度、位置、力矩、工作空间和碰撞风险限制；
- watchdog、safe stop、故障锁存和受控复位；
- 对 ActionCandidate 做 accept、clamp 或 reject；
- 生成不可修改的安全事件。

依赖：只依赖硬件安全接口、已验证的限制配置和本地单调时钟，不依赖模型或云端。

Day 1 只实现仿真 Safety Supervisor 和接口，不宣称实现认证级安全功能。

#### 02 Hardware Abstraction Layer（HAL）

职责：

- 统一不同电机、机械臂、移动底盘、相机和传感器接口；
- 明确单位、坐标系、时间戳、标定版本和设备能力；
- 提供仿真 adapter 与真实硬件 adapter 的同构接口；
- 暴露 robot profile，不把硬件细节泄漏给模型和任务层。

#### 03 Real-time Control

职责：

- 状态估计、轨迹插值和 servo/control loop；
- 满足现场控制周期和最大允许 jitter；
- 消费经过 Safety 检查的目标，而不是直接消费云端消息；
- 记录控制延迟、deadline miss 和驱动状态。

ROS 2 实时指南强调实时循环必须避免 page fault、动态分配和无期限阻塞。本模块因此不能依赖网络返回或普通云服务时延。

#### 04 Robot Runtime

职责：

- 任务状态机、episode 生命周期和 task/episode/request ID；
- 采集 HAL observation，并构造 ObservationEnvelope；
- 管理 action chunk、重新规划频率和动作队列；
- 管理 CONNECTED、DEGRADED、DEGRADED_SAFE 等运行状态；
- 将候选动作交给 Safety，而不直接操作驱动。

该模块可吸收 openpi-client 的 `Environment`、`Agent`、`Runtime` 和 `ActionChunkBroker` 思路，但需要增加 deadline、失败状态和持久化事件。

#### 05 Edge Gateway

职责：

- 设备身份、mTLS、消息大小限制、限流和重连退避；
- 在线推理协议适配和云端 endpoint 选择；
- episode 本地环形缓冲、断点续传和 artifact cache；
- Fleet 期望状态拉取、结果回报和本地发布代理；
- 本地 telemetry collector 和诊断包生成。

### 5.2 Cloud：云端智能与运营域

#### 06 Inference Plane

职责：

- 认证、租户/机器人限流、deadline 和请求大小检查；
- 按 model_release_id、robot profile 和区域路由到已预热 GPU replica；
- 返回 ActionCandidate、valid_until、模型版本和服务耗时；
- 防止旧请求、重复请求和迟到响应污染当前 episode；
- 聚合 P50/P95/P99 延迟、deadline miss 和错误码。

Day 1 使用确定性 Policy Mock。首版真实推理可复用 openpi WebSocket + msgpack，但必须补上 mTLS、版本、deadline、ID 和重连语义；契约保持与传输协议无关，未来可迁移到 gRPC/QUIC。

#### 07 Data Platform

职责：

- 对象存储保存不可变原始 episode、视频和诊断包；
- metadata catalog 保存 robot、task、时间、版本、结果和文件位置；
- 数据质量、去重、采样、隐私和标注状态；
- dataset snapshot 和 data lineage；
- 失败、人工干预和罕见场景优先回流。

#### 08 Training & Evaluation

职责：

- 从不可变 dataset snapshot 启动训练；
- 保存代码提交、config、seed、依赖、训练指标和 checkpoint；
- 运行 held-out、replay、OOD、仿真和故障注入评测；
- 生成结构化 evaluation report；
- 不直接发布到机器人，只向 Model Registry 提交候选产物。

#### 09 Model Registry & Release

职责：

- 把权重、config、transforms、normalization、schema、兼容 robot profile 和评测报告绑定为 ModelRelease；
- 保存内容 digest、构建来源、签名和 rollback release；
- 管理 Draft、Evaluating、Approved、Canary、Production、Revoked 状态；
- 禁止只替换权重而继续复用未知配置或统计量。

#### 10 Fleet Control Plane

职责：

- 设备注册、证书、分组、能力和当前版本；
- 期望配置、软件/模型 release、灰度批次和维护窗口；
- 机器人主动 pull、安装结果回报、自动停止扩量和回滚；
- 远程诊断请求、权限、审批和审计；
- 不直接发送电机命令。

#### 11 Observability & Audit

职责：

- metrics、structured logs、distributed traces 和 safety events；
- 使用 robot_id、task_id、episode_id、request_id、model_release_id、software_release_id 贯通边缘和云端；
- 任务成功率、人工干预率、推理尾延迟、断线率、Safety reject/clamp、温度和资源指标；
- SLO、告警、事件时间线和诊断包；
- 发布、复位、远程操作和权限变化的审计日志。

### 5.3 Shared：契约与开发者平台

#### 12 Shared Contracts & Developer Platform

职责：

- Observation、ActionCandidate、Task、ArtifactManifest 和 event schema；
- robot profile、单位、坐标系和兼容策略；
- Edge/Cloud SDK、仿真 fixture、录制回放和故障注入；
- config schema、代码生成和 compatibility tests；
- CI、artifact build、SBOM/签名入口和 ADR；
- 本地开发环境和最小 Docker Compose。

该模块不承载生产运行时决策，只提供契约和工具。

## 6. 四平面通信设计

### 6.1 边缘实时执行平面

范围：传感器、状态估计、Safety、轨迹控制和驱动。

特点：毫秒级周期、确定 deadline、现场总线/进程内/shared memory、不能经过公网。

### 6.2 在线推理平面

范围：ObservationEnvelope 到 ActionCandidate。

特点：低频 1–10 Hz、硬 deadline、双向认证、消息大小上限、模型版本固定、迟到响应丢弃。

建议消息包含：

- robot_id、task_id、episode_id、request_id；
- monotonic timestamp 和 deadline；
- observation schema version、robot profile 和 calibration version；
- images、state、task/prompt；
- 返回的 model_release_id、actions、valid_until 和 server timing。

### 6.3 批量数据平面

范围：视频、传感器序列、动作、事件、回放和诊断包。

特点：异步、允许延迟、可采样、可断点续传，不阻塞机器人运行。

推荐流程：

```text
episode 本地落盘
  → 生成 manifest、chunk 和 checksum
  → 网络可用时异步上传
  → 对象存储保存文件
  → metadata catalog 保存索引和血缘
```

### 6.4 Fleet 管理平面

范围：设备身份、期望配置、软件/模型发布、健康上报和诊断。

特点：秒/分钟级、强身份、签名 artifact、全量审计。机器人主动拉取期望状态，不暴露公网入站控制端口。

## 7. 核心契约

### 7.1 ObservationEnvelope

封装机器人 observation 和生产元数据。字段语义必须明确单位、坐标系、图像编码、时间戳来源和 calibration version。

### 7.2 ActionCandidate

云端模型提出的动作候选。它必须包含 request_id、model_release_id、valid_until、action frame/unit 和 action chunk；边缘端有权丢弃、clamp 或 reject。

### 7.3 Task

描述任务目标、允许区域、速度/力限制、超时、成功条件和恢复策略。自然语言 prompt 只是 Task 的一个字段，不能代替结构化安全约束。

### 7.4 ModelRelease

至少绑定：

- release_id；
- weights digest；
- model config digest；
- transforms digest；
- normalization statistics digest；
- observation/action schema version；
- compatible robot profiles；
- dataset snapshot ID；
- source commit 和 build provenance；
- evaluation report ID；
- runtime 最低版本；
- artifact signature；
- rollback release ID。

### 7.5 ArtifactManifest

描述软件、模型、配置或 episode 文件的内容、digest、大小、版本、签名和依赖。Fleet 和批量数据平面复用同一套内容寻址和完整性思想，但权限与生命周期不同。

### 7.6 兼容策略

- 当前版本必须兼容前一个 minor 版本；
- 未知字段可以忽略，但字段语义变化必须升 major；
- 未知 major version、单位、坐标系或 robot profile 必须拒绝；
- schema compatibility test 在 CI 中执行；
- 数据记录必须保存当时的 schema 和 release IDs，保证未来可回放。

## 8. 运行状态机

### 8.1 正常状态

- `BOOT`：校验软件、模型、配置和签名；
- `SELF_CHECK`：检查驱动、传感器、标定和安全回路；
- `READY`：等待任务；
- `RUNNING`：正常执行；
- `PAUSED`：可恢复的人工暂停。

### 8.2 退化和故障状态

- `CONNECTED`：推理在 deadline 内返回；
- `DEGRADED`：偶发超时，降低速度或缩短 chunk 执行；
- `DEGRADED_SAFE`：连续超时/断网，安全中止并停止或回安全位；
- `FAULT_LATCHED`：Safety、驱动、状态或传感器严重异常，禁止自动恢复。

`FAULT_LATCHED` 只能通过明确权限和受控流程复位。云端恢复连接不能自动解除锁存故障。

## 9. 数据—模型—生产闭环

```text
Production Episodes
  → Ingest & Catalog
  → Curate & Label
  → Immutable Dataset Snapshot
  → Train
  → Offline Eval
  → Simulation / Replay / Fault Injection
  → Shadow（只预测不控制）
  → Canary（1 台围栏机器人、低速）
  → Gradual Rollout
  → Production Episodes
```

门禁指标分四类：

- 安全：越界候选、Safety reject/clamp、碰撞、E-stop、锁存故障；
- 系统：P50/P95/P99 推理延迟、deadline miss、断线、CPU/GPU/温度；
- 任务：成功率、完成时间、人工干预和重试；
- 数据：相机/状态覆盖、漂移、NaN、标签质量和场景分布。

任何门禁失败都不能进入下一阶段。生产指标恶化时停止扩量并回滚。

## 10. Day 1 Monorepo 骨架

```text
robot-platform/
├── contracts/                  # schema、IDL、compatibility tests
├── edge/
│   ├── hal/                    # 硬件接口与仿真 adapter
│   ├── control/                # 状态估计和轨迹接口
│   ├── safety/                 # supervisor、limits、watchdog
│   ├── runtime/                # task 状态机、action broker
│   └── gateway/                # cloud client、本地队列、telemetry
├── cloud/
│   ├── inference-api/          # 推理契约 endpoint
│   └── policy-mock/            # 确定性测试策略
├── simulation/                 # 确定性环境与测试场景
├── observability/              # event schema 和 collector 配置
├── deploy/                     # 本地 compose 和镜像 manifest
├── tests/                      # contract、integration、fault injection
└── docs/adr/                   # 架构决策记录
```

### 10.1 Day 1 必须完成

- Observation、ActionCandidate、Task、ArtifactManifest v1；
- 仿真 HAL、Robot Runtime 和 Safety Supervisor；
- Policy Mock、deadline 和断网状态；
- 结构化 event 和 episode replay；
- CI：单元、契约、闭环和故障注入；
- edge-runtime 与 policy-mock 两个可复现镜像；
- ADR：实时域、网络边界、版本策略、日志关联键。

### 10.2 Day 1 明确不做

- Kubernetes、service mesh、多云和多区域；
- 完整数据湖、标注平台和 GPU 调度；
- 真实 OTA、远程接管和生产 Fleet 控制面；
- 真实 VLA 模型；
- 认证级 Safety PLC 实现；
- 为所有未知机器人预先统一消息；
- 与人共享空间的协作安全认证。

## 11. CI 与测试体系

### 11.1 CI 定义

CI（Continuous Integration，持续集成）在每次提交后自动构建和验收代码。

Day 1 CI 必须：

1. 静态检查和单元测试；
2. schema compatibility test；
3. 构建 edge-runtime 和 policy-mock 镜像；
4. 启动仿真闭环；
5. 注入断网、deadline miss、坏 schema、旧 release 和越界动作；
6. 验证状态机、安全拒绝和 episode 回放；
7. 只有全部通过才允许合并或构建候选 artifact。

### 11.2 测试金字塔

从低成本到高成本：

1. 单元、属性、契约和静态检查；
2. 仿真、回放和故障注入；
3. Hardware-in-the-Loop；
4. 围栏真实机器人验收和 Canary。

真实机器人测试不能替代下层自动化测试。

## 12. 安全与供应链

- 每台机器人使用独立设备身份和可轮换证书；
- 在线推理和 Fleet 通信使用 mTLS；
- 机器人默认无公网入站端口和共享 SSH 密钥；
- 软件、模型、配置和固件使用内容 digest 与签名 manifest；
- 发布记录构建来源、依赖、SBOM、审批人和目标设备；
- 新版本必须支持灰度、停止扩量和回滚；
- Safety 配置、复位和远程诊断采用最小权限并保存审计；
- 后续 OTA 设计参考 TUF/Uptane 风格的签名元数据与回滚保护，不在 Day 1 自研完整更新系统。

## 13. 0–90 天路线图

### Day 1：固定边界

- Monorepo、contracts、仿真 HAL；
- Robot Runtime、Safety Supervisor、Policy Mock；
- CI 正常/故障闭环和 episode replay。

### Day 2–14：接第一台机器人

- 真实 HAL 和 robot profile；
- 现场安全风险评估；
- 围栏、低速、单任务；
- 真实 episode 记录和远程推理 MVP。

### Day 15–45：建立学习闭环

- 对象存储、metadata catalog 和 dataset snapshot；
- 训练、回放、仿真评测和 ModelRelease；
- Shadow、Canary、基础 Fleet/OTA 和发布审计。

### Day 46–90：小规模生产

- 3–5 台机器人；
- SLO、告警、诊断包和值班流程；
- 灰度、回滚和断网演练；
- 故障手册和受控复位；
- 第二种 robot profile 验证通用边界。

## 14. 飞书调研报告设计

创建一篇飞书文档，标题为 `机器人公司 Day 1 基础设施：从 openpi 到可生产通用平台`。报告主要由 8–10 张 Mermaid 画板和配套短文构成。

建议图表：

1. 三种总体路线比较图；
2. 边缘—云总体模块框图；
3. Day 1 最薄垂直切片流程图；
4. 四平面数据传输图；
5. 在线推理 sequence diagram；
6. 机器人运行状态机；
7. 数据—模型—发布闭环；
8. 测试金字塔；
9. Monorepo 模块树；
10. 0–90 天路线图。

每张图后必须包含：

- 图中关键词的中文解释；
- Day 1 做什么、暂不做什么；
- 模块 owner 和输入输出；
- 主要失败模式和验收信号。

报告正文结构：

1. 执行摘要；
2. 已确认约束；
3. 方案比较与决策；
4. 总体模块架构；
5. 四平面通信；
6. 核心契约和状态机；
7. 数据与模型闭环；
8. CI、测试和可观测性；
9. Day 1 仓库骨架；
10. 0–90 天路线；
11. 风险清单；
12. 术语表与来源。

## 15. 研究依据与边界

已读取并用于本设计的材料：

- openpi 的 `README.md`、`docs/remote_inference.md`、`src/openpi/serving/websocket_policy_server.py`、`packages/openpi-client` runtime 与 WebSocket client、训练配置和 Docker Compose 示例；
- ROS 2 官方 Real-Time Programming 指南：实时循环需要满足 deadline，并避免 page fault、动态内存分配和无期限阻塞；
- ISO 10218-1/2 的机器人本体与系统集成安全边界；
- ISO/TS 15066 对协作式工业机器人系统的补充安全要求。

后续实现阶段的参考候选包括 OpenTelemetry、SLSA、TUF/Uptane 和 MCAP。外部文档服务在本次调研中部分连接失败，因此正式飞书报告只把已核实材料作为事实依据；其余候选明确标注为后续技术选型参考，不伪装成已验证结论。

## 16. 验收标准

- 书面设计明确 12 个模块的职责和禁止依赖；
- 明确四类通信平面的实时性、可靠性和安全语义；
- 明确 Observation、ActionCandidate、Task、ModelRelease 和 ArtifactManifest 契约；
- 明确正常、退化、安全停止和锁存故障状态；
- Day 1 具有可执行的 Monorepo 骨架、CI 闭环和非目标清单；
- 报告包含 8–10 张可读的框图/流程图，并配中文术语解释；
- 飞书文档包含 openpi、ROS 2 和 ISO 的已核实来源；
- 不把云端推理放入硬实时或安全关键路径；
- 不把 Day 90 能力伪装成 Day 1 必须完成。
