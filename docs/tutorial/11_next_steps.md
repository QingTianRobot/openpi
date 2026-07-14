# 第 11 章：规划下一阶段实验

[← 上一章：服务与闭环评测](10_serve_and_evaluate.md) · [课程入口：README](README.md) · [下一页：课程术语表 →](glossary.md)

## 本章目标

完成本章后，你应该能够：

- 盘点自己已经跑通的数据、训练、checkpoint、服务与评测能力；
- 在“扩大数据与正式评测”“完整微调 `pi05_libero`”“接入自有数据或机器人”三条路线中做出有依据的选择；
- 为每条路线分别写清保持不变的协议、允许改变的组件、资源前提、分阶段里程碑、回滚方式和成功证据；
- 区分工程闭环、模型质量和真实机器人安全，不用单次 loss、单个视频或一次成功替代正式评测；
- 为下一次实验留下可复查、可中止、可恢复的记录。

遇到陌生术语可查[课程术语表](glossary.md)；出现跨层故障时查[故障排查手册](troubleshooting.md)。

## 前置条件

本章以你已经完成第 01–10 章为前提，至少保存了以下证据：

1. 一份能离线读取的小规模 LeRobot 数据集及字段、shape、dtype、动作语义检查；
2. 与数据/config 对应且数值有限的 normalization statistics；
3. 一次短程 LoRA 训练日志，以及至少一个包含 `train_state`、`params`、`assets` 的已提交 checkpoint；
4. 一次 Policy Server 加载记录和一次 LIBERO episode 推进记录；
5. 明确区分 `pi0_libero_tutorial`（π₀ LoRA 教学路线）与 `pi05_libero`（π₀.₅ 完整微调路线）。

如果这些证据尚不齐，不必直接扩大规模。先回到相应章节补齐验收，通常比把同一错误放大到更多数据、更多 GPU 或真实机器人上更省时间。

## 核心概念

### 你已经具备的能力

完成主线并不意味着模型已经达到可部署性能，但意味着你已建立一条可验证的工程链路：

```text
episode / frame
    → LeRobot dataset
    → repack / data / model transforms
    → normalization statistics
    → training state + params + assets
    → Policy Server
    → WebSocket client
    → action chunk
    → LIBERO episode 与视频证据
```

这条链路中最值得保留的不是某个命令，而是四类契约：

- **数据契约**：字段名、相机语义、state/action 维度、dtype、单位、坐标系、控制频率和 prompt 来源；
- **模型契约**：模型类型、action dimension、action horizon、tokenization、delta/absolute 语义和参数树；
- **资产契约**：config name、repo id、asset id、normalization statistics、基础权重和 checkpoint step 一一对应；
- **评测契约**：固定任务集合、初始条件、trial 数、随机种子、超时/步数上限、成功定义和原始日志/视频保存方式。

下一阶段的原则是：**一次只主动改变少量因素，其他契约保持可追踪**。否则结果变化时无法判断来自数据、transform、训练、checkpoint 还是评测条件。

### 三条路线怎样选择

| 你的主要问题 | 推荐路线 | 首个可交付成果 | 主要风险 |
| --- | --- | --- | --- |
| 教学闭环已通，想知道模型质量是否随数据增长而改善 | 路线 A：扩大数据、任务与正式评测 | 一份冻结协议的 baseline 与分任务指标表 | 数据分布和评测条件同时变化，导致结论不可比较 |
| 有高算力并要研究 π₀.₅ 的 LIBERO 完整微调 | 路线 B：`pi05_libero` 全参数训练 | 静态配置审计、短 smoke run 和资源曲线 | 显存/算力成本高；单卡或 30k step 不保证复现官方结果 |
| 有自有数据、仿真平台或真实机器人 | 路线 C：自有平台接入 | 离线协议样本、transform 单测和 server replay | 单位/坐标系/动作边界错误可能损坏设备或伤人 |

如果目标仍不清楚，优先选路线 A。它能先建立可信的 baseline 和评测方法，之后无论换模型还是换机器人，都有对照组。

## 源码地图

| 文件 | 下一阶段阅读重点 |
| --- | --- |
| [`src/openpi/training/config.py`](../../src/openpi/training/config.py) | `pi05_libero`、数据 factory、训练参数、EMA、FSDP 与权重来源 |
| [`src/openpi/policies/libero_policy.py`](../../src/openpi/policies/libero_policy.py) | LIBERO 输入/输出字段、padding 与动作维度裁剪 |
| [`src/openpi/transforms.py`](../../src/openpi/transforms.py) | repack、delta/absolute、normalize、tokenize 与 padding |
| [`scripts/train.py`](../../scripts/train.py) | 数据首 batch、JIT、训练循环、最终 batch 获取与异步保存时序 |
| [`examples/libero/main.py`](../../examples/libero/main.py) | suite/trial、`replan_steps`、action chunk、episode 和视频输出 |
| [`examples/aloha_real/README.md`](../../examples/aloha_real/README.md) | ALOHA 数据转换、已有 normalization assets 与远程推理入口 |
| [`examples/droid/README_train.md`](../../examples/droid/README_train.md) | 大规模 RLDS 与小规模自定义 DROID/LeRobot 两种训练路线 |
| [`examples/ur5/README.md`](../../examples/ur5/README.md) | 只作为 inputs/outputs、repack 和 delta action 的概念草图；其中部分 API 调用已落后于当前源码，不能逐字复制 |

## 动手实践

先为任何路线复制下面这张实验卡，不要直接从“启动长训练”开始：

| 项目 | 必须记录的内容 |
| --- | --- |
| 实验问题 | 这次只想验证什么；什么结果会否定当前假设 |
| 保持不变 | 数据/模型/资产/评测契约中被冻结的字段 |
| 主动改变 | 数据量、任务、模型配置、训练预算或机器人适配中的具体一项 |
| 资源上限 | GPU 型号/数量、显存、时间、磁盘、网络与人工监督预算 |
| 中止条件 | NaN/Inf、OOM、协议断言失败、设备越界、日志缺失等 |
| 成功证据 | 文件、命令输出、指标表、日志、视频或回放的精确位置 |
| 回滚点 | 原配置、原 stats、原 checkpoint、原服务端口和原控制程序 |

### 路线 A：扩大 episode、任务、训练步数并建立正式评测

#### 保持不变的协议

- 先冻结一套已通过第 07–10 章验收的字段、transform、config/checkpoint 配对和评测客户端；
- 同一轮对比使用相同任务 suite、trial 数、成功定义、`replan_steps`、步数上限和视频命名规则；
- 训练集、验证集和评测任务按 episode/task 划分，避免同一轨迹片段泄漏到不同集合；
- 每个结果保留 config diff、数据清单、stats 路径、checkpoint step、服务日志和逐 trial 结果。

#### 允许改变的组件

- episode 数、任务覆盖、示范来源、数据过滤和采样权重；
- normalization statistics 的抽样规模与是否按新数据重算；
- batch、训练步数、学习率和保存间隔，但每次改变都使用新的实验名；
- 评测 trial 数和 suite 范围，但只有冻结成新协议后才能与同协议结果横向比较。

#### 前提与分阶段里程碑

| 阶段 | 做什么 | 可观察成功证据 |
| --- | --- | --- |
| A0 基线冻结 | 用现有 checkpoint 重跑固定小评测 | 每个 trial 有任务名、seed/初始条件、成功布尔值、日志与独立视频目录 |
| A1 数据扩展 | 逐批加入 episode/task，重复离线字段、有限性、动作与泄漏检查 | 新旧数据清单可 diff；抽样样本仍满足相同协议 |
| A2 统计与短训练 | 对新数据明确选择重算或复用 stats；先做少量 step smoke test | 首 batch、loss/grad/param norm 有限；新 checkpoint 组件完整 |
| A3 受控长训练 | 使用新实验名扩大 step，持续记录资源与异常 | 进程正常等待异步保存；各候选 step 可加载 |
| A4 正式评测 | 用冻结评测协议评估多个 checkpoint，并报告分任务与总体结果 | 原始 trial 表、成功率、失败类型和置信区间/样本数同时可查 |

正式评测不要只报平均成功率。至少同时报告任务数、每任务 trial 数、成功/失败计数、版本/commit、checkpoint、评测参数和失败分类。数据很少时应直接给出计数，不用过度解释小样本百分比。

#### 回滚与安全

旧数据、stats、checkpoint 和 baseline 评测目录保持只读或显式归档；新实验使用新 config/exp name，不用 `--overwrite` 复用旧目录。若扩展数据后出现字段漂移或 NaN，回退到最后一批已验收数据并二分定位新增批次，而不是清空所有缓存或删除整个数据集。

### 路线 B：在 A100/H100 上完整微调 `pi05_libero`

当前 [`pi05_libero`](../../src/openpi/training/config.py) 的关键事实是：

| 字段 | 当前值 | 含义 |
| --- | --- | --- |
| 模型 | `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)` | 使用 π₀.₅；动作块长度为 10；LIBERO 状态不走默认离散 state 输入 |
| 数据 | `physical-intelligence/libero` | `prompt_from_task=True`，`extra_delta_transform=False` |
| batch | `256` | 全局 batch；必须与实际设备拓扑和 sharding 共同设计 |
| EMA | `0.999` | checkpoint 推理参数会使用 EMA 路径，语义不同于教学 LoRA 的 `ema_decay=None` |
| 初始参数 | `gs://openpi-assets/checkpoints/pi05_base/params` | 必须与 π₀.₅ 参数树匹配 |
| 训练范围 | 没有 LoRA variant，也没有 LoRA freeze filter | 这是全参数训练，不是第 08–10 章的低显存 LoRA |
| 默认步数 | `30_000` | 只是仓库配置，不是“必然复现官方指标”的承诺 |

这一路线计算量和显存需求都高。不能承诺一张 A100/H100 足以容纳当前 batch/优化器状态，也不能承诺 30k step 在你的数据、硬件和软件版本上复现官方 LIBERO 结果。增加 GPU 也不会自动得到正确 FSDP；多 GPU mesh、`fsdp_devices`、batch 可整除性、状态切分、通信成本和 checkpoint 恢复必须作为独立设计验证。

#### 保持不变与需要改变

- **保持不变**：π₀.₅ 模型/参数树、10-step action horizon、`discrete_state_input=False`、`extra_delta_transform=False`、LIBERO 7D 动作语义、匹配的 stats/assets 和固定评测协议。
- **需要改变或确认**：本地数据可达性、stats 来源、GPU 拓扑、global batch、FSDP/数据并行策略、实验名、保存频率、监控与预算；任何 batch/sharding 改动都要形成新实验记录。
- **不能继承**：`pi0_libero_tutorial` 的 π₀ LoRA checkpoint、50-step horizon、`pi0_base` 参数或 `extra_delta_transform=True` 统计。

#### 分阶段里程碑

1. **静态审计**：只读取 config，记录上表字段、可见设备、预计 checkpoint 位置和参数来源，不创建模型。
2. **数据与 assets 审计**：离线抽样验证字段、7D action、transform 输出与 stats；目标 stats 已存在时非零退出，不静默覆盖。
3. **资源 smoke test**：用新实验名运行极少 step，证明首 batch、模型初始化、一次 JIT/更新和 checkpoint 保存链路可工作；这不是性能实验。
4. **拓扑实验**：若单卡不可行，单独设计并验证多 GPU/FSDP；用实际设备内存和 sharding 日志证明状态确实按预期放置。
5. **受控训练与候选选择**：按预算运行，保存有限指标、资源曲线和已提交 checkpoint；候选 checkpoint 必须先 reload，再进入路线 A 的正式评测。

#### 回滚与成功证据

为每次拓扑、batch 或学习率变化使用不同 config/exp name。遇到 OOM、NaN/Inf、参数树或 stats 不匹配时保留首个错误、设备信息和现场目录，回退到上一个通过的 smoke checkpoint；不要用 π₀ checkpoint“试着加载”，也不要覆盖旧实验。

成功证据不是“命令跑了 30k step”，而是：配置与参数/assets 一致；资源上限可持续；指标有限；异步保存正常结束；checkpoint 可重新加载；固定评测协议下有逐 trial 结果。是否达到研究目标由预先写下的评测阈值决定。

### 路线 C：接入自有数据、仿真平台或真实机器人

这条路线先做协议适配，再谈模型性能。实现时优先以当前源码和仍在维护的 LIBERO/ALOHA/DROID 路径为准：

1. [`LeRobotLiberoDataConfig`](../../src/openpi/training/config.py) 与 [`libero_policy.py`](../../src/openpi/policies/libero_policy.py)：核对当前 `DataConfigFactory`、transform 和在线 observation schema；
2. [`examples/libero/README.md`](../../examples/libero/README.md)：把仿真闭环作为接入和回归测试基准；
3. [`examples/aloha_real/README.md`](../../examples/aloha_real/README.md)：理解真实平台数据、已有 stats 和远程服务边界；
4. [`examples/droid/README_train.md`](../../examples/droid/README_train.md) 与 [`examples/droid/README.md`](../../examples/droid/README.md)：比较 LeRobot/RLDS 数据路线，并理解机器人控制机与 GPU Policy Server 分离；
5. [`examples/ur5/README.md`](../../examples/ur5/README.md)：只用于建立 inputs/outputs、repack 和 delta action 的概念。它相对当前 API 已过时；实现必须遵循当前 `DataConfigFactory.create(assets_dirs, model_config)`、`create_base_config(assets_dirs, model_config)` 和 `model_config` 驱动 transform/normalization 的签名，不能逐字复制其中代码。

#### 保持不变的边界

- Policy Server/WebSocket 仍只交换经过定义的 observation、prompt 和 action chunk；
- `DataConfig.repack_transforms` 只把离线 dataset 样本映射成机器人/环境 schema，属于训练数据加载路径；正常在线推理客户端必须直接发送 **post-repack** schema，不应再套用 dataset repack；
- data transforms、model transforms、normalization 和动作协议在训练/推理间按各自 input/output 方向共享；输出端执行所需的 unnormalize 与 delta/absolute 逆变换；
- 不要对在线 observation 做“双重 repack”。若推理端字段仍是 dataset 原始 key，应先修客户端协议；不要把训练 repack 再塞进 live policy 来掩盖 schema 漂移；
- config、基础参数、stats/assets、action dimension/horizon 必须匹配；
- 所有真实机器人动作在进入底层控制前经过独立的单位、速率、位置/速度/力矩、工作空间和夹爪限制。

#### 需要替换的组件

- 数据转换器、相机键、state/action 拼接、控制频率、单位、坐标系和 prompt 来源；
- 平台专用 `Inputs`/`Outputs`、`DataConfigFactory`、repack/data transforms 与 action mask；
- stats 复用或重算策略、客户端 observation 采集、action 执行和错误处理；
- 仿真/回放环境、设备限位、看门狗、急停和人工监督流程。

#### 安全优先的分阶段里程碑

| 阶段 | 允许发生什么 | 必须看到的证据 |
| --- | --- | --- |
| C0 离线数据检查 | 不启动 Policy Server，不连接执行器 | episode 边界、时间戳、图像、state/action、单位、频率和有限性报告 |
| C1 transform 单元测试 | 只对固定样本运行纯函数 | 输入/输出 key、shape、dtype、padding、delta/absolute 往返与边界样例断言通过 |
| C2 server replay | 服务端接收录制 observation，但 action 不下发机器人 | 多帧请求可重复；action chunk shape/有限性/延迟有记录 |
| C3 仿真或影子模式 | action 进入仿真，或真实控制器只记录不执行 | 无越界；replan、超时、断连和陈旧 action 被正确处理 |
| C4 受限真机测试 | 低速、空载/软物体、隔离工作区、现场监护 | 物理限位、软件限位、看门狗和急停均独立测试；每次动作可中止 |
| C5 扩大任务范围 | 只在前一阶段重复通过后逐步放宽 | 正式风险评估、逐 trial 记录和明确的停止标准 |

本教程不提供绕过设备安全系统、直接下发无限制动作或无人监督部署的步骤。急停必须独立于模型和网络工作；WebSocket 断开、响应超时、action 非有限、时间戳过期或超限时，底层控制器应进入平台定义的安全状态，而不是继续执行旧 action chunk。

#### 回滚与成功证据

每一阶段都保留上一阶段可运行版本和录制样本。任何协议变化先回到 C0/C1，任何控制器变化先回到影子模式。成功证据包括 transform 测试、server replay、仿真回归、限制器/超时/急停测试记录；“机器人动了”本身不是验收。

### 推荐进阶阅读顺序

先以已提交的课程章节和当前源码为主：

1. [第 04 章：openpi 架构](04_openpi_architecture.md)与 [`src/openpi/policies/policy_config.py`](../../src/openpi/policies/policy_config.py)：连接一次请求、transform 与 Policy；
2. [第 06 章：JAX 与 π₀ 基础](06_jax_and_pi0_basics.md)与 [`scripts/train.py`](../../scripts/train.py)：准备修改训练、JIT、sharding 或性能路径；
3. [`src/openpi/models/pi0.py`](../../src/openpi/models/pi0.py)与 [`src/openpi/models/pi0_config.py`](../../src/openpi/models/pi0_config.py)：理解模型建立、参数状态、loss 与动作采样；
4. [Normalization statistics](../norm_stats.md)：决定复用还是重算机器人统计量；
5. [Remote inference](../remote_inference.md)：设计机器人控制机与 GPU 服务端的接口边界。

本工作区若已有以下**可选本地进阶资料**，可按代码路径打开；它们不是已提交教程的必需依赖：`docs/architecture_zh.md`、`docs/jax_in_openpi_zh.md`、`docs/model_build_and_inference_zh.md`。

## 预期结果

完成本章后，你应留下“一页后续实验计划”，其中至少包含：

- 选择的路线和暂不选择其他路线的理由；
- 数据、config、transforms、checkpoint/assets 和评测五项的“保持/替换”决定；
- 分阶段里程碑，每个阶段一个可观察成功信号和一个中止条件；
- 资源预算、唯一实验名、输出路径、回滚点和安全责任人；
- 不把单卡、30k step、短程 loss 或一次 rollout 写成性能保证。

本章没有运行完整 π₀.₅ 训练、多 GPU/FSDP 实验或真实机器人测试；所有重型命令和物理安全验证都必须在你的受控环境中另行执行并记录。

## 常见问题

### 1. 我应该直接把教学训练从 500 step 改成 30k 吗？

不应只改一个数字。先冻结 baseline，扩大数据与统计量检查，再用新实验名完成短 smoke run。训练预算、数据量和评测协议要一起设计；否则更长训练只会产生更贵但无法解释的结果。

### 2. `pi05_libero` 名字里没有 LoRA，能否沿用教学 LoRA checkpoint？

不能。它是 π₀.₅ 全参数配置，从 `pi05_base/params` 初始化；教学 checkpoint 是 π₀ LoRA 模型图。模型类型、参数树、horizon、delta transform 和 assets 都不同。

### 3. 多张 H100 是否意味着配置无需修改？

不意味着。global batch 必须可分，默认 sharding 不会自动满足你的显存目标。需要明确设计 mesh/FSDP、检查实际 sharding 和单卡峰值，并验证 checkpoint 恢复。

### 4. 自有机器人已经能接收 action，可以跳过回放和仿真吗？

不能。字段、单位、坐标系或超时错误在真机上可能造成危险。先离线检查、transform 单测、server replay、仿真/影子模式，再在独立限位和急停下做受限测试。

## 本章验收

- [ ] 已从三条路线中选择一条，并写出实验问题与否定条件；
- [ ] 已列出数据、模型、资产和评测四类保持不变的协议；
- [ ] 已列出需要替换的组件、前置资源和唯一实验输出路径；
- [ ] 至少有三个分阶段里程碑，每个都有成功证据、中止条件和回滚点；
- [ ] 若选择 `pi05_libero`，已准确记录 π₀.₅、horizon 10、`discrete_state_input=False`、`extra_delta_transform=False`、batch 256、EMA 0.999、`pi05_base/params` 和全参数训练；
- [ ] 没有承诺单张 A100/H100 或 30k step 能复现官方结果；多 GPU/FSDP 被列为独立设计与验证任务；
- [ ] 若涉及自有机器人，计划顺序是离线数据 → transform 测试 → server replay/仿真 → 限位、看门狗与急停下的受限测试；
- [ ] 正式评测会保存逐 trial 原始记录，而不只报告平均值或挑选成功视频。

## 思考题

### 1. 为什么扩大数据与扩大评测 trial 不应在同一个未记录的步骤里完成？

<details>
<summary>参考答案（展开查看）</summary>

两者会同时改变训练分布和测量噪声。若结果变化，就无法判断来自模型、任务覆盖还是评测抽样。应先冻结 baseline 协议，再分别记录数据版本与评测协议版本。

</details>

### 2. 为什么 `pi05_libero` 的 30k step 不是复现承诺？

<details>
<summary>参考答案（展开查看）</summary>

训练结果还取决于精确数据版本、过滤/采样、stats、软件 commit、设备拓扑、优化器执行、随机性和评测协议。配置中的步数只定义一次运行预算，不能单独保证官方成功率。

</details>

### 3. 为什么真实机器人客户端不能把上一段 action chunk 在断连后继续无限执行？

<details>
<summary>参考答案（展开查看）</summary>

旧 action 已不再对应最新 observation，环境也可能已经变化。客户端和底层控制器需要时间戳、超时、限位、看门狗和安全停止；这些机制必须独立于模型服务可用性。

</details>

### 4. 一个 checkpoint 能重新加载，为什么仍不能证明模型质量？

<details>
<summary>参考答案（展开查看）</summary>

重新加载只证明参数、配置和 assets 的工程兼容性。质量需要在预先冻结的任务、trial、成功标准和版本条件下进行独立评测，并保留失败样本与原始计数。

</details>

[← 上一章：服务与闭环评测](10_serve_and_evaluate.md) · [课程入口：README](README.md) · [下一页：课程术语表 →](glossary.md)
