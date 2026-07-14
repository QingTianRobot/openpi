# openpi 课程术语表

[课程入口：README](README.md) · [第 11 章：规划下一阶段实验](11_next_steps.md) · [故障排查手册](troubleshooting.md)

本页给出英文、中文和 openpi 项目语境中的最短解释。机器人领域的同一个词在不同论文或平台中可能有更窄或更宽的定义；实际实现以当前 config、数据字段、transform 和客户端协议为准。

## 英文字母索引

[A](#index-a) · [C](#index-c) · [D](#index-d) · [E](#index-e) · [F](#index-f) · [G](#index-g) · [J](#index-j) · [L](#index-l) · [N](#index-n) · [O](#index-o) · [P](#index-p) · [R](#index-r) · [S](#index-s) · [T](#index-t) · [V](#index-v) · [W](#index-w) · [X](#index-x)

<a id="index-a"></a>

**A**：[Action](#term-action) · [Action chunk](#term-action-chunk) · [Action horizon](#term-action-horizon) · [Assets](#term-assets)

<a id="index-c"></a>

**C**：[Checkpoint](#term-checkpoint)

<a id="index-d"></a>

**D**：[Data transform](#term-data-transform) · [Delta/absolute action](#term-delta-absolute-action) · [Demonstration](#term-demonstration)

<a id="index-e"></a>

**E**：[EMA](#term-ema) · [Episode](#term-episode)

<a id="index-f"></a>

**F**：[Flax NNX](#term-flax-nnx) · [Flow matching](#term-flow-matching) · [FSDP](#term-fsdp)

<a id="index-g"></a>

**G**：[GraphDef](#term-graphdef)

<a id="index-j"></a>

**J**：[JAX](#term-jax) · [JIT](#term-jit)

<a id="index-l"></a>

**L**：[LeRobot](#term-lerobot) · [LIBERO](#term-libero) · [LoRA](#term-lora)

<a id="index-n"></a>

**N**：[Normalization](#term-normalization) · [Normalization statistics](#term-normalization-statistics)

<a id="index-o"></a>

**O**：[Observation](#term-observation)

<a id="index-p"></a>

**P**：[Policy](#term-policy) · [Policy Server](#term-policy-server) · [PRNG key](#term-prng-key) · [Prompt](#term-prompt) · [PyTree](#term-pytree)

<a id="index-r"></a>

**R**：[replan_steps](#term-replan-steps) · [Repack transform](#term-repack-transform) · [RLDS](#term-rlds)

<a id="index-s"></a>

**S**：[State（机器人）](#term-robot-state) · [State（NNX）](#term-nnx-state)

<a id="index-t"></a>

**T**：[Token](#term-token) · [Trajectory](#term-trajectory) · [Transform](#term-transform)

<a id="index-v"></a>

**V**：[VLA](#term-vla)

<a id="index-w"></a>

**W**：[WebSocket](#term-websocket)

<a id="index-x"></a>

**X**：[XLA](#term-xla)

## 机器人、数据与控制

参见[第 01 章](01_vla_and_robotics_basics.md)、[第 05 章](05_data_and_transforms.md)和[第 07 章](07_prepare_small_dataset.md)。

<a id="term-observation"></a>

### Observation — 观测

策略在一个时刻收到的信息集合；在 LIBERO 中通常包含多路图像、`observation/state` 和任务文字，而不只是相机画面。

<a id="term-robot-state"></a>

### State — 机器人状态

描述机器人当前构型或传感状态的数值向量，例如关节、末端或夹爪信息；它是 observation 的一部分。它与下文 Flax NNX 的 [`State`](#term-nnx-state) 不是同一概念。

<a id="term-action"></a>

### Action — 动作

策略交给环境或控制器的命令；LIBERO 单步动作是 7 维，具体坐标系、单位和夹爪约定必须由平台协议定义。

<a id="term-action-chunk"></a>

### Action chunk — 动作块

模型一次输出的连续多步动作数组，shape 通常是 `(action_horizon, action_dim)`；客户端可以执行其中若干步后重新请求策略。

<a id="term-action-horizon"></a>

### Action horizon — 动作预测时域

一个 action chunk 包含的预测步数；例如当前 `pi05_libero` 是 10，而教学 π₀ 配置是 50，不能只换 checkpoint 而忽略该差异。

<a id="term-replan-steps"></a>

### `replan_steps` — 重规划步数

客户端从一个 action chunk 中实际执行多少步后再次请求策略；它必须不大于服务端返回的有效 horizon，并影响控制延迟与反馈频率。

<a id="term-delta-absolute-action"></a>

### Delta action / Absolute action — 增量动作 / 绝对动作

绝对动作给出目标值，增量动作给出相对当前 state 的变化量；训练输入变换和推理输出逆变换必须使用相同维度 mask、单位和约定。

<a id="term-episode"></a>

### Episode — 回合

环境从 reset 到成功、失败或达到上限的一次完整尝试；数据划分和正式评测通常应以 episode 为单位，避免轨迹泄漏。

<a id="term-trajectory"></a>

### Trajectory — 轨迹

按时间排列的 observation/action 序列；可以覆盖整个 episode，也可以只是其中一个片段。

<a id="term-demonstration"></a>

### Demonstration — 示范

被当作学习范例的 trajectory，通常来自人类遥操作、专家策略或筛选后的成功数据。

<a id="term-policy"></a>

### Policy — 策略

把当前 observation 和 prompt 映射为 action 或 action chunk 的决策模型；在 openpi 中还包含前后处理 transforms 和 normalization。

<a id="term-vla"></a>

### VLA (Vision-Language-Action) — 视觉-语言-动作模型

联合使用视觉观测、语言任务和动作输出的策略模型；π₀/π₀.₅ 是本课程中的 VLA 实例。

<a id="term-libero"></a>

### LIBERO — 机器人操作仿真基准

本课程使用的 MuJoCo 任务集合和评测环境；它提供 observation、接收 7D action，并按 episode 记录成功与视频。

## 语言与模型目标

参见[第 01 章](01_vla_and_robotics_basics.md)和[第 06 章](06_jax_and_pi0_basics.md)。

<a id="term-prompt"></a>

### Prompt — 提示词 / 任务指令

告诉策略要完成什么的文字，例如 “pick up the mug”；在 LeRobot 路线中通常从 task 字段读取并在模型变换中 tokenize。

<a id="term-token"></a>

### Token — 词元

分词器把文字或离散信息编码后的整数单元；序列会被截断或 padding 到模型配置的最大 token 长度。

<a id="term-flow-matching"></a>

### Flow matching — 流匹配

π₀ 系列用于学习从噪声动作到数据动作连续向量场的训练目标；推理时模型通过数值积分逐步生成 action chunk。

## 数据格式、变换与归一化

参见[第 05 章](05_data_and_transforms.md)、[第 07 章](07_prepare_small_dataset.md)、[第 08 章](08_configure_finetuning.md)和[Normalization statistics 深读](../norm_stats.md)。

<a id="term-rlds"></a>

### RLDS (Reinforcement Learning Datasets) — 强化学习数据集格式

以 episode/step 组织机器人序列数据的生态格式；仓库的大规模 DROID 路线直接使用 RLDS，而课程把小规模 LIBERO 数据转换为 LeRobot。

<a id="term-lerobot"></a>

### LeRobot — 机器人数据集与工具生态

openpi 小规模自定义数据路线使用的数据格式/加载接口；repo id、cache root、metadata、episode 和视频共同构成可读取数据集。

<a id="term-transform"></a>

### Transform — 变换

对字典样本执行字段或数值转换的可组合函数；训练与推理复用相应输入变换，输出侧执行必要逆变换。

<a id="term-repack-transform"></a>

### Repack transform — 字段重组变换

把数据集原始嵌套 key 映射为策略适配器期望的 key；它改变字段组织，不应暗中改变数值语义。

<a id="term-data-transform"></a>

### Data transform — 数据变换

把平台字段转成通用模型字段，并处理图像、state/action、delta/absolute 等平台语义；它位于 repack 之后、model transform 之前。

<a id="term-model-transform"></a>

### Model transform — 模型变换

把通用字段变成特定模型输入，例如 resize、tokenize、padding；它由模型配置决定，不能随意与另一模型类型混用。

<a id="term-normalization"></a>

### Normalization — 归一化

用训练数据统计量把 state/action 缩放到模型更适合学习的尺度；推理输出必须用匹配的统计量 unnormalize。

<a id="term-normalization-statistics"></a>

### Normalization statistics — 归一化统计量

训练数据各维的 mean、std、quantile 等统计；它们按 asset id 保存，并与机器人、动作协议、数据版本和 checkpoint 配套。

<a id="term-assets"></a>

### Assets — 配置资产

checkpoint 旁的非参数推理依赖，本课程主要指 normalization statistics；`params` 可读并不代表缺少或错配 assets 时仍能正确推理。

## 训练、参数与分布式执行

参见[第 08 章](08_configure_finetuning.md)、[第 09 章](09_run_finetuning.md)和[第 11 章](11_next_steps.md)。

<a id="term-checkpoint"></a>

### Checkpoint — 检查点

训练过程保存的可恢复快照；本课程依赖 `train_state`、`params`、`assets` 三类组件，服务主要加载匹配 config 的 `params` 与 `assets`。

<a id="term-lora"></a>

### LoRA (Low-Rank Adaptation) — 低秩适配

在部分线性层加入小型低秩可训练参数的微调方法；它降低一部分训练状态成本，但不等于模型只占很少显存，也不等于所有非 adapter 参数都必然冻结。

<a id="term-ema"></a>

### EMA (Exponential Moving Average) — 指数移动平均

对训练参数维护平滑副本的方法；`pi05_libero` 使用 `ema_decay=0.999`，而教学 LoRA 关闭 EMA，checkpoint 的推理参数语义因此不同。

<a id="term-fsdp"></a>

### FSDP (Fully Sharded Data Parallel) — 完全分片数据并行

把模型参数、梯度或优化器状态跨设备切分以降低单卡占用的策略；多 GPU 不会自动变成正确 FSDP，必须设计设备数、mesh、batch 和恢复路径。

## JAX 与 Flax NNX

参见[第 06 章](06_jax_and_pi0_basics.md)、[JAX 深读](../jax_in_openpi_zh.md)和[模型建立与推理深读](../model_build_and_inference_zh.md)。

<a id="term-jax"></a>

### JAX — 数值计算与变换框架

openpi 的训练和主要模型执行框架，提供 NumPy 风格数组、自动微分、JIT、设备放置和 sharding。

<a id="term-pytree"></a>

### PyTree — 树状容器

由 list/tuple/dict/dataclass 等节点和数组叶子组成的嵌套结构；JAX 用它统一表示 batch、参数、梯度和优化器状态。

<a id="term-prng-key"></a>

### PRNG key — 伪随机数键

JAX 显式传递和 split 的随机状态；模型初始化、噪声和采样依赖 key，而不是隐式全局随机生成器。

<a id="term-jit"></a>

### JIT (Just-In-Time compilation) — 即时编译

把 JAX 函数按输入 shape/dtype 编译成设备程序；第一次调用可能较慢，输入签名变化也可能触发重新编译。

<a id="term-xla"></a>

### XLA (Accelerated Linear Algebra) — 加速线性代数编译器

JAX 后端用来优化并生成 CPU/GPU/TPU 程序的编译器；许多编译、显存和设备错误会出现在 XLA 层。

<a id="term-flax-nnx"></a>

### Flax NNX — Flax 的对象化神经网络 API

openpi 用它定义模块，并把模型图结构与可变数组状态拆开处理。

<a id="term-graphdef"></a>

### GraphDef — NNX 图定义

描述模块结构和静态关系、但不携带全部可变参数数组的对象；可与 NNX `State` 合并重建模型。

<a id="term-nnx-state"></a>

### State — NNX 模型状态

Flax NNX 中保存参数、batch statistics 等数组叶子的树状状态。它是程序对象，**不是**机器人 observation 中描述关节/夹爪的 [`state`](#term-robot-state)。

## 服务与通信

参见[第 04 章](04_openpi_architecture.md)、[第 10 章](10_serve_and_evaluate.md)和[远程推理文档](../remote_inference.md)。

<a id="term-policy-server"></a>

### Policy Server — 策略服务器

加载 config、checkpoint 参数和 assets，接收 observation 请求并返回 action chunk 的服务进程；模型端和机器人/仿真端可以运行在不同环境。

<a id="term-websocket"></a>

### WebSocket — 全双工长连接协议

openpi 轻量客户端与 Policy Server 交换 observation/action 的通信通道；连接成功只证明网络可达，不证明字段、shape、动作或任务质量正确。

## 进一步查阅

- [项目架构深读](../architecture_zh.md)
- [JAX 在 openpi 中的使用](../jax_in_openpi_zh.md)
- [模型建立与推理深读](../model_build_and_inference_zh.md)
- [Normalization statistics](../norm_stats.md)
- [Remote inference](../remote_inference.md)

[课程入口：README](README.md) · [第 11 章：规划下一阶段实验](11_next_steps.md) · [故障排查手册](troubleshooting.md)
