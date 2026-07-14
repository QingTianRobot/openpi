# openpi 零基础章节式教程：课程大纲

[← 返回课程入口 README.md](README.md) · [开始第 01 章 →](01_vla_and_robotics_basics.md)

## 课程目标与配置边界

本课程面向具备 Python 基础、但没有机器人学、JAX 或 VLA 经验的学习者。目标不是复现官方训练指标，而是理解 openpi 的关键抽象，并完成“预训练策略运行 → 小数据准备 → normalization statistics → 教学 LoRA → checkpoint 服务 → LIBERO 评测”的可检查工程闭环。

课程包含两条不可混用 checkpoint 的配置路线：

- 第 03 章使用 `pi05_libero` 及其官方 π₀.₅ checkpoint，只用于先跑通预训练演示；
- 第 08–10 章使用课程中新增的 `pi0_libero_tutorial` 及其 π₀ LoRA checkpoint；
- 启动服务时，配置名、模型结构、normalization assets 和 checkpoint 必须来自同一条路线。

贯穿课程的概念数据流是：

```text
LIBERO observation → WebSocket client → Policy Server → LiberoInputs → normalize / resize / tokenize / pad → π₀ or π₀.₅ → unnormalize / LiberoOutputs → action chunk → LIBERO environment
```

前半段把仿真产生的图像、状态和文字指令变成模型输入；模型一次生成一段动作；后半段把动作恢复到 LIBERO 使用的尺度和格式，再交回仿真执行。各章会从机器人概念、源码组件、数据字段和训练过程四个角度重复观察这条链路。

## 资源与时间预算

| 资源 | 最低课程假设 | 用途 |
| --- | --- | --- |
| 操作系统 | Ubuntu 22.04 | 仓库已验证的宿主系统 |
| 主环境 | Python 3.11 + `uv` | 模型、数据处理、训练和 Policy Server |
| 仿真环境 | Docker | 隔离 LIBERO/MuJoCo 及其依赖 |
| GPU | NVIDIA A100 或 H100 | π₀/π₀.₅ 推理和 LoRA 教学训练 |
| 磁盘 | 至少 100 GB 可用空间 | 数据集、模型、Docker 镜像、JAX 缓存和 checkpoint |

有效学习预算为 **16–24 小时**。下表给出的阅读与操作时间合计约 **21.5 小时**，但不包含模型或数据下载、Docker 镜像构建、第一次 JIT 编译和训练运行的等待时间。这些阶段可能明显更久，应单独预留机器时间。

## 章节约定

第 01–11 章都包含上一章/课程入口/下一章导航、本章目标、前置条件、核心概念、源码地图、动手实践、预期结果、常见问题、本章验收和思考题。所有命令默认从仓库根目录执行；需要并行终端时，命令会明确标为“模型/训练端”或“LIBERO 仿真端”。

## 第 01–11 章教学安排

| 章节 | 学习目标 | 核心概念 | 实践任务 | 预计阅读 / 操作时间 | 学习产物 | 客观验收标准 |
| --- | --- | --- | --- | --- | --- | --- |
| [01 VLA 与机器人学习基础](01_vla_and_robotics_basics.md) | 能用自己的话解释机器人策略的输入、输出和模仿学习目标 | observation、action、episode、demonstration、policy、VLA、action chunk | 读取 LIBERO 示例 observation，记录 key、shape 和 dtype，并解释模型为何一次输出多步动作 | 60 分钟 / 30 分钟 | 一份字段与 shape 说明 | 能正确指出示例中的 8 维 state、7 维单步 action，并说明 action chunk 是按时间排列的多步动作 |
| [02 环境安装与自检](02_environment_setup.md) | 建立彼此隔离且可通信的模型/训练端与 LIBERO 仿真端 | Python 3.11、uv、CUDA/JAX、Docker、子模块、缓存 | 检查系统、GPU、JAX device、Docker Compose、子模块和磁盘空间 | 60 分钟 / 60 分钟 | 环境自检记录 | 输出显示 Ubuntu 22.04、Python 3.11、JAX 至少发现一个 GPU、Docker Compose 可用，且可用磁盘不少于 100 GB |
| [03 第一次 LIBERO 推理](03_first_libero_inference.md) | 使用官方策略完成第一次可观察的仿真闭环 | `pi05_libero`、π₀.₅ checkpoint、Policy Server、WebSocket client、action queue | 分别启动服务端和 LIBERO 客户端，运行至少一个 trial 并保存关键日志或视频 | 45 分钟 / 60 分钟 | 预训练服务日志、客户端日志和一次评测记录 | 日志证明 `pi05_libero` checkpoint 成功加载、客户端建立 WebSocket 连接、收到非空 actions，且 LIBERO episode 至少推进一个控制周期 |
| [04 openpi 架构](04_openpi_architecture.md) | 沿一次请求定位客户端、服务、Policy、Transform 和模型的职责 | 序列化、Policy Server、Policy、`LiberoInputs`、模型采样、`LiberoOutputs` | 从 LIBERO 请求入口追到动作响应，画出组件图并标注关键源码入口 | 75 分钟 / 45 分钟 | 一张带源码位置的数据流图 | 能分别指出负责网络传输、输入适配、模型推理和输出适配的组件，并按正确顺序复述完整数据流 |
| [05 数据与变换](05_data_and_transforms.md) | 理解原始字段如何变为模型输入，以及动作如何恢复到环境尺度 | image/state/prompt/action、repack、normalize、resize、tokenize、pad、action mask | 对同一样本记录变换前后 key、shape、dtype 和数值范围，核对 normalization statistics | 90 分钟 / 60 分钟 | 变换前后对照表 | 对照表中不存在缺失必需字段；图像、state、prompt 和 actions 的 shape/dtype 与配置一致；state/actions 使用匹配数据源的统计量 |
| [06 JAX 与 π₀ 基础](06_jax_and_pi0_basics.md) | 建立足以阅读训练与推理路径的 JAX、NNX 和 flow matching 直觉 | array、PyTree、PRNG key、JIT、Flax NNX、loss、采样、flow matching | 运行小型 JAX/PyTree/JIT 实验，并把概念对应到 π₀ 的训练与采样入口 | 90 分钟 / 45 分钟 | 一份 JAX 到 openpi 的概念映射 | JAX 实验在 GPU 上运行；能区分首次 JIT 编译与程序卡死，并能指出训练 loss 路径和动作采样路径用途不同 |
| [07 准备小规模数据](07_prepare_small_dataset.md) | 把少量 LIBERO RLDS episode 转为可检查的 LeRobot 数据集 | RLDS、LeRobot、episode/frame、feature schema、动作语义、数据质量 | 选择少量 episode，执行转换并检查 metadata、图像方向、NaN、动作范围和 prompt | 45 分钟 / 90 分钟 | 一个小规模 LeRobot 数据集和质量检查表 | metadata 显示非零 episode/frame；两路图像、8 维 state、7 维 actions 和非空 task 均存在，抽样数据无 NaN |
| [08 配置教学微调](08_configure_finetuning.md) | 让小数据、输入输出变换、LoRA 模型和归一化统计通过一个配置对齐 | `TrainConfig`、`pi0_libero_tutorial`、LoRA、freeze filter、data config、normalization assets | 新增并打印教学配置，使用小数据计算 state/action normalization statistics | 75 分钟 / 60 分钟 | 教学配置改动和对应统计量目录 | `_config.get_config("pi0_libero_tutorial")` 可加载且指向教学数据；LoRA variants 前后一致；生成的统计中同时包含 state 与 actions |
| [09 运行短程 LoRA](09_run_finetuning.md) | 启动、观察并安全恢复一次短程教学训练 | step、batch、loss、JIT、EMA、checkpoint、resume、overwrite | 使用 `pi0_libero_tutorial` 运行短程训练，记录首个有限 loss、末尾 step 和 checkpoint 结构 | 45 分钟 / 60 分钟 | 训练日志和至少一个教学 checkpoint | 日志中至少一个 loss 为有限值；训练写出预期末尾 step 的 checkpoint；checkpoint 目录含可供加载的参数与 assets；训练等待时间不计入操作时长 |
| [10 服务与闭环评测](10_serve_and_evaluate.md) | 用自己的教学 checkpoint 驱动 LIBERO，并验证训练到部署的接口一致 | checkpoint loader、Policy Server、config/assets 配对、replan、evaluation | 用 `pi0_libero_tutorial` 加载教学 checkpoint，在另一终端运行 LIBERO trial 并记录结果 | 45 分钟 / 75 分钟 | 服务启动记录、评测日志和可选回放视频 | 服务端无参数树或统计量不匹配错误；客户端收到非空 action chunk；LIBERO episode 至少推进一个控制周期。任务失败不等于工程闭环失败 |
| [11 下一步](11_next_steps.md) | 为更大数据、完整训练、自有数据或真实机器人制定可验证的迁移路线 | 数据扩展、完整微调、协议边界、机器人适配、安全与评测 | 选择一条扩展路线，列出保持不变的协议、需要替换的组件、资源风险和分阶段验收 | 45 分钟 / 30 分钟 | 一页后续实验计划 | 计划至少列出数据、config、transforms、checkpoint 和评测五项的“保留/替换”决定，并为每个阶段给出一个可观察验收信号 |

## 三个里程碑

| 里程碑 | 完成位置 | 必须留下的证据 |
| --- | --- | --- |
| M1：预训练闭环 | 第 03 章结束 | `pi05_libero` 正确加载；WebSocket 客户端收到动作；LIBERO episode 推进；保存日志或视频 |
| M2：小数据 + 教学配置 | 第 08 章结束 | 小规模 LeRobot 数据通过字段与质量检查；`pi0_libero_tutorial` 可加载；对应 state/action normalization statistics 已生成 |
| M3：教学 checkpoint 闭环 | 第 10 章结束 | 使用 `pi0_libero_tutorial` 加载教学 LoRA checkpoint；客户端收到 action chunk；LIBERO episode 推进并保存评测记录 |

M1 和 M3 使用不同模型路线。验收时必须从日志同时核对配置名和 checkpoint 路径，不能把 `pi05_libero` 的 checkpoint 交给 `pi0_libero_tutorial`，也不能反向混用。

## 课程总体验收

完成主线后，用以下结果判断是否真正跑通，而不是仅仅“命令没有报错”：

1. 能从 LIBERO observation 开始，按顺序解释 WebSocket、Policy Server、`LiberoInputs`、模型和 `LiberoOutputs` 的职责；
2. 能展示一份小规模 LeRobot 数据集，说明 image、state、prompt 和 actions 的 shape 与语义；
3. 能证明 normalization statistics 与 `pi0_libero_tutorial` 使用的是同一数据源；
4. 能展示有限 loss、可加载 checkpoint 和对应配置名；
5. 能分别用官方预训练 checkpoint 和教学 checkpoint 让 LIBERO 客户端收到动作并推进 episode；
6. 能根据错误发生在安装、数据、变换、归一化、训练、checkpoint、服务或仿真中的哪一层，先选择一条收集证据的命令。

返回[课程入口 README.md](README.md)，或进入[第 01 章](01_vla_and_robotics_basics.md)开始学习。
