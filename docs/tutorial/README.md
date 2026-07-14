# openpi 零基础章节式教程

> **先记住两条互不兼容的实践路线：**第 03 章的预训练演示使用 `pi05_libero`（π₀.₅）；第 08 章开始的教学 LoRA 使用后续创建的 `pi0_libero_tutorial`（π₀）。两者的模型结构、配置和 checkpoint 必须一一对应，**绝不能混用 checkpoint**。

[查看课程大纲](00_course_outline.md) · [立即开始：第 01 章](01_vla_and_robotics_basics.md)

## 这门课解决什么问题

这是一门面向 openpi 初学者的中文实践课。它不会先要求你读完模型论文或大段源码，而是沿着“运行官方策略 → 看懂数据流 → 准备小数据 → 配置 LoRA → 短程训练 → 服务自己的 checkpoint → 回到 LIBERO 闭环评测”的路线逐步推进。

完成课程后，你应能独立跑通以下端到端闭环：在 LIBERO 中产生 observation，通过 WebSocket 请求 openpi Policy Server，获得 action chunk；随后把一小份 LIBERO 数据转换为训练数据，生成归一化统计，完成一次教学 LoRA 训练，并用自己的 checkpoint 再次驱动 LIBERO 环境。短程训练用于验证工程链路，不承诺复现官方 benchmark 成功率。

## 适合谁

本课程假设你：

- 已掌握 Python 基础语法，能运行脚本、阅读字典和数组 shape；
- **没有**机器人学、JAX 或视觉-语言-动作模型（VLA）经验；
- 愿意通过命令输出、数据样本和小实验建立概念，而不是只读理论；
- 不使用真实机器人，所有闭环实践都在 LIBERO 仿真中完成。

## 资源与时间预算

| 项目 | 课程假设 |
| --- | --- |
| 操作系统 | Ubuntu 22.04 |
| 模型/训练主环境 | 仓库的 Python 3.11 `uv` 环境 |
| 仿真环境 | Docker 中的 LIBERO 客户端与 MuJoCo |
| GPU | 单卡 NVIDIA A100 或 H100 |
| 磁盘 | 至少 100 GB 可用空间，用于模型、数据、镜像、缓存和 checkpoint |
| 有效学习时间 | 16–24 小时，可跨多天完成 |

16–24 小时只统计阅读、观察输出、编辑配置和执行练习的时间。模型/数据下载、Docker 镜像构建、第一次 JIT 编译以及训练本身的等待时间均不计入学习时长。

## 一次请求的数据流

课程会反复使用下面这条概念数据流：

```text
LIBERO observation → WebSocket client → Policy Server → LiberoInputs → normalize / resize / tokenize / pad → π₀ or π₀.₅ → unnormalize / LiberoOutputs → action chunk → LIBERO environment
```

对初学者来说，可以把它理解为：仿真环境先提供相机图像、机器人状态和任务文字；客户端把这些 observation 发给远端策略服务；`LiberoInputs` 把 LIBERO 字段整理成模型认识的格式，再依次做数值归一化、图像缩放、文字分词和定长补齐。π₀ 或 π₀.₅ 一次预测一段连续动作，后处理再把动作还原到 LIBERO 的数值尺度和字段格式。客户端按顺序执行这段 action chunk，环境由此向前推进，并产生下一轮 observation。

## 课程导航（15 个 Markdown 文件）

| 文件 | 主题 | 建议用法 |
| --- | --- | --- |
| [README.md](README.md) | 课程入口、路线、运行环境与学习产物 | 当前页面 |
| [00_course_outline.md](00_course_outline.md) | 完整大纲、课时、里程碑与验收标准 | 开始前阅读 |
| [01_vla_and_robotics_basics.md](01_vla_and_robotics_basics.md) | observation、action、模仿学习、VLA 与 action chunk | 主线 |
| [02_environment_setup.md](02_environment_setup.md) | Ubuntu、Python、GPU、uv、Docker 与环境自检 | 主线 |
| [03_first_libero_inference.md](03_first_libero_inference.md) | 使用 `pi05_libero` 完成首次预训练闭环 | 主线里程碑 |
| [04_openpi_architecture.md](04_openpi_architecture.md) | 沿一次请求认识客户端、服务、Policy、Transform 和模型 | 主线 |
| [05_data_and_transforms.md](05_data_and_transforms.md) | 字段、shape、图像、prompt、归一化与 padding | 主线 |
| [06_jax_and_pi0_basics.md](06_jax_and_pi0_basics.md) | 训练所需的 JAX、PyTree、JIT、NNX 与 flow matching 直觉 | 主线/可回读 |
| [07_prepare_small_dataset.md](07_prepare_small_dataset.md) | 将少量 LIBERO RLDS episode 转为 LeRobot 数据集 | 主线 |
| [08_configure_finetuning.md](08_configure_finetuning.md) | 创建 `pi0_libero_tutorial`、配置 LoRA 并计算统计量 | 主线里程碑 |
| [09_run_finetuning.md](09_run_finetuning.md) | 运行短程训练、阅读日志并管理 checkpoint | 主线 |
| [10_serve_and_evaluate.md](10_serve_and_evaluate.md) | 服务教学 checkpoint 并完成 LIBERO 闭环评测 | 主线里程碑 |
| [11_next_steps.md](11_next_steps.md) | 扩大训练、自有数据和真实机器人迁移路线 | 扩展阅读 |
| [glossary.md](glossary.md) | 中英文术语、缩写和源码名词速查 | 随时查阅 |
| [troubleshooting.md](troubleshooting.md) | 按安装、数据、训练、服务和仿真层级排错 | 遇到问题时查阅 |

## 两条学习路径

**主线动手路径：**先读[课程大纲](00_course_outline.md)，再按第 01 章到第 10 章顺序完成。第 03 章先得到一个可运行的预训练闭环；第 07–09 章准备小数据并训练；第 10 章用自己的教学 checkpoint 再次闭环。最后用第 11 章规划下一次实验。

**可选阅读路径：**如果你暂时只想建立源码地图，不准备训练，可集中阅读第 01、04、05、06、11 章；`glossary.md` 和 `troubleshooting.md` 按需查阅。这条路径不要求修改模型核心实现，也不替代第 02、03 章的环境与闭环验收。

## 双环境标签约定

后续命令默认从**仓库根目录**执行。需要多个终端时，每段命令都会带以下标签：

| 标签 | 在哪里运行 | 负责什么 |
| --- | --- | --- |
| **模型/训练端** | Ubuntu 22.04 宿主机的 Python 3.11 `uv` 环境 | 数据处理、归一化统计、JAX/LoRA 训练、checkpoint 加载和 Policy Server |
| **LIBERO 仿真端** | 项目提供的 Docker 环境 | LIBERO/MuJoCo 仿真、WebSocket client、episode 推进和评测记录 |

两端只通过 WebSocket 交换 observation 与 action。不要为了运行 LIBERO，把它的旧版依赖直接安装进模型/训练主环境。

## 你将留下的学习产物

完成主线后，应能保存并检查：

- 一份 observation/action 字段、shape、dtype 和动作语义说明；
- 一个小规模 LeRobot 数据集及其质量检查记录；
- 与 `pi0_libero_tutorial` 数据源对应的 normalization statistics；
- 一次短程 LoRA 训练日志和至少一个可加载 checkpoint；
- 一次 Policy Server 启动记录；
- 预训练 checkpoint 与教学 checkpoint 各一次 LIBERO 闭环评测记录；
- 一份后续扩大数据、训练或接入自有平台的实验计划。

## 每章的固定结构

第 01–11 章统一提供：上一章/课程入口/下一章导航、本章目标、前置条件、核心概念、源码地图、动手实践、预期结果、常见问题、本章验收和思考题。验收以可观察的文件、命令输出或行为为准；遇到问题时先收集证据，再按数据流定位。

准备好后，先用[课程大纲](00_course_outline.md)确认资源与时间，再进入[第 01 章：VLA 与机器人学习基础](01_vla_and_robotics_basics.md)。
