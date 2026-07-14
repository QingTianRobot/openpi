# openpi 零基础章节式教程设计

## 1. 背景与目标

本项目已有完整的 openpi 训练、推理、服务与 LIBERO 仿真代码，也已有三篇中文源码深读文档：

- `docs/architecture_zh.md`
- `docs/jax_in_openpi_zh.md`
- `docs/model_build_and_inference_zh.md`

这些材料适合进阶查阅，但缺少面向初学者的学习顺序、连续实践主线和逐章验收。本次工作将在 `docs/tutorial/` 中新增一套可独立学习的章节式教程，把现有仓库组织成教学示范。

目标学习者已经掌握 Python 基础语法，但没有机器人、JAX 或视觉-语言-动作模型经验。教程默认使用 Ubuntu、NVIDIA A100/H100 和 LIBERO 仿真环境，不依赖真实机器人。

课程按 16–24 小时的独立学习内容设计，模型与数据下载、首次 JIT 编译和短程训练时间不计入阅读与练习时长。各章提供独立检查点，学习者可以跨多天完成。

完成教程后，学习者应能：

1. 解释机器人 observation、action、action chunk、模仿学习和 VLA 的基本含义；
2. 说明 openpi 从 LIBERO observation 到模型动作输出的完整数据流；
3. 运行官方预训练策略并在 LIBERO 中观察推理结果；
4. 将一小份 LIBERO RLDS 数据转换成 LeRobot 格式；
5. 检查字段、形状、动作语义和归一化统计；
6. 使用仓库已有的 LoRA 路径完成一次短程微调；
7. 保存、加载和服务自己的 checkpoint，并完成一次仿真闭环评测；
8. 根据错误出现的层级选择合适的排查方法。

## 2. 教学方法

采用螺旋式任务主线，而不是先集中讲完理论或直接堆叠命令。学习顺序为：

```text
运行预训练模型
    ↓
认识 observation 与 action
    ↓
理解 openpi 数据流和核心抽象
    ↓
转换小规模 LIBERO 数据
    ↓
配置数据变换与归一化
    ↓
执行短程 LoRA 微调
    ↓
加载 checkpoint 并启动 Policy Server
    ↓
在 LIBERO 中完成闭环评测
```

每次引入概念都要立即对应一个可观察的仓库对象、命令输出或小实验。现有三篇中文深度文档作为进阶阅读链接，不在入门正文中重复大段源码细节。

## 3. 课程结构

教程文件放在 `docs/tutorial/`：

| 文件 | 职责 |
| --- | --- |
| `README.md` | 课程入口、导航、推荐顺序和最终成果 |
| `00_course_outline.md` | 教学大纲、先修条件、时间与资源预算、总体验收标准 |
| `01_vla_and_robotics_basics.md` | 机器人学习、模仿学习、VLA、状态、动作与 action chunk |
| `02_environment_setup.md` | Ubuntu、GPU、uv、CUDA、子模块、依赖安装与环境自检 |
| `03_first_libero_inference.md` | 使用官方 π₀.₅ checkpoint 完成第一次 LIBERO 推理 |
| `04_openpi_architecture.md` | 沿一次请求认识客户端、服务、Policy、Transform 和模型 |
| `05_data_and_transforms.md` | 字段协议、图像、状态、动作、prompt、padding 和 normalization |
| `06_jax_and_pi0_basics.md` | 理解训练所需的 JAX、PyTree、JIT、Flax NNX 和 flow matching |
| `07_prepare_small_dataset.md` | 选择少量 LIBERO episode 并转换为 LeRobot 数据集 |
| `08_configure_finetuning.md` | 配置数据源、变换、归一化统计、LoRA 和训练参数 |
| `09_run_finetuning.md` | 运行短程训练、阅读日志、检查 loss 并管理 checkpoint |
| `10_serve_and_evaluate.md` | 服务教学 checkpoint，并在 LIBERO 中完成闭环评测 |
| `11_next_steps.md` | 完整训练、自有数据和真实机器人接入的扩展路线 |
| `glossary.md` | 中英文术语、缩写和源码名词速查 |
| `troubleshooting.md` | 按安装、数据、JAX、显存、checkpoint、服务和仿真分类排错 |

课程正文不要求修改模型核心实现。需要学习者修改数据转换常量或训练配置时，章节提供完整代码片段、准确位置、修改原因和恢复方式，使改动本身成为数据适配练习。

## 4. 逐章教学模板

每一章使用相同结构：

1. **本章目标**：列出完成后能够执行或解释的事项；
2. **前置条件**：指出必须先完成的章节、文件或环境状态；
3. **核心概念**：只讲完成本章实践所需的最少理论；
4. **源码地图**：链接到仓库中的准确文件并说明阅读重点；
5. **动手实践**：给出可复制命令、代码和操作顺序；
6. **预期结果**：展示关键输出特征及其含义，不依赖易变化的完整日志；
7. **常见问题**：优先给出诊断命令，再解释可能原因和处理办法；
8. **本章验收**：提供可客观判断的检查清单；
9. **思考题**：使用少量问题帮助学习者连接相邻章节。

教程命令以仓库根目录为默认工作目录。必须在其他目录执行时，正文会明确说明。下载量大、耗时长、需要多终端或会覆盖本地数据的命令必须在执行前给出提示。

## 5. 模型与实践路线

第一次仿真推理使用仓库默认的 `pi05_libero` 和官方 checkpoint，使学习者尽早看到完整闭环并建立直观认识。

教学微调主线使用现有 `pi0_libero_low_mem_finetune` 配置。选择该路径的原因是：

- 它是仓库内已有并带注释的 LIBERO LoRA 示例；
- 冻结参数和 LoRA 的边界清晰，适合第一次训练；
- 短程教学运行的时间和显存成本低于完整微调；
- 学习者可以把重点放在数据、变换、训练状态和 checkpoint 上。

教程会明确指出：预训练演示使用 π₀.₅，LoRA 教学微调使用 π₀，两者不是同一个 checkpoint 或模型配置。`11_next_steps.md` 再说明如何在 A100/H100 上扩展到 `pi05_libero` 完整微调，避免初学者误把两个配置混用。

短程训练用于验证工程闭环，不承诺复现官方 30k-step 结果。默认成功标准是训练与部署链路正确，而不是达到官方 LIBERO 成功率。

训练与仿真使用两个明确隔离的运行环境：openpi 模型、数据处理和训练运行在仓库的 Python 3.11 `uv` 环境；LIBERO 客户端优先运行在项目提供的 Docker 环境。两侧只通过 WebSocket 交换 observation 和 action。教程中的多终端命令会分别标注“模型/训练端”和“LIBERO 仿真端”，避免把旧版仿真依赖安装进主训练环境。

## 6. 数据流与学习产物

教程围绕一份可追踪的数据样本展开：

```text
LIBERO RLDS episode
    → LeRobot frame/episode
    → repack transforms
    → LiberoInputs
    → Normalize / resize / tokenize / pad
    → Observation + action chunk
    → π₀ loss 或动作采样
    → Unnormalize / LiberoOutputs
    → WebSocket response
    → LIBERO 环境动作
```

学习者在课程中应得到以下可检查产物：

- 一个小规模 LeRobot 数据集；
- 一份对应的 normalization statistics；
- 一次短程训练日志；
- 至少一个可加载的 checkpoint；
- 一次 Policy Server 启动记录；
- 一次 LIBERO 仿真评测结果；
- 一份对 observation/action 字段和 shape 的说明。

## 7. 错误处理设计

教程不把所有错误混成一个常见问题列表，而是按数据流定位：

| 层级 | 典型问题 | 首要证据 |
| --- | --- | --- |
| 安装 | Python/CUDA/依赖不匹配 | 版本和设备检查输出 |
| 数据 | 路径、字段、episode 或图像格式错误 | 数据集元信息与单样本打印 |
| 变换 | key、shape、dtype、动作维度不匹配 | 变换前后结构对比 |
| 归一化 | statistics 缺失或数据源不一致 | assets 路径和统计 shape |
| 训练 | OOM、NaN、JIT 编译慢 | 显存、loss、首轮编译日志 |
| checkpoint | 配置名、参数树或 step 路径错误 | checkpoint 目录结构与加载错误 |
| 服务 | 端口、地址或模型加载失败 | 服务端启动日志和端口检查 |
| 仿真 | EGL/GLX、子模块或任务集问题 | LIBERO 客户端错误和渲染后端 |

正文必须区分“首次 JIT 编译较慢”和“程序卡死”，区分“短程训练任务成功率低”和“训练链路故障”，并提醒学习者不要在不知道原因时直接删除 checkpoint 或依赖缓存。

## 8. 验证与质量标准

文档实施完成后进行以下验证：

1. 检查 `docs/tutorial/README.md` 能到达全部章节，全部章节能返回课程入口；
2. 检查所有仓库相对链接的目标文件存在；
3. 检查命令中的脚本路径、配置名和主要参数与当前源码一致；
4. 检查代码块语言标签和 shell 续行格式；
5. 检查教程中不存在未完成标记或未解释的占位符；
6. 检查 π₀、π₀.₅、JAX、LoRA、checkpoint 等术语在首次出现时有解释或词汇表链接；
7. 运行不需要下载模型或启动 GPU 训练的轻量检查命令；
8. 对无法在文档构建阶段完整执行的下载、训练和仿真命令，明确前置条件、预期结果和失败诊断方法。

逐章验收至少覆盖：

- JAX 能发现预期 GPU；
- 预训练 Policy Server 与 LIBERO 客户端能建立连接；
- 转换后样本字段和 shape 符合配置；
- normalization statistics 成功生成；
- 训练 loss 为有限值，checkpoint 被写入预期目录；
- 教学 checkpoint 能被服务端重新加载；
- LIBERO 客户端能接收动作并推进 episode。

## 9. 范围边界

本次教程包含零基础概念、环境配置、LIBERO 预训练推理、数据适配、LoRA 教学微调、服务与评测。不包含以下内容：

- 复现官方完整训练指标；
- 深入推导全部 flow matching 数学；
- 重写 π₀、π₀.₅ 或 π₀-FAST 模型；
- 配置真实机器人硬件和安全控制；
- 多节点分布式训练；
- 为所有 ALOHA、DROID、UR5 示例分别编写完整课程。

这些主题只在扩展章节提供路线和现有文档链接，以保持主线聚焦且可以独立完成。
