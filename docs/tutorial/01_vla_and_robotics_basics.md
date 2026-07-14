# 第 01 章：机器人学习与 VLA 基础

[← 上一章：课程大纲](00_course_outline.md) · [课程入口：README](README.md) · [下一章：环境安装与自检 →](02_environment_setup.md)

## 本章目标

完成本章后，你应该能够：

- 用一个 LIBERO 单臂操作任务解释 observation、state、action、episode、trajectory、demonstration 和 policy；
- 读出 openpi 的 LIBERO 示例字段及 shape：两张 `224×224×3` 图像、8 维 state、字符串 prompt；
- 说明 LIBERO 环境为什么接收 7 维单步 action，以及这个维度在源码中的依据；
- 用直觉语言区分模仿学习、VLA 和普通的“看一眼、输出一步”控制；
- 解释 action chunk 的作用，以及客户端为什么只执行其中的 `replan_steps` 步就重新规划；
- 概念性地区分 π₀/π₀.₅ 的连续动作 flow matching 与 π₀-FAST 的动作 token 自回归生成。

本章只建立共同语言和数据直觉，不推导公式，也不要求启动模型服务或 LIBERO 仿真。若尚未配置仓库的 Python 依赖，可以先阅读并完成纸面练习，再在[第 02 章](02_environment_setup.md)完成环境安装后回来运行命令。

## 前置条件

你只需要：

- 会阅读 Python 字典、字符串和 NumPy 数组；
- 知道 `shape=(8,)` 表示一条包含 8 个数的一维向量；
- 能在终端切换到 openpi 仓库根目录。

本章命令运行在 **Ubuntu 宿主机的 Python 3.11 `uv` 环境**，不需要启动 Compose 中的 `openpi_server` 或 LIBERO `runtime`。这里的“模型/训练端”和“LIBERO 仿真端”是数据流中的逻辑角色；后续闭环评测时才会把它们放进对应容器。

相关英文缩写和中英文对照可随时查看[课程术语表](glossary.md)。

## 核心概念

### 从一个 LIBERO 单臂任务开始

设想桌上有一个碗和一个小物体，任务文字是：

> 把小物体放进碗里。

LIBERO 仿真中的单臂机器人需要看到桌面和自己的夹爪，判断手臂当前姿态，然后连续完成靠近、抓取、抬起、移动和松开。这个例子足以串起机器人学习最常用的概念。

#### Observation：此刻能拿到的全部输入

[Observation（观测）](glossary.md)是策略在某个时刻收到的信息集合，而不只是一张图片。本项目的 LIBERO 请求包含：

| 字段 | shape / 类型 | 初学者可以怎样理解 |
| --- | --- | --- |
| `observation/image` | `(224, 224, 3)` | 第三人称相机看到的桌面场景，最后一维是 RGB 三个颜色通道 |
| `observation/wrist_image` | `(224, 224, 3)` | 腕部相机看到的近距离画面 |
| `observation/state` | `(8,)` | 机器人自身的低维状态 |
| `prompt` | Python 字符串 | 自然语言任务，例如“把小物体放进碗里” |

两张图像回答“外界看起来怎样”，state 回答“机器人自己在哪里”，prompt 回答“这次要做什么”。同一张桌面图像配上不同 prompt，期望动作可能完全不同，因此文字不是装饰，而是策略输入的一部分。

#### State：观测中的机器人数字摘要

[State（状态）](glossary.md)在这里特指 `observation/state` 这条 8 维向量。LIBERO 客户端在 `examples/libero/main.py` 中把下面三部分拼接起来：

```text
末端执行器位置 3 维 + 末端姿态的轴角表示 3 维 + 夹爪关节位置 2 维 = 8 维
```

“末端执行器”可以先理解为机械臂最前端、安装夹爪的部位。图像能提供丰富的场景信息，但从像素精确恢复机器人姿态并不容易；直接提供 state，能让策略更清楚地知道手臂和夹爪当前处于什么状态。

注意：真实系统的 state 不一定是 8 维。维度由机器人硬件、控制接口和数据集约定共同决定，不能看到“机器人 state”就默认它总是 8 维。

#### Action：机器人下一步要执行的命令

[Action（动作）](glossary.md)是策略交给环境的控制命令。LIBERO 的单步 action 是 7 维：通常可直觉地看成末端平移 3 维、末端旋转 3 维和夹爪控制 1 维。

```text
单步 action: [移动 x, 移动 y, 移动 z, 旋转 1, 旋转 2, 旋转 3, 夹爪]
                    └────────────── 7 个数 ──────────────┘
```

具体数值是绝对目标、相对增量还是经过归一化的控制量，要以环境和数据变换约定为准。本章只需记住：**LIBERO 接口需要 7 维动作**。项目中的 `LiberoOutputs` 明确取模型输出的前 7 维；模型为了兼容多种机器人可能使用更大的内部 action 维度，剩余部分只是 padding，不能据此改变 LIBERO 环境接口。

#### Episode、trajectory 与 demonstration

这三个词都描述“一段时间”，但侧重点不同：

- [Episode（回合）](glossary.md)：环境从 reset 开始，到任务成功、失败或达到步数上限为止的一次完整尝试。
- [Trajectory（轨迹）](glossary.md)：按时间排列的数据序列，例如 `(observation₀, action₀, observation₁, action₁, ...)`。它可以是一整个 episode，也可以是其中一段。
- [Demonstration（示范）](glossary.md)：被当作学习范例的 trajectory，通常由人类遥操作、专家策略或经过筛选的成功数据产生。

因此，不是每个 episode 都一定是 demonstration。一次随机乱动后失败的 episode 仍然是一段轨迹，但通常不会直接当作“专家示范”教给模仿学习模型。

#### Policy：从所见和所求到所做

[Policy（策略）](glossary.md)可以先理解为一个决策函数：输入当前 observation 和任务 prompt，输出下一步 action，或者输出一段 action chunk。

```text
policy(两张图像, 8 维 state, prompt) → 一个或多个 7 维 action
```

策略不是预先写死的“先左移 2 厘米，再闭合夹爪”脚本。学习型 policy 从示范数据中归纳：看到什么场景、接到什么指令时，哪些动作更像专家会做的动作。

### 模仿学习：把专家轨迹变成训练题

[Imitation learning（模仿学习）](glossary.md)的核心任务很像“看例题学解法”：数据集中保存专家在每个时刻看到了什么、任务是什么、随后做了什么；训练时让模型根据 observation 和 prompt 预测专家 action，并不断缩小预测与示范之间的差异。

以“把小物体放进碗里”为例，一条 demonstration 可能包含：

1. 图像中物体在夹爪左前方，专家向左前方移动；
2. 物体进入夹爪范围，专家闭合夹爪；
3. 物体被抓住，专家抬高并移向碗；
4. 物体位于碗上方，专家松开夹爪。

模型不会只背诵这一条坐标序列。足够多样的数据会帮助它学习“图像和状态呈现这种关系、文字要求完成这种任务时，接下来应采取怎样的动作”。不过，模仿学习通常更擅长数据覆盖过的情况；场景偏离示范太远时，错误可能逐步积累。

### VLA：看图、读指令，再规划动作

[VLA（Vision-Language-Action）](glossary.md)把三类信息接到同一个策略中：

- Vision：两路相机图像告诉模型物体、桌面、碗和夹爪在哪里；
- Language：prompt 告诉模型目标是“放进碗里”，而不是“推到桌边”；
- Action：模型输出机器人可以执行的控制序列。

一个直觉例子是：模型看到桌面图像和腕部近景，读到“把小物体放进碗里”，结合 8 维 state 判断夹爪当前姿态，然后规划“靠近、对准、闭合、抬起……”的若干步动作。这里的“规划”不是说模型一定显式生成文字计划，而是它一次给出一小段时间上相互衔接的动作。

### Action chunk：一次预测一小段动作

如果 policy 每次只预测一个 7 维 action，控制循环就是：看一次 → 预测一步 → 执行一步 → 再看一次。这种方式反馈及时，但每个环境步都要完成一次模型推理，延迟较高；相邻动作还可能因为每次预测的小波动而不够连贯。

[Action chunk（动作块）](glossary.md)则让模型一次输出按时间排列的多步动作：

```text
action chunk shape: (action_horizon, 7)

时间       t        t+1       t+2                 t+H-1
动作    action₀ → action₁ → action₂ → ... → actionH₋₁
```

它与“把一个动作复制很多次”不同：chunk 中每一行都可以不同，共同描述一个短时动作意图。这样做有两个主要好处：

- 一次推理可支持多个控制步，减少必须调用大模型的频率；
- 模型能联合预测相邻步骤，动作通常更连贯，也更容易表达“先靠近再闭合夹爪”这样的短时结构。

代价是：预测越远，未来越可能因物体移动、接触误差或前几步执行偏差而过时。因此 openpi 的 LIBERO 客户端采用“短执行、再规划”：模型返回完整 chunk，但客户端只把前 `replan_steps` 个动作放入队列；执行完这些动作后，用新的图像和 state 再请求一个 chunk。当前示例默认 `replan_steps=5`。

这是一种滚动闭环：既利用多步预测的连贯性，又定期用最新 observation 修正计划。`replan_steps` 不是 action chunk 的总长度；它是每轮实际执行的前缀长度，并且必须小于或等于模型返回的 chunk 长度。

### π₀、π₀.₅ 与 π₀-FAST：两种动作生成思路

这三个名字都属于 openpi 的 VLA 模型路线，但动作生成方式不能混为一谈：

| 模型 | 动作表示与生成方式 | 本章直觉 |
| --- | --- | --- |
| π₀ / π₀.₅ | 连续 action，通过 **flow matching** 生成 | 从随机连续动作出发，多次修正，逐渐得到一段可执行的连续 action chunk |
| π₀-FAST | 把动作编码为离散 **action tokens**，再自回归生成 | 像语言模型逐个生成词元一样，根据已有输入和已生成 token 继续生成后续动作 token，最后解码为动作 |

π₀.₅ 仍然走 `pi0.py` 中的连续动作 flow matching 路线，并不是 FAST 的 token 路线。π₀-FAST 的“自回归”表示后续 token 的生成依赖前面已有的 token。现在只需建立这个概念边界；flow matching 的训练与采样直觉会在[第 06 章](06_jax_and_pi0_basics.md)展开。

## 源码地图

阅读源码时先带着“字段在哪里产生、在哪里适配、在哪里生成动作”三个问题，不必从模型文件第一行读到最后一行。

| 源码入口 | 本章应该定位什么 |
| --- | --- |
| [`src/openpi/policies/libero_policy.py`](../../src/openpi/policies/libero_policy.py) | `make_libero_example()` 给出两张图像、8 维 state 和 prompt；`LiberoInputs` 把 LIBERO 字段映射为模型字段；`LiberoOutputs` 只返回前 7 维动作 |
| [`examples/libero/main.py`](../../examples/libero/main.py) | 从仿真 observation 构造 8 维 state；请求 action chunk；只执行前 `replan_steps` 步并在队列清空后重新规划 |
| [`src/openpi/models/model.py`](../../src/openpi/models/model.py) | `Observation`、`Actions`、`action_dim`、`action_horizon` 和统一的 `sample_actions` 接口 |
| [`src/openpi/models/pi0.py`](../../src/openpi/models/pi0.py) | π₀/π₀.₅ 的连续动作 flow matching 训练与 action chunk 采样入口 |
| [`src/openpi/models/pi0_fast.py`](../../src/openpi/models/pi0_fast.py) | π₀-FAST 的 action token 损失和自回归解码入口 |

把环境接口和模型内部接口分开看很重要：LIBERO 原始 state 是 8 维、单步 action 是 7 维；进入通用模型前，transform 可以把它们补齐到配置要求的内部维度，输出时再由 `LiberoOutputs` 截回 7 维。

## 动手实践

### 实践 1：打印 LIBERO 示例 observation 的 shape

在 **仓库根目录**运行：

```bash
uv run python -c "from openpi.policies.libero_policy import make_libero_example; x=make_libero_example(); print({k: getattr(v, 'shape', type(v).__name__) for k, v in x.items()})"
```

这条命令只创建随机示例并打印字典中每个值的 shape 或类型，不启动模型、不下载 checkpoint，也不运行 LIBERO。随机的是数组内容，shape 是固定的。

观察时回答三个问题：

1. 哪个字段是 `(8,)`？
2. 哪两个字段都是 `(224, 224, 3)`？
3. 为什么 `prompt` 显示类型名而不是数组 shape？

如果依赖尚未安装，命令可能在 import 阶段报 `ModuleNotFoundError`。这不改变本章的源码结论；保存报错信息，完成第 02 章环境配置后再运行。不要把“阅读源码确认 shape”写成“命令已经成功运行”。

### 实践 2：小型 shape 阅读题

假设策略返回一个 shape 为 `(50, 7)` 的 action chunk，客户端配置 `replan_steps=5`。先不要看答案，填写：

| 问题 | 你的答案 |
| --- | --- |
| chunk 中有多少个时间步？ |  |
| 每个时间步有多少个控制量？ |  |
| 本轮会放入执行队列多少步？ |  |
| 执行队列清空后，是否直接继续执行旧 chunk 的第 6 步？ |  |

<details>
<summary>展开参考答案</summary>

chunk 有 50 个时间步，每步 7 个控制量。本轮只把前 5 步放入队列；这 5 步执行完后，客户端使用最新 observation 重新请求 chunk，而不是直接继续旧 chunk 的第 6 步。

</details>

### 实践 3：画出一次滚动规划的数据流

先遮住下面最后一行，用本章字段补全数据流：

```text
LIBERO 当前时刻
  ├─ ?：桌面第三人称图像
  ├─ ?：腕部图像
  ├─ ?：8 维机器人状态
  └─ ?：自然语言任务
          ↓
        policy
          ↓
  shape = (?, ?) 的 action chunk
          ↓
  执行前 ? 步 → 获得新 observation → 再规划
```

参考填写：

```text
observation/image + observation/wrist_image + observation/state + prompt
    → policy → (action_horizon, 7) action chunk
    → 执行前 replan_steps 步 → 新 observation → 重新请求
```

## 预期结果

实践 1 成功时，输出的具体排版可能略有不同，但应包含这些事实：

```text
'observation/state': (8,)
'observation/image': (224, 224, 3)
'observation/wrist_image': (224, 224, 3)
'prompt': 'str'
```

示例字典没有 `actions` 字段是正常的：它展示的是推理请求输入。7 维 action 是 policy 推理后的输出，维度依据可在 `LiberoOutputs` 的 `[..., :7]` 以及 LIBERO 客户端的 `env.step(action.tolist())` 路径中确认。

完成三项实践后，你应留下至少一份简短记录，包含：

- 四个输入字段的名称、shape 或类型；
- “8 维 state 从哪里拼接出来、7 维 action 在哪里被截取”的源码位置；
- 一句话解释 action chunk 与 `replan_steps` 的关系。

## 常见问题

### 1. 为什么有两张真实图像，但模型代码看起来支持三路图像？

LIBERO 请求提供第三人称和腕部两张图像。`LiberoInputs` 会为不存在的右腕相机创建全零占位图，并设置相应 mask，以适配通用模型输入结构。占位图不代表 LIBERO 实际又拍了一张相机图像。

### 2. 8 维 state 和 7 维 action 为什么不相等？

state 描述“现在是什么状态”，action 描述“下一步怎样控制”，两者含义不同，维度没有必须相等的规则。本例中 state 的夹爪关节位置占 2 维，而 action 用 1 个控制量表达夹爪命令。

### 3. 为什么模型配置里可能看到 32 维 action？LIBERO 不是 7 维吗？

32 维是某些通用模型配置的内部统一宽度。LIBERO 的 7 维 action 会被 padding 到内部宽度，推理后再由 `LiberoOutputs` 取回前 7 维。环境接口仍然是 7 维。

### 4. Action chunk 越长越好吗？

不一定。较长 chunk 可以降低推理调用频率并表达更长的短时计划，但远端动作更容易因执行误差或场景变化而过时。实际系统还会结合推理延迟、控制频率和任务接触复杂度选择 `action_horizon` 与 `replan_steps`。

### 5. 本章命令失败，是否说明项目代码有问题？

不一定。若是 `ModuleNotFoundError`、Python 版本或环境路径问题，先按第 02 章完成依赖和环境自检。记录完整报错比反复盲目重试更有价值。若命令已经进入函数并打印出字段，则本章 observation shape 实践已经完成。

## 本章验收

不要只判断“读完了”，请逐项完成以下验收：

- [ ] 能用 LIBERO 单臂任务解释 observation、state、action、episode、trajectory、demonstration 和 policy，并指出不是每个 episode 都是 demonstration；
- [ ] 能写出两张图像 `(224, 224, 3)`、state `(8,)`、prompt 字符串和单步 action 7 维；
- [ ] 能在 `examples/libero/main.py` 指出 8 维 state 来自 `3 + 3 + 2` 的拼接，在 `src/openpi/policies/libero_policy.py` 指出输出为什么是 7 维；
- [ ] 能解释 action chunk 是按时间排列的多步动作，而不是同一步动作的复制；
- [ ] 能解释为什么使用 action chunk：减少逐步推理开销、联合产生更连贯的短时动作，同时通过重新规划减轻旧计划过时的问题；
- [ ] 能说明客户端只执行预测 chunk 的前 `replan_steps` 步，随后基于新 observation 重新规划；
- [ ] 能明确说出：π₀/π₀.₅ 使用连续动作 flow matching，π₀-FAST 使用 action tokens 的自回归生成。

其中“为什么使用 action chunk”和“observation/action 维度来自哪里”是进入下一章前必须口头讲清楚的两项。

## 思考题

### 1. 同一张桌面图像分别配上“把物体放进碗里”和“把物体推到左边”，policy 的输出应该相同吗？为什么？

<details>
<summary>展开参考答案</summary>

不应该。Vision 描述当前场景，Language 指定目标；场景相同但目标不同，合适的动作序列也会不同。这正是 VLA 同时使用视觉和语言的原因。

</details>

### 2. 一次失败的 LIBERO rollout 是 episode、trajectory 还是 demonstration？

<details>
<summary>展开参考答案</summary>

它仍然是一次 episode，也产生了一条 trajectory。它是否被称为 demonstration 取决于数据选择目的；若 demonstration 指提供给模仿学习的专家范例，普通失败 rollout 通常不算。

</details>

### 3. 如果模型预测 50 步，但机器人执行到第 3 步时物体被碰偏，为什么不一定要执行完剩余 47 步？

<details>
<summary>展开参考答案</summary>

旧 chunk 基于预测前的 observation，物体被碰偏后它已经部分过时。客户端只执行较短前缀并重新观测、重新规划，可以让新 chunk 对变化作出修正。

</details>

### 4. π₀-FAST 的“FAST”是否意味着它只是把 π₀ 的连续动作采样循环少运行几次？

<details>
<summary>展开参考答案</summary>

不是。本章需要抓住的是表示和生成范式不同：π₀/π₀.₅ 用 flow matching 生成连续 action，π₀-FAST 把动作编码为 token 并自回归生成。两者不是简单的“同一算法多跑或少跑几次”。

</details>

[← 上一章：课程大纲](00_course_outline.md) · [课程入口：README](README.md) · [下一章：环境安装与自检 →](02_environment_setup.md)
