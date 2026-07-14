# 第 05 章：数据协议与 Transform

[← 上一章：openpi 架构与一次请求的数据流](04_openpi_architecture.md) · [课程入口：README](README.md) · [下一章：JAX、NNX 与 π₀ 基础 →](06_jax_and_pi0_basics.md)

## 本章目标

完成本章后，你应该能够：

- 区分 LIBERO 训练数据集 key、在线推理请求 key 和模型统一 `Observation` 字段；
- 按真实顺序解释训练输入、推理输入和推理输出三条 transform 管线；
- 说明 `repack_transforms`、`data_transforms`、normalization 和 `model_transforms` 各自解决什么问题；
- 追踪 8 维 state、7 维 action 怎样进入内部模型维度，以及 action 为什么最终又变回 7 维；
- 解释 normalization statistics、delta/absolute action 语义和 checkpoint/config 为什么必须配套；
- 面对错 key、错 shape、错统计量或错动作语义时，先在正确的层收集证据。

本章仍用第 03 章中 `serve_policy.py --env LIBERO` 默认加载的官方 **JAX `pi05_libero`** 预训练路径作为主表和 shape 示例。后续 LoRA 教学会另建 π₀ 配置 `pi0_libero_tutorial`；本章不会预设它必须照搬某个旧配置的 delta 选项。若 checkpoint 被识别为 PyTorch，图像在 `Observation` 内部的布局会不同，后文单独说明。

## 前置条件

开始前应已完成[第 04 章](04_openpi_architecture.md)，并能指出一次请求中的传输、Policy input transforms、`Observation.from_dict()`、模型采样和 output transforms。

还应记住第 01 章中的两个环境接口事实：

```text
LIBERO state  = 末端位置 3 + axis-angle 3 + 夹爪关节位置 2 = 8 维
LIBERO action = 末端平移 3 + 末端旋转 3 + 夹爪控制 1       = 7 维
```

本章练习以静态源码搜索和手工追踪字典为主，不要求安装 LIBERO、下载 tokenizer 或加载 checkpoint。所有命令默认从仓库根目录运行。

## 核心概念

### 同一份信息会经历三套名字

初学者最容易把下面三个时刻混在一起：

1. **训练数据集样本：**转换后的 LeRobot frame 使用 `image`、`wrist_image`、`state`、`actions` 和 task；
2. **在线推理请求：**LIBERO 客户端直接发送 `observation/image`、`observation/wrist_image`、`observation/state` 和 `prompt`，没有输入 `actions`；
3. **模型统一协议：**`LiberoInputs` 与 model transforms 产生 `image`、`image_mask`、`state`、token 及其 mask，随后建立 `Observation`。训练时 action chunk 作为单独的 `Actions` 张量交给模型，不是 `Observation` 的成员。

`repack_transforms` 的目的，就是先让训练样本看起来像在线请求；之后两条路径才能复用同一套 `data_transforms`。

### LIBERO 字段对照表

下表以当前 [`convert_libero_data_to_lerobot.py`](../../examples/libero/convert_libero_data_to_lerobot.py)、[`LeRobotLiberoDataConfig`](../../src/openpi/training/config.py)、[`LiberoInputs`](../../src/openpi/policies/libero_policy.py) 和默认 JAX `pi05_libero` 为依据。`H` 表示 action horizon；该预训练配置中 `H=10`、内部 `action_dim=32`、`max_token_len=200`。

| 原始训练 key / 推理 key | repack / adapter 后的 key | 统一 `Observation` / action 字段 | shape（关键阶段） | dtype（关键阶段） | 语义 |
| --- | --- | --- | --- | --- | --- |
| 训练：`image`；推理：`observation/image` | 训练 repack 为 `observation/image`；`LiberoInputs` 变为 `image/base_0_rgb` | `Observation.images["base_0_rgb"]` | 训练存储 `(256,256,3)`；推理请求 `(224,224,3)`；JAX 最终 batch `(1,224,224,3)` | 训练存储类型 `image`，LeRobot 读取时通常是浮点 CHW；adapter 转 HWC `uint8`；JAX 最终 `float32`、范围 `[-1,1]` | 第三人称场景 RGB |
| 训练：`wrist_image`；推理：`observation/wrist_image` | 训练 repack 为 `observation/wrist_image`；adapter 变为 `image/left_wrist_0_rgb` | `Observation.images["left_wrist_0_rgb"]` | 同上 | 同上 | 腕部相机 RGB |
| 两侧都没有右腕图像 | `LiberoInputs` 创建 `image/right_wrist_0_rgb = zeros_like(base_image)` | `Observation.images["right_wrist_0_rgb"]` | JAX 最终 batch `(1,224,224,3)` | `uint8` 占位图，进入 JAX `Observation` 后为 `float32` | 通用三相机协议的占位，不是第三台真实相机 |
| 训练：`state`；推理：`observation/state` | 训练 repack 为 `observation/state`；adapter 变为 `state`；model transform padding | `Observation.state` | 原始 `(8,)`；`pi05_libero` padding 后 `(32,)`；最终 batch `(1,32)` | 训练与在线请求都应显式提供 `float32`；模型规格也是 `float32` | 末端位置 3 + axis-angle 3 + 两个夹爪关节位置 |
| 训练：逐 frame `actions`；推理输入：无 | repack 后仍为 `actions`；adapter 原样带入；loader 组成 chunk，model transform padding | 训练的 `Actions`；推理时是 π₀.₅ 连续模型输出再进入 output transforms | frame `(7,)`；训练 chunk `(H,7)`；内部 `(1,H,32)`；返回 `(H,7)` | 训练 action、连续模型输出和环境 action 都应为 `float32` | 每步 3 平移 + 3 旋转 + 1 夹爪控制；按时间排列成 chunk |
| 训练：frame 写入 `task`，loader 通过 `task_index` 查任务文本；推理：`prompt` | `PromptFromLeRobotTask` 先生成 `prompt`；repack/adapter 保留 `prompt`；model transform 生成 token | `Observation.tokenized_prompt` | 原始是一个字符串；最终 `(1,200)`（`pi05_libero`） | 原始 `str`；最终为整数 token（模型规格 `int32`） | 当前任务的自然语言指令 |
| 原始两侧均无 `image_mask` | `LiberoInputs` 创建 `image_mask/base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb` | `Observation.image_masks[...]` | 每路未 batch 标量；最终各 `(1,)` | `bool` | 标记图像是否有效；π₀/π₀.₅ 的 LIBERO 右腕占位为 `False`，两路真实图像为 `True` |
| 原始两侧均无 token mask | `TokenizePrompt` 生成 `tokenized_prompt_mask` | `Observation.tokenized_prompt_mask` | 最终 `(1,200)`（`pi05_libero`） | `bool` | `True` 表示真实 token，`False` 表示补齐位置 |
| 仅 π₀-FAST model transform 生成 | `token_ar_mask`、`token_loss_mask` | `Observation.token_ar_mask`、`token_loss_mask` | 长度由 FAST 的 `max_token_len` 决定 | `int` / `bool` | 控制自回归注意关系和训练 loss 位置；当前 `pi05_libero` 请求中不使用 |

表中 shape 不能脱离阶段阅读。例如，训练文件中的图像是 256×256，在线客户端已提前变成 224×224，而最终 `Observation` 还多了 batch 维。看到其中一个 shape，不能断言其他阶段也完全相同。

### 协议布局与后端内部布局

网络协议和 `LiberoInputs` 使用 HWC 图像；Policy 加 batch 后，进入 `Observation.from_dict()` 前是 BHWC。随后后端分支不同：

| 阶段 | JAX 默认 `pi05_libero` | PyTorch checkpoint 路径 |
| --- | --- | --- |
| 请求/adapter 协议 | HWC `uint8` | HWC `uint8` |
| Policy 加 batch | BHWC `uint8` | BHWC `torch.uint8` |
| `Observation.from_dict()` 后 | BHWC `float32`，范围 `[-1,1]` | 转为 BCHW `torch.float32`，范围 `[-1,1]` |

因此主表的最终 BHWC shape 只描述本课程默认 JAX 路径，不能外推到 PyTorch 内部。state 和训练 action 也应在协议入口显式构造成 `float32`；Policy 的 `jnp.asarray`/`torch.from_numpy` 是后端转换，不是“任意浮点 dtype 都保证自动变成 float32”的接口承诺。调试时应在发送前检查 dtype，而不是依赖某台机器的 JAX x64 或 PyTorch 行为。

### 三条 transform 管线的准确顺序

#### 训练输入

[`training/data_loader.py`](../../src/openpi/training/data_loader.py) 的 `transform_dataset()` 和 `transform_iterable_dataset()` 按下列顺序组装：

```text
dataset sample
  → repack_transforms.inputs
  → data_transforms.inputs
  → Normalize
  → model_transforms.inputs
  → Observation + Actions
```

对 LIBERO 来说：

- repack 把 `image` 等数据集 key 改成在线协议的 `observation/image` 等 key；
- data transforms 用 `LiberoInputs` 建立三路图像、mask、state、actions 和 prompt，必要时再做额外动作语义变换；
- normalization 处理 state 和 actions；
- model transforms resize、tokenize，并在 π₀/π₀.₅ 路径把 state/actions 补到模型 action dimension。

#### 推理输入

[`policy_config.create_trained_policy()`](../../src/openpi/policies/policy_config.py) 的标准服务路径组装为：

```text
inference request
  → InjectDefaultPrompt（仅在缺 prompt 且配置了默认值时生效）
  → data_transforms.inputs
  → Normalize
  → model_transforms.inputs
  → Observation.from_dict()
```

这里**没有使用 `data_config.repack_transforms`**。原因不是漏做，而是环境已经直接发送 `observation/...` 在线 key。`create_trained_policy()` 确实允许调用者额外传入一个可选 `repack_transforms` 参数，但默认 LIBERO 服务没有传它；不要把这个扩展点误认为训练数据集 repack 会自动作用于每次在线请求。

#### 推理输出

同一个函数实际组装的输出顺序是：

```text
model-native output
  → model_transforms.outputs（模型专用输出变换/解码，如 FAST action token 解码）
  → Unnormalize
  → data_transforms.outputs（机器人/环境动作语义与字段）
  → 可选 repack_transforms.outputs
  → response dict
```

这些输出 transforms 不一定与输入 transforms 一一互逆。例如 resize 没有对应的“反 resize”，而 π₀-FAST 有一个只在输出侧需要的 `ExtractFASTActions`。模型分支应分开理解：

```text
π₀ / π₀.₅:
sample_actions → 标准化连续 (B,H,D) → 去 batch → model-specific outputs（通常为空） → Unnormalize

π₀-FAST:
sample_actions → token-ID 序列 → 去 batch → ExtractFASTActions → 连续 (H,D) → Unnormalize
```

对 `pi05_libero` 而言，model-specific output transforms 为空，`extra_delta_transform=False`，所以关键路径是：

```text
(H,32) model actions → Unnormalize → LiberoOutputs → (H,7) actions
```

如果某套 LIBERO 配置启用了额外 delta transform，`Group.push()` 会把逆变换放到已有输出适配之前，因此 data outputs 的顺序是：

```text
AbsoluteActions → LiberoOutputs
```

先恢复动作语义，再裁成环境需要的前 7 维。顺序颠倒会让逆变换缺少 `state` 或错误处理维度。

### 三组 transform 的边界

| 分组 | 训练 | 推理输入 | 推理输出 | LIBERO 例子 |
| --- | --- | --- | --- | --- |
| `repack_transforms` | 使用 | 默认不使用数据配置里的 repack | 仅显式传入可选 repack 时使用 | `image → observation/image`、`actions → actions` |
| `data_transforms` | 使用 | 使用 | 使用 outputs | `LiberoInputs`、可选 `DeltaActions`；输出侧可选 `AbsoluteActions`、`LiberoOutputs` |
| normalization | state + actions | 通常只有 state | actions 反归一化 | z-score 或 quantile，由配置和统计资产决定 |
| `model_transforms` | 使用 | 使用 | 使用模型专用 outputs | resize、prompt tokenize、padding；FAST 还编码/解码动作 token |

判断某个变换属于哪一层，可以问：

- 只是数据集列名与在线 key 不同？放在 repack；
- 与某个机器人、坐标系或动作接口有关？放在 data transforms；
- 与 π₀/π₀.₅/π₀-FAST 固定输入规格有关？放在 model transforms；
- 是训练分布的数值尺度？由 normalization statistics 处理。

### `LiberoInputs` 做什么，又没有做什么

[`LiberoInputs.__call__()`](../../src/openpi/policies/libero_policy.py) 会：

- 把训练读取到的浮点 CHW 图像转换为 `uint8` HWC；在线 `uint8` HWC 图像则直接通过；
- 把 base/wrist 图像映射到模型约定的相机名；
- 用全零图像补缺失的右腕视角，并建立 image masks；
- 把 `observation/state` 改名为 `state`；
- 训练样本存在 `actions` 时原样带入；存在 `prompt` 时原样带入。

源码在 actions 附近的注释提到 padding，但这个类的实际语句只是 `inputs["actions"] = data["actions"]`。真正把 state 和 actions 补到模型 action dimension 的是 [`PadStatesAndActions`](../../src/openpi/transforms.py)，它由 `ModelTransformFactory` 放在 π₀/π₀.₅ 的 **model transforms** 中。

也要区分两种“补齐”：`LiberoInputs` 用零图像补一台不存在的相机；`PadStatesAndActions` 用零数值补 state/action 的最后一维。它们不是同一个步骤。

### 7D action、8D state 与内部 32D

`pi05_libero` 使用 `Pi0Config` 的默认 `action_dim=32`。数据进入模型前：

```text
state:   (8,)   → PadStatesAndActions(32) → (32,)
actions: (H,7)  → PadStatesAndActions(32) → (H,32)   （仅训练输入有 actions）
```

`pad_to_dim()` 只在当前维度小于目标时在末尾补零；它不会把 8 维 state 解释成 7 维 action，也不会改变前 7/8 维的语义。模型采样后，`LiberoOutputs` 明确执行 `data["actions"][..., :7]`，所以环境得到 `(H,7)`。

`LiberoOutputs` 的裁剪只是协议恢复，不是“模型只预测了 7 维”。模型实际输出内部宽度，padding 维度也有数值；它们对 LIBERO 环境没有定义，必须丢弃。

### normalization statistics 是行为契约

normalization 不只是让数值“更好看”。统计量由某个数据集、字段排列和动作语义计算，并与 checkpoint 一起定义模型见过的数值空间：

```text
训练：环境数值 --Normalize(训练统计)--> 模型学习的数值
推理：环境 state --同一 Normalize--> 模型输入
输出：连续模型 action --同一 Unnormalize--> 环境动作尺度
```

当前配置工厂中，π₀ 默认使用 mean/std，π₀.₅ 与 π₀-FAST 默认使用 quantile statistics。`create_trained_policy()` 优先从 checkpoint 的 `assets/<asset_id>` 加载统计量，正是为了与原训练过程一致。详细背景见[Normalization statistics](../norm_stats.md)。

下面这些对象必须一起核对：

- 数据集及其 `asset_id`；
- state/action 的维度和顺序；
- action 是 delta 还是 absolute，夹爪维度怎样定义；
- 配置选择 z-score 还是 quantile；
- checkpoint 参数和随附的 assets。

最危险的错误不是总会崩溃。另一份统计量如果恰好也有兼容 shape，代码可能正常返回 `(H,7)`，数值也像普通小数，但尺度、中心或夹爪含义已经错了。这就是“shape 看起来合理、动作却完全不对”。

### delta 与 absolute：以配置为准，不按模型名字猜

LIBERO 原始 action 已经是 delta action。`LeRobotLiberoDataConfig.extra_delta_transform` 是为了兼容一些历史配置/checkpoint：启用后，`DeltaActions` 会对 mask 选中的前 6 维再减去当前 state 的对应值，输出侧用 `AbsoluteActions` 做逆操作，夹爪第 7 维保持原约定。

当前配置注册表给出了一个需要刻意记住的差别：

| 命名配置 | `extra_delta_transform` | 怎样理解 |
| --- | ---: | --- |
| `pi0_libero`、`pi0_libero_low_mem_finetune` 等旧 LIBERO 示例 | `True` | 虽然 raw LIBERO action 已是 delta，仍保留历史训练管线的额外变换，以匹配相应 checkpoint/训练约定 |
| `pi05_libero` | `False` | 不再施加额外 delta 变换；第 03 章的官方预训练推理走这条路径 |

这不是“所有 π₀ 永远设 True、所有 π₀.₅ 永远设 False”的架构定律。它是具体数据与 checkpoint 的兼容选择。第 08 章创建 `pi0_libero_tutorial` 时，必须明确记录所采用的动作语义并让训练、统计量计算和部署保持相同，不能只根据模型名字复制布尔值。

### action horizon、chunk、tokens 与 masks

**Action horizon** 是模型一次覆盖多少个未来时间步。训练 data loader 根据它读取一段 action sequence；模型输出的 action chunk shape 是：

```text
(batch, action_horizon, action_dim)
```

去掉单请求的 batch、裁回 LIBERO 维度后是 `(action_horizon, 7)`。第 03 章客户端的 `replan_steps=5` 只执行 chunk 前 5 步，而 `pi05_libero` 的 horizon 是 10；horizon 是模型预测长度，replan steps 是客户端实际使用长度，两者不是同一参数。

**Image masks** 告诉模型某路图像是真实输入还是 padding。**Token mask** 告诉模型语言序列中哪些位置是真实 token、哪些是补齐。π₀-FAST 还需要 autoregressive mask 和 loss mask；普通 π₀/π₀.₅ 的 `Observation` 对这些字段可为 `None`。

不要在原始 `make_libero_example()` 字典中寻找这些最终字段。原始推理样本只有两张图、8 维 state 和 prompt；mask、第三路占位图、token、padding 和 batch 维都是后续 transform/Policy 才加入的。

## 源码地图

| 文件 | 本章阅读重点 |
| --- | --- |
| [`examples/libero/convert_libero_data_to_lerobot.py`](../../examples/libero/convert_libero_data_to_lerobot.py) | 训练数据 `image`、`wrist_image`、`state(8)`、`actions(7)` 和 task 从哪里来 |
| [`examples/libero/main.py`](../../examples/libero/main.py) | 在线请求的 `observation/...` key、224×224 `uint8` 图像、8 维 state、prompt 和 action chunk 消费 |
| [`src/openpi/training/config.py`](../../src/openpi/training/config.py) | `LeRobotLiberoDataConfig`、三组 transforms、`extra_delta_transform`、`pi05_libero` |
| [`src/openpi/training/data_loader.py`](../../src/openpi/training/data_loader.py) | 训练 transform 的真实组装顺序和 action horizon 读取 |
| [`src/openpi/policies/libero_policy.py`](../../src/openpi/policies/libero_policy.py) | `LiberoInputs` 的字段/图像/mask 适配与 `LiberoOutputs` 的 7 维裁剪 |
| [`src/openpi/policies/policy_config.py`](../../src/openpi/policies/policy_config.py) | 推理 input/output transforms 与 checkpoint norm stats 的组装顺序 |
| [`src/openpi/transforms.py`](../../src/openpi/transforms.py) | `RepackTransform`、`Normalize`、`DeltaActions`、tokenize、`PadStatesAndActions` |
| [`src/openpi/models/model.py`](../../src/openpi/models/model.py) | 最终 `Observation` 字段、JAX/PyTorch 图像布局分支、公共模型接口 |
| [`src/openpi/models/pi0_config.py`](../../src/openpi/models/pi0_config.py) | π₀/π₀.₅ 的 `action_dim`、`action_horizon`、`max_token_len` 和输入规格 |
| [`src/openpi/models/pi0_fast.py`](../../src/openpi/models/pi0_fast.py) | FAST `sample_actions()` 怎样自回归返回 token-ID 序列 |

更完整的系统背景可读[项目架构深读](../architecture_zh.md)和[模型建立与推理深读](../model_build_and_inference_zh.md)；遇到术语或跨层故障，可查后续补齐的[课程术语表](glossary.md)与[故障排查手册](troubleshooting.md)。

## 动手实践

### 实践 1：只靠符号搜索确认三条顺序

从仓库根目录运行：

```bash
rg -n -A 20 '^def transform_dataset' src/openpi/training/data_loader.py
rg -n -B 2 -A 16 '^    return _policy\.Policy' src/openpi/policies/policy_config.py
rg -n -A 22 '^class Group' src/openpi/transforms.py
rg -n -A 14 '^class ExtractFASTActions' src/openpi/transforms.py
```

先独立写出三条顺序，再展开参考答案：

<details>
<summary>参考答案（展开查看）</summary>

```text
training input = repack → data → normalize → model
inference input = data → normalize → model              （标准请求无 dataset repack）
inference output = model-specific output/decoding → unnormalize → data output
```

默认 prompt 注入位于推理 data transforms 前；可选外部 repack 只有显式传给 `create_trained_policy()` 时才存在。输出 transforms 是模型专用处理，并不保证与输入 transforms 一一互逆。

</details>

### 实践 2：手工追踪一个字典

先定位定义：

```bash
rg -n -A 8 '^class LeRobotLiberoDataConfig' src/openpi/training/config.py
rg -n -A 8 '^class ModelTransformFactory' src/openpi/training/config.py
rg -n -B 2 -A 12 '"observation/image": "image"' src/openpi/training/config.py
rg -n -B 2 -A 8 'inputs=\[libero_policy\.LiberoInputs' src/openpi/training/config.py
rg -n -A 42 '^class LiberoInputs' src/openpi/policies/libero_policy.py
rg -n -B 2 -A 10 '^        if "actions" in data:' src/openpi/policies/libero_policy.py
rg -n -A 14 '^class LiberoOutputs' src/openpi/policies/libero_policy.py
rg -n -A 10 '^class PadStatesAndActions' src/openpi/transforms.py
```

在纸面或笔记中从这个**推理请求**开始：

```text
observation/image:       uint8 (224,224,3)
observation/wrist_image: uint8 (224,224,3)
observation/state:       float32 (8,)
prompt:                  str
```

每经过一个符号只写“key/shape/dtype 有什么变化”。完成后再展开参考路径。

<details>
<summary>参考路径（展开查看）</summary>

`LiberoInputs` 产生三路 image、三路 image mask、8 维 state 和 prompt；normalization 处理 state；model transforms resize/tokenize，并把 state 补到 32 维。Policy 加 batch 后，默认 JAX `Observation` 图像为 BHWC。π₀.₅ 采样连续 `(1,10,32)`，Policy 去 batch、反归一化，`LiberoOutputs` 再裁成 `(10,7)`。推理输入始终没有 `actions`。

</details>

这个练习故意不执行 Python import。实例化完整 data config/tokenizer 可能依赖已同步的 openpi 环境并访问 tokenizer assets；静态源码足以验证顺序，也不会在无意中下载资源。若你自行改成运行式实验，应使用现有环境的 `uv run --no-sync ...`，并把“期望结果”和“本机已验证结果”分开记录。

### 实践 3：比较动作语义配置

```bash
rg -n -B 3 -A 9 'name="pi0_libero"' src/openpi/training/config.py
rg -n -B 3 -A 9 'name="pi0_libero_low_mem_finetune"' src/openpi/training/config.py
rg -n -B 3 -A 9 'name="pi05_libero"' src/openpi/training/config.py
rg -n -B 2 -A 9 '^        if self.extra_delta_transform' src/openpi/training/config.py
rg -n -A 18 '^class DeltaActions' src/openpi/transforms.py
rg -n -A 18 '^class AbsoluteActions' src/openpi/transforms.py
```

请不要只抄 `True/False`。对每个配置记录：raw action 已是什么语义、输入侧是否再做变换、输出侧怎样恢复对应动作语义、它必须与哪套 checkpoint/统计量保持一致。

### 实践 4：建立 shape/dtype 调试清单

复制下面的清单到实验记录，并为自己的样本填证据：

- [ ] 训练 key 是 `image/wrist_image/state/actions/task`，在线 key 是 `observation/.../prompt`，没有把二者混传；
- [ ] 两路图像经过 adapter 后都是 HWC、3 通道、`uint8`，方向与训练预处理一致；
- [ ] Policy 加 batch 后，默认 JAX 路径保持 BHWC；PyTorch 路径在 `Observation.from_dict()` 中转为 BCHW；
- [ ] image mask 的 key 与 image key 完全一致，右腕占位 mask 符合当前模型类型；
- [ ] 原始 LIBERO state 最后一维是 8，顺序为 `3+3+2`；
- [ ] 在线 state 与训练/环境 action 数组显式为 `float32`，没有依赖后端隐式降精度；
- [ ] 原始单步 action 最后一维是 7，训练 loader 产生的时间维长度等于 action horizon；
- [ ] π₀/π₀.₅ model transforms 后 state/action 最后一维等于配置 `action_dim`；
- [ ] prompt 非空，token 与 token mask 同时存在且长度等于 `max_token_len`；
- [ ] 没有 NaN/Inf，图像范围、state/action 数值范围与当前阶段相符；
- [ ] normalization assets 的 `asset_id`、字段维度、量化方式和 checkpoint 配套；
- [ ] `extra_delta_transform` 与训练/部署使用同一动作语义；
- [ ] 当前模型原生输出已辨认：π₀/π₀.₅ 是连续 chunk，π₀-FAST 是待 `ExtractFASTActions` 解码的 token ID；
- [ ] 返回 action shape 是 `(action_horizon, 7)`，再由客户端按 `replan_steps` 截取执行。

## 预期结果

完成练习后，重点检查这四个易错点，而不是再抄一遍整条管线：

| 易错点 | 可观察的正确结果 |
| --- | --- |
| 把训练 key 当在线 key | 追踪记录中先出现 `image → observation/image` 的 training-only repack；在线记录直接从 `observation/image` 开始 |
| 把协议 HWC 当成所有后端内部布局 | 记录中明确写出 JAX BHWC 与 PyTorch BCHW 的分叉位置是 `Observation.from_dict()` |
| 把所有 `sample_actions` 输出当连续 `(H,D)` | π₀/π₀.₅ 记录连续 chunk；π₀-FAST 记录 token ID → `ExtractFASTActions` → 连续 chunk |
| 只看 `(H,7)` 就判定动作正确 | 记录同时包含 norm asset 来源与 `extra_delta_transform`，证明尺度和语义也已核对 |

### 本文的验证边界

本章已静态核对 converter、data loader、config、Policy 组装、LIBERO adapter、通用 transforms 和模型规格。没有实例化 tokenizer、读取真实 LeRobot sample、加载 `pi05_libero` assets 或运行 GPU 推理；表中的最终 shape 来自配置和源码契约，不是本机 runtime dump。

## 常见问题

排错时统一使用“症状 → 最早失败层 → 源码/样本证据 → 只改一个变量”的顺序。不要因为最终动作不好看就同时改 key、统计量和 delta 选项。

### 1. 报 `KeyError: observation/image`，从哪里查？

先确认错误发生在 `LiberoInputs`，再对比两侧 key：

```bash
rg -n -B 2 -A 12 '^                        element = \{' examples/libero/main.py
rg -n -B 2 -A 12 'base_image = _parse_image' src/openpi/policies/libero_policy.py
rg -n -B 2 -A 10 '"observation/image": "image"' src/openpi/training/config.py
```

训练样本若仍叫 `image`，检查 repack 是否进入 data loader；在线请求若发送 `image`，应修客户端协议，而不是给默认推理偷偷套上训练 repack。

### 2. 报 shape/type 错误，应该从模型开始看吗？

先记录每层最后一个正确 shape。若 `LiberoInputs` 前图像是 CHW 浮点，它会负责转 HWC `uint8`；若 adapter 后仍不是 3 通道，先修数据。若 state/action 到 model transform 前是 8/7 维但之后不是 `action_dim`，再查 `PadStatesAndActions` 和当前 `ModelType`。只有最终 `Observation` 已符合 `inputs_spec()` 后，才进入模型内部排查。

### 3. action shape 正确但机器人乱动，为什么？

正确 `(H,7)` 不能排除尺度和语义错误。按顺序记录：config 名、checkpoint 路径、`asset_id`、stats 文件来源、`use_quantile_norm`、`extra_delta_transform`，再比较训练和服务配置。统计量或 delta 语义错配都可能产生有限、平滑、shape 正确但物理含义错误的动作。

### 4. 归一化统计缺失与统计错误有什么区别？

训练 data loader 在需要统计但找不到时会明确报错；部署加载也会要求有效 `asset_id`。而“加载了错误但 shape 兼容的统计量”可能不报错。前者看异常和文件路径，后者必须靠 provenance：统计由哪个 config、哪个数据集、哪种 action 语义计算。

### 5. 是否可以通过切换 `extra_delta_transform` 试到能动为止？

不建议。先用 `rg` 和实验记录确认训练该 checkpoint 时的配置，再保持服务一致。盲试会把动作语义、统计量和模型能力混成一个变量，即使偶然能动也无法解释或复现。

### 6. 为什么右腕全零图有时 mask 是 `True`？

`LiberoInputs` 对 π₀/π₀.₅ 将缺失右腕 mask 设为 `False`，对 π₀-FAST 按其约定设为 `True`。先检查 `model_type`，不要脱离模型协议把所有 padding 图像一律设成相同 mask。

更多归一化资料见 [`docs/norm_stats.md`](../norm_stats.md)；跨层排错见后续补齐的[故障排查手册](troubleshooting.md)。

## 本章验收

进入下一章前，提交或展示以下可观察证据：

- [ ] 一份实践 1 的命令输出，并在旁边标出 training input、inference input、model-specific output 三条顺序；
- [ ] 一张手工追踪表，至少包含“原始请求、`LiberoInputs` 后、normalization 后、model transforms 后、`Observation`、环境输出”六行及各自 key/shape/dtype；
- [ ] 追踪表中明确记录默认 JAX 图像为 BHWC，并引用源码说明 PyTorch 分支会转为 BCHW；
- [ ] 一份 `pi0_libero` 与 `pi05_libero` 配置摘录，记录各自 `extra_delta_transform`、raw LIBERO action 语义和配套 checkpoint/statistics 约束；
- [ ] 一句带源码符号的 FAST 分支说明：`sample_actions` 产生 token ID，`ExtractFASTActions` 在 `Unnormalize` 前解码为连续动作；
- [ ] 一张四项诊断卡：wrong key → repack/adapter，wrong shape/dtype → adapter/backend/model spec，wrong stats → asset/config/checkpoint，wrong action semantics → delta/absolute transforms。

## 思考题

### 1. 为什么在线请求不应该直接复用训练数据集的 `image` key？

<details>
<summary>参考答案（展开查看）</summary>

训练数据先通过 repack 把 `image` 改成在线协议的 `observation/image`，之后才进入共享 `LiberoInputs`。在线环境本来就应发送目标协议；直接发训练 key 会把数据集存储格式泄漏到服务接口，并导致 adapter 缺 key。

</details>

### 2. 为什么 wrong normalization statistics 可能比缺文件更难发现？

<details>
<summary>参考答案（展开查看）</summary>

缺文件通常立即报错；错误统计量如果 key 和 shape 兼容，Normalize/Unnormalize 仍能完成，返回 action 也有正确 shape。错误只体现在数值尺度和物理行为，因此必须记录统计资产来源与动作语义，不能只看程序是否运行。

</details>

### 3. `LiberoOutputs` 已裁成 7 维，为什么输出前仍需要正确的 32 维反归一化？

<details>
<summary>参考答案（展开查看）</summary>

模型是在内部 32 维协议中采样，Unnormalize 按训练统计恢复有定义的动作维度，并对 padding 维保持合适默认处理。之后 `LiberoOutputs` 才选择环境定义的前 7 维。先随意裁剪会改变既定输出管线，也可能破坏需要 state/完整字典的动作逆变换。

</details>

### 4. 为什么不能从“π₀”三个字符直接决定 `extra_delta_transform=True`？

<details>
<summary>参考答案（展开查看）</summary>

该布尔值描述的是具体数据与历史 checkpoint 的动作协议，不是模型架构的数学必然。当前旧 π₀ LIBERO 示例为兼容训练历史设为 `True`，`pi05_libero` 为 `False`；新教学配置仍必须依据自己的训练数据、统计量和部署约定明确选择并保持一致。

</details>

[← 上一章：openpi 架构与一次请求的数据流](04_openpi_architecture.md) · [课程入口：README](README.md) · [下一章：JAX、NNX 与 π₀ 基础 →](06_jax_and_pi0_basics.md)
