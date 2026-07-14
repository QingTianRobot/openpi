# 第 08 章：配置教学微调与计算归一化统计

[← 上一章：准备小规模 LIBERO 数据集](07_prepare_small_dataset.md) · [课程入口：README](README.md) · [下一章：运行教学微调 →](09_run_finetuning.md)

## 本章目标

完成本章后，你应该能够：

- 用初学者语言解释 `TrainConfig` 中模型、数据、权重、可训练参数、优化、日志、checkpoint 和 assets 等字段的职责；
- 在专用练习分支中基于 `pi0_libero_low_mem_finetune` 添加唯一的 `pi0_libero_tutorial` 配置，并检查源码 diff；
- 说明为什么模型使用的两个 LoRA variant 与生成 `freeze_filter` 的两个 variant 必须完全相同；
- 区分本课程的 π₀ LoRA 与第 03 章使用的 `pi05_libero`，并解释为什么二者 checkpoint 不能混用；
- 不创建模型、不下载权重，只通过配置注册表检查教学配置；
- 解释 normalization statistics 的计算管线、保存路径和 `--max-frames` 的准确行为；
- 在写入前检查精确目标，在写入后验证 `state`/`actions` 的 key、shape 和有限性。

本章包含一项**由学习者完成的源码修改练习**。本课程文档不会替你修改 [`config.py`](../../src/openpi/training/config.py)；请在自己的专用分支中插入配置、检查 diff，再运行统计命令。

## 前置条件

开始前应完成[第 07 章](07_prepare_small_dataset.md)，并已经得到本地 LeRobot 数据集：

```text
openpi-tutorial/libero-small
```

它应至少满足：两路图像可读、state 为 8 维、actions 为 7 维、task 非空、数值不含 NaN/Inf。归一化统计会忠实总结送入统计管线的数值；如果数据协议本身错误，统计文件只会把错误固定下来。

本章命令从仓库根目录运行，使用宿主机 Python 3.11 `uv` 主环境：

```bash
pwd
uv run --no-sync python --version
```

`--no-sync` 表示不在命令执行时重新解析或下载依赖。若 import 或本地数据检查失败，应先回到第 02、07 章修复环境与数据，不要靠删除 cache 或改统计脚本绕过错误。

## 核心概念

### `TrainConfig` 是一次训练的总装配单

[`TrainConfig`](../../src/openpi/training/config.py) 不是只保存“训练多少步”的参数表。它把模型结构、数据来源、transform、归一化 assets、初始权重、参数冻结方式、优化过程和输出位置装配成一个可按名字查找的训练方案。

| 字段 | 初学者可以怎样理解 | 本章关注点 |
| --- | --- | --- |
| `name` | 配置在注册表中的唯一名字 | 使用 `pi0_libero_tutorial`；也参与 assets/checkpoint 路径，后续 Policy Server 还要用同一名字找配置 |
| `model` | 要创建哪一种模型，以及 action 维度、horizon、Gemma variant 等结构参数 | 使用 π₀ 的两个 LoRA variant，而不是 π₀.₅ |
| `data` | 数据 repo id，以及怎样 repack 字段、生成 prompt、处理 LIBERO 输入/动作 | 指向 `openpi-tutorial/libero-small`，继承现有 LIBERO π₀ 训练语义 |
| `weight_loader` | 模型创建后，从哪个 checkpoint 装入已有参数 | 从 `pi0_base` 初始化；LoRA 新参数由当前模型结构补齐 |
| `freeze_filter` | 哪些参数被冻结、不更新 | 由与 `model` 完全相同的 LoRA variant 生成 |
| `trainable_filter` | 哪些参数真正可训练 | 它不是单独填写的字段，而是 `Param AND NOT freeze_filter` 的派生属性 |
| `lr_schedule` | 每一步用多大的学习率 | 本章沿用默认 cosine warmup/decay，只做工程链路演示 |
| `optimizer` | 怎样根据梯度更新可训练参数 | 本章沿用默认 AdamW 和全局梯度裁剪 |
| `ema_decay` | 是否维护参数的指数移动平均副本 | LoRA 示例设为 `None`，关闭 EMA |
| `batch_size` | 每个全局训练 step 使用多少条样本 | 设为 8，优先降低教学运行的资源压力 |
| `num_train_steps` | 执行多少个训练 batch/step | 设为 500；这是工程演示，不是性能目标 |
| `log_interval` | 每隔多少 step 记录一次指标 | 设为 10，短训练中也能看到足够日志 |
| `save_interval` | 每隔多少 step 保存 checkpoint | 设为 100 |
| `keep_period` | 哪些周期 checkpoint 不被清理 | 设为 100；本章每次周期保存都落在保留周期上 |
| `wandb_enabled` | 是否启用 Weights & Biases 远端实验记录 | 设为 `False`，先完成纯本地教学闭环 |
| `assets_base_dir` / `assets_dirs` | normalization statistics 等非权重资产放在哪里 | 默认基目录是 `./assets`，派生目录是 `assets/<config-name>` |
| `checkpoint_base_dir` / `checkpoint_dir` | 训练权重保存在哪里 | 默认是 `checkpoints/<config-name>/<exp-name>`；`exp_name` 在训练章提供 |

未在配置块里显式写出的 `lr_schedule` 和 `optimizer` 会继承 `TrainConfig` 默认值。当前默认学习率计划先 warmup 1000 step，而本章只训练 500 step，所以教学运行结束时仍在 warmup 阶段。这再次说明：500 step 用于验证数据、梯度、日志和 checkpoint 链路，不用于追求或比较 LIBERO 成功率。

### 配置名不是随手写的标签

`_config.get_config("pi0_libero_tutorial")` 会返回一个同时确定以下内容的对象：

```text
配置名
  ├─ π₀/π₀.₅ 与模型 variant
  ├─ 数据 repo id
  ├─ repack / data / model transforms
  ├─ normalization statistics 的查找与写入路径
  ├─ 初始 checkpoint 与冻结规则
  └─ 训练 checkpoint 目录和后续 Policy Server 的加载契约
```

名称本身不会自动证明这些组件彼此兼容，但它是后续命令重新找到这整套装配的入口。若训练时改了名字或模型、服务时仍使用旧配置，即使路径中存在 checkpoint，也可能得到缺 key、shape 不匹配、错误 normalization 或错误动作语义。

因此每个不同实验应使用唯一配置名和独立路径。不要把教学统计覆盖到 `pi0_libero_low_mem_finetune`，也不要让 `pi0_libero_tutorial` 在 `_CONFIGS` 中出现两次；文件末尾会主动拒绝重复名称。

### 本课程为什么选择 π₀ LoRA

本课程创建 `pi0_libero_tutorial`，做法是从现有 `pi0_libero_low_mem_finetune` 复制结构，再只缩小数据与工程规模：

- 模型是 π₀，`model_type` 为 `pi0`；
- PaliGemma 主干使用 `gemma_2b_lora`；
- action expert 使用 `gemma_300m_lora`；
- 初始权重来自 `gs://openpi-assets/checkpoints/pi0_base/params`；
- 数据改为第 07 章准备的 `openpi-tutorial/libero-small`；
- batch 改为 8，训练改为 500 step，关闭 EMA 与 W&B。

LoRA 可以把需要更新的参数限制在较小的 adapter 子集，但它不代表训练不需要创建基础模型，也不保证任意 GPU 都不会 OOM。`low_mem` 是相对完整微调而言，实际内存仍受模型、batch、JAX 编译、优化器状态和设备影响。

### `model` 与 `freeze_filter` 的 variant 必须逐字一致

`Pi0Config.get_freeze_filter()` 会根据 `paligemma_variant` 和 `action_expert_variant` 中是否包含 `lora` 来构造 NNX filter。`TrainConfig.trainable_filter` 再取“所有参数中不属于 freeze filter 的部分”。

本章必须让下面两次构造使用相同参数：

```python
pi0_config.Pi0Config(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)
```

第一次决定模型实际有哪些基础参数与 LoRA 参数；第二次决定哪些路径被冻结。如果一边是 LoRA variant，另一边是普通 variant，训练可能错误冻结 LoRA、错误放开基础权重，或让可训练集合与预期不符。不要凭“看起来都是 π₀”省略核对。

### `pi0_libero_tutorial` 与 `pi05_libero` 是两条不可混用的路线

第 03 章预训练闭环使用 `pi05_libero`，也就是 π₀.₅。本章则明确使用 π₀ LoRA。当前源码中二者至少有这些差异：

| 项目 | `pi0_libero_tutorial` | `pi05_libero` |
| --- | --- | --- |
| 模型类型 | π₀ | π₀.₅ |
| `pi05` | `False`（默认） | `True` |
| action horizon | 50（π₀ 默认） | 10 |
| action transform | `extra_delta_transform=True` | `extra_delta_transform=False` |
| 初始权重 | `pi0_base/params` | `pi05_base/params` |
| 本章训练方式 | 两个 LoRA variant | 现有配置不是本章 LoRA 教学配置 |

checkpoint 保存的是与模型结构和参数路径对应的权重，不是可随意套到另一个名字上的通用文件。不要用 `pi05_libero` 加载本章 π₀ checkpoint，也不要把 `pi05_base` 当作本章初始权重；反方向同样不行。

### normalization statistics 是数据协议的一部分

训练和推理都要用同一组统计量把 state/actions 映射到模型所用尺度，再在输出侧反归一化。统计必须对应实际训练数据以及相同的动作语义。

本章 `compute_norm_stats.py` 的准确管线是：

```text
config name
  → TrainConfig
  → data factory.create(...)
  → LeRobot dataset + action chunk
  → prompt_from_task（若启用）
  → repack transforms
  → LIBERO data transforms
  → RemoveStrings
  → RunningStats(state, actions)
  → assets/<config-name>/<repo-id>/norm_stats.json
```

这里有两个容易混淆的边界：

1. 统计脚本会执行 repack 和 data transforms，所以 `extra_delta_transform=True` 会影响 action statistics；
2. 统计脚本**不会**执行 `Normalize`，也不会执行 model transforms。因此 resize、tokenize、把 8D/7D pad 到模型 32D 等步骤不参与原始统计。

`RemoveStrings` 只移除 prompt 等字符串字段，因为 JAX 和数值统计不需要它们。随后 `RunningStats` 把除最后一维外的维度展平：state 得到 8 维统计，action chunk 得到 7 维统计。当前 `normalize.save` 写出 `norm_stats.json`，其中包含 `mean`、`std`、`q01`、`q99`；π₀ 默认使用 mean/std 的 z-score 路径，但保存文件仍包含 quantile 信息。

## 源码地图

| 文件或资料 | 本章阅读重点 |
| --- | --- |
| [`src/openpi/training/config.py`](../../src/openpi/training/config.py) | `TrainConfig`、`LeRobotLiberoDataConfig`、LIBERO 配置区、唯一名称检查、assets/checkpoint 派生路径 |
| [`src/openpi/models/pi0_config.py`](../../src/openpi/models/pi0_config.py) | π₀/π₀.₅ 开关、默认 action horizon、两个 Gemma variant 与 `get_freeze_filter()` |
| [`src/openpi/training/weight_loaders.py`](../../src/openpi/training/weight_loaders.py) | `CheckpointWeightLoader` 如何载入基础权重并补齐缺失的 LoRA 参数 |
| [`src/openpi/training/optimizer.py`](../../src/openpi/training/optimizer.py) | 默认 cosine learning-rate schedule、AdamW 与 gradient clipping |
| [`scripts/compute_norm_stats.py`](../../scripts/compute_norm_stats.py) | 数据集创建、transform 边界、`max_frames`、`RunningStats` 与输出路径 |
| [`src/openpi/training/data_loader.py`](../../src/openpi/training/data_loader.py) | LeRobot metadata、action horizon、task→prompt 与完整训练 transform 顺序 |
| [`src/openpi/shared/normalize.py`](../../src/openpi/shared/normalize.py) | `RunningStats`、JSON 序列化、`save()`/`load()` 和 `norm_stats.json` 格式 |
| [第 07 章：准备小规模 LIBERO 数据集](07_prepare_small_dataset.md) | `openpi-tutorial/libero-small` 的生成、metadata、8D/7D 与数据质量检查 |
| [`docs/norm_stats.md`](../norm_stats.md) | 何时复用预训练统计、何时为新数据计算统计，以及 action-space 约定 |

遇到陌生缩写可查后续补齐的[课程术语表](glossary.md)；遇到环境、数据、训练或 checkpoint 错误可查后续补齐的[故障排查手册](troubleshooting.md)。

## 动手实践

### 实践 1：建立配置练习分支并定位插入位置

先确认当前修改，再创建只用于本练习的分支：

```bash
git status --short
git switch -c tutorial/pi0-libero-config
rg -n 'Fine-tuning Libero configs|pi0_libero_low_mem_finetune|pi05_libero' \
  src/openpi/training/config.py
```

如果你已经位于自己的练习分支，不要机械地再次创建同名分支；记录当前分支即可。不要在这一步修改 `pi0_config.py`、`data_loader.py` 或统计脚本。本章只要求在 `_CONFIGS` 的 LIBERO 区域增加一个 `TrainConfig` 项。

### 实践 2：插入完整教学配置

打开 [`src/openpi/training/config.py`](../../src/openpi/training/config.py)，在 `_CONFIGS` 的 “Fine-tuning Libero configs” 区域中，把下面完整 block 作为一个新的列表元素插入。不要替换或改名现有配置。

```python
TrainConfig(
    name="pi0_libero_tutorial",
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="openpi-tutorial/libero-small",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=True,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    batch_size=8,
    num_train_steps=500,
    log_interval=10,
    save_interval=100,
    keep_period=100,
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,
    wandb_enabled=False,
),
```

这里的 `prompt_from_task=True` 位于 `base_config=DataConfig(...)` 内，因为 `LeRobotLiberoDataConfig` 自身是 factory，实际 `DataConfig` 才拥有这个字段。把它直接写成 `LeRobotLiberoDataConfig(prompt_from_task=True)` 与当前 dataclass API 不匹配。

`extra_delta_transform=True` 是为了与现有 π₀ LIBERO checkpoint/config 路线保持一致：它会对前 6 个 action 维度再执行基于 state 的 delta 变换，夹爪维度保持原语义。LIBERO 原始动作本已是 delta，因此这不是可对所有数据照抄的通用规则；本章保留它是兼容现有旧 π₀ checkpoint 训练语义的明确教学选择。切换到别的 checkpoint、数据协议或 π₀.₅ 时必须重新论证，不能靠试到机器人“能动”为止。

### 实践 3：检查 diff、唯一名称和意外修改

编辑后先看证据，不要直接训练：

```bash
git diff -- src/openpi/training/config.py
git diff --check -- src/openpi/training/config.py
rg -n 'name="pi0_libero_tutorial"' src/openpi/training/config.py
test "$(rg -c 'name="pi0_libero_tutorial"' src/openpi/training/config.py)" -eq 1
git status --short
```

人工逐项检查：

- diff 只增加一个 `TrainConfig`，没有改动现有 `pi0_libero_low_mem_finetune` 或 `pi05_libero`；
- repo id 精确为 `openpi-tutorial/libero-small`；
- `model=` 和 `freeze_filter=` 中两个 variant 逐字一致；
- 初始权重路径是 `pi0_base/params`，没有写成 `pi05_base/params`；
- `num_train_steps=500`、`batch_size=8`、三个 interval/period 和两个关闭项都正确；
- 名称只出现一次，没有空白错误或重复注册。

### 实践 4：只发现配置，不实例化或下载模型

下面的检查只导入配置注册表并读取 dataclass/factory 字段；它不会调用 `config.model.create(...)`，不会分配模型参数，也不会执行 `weight_loader.load(...)` 下载 GCS 权重：

```bash
uv run --no-sync python - <<'PY'
from openpi.training import config as _config

config = _config.get_config("pi0_libero_tutorial")
data_factory = config.data

assert isinstance(data_factory, _config.LeRobotLiberoDataConfig)
assert data_factory.base_config is not None

print("name:", config.name)
print("data factory:", type(data_factory).__name__)
print("repo id:", data_factory.repo_id)
print("prompt from task:", data_factory.base_config.prompt_from_task)
print("extra delta transform:", data_factory.extra_delta_transform)
print("batch size:", config.batch_size)
print("train steps:", config.num_train_steps)
print("model type:", config.model.model_type.value)
print("action dim:", config.model.action_dim)
print("action horizon:", config.model.action_horizon)
print("PaliGemma variant:", config.model.paligemma_variant)
print("action expert variant:", config.model.action_expert_variant)
print("freeze filter:", config.freeze_filter)
print("trainable filter:", config.trainable_filter)
print("base weights:", config.weight_loader.params_path)
print("assets dir:", config.assets_dirs)
PY
```

关键输出应包括：`pi0`、action horizon 50、两个 `_lora` variant、repo id、batch 8、steps 500，以及非空的 freeze/trainable filter 表达。`action_dim` 会显示模型内部维度 32；这不否定数据的 8D state/7D action，因为 model transforms 会在归一化之后 padding。不要为了“确认配置”实例化数十亿参数模型。

### 实践 5：写入前检查精确统计目标

统计脚本最终使用：

```text
config.assets_dirs / data_config.repo_id
```

默认从仓库根目录运行时，本章目标是：

```text
assets/pi0_libero_tutorial/openpi-tutorial/libero-small/norm_stats.json
```

repo id 含 `/`，所以它会形成两层子目录。先用配置值计算目标并只列出该目标内部的内容：

```bash
uv run --no-sync python - <<'PY'
from openpi.training import config as _config

config = _config.get_config("pi0_libero_tutorial")
repo_id = config.data.repo_id
assert isinstance(repo_id, str) and repo_id

target = config.assets_dirs / repo_id
norm_file = target / "norm_stats.json"

print("target directory:", target)
print("target exists:", target.exists())
print("norm file:", norm_file)
print("norm file exists:", norm_file.exists())

if target.exists():
    for path in sorted(target.rglob("*")):
        print("existing:", path)
PY
```

若目标中已经有统计文件，先确认它来自哪份数据、是否被 checkpoint 或实验引用。需要保留时，应停止本次写入并备份这个**精确目标目录**，或者为新实验选择新的唯一配置名。不要运行 `rm -rf assets`，也不要清空整个 Hugging Face/openpi cache。专用配置名的价值之一，就是让教学统计进入独立目录。

### 实践 6：计算最多 2000 frame 的教学统计

确认本地数据和目标后运行：

```bash
uv run --no-sync scripts/compute_norm_stats.py --config-name pi0_libero_tutorial --max-frames 2000
```

`--max-frames 2000` 是速度选择，不是统计质量标准。对于 LeRobot 路径，当前脚本的行为是：

- 当 `max_frames < len(dataset)` 时，使用 `max_frames // batch_size` 个 batch 并开启 shuffle；
- 否则使用 `len(dataset) // batch_size` 个 batch，且不 shuffle；
- 不足一个完整 batch 的余数不会进入统计。

本章 batch 为 8，因此 2000 恰好对应 250 个完整 batch。若设成小于 8，会得到 0 个 batch，最终无法产生至少两个向量所需的统计。完整训练应让统计覆盖真实训练数据分布；2000 frame 只帮助先验证工程链路，少量或偏置抽样可能低估长尾状态与动作范围。

如果第 07 章的数据只保存在本地 LeRobot cache，并且你希望错误 repo id 立即失败而不是回退到 Hugging Face Hub，可用本地离线解析：

```bash
HF_HUB_OFFLINE=1 uv run --no-sync scripts/compute_norm_stats.py \
  --config-name pi0_libero_tutorial --max-frames 2000
```

`HF_HUB_OFFLINE=1` 只影响 Hugging Face/LeRobot 的 Hub 访问。归一化命令不会调用 `weight_loader.load()`，所以它本来就不会下载 `gs://.../pi0_base/params`；设置 Hugging Face 离线模式也不是用来阻止 GCS 权重下载的开关。若本地 dataset metadata/layout 不完整，离线模式会直接暴露问题。

### 实践 7：读取实际 JSON 并验证 key、shape 与有限性

成功后先确认日志中的 `Writing stats to:` 与实践 5 的目标一致，再运行：

```bash
uv run --no-sync python - <<'PY'
import json
import numpy as np

from openpi.shared import normalize
from openpi.training import config as _config

config = _config.get_config("pi0_libero_tutorial")
repo_id = config.data.repo_id
assert isinstance(repo_id, str) and repo_id

directory = config.assets_dirs / repo_id
path = directory / "norm_stats.json"
assert path.is_file(), path

raw = json.loads(path.read_text())
print("file:", path)
print("JSON top-level keys:", sorted(raw))
assert set(raw) == {"norm_stats"}

stats = normalize.load(directory)
print("stat keys:", sorted(stats))
assert set(stats) == {"state", "actions"}

expected_shapes = {"state": (8,), "actions": (7,)}
for key, expected_shape in expected_shapes.items():
    item = stats[key]
    for field in ("mean", "std", "q01", "q99"):
        value = getattr(item, field)
        assert value is not None, (key, field)
        array = np.asarray(value)
        print(key, field, "shape=", array.shape, "finite=", bool(np.isfinite(array).all()))
        assert array.shape == expected_shape, (key, field, array.shape)
        assert np.isfinite(array).all(), (key, field)
    assert (np.asarray(item.std) >= 0).all(), key
PY
```

shape 正确和全部有限是必要条件，不是充分条件。还应把 mean/std/quantile 的数值范围与第 07 章抽样记录、动作坐标系、夹爪约定和 `extra_delta_transform=True` 的语义一起判断。极小的 std 可能是真正的常量维，也可能意味着数据过少、某个字段被冻结或转换错误；先定位维度和样本证据，不要直接手改 JSON。

### 实践 8：记录进入训练章前的一致性清单

把下面信息保存到自己的实验记录中：

| 契约 | 本章应记录的值 | 为什么后续还要复核 |
| --- | --- | --- |
| config name | `pi0_libero_tutorial` | 第 09 章训练和第 10 章 Policy Server 都要用同一配置入口 |
| repo id | `openpi-tutorial/libero-small` | 决定本地数据解析和 stats 子目录 |
| 数据协议 | 8D state、7D actions | stats 应分别为 `(8,)`、`(7,)`，之后才 pad 到模型 32D |
| 动作语义 | LIBERO 数据 + `extra_delta_transform=True` | 统计、训练与推理输出逆变换必须一致 |
| 模型 | π₀，horizon 50 | 不能拿 π₀.₅ 的 horizon/config/checkpoint 替代 |
| LoRA variants | `gemma_2b_lora` + `gemma_300m_lora` | 模型结构与 freeze filter 必须逐字一致 |
| 初始权重 | `pi0_base/params` | 决定参数树兼容性；不是 `pi05_base` |
| stats 路径 | `assets/pi0_libero_tutorial/openpi-tutorial/libero-small/norm_stats.json` | 训练 loader 和 checkpoint assets 要找到同一份统计 |
| 教学规模 | batch 8、500 step、最多 2000 frame stats | 只验证工程闭环，不是性能承诺 |

若任意一项发生变化，应把它当成一个新实验重新检查，而不是只改最后一条训练命令。

## 预期结果

本文编写环境没有准备本地 `openpi-tutorial/libero-small` 数据与完整运行依赖，也没有实际执行 `compute_norm_stats.py`。因此下面是**条件式预期**，不是本机实测记录。

如果你已按第 07 章准备正确数据、在练习分支插入配置并完成依赖同步，则应看到：

1. `_config.get_config("pi0_libero_tutorial")` 能成功返回唯一配置；
2. 发现脚本显示 repo id、batch 8、steps 500、模型类型 `pi0`、horizon 50 和两个 LoRA variant；
3. 发现阶段没有模型参数分配，也没有基础 checkpoint 下载；
4. 统计命令读取本地 LeRobot 数据并打印 250 个 batch 左右的进度（当数据长度大于 2000 且 batch 为 8 时）；
5. 日志打印 `Writing stats to:`，路径以 `assets/pi0_libero_tutorial/openpi-tutorial/libero-small` 结尾；
6. 目标下出现 `norm_stats.json`，顶层为 `norm_stats`，内部恰有 `state` 和 `actions`；
7. state 的 mean/std/q01/q99 shape 均为 `(8,)`，actions 对应 shape 均为 `(7,)`，且数值全部有限。

实际耗时、进度速度、均值、标准差和 quantile 取决于本地数据内容、抽样顺序、CPU/磁盘性能及 LeRobot 版本。保存命令、完整日志、配置 diff 和检查输出，作为进入第 09 章的证据。

## 常见问题

### 1. `Config 'pi0_libero_tutorial' not found` 或提示重复名称

先运行 `rg -n 'pi0_libero_tutorial' src/openpi/training/config.py`。找不到说明配置未插入当前 checkout、名字拼错或命令不在预期仓库；出现两次 `name=` 则会触发 `_CONFIGS` 唯一性检查。再看 `git diff` 和 `git branch --show-current`，不要先改 `get_config()`。

### 2. 本地数据存在，却仍访问 Hub、要求登录或报 repo not found

打印 `HF_LEROBOT_HOME`、repo id 和第 07 章的精确 root，确认本地布局确实位于 LeRobot 当前版本期望的位置。需要纯本地诊断时把 `HF_HUB_OFFLINE=1` 放在 `uv run` 前。离线后失败说明本地 metadata/cache 契约不完整，不应通过登录掩盖路径错误。

### 3. 训练以后报 `Normalization stats not found`

比较三项证据：训练使用的 config name、`config.assets_dirs / repo_id`、磁盘上的 `norm_stats.json`。常见原因是统计时用了另一个配置名/repo id、从别的工作目录运行导致相对基目录不同，或文件被手工移动。不要把任意 stats 复制到报错路径；先确认数据和动作协议真的相同。

### 4. 统计进度为 0、很快失败，或样本数比 `max_frames` 小

检查 `len(dataset)`、`batch_size` 和 `max_frames // batch_size`。当前实现只处理完整 batch；`max_frames < batch_size` 会得到 0 batch，数据长度小于一个 batch 也无法产生有效统计。把数值调大前先确认数据集不是空的、episode/frame 转换没有失败。

### 5. 报 `unexpected keyword argument prompt_from_task` 或字段访问错误

对照当前 dataclass：factory 是 `LeRobotLiberoDataConfig`，`prompt_from_task` 属于嵌套的 `DataConfig`。正确写法是 `base_config=DataConfig(prompt_from_task=True)`。配置发现脚本直接读取 `data_factory.base_config.prompt_from_task`，不必实例化模型。

### 6. 检查发现 NaN/Inf、shape 不是 `(8,)`/`(7,)`

先回到第 07 章检查具体 frame 的 state/actions，再核对 repack key 和 `LiberoInputs`。保存完整异常、出错 key、真实 shape、episode/frame 索引与原始值。不要通过删除异常维度、`nan_to_num` 或手改统计 JSON 让断言表面通过。

### 7. action 统计范围很奇怪，是否切换 `extra_delta_transform`？

不要盲切。先对照数据原始动作语义、`DeltaActions` 公式、所选基础 checkpoint 和推理输出逆变换。本章的 `True` 是继承现有旧 π₀ LIBERO 路线的兼容选择；π₀.₅ 现有配置是 `False`。改变它会同时改变 stats 和训练目标，应作为新实验完整重算并验证。

### 8. LoRA 训练却更新了过多参数，或没有 LoRA 参数更新

打印 model 的两个 variant、`freeze_filter` 和 `trainable_filter`，再逐字比较配置块中的两次 `Pi0Config(...)`。不要只比较类名。若 variant 不一致，先修配置并重新开始实验；已有错误 checkpoint 不应继续当作正确 LoRA 结果。

### 9. 写统计时报权限不足或磁盘空间不足

打印 `config.assets_dirs`、`df -h .`、精确目标的父目录权限和已有文件大小。只处理本章专用目标，不要用 sudo 改写整个仓库/cache 的所有权，也不要清空未知 assets。若目标已有价值，先停止并备份或换唯一配置名。

## 本章验收

进入下一章前，提交或展示以下证据：

- [ ] 在专用练习分支中只为 `_CONFIGS` 增加一个 `pi0_libero_tutorial` 项，`git diff --check` 通过；
- [ ] 配置名称唯一，现有 `pi0_libero_low_mem_finetune`、`pi05_libero` 和其他配置未被修改；
- [ ] repo id 精确为 `openpi-tutorial/libero-small`，batch 为 8，steps 为 500；
- [ ] 模型与 freeze filter 都精确使用 `gemma_2b_lora` + `gemma_300m_lora`；
- [ ] 初始 checkpoint 是 `pi0_base/params`，并能说明它不能与 `pi05_base`/`pi05_libero` 混用；
- [ ] 配置发现脚本能输出 factory 字段、模型类型、horizon、variant 和 freeze/trainable 语义，且没有实例化/下载模型；
- [ ] 写入前已打印精确目标，没有静默覆盖有价值的旧统计；
- [ ] `norm_stats.json` 位于 `assets/pi0_libero_tutorial/openpi-tutorial/libero-small/`；
- [ ] stats 恰有 `state`/`actions`，所有统计数组均有限，shape 分别为 `(8,)`/`(7,)`；
- [ ] 能说明统计执行 repack/data transforms，但不执行 Normalize/model transforms；
- [ ] 能说明 `--max-frames 2000` 是速度折中，500 step 是工程演示，二者都不是性能目标；
- [ ] 已记录 config、repo、7D/8D、delta、LoRA、stats 路径和后续 Policy Server 名称的一致性清单。

## 思考题

### 1. 为什么 `pi0_libero_tutorial` 不只是一个显示名称？

<details>
<summary>参考答案（展开查看）</summary>

因为 `get_config()` 用它取回模型、数据 factory、transform、基础权重、冻结规则和训练参数；`assets_dirs` 与 checkpoint 路径也含配置名，后续训练和 Policy Server 还要用同一入口恢复这套契约。名称相同不自动保证内容正确，但改名或用错名会让资产路径和装配一起改变。

</details>

### 2. 为什么 model 和 freeze filter 要分别写两次完全相同的 LoRA variant？

<details>
<summary>参考答案（展开查看）</summary>

`model=` 的 `Pi0Config` 决定真实参数树中哪些模块具有 LoRA 参数；`get_freeze_filter()` 所属的另一个 `Pi0Config` 根据 variant 生成冻结路径。两者不一致时，实际参数树和冻结规则会描述不同模型，可能错误训练基础权重或冻结 adapter。`trainable_filter` 只是对 freeze filter 取反，不会自动修正这种不一致。

</details>

### 3. 为什么计算统计时不能先执行 `Normalize` 和 model transforms？

<details>
<summary>参考答案（展开查看）</summary>

`Normalize` 本身就需要待计算的统计，先执行会形成循环依赖。model transforms 负责 resize、tokenize 和把 state/actions pad 到模型内部维度，它们不代表机器人原始数值协议。统计应在 repack 与机器人/数据语义变换之后、归一化与模型格式化之前得到 8D state 和 7D actions 的真实分布。

</details>

### 4. 为什么 2000 frame 和 500 step 都不能证明模型性能？

<details>
<summary>参考答案（展开查看）</summary>

2000 frame 可能没有覆盖完整任务、episode 和长尾动作分布；500 step 甚至短于默认 1000-step learning-rate warmup。它们足以帮助发现数据加载、统计、梯度、日志和 checkpoint 链路问题，但不能代替覆盖完整分布的统计、充分训练、独立评测和多次实验。

</details>

### 5. 为什么 `pi05_libero` 能跑 LIBERO，也不能直接加载本章 checkpoint？

<details>
<summary>参考答案（展开查看）</summary>

“都用于 LIBERO”只说明任务环境相同，不说明模型结构相同。π₀.₅ 使用不同的模型模式、state/token 处理、horizon、action transform 和基础参数树；本章是 π₀ 的两个 LoRA variant，并从 `pi0_base` 初始化。checkpoint 必须与创建它的模型/config/normalization/action 语义匹配，不能按环境名称互换。

</details>

[← 上一章：准备小规模 LIBERO 数据集](07_prepare_small_dataset.md) · [课程入口：README](README.md) · [下一章：运行教学微调 →](09_run_finetuning.md)
