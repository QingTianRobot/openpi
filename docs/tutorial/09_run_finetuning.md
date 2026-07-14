# 第 09 章：运行短程 LoRA 微调与管理 checkpoint

[← 上一章：配置教学微调](08_configure_finetuning.md) · [课程入口：README](README.md) · [下一章：服务与闭环评测 →](10_serve_and_evaluate.md)

## 本章目标

完成本章后，你应该能够：

- 在创建模型和下载基础权重之前，检查教学配置、本地数据、归一化统计、JAX 设备、全局 batch、磁盘和 checkpoint 目标；
- 解释 `scripts/train.py` 从 mesh、首个 batch、基础权重、JIT `train_step` 到异步 checkpoint 的完整执行顺序；
- 使用 `pi0_libero_tutorial` 启动一次 500-step 的 π₀ LoRA 工程演示，并区分“首次编译慢”“数据等待”“进程退出”“OOM”和“NaN/Inf”；
- 从日志中找到有限的 `loss`、`grad_norm`、`param_norm`，但不把短程 loss 或单次运行误当成模型性能结论；
- 说明为什么当前配置会请求在 step 100、200、300、400 和最终 499 保存，并以实际目录为准检查 manager 的保留结果；
- 检查 checkpoint 的 `train_state`、`params` 和 `assets` 组件，确认 normalization assets 随 checkpoint 保存；
- 在中断后用 `--resume` 恢复最新已提交的训练状态，并准确说明当前源码不会恢复数据迭代器位置；
- 在不删除未知目录、不清空 cache 的前提下处理已有实验、网络、显存、数值和异步保存问题。

遇到陌生缩写可查后续补齐的[课程术语表](glossary.md)；需要跨章节定位故障时，可查后续补齐的[故障排查手册](troubleshooting.md)。

## 前置条件

本章不是“先运行看看”的入口。开始前必须已经完成[第 08 章](08_configure_finetuning.md)，并保存以下真实产物：

1. `src/openpi/training/config.py` 中存在且只存在一个 `pi0_libero_tutorial`；
2. `openpi-tutorial/libero-small` 已经是当前固定 LeRobot 版本可以纯本地读取的数据集；
3. `assets/pi0_libero_tutorial/openpi-tutorial/libero-small/norm_stats.json` 存在，包含有限的 8D `state` 与 7D `actions` 统计；
4. 教学配置的全局 `batch_size=8`、`num_train_steps=500`、`log_interval=10`、`save_interval=100`、`keep_period=100`、`ema_decay=None`、`wandb_enabled=False`；
5. 基础参数路径仍是 `gs://openpi-assets/checkpoints/pi0_base/params`，模型和 freeze filter 仍使用相同的两个 π₀ LoRA variant。

### 训练硬件与空间不是可省略的条件

本课程把 **NVIDIA A100 或 H100** 作为教学训练目标。显存容量、可见卡数、driver、CUDA/JAX 版本、模型 sharding、batch shape 和机器上的其他进程都会影响能否运行。配置名含 `low_mem`、训练采用 LoRA，都只表示相对完整微调减少了一部分梯度和优化器状态；基础权重、未冻结参数、激活和输入 batch 仍占显存，不能据此承诺“任意低显存卡总能放下”。

`batch_size=8` 是每个训练 step 的**全局 batch**，必须能被 `jax.device_count()` 整除。1、2、4、8 张可见设备满足这个条件，3、6、16 张不满足。第 08 章配置默认 `fsdp_devices=1`，多卡主要做数据并行并在每张卡保留完整训练状态；增加卡数不自动降低单卡模型内存。

还要为以下内容预留磁盘：

- `pi0_base` 参数第一次从 GCS 下载到 openpi cache 时的空间；
- JAX compilation cache；
- `checkpoints/pi0_libero_tutorial/libero_lora_demo/` 下的训练状态、推理参数和 assets；
- 若你选择保留旧实验，旧目录的显式备份空间。

本章所有命令默认从仓库根目录、宿主机 Python 3.11 `uv` 环境运行。本文编写环境没有准备课程本地数据、A100/H100 和完整训练依赖，因此**没有实际启动 500-step 重型训练**；后文日志与目录均为按当前源码推导的条件式预期，不是伪造的本机实测数字。

## 核心概念

### 一个训练 step 实际做了什么

对本章配置，每次 `train_step` 都会先用 `state.model_def` 与 `state.params` 合并出 NNX 模型并切到训练模式。模型对 observation 和 actions 计算逐块 loss，代码取均值得到一个标量；`nnx.value_and_grad` 只对 `config.trainable_filter` 命中的状态求梯度。随后优化器计算 updates，`optax.apply_updates` 更新可训练参数，再把新参数写回完整模型状态。

训练状态中的 `step` 每次加 1。`ema_decay=None`，因此本章不会维护第二份 EMA 参数；源码里的 EMA 分支存在，但在这个教学配置中被跳过。每次调用同时返回三个诊断量：

| 指标 | 它回答什么 | 不能单独证明什么 |
| --- | --- | --- |
| `loss` | 当前或当前日志窗口内的训练目标值是否为有限数 | 不能证明 LIBERO 成功率，也不要求逐点单调下降 |
| `grad_norm` | 本次可训练参数梯度的全局范数是否有限、是否明显异常 | 不能仅凭“较小”判断已收敛 |
| `param_norm` | 过滤后的 kernel 参数全局范数是否有限 | 不是 LoRA 参数量统计，也不是性能指标 |

`loss` 会受 batch 抽样、任务、动作和优化器状态影响，短程中上升或波动并不自动表示训练失败。相反，只要任一关键指标出现 `nan` 或 `inf`，就不满足本章验收；应保留现场并检查数据、统计、精度、学习率与梯度，而不是继续把坏 checkpoint 当作成果。

### JIT 为什么让第一次等待特别长

`scripts/train.py` 先通过 shape evaluation 和 sharding 规划初始化训练状态，再加载基础参数并 JIT 初始化；训练循环外又把 `train_step` 包装为 `jax.jit`。第一次真正调用时，JAX 需要针对当前模型、shape、dtype、mesh 和 sharding 编译可执行程序。这个等待可能明显长于后续 step。

在本章没有另设 allocator/preallocate 开关时，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 会让 JAX 默认的预分配 allocator 尝试预留约 90% 的可见 GPU 内存。它不是运行期显存的硬上限，也不会让本来放不下的模型必然运行；在共享 GPU 上还可能很早就激进占用大量空闲显存，挤压其他进程。判断是否 OOM，必须看实际 allocator 设置、进程、`nvidia-smi` 和错误日志。

### checkpoint 的 step 名与训练状态 step 不完全相同

训练循环使用零基索引：

```text
for step in range(start_step, num_train_steps):
    训练并把 train_state.step 加 1
    按需记录日志
    batch = next(data_iter)
    再判断是否以循环变量 step 保存
```

全新运行时 `start_step=0`，循环变量是 0 到 499。保存条件是：

```text
(step % save_interval == 0 and step > start_step)
or step == num_train_steps - 1
```

因此在 `num_train_steps=500`、`save_interval=100` 时，**只要每一步后面的 `next(data_iter)` 成功返回**，代码会请求保存：

```text
100, 200, 300, 400, 499
```

不会在 0 保存；最终目录名是 `499`，其中训练状态的内部 `step` 已经是 500。需要注意，源码甚至会在最终 step 更新完成后先取一个不会再用于训练的额外 batch，再进入 499 的保存判断。若数据 worker、I/O 或 transform 在这次额外获取中阻塞或报错，最终 499 根本不会提交给 checkpoint manager。

`CheckpointManager` 配置了 `max_to_keep=1` 和 `keep_period=100`，意图让命中保留周期的 checkpoint 不被普通清理，同时保留最新结果。不过 Orbax 版本、成功提交历史和异常中断都会影响最终可见目录；课程验收应列出**实际 manager 已提交的目录**，不要只凭配置宣称所有五个目录一定存在。

保存是异步的。训练循环发出 `save()` 后可以继续，结束时会打印 `Waiting for checkpoint manager to finish` 并调用 `wait_until_finished()`。看到进度条结束不等于最后一次写盘已经完成；应等进程正常退出，再检查 step 目录和时间戳。

### `HF_HUB_OFFLINE=1` 与 GCS 权重下载是两条链路

本地教学数据使用形如 Hub repo id 的 `openpi-tutorial/libero-small`。如果当前固定的 Hugging Face/LeRobot 版本与本地 cache 布局兼容，把 `HF_HUB_OFFLINE=1` 设置在 Python 进程启动前，可以防止 repo id 拼错或本地内容残缺时悄悄回退到 Hugging Face Hub。

这个变量只约束 Hugging Face Hub 客户端，不会阻止 openpi 的 downloader 访问：

```text
gs://openpi-assets/checkpoints/pi0_base/params
```

所以第一次训练仍可能下载基础 checkpoint。若机器需要纯离线训练，必须提前确认 GCS 参数已经存在于 `OPENPI_DATA_HOME` 指向的 cache；不要误以为 `HF_HUB_OFFLINE=1` 会覆盖 GCS。

## 源码地图

| 文件 | 本章阅读入口 | 作用 |
| --- | --- | --- |
| [`scripts/train.py`](../../scripts/train.py) | `main()` | 检查 batch、创建 mesh/checkpoint/data loader、取首个 batch、初始化状态、JIT step、记录日志和保存 |
| [`scripts/train.py`](../../scripts/train.py) | `init_train_state()` | 创建模型 shape、规划 sharding、加载/合并基础参数、建立优化器状态和可选 EMA |
| [`scripts/train.py`](../../scripts/train.py) | `train_step()` | 合并模型、计算 loss/梯度、更新可训练状态并产生三个指标 |
| [`src/openpi/training/config.py`](../../src/openpi/training/config.py) | `TrainConfig`、`cli()` | 定义全局 batch、step/日志/保存间隔、checkpoint 路径及 `--resume`/`--overwrite` 互斥 |
| [`src/openpi/training/data_loader.py`](../../src/openpi/training/data_loader.py) | `create_data_loader()` | 由 repo id 创建数据集，应用 transforms 与 normalization，产生全局训练 batch |
| [`src/openpi/training/checkpoints.py`](../../src/openpi/training/checkpoints.py) | `initialize_checkpoint_dir()` | 处理新目录、覆盖、恢复和 Orbax retention 选项 |
| [`src/openpi/training/checkpoints.py`](../../src/openpi/training/checkpoints.py) | `save_state()`、`restore_state()` | 异步保存/恢复训练状态和推理参数，并把 normalization statistics 写入 assets |
| [`src/openpi/training/weight_loaders.py`](../../src/openpi/training/weight_loaders.py) | `CheckpointWeightLoader` | 从本地或 `gs://` 参数目录加载基础权重，并补入新 LoRA 参数 |

### 按源码顺序阅读一次完整启动

`main()` 的实际顺序很重要，因为它能帮助判断程序卡在哪一层：

1. 初始化日志，检查 global batch 能否被可见 JAX 设备数整除；
2. 设置 JAX compilation cache，拆分随机数，建立 mesh、data sharding 与 replicated sharding；
3. 初始化 checkpoint manager；本地 W&B 被禁用，但仍走 disabled mode；
4. 创建 data loader，立刻取出第一个 batch，并打印 observation/actions 的数组树信息；
5. 从首个 batch 的多路相机图像拼接最多 5 张 sanity-check 图，调用 disabled W&B logger；
6. 通过 `init_train_state()` 规划状态 shape/sharding；新训练会创建模型、加载 `pi0_base`、合并 LoRA 初始参数并建立 trainable optimizer state；
7. `jax.block_until_ready()` 等初始化完成，再打印参数树信息；恢复训练则在此后从 manager 恢复最新状态；
8. 用当前 sharding JIT 包装 `train_step`，进入零基训练循环；
9. 每步合并模型、计算 loss/grads、更新 trainable state；本章 EMA 关闭；
10. 每 10 个循环 step 汇总并打印 `loss`、`grad_norm`、`param_norm`，同时写入 disabled W&B；
11. 按上文的保存时序推进下一 batch 与异步 checkpoint；循环结束后等待所有保存完成。

这也解释了为什么“还没看到 Step 0”不只可能是 JIT：首个 batch 和基础权重初始化都发生在训练指标之前，应结合最后一条日志、GPU/CPU/磁盘活动判断。

## 动手实践

### 实践 1：只发现配置，不创建模型

先确认第 08 章的配置真的在当前 checkout 中。下面脚本只读取注册表与 dataclass 字段，不调用 `config.model.create(...)`，也不调用 `weight_loader.load(...)`：

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python - <<'PY'
import dataclasses

from openpi.training import config as _config

config = _config.get_config("pi0_libero_tutorial")
run_config = dataclasses.replace(config, exp_name="libero_lora_demo")

print("config:", config.name)
print("repo id:", config.data.repo_id)
print("model type:", config.model.model_type)
print("variants:", config.model.paligemma_variant, config.model.action_expert_variant)
print("global batch:", config.batch_size)
print("steps/log/save/keep:", config.num_train_steps, config.log_interval, config.save_interval, config.keep_period)
print("EMA:", config.ema_decay)
print("W&B enabled:", config.wandb_enabled)
print("FSDP devices:", config.fsdp_devices)
print("weight loader:", config.weight_loader)
print("checkpoint target:", run_config.checkpoint_dir)

assert config.name == "pi0_libero_tutorial"
assert config.data.repo_id == "openpi-tutorial/libero-small"
assert config.model.paligemma_variant == "gemma_2b_lora"
assert config.model.action_expert_variant == "gemma_300m_lora"
assert config.batch_size == 8
assert config.fsdp_devices == 1
assert config.num_train_steps == 500
assert (config.log_interval, config.save_interval, config.keep_period) == (10, 100, 100)
assert config.ema_decay is None
assert config.wandb_enabled is False
assert str(config.weight_loader.params_path) == "gs://openpi-assets/checkpoints/pi0_base/params"
PY
```

若这里触发模型分配或权重下载，说明你运行的不是这段脚本，或本地配置代码引入了额外副作用；先检查 diff，不要继续训练。

### 实践 2：纯本地检查数据和有限统计

训练 loader 按 repo id 查找数据。先打印当前固定 LeRobot 版本实际使用的根目录，并显式传入 `root` 读取 metadata，避免预检阶段回退到 Hub：

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python - <<'PY'
import inspect
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from openpi.shared import normalize
from openpi.training import config as _config

repo_id = "openpi-tutorial/libero-small"
root = Path(HF_LEROBOT_HOME) / repo_id
print("local dataset:", root)
assert root.is_dir(), root

signature = inspect.signature(LeRobotDatasetMetadata)
if "root" not in signature.parameters:
    raise RuntimeError(
        f"Pinned LeRobotDatasetMetadata has no root parameter: {signature}; "
        "adapt the local-only check instead of allowing Hub fallback."
    )
meta = LeRobotDatasetMetadata(repo_id, root=root)
print("fps:", meta.fps)
print("tasks:", len(meta.tasks))
assert meta.tasks

config = _config.get_config("pi0_libero_tutorial")
stats_dir = config.assets_dirs / repo_id
stats_file = stats_dir / "norm_stats.json"
print("norm stats:", stats_file)
assert stats_file.is_file(), stats_file

stats = normalize.load(stats_dir)
assert set(stats) == {"state", "actions"}, sorted(stats)
for key, expected_shape in {"state": (8,), "actions": (7,)}.items():
    for field in ("mean", "std", "q01", "q99"):
        value = np.asarray(getattr(stats[key], field))
        print(key, field, value.shape, "finite=", bool(np.isfinite(value).all()))
        assert value.shape == expected_shape, (key, field, value.shape)
        assert np.isfinite(value).all(), (key, field)
PY
```

如果离线模式失败而联网模式成功，不要立刻关闭离线检查继续训练。这通常表示本地 root、metadata、视频/data 文件或 cache 契约不完整；回到第 07 章检查精确路径。

### 实践 3：检查 JAX 看见的 GPU 与 batch 整除关系

```bash
uv run --no-sync python - <<'PY'
import jax

from openpi.training import config as _config

config = _config.get_config("pi0_libero_tutorial")
devices = jax.devices()
print("device count:", len(devices))
for index, device in enumerate(devices):
    print(index, "platform=", device.platform, "kind=", device.device_kind)

assert devices, "JAX did not expose any device"
assert all(device.platform == "gpu" for device in devices), devices
assert config.fsdp_devices == 1, f"expected tutorial fsdp_devices=1, got {config.fsdp_devices}"
assert config.batch_size % len(devices) == 0, (
    f"global batch {config.batch_size} is not divisible by {len(devices)} visible devices"
)
print("global batch per visible device:", config.batch_size // len(devices))
PY
```

人工确认 `device_kind` 是本课程目标的 A100/H100。若 JAX 只看到 CPU、看见的卡数与 `nvidia-smi` 不一致，或设备数不能整除 8，先修正可见设备与环境。不要把训练命令启动后才出现的 batch 错误当作显存错误。

### 实践 4：检查磁盘、父目录和唯一目标

本章实验名固定为 `libero_lora_demo`，默认目标是：

```text
checkpoints/pi0_libero_tutorial/libero_lora_demo/
```

先检查仓库所在文件系统，再只创建目标的父目录：

```bash
set -euo pipefail

CHECKPOINT_DIR=checkpoints/pi0_libero_tutorial/libero_lora_demo
CHECKPOINT_PARENT=$(dirname "$CHECKPOINT_DIR")

df -h .
mkdir -p "$CHECKPOINT_PARENT"
if ! test -w "$CHECKPOINT_PARENT"; then
  echo "CHECKPOINT PARENT IS NOT WRITABLE: $CHECKPOINT_PARENT" >&2
  exit 1
fi

if test -e "$CHECKPOINT_DIR"; then
  echo "REFUSING TO OVERWRITE EXISTING EXPERIMENT: $CHECKPOINT_DIR" >&2
  find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 2 -print | sort
  exit 1
fi

echo "preflight passed: target does not exist: $CHECKPOINT_DIR"
```

目标存在时命令以非零状态结束。此时只做以下三种明确决定之一：

- 为新实验换一个尚不存在的 `--exp-name`；
- 如果它就是被中断的同一次训练，先检查最新 committed step，再按“实践 9”使用 `--resume`；
- 如果必须保留旧实验并重用名称，为这个**精确目录**选择不存在的备份名，先检查再 `mv`，之后重新运行 guard。

一个不删除数据的备份模板是：

```bash
set -euo pipefail

CHECKPOINT_DIR=checkpoints/pi0_libero_tutorial/libero_lora_demo
BACKUP_DIR=checkpoints/pi0_libero_tutorial/libero_lora_demo.backup-20260715

if ! test -e "$CHECKPOINT_DIR"; then
  echo "SOURCE EXPERIMENT DOES NOT EXIST: $CHECKPOINT_DIR" >&2
  exit 1
fi
if test -e "$BACKUP_DIR"; then
  echo "BACKUP TARGET ALREADY EXISTS: $BACKUP_DIR" >&2
  exit 1
fi

mv -- "$CHECKPOINT_DIR" "$BACKUP_DIR"
echo "backup created: $BACKUP_DIR"
```

日期只是示例，必须换成你的唯一名称。不要执行 `rm -rf checkpoints`，也不要为了重跑而清空整个 openpi/Hugging Face/JAX cache。

### 实践 5：用无覆盖模式启动教学训练

当前 `initialize_checkpoint_dir()` 在目标不存在时会自行创建目录；若目录已存在而没有显式 flag，它会抛出 `FileExistsError`。因此首次运行的推荐主线是：先由 guard 拒绝已有目录，再**不带 `--overwrite`** 启动。即使另一个进程在 guard 与 manager 初始化之间抢先创建同名目录，manager 仍会失败退出，而不是删除对方的结果。

```bash
set -euo pipefail

CHECKPOINT_DIR=checkpoints/pi0_libero_tutorial/libero_lora_demo
if test -e "$CHECKPOINT_DIR"; then
  echo "REFUSING TO OVERWRITE EXISTING EXPERIMENT: $CHECKPOINT_DIR" >&2
  exit 1
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 HF_HUB_OFFLINE=1 uv run --no-sync scripts/train.py pi0_libero_tutorial --exp-name libero_lora_demo
```

首次运行没有技术理由添加 `--overwrite`。这条无覆盖命令也是后文所有新实验的默认模板；若要另存日志，只换一个不存在的新 `--exp-name`。

只有为了兼容已有外部运行方案、并且你已经按实践 4 把旧实验完整备份到另一个明确路径后，才可能看到下面的高风险兼容写法：

```bash
set -euo pipefail

CHECKPOINT_DIR=checkpoints/pi0_libero_tutorial/libero_lora_demo
if test -e "$CHECKPOINT_DIR"; then
  echo "TARGET STILL EXISTS AFTER BACKUP; REFUSING RISKY OVERWRITE: $CHECKPOINT_DIR" >&2
  exit 1
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 HF_HUB_OFFLINE=1 uv run --no-sync scripts/train.py pi0_libero_tutorial --exp-name libero_lora_demo --overwrite
```

它仍然存在 guard 与 manager 初始化之间的竞态：若别的进程恰好创建同名目标，`overwrite=True` 会清空它。因此课程不推荐这条命令，也不能把“已经备份”理解成竞态保护。恢复已有训练只用 `--resume`，绝不带 `--overwrite`。

第一次运行可能先从 `gs://openpi-assets/checkpoints/pi0_base/params` 下载基础参数；网络、GCS 工具与 `OPENPI_DATA_HOME` cache 权限都会影响这一阶段。权重到位后，第一次模型初始化和第一次 JIT 仍可能较慢。不要仅凭几分钟没有新的 `Step ...` 行就杀进程；先按下一实践收集证据。

### 实践 6：在另一个终端监控而不干扰训练

首先看 GPU 与训练进程：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv -l 5
```

另一个终端可以运行一次只读进程查询：

```bash
pgrep -af 'scripts/train.py pi0_libero_tutorial'
```

再结合训练终端的最后一条日志判断：

| 现象 | 更可能处于什么阶段 | 下一条证据 |
| --- | --- | --- |
| 进程存在，尚未打印 `Initialized data loader`，GPU 利用率低 | 本地数据解析、首 batch、worker 或 I/O 等待 | 看 `data_config`/`local_batch_size` 后是否还有异常；检查 CPU、磁盘和 dataset root |
| 已打印 data loader/tree，正在下载或初始化参数 | GCS 基础权重或训练状态初始化 | 看 downloader 日志、网络、cache 空间与权限 |
| 进程存在，GPU 已占用较多显存，首个 `Step` 尚未出现 | 模型初始化或首次 JIT 编译 | 等待并观察 GPU/CPU 是否仍有活动，不凭墙钟猜死锁 |
| `pgrep` 无结果 | 进程已经退出，不是“仍在编译” | 回到训练终端看退出码与最后 traceback |
| 日志出现 `RESOURCE_EXHAUSTED`、`CUDA_ERROR_OUT_OF_MEMORY` 或明确 OOM | 显存不足 | 记录设备、显存峰值、batch/sharding 和其他进程；不要把 fraction 当保证 |
| 已有 step 日志，但指标出现 `nan`/`inf` | 数值或数据问题 | 停止把后续 checkpoint 当有效成果，保存首个异常 step 与前一窗口指标 |

如果希望保留完整终端日志，可以在下一次**新实验名**中显式启用 shell 的 pipeline 失败传播并用 `tee`；不要为了加日志而覆盖当前实验：

```bash
set -euo pipefail

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 HF_HUB_OFFLINE=1 \
  uv run --no-sync scripts/train.py pi0_libero_tutorial \
  --exp-name libero_lora_demo_v2 2>&1 | tee /tmp/libero_lora_demo_v2.log
```

### 实践 7：阅读条件式预期日志

若配置、数据、设备、基础权重和 JIT 都成功，日志顺序应大致包含：

1. 主机信息、`data_config` 与 local batch；
2. `Initialized data loader:` 后的 observation/actions 数组树；
3. 训练状态 sharding/tree 信息；
4. `Initialized train state:` 后的参数树；
5. `Step 0:`，随后按配置每 10 个循环 step 打印一次窗口指标；
6. 结束时 `Waiting for checkpoint manager to finish`，然后进程正常退出。

每条 step 行的字段名来自源码，应包含：

```text
loss=...
grad_norm=...
param_norm=...
```

本教程不提供虚构数值或“合格 loss 阈值”。你的证据应记录至少一条实际日志，并验证三个字段都是有限数。`Step 10` 等后续行是窗口均值；loss 不必单调下降。500 step 甚至仍处于第 08 章所述默认 learning-rate warmup 范围内，它只验证数据、梯度、更新、日志和保存链路，不是正式收敛训练，也没有官方 LIBERO 成功率要求。

如果保存了日志，可用只读命令定位指标和异常：

```bash
set -euo pipefail

LOG=/tmp/libero_lora_demo_v2.log
if ! test -f "$LOG"; then
  echo "LOG FILE DOES NOT EXIST: $LOG" >&2
  exit 1
fi
if ! rg -n 'Initialized data loader|Initialized train state|Step [0-9]+:|Waiting for checkpoint|nan|inf|OOM|RESOURCE_EXHAUSTED' "$LOG"; then
  echo "NO EXPECTED TRAINING EVIDENCE FOUND IN: $LOG" >&2
  exit 1
fi
```

最终仍要人工或用解析脚本确认数值有限；仅仅匹配到字符串 `loss=` 不够。

### 实践 8：安全检查实际 checkpoint 目录与 assets

训练期间或正常退出后，按目录 basename 过滤出纯数字 step，再做数字排序，不假定 retention 结果。训练仍在进行时，这些只是写盘进度证据；只有进程完成 `wait_until_finished()` 并正常退出后，才把组件检查当作验收：

```bash
set -euo pipefail

CHECKPOINT_ROOT=checkpoints/pi0_libero_tutorial/libero_lora_demo
if ! test -d "$CHECKPOINT_ROOT"; then
  echo "CHECKPOINT ROOT DOES NOT EXIST: $CHECKPOINT_ROOT" >&2
  exit 1
fi

NUMERIC_STEPS=$(
  find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | awk '/^[0-9]+$/' | sort -n
)
if test -z "$NUMERIC_STEPS"; then
  echo "NO NUMERIC CHECKPOINT STEP FOUND UNDER: $CHECKPOINT_ROOT" >&2
  exit 1
fi

echo "numeric step directories:"
printf '%s\n' "$NUMERIC_STEPS"
while IFS= read -r step; do
  stat -c '%y %n' "$CHECKPOINT_ROOT/$step"
done <<< "$NUMERIC_STEPS"

LATEST_STEP=$(printf '%s\n' "$NUMERIC_STEPS" | tail -n 1)
STEP_DIR="$CHECKPOINT_ROOT/$LATEST_STEP"

echo "latest visible numeric step directory: $STEP_DIR"
for component in train_state params assets; do
  if ! test -e "$STEP_DIR/$component"; then
    echo "MISSING CHECKPOINT COMPONENT: $STEP_DIR/$component" >&2
    exit 1
  fi
done

CHECKPOINT_NORM_STATS="$STEP_DIR/assets/openpi-tutorial/libero-small/norm_stats.json"
if ! test -f "$CHECKPOINT_NORM_STATS"; then
  echo "MISSING CHECKPOINT NORM STATS: $CHECKPOINT_NORM_STATS" >&2
  exit 1
fi

find "$STEP_DIR" -mindepth 1 -maxdepth 2 -print | sort
```

这段命令只把 basename 完全匹配 `^[0-9]+$` 的一级目录放进 `NUMERIC_STEPS`；临时目录、日志文件和其他名称不会参与 `sort -n` 或 latest-step 选择。`set -euo pipefail` 让 `find`、`awk`、`sort`、`stat` 或组件检查中的失败立即传播，而空结果由显式 guard 退出 1。不要猜 Orbax 在三个组件内部使用的文件名。

三个 item 的职责是：

- `train_state`：优化器状态、内部 step，以及恢复训练所需的非推理状态；
- `params`：从完整状态拆出的推理参数，后续 `CheckpointWeightLoader`/Policy Server 从 `<step>/params` 加载；本章 EMA 关闭，所以这里保存当前参数而不是 EMA 副本；
- `assets`：`save_state()` 从 data config 取得 normalization statistics，并通过 `asset_id` 写到 `assets/openpi-tutorial/libero-small/norm_stats.json`。

不要把 `wandb_id.txt`、Orbax metadata 或组件内部文件名写成跨版本固定契约。本课程只依赖当前 manager 注册的 `train_state`、`params`、`assets` item，以及 policy loader 使用的 `<step>/params` 和对应 asset id。单靠 `find` 看见数字目录并不能在异步写入期间证明它已经提交；正常退出后的 wait、完整组件和后续 manager 恢复共同构成证据。

若 500 step 完整正常结束，最大的数字目录应为 `499`；若只运行到中途，则至少一个已提交 checkpoint 也可作为“保存链路工作”的证据，但不等于完成全程。只有进程在 `wait_until_finished()` 后正常退出，最终保存才满足完整运行验收。

### 实践 9：中断后恢复，而不是覆盖

先重新运行实践 2 与实践 8，再用下面的 fail-fast guard 和相同 config/exp name 恢复：

```bash
set -euo pipefail

DATASET_ROOT=$(
  HF_HUB_OFFLINE=1 uv run --no-sync python - <<'PY'
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

print(Path(HF_LEROBOT_HOME) / "openpi-tutorial/libero-small")
PY
)
ORIGINAL_NORM_STATS=assets/pi0_libero_tutorial/openpi-tutorial/libero-small/norm_stats.json
CHECKPOINT_ROOT=checkpoints/pi0_libero_tutorial/libero_lora_demo

if ! test -d "$DATASET_ROOT"; then
  echo "ORIGINAL LOCAL DATASET DOES NOT EXIST: $DATASET_ROOT" >&2
  exit 1
fi
if ! test -f "$ORIGINAL_NORM_STATS"; then
  echo "ORIGINAL NORM STATS DO NOT EXIST: $ORIGINAL_NORM_STATS" >&2
  exit 1
fi
if ! test -d "$CHECKPOINT_ROOT"; then
  echo "CHECKPOINT ROOT DOES NOT EXIST: $CHECKPOINT_ROOT" >&2
  exit 1
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 HF_HUB_OFFLINE=1 uv run --no-sync scripts/train.py pi0_libero_tutorial --exp-name libero_lora_demo --resume
```

不要添加 `--overwrite`。`TrainConfig.__post_init__()` 明确禁止同时 resume 和 overwrite；语义上二者也相反：resume 要保留已有状态，overwrite 会清空目录。

恢复时，checkpoint manager 默认选择最新已提交 step，`restore_state()` 合并 `train_state` 与 `params`。例如目录名是 `400` 时，保存内容中的内部 state step 已是 401，循环会从 401 继续。若最终 `499` 已保存，内部 state step 是 500，而配置仍是 `num_train_steps=500`，`range(500, 500)` 为空，程序不会再执行训练 step；只有你明确修改配置增加总步数后，才会继续。这属于新实验设计决定，应记录 config diff，不要暗中把 500 改大。

checkpoint 内的 `assets` 不会在 resume 前自动还原到 `config.assets_dirs`。源码先用原始路径创建 loader、加载 stats 并取首 batch，之后才调用 `restore_state()`；所以仅保留 checkpoint 目录、却移动了本地 dataset 或原始 stats，恢复仍会在 restore 前失败。

另一个限制是：`save_state()` 没有把 data loader/iterator 位置注册为 checkpoint item，`restore_state()` 还会显式 `del data_loader`。因此 resume 会恢复最新模型、优化器和 step，**不会恢复到中断前的精确数据 batch 位置**；新建的 shuffled loader 会从自己的初始迭代流程重新开始。不要把函数参数中出现 `data_loader` 误读成“已保存数据加载器状态”。

如果实验目录存在但从未成功提交第一个 checkpoint，manager 会记录“does not contain any checkpoints”并取消实际恢复；此时会重新初始化训练状态。先看日志与目录，再决定是保留现场、换新实验名还是明确重新开始，不要手工把半写入目录冒充可恢复 checkpoint。

## 预期结果

因为本文环境没有运行重型训练，以下只有在你的 A100/H100、数据、统计、依赖、网络/cache 与磁盘条件都满足时才成立：

- 配置发现、数据/统计、GPU/batch 和目标 guard 全部退出 0；
- 训练先打印 data loader/tree 和 train state 信息，随后按 10-step 间隔出现实际的 `loss`、`grad_norm`、`param_norm`；
- 至少一组日志指标全部有限，没有 `nan`/`inf`；loss 可以波动，不要求单调下降；
- 保存条件依次请求 100、200、300、400、499；完整正常运行会等待异步 manager，且最新有效 step 为 499；
- `checkpoints/pi0_libero_tutorial/libero_lora_demo/` 下至少有一个 manager 已提交的数字 step；
- 最新有效 step 下存在 `train_state`、`params`、`assets`，且 assets 中包含本配置 asset id 对应的 `norm_stats.json`；
- `<step>/params` 与匹配的 `pi0_libero_tutorial`/assets 已具备第 10 章进行实际 reload 验证的输入条件；
- 没有覆盖任何无关实验，也没有以删除整个 cache 或 checkpoint 根目录作为修复手段。

## 常见问题

### 1. 找不到 `pi0_libero_tutorial`

先运行 `rg -n 'name="pi0_libero_tutorial"' src/openpi/training/config.py` 和实践 1。没有结果说明第 08 章改动不在当前 checkout；多于一个定义会触发名称唯一性问题。再看 `git branch --show-current`、`git status --short` 和配置 diff，不要先改训练脚本绕过注册表。

### 2. 本地数据或 normalization statistics 找不到

保留 `HF_HUB_OFFLINE=1`，打印 `HF_LEROBOT_HOME`、精确 repo root、`config.assets_dirs` 和 stats 文件。数据缺失回第 07 章，stats 缺失或非有限回第 08 章。若错误发生在 `--resume`，checkpoint assets 也不能替代这些原始路径，按实践 9 重新预检。不要联网后让同名远端 repo 掩盖本地路径错误，也不要把别的配置的 stats 复制过来凑路径。

### 3. 离线数据检查失败，但我明明看到目录

目录存在不代表 pinned LeRobot 所需 metadata、data、video 和索引完整，也不代表构造器按同一 cache root 查找。使用实践 2 的 `root=root` metadata 检查；若 API 签名不同，检查仓库锁定的 LeRobot commit 和实际签名后调整本地检查，不要省略 root 后静默访问 Hub。

### 4. 基础 checkpoint 下载失败

确认失败 URL 是 `gs://openpi-assets/checkpoints/pi0_base/params`，再检查 DNS/代理、`gsutil` 或 gcsfs 路径、`OPENPI_DATA_HOME`、磁盘和 cache 权限。`HF_HUB_OFFLINE=1` 不是 GCS 离线开关。不要把 `pi05_base` 或陌生本地参数替换进去；模型结构不匹配会把网络问题变成参数树问题。

### 5. `Batch size 8 must be divisible...`

记录 `jax.devices()`、`jax.device_count()`、`CUDA_VISIBLE_DEVICES` 和 global batch。调整可见设备到能整除 8 的 A100/H100 组合，或回到配置设计阶段创建清楚记录的新 batch/FSDP 实验。不要临时把 batch 当成“每卡 batch”随手相乘。

### 6. LoRA 训练仍然 OOM

先保存 GPU 型号/容量、可见设备数、峰值、其他进程、global batch、`fsdp_devices`、allocator 设置、最后日志和完整 OOM。在默认预分配 allocator 下，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 会尝试预留约 90% 可见显存，但它不是硬上限，还可能在共享 GPU 上激进占用空闲显存；默认 `fsdp_devices=1` 也不会把模型自动切到多卡。需要改 batch、FSDP、精度或硬件时，把它作为新实验配置并使用新 exp name，不要在同一 checkpoint 目录暗中改变契约。

### 7. `loss`、`grad_norm` 或 `param_norm` 出现 NaN/Inf

记录首个异常 step、前一个有限窗口、原始数据有限性、stats 的 mean/std/quantile、学习率配置和梯度日志。停止把后续 checkpoint 当作有效结果；先用第 07、08 章脚本验证输入与 stats。不要通过过滤日志、重命名 checkpoint 或只看 loss 某一项掩盖异常。

### 8. 很久没有 step 日志

按实践 6 区分阶段：首 batch 之前看数据/worker/I/O；data loader 之后看 GCS 下载和模型初始化；GPU 已分配且进程存活时考虑首次 JIT；`pgrep` 无结果则检查退出码。只有明确 traceback 才能判断异常类型。不要仅靠等待时间断言“编译卡死”。

### 9. checkpoint 写入时报权限不足或磁盘满

运行 `df -h .`、`stat` 目标父目录和实践 8 的时间戳检查，确认问题只影响本章实验路径。释放空间时先识别自己拥有且可重建的具体产物，或把新实验定向到有空间的 `checkpoint_base_dir`。不要 sudo 改写整个仓库/cache 所有权，也不要删除未知实验。

### 10. 已有目录导致 `FileExistsError`

这是保护行为。列出目录、数字 step 和时间戳，然后选择新 exp name、明确 `--resume` 或显式备份。不要因为“只是教学”就补上无 guard 的 `--overwrite`。

### 11. 同时传 `--resume --overwrite` 报错

删除其中一个 flag 前先确定真实意图。继续同一次实验只保留 `--resume`；开始新实验应使用不存在的新 exp name。二者同时出现由配置源码主动拒绝，不应修改 `__post_init__()` 绕过。

### 12. 接近结束却没有最终 499，或在异步保存期间中断

先按上文保存时序检查 traceback：停在最终 `next(data_iter)` 就排查 dataset/worker/I/O/transform；已进入 save/wait 后中断才按异步提交处理。保留目录现场，用 manager 可见的数字 step、组件和时间戳判断最新 committed checkpoint；半写入内容不应手工改名冒充有效 step。若一个 checkpoint 都未提交，源码会取消实际 resume 并重新初始化。

## 本章验收

进入第 10 章前，提交或展示以下证据：

- [ ] 当前 checkout 中唯一存在 `pi0_libero_tutorial`，并且实践 1 没有创建模型或下载权重；
- [ ] 本地 `openpi-tutorial/libero-small` 能在 `HF_HUB_OFFLINE=1` 下读取 metadata；
- [ ] 8D state/7D actions 的 mean/std/q01/q99 shape 正确且全部有限；
- [ ] 两个 variant 精确为 `gemma_2b_lora`/`gemma_300m_lora`，`fsdp_devices=1`；`jax.devices()` 显示课程目标 A100/H100，global batch 8 能被可见设备数整除；
- [ ] 已记录磁盘空间和精确 checkpoint 目标，目标存在时 guard 非零退出；
- [ ] 首次运行使用不存在的新实验目录且不带 `--overwrite`；没有覆盖、删除或混用无关实验；
- [ ] 实际日志包含至少一组有限的 `loss`、`grad_norm`、`param_norm`；没有用虚构数字或单调下降作为验收；
- [ ] 至少一个 manager checkpoint 已提交；若完整运行 500 step，进程在异步 wait 后正常退出且最新数字 step 为 499；
- [ ] 最新有效 step 中存在 `train_state`、`params` 和 `assets/openpi-tutorial/libero-small/norm_stats.json`；
- [ ] 能说明保存请求是 100、200、300、400、499，但 retention 结果以实际 manager 目录为准；
- [ ] `<step>/params`、匹配的配置名与 normalization assets 已准备好供第 10 章实际 reload；
- [ ] 明确知道本章不要求官方 loss 阈值或 LIBERO success-rate，500 step 只验收工程闭环。

## 思考题

### 1. 为什么本章日志里的 loss 不要求每 10 step 都下降？

<details>
<summary>参考答案（展开查看）</summary>

每个日志窗口包含不同的 shuffled batch、任务和动作，随机抽样会造成波动；优化器更新也不保证每个局部窗口都下降。500 step 仍是短程工程演示。验收关注指标为有限数、训练确实更新并能保存，而性能趋势需要更长训练、多次实验和独立评测。

</details>

### 2. 为什么完整 500-step 运行的最终目录叫 `499`？

<details>
<summary>参考答案（展开查看）</summary>

Python 循环变量从 0 到 499，checkpoint manager 用这个循环变量命名目录；每次 `train_step` 又把状态内部 step 加 1。因此完整正常运行的最终目录是 499，内部训练状态已经完成 500 次更新。恢复它时 `start_step=500`，若总步数仍为 500，循环为空。

</details>

### 3. `HF_HUB_OFFLINE=1` 为什么不能保证训练完全不联网？

<details>
<summary>参考答案（展开查看）</summary>

它控制 Hugging Face Hub 客户端，用于防止本地 LeRobot repo id 回退到 Hub。基础参数由 openpi downloader 从 `gs://openpi-assets/...` 获取，属于 GCS 链路，不受这个变量控制。完全离线前还要确认基础参数已在 `OPENPI_DATA_HOME` cache 中。

</details>

### 4. 为什么 `--resume` 不能与 `--overwrite` 一起使用？

<details>
<summary>参考答案（展开查看）</summary>

resume 的前提是保留已有 checkpoint 并恢复最新状态；overwrite 会清空同一实验目录，两者语义冲突，`TrainConfig.__post_init__()` 也会主动报错。恢复使用 `--resume`；新实验使用不存在的新 exp name，并先通过目标 guard。

</details>

### 5. 为什么看到 `save_state(..., data_loader, ...)` 仍不能说数据迭代位置会恢复？

<details>
<summary>参考答案（展开查看）</summary>

当前 manager item 只有 `assets`、`train_state`、`params`，没有 data-loader iterator item；`restore_state()` 还显式丢弃 data loader 参数。因此 resume 会恢复模型、优化器和 step，却不会恢复精确 iterator/batch 位置。

</details>

[← 上一章：配置教学微调](08_configure_finetuning.md) · [课程入口：README](README.md) · [下一章：服务与闭环评测 →](10_serve_and_evaluate.md)
