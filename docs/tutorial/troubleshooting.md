# openpi 课程故障排查手册

[课程入口：README](README.md) · [课程术语表](glossary.md) · [环境安装](02_environment_setup.md) · [数据与变换](05_data_and_transforms.md) · [训练](09_run_finetuning.md) · [服务与评测](10_serve_and_evaluate.md)

本手册按数据流分层排错。每个问题都先收集证据，再做范围最小、可回滚的处理，最后回到对应章节验收。除特别说明外，命令从仓库根目录运行；Python 命令统一使用 `uv run --no-sync`，避免排错过程中隐式同步环境。

本文编写环境没有运行依赖下载、Docker 镜像构建、Docker GPU smoke test、模型下载、完整训练、Policy Server 或 LIBERO 闭环等重型命令。下面的命令是按当前仓库和教程验证过的诊断路径，不是本机重型运行记录。

## 快速症状索引

| 症状 | 先看分类 |
| --- | --- |
| `ModuleNotFoundError`、Python 版本错误、子模块缺失 | [1. 安装](#1-安装) |
| JAX 只看到 CPU、CUDA/XLA 报错、OOM | [2. GPU/JAX](#2-gpujax) |
| repo not found、离线读取失败、episode/frame 异常 | [3. 数据](#3-数据) |
| `KeyError`、shape/dtype 错、动作方向或尺度错误 | [4. Transform](#4-transform) |
| stats 找不到、维度不符、NaN/Inf、覆盖警告 | [5. Normalization](#5-normalization) |
| 很久没有 step、指标 NaN/Inf、batch/FSDP 错 | [6. 训练](#6-训练) |
| checkpoint 不完整、无法加载、resume 失败 | [7. Checkpoint](#7-checkpoint) |
| 端口占用、连接拒绝、服务端/客户端配置不一致 | [8. 服务与网络](#8-服务与网络) |
| EGL/GLX、MuJoCo、action horizon、视频缺失 | [9. LIBERO 渲染与评测](#9-libero-渲染与评测) |

## 通用排错与升级规则

先确定最后一个成功层级：安装 → JAX/GPU → 原始数据 → repack → data transform → normalization/model transform → 训练 → checkpoint → Policy Server → WebSocket → LIBERO action/episode。首个失败层通常比最后一条连锁错误更有价值。

在修改任何状态、覆盖文件或重新运行重型命令前，至少保存：

- 完整命令、退出码和从首个异常开始的原始输出/traceback；
- 当前 commit、`git status --short`、config name 与相关 config diff；
- 数据 repo id/root、stats 路径、checkpoint 根目录与实际 step；
- `jax.devices()`、GPU 型号/显存、`CUDA_VISIBLE_DEVICES` 与 batch/FSDP 设置；
- Policy Server/客户端参数、端口、Compose project name、服务端和客户端日志；
- 出错 episode/frame/task、输入 key/shape/dtype/单位，以及输出 action 的 shape/有限性。

若同一错误在完成本分类的只读检查后仍无法解释，把上述证据连同“最后一个通过的章节验收”和“首个失败断言”交给维护者。不要先做无范围的 `rm`、cache prune、`git reset`、全局权限修改或静默覆盖；这些操作会同时删除现场和可用回滚点。

## 1. 安装

### 1.1 Python/依赖导入失败

**症状**

`ModuleNotFoundError`、解释器不是 Python 3.11、`uv run` 选择了意外环境，或主项目与 LIBERO 旧依赖混装后出现版本冲突。

**证据命令/日志**

```bash
pwd -P
git status --short
uv --version
uv run --no-sync python --version
uv run --no-sync python - <<'PY'
import sys
print(sys.executable)
print(sys.version)
PY
```

保留第一个 import traceback。若曾在根 `.venv` 中安装 `examples/libero/requirements.txt`，把这一步也写入记录。

**常见原因**

没有按仓库锁定依赖完成安装、从错误 checkout/目录运行、`.venv` 来自另一 Python 小版本，或把 LIBERO Python 3.8 依赖装进主 Python 3.11 环境。

**处理**

先对照 [`pyproject.toml`](../../pyproject.toml) 和[第 02 章安装步骤](02_environment_setup.md#实践-2按仓库-readme-安装主环境)。需要同步时只在确认当前目录和网络/磁盘后执行仓库规定的 `GIT_LFS_SKIP_SMUDGE=1 uv sync`；LIBERO 继续使用独立 Docker/环境。不要靠在错误环境里逐个 `pip install` 掩盖锁文件或 Python 版本问题。

**回到哪个验收**

回到[第 02 章验收](02_environment_setup.md#本章验收)，重新确认 Python 3.11、主项目 import 和双运行时边界。

### 1.2 子模块状态异常

**症状**

构建或 import 报第三方目录缺失；`git submodule status` 行首不是空格。

**证据命令/日志**

```bash
git submodule status
git status --short
git diff --submodule=log
```

`-` 表示未初始化；`+` 表示子模块 checkout 与父仓库记录的 commit 不同；`U` 表示冲突。三者含义不同，不能都用同一个“重新初始化”动作处理。

**常见原因**

克隆后未递归初始化、用户正在子模块中开发、父仓库切换后子模块仍在另一 commit，或合并冲突尚未解决。

**处理**

只有行首为 `-` 且确认没有本地子模块工作时，才按仓库说明运行 `git submodule update --init --recursive`。遇到 `+` 先检查子模块内状态和预期 commit；遇到 `U` 保留冲突现场并按版本控制流程解决。不要使用 blanket reset 或删除整个子模块目录。

**回到哪个验收**

回到[第 02 章验收](02_environment_setup.md#本章验收)，要求课程所需子模块行首为一个空格。

## 2. GPU/JAX

### 2.1 JAX 只看到 CPU，或容器看不到 GPU

**症状**

`jax.devices()` 只有 CPU；出现 CUDA driver/toolkit 错误；宿主机能用 GPU，但 Compose/容器不能请求 GPU。

**证据命令/日志**

```bash
nvidia-smi
uv run --no-sync python - <<'PY'
import jax
print(jax.devices())
print("device_count:", jax.device_count())
print("default_backend:", jax.default_backend())
PY
docker compose version
```

在允许拉取镜像、磁盘充足且获授权时，再运行真实 Docker GPU smoke test：

```bash
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

该命令首次可能下载镜像；未运行就应把“容器 GPU”记录为未验证。

**常见原因**

宿主机驱动不可用、JAX 安装成 CPU 路径、`CUDA_VISIBLE_DEVICES` 屏蔽设备、NVIDIA Container Toolkit 未配置，或容器 runtime 没有 GPU 权限。

**处理**

按“宿主机 `nvidia-smi` → JAX backend → Docker smoke test”逐层修复；不要用重装全部依赖替代层级判断。驱动和 Container Toolkit 由主机管理员按组织规范处理，保留版本与完整错误。

**回到哪个验收**

回到[第 02 章验收](02_environment_setup.md#本章验收)，再进入[第 06 章验收](06_jax_and_pi0_basics.md#本章验收)。

### 2.2 OOM、设备数与 batch/FSDP 不匹配

**症状**

`RESOURCE_EXHAUSTED`、CUDA OOM、global batch 不能被设备数整除，或增加 GPU 后单卡显存仍未下降。

**证据命令/日志**

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
uv run --no-sync python - <<'PY'
import jax
from openpi.training import config as _config

config = _config.get_config("pi0_libero_tutorial")
print("devices:", jax.devices())
print("global batch:", config.batch_size)
print("fsdp_devices:", config.fsdp_devices)
PY
```

同时保存首个 OOM 前的日志、其他 GPU 进程和 allocator 环境变量。

**常见原因**

global batch 误当作每卡 batch；默认 `fsdp_devices=1` 只做数据并行并复制完整状态；模型、优化器、EMA、激活和首 batch 共同超出显存；共享 GPU 被其他进程占用。

**处理**

先让设备数整除 global batch。batch、精度、模型、EMA 或 FSDP 变化都作为新 config/exp 实验；通过实际 sharding 日志和单卡峰值验证效果。不要仅凭“多卡”或配置名含 `low_mem` 推断模型已分片，也不要在同一 checkpoint 目录暗中改变拓扑契约。

**回到哪个验收**

教学 LoRA 回到[第 08 章验收](08_configure_finetuning.md#本章验收)和[第 09 章验收](09_run_finetuning.md#本章验收)；完整 π₀.₅ 路线回到[第 11 章路线 B](11_next_steps.md#路线-b在-a100h100-上完整微调-pi05_libero)。

## 3. 数据

### 3.1 本地数据存在，但仍访问 Hub 或报 repo not found

**症状**

课程 repo id 要求登录、联网查找失败，或肉眼看到目录但 LeRobot metadata 构造失败。

**证据命令/日志**

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python - <<'PY'
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

root = Path(HF_LEROBOT_HOME) / "openpi-tutorial/libero-small"
print("HF_LEROBOT_HOME:", HF_LEROBOT_HOME)
print("dataset root:", root)
print("exists:", root.exists())
if root.exists():
    for path in sorted(root.iterdir()):
        print(path.name)
PY
```

然后用第 07/09 章显式 `root=` 的 metadata 检查保存完整 traceback。

**常见原因**

repo id 与 cache 布局不一致、metadata/data/video/index 不完整、运行用户的 cache root 不同，或未设置离线模式时同名 repo 回退 Hub。

**处理**

保持 `HF_HUB_OFFLINE=1`，修正本地 repo root、metadata 或转换产物。不要登录后让远端同名数据掩盖本地错误，也不要移动未知 cache 目录来试错。

**回到哪个验收**

回到[第 07 章验收](07_prepare_small_dataset.md#本章验收)，再重做[第 09 章本地数据预检](09_run_finetuning.md#实践-2纯本地检查数据和有限统计)。

### 3.2 episode/frame 数量、边界或样本内容异常

**症状**

转换后数据为空、episode 数不符、action 与 observation 时间错位、图像缺帧，或抽样值出现 NaN/Inf。

**证据命令/日志**

```bash
rg -n 'REPO_NAME|RAW_DATASET_NAMES|max_episodes_per_dataset|push_to_hub' \
  examples/libero/convert_libero_data_to_lerobot.py docs/tutorial/07_prepare_small_dataset.md
git diff -- examples/libero/convert_libero_data_to_lerobot.py
df -h .
```

仓库原始转换器用 `RAW_DATASET_NAMES` 选择 RLDS source，并用 `REPO_NAME` 决定 LeRobot 输出；第 07 章的小数据改动再向同一脚本的 `main()` 加入 `max_episodes_per_dataset`。若当前脚本中没有后者，说明限量转换改动尚未应用到这个 checkout，而不是 episode 上限“失效”。保存具体 episode/frame 索引、原始 key、shape/dtype、时间戳和首个非有限值；不要只保留聚合计数。

**常见原因**

源 RLDS 版本/路径不一致、筛选常量选错、转换中断留下半成品、时间轴或 episode 边界映射错误，或磁盘不足导致视频/数据缺失。

**处理**

对一个已知 episode 做源数据与 LeRobot frame 的逐字段对照，再扩展到全量检查。新转换使用独立目标 repo id/root；保留旧产物作为对照，不直接覆盖。

**回到哪个验收**

回到[第 07 章验收](07_prepare_small_dataset.md#本章验收)，要求 episode/frame 计数、字段、有限性、图像和动作语义均有证据。

## 4. Transform

### 4.1 `KeyError`、shape/dtype 或相机字段不匹配

**症状**

repack、`LiberoInputs` 或模型前处理报缺 key；图像是 `(C,H,W)` 而不是 `(H,W,C)`；state/action 维度错误。

**证据命令/日志**

```bash
rg -n 'RepackTransform|LiberoInputs|LiberoOutputs|observation.state|observation.images|actions' \
  src/openpi/training/config.py src/openpi/policies/libero_policy.py
git diff -- src/openpi/training/config.py src/openpi/policies/libero_policy.py
```

在首个失败样本处打印变换前后 `sorted(keys)`、shape、dtype 和数值范围；记录具体 episode/frame。

**常见原因**

数据集原始 key 与 repack 映射不一致、平台 adapter 期待另一嵌套结构、图像通道顺序错误，或把 padding 后的模型维度误当作原始数据维度。

**处理**

按 repack → data transform → model transform 顺序逐层运行同一个固定样本，每层只修自己的契约。不要在模型内部增加临时 key fallback 掩盖数据协议错误。

**回到哪个验收**

回到[第 05 章验收](05_data_and_transforms.md#本章验收)，自有平台还要回到[第 11 章 C1 transform 单测](11_next_steps.md#安全优先的分阶段里程碑)。

### 4.2 动作方向、尺度或 delta/absolute 语义错误

**症状**

shape 正确但动作幅度离谱、方向相反、夹爪维被错误做差分，或训练正常而闭环立即越界。

**证据命令/日志**

```bash
rg -n 'extra_delta_transform|DeltaActions|AbsoluteActions|make_bool_mask' \
  src/openpi/training/config.py src/openpi/transforms.py examples/ur5/README.md
```

保存原始 state/action、delta 后值、输出逆变换后的值、单位、坐标系、控制频率和 mask。用固定样本做 delta→absolute 往返检查。

**常见原因**

训练和推理使用不同 `extra_delta_transform`；错误维度 mask；角度/长度单位混用；控制频率变化但动作定义未调整；π₀ 教学配置与 π₀.₅ LIBERO 配置混用。

**处理**

先停止闭环执行，在离线样本和仿真中修复双向 transform。改变动作语义后必须重算或明确复用 stats，并作为新实验重新训练/评测。真实机器人不得靠降低速度掩盖坐标系错误。

**回到哪个验收**

回到[第 05 章验收](05_data_and_transforms.md#本章验收)和[第 10 章验收](10_serve_and_evaluate.md#本章验收)。

## 5. Normalization

### 5.1 stats 找不到、shape 不符或含 NaN/Inf

**症状**

`Normalization stats not found`、state/action 统计维度不匹配、标准差异常，或加载后出现 NaN/Inf。

**证据命令/日志**

```bash
find assets -path '*norm_stats.json' -type f -print | sort
rg -n 'repo_id|asset_id|assets_dir|extra_delta_transform' src/openpi/training/config.py
git status --short
```

使用[第 08 章检查脚本](08_configure_finetuning.md#实践-7读取实际-json-并验证-keyshape-与有限性)记录精确 config、repo id、asset id、stats 路径、各数组 shape 与有限性。

**常见原因**

统计和训练使用不同 config/repo id；stats 来自另一机器人或动作协议；文件被移动；样本太少或输入已有非有限值；delta 语义改变后沿用旧 stats。

**处理**

让数据、transform、config 和 asset id 一一对应。不要把任意 `norm_stats.json` 复制到报错路径，也不要手改 JSON 数值。需要重算时先创建新实验身份或显式备份旧文件。

**回到哪个验收**

回到[第 08 章验收](08_configure_finetuning.md#本章验收)，确认 8D state/7D action（教学 LIBERO）等实际协议的 shape 和有限性。

### 5.2 计算 stats 时提示将覆盖已有文件

**症状**

preflight 报 `REFUSING TO OVERWRITE`，或目标 `norm_stats.json` 已存在但来源不明。

**证据命令/日志**

```bash
TARGET=assets/pi0_libero_tutorial/openpi-tutorial/libero-small/norm_stats.json
if test -e "$TARGET"; then
  stat -c '%y %s %n' "$TARGET"
else
  echo "target does not exist: $TARGET"
fi
git status --short -- "$TARGET"
```

**常见原因**

重复使用同一 config/repo id、上次统计已经成功、旧实验遗留，或当前 checkout 指向相同 assets 目录。

**处理**

把 guard 视为正常保护。查明旧文件对应的数据和协议；需要保留时为精确文件/目录选择不存在的备份名，或为新实验使用唯一 config/repo id，然后重新通过 guard。不要删除 guard、改成 warning 或静默覆盖。

**回到哪个验收**

回到[第 08 章验收](08_configure_finetuning.md#本章验收)中的 stats 写入保护与一致性清单。

## 6. 训练

### 6.1 很久没有出现 step 日志

**症状**

训练进程看似停住，尚未出现 `Step ...`，或 GPU 利用率与预期不符。

**证据命令/日志**

```bash
pgrep -af 'scripts/train.py'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
rg -n 'Initialized data loader|Initialized train state|Step [0-9]+:|Waiting for checkpoint|Traceback' /tmp/openpi-training.log
```

把日志路径替换为本次实际文件；没有日志文件时保存训练终端从启动到当前的完整输出。

**常见原因**

本地数据/worker/I/O 卡在首 batch；基础权重下载；模型初始化；第一次 JIT；进程已退出但终端滚动位置造成误判。

**处理**

依据最后成功日志分层：首 batch 前查数据；data loader 后查权重/cache；GPU 已分配且进程存活时查 JIT；进程不存在则看退出码和 traceback。不要仅按等待分钟数判断“编译死锁”，也不要同时启动第二个同名训练。

**回到哪个验收**

回到[第 09 章验收](09_run_finetuning.md#本章验收)，确认至少一组实际且有限的训练指标。

### 6.2 loss/grad_norm/param_norm 出现 NaN/Inf

**症状**

训练日志中的任一关键指标首次变成非有限值，或数值在某一步骤突然爆炸。

**证据命令/日志**

```bash
rg -n -i 'step [0-9]+:|loss|grad_norm|param_norm|nan|inf' /tmp/openpi-training.log
git diff -- src/openpi/training/config.py
```

保存首个异常 step、之前的有限窗口、学习率、输入 batch 索引、原始数据有限性和 stats 数值范围。

**常见原因**

原始数据或 stats 非有限、极小 std、动作单位/transform 错误、学习率或梯度设置不合适，或错误恢复了不兼容状态。

**处理**

停止把异常后的 checkpoint 当作有效候选。先回到数据和 stats 的固定样本检查，再把优化器/学习率变化作为新实验。不要只过滤日志、`nan_to_num` 输入或重命名异常 checkpoint 让表面验收通过。

**回到哪个验收**

依次回到[第 07 章验收](07_prepare_small_dataset.md#本章验收)、[第 08 章验收](08_configure_finetuning.md#本章验收)和[第 09 章验收](09_run_finetuning.md#本章验收)。

## 7. Checkpoint

### 7.1 最终 step 缺失或 checkpoint 组件不完整

**症状**

500-step 教学运行没有 `499`，数字目录存在但缺 `train_state`/`params`/`assets`，或训练结束前中断。

**证据命令/日志**

```bash
CHECKPOINT_ROOT=checkpoints/pi0_libero_tutorial/libero_lora_demo
find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 2 -print | sort
rg -n 'Saving checkpoint|Waiting for checkpoint manager|Traceback' /tmp/openpi-training.log
```

检查每个数字 step 的时间戳，并确认进程是否到达 `wait_until_finished()` 后正常退出。

**常见原因**

Orbax 异步保存尚未提交；训练中断；磁盘/权限失败；或最终更新后那次额外 `next(data_iter)` 在进入 499 保存判断前阻塞/报错。当前训练循环会先获取下一 batch，再判断保存；最终 batch fetch 失败时不会请求最终 499。

**处理**

保留现场，以 checkpoint manager 已提交的 step 和完整三组件为准。若 traceback 在最终 batch fetch，排查 dataset/worker/I/O/transform；若在 save/wait，排查磁盘和异步提交。不要把临时目录手工改名为数字 step，也不要假定配置中列出的所有保存点都必然存在。

**回到哪个验收**

回到[第 09 章验收](09_run_finetuning.md#本章验收)，完整运行要求 wait 后正常退出且最新有效 step 为 499。

### 7.2 服务加载时报参数树、config 或 assets 不兼容

**症状**

加载 checkpoint 时出现 missing/unexpected key、shape mismatch、模型类型错误或 normalization stats 缺失。

**证据命令/日志**

```bash
rg -n 'name="pi0_libero_tutorial"|name="pi05_libero"|weight_loader|action_horizon|extra_delta_transform' \
  src/openpi/training/config.py
find checkpoints -path '*/params' -o -path '*/assets' | sort
```

记录服务命令中的 config name、精确 checkpoint step、`params` 路径、asset id/stats 路径和完整加载 traceback。

**常见原因**

把 π₀ LoRA checkpoint 交给 `pi05_libero`；把 π₀.₅ 参数交给 π₀；只复制 `params` 没有配套 assets；checkpoint 的 repo/asset id 与当前 data config 漂移；horizon/variant 改动后仍沿用旧 step。

**处理**

把 config、模型图、`params`、`assets` 当作一个兼容单元。回到生成该 checkpoint 的 commit/config，验证三组件和 stats 后再服务。不要通过忽略 missing key、裁剪参数树或拿别的 stats 凑路径来绕过错误。

**回到哪个验收**

回到[第 10 章 checkpoint preflight](10_serve_and_evaluate.md#实践-2fail-fast-检查-configcheckpoint-与-assets)和[第 10 章验收](10_serve_and_evaluate.md#本章验收)。

### 7.3 `--resume` 仍报本地数据或原始 stats 缺失

**症状**

checkpoint 目录和其中 assets 都存在，但恢复在 `restore_state()` 前就因 dataset/stats 失败。

**证据命令/日志**

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python - <<'PY'
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

print(Path(HF_LEROBOT_HOME) / "openpi-tutorial/libero-small")
PY
test -f assets/pi0_libero_tutorial/openpi-tutorial/libero-small/norm_stats.json
find checkpoints/pi0_libero_tutorial/libero_lora_demo -mindepth 1 -maxdepth 1 -type d -print | sort
```

**常见原因**

当前源码在恢复 state 前先用 config 创建 data loader、读取原始 stats 并取得首 batch；checkpoint assets 不会自动还原到 `config.assets_dirs`。此外 resume 不保存/恢复数据迭代器位置。

**处理**

恢复原实验所用的本地 dataset 和原始 norm stats 路径，保持相同 config/exp name，只使用 `--resume`，不加 `--overwrite`。若数据版本已改变，把它视为新实验，而不是强行续接旧 optimizer state。

**回到哪个验收**

回到[第 09 章恢复实践](09_run_finetuning.md#实践-9中断后恢复而不是覆盖)和[第 09 章验收](09_run_finetuning.md#本章验收)。

## 8. 服务与网络

### 8.1 端口占用、连接拒绝或客户端一直等待

**症状**

Policy Server bind 失败、客户端 `Waiting for server...`、连接被拒绝，或服务端与客户端使用了不同端口。

**证据命令/日志**

在 `set -o pipefail` 下不要使用 `ss -ltn | rg -q ...`；`rg -q` 可能提前关闭管道，让上游 SIGPIPE 混入判断。先捕获完整监听表：

```bash
set -euo pipefail

PORT=8000
SS_LISTEN=$(ss -ltn)
if rg ":${PORT}\\b" <<< "$SS_LISTEN" >/dev/null; then
  echo "HOST PORT ${PORT} IS ALREADY IN USE" >&2
else
  echo "HOST PORT ${PORT} APPEARS FREE"
fi

docker compose -f examples/libero/compose.yml ps -a
docker compose -f examples/libero/compose.yml logs --tail=300 openpi_server runtime
```

共享 host 还要记录精确 Compose `-p` project、服务端 `--port` 和客户端 `--args.port`。

**常见原因**

host network 上已有进程监听端口；只修改了服务端或客户端一侧；查看日志时漏掉启动所用 project name；服务端仍在加载/JIT；服务进程已经退出。

**处理**

选择经团队确认的空闲端口，并让 server/client 同时修改；所有 `up/ps/logs/down` 使用同一 project name。先从服务端加载完成与 `Connection from ... opened` 证明连接，再判断 action 协议。不要因为客户端曾等待结束就宣称闭环通过。

**回到哪个验收**

回到[第 10 章验收](10_serve_and_evaluate.md#本章验收)，要求服务加载、连接、action 和 episode 分层证据齐全。

### 8.2 Compose 在共享/不可信主机上的安全边界不合适

**症状**

准备在多人机器运行默认 Compose，但尚未审计 `network_mode: host`、`privileged: true` 和宿主机写挂载；或不同项目互相影响端口/文件。

**证据命令/日志**

```bash
set -euo pipefail

COMPOSE_CONFIG=$(docker compose -f examples/libero/compose.yml config)
rg -n 'network_mode: host|privileged: true|source:|target:|/app|openpi_assets' <<< "$COMPOSE_CONFIG"
```

记录展开后的 host source、容器 target、读写模式、GPU、project name 和端口，不只阅读原始 YAML。

**常见原因**

把容器当作强隔离边界。host network 不隔离端口并暴露宿主网络命名空间；privileged 容器权限很高；仓库或 cache 的可写 bind mount 允许容器修改宿主文件。

**处理**

默认 Compose 只用于获授权、可信、独占或经过管理员评审的开发主机。共享环境使用唯一 project/端口并最小化写挂载；需要改变 privilege/network/mount 时由平台负责人设计 override 并验证功能与安全。不要在不可信多租户主机直接运行默认配置。

**回到哪个验收**

回到[第 10 章静态 Compose 安全检查](10_serve_and_evaluate.md#实践-3静态检查-compose-安全边界和默认端口)与[第 10 章验收](10_serve_and_evaluate.md#本章验收)。

## 9. LIBERO 渲染与评测

### 9.1 EGL/GLX、MuJoCo 或显示连接失败

**症状**

EGL 初始化失败、OpenGL context 错误、GLX 无法连接 X server，或 runtime 在 episode 开始前退出。

**证据命令/日志**

```bash
docker compose -f examples/libero/compose.yml logs --tail=300 runtime
docker compose -f examples/libero/compose.yml config | rg -n 'MUJOCO_GL|PYOPENGL_PLATFORM|DISPLAY|X11'
```

保存首个 EGL/GLX/MuJoCo traceback和展开后的两个渲染变量。

**常见原因**

GPU/container runtime 问题、EGL 设备不可用，或只把 `MUJOCO_GL` 改为 `glx` 而 `PYOPENGL_PLATFORM` 仍是 `egl`；GLX 还依赖正确的 `DISPLAY`、X11 socket 和授权。

**处理**

EGL 是默认路径，先修 GPU/EGL 层。只有日志证明需要时，才严格按[第 02 章“EGL 是 Compose 默认值，GLX 是回退路径”](02_environment_setup.md#egl-是-compose-默认值glx-是回退路径)使用临时 override 同时设置两个变量，并理解 `xhost` 放宽权限的风险和反向撤销方式。不要一开始就全局切 GLX。

**回到哪个验收**

回到[第 03 章验收](03_first_libero_inference.md#本章验收)或[第 10 章验收](10_serve_and_evaluate.md#本章验收)，确认 episode 实际推进。

### 9.2 已连接但 action 断言失败、episode 不推进或没有新视频

**症状**

服务端出现连接日志，但客户端 `Caught exception`、action chunk 太短/维度错误、只执行 dummy action、episode 没结束，或视频目录只有旧文件。

**证据命令/日志**

```bash
rg -n 'replan_steps|len\(action_chunk\)|action_plan.extend|Caught exception|Starting episode|Success:' \
  examples/libero/main.py
rg -n 'action_horizon|return \{"actions"' \
  src/openpi/models/pi0_config.py src/openpi/policies/libero_policy.py
docker compose -f examples/libero/compose.yml logs --tail=400 openpi_server runtime
```

检查本次唯一视频目录，而不是共享历史目录：

```bash
: "${VIDEO_OUT_PATH:?Export the exact path from this run}"
find "$VIDEO_OUT_PATH" -maxdepth 1 -type f -name 'rollout_*.mp4' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
```

**常见原因**

config/checkpoint horizon 不匹配；`replan_steps` 大于有效 chunk；输出未裁到平台 action 维度；字段/数值异常；episode 尚未完成；沿用旧视频目录把历史 MP4 当成新证据。

**处理**

先确认服务端无参数/assets 错，再确认返回 action shape、有限性、`len(action_chunk) >= replan_steps` 和平台 7D 输出。每次评测使用不存在的唯一视频目录，等待 episode 结束后检查 `Success: True/False` 和新 MP4。不要把连接打开或单个成功视频等同于正式 success rate。

**回到哪个验收**

回到[第 10 章 action/episode 证据实践](10_serve_and_evaluate.md#实践-6收集连接action-与-episode-证据)和[第 10 章验收](10_serve_and_evaluate.md#本章验收)。

## 仍然无法定位时

提交问题前，把通用升级规则中的证据打包成一个只读记录，并明确回答：

1. 最后通过的是哪个章节验收？
2. 首个失败命令和首个异常是什么？
3. config、dataset、stats、checkpoint step、params/assets 是否来自同一实验？
4. 问题能否用固定离线样本复现，还是只在 GPU、服务或 LIBERO 闭环出现？
5. 为排错已经改变过哪些状态，哪些原始现场仍被保留？

这五项通常足以让维护者把问题路由到正确层级，也能避免反复建议清 cache、重装或重跑长任务。

[课程入口：README](README.md) · [课程术语表](glossary.md) · [环境安装](02_environment_setup.md) · [数据与变换](05_data_and_transforms.md) · [训练](09_run_finetuning.md) · [服务与评测](10_serve_and_evaluate.md)
