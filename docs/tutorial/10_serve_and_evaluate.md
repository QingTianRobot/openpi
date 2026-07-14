# 第 10 章：服务教学 checkpoint 与完成 LIBERO 闭环评测

[← 上一章：运行短程 LoRA 微调](09_run_finetuning.md) · [课程入口：README](README.md) · [下一章：规划后续实验 →](11_next_steps.md)

## 本章目标

完成本章后，你应该能够：

- 从 `checkpoints/pi0_libero_tutorial/libero_lora_demo/` 中选择一个实际存在、组件完整且有训练证据的数字 step；
- 解释为什么教学 checkpoint 必须与 `pi0_libero_tutorial` 一起加载，不能改用 `pi05_libero`；
- 说明 `create_trained_policy()` 怎样用 config 创建 LoRA 模型图、从 checkpoint 读取推理参数与 normalization assets；
- 区分推理所需的 `params/assets` 与只用于继续训练的 `train_state`；
- 在可信、获授权的 Docker/GPU 主机上安全启动 `openpi_server` 与 LIBERO `runtime`；
- 在共享 host 上让 Compose project name、服务端端口和客户端端口保持一致；
- 从服务端连接日志、客户端 episode 进度、action 错误与视频文件判断闭环是否真的推进；
- 区分“策略完成任务”与“服务、通信、动作和仿真接口正确”；
- 用证据定位 config、checkpoint、assets、缓存、端口、GPU、渲染和视频写入问题。

遇到陌生缩写可查后续补齐的[课程术语表](glossary.md)；跨章节问题可查后续补齐的[故障排查手册](troubleshooting.md)。

## 前置条件

开始前应已经完成[第 08 章](08_configure_finetuning.md)和[第 09 章](09_run_finetuning.md)，并确认：

1. 当前 checkout 的 `src/openpi/training/config.py` 仍包含唯一且内容未漂移的 `pi0_libero_tutorial`；
2. 第 09 章已经验证至少一个数字 step 的 `train_state`、`params`、`assets` 和 checkpoint norm stats；
3. 只有在完整执行 500 个训练 step、最终 batch 获取与 save 均成功、异步 manager 正常完成时，才把 `499` 当作最终 checkpoint；
4. 若没有可信的 `499`，应从实际数字目录中选择另一个组件完整、并能与训练日志对应的 step；
5. Docker Compose、NVIDIA Container Toolkit、LIBERO 子模块和默认 EGL 路径已经通过第 02、03 章的环境检查；
6. 仓库根目录、Docker 数据目录、`${OPENPI_DATA_HOME:-$HOME/.cache/openpi}` 和视频输出目录有足够空间；
7. 默认 host 端口 `8000` 或共享方案使用的端口未被其他进程占用。

本章命令必须从 **openpi 仓库根目录**运行。Compose 会把命令执行时的 `$PWD` 可写挂载为两个容器中的 `/app`；若从错误目录启动，容器就看不到课程 config 或 checkpoint。

> **安全边界：**[`examples/libero/compose.yml`](../../examples/libero/compose.yml) 为 `runtime` 设置了 `privileged: true`，并把当前仓库通过 `$PWD:/app` 可写挂载到 `runtime` 与 `openpi_server`。只应在你获准使用 Docker/GPU 的主机上运行可信 checkout。共享 Docker host 上必须协调唯一 project name、空闲 host 端口和 GPU 使用；`up`、`ps`、`logs`、`exec`、`down` 必须始终指向同一个 project。

本文编写环境没有课程 checkpoint、完整容器依赖和可用于重型评测的 GPU，因此**没有实际运行** `docker compose ... up --build`、加载模型或执行 LIBERO episode。后文运行结果都是按当前源码推导、由学习者现场验证的条件式预期，不是伪造日志。

## 核心概念

### config 与 checkpoint 共同定义一份可加载策略

教学 checkpoint 不是一袋可以交给任意 LIBERO config 的数组。服务端显式 checkpoint 路径会调用：

```text
create_trained_policy(train_config, checkpoint_dir)
```

其中 config 与 checkpoint 分工如下：

| 部分 | 来自哪里 | 决定什么 |
| --- | --- | --- |
| 模型图 | `pi0_libero_tutorial.model` | π₀、两个 LoRA variant、参数树、action horizon 50、内部 action dim 32 |
| 输入/输出 transform | `pi0_libero_tutorial.data` | LIBERO 字段、8D state、7D action、prompt、图像与 extra delta 动作语义 |
| 参数值 | `<step>/params` | 基础参数、LoRA 参数及本次训练后的完整推理参数 |
| normalization | `<step>/assets/<asset_id>/norm_stats.json` | 与训练数据和动作语义配套的 state/action 归一化与反归一化 |
| 训练恢复状态 | `<step>/train_state` | optimizer、训练 step 等；Policy Server 推理不会读取 |

[`policy_config.py`](../../src/openpi/policies/policy_config.py) 对 JAX checkpoint 执行 `train_config.model.load(restore_params(checkpoint_dir / "params"))`。随后它从 config 创建 data/model transforms，但 normalization 明确从 `checkpoint_dir / "assets"` 和 config 的 `asset_id` 加载，保证服务使用训练时随 checkpoint 保存的 stats。

因此必须使用：

```text
config:     pi0_libero_tutorial
checkpoint: checkpoints/pi0_libero_tutorial/libero_lora_demo/<step>
```

不能因为评测环境都是 LIBERO 就换成 `pi05_libero`。π₀.₅ 的模型模式、参数树、state/token 处理、action horizon 与动作 transform 都不同；错配可能在 load 时出现 key/shape/tree 错误，也可能把错误 normalization 或动作语义带到仿真。

### 从 checkpoint 到 action chunk 的数据流

[`scripts/serve_policy.py`](../../scripts/serve_policy.py) 解析 `policy:checkpoint` 后，用 config name 和目录创建 Policy，再启动 [`WebsocketPolicyServer`](../../src/openpi/serving/websocket_policy_server.py)。一次请求的路径是：

```text
LIBERO observation
  → runtime 构造两路图像、8D state、prompt
  → WebsocketClientPolicy 发送 msgpack
  → WebsocketPolicyServer 调用 Policy.infer()
  → config 的输入 transforms + checkpoint normalization
  → π₀ LoRA 模型 sample_actions
  → 反向 transforms + unnormalize + LIBERO 7D 输出
  → 返回 actions[action_horizon, 7]
  → runtime 取前 replan_steps 个动作推进环境
```

教学 `Pi0Config` 未改写 `action_horizon`，所以沿用 π₀ 默认值 50；客户端默认 `replan_steps=5`。当前组合满足 `50 >= 5`。客户端仍会对真实响应执行：

```text
len(action_chunk) >= replan_steps
```

不要只凭静态 config 宣称响应正确。运行时还要检查服务端是否返回错误帧、客户端是否打印 `Caught exception`、episode 是否越过等待阶段并持续推进。

### 两个 Compose service 为什么能看到同一 checkout

Compose 定义了两个 host-network service：

| service | 容器内命令 | 本章职责 |
| --- | --- | --- |
| `openpi_server` | `uv run scripts/serve_policy.py $SERVER_ARGS` | 读取 `/app` 中的课程 config 与本地 checkpoint，加载模型并监听端口 |
| `runtime` | `python examples/libero/main.py $CLIENT_ARGS` | 启动 LIBERO、连接服务端、发送 observation、执行动作并保存视频 |

两个容器都挂载 `$PWD:/app`。只要从仓库根目录启动，学习者在宿主机添加的 `pi0_libero_tutorial` 和 `./checkpoints/.../<step>` 会以相同相对路径出现在 `/app` 中。服务镜像虽然在 build 阶段安装依赖，但运行时 `/app` 是宿主机当前 checkout，不是镜像构建时的一份旧源码副本。

本地 checkpoint 参数不需要重新从 GCS 下载。不过模型 transform 会创建 PaliGemma tokenizer，当前源码可能从 `gs://big_vision/paligemma_tokenizer.model` 获取它；Docker build 也可能下载镜像与 Python 依赖。Compose 把宿主机 `${OPENPI_DATA_HOME:-~/.cache/openpi}` 挂载为服务端 `/openpi_assets`，供这些 openpi 资源复用。`HF_HUB_OFFLINE=1` 不会阻止 GCS 下载。

### 工程闭环与任务成功率是两个验收层级

`libero_spatial` 是包含多个 task 的 suite。`--args.num-trials-per-task 1` 表示**每个 task 各运行一次**，不是整个 suite 只运行一个 episode。客户端会遍历 suite 的多个 task，并把已结束 episode 的 MP4 写到：

```text
data/libero/videos/
```

500-step、小数据 LoRA 是工程教学，不是正式性能训练。`Success: False` 可以与以下事实同时成立：checkpoint 成功加载、WebSocket 正常、actions 返回、仿真推进、视频写出。任务 success rate 衡量策略质量；本章验收的是训练到服务再到仿真的接口闭环，不要求达到官方或预训练模型的成功率。

## 源码地图

| 文件或章节 | 阅读重点 |
| --- | --- |
| [`scripts/serve_policy.py`](../../scripts/serve_policy.py) | `policy:checkpoint` CLI、default policy、端口与 WebSocket Server 创建 |
| [`src/openpi/policies/policy_config.py`](../../src/openpi/policies/policy_config.py) | `create_trained_policy()` 怎样加载 params、checkpoint assets 与 transforms |
| [`src/openpi/policies/policy.py`](../../src/openpi/policies/policy.py) | `Policy.infer()` 怎样变换 observation、采样 actions、反变换并返回 timing |
| [`src/openpi/serving/websocket_policy_server.py`](../../src/openpi/serving/websocket_policy_server.py) | `Connection ... opened`、请求解码、策略调用、响应和服务端 traceback |
| [`packages/openpi-client/src/openpi_client/websocket_client_policy.py`](../../packages/openpi-client/src/openpi_client/websocket_client_policy.py) | 客户端等待、连接、metadata、请求与错误帧处理 |
| [`examples/libero/compose.yml`](../../examples/libero/compose.yml) | privileged、可写挂载、host network、GPU、cache、EGL 与参数传递 |
| [`examples/libero/main.py`](../../examples/libero/main.py) | suite/task/episode 循环、observation、action horizon 断言、视频与成功率 |
| [`examples/libero/README.md`](../../examples/libero/README.md) | 上游 Docker 命令和预训练 benchmark 背景 |
| [`docs/remote_inference.md`](../remote_inference.md) | Policy Server 与轻量客户端的通用远程推理模式 |
| [第 03 章](03_first_libero_inference.md) | 官方 `pi05_libero` 预训练闭环、Compose 安全和渲染排错 |
| [第 08 章](08_configure_finetuning.md) | `pi0_libero_tutorial` 的模型、data transform、动作语义和 stats 契约 |
| [第 09 章](09_run_finetuning.md) | checkpoint step、组件、异步保存和恢复证据 |

## 动手实践

以下命令都从仓库根目录执行。先完成 checkpoint/config 预检，再启动 Compose；不要把“容器里报错”当成发现路径拼错的第一道检查。

### 实践 1：选择一个有证据的 checkpoint step

如果第 09 章完整运行并明确验证最终保存，可在当前 shell 中选择：

```bash
export CHECKPOINT_STEP=499
```

如果没有可信的 499，不要创建同名目录或把其他 step 复制成 499。用下面命令按数字倒序寻找第一个结构完整的候选；它同时要求 `train_state`、`params` 和 checkpoint norm stats 存在：

```bash
set -euo pipefail

CHECKPOINT_ROOT=checkpoints/pi0_libero_tutorial/libero_lora_demo
ASSET_REL=openpi-tutorial/libero-small/norm_stats.json

if ! test -d "$CHECKPOINT_ROOT"; then
  echo "CHECKPOINT ROOT DOES NOT EXIST: $CHECKPOINT_ROOT" >&2
  exit 1
fi

NUMERIC_STEPS=$(
  find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | awk '/^[0-9]+$/' | sort -nr
)
if test -z "$NUMERIC_STEPS"; then
  echo "NO NUMERIC CHECKPOINT STEPS FOUND: $CHECKPOINT_ROOT" >&2
  exit 1
fi

SELECTED_STEP=""
while IFS= read -r step; do
  step_dir="$CHECKPOINT_ROOT/$step"
  if test -e "$step_dir/train_state" \
    && test -e "$step_dir/params" \
    && test -f "$step_dir/assets/$ASSET_REL"; then
    SELECTED_STEP="$step"
    break
  fi
done <<< "$NUMERIC_STEPS"

if test -z "$SELECTED_STEP"; then
  echo "NO STRUCTURALLY COMPLETE CHECKPOINT STEP FOUND" >&2
  exit 1
fi

export CHECKPOINT_STEP="$SELECTED_STEP"
echo "selected structural candidate: $CHECKPOINT_STEP"
```

“结构完整”仍不是训练质量证明。把输出 step 与第 09 章实际日志、save/wait 和时间戳对应；如果目录来自中断现场而 manager 未认可，不要用于服务。

### 实践 2：fail-fast 检查 config、checkpoint 与 assets

下面 preflight 不创建模型、不加载数十亿参数，也不访问 LeRobot dataset。它检查当前工作目录、配置唯一性、关键 config 字段、checkpoint 顶层组件，以及 checkpoint 中 8D/7D stats 的 shape 与有限性：

```bash
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
if test "$(pwd -P)" != "$(cd "$REPO_ROOT" && pwd -P)"; then
  echo "RUN FROM REPOSITORY ROOT: $REPO_ROOT" >&2
  exit 1
fi

: "${CHECKPOINT_STEP:?Set CHECKPOINT_STEP to 499 or a verified numeric step first}"
if ! [[ "$CHECKPOINT_STEP" =~ ^[0-9]+$ ]]; then
  echo "CHECKPOINT_STEP MUST BE NUMERIC: $CHECKPOINT_STEP" >&2
  exit 1
fi

CONFIG_COUNT=$(rg -c 'name="pi0_libero_tutorial"' src/openpi/training/config.py || true)
if test "${CONFIG_COUNT:-0}" -ne 1; then
  echo "EXPECTED EXACTLY ONE pi0_libero_tutorial CONFIG, FOUND: ${CONFIG_COUNT:-0}" >&2
  exit 1
fi

export CHECKPOINT_DIR="checkpoints/pi0_libero_tutorial/libero_lora_demo/$CHECKPOINT_STEP"
for component in train_state params assets; do
  if ! test -e "$CHECKPOINT_DIR/$component"; then
    echo "MISSING CHECKPOINT COMPONENT: $CHECKPOINT_DIR/$component" >&2
    exit 1
  fi
done

CHECKPOINT_NORM_STATS="$CHECKPOINT_DIR/assets/openpi-tutorial/libero-small/norm_stats.json"
if ! test -f "$CHECKPOINT_NORM_STATS"; then
  echo "MISSING CHECKPOINT NORM STATS: $CHECKPOINT_NORM_STATS" >&2
  exit 1
fi

sha256sum src/openpi/training/config.py
HF_HUB_OFFLINE=1 uv run --no-sync python - <<'PY'
import os
from pathlib import Path

import numpy as np

from openpi.models import pi0_config
from openpi.shared import normalize
from openpi.training import config as _config

config = _config.get_config("pi0_libero_tutorial")
checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"])

assert config.name == "pi0_libero_tutorial"
assert config.model.model_type.value == "pi0"
assert config.model.pi05 is False
assert config.model.paligemma_variant == "gemma_2b_lora"
assert config.model.action_expert_variant == "gemma_300m_lora"
assert config.model.action_horizon == 50
expected_filter = pi0_config.Pi0Config(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
).get_freeze_filter()
assert repr(config.freeze_filter) == repr(expected_filter)
assert config.data.repo_id == "openpi-tutorial/libero-small"
assert config.data.extra_delta_transform is True
assert config.data.base_config.prompt_from_task is True

stats_dir = checkpoint_dir / "assets" / config.data.repo_id
stats = normalize.load(stats_dir)
assert set(stats) == {"state", "actions"}, sorted(stats)
for key, shape in {"state": (8,), "actions": (7,)}.items():
    for field in ("mean", "std", "q01", "q99"):
        value = np.asarray(getattr(stats[key], field))
        print(key, field, "shape=", value.shape, "finite=", bool(np.isfinite(value).all()))
        assert value.shape == shape, (key, field, value.shape)
        assert np.isfinite(value).all(), (key, field)

print("config:", config.name)
print("checkpoint:", checkpoint_dir)
print("action horizon:", config.model.action_horizon)
print("preflight passed without model instantiation")
PY
```

`HF_HUB_OFFLINE=1` 让这次宿主机静态检查不应回退 Hugging Face Hub。服务创建 policy 时不会实例化 LeRobot dataset；它只用 data factory 创建 transforms 与 asset id，所以本地训练 dataset 不再是推理前置。实际 Policy Server 仍可能通过 openpi downloader 获取 GCS tokenizer，离线要求应单独验证 `${OPENPI_DATA_HOME}` cache。

`train_state` 在这里被检查，是为了确认候选 step 与第 09 章训练产物一致；`create_trained_policy()` 推理只读取 `params` 与 `assets`，不会恢复 optimizer 或训练 step。

### 实践 3：静态检查 Compose 安全边界和默认端口

在独占、可信的开发主机上使用默认端口前，确认端口未被监听，并查看 Compose 展开后的关键字段：

```bash
set -euo pipefail

if ss -ltn | rg -q ':8000\b'; then
  echo "HOST PORT 8000 IS ALREADY IN USE" >&2
  exit 1
fi

export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi0_libero_tutorial --policy.dir ./checkpoints/pi0_libero_tutorial/libero_lora_demo/499"
export CLIENT_ARGS="--args.task-suite-name libero_spatial --args.num-trials-per-task 1"

docker compose -f examples/libero/compose.yml config \
  | rg -n 'privileged: true|network_mode: host|SERVER_ARGS|CLIENT_ARGS|/app|OPENPI_DATA_HOME|openpi_assets'
```

如果你选择的不是 499，把 `SERVER_ARGS` 中的末尾 step 改为已验证的 `$CHECKPOINT_STEP`：

```bash
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi0_libero_tutorial --policy.dir ./checkpoints/pi0_libero_tutorial/libero_lora_demo/$CHECKPOINT_STEP"
```

检查展开结果中的 `/app` host source 是否就是当前仓库，确认 `runtime` 仍是 privileged，两个 service 使用 host network，且服务端 cache 挂载符合预期。`docker compose config` 不构建镜像或加载模型，只验证静态合成结果。

### 实践 4：启动默认教学闭环

若完整运行已验证 499，默认教学命令是：

```bash
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi0_libero_tutorial --policy.dir ./checkpoints/pi0_libero_tutorial/libero_lora_demo/499"
export CLIENT_ARGS="--args.task-suite-name libero_spatial --args.num-trials-per-task 1"
docker compose -f examples/libero/compose.yml up --build
```

Tyro 会把 `policy:checkpoint` 解析为显式 checkpoint policy，并把 `--policy.config`、`--policy.dir` 填入对应字段。此时 `create_policy()` 直接走 `Checkpoint` 分支，不会读取 `args.env` 的默认 checkpoint 映射；所以 `--env LIBERO` 对显式 policy 实际未参与选择，保留它是无害的环境意图标记。更精简的等价服务参数可以省略 `--env LIBERO`，但本课程保留它以与 LIBERO Compose 命令一致。

如果选择了其他 step，使用实践 3 中含 `$CHECKPOINT_STEP` 的 `SERVER_ARGS` 后再执行同一 `docker compose ... up --build`。

第一次运行仍可能经历镜像拉取、apt/uv 依赖安装、tokenizer GCS 下载、模型加载和首次 JIT。`./checkpoints/...` 是 `/app` 中的本地 bind-mount 路径，不应被重新当作远端 checkpoint 下载；日志若显示本地路径不存在，应先检查 `$PWD:/app` 和 step 选择。

### 实践 5：共享 host 使用唯一 project 与一致端口

共享 Docker host 不要使用默认 project 和 8000。以下示例使用端口 18000，并在每条 Compose 命令中复用同一个 project name：

```bash
set -euo pipefail

if ss -ltn | rg -q ':18000\b'; then
  echo "HOST PORT 18000 IS ALREADY IN USE" >&2
  exit 1
fi

export SERVER_ARGS="--env LIBERO --port 18000 policy:checkpoint --policy.config pi0_libero_tutorial --policy.dir ./checkpoints/pi0_libero_tutorial/libero_lora_demo/499"
export CLIENT_ARGS="--args.port 18000 --args.task-suite-name libero_spatial --args.num-trials-per-task 1"

docker compose -p "openpi-tutorial-$USER-lora" -f examples/libero/compose.yml config \
  | rg -n '18000|SERVER_ARGS|CLIENT_ARGS|network_mode: host'
docker compose -p "openpi-tutorial-$USER-lora" -f examples/libero/compose.yml up --build
```

如果同一 Unix 账号由多人共享，请把 `openpi-tutorial-$USER-lora` 换成团队确认的唯一后缀。若 step 不是 499，也必须同步替换服务参数。`-p` 只隔离 Compose 资源名称；host network 的端口冲突必须由服务端 `--port 18000` 与客户端 `--args.port 18000` 一起解决。

另一个终端查看状态和日志时，也要使用同一 project：

```bash
docker compose -p "openpi-tutorial-$USER-lora" -f examples/libero/compose.yml ps -a
docker compose -p "openpi-tutorial-$USER-lora" -f examples/libero/compose.yml logs --tail=300 openpi_server runtime
```

不要用不带 `-p` 的 `ps/logs/down` 判断这个命名项目。

### 实践 6：收集连接、action 与 episode 证据

默认项目可以在另一个终端运行：

```bash
docker compose -f examples/libero/compose.yml ps -a
docker compose -f examples/libero/compose.yml logs --tail=400 openpi_server runtime
find data/libero/videos -maxdepth 1 -type f -name 'rollout_*.mp4' \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
```

共享项目给前两条命令加同一个 `-p "openpi-tutorial-$USER-lora"`。重点寻找以下证据：

1. 服务端完成 `Loading model...`，没有参数 tree/shape 或 norm stats 错误，并进入创建 server 的路径；
2. 客户端可能先打印 `Task suite: libero_spatial` 与 `Waiting for server...`；轻量客户端**没有**显式 `Connected` 日志；
3. 可靠的连接证据是服务端 `Connection from ... opened`，或客户端等待后继续到 `Task: ...` 与 `Starting episode...`；
4. `Starting episode...` 之后前 10 个 timestep 是 dummy action 等待。要证明真实 policy actions 返回，应继续确认服务端没有错误帧、客户端没有 `Caught exception`/action horizon 断言，并观察 episode 持续推进或结束；
5. 当前 config 的静态 horizon 是 50，客户端断言实际 `len(action_chunk) >= 5`；随后 `LiberoOutputs` 只返回前 7 个 action 维度供环境执行；
6. 已结束 episode 会记录 `Success: True/False`、累计进度，并在 `data/libero/videos/` 留下 success/failure MP4。

用源码再次核对 action 契约：

```bash
rg -n 'action_horizon: int = 50' src/openpi/models/pi0_config.py
rg -n 'replan_steps|len\(action_chunk\)|action_plan.extend|Caught exception' examples/libero/main.py
rg -n 'return \{"actions".*:7' src/openpi/policies/libero_policy.py
```

若服务端连接已经打开，但客户端很快打印 `Caught exception`，不要把“曾经连接”误当作 action 接口通过。保留服务端 traceback 与客户端首个错误，按常见问题定位字段、shape、horizon 或仿真 action。

### 实践 7：理解 suite、trial、视频与成功率

默认客户端参数：

```text
task suite:            libero_spatial
trials per task:       1
replan steps:          5
video output:          data/libero/videos
```

主循环仍遍历 `task_suite.n_tasks`。因此即使每个 task 只有一个 trial，完整 suite 也会运行多个 episode，可能耗时很长。只做早期 smoke check 时，可以在真实 action 请求成功且 episode 已推进后安全停止；要完成本章验收，至少让一个 episode 走到 success/failure 记录与 MP4 写出，但不必等待整个 suite。

教学 checkpoint 在小数据上只训练 500 step，任务全部失败并不意外。不要把本章结果与 [`examples/libero/README.md`](../../examples/libero/README.md) 中特定官方预训练 checkpoint 的成功率表直接比较；训练规模、模型路线、数据和评测目的都不同。

### 实践 8：停止并清理本次 Compose 项目

在 `up` 终端按 `Ctrl+C`，等待 Compose 停止。默认项目使用：

```bash
docker compose -f examples/libero/compose.yml ps -a
docker compose -f examples/libero/compose.yml down
unset SERVER_ARGS CLIENT_ARGS CHECKPOINT_STEP CHECKPOINT_DIR
```

共享项目必须继续使用相同 project name：

```bash
docker compose -p "openpi-tutorial-$USER-lora" -f examples/libero/compose.yml ps -a
docker compose -p "openpi-tutorial-$USER-lora" -f examples/libero/compose.yml down
unset SERVER_ARGS CLIENT_ARGS CHECKPOINT_STEP CHECKPOINT_DIR
```

`down` 不会删除宿主机 bind mount 中的 checkpoint、课程源码、视频或 `${OPENPI_DATA_HOME}` cache。不要使用 `down --rmi all`、`docker system prune` 或删除整个 cache 来完成本章清理。

### 实践 9（可选）：单独运行 π₀.₅ 预训练参考

只有在教学项目已经停止、你明确需要工程参考且资源允许时，才单独运行预训练配置。使用不同 config/checkpoint、project、端口和视频目录，避免混合证据：

```bash
export SERVER_ARGS="--env LIBERO --port 18001 policy:checkpoint --policy.config pi05_libero --policy.dir gs://openpi-assets/checkpoints/pi05_libero"
export CLIENT_ARGS="--args.port 18001 --args.task-suite-name libero_spatial --args.num-trials-per-task 1 --args.video-out-path data/libero/videos_pi05_reference"
docker compose -p "openpi-tutorial-$USER-pi05-reference" -f examples/libero/compose.yml up --build
```

结束后使用相同 project 清理：

```bash
docker compose -p "openpi-tutorial-$USER-pi05-reference" -f examples/libero/compose.yml down
unset SERVER_ARGS CLIENT_ARGS
```

这是另一套 `pi05_libero` + 官方 checkpoint 运行，不是教学 checkpoint 的替代 config，也不能把两边 params/assets 互换。可以比较启动、连接与行为现象，但不要把一次短测外推成官方 success rate，也不要把两个视频目录或日志合并成同一实验。

## 预期结果

在学习者主机满足 checkpoint、Docker、GPU、缓存、端口与 LIBERO 条件时，应现场得到以下证据：

| 证据 | 能证明什么 | 不能单独证明什么 |
| --- | --- | --- |
| preflight 输出 config、step、horizon 和有限 stats | config/路径/assets 的静态契约一致 | 模型一定能在当前 GPU 上完成首次推理 |
| 服务端完成模型加载且无 tree/shape/norm 错误 | `pi0_libero_tutorial` 与 checkpoint 可组成 Policy | WebSocket 客户端已经收到 action |
| 服务端出现 `Connection from ... opened` | 客户端与服务端 host/port 可达 | action 内容或仿真字段一定正确 |
| 客户端越过等待阶段，episode 持续推进且无 `Caught exception` | 至少一次 observation/action 交换可以驱动控制循环 | 策略会成功完成 task |
| 已结束 episode 的日志和新 MP4 | 仿真、动作执行和视频写入闭环完成 | 整个 suite 已跑完或达到官方成功率 |
| action horizon 断言未触发 | 实际返回 chunk 至少覆盖 `replan_steps=5` | 动作质量或坐标语义一定最优 |

`Success: False` 不会自动使本章失败。只要教学 checkpoint 加载、客户端连接、actions 返回、episode 至少推进并留下日志/视频，且没有字段、shape、config 或 normalization 错误，工程验收即可成立。

## 常见问题

排错顺序统一为：保留现场 → 确认使用的 Compose project → 查看 service 状态 → 分开读服务端/客户端首个错误 → 再修改一个变量。共享方案中的命令必须补上同一个 `-p "openpi-tutorial-$USER-lora"`。

### 1. 容器内找不到 `pi0_libero_tutorial`

先确认宿主机当前目录与 Compose 挂载，而不是修改 `serve_policy.py`：

```bash
pwd -P
git rev-parse --show-toplevel
rg -n 'name="pi0_libero_tutorial"' src/openpi/training/config.py
docker compose -f examples/libero/compose.yml config | rg -n '/app|SERVER_ARGS'
docker compose -f examples/libero/compose.yml logs --tail=200 openpi_server
```

若 `$PWD:/app` 指向错误目录，回到仓库根目录重启正确 project。镜像运行时使用 bind-mounted `/app`；单纯重复 build 不会修复错误的启动目录。若 config 在宿主机本来就缺失或字段漂移，回第 08 章恢复准确内容。

### 2. checkpoint 路径或组件不存在

打印本章实际选择，不要凭记忆补 `/499`：

```bash
printf 'CHECKPOINT_STEP=%s\n' "${CHECKPOINT_STEP:-<unset>}"
printf 'CHECKPOINT_DIR=%s\n' "${CHECKPOINT_DIR:-<unset>}"
find checkpoints/pi0_libero_tutorial/libero_lora_demo \
  -mindepth 1 -maxdepth 2 -print | sort
docker compose -f examples/libero/compose.yml logs --tail=250 openpi_server
```

Policy loader 需要 step 目录下的 `params` 与 `assets/<asset_id>/norm_stats.json`；不要把 `.../params` 本身传给 `--policy.dir`。`train_state` 不参与推理，但缺失时说明该目录不符合本课程完整训练产物预检，应回第 09 章确认来源。

### 3. 参数 tree、key 或 shape 不匹配

优先核对服务日志中的 config/path，再重复实践 2 的静态断言。最常见错误是用 `pi05_libero` 创建模型图却加载 π₀ LoRA params，或模型与 freeze-filter variant 在训练后被修改。不要通过删除 key、改 checkpoint 文件或换成能“勉强 load”的 config 掩盖问题；恢复生成该 checkpoint 的精确 config。

### 4. normalization assets 或 action 语义错误

检查精确 asset id 和 checkpoint 文件：

```bash
CHECKPOINT_DIR="checkpoints/pi0_libero_tutorial/libero_lora_demo/${CHECKPOINT_STEP:-499}"
find "$CHECKPOINT_DIR/assets" -maxdepth 4 -type f -print | sort
rg -n 'repo_id="openpi-tutorial/libero-small"|extra_delta_transform=True' \
  src/openpi/training/config.py
docker compose -f examples/libero/compose.yml logs --tail=250 openpi_server
```

服务 normalization 来自 checkpoint assets，不应拿 `pi05_libero` 或其他数据的 stats 替换。即使 shape 相同，错误动作坐标系、delta 约定或夹爪语义也可能让仿真动作错误。

### 5. 模型加载、tokenizer 或 cache 下载失败

本地 params 不代表启动完全不联网。区分 Docker build 依赖、PaliGemma tokenizer GCS 和 cache 权限：

```bash
docker compose -f examples/libero/compose.yml logs --tail=300 openpi_server
docker compose -f examples/libero/compose.yml config | rg -n 'OPENPI_DATA_HOME|openpi_assets'
printf 'OPENPI_DATA_HOME=%s\n' "${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
df -h "${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
ls -ld "${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
```

根据首个 URL 与 traceback 区分 registry/apt/Python 依赖、`gs://big_vision/...tokenizer...`、磁盘和权限。`HF_HUB_OFFLINE=1` 不阻止 GCS；不要清空整个 cache 或把远端失败误判成 checkpoint 参数损坏。

### 6. 客户端一直等待，或 host 端口冲突

`depends_on` 只保证容器启动顺序，不保证模型 ready。客户端遇到 `ConnectionRefusedError` 会每 5 秒重试，且没有 `Connected` 日志。先检查：

```bash
docker compose -f examples/libero/compose.yml ps -a
docker compose -f examples/libero/compose.yml logs --tail=300 openpi_server runtime
ss -ltnp | rg ':8000\b'
```

host network 没有 published-port 映射，`docker ps` 的 Ports 为空也可能正常。若换成 18000，两侧参数和 `ss` 筛选必须一起改；命名项目的 `ps/logs/down` 也必须使用同一 `-p`。

### 7. action horizon、observation 字段或环境 action 报错

保留客户端 `Caught exception` 和服务端 traceback，再核对：

```bash
rg -n 'observation/image|observation/wrist_image|observation/state|prompt' examples/libero/main.py
rg -n 'action_horizon: int = 50' src/openpi/models/pi0_config.py
rg -n 'replan_steps|len\(action_chunk\)|action.tolist' examples/libero/main.py
rg -n 'class LiberoInputs|class LiberoOutputs|:7' src/openpi/policies/libero_policy.py
```

教学 config 应返回长度 50、每步 7D 的 LIBERO action。不要删除 horizon 断言；若 shape 正确但环境行为异常，再检查 config、norm stats 与 delta/absolute 动作语义。

### 8. Docker 容器看不到 GPU

分层检查宿主机、Docker runtime 和两个 service：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
docker compose -f examples/libero/compose.yml config | rg -n 'driver: nvidia|capabilities|count:'
docker compose -f examples/libero/compose.yml logs --tail=200 openpi_server runtime
```

独立 CUDA 容器失败时先修复 driver/NVIDIA Container Toolkit。不要通过修改模型、切 CPU 或反复删除镜像掩盖 Docker GPU 问题。

### 9. EGL、GLX 或 X11 渲染失败

基础 Compose 默认 `MUJOCO_GL=egl`，并硬编码 `PYOPENGL_PLATFORM=egl`。先收集：

```bash
docker compose -f examples/libero/compose.yml config \
  | rg -n 'MUJOCO_GL|PYOPENGL_PLATFORM|DISPLAY|MUJOCO_EGL_DEVICE_ID'
docker compose -f examples/libero/compose.yml logs --tail=300 runtime
```

只有证据指向 EGL 后端时，才按[第 02 章的完整 GLX override](02_environment_setup.md#egl-是-compose-默认值glx-是回退路径)同时把 `MUJOCO_GL` 与 `PYOPENGL_PLATFORM` 改为 `glx`，并处理 `DISPLAY`、X11 socket 和授权。只设置 `MUJOCO_GL=glx` 是不完整切换。

### 10. 视频目录不可写或没有新视频

检查 runtime 日志、宿主机路径和可写性：

```bash
docker compose -f examples/libero/compose.yml logs --tail=300 runtime
ls -ld data data/libero data/libero/videos 2>/dev/null
find data/libero/videos -maxdepth 1 -type f -name 'rollout_*.mp4' \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
```

视频只在 episode 结束路径写出；刚出现 `Starting episode...` 时没有 MP4 不一定是权限错误。若日志明确是 permission/disk 问题，只修复本章精确目录或改 `--args.video-out-path` 到明确可写的新目录，不要递归改写整个仓库所有权。

## 本章验收

进入第 11 章前，提交或展示以下证据：

- [ ] `CHECKPOINT_STEP` 来自完整 499 证据或另一个经验证的数字 step，没有伪造/复制 499；
- [ ] preflight 证明 config 唯一且仍是 π₀ 双 LoRA、horizon 50、教学 repo 与 `extra_delta_transform=True`；
- [ ] checkpoint step 中存在 `train_state`、`params` 和正确 asset id 的有限 norm stats；能说明推理不读取 `train_state`；
- [ ] `SERVER_ARGS` 使用 `pi0_libero_tutorial` 与匹配 step，未用 `pi05_libero` 加载教学 params；
- [ ] Compose 静态展开确认 trusted checkout 被挂载到 `/app`，并理解 privileged 与可写挂载风险；
- [ ] 默认或共享项目的服务端/客户端端口一致；`up`、`ps`、`logs`、`down` 使用同一 project name；
- [ ] 服务端成功加载教学 checkpoint，没有参数 tree/shape、config 或 norm stats 错误；
- [ ] 服务端出现 `Connection from ... opened`，或客户端越过等待进入 task/episode；没有把不存在的 `Connected` 日志当要求；
- [ ] 至少一个真实 policy action 请求成功，chunk 未触发 `len(action_chunk) < replan_steps`，episode 至少推进一个控制周期；
- [ ] 至少一个 episode 走到 success/failure 记录，且 `data/libero/videos/` 中有对应 MP4；
- [ ] 能说明 `num-trials-per-task=1` 仍遍历 suite 的多个 task；
- [ ] 明确知道教学 500-step checkpoint 任务失败不等于工程失败，本章没有官方 success-rate 要求；
- [ ] cleanup 没有删除 checkpoint、视频、源码或整个 cache，也没有对别人的 Compose project 执行 `down`。

## 思考题

### 1. 为什么推理需要 `params` 和 `assets`，却不需要 `train_state`？

<details>
<summary>参考答案（展开查看）</summary>

推理只需创建模型图、载入参数、完成输入归一化和输出反归一化。optimizer、训练 step 等状态只用于继续训练，所以 `create_trained_policy()` 直接读取 `<step>/params` 与 `<step>/assets`，不会调用训练恢复函数加载 `train_state`。

</details>

### 2. 为什么显式 `policy:checkpoint` 时保留 `--env LIBERO` 仍不会切回默认 π₀.₅？

<details>
<summary>参考答案（展开查看）</summary>

Tyro 已把 policy 解析成 `Checkpoint` 后，`create_policy()` 直接使用 `--policy.config` 与 `--policy.dir`。只有 policy 是 `Default` 时才根据 `args.env` 查询默认 checkpoint。因此这里的 `--env LIBERO` 是无害环境标签，真正选择教学策略的是显式 config/dir。

</details>

### 3. 为什么 `--args.num-trials-per-task 1` 不等于只运行一个 episode？

<details>
<summary>参考答案（展开查看）</summary>

外层循环遍历 suite 的所有 task，内层循环才按 `num_trials_per_task` 为每个 task 创建 episode。设为 1 只把每个 task 的重复次数降为一次；`libero_spatial` 仍包含多个 task。

</details>

### 4. 为什么 `Success: False` 仍可能通过本章验收？

<details>
<summary>参考答案（展开查看）</summary>

任务成功率衡量策略质量，而本章先验收 config/checkpoint 加载、WebSocket、observation/action、仿真推进和视频写入。500-step 小数据 LoRA 可能无法完成任务；只要接口闭环正确且证据完整，任务失败不等于服务失败。

</details>

### 5. 为什么不能把 `pi05_libero` 的 config 与教学 checkpoint 组合？

<details>
<summary>参考答案（展开查看）</summary>

config 决定模型图、参数树、state/token 处理、horizon 和 transforms；checkpoint 保存的是与生成它的图和动作语义配套的值与 stats。π₀.₅ config 与 π₀ 双 LoRA checkpoint 描述不同策略，混用会造成参数/shape 错误或更隐蔽的 normalization、动作语义错误。

</details>

[← 上一章：运行短程 LoRA 微调](09_run_finetuning.md) · [课程入口：README](README.md) · [下一章：规划后续实验 →](11_next_steps.md)
