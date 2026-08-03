# Dobot Nova2 π0.5 Docker 训练记录

本文记录从 Docker 网络环境配置、镜像构建、LeRobot v3 数据接入，到 Dobot Nova2 的 π0.5 action-only 微调、指标记录和恢复训练的完整流程。后续训练状态和问题继续追加到文末的“运行记录”。

## 1. 训练目标

- 数据集：`/home/ps/DZK_repos/dobot/dobot_nova2_ros2_driver/data_collection/lerobot_tape_pi05`
- 基座：π0.5 base
- 冻结：SigLIP vision encoder 和 PaliGemma language backbone
- 训练：π0.5 action expert、action input/output projection、time MLP
- 内部 action 维度：32（必须与预训练 checkpoint 一致）
- Dobot action 维度：7；输入时填充至 32，推理输出时截取前 7 维
- action horizon：16
- batch size：8
- 训练步数：20,000
- optimizer：AdamW，gradient clipping 1.0
- learning rate：1,000 步 warmup 至 `5e-5`，20,000 步 cosine decay 至 `5e-6`
- EMA：关闭
- W&B：关闭，使用本地 CSV 和 PNG 曲线
- checkpoint：每 1,000 步保存，5,000 步的倍数长期保留

训练配置名：

```text
pi05_dobot_tape_action_only
```

实验名：

```text
dobot_tape_action_only_v1
```

## 2. Docker 网络问题和最终配置

最初 `opid-build` 分别在 Docker Hub Dockerfile frontend、NVIDIA CUDA 镜像和 GHCR uv 镜像处出现 `TLS handshake timeout`、`EOF`。根因是 Docker daemon 代理端口和 shell 中大小写代理变量不一致，同时 Go HTTP/2 经当前代理连接 registry 不稳定。

Docker systemd drop-in 最终为 `/etc/systemd/system/docker.service.d/http-proxy.conf`：

```ini
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:17891"
Environment="HTTPS_PROXY=http://127.0.0.1:17891"
Environment="NO_PROXY=localhost,127.0.0.1,host.docker.internal"
Environment="GODEBUG=http2client=0"
```

`/etc/docker/daemon.json` 中保留 NVIDIA runtime，并设置：

```json
{
  "runtimes": {
    "nvidia": {
      "args": [],
      "path": "nvidia-container-runtime"
    }
  },
  "proxies": {
    "http-proxy": "http://127.0.0.1:17891",
    "https-proxy": "http://127.0.0.1:17891",
    "no-proxy": "localhost,127.0.0.1,host.docker.internal"
  }
}
```

注意：最终没有把 `ghcr.io` 放进 `no-proxy`。当前环境中 GHCR 直连会出现 `EOF`，应通过代理访问。

修改后执行：

```zsh
sudo systemctl daemon-reload
sudo systemctl restart docker
systemctl show docker --property=Environment
docker info | sed -n '/Proxy/,+3p'
```

分别预拉取并验证 registry：

```zsh
docker pull docker/dockerfile:1.7
docker pull ghcr.io/astral-sh/uv:0.5.1
docker pull nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
```

仓库的 `functions.zsh` 已调整代理优先级：显式的 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 优先于可能残留且指向旧端口 7890 的小写变量。

使用国内 apt/PyPI 镜像构建：

```zsh
source "$PWD/functions.zsh"
export OPENPI_UBUNTU_APT_MIRROR=https://mirrors.aliyun.com/ubuntu
export OPENPI_UBUNTU_SECURITY_MIRROR=https://mirrors.aliyun.com/ubuntu
export OPENPI_UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
opid-build
```

`openpi:dev` 镜像已成功构建。

## 3. 数据集检查

当前数据是 LeRobot v3.0：

- 31 episodes
- 3,453 frames
- 10 FPS
- task：`pick up the tape roll`
- `observation.images.global`：480×640 RGB video
- `observation.images.wrist`：530×848 RGB video
- `observation.state`：13 维
- `action`：7 维

state 的 13 维依次为 6 个关节位置、TCP 的 6 维 pose、夹爪开度。训练输入选取 6 个关节位置和夹爪开度，共 7 维；TCP pose 不进入 state token。

action 的 7 维依次为归一化的 `x/y/z/rx/ry/rz` jog command 和夹爪目标。当前数据的 `rx/ry/rz` 始终为 0，因此本数据不能教会模型旋转；其余平移和夹爪维度有正常变化。

未来数据继续使用 LeRobot v3.0，并保持字段名、维度、单位和 action 顺序不变。不要为了当前 OpenPI 锁定的旧 LeRobot Python 依赖而降级采集格式。

## 4. LeRobot v3 本地适配

OpenPI 锁定的 LeRobot commit 不能直接读取当前 v3.0 数据。直接升级到较新的 LeRobot 会与仓库锁定的 NumPy、Transformers 和 Hugging Face Hub 版本冲突，因此增加了只读本地适配层：

- `src/openpi/training/lerobot_v3_dataset.py`
- 使用 PyArrow 读取 Parquet
- 使用 TorchCodec 随机读取 MP4/AV1 视频帧
- 按 episode 边界构造 16 帧 action chunk
- episode 尾部使用最后一个 action 补齐 horizon
- 归一化统计阶段返回占位图像，避免无意义的视频解码

数据集通过 `compose.dobot.yaml` 只读挂载：

```yaml
services:
  openpi:
    volumes:
      - type: bind
        source: ${OPENPI_DOBOT_DATASET_DIR:?set_OPENPI_DOBOT_DATASET_DIR}
        target: /datasets/lerobot_tape_pi05
        read_only: true
```

## 5. 冻结参数核验

实际枚举得到 19 个可训练参数叶，只包括：

- `PaliGemma/llm/..._1...`：π0.5 action expert
- `action_in_proj`
- `action_out_proj`
- `time_mlp_in`
- `time_mlp_out`

SigLIP 路径和没有 `_1` 后缀的 PaliGemma language backbone 路径均被冻结，并在训练 state 中转为 bfloat16。action expert 和 projection head 保持 float32 训练。

## 6. 启动 GPU 容器

每个新 shell 先设置数据集路径：

```zsh
export OPENPI_DOBOT_DATASET_DIR=/home/ps/DZK_repos/dobot/dobot_nova2_ros2_driver/data_collection/lerobot_tape_pi05
```

启动：

```zsh
docker compose \
  -f compose.yaml \
  -f compose.gpu.yaml \
  -f compose.dobot.yaml \
  up -d openpi
```

验证 GPU：

```zsh
docker compose \
  -f compose.yaml \
  -f compose.gpu.yaml \
  -f compose.dobot.yaml \
  exec openpi nvidia-smi
```

已验证 GPU 0 为 NVIDIA RTX 5000 Ada，显存约 32 GB。

## 7. 归一化统计

在容器中运行：

```zsh
docker compose \
  -f compose.yaml \
  -f compose.gpu.yaml \
  -f compose.dobot.yaml \
  exec openpi zsh -lc \
  'source /usr/local/share/openpi/functions.zsh && opic-norm --config-name pi05_dobot_tape_action_only'
```

输出位置：

```text
/home/ps/DZK_repos/openpi-docker-data/assets/
  pi05_dobot_tape_action_only/local/dobot_nova2_tape_pi05/norm_stats.json
```

已完成全部 3,453 帧统计。旋转 action 三轴的 `std` 和分位范围为 0；quantile normalize 使用 `1e-6` 防止除零，因此不会产生 NaN，但这些轴只能学习恒定的零旋转命令。

## 8. 启动训练

后台训练命令：

```zsh
docker compose \
  -f compose.yaml \
  -f compose.gpu.yaml \
  -f compose.dobot.yaml \
  exec -d -T openpi zsh -lc \
  'source /usr/local/share/openpi/functions.zsh && \
   opic-train pi05_dobot_tape_action_only \
     --exp-name=dobot_tape_action_only_v1 \
     > /workspace/wandb/dobot_tape_action_only_v1.log 2>&1'
```

只有明确需要清空同名实验时才加 `--overwrite`。已有 checkpoint 需要继续训练时使用：

```zsh
opic-train pi05_dobot_tape_action_only \
  --exp-name=dobot_tape_action_only_v1 \
  --resume
```

## 9. 指标、曲线和 checkpoint

训练器每 100 步保存一次本地指标：

```text
/home/ps/DZK_repos/openpi-docker-data/checkpoints/
  pi05_dobot_tape_action_only/dobot_tape_action_only_v1/
    metrics.csv
    training_curves.png
```

`metrics.csv` 字段：

```text
step,loss,grad_norm,param_norm,learning_rate
```

`training_curves.png` 包含 loss、gradient norm、parameter norm 和 learning rate 四张曲线，并在训练中持续更新。

完整文本日志：

```text
/home/ps/DZK_repos/openpi-docker-data/wandb/dobot_tape_action_only_v1.log
```

实时查看：

```zsh
tail -f /home/ps/DZK_repos/openpi-docker-data/wandb/dobot_tape_action_only_v1.log
```

checkpoint 根目录：

```text
/home/ps/DZK_repos/openpi-docker-data/checkpoints/
  pi05_dobot_tape_action_only/dobot_tape_action_only_v1/
```

π0.5 base 权重首次下载约 11.6 GiB，缓存位置：

```text
/home/ps/DZK_repos/openpi-docker-data/cache/openpi/openpi-assets/checkpoints/pi05_base/params
```

缓存已完成，后续启动不会重复下载。

## 10. 本次修改的仓库文件

- `functions.zsh`：修正 shell 代理变量优先级
- `compose.dobot.yaml`：只读挂载 Dobot 数据集
- `src/openpi/policies/dobot_policy.py`：Dobot state/image/action 映射
- `src/openpi/training/lerobot_v3_dataset.py`：LeRobot v3 本地只读适配
- `src/openpi/training/data_loader.py`：接入本地 v3 dataset
- `src/openpi/training/config.py`：action-only π0.5 配置和冻结规则
- `scripts/compute_norm_stats.py`：统计时跳过视频解码
- `scripts/train.py`：本地 CSV 指标和 PNG 训练曲线
- `docs/dobot_pi05_docker_training.md`：本文档

## 11. 运行记录

### 2026-07-29：环境和数据接入

- 修复 Docker daemon 的 17891 代理和 Go HTTP/2 registry 问题。
- 成功拉取 Dockerfile frontend、GHCR uv 和固定 digest 的 CUDA 镜像。
- `opid-build` 成功生成 `openpi:dev`。
- 确认数据为 LeRobot v3.0，并实现 PyArrow + TorchCodec 本地适配。
- 验证单样本、双相机视频随机读取、16×7 action chunk 和 prompt。
- 验证冻结后只有 19 个 action 相关参数叶可训练。
- 完成 3,453 帧归一化统计。

### 首次训练尝试：action 维度修正

- 最初把模型内部 `action_dim` 设为 7。
- π0.5 base checkpoint 的 `action_in_proj/kernel` 是 `(32, 1024)`，导致与 7 维模型的 `(7, 1024)` 形状不匹配。
- 修正为内部 32 维；Dobot 7 维 action 在模型变换阶段填充到 32，输出再截回 7。
- 修正后成功加载 12.5 GiB checkpoint，数据 batch 形状为 state `(8, 32)`、action `(8, 16, 32)`，并开始执行训练 step。

### 指标记录增强

- 发现禁用 W&B 时原始训练器没有可靠的本地结构化指标文件。
- 增加 `metrics.csv`、`training_curves.png` 和文本日志中的指标输出。
- 临时测试已验证 CSV 内容正确，PNG 可成功生成。

### 正式训练启动

- 启用本地指标记录后重新从 step 0 启动 `dobot_tape_action_only_v1`。
- Step 0：loss `0.131098`，gradient norm `0.548557`，learning rate `4.99494e-08`。
- Step 100：loss `0.101450`，gradient norm `0.427497`，learning rate `5.04496e-06`。
- Step 200：loss `0.031914`，gradient norm `0.255380`，learning rate `1.00400e-05`。
- 前 200 步 loss 和 gradient norm 均为有限值且下降，没有发现 NaN/OOM。
- 稳态速度约 `1.0–1.2 s/step`，按当前速度预计约 6 小时完成 20,000 步。
- GPU 显存约 `30.3/32.8 GB`，利用率 100%，观测温度 79°C。
- 已人工检查 `training_curves.png`，四个 panel 均可正常读取并包含 step 0 和 step 100 数据点。
- 训练保持后台运行。

### 2026-07-30 14:15：中途状态复核

- 训练并未完成；现场日志约为 `12,400/20,000`，训练进程仍在运行，GPU 利用率 100%。
- 最新结构完整的 checkpoint 为 step 12,000；同时按 `keep_period=5,000` 保留了 step 5,000 和 10,000。
- 每个完整 checkpoint 约 8.8 GiB，均包含 `params`、`train_state`、`assets` 和 Dobot normalization stats。
- 已记录 125 个指标点。step 0 loss 为 `0.131098`；截至本次检查，最低 loss 为 step 12,200 的 `0.007720`，step 12,400 loss 为 `0.008244`。
- loss 曲线先快速下降，再平稳缓慢下降；gradient norm 有小幅波动但保持有限，没有 NaN、Inf、OOM 或异步 checkpoint 错误。
- 当时预计剩余约 2 小时 17 分钟。只有日志完成到 step 19,999、最终 checkpoint 写入完成并且训练进程退出，才算 20,000 步训练完成。
- 部署前修正了 Dobot inference observation：真实推理不需要提供 ground-truth `actions`；训练样本仍可提供 actions。新增的两项单元测试均通过。

### 2026-07-30：checkpoint 磁盘空间故障和恢复

- 训练在 step 18,000 附近退出，根因是 step 17,000 异步 checkpoint 写入时出现 `RESOURCE_EXHAUSTED` / `OS error 28: ENOSPC No space left on device`。
- 这不是 GPU OOM，也不是 loss NaN；step 16,000 checkpoint 完整可读，step 17,000 只有不完整的 `17000.orbax-checkpoint-tmp-64` 临时目录。
- 删除了这个已确认损坏的 3.7 GiB 临时目录；当时检查宿主机 checkpoint 文件系统已有约 202 GiB 可用空间。
- 训练器增加 resume 时的本地指标截断逻辑，防止把没有对应 checkpoint 的 step 17,000–18,000 指标混入恢复后的曲线。
- `scripts/train_test.py::test_truncate_local_metrics` 通过，Ruff 检查通过。
- 已使用 `--resume` 成功从 step 16,000 恢复，日志确认参数和 optimizer state 均从该 checkpoint 读取，随后继续进入 `16.0kit/20.0kit`。
- 恢复后的指标文件已保留 step 0 到 16,000；最终仍需完成 step 19,999 并写入最终 checkpoint。
- 恢复后的 step 16,100 loss 为 `0.006883`、gradient norm 为 `0.065171`，训练继续正常推进。

## 12. 训练完成后的验收和部署顺序

### 12.1 确认真正完成

本训练循环执行 step 0 到 19,999，因此最终目录名应为 `19999`，不是 `20000`。验收时同时检查：

```zsh
tail -n 80 /home/ps/DZK_repos/openpi-docker-data/wandb/dobot_tape_action_only_v1.log

find /home/ps/DZK_repos/openpi-docker-data/checkpoints/\
pi05_dobot_tape_action_only/dobot_tape_action_only_v1 \
  -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort -n
```

必须看到以下证据：

- 进度到 `20.0kit/20.0kit`；
- 日志出现 `Waiting for checkpoint manager to finish`；
- step `19999` 的异步写入完成且没有 background save error；
- `19999/{params,train_state,assets}` 全部存在；
- 训练 Python 进程已经退出。

### 12.2 启动最终 checkpoint 的 Policy Server

训练结束、GPU 显存释放后，从仓库根目录执行：

```zsh
export OPENPI_DOBOT_DATASET_DIR=/home/ps/DZK_repos/dobot/dobot_nova2_ros2_driver/data_collection/lerobot_tape_pi05

docker compose \
  -f compose.yaml \
  -f compose.gpu.yaml \
  -f compose.dobot.yaml \
  exec -d -T openpi zsh -lc \
  'source /usr/local/share/openpi/functions.zsh && \
   opic-serve --port=8000 --record \
     policy:checkpoint \
     --policy.config=pi05_dobot_tape_action_only \
     --policy.dir=/workspace/checkpoints/pi05_dobot_tape_action_only/dobot_tape_action_only_v1/19999 \
     > /workspace/wandb/dobot_tape_action_only_v1_serve.log 2>&1'
```

服务端通过宿主机端口 8000 提供 WebSocket inference。`--record` 会保留请求和输出，便于首次联调定位字段或尺度问题。

### 12.3 先做离线 inference smoke test

不要直接让机器人运动。先从数据集中取真实观测请求 Policy Server，并验证：

- 输入 `state` 是原始 13 维，顺序与采集元数据一致；
- `global_image`、`wrist_image` 是 HWC uint8；
- prompt 使用 `pick up the tape roll`；
- 返回 `actions.shape == (16, 7)`；
- 所有 action 有限且反归一化后尺度合理；
- 第 4–6 维旋转 action 应接近 0，因为本训练集没有旋转示范。

Policy Server 请求使用的 Dobot observation schema：

```python
observation = {
    "state": state_13d,
    "global_image": global_image_uint8,
    "wrist_image": wrist_image_uint8,
    "prompt": "pick up the tape roll",
}
```

### 12.4 再接 ROS 2，并限制首次上机风险

机器人端安装 `packages/openpi-client`，通过 `WebsocketClientPolicy(host=<训练机IP>, port=8000)` 请求 action chunk。首次上机必须：

- 保留急停、关节/TCP workspace limit 和碰撞保护；
- 对六维 jog command 再做 `[-1, 1]` clamp；
- 使用低速度比例；
- 第一次只执行 action chunk 的 1–4 步，然后重新规划，不要直接开环执行全部 16 步；
- 先空载、远离桌面和相机支架验证坐标符号，再尝试胶带抓取；
- 单独核对夹爪第 7 维的 0/1 方向和物理开合方向。

### 12.5 不只看训练 loss，比较 checkpoint 的真实成功率

31 个 episode 全部参与了训练，目前没有独立 validation set，因此较低 training loss 不能证明真实抓取成功。建议至少比较 step 10,000、15,000 和 19,999：

- 每个 checkpoint 在未参与采集布置的胶带位置上执行 10–20 次；
- 记录抓取成功率、碰撞/越界次数、完成时间和人工接管次数；
- 选择真实成功率最好且最稳定的 checkpoint，而不是机械地选择训练 loss 最低者。

如果任务需要腕部旋转，必须补采包含非零 `rx/ry/rz` action 的示范。数据格式仍保持 LeRobot v3.0，不需要换版本。

## 13. 最终完成验收

### 2026-07-30：训练完成

- 恢复训练从 step 16,000 继续到 step 19,999，最终进度达到 `20.0kit/20.0kit`。
- 最终 checkpoint：
  `checkpoints/pi05_dobot_tape_action_only/dobot_tape_action_only_v1/19999`
- 最终 checkpoint 约 8.8 GiB，其中 `params` 约 5.8 GiB、`train_state` 约 3.0 GiB，`assets/local/dobot_nova2_tape_pi05/norm_stats.json` 存在。
- Orbax 日志确认 step 19,999 异步保存完成，`No errors found in background save thread`，训练 Python 进程已退出。
- 恢复阶段之后没有新的 traceback、ENOSPC、NaN、Inf 或 OOM。
- 指标文件共 200 个记录点（step 0 到 19,900）。全程最低记录 loss 为 step 19,000 的 `0.005509`；最后记录 step 19,900 loss 为 `0.006732`，gradient norm 为 `0.096269`。
- 最终 checkpoint 已做一次真实数据帧离线推理：输出 shape 为 `(16, 7)`，全部有限；首次 JIT 推理约 8.3 秒；旋转三轴最大绝对值约 `3.6e-9`，与数据中旋转 action 恒为 0 一致。
- 该次推理输出最小/最大值约 `-1.006/1.004`，上机执行前仍需按机器人安全边界 clamp，不能直接把网络输出当作已限幅的物理命令。

### 2026-07-31：Policy Server 全链路离线验收

- 使用最终 step 19,999 checkpoint 启动带 `--record` 的 Policy Server，日志确认参数和 checkpoint 内的 Dobot normalization stats 均加载成功。
- 仅从容器本机 `127.0.0.1` 建立 WebSocket 客户端连接，输入来自数据集的真实观测帧；本次检查没有连接 ROS 2、机器人控制器或发送任何真机命令。
- 客户端提交原始 13 维 state、两路 HWC uint8 图像和任务 prompt；服务记录保存的是 policy transform 后的 13 维 state 及 CHW 图像。
- 三份记录的输出均为 `(16, 7)` 且全部有限。首次 JIT 模型推理约 `8001 ms`；暖态模型推理约 `74 ms`，对应客户端 WebSocket 往返约 `115–131 ms`。
- 三份记录中的输出总体范围达到约 `[-1.006, 1.026]`，再次证明真机适配器必须在执行前对 jog action 做安全限幅，不能假设模型输出天然位于 `[-1, 1]`。
- 请求、输入和输出记录保存在仓库的 `policy_records/step_0.npy` 到 `step_2.npy`，每份记录包含两路图像、state、prompt、actions 和 policy inference timing。
- 验收完成后已向临时 Policy Server 发送 `TERM`。复核确认 `serve_policy.py` 和 GPU compute process 均已退出，GPU 显存从约 24.5 GiB 回落至约 1.1 GiB。

至此，`真实数据帧 -> WebSocket 客户端 -> Policy Server -> 最终 checkpoint -> 16x7 action chunk` 的离线链路已经通过。下一阶段才是 ROS 2 真机适配和低风险联调，仍需执行 12.4 节中的限幅、低速、短 chunk、急停和坐标方向检查。

## 14. 2026-07-31：ServoP 数据重新训练

### 14.1 数据验收

本轮训练只使用 ServoP 重新采集并接受的数据：

```text
/home/ps/DZK_repos/dobot/dobot_nova2_ros2_driver/data_collection_servo_p/lerobot_pi05_servo_p
```

使用 Dobot 驱动仓库的 `make servo-data-lerobot-validate` 验证通过。数据集为 LeRobot v3.0，`repo_id` 为
`local/dobot_nova2_tape_pi05_servo_p`，包含 42 个 accepted episodes、4,389 帧，采样率 10 FPS；state 为
13 维，action 为 7 维 ServoP，两路图像分别为 wrist `3x530x848` 和 global `3x480x640`。原始采集目录中的
8 个 rejected episode 和 1 个 incomplete episode 没有进入训练集。

ServoP action 的物理语义与上一轮 MoveJog action 不同，因此本轮没有复用旧数据集、旧 normalization stats
或旧训练 checkpoint。

### 14.2 训练配置

新增独立配置 `pi05_dobot_tape_servo_p_action_only`，实验名为
`dobot_tape_servo_p_action_only_v1`。配置沿用上一轮 action-only 方案：

- 以 pi0.5 base checkpoint 初始化，内部 action dimension 为 32，action horizon 为 16；
- batch size 为 8；
- 冻结 SigLIP vision encoder 和 PaliGemma language backbone；
- 只训练 action expert、action projection 和 timestep MLP；
- AdamW，gradient clipping 为 1.0；
- cosine learning rate，warmup 1,000 步，peak LR `5e-5`，final LR `5e-6`；
- EMA 和 W&B 均关闭；
- 每 1,000 步保存 checkpoint，每 5,000 步长期保留；
- 为覆盖约 10 小时训练窗口，本轮总步数和 LR decay steps 均设为 30,000。

新 normalization stats 已根据全部 4,389 帧单独计算，位置为：

```text
/home/ps/DZK_repos/openpi-docker-data/assets/pi05_dobot_tape_servo_p_action_only/local/dobot_nova2_tape_pi05_servo_p/norm_stats.json
```

### 14.3 启动与监控

Docker 容器的数据集挂载变量必须指向新的 ServoP LeRobot 目录：

```zsh
export OPENPI_DOBOT_DATASET_DIR=/home/ps/DZK_repos/dobot/dobot_nova2_ros2_driver/data_collection_servo_p/lerobot_pi05_servo_p
```

训练通过以下命令在 Docker 容器中后台启动：

```zsh
opic-train pi05_dobot_tape_servo_p_action_only \
  --exp-name=dobot_tape_servo_p_action_only_v1
```

训练日志、结构化指标、曲线和 checkpoint 分别位于：

```text
/home/ps/DZK_repos/openpi-docker-data/wandb/dobot_tape_servo_p_action_only_v1.log
/home/ps/DZK_repos/openpi-docker-data/checkpoints/pi05_dobot_tape_servo_p_action_only/dobot_tape_servo_p_action_only_v1/metrics.csv
/home/ps/DZK_repos/openpi-docker-data/checkpoints/pi05_dobot_tape_servo_p_action_only/dobot_tape_servo_p_action_only_v1/training_curves.png
/home/ps/DZK_repos/openpi-docker-data/checkpoints/pi05_dobot_tape_servo_p_action_only/dobot_tape_servo_p_action_only_v1/
```

启动后检查结果：

- step 0：loss `0.118185`，gradient norm `0.641036`，learning rate `4.99494e-08`；
- step 100：loss `0.100455`，gradient norm `0.422921`，learning rate `5.04496e-06`；
- 训练速率稳定在约 `1.0–1.1 s/step`，预计约 9 小时完成；
- 启动时 GPU 利用率 100%，显存约 30.4/32.8 GiB；
- 当前未发现 traceback、OOM、NaN 或 Inf。

### 14.4 完成验收

本训练循环执行 step 0 到 29,999，因此最终 checkpoint 目录应为 `29999`。训练完成后必须同时确认：

- 日志进度到 `30.0kit/30.0kit`；
- 日志出现 checkpoint manager 等待和 step 29,999 保存完成记录；
- `29999/{params,train_state,assets}` 均完整存在；
- 后台 checkpoint 保存线程没有错误，训练进程已经退出；
- `metrics.csv` 和 `training_curves.png` 完整生成；
- 全程没有 ENOSPC、OOM、NaN、Inf 或 traceback。

在完成这些检查及真实数据帧的离线 inference smoke test 前，不把新 checkpoint 接入真机运动链路。

### 14.5 训练完成与部署验收

训练在 2026-07-31 完成全部 30,000 步，总耗时约 8 小时 38 分。最终 checkpoint 为：

```text
/home/ps/DZK_repos/openpi-docker-data/checkpoints/
  pi05_dobot_tape_servo_p_action_only/dobot_tape_servo_p_action_only_v1/29999
```

Orbax 日志确认 step 29,999 异步保存完成，后台保存线程没有错误。长期保留的 checkpoint 为
5,000、10,000、15,000、20,000、25,000 和 29,999。最后记录的 step 29,900 loss 为
`0.00519082`，gradient norm 为 `0.085444`，没有发现 OOM、NaN、Inf、ENOSPC 或训练 traceback。

2026-08-01 使用真实双相机和机器人状态完成 dry-run 部署验收。Policy Server 正确加载 step
29,999 和 ServoP 专用 normalization stats；首次 JIT inference 约 8.24 秒，暖态约 109 ms，
返回的 action shape 为 `(16,7)` 且全部有限。dry-run 没有使能或移动机械臂。

部署验收同时发现训练前的数据检查只验证了 LeRobot schema、shape 和数值有限性，没有验证
action 与物理状态的时间语义。42 个 accepted episode 的真实夹爪均从 95 mm 打开状态开始，
但其中 38 个 episode 的第七维 action 从第 0 帧就错误标为关闭；只有 4 个 episode 正确记录
“先打开、再关闭”。因此 v1 checkpoint 会在开始时预测关闭夹爪，当前禁止 armed 实机运行。

根因是采集节点自动打开夹爪并归位后，手柄节点保留上一轮的关闭目标。Dobot 手柄节点已改为
在 Start 成功时从夹爪反馈重新同步二值 action target，防止后续数据继续出现相同问题。现有
数据仍需重标或重采，并重新训练新的 checkpoint。以后的数据验收必须增加以下语义检查：

- 每个 episode 的初始 gripper observation 和 gripper target 必须一致；
- 打开到关闭的 action 跳变应领先物理 opening 下降一个合理的执行延迟；
- 输出每个 episode 的首次开/关索引及全数据集开关比例，不能只看 min/max/quantile；
- dry-run 必须验证初始 action 与任务阶段相符，不能仅验证 shape 和 finite。

## 15. 2026-08-03：Pen-box ServoP v2 长时训练（当前有效）

本轮只使用新的 pen-box ServoP v2 数据，不混用 tape、MoveJog 或 ServoP v1 数据：

```text
/home/ps/DZK_repos/dobot/dobot_nova2_ros2_driver/data/collections/servo_p_v2/lerobot_pi05_servo_p_v2
```

数据验收结果：LeRobot v3.0，108 个 accepted episode、23,047 帧、10 FPS；state 13 维、action
7 维，包含腕部和全局两路 RGB 图像。每个 episode 的夹爪语义为“起始打开、接近目标时关闭、任务
结束时打开”，并已检查 action 与实际夹爪反馈的时间顺序。

训练配置和实验：

```text
config:  pi05_dobot_pen_box_servo_p_action_only
task:    pick the pen and put it in the box
exp:     dobot_pen_box_servo_p_action_only_v1_long
steps:   1,000,000（手动停止）
batch:   8
```

SigLIP vision encoder 和 PaliGemma language backbone 冻结，只训练 action expert、action 投影层
和 timestep MLP。训练使用独立 normalization stats，所有 loss、gradient norm 和学习率均写入
本地 CSV/PNG；W&B 只保留兼容目录，不作为训练成功判据。

本次验收采用的 checkpoint 为 step `135000`：

```text
/home/ps/openpi-docker-data/checkpoints/
  pi05_dobot_pen_box_servo_p_action_only/
  dobot_pen_box_servo_p_action_only_v1_long/135000
```

结构化指标、曲线和日志：

```text
/home/ps/openpi-docker-data/checkpoints/pi05_dobot_pen_box_servo_p_action_only/
  dobot_pen_box_servo_p_action_only_v1_long/metrics.csv
  dobot_pen_box_servo_p_action_only_v1_long/training_curves.png
/home/ps/openpi-docker-data/wandb/dobot_pen_box_servo_p_action_only_v1_long.log
```

验收结果：checkpoint 的 `params`、`train_state` 和 `assets` 完整；loss/gradient norm 全程有限；
Policy Server 离线 smoke test 返回有限的 `(16, 7)` action；暖态 WebSocket 推理约 100 ms；Dobot
真机已完成回起点、夹爪初始化、ServoP 运动和夹爪动作的闭环验证。当前实际启动入口统一放在
Dobot 仓库，见其 README 的“π0.5 训练与实机部署”章节。
