# 第 03 章：第一次 LIBERO 预训练推理

[← 上一章：环境安装与双运行时自检](02_environment_setup.md) · [课程入口：README](README.md) · [下一章：openpi 架构与一次请求的数据流 →](04_openpi_architecture.md)

## 本章目标

完成本章后，你应该能够：

- 用仓库推荐的 Docker Compose 工作流同时启动 LIBERO 仿真客户端和 openpi Policy Server；
- 说明 `runtime` 与 `openpi_server` 两个 service 的职责，以及 observation 和 action 如何通过 WebSocket 交换；
- 知道 `--env LIBERO` 为什么会加载 `pi05_libero` 与官方预训练 checkpoint；
- 从 `examples/libero/main.py` 中找出图像、机器人状态、语言指令和 action queue 的构造过程；
- 用日志、容器状态和回放视频判断闭环是否真的开始运行；
- 区分首次镜像构建、checkpoint 下载、JIT 编译的正常等待与真正的故障。

本章只运行仓库默认的 LIBERO **预训练推理**：配置是 `pi05_libero`，模型属于 π₀.₅。后续 LoRA 教学会创建 `pi0_libero_tutorial`，它属于 π₀；两套配置及 checkpoint 不能混用。遇到陌生术语可查后续补齐的[课程术语表](glossary.md)，跨章节问题可查后续补齐的[故障排查手册](troubleshooting.md)。

## 前置条件

开始前应已完成[第 02 章](02_environment_setup.md)的环境检查，并确认：

- 当前目录是 openpi 仓库根目录；
- `git submodule status` 中课程所需子模块已初始化，并位于父仓库记录的 commit；
- `docker compose version` 可用；
- 宿主机与 Docker 容器都能访问 NVIDIA GPU；
- 仓库所在文件系统、Docker 数据目录和 `${OPENPI_DATA_HOME:-$HOME/.cache/openpi}` 有足够空间；
- 网络能够获取 Docker 镜像、Python 构建依赖和 `gs://openpi-assets/checkpoints/pi05_libero`；
- 宿主机的 `8000` 端口未被其他服务占用。

本章命令会构建两个镜像、下载模型并启动多轮仿真，时间和磁盘开销都明显大于上一章的静态检查。共享服务器上运行前，先确认 GPU、端口和 Docker 使用约定。

> **安全边界：**当前 Compose 为 `runtime` 设置了 `privileged: true`，并把当前仓库通过 `$PWD:/app` **可写挂载**到 `runtime` 与 `openpi_server` 两个容器。容器会直接执行该 checkout 中的脚本，因此只应在获准使用 Docker/GPU 的主机上运行你信任的仓库内容。共享 Docker host 上还必须先协调唯一的 Compose project name 和 host 端口；执行 `up`、`ps`、`logs` 或 `down` 时始终复用同一个 project name，避免与别人的容器重名、占用同一 host 端口，或对错误的 Compose 项目执行 `down`。

## 核心概念

### 推荐的双服务闭环

[`examples/libero/compose.yml`](../../examples/libero/compose.yml) 定义了两个相互配合的 service：

| service | 容器内启动内容 | 职责 |
| --- | --- | --- |
| `openpi_server` | [`scripts/serve_policy.py`](../../scripts/serve_policy.py) | 加载模型与 checkpoint，接收 observation，运行策略推理并返回 action chunk |
| `runtime` | [`examples/libero/main.py`](../../examples/libero/main.py) | 启动 LIBERO/MuJoCo 环境，用轻量 `openpi-client` 发送 observation、接收动作、推进 episode 并保存视频 |

模型服务镜像的 [`serve_policy.Dockerfile`](../../scripts/docker/serve_policy.Dockerfile) 最终执行 `scripts/serve_policy.py $SERVER_ARGS`。当 `SERVER_ARGS="--env LIBERO"` 且没有显式指定其他 policy 时，[`EnvMode.LIBERO`](../../scripts/serve_policy.py) 会映射到：

```text
config:     pi05_libero
checkpoint: gs://openpi-assets/checkpoints/pi05_libero
```

`runtime` 镜像则运行 `examples/libero/main.py $CLIENT_ARGS`。它不加载 JAX 模型，而是使用轻量的 [`WebsocketClientPolicy`](../../packages/openpi-client/src/openpi_client/websocket_client_policy.py) 连接服务端。

两个 service 都使用 `network_mode: host`。服务端默认监听 `0.0.0.0:8000`，客户端默认连接 `0.0.0.0:8000`；在 Linux 的 host network 下，它们共享宿主机网络命名空间，因此通过同一个 `8000` 端口建立 WebSocket 连接。客户端把 observation 编码后发送，服务端的 [`WebsocketPolicyServer`](../../src/openpi/serving/websocket_policy_server.py) 调用策略推理，再把包含 `actions` 的响应发回客户端。

### 一次 observation 包含什么

`examples/libero/main.py` 在每次需要重新规划时构造四个输入字段：

| 字段 | 来自哪里 | 客户端预处理 |
| --- | --- | --- |
| `observation/image` | 第三人称 `agentview_image` | 高、宽两个轴都反转，也就是旋转 180°；再等比例缩放并 padding 到 `224 × 224`，转换为 `uint8` |
| `observation/wrist_image` | 腕部 `robot0_eye_in_hand_image` | 同样旋转 180°、缩放并 padding 到 `224 × 224`、转换为 `uint8` |
| `observation/state` | 机器人本体状态 | 拼接末端执行器位置、由四元数转换得到的 3 维 axis-angle、夹爪关节位置 `gripper_qpos` |
| `prompt` | 当前 LIBERO task 的自然语言描述 | 转成字符串，告诉策略当前要完成什么任务 |

旋转不是为了“看起来正”，而是为了与训练预处理保持一致。图像尺寸也不是直接拉伸到正方形，而是通过 `resize_with_pad` 尽量保留长宽比。

### action chunk 与重新规划

客户端维护一个 `collections.deque`，源码变量名是 `action_plan`。只有队列为空时，它才把最新 observation 发给服务端请求一个 action chunk。默认 `replan_steps=5`，因此客户端只把返回 chunk 的前 5 个动作放入队列，然后逐个执行；第 5 个动作执行完、队列再次变空后，才基于新的 observation 发起下一次请求。

这是一轮重规划周期的数据流。图中从 LIBERO 启动后的等待步结束处开始：

```text
LIBERO 环境          runtime / openpi-client              openpi_server / Policy
    |                         |                                      |
    |--- 新 observation ----->|                                      |
    |                         |-- 旋转图像、resize/pad 到 224         |
    |                         |-- 拼接 state，加入 task prompt        |
    |                         |-- WebSocket: observation ------------>|
    |                         |                                      |-- 策略推理
    |                         |<-- WebSocket: action chunk -----------|
    |                         |-- 取前 replan_steps=5 个动作入队      |
    |<-- action 1 ------------|                                      |
    |--- 新 observation ----->|  队列未空，不请求策略                 |
    |<-- action 2 ... 5 ------|                                      |
    |--- 新 observation ----->|  队列已空，进入下一轮重规划           |
```

客户端有一条显式断言：策略返回的 action horizon 必须不短于 `replan_steps`。仓库默认 `pi05_libero` 的 action horizon 是 10，可以满足默认的 5 步重规划；换成自定义配置时必须重新核对这项约束。

## 源码地图

第一次运行不要求读懂整个模型，只需沿服务启动和一次请求的路径阅读：

| 文件 | 阅读重点 |
| --- | --- |
| [`examples/libero/README.md`](../../examples/libero/README.md) | 上游推荐的 Docker 工作流、参数入口和 benchmark 说明 |
| [`examples/libero/compose.yml`](../../examples/libero/compose.yml) | 两个 service、host network、GPU、缓存挂载、默认 EGL 与环境变量传递 |
| [`examples/libero/main.py`](../../examples/libero/main.py) | LIBERO episode 循环、observation 构造、action queue、日志与视频输出 |
| [`scripts/serve_policy.py`](../../scripts/serve_policy.py) | `EnvMode.LIBERO` 的默认 config/checkpoint、端口和 Policy Server 创建 |
| [`scripts/docker/serve_policy.Dockerfile`](../../scripts/docker/serve_policy.Dockerfile) | `openpi_server` 镜像的 Python 环境和最终启动命令 |
| [`src/openpi/serving/websocket_policy_server.py`](../../src/openpi/serving/websocket_policy_server.py) | 服务端如何接收 observation、调用 `policy.infer()` 并返回结果 |
| [`packages/openpi-client/src/openpi_client/websocket_client_policy.py`](../../packages/openpi-client/src/openpi_client/websocket_client_policy.py) | 轻量客户端如何等待服务、编码请求和解码响应 |

建议先读 `compose.yml` 和两个启动入口，再顺着 WebSocket 客户端/服务端看一轮请求；模型内部结构留到下一章。

## 动手实践

以下命令都从 openpi 仓库根目录运行。

### 实践 1：启动最小的预训练评测

在独占的开发主机上，可以设置默认 LIBERO policy，并把每个 task 的 trial 数从默认 50 降到 1：

```bash
export SERVER_ARGS="--env LIBERO"
export CLIENT_ARGS="--args.task-suite-name libero_spatial --args.num-trials-per-task 1"
docker compose -f examples/libero/compose.yml up --build
```

共享 Docker host 上不要直接照搬默认 project name 和 `8000` 端口。先与其他使用者确认一个未占用的端口和唯一项目名，例如把两侧都改为 `18000`，并在所有 Compose 命令中复用 `-p "openpi-tutorial-$USER"`：

```bash
export SERVER_ARGS="--env LIBERO --port 18000"
export CLIENT_ARGS="--args.port 18000 --args.task-suite-name libero_spatial --args.num-trials-per-task 1"
docker compose -p "openpi-tutorial-$USER" -f examples/libero/compose.yml up --build
```

若同一个 Unix 账号也被多人共享，应再为 project name 加上团队约定的唯一后缀。端口和项目名是两件事：`-p` 隔离 Compose 资源名称，`--port`/`--args.port` 避免 host network 的监听冲突；两侧端口必须相同。

这里的“最小”指减少每个 task 的重复 trial，并不表示只运行一个总 episode。`libero_spatial` 是一个包含多个 task 的 suite；程序仍会遍历 suite 中的多个 task，而 `--args.num-trials-per-task 1` 表示**每个 task 各运行 1 次**。如果只是确认闭环，看到至少一个 episode 已推进并留下证据后，可以按后文方式安全停止，不必等待整个 suite 完成。

首次运行可能依次经历：下载基础镜像、构建两个镜像、下载 checkpoint、加载参数，以及模型第一次推理的 JAX/XLA JIT 编译。这些阶段可能耗时较长；不要只因短时间没有 episode 输出就判定为卡死，先观察“常见问题”中的状态和日志证据。

### 实践 2：观察服务端与客户端证据

`docker compose up` 会把两个 service 的输出交错显示。不要期待不同版本逐字输出一份完全相同的日志；应识别下面这些语义证据：

- `openpi_server` 一侧先出现 checkpoint/配置加载相关信息，随后出现创建服务、监听端口或接受 WebSocket 连接的迹象；
- 客户端的 `Task suite: ...` 在构造 WebSocket 客户端**之前**打印，因此它可能先于 `Waiting for server at ...` 出现，不能单独证明已经连接；
- 轻量客户端没有显式的 `Connected` 日志。可靠的连接证据是等待信息之后客户端继续打印 `Task: ...` 与 `Starting episode...`，或者服务端打印源码中的 `Connection from ... opened`；
- 建立连接后，客户端继续打印当前任务的自然语言描述、episode 开始/完成进度、`Success: True` 或 `Success: False`，以及累计 episode/成功数；
- 每个已结束的 episode 会在宿主机 `data/libero/videos/` 下保存一个以 `success` 或 `failure` 结尾的 MP4 回放。

另开终端可以收集容器与文件证据：

```bash
docker compose -f examples/libero/compose.yml ps
docker compose -f examples/libero/compose.yml logs --tail=200 openpi_server runtime
find data/libero/videos -maxdepth 1 -type f -name 'rollout_*.mp4' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
```

若使用了共享主机示例，前两条命令也必须带相同的 project name：

```bash
docker compose -p "openpi-tutorial-$USER" -f examples/libero/compose.yml ps
docker compose -p "openpi-tutorial-$USER" -f examples/libero/compose.yml logs --tail=200 openpi_server runtime
```

`Success: False` 说明策略没有在该 rollout 内完成任务，不等于 WebSocket、图像预处理或仿真环境部署失败。本章先验收工程闭环，不要求复现上游成功率。

### 实践 3：完成源码观察任务

用符号名搜索关键逻辑，并让 `rg` 同时显示上下文。这样即使源码行号变化，观察任务仍然有效：

```bash
rg -n -C 4 'agentview_image|robot0_eye_in_hand_image|resize_with_pad' examples/libero/main.py
rg -n -C 7 '"observation/state"|robot0_eef_pos|_quat2axisangle|robot0_gripper_qpos|"prompt"' examples/libero/main.py
rg -n -C 6 'replan_steps|action_plan|client\.infer' examples/libero/main.py
rg -n -C 3 'task_description = task\.language' examples/libero/main.py
```

在输出中定位并解释：

1. 图像旋转：为什么 `[::-1, ::-1]` 同时反转两个图像轴，以及为什么主视角和腕部视角都这样处理；
2. state 拼接：指出 `robot0_eef_pos`、`_quat2axisangle(robot0_eef_quat)` 和 `robot0_gripper_qpos` 的顺序；
3. prompt：指出它来自 `task_description`，而 `task_description` 又来自当前 task 的 `language`；
4. `replan_steps`：找出默认值 5、chunk 长度断言和只截取前 5 步的位置；
5. action queue：找出 `deque` 的创建、`if not action_plan`、`extend` 和 `popleft`，说明为什么队列未空时不会再次请求服务端。

验收时不要只报行号；请用自己的话画出或复述“一次 observation 请求换回一个 action chunk，执行 5 步后再请求”的闭环。

### 实践 4：停止与清理本次运行

在运行 `docker compose up` 的终端按 `Ctrl+C`。等待 Compose 完成停止流程，再使用与 `up` **完全相同的 project name** 检查和清理。

如果使用了默认最小命令：

```bash
docker compose -f examples/libero/compose.yml ps
docker compose -f examples/libero/compose.yml down
unset SERVER_ARGS CLIENT_ARGS
```

如果使用了共享主机的命名项目与 `18000` 端口示例：

```bash
docker compose -p "openpi-tutorial-$USER" -f examples/libero/compose.yml ps
docker compose -p "openpi-tutorial-$USER" -f examples/libero/compose.yml down
unset SERVER_ARGS CLIENT_ARGS
```

不要在命名项目启动后改用不带 `-p` 的 `ps` 或 `down`。如果 `ps` 没有列出运行中的 service，说明本次进程已停止；想同时查看已停止的容器，可追加 `-a`。`down` 的默认范围是停止并移除所选 Compose 项目的容器及其可移除网络资源。它不会删除已构建镜像、Docker build cache，也不会删除宿主机 bind mount 中的 `data/libero/videos` 或 `${OPENPI_DATA_HOME:-$HOME/.cache/openpi}` checkpoint 缓存。最后 `unset` 可避免本章参数意外影响同一 shell 中的后续 Compose 运行。本章没有理由使用 `down --rmi all`、`docker system prune` 或直接删除整个模型缓存。

## 预期结果

完成一次有效的最小运行后，应能提供以下证据：

| 证据 | 能证明什么 | 不能单独证明什么 |
| --- | --- | --- |
| 两个 service 都启动，服务端有创建/监听相关日志 | Compose 和 Policy Server 已进入运行路径 | checkpoint 一定已完成首次推理 |
| 等待信息后客户端继续打印 `Task:`/`Starting episode...`，或服务端打印 `Connection from ... opened` | WebSocket 的 host/port 路径可达 | observation 字段和动作 shape 一定正确 |
| 日志出现 task 描述和 episode 开始/累计进度 | LIBERO task suite 与仿真循环已经推进 | 策略一定会成功完成任务 |
| 日志出现 success/failure，且目录中有新 MP4 | 至少一个 episode 已结束并保存回放 | 整个 suite 已跑完或达到官方成功率 |
| 后续重规划没有 action 长度断言 | 返回的 action horizon 至少覆盖当前 `replan_steps` | 自定义 checkpoint 与其他配置也一定兼容 |

### 本文的验证边界

本文作者在当前环境中只静态核对了 Compose、Dockerfile、服务端、客户端和 LIBERO 主循环源码，并执行了文档文本与 Git 差异检查。**没有在当前环境运行** `docker compose ... up --build`，也没有实际下载 `pi05_libero` checkpoint、触发首次 JIT 或跑完 LIBERO episode。因此上面的运行现象是学习者应现场收集的验收证据，不是作者声称已经在这台机器得到的实测日志。

## 常见问题

排错顺序统一为“保留现场 → 看 service 状态 → 分开读服务端/客户端日志 → 再修改一个变量”。不要把清空缓存或重装所有依赖当作第一步。

下面的多数排错命令按默认项目写出；如果你用 `-p "openpi-tutorial-$USER"` 启动，就必须给每一条 `docker compose` 排错命令补上同一个 `-p`。不要用默认项目的 `ps`/`logs` 去判断命名项目的状态。

### 1. 首次 JIT 很慢，还是程序真的挂住了？

先记录容器状态、最近日志和 GPU 活动：

```bash
docker compose -f examples/libero/compose.yml ps -a
docker compose -f examples/libero/compose.yml logs --since=10m --timestamps openpi_server runtime
nvidia-smi
docker stats --no-stream
```

若服务端已加载 checkpoint、客户端已连接，而且第一次 `infer` 前后 GPU/CPU 仍有活动，长等待可能来自首次 JIT。若某个容器已经退出、日志持续重复同一异常、GPU 完全不可见，或服务端根本没有进入加载/监听路径，就按对应错误继续定位。JIT 慢和“容器已崩溃”应由证据区分，不能只看墙上时间。

### 2. checkpoint 下载失败、很慢或缓存不可写

先看服务端日志、挂载展开结果、空间和缓存目录权限：

```bash
docker compose -f examples/libero/compose.yml logs --tail=300 openpi_server
docker compose -f examples/libero/compose.yml config | rg -n 'OPENPI_DATA_HOME|openpi_assets'
printf 'OPENPI_DATA_HOME=%s\n' "${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
df -h "${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
du -sh "${OPENPI_DATA_HOME:-$HOME/.cache/openpi}" 2>/dev/null
ls -ld "${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
```

根据日志区分 DNS/代理/对象存储访问、磁盘不足和目录权限问题。缓存存在并增长通常说明下载正在产生进度；不要在下载中途或不了解内容时删除整个缓存。确需修复残缺文件时，也应先记录具体路径和错误，再只处理被证据指向的对象。

### 3. `8000` 端口被占用

host network 意味着端口冲突直接发生在宿主机。以 `ss` 的监听 socket 为主要证据，再检查可疑容器的状态、host PID 与日志：

```bash
ss -ltnp | rg ':8000\b'
docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}'
printf '输入候选容器名: '; read -r CONTAINER_NAME
docker inspect --format '{{.Name}} pid={{.State.Pid}} network={{.HostConfig.NetworkMode}}' "$CONTAINER_NAME"
docker logs --tail=100 "$CONTAINER_NAME"
unset CONTAINER_NAME
```

使用 `network_mode: host` 时没有 Docker published-port 映射，因此 `docker ps` 的 `{{.Ports}}` 可能为空，即使进程正在监听 host 端口。不要据此判断端口无人使用；先从 `ss` 取得监听 PID，再把 `CONTAINER_NAME` 改成上一步发现的候选容器，用 `docker inspect` 的 host PID 和日志交叉确认归属。若实际使用 `18000` 或其他端口，也要把 `ss` 的筛选值同步替换。

确认归属后停止你有权停止的旧服务，或者让服务端和客户端一起改用同一个空闲端口，并给 Compose 项目一个唯一名称，例如复用前面的共享主机方案：

```bash
export SERVER_ARGS="--env LIBERO --port 18000"
export CLIENT_ARGS="--args.port 18000 --args.task-suite-name libero_spatial --args.num-trials-per-task 1"
docker compose -p "openpi-tutorial-$USER" -f examples/libero/compose.yml up --build
```

只改一侧端口会造成连接失败；后续 `ps`、`logs` 和 `down` 也必须继续使用 `-p "openpi-tutorial-$USER"`。

### 4. 客户端一直显示等待服务端

`depends_on` 只规定容器启动顺序，不保证模型已经加载完成。轻量客户端遇到 `ConnectionRefusedError` 会每 5 秒重试，因此 checkpoint 加载期间出现等待信息可以是正常现象。注意 `Task suite: ...` 在连接前就会打印，且客户端没有单独的 `Connected` 日志；只有等待后继续进入 `Task: ...`/`Starting episode...`，或服务端出现 `Connection from ... opened`，才是可靠的连接证据。先检查服务端是否仍在运行和推进：

```bash
docker compose -f examples/libero/compose.yml ps -a
docker compose -f examples/libero/compose.yml logs --tail=300 openpi_server
```

如果服务端已经退出，修复它的首个根因；如果仍在加载或 JIT，保留日志并继续观察；如果服务端已经监听却仍无法连接，再核对两侧端口和是否仍使用 host network。

### 5. Docker 容器看不到 GPU

分层检查宿主机、Docker runtime 和两个 service：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
docker compose -f examples/libero/compose.yml config | rg -n 'driver: nvidia|capabilities|count:'
docker compose -f examples/libero/compose.yml logs --tail=200 openpi_server runtime
```

如果独立 CUDA 容器都失败，应先修复 Docker/NVIDIA Container Toolkit，而不是修改 openpi 模型代码。service 已经运行时，还可以用 `docker compose ... exec <service> nvidia-smi` 分别核对容器内可见性。

### 6. LIBERO 包、BDDL 或初始状态文件找不到

先检查子模块状态与目标目录：

```bash
git submodule status
test -d third_party/libero/libero/libero && printf 'LIBERO source exists\n'
```

行首 `-` 表示未初始化；标准修复是：

```bash
git submodule update --init --recursive
```

行首 `+` 或 `U` 需要先确认本地改动、父仓库记录的 commit 或冲突，不能用反复重建镜像掩盖源码状态问题。

### 7. action horizon 比 `replan_steps` 短

客户端会明确报错，说明希望每次执行若干步，但策略只返回了更短的 chunk。先核对客户端默认值、当前模型配置和实际返回长度：

```bash
rg -n 'replan_steps|len\(action_chunk\)|action_plan.extend' examples/libero/main.py
rg -n 'name="pi05_libero"|action_horizon' src/openpi/training/config.py
```

仓库默认组合是 horizon 10、`replan_steps=5`。自定义 policy 时，应让 `replan_steps` 不大于该 policy 的 action horizon；不要删除断言，也不要假设名为 LIBERO 的任意 checkpoint 都有相同长度。

### 8. EGL 或 GLX 渲染失败

当前 Compose 的默认值是 EGL：`MUJOCO_GL` 可由环境变量展开，但 `PYOPENGL_PLATFORM=egl` 是硬编码的。因此只设置 `MUJOCO_GL=glx` 是**部分切换**，不能证明 PyOpenGL 已完整改用 GLX。先收集展开配置与 runtime 日志：

```bash
docker compose -f examples/libero/compose.yml config | rg -n 'MUJOCO_GL|PYOPENGL_PLATFORM|DISPLAY'
docker compose -f examples/libero/compose.yml logs --tail=300 runtime
```

默认 EGL 失败且证据确实指向渲染后端时，按[第 02 章的完整 GLX override](02_environment_setup.md#egl-是-compose-默认值glx-是回退路径)同时覆盖 `MUJOCO_GL=glx` 与 `PYOPENGL_PLATFORM=glx`，并核对 X11 的 `DISPLAY`、socket 和授权。不要只设置 `MUJOCO_GL` 后就宣称已经完整切换。

## 本章验收

进入下一章前，逐项提供终端输出、文件或口头解释：

- [ ] `docker compose ... ps` 或对应日志证明 `openpi_server` 与 `runtime` 都曾启动；
- [ ] 服务端日志证明它进入默认 LIBERO policy 的加载与监听路径，且你能说出配置 `pi05_libero` 和 checkpoint `gs://openpi-assets/checkpoints/pi05_libero`；
- [ ] 能说明 `Task suite:` 可能先于等待日志，客户端没有显式 `Connected`；等待后出现 `Task:`/`Starting episode...` 或服务端出现 `Connection from ... opened` 才能证明已经连接；
- [ ] 至少一个 episode 已开始并推进，日志中没有 observation 字段、WebSocket 或 action shape 的立即错误；
- [ ] 至少能提供一个已结束 episode 的 success/failure 日志，并在 `data/libero/videos` 找到对应回放；成功率高低不是本章验收条件；
- [ ] 能指出图像旋转、state 拼接、prompt、`replan_steps` 和 action queue 的源码位置并解释作用；
- [ ] 能说明“每个 task 1 个 trial”不等于“整个 suite 只有 1 个 episode”；
- [ ] 能用 `Ctrl+C` 停止运行，并说明 `docker compose ... down` 会清理什么、保留什么；
- [ ] 能说明 `runtime` 的 `privileged: true` 与两个容器的可写 `/app` 挂载意味着什么；共享 host 上能为 `up`、`ps`、`logs`、`down` 复用唯一 project name，并为服务端和客户端设置同一个空闲 host 端口；
- [ ] 能说明本章环境验收不要求成功率达标，`Success: False` 仍可能对应一个正确运行的工程闭环；
- [ ] 能说明本章的 `pi05_libero` 预训练 checkpoint 不能与后续 `pi0_libero_tutorial` 混用。

## 思考题

### 1. 为什么客户端不在每个仿真 step 都请求一次策略？

<details>
<summary>参考答案（展开查看）</summary>

策略一次返回 action chunk。客户端把前 `replan_steps=5` 步放进队列，逐步执行，可以减少服务请求和推理开销；执行完这 5 步后再用新 observation 重规划，又能避免长时间完全开环。

</details>

### 2. 为什么 `--args.num-trials-per-task 1` 仍可能运行很久？

<details>
<summary>参考答案（展开查看）</summary>

它把每个 task 的 trial 数设为 1，但主循环仍遍历 task suite 中的多个 task。总 episode 数取决于 suite 的 task 数，而不是只看这个参数中的数字 1；首次构建、checkpoint 下载和 JIT 还会增加启动时间。

</details>

### 3. 客户端已经打印 `Success: False`，为什么本章的环境验收仍可能通过？

<details>
<summary>参考答案（展开查看）</summary>

`Success: False` 是任务结果，不是通信结果。只要服务加载、WebSocket 连接、observation/action 交换、仿真推进和回放保存都正常，工程闭环已经成立。本章不以单个 rollout 的策略成功率作为环境安装标准。

</details>

### 4. 为什么只设置 `MUJOCO_GL=glx` 不能证明已完整切换到 GLX？

<details>
<summary>参考答案（展开查看）</summary>

基础 Compose 会展开 `MUJOCO_GL`，但同时硬编码 `PYOPENGL_PLATFORM=egl`。只改前者会留下互相矛盾的渲染选择器；完整回退需要用 Compose override 同时设置二者，并满足 GLX 的 X11 条件。

</details>

[← 上一章：环境安装与双运行时自检](02_environment_setup.md) · [课程入口：README](README.md) · [下一章：openpi 架构与一次请求的数据流 →](04_openpi_architecture.md)
