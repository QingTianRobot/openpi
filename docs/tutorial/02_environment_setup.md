# 第 02 章：环境安装与双运行时自检

[← 上一章：机器人学习与 VLA 基础](01_vla_and_robotics_basics.md) · [课程入口：README](README.md) · [下一章：第一次 LIBERO 推理 →](03_first_libero_inference.md)

## 本章目标

完成本章后，你应该能够：

- 按仓库 `README.md` 初始化子模块，并安装 Python 3.11 的主项目 `uv` 环境；
- 解释为什么 openpi 主环境与 LIBERO/MuJoCo 评测环境不能混装依赖；
- 说明推荐评测时 `openpi_server` 与 `runtime` 两个容器分别做什么，以及它们如何通过 WebSocket 通信；
- 用 `nvidia-smi`、JAX、Docker Compose、子模块和磁盘命令完成一次证据优先的环境自检；
- 知道 checkpoint、数据集、Docker 镜像、训练产物和评测视频大致会占用哪些目录；
- 在 EGL 渲染失败时有条件地回退到 GLX，而不是一开始就修改默认渲染路径。

本章只准备环境，不下载完整训练数据、不启动 LIBERO 闭环评测，也不修改模型源码。遇到英文缩写可查[课程术语表](glossary.md)；遇到跨章节故障可继续查[故障排查手册](troubleshooting.md)。这两个页面是课程后续章节会补齐的配套索引。

## 前置条件

本课程采用下面的支持假设。其他配置不一定完全不能运行，但需要你自行处理版本、显存或平台兼容问题。

| 项目 | 本课程假设 | 为什么重要 |
| --- | --- | --- |
| 操作系统 | Ubuntu 22.04 | openpi 仓库明确说明在该版本上测试；本教程不覆盖 Windows、macOS 或其他 Linux 发行版的差异 |
| GPU | 单卡 NVIDIA A100 或 H100 | 为后续 π₀/π₀.₅ 推理和教学 LoRA 留出足够显存；宿主机需要可工作的 NVIDIA 驱动 |
| 主项目 Python | Python 3.11 | 根目录 `pyproject.toml` 要求 Python `>=3.11`，课程统一用 3.11，避免不同小版本带来的依赖分歧 |
| 容器工具 | Docker Engine 与 Docker Compose | 推荐的 LIBERO 评测由 `examples/libero/compose.yml` 同时管理模型服务和仿真客户端 |
| 可用磁盘 | 至少 100 GB | 需要容纳 Python 包、模型缓存、数据集、Docker 镜像与层、训练 checkpoint 和评测视频 |

此外还需要：

- 能从当前机器访问 GitHub；后续下载模型和数据时还可能访问 Hugging Face 与模型资源存储；
- 已安装 Git 和 [`uv`](https://docs.astral.sh/uv/)；
- Docker 已配置 NVIDIA Container Toolkit，使容器能够请求 GPU；
- 已进入 openpi 仓库根目录，也就是能看到 `pyproject.toml`、`examples/` 和 `src/` 的目录。

本章不会指导安装 NVIDIA 驱动或 Docker 本身，因为这些操作依赖机器权限和组织环境。先保留完整错误信息，再按本章“常见问题”判断故障发生在宿主机驱动、Python 环境还是容器层。

### 安装前检查清单

开始安装前逐项确认：

- [ ] 宿主机是 Ubuntu 22.04；
- [ ] 计划使用 NVIDIA A100 或 H100，且当前用户能运行 `nvidia-smi`；
- [ ] `uv --version` 能输出版本号；
- [ ] `docker compose version` 能输出 Compose 版本；
- [ ] 已准备运行一次真实的 Docker GPU smoke test；若暂缓镜像拉取，就把容器 GPU 状态记录为“未验证”，且不把 preflight 视为通过；
- [ ] 当前目录是仓库根目录；
- [ ] `df -h .` 显示至少 100 GB 可用空间；
- [ ] 网络或代理允许 Git 访问 GitHub；
- [ ] 清楚主项目使用 Python 3.11，LIBERO 客户端使用隔离的 Python 3.8 运行时；
- [ ] 没有计划把 `examples/libero/requirements.txt` 或 `third_party/libero/requirements.txt` 安装进根目录 `.venv`。

最后一项尤其重要：LIBERO 依赖集合较旧，并且官方示例明确为它创建 Python 3.8 环境。把它安装进主 `uv` 环境，可能覆盖 JAX、NumPy、PyTorch 或其他训练依赖，让问题变成难以定位的版本冲突。

## 核心概念

### 一套仓库，两类运行时职责

openpi 的“训练/模型环境”和“仿真环境”关注点不同。最稳妥的做法不是强行统一版本，而是让它们通过稳定的 WebSocket 接口交换 observation 和 action。

| 运行时上下文 | Python 与位置 | 主要职责 | 本课程怎样使用 |
| --- | --- | --- | --- |
| 宿主机的主项目环境 | 仓库根目录的 Python 3.11 `uv` 环境 | 数据转换、normalization statistics、训练、配置检查和轻量本地命令 | 本章安装；后续数据与训练章节反复使用 |
| 推荐评测的模型端 | Compose 的 `openpi_server` 容器 | 加载模型/checkpoint，启动 Policy Server，接收 observation 并返回 action chunk | 第 03 章由 `examples/libero/compose.yml` 启动 |
| 推荐评测的仿真端 | Compose 的 `runtime` 容器，内部 Python 3.8 | 运行 LIBERO、MuJoCo 和轻量 `openpi-client`，产生 observation、发送请求并执行 action | 第 03 章与 `openpi_server` 一起启动 |

这里有两个容易混淆的点：

1. “模型端”是职责，不等于“永远运行在宿主机”。不用 Docker 的手工方案可以在宿主机启动 Policy Server，但**本课程推荐的闭环评测会在 `openpi_server` 容器中运行它**。
2. 宿主机主环境仍然有价值。数据转换、统计量计算、训练和无需 LIBERO 的轻量检查都在 Python 3.11 `uv` 环境完成；评测时再把模型服务与仿真交给 Compose 管理。

推荐评测的数据流可以简化为：

```text
runtime 容器（Python 3.8，LIBERO/MuJoCo）
    └─ observation ──WebSocket──> openpi_server 容器（Policy Server）
                                 └─ action chunk ──WebSocket──> runtime 容器
```

[`examples/libero/compose.yml`](../../examples/libero/compose.yml) 为两个服务都设置了 `network_mode: host`。LIBERO 客户端默认连接端口 `8000`，因此两个容器通过宿主机网络命名空间中的同一端口通信，不需要把 LIBERO 的旧 Python 依赖塞进模型环境。具体启动参数将在[第 03 章](03_first_libero_inference.md)使用。

### 为什么安装后的小命令使用 `uv run --no-sync`

`uv run` 不只是“找 Python 并运行命令”。默认情况下，它会检查项目环境，并可能协调或同步依赖。本仓库把 `lerobot` 固定到一个 GitHub commit，因此同步阶段可能访问 GitHub；网络不稳定时，一个原本很轻量的检查也可能停在依赖获取上。

安装阶段应该允许同步，因此使用仓库 `README.md` 给出的 `uv sync`。安装完成后，本教程的轻量命令通常写成：

```bash
uv run --no-sync python -c "..."
```

`--no-sync` 表示“使用现有环境，不在这次命令前重新解析或安装”。它不会修复缺包：如果环境尚未正确安装，命令会如实报 `ModuleNotFoundError` 或解释器错误，此时应回到安装步骤处理。

### `GIT_LFS_SKIP_SMUDGE=1` 在做什么

仓库安装命令在 `uv sync` 和 editable install 前临时设置：

```text
GIT_LFS_SKIP_SMUDGE=1
```

这是 Git LFS 的环境变量。仓库 `README.md` 指出，拉取作为 Git 依赖的 LeRobot 时需要它，以避免 Git 在 checkout 过程中自动下载 LFS 管理的大文件。它只作用于当前命令，不等于“以后永远不下载 checkpoint 或数据”；模型、数据集和其他资源会在真正需要时通过各自的下载路径获取。

### 磁盘不只被一个缓存占用

100 GB 是为多个类别共同预留的课程下限，不是某一个固定文件的精确下载量。实际占用取决于选择的 checkpoint、数据规模、镜像是否复用、训练保存频率和视频数量。

| 位置或类别 | 保存什么 | 初学者应注意什么 |
| --- | --- | --- |
| `${OPENPI_DATA_HOME}`，默认 `~/.cache/openpi` | openpi 自动下载的 checkpoint 和相关模型资源 | Compose 会把它挂载进 `openpi_server` 的 `/openpi_assets`，避免每次容器运行都重复下载 |
| Hugging Face / LeRobot 数据缓存 | Hub 元数据、数据集文件、LeRobot episode 与视频等 | 具体根目录受 `HF_HOME`、LeRobot 版本和数据转换参数影响；不要假设它一定与 `OPENPI_DATA_HOME` 相同 |
| Docker images、layers 与 build cache | LIBERO 镜像、Policy Server 镜像及构建中间层 | 即使仓库目录很小，Docker 自己的数据目录也可能占用大量空间 |
| `assets/` | 后续生成的 normalization statistics 等训练配置资产 | 新克隆仓库里可能还不存在；第 08 章会生成课程资产 |
| `checkpoints/` | 后续训练写出的参数、优化器状态和随 checkpoint 复制的 assets | 保存多个 step 会持续增长；第 09 章会解释实验目录结构 |
| `data/libero/videos/` | LIBERO 客户端保存的评测回放 | trial 越多，视频越多；目录通常在首次评测时创建 |

先用 `df`、`du` 和 `docker system df` 查清楚空间去向，再决定迁移、归档或清理哪一类内容。不要在不知道用途时直接删除整个缓存目录；正在下载的文件、后续可复用的镜像层或仍需加载的 checkpoint 都可能在里面。

### EGL 是 Compose 默认值，GLX 是回退路径

[`examples/libero/compose.yml`](../../examples/libero/compose.yml) 设置：

```text
MUJOCO_GL=${MUJOCO_GL:-egl}
PYOPENGL_PLATFORM=egl
```

因此无额外配置时使用 EGL，这是无窗口/服务器环境更合适的默认路径。上游 [LIBERO README](../../examples/libero/README.md) 给出的简写是：

```bash
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

这条命令只把可展开的 `MUJOCO_GL` 改为 `glx`；当前基础 Compose 文件仍把 `PYOPENGL_PLATFORM` **硬编码为 `egl`**。因此它是上游提供的部分切换方式，不能保证 PyOpenGL 也完整切到 GLX。

若日志证明需要完整 GLX 回退，可在仓库外创建临时 override，例如 `/tmp/openpi-libero-glx.override.yml`：

```yaml
services:
  runtime:
    environment:
      MUJOCO_GL: glx
      PYOPENGL_PLATFORM: glx
```

随后同时加载基础文件与 override：

```bash
SERVER_ARGS="--env LIBERO" docker compose \
  -f examples/libero/compose.yml \
  -f /tmp/openpi-libero-glx.override.yml \
  up --build
```

Compose 会按环境变量名称合并配置，使 `runtime` 中的两个选择器都成为 `glx`。可以先把最后一行 `up --build` 临时换成 `config`，检查展开结果中两个变量是否都为 `glx`。

GLX 依赖 X11。它通常需要正确的 `DISPLAY`、挂载的 `/tmp/.X11-unix` 和允许容器连接 X server 的权限；官方 [LIBERO README](../../examples/libero/README.md) 给出了 `sudo xhost +local:docker`。该命令会放宽本机 X server 的访问权限，只应在确实使用 GLX、理解机器安全边界时执行。使用结束后，通常由图形会话的所有者在同一个 `DISPLAY`/`XAUTHORITY` 上运行精确的反向命令：

```bash
xhost -local:docker
```

EGL 正常时不需要先执行 `xhost`。

## 源码地图

本章不要求通读源码，只需知道每个环境事实从哪里核对。

| 文件 | 本章阅读重点 |
| --- | --- |
| [`README.md`](../../README.md) | Ubuntu 支持范围、GPU 需求、子模块与 `uv` 安装命令、`OPENPI_DATA_HOME` 默认行为 |
| [`pyproject.toml`](../../pyproject.toml) | 主项目要求 Python `>=3.11`，JAX CUDA 依赖，以及固定到 GitHub commit 的 `lerobot` 来源 |
| [`examples/libero/compose.yml`](../../examples/libero/compose.yml) | `runtime` 与 `openpi_server` 服务、host network、GPU reservation、默认 EGL 和模型缓存挂载 |
| [`examples/libero/Dockerfile`](../../examples/libero/Dockerfile) | LIBERO 容器显式创建 Python 3.8 环境，并安装独立的 LIBERO/MuJoCo 依赖 |
| [`examples/libero/README.md`](../../examples/libero/README.md) | 推荐 Docker 启动方式、GLX 回退与不推荐的手工 Python 3.8 方案 |
| [`examples/libero/main.py`](../../examples/libero/main.py) | WebSocket 客户端默认 host/port 与 `data/libero/videos` 输出路径 |
| [`src/openpi/shared/download.py`](../../src/openpi/shared/download.py) | `OPENPI_DATA_HOME` 和默认 `~/.cache/openpi` 的实际实现 |

阅读时把“仓库里声明了什么”与“当前机器真的具备什么”分开：文件能证明版本和服务设计，不能证明你的驱动、GPU、网络和磁盘已经可用。

## 动手实践

以下命令默认都在 openpi 仓库根目录运行。先做不会修改项目的检查，再安装依赖。

### 实践 1：记录安装前的机器状态

逐条运行，不要只保存最后一行输出：

```bash
nvidia-smi
uv --version
docker compose version
git submodule status
df -h .
```

观察重点：

- `nvidia-smi` 应列出至少一张计划使用的 NVIDIA GPU，而不是驱动通信错误；
- `uv --version` 应输出可识别的版本；
- `docker compose version` 应表明 Compose 子命令可用；
- `git submodule status` 中课程所需子模块的行首应为空格；`-` 表示未初始化，`+` 表示 checkout 与父仓库记录的 commit 不同，`U` 表示冲突；
- `df -h .` 的 `Avail` 应至少为 100 GB，同时留意仓库和 Docker 数据目录可能不在同一文件系统。

如果此时子模块行首是 `-`，不需要慌张：这正是下一步要修复的状态。`+` 和 `U` 也不是标准通过状态；应先核对本地工作与预期 commit。若 `nvidia-smi`、Compose 或磁盘条件不满足，应先解决宿主机前置条件，避免在大型依赖安装或镜像构建中途才失败。

Compose 命令存在还不能证明容器可以使用 GPU。运行一个不依赖本仓库镜像的 smoke test：

```bash
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

首次运行可能需要从镜像仓库拉取 CUDA base image，因此会使用网络、时间和 Docker 磁盘。预期输出与宿主机 `nvidia-smi` 类似，并列出目标 GPU。若你决定暂缓拉取或运行，就必须把“Docker GPU 访问”记录为**未验证**，本章 preflight 也尚未完整通过。

### 实践 2：按仓库 README 安装主环境

先初始化所有嵌套子模块：

```bash
git submodule update --init --recursive
```

然后让 `uv` 创建/同步根目录的 Python 3.11 项目环境，并以 editable 方式安装当前仓库：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

这几步可能访问 GitHub 和 Python 包索引，并占用较长时间。不要同时在另一个终端修改 `.venv` 或向其中安装 LIBERO 的 Python 3.8 依赖。安装结束后，可用下面的轻量命令确认实际解释器：

```bash
uv run --no-sync python --version
```

预期是 Python 3.11.x。若不是，先处理解释器选择，再继续 JAX 检查。

### 实践 3：确认子模块、JAX 与 GPU

再次查看子模块：

```bash
git submodule status
```

课程标准要求所需子模块行首的第一个字符是**空格**，而不只是“没有 `-`”。三种非空白前缀的含义是：

- `-`：子模块未初始化；
- `+`：当前 checkout 与父仓库记录的 commit 不同；
- `U`：子模块存在合并冲突。

默认应把课程所需的 `third_party/libero` 等子模块对齐到父仓库记录的 commit。若你有意保留非空白状态，继续前必须记录子模块路径、前缀、实际 commit、偏离原因和对后续课程的影响；它属于有意识的例外，不应被写成标准 preflight 已通过。

随后让已经安装好的 JAX 列出设备：

```bash
uv run --no-sync python -c "import jax; print(jax.devices())"
```

预期列表中至少出现一个 GPU 设备。只看到 `CpuDevice` 表示 Python 可以导入 JAX，但没有建立课程需要的 GPU 执行路径；“能 import”不等于“GPU 已配置完成”。

这里必须使用安装后的 `--no-sync` 命令。若改用普通 `uv run`，`uv` 可能先同步依赖并访问 GitHub，网络错误会遮住你真正想检查的 JAX 设备状态。

### 实践 4：检查缓存与可增长目录

先查看环境变量和各类存储的当前状态：

```bash
printf 'OPENPI_DATA_HOME=%s\n' "${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
printf 'HF_HOME=%s\n' "${HF_HOME:-未显式设置，使用库默认值}"
du -sh "${OPENPI_DATA_HOME:-$HOME/.cache/openpi}" "$HOME/.cache/huggingface" assets checkpoints data/libero/videos 2>/dev/null
docker system df
df -h .
```

某些目录尚不存在时，`du` 不显示它们是正常的。例如新安装后通常还没有本课程的 `assets/`、`checkpoints/` 和评测视频；它们会在后续章节生成。记录当前基线，之后才能判断增长来自模型、数据、训练还是视频。

如果 Docker 数据根目录位于另一个分区，`df -h .` 不能代表 Docker 的剩余空间。可以先用只读信息命令确认 Docker Root Dir，再对相应挂载点检查：

```bash
docker info
```

### 实践 5：在真正启动评测前检查 Compose 配置

本章不启动或构建容器，只让 Compose 展开配置：

```bash
docker compose -f examples/libero/compose.yml config
```

在输出中确认：

- 存在 `openpi_server` 和 `runtime` 两个 service；
- 两者使用 host networking；
- `runtime` 的 `MUJOCO_GL` 默认展开为 `egl`；
- 两个服务都声明了 NVIDIA GPU reservation；
- `openpi_server` 把宿主机的 openpi 缓存挂载到容器内 `/openpi_assets`。

这一步只能证明 Compose 文件能被解析，不能证明镜像已构建、容器能看到 GPU，也不能证明 Policy Server 已经启动。真正的双容器运行留到下一章。

## 预期结果

### 命令输出解释表

| 命令或现象 | 可以得出的结论 | 仍然不能证明什么 |
| --- | --- | --- |
| `nvidia-smi` 列出 A100/H100 | 宿主机能与 NVIDIA 驱动通信 | JAX 和 Docker 一定能使用 GPU |
| `uv --version` 输出版本 | `uv` 命令可用 | 项目依赖已经安装完整 |
| `docker compose version` 输出版本 | Compose 插件可用 | Docker daemon、GPU runtime 和镜像构建一定正常 |
| `docker run --rm --gpus all ... nvidia-smi` 列出 GPU | Docker daemon 能通过 NVIDIA runtime 把 GPU 暴露给这个测试容器 | LIBERO 与模型镜像已经构建或应用服务已经正常启动 |
| `git submodule status` 所需行首为空格 | 子模块已初始化，并位于父仓库记录的 commit | 子模块内部依赖或后续 LIBERO 运行一定正常 |
| `uv run --no-sync python --version` 显示 3.11.x | 主环境解释器符合课程约定 | JAX CUDA 后端可用 |
| `jax.devices()` 至少列出一个 GPU | 当前 `uv` 环境中的 JAX 看到了 GPU | Docker 中的两个服务也一定能看到 GPU |
| `docker compose ... config` 列出两个服务和 GPU reservation | Compose 文件解析成功，声明符合预期 | 容器已启动或 GPU 分配已实际生效 |
| `df -h .` 的 `Avail` ≥ 100 GB | 仓库所在文件系统达到课程磁盘下限 | Docker 数据目录或其他挂载点也有同样空间 |

安装与自检完成后，你应该留下一个简短记录，至少包含：

```text
宿主机：Ubuntu 22.04，GPU 型号，nvidia-smi 状态
主环境：Python 3.11.x，uv 版本，jax.devices() 摘要
容器：Docker Compose 版本，独立 CUDA 容器 GPU smoke test，Compose 配置可解析
子模块：课程所需行首为空格；任何非空白例外均记录路径、commit、原因和影响
磁盘：仓库分区可用空间，Docker 存储概况
渲染计划：默认 EGL；完整 GLX override 同时设置 MUJOCO_GL 与 PYOPENGL_PLATFORM
```

### 本文的验证边界

本章中的安装和预检命令是给学习者在目标机器上执行的操作说明，不是伪造的成功日志。编写本章时做了两类核对：

- **静态核对**：从 `README.md`、`pyproject.toml`、LIBERO `Dockerfile` 和 `compose.yml` 确认 Python 3.11/3.8 边界、安装命令、两个 service、host networking、缓存挂载和 EGL 默认值；
- **当前编写环境实测（2026-07-14）**：`uv --version` 与 `docker compose version` 可运行，仓库分区空间超过 100 GB；但子模块状态行首为 `-`，`nvidia-smi` 无法与驱动通信。因此该环境**没有通过**完整 preflight。

本章编写过程中没有运行或通过 `docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi`，也没有把 `uv run --no-sync python -c "import jax; print(jax.devices())"` 的结果作为已通过证据。因此不声称当前环境的 Docker GPU 或 JAX/GPU 检查成功。你的验收必须以自己目标机器的实际输出为准。

## 常见问题

下面每个问题都先收集证据，再决定处理方向。不要只根据报错末尾一句就重装所有组件。

### 1. 主环境用了错误的 Python 版本

先确认三个事实：

```bash
uv run --no-sync python --version
uv run --no-sync python -c "import sys; print(sys.executable)"
uv python list
```

若第一条不是 3.11.x，说明当前项目环境不符合课程约定。不要在 Python 3.8 的 LIBERO 环境里继续安装主项目依赖，也不要靠修改 `PATH` 猜测修复。先记录 `.venv` 中是否有手工安装且需要保留的内容，再让 `uv` 明确选择 3.11 并重新同步项目环境：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 2. `jax.devices()` 没有 GPU

按层比较：

```bash
nvidia-smi
uv run --no-sync python -c "import jax; print(jax.default_backend()); print(jax.devices())"
uv pip show jax jaxlib
```

- 如果 `nvidia-smi` 失败，先修宿主机驱动或 GPU 分配；Python 重装通常不能修复宿主机无法访问 GPU。
- 如果 `nvidia-smi` 成功但 JAX 只有 CPU，检查当前解释器、`jax`/`jaxlib` 安装来源、驱动兼容性和安装日志。
- 如果 import 本身失败，回看 `uv sync` 是否完整结束，而不是直接把它归类成 CUDA 错误。

保留完整 traceback 和 `nvidia-smi` 输出，查阅[故障排查手册](troubleshooting.md)的 JAX/GPU 分类。

### 3. 子模块缺失，LIBERO 文件找不到

先运行：

```bash
git submodule status
```

行首 `-` 表示未初始化；运行：

```bash
git submodule update --init --recursive
```

课程所需子模块的标准通过状态是行首为空格。若行首是 `+`，表示当前子模块 checkout 与父仓库记录的 commit 不同；若是 `U`，表示有冲突。不要把这两种情况当成“已经完全正常”，也不要覆盖自己在子模块中的改动。先用 `git -C third_party/libero status --short` 等命令确认是否有本地工作，再决定如何对齐。

如果确实要保留 `+`、`U` 或其他非空白状态，继续前至少记录：子模块路径、前缀、当前 commit、父仓库期望 commit、保留原因和可能影响。这是显式偏离课程基线，而不是标准验收通过。

### 4. `uv sync` 因网络、GitHub 或 LeRobot Git 依赖失败

先保存完整日志，并确认你使用了仓库要求的环境变量：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

然后单独检查 GitHub Git 访问，而不是反复删除环境或缓存：

```bash
git ls-remote https://github.com/huggingface/lerobot.git HEAD
uv cache dir
```

如果 `git ls-remote` 也失败，优先检查 DNS、代理、证书、组织防火墙或 GitHub 可达性；如果它成功而 `uv sync` 失败，再从完整日志定位包索引、固定 commit、构建工具或磁盘错误。普通 `uv run` 也可能触发同一同步路径，所以环境安装完成后的轻量检查使用 `uv run --no-sync`。

### 5. Docker Compose 能解析，但容器看不到 GPU

先确认宿主机，再确认容器层：

```bash
nvidia-smi
docker info
docker compose -f examples/libero/compose.yml config
```

实际 GPU 验收应运行独立 smoke test：

```bash
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

该命令首次运行可能拉取镜像。若因为网络、时间或磁盘原因暂缓执行，Docker GPU 访问仍是“未验证”，不能勾选本章完整 preflight。若独立 smoke test 成功，并且已经完成本项目镜像构建，还可进一步检查模型服务镜像：

```bash
docker compose -f examples/libero/compose.yml run --rm openpi_server nvidia-smi
```

宿主机成功、容器失败通常指向 NVIDIA Container Toolkit、Docker runtime 配置或 Compose GPU reservation，而不是 JAX 模型代码。`docker compose ... config` 只证明声明存在，不能证明 daemon 已成功分配 GPU。

### 6. 安装、构建或训练时磁盘耗尽

分别查看仓库分区、Docker 和常见缓存：

```bash
df -h .
docker system df
du -sh "${OPENPI_DATA_HOME:-$HOME/.cache/openpi}" "$HOME/.cache/huggingface" assets checkpoints data/libero/videos 2>/dev/null
```

根据证据判断增长来自 checkpoint、数据集、镜像层、build cache 还是视频。先确认文件是否仍被当前实验引用，再迁移或归档；不要把“清空所有缓存”当成第一步，也不要删除唯一的训练 checkpoint。

### 7. MuJoCo 出现 EGL 或 GLX 错误

先记录 `runtime` 日志中的第一处图形初始化错误：

```bash
docker compose -f examples/libero/compose.yml logs runtime
```

如果默认 EGL 报错，核对 GPU 可见性以及 Compose 中的 `MUJOCO_EGL_DEVICE_ID`、`NVIDIA_DRIVER_CAPABILITIES` 和 `PYOPENGL_PLATFORM`。只有证据指向 EGL 路径时，才回退 GLX。

注意：只在命令前设置 `MUJOCO_GL=glx` 不会覆盖基础 Compose 中硬编码的 `PYOPENGL_PLATFORM=egl`。完整回退应使用前文的双文件 Compose override，让 `runtime` 中 `MUJOCO_GL` 和 `PYOPENGL_PLATFORM` **同时**等于 `glx`，并先用 `docker compose ... config` 核对展开值。

GLX 仍失败时检查：

```bash
printf 'DISPLAY=%s\n' "$DISPLAY"
ls -ld /tmp/.X11-unix
```

GLX 需要 X11 socket 与访问权限。按 [LIBERO README](../../examples/libero/README.md) 处理 `xhost`，同时遵守共享服务器的安全策略。授权使用结束后，通常由图形会话所有者在同一个 `DISPLAY`/`XAUTHORITY` 上运行 `xhost -local:docker` 撤销。若机器本来就是无桌面的计算节点，优先修复 EGL，而不是强行搭建 GLX/X11。

## 本章验收

进入下一章前，逐项提供命令输出或口头解释：

- [ ] 能说出课程支持假设：Ubuntu 22.04、A100/H100、主项目 Python 3.11、Docker Compose、至少 100 GB 可用磁盘；
- [ ] `git submodule status` 的课程所需子模块行首为空格；若有意保留非空白状态，已记录路径、前缀、实际/期望 commit、原因和影响，并明确标为课程基线例外；
- [ ] `uv run --no-sync python --version` 输出 Python 3.11.x；
- [ ] `uv run --no-sync python -c "import jax; print(jax.devices())"` 至少列出一个 GPU；
- [ ] `docker compose version` 可用，`docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi` 能在容器中列出 GPU，且 `docker compose -f examples/libero/compose.yml config` 能解析；
- [ ] 能解释宿主机主 `uv` 环境负责数据/统计/训练，推荐评测由 `openpi_server` 和 Python 3.8 `runtime` 两个容器共同完成；
- [ ] 能明确说出 Policy Server 在推荐 Compose 工作流中运行于 `openpi_server` 容器，而不是永远运行在宿主机；
- [ ] 能解释为什么不能把 LIBERO 的旧依赖安装进主 `.venv`；
- [ ] 能指出 `OPENPI_DATA_HOME`、Hugging Face/LeRobot 缓存、Docker 存储、`assets/`、`checkpoints/` 和 `data/libero/videos/` 的不同用途；
- [ ] 能说明默认渲染是 EGL；完整 GLX 回退要同时覆盖 `MUJOCO_GL=glx` 与 `PYOPENGL_PLATFORM=glx`，并在使用后用 `xhost -local:docker` 撤销临时 X11 授权。

若 GPU、子模块或磁盘任一项未通过，不要把“Compose 文件能解析”当成本章完成。第 03 章会实际构建镜像、加载模型并运行 LIBERO，这些前置问题到那时只会更难区分。

## 思考题

### 1. 为什么推荐评测已经使用 Docker，仍然要安装宿主机 Python 3.11 `uv` 环境？

<details>
<summary>展开参考答案</summary>

课程后续的数据转换、normalization statistics、配置检查和训练都在宿主机主环境完成。Docker Compose 主要隔离推荐闭环评测中的模型服务与 LIBERO/MuJoCo 客户端；它不是对整个课程主环境的替代。

</details>

### 2. `docker compose ... config` 成功，能否证明 LIBERO 已经可以使用 GPU？

<details>
<summary>展开参考答案</summary>

不能。它只证明 Compose YAML 可以解析，并能看到 GPU reservation 声明。还需要宿主机 `nvidia-smi`、NVIDIA Container Toolkit 和实际容器内的 GPU 检查才能证明分配生效。

</details>

### 3. 普通 `uv run` 在一个简单 import 检查前出现 GitHub 网络错误，为什么不一定是被检查模块的代码有问题？

<details>
<summary>展开参考答案</summary>

普通 `uv run` 可能先协调或同步项目依赖；本仓库的 LeRobot 来源是固定 GitHub commit，因此运行目标代码之前就可能访问网络并失败。完成安装后使用 `uv run --no-sync`，可以把“现有环境中的 import 是否成功”与“是否能重新同步依赖”分开检查。

</details>

### 4. EGL 报错后只设置 `MUJOCO_GL=glx`，为什么还不能确认已经完整切换到 GLX？

<details>
<summary>展开参考答案</summary>

基础 Compose 除了可展开的 `MUJOCO_GL`，还硬编码了 `PYOPENGL_PLATFORM=egl`。只设置前者会留下互相矛盾的渲染选择器，不能保证 PyOpenGL 已切到 GLX。完整回退需要用 Compose override 同时把两者设为 `glx`，再核对 X11 socket、`DISPLAY` 与权限；用完临时授权后运行 `xhost -local:docker`。

</details>

[← 上一章：机器人学习与 VLA 基础](01_vla_and_robotics_basics.md) · [课程入口：README](README.md) · [下一章：第一次 LIBERO 推理 →](03_first_libero_inference.md)
