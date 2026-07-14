# openpi 零基础章节式教程 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `docs/tutorial/` 中交付一套面向具备 Python 基础的 openpi 初学者的中文章节式教程，使学习者能够在 A100/H100 上完成 LIBERO 预训练推理、小数据适配、LoRA 微调、策略服务和仿真评测闭环。

**Architecture:** 教程使用“先跑通、再拆解、后改造”的螺旋主线。主训练环境使用仓库 Python 3.11 `uv` 环境，LIBERO 仿真优先使用项目 Docker 环境，二者通过 WebSocket 解耦；教程只新增文档，不修改模型核心代码，数据转换脚本和训练配置的改动以完整学习练习呈现。

**Tech Stack:** Markdown、Python 3.11、uv、JAX/Flax NNX、LeRobot、Docker Compose、LIBERO、WebSocket、π₀/π₀.₅、LoRA

---

## 文件职责图

**新增课程文件：**

- `docs/tutorial/README.md`：课程入口、导航、路径选择和最终产物。
- `docs/tutorial/00_course_outline.md`：完整大纲、课时、资源、里程碑与总体验收。
- `docs/tutorial/01_vla_and_robotics_basics.md`：机器人学习和 VLA 入门。
- `docs/tutorial/02_environment_setup.md`：主环境、Docker 仿真环境与设备自检。
- `docs/tutorial/03_first_libero_inference.md`：官方 π₀.₅ checkpoint 的首次闭环推理。
- `docs/tutorial/04_openpi_architecture.md`：一次推理请求经过的组件和代码入口。
- `docs/tutorial/05_data_and_transforms.md`：数据协议、变换、归一化和 action chunk。
- `docs/tutorial/06_jax_and_pi0_basics.md`：理解训练流程所需的 JAX、NNX 和 flow matching。
- `docs/tutorial/07_prepare_small_dataset.md`：小规模 RLDS 到 LeRobot 转换练习。
- `docs/tutorial/08_configure_finetuning.md`：教学 LoRA 配置与 normalization statistics。
- `docs/tutorial/09_run_finetuning.md`：短程训练、指标、checkpoint 和恢复。
- `docs/tutorial/10_serve_and_evaluate.md`：服务教学 checkpoint 并进行 LIBERO 评测。
- `docs/tutorial/11_next_steps.md`：完整微调、自有数据和真机迁移路线。
- `docs/tutorial/glossary.md`：中英文术语和缩写速查。
- `docs/tutorial/troubleshooting.md`：按数据流分层的故障诊断手册。

**只读引用，不纳入修改：**

- `README.md`
- `examples/libero/README.md`
- `examples/libero/main.py`
- `examples/libero/compose.yml`
- `examples/libero/convert_libero_data_to_lerobot.py`
- `scripts/compute_norm_stats.py`
- `scripts/train.py`
- `scripts/serve_policy.py`
- `src/openpi/models/model.py`
- `src/openpi/models/pi0.py`
- `src/openpi/policies/policy.py`
- `src/openpi/policies/libero_policy.py`
- `src/openpi/training/config.py`
- `src/openpi/training/data_loader.py`
- `docs/architecture_zh.md`
- `docs/jax_in_openpi_zh.md`
- `docs/model_build_and_inference_zh.md`

现有 `.vscode/settings.json` 和三篇未跟踪中文文档属于用户工作。所有提交必须使用精确文件路径，只暂存本计划新增的 `docs/tutorial/` 文件。

除课程入口、总纲、术语表和排错手册外，第 01–11 章统一包含：上一章/课程入口/下一章导航、本章目标、前置条件、核心概念、源码地图、动手实践、预期结果、常见问题、本章验收和思考题。命令默认从仓库根目录运行；多终端命令必须标注“模型/训练端”或“LIBERO 仿真端”。

### Task 1: 建立课程入口和教学大纲

**Files:**

- Create: `docs/tutorial/README.md`
- Create: `docs/tutorial/00_course_outline.md`

- [ ] **Step 1: 写课程入口页**

在 `README.md` 中写入以下完整结构：课程定位、适用人群、硬件假设、16–24 小时时间预算、最终闭环、15 个文件的导航表、主线与只读路径、环境标签约定、学习产物和开始学习链接。首屏必须明确：预训练演示使用 `pi05_libero`，教学 LoRA 使用 `pi0_libero_tutorial`，两者不可混用 checkpoint。

课程入口的数据流使用下面的精确顺序：

```text
LIBERO observation
  → WebSocket client
  → Policy Server
  → LiberoInputs
  → normalize / resize / tokenize / pad
  → π₀ 或 π₀.₅
  → unnormalize / LiberoOutputs
  → action chunk
  → LIBERO environment
```

- [ ] **Step 2: 写课程总纲**

在 `00_course_outline.md` 中为第 01–11 章分别列出学习目标、核心概念、实践任务、预计阅读/操作时间、产物和验收。设置三个里程碑：第 03 章完成预训练闭环，第 08 章完成自有小数据与配置，第 10 章完成教学 checkpoint 闭环。

总纲明确资源预算：Ubuntu 22.04、Python 3.11 主环境、Docker、A100/H100、至少 100 GB 可用磁盘；下载、镜像构建、首次 JIT 和训练时间不计入 16–24 小时学习时间。

- [ ] **Step 3: 检查入口和总纲互链**

Run:

```bash
rg -n '00_course_outline|README.md|pi05_libero|pi0_libero_tutorial' docs/tutorial/README.md docs/tutorial/00_course_outline.md
```

Expected: 两个文件均有导航链接，课程入口同时解释两个配置名。

- [ ] **Step 4: 提交课程骨架**

```bash
git add docs/tutorial/README.md docs/tutorial/00_course_outline.md
git commit -m "docs: add openpi tutorial outline"
```

### Task 2: 编写机器人学习与 VLA 基础

**Files:**

- Create: `docs/tutorial/01_vla_and_robotics_basics.md`

- [ ] **Step 1: 写 observation、action 和 episode 的最小模型**

使用 LIBERO 单臂任务解释 observation、7 维 action、episode、trajectory、demonstration 和 policy。给出具体字段示例：两路 `224×224×3` 图像、8 维 state、自然语言 prompt、7 维 action。

- [ ] **Step 2: 写模仿学习、VLA 和 action chunk**

用“看图和指令后一次规划若干步动作”的例子解释 VLA 与 action chunk；明确 π₀/π₀.₅ 使用连续动作 flow matching，π₀-FAST 使用动作 token 自回归生成。本章只给直觉，不展开数学推导。

- [ ] **Step 3: 加入可运行的数据形状练习**

练习使用 `src/openpi/policies/libero_policy.py` 中的 `make_libero_example()`，要求学习者打印 key、shape 和 dtype，并将结果与本章协议表对照。

```bash
uv run python -c "from openpi.policies.libero_policy import make_libero_example; x=make_libero_example(); print({k: getattr(v, 'shape', type(v).__name__) for k, v in x.items()})"
```

Expected: state 为 `(8,)`，两路图像为 `(224, 224, 3)`，prompt 为字符串。

- [ ] **Step 4: 添加本章验收与思考题**

验收要求学习者能口头解释“为什么模型输出 action chunk 而不是只输出一步”，并能指出 observation 与 action 的维度来源。

- [ ] **Step 5: 提交基础章节**

```bash
git add docs/tutorial/01_vla_and_robotics_basics.md
git commit -m "docs: teach robotics and VLA foundations"
```

### Task 3: 编写环境安装与双运行时说明

**Files:**

- Create: `docs/tutorial/02_environment_setup.md`

- [ ] **Step 1: 写主训练环境安装流程**

按仓库 `README.md` 写子模块、uv 和项目依赖安装，明确 Python 3.11、Ubuntu 22.04 和 `GIT_LFS_SKIP_SMUDGE=1` 的原因。

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

- [ ] **Step 2: 写 GPU 与软件自检**

```bash
nvidia-smi
uv --version
uv run python -c "import jax; print(jax.devices())"
docker compose version
git submodule status
```

Expected: JAX 输出至少一个 GPU 设备，子模块行首不出现 `-`，Docker Compose 可用。

- [ ] **Step 3: 解释双运行时边界**

用表格对比主 `uv` 环境和 LIBERO Docker：前者使用 Python 3.11、JAX、训练与 Policy Server；后者使用 Python 3.8、LIBERO/MuJoCo 和轻量 `openpi-client`。明确不要把 LIBERO 旧依赖安装进主环境。

- [ ] **Step 4: 写缓存、磁盘和渲染检查**

解释 `OPENPI_DATA_HOME`、Hugging Face/LeRobot 数据缓存、Docker 镜像和 checkpoint 的磁盘用途；分别记录 EGL 默认路径和 `MUJOCO_GL=glx` 回退条件。

- [ ] **Step 5: 提交环境章节**

```bash
git add docs/tutorial/02_environment_setup.md
git commit -m "docs: add openpi tutorial environment setup"
```

### Task 4: 编写第一次 LIBERO 预训练推理

**Files:**

- Create: `docs/tutorial/03_first_libero_inference.md`

- [ ] **Step 1: 解释两容器闭环**

说明 `runtime` 容器运行 LIBERO 客户端，`openpi_server` 容器加载 `pi05_libero`，两者使用 host network 的 8000 端口通信。链接 `examples/libero/compose.yml`、`examples/libero/main.py` 和 `scripts/serve_policy.py`。

- [ ] **Step 2: 写最小预训练启动命令**

```bash
export SERVER_ARGS="--env LIBERO"
export CLIENT_ARGS="--args.task-suite-name libero_spatial --args.num-trials-per-task 1"
docker compose -f examples/libero/compose.yml up --build
```

Expected: 服务端加载 `pi05_libero`，客户端打印 task description、episode 进度和 success 状态，并在 `data/libero/videos` 产生回放视频。

- [ ] **Step 3: 写 GLX 回退和安全停止方法**

出现 EGL 初始化错误时使用：

```bash
MUJOCO_GL=glx docker compose -f examples/libero/compose.yml up --build
```

说明 `Ctrl+C` 后运行 `docker compose -f examples/libero/compose.yml ps` 检查容器状态；不建议用删除镜像或缓存作为首个排错动作。

- [ ] **Step 4: 添加观察任务**

要求学习者在 `examples/libero/main.py` 中定位图像翻转、state 拼接、prompt、`replan_steps` 和 action queue，并画出一轮请求的输入输出。

- [ ] **Step 5: 提交首次推理章节**

```bash
git add docs/tutorial/03_first_libero_inference.md
git commit -m "docs: add first LIBERO inference lesson"
```

### Task 5: 编写 openpi 架构与数据变换课程

**Files:**

- Create: `docs/tutorial/04_openpi_architecture.md`
- Create: `docs/tutorial/05_data_and_transforms.md`

- [ ] **Step 1: 按一次请求写架构章节**

从 `WebsocketClientPolicy.infer()` 开始，依次解释 WebSocket server、`policy_config.create_trained_policy()`、`Policy.infer()`、input transforms、`BaseModel.sample_actions()` 和 output transforms。每个组件给出“做什么、输入输出、依赖什么”三项说明。

- [ ] **Step 2: 加入目录地图和源码阅读顺序**

目录地图覆盖 `packages/openpi-client`、`src/openpi/serving`、`src/openpi/policies`、`src/openpi/transforms.py`、`src/openpi/models` 和 `src/openpi/training`。进阶链接指向 `../architecture_zh.md` 和 `../model_build_and_inference_zh.md`。

- [ ] **Step 3: 写 LIBERO 字段协议**

在数据章节用表格列出原始 key、`RepackTransform` 后 key、统一 `Observation` 字段、shape、dtype 和语义。必须覆盖 image、wrist image、state、actions、prompt 和 masks。

- [ ] **Step 4: 写三类 transform 与 normalization**

解释 `repack_transforms` 只用于数据集、`data_transforms` 同时用于训练和推理、`model_transforms` 负责 resize/tokenize/pad；解释 normalization statistics 必须和数据源、动作定义、训练配置一致。

- [ ] **Step 5: 写 action delta 和 padding 风险**

以 `LeRobotLiberoDataConfig.extra_delta_transform`、`LiberoInputs` 和 `LiberoOutputs` 为例，说明 7 维 LIBERO action 如何 padding 到模型维度再裁回 7 维，以及旧 π₀ checkpoint 使用额外 delta transform、π₀.₅ 配置不使用该变换的区别。

- [ ] **Step 6: 提交架构和数据章节**

```bash
git add docs/tutorial/04_openpi_architecture.md docs/tutorial/05_data_and_transforms.md
git commit -m "docs: explain openpi architecture and data flow"
```

### Task 6: 编写 JAX、NNX 与 π₀ 最小知识章节

**Files:**

- Create: `docs/tutorial/06_jax_and_pi0_basics.md`

- [ ] **Step 1: 写 NumPy 到 JAX 的迁移直觉**

解释 array、device、PyTree、显式 PRNG key、`jax.jit` 和首次编译。用 `Observation` 是 PyTree、训练 batch 是 `(Observation, Actions)` 的项目实例连接概念。

- [ ] **Step 2: 写 Flax NNX 状态模型**

解释 `nnx.Module`、`GraphDef`、参数 `State`、`nnx.split/merge`、trainable filter 和 LoRA 冻结边界，链接 `scripts/train.py` 与 `src/openpi/shared/nnx_utils.py`。

- [ ] **Step 3: 写 flow matching 训练和采样直觉**

使用“真实动作与噪声之间的带时间插值”解释 loss，使用“从噪声逐步积分到动作”解释推理。包含下面四行核心关系，并逐项解释变量：

```python
x_t = time * noise + (1 - time) * actions
u_t = noise - actions
v_t = model(observation, x_t, time)
loss = mean(square(v_t - u_t))
```

- [ ] **Step 4: 写 JIT、显存和调试边界**

明确首次调用慢不等于卡死，shape 改变可能触发重新编译，冻结参数仍占权重显存，batch size 与参数显存是不同问题。进阶链接指向 `../jax_in_openpi_zh.md`。

- [ ] **Step 5: 提交 JAX 章节**

```bash
git add docs/tutorial/06_jax_and_pi0_basics.md
git commit -m "docs: teach JAX and pi0 fundamentals"
```

### Task 7: 编写小规模 LIBERO 数据转换练习

**Files:**

- Create: `docs/tutorial/07_prepare_small_dataset.md`

- [ ] **Step 1: 写 RLDS、LeRobot 和字段映射说明**

解释原始 LIBERO RLDS episode、LeRobot frame/episode、fps 和 task 字段。把转换脚本 features 与 `LeRobotLiberoDataConfig.repack_transform` 一一对应。

- [ ] **Step 2: 给出安全的小数据修改补丁**

教程要求学习者先创建分支或保存 diff，再将 `examples/libero/convert_libero_data_to_lerobot.py` 的数据集名改为单个 `libero_spatial_no_noops`，输出 repo id 改为 `openpi-tutorial/libero-small`，并为 `main()` 添加 `max_episodes_per_dataset: int | None = 10`。循环中使用 `enumerate()`，在达到上限时 `break`。

必须展示完整的修改后 `main()`：

```python
REPO_NAME = "openpi-tutorial/libero-small"
RAW_DATASET_NAMES = ["libero_spatial_no_noops"]


def main(
    data_dir: str,
    *,
    push_to_hub: bool = False,
    max_episodes_per_dataset: int | None = 10,
):
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode_index, episode in enumerate(raw_dataset):
            if max_episodes_per_dataset is not None and episode_index >= max_episodes_per_dataset:
                break
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
```

章节明确原脚本会删除同名输出目录，运行前必须确认 `REPO_NAME`，且教学数据不推送到 Hub。

- [ ] **Step 3: 写转换命令和数据检查**

```bash
export LIBERO_RLDS_DIR=/data/libero_rlds
uv run examples/libero/convert_libero_data_to_lerobot.py \
  --data-dir "$LIBERO_RLDS_DIR" \
  --max-episodes-per-dataset 10
```

随后使用 `LeRobotDatasetMetadata("openpi-tutorial/libero-small")` 打印 fps、episode 数、frame 数、features 和 tasks。验收要求两路图像、8 维 state、7 维 actions 和非空 task 均存在。

- [ ] **Step 4: 写数据质量检查表**

检查 episode 边界、图像方向、dtype、NaN、动作范围、prompt 解码和演示长度；说明数据能加载不代表动作语义正确。

- [ ] **Step 5: 提交数据准备章节**

```bash
git add docs/tutorial/07_prepare_small_dataset.md
git commit -m "docs: add small LIBERO dataset lesson"
```

### Task 8: 编写教学微调配置与归一化课程

**Files:**

- Create: `docs/tutorial/08_configure_finetuning.md`

- [ ] **Step 1: 逐项解释 `TrainConfig`**

解释 model、data、weight loader、freeze filter、EMA、batch size、steps、logging、checkpoint 和 W&B 开关。明确配置名同时绑定模型结构、数据协议、normalization assets 和 checkpoint 加载方式。

- [ ] **Step 2: 给出完整教学配置块**

教程要求学习者在 `_CONFIGS` 的 LIBERO 配置区域加入以下配置，且模型与 freeze filter 使用完全相同的 LoRA variants：

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

- [ ] **Step 3: 写配置发现和 shape 自检**

使用 `_config.get_config("pi0_libero_tutorial")` 打印 config name、repo id、batch size、steps、model type 和 action horizon。Expected: repo id 为 `openpi-tutorial/libero-small`，steps 为 500。

- [ ] **Step 4: 写 normalization statistics 命令**

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_libero_tutorial \
  --max-frames 2000
```

Expected: 输出路径位于 `assets/pi0_libero_tutorial/openpi-tutorial/libero-small`，state 与 actions 均有统计量。解释 `max_frames` 只用于教学提速，正式训练应覆盖完整数据分布。

- [ ] **Step 5: 写配置一致性验收**

要求学习者检查 repo id、动作 delta 语义、LoRA model/freeze filter、统计量路径和后续 Policy Server config name 五者一致。

- [ ] **Step 6: 提交配置章节**

```bash
git add docs/tutorial/08_configure_finetuning.md
git commit -m "docs: add tutorial finetuning configuration"
```

### Task 9: 编写短程 LoRA 微调与 checkpoint 课程

**Files:**

- Create: `docs/tutorial/09_run_finetuning.md`

- [ ] **Step 1: 写训练前检查**

检查 GPU、数据元信息、normalization assets、checkpoint 输出目录和磁盘。解释 `--overwrite` 会覆盖同名实验，首次教学运行使用固定实验名 `libero_lora_demo`。

- [ ] **Step 2: 写训练命令**

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi0_libero_tutorial \
  --exp-name libero_lora_demo \
  --overwrite
```

Expected: 首先打印 data loader 和 train state 结构；首次 JIT 后每 10 step 输出 loss、grad norm 和 param norm；在 100、200、300、400 和最终 499 附近保存 checkpoint。

- [ ] **Step 3: 解释训练日志和异常证据**

分别解释 loss、grad norm、param norm、JIT 时间、数据加载等待、OOM 和 NaN。明确 500 step 只验证链路，loss 不保证单调下降，仿真成功率不用于和官方 30k-step checkpoint 比较。

- [ ] **Step 4: 写 checkpoint 目录和恢复流程**

目标目录为 `checkpoints/pi0_libero_tutorial/libero_lora_demo/`。说明最终 step 为 `499` 的原因，并给出不带 `--overwrite`、改用 `--resume` 的恢复命令；解释 `resume` 与 `overwrite` 不能同时使用。

- [ ] **Step 5: 写训练验收**

验收要求 loss 和 norms 为有限值、至少一个 checkpoint 存在、checkpoint 中包含推理参数和 assets，且训练进程正常等待异步保存完成。

- [ ] **Step 6: 提交训练章节**

```bash
git add docs/tutorial/09_run_finetuning.md
git commit -m "docs: add LoRA finetuning lesson"
```

### Task 10: 编写教学 checkpoint 服务与仿真评测

**Files:**

- Create: `docs/tutorial/10_serve_and_evaluate.md`

- [ ] **Step 1: 写训练配置和 checkpoint 匹配规则**

明确服务时必须使用 `pi0_libero_tutorial`，不能用首次演示的 `pi05_libero` 加载 π₀ LoRA checkpoint。解释 checkpoint 的 assets 会为 Policy 提供 normalization statistics。

- [ ] **Step 2: 写 Docker Compose 启动命令**

```bash
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi0_libero_tutorial --policy.dir ./checkpoints/pi0_libero_tutorial/libero_lora_demo/499"
export CLIENT_ARGS="--args.task-suite-name libero_spatial --args.num-trials-per-task 1"
docker compose -f examples/libero/compose.yml up --build
```

Expected: server 从教学目录加载 checkpoint，client 连接 8000 端口并接收至少 5 步 action chunk 供 `replan_steps=5` 使用。

- [ ] **Step 3: 写评测结果解释**

说明 LIBERO 会遍历 task suite，记录每个 episode 的 success 并保存视频。教学验收是完整 episode 能推进且无字段/shape/服务错误；短程小数据模型失败不等于部署失败。

- [ ] **Step 4: 写服务分层排错**

依次检查 checkpoint 路径、config name、assets、server 日志、8000 端口、client host、action horizon、EGL/GLX 和回放目录。所有排错先收集证据，再修改状态。

- [ ] **Step 5: 提交服务与评测章节**

```bash
git add docs/tutorial/10_serve_and_evaluate.md
git commit -m "docs: add policy serving and evaluation lesson"
```

### Task 11: 编写扩展路线、术语表和排错手册

**Files:**

- Create: `docs/tutorial/11_next_steps.md`
- Create: `docs/tutorial/glossary.md`
- Create: `docs/tutorial/troubleshooting.md`

- [ ] **Step 1: 写三条扩展路线**

`11_next_steps.md` 分别给出：扩大数据和训练步数、在 A100/H100 上使用 `pi05_libero` 完整微调、自有机器人/数据接入。每条路线列出必须保持不变的协议和需要替换的组件，不承诺官方指标。

- [ ] **Step 2: 写术语表**

`glossary.md` 至少包含 observation、state、action、action chunk、episode、trajectory、policy、VLA、flow matching、token、prompt、RLDS、LeRobot、transform、normalization、checkpoint、LoRA、JAX、PyTree、JIT、XLA、Flax NNX、FSDP、Policy Server 和 WebSocket。每项包含英文、中文和一句项目语境解释。

- [ ] **Step 3: 写分层排错手册**

`troubleshooting.md` 按安装、GPU/JAX、数据、transform、normalization、训练、checkpoint、服务、LIBERO 渲染九类组织。每项使用“症状 → 证据命令 → 常见原因 → 处理”的固定格式。

- [ ] **Step 4: 加入进阶文档链接**

从扩展章节和术语表链接 `../architecture_zh.md`、`../jax_in_openpi_zh.md`、`../model_build_and_inference_zh.md`、`../norm_stats.md` 与 `../remote_inference.md`。

- [ ] **Step 5: 提交参考章节**

```bash
git add docs/tutorial/11_next_steps.md docs/tutorial/glossary.md docs/tutorial/troubleshooting.md
git commit -m "docs: add tutorial references and troubleshooting"
```

### Task 12: 完成全套文档验证与交付检查

**Files:**

- Verify: `docs/tutorial/*.md`
- Verify: `docs/superpowers/specs/2026-07-14-openpi-tutorial-design.md`

- [ ] **Step 1: 检查文件集合**

Run:

```bash
rg --files docs/tutorial | sort
```

Expected: 正好列出设计中的 15 个 Markdown 文件。

- [ ] **Step 2: 检查未完成标记和空章节**

Run:

```bash
rg -n '[T]BD|[T]ODO|待补充|以后再写|占位' docs/tutorial
```

Expected: 无输出。逐个文件运行 `wc -l`，确认每章都包含目标、实践、预期结果、排错和验收，而不是只有标题。

- [ ] **Step 3: 检查本地 Markdown 链接**

Run:

```bash
python3 -c 'import pathlib,re,sys; files=list(pathlib.Path("docs/tutorial").glob("*.md")); bad=[]; pattern=re.compile(r"\[[^]]*\]\(([^)]+)\)"); [(bad.append((str(f),u)) if not (f.parent/u.split("#",1)[0]).resolve().exists() else None) for f in files for u in pattern.findall(f.read_text()) if u and not u.startswith(("http://","https://","mailto:","#"))]; print(f"checked={len(files)} broken={len(bad)}"); [print(f"{f}: {u}") for f,u in bad]; sys.exit(bool(bad))'
```

Expected: `checked=15 broken=0`。

- [ ] **Step 4: 静态核对配置名、脚本路径和 CLI 参数**

Run:

```bash
rg -n 'name="pi0_libero_low_mem_finetune"|name="pi05_libero"' src/openpi/training/config.py
rg -n 'pi0_libero_tutorial|compute_norm_stats.py|scripts/train.py|scripts/serve_policy.py' docs/tutorial
rg -n 'task_suite_name|num_trials_per_task|replan_steps' examples/libero/main.py
```

Expected: 上游配置与脚本字段存在，教程中的教学配置名在第 08–10 章保持一致。

- [ ] **Step 5: 运行轻量示例并检查 Markdown diff**

Run:

```bash
uv run python -c "from openpi.policies.libero_policy import make_libero_example; x=make_libero_example(); assert x['observation/state'].shape == (8,); assert x['observation/image'].shape == (224, 224, 3); assert x['observation/wrist_image'].shape == (224, 224, 3); print('LIBERO example OK')"
git diff --check -- docs/tutorial
```

Expected: 打印 `LIBERO example OK`，`git diff --check` 无输出。若依赖尚未安装，记录该限制，不得声称该轻量示例已经验证通过。

- [ ] **Step 6: 按设计逐项审阅范围和术语一致性**

对照设计文档第 3、5、7、8、9 节，确认章节齐全、π₀/π₀.₅ 路径区分清楚、双运行时边界明确、排错按数据流组织、未把完整训练指标写成教学承诺。

- [ ] **Step 7: 提交最终导航修正**

如验证过程中修正了链接、术语或命令，只暂存 `docs/tutorial/`：

```bash
git add docs/tutorial
git commit -m "docs: verify openpi tutorial navigation"
```

如果验证未产生修改，则跳过该提交，并在交付说明中列出实际运行的检查命令和结果。
