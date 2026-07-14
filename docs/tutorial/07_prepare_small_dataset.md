# 第 07 章：准备小规模 LIBERO 数据集

[← 上一章：JAX、NNX 与 π₀ 基础](06_jax_and_pi0_basics.md) · [课程入口：README](README.md) · [下一章：配置教学微调 →](08_configure_finetuning.md)

## 本章目标

完成本章后，你应该能够：

- 区分 RLDS 的 episode/step 与 LeRobot 的 episode/frame，并解释 `fps=10` 和 task/language 字段怎样保留时间与任务语义；
- 逐项核对 LIBERO RLDS 到 LeRobot 的字段映射：两路 256×256×3 图像、8 维 state、7 维 actions 和非空 task；
- 说明数据转换是协议适配，不是把若干文件复制到另一个目录；
- 在不写出 LeRobot 数据集的前提下，用文件系统和 `tfds.load` 对原始数据做预检；
- 在专用 Git 分支上把现有转换器改成最多 10 个 episode 的教学版本，并在运行前识别它会删除的精确输出目录；
- 用 `LeRobotDatasetMetadata`、实际 frame 和图像预览检查结构、数值与语义，而不把“能加载”误当成“数据正确”。

本章是**源码修改练习**：学习者会临时修改转换脚本，但本课程仓库中的转换器、训练配置和数据 loader 不需要由本章文档代替。请把练习 diff 留在专用分支，理解后再决定是否保留。

## 前置条件

开始前应已完成[第 02 章](02_environment_setup.md)的宿主机环境安装，并读完[第 05 章](05_data_and_transforms.md)。尤其要知道：转换后的字段还会经过 repack、LIBERO data transform、normalization 和 model transform，不能把磁盘中的 256×256 图像直接等同于模型最终看到的张量。

### 使用宿主机 Python 3.11 主环境

本章转换使用仓库根目录的 **Python 3.11 `uv` 主环境**。无论是 Docker 内的 LIBERO 评测客户端，还是 `examples/libero/.venv` 对应的无 Docker Python 3.8 评测环境，都不要拿来做转换；它们服务于旧版 LIBERO/MuJoCo 仿真，不是数据准备环境。

[`pyproject.toml`](../../pyproject.toml) 把 TensorFlow、TFDS 和 `dlimp` 放在可选的 `rlds` dependency group 中。若第 02 章只同步了主依赖，从仓库根目录运行一次：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --group rlds
```

`GIT_LFS_SKIP_SMUDGE=1` 用于避免同步固定 Git 版本的 LeRobot 依赖时自动拉取无关的 Git LFS 大文件。完成同步后，本章运行命令统一使用 `uv run --no-sync`；它假设依赖已经准备好，不会在转换途中重新解析或下载环境。

先验证解释器和导入：

```bash
uv run --no-sync python --version
uv run --no-sync python -c \
  'import tensorflow as tf, tensorflow_datasets as tfds; print(tf.__version__, tfds.__version__)'
```

Python 应为 3.11，且两个 import 均成功。不要用临时的系统 `pip install` 把包塞进另一个解释器，否则后续命令可能仍然找不到依赖。

### 原始数据由学习者准备，本章不自动下载

转换器和项目 README 指向公开的 [modified LIBERO RLDS 数据集](https://huggingface.co/datasets/openvla/modified_libero_rlds)。请自行阅读数据集说明、确认许可与磁盘空间，再把需要的 `libero_spatial_no_noops` 放到本机目录。本章不会自动下载数据，也不会假设 `/data/libero_rlds` 已存在。

`--data-dir` 必须指向能被下面调用识别的 **TFDS/RLDS 根目录**：

```python
tfds.load("libero_spatial_no_noops", data_dir=data_dir, split="train")
```

它通常不是某个任意图片目录，也不一定是版本子目录本身。目录中应保留 TFDS builder、版本、shard 和 `dataset_info` 等布局；单独复制 `.tfrecord` 而丢失元数据，通常不能满足转换器的读取契约。

## 核心概念

### RLDS episode/step 与 LeRobot episode/frame

RLDS 把一次完整演示表示为一个 **episode**，其中 `episode["steps"]` 是按时间排列的 step 数据集。每个 step 包含当时的 observation、执行的 action 和语言指令。episode 边界非常重要：末帧之后不能继续拼接下一次演示的动作。

LeRobot 也保留 episode，但写入接口以 **frame** 为单位：

```text
RLDS episode
  ├─ step 0 ── add_frame(...) ┐
  ├─ step 1 ── add_frame(...) ├─ LeRobot episode
  └─ step N ── add_frame(...) ┘
                 save_episode()
```

转换器为一个 RLDS episode 的所有 step 调用 `dataset.add_frame(...)`，然后只调用一次 `dataset.save_episode()`。如果把 `save_episode()` 放进 step 循环，每帧都会错误地变成独立 episode；如果忘记调用，episode 边界和尾部数据都可能不正确。

这里的 `fps=10` 表示相邻 frame 的标称时间间隔是 0.1 秒。训练 loader 在 [`data_loader.py`](../../src/openpi/training/data_loader.py) 中读取 `dataset_meta.fps`，把 action horizon 转成 `0/fps, 1/fps, ...` 的时间偏移。因此 fps 不只是展示视频的参数；写错它会改变 action chunk 的时间索引。以 horizon 10 为例，十个索引的偏移为 0.0 到 0.9 秒。

### task/language 是训练条件，不是装饰性文本

原始 step 的 `language_instruction` 是字节串，当前转换器用 `.decode()` 得到 Python 字符串，并作为 `task` 传给 LeRobot。LeRobot 会维护 task 映射并在 frame 中使用 `task_index`；openpi 的 `PromptFromLeRobotTask(dataset_meta.tasks)` 再把索引恢复成训练输入中的 `prompt`。

因此链路是：

```text
RLDS language_instruction bytes
  → UTF-8 decode
  → LeRobot task / task_index + metadata.tasks
  → openpi prompt
  → tokenizer
```

空字符串、错误编码、截断或把所有 task 错写成同一句话，都可能让数据“可以加载”却失去语言条件。不要为消除 decode 异常直接使用 `errors="ignore"`；先检查原始类型和真实编码。

### 字段映射必须与训练协议逐项对齐

当前 [`convert_libero_data_to_lerobot.py`](../../examples/libero/convert_libero_data_to_lerobot.py) 的映射如下。

| RLDS step 来源 | LeRobot 写入 key | 声明的存储 feature | openpi 后续读取 |
| --- | --- | --- | --- |
| `step["observation"]["image"]` | `image` | `dtype="image"`，`shape=(256,256,3)` | repack 为 `observation/image`，再成为 `base_0_rgb` |
| `step["observation"]["wrist_image"]` | `wrist_image` | `dtype="image"`，`shape=(256,256,3)` | repack 为 `observation/wrist_image`，再成为 `left_wrist_0_rgb` |
| `step["observation"]["state"]` | `state` | `float32[8]` | repack 为 `observation/state` |
| `step["action"]` | `actions` | `float32[7]` | 保持为 `actions`，再进入 action chunk/transform |
| `step["language_instruction"].decode()` | `task` | LeRobot task 映射，而非上表中的数值/图像 feature | `task_index → metadata.tasks → prompt` |

两路图像在转换数据集中声明为 256×256×3；这是**模型 resize 之前**的存储协议。后续 `ModelTransformFactory` 使用 `ResizeImages(224, 224)`，`LiberoInputs` 还会把 LeRobot 解码得到的图像解析成 `uint8 HWC`。磁盘 metadata 的 `(256,256,3)`、loader 可能返回的 `float32 CHW`，以及模型最终的 224×224 输入是三个不同阶段，详见[第 05 章](05_data_and_transforms.md)。

### 转换是协议适配，不是文件复制

一次正确转换至少做了这些工作：

1. 从 TFDS/RLDS 的嵌套 episode/step 数据流读取样本；
2. 重命名并验证 observation/action 字段与维度；
3. 解码语言并建立 task 索引；
4. 按 frame 写入图像、state 和 actions；
5. 用 `save_episode()` 保留演示边界；
6. 写入 fps、robot type、features、episode/frame 索引等 LeRobot metadata；
7. 按 LeRobot/Hugging Face 的本地数据集布局保存，供 loader 和可选 Hub 上传使用。

只复制 TFRecord、图片或目录，无法自动产生这些协议关系。反过来，即使结构转换成功，动作坐标系、夹爪方向、相机朝向或语言语义仍可能错误，所以必须再做数据质量检查。

### LeRobot 输出目录与 Hugging Face 缓存

当前转换器用 `HF_LEROBOT_HOME / REPO_NAME` 组成输出路径。它通常落在用户级 Hugging Face/LeRobot cache 中；`REPO_NAME="openpi-tutorial/libero-small"` 会在该根目录下形成两层相对路径。具体根目录受 LeRobot 固定版本和 Hugging Face 环境变量影响；它不等于 openpi checkpoint 的 `OPENPI_DATA_HOME`，也不应在教程里写死成某台机器的绝对路径。

运行前用当前环境中的常量发现真实路径：

```bash
uv run --no-sync python - <<'PY'
import os
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

repo_id = "openpi-tutorial/libero-small"
output_path = Path(HF_LEROBOT_HOME) / repo_id
print("HF_HOME env:", os.environ.get("HF_HOME", "<not set>"))
print("HF_LEROBOT_HOME:", HF_LEROBOT_HOME)
print("exact output path:", output_path)
print("already exists:", output_path.exists())
PY
```

可进一步阅读项目固定的 [LeRobot commit](https://github.com/huggingface/lerobot/tree/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5) 和 [Hugging Face cache 指南](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)。以当前环境打印出的 `HF_LEROBOT_HOME` 为准，不要通过猜测路径去清理整个缓存。

## 源码地图

| 文件或资料 | 本章阅读重点 |
| --- | --- |
| [`examples/libero/convert_libero_data_to_lerobot.py`](../../examples/libero/convert_libero_data_to_lerobot.py) | RLDS 读取、features、`add_frame/save_episode`、删除输出目录和可选 Hub push |
| [`src/openpi/training/config.py`](../../src/openpi/training/config.py) | `LeRobotLiberoDataConfig` 怎样把五个训练字段 repack 到 LIBERO 协议，并在 model transform 中 resize |
| [`src/openpi/training/data_loader.py`](../../src/openpi/training/data_loader.py) | `LeRobotDatasetMetadata`、fps 到 action 时间偏移、task 到 prompt、完整 transform 顺序 |
| [`src/openpi/policies/libero_policy.py`](../../src/openpi/policies/libero_policy.py) | 两路图像解析、第三路占位、8 维 state、prompt 和 7 维输出裁剪 |
| [`pyproject.toml`](../../pyproject.toml) | Python 3.11、固定 LeRobot commit 和 `rlds` dependency group |
| [第 05 章：数据协议与 Transform](05_data_and_transforms.md) | 从 LeRobot 字段到模型 `Observation/Actions` 的后续路径 |
| [modified LIBERO RLDS](https://huggingface.co/datasets/openvla/modified_libero_rlds) | 原始数据下载入口与数据集说明；本章不自动下载 |

遇到陌生缩写可查后续补齐的[课程术语表](glossary.md)；遇到 TFDS、缓存、磁盘或数据质量问题可查后续补齐的[故障排查手册](troubleshooting.md)。

## 动手实践

### 实践 1：对原始 RLDS 做只读预检

先把下面路径改成你的真实 TFDS/RLDS 根目录：

```bash
export LIBERO_RLDS_DIR=/data/libero_rlds
test -d "$LIBERO_RLDS_DIR"
find "$LIBERO_RLDS_DIR" -maxdepth 3 -type f \
  \( -name dataset_info.json -o -name features.json \) -print | head
df -h "$LIBERO_RLDS_DIR"
```

`test -d` 失败时不要继续。文件系统检查只能说明“看起来像 TFDS 目录”；真正的读取契约仍要由 `tfds.load` 验证。下面只读取第一个 episode/step，不创建 LeRobot 输出：

```bash
uv run --no-sync python - <<'PY'
import os
from pathlib import Path

import tensorflow_datasets as tfds

name = "libero_spatial_no_noops"
root = Path(os.environ["LIBERO_RLDS_DIR"]).expanduser()
assert root.is_dir(), root

raw_dataset = tfds.load(
    name,
    data_dir=str(root),
    split="train",
    shuffle_files=False,
    download=False,
)
episode = next(iter(raw_dataset.take(1)))
step = next(episode["steps"].as_numpy_iterator())

print("dataset:", name)
print("data_dir:", root.resolve())
print("episode keys:", sorted(episode))
print("step keys:", sorted(step))
print("observation keys:", sorted(step["observation"]))
print("image:", step["observation"]["image"].shape, step["observation"]["image"].dtype)
print("wrist_image:", step["observation"]["wrist_image"].shape, step["observation"]["wrist_image"].dtype)
print("state:", step["observation"]["state"].shape, step["observation"]["state"].dtype)
print("action:", step["action"].shape, step["action"].dtype)
print("language raw:", repr(step["language_instruction"]))
print("language decoded:", step["language_instruction"].decode())
PY
```

记录真实输出。预期应能看到两路图像、8 维 state、7 维 action 和可解码的非空语言；如果 shape 或 dtype 不同，不要先改 LeRobot feature 声明来“压过”异常，应先确认下载版本和 TFDS 根目录是否正确。

### 实践 2：建立专用分支并确认删除风险

原转换器包含：

```python
output_path = HF_LEROBOT_HOME / REPO_NAME
if output_path.exists():
    shutil.rmtree(output_path)
```

也就是说，每次运行都会递归删除**同一 repo id 对应的现有本地输出**。在修改源码前创建专用分支：

```bash
git switch -c tutorial/libero-small-conversion
git status --short
```

如果你已经在自己的练习分支上，可以继续使用它，但要单独保存 `git diff -- examples/libero/convert_libero_data_to_lerobot.py`。运行前必须完成以下安全检查：

- 确认 `REPO_NAME` 精确为专用教学 id，而不是已有的个人或团队重要数据集 id；
- 用上一节的 Python 片段打印精确 `output_path`；
- 若该路径已存在且有价值，先停止，备份/重命名它或换一个全新的 repo id；
- 不要为了重跑而删除整个 Hugging Face、LeRobot 或 openpi cache；
- 确认本次命令没有 `--push-to-hub`，避免把练习数据意外发布。

### 实践 3：把转换器改成最多 10 个 episode

编辑 [`examples/libero/convert_libero_data_to_lerobot.py`](../../examples/libero/convert_libero_data_to_lerobot.py)，保留现有 imports，并用下面的常量、`main()` 和入口替换原对应部分：

```python
REPO_NAME = "openpi-tutorial/libero-small"
RAW_DATASET_NAMES = ["libero_spatial_no_noops"]


def main(
    data_dir: str,
    *,
    push_to_hub: bool = False,
    max_episodes_per_dataset: int | None = 10,
):
    if max_episodes_per_dataset is not None and max_episodes_per_dataset < 1:
        raise ValueError("max_episodes_per_dataset must be None or at least 1")

    # Open and minimally validate every source before touching an existing output.
    # download=False makes a wrong/missing local TFDS dataset fail instead of downloading it.
    raw_datasets = {}
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(
            raw_dataset_name,
            data_dir=data_dir,
            split="train",
            download=False,
        )
        first_episode = next(iter(raw_dataset.take(1)), None)
        if first_episode is None:
            raise ValueError(f"Raw dataset is empty: {raw_dataset_name}")
        try:
            first_step = next(first_episode["steps"].as_numpy_iterator())
        except StopIteration as exc:
            raise ValueError(f"First episode has no steps: {raw_dataset_name}") from exc
        _ = first_step["observation"]["image"]
        _ = first_step["observation"]["wrist_image"]
        _ = first_step["observation"]["state"]
        _ = first_step["action"]
        instruction = first_step["language_instruction"]
        if not isinstance(instruction, bytes):
            raise TypeError(
                f"Expected bytes language_instruction in {raw_dataset_name}, got {type(instruction).__name__}"
            )
        if not instruction.decode().strip():
            raise ValueError(f"First language instruction is empty: {raw_dataset_name}")
        raw_datasets[raw_dataset_name] = raw_dataset

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

    for raw_dataset_name, raw_dataset in raw_datasets.items():
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


if __name__ == "__main__":
    tyro.cli(main)
```

`max_episodes_per_dataset=0` 或负数通常是输入错误，所以代码在任何 I/O 前拒绝它。随后用 `download=False` 打开并最小读取每个本地 source，确认首个 episode/step、所需字段和语言可用；数据集名称拼错、目录缺失或 source 为空时，会在删除输出之前失败。这里仍只是首样本预检，后续质量检查不能省略。

只有全部 source 预检通过，代码才处理 `HF_LEROBOT_HOME / REPO_NAME`。`enumerate()` 给当前 raw dataset 的 episode 编号；索引达到 10 时先 `break`，所以默认最多保存索引 0–9。`None` 表示不设上限，但本章不要用它跑完整数据。原来的 Hub push block 和 `rlds` tag 被保留，只是教学命令不传 `--push-to-hub`，因此不会执行。

先核对 diff，不要立刻运行：

```bash
git diff -- examples/libero/convert_libero_data_to_lerobot.py
rg -n 'REPO_NAME|RAW_DATASET_NAMES|max_episodes|rmtree|push_to_hub' \
  examples/libero/convert_libero_data_to_lerobot.py
```

### 实践 4：运行小规模转换

再次确认 `LIBERO_RLDS_DIR` 已改成真实路径，并确认打印出的输出目录没有需要保留的数据。然后从仓库根目录运行：

```bash
export LIBERO_RLDS_DIR=/data/libero_rlds
uv run --no-sync examples/libero/convert_libero_data_to_lerobot.py \
  --data-dir "$LIBERO_RLDS_DIR" \
  --max-episodes-per-dataset 10
```

`--no-sync` 只适用于已经执行过 `uv sync --group rlds` 的环境。命令没有 `--push-to-hub`，所以 `push_to_hub=False`；它只应写本地 LeRobot 数据集。

### 实践 5：读取 metadata，而不是猜 API

openpi 当前源码直接使用 `LeRobotDatasetMetadata.fps` 和 `.tasks`。不同 LeRobot 版本对 episode/frame 总数等公开属性的命名可能变化，因此先打印 `dir(meta)`，再用 `getattr` 检查候选属性：

```bash
uv run --no-sync python - <<'PY'
import inspect
from collections.abc import Mapping
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

repo_id = "openpi-tutorial/libero-small"
root = Path(HF_LEROBOT_HOME) / repo_id
assert root.is_dir(), f"Local LeRobot dataset does not exist: {root}"

signature = inspect.signature(LeRobotDatasetMetadata)
if "root" not in signature.parameters:
    raise RuntimeError(
        f"Pinned LeRobotDatasetMetadata signature has no root parameter: {signature}. "
        "Inspect the pinned API and adapt this local-only check; do not omit root and fall back to the Hub."
    )

meta = LeRobotDatasetMetadata(repo_id, root=root)

print("local path:", root)
print("public attributes:")
print([name for name in dir(meta) if not name.startswith("_")])

for name in ("fps", "total_episodes", "total_frames", "features", "tasks"):
    print(f"{name}:", getattr(meta, name, "<not exposed by this pinned API>"))

assert getattr(meta, "fps") == 10
tasks = getattr(meta, "tasks")
assert tasks and all(isinstance(text, str) and text.strip() for text in tasks.values())

episode_count = getattr(meta, "total_episodes", None)
if episode_count is not None:
    assert 1 <= episode_count <= 10

features = getattr(meta, "features", None)
assert isinstance(features, Mapping), f"metadata.features is missing or not a mapping: {features!r}"

expected = {
    "image": ("image", (256, 256, 3)),
    "wrist_image": ("image", (256, 256, 3)),
    "state": ("float32", (8,)),
    "actions": ("float32", (7,)),
}
present_converter_keys = set(features).intersection(expected)
assert present_converter_keys == set(expected), (
    f"converter feature keys mismatch: expected {sorted(expected)}, "
    f"found {sorted(present_converter_keys)}; all metadata keys={sorted(features)}"
)


def field(spec, name):
    value = spec.get(name) if isinstance(spec, Mapping) else getattr(spec, name, None)
    assert value is not None, f"Feature spec has no {name!r}: {spec!r}"
    return value


for key, (expected_dtype, expected_shape) in expected.items():
    spec = features[key]
    actual_dtype = str(field(spec, "dtype"))
    actual_shape = tuple(field(spec, "shape"))
    assert actual_dtype == expected_dtype, (key, actual_dtype, expected_dtype)
    assert actual_shape == expected_shape, (key, actual_shape, expected_shape)
    print(f"feature[{key!r}]: dtype={actual_dtype}, shape={actual_shape}")
PY
```

本地文档环境没有安装该固定 LeRobot commit，不能靠 import 预先证明构造签名；片段因此先检查 `inspect.signature`，只有确认 `root` 参数存在才构造对象。若断言失败，应根据打印出的固定版本签名调整**显式本地根目录**调用，不能删掉 `root` 参数让库静默查询 Hub。

`total_episodes`、`total_frames` 等计数字段仍用 `getattr` 探测，因为不同 LeRobot 版本可能采用不同公开名称；`fps`、`tasks` 和 `features` 则是本练习的硬契约。features 允许 LeRobot 自己添加 `timestamp`、`frame_index`、`episode_index`、`index`、`task_index` 等系统字段，但转换器负责的四个 key 必须完整存在，并逐一严格匹配 dtype/shape。不要把 `"<not exposed...>"` 当错误结论；它只适用于可选计数字段。

### 实践 6：检查 frame、数值、任务分布和图像方向

下面的检查会遍历这个小数据集，并把第一帧两路图像写到 `/tmp` 供人工查看；它不会修改仓库或上传 Hub：

```bash
uv run --no-sync python - <<'PY'
import inspect
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

repo_id = "openpi-tutorial/libero-small"
root = Path(HF_LEROBOT_HOME) / repo_id
assert root.is_dir(), f"Local LeRobot dataset does not exist: {root}"

for constructor in (LeRobotDatasetMetadata, LeRobotDataset):
    signature = inspect.signature(constructor)
    if "root" not in signature.parameters:
        raise RuntimeError(
            f"Pinned {constructor.__name__} signature has no root parameter: {signature}. "
            "Adapt this local-only check to the pinned API; do not fall back to the Hub."
        )

meta = LeRobotDatasetMetadata(repo_id, root=root)
dataset = LeRobotDataset(repo_id, root=root)


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def image_to_hwc_uint8(value):
    image = as_numpy(value)
    if image.ndim != 3:
        raise AssertionError(f"unexpected image ndim: {image.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    assert image.shape == (256, 256, 3), image.shape

    minimum = float(image.min())
    maximum = float(image.max())
    tolerance = 1e-3
    if np.issubdtype(image.dtype, np.floating):
        assert np.isfinite(image).all()
        if minimum >= -tolerance and maximum <= 1.0 + tolerance:
            image = np.clip(image, 0.0, 1.0) * 255.0
        elif minimum >= -tolerance and maximum <= 255.0 + tolerance:
            image = np.clip(image, 0.0, 255.0)
        else:
            raise AssertionError(
                f"unsupported float image range [{minimum}, {maximum}] for dtype={image.dtype}"
            )
    elif np.issubdtype(image.dtype, np.integer):
        if minimum < 0.0 or maximum > 255.0:
            raise AssertionError(
                f"unsupported integer image range [{minimum}, {maximum}] for dtype={image.dtype}"
            )
    else:
        raise AssertionError(f"unsupported image dtype={image.dtype}")
    return image.astype(np.uint8)


episode_frames = Counter()
task_frames = Counter()
last_frame_index = {}
states = []
actions = []

for row_index in range(len(dataset)):
    row = dataset[row_index]
    if row_index == 0:
        print("row keys:", sorted(row))

    episode_index = int(as_numpy(row["episode_index"]).item())
    frame_index = int(as_numpy(row["frame_index"]).item())
    task_index = int(as_numpy(row["task_index"]).item())

    expected_frame = last_frame_index.get(episode_index, -1) + 1
    assert frame_index == expected_frame, (episode_index, frame_index, expected_frame)
    last_frame_index[episode_index] = frame_index
    episode_frames[episode_index] += 1
    task_frames[task_index] += 1

    state = as_numpy(row["state"])
    action = as_numpy(row["actions"])
    assert state.shape[-1] == 8 and np.isfinite(state).all(), state
    assert action.shape[-1] == 7 and np.isfinite(action).all(), action
    states.append(state)
    actions.append(action)

    for camera in ("image", "wrist_image"):
        preview = image_to_hwc_uint8(row[camera])
        if row_index == 0:
            Image.fromarray(preview).save(f"/tmp/libero-small-{camera}.png")

states = np.stack(states)
actions = np.stack(actions)
tasks = getattr(meta, "tasks")

print("frames:", len(dataset))
print("episode frame counts:", dict(sorted(episode_frames.items())))
print("task frame counts:", {tasks[index]: count for index, count in task_frames.items()})
print("state min/max:", states.min(axis=0), states.max(axis=0))
print("action min/max:", actions.min(axis=0), actions.max(axis=0))
print("action q01/q99:", np.quantile(actions, 0.01, axis=0), np.quantile(actions, 0.99, axis=0))
print("previews: /tmp/libero-small-image.png and /tmp/libero-small-wrist_image.png")

assert 1 <= len(episode_frames) <= 10
assert len(dataset) == sum(episode_frames.values())
assert tasks and all(text.strip() for text in tasks.values())
PY
```

图像转换只接受有限浮点的 `[0,1]` 或 `[0,255]` 范围，以及值位于 `[0,255]` 的整数图像；`1e-3` tolerance 只吸收边界附近的浮点误差。其他范围会连同 min/max 明确失败，不会被无条件 `clip` 成一张看似正常的图。

若当前固定 LeRobot 版本返回的系统索引 key 与片段不同，先根据第一行 `row keys` 调整候选 key；不要伪造 episode 边界。检查结果至少回答：

- episode 数与每个 episode 的 frame 数是否合理，`frame_index` 是否在新 episode 从 0 重新开始；
- `image` 与 `wrist_image` 是否都存在，metadata 存储 shape 是否为 256×256×3，loader 的 CHW/HWC 转换是否符合预期；
- state 最后一维是否为 8，actions 最后一维是否为 7，二者是否全为有限数；
- 各动作维度的 min/max 与 1%/99% 分位是否有尖峰、常量维或明显离群值；
- tasks 是否能由 `task_index` 找回非空文本，task 分布是否符合所选演示；
- 两张 `/tmp` 预览是否可读、方向是否正确、相机是否对调、颜色是否异常；
- episode 长度是否出现 0、1 或异常巨大的值。

数值范围没有脱离机器人与数据版本的通用“正确阈值”。发现离群值时应定位具体 episode/frame，结合图像与动作语义检查，而不是直接裁剪。**加载成功只证明文件和基础 schema 可读，不证明坐标系、动作方向、夹爪约定、时间对齐、相机方向或语言条件正确。**

## 预期结果

本章文档编写环境没有下载原始 LIBERO RLDS，也没有实际运行 TensorFlow 转换，因此以下是**条件式预期**，不是本机实测记录。

如果你已经准备了正确的 `libero_spatial_no_noops` TFDS/RLDS 目录、同步了 `rlds` 依赖，并按默认上限运行，则应看到：

1. `HF_LEROBOT_HOME / "openpi-tutorial/libero-small"` 下出现一个本地 LeRobot 数据集，检查代码先断言该目录存在，再显式传入 `root=root`；
2. metadata 的 fps 为 10，schema 断言确认 features 包含 `image`、`wrist_image`、`state` 和 `actions`；
3. 两路存储图像均声明为 256×256×3，state 为 8 维，actions 为 7 维；
4. episode 数为 1–10；若原始 split 至少有 10 个可读 episode，默认教学运行通常得到 10 个；
5. tasks 非空，frame 中的 task index 能映射回语言指令；
6. state/actions 不含 NaN 或 Inf，样本图像能生成可读预览；
7. 没有任何 Hub 上传，因为命令未启用 `--push-to-hub`。

实际 frame 数、每个 episode 长度、task 种类、动作范围、耗时和磁盘占用依赖你下载的数据版本与机器。请保存自己的命令、metadata 输出、质量检查结果和转换器 diff 作为证据。

## 常见问题

### 1. `DatasetNotFoundError`、builder not found 或 split 不存在

先核对名称必须是 `libero_spatial_no_noops`，再打印 `LIBERO_RLDS_DIR` 的绝对路径和 `dataset_info.json` 位置。`data_dir` 应是 `tfds.load(name, data_dir=...)` 能识别的根，而不是随意的父目录、单个 shard 或错误版本子目录。教学版 `main()` 和实践 1 都使用 `download=False`：缺少本地数据时应立即失败，不会替你联网下载。先让只读预检成功，再运行转换。

### 2. 缺少 `tensorflow_datasets`、TensorFlow 或 `dlimp`

确认当前命令在仓库根目录和 Python 3.11 主环境中执行，然后运行：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --group rlds
uv run --no-sync python -c 'import tensorflow, tensorflow_datasets, dlimp; print("rlds deps ok")'
```

不要切到 LIBERO Python 3.8 仿真环境，也不要把两个环境混成一个。

### 3. 磁盘空间不足

原始 RLDS 和转换后的 LeRobot 图像/视频会同时占空间。先对原始目录、打印出的 `HF_LEROBOT_HOME` 和当前文件系统运行 `du -sh`/`df -h`。清理时只处理你明确拥有且可重建的目标；不要用“删除整个 cache”作为第一步。

### 4. 为什么现有数据突然消失？

仓库原转换器会对 `HF_LEROBOT_HOME / REPO_NAME` 执行 `shutil.rmtree`。本章教学版先打开并最小读取所有 source，再进入这一步，所以名称拼错、目录缺失或空 source 不会先毁掉旧输出；但只要 source 预检通过，同名输出仍会被删除。如果 `REPO_NAME` 复用了有价值的 id，立即停止后续重跑并检查备份；始终使用专用 id、打印精确路径，并在存在时先备份/重命名。

### 5. image writer 报进程、线程、semaphore 或文件句柄错误

脚本默认使用 10 个 image writer threads 和 5 个 processes，受容器、ulimit、共享内存和操作系统进程限制影响。先保存完整异常并确认没有遗留转换进程；在练习分支中把并发数保守地降到例如 2 个 threads、1 个 process，再只重跑小数据。不要把 worker 错误误判为 TFDS 字段错误。

### 6. 转换中断后留下半成品

先打印并检查精确输出路径，确认其中没有需要保留的成功数据或人工修改。原脚本下一次启动会删除同名目标再重建；若你要保留现场用于诊断，先重命名该**单一教程输出目录**。不要清空整个 LeRobot/HF cache，也不要在前一个 writer 仍运行时启动第二次转换。

### 7. `language_instruction.decode()` 失败或 task 为空

回到实践 1 打印 `type`/`repr`。当前脚本假设值是 UTF-8 bytes；如果实际数据版本不同，应先确认编码和字段语义。不要静默忽略非法字节。转换后同时检查 `metadata.tasks`、frame 的 `task_index` 和 task 分布。

### 8. 为什么 Hugging Face 要求登录？

本地转换和本地检查都不需要 Hub 登录。检查片段会先断言 `root` 存在、核对构造签名，再显式传入 `root=root`；如果固定 API 不支持这个参数，片段会明确失败，而不是删掉参数后尝试 Hub。只有显式传入 `--push-to-hub`，程序才应执行上传 block，此时才需要 Hugging Face 身份验证、目标 namespace 权限，并需要重新判断 public/private 设置。若本章命令请求网络或认证，先检查 `root`、构造调用和完整命令。

## 本章验收

进入下一章前，提交或展示以下证据：

- [ ] 在专用 Git 分支中完成转换器练习，并能解释 `REPO_NAME`、单一 raw dataset 和 episode 上限的 diff；
- [ ] `tfds.load(..., download=False)` 只读预检能找到本地原始 split，并显示两路图像、8 维 state、7 维 action 和可解码语言；
- [ ] 打印并人工确认精确 `HF_LEROBOT_HOME / REPO_NAME`，没有覆盖有价值的现有 repo id；
- [ ] 教学版 `main()` 在 `rmtree` 前验证上限并打开/最小读取所有 source，错误 source 不会触发旧输出删除；
- [ ] metadata/data 检查先断言本地 root 存在并验证两个构造器都支持 `root`，随后以 `root=root` 构造，没有 Hub 回退；
- [ ] schema 硬断言通过：两路 `image` feature 都是 `image[256,256,3]`，`state` 是 `float32[8]`，`actions` 是 `float32[7]`；
- [ ] episode 数为 1–10（或与你显式选择的更小数量一致），边界与 frame 数可解释；
- [ ] tasks 非空，`task_index` 能恢复文本，task 分布与源演示一致；
- [ ] state/actions 全为有限数，动作范围/离群值已经记录并人工判断；
- [ ] `image` 与 `wrist_image` 样本都可读取，范围属于 `[0,1]` 浮点、`[0,255]` 浮点或合法整数之一，且 shape、颜色、方向和相机身份正确；
- [ ] 命令没有 `--push-to-hub`，确认没有意外上传；
- [ ] 能说明“数据可加载”与“动作/时间/语言语义正确”的差别。

## 思考题

### 1. 为什么把 RLDS 文件复制进 LeRobot cache 不能替代转换？

<details>
<summary>参考答案（展开查看）</summary>

因为两个格式的组织协议不同。转换器不仅移动数据，还要展开 RLDS episode/step、映射字段、编码图像、解码语言、建立 task index、写 fps 与 feature metadata，并用 `save_episode()` 保留边界。缺少这些关系时，loader 即使碰巧看见文件，也无法按 LeRobot 的时间、任务和 schema 契约读取。

</details>

### 2. `fps=10` 写错为什么会影响训练，而不只是视频播放速度？

<details>
<summary>参考答案（展开查看）</summary>

openpi 的 loader 用 `dataset_meta.fps` 把 action horizon 的离散索引换成秒级 `delta_timestamps`。写错 fps 会让相同的 10 个 action 索引代表错误的时间跨度，从而破坏 observation 与未来 action chunk 的对齐；它不是纯展示字段。

</details>

### 3. 为什么转换器的 features 字典没有显式声明 `task`，训练仍能得到 prompt？

<details>
<summary>参考答案（展开查看）</summary>

当前转换器把 task 字符串随 frame 交给 LeRobot，LeRobot 将任务维护为 metadata 中的 task 映射，并在 frame 中保存 `task_index`。`data_loader.py` 在 `prompt_from_task=True` 时使用 `PromptFromLeRobotTask(dataset_meta.tasks)`，把索引恢复成 `prompt`。task 走的是任务元数据协议，不是普通 8 维/7 维数值 feature。

</details>

### 4. 为什么 metadata、shape 和 finite 检查全部通过，仍不能证明数据适合训练？

<details>
<summary>参考答案（展开查看）</summary>

这些检查只能证明结构与基本数值有效。动作可能使用错误坐标系或符号，夹爪开合可能相反，图像可能旋转/镜像或两路相机对调，语言也可能与演示不匹配。还要结合 episode 边界、动作分布、任务分布和图像/动作的人工抽查验证语义。

</details>

[← 上一章：JAX、NNX 与 π₀ 基础](06_jax_and_pi0_basics.md) · [课程入口：README](README.md) · [下一章：配置教学微调 →](08_configure_finetuning.md)
