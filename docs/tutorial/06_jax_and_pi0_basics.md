# 第 06 章：JAX、NNX 与 π₀ 基础

[← 上一章：数据协议与 Transform](05_data_and_transforms.md) · [课程入口：README](README.md) · [下一章：准备小规模数据集 →](07_prepare_small_dataset.md)

## 本章目标

完成本章后，你应该能够：

- 从本项目的训练路径解释 NumPy 数组与 JAX 数组的差别：设备位置、不可变更新、PyTree、显式随机 key 和 `jax.jit`；
- 说明 `Observation` 为什么是 PyTree，以及一个训练 batch 为什么写成 `(Observation, Actions)`；
- 读懂 `jax.ShapeDtypeStruct` 表达的 shape/dtype 契约，并解释稳定 shape 对 JIT 与 sharding 的意义；
- 用自己的话描述 NNX 的 `Module → split → GraphDef + State → merge`；
- 追踪根 PRNG key 怎样分成初始化、图像增强、噪声和时间 key；
- 用四个概念关系解释 π₀/π₀.₅ 的 flow matching 训练目标与反向采样；
- 说明 LoRA 减少了哪些训练内存，又没有减少哪些内存；
- 面对第一次 JIT 很慢、OOM、NaN 或反复编译时，先收集证据再调整配置。

本章只建立完成后续微调所需的最小知识，不试图替代完整 JAX 或扩散/流模型课程。更深入的实现细节可读[项目中的 JAX 深读](../jax_in_openpi_zh.md)。

## 前置条件

开始前应已完成[第 02 章](02_environment_setup.md)的环境安装，并读完[第 05 章](05_data_and_transforms.md)，知道模型看到的是统一 `Observation` 和定长 action chunk，而不是 LIBERO 原始字典。

本章包含两类练习：

- **静态源码练习**只需要 `rg`，可以直接验证仓库中的符号与调用关系；
- **JAX 运行练习**只有在第 02 章环境已经同步完成后才运行，统一使用 `uv run --no-sync`，避免练习过程临时改依赖或触发解析。

当前文档编写环境没有完成全套 JAX runtime/GPU 验证，因此后文 Python 片段标注的是**预期现象**，不是本机实测输出。不要把文档中的期望值改写成自己的“已验证结果”；应保存本机命令、设备列表和实际输出作为证据。

## 核心概念

### 从 NumPy 到 JAX：先改变执行模型，而不是只换 import

NumPy 代码通常在 CPU 上立即执行，数组也经常原地修改。JAX 的 `jax.Array` 可能位于 CPU、GPU 或 TPU；JAX 还可以先追踪数组运算，再把整段计算编译给设备执行。因此，把 `numpy` 改成 `jax.numpy` 后，需要同时接受以下约定。

#### JAX 数组不可变

JAX 不支持 `x[0] = value` 这种原地写法。函数式更新写成：

```python
y = x.at[0].set(value)
```

这里 `x` 仍表示旧值，`y` 表示新值。编译器在安全时仍可复用底层 buffer，所以“源码不可变”不等于“一定多复制一份设备内存”。在 openpi 训练中，同样的思路体现在 `dataclasses.replace(state, ...)`、`optax.apply_updates(...)` 和返回新的 `TrainState`；不要依赖 Python 全局变量偷偷改变训练状态。

#### 数组有设备位置

`np.ndarray` 是主机内存中的数组；`jax.device_put(...)` 可以显式把数据放到 JAX 设备。`jnp` 运算的结果也是 `jax.Array`，但数据传输和设备计算可能异步发生。计时或确认错误位置时，应对结果调用 `block_until_ready()`，否则测到的可能只是任务提交时间。

项目的数据 loader 产生 batch，JIT 训练步再按指定 sharding 接收它。初学阶段不必手工移动每个叶子，但必须知道“Python 已经返回”不一定代表 GPU 已完成。

#### PyTree 是 JAX 眼中的嵌套数据

PyTree 不是一种新的数组类型，而是一套“容器结构 + 数组叶子”的约定。例如下面都是 PyTree：

```text
dict[str, array]
tuple[Observation, Actions]
TrainState(step, params, opt_state, ...)
```

JAX 可以保持容器结构不变，让不同 transform 处理它们支持的叶子。例如 `tree.map` 可以遍历所有叶子，设备放置、shape 推导和 sharding 主要作用于数组叶子；`grad` 则不是“对 bool、整数或静态字段都求导”。在本项目中，`DiffState` 与 trainable filter 先选出参数子集，梯度针对其中可微的浮点或复数数组值。树的**结构**也属于编译输入的一部分：不仅数组 shape/dtype 变化可能触发新编译，字典 key、可选字段是否为 `None` 等结构变化也可能改变 trace。

### `Observation` 与训练 batch 都是 PyTree

[`Observation`](../../src/openpi/models/model.py) 用 `flax.struct.dataclass` 定义。它既是字段清楚的不可变记录，也是已经注册给 JAX 的 PyTree 容器。它的叶子包括：

- `images` 字典中的各路图像数组；
- `image_masks` 字典中的 mask；
- `state`；
- 可选的 prompt token、token mask；
- π₀-FAST 专用的 autoregressive mask 与 loss mask。

π₀/π₀.₅ 的训练 batch 不是“把 action 塞进 Observation”，而是：

```text
batch = (observation, actions)

observation: Observation
actions:     float32[batch, action_horizon, action_dim]
```

这个 tuple 本身也是 PyTree。`scripts/train.py` 中的 `train_step()` 先执行 `observation, actions = batch`，再把两者分别交给 `model.compute_loss(...)`。这与第 05 章的数据协议一致：observation 是条件，actions 是监督目标。

`flax.struct.dataclass` 的实例采用函数式更新，但字段里的嵌套 Python 容器仍应当作结构数据使用；训练中不要一边 trace 一边增删图像 key。

### `ShapeDtypeStruct`：只描述契约，不分配真实 batch

`jax.ShapeDtypeStruct(shape, dtype)` 是一个抽象叶子：它说“这里将有一个这种 shape/dtype 的数组”，但不会真的分配对应设备 buffer。模型配置的 `inputs_spec()` 用它描述三路图像、mask、state、token 和 actions；训练初始化又通过 `jax.eval_shape(...)` 得到 `TrainState` 的抽象结构，再据此计算 sharding。

以 π₀/π₀.₅ 默认协议为例，关键规格可概括为：

```text
image:   float32[B, 224, 224, 3]
mask:    bool[B]
state:   float32[B, action_dim]
tokens:  int32[B, max_token_len]
actions: float32[B, action_horizon, action_dim]
```

稳定 shape/dtype 有三层价值：

1. 数据和模型能在真正分配大数组前核对接口；
2. JIT 可以复用已经编译的 executable；
3. sharding 可以提前判断哪些维度能按 mesh 切分。

因此，训练中通常把图像、token 和 action chunk 补成固定长度。若最后一个 batch 突然变小、token 长度变化、`float32` 变成 `float64`，或某次缺了一路相机，不只可能报 shape 错，还可能产生新的编译。

> **立即观察 1：PyTree 与 shape 契约。**下面两个搜索只输出定义位置和规格叶子，不需要导入 JAX：
>
> ```bash
> rg -n '@struct\.dataclass|class Observation|^Actions =' src/openpi/models/model.py
> rg -n 'ShapeDtypeStruct' src/openpi/models/pi0_config.py
> ```
>
> 暂停一分钟：画出 `(Observation, Actions)` 的两层树，并圈出 bool/int 叶子。说明为什么“它是 PyTree”不等于“每个叶子都参与梯度”。

### 显式 PRNG key：随机性也是数据流

JAX 没有供模型代码随意读取的隐式全局随机状态。随机函数需要显式 key；一个 key 不会被“用掉并自动前进”。如果对同一个 key 重复调用同一个随机函数，会得到相同结果。因此正确模式是从根 key 派生用途明确的新 key。

openpi 训练的主线是：

```text
root = jax.random.key(config.seed)
  ├─ train_rng
  │    └─ fold_in(step) → 当前 step 的 train_rng
  │         ├─ preprocess_rng → 训练图像增强
  │         ├─ noise_rng      → 与 actions 同 shape 的高斯噪声
  │         └─ time_rng       → 每个样本的 flow time
  └─ init_rng
       └─ model_rng → 参数初始化
```

这里有两种常见派生方式：

- `jax.random.split(key, n)` 生成多个并列用途的子 key；
- `jax.random.fold_in(key, step)` 把 step 等整数稳定混入 key，适合从固定训练根 key 得到“第 N 步的随机性”。

在 [`scripts/train.py`](../../scripts/train.py) 中，根 key 先拆为 `train_rng` 和 `init_rng`，训练步再将 `state.step` fold in；在 [`pi0.py`](../../src/openpi/models/pi0.py) 的 `compute_loss()` 中，当前步 key 又拆成 augmentation、noise、time 三个 key。实际 flow time 不是均匀采样，而是 `Beta(1.5, 1) * 0.999 + 0.001`。这样随机性可复现、可并行，也不会因为某个函数内部多抽一次随机数而悄悄改变所有后续结果。

> **立即观察 2：从 seed 追到真实 time 分布。**
>
> ```bash
> rg -n 'random\.key|random\.split|random\.fold_in' scripts/train.py
> rg -n -A 8 'preprocess_rng, noise_rng, time_rng' src/openpi/models/pi0.py
> ```
>
> 第二个搜索应同时显示三路 key、Gaussian noise 和缩放后的 Beta time。后文小脚本中的 `uniform` 只演示“不同 key 对应不同随机用途”，不是 π₀ 的真实 time 采样公式。

### Flax NNX：把可调用模型拆成结构与状态

π₀ 模型继承 `nnx.Module`。对初学者，可以把一个 NNX module 看成两部分：

```text
model (nnx.Module)
  ├─ GraphDef：模块拓扑、子模块关系和重建所需的静态定义
  └─ State：参数等变量节点当前保存的值
```

对应的基本操作是：

```python
graphdef, state = nnx.split(model)
model = nnx.merge(graphdef, state)
```

`split` 得到可进入 JAX transform 的、以数组值为叶子的 State/PyTree；`merge` 在需要调用 `model.compute_loss()` 时重建可调用的 module graph。它不是序列化格式转换，也不是复制出两个独立模型。

[`training/utils.py`](../../src/openpi/training/utils.py) 的 `TrainState` 保存：

- `params: nnx.State`：当前模型状态；
- `model_def: nnx.GraphDef`：重建模型的图定义；
- `opt_state`：Optax 优化器状态；
- `step`；
- 可选 `ema_params`；
- 作为静态字段保存的优化器变换 `tx` 与 EMA 系数。

<details>
<summary>进阶：为什么还有 <code>module_jit</code>？</summary>

[`nnx_utils.module_jit`](../../src/openpi/shared/nnx_utils.py) 先在 JIT 外 `split`，编译函数内部再 `merge`，从而避免每次调用都遍历可变 module graph。它冻结创建 wrapper 时捕获的 state；module 内部 mutation 不会被带回 wrapper 外，因此不能把它误当训练状态更新器。初读训练主线时只需记住标准 `split/merge` 即可。

</details>

### filter、`PathRegex` 与 LoRA 的可训练子集

NNX filter 用变量类型和路径规则选择 State 的一部分。训练配置中的核心关系是：

```text
trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))
```

也就是：先限定为 `nnx.Param`，再排除 freeze filter 命中的参数。课程后续使用的低内存示例是精确配置名 `pi0_libero_low_mem_finetune`：它选择 LoRA model variants，用对应 `get_freeze_filter()`，并关闭 EMA。

LoRA variant 在基础权重旁新增低秩参数；`get_freeze_filter()` 的语义是冻结所选基础 LLM 路径，同时不把 LoRA 路径纳入冻结集合。`trainable_filter` 在 `nnx.Param` 范围内取它的补集，所以静态源码可以证明**规则怎样组合**，却不能列出本次运行中所有实际叶子路径。具体路径依赖已经实例化的 NNX module graph，必须创建模型后检查其 `nnx.State`；本章不要求为此加载大模型。

`scripts/train.py` 还做了三件与内存直接相关的事：

1. 冻结参数转成 `bfloat16`；
2. `tx.init(...)` 只为 `params.filter(trainable_filter)` 建立优化器状态；
3. `nnx.DiffState(..., trainable_filter)` 只对可训练子集求梯度。

这会减少可训练子集的梯度和优化器状态；冻结权重、forward 激活和 batch 内存不会因此消失。完整内存边界见本节后面的资源表。

<details>
<summary>进阶：<code>PathRegex</code> 怎样参与选择？</summary>

[`PathRegex`](../../src/openpi/shared/nnx_utils.py) 把变量路径用 `/` 连接，再用正则表达式做 `fullmatch`。π₀ 配置用它描述 `llm`、action expert 和 `lora` 等路径规则。这里读到的是 filter 的布尔语义；只有对实例化后的 State 遍历路径，才能得到某个具体模型的实际命中列表。

</details>

> **立即观察 3：只验证 filter 语义，不猜实际路径。**
>
> ```bash
> rg -n -B 2 -A 20 'name="pi0_libero_low_mem_finetune"' src/openpi/training/config.py
> rg -n -A 38 '^    def get_freeze_filter' src/openpi/models/pi0_config.py
> rg -n -A 4 '^    def trainable_filter' src/openpi/training/config.py
> rg -n 'nnx\.(split|merge)|nnx\.graphdef|nnx\.state' \
>   scripts/train.py src/openpi/training/utils.py src/openpi/shared/nnx_utils.py
> ```
>
> 写下三条已由源码支持的结论：配置使用什么 model variants、freeze filter 怎样取得、trainable filter 怎样由补集形成。不要从这些搜索抄出一份“实际可训练路径清单”；那需要先实例化模型并遍历 NNX State。

### π₀/π₀.₅ flow matching 的最小直觉

π₀ 与 π₀.₅ 在网络如何编码 state、如何注入时间等方面不同，但当前 JAX 实现共享同一条连续 action flow 训练/采样路径。下面四行是**概念化伪代码**；前三个中间量和平方误差对应 `pi0.py`，`model(...)` 则省略了真实的图像/语言 prefix、action suffix、attention 和 projection：

```text
x_t  = time * noise + (1 - time) * actions
u_t  = noise - actions
v_t  = model(observation, x_t, time)
loss = mean(square(v_t - u_t))
```

变量含义如下：

- `actions`：归一化后的真实 action chunk，shape 为 `(B,H,D)`；
- `noise`：与 actions 同 shape 的高斯噪声；
- `time`：每个样本一个按 `Beta(1.5,1) * 0.999 + 0.001` 采样的时间值，再广播到 action 维；
- `x_t`：真实动作与噪声之间的插值；`time=0` 靠近动作，`time=1` 靠近噪声；
- `u_t`：这条直线路径随时间增加时的真实速度，即 `noise - actions`；
- `v_t`：模型在 observation 条件下，对当前位置和时间预测的速度。

为什么预测速度？训练时可以从任意 `time` 构造一个带监督的 `x_t` 和目标速度，教模型学到整个向量场；推理时就能从一个新噪声出发，沿模型给出的速度场逐步走回动作分布。

本仓库 `sample_actions()` 采用 `t=1` 是噪声、`t=0` 是动作的约定。它先采样 noise，再令 `dt = -1 / num_steps`，在 `jax.lax.while_loop` 中重复：

```text
v_t  = model(observation, x_t, time)
x_t  = x_t + dt * v_t
time = time + dt
```

训练目标 `u_t` 描述“时间从动作走向噪声”的正方向；采样时 `dt` 为负，因此沿反方向从噪声积分到动作。`jax.lax.while_loop` 让循环留在 JAX 计算图内并能随外层调用一起编译，而不是让 Python 每步驱动设备。

这里不需要继续推导微分方程。进入后续训练前，只要能说清：**训练随机挑路径上的点并预测速度；推理从噪声开始，用负时间步积分到 action chunk。**

### π₀-FAST 是另一条训练与生成路径

不要把 π₀-FAST 的名字理解成“同一 flow sampler 的加速开关”。[`pi0_fast.py`](../../src/openpi/models/pi0_fast.py) 把动作表示为 token，训练目标是 next-token autoregressive cross-entropy，生成时逐 token 解码，再由 model output transform 把 token ID 还原为连续 action。其生成循环在**当前迭代中所有 batch 元素都生成 EOS**，或达到最大解码步数时停止。

因此：

```text
π₀ / π₀.₅：连续 action + flow matching + 从噪声积分
π₀-FAST：动作 token + 自回归 next-token 训练/生成 + token 解码
```

前述 noise/time target 和从 `t=1` 积分到 `t=0` 的循环只描述 π₀/π₀.₅ flow 路径。π₀-FAST 虽也用 `jax.lax.while_loop` 编译生成循环，但循环体是在预测下一个 token，不是在积分速度场。

> **暂停练习 4：区分两个 target、两个循环。**
>
> ```bash
> rg -n 'x_t =|u_t =|v_t =|return jnp\.mean\(jnp\.square' src/openpi/models/pi0.py
> rg -n -B 6 -A 4 'x_0, _ = jax\.lax\.while_loop' src/openpi/models/pi0.py
> rg -n -A 36 '^    def compute_loss' src/openpi/models/pi0_fast.py
> ```
>
> 不看正文写两句：π₀/π₀.₅ 的监督 target 是什么；π₀-FAST 的监督 target 是什么。再说明两个 `while_loop` 分别迭代连续 `x_t` 还是离散 token。

### 一个训练步实际发生什么

[`train_step`](../../scripts/train.py) 可以压缩成以下顺序：

```text
GraphDef + full State
  → nnx.merge 得到 model，并切到 train 模式
  → 从 batch 拆出 Observation 与 Actions
  → fold_in(step) 得到本步随机 key
  → 对 trainable_filter 选中的 State 计算 loss 与梯度
  → Optax 根据梯度和 opt_state 生成 updates
  → apply_updates 更新可训练参数
  → nnx.update 写回 model，再抽取完整 State
  → step + 1，并可选更新 EMA
  → 返回新的 TrainState 与 loss/grad_norm/param_norm
```

外层 `jax.jit` 把这个步骤编译，并通过 `in_shardings`/`out_shardings` 约束 train state 与 batch 放置。初学者可先把 sharding 理解成“同一 PyTree 的不同数组叶子怎样分布到多张设备”，无需先掌握每个 `PartitionSpec` 细节。

### `jax.jit`：第一次调用包含追踪和编译

`jax.jit(function)` 返回一个编译调用入口。第一次遇到某种输入 PyTree 结构、shape、dtype 和相关静态参数时，JAX 需要追踪 Python 运算、生成中间表示并由 XLA 编译；之后兼容输入才能复用结果。

在本项目中，训练状态初始化被 JIT，`train_step` 也通过 `jax.jit(...)` 编译，并显式声明 input/output sharding。于是常见现象是：

```text
数据 loader 已就绪
→ 抽象初始化与 sharding 规划
→ 第一次大模型编译较慢
→ 稳定 shape 的后续 step 明显更规律
```

第一次慢不能只凭“终端暂时不打印”判断为挂死。更有用的证据包括：进程是否还在、CPU 是否忙于编译、GPU/内存是否变化、是否出现编译日志、第二个同 shape step 是否复用 executable。反过来，如果 shape/dtype 不断变化，可能每隔一段时间又编译一次。

调试时要记住两个边界：

- JAX 默认可能异步执行；精确定位或计时时用 `.block_until_ready()`；
- Python `print` 只在 trace 时执行或看见 tracer，已编译计算内部需要时可用 `jax.debug.print`，但它会增加开销。

### JIT、显存和 sharding 的边界

判断训练资源时至少分开四类：

| 类别 | 主要受什么影响 | LoRA/冻结是否消除 |
| --- | --- | --- |
| 模型参数 | 模型规模、dtype、复制或 FSDP 分片 | 不消除；冻结权重仍存在，可能降精度/分片 |
| 梯度与优化器状态 | 可训练参数数量、优化器 | 显著减少，因为只覆盖 trainable subset |
| EMA 参数 | `ema_decay` 是否启用、参数树规模 | 不自动消除；低内存配置可显式关闭 |
| batch 与激活 | batch size、图像/token/action shape、网络中间结果 | 通常仍存在，LoRA 不等于无激活 |

`fsdp_devices` 不是“填得越大越省显存”的通用旋钮。它大于 1 时需要总设备数可以整除该值，可能降低部分每设备参数/状态内存，也可能增加通信；全局 `batch_size` 仍必须能被总设备数整除，而增大 batch 会增加激活内存。没有 GPU 数量、模型 variant、batch shape 和 sharding 日志时，不能承诺精确显存数字。

<details>
<summary>进阶：当前 FSDP 实现具体选择哪些数组？</summary>

- `fsdp_devices=1` 时 FSDP 轴大小为 1，数组按复制处理；
- mesh 的另一轴是 batch/data-parallel 轴，训练数据按 mesh 规则分布；
- `fsdp_sharding()` 只尝试切分达到大小阈值、至少二维且存在可整除维度的数组；
- 标量、向量、小数组或找不到可整除轴的数组仍复制。

这些规则解释了为什么“启用 FSDP”不等于所有参数和优化器叶子都平均切开。

</details>

> **暂停练习 5：把编译证据与内存证据分开。**
>
> ```bash
> rg -n -A 12 'ptrain_step = jax\.jit' scripts/train.py
> rg -n -A 16 '^def make_mesh' src/openpi/training/sharding.py
> rg -n 'batch_size % jax\.device_count|fsdp_devices' \
>   scripts/train.py src/openpi/training/config.py
> ```
>
> 写两列笔记：左列只记录 first JIT、shape/dtype 与编译日志；右列只记录参数、梯度/优化器、EMA、batch/激活和 sharding。不要用“卡住了”或“OOM”替代这些证据。

## 源码地图

| 文件 | 本章阅读重点 |
| --- | --- |
| [`src/openpi/models/model.py`](../../src/openpi/models/model.py) | `Observation`、`Actions`、`BaseModel` 公共接口与 PyTree 数据协议 |
| [`src/openpi/models/pi0_config.py`](../../src/openpi/models/pi0_config.py) | `inputs_spec()` 的 `ShapeDtypeStruct` 和 LoRA freeze filter |
| [`src/openpi/models/pi0.py`](../../src/openpi/models/pi0.py) | π₀/π₀.₅ 的随机 key 拆分、flow loss 与 `while_loop` 采样 |
| [`scripts/train.py`](../../scripts/train.py) | 根 key、NNX split/merge、trainable filter、训练步 JIT 与 EMA |
| [`src/openpi/training/utils.py`](../../src/openpi/training/utils.py) | `TrainState` 中的 State、GraphDef、Optax state 与静态字段 |
| [`src/openpi/shared/nnx_utils.py`](../../src/openpi/shared/nnx_utils.py) | `module_jit`、`PathRegex` 和按 filter 映射 State |
| [`src/openpi/training/sharding.py`](../../src/openpi/training/sharding.py) | device mesh、data/FSDP 轴和大数组分片条件 |
| [`src/openpi/models/pi0_fast.py`](../../src/openpi/models/pi0_fast.py) | π₀-FAST 的 next-token loss 与自回归生成，作为对照分支 |
| [JAX in openpi 深读](../jax_in_openpi_zh.md) | 本章之后继续阅读 JAX transform、编译和工程实现 |

遇到陌生缩写可查后续补齐的[课程术语表](glossary.md)；遇到环境、编译、显存或训练数值问题可查后续补齐的[故障排查手册](troubleshooting.md)。

## 动手实践

### 实践 1：确认设备与不可变更新

仅在第 02 章环境已经完成后，从仓库根目录运行：

```bash
uv run --no-sync python - <<'PY'
import numpy as np
import jax
import jax.numpy as jnp

host = np.arange(4, dtype=np.float32)
x = jax.device_put(host)
y = x.at[0].set(99.0)

print("devices:", jax.devices())
print("x devices:", x.devices())
print("x:", x)
print("y:", y)
print("x unchanged:", bool(jnp.array_equal(x, jnp.arange(4, dtype=jnp.float32))))
PY
```

**预期现象（当前文档环境未运行）：**输出至少列出一个 JAX device；`x` 仍为 `[0,1,2,3]`，`y` 的第一个元素为 `99`。若只看到 CPU，这个小练习仍可运行，但不能据此证明后续大模型会使用 GPU。

### 实践 2：观察 PyTree、显式随机 key 与抽象 shape

```bash
uv run --no-sync python - <<'PY'
import jax
import jax.numpy as jnp

batch = (
    {"state": jnp.zeros((2, 32), dtype=jnp.float32)},
    jnp.zeros((2, 10, 32), dtype=jnp.float32),
)
leaves, tree_def = jax.tree_util.tree_flatten(batch)
print("tree:", tree_def)
print("leaves:", [(x.shape, x.dtype) for x in leaves])

root = jax.random.key(0)
augment_key, noise_key, time_key = jax.random.split(root, 3)
print("noise:", jax.random.normal(noise_key, (2,)))
print("uniform key-use demo:", jax.random.uniform(time_key, (2,)))
print("same key repeats:", jax.random.normal(noise_key, (2)))

action_spec = jax.ShapeDtypeStruct((2, 10, 32), jnp.float32)
loss_spec = jax.eval_shape(lambda a: jnp.mean(jnp.square(a), axis=-1), action_spec)
print("abstract action:", action_spec)
print("abstract loss:", loss_spec)
PY
```

**预期现象（当前文档环境未运行）：**tuple/dict 结构被保留，叶子是两个数组；对同一个 `noise_key` 重复调用得到相同数值；`eval_shape` 只返回 loss 的抽象 `(2,10)` shape/dtype，不计算真实 batch。`augment_key` 未实际做图像增强，`uniform` 也只演示 `time_key` 是独立用途；π₀ 的真实 time 分布必须以 [`compute_loss()`](../../src/openpi/models/pi0.py) 中缩放后的 Beta 采样为准。

### 实践 3：观察第一次 JIT 与同步边界

```bash
uv run --no-sync python - <<'PY'
import time
import jax
import jax.numpy as jnp

@jax.jit
def step(x):
    return jnp.tanh(x @ x.T).sum()

x = jnp.ones((512, 512), dtype=jnp.float32)
for label in ("first", "second"):
    start = time.perf_counter()
    value = step(x).block_until_ready()
    elapsed = time.perf_counter() - start
    print(label, "seconds=", round(elapsed, 4), "value=", float(value))
PY
```

**预期现象（当前文档环境未运行）：**第一次包含 trace/compile，通常比第二次慢，但具体差距取决于设备、缓存和系统负载，不把“第一次一定慢 N 倍”当验收条件。`.block_until_ready()` 保证两次都测到计算完成，而不只是异步提交。

若要收集编译证据，可以临时为同一练习增加 `JAX_LOG_COMPILES=1` 环境变量；日志较多，保存相关的函数名、shape/dtype 和时间即可，不要把完整编译日志粘进学习笔记。

### 实践 4：用有限上下文定位训练主线

以下搜索按符号锚定，并限制上下文，避免把整个大文件输出到终端：

```bash
rg -n -A 48 '^def train_step' scripts/train.py
rg -n -A 34 '^    def compute_loss' src/openpi/models/pi0.py
rg -n -A 68 '^    def sample_actions' src/openpi/models/pi0.py
rg -n 'nnx\.(split|merge)|nnx\.graphdef|nnx\.state' \
  scripts/train.py src/openpi/training/utils.py src/openpi/shared/nnx_utils.py
```

请从输出中标注：

1. batch 在哪里拆为 `observation, actions`；
2. `fold_in` 与三路 `split` 分别在哪里；
3. `x_t`、`u_t`、`v_t`、平方误差在哪里；
4. 采样的负 `dt`、`step`、`cond` 和 `jax.lax.while_loop` 在哪里；
5. model 在哪里 `merge`，完整 State 在哪里重新抽取。

### 实践 5：核对可训练 filter 与 FAST 分支

```bash
rg -n -B 2 -A 20 'name="pi0_libero_low_mem_finetune"' src/openpi/training/config.py
rg -n -A 38 '^    def get_freeze_filter' src/openpi/models/pi0_config.py
rg -n -A 4 '^    def trainable_filter' src/openpi/training/config.py
rg -n 'PathRegex|DiffState|params\.filter|tx\.init' \
  scripts/train.py src/openpi/shared/nnx_utils.py
rg -n -A 36 '^    def compute_loss' src/openpi/models/pi0_fast.py
rg -n -A 20 '^        def step' src/openpi/models/pi0_fast.py
rg -n -A 18 '# Decode one step' src/openpi/models/pi0_fast.py
rg -n -B 8 -A 4 'jax\.lax\.while_loop' src/openpi/models/pi0_fast.py
```

先为 `pi0_libero_low_mem_finetune` 写三条**规则语义**：model variants、freeze filter 来源、`Param AND NOT freeze_filter`。这些静态搜索没有实例化 NNX State，因此不要声称它们列出了实际可训练参数路径。

再用最后三个搜索记录 FAST 的生成证据：`step` 从 `last_logit` 采样或取 argmax 并写入 `output_tokens`，采样出的 token 又被嵌入以计算下一步 `last_logit`；循环在当前迭代中所有 batch 元素都生成 EOS，或达到最大步数时停止。最后写一句 FAST 对照：它的 loss 是 token cross-entropy，`sample_actions` 是自回归 token generation，不使用 π₀/π₀.₅ 的 flow noise/time target。

## 预期结果

完成练习后，你的实验记录应有五项，而不是重复抄写整章：

1. device/immutable 与 PyTree/shape 小脚本的实际输出；
2. 同 shape 的 first/second JIT 同步计时；
3. 一张 `root key → step key → preprocess/noise/Beta time` 追踪图；
4. 一张 `GraphDef/State → loss/grad → Optax → full State/EMA` 训练步图；
5. 一页对照：`pi0_libero_low_mem_finetune` 的 filter **语义**、π₀/π₀.₅ flow target、π₀-FAST token target，以及参数/优化器/EMA/激活四类内存证据。

### 本文的验证边界

本章已静态核对 `Observation`、π₀/π₀.₅ loss 与 sampling、训练步、NNX helpers、LoRA filter 和 sharding 源码。文中的 `rg` 锚点来自当前仓库。JAX Python 片段、GPU device placement、首次编译耗时、显存占用和多设备 FSDP 没有在当前文档环境实际运行；这些结果必须由学习者在已完成第 02 章的机器上记录，不能由本文替代。

## 常见问题

排错时按“症状 → 最早可观察证据 → 对照源码契约 → 只改一个变量”的顺序。不要同时改 batch、dtype、FSDP 和模型 variant，否则即使恢复运行也无法知道原因。

### 1. `import jax` 或 `import openpi` 失败怎么办？

先确认位于仓库根目录、已完成第 02 章依赖同步，并使用：

```bash
uv run --no-sync python -c 'import jax; print(jax.__version__, jax.devices())'
```

记录完整异常、Python 路径和命令。不要为了绕过一次 import error，直接在环境外执行 `pip install` 或临时替换 lock 中的 JAX 包；那会让后续结果无法与课程环境比较。

### 2. 为什么只看到 CPU？

`jax.devices()` 是第一条证据。若只有 CPU，小型数组练习仍然有效；后续训练则要继续核对第 02 章要求的驱动、容器/宿主角色、JAX accelerator build 和设备可见性。不要仅凭 `nvidia-smi` 能看到显卡就断言当前 `uv run` 进程一定使用 GPU，也不要把 CPU 上的慢误判为 JIT 挂死。

### 3. 第一个训练 step 很久没有输出，是在编译还是挂了？

先记录最后一条日志，区分 data loader、train state 初始化和第一次 `ptrain_step`。必要时开启 `JAX_LOG_COMPILES=1` 重现一个最小步骤，并观察进程 CPU、GPU/内存、异常和后续同 shape step。编译通常会留下高 CPU、编译日志或完成后的后续 step；真正的阻塞可能停在数据读取、外部资源、设备初始化或通信。只有“等了很久”不是足够证据。

### 4. OOM 时 LoRA 为什么没有解决全部问题？

回到核心概念的四类内存表，先记录 OOM 出现在初始化、forward/backward 还是 optimizer update。forward OOM 要重点看冻结权重、batch/激活和输入 shape；update 附近还要看优化器与 EMA。降低 batch size 常能减少激活，但仍须满足设备整除约束；`fsdp_devices` 只有在多设备和合法 mesh 下才值得评估。

### 5. loss 变成 NaN 应该先改什么？

先保存 NaN 首次出现的 step、loss/grad norm、batch shape/dtype，以及 observation/actions 是否含 NaN/Inf；再核对 normalization statistics、学习率和动作尺度。`JAX_DEBUG_NANS=1` 可用于更小的复现实验，但可能显著减慢或改变执行方式。先找最早产生非有限值的位置，不要一上来清缓存、重装 JAX 或同时降低所有超参数。

### 6. 为什么训练中反复出现编译？

对比每次编译对应的 PyTree 结构、shape、dtype 和静态参数。常见来源包括最后一个 batch 大小变化、token/action 长度未固定、图像 key 变化、`float32/float64` 混用或把 Python 标量当静态分支。先让输入契约稳定，再判断缓存是否复用；删除编译缓存不是修复动态 shape 的方法。

### 7. `fsdp_devices=2` 是否表示使用两张 GPU？

不是。它表示 mesh 的 FSDP 轴大小为 2；总 device count 还必须能被 2 整除。用 `make_mesh()` 和启动时的 sharding 日志确认布局，具体数组选择规则见前面的折叠进阶说明。

更多跨层诊断路径见后续补齐的[故障排查手册](troubleshooting.md)。

## 本章验收

进入下一章前，提交或展示以下证据：

- [ ] 用实际数组解释 device placement、不可变更新，以及 `(Observation, Actions)` PyTree/`ShapeDtypeStruct` 契约；
- [ ] 从 root key 追到 `split/fold_in` 和 augmentation/noise/Beta time，说明不存在隐式全局随机状态；
- [ ] 用 `GraphDef + State` 解释 `nnx.split/merge`，并只陈述 filter 语义，不把静态搜索冒充实际 State 路径；
- [ ] 不看答案写出 `x_t`、`u_t`、`v_t` 和平方误差，并解释负 `dt` 的采样方向；
- [ ] 区分 π₀/π₀.₅ flow target 与 π₀-FAST autoregressive token target；
- [ ] 解释 first JIT 为什么慢，以及怎样用同步、编译日志和稳定输入辨别编译与阻塞；
- [ ] 说明 LoRA 减少 trainable/optimizer memory，但冻结权重与 batch/activation memory 仍然存在。

## 思考题

### 1. 为什么同一个 `noise_key` 连续调用两次 `jax.random.normal` 会得到相同结果？

<details>
<summary>参考答案（展开查看）</summary>

JAX key 是显式输入，不是会自动前进的全局随机状态。同一个函数、key、shape 和 dtype 对应同一结果。需要新随机样本时，应从父 key `split` 新 key，或把 step 等标识 `fold_in`，并为 augmentation、noise、time 等用途保留独立 key。

</details>

### 2. `u_t = noise - actions` 指向噪声，为什么采样却能得到动作？

<details>
<summary>参考答案（展开查看）</summary>

`u_t` 是时间从 0 增加到 1 时，也就是从 action 走向 noise 的速度。采样从 `time=1` 的 noise 开始，使用负步长 `dt`，所以 `x_t + dt * v_t` 沿速度场的反方向积分，最终接近 `time=0` 的 action。

</details>

### 3. 为什么静态 `rg` 能解释 filter 语义，却不能列出实际可训练路径？

<details>
<summary>参考答案（展开查看）</summary>

静态搜索能看到 `freeze_filter`、`trainable_filter = Param AND NOT freeze_filter` 以及配置选择了哪些 model variants，因此可以解释规则如何组合。但具体叶子路径由实际实例化的 NNX module graph 决定；只有创建模型并遍历它的 `nnx.State`，才能列出本次实例的命中结果。

</details>

### 4. 为什么训练 batch 是 `(Observation, Actions)`，而不是把 actions 塞进 `Observation`？

<details>
<summary>参考答案（展开查看）</summary>

`Observation` 表示模型生成动作时可见的条件：图像、mask、state 和 prompt；`Actions` 是训练监督目标。把二者组成 tuple 后，整个 batch 仍是 PyTree，`train_step()` 可以明确拆开条件与标签，再交给 `compute_loss()`。推理时没有真实 actions，也就不会把训练标签误当 observation 字段。

</details>

[← 上一章：数据协议与 Transform](05_data_and_transforms.md) · [课程入口：README](README.md) · [下一章：准备小规模数据集 →](07_prepare_small_dataset.md)
