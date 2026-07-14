# 第 04 章：openpi 架构与一次请求的数据流

[← 上一章：第一次 LIBERO 预训练推理](03_first_libero_inference.md) · [课程入口：README](README.md) · [下一章：数据协议与 Transform →](05_data_and_transforms.md)

## 本章目标

完成本章后，你应该能够：

- 从 `WebsocketClientPolicy.infer()` 开始，按真实调用顺序追踪一次 observation 如何变成 action chunk；
- 区分网络传输、输入预处理、模型采样、输出后处理和配置组装分别由哪个组件负责；
- 说明 Policy 在服务启动阶段加载，WebSocket 请求处理循环会复用它，而不是每来一个请求就重新加载模型；
- 理解为什么 `openpi-client` 与 JAX/Torch 模型环境分开发布，以及这种边界如何支持两个运行时；
- 使用基于符号名的 `rg` 命令自行核对调用链，而不依赖容易变化的行号。

本章沿用第 03 章的 `pi05_libero` 预训练示例。后续课程会创建 π₀ 的 `pi0_libero_tutorial`；模型配置、checkpoint 和 normalization assets 仍然必须配套，不能因为两者都服务 LIBERO 就混用。

## 前置条件

开始前应已完成[第 03 章](03_first_libero_inference.md)，至少能解释：

- LIBERO `runtime` 构造 observation，并通过 WebSocket 请求策略；
- `openpi_server` 加载 `pi05_libero` 与对应 checkpoint；
- 一次推理返回的是 action chunk，客户端只执行其中若干步后重新规划；
- 课程中的“模型/训练端”和“LIBERO 仿真端”是逻辑角色，不保证永远对应宿主机、某个固定容器或两台机器。

本章只做源码导航，不要求下载 checkpoint、启动 GPU 推理或修改模型代码。所有命令默认从仓库根目录运行。

## 核心概念

### 先分清启动期与请求期

[`scripts/serve_policy.py`](../../scripts/serve_policy.py) 的 `main()` 先调用 `create_policy(args)`。默认 LIBERO 路径会取得命名配置 `pi05_libero`，再调用 [`policy_config.create_trained_policy()`](../../src/openpi/policies/policy_config.py) 下载或定位 checkpoint、加载模型参数、读取 normalization statistics，并组装一个 [`Policy`](../../src/openpi/policies/policy.py) 对象。完成这些工作后，脚本才把该对象交给 `WebsocketPolicyServer` 并进入 `serve_forever()`。

因此有两个不同阶段：

```text
启动期（一次）
config + checkpoint + norm stats
    -> create_trained_policy()
    -> Policy(model, input transforms, output transforms)
    -> WebsocketPolicyServer(policy)
    -> 开始监听

请求期（重复）
收到 observation
    -> 已存在的 self._policy.infer(obs)
    -> 返回 action chunk
```

除非进程被重启或上层显式换掉 Policy，请求处理循环不会为每次请求重新创建模型。第一次请求仍可能因为 JAX/XLA JIT 编译而较慢，但“首次编译”不等于“再次加载 checkpoint”。

### 一次请求的紧凑时序图

下面从客户端已经连接、服务端已经加载好 Policy 的时刻开始：

```text
LIBERO / 调用方       WebsocketClientPolicy       WebsocketPolicyServer          Policy / model
      |                        |                            |                           |
      |-- observation dict --->|                            |                           |
      |                        |-- msgpack bytes ---------->|                           |
      |                        |                            |-- unpack obs ------------>|
      |                        |                            |                           |-- input transforms
      |                        |                            |                           |-- Observation.from_dict()
      |                        |                            |                           |-- BaseModel.sample_actions()
      |                        |                            |                           |-- model-specific output transforms / decoding
      |                        |                            |                           |-- unnormalize + environment outputs
      |                        |                            |<-- result dict -----------|
      |                        |<-- msgpack response -------|                           |
      |<-- actions + timing ---|                            |                           |
```

连接刚建立时还有一次 metadata 握手。标准 [`serve_policy.py`](../../scripts/serve_policy.py) 路径先读取 `policy.metadata`，把它作为 `metadata=` 参数传给 server；server 保存为 `self._metadata`，并在连接建立后发送这份 `_metadata`。客户端在构造函数中接收并保存。它不属于每次 `infer()` 的 observation/action 往返。

### 八个组件各自负责什么

| 组件 | 做什么 | 输入 → 输出 | 依赖什么 |
| --- | --- | --- | --- |
| [`WebsocketClientPolicy.infer()`](../../packages/openpi-client/src/openpi_client/websocket_client_policy.py) | 把调用方字典编码成可传输字节，同步发送并等待一条响应 | observation `dict` → msgpack bytes → response `dict` | `websockets`、`msgpack_numpy`；不知道 JAX/Torch 模型内部 |
| [`WebsocketPolicyServer._handler()`](../../src/openpi/serving/websocket_policy_server.py) | 管理连接、解包请求、调用已加载 Policy、附加服务端耗时并编码响应 | request bytes → observation `dict` → result bytes | 一个在启动时传入的 `BasePolicy` 对象 |
| [`scripts/serve_policy.py`](../../scripts/serve_policy.py) | 解析服务参数，选择 config/checkpoint，先创建 Policy，再启动 WebSocket server | CLI 参数 → 已加载 Policy + 监听中的 server | `training/config.py`、`policy_config.py`、serving 层 |
| [`create_trained_policy()`](../../src/openpi/policies/policy_config.py) | 加载 JAX 或 PyTorch checkpoint、读取统计量、从数据配置取得 transform，并组装 Policy | `TrainConfig` + checkpoint 路径 → `Policy` | model config、checkpoint、data config、norm stats、transform 定义 |
| [`Policy.infer()`](../../src/openpi/policies/policy.py) | 编排一次本地推理：输入变换、加 batch/device 转换、模型采样、去 batch、输出变换 | 环境字典 → 机器人/环境可用的结果字典 | 已加载 `BaseModel`、输入/输出 transform、JAX 或 PyTorch runtime |
| 输入 transforms | 把环境协议逐步变成模型协议：LIBERO 字段适配、归一化、resize、tokenize、pad | 原始 observation → 统一的嵌套字典 | `DataConfig`、normalization assets、[`transforms.py`](../../src/openpi/transforms.py) |
| [`Observation.from_dict()`](../../src/openpi/models/model.py) | 把嵌套字典收束为模型公共 `Observation`；也把 `uint8` 图像换算到 `[-1, 1]` 浮点范围 | 已变换字典 → `Observation` | 固定字段约定：images、image masks、state、tokens/masks |
| [`BaseModel.sample_actions()`](../../src/openpi/models/model.py) 与输出 transforms | π₀/π₀.₅ 生成标准化连续 action chunk；π₀-FAST 先生成 token ID，再由模型专用输出 transform 解码；最后反归一化并还原环境动作字段 | `Observation` → 连续 action 或 token ID → `{"actions": ...}` | π₀/π₀.₅/π₀-FAST 实现、模型专用输出解码、机器人输出变换 |

`Policy.infer()` 都把模型返回值暂存在 `outputs["actions"]`，但这个 key 在模型专用输出处理前并不保证已经是连续动作：π₀/π₀.₅ 直接返回标准化的连续 `(B, H, D)` action chunk；π₀-FAST 返回 token-ID 序列，`model_transforms.outputs` 中的 `ExtractFASTActions` 先把它解码为连续 `(H, D)` 动作，之后才进入 `Unnormalize`。输出 transforms 是“模型专用处理/解码”，不一定与输入 transforms 一一互逆。

这张表可以简化成五种责任：

- **传输：**`WebsocketClientPolicy` 与 `WebsocketPolicyServer`；
- **预处理：**Policy 的 input transforms；
- **模型采样：**`BaseModel.sample_actions()`；
- **后处理：**Policy 的 output transforms；
- **配置与组装：**`training/config.py` 和 `policy_config.create_trained_policy()`。

### `Policy.infer()` 内部为什么还有一步 `Observation.from_dict()`

transform 的中间结果仍是容易组合、容易改 key 的普通字典。`Policy.infer()` 在输入变换完成后给每个数组增加 batch 维，并转为 JAX array 或 PyTorch tensor，然后调用 `Observation.from_dict()` 建立统一结构。模型因此只面对稳定字段：

```text
Observation
├── images
├── image_masks
├── state
├── tokenized_prompt
├── tokenized_prompt_mask
├── token_ar_mask       （π0-FAST 可用）
└── token_loss_mask     （π0-FAST 可用）
```

这一边界很重要：环境适配器可以使用 `observation/image` 等机器人字段，模型实现不必认识这些环境专用 key；反过来，网络层也不必知道 `Observation` 里的视觉 token 或 action expert。

### 轻量客户端与两个运行时

[`packages/openpi-client/pyproject.toml`](../../packages/openpi-client/pyproject.toml) 只声明 NumPy、Pillow、msgpack、websockets 等轻量依赖，没有 JAX、Flax、Torch、Orbax 或完整模型栈。根项目的 [`pyproject.toml`](../../pyproject.toml) 才包含模型训练和推理所需的重依赖。

这让控制/仿真运行时只安装 `openpi-client`，而模型运行时安装完整 openpi。两侧通过字典协议和 WebSocket 解耦：

```text
控制或仿真运行时                         模型运行时
环境依赖 + openpi-client   <—— WebSocket ——>  openpi + JAX/Torch + checkpoint
```

第 03 章推荐把两者放进 Compose 的两个容器，这是“双运行时”教学模型的一种具体部署。边界本身并不规定 Policy Server 必须在宿主机、容器或远端机器；只要网络可达、协议和数据语义一致，就可以选择合适的部署位置。

## 源码地图

### 按目录看职责

| 路径 | 这一层回答的问题 | 本章入口 |
| --- | --- | --- |
| [`packages/openpi-client`](../../packages/openpi-client) | 调用方怎样在不安装模型栈的情况下发送 observation、接收 actions？ | `websocket_client_policy.py`、`msgpack_numpy.py` |
| [`src/openpi/serving`](../../src/openpi/serving) | 网络请求怎样转成一次 Policy 调用？ | `websocket_policy_server.py` |
| [`src/openpi/policies`](../../src/openpi/policies) | 模型、transform 和机器人适配怎样组合成统一 `infer()`？ | `policy.py`、`policy_config.py`、`libero_policy.py` |
| [`src/openpi/transforms.py`](../../src/openpi/transforms.py) | 字段重排、归一化、resize、tokenize、delta/absolute、padding 怎样组合？ | `Group`、`compose`、`Normalize`、`PadStatesAndActions` |
| [`src/openpi/models`](../../src/openpi/models) | 统一 `Observation`/`BaseModel` 接口和具体 π 模型如何定义？ | `model.py`、`pi0_config.py`、`pi0.py`、`pi0_fast.py` |
| [`src/openpi/training`](../../src/openpi/training) | 命名配置怎样绑定模型、数据、统计量、训练与部署契约？ | `config.py`、`data_loader.py`、`checkpoints.py` |

### 建议阅读顺序

1. [`websocket_client_policy.py`](../../packages/openpi-client/src/openpi_client/websocket_client_policy.py)：找 `WebsocketClientPolicy.infer`；
2. [`websocket_policy_server.py`](../../src/openpi/serving/websocket_policy_server.py)：找 `_handler` 和 `self._policy.infer`；
3. [`serve_policy.py`](../../scripts/serve_policy.py)：确认 `create_policy` 发生在创建 server 之前；
4. [`policy_config.py`](../../src/openpi/policies/policy_config.py)：看 checkpoint、norm stats 和两组 transform 的组装；
5. [`policy.py`](../../src/openpi/policies/policy.py)：看 `Policy.infer` 的实际执行顺序；
6. [`transforms.py`](../../src/openpi/transforms.py) 与 [`libero_policy.py`](../../src/openpi/policies/libero_policy.py)：看每一步怎样改字典；
7. [`models/model.py`](../../src/openpi/models/model.py)：看 `Observation.from_dict` 和 `BaseModel.sample_actions` 契约；
8. [`training/config.py`](../../src/openpi/training/config.py)：回到最上层，确认 `pi05_libero` 选择了什么模型和数据配置。

若想继续阅读模型内部，先沿上面的当前源码入口和[第 06 章](06_jax_and_pi0_basics.md)继续。本工作区若存在 `docs/architecture_zh.md` 或 `docs/model_build_and_inference_zh.md`，可把它们作为可选本地深读资料；它们不是已提交教程的必需依赖。

## 动手实践

### 实践 1：用符号名还原启动期

从仓库根目录运行：

```bash
rg -n -A 20 '^def main' scripts/serve_policy.py
rg -n -B 2 -A 12 'logging\.info\("Loading model' src/openpi/policies/policy_config.py
rg -n -B 2 -A 16 '^    return _policy\.Policy' src/openpi/policies/policy_config.py
rg -n -A 13 '^    def __init__' src/openpi/serving/websocket_policy_server.py
rg -n -A 28 '^    async def _handler' src/openpi/serving/websocket_policy_server.py
```

在输出中按顺序指出：

1. `main()` 在哪里完成 `create_policy(args)`；
2. `create_trained_policy()` 在哪里加载模型与 normalization statistics；
3. Policy 在哪里传给 `WebsocketPolicyServer` 并保存为 `self._policy`；
4. `_handler()` 的循环为什么只复用 `self._policy.infer(obs)`。

如果你的结论是“服务端每次收到请求都会运行 `create_trained_policy()`”，请回到调用位置重新检查：`_handler()` 中没有该调用。

### 实践 2：从客户端逐跳追踪一个请求

继续使用符号搜索，不记录固定行号：

```bash
rg -n -A 10 '^    def infer' packages/openpi-client/src/openpi_client/websocket_client_policy.py
rg -n -A 28 '^    async def _handler' src/openpi/serving/websocket_policy_server.py
rg -n -A 40 '^    def infer\(self, obs: dict, \*, noise:' src/openpi/policies/policy.py
rg -n -A 20 '^    def from_dict' src/openpi/models/model.py
rg -n -A 2 '^    def sample_actions' src/openpi/models/model.py src/openpi/models/pi0_fast.py
```

把搜索结果整理成一行，不必抄源码：

```text
客户端 pack/send → 服务端 recv/unpack → Policy.infer → input transforms
→ Observation.from_dict → sample_actions → output transforms
→ 服务端 pack/send → 客户端 recv/unpack
```

然后在每个箭头上标注数据类型，例如 `dict`、msgpack bytes、batched array、`Observation`、action `dict`。这是本章最重要的源码导航产物。

### 实践 3：核对配置怎样进入部署

```bash
rg -n -B 2 -A 5 'EnvMode\.LIBERO: Checkpoint' scripts/serve_policy.py
rg -n -B 3 -A 10 'name="pi05_libero"' src/openpi/training/config.py
rg -n -B 2 -A 16 '^    return _policy\.Policy' src/openpi/policies/policy_config.py
```

请回答：命名配置在哪里选择模型结构、action horizon 和 LIBERO 数据适配？checkpoint 路径在哪里选择？真正执行 transform 列表的是哪个对象？如果换成自定义 checkpoint，为什么 config、参数和 assets 必须一起核对？

## 预期结果

完成静态练习后，你应得到一张类似下面的责任清单：

| 观察到的证据 | 可以得出的结论 |
| --- | --- |
| `serve_policy.main()` 先创建 Policy，再创建并启动 server | 模型在服务请求到来前加载 |
| server 构造函数保存 `self._policy`，handler 循环调用其 `infer` | 多个请求复用同一 Policy 对象 |
| 客户端和服务端都调用 `msgpack_numpy` | 传输层负责字典/NumPy 与 bytes 的边界 |
| `Policy.infer()` 先 `_input_transform`，后 `Observation.from_dict` 和 `_sample_actions` | 预处理、统一模型协议、模型采样有清楚顺序 |
| `_output_transform` 在去掉 batch 维之后执行 | 后处理面对单个请求的模型原生输出；π₀-FAST 会先在这里把 token ID 解码为连续 action |
| `training/config.py` 绑定 model 和 data factory | 配置既影响训练，也影响部署时的 shape、transform 和统计量 |

### 本文的验证边界

本章内容依据当前 checkout 的源码静态核对，命令均为只读搜索。没有启动 Policy Server、下载 checkpoint 或执行模型采样，因此不声称当前机器已经测得运行时 action 或延迟；第 03 章的运行证据与本章的调用链证据应分开保存。

## 常见问题

### 1. 为什么 server 文件里没有模型类？

先搜索依赖注入证据：

```bash
rg -n -A 13 '^    def __init__' src/openpi/serving/websocket_policy_server.py
```

server 只依赖 `BasePolicy` 接口，不负责理解 π₀ 或 π₀.₅。模型和 transform 已在 `create_trained_policy()` 中组装进 Policy；这种分层让同一个传输层可以服务不同策略。

### 2. 第一次请求很慢，是否说明每次都重新加载模型？

不要只凭耗时判断。先看日志中 checkpoint 加载发生在监听之前，后续请求是否重复出现同样的加载日志；再检查 `_handler()` 是否调用加载函数。常见情况是第一次 `sample_actions` 触发 JAX/XLA 编译，而不是重新读取 checkpoint。

### 3. `Policy` 和 `BaseModel` 有什么区别？

`BaseModel` 的公共推理职责是“给定统一 `Observation`，生成该模型的原生输出”。π₀/π₀.₅ 的原生输出是连续 action chunk；π₀-FAST 的原生输出是 token ID，需要 `ExtractFASTActions` 解码。`Policy` 还负责环境字段适配、normalization、batch/device、模型专用输出处理、机器人输出适配和 timing。把一个模型直接当成完整机器人策略，会漏掉这些协议步骤。

### 4. 为什么不把模型依赖直接装进客户端？

先比较两个依赖文件：

```bash
sed -n '/dependencies = \[/,/^\]/p' packages/openpi-client/pyproject.toml
sed -n '/dependencies = \[/,/^\]/p' pyproject.toml
```

控制或仿真环境通常有自己的 Python、驱动和实时依赖。轻量客户端减少依赖冲突，也让 GPU 模型运行时独立扩缩容。代价是两侧必须严格维护网络可达性和字典协议一致性。

### 5. WebSocket server 是否一定运行在另一台主机？

不是。WebSocket 是进程边界，不是固定部署地点。它可以连接同机进程、两个容器或远端主机。排错时应先记录当前实际的 host、port、容器网络和启动命令，不要从“client/server”名称推断物理位置。

更多跨章节排错入口见[故障排查手册](troubleshooting.md)，术语见[课程术语表](glossary.md)。

## 本章验收

进入下一章前，逐项口头解释或保存搜索输出：

- [ ] 能从 `WebsocketClientPolicy.infer()` 开始，按正确顺序复述直到客户端得到响应的完整链路；
- [ ] 能指出负责**传输**的是 WebSocket client/server；
- [ ] 能指出负责**预处理**的是 Policy input transforms；
- [ ] 能指出负责**模型采样**的是 `BaseModel.sample_actions()`；
- [ ] 能指出负责**后处理**的是 Policy output transforms；
- [ ] 能区分 π₀/π₀.₅ 的连续 action 输出与 π₀-FAST 的 token ID 输出，并指出 FAST 在反归一化前由 `ExtractFASTActions` 解码；
- [ ] 能指出负责**配置与组装**的是 `training/config.py` 和 `policy_config.create_trained_policy()`；
- [ ] 能用源码说明 Policy 在启动期加载、在请求期复用；
- [ ] 能解释 `Observation.from_dict()` 为什么位于 transform 字典与模型对象之间；
- [ ] 能解释轻量 `openpi-client` 与模型运行时分离的依赖收益，并且不会把 server 误说成永远运行在宿主机。

## 思考题

### 1. 如果客户端能连接但服务端报告缺少 `observation/state`，应先看模型内部还是协议边界？

<details>
<summary>参考答案（展开查看）</summary>

先看客户端构造的字典、msgpack 解包结果和 input transforms。缺 key 发生在统一 `Observation` 建立之前，优先属于环境协议或数据适配问题；没有证据时不应先修改模型网络。

</details>

### 2. 为什么 `training/config.py` 会出现在推理架构里？

<details>
<summary>参考答案（展开查看）</summary>

命名 `TrainConfig` 同时绑定模型结构、数据配置、action dimension/horizon 和 normalization assets。`create_trained_policy()` 用它重建与训练一致的模型和 transform，因此它也是部署契约的一部分，而不只是优化器参数集合。

</details>

### 3. 如果要在同一进程里直接调用 Policy，还必须经过 WebSocket 吗？

<details>
<summary>参考答案（展开查看）</summary>

不必须。WebSocket 层解决跨进程或跨运行时传输；`Policy.infer()` 本身是本地 Python 接口。绕过网络不会取消输入/output transform、normalization 和模型协议要求。

</details>

[← 上一章：第一次 LIBERO 预训练推理](03_first_libero_inference.md) · [课程入口：README](README.md) · [下一章：数据协议与 Transform →](05_data_and_transforms.md)
