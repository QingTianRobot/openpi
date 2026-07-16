# openpi 控制方式与频率飞书专题 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有飞书报告第 5 章末尾增加一张 openpi Action Chunk 控制流程图、两张频率表和源码解释，并完成结构与事实验证。

**Architecture:** 从 openpi 当前源码提取频率、horizon、重规划和 timing 证据；回读飞书第 5 章最新 block ID；使用一次 XML `block_insert_after` 插入完整专题，避免多次写入导致顺序变化；最后回读全文验证位置、画板、数字、风险说明和来源链接。

**Tech Stack:** openpi 本地源码、`lark-cli docs`、飞书 Docx XML、Mermaid whiteboard、只读 JSON 验证

---

### Task 1: 核对源码与飞书锚点

**Files:**
- Read: `docs/norm_stats.md`
- Read: `docs/remote_inference.md`
- Read: `examples/aloha_real/main.py`
- Read: `examples/aloha_sim/main.py`
- Read: `examples/droid/main.py`
- Read: `examples/droid/README.md`
- Read: `examples/libero/main.py`
- Read: `src/openpi/models/pi0_config.py`
- Read: `src/openpi/models/pi0_fast.py`
- Read: `src/openpi/training/config.py`

- [ ] **Step 1: 确认频率与 chunk 参数**

Run: `rg -n 'Control frequencies|control frequency|max_hz=|action_horizon|open_loop_horizon|replan_steps|0.5 - 1 sec' docs examples src packages/openpi-client`

Expected: 找到 15/20/50 Hz、ALOHA 25/10 步、DROID 8 步、LIBERO 5 步、模型 horizon 和 DROID 延迟证据。

- [ ] **Step 2: 获取第 5 章最新 block 结构**

先从目录回读结果取得 `chapter5_id`，再运行：`lark-cli docs +fetch --as user --doc O44SdhpWXobXdOxIvyGcWIx1nmf --scope section --start-block-id "$chapter5_id" --detail full`

Expected: 返回第 5 章全部内容，确定第 6 章之前最后一个顶层 block ID。

### Task 2: 生成专题 XML

**Files:**
- Create: `/tmp/robotics-company-day1-feishu/openpi-control-frequency.xml`

- [ ] **Step 1: 编写流程图、表格和说明**

XML 必须包含：专题 h2、Mermaid 控制机制图、平台频率表、Action Horizon 表、名义与实际频率公式、同步阻塞解释、DROID 源码风险 callout 和官方 GitHub 来源列表。

- [ ] **Step 2: 静态检查内容**

Run: `rg -n 'Action Chunk|50 Hz|20 Hz|15 Hz|1.875 Hz|0.5–1 秒|policy_timing|server_timing|pi05_droid|\(10, 8\)|action horizon' /tmp/robotics-company-day1-feishu/openpi-control-frequency.xml`

Expected: 所有关键事实均存在，不含占位内容或未转义的 XML 特殊字符。

### Task 3: 写入飞书

**Files:**
- Read: `/tmp/robotics-company-day1-feishu/openpi-control-frequency.xml`

- [ ] **Step 1: 局部插入完整专题**

Run:

```bash
lark-cli docs +update --as user \
  --doc O44SdhpWXobXdOxIvyGcWIx1nmf \
  --command block_insert_after \
  --block-id "$anchor_block_id" \
  --content @openpi-control-frequency.xml
```

Expected: `ok=true`、`result=success`、warnings 为空，并返回一个新的 whiteboard block。

### Task 4: 回读与验证

**Files:**
- Create: `/tmp/robotics-company-day1-feishu/openpi-control-frequency-verification.json`

- [ ] **Step 1: 回读全文**

Run: `lark-cli docs +fetch --as user --doc O44SdhpWXobXdOxIvyGcWIx1nmf --detail full`

Expected: 新专题出现在第 5 章与第 6 章之间，一级标题仍为 11 个，唯一 whiteboard token 为 11 个。

- [ ] **Step 2: 验证事实和保真度**

检查专题标题、流程图、两张表、15/20/50 Hz、ALOHA/DROID/LIBERO、Action Horizon、0.5–1 秒、同步阻塞、timing 字段、DROID horizon 风险和官方来源链接全部存在；原有 10 个“图解：”标题、关键术语与来源仍保留。

- [ ] **Step 3: 保存验证报告**

将 revision、章节数、画板数、专题位置、事实检查和原有内容检查写入验证 JSON。Expected: 顶层 `ok=true`。
