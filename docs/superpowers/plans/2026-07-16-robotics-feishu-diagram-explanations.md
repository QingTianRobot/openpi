# 飞书机器人基础设施报告图解增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为现有飞书机器人基础设施报告的全部 10 张图增加结构化文字解读，并验证文档结构与关键内容未被破坏。

**Architecture:** 先回读最新文档并提取画板及相邻 block ID；再按章节逐一使用 XML `block_insert_after` 插入说明。Day 1 章节的两张连续画板分别处理。最后回读全文，统计画板、解释标题、关键词和来源。

**Tech Stack:** `lark-cli docs`、飞书 Docx XML、Python 只读校验、现有 `/tmp/robotics-company-day1-feishu` 工作目录

---

### Task 1: 获取最新文档结构

**Files:**
- Read: `/tmp/robotics-company-day1-feishu/fetched.xml`
- Create: `/tmp/robotics-company-day1-feishu/explanations/targets.json`

- [ ] **Step 1: 全量获取最新文档及 block ID**

Run: `lark-cli docs +fetch --as user --doc O44SdhpWXobXdOxIvyGcWIx1nmf --detail full`

Expected: `ok=true`，返回 11 个一级标题和 10 个唯一 whiteboard token/block ID。

- [ ] **Step 2: 建立 10 张图的目标清单**

保存章节、画板 block ID、画板 token、后继 block ID和解释标题。Expected: `decision`、`architecture`、`day1_loop`、`day1_repo`、`transport`、`inference`、`state_machine`、`ml_loop`、`quality`、`roadmap` 十个键全部存在。

### Task 2: 生成图解 XML

**Files:**
- Create: `/tmp/robotics-company-day1-feishu/explanations/*.xml`

- [ ] **Step 1: 为每张图编写说明**

每个文件包含一个二级标题和四个短段落或项目符号，分别回答读图顺序、箭头/区域、关键边界和工程含义；每张图约 200–300 字。

- [ ] **Step 2: 做内容静态检查**

Run: `rg -n '读图顺序|箭头|边界|落地' /tmp/robotics-company-day1-feishu/explanations/*.xml`

Expected: 10 个说明文件均命中结构化解释关键词，且不存在占位内容或空段落。

### Task 3: 写入飞书文档

**Files:**
- Read: `/tmp/robotics-company-day1-feishu/explanations/*.xml`

- [ ] **Step 1: 按画板 block ID 局部插入**

对 10 张图依次运行：

```bash
lark-cli docs +update --as user \
  --doc O44SdhpWXobXdOxIvyGcWIx1nmf \
  --command block_insert_after \
  --block-id '<whiteboard_block_id>' \
  --content -
```

通过 stdin 传入对应 XML。Expected: 每次 `ok=true`，无 warnings。

- [ ] **Step 2: 处理连续画板的顺序**

Day 1 第一张图说明写入后重新 fetch 该章节，再定位第二张画板，避免插入顺序颠倒或复用失效 ID。

### Task 4: 最终回读验证

**Files:**
- Create: `/tmp/robotics-company-day1-feishu/explanations/verification.json`

- [ ] **Step 1: 回读全文**

Run: `lark-cli docs +fetch --as user --doc O44SdhpWXobXdOxIvyGcWIx1nmf --detail full`

Expected: 获取成功，仍有 11 个一级标题和 10 个唯一 whiteboard token。

- [ ] **Step 2: 验证新增解释**

检查 10 个解释标题全部存在，每个区块均包含读图顺序、箭头或区域、边界和工程落地含义；同时验证 `Safety Kernel`、`ObservationEnvelope`、`ActionCandidate`、`ModelRelease`、`FAULT_LATCHED`、`CI`、四平面和原有来源链接仍然存在。

- [ ] **Step 3: 保存验证结果**

将计数、标题列表、关键词检查、来源检查和文档 revision 写入 `verification.json`。Expected: 顶层 `ok=true`。
