# openpi 教程飞书发布实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前飞书用户“我的空间”根目录发布一套包含 15 篇文档的 openpi 中文章节式教程，并重建教程导航与源码链接。

**Architecture:** 先创建目标文件夹和 15 篇空文档以获得稳定 URL，再生成只存在于临时目录的 Markdown 发布稿。发布稿把教程内相对链接映射到飞书 URL，把仓库相对路径映射到 GitHub 主分支；正文按二级标题分段写入，最后逐篇读取核验。

**Tech Stack:** `lark-cli` Drive/Docs shortcuts、Python 3 标准库、Markdown、飞书云文档 Docx

---

### Task 1: 本地内容与飞书身份预检

**Files:**
- Read: `docs/tutorial/*.md`
- Read: `docs/superpowers/specs/2026-07-15-openpi-tutorial-feishu-publishing-design.md`

- [ ] **Step 1: 核对输入文件集合**

Run: `rg --files docs/tutorial | sort`

Expected: 恰好列出 README、课程大纲、第 01–11 章、术语表和故障排查手册，共 15 个 Markdown 文件。

- [ ] **Step 2: 检查飞书用户登录态与权限**

Run: `LARKSUITE_CLI_NO_UPDATE_NOTIFIER=1 LARKSUITE_CLI_NO_SKILLS_NOTIFIER=1 lark-cli auth status --json --verify`

Expected: `verified=true`，且 `identities.user.status` 与 `identities.user.tokenStatus` 表示用户登录有效。若缺权限，按错误中的最小 scope 发起 split-flow 授权，不对 bot 身份执行用户登录。

### Task 2: 创建飞书目录与文档清单

**Files:**
- Create: `/tmp/openpi-feishu-tutorial/manifest.json`

- [ ] **Step 1: 创建根目录文件夹**

Run: `lark-cli drive +create-folder --as user --name "openpi 零基础章节式教程"`

Expected: 返回 `folder_token` 和可访问的文件夹 URL。

- [ ] **Step 2: 创建 15 篇空文档**

按以下固定前缀映射读取源文件一级标题：

```text
README.md=00
00_course_outline.md=01
01_vla_and_robotics_basics.md=02
02_environment_setup.md=03
03_first_libero_inference.md=04
04_openpi_architecture.md=05
05_data_and_transforms.md=06
06_jax_and_pi0_basics.md=07
07_prepare_small_dataset.md=08
08_configure_finetuning.md=09
09_run_finetuning.md=10
10_serve_and_evaluate.md=11
11_next_steps.md=12
glossary.md=13
troubleshooting.md=14
```

对每个源文件设置 `title="$prefix ${h1#\# }"`，然后执行：

```bash
lark-cli docs +create --as user \
  --title "$title" \
  --parent-token "$folder_token"
```

Expected: 每次返回唯一的 `document_id` 和 `url`，且文档归属目标文件夹。

- [ ] **Step 3: 保存发布清单**

将 `folder_token`、文件夹 URL，以及每个源文件的 `title`、`document_id`、`url` 写入 `/tmp/openpi-feishu-tutorial/manifest.json`。清单不得包含 access token 或 app secret。

Expected: 清单包含 15 个文档条目，源文件名无重复，文档 URL 无重复。

### Task 3: 生成飞书 Markdown 发布稿

**Files:**
- Create: `/tmp/openpi-feishu-tutorial/publish/*.md`
- Read: `docs/tutorial/*.md`

- [ ] **Step 1: 构建链接映射**

从 `manifest.json` 建立 `README.md`、`00_course_outline.md`、第 01–11 章、`glossary.md`、`troubleshooting.md` 到飞书 URL 的映射。

- [ ] **Step 2: 转换教程内链接**

仅在 Markdown 围栏代码块之外转换链接：

- `*.md` 或 `*.md#标题锚点` 且目标位于 `docs/tutorial`：替换为对应飞书文档 URL；跨文档标题锚点不拼接 Markdown slug。
- `../`、`../../` 指向仓库其他文件：解析为仓库根路径 `repo_path` 并替换为 `https://github.com/Physical-Intelligence/openpi/blob/main/$repo_path`，保留 GitHub 标题锚点。
- `https://`、`http://`、`#本页锚点`：保持不变。

Expected: 发布稿不再包含指向教程其他 Markdown 文件的相对链接；源码路径均落在官方 GitHub 仓库主分支。

- [ ] **Step 3: 保真检查**

Run: `wc -l docs/tutorial/*.md /tmp/openpi-feishu-tutorial/publish/*.md`

Expected: 每篇发布稿与源文档行数相同；一级标题、代码围栏数量和表格分隔行数量一致。

### Task 4: 分段写入 15 篇飞书文档

**Files:**
- Read: `/tmp/openpi-feishu-tutorial/publish/*.md`

- [ ] **Step 1: 按二级标题切分正文**

对每篇发布稿，在围栏代码块之外以 `## ` 标题为边界切分。第一段包含一级标题和导语，后续每段以一个二级标题开头；禁止在代码围栏或 Markdown 表格行中切分。

- [ ] **Step 2: 写入第一段**

Run:

```bash
lark-cli docs +update --as user \
  --doc "$document_id" \
  --command overwrite \
  --doc-format markdown \
  --content -
```

通过 stdin 输入第一段。Expected: `result=success`，无致命 warning。

- [ ] **Step 3: 依次追加剩余段落**

Run:

```bash
lark-cli docs +update --as user \
  --doc "$document_id" \
  --command append \
  --doc-format markdown \
  --content -
```

通过 stdin 按原顺序输入每个剩余分段。Expected: 所有更新均返回 `result=success`；失败时只重试当前文档当前分段。

### Task 5: 远端核验与交付

**Files:**
- Read: `/tmp/openpi-feishu-tutorial/manifest.json`

- [ ] **Step 1: 核验目标文件夹清单**

调用 `drive files list` 前先读取对应参数模板并执行 schema；分页列出目标文件夹内容。

Expected: 目标文件夹包含恰好 15 篇本次创建的 Docx，标题与设计一致。

- [ ] **Step 2: 核验每篇标题和关键内容**

对 15 篇文档执行：

```bash
lark-cli docs +fetch --as user \
  --doc "$document_id" \
  --doc-format markdown \
  --detail simple
```

Expected: 文档含原一级标题；课程入口包含“这门课解决什么问题”，术语表包含“VLA”，故障排查手册包含“故障排查”；每篇至少有一个二级标题。

- [ ] **Step 3: 抽查导航和源码链接**

在课程入口、课程大纲、第 05 章、故障排查手册中抽查：教程导航链接为飞书 Docx URL，源码引用为官方 GitHub URL，不再出现 `](README.md)`、`](../` 或 `](../../`。

- [ ] **Step 4: 返回结果**

向用户提供目标文件夹 URL、成功发布的文档数量、核验结果；若有降级项，明确列出受影响文档和表达形式。
