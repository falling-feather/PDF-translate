# PDF 学术文献翻译工具

面向**英文学术 PDF** 的工具集：自动拆分正文与参考文献、按页分块翻译、本地 **memory/** 记忆（术语与摘要）、支持 **CLI** 与 **Web 工作台**（多用户、管理员配置 API、审计日志）。合并输出为 **Markdown**，并导出链接索引等辅助文件。

---

## 功能概览

| 能力 | 说明 |
|------|------|
| 正文 / 参考文献拆分 | 基于标题启发式识别 References 等；可选「尾部 15% 视为参考文献」兜底 |
| 分块翻译 | 每块 1–3 页，块间可配置重叠页，控制单次上下文规模 |
| 多翻译后端 | `echo`（联调）、OpenAI 兼容（含 **DeepSeek**）、Ollama、DeepL、`hybrid`（初稿 + 润色） |
| 记忆注入 | `memory/` 下术语表、分块摘要、风格说明等；串联模式带前文摘要与译文段尾衔接 |
| 可选译前巡视 | 开启 `PDF_TRANSLATE_SURVEY_ENABLED` 时，经**硅基流动**草拟术语并写入 `glossary`，产出 `output/survey/*.json`；后续规划见 [docs/02-更新规划.md](./docs/02-更新规划.md) |
| 串联 / 并联 | 串联：顺序翻译、记忆递进；并联：分批并行请求，按序拼接（更快，衔接略弱） |
| 段尾顺延（串联 + LLM） | 非最后一块可要求模型输出 `《&fenduan&》` / `《&fenju&》` 与未译英文尾段，写入 `deferred_source_carry.txt`，下块接续；合并后仅做标识符替换，不再调用模型审视全文 |
| 任务控制 | Web 端可**终止**翻译；支持断点续译（`output/state.json`） |
| 下载命名 | 译文 `.md` 与打包 `.zip` 按原 PDF 文件名生成中文友好名（如 `论文名（翻译版）.md`） |
| 译后质量闭环 | 本地生成 QA 报告、局部修复计划和双语 HTML，覆盖数字、引用、表格形状、术语一致性与术语冲突等基础检查 |
| 资料包 | zip 内路径使用中文分类：`译文/`、`质量/`、`设置/`、`记忆/`、`关键词/`、`原文/` 等 |

---

## 技术栈

- **Python 3.10+**：核心库与 FastAPI 服务  
- **PyMuPDF**：PDF 文本与拆分  
- **Typer**：命令行  
- **httpx**：翻译 API 调用与重试  
- **SQLite + bcrypt + JWT**：Web 用户与登录  
- **Vue 3 + Vite**：前端（构建后由后端静态托管）

---

## 仓库结构（主要部分）

```
pdf translate/
├── pyproject.toml          # 包定义与依赖
├── start_web.bat           # Windows：检查前端产物后启动 Web（环境变量见脚本内注释）
├── build_frontend.bat      # Windows：npm install + build，输出到 server/static
├── README.md               # 本说明（含试运行与部署要点）
├── SETUP_MANUAL.md         # 安装与 API / 环境变量详解
├── docs/                   # 开发者文档、更新规划、专利优化方向、测试验收
├── pdf_translate/          # Python 包
│   ├── cli.py              # Typer CLI
│   ├── pipeline.py         # 拆分、翻译主流程
│   ├── pipeline_merge.py   # 块合并与标识符收尾
│   ├── memory_store.py     # memory/ 读写
│   ├── chunking.py         # 按页分块
│   ├── extractors/         # DocumentIR 与本地结构抽取
│   ├── chunkers/           # 结构感知分段器
│   ├── qa/                 # 结构不变量与 QA 报告
│   ├── exporters/          # 双语 HTML 与未来 PDF 输出
│   ├── vision/             # 图像页、扫描页、OCR/VLM 路由
│   ├── deferral_markers.py # 段尾顺延协议与合并替换
│   ├── translators/        # 各翻译后端
│   └── server/             # FastAPI、任务队列、静态资源 static/
├── frontend/               # Vue 源码（需 build 进 server/static）
└── data/                   # 默认数据根（可通过环境变量修改）
    ├── app.db              # 用户、任务元数据、审计、KV 配置
    └── web_jobs/<job_id>/  # 每个上传任务的工作目录
```

---

## 试运行（推荐顺序）

用于在本机快速验收 Web 功能：

1. **Python 虚拟环境**（若尚未创建）：`python -m venv .venv`，激活后 `pip install -e .`。
2. **构建前端**（若 `pdf_translate/server/static/index.html` 不存在）：
   - Windows：双击 **`build_frontend.bat`**；
   - 或手动：`cd frontend` → `npm install` → `npm run build`。
3. **启动服务**：Windows 双击 **`start_web.bat`**；其它平台见下文「启动服务」。
4. 浏览器打开 **`http://127.0.0.1:901`**（端口可在 `start_web.bat` 或环境变量 `PDF_TRANSLATE_WEB_PORT` 中修改）。
5. 使用登录页账号进入；**数据库为空**时，首个管理员由 `PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD` 创建（见 bat 内说明），**务必在正式环境修改密码**。
6. **用户工作台**：进入后会刷新并行窗口状态；未收藏任务超过 **24 小时**会在进入时从库中清理并删除对应 `web_jobs` 目录（进行中任务会跳过）；「我的任务」支持**收藏**（每用户最多 3 条），取消收藏后任务回到列表且创建时间更新。
7. **管理端**：`/admin` 可配置 API；操作日志以**中文摘要**展示，可展开查看原始 JSON 详情。

---

## 快速开始

### 1. Python 环境

```bash
cd /path/to/pdf-translate
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -e .
```

验证：

```bash
pdf-translate --help
```

Web 入口为 `pdf-translate-web` 或 `python -m pdf_translate.server`（直接启动服务，无子命令帮助）。

### 2. 配置 API（可选）

复制或新建 `.env`，至少配置默认后端与对应密钥。最小离线联调：

```env
PDF_TRANSLATE_BACKEND=echo
```

OpenAI / DeepSeek / Ollama / DeepL 等完整列表与说明见 **[SETUP_MANUAL.md](./SETUP_MANUAL.md)**。

### 3. Web 工作台

**构建前端**（需 Node.js LTS）：

```bash
cd frontend
npm install
npm run build
```

产物输出到 `pdf_translate/server/static/`。

**启动服务**：

```bash
pdf-translate-web
```

默认 **`http://0.0.0.0:901`**。浏览器访问 `http://127.0.0.1:901`，API 文档：`/docs`。

**Windows**：先按需运行 **`build_frontend.bat`** 构建静态页，再运行 **`start_web.bat`**（脚本内注释含 `PDF_TRANSLATE_DATA`、`PDF_TRANSLATE_WEB_PORT`、`PDF_TRANSLATE_WEB_HOST`、`PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD` 等）。

### 4. 环境变量（Web 与数据）

| 变量 | 说明 |
|------|------|
| `PDF_TRANSLATE_DATA` | 数据根目录，默认 `./data`；其下为 `app.db` |
| `PDF_TRANSLATE_WEB_DATA` | 任务工作目录根，默认 `<DATA>/web_jobs` |
| `PDF_TRANSLATE_WEB_PORT` | 监听端口，默认 `901` |
| `PDF_TRANSLATE_WEB_HOST` | 默认 `0.0.0.0`；仅本机可设为 `127.0.0.1` |
| `PDF_TRANSLATE_WEB_RELOAD` | 设为 `1` / `true` 时 uvicorn 自动重载（开发用） |
| `PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD` | **仅在库中无任何用户时**用于创建首个管理员 |
| `PDF_TRANSLATE_ADMIN_USERNAME` | 可选，保留管理员用户名（默认见代码/文档） |
| `PDF_TRANSLATE_PARALLEL_WORKERS` | 并联模式下每批线程数上限相关默认（见服务端逻辑） |

更多 HTTP 超时、重试、各厂商 API 变量见 **SETUP_MANUAL.md**。

---

## Web 使用说明

- **登录**：普通用户与管理员共用登录页；管理员可进入 `/admin`。  
- **用户**：上传 PDF，选择每块页数、重叠、翻译后端（若管理员允许）、**串联 / 并联**、并联并发数；可**终止**进行中的任务；完成后查看**总用时**，下载 **Markdown** 或 **zip 资料包**。  
- **我的任务**：列表时间按本地时区可读格式显示；支持**收藏**（最多 3 条），收藏区与主列表风格一致；**每次进入翻译工作台**会清空并行窗口缓存、拉取列表，并对**未收藏且创建超过 24 小时**的任务执行库与磁盘清理（`queued`/`running` 不删）。  
- **管理员**：配置各后端 API Key（可与环境变量叠加/覆盖规则见设置逻辑）、启用后端列表、是否开放注册；审计日志为**自然语言摘要**，可展开看完整详情；可浏览全部任务并下载产物。  
- **断点续译**：同一任务目录下保留 `output/state.json` 与块文件即可在再次运行时跳过已完成块（具体以后端任务是否复用同一工作目录为准；Web 每次上传为新任务目录）。

---

## 命令行（CLI）

```bash
pdf-translate init <工作目录>
pdf-translate split <输入.pdf> <工作目录> [--tail-fallback]
pdf-translate translate <工作目录> [-b openai] [--pages 3] [--overlap 1] [--chunk-strategy page|structure] [--max-chunks N]
pdf-translate links <工作目录>
pdf-translate run <输入.pdf> <工作目录>   # init + split + translate 一键
```

CLI 的 `translate` / `run` 默认使用 **串联** 与当前 `pipeline` 默认参数；`--chunk-strategy structure` 为实验性结构分段路径，会基于 `DocumentIR` 结构块生成翻译块。**并联、Web 专属选项**以 Web 表单为准。详见 `pdf-translate translate --help`。

---

## 流水线与产物（单任务目录）

1. **拆分**：`split/main.pdf`、`split/references.pdf`（若有）、`split/manifest.json`  
2. **分块翻译**：`output/chunks/c0000.md` …（含 YAML 元数据 + 译文正文）  
3. **状态**：`output/state.json`、`output/chunks_manifest.json`、`output/run_log.jsonl`  
4. **合并稿**：`output/translated_full.md`（合并时默认去掉各块 YAML 头，并对顺延标识符做替换）  
5. **记忆**：`memory/glossary.json`、`chunk_summaries.json`、`style_notes.yaml` 等  
6. **结构层**：`output/document_ir.json`、`output/structure_chunks_manifest.json`、`output/structure_qa.json`、`output/vision_route.json`（结构识别、边界保护式结构分段预览、表格维度、图注/表注/脚注归属、页面边界碎片、页级 OCR/VLM 路由摘要）
7. **译后 QA 与修复计划**：`output/qa_report.json`、`output/qa_report.md`、`output/repair_plan.json`、`output/repair_plan.md`（数字/引用/表格/术语一致性等检查，以及局部修复建议）
8. **阅读输出**：`output/bilingual.html`（按块展示原文、译文、QA 问题和局部修复建议）
9. **链接**：`output/links_index.csv`

### memory/ 目录（CLI 工作目录）

`pdf-translate init` 后会在工作目录生成 `memory/`，常见文件：

| 文件 | 用途 |
|------|------|
| `glossary.json` | 术语表（英中对照、别名、页码等） |
| `chunk_summaries.json` | 各块短摘要，供串联翻译时注入上下文 |
| `style_notes.yaml` | 风格偏好（人称、时态等） |
| `pending_review.json` | 待确认的术语冲突、同译名复核或边界问题（若有工作流写入） |

---

## 服务器部署要点（Windows / 云主机）

对外提供服务时，除 **README「试运行」** 与 **SETUP_MANUAL** 中的安装步骤外，建议：

1. **独立数据盘**：设置环境变量 `PDF_TRANSLATE_DATA` 指向固定目录（如 `C:\pdf-translate-data`），其下会有 `app.db` 与 `web_jobs/`。首次无用户时可用 `PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD` 创建管理员。
2. **监听地址**：公网访问设置 `PDF_TRANSLATE_WEB_HOST=0.0.0.0`，`PDF_TRANSLATE_WEB_PORT=901`（或你选用的端口）。
3. **防火墙与安全组**：在云厂商安全组与本机「入站规则」中放行对应 **TCP 端口**；生产环境请修改默认引导密码并限制 CORS（见 `PDF_TRANSLATE_CORS_ORIGINS`）。
4. **更新流程**：`git pull` → 激活 venv → `pip install -e .` → `cd frontend && npm ci && npm run build`（若仓库内未带最新 `server/static`）→ 重新启动 `python -m pdf_translate.server`。
5. **HTTPS / 域名**：可在前级使用 Caddy、IIS 或 Nginx 反代到本机端口；与应用程序配置相互独立。

---

## 限制与提示

- 翻译质量与费用取决于所选模型与分块策略；**长块**首次请求可能较慢，服务端已对常见 HTTP 失败做退避重试。  
- **段尾顺延**依赖模型严格按协议输出标识符与英文尾段；**DeepL / echo / hybrid** 路径不会启用该协议。  
- 大文件请注意上传体积上限（Web 端当前策略见 `routes_web` 实现）。  
- 生产部署请修改默认密码、限制 CORS、并考虑 HTTPS 与防火墙策略。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [SETUP_MANUAL.md](./SETUP_MANUAL.md) | 安装步骤、前端构建、环境变量全表、DeepSeek / 代理、CLI 顺序等 |
| [docs/00-方向总览.md](./docs/00-方向总览.md) | 正式文档入口，汇总算法流程、表格/图像/分段/输出形态等方向 |
| [docs/01-开发者文档.md](./docs/01-开发者文档.md) | 架构、目录边界、核心管线、扩展规则与安全边界 |
| [docs/02-更新规划.md](./docs/02-更新规划.md) | 本地 OCR、结构感知分段、QA、术语确认台、HTML/PDF 输出等后续规划 |
| [docs/03-专利优化方向.md](./docs/03-专利优化方向.md) | 面向专利目标的技术方案、原型模块与实验材料清单 |
| [docs/04-竞品与技术边界.md](./docs/04-竞品与技术边界.md) | 类似工具、开源路线、云服务能力和差异化定位 |
| [docs/05-测试与验收指南.md](./docs/05-测试与验收指南.md) | 自动化测试、手工回归、质量指标和专利原型实验记录 |
| [docs/06-发布历史.md](./docs/06-发布历史.md) | 已完成能力、验证事实和仍需后续处理的风险 |

**维护者本地（默认不入库）**：根目录 `.gitignore` 会排除 **`plans/`** 目录及 **`DEPLOY_WINDOWS_SERVER.md`**、**`PROJECT_DESIGN.md`**。`plans/` 仅作为本机草稿来源；正式协作、后续规划和专利准备材料以 `docs/` 为准。

**版本库不跟踪的内容**（见 `.gitignore`）：`.venv`、`data/`、`frontend/node_modules`、`.env`、本地素材草稿 `sucai/`、上述规划类路径等。  
**随仓库提供的构建产物**：`pdf_translate/server/static/` 为一次前端构建结果；修改 Vue 后请在 `frontend` 下执行 `npm run build` 再提交对应静态文件。

---

## 许可证

若未单独提供许可证文件，默认以项目所有者声明为准；使用第三方 API 时请遵守各服务商条款。
