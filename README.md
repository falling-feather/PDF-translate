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
| 串联 / 并联 | 串联：顺序翻译、记忆递进；并联：分批并行请求，按序拼接（更快，衔接略弱） |
| 段尾顺延（串联 + LLM） | 非最后一块可要求模型输出 `《&fenduan&》` / `《&fenju&》` 与未译英文尾段，写入 `deferred_source_carry.txt`，下块接续；合并后仅做标识符替换，不再调用模型审视全文 |
| 任务控制 | Web 端可**终止**翻译；支持断点续译（`output/state.json`） |
| 下载命名 | 译文 `.md` 与打包 `.zip` 按原 PDF 文件名生成中文友好名（如 `论文名（翻译版）.md`） |
| 资料包 | zip 内路径使用中文分类：`译文/`、`设置/`、`记忆/`、`关键词/`、`原文/` 等 |

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
├── start_web.bat           # Windows 一键启动 Web（端口、数据目录、引导管理员密码）
├── README.md               # 本说明
├── SETUP_MANUAL.md         # 安装与 API / 环境变量详解
├── PROJECT_DESIGN.md       # 设计思路与记忆目录约定
├── pdf_translate/          # Python 包
│   ├── cli.py              # Typer CLI
│   ├── pipeline.py         # 拆分、翻译主流程
│   ├── pipeline_merge.py   # 块合并与标识符收尾
│   ├── memory_store.py     # memory/ 读写
│   ├── chunking.py         # 按页分块
│   ├── deferral_markers.py # 段尾顺延协议与合并替换
│   ├── translators/        # 各翻译后端
│   └── server/             # FastAPI、任务队列、静态资源 static/
├── frontend/               # Vue 源码（需 build 进 server/static）
└── data/                   # 默认数据根（可通过环境变量修改）
    ├── app.db              # 用户、任务元数据、审计、KV 配置
    └── web_jobs/<job_id>/  # 每个上传任务的工作目录
```

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

**Windows**：可直接运行项目根目录 **`start_web.bat`**（内含 `PDF_TRANSLATE_DATA`、`PDF_TRANSLATE_WEB_PORT`、`PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD` 等示例环境变量）。

### 4. 环境变量（Web 与数据）

| 变量 | 说明 |
|------|------|
| `PDF_TRANSLATE_DATA` | 数据根目录，默认 `./data`；其下为 `app.db` |
| `PDF_TRANSLATE_WEB_DATA` | 任务工作目录根，默认 `<DATA>/web_jobs` |
| `PDF_TRANSLATE_WEB_PORT` | 监听端口，默认 `901` |
| `PDF_TRANSLATE_WEB_HOST` | 默认 `0.0.0.0` |
| `PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD` | **仅在库中无任何用户时**用于创建首个管理员 |
| `PDF_TRANSLATE_ADMIN_USERNAME` | 可选，保留管理员用户名（默认见代码/文档） |
| `PDF_TRANSLATE_PARALLEL_WORKERS` | 并联模式下每批线程数上限相关默认（见服务端逻辑） |

更多 HTTP 超时、重试、各厂商 API 变量见 **SETUP_MANUAL.md**。

---

## Web 使用说明

- **登录**：普通用户与管理员共用登录页；管理员可进入 `/admin`。  
- **用户**：上传 PDF，选择每块页数、重叠、翻译后端（若管理员允许）、**串联 / 并联**、并联并发数；可**终止**进行中的任务；完成后查看**总用时**，下载 **Markdown** 或 **zip 资料包**。  
- **管理员**：配置各后端 API Key（可与环境变量叠加/覆盖规则见设置逻辑）、启用后端列表、是否开放注册；查看审计与任务列表；下载任务产物。  
- **断点续译**：同一任务目录下保留 `output/state.json` 与块文件即可在再次运行时跳过已完成块（具体以后端任务是否复用同一工作目录为准；Web 每次上传为新任务目录）。

---

## 命令行（CLI）

```bash
pdf-translate init <工作目录>
pdf-translate split <输入.pdf> <工作目录> [--tail-fallback]
pdf-translate translate <工作目录> [-b openai] [--pages 3] [--overlap 1] [--max-chunks N]
pdf-translate links <工作目录>
pdf-translate run <输入.pdf> <工作目录>   # init + split + translate 一键
```

CLI 的 `translate` / `run` 默认使用 **串联** 与当前 `pipeline` 默认参数；**并联、Web 专属选项**以 Web 表单为准。详见 `pdf-translate translate --help`。

---

## 流水线与产物（单任务目录）

1. **拆分**：`split/main.pdf`、`split/references.pdf`（若有）、`split/manifest.json`  
2. **分块翻译**：`output/chunks/c0000.md` …（含 YAML 元数据 + 译文正文）  
3. **状态**：`output/state.json`、`output/chunks_manifest.json`、`output/run_log.jsonl`  
4. **合并稿**：`output/translated_full.md`（合并时默认去掉各块 YAML 头，并对顺延标识符做替换）  
5. **记忆**：`memory/glossary.json`、`chunk_summaries.json`、`style_notes.yaml` 等  
6. **链接**：`output/links_index.csv`  

设计背景与记忆字段含义见 **[PROJECT_DESIGN.md](./PROJECT_DESIGN.md)**。

---

## 限制与提示

- 翻译质量与费用取决于所选模型与分块策略；**长块**首次请求可能较慢，服务端已对常见 HTTP 失败做退避重试。  
- **段尾顺延**依赖模型严格按协议输出标识符与英文尾段；**DeepL / echo / hybrid** 路径不会启用该协议。  
- 大文件请注意上传体积上限（Web 端当前策略见 `routes_web` 实现）。  
- 生产部署请修改默认密码、限制 CORS、并考虑 HTTPS 与防火墙策略。

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [SETUP_MANUAL.md](./SETUP_MANUAL.md) | 安装步骤、前端构建、环境变量全表、DeepSeek / 代理等 |
| [PROJECT_DESIGN.md](./PROJECT_DESIGN.md) | 需求与架构、memory 设计、工作计划 |

---

## 许可证

若未单独提供许可证文件，默认以项目所有者声明为准；使用第三方 API 时请遵守各服务商条款。
