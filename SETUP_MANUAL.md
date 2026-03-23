# 安装与 API 配置说明

本文档说明如何安装本工具、配置环境变量，以及可选的手动步骤（Ollama、DeepL、OpenAI 兼容接口等）。

---

## 1. 环境要求

- Python **3.10+**
- 网络：使用 `openai` / `deepl` / `hybrid` 时需能访问对应 API；`echo` 与仅 `split` / `links` **不需要**外网。

---

## 2. 安装

在项目根目录（含 `pyproject.toml` 的目录）执行：

```bash
cd "d:\代码玩具测试\pdf translate"
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

安装完成后可使用：

```bash
pdf-translate --help
```

或（无需安装 editable 时）：

```bash
pip install -r requirements.txt
python -m pdf_translate --help
```

### 2.1 Web 工作台（FastAPI + Vue）

1. **构建前端**（需本机安装 [Node.js](https://nodejs.org/) LTS）：

```bash
cd frontend
npm install
npm run build
```

构建产物输出到 `pdf_translate/server/static/`，由后端一并托管。

2. **启动 Web 服务**（在项目根目录、已 `pip install -e .`）：

```bash
pdf-translate-web
```

默认监听 **`http://0.0.0.0:901`**（全网卡）。浏览器访问本机 `http://127.0.0.1:901`；团队同学在同一局域网可使用 `http://<你的内网IP>:901`。请在 Windows 防火墙中放行 **901** 端口（或设置环境变量 `PDF_TRANSLATE_WEB_PORT`）。

**一键启动（Windows）**：

- 若尚未构建前端：先双击 **`build_frontend.bat`**（在 `frontend` 下执行 `npm install` 与 `npm run build`，产物写入 `pdf_translate/server/static/`）。
- 再双击 **`start_web.bat`**：会检查静态页是否存在、设置端口与数据目录（变量见脚本内注释）、`pip install -e .` 后启动。首次启动若 `data\app.db` 不存在，会用 `PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD` 创建首个管理员。

**登录与权限**：所有用户从 `/login` 登录；账号存于本地 SQLite（`data/app.db`）。管理员进入 `/admin` 可配置 API 密钥、启用后端、是否开放注册；**审计日志**以中文摘要展示，可展开查看 JSON 详情；可下载全部任务产物。

**用户工作台**：每次进入翻译工作台会清空本地并行窗口缓存，并调用接口清理**未收藏且超过 24 小时**的历史任务（删库并删除 `web_jobs/<job_id>/`，进行中的任务不删）。「我的任务」支持**收藏**（每用户最多 3 条），取消收藏后任务回到主列表且 `created_at` 更新为取消时刻。

3. **前端开发联调**（可选）：终端 A 运行 `pdf-translate-web`；终端 B 在 `frontend` 下执行 `npm run dev`，Vite 会把 `/api` 代理到 **901**。

4. **数据目录**：每个上传任务的工作目录默认在 **`data/web_jobs/<job_id>/`**（可用环境变量 `PDF_TRANSLATE_WEB_DATA` 修改绝对路径）。

5. **API 文档**：服务启动后访问 **`http://127.0.0.1:901/docs`**（OpenAPI）。

6. **DeepSeek**：在管理后台填写 `deepseek_api_key`（及可选 Base URL / Model），并在「允许的后端」中勾选 `deepseek`；用户端翻译时选择后端 `deepseek` 即可。亦可通过环境变量 `DEEPSEEK_API_KEY` 提供密钥（后台非空值会覆盖环境变量）。

7. **流畅度**：翻译过程中每完成一块会将 **`output/translated_full.md` 增量更新**，用户页在任务未完成时也可 **「下载已译部分」**；对端断连、429/502/503/504 等会自动退避重试（见 `PDF_TRANSLATE_HTTP_RETRIES`）。

---

## 3. 环境变量（`.env` 推荐）

在项目目录或上层目录放置 `.env`（程序通过 `python-dotenv` 自动加载），常用项如下：

| 变量 | 说明 |
|------|------|
| `PDF_TRANSLATE_BACKEND` | 默认后端：`echo`（联调）、`openai`、`ollama`、`deepl`、`hybrid` |
| `OPENAI_API_KEY` | OpenAI 或兼容服务的密钥 |
| `OPENAI_BASE_URL` | 默认 `https://api.openai.com/v1`；国内中转或 Azure 请改为对应 base URL |
| `OPENAI_MODEL` | 默认 `gpt-4o-mini` |
| `OLLAMA_BASE_URL` | 默认 `http://127.0.0.1:11434/v1` |
| `OLLAMA_MODEL` | 例如 `llama3.2`，需与本机 `ollama pull` 的模型名一致 |
| `DEEPL_API_KEY` | DeepL Free/Pro 的 auth key |
| `DEEPL_API_URL` | Free 一般为 `https://api-free.deepl.com/v2/translate`；Pro 为 `https://api.deepl.com/v2/translate` |
| `HTTP_TIMEOUT_S` | HTTP 超时秒数（读超时为主），默认 **`240`**（长块翻译建议 ≥180） |
| `PDF_TRANSLATE_HTTP_RETRIES` | 对 DeepSeek/OpenAI 兼容与 DeepL 的 HTTP 失败自动重试次数，默认 **`4`**（含首次请求，即最多约 4 次尝试） |

**最小可运行（不调用外网）：**

```env
PDF_TRANSLATE_BACKEND=echo
```

**OpenAI 官方：**

```env
PDF_TRANSLATE_BACKEND=openai
OPENAI_API_KEY=sk-...
```

**本地 Ollama（需先安装并启动 Ollama，且 `ollama pull` 过模型）：**

```env
PDF_TRANSLATE_BACKEND=ollama
OLLAMA_MODEL=llama3.2
```

**DeepL 仅机器翻译：**

```env
PDF_TRANSLATE_BACKEND=deepl
DEEPL_API_KEY=你的密钥
```

**Hybrid（DeepL 初稿 + OpenAI 兼容 API 中译校对）：**

```env
PDF_TRANSLATE_BACKEND=hybrid
DEEPL_API_KEY=...
OPENAI_API_KEY=...
```

---

## 4. 手动搭建：Ollama

1. 从 [https://ollama.com](https://ollama.com) 安装 Ollama 并确保服务运行。
2. 终端执行：`ollama pull llama3.2`（或你选用的模型）。
3. 确认浏览器或 `curl http://127.0.0.1:11434` 可访问。
4. `.env` 中设置 `PDF_TRANSLATE_BACKEND=ollama` 与 `OLLAMA_MODEL`。

---

## 5. 手动搭建：DeepL API

1. 在 DeepL 开发者页面申请 API Key（区分 Free / Pro 端点）。
2. 将 `DEEPL_API_URL` 设为与账号类型一致的 v2 translate 地址（见上表）。
3. 设置 `DEEPL_API_KEY`。

---

## 6. CLI 使用顺序（与 PROJECT_DESIGN.md 一致）

```bash
# 1) 初始化工作目录（生成 memory/）
pdf-translate init ./my-paper

# 2) 拆分正文与参考文献 PDF
pdf-translate split ./article.pdf ./my-paper

# 可选：未检测到 References 标题时，用最后约 15% 页作为参考文献
pdf-translate split ./article.pdf ./my-paper --tail-fallback

# 3) 按块翻译（默认后端由 PDF_TRANSLATE_BACKEND 决定）
pdf-translate translate ./my-paper --backend echo

# 4) 导出正文中各页超链接索引 CSV（便于后续 HTML/对照）
pdf-translate links ./my-paper
```

一键：`pdf-translate run ./article.pdf ./my-paper --backend echo`

---

## 7. 工作目录结构说明

**术语表页码**：`memory/glossary.json` 中 `first_page` 表示该术语在 **`split/main.pdf` 中的页码（从 1 开始）**，用于按块注入相关术语。

```
my-paper/
  memory/
    glossary.json
    entities.json
    chunk_summaries.json
    style_notes.yaml
    pending_review.json
    running_summary.md
  split/
    main.pdf
    references.pdf   # 若有
    manifest.json
  output/
    chunks_manifest.json
    chunks/c0000.md, c0001.md, …（每块独立 Markdown）
    translated_full.md
    state.json
    run_log.jsonl
    links_index.csv   # 执行 links 后
```

---

## 8. 常见问题

- **提示缺少 API Key**：将 `PDF_TRANSLATE_BACKEND` 设为 `echo` 做流程验证，或按上表配置密钥。
- **参考文献未拆出**：PDF 中标题可能不是 `References` / `Bibliography` 等，可尝试 `--tail-fallback` 或后续在配置中增加自定义规则（开发中）。
- **Azure OpenAI**：使用兼容 Chat Completions 的 endpoint，将 `OPENAI_BASE_URL` 与 `OPENAI_API_KEY` 设为 Azure 提供的值，并选用部署名作为 `OPENAI_MODEL`。

---

如有新的后端或企业代理规则，可在 `pdf_translate/translators/factory.py` 中扩展 `build_translator`。
