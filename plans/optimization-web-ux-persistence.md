# Web 交互、任务持久化与后续深化

## 已落实（本轮）

### 1. 翻译后端：由自由填写改为固定下拉

- **动机**：用户未读说明时易填 `openai` 等未接入的后端，导致报错或困惑。
- **实现**：
  - 用户端仅展示管理端 **已启用** 的后端；首项为「使用服务器默认」。
  - API `GET /api/user/backends` 增加 `labels`（中文说明），与 `enabled`、`default_backend` 一致。
- **相关文件**：`frontend/src/views/UserView.vue`，`pdf_translate/server/routes_web.py`（`BACKEND_UI_LABELS`）。

### 2.「我的任务」刷新后丢失 — 根因与修复

- **根因**：`jobs_meta` 表**没有** `id` 列，但 `list_jobs_for_user` / `list_all_jobs` 使用了 `ORDER BY id DESC`，SQLite 报错，接口失败；前端 `loadMyJobs` 在 `!r.ok` 时静默返回，表现为「无记录」；管理端任务列表同样受影响。
- **修复**：改为 `ORDER BY created_at DESC`（`pdf_translate/server/database.py`）。
- **体验增强**：
  - 任务列表加载失败时展示明确错误文案，便于排查。
  - 「我的任务」每行增加 **「查看状态 / 下载」**，可在刷新后重新拉取 `/api/jobs/{id}` 并恢复「当前任务」卡片（进行中则自动轮询）。

### 3. 数据实际保存在哪

- **元数据**：`app.db` 的 `jobs_meta`（任务 id、用户、文件名、创建时间）— 持久化，与服务重启无关。
- **运行状态与产物**：`PDF_TRANSLATE_WEB_DATA`（默认 `<DATA>/web_jobs/<job_id>/`）下的文件与 `web_status.json`；进程内 `JobRegistry` 会在启动时 `hydrate_from_disk` 尽量恢复内存中的任务对象。

---

## 可选后续：约 24 小时保留策略

**需求理解**：限制磁盘占用、合规「短期留存」，而非长期网盘。

| 方案 | 做法 | 优点 | 注意 |
|------|------|------|------|
| **A. 本地 TTL（推荐优先评估）** | 定时任务或每次创建任务时清理：`created_at` 早于 `now - 24h` 的 `jobs_meta` 行 + 对应 `web_jobs/<id>/` 目录 | 简单、无外部依赖 | 需明确是否同时删审计里关联的 `job_id`；与用户「刚想下载」竞态可用「宽限期」或仅删目录保留元数据一行提示已过期 |
| **B. 仅清理工作目录** | DB 保留记录更久，只删大文件 | 列表可查历史，省空间 | 用户点击下载需友好提示已过期 |
| **C. 发布到 GitHub 等** | 将 zip/md 推到私有仓库 | 异地备份 | 成本高：OAuth、LFS、大文件与论文版权/隐私；维护复杂，**一般不优先于 A** |

环境变量草案：`PDF_TRANSLATE_JOB_RETENTION_HOURS`（默认空表示不自动删）。

---

## 管理端与其它 UX 深化（ backlog ）

- **默认后端与启用列表联动校验**：保存设置时若 `default_backend` 不在 `enabled_backends` 内则警告或自动修正。
- **任务列表展示状态**：在 `jobs_meta` 增加 `last_status` / `updated_at`（任务线程在状态变更时写回 DB），管理端与用户端列表可显示「进行中 / 完成 / 失败」而无需逐个查磁盘。
- **WebSocket 或 SSE**：替代轮询进度，减轻长文翻译时的请求频率。
- **上传前校验**：前端根据 `Accept` 与大小提示，与后端 120MB 一致。
- **国际化**：若存在海外用户，可将 `labels` 与错误信息做成可切换语言。

---

## 实现记录

- 2025-03：`ORDER BY created_at` 修复列表；用户端后端下拉与 `labels`；我的任务「查看状态 / 下载」。
