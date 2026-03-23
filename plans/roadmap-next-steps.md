# 执行路线图（Next Steps）

本路线图把现有 plans 转成可执行顺序，按“用户感知收益 / 实现成本 / 技术风险”综合排序。

## P0：两周内可落地（先做）

### 1) OCR 检测与提示（不先做整页 OCR）

- 目标：先解决“扫描 PDF 看起来像失败”的认知问题。
- 输出：
  - 上传或任务启动时给出“疑似扫描件”的明确提示；
  - CLI 同步给出警告或退出码。
- 依赖：`optimization-ocr.md` 阶段 A。

### 2) ToC 文本规范化（行内点线折叠 + 目录页启发式）

- 目标：优先降低目录排版错乱。
- 输出：
  - `sanitize_page_text_for_translation()`（或等效统一入口）；
  - 加 2~3 个回归样例（目录页、正文页、误伤防护）。
- 依赖：`optimization-toc-layout.md` 阶段 A。

### 3) 服务端状态可见性提升（最小 DB 扩展）

- 目标：让用户/管理端列表直接看到任务状态，而非仅靠内存/磁盘。
- 输出：
  - 新表（如 `jobs_state`）；
  - `JobRegistry.update()` 同步写状态（可节流）；
  - 列表接口返回 `status/phase/updated_at`。
- 依赖：`optimization-server-architecture.md` P0-1。

## P1：一到两个月（中期）

### 4) TTL 清理器（24h 可配置）

- 目标：控制磁盘占用，符合短期留存预期。
- 输出：
  - 环境变量 `PDF_TRANSLATE_JOB_RETENTION_HOURS`；
  - 定时清理 + 审计日志；
  - 下载时“已过期”友好提示。
- 依赖：`optimization-web-ux-persistence.md` 与 `optimization-server-architecture.md`。

### 5) Zip 流式下载

- 目标：降低大任务下载内存峰值。
- 输出：
  - 下载接口改 `StreamingResponse`；
  - 压测并发下载场景。
- 依赖：`optimization-server-architecture.md` P0-2。

### 6) JWT 过期控制

- 目标：提升部署安全性。
- 输出：
  - token 增加 `exp/iat`；
  - 前端处理登录过期（跳转登录页 + 友好提示）。
- 依赖：`optimization-server-architecture.md` P0-3。

## P2：探索项（并行 PoC）

### 7) 位置一致译文 PDF（bbox overlay）

- 目标：在文字层正常 PDF 上生成“位置高一致”的译文 PDF。
- 输出：
  - M0：单栏页面 bbox line 映射 + 覆盖绘制；
  - M1：自动换行与字号缩放策略；
  - M2：目录页专项策略。
- 依赖：`optimization-pdf-reflow.md`。

---

## 暂缓项（不建议优先）

- 像素级完全一致的 PDF 重建；
- 全量 OCR 默认开启（会显著增加延迟与成本）；
- 过早引入 MySQL 迁移（当前瓶颈主要不在 DB）。

