# 优化计划索引

本目录存放与 **PDF 学术翻译工具** 相关的讨论整理与后续可实施方案，供迭代时对照。

## 阅读顺序建议

| 文档 | 内容 |
|------|------|
| [discussed-context.md](./discussed-context.md) | 当前已讨论过的问题、结论与代码现状锚点 |
| [optimization-ocr.md](./optimization-ocr.md) | 扫描版 / 图片型 PDF 与整页 OCR 方案 |
| [optimization-toc-layout.md](./optimization-toc-layout.md) | 目录（ToC）点线与排版错乱治理 |
| [optimization-pipeline-future.md](./optimization-pipeline-future.md) | 其它可行后期优化（未在本次对话细谈但值得排队） |
| [optimization-web-ux-persistence.md](./optimization-web-ux-persistence.md) | Web 后端选择、任务列表持久化、TTL 与管理层深化 |
| [optimization-server-architecture.md](./optimization-server-architecture.md) | 服务端稳定性、状态持久化、安全与性能优化 |
| [optimization-pdf-reflow.md](./optimization-pdf-reflow.md) | 位置一致译文 PDF：bbox 映射与覆盖/重排路线 |
| [roadmap-next-steps.md](./roadmap-next-steps.md) | 按优先级排好的执行路线图（建议从这里开工） |

## 当前状态（2026-03）

- **已完成（已落地）**：
  - 用户页后端选择从自由输入改为下拉；
  - 我的任务列表（`created_at` 排序、本地可读时间格式）；
  - 用户页表单缓存、API 翻译面板、页脚支持弹窗；进入工作台时清空并行任务缓存；
  - 注册页管理员联系提示弹窗；
  - 默认后端 DeepSeek（脚本与默认配置）；
  - **未收藏任务 24h TTL**：进入工作台时清理数据库与 `web_jobs` 目录（活跃任务跳过）；
  - **任务收藏**：每用户最多 3 条，取消收藏后回到主列表并刷新 `created_at`；
  - 管理端审计日志：**中文摘要** + 可展开原始详情。
- **进行中 / 排队（方案已有，待实装）**：
  - OCR 检测与按需 OCR 分支；
  - ToC 与复杂版式规范化；
  - 服务端状态双写 DB、流式下载等深化；
  - 位置一致译文 PDF（bbox overlay / reflow）。

## 与主仓库文档的关系

| 类型 | 位置 |
|------|------|
| 功能说明与试运行步骤 | 根目录 `README.md` |
| 安装与环境变量详解 | `SETUP_MANUAL.md` |
| 架构愿景与 memory 约定 | `PROJECT_DESIGN.md` |
| Windows 部署 | `DEPLOY_WINDOWS_SERVER.md` |

`plans/` 记录 **待做 / 可选** 技术路线与讨论整理；**当前已落地能力**以根目录 `README.md` 的「试运行」「Web 使用说明」为准。
