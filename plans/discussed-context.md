# 讨论整理：问题背景与结论

本文档固化我们在对话中达成的一致判断，并指向代码中相关位置，便于实现时快速定位。

## 一、项目在做什么（简要）

- **目标**：英文学术 PDF → 拆分正文与参考文献 → 按页分块 → 多后端翻译（含 DeepSeek 等 OpenAI 兼容 API）→ 合并 Markdown，并配合本地 `memory/` 术语与摘要。
- **入口**：CLI（`pdf_translate/cli.py`）与 Web（FastAPI + Vue，构建产物在 `pdf_translate/server/static/`）。

## 二、已识别的两个主要问题

### 2.1 图片型 / 扫描型 PDF（无文字层）

- **现象**：论文以插入图片方式录入文字，或扫描件，PDF 内无可选中文本层。
- **根因**：当前正文抽取使用 PyMuPDF `page.get_text("text")`，见 `pdf_translate/pipeline.py` 中 `_page_rows_for_main`；仅读文字层，无 OCR。
- **结论**：在保持现有流水线（分块 → `TranslationRequest` → 翻译器）的前提下，**无法仅靠调 prompt 解决**；需要 **检测** 与可选 **OCR 文本源**（或人工预处理 PDF）。

### 2.2 目录（ToC）等特殊版式翻译后排版错乱

- **现象**：目录引导点（leader dots）变成多行纯点；章节号、标题、页码被拆成多行，与纸质目录「一行一条」不一致。
- **根因（两层）**：
  1. **抽取**：多个独立 text block，`get_text("text")` 的行序/块序未必等于视觉上一行。
  2. **翻译**：模型可能照抄或扩写点线；输入已碎时输出更碎。
- **已有缓解**：`pdf_translate/text_sanitize.py` 的 `collapse_toc_dot_leaders` 在送入模型前删除「整行几乎全是点」且长度 ≥ 24 的行；在 `pipeline.py` 的翻译路径中已调用。
- **结论**：**仍有优化空间**；需在「删纯点行」之外增加 **目录页启发式、行内点线折叠、可选基于 bbox 的同行合并**，并考虑 **译后轻量清洗**。

## 三、整页 OCR 与 DeepSeek 方案的关系（结论摘要）

- **API / 协议**：DeepSeek 走 OpenAI 兼容 `chat/completions`（`pdf_translate/translators/openai_compatible.py`），改为 OCR 供给正文 **不要求改接口形态**。
- **实际影响**：
  - 扫描版：从「几乎不调 API」变为「正常调用量」，费用与时间 **预期显著上升**。
  - 若对已有文字层 PDF **一律 OCR**：输入往往更噪、更长，**输入 token 与费用** 易增；公式与双栏 **错误阅读顺序** 会拖累译文质量。
- **推荐策略**：**默认文字层**；仅低文字密度（或用户勾选）时走 OCR。详见 [optimization-ocr.md](./optimization-ocr.md)。

## 四、代码锚点速查

| 主题 | 文件与符号 |
|------|------------|
| 按页取文 | `pipeline.py` → `_page_rows_for_main` |
| 分块 | `chunking.py` → `build_text_chunks` |
| 目录点线（当前） | `text_sanitize.py` → `collapse_toc_dot_leaders` |
| 翻译入口（含 DeepSeek） | `translators/openai_compatible.py` |
| 参考文献拆分 | `pdf_structure.py` |

## 五、补充（Web 任务列表）

- 用户「我的任务」与管理端任务列表依赖 `jobs_meta`；若列表始终为空，需确认查询使用合法排序列（曾用不存在的 `id` 会导致接口失败，见 `optimization-web-ux-persistence.md`）。
