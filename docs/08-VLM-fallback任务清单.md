# VLM fallback 任务清单

> 文档定位：记录 OCR 后置视觉复核任务清单的设计、验收方式和后续规划。  
> 最后更新：2026-07-06  
> 关联产物：`output/vlm_tasks.json`、`output/vlm_review.json`、`output/vlm_review.md`、`output/vlm_results.json`、`output/vlm_apply.json`、`output/vlm_apply.md`、`output/vlm_retranslation_plan.json`、`output/vlm_retranslation_plan.md`、`output/experiment_metrics.json`、Web 任务产物摘要与下载入口、ZIP 资料包。

## 当前状态

当前已新增 `vlm_tasks.json` 产物，并已接入 Web 任务状态、用户端下载入口和管理端 artifact 下载入口。它不直接调用外部 VLM，也不修改正式译文，而是在本地 OCR 执行、OCR 回写和 OCR 候选 QA 之后，把确实需要视觉复核的任务物化出来。用户完成逐项复核后，可显式应用 `vlm_results.json`，刷新 OCR 回写、候选 QA 和晋级 IR，并生成 `vlm_apply.json/md` 作为审计报告，同时生成 `vlm_retranslation_plan.json/md` 标出受影响 chunk。

这一步解决的是“路由阶段只是预估需要 VLM，但 OCR 之后没有明确待办”的问题。现在系统可以区分：

- `ocr_tasks.json`：路由期调度任务，说明哪些页面或区域需要 OCR。
- `vlm_tasks.json`：OCR 后置复核任务，说明哪些 OCR 任务失败、低置信、缺结果或结构门禁异常，需要交给 VLM 或人工复核。
- `vlm_apply.json/md`：复核结果应用报告，说明 VLM 结果替换或追加了哪些 OCR 结果，以及后续回写、QA、晋级是否成功。
- `vlm_retranslation_plan.json/md`：应用后影响分块计划，说明哪些已翻译 chunk 因结构事实变化建议候选重译，且不覆盖既有译文。

Web 工作台现在会在任务产物摘要中显示 `VLM复核`，用户可通过 `/api/jobs/{job_id}/download/vlm-tasks.json` 下载，管理员可通过 `kind=vlm_fallback_tasks` 下载同一份 JSON。任务完成审计详情也会记录 `vlm_fallback_tasks` 路径，方便后续整理专利证据链。

## 触发条件

当前会生成 VLM fallback 任务的情况包括：

- OCR 结果缺失，且原任务仍有视觉证据。
- OCR 回写被拒绝，例如 `low_confidence`、`empty_text`、`result_not_succeeded`。
- OCR 候选 QA 为 `needs_review` 或 `blocked`。
- 表格、公式等结构化门禁发现异常，例如缺少单元格 bbox、缺少 locked token、公式编号缺失。

系统会排除普通文本段落的低置信任务，避免把所有普通 warning 都推给 VLM。

## 任务内容

每条 `vlm_tasks.json` 任务会保留：

- 来源 OCR 任务 ID：`source_ocr_task_id`
- 页码、block id、block type、bbox
- 输入图片、页面预览图、裁剪尺寸
- 触发原因：`trigger_reasons`
- 来源阶段：`source_stages`
- 建议输出字段：`expected_outputs`
- 复核目标：`review_goals`
- 表格/公式上下文和结构契约
- fallback 策略说明

真实接入 VLM 时，结果仍应回写成 `ocr-results-v1`，并使用原始 `source_ocr_task_id`，这样可以复用现有 OCR 回写、候选 QA 和晋级链路。`manual_vlm_review` 与 `source_vlm_review_id:*` 这类字段是溯源标记，属于 trace-only warning，不会因为审计标记本身阻断候选晋级。

## 指标与验收

`experiment_metrics.json` 已新增后置 fallback 指标：

- `vlm_fallback_after_ocr_gate_count`
- `vlm_fallback_ready_task_count`
- `vlm_fallback_blocked_task_count`
- `vlm_fallback_ocr_low_confidence_task_count`
- `vlm_fallback_ocr_missing_result_task_count`
- `vlm_fallback_candidate_gate_task_count`
- `vlm_fallback_structured_gate_task_count`

同时新增 breakdown：

- `vlm_fallback_reason_counts`
- `vlm_fallback_source_stage_counts`
- `vlm_fallback_review_goal_counts`
- `vlm_fallback_expected_output_counts`

验收时重点看三件事：

1. OCR 后确实产生 `output/vlm_tasks.json`。
2. 低置信、空文本、缺失结果、结构门禁异常能被归入明确原因。
3. 普通文本 warning 不会被误判为 VLM fallback 任务。
4. Web 用户端和管理端都能看到 `VLM复核` 产物，并能下载 `vlm_tasks.json`。

## 专利价值

这项能力适合作为“OCR/VLM 混合路由闭环”的证据点。它的价值不在于简单调用视觉模型，而在于形成了一个可审计流程：

```text
页面/区域路由
  -> 本地 OCR 或结构化 OCR
  -> OCR 回写门禁
  -> OCR 候选结构 QA
  -> VLM fallback 任务物化
  -> 后续 VLM/人工复核
  -> 复用 OCR 回写与晋级链路
  -> 受影响 chunk 重译计划
```

相比“文字少就调用图像模型”，这个流程能说明系统根据 OCR 结果质量、结构门禁和视觉证据来决定是否升级处理，更适合写成算法层级的创新点。

## 下一步规划

- 接入真实 VLM executor，但保持默认不自动调用外部模型。
- 已支持 Web 逐项人工复核：用户端和管理端可读取 `vlm_tasks.json` 生成 `vlm_review.json/md`，在弹窗中查看页码、bbox、裁剪图路径、触发原因和建议输出字段，填写识别文本、置信度、语言和备注，并按单条写回 `accept_result`、`mark_unusable`、`needs_revision` 或 `clear`。
- 已支持把人工/VLM 复核结果转换为 `ocr-results-v1`：接受单条结果后会生成 `output/vlm_results.json`，其中 `task_id` 使用原始 `source_ocr_task_id`，可继续复用现有 OCR 回写、候选 QA 和晋级链路。
- 已支持显式应用复核结果：用户端和管理端可执行“应用VLM”，系统会把 VLM 结果按 `task_id` 替换同一 OCR 任务的旧结果，保留其他 OCR 结果，并重跑 `ocr_writeback.json`、`ocr_candidate_qa.*`、`ocr_candidate_promotion.*` 和 `document_ir_promoted.json`。
- 已支持应用后重译计划：系统会读取 active `chunks_manifest.json`，优先按 block id、其次按页码范围定位受影响 chunk，写出 `vlm_retranslation_plan.json/md`；用户端和管理端都可下载，ZIP 资料包会归档为 `质量/VLM重译计划.*`。
- 批量复核已支持 `mark_unusable`、`needs_revision` 和 `clear`；批量接受仍要求逐项填写识别文本，避免无文本结果进入回写链路。
- 后续接入真实 VLM executor 时，应复用同一份 `vlm_review.json` / `vlm_results.json` 契约，而不是绕过现有 OCR 门禁。
- 用真实扫描论文、图片型表格和公式密集样本校准触发阈值。
- 将 VLM fallback 成功率、误触发率和结构修复率纳入批量实验报告。

## 本轮完成：Web 复核与回写结果

本轮已把 VLM fallback 从“只能下载任务清单”推进到“可逐项复核并生成回写结果”的闭环：

- 新增 `output/vlm_review.json` 与 `output/vlm_review.md`，记录每个 VLM/人工复核项的人工决策、识别文本、置信度、语言、备注、审核人、审核时间和有效状态。
- 新增 `output/vlm_results.json`，把已接受的复核文本整理为 `ocr-results-v1`，并保留 `manual_vlm_review`、来源复核 ID、结构化表格/公式字段等审计信息。
- 新增用户端下载 `/api/jobs/{job_id}/download/vlm-review.md`、`/api/jobs/{job_id}/download/vlm-results.json`，管理端 artifact 新增 `vlm_fallback_review` 和 `vlm_fallback_results`。
- 新增 Web 复核弹窗，支持筛选待审/需复查/已接受/不可用/缺视觉证据项，按类型和关键词检索，单条接受并生成回写结果，或批量标记需复查/不可用/清空。
- 任务状态摘要新增 `vlm_fallback_review_*` 和 `vlm_fallback_results_*` 字段；当仍有待复核项时，`artifact_warnings` 会出现 `vlm_fallback_review_required_items`。

## 本轮完成：VLM 结果应用

本轮已把 `vlm_results.json` 从“可下载回写结果”推进到“可显式应用到 OCR 链路”的状态：

- 新增 `pdf_translate.vision.vlm_apply`，负责合并原 OCR 结果与 VLM 复核结果，VLM 结果按 `task_id` 优先覆盖旧结果，其他 OCR 结果保留。
- 新增 `POST /api/jobs/{job_id}/vlm-fallback-results/apply`，应用后刷新 `ocr_results.json`、`ocr_writeback.json`、`document_ir_ocr.json`、`ocr_candidate_qa.*`、`ocr_candidate_promotion.*` 和 `document_ir_promoted.json`。
- 新增 `output/vlm_apply.json` 与 `output/vlm_apply.md`，记录 VLM 结果数、替换数、追加数、回写接受数、QA 候选数、晋级候选数和 canonical 结构晋级数。
- 新增 `output/vlm_retranslation_plan.json` 与 `output/vlm_retranslation_plan.md`，按 block id 或页码范围定位受影响 chunk，记录建议重译数量、未映射任务和映射方式。
- 用户端和管理端新增“应用VLM”“VLM应用报告”与“VLM重译计划”入口；管理端 artifact 新增 `vlm_fallback_apply` 和 `vlm_retranslation_plan`，ZIP 资料包新增 VLM 复核报告、回写结果、应用报告和重译计划的中文归档名。
- 该动作只刷新结构侧派生产物和重译计划，不自动重翻全文，也不覆盖正式译文；后续可在晋级 IR 与计划基础上继续设计候选重译或正式稿发布策略。
