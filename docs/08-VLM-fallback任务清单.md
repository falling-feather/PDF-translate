# VLM fallback 任务清单

> 文档定位：记录 OCR 后置视觉复核任务清单的设计、验收方式和后续规划。  
> 最后更新：2026-07-06  
> 关联产物：`output/vlm_tasks.json`、`output/experiment_metrics.json`、ZIP 资料包。

## 当前状态

本轮已新增 `vlm_tasks.json` 产物。它不直接调用外部 VLM，也不修改正式译文，而是在本地 OCR 执行、OCR 回写和 OCR 候选 QA 之后，把确实需要视觉复核的任务物化出来。

这一步解决的是“路由阶段只是预估需要 VLM，但 OCR 之后没有明确待办”的问题。现在系统可以区分：

- `ocr_tasks.json`：路由期调度任务，说明哪些页面或区域需要 OCR。
- `vlm_tasks.json`：OCR 后置复核任务，说明哪些 OCR 任务失败、低置信、缺结果或结构门禁异常，需要交给 VLM 或人工复核。

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

真实接入 VLM 时，结果仍应回写成 `ocr-results-v1`，并使用原始 `source_ocr_task_id`，这样可以复用现有 OCR 回写、候选 QA 和晋级链路。

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
```

相比“文字少就调用图像模型”，这个流程能说明系统根据 OCR 结果质量、结构门禁和视觉证据来决定是否升级处理，更适合写成算法层级的创新点。

## 下一步规划

- 接入真实 VLM executor，但保持默认不自动调用外部模型。
- 支持把 VLM 返回结果转换为 `ocr-results-v1`。
- 增加人工复核界面，允许用户查看 `vlm_tasks.json` 中的页面预览、bbox 和建议输出字段。
- 用真实扫描论文、图片型表格和公式密集样本校准触发阈值。
- 将 VLM fallback 成功率、误触发率和结构修复率纳入批量实验报告。
