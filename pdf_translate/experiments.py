from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pdf_translate import pipeline
from pdf_translate.config import AppConfig

SCHEMA_VERSION = "batch-experiment-v1"

SUMMARY_FIELDS: dict[str, list[str]] = {
    "quality": [
        "page_count",
        "chunk_count",
        "table_count",
        "translation_issue_count",
        "table_shape_error_count",
        "table_cell_token_error_count",
        "missing_table_locked_token_count",
        "page_boundary_fragment_count",
        "split_boundary_count",
        "protected_boundary_count",
        "ocr_task_count",
        "ocr_candidate_promoted_count",
        "repair_request_count",
        "repair_merge_applied_count",
        "post_repair_issue_count",
    ],
    "rates": [
        "table_reconstruction_ready_rate",
        "table_cell_token_error_rate",
        "table_locked_token_missing_rate",
        "split_boundary_rate",
        "protected_boundary_rate",
        "active_split_reduction_rate_vs_baseline",
        "entity_missing_rate",
        "routed_page_rate",
        "ocr_ready_task_rate",
        "ocr_candidate_promotion_rate",
        "qa_issue_per_chunk",
        "repair_merge_apply_rate",
        "post_repair_issue_reduction_rate",
    ],
    "performance": [
        "total_elapsed_ms",
        "translation_elapsed_ms",
        "translation_request_count",
        "http_attempt_count",
        "http_retry_count",
        "estimated_request_token_count",
        "estimated_total_token_count",
        "estimated_total_cost",
        "estimated_cost_per_chunk",
    ],
}

COMPARISON_FIELDS = [
    ("quality", "translation_issue_count"),
    ("quality", "table_shape_error_count"),
    ("quality", "table_cell_token_error_count"),
    ("rates", "split_boundary_rate"),
    ("rates", "protected_boundary_rate"),
    ("rates", "active_split_reduction_rate_vs_baseline"),
    ("performance", "total_elapsed_ms"),
    ("performance", "translation_request_count"),
    ("performance", "estimated_total_cost"),
]


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    chunk_strategy: Literal["page", "structure"]
    execute_ocr: bool = False
    execute_repair_requests: bool = False


def _safe_id(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "-", value.strip())
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    return text or "item"


def parse_variant_spec(spec: str) -> ExperimentVariant:
    raw = spec.strip().lower().replace("_", "-")
    if not raw:
        raise ValueError("empty experiment variant")
    parts = [part for part in raw.split("+") if part]
    base = parts[0]
    if base not in ("page", "structure"):
        raise ValueError(f"unknown experiment variant base: {base}")

    flags = set(parts[1:])
    known_flags = {"ocr", "execute-ocr", "repair", "repairs", "execute-repairs"}
    unknown_flags = flags - known_flags
    if unknown_flags:
        raise ValueError(f"unknown experiment variant flags: {', '.join(sorted(unknown_flags))}")

    execute_ocr = bool(flags & {"ocr", "execute-ocr"})
    execute_repair_requests = bool(flags & {"repair", "repairs", "execute-repairs"})
    name_parts = [base]
    if execute_ocr:
        name_parts.append("ocr")
    if execute_repair_requests:
        name_parts.append("repair")
    return ExperimentVariant(
        name="+".join(name_parts),
        chunk_strategy=base,  # type: ignore[arg-type]
        execute_ocr=execute_ocr,
        execute_repair_requests=execute_repair_requests,
    )


def parse_variant_specs(specs: str | list[str] | tuple[str, ...]) -> list[ExperimentVariant]:
    if isinstance(specs, str):
        raw_specs = [part.strip() for part in specs.split(",")]
    else:
        raw_specs = []
        for item in specs:
            raw_specs.extend(part.strip() for part in item.split(","))

    variants: list[ExperimentVariant] = []
    seen: set[str] = set()
    for raw in raw_specs:
        if not raw:
            continue
        variant = parse_variant_spec(raw)
        if variant.name not in seen:
            variants.append(variant)
            seen.add(variant.name)
    if not variants:
        raise ValueError("at least one experiment variant is required")
    return variants


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_subset(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for group, fields in SUMMARY_FIELDS.items():
        raw_group = metrics.get(group)
        if not isinstance(raw_group, dict):
            raw_group = {}
        summary[group] = {field: raw_group.get(field, 0) for field in fields}
    return summary


def _metric_value(record: dict[str, Any], group: str, field: str) -> Any:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    raw_group = metrics.get(group)
    if not isinstance(raw_group, dict):
        return None
    return raw_group.get(field)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _aggregate_records(records: list[dict[str, Any]], variants: list[ExperimentVariant]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for variant in variants:
        variant_records = [record for record in records if record.get("variant") == variant.name]
        succeeded = [record for record in variant_records if record.get("status") == "succeeded"]
        averages: dict[str, dict[str, float]] = {}
        for group, fields in SUMMARY_FIELDS.items():
            averages[group] = {}
            for field in fields:
                values = [
                    number
                    for record in succeeded
                    if (number := _numeric(_metric_value(record, group, field))) is not None
                ]
                averages[group][field] = _mean(values)
        aggregates.append(
            {
                "variant": variant.name,
                "chunk_strategy": variant.chunk_strategy,
                "execute_ocr": variant.execute_ocr,
                "execute_repair_requests": variant.execute_repair_requests,
                "run_count": len(variant_records),
                "succeeded_count": len(succeeded),
                "failed_count": len(variant_records) - len(succeeded),
                "averages": averages,
            }
        )
    return aggregates


def _compare_to_baseline(records: list[dict[str, Any]], baseline_variant: str) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_sample.setdefault(str(record.get("sample_id", "")), []).append(record)

    for sample_id, sample_records in sorted(by_sample.items()):
        baseline = next(
            (
                record
                for record in sample_records
                if record.get("variant") == baseline_variant and record.get("status") == "succeeded"
            ),
            None,
        )
        if baseline is None:
            continue
        for record in sample_records:
            if record.get("variant") == baseline_variant or record.get("status") != "succeeded":
                continue
            deltas: dict[str, float] = {}
            for group, field in COMPARISON_FIELDS:
                before = _numeric(_metric_value(baseline, group, field))
                after = _numeric(_metric_value(record, group, field))
                if before is None or after is None:
                    continue
                deltas[f"{group}.{field}"] = round(after - before, 4)
            comparisons.append(
                {
                    "sample_id": sample_id,
                    "baseline_variant": baseline_variant,
                    "variant": record.get("variant"),
                    "deltas": deltas,
                }
            )
    return comparisons


def _record_paths(work_dir: Path, output_dir: Path) -> dict[str, str]:
    files = {
        "experiment_metrics": work_dir / "output" / "experiment_metrics.json",
        "run_metrics": work_dir / "output" / "run_metrics.json",
        "cost_estimate": work_dir / "output" / "cost_estimate.json",
        "translated_full": work_dir / "output" / "translated_full.md",
        "bilingual_html": work_dir / "output" / "bilingual.html",
    }
    result: dict[str, str] = {}
    for key, path in files.items():
        try:
            result[key] = str(path.relative_to(output_dir)).replace("\\", "/")
        except ValueError:
            result[key] = str(path)
    return result


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def write_batch_experiment_markdown(report: dict[str, Any], path: Path) -> Path:
    lines = [
        "# 批量实验汇总",
        "",
        f"- 生成时间：{report.get('created_at')}",
        f"- 样本数：{report.get('sample_count')}",
        f"- 运行数：{report.get('run_count')}",
        f"- 成功数：{report.get('succeeded_count')}",
        f"- 失败数：{report.get('failed_count')}",
        "",
        "## 策略均值",
        "",
        "| 策略 | 成功/总数 | 平均 issue | 平均边界切开率 | 平均边界保护率 | 平均耗时 ms | 平均估算成本 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report.get("aggregates", []):
        averages = item.get("averages", {})
        quality = averages.get("quality", {})
        rates = averages.get("rates", {})
        performance = averages.get("performance", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("variant", "")),
                    f"{item.get('succeeded_count', 0)}/{item.get('run_count', 0)}",
                    _format_number(quality.get("translation_issue_count", 0)),
                    _format_number(rates.get("split_boundary_rate", 0)),
                    _format_number(rates.get("protected_boundary_rate", 0)),
                    _format_number(performance.get("total_elapsed_ms", 0)),
                    _format_number(performance.get("estimated_total_cost", 0)),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 单次运行",
            "",
            "| 样本 | 策略 | 状态 | 工作目录 | 指标文件 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for record in report.get("records", []):
        files = record.get("files", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.get("sample_id", "")),
                    str(record.get("variant", "")),
                    str(record.get("status", "")),
                    str(record.get("work_dir", "")),
                    str(files.get("experiment_metrics", "")),
                ]
            )
            + " |"
        )

    if report.get("comparisons"):
        lines.extend(
            [
                "",
                "## 相对基线差值",
                "",
                "差值为“当前策略 - 基线策略”。负数通常表示 issue、耗时或成本下降；正数对保护率和降幅类指标通常更好。",
                "",
                "| 样本 | 基线 | 策略 | issue 差值 | 边界切开率差值 | 边界保护率差值 | 耗时差值 ms |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in report["comparisons"]:
            deltas = item.get("deltas", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("sample_id", "")),
                        str(item.get("baseline_variant", "")),
                        str(item.get("variant", "")),
                        _format_number(deltas.get("quality.translation_issue_count", "")),
                        _format_number(deltas.get("rates.split_boundary_rate", "")),
                        _format_number(deltas.get("rates.protected_boundary_rate", "")),
                        _format_number(deltas.get("performance.total_elapsed_ms", "")),
                    ]
                )
                + " |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_batch_experiment(
    pdfs: list[Path],
    output_dir: Path,
    cfg: AppConfig,
    *,
    variants: list[ExperimentVariant] | None = None,
    backend: str | None = None,
    pages_per_chunk: int = 3,
    overlap_pages: int = 1,
    max_chunks: int | None = None,
    tail_fallback: bool = False,
    translate_mode: Literal["serial", "parallel"] = "serial",
    parallel_workers: int = 4,
    resume: bool = False,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    if not pdfs:
        raise ValueError("at least one PDF is required")
    variants = variants or parse_variant_specs("page,structure")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend or cfg.default_translator,
        "pages_per_chunk": pages_per_chunk,
        "overlap_pages": overlap_pages,
        "max_chunks": max_chunks,
        "tail_fallback": tail_fallback,
        "translate_mode": translate_mode,
        "parallel_workers": parallel_workers,
        "variants": [variant.__dict__ for variant in variants],
        "samples": [
            {
                "sample_id": f"{index:03d}-{_safe_id(pdf.stem)}",
                "source_pdf": str(pdf.resolve()),
            }
            for index, pdf in enumerate(pdfs, start=1)
        ],
    }
    (output_dir / "batch_experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    for index, pdf in enumerate(pdfs, start=1):
        sample_id = f"{index:03d}-{_safe_id(pdf.stem)}"
        for variant in variants:
            work_dir = output_dir / "runs" / sample_id / _safe_id(variant.name)
            record: dict[str, Any] = {
                "sample_id": sample_id,
                "source_pdf": str(pdf.resolve()),
                "variant": variant.name,
                "chunk_strategy": variant.chunk_strategy,
                "execute_ocr": variant.execute_ocr,
                "execute_repair_requests": variant.execute_repair_requests,
                "work_dir": str(work_dir.relative_to(output_dir)).replace("\\", "/"),
                "status": "pending",
            }
            try:
                pipeline.init_workdir(work_dir)
                pipeline.run_split(pdf, work_dir, use_tail_if_no_heading=tail_fallback)
                pipeline.run_translate(
                    work_dir,
                    cfg,
                    backend=backend,
                    pages_per_chunk=pages_per_chunk,
                    overlap_pages=overlap_pages,
                    resume=resume,
                    max_chunks=max_chunks,
                    translate_mode=translate_mode,
                    parallel_workers=parallel_workers,
                    chunk_strategy=variant.chunk_strategy,
                    execute_repair_requests=variant.execute_repair_requests,
                    execute_ocr=variant.execute_ocr,
                )
                metrics_path = work_dir / "output" / "experiment_metrics.json"
                metrics = _read_json(metrics_path)
                record["status"] = "succeeded"
                record["metrics"] = _summary_subset(metrics)
                record["files"] = _record_paths(work_dir, output_dir)
            except Exception as exc:  # pragma: no cover - exercised through integration failures
                record["status"] = "failed"
                record["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                }
                records.append(record)
                if stop_on_error:
                    raise
                continue
            records.append(record)

    aggregates = _aggregate_records(records, variants)
    baseline_variant = variants[0].name
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend or cfg.default_translator,
        "sample_count": len(pdfs),
        "variant_count": len(variants),
        "run_count": len(records),
        "succeeded_count": sum(1 for record in records if record.get("status") == "succeeded"),
        "failed_count": sum(1 for record in records if record.get("status") == "failed"),
        "baseline_variant": baseline_variant,
        "records": records,
        "aggregates": aggregates,
        "comparisons": _compare_to_baseline(records, baseline_variant),
        "manifest_file": "batch_experiment_manifest.json",
    }
    (output_dir / "batch_experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_batch_experiment_markdown(report, output_dir / "batch_experiment_summary.md")
    return report
