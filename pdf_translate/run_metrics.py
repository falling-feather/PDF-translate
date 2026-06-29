from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

RUN_LOG_EVENT_SCHEMA_VERSION = "run-log-event-v1"
RUN_METRICS_SCHEMA_VERSION = "run-metrics-v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def elapsed_ms_since(start_perf: float) -> int:
    return max(0, int(round((perf_counter() - start_perf) * 1000)))


def estimate_token_count(char_count: int | float | None) -> int:
    chars = max(0, int(char_count or 0))
    if not chars:
        return 0
    return (chars + 3) // 4


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _http_retry_summary(
    event: dict[str, Any],
    retry_events: list[dict[str, Any]],
) -> dict[str, int]:
    if retry_events:
        return {
            "http_attempt_count": len(retry_events),
            "http_retry_count": sum(1 for item in retry_events if bool(item.get("will_retry"))),
            "http_failed_attempt_count": sum(
                1 for item in retry_events if str(item.get("status") or "") != "success"
            ),
            "http_retryable_error_count": sum(
                1 for item in retry_events if str(item.get("status") or "") == "retryable_error"
            ),
            "http_fatal_error_count": sum(
                1 for item in retry_events if str(item.get("status") or "") == "fatal_error"
            ),
        }
    return {
        "http_attempt_count": _as_int(event.get("http_attempt_count")),
        "http_retry_count": _as_int(event.get("http_retry_count")),
        "http_failed_attempt_count": _as_int(event.get("http_failed_attempt_count")),
        "http_retryable_error_count": _as_int(event.get("http_retryable_error_count")),
        "http_fatal_error_count": _as_int(event.get("http_fatal_error_count")),
    }


def build_run_metrics(
    events: list[dict[str, Any]],
    *,
    doc_id: str | None = None,
    pipeline_variant: str | None = None,
    backend: str | None = None,
    translate_mode: str | None = None,
    parallel_workers: int | None = None,
    page_count: int | None = None,
    chunk_count: int | None = None,
    completed_chunk_count: int | None = None,
    total_elapsed_ms: int | None = None,
) -> dict[str, Any]:
    stage_elapsed = Counter()
    stage_counts = Counter()
    translator_counts = Counter()
    skip_reasons = Counter()
    chunks: list[dict[str, Any]] = []

    for event in events:
        event_type = str(event.get("event_type") or "")
        phase = str(event.get("phase") or "unknown")
        elapsed = _as_int(event.get("elapsed_ms"))
        if event_type in {"stage", "stage_error"}:
            stage_elapsed[phase] += elapsed
            stage_counts[phase] += 1
        elif event_type == "chunk_translation":
            translator = str(event.get("translator") or "unknown")
            translator_counts[translator] += 1
            retry_events = [
                item
                for item in (event.get("http_retry_events") or [])
                if isinstance(item, dict)
            ]
            retry_summary = _http_retry_summary(event, retry_events)
            chunks.append(
                {
                    "chunk_id": event.get("chunk_id"),
                    "chunk_index": event.get("chunk_index"),
                    "pages_1based": event.get("pages_1based") or [],
                    "translator": translator,
                    "elapsed_ms": elapsed,
                    "source_char_count": _as_int(event.get("source_char_count")),
                    "context_char_count": _as_int(event.get("context_char_count")),
                    "request_char_count": _as_int(event.get("request_char_count")),
                    "translated_char_count": _as_int(event.get("translated_char_count")),
                    "estimated_request_token_count": _as_int(
                        event.get("estimated_request_token_count")
                    ),
                    "estimated_translated_token_count": _as_int(
                        event.get("estimated_translated_token_count")
                    ),
                    **retry_summary,
                    "http_retry_events": retry_events,
                }
            )
        elif event_type == "chunk_skipped":
            skip_reasons[str(event.get("reason") or "unknown")] += 1

    translation_elapsed_ms = sum(chunk["elapsed_ms"] for chunk in chunks)
    source_char_count = sum(chunk["source_char_count"] for chunk in chunks)
    context_char_count = sum(chunk["context_char_count"] for chunk in chunks)
    request_char_count = sum(chunk["request_char_count"] for chunk in chunks)
    translated_char_count = sum(chunk["translated_char_count"] for chunk in chunks)
    estimated_request_token_count = sum(chunk["estimated_request_token_count"] for chunk in chunks)
    estimated_translated_token_count = sum(
        chunk["estimated_translated_token_count"] for chunk in chunks
    )
    http_attempt_count = sum(chunk["http_attempt_count"] for chunk in chunks)
    http_retry_count = sum(chunk["http_retry_count"] for chunk in chunks)
    http_failed_attempt_count = sum(chunk["http_failed_attempt_count"] for chunk in chunks)
    http_retryable_error_count = sum(chunk["http_retryable_error_count"] for chunk in chunks)
    http_fatal_error_count = sum(chunk["http_fatal_error_count"] for chunk in chunks)
    translation_request_count = len(chunks)
    max_chunk_elapsed_ms = max((chunk["elapsed_ms"] for chunk in chunks), default=0)

    summary = {
        "page_count": _as_int(page_count),
        "chunk_count": _as_int(chunk_count),
        "completed_chunk_count": _as_int(completed_chunk_count)
        or translation_request_count,
        "skipped_chunk_count": sum(skip_reasons.values()),
        "translation_request_count": translation_request_count,
        "total_elapsed_ms": _as_int(total_elapsed_ms),
        "stage_elapsed_ms": dict(sorted(stage_elapsed.items())),
        "translation_elapsed_ms": translation_elapsed_ms,
        "avg_chunk_elapsed_ms": _rate(translation_elapsed_ms, translation_request_count),
        "max_chunk_elapsed_ms": max_chunk_elapsed_ms,
        "source_char_count": source_char_count,
        "context_char_count": context_char_count,
        "request_char_count": request_char_count,
        "translated_char_count": translated_char_count,
        "estimated_source_token_count": estimate_token_count(source_char_count),
        "estimated_context_token_count": estimate_token_count(context_char_count),
        "estimated_request_token_count": estimated_request_token_count,
        "estimated_translated_token_count": estimated_translated_token_count,
        "estimated_total_token_count": estimated_request_token_count
        + estimated_translated_token_count,
        "http_attempt_count": http_attempt_count,
        "http_retry_count": http_retry_count,
        "http_failed_attempt_count": http_failed_attempt_count,
        "http_retryable_error_count": http_retryable_error_count,
        "http_fatal_error_count": http_fatal_error_count,
        "request_chars_per_second": _rate(request_char_count * 1000, translation_elapsed_ms),
        "translated_chars_per_second": _rate(
            translated_char_count * 1000,
            translation_elapsed_ms,
        ),
    }

    return {
        "schema_version": RUN_METRICS_SCHEMA_VERSION,
        "doc_id": doc_id or "unknown",
        "pipeline_variant": pipeline_variant or "unknown",
        "backend": backend or "unknown",
        "translate_mode": translate_mode or "unknown",
        "parallel_workers": _as_int(parallel_workers),
        "summary": summary,
        "breakdowns": {
            "stage_counts": dict(sorted(stage_counts.items())),
            "translator_counts": dict(sorted(translator_counts.items())),
            "skip_reasons": dict(sorted(skip_reasons.items())),
        },
        "chunks": chunks,
    }


@dataclass
class RunMetricsRecorder:
    log_path: Path
    reset: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)
    started_perf: float = field(default_factory=perf_counter)

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.reset:
            self.log_path.write_text("", encoding="utf-8")

    def record(self, event_type: str, phase: str, **payload: Any) -> dict[str, Any]:
        event = {
            "schema_version": RUN_LOG_EVENT_SCHEMA_VERSION,
            "timestamp": utc_now_iso(),
            "event_type": event_type,
            "phase": phase,
        }
        event.update({key: _jsonable(value) for key, value in payload.items()})
        self.events.append(event)
        with self.log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    @contextmanager
    def stage(self, phase: str, **payload: Any) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        except Exception as exc:
            self.record(
                "stage_error",
                phase,
                elapsed_ms=elapsed_ms_since(started),
                error_type=type(exc).__name__,
                **payload,
            )
            raise
        else:
            self.record("stage", phase, elapsed_ms=elapsed_ms_since(started), **payload)

    def record_chunk_translation(
        self,
        *,
        chunk_id: str,
        pages_1based: list[int],
        translator: str,
        elapsed_ms: int,
        source_char_count: int,
        context_char_count: int,
        request_char_count: int,
        translated_char_count: int,
        chunk_index: int | None = None,
        chunk_total: int | None = None,
        raw_translated_char_count: int | None = None,
        deferred_char_count: int = 0,
        http_retry_events: list[dict[str, Any]] | None = None,
        prompt_version: str | None = None,
        prompt_fingerprint: str | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        retry_events = [item for item in (http_retry_events or []) if isinstance(item, dict)]
        retry_summary = _http_retry_summary({}, retry_events)
        return self.record(
            "chunk_translation",
            "translation",
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            pages_1based=pages_1based,
            translator=translator,
            mode=mode,
            elapsed_ms=elapsed_ms,
            source_char_count=source_char_count,
            context_char_count=context_char_count,
            request_char_count=request_char_count,
            translated_char_count=translated_char_count,
            raw_translated_char_count=raw_translated_char_count
            if raw_translated_char_count is not None
            else translated_char_count,
            deferred_char_count=deferred_char_count,
            estimated_source_token_count=estimate_token_count(source_char_count),
            estimated_context_token_count=estimate_token_count(context_char_count),
            estimated_request_token_count=estimate_token_count(request_char_count),
            estimated_translated_token_count=estimate_token_count(translated_char_count),
            **retry_summary,
            http_retry_events=retry_events,
            prompt_version=prompt_version,
            prompt_fingerprint=prompt_fingerprint,
        )

    def record_chunk_skipped(
        self,
        *,
        chunk_id: str,
        pages_1based: list[int],
        reason: str,
        chunk_index: int | None = None,
        chunk_total: int | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        return self.record(
            "chunk_skipped",
            "translation",
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            pages_1based=pages_1based,
            reason=reason,
            mode=mode,
        )

    def write_summary(
        self,
        path: Path,
        *,
        doc_id: str | None = None,
        pipeline_variant: str | None = None,
        backend: str | None = None,
        translate_mode: str | None = None,
        parallel_workers: int | None = None,
        page_count: int | None = None,
        chunk_count: int | None = None,
        completed_chunk_count: int | None = None,
    ) -> dict[str, Any]:
        metrics = build_run_metrics(
            self.events,
            doc_id=doc_id,
            pipeline_variant=pipeline_variant,
            backend=backend,
            translate_mode=translate_mode,
            parallel_workers=parallel_workers,
            page_count=page_count,
            chunk_count=chunk_count,
            completed_chunk_count=completed_chunk_count,
            total_elapsed_ms=elapsed_ms_since(self.started_perf),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics
