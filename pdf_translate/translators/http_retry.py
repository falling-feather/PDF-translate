"""HTTP 调用重试：应对对端提前断连、429/502/503/504 等。"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

import httpx

T = TypeVar("T")
HTTP_RETRY_EVENT_SCHEMA_VERSION = "http-retry-event-v1"

_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
_RETRYABLE_EXCEPTIONS = (
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
)
_captured_events: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "pdf_translate_http_retry_events",
    default=None,
)


def max_attempts() -> int:
    try:
        n = int(os.getenv("PDF_TRANSLATE_HTTP_RETRIES", "4"))
        return max(1, min(n, 12))
    except ValueError:
        return 4


def _sleep_backoff(attempt: int) -> None:
    base = min(90.0, (2**attempt) + random.uniform(0, 0.8))
    time.sleep(base)


@contextmanager
def capture_http_retry_events() -> Iterator[list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    token = _captured_events.set(events)
    try:
        yield events
    finally:
        _captured_events.reset(token)


def _record_retry_event(
    *,
    context: str,
    attempt_index: int,
    max_attempts: int,
    status: str,
    elapsed_ms: int,
    will_retry: bool,
    error: BaseException | None = None,
    status_code: int | None = None,
) -> None:
    events = _captured_events.get()
    if events is None:
        return
    events.append(
        {
            "schema_version": HTTP_RETRY_EVENT_SCHEMA_VERSION,
            "context": context,
            "attempt_index": attempt_index,
            "max_attempts": max_attempts,
            "status": status,
            "elapsed_ms": max(0, int(elapsed_ms)),
            "will_retry": bool(will_retry),
            "error_type": type(error).__name__ if error is not None else "",
            "status_code": status_code,
        }
    )


def summarize_http_retry_events(events: list[dict[str, Any]] | None) -> dict[str, int]:
    retry_events = [event for event in (events or []) if isinstance(event, dict)]
    return {
        "http_attempt_count": len(retry_events),
        "http_retry_count": sum(1 for event in retry_events if bool(event.get("will_retry"))),
        "http_failed_attempt_count": sum(
            1 for event in retry_events if str(event.get("status") or "") != "success"
        ),
        "http_retryable_error_count": sum(
            1 for event in retry_events if str(event.get("status") or "") == "retryable_error"
        ),
        "http_fatal_error_count": sum(
            1 for event in retry_events if str(event.get("status") or "") == "fatal_error"
        ),
    }


def call_with_http_retry(
    op: Callable[[], T],
    *,
    context: str = "HTTP",
) -> T:
    last: BaseException | None = None
    attempts = max_attempts()
    for attempt in range(attempts):
        started = time.perf_counter()
        try:
            result = op()
        except httpx.HTTPStatusError as e:
            last = e
            response = getattr(e, "response", None)
            code = int(response.status_code) if response is not None else None
            retryable = code in _RETRYABLE_STATUS_CODES
            will_retry = retryable and attempt < attempts - 1
            _record_retry_event(
                context=context,
                attempt_index=attempt + 1,
                max_attempts=attempts,
                status="retryable_error" if retryable else "fatal_error",
                elapsed_ms=round((time.perf_counter() - started) * 1000),
                will_retry=will_retry,
                error=e,
                status_code=code,
            )
            if not retryable:
                raise
        except _RETRYABLE_EXCEPTIONS as e:
            last = e
            will_retry = attempt < attempts - 1
            _record_retry_event(
                context=context,
                attempt_index=attempt + 1,
                max_attempts=attempts,
                status="retryable_error",
                elapsed_ms=round((time.perf_counter() - started) * 1000),
                will_retry=will_retry,
                error=e,
            )
        else:
            _record_retry_event(
                context=context,
                attempt_index=attempt + 1,
                max_attempts=attempts,
                status="success",
                elapsed_ms=round((time.perf_counter() - started) * 1000),
                will_retry=False,
            )
            return result
        if will_retry:
            _sleep_backoff(attempt)
    assert last is not None
    raise RuntimeError(f"{context} 在 {attempts} 次重试后仍失败: {last}") from last
