"""HTTP 调用重试：应对对端提前断连、429/502/503/504 等。"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from typing import TypeVar

import httpx

T = TypeVar("T")


def max_attempts() -> int:
    try:
        n = int(os.getenv("PDF_TRANSLATE_HTTP_RETRIES", "4"))
        return max(1, min(n, 12))
    except ValueError:
        return 4


def _sleep_backoff(attempt: int) -> None:
    base = min(90.0, (2**attempt) + random.uniform(0, 0.8))
    time.sleep(base)


def call_with_http_retry(
    op: Callable[[], T],
    *,
    context: str = "HTTP",
) -> T:
    last: BaseException | None = None
    attempts = max_attempts()
    for attempt in range(attempts):
        try:
            return op()
        except httpx.HTTPStatusError as e:
            last = e
            code = e.response.status_code
            if code not in (429, 502, 503, 504):
                raise
        except (
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
        ) as e:
            last = e
        if attempt < attempts - 1:
            _sleep_backoff(attempt)
    assert last is not None
    raise RuntimeError(f"{context} 在 {attempts} 次重试后仍失败: {last}") from last
