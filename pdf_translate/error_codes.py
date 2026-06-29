from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import httpx

ERROR_CODE_SCHEMA_VERSION = "error-code-v1"


@dataclass(frozen=True)
class ErrorSpec:
    code: str
    category: str
    retryable: bool
    user_message: str
    next_step: str


@dataclass(frozen=True)
class ErrorInfo:
    code: str
    category: str
    retryable: bool
    user_message: str
    next_step: str
    detail: str = ""
    source: str = ""
    http_status: int | None = None
    exception_type: str = ""

    def with_context(
        self,
        *,
        detail: str | None = None,
        source: str | None = None,
        http_status: int | None = None,
        exception: BaseException | None = None,
    ) -> ErrorInfo:
        return replace(
            self,
            detail=self.detail if detail is None else detail,
            source=self.source if source is None else source,
            http_status=self.http_status if http_status is None else http_status,
            exception_type=(
                self.exception_type if exception is None else type(exception).__name__
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ERROR_CODE_SCHEMA_VERSION,
            "code": self.code,
            "category": self.category,
            "retryable": self.retryable,
            "user_message": self.user_message,
            "next_step": self.next_step,
            "detail": self.detail,
            "source": self.source,
            "http_status": self.http_status,
            "exception_type": self.exception_type,
        }


ERROR_SPECS: dict[str, ErrorSpec] = {
    "CONFIG_MISSING_API_KEY": ErrorSpec(
        code="CONFIG_MISSING_API_KEY",
        category="config",
        retryable=False,
        user_message="API key is missing.",
        next_step="Configure the required API key, then retry the task.",
    ),
    "CONFIG_INVALID_BACKEND": ErrorSpec(
        code="CONFIG_INVALID_BACKEND",
        category="config",
        retryable=False,
        user_message="The selected translation backend is invalid.",
        next_step="Choose an enabled backend or update the backend configuration.",
    ),
    "CONFIG_INVALID_OPTION": ErrorSpec(
        code="CONFIG_INVALID_OPTION",
        category="config",
        retryable=False,
        user_message="The task configuration is invalid.",
        next_step="Check the submitted options and administrator settings.",
    ),
    "HTTP_RATE_LIMIT": ErrorSpec(
        code="HTTP_RATE_LIMIT",
        category="rate_limit",
        retryable=True,
        user_message="The upstream API rate limit was reached.",
        next_step="Wait for the quota window to reset or reduce concurrency.",
    ),
    "HTTP_TIMEOUT": ErrorSpec(
        code="HTTP_TIMEOUT",
        category="timeout",
        retryable=True,
        user_message="The upstream API request timed out.",
        next_step="Retry later, increase timeout, or reduce chunk size.",
    ),
    "HTTP_AUTH_ERROR": ErrorSpec(
        code="HTTP_AUTH_ERROR",
        category="auth",
        retryable=False,
        user_message="The upstream API rejected authentication.",
        next_step="Check whether the API key, base URL, and model are valid.",
    ),
    "HTTP_CLIENT_ERROR": ErrorSpec(
        code="HTTP_CLIENT_ERROR",
        category="http_client",
        retryable=False,
        user_message="The upstream API rejected the request.",
        next_step="Check request configuration, model name, and account permissions.",
    ),
    "HTTP_SERVER_ERROR": ErrorSpec(
        code="HTTP_SERVER_ERROR",
        category="http_server",
        retryable=True,
        user_message="The upstream API returned a server error.",
        next_step="Retry later or switch to another enabled backend.",
    ),
    "HTTP_NETWORK_ERROR": ErrorSpec(
        code="HTTP_NETWORK_ERROR",
        category="network",
        retryable=True,
        user_message="The upstream API connection failed.",
        next_step="Check network access, proxy settings, and upstream availability.",
    ),
    "API_RESPONSE_INVALID": ErrorSpec(
        code="API_RESPONSE_INVALID",
        category="api_response",
        retryable=False,
        user_message="The upstream API response could not be parsed.",
        next_step="Check model compatibility and captured upstream diagnostics.",
    ),
    "PDF_PARSE_ERROR": ErrorSpec(
        code="PDF_PARSE_ERROR",
        category="pdf_parse",
        retryable=False,
        user_message="The PDF could not be parsed.",
        next_step="Check whether the PDF is corrupted, encrypted, or image-only.",
    ),
    "TASK_CANCELLED": ErrorSpec(
        code="TASK_CANCELLED",
        category="cancelled",
        retryable=False,
        user_message="The task was cancelled by request.",
        next_step="Submit a new task if translation should continue.",
    ),
    "PIPELINE_ERROR": ErrorSpec(
        code="PIPELINE_ERROR",
        category="pipeline",
        retryable=False,
        user_message="The translation pipeline failed.",
        next_step="Inspect run logs and retry after fixing the failed phase.",
    ),
    "UNKNOWN_ERROR": ErrorSpec(
        code="UNKNOWN_ERROR",
        category="unknown",
        retryable=False,
        user_message="An unknown error occurred.",
        next_step="Inspect logs and preserve the work directory for diagnosis.",
    ),
}


class PdfTranslateError(RuntimeError):
    def __init__(self, info: ErrorInfo) -> None:
        self.error_info = info
        message = info.detail or info.user_message
        super().__init__(message)


def make_error_info(
    code: str,
    *,
    detail: str = "",
    source: str = "",
    http_status: int | None = None,
    exception: BaseException | None = None,
) -> ErrorInfo:
    spec = ERROR_SPECS.get(code, ERROR_SPECS["UNKNOWN_ERROR"])
    return ErrorInfo(
        code=spec.code,
        category=spec.category,
        retryable=spec.retryable,
        user_message=spec.user_message,
        next_step=spec.next_step,
        detail=detail,
        source=source,
        http_status=http_status,
        exception_type=type(exception).__name__ if exception is not None else "",
    )


def error_info_from_http_status(
    status_code: int | None,
    *,
    detail: str = "",
    source: str = "",
    exception: BaseException | None = None,
) -> ErrorInfo:
    if status_code == 429:
        code = "HTTP_RATE_LIMIT"
    elif status_code in (401, 403):
        code = "HTTP_AUTH_ERROR"
    elif status_code == 408:
        code = "HTTP_TIMEOUT"
    elif status_code is not None and status_code >= 500:
        code = "HTTP_SERVER_ERROR"
    elif status_code is not None and status_code >= 400:
        code = "HTTP_CLIENT_ERROR"
    else:
        code = "HTTP_NETWORK_ERROR"
    return make_error_info(
        code,
        detail=detail,
        source=source,
        http_status=status_code,
        exception=exception,
    )


def _looks_like_pdf_parse_error(exc: BaseException) -> bool:
    mod = type(exc).__module__.lower()
    name = type(exc).__name__.lower()
    return (
        mod.startswith("fitz")
        or mod.startswith("pymupdf")
        or "filedataerror" in name
        or "document" in name and "error" in name
    )


def error_info_from_exception(
    exc: BaseException,
    *,
    source: str = "",
    detail: str | None = None,
) -> ErrorInfo:
    resolved_detail = str(exc) if detail is None else detail
    if isinstance(exc, PdfTranslateError):
        return exc.error_info.with_context(
            detail=resolved_detail or exc.error_info.detail,
            source=source or exc.error_info.source,
        )
    if isinstance(exc, httpx.HTTPStatusError):
        response = getattr(exc, "response", None)
        status = int(response.status_code) if response is not None else None
        return error_info_from_http_status(
            status,
            detail=resolved_detail,
            source=source,
            exception=exc,
        )
    if isinstance(exc, httpx.TimeoutException):
        return make_error_info(
            "HTTP_TIMEOUT",
            detail=resolved_detail,
            source=source,
            exception=exc,
        )
    if isinstance(exc, (httpx.NetworkError, httpx.RemoteProtocolError, httpx.TransportError)):
        return make_error_info(
            "HTTP_NETWORK_ERROR",
            detail=resolved_detail,
            source=source,
            exception=exc,
        )
    if type(exc).__name__ == "JobCancelled":
        return make_error_info(
            "TASK_CANCELLED",
            detail=resolved_detail,
            source=source,
            exception=exc,
        )
    if _looks_like_pdf_parse_error(exc):
        return make_error_info(
            "PDF_PARSE_ERROR",
            detail=resolved_detail,
            source=source,
            exception=exc,
        )
    if isinstance(exc, ValueError):
        lowered = resolved_detail.lower()
        if "api key" in lowered or "api_key" in lowered:
            code = "CONFIG_MISSING_API_KEY"
        elif "backend" in lowered or "translator" in lowered:
            code = "CONFIG_INVALID_BACKEND"
        else:
            code = "CONFIG_INVALID_OPTION"
        return make_error_info(code, detail=resolved_detail, source=source, exception=exc)
    return make_error_info(
        "UNKNOWN_ERROR",
        detail=resolved_detail,
        source=source,
        exception=exc,
    )


def error_payload(exc: BaseException, *, source: str = "") -> dict[str, Any]:
    return error_info_from_exception(exc, source=source).to_dict()
