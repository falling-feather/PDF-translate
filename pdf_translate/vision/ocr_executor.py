from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from pdf_translate.vision.ocr_writeback import OCR_RESULTS_SCHEMA_VERSION

SCHEMA_VERSION = "ocr-execution-v1"
DEFAULT_ENGINE = "tesseract_cli"
DEFAULT_LANGUAGE = "eng"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_CONFIDENCE = 0.6
SUPPORTED_ENGINES = {DEFAULT_ENGINE}

CommandRunner = Callable[[list[str], int], subprocess.CompletedProcess[str]]


def _tasks(ocr_tasks: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(ocr_tasks, dict):
        return []
    raw = ocr_tasks.get("tasks")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _resolve_input_path(work_dir: Path, input_path: str) -> Path:
    path = Path(input_path)
    if path.is_absolute():
        return path
    return work_dir / "output" / path


def _task_bbox(task: dict[str, Any]) -> list[Any]:
    raw = task.get("bbox")
    return list(raw) if isinstance(raw, list) else []


def _psm_for_task(task: dict[str, Any]) -> str:
    if str(task.get("scope") or "") == "page":
        return "3"
    return "6"


def _default_runner(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def _result(
    task: dict[str, Any],
    *,
    status: str,
    text: str = "",
    confidence: float = 0.0,
    engine: str,
    language: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": str(task.get("task_id") or ""),
        "status": status,
        "text": text,
        "confidence": confidence,
        "engine": engine,
        "language": language,
        "bbox": _task_bbox(task),
        "warnings": list(warnings or []),
        "page_no": int(task.get("page_no") or 0),
        "block_id": str(task.get("block_id") or ""),
        "input_path": str(task.get("input_path") or ""),
    }


def _engine_binary(engine: str, command: str | None, runner: CommandRunner | None) -> str:
    if command:
        return command
    if runner is not None:
        return "tesseract"
    return shutil.which("tesseract") or ""


def execute_ocr_tasks(
    ocr_tasks: dict[str, Any] | None,
    work_dir: Path,
    *,
    engine: str = DEFAULT_ENGINE,
    language: str = DEFAULT_LANGUAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    command: str | None = None,
    command_runner: CommandRunner | None = None,
    default_confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Execute ready OCR tasks through an optional local command-line OCR engine."""
    task_list = _tasks(ocr_tasks)
    resolved_engine = engine or DEFAULT_ENGINE
    resolved_language = language or DEFAULT_LANGUAGE
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    executed_commands: list[dict[str, Any]] = []

    if resolved_engine not in SUPPORTED_ENGINES:
        for task in task_list:
            results.append(
                _result(
                    task,
                    status="failed",
                    engine=resolved_engine,
                    language=resolved_language,
                    warnings=["unsupported_ocr_engine"],
                )
            )
        status_counts = Counter(str(item.get("status") or "unknown") for item in results)
        return _payload(
            ocr_tasks,
            results,
            skipped,
            executed_commands,
            engine=resolved_engine,
            language=resolved_language,
            engine_available=False,
            binary="",
            status_counts=status_counts,
        )

    binary = _engine_binary(resolved_engine, command, command_runner)
    runner = command_runner or _default_runner
    engine_available = bool(binary)

    for task in task_list:
        task_status = str(task.get("status") or "")
        if task_status != "pending_engine":
            item = _result(
                task,
                status="skipped",
                engine=resolved_engine,
                language=resolved_language,
                warnings=[f"task_status_{task_status or 'unknown'}"],
            )
            results.append(item)
            skipped.append(item)
            continue

        if not engine_available:
            results.append(
                _result(
                    task,
                    status="failed",
                    engine=resolved_engine,
                    language=resolved_language,
                    warnings=["ocr_engine_not_found"],
                )
            )
            continue

        input_path = _resolve_input_path(work_dir, str(task.get("input_path") or ""))
        if not input_path.is_file():
            results.append(
                _result(
                    task,
                    status="failed",
                    engine=resolved_engine,
                    language=resolved_language,
                    warnings=["input_file_missing"],
                )
            )
            continue

        cmd = [
            binary,
            str(input_path),
            "stdout",
            "-l",
            resolved_language,
            "--psm",
            _psm_for_task(task),
        ]
        executed_commands.append(
            {
                "task_id": str(task.get("task_id") or ""),
                "input_path": str(task.get("input_path") or ""),
                "engine": resolved_engine,
                "language": resolved_language,
                "psm": cmd[-1],
            }
        )
        try:
            completed = runner(cmd, timeout_seconds)
        except subprocess.TimeoutExpired:
            results.append(
                _result(
                    task,
                    status="failed",
                    engine=resolved_engine,
                    language=resolved_language,
                    warnings=["ocr_timeout"],
                )
            )
            continue
        except OSError as exc:
            results.append(
                _result(
                    task,
                    status="failed",
                    engine=resolved_engine,
                    language=resolved_language,
                    warnings=[f"ocr_command_error:{type(exc).__name__}"],
                )
            )
            continue

        text = (completed.stdout or "").strip()
        if completed.returncode != 0:
            results.append(
                _result(
                    task,
                    status="failed",
                    text=text,
                    engine=resolved_engine,
                    language=resolved_language,
                    warnings=["ocr_command_failed"],
                )
            )
            continue
        if not text:
            results.append(
                _result(
                    task,
                    status="failed",
                    engine=resolved_engine,
                    language=resolved_language,
                    warnings=["ocr_empty_text"],
                )
            )
            continue
        results.append(
            _result(
                task,
                status="succeeded",
                text=text,
                confidence=default_confidence,
                engine=resolved_engine,
                language=resolved_language,
                warnings=["confidence_estimated"],
            )
        )

    status_counts = Counter(str(item.get("status") or "unknown") for item in results)
    return _payload(
        ocr_tasks,
        results,
        skipped,
        executed_commands,
        engine=resolved_engine,
        language=resolved_language,
        engine_available=engine_available,
        binary=binary,
        status_counts=status_counts,
    )


def _payload(
    ocr_tasks: dict[str, Any] | None,
    results: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    executed_commands: list[dict[str, Any]],
    *,
    engine: str,
    language: str,
    engine_available: bool,
    binary: str,
    status_counts: Counter[str],
) -> dict[str, Any]:
    attempted_count = len(executed_commands)
    succeeded_count = status_counts.get("succeeded", 0)
    failed_count = status_counts.get("failed", 0)
    return {
        "schema_version": OCR_RESULTS_SCHEMA_VERSION,
        "doc_id": str((ocr_tasks or {}).get("doc_id") or ""),
        "source": "local_ocr_executor",
        "results": results,
        "execution": {
            "schema_version": SCHEMA_VERSION,
            "summary": {
                "task_count": len(_tasks(ocr_tasks)),
                "attempted_task_count": attempted_count,
                "succeeded_task_count": succeeded_count,
                "failed_task_count": failed_count,
                "skipped_task_count": len(skipped),
                "engine_available": engine_available,
                "engine": engine,
                "language": language,
                "status_counts": dict(status_counts),
            },
            "engine": {
                "type": engine,
                "binary": binary,
                "language": language,
            },
            "commands": executed_commands,
            "skipped_tasks": skipped,
        },
    }


def write_ocr_execution_results(
    ocr_tasks: dict[str, Any] | None,
    work_dir: Path,
    path: Path,
    *,
    engine: str = DEFAULT_ENGINE,
    language: str = DEFAULT_LANGUAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    command: str | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    payload = execute_ocr_tasks(
        ocr_tasks,
        work_dir,
        engine=engine,
        language=language,
        timeout_seconds=timeout_seconds,
        command=command,
        command_runner=command_runner,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
