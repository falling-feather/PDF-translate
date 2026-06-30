from __future__ import annotations

import asyncio
import shutil
import unittest
from pathlib import Path

from fastapi import HTTPException

from pdf_translate.server.routes_web import (
    PDF_UPLOAD_MAGIC,
    UPLOAD_READ_CHUNK_BYTES,
    _save_pdf_upload_streaming,
)


class FakeUpload:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size is None or size < 0:
            size = len(self.data) - self.offset
        chunk = self.data[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class WebUploadSecurityTests(unittest.TestCase):
    def _root(self, name: str) -> Path:
        root = Path.cwd() / "test-output" / "web-upload-security" / name
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def test_save_pdf_upload_streams_valid_pdf(self) -> None:
        data = PDF_UPLOAD_MAGIC + b"1.7\n" + b"x" * (UPLOAD_READ_CHUNK_BYTES + 7)
        upload = FakeUpload(data)
        dest = self._root("valid") / "input.pdf"
        uploaded_bytes = asyncio.run(
            _save_pdf_upload_streaming(
                upload,
                dest,
                max_bytes=len(data) + 1,
                max_mb=2,
            )
        )

        self.assertEqual(uploaded_bytes, len(data))
        self.assertEqual(dest.read_bytes(), data)
        self.assertEqual(upload.read_sizes[0], len(PDF_UPLOAD_MAGIC))
        self.assertIn(UPLOAD_READ_CHUNK_BYTES, upload.read_sizes[1:])

    def test_save_pdf_upload_rejects_non_pdf_magic(self) -> None:
        upload = FakeUpload(b"not a pdf")
        dest = self._root("invalid-magic") / "input.pdf"
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(
                _save_pdf_upload_streaming(
                    upload,
                    dest,
                    max_bytes=1024,
                    max_mb=1,
                )
            )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertFalse(dest.exists())

    def test_save_pdf_upload_removes_partial_file_when_over_limit(self) -> None:
        upload = FakeUpload(PDF_UPLOAD_MAGIC + b"x" * 32)
        dest = self._root("over-limit") / "input.pdf"
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(
                _save_pdf_upload_streaming(
                    upload,
                    dest,
                    max_bytes=len(PDF_UPLOAD_MAGIC) + 1,
                    max_mb=1,
                )
            )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
