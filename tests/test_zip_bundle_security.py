from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from pdf_translate.zip_bundle import iter_bundle_files


class ZipBundleSecurityTests(unittest.TestCase):
    def _root(self) -> Path:
        root = Path.cwd() / "test-output" / "zip-bundle-security"
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _make_symlink_or_skip(self, target: Path, link: Path, *, target_is_dir: bool = False) -> None:
        try:
            link.symlink_to(target, target_is_directory=target_is_dir)
        except NotImplementedError as exc:
            self.skipTest(f"Symlink not supported in this environment: {exc}")
        except PermissionError as exc:
            self.skipTest(f"Symlink creation denied in this environment: {exc}")
        except OSError as exc:
            # Windows may return a privilege-related error when developer mode is off.
            if getattr(exc, "winerror", None) == 1314:
                self.skipTest(f"Symlink privilege missing in this environment: {exc}")
            raise

    def test_iter_bundle_files_keeps_regular_files(self) -> None:
        root = self._root() / "root"
        memory = root / "memory"
        repairs = root / "output" / "repairs"
        memory.mkdir(parents=True)
        repairs.mkdir(parents=True)
        translated = root / "output" / "translated_full.md"
        translated.parent.mkdir(parents=True, exist_ok=True)
        translated.write_text("normal output", encoding="utf-8")
        translated_pdf = root / "output" / "translated_full.pdf"
        translated_pdf.write_bytes(b"%PDF-1.4 normal output")
        (memory / "safe.md").write_text("normal memory", encoding="utf-8")
        (repairs / "safe.md").write_text("normal repair", encoding="utf-8")

        included = {
            path.relative_to(root).as_posix()
            for path in iter_bundle_files(root)
        }

        self.assertIn("output/translated_full.md", included)
        self.assertIn("output/translated_full.pdf", included)
        self.assertIn("memory/safe.md", included)
        self.assertIn("output/repairs/safe.md", included)

    def test_iter_bundle_files_ignores_symlink_path_escape(self) -> None:
        created_links: list[Path] = []
        base = self._root()
        work_root = base / "root"
        outside_root = base / "outside"
        work_root.mkdir()
        outside_root.mkdir()
        try:
            root = work_root
            output = root / "output"
            memory = root / "memory"
            repairs = output / "repairs"
            output.mkdir(parents=True)
            memory.mkdir()
            repairs.mkdir()

            normal_output = output / "translated_full.md"
            normal_output.write_text("normal output", encoding="utf-8")
            normal_pdf = output / "translated_full.pdf"
            normal_pdf.write_bytes(b"%PDF-1.4 normal output")
            normal_memory = memory / "safe.md"
            normal_memory.write_text("normal memory", encoding="utf-8")
            normal_repair = repairs / "safe.md"
            normal_repair.write_text("normal repair", encoding="utf-8")

            outside_dir_path = outside_root / "outside"
            outside_dir_path.mkdir()
            outside_file = outside_dir_path / "outside.txt"
            outside_file.write_text("outside", encoding="utf-8")

            linked_file = memory / "linked.md"
            self._make_symlink_or_skip(outside_file, linked_file)
            created_links.append(linked_file)
            linked_dir = repairs / "linked"
            self._make_symlink_or_skip(outside_dir_path, linked_dir, target_is_dir=True)
            created_links.append(linked_dir)

            included = {
                path.relative_to(root).as_posix()
                for path in iter_bundle_files(root)
            }

            self.assertIn("output/translated_full.md", included)
            self.assertIn("output/translated_full.pdf", included)
            self.assertIn("memory/safe.md", included)
            self.assertIn("output/repairs/safe.md", included)
            self.assertNotIn("memory/linked.md", included)
            self.assertNotIn("output/repairs/linked", included)
        finally:
            for link in created_links:
                try:
                    link.unlink()
                except OSError:
                    shutil.rmtree(link, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
