from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import fitz

from pdf_translate.config import AppConfig
from pdf_translate.experiments import parse_variant_specs, run_batch_experiment


class BatchExperimentTests(unittest.TestCase):
    def test_parse_variant_specs_deduplicates_and_sets_flags(self) -> None:
        variants = parse_variant_specs("page, structure, structure+ocr+repair, structure")

        self.assertEqual([variant.name for variant in variants], ["page", "structure", "structure+ocr+repair"])
        self.assertEqual(variants[0].chunk_strategy, "page")
        self.assertEqual(variants[1].chunk_strategy, "structure")
        self.assertTrue(variants[2].execute_ocr)
        self.assertTrue(variants[2].execute_repair_requests)

    def test_run_batch_experiment_writes_patent_evidence_summary(self) -> None:
        root = Path.cwd() / "test-output" / "batch-experiment"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            pdf_path = root / "sample.pdf"
            doc = fitz.open()
            p1 = doc.new_page(width=595, height=842)
            p1.insert_text((72, 72), "1 Introduction\nSmith et al. (2024) evaluated BERT.")
            p2 = doc.new_page(width=595, height=842)
            p2.insert_text((72, 72), "Table 1: Results\nModel Acc F1 N\nA 91.2 88.1 120")
            pdf_path.write_bytes(doc.tobytes())
            doc.close()

            output_dir = root / "experiment"
            report = run_batch_experiment(
                [pdf_path],
                output_dir,
                AppConfig.from_env(),
                variants=parse_variant_specs("page,structure"),
                backend="echo",
                pages_per_chunk=1,
                overlap_pages=0,
                max_chunks=1,
            )

            summary_json = output_dir / "batch_experiment_summary.json"
            summary_md = output_dir / "batch_experiment_summary.md"
            manifest = output_dir / "batch_experiment_manifest.json"
            self.assertTrue(summary_json.is_file())
            self.assertTrue(summary_md.is_file())
            self.assertTrue(manifest.is_file())
            self.assertEqual(report["schema_version"], "batch-experiment-v1")
            self.assertEqual(report["sample_count"], 1)
            self.assertEqual(report["run_count"], 2)
            self.assertEqual(report["succeeded_count"], 2)
            self.assertEqual(report["failed_count"], 0)
            self.assertEqual(report["baseline_variant"], "page")
            self.assertEqual({item["variant"] for item in report["aggregates"]}, {"page", "structure"})
            self.assertEqual(report["comparisons"][0]["variant"], "structure")

            loaded = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(loaded["records"][0]["status"], "succeeded")
            self.assertIn("metrics", loaded["records"][0])
            self.assertIn("translation_issue_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("total_elapsed_ms", loaded["records"][0]["metrics"]["performance"])
            self.assertIn("runs/001-sample/page/output/experiment_metrics.json", summary_json.read_text(encoding="utf-8"))
            self.assertIn("批量实验汇总", summary_md.read_text(encoding="utf-8"))
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)


if __name__ == "__main__":
    unittest.main()
