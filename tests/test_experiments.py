from __future__ import annotations

import csv
import json
import shutil
import unittest
from pathlib import Path

import fitz

from pdf_translate.config import AppConfig
from pdf_translate.experiments import (
    load_sample_metadata,
    parse_variant_specs,
    run_batch_experiment,
    write_batch_experiment_evidence,
    write_sample_manifest,
)


class BatchExperimentTests(unittest.TestCase):
    @staticmethod
    def _write_sample_pdf(path: Path) -> None:
        doc = fitz.open()
        p1 = doc.new_page(width=595, height=842)
        p1.insert_text((72, 72), "1 Introduction\nSmith et al. (2024) evaluated BERT.")
        p2 = doc.new_page(width=595, height=842)
        p2.insert_text((72, 72), "Table 1: Results\nModel Acc F1 N\nA 91.2 88.1 120")
        path.write_bytes(doc.tobytes())
        doc.close()

    @staticmethod
    def _write_table_heavy_pdf(path: Path) -> None:
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        text = "\n".join(
            [
                "Table 1: Main results",
                "Model Acc F1 N",
                "A 91.2 88.1 120",
                "B 92.4 89.7 120",
                "Table 2: Robustness",
                "Group Mean SD N",
                "X 10.2 1.1 40",
                "Y 11.5 1.4 40",
            ]
        )
        page.insert_text((72, 72), text)
        path.write_bytes(doc.tobytes())
        doc.close()

    @staticmethod
    def _write_scanned_like_pdf(path: Path) -> None:
        doc = fitz.open()
        doc.new_page(width=595, height=842)
        path.write_bytes(doc.tobytes())
        doc.close()

    def test_parse_variant_specs_deduplicates_and_sets_flags(self) -> None:
        variants = parse_variant_specs("page, structure, structure+ocr+repair, structure")

        self.assertEqual([variant.name for variant in variants], ["page", "structure", "structure+ocr+repair"])
        self.assertEqual(variants[0].chunk_strategy, "page")
        self.assertEqual(variants[1].chunk_strategy, "structure")
        self.assertTrue(variants[2].execute_ocr)
        self.assertTrue(variants[2].execute_repair_requests)

    def test_write_sample_manifest_classifies_pdf_samples(self) -> None:
        root = Path.cwd() / "test-output" / "sample-manifest"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            table_pdf = root / "table-heavy.pdf"
            scanned_pdf = root / "scan.pdf"
            self._write_table_heavy_pdf(table_pdf)
            self._write_scanned_like_pdf(scanned_pdf)

            manifest_path = root / "samples.csv"
            report_path = root / "samples.json"
            manifest = write_sample_manifest([table_pdf, scanned_pdf], manifest_path, report_path=report_path)

            self.assertTrue(manifest_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertEqual(manifest["schema_version"], "experiment-sample-manifest-v1")
            self.assertEqual(manifest["sample_count"], 2)
            by_name = {Path(sample["source_pdf"]).name: sample for sample in manifest["samples"]}
            self.assertEqual(by_name["table-heavy.pdf"]["pdf_type"], "table-heavy")
            self.assertIn("table", by_name["table-heavy.pdf"]["tags"])
            self.assertEqual(by_name["scan.pdf"]["pdf_type"], "scanned")
            self.assertIn("scanned", by_name["scan.pdf"]["tags"])

            metadata = load_sample_metadata(manifest_path)
            self.assertEqual(metadata["table-heavy.pdf"]["pdf_type"], "table-heavy")
            self.assertEqual(metadata["scan.pdf"]["pdf_type"], "scanned")
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_run_batch_experiment_writes_patent_evidence_summary(self) -> None:
        root = Path.cwd() / "test-output" / "batch-experiment"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            pdf_path = root / "sample.pdf"
            self._write_sample_pdf(pdf_path)

            metadata_path = root / "samples.csv"
            metadata_path.write_text(
                "source_pdf,sample_id,pdf_type,tags,notes\n"
                "sample.pdf,sample-table,table-heavy,table;entity,用于表格与实体保留实验\n",
                encoding="utf-8",
            )

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
                sample_metadata=load_sample_metadata(metadata_path),
            )

            summary_json = output_dir / "batch_experiment_summary.json"
            summary_md = output_dir / "batch_experiment_summary.md"
            manifest = output_dir / "batch_experiment_manifest.json"
            review_csv = output_dir / "batch_experiment_review.csv"
            self.assertTrue(summary_json.is_file())
            self.assertTrue(summary_md.is_file())
            self.assertTrue(manifest.is_file())
            self.assertTrue(review_csv.is_file())
            self.assertEqual(report["schema_version"], "batch-experiment-v1")
            self.assertEqual(report["sample_count"], 1)
            self.assertEqual(report["run_count"], 2)
            self.assertEqual(report["succeeded_count"], 2)
            self.assertEqual(report["failed_count"], 0)
            self.assertEqual(report["baseline_variant"], "page")
            self.assertEqual({item["variant"] for item in report["aggregates"]}, {"page", "structure"})
            self.assertEqual(report["comparisons"][0]["variant"], "structure")
            self.assertEqual(report["samples"][0]["sample_id"], "sample-table")
            self.assertEqual(report["samples"][0]["pdf_type"], "table-heavy")
            self.assertEqual(report["samples"][0]["tags"], ["table", "entity"])

            loaded = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(loaded["records"][0]["status"], "succeeded")
            self.assertEqual(loaded["records"][0]["pdf_type"], "table-heavy")
            self.assertIn("metrics", loaded["records"][0])
            self.assertIn("translation_issue_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("structure_hint_chunk_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("structure_hint_avg_char_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("structure_hint_chunk_rate", loaded["records"][0]["metrics"]["rates"])
            self.assertIn("structure_hint_locked_token_per_chunk", loaded["records"][0]["metrics"]["rates"])
            self.assertIn(
                "structure_hint_merged_cell_candidate_type_counts",
                loaded["records"][0]["metrics"]["breakdowns"],
            )
            self.assertIn("table_merged_cell_candidate_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("table_merged_cell_candidate_rate", loaded["records"][0]["metrics"]["rates"])
            self.assertIn("table_merged_cell_candidate_type_counts", loaded["records"][0]["metrics"]["breakdowns"])
            self.assertIn("table_footnote_cell_binding_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("table_footnote_cell_binding_rate", loaded["records"][0]["metrics"]["rates"])
            self.assertIn("table_chain_reject_reason_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("table_chain_reject_reason_counts", loaded["records"][0]["metrics"]["breakdowns"])
            self.assertIn("repair_merge_table_targeted_patch_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("repair_merge_table_targeted_patch_rate", loaded["records"][0]["metrics"]["rates"])
            self.assertIn("repair_merge_strategy_counts", loaded["records"][0]["metrics"]["breakdowns"])
            self.assertIn("ocr_structured_table_candidate_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("ocr_structured_table_gate_review_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("ocr_structured_table_gate_pass_rate", loaded["records"][0]["metrics"]["rates"])
            self.assertIn("ocr_table_cell_bbox_coverage_rate", loaded["records"][0]["metrics"]["rates"])
            self.assertIn(
                "ocr_candidate_structured_table_gate_issue_counts",
                loaded["records"][0]["metrics"]["breakdowns"],
            )
            self.assertIn("breakdowns", loaded["aggregates"][0])
            self.assertIn("structure_hint_merged_cell_candidate_type_counts", loaded["aggregates"][0]["breakdowns"])
            self.assertIn("table_merged_cell_candidate_type_counts", loaded["aggregates"][0]["breakdowns"])
            self.assertIn("repair_merge_strategy_counts", loaded["aggregates"][0]["breakdowns"])
            self.assertIn("table_chain_reject_reason_category_counts", loaded["aggregates"][0]["breakdowns"])
            self.assertIn("ocr_candidate_structured_table_gate_issue_counts", loaded["aggregates"][0]["breakdowns"])
            self.assertIn("rates.ocr_structured_table_gate_pass_rate", loaded["comparisons"][0]["deltas"])
            self.assertIn("total_elapsed_ms", loaded["records"][0]["metrics"]["performance"])
            self.assertIn("runs/sample-table/page/output/experiment_metrics.json", summary_json.read_text(encoding="utf-8"))
            self.assertIn("translated_pdf", loaded["records"][0]["files"])
            self.assertIn("runs/sample-table/page/output/translated_full.pdf", summary_json.read_text(encoding="utf-8"))
            summary_text = summary_md.read_text(encoding="utf-8")
            self.assertIn("OCR structured table gate", summary_text)
            self.assertIn("批量实验汇总", summary_text)
            self.assertIn("续表拒绝类别", summary_text)
            self.assertIn("平均合并候选", summary_text)
            review_text = review_csv.read_text(encoding="utf-8-sig")
            self.assertIn("human_score", review_text)
            self.assertIn("human_score_markdown", review_text)
            self.assertIn("human_score_html", review_text)
            self.assertIn("human_score_pdf", review_text)
            self.assertIn("human_score_table_readability", review_text)
            self.assertIn("human_score_figure_footnote_layout", review_text)
            self.assertIn("human_score_terminology_consistency", review_text)
            self.assertIn("human_score_structure_coherence", review_text)
            self.assertIn("include_in_patent_evidence", review_text)
            self.assertIn("patent_evidence_notes", review_text)
            self.assertIn("table_merged_cell_candidate_types", review_text)
            self.assertIn("table_chain_reject_reason_categories", review_text)
            self.assertIn("ocr_structured_table_gate_pass_rate", review_text)
            self.assertIn("ocr_table_cell_bbox_coverage_rate", review_text)
            self.assertIn("ocr_structured_table_gate_issues", review_text)
            self.assertIn("translated_pdf", review_text)
            self.assertIn("table-heavy", review_text)
            self.assertIn("sample-table", review_text)
            with review_csv.open("r", encoding="utf-8-sig", newline="") as review_file:
                review_rows = list(csv.DictReader(review_file))
            self.assertEqual(len(review_rows), 2)
            self.assertEqual(review_rows[0]["include_in_patent_evidence"], "")
            self.assertEqual(review_rows[0]["human_score_pdf"], "")
            self.assertIn("ocr_structured_table_gate_pass_rate", review_rows[0])
            review_rows[0].update(
                {
                    "human_score": "4.5",
                    "human_score_table_readability": "4",
                    "human_score_structure_coherence": "5",
                    "include_in_patent_evidence": "是",
                    "patent_evidence_notes": "结构化表格 OCR 门禁可作为证据",
                    "reviewer": "导师",
                    "ocr_structured_table_candidate_count": "2",
                    "ocr_structured_table_gate_review_count": "1",
                    "ocr_structured_table_gate_pass_rate": "0.75",
                    "ocr_table_cell_bbox_coverage_rate": "0.5",
                    "ocr_structured_table_gate_issues": "structured_table_missing_cell_bboxes:1",
                }
            )
            with review_csv.open("w", encoding="utf-8-sig", newline="") as review_file:
                writer = csv.DictWriter(review_file, fieldnames=list(review_rows[0].keys()))
                writer.writeheader()
                writer.writerows(review_rows)

            evidence = write_batch_experiment_evidence(summary_json, review_csv, output_dir)
            evidence_json = output_dir / "batch_experiment_evidence.json"
            evidence_md = output_dir / "batch_experiment_evidence.md"
            self.assertTrue(evidence_json.is_file())
            self.assertTrue(evidence_md.is_file())
            self.assertEqual(evidence["schema_version"], "batch-experiment-evidence-v1")
            self.assertEqual(evidence["included_count"], 1)
            self.assertEqual(evidence["score_averages"]["human_score"]["average"], 4.5)
            self.assertEqual(
                evidence["ocr_structured_table_gate_summary"]["gate_issue_counts"][
                    "structured_table_missing_cell_bboxes"
                ],
                1,
            )
            self.assertEqual(evidence["evidence_candidates"][0]["reviewer"], "导师")
            self.assertIn("translated_pdf", evidence["evidence_candidates"][0]["files"])
            self.assertIn("结构化表格 OCR 门禁可作为证据", evidence_md.read_text(encoding="utf-8"))
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_sample_metadata_supports_positional_json_and_relative_tsv(self) -> None:
        root = Path.cwd() / "test-output" / "batch-experiment-metadata"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            json_pdf = root / "json-sample.pdf"
            tsv_pdf = root / "tsv-sample.pdf"
            self._write_sample_pdf(json_pdf)
            self._write_sample_pdf(tsv_pdf)

            json_manifest = root / "samples.json"
            json_manifest.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "json-positional",
                            "pdf_type": "formula-heavy",
                            "tags": ["formula", "ocr"],
                            "notes": "无 source_pdf 时按 PDF 入参顺序补位",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            json_report = run_batch_experiment(
                [json_pdf],
                root / "json-experiment",
                AppConfig.from_env(),
                variants=parse_variant_specs("page"),
                backend="echo",
                pages_per_chunk=1,
                overlap_pages=0,
                max_chunks=1,
                sample_metadata=load_sample_metadata(json_manifest),
            )
            self.assertEqual(json_report["samples"][0]["sample_id"], "json-positional")
            self.assertEqual(json_report["samples"][0]["pdf_type"], "formula-heavy")
            self.assertEqual(json_report["samples"][0]["tags"], ["formula", "ocr"])

            tsv_manifest = root / "samples.tsv"
            tsv_manifest.write_text(
                "source_pdf\tsample_id\tpdf_type\ttags\tnotes\n"
                "tsv-sample.pdf\ttsv-relative\tscanned\tscan|table\tTSV 相对路径匹配\n",
                encoding="utf-8",
            )
            tsv_report = run_batch_experiment(
                [tsv_pdf],
                root / "tsv-experiment",
                AppConfig.from_env(),
                variants=parse_variant_specs("page"),
                backend="echo",
                pages_per_chunk=1,
                overlap_pages=0,
                max_chunks=1,
                sample_metadata=load_sample_metadata(tsv_manifest),
            )
            self.assertEqual(tsv_report["samples"][0]["sample_id"], "tsv-relative")
            self.assertEqual(tsv_report["samples"][0]["pdf_type"], "scanned")
            self.assertEqual(tsv_report["samples"][0]["tags"], ["scan", "table"])
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)


if __name__ == "__main__":
    unittest.main()
