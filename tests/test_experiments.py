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

    @staticmethod
    def _write_annotation_entity_pdf(path: Path) -> None:
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        text = "\n".join(
            [
                "Smith (2024) and Lee (2023) evaluated BERT, GPT-4, ImageNet, and CLIP.",
                "Stanford University collaborated with Massachusetts Institute of Technology.",
                "Fig. 1 Overview of the proposed workflow.",
                "Fig. 2 Error analysis for entity preservation.",
                "Fig. 3 Annotation layout examples.",
                "* Corresponding author: scholar@stanford.edu",
                "1 Department of Computer Science, Stanford University",
                "2 School of Engineering, Massachusetts Institute of Technology",
                "Author contributions: Smith designed the study and Lee verified the system.",
            ]
        )
        page.insert_text((72, 72), text)
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
            annotation_pdf = root / "annotation-entity.pdf"
            self._write_table_heavy_pdf(table_pdf)
            self._write_scanned_like_pdf(scanned_pdf)
            self._write_annotation_entity_pdf(annotation_pdf)

            manifest_path = root / "samples.csv"
            report_path = root / "samples.json"
            markdown_path = root / "samples.md"
            manifest = write_sample_manifest(
                [table_pdf, scanned_pdf, annotation_pdf],
                manifest_path,
                report_path=report_path,
                markdown_path=markdown_path,
            )

            self.assertTrue(manifest_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertTrue(markdown_path.is_file())
            self.assertEqual(manifest["schema_version"], "experiment-sample-manifest-v1")
            self.assertEqual(manifest["sample_count"], 3)
            by_name = {Path(sample["source_pdf"]).name: sample for sample in manifest["samples"]}
            self.assertEqual(by_name["table-heavy.pdf"]["pdf_type"], "table-heavy")
            self.assertIn("table", by_name["table-heavy.pdf"]["tags"])
            self.assertEqual(by_name["scan.pdf"]["pdf_type"], "scanned")
            self.assertIn("scanned", by_name["scan.pdf"]["tags"])
            self.assertEqual(by_name["annotation-entity.pdf"]["pdf_type"], "annotation-entity-heavy")
            self.assertIn("annotation", by_name["annotation-entity.pdf"]["tags"])
            self.assertIn("entity", by_name["annotation-entity.pdf"]["tags"])
            self.assertGreaterEqual(
                by_name["annotation-entity.pdf"]["metrics"]["annotation_marker_count"],
                2,
            )
            self.assertGreaterEqual(
                by_name["annotation-entity.pdf"]["metrics"]["entity_candidate_count"],
                6,
            )
            coverage = manifest["summary"]["coverage"]
            self.assertEqual(coverage["counts"]["table-heavy"], 1)
            self.assertEqual(coverage["counts"]["scanned"], 1)
            self.assertEqual(coverage["counts"]["annotation-entity-heavy"], 1)
            self.assertFalse(coverage["ready_for_patent_batch"])
            self.assertIn("normal", coverage["missing_counts"])
            markdown_text = markdown_path.read_text(encoding="utf-8")
            self.assertIn("# 跑批样本覆盖度报告", markdown_text)
            self.assertIn("注释/实体密集论文", markdown_text)
            self.assertIn("annotation-entity-heavy", markdown_text)
            self.assertIn("annotations=", markdown_text)
            self.assertIn("entities=", markdown_text)
            csv_text = manifest_path.read_text(encoding="utf-8-sig")
            self.assertIn("confirmed_pdf_type", csv_text)
            self.assertIn("include_in_patent_batch", csv_text)

            metadata = load_sample_metadata(manifest_path)
            self.assertEqual(metadata["table-heavy.pdf"]["pdf_type"], "table-heavy")
            self.assertEqual(metadata["scan.pdf"]["pdf_type"], "scanned")
            self.assertEqual(metadata["annotation-entity.pdf"]["pdf_type"], "annotation-entity-heavy")

            with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                if row["source_pdf"].endswith("annotation-entity.pdf"):
                    row["confirmed_pdf_type"] = "normal"
                    row["confirmed_tags"] = "normal;manual-confirmed"
                    row["include_in_patent_batch"] = "否"
                    row["reviewer"] = "导师"
                    row["review_notes"] = "确认不作为注释实体密集样本"
            with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            confirmed_metadata = load_sample_metadata(manifest_path)
            self.assertEqual(confirmed_metadata["annotation-entity.pdf"]["pdf_type"], "normal")
            self.assertEqual(confirmed_metadata["annotation-entity.pdf"]["tags"], ("normal", "manual-confirmed"))
            self.assertEqual(confirmed_metadata["annotation-entity.pdf"]["include_in_patent_batch"], "否")
            self.assertEqual(confirmed_metadata["annotation-entity.pdf"]["reviewer"], "导师")
            self.assertEqual(confirmed_metadata["annotation-entity.pdf"]["review_notes"], "确认不作为注释实体密集样本")
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
            excluded_pdf = root / "excluded.pdf"
            self._write_sample_pdf(pdf_path)
            self._write_sample_pdf(excluded_pdf)

            metadata_path = root / "samples.csv"
            metadata_path.write_text(
                "source_pdf,sample_id,pdf_type,tags,confirmed_pdf_type,confirmed_tags,"
                "include_in_patent_batch,reviewer,review_notes,notes\n"
                "sample.pdf,sample-table,normal,normal,table-heavy,table;entity,"
                "是,导师,确认作为表格样本,用于表格与实体保留实验\n"
                "excluded.pdf,sample-excluded,normal,normal,normal,normal,"
                "否,导师,暂不进入本轮专利跑批,用于过滤测试\n",
                encoding="utf-8",
            )

            output_dir = root / "experiment"
            report = run_batch_experiment(
                [pdf_path, excluded_pdf],
                output_dir,
                AppConfig.from_env(),
                variants=parse_variant_specs("page,structure"),
                backend="echo",
                pages_per_chunk=1,
                overlap_pages=0,
                max_chunks=1,
                sample_metadata=load_sample_metadata(metadata_path),
                patent_batch_only=True,
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
            self.assertEqual(report["input_pdf_count"], 2)
            self.assertTrue(report["sample_filter"]["patent_batch_only"])
            self.assertEqual(report["sample_filter"]["selected_sample_count"], 1)
            self.assertEqual(report["sample_filter"]["skipped_sample_count"], 1)
            self.assertEqual(report["run_count"], 2)
            self.assertEqual(report["succeeded_count"], 2)
            self.assertEqual(report["failed_count"], 0)
            self.assertEqual(report["baseline_variant"], "page")
            self.assertEqual({item["variant"] for item in report["aggregates"]}, {"page", "structure"})
            self.assertEqual(report["comparisons"][0]["variant"], "structure")
            self.assertEqual(report["samples"][0]["sample_id"], "sample-table")
            self.assertNotIn("sample-excluded", json.dumps(report, ensure_ascii=False))
            self.assertEqual(report["samples"][0]["pdf_type"], "table-heavy")
            self.assertEqual(report["samples"][0]["tags"], ["table", "entity"])
            self.assertEqual(report["samples"][0]["include_in_patent_batch"], "是")
            self.assertEqual(report["samples"][0]["reviewer"], "导师")
            self.assertEqual(report["samples"][0]["review_notes"], "确认作为表格样本")

            loaded = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(loaded["records"][0]["status"], "succeeded")
            self.assertEqual(loaded["sample_filter"]["skipped_sample_count"], 1)
            self.assertEqual(loaded["records"][0]["pdf_type"], "table-heavy")
            self.assertEqual(loaded["records"][0]["include_in_patent_batch"], "是")
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
            self.assertIn("repair_patch_review_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("repair_patch_review_required_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("repair_patch_review_blocking_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("repair_patch_review_required_rate", loaded["records"][0]["metrics"]["rates"])
            self.assertIn(
                "repair_patch_review_default_decision_counts",
                loaded["records"][0]["metrics"]["breakdowns"],
            )
            self.assertIn("repair_publish_confirmed", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("repair_publish_published", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("repair_publish_open_issue_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("repair_publish_rate", loaded["records"][0]["metrics"]["rates"])
            self.assertIn("repair_publish_status_counts", loaded["records"][0]["metrics"]["breakdowns"])
            self.assertIn("ocr_structured_table_candidate_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("ocr_structured_table_gate_review_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("ocr_structured_table_promotion_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("ocr_structured_table_gate_pass_rate", loaded["records"][0]["metrics"]["rates"])
            self.assertIn("ocr_structured_table_promotion_rate", loaded["records"][0]["metrics"]["rates"])
            self.assertIn("ocr_table_cell_bbox_coverage_rate", loaded["records"][0]["metrics"]["rates"])
            self.assertIn(
                "ocr_candidate_structured_table_gate_issue_counts",
                loaded["records"][0]["metrics"]["breakdowns"],
            )
            self.assertIn("ocr_structured_formula_candidate_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("ocr_structured_formula_gate_review_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("ocr_structured_formula_promotion_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("ocr_structured_formula_token_count", loaded["records"][0]["metrics"]["quality"])
            self.assertIn("ocr_structured_formula_gate_pass_rate", loaded["records"][0]["metrics"]["rates"])
            self.assertIn("ocr_structured_formula_promotion_rate", loaded["records"][0]["metrics"]["rates"])
            self.assertIn(
                "ocr_candidate_structured_formula_gate_issue_counts",
                loaded["records"][0]["metrics"]["breakdowns"],
            )
            self.assertIn("breakdowns", loaded["aggregates"][0])
            self.assertIn("structure_hint_merged_cell_candidate_type_counts", loaded["aggregates"][0]["breakdowns"])
            self.assertIn("table_merged_cell_candidate_type_counts", loaded["aggregates"][0]["breakdowns"])
            self.assertIn("repair_merge_strategy_counts", loaded["aggregates"][0]["breakdowns"])
            self.assertIn("repair_patch_review_default_decision_counts", loaded["aggregates"][0]["breakdowns"])
            self.assertIn("repair_publish_status_counts", loaded["aggregates"][0]["breakdowns"])
            self.assertIn("table_chain_reject_reason_category_counts", loaded["aggregates"][0]["breakdowns"])
            self.assertIn("ocr_candidate_structured_table_gate_issue_counts", loaded["aggregates"][0]["breakdowns"])
            self.assertIn("ocr_candidate_structured_formula_gate_issue_counts", loaded["aggregates"][0]["breakdowns"])
            self.assertIn("rates.ocr_structured_table_gate_pass_rate", loaded["comparisons"][0]["deltas"])
            self.assertIn("rates.ocr_structured_table_promotion_rate", loaded["comparisons"][0]["deltas"])
            self.assertIn("rates.ocr_structured_formula_gate_pass_rate", loaded["comparisons"][0]["deltas"])
            self.assertIn("rates.ocr_structured_formula_promotion_rate", loaded["comparisons"][0]["deltas"])
            self.assertIn("rates.repair_patch_review_required_rate", loaded["comparisons"][0]["deltas"])
            self.assertIn("rates.repair_publish_rate", loaded["comparisons"][0]["deltas"])
            self.assertIn("total_elapsed_ms", loaded["records"][0]["metrics"]["performance"])
            self.assertIn("runs/sample-table/page/output/experiment_metrics.json", summary_json.read_text(encoding="utf-8"))
            self.assertIn("translated_pdf", loaded["records"][0]["files"])
            self.assertIn("runs/sample-table/page/output/translated_full.pdf", summary_json.read_text(encoding="utf-8"))
            self.assertIn("repair_publish", loaded["records"][0]["files"])
            self.assertIn("runs/sample-table/page/output/repair_publish.json", summary_json.read_text(encoding="utf-8"))
            self.assertIn("repair_patch_review", loaded["records"][0]["files"])
            self.assertIn(
                "runs/sample-table/page/output/repair_patch_review.json",
                summary_json.read_text(encoding="utf-8"),
            )
            summary_text = summary_md.read_text(encoding="utf-8")
            self.assertIn("OCR structured table gate", summary_text)
            self.assertIn("OCR structured formula gate", summary_text)
            self.assertIn("批量实验汇总", summary_text)
            self.assertIn("仅运行人工纳入样本：是", summary_text)
            self.assertIn("跳过样本数：1", summary_text)
            self.assertIn("续表拒绝类别", summary_text)
            self.assertIn("平均合并候选", summary_text)
            self.assertIn("确认作为表格样本", summary_text)
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
            self.assertIn("ocr_structured_table_promotion_rate", review_text)
            self.assertIn("ocr_table_cell_bbox_coverage_rate", review_text)
            self.assertIn("ocr_structured_table_gate_issues", review_text)
            self.assertIn("ocr_structured_formula_candidate_count", review_text)
            self.assertIn("ocr_structured_formula_gate_pass_rate", review_text)
            self.assertIn("ocr_structured_formula_promotion_rate", review_text)
            self.assertIn("ocr_structured_formula_gate_issues", review_text)
            self.assertIn("repair_patch_review_count", review_text)
            self.assertIn("repair_patch_review_required_count", review_text)
            self.assertIn("repair_patch_review_default_decision_counts", review_text)
            self.assertIn("repair_patch_review_report", review_text)
            self.assertIn("repair_publish_confirmed", review_text)
            self.assertIn("repair_publish_published", review_text)
            self.assertIn("repair_publish_open_issue_count", review_text)
            self.assertIn("repair_publish_status_counts", review_text)
            self.assertIn("repair_publish_report", review_text)
            self.assertIn("repair_published_full", review_text)
            self.assertIn("translated_pdf", review_text)
            self.assertIn("table-heavy", review_text)
            self.assertIn("sample-table", review_text)
            self.assertIn("sample_include_in_patent_batch", review_text)
            self.assertIn("sample_review_notes", review_text)
            with review_csv.open("r", encoding="utf-8-sig", newline="") as review_file:
                review_rows = list(csv.DictReader(review_file))
            self.assertEqual(len(review_rows), 2)
            self.assertEqual(review_rows[0]["include_in_patent_evidence"], "")
            self.assertEqual(review_rows[0]["human_score_pdf"], "")
            self.assertEqual(review_rows[0]["sample_include_in_patent_batch"], "是")
            self.assertEqual(review_rows[0]["sample_reviewer"], "导师")
            self.assertEqual(review_rows[0]["sample_review_notes"], "确认作为表格样本")
            self.assertIn("ocr_structured_table_gate_pass_rate", review_rows[0])
            self.assertIn("ocr_structured_table_promotion_rate", review_rows[0])
            self.assertIn("ocr_structured_formula_gate_pass_rate", review_rows[0])
            self.assertIn("ocr_structured_formula_promotion_rate", review_rows[0])
            self.assertIn("repair_patch_review_count", review_rows[0])
            self.assertIn("repair_patch_review_report", review_rows[0])
            self.assertTrue(review_rows[0]["repair_patch_review_report"].endswith("output/repair_patch_review.json"))
            self.assertIn("repair_publish_confirmed", review_rows[0])
            self.assertIn("repair_publish_report", review_rows[0])
            self.assertTrue(review_rows[0]["repair_publish_report"].endswith("output/repair_publish.json"))
            review_rows[0].update(
                {
                    "human_score": "4.5",
                    "human_score_table_readability": "4",
                    "human_score_structure_coherence": "5",
                    "include_in_patent_evidence": "是",
                    "patent_evidence_notes": "结构化表格与公式 OCR 门禁可作为证据",
                    "reviewer": "导师",
                    "ocr_structured_table_candidate_count": "2",
                    "ocr_structured_table_gate_review_count": "1",
                    "ocr_structured_table_gate_pass_rate": "0.75",
                    "ocr_structured_table_promotion_count": "1",
                    "ocr_structured_table_promotion_rate": "0.5",
                    "ocr_table_cell_bbox_coverage_rate": "0.5",
                    "ocr_structured_table_gate_issues": "structured_table_missing_cell_bboxes:1",
                    "ocr_structured_formula_candidate_count": "3",
                    "ocr_structured_formula_gate_review_count": "1",
                    "ocr_structured_formula_gate_pass_rate": "0.667",
                    "ocr_structured_formula_promotion_count": "1",
                    "ocr_structured_formula_promotion_rate": "0.333",
                    "ocr_structured_formula_token_count": "8",
                    "ocr_structured_formula_equation_label_count": "2",
                    "ocr_structured_formula_gate_issues": "structured_formula_missing_equation_labels:1",
                    "repair_patch_review_count": "3",
                    "repair_patch_review_safe_count": "2",
                    "repair_patch_review_required_count": "1",
                    "repair_patch_review_blocking_count": "1",
                    "repair_patch_review_safe_rate": "0.667",
                    "repair_patch_review_required_rate": "0.333",
                    "repair_patch_review_default_decision_counts": "approve_candidate:2; manual_review_required:1",
                    "repair_patch_review_risk_counts": "low:2; high:1",
                    "repair_publish_confirmed": "是",
                    "repair_publish_published": "否",
                    "repair_publish_open_issue_count": "2",
                    "repair_publish_rate": "0",
                    "repair_publish_status_counts": "pending_confirmation:1",
                }
            )
            for field in [
                "repair_patch_review_count",
                "repair_patch_review_safe_count",
                "repair_patch_review_required_count",
                "repair_patch_review_blocking_count",
                "repair_patch_review_safe_rate",
                "repair_patch_review_required_rate",
                "repair_patch_review_default_decision_counts",
                "repair_patch_review_risk_counts",
            ]:
                review_rows[1][field] = ""
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
            self.assertEqual(
                evidence["ocr_structured_table_gate_summary"]["structured_table_promotion_count_total"],
                1.0,
            )
            self.assertEqual(
                evidence["ocr_structured_table_gate_summary"]["structure_promotion_rate"]["average"],
                0.25,
            )
            self.assertEqual(
                evidence["ocr_structured_formula_gate_summary"]["gate_issue_counts"][
                    "structured_formula_missing_equation_labels"
                ],
                1,
            )
            self.assertEqual(
                evidence["ocr_structured_formula_gate_summary"]["structured_formula_candidate_count_total"],
                3,
            )
            self.assertEqual(
                evidence["ocr_structured_formula_gate_summary"]["structured_formula_token_count_total"],
                8,
            )
            self.assertEqual(
                evidence["ocr_structured_formula_gate_summary"]["structured_formula_promotion_count_total"],
                1.0,
            )
            self.assertEqual(
                evidence["ocr_structured_formula_gate_summary"]["structure_promotion_rate"]["average"],
                0.1665,
            )
            self.assertEqual(evidence["repair_publish_summary"]["confirmed_count_total"], 1)
            self.assertEqual(evidence["repair_publish_summary"]["published_count_total"], 0)
            self.assertEqual(evidence["repair_publish_summary"]["open_issue_count_total"], 2)
            self.assertEqual(evidence["repair_publish_summary"]["status_counts"]["pending_confirmation"], 2)
            self.assertEqual(evidence["repair_patch_review_summary"]["patch_count_total"], 3)
            self.assertEqual(evidence["repair_patch_review_summary"]["safe_count_total"], 2)
            self.assertEqual(evidence["repair_patch_review_summary"]["required_count_total"], 1)
            self.assertEqual(evidence["repair_patch_review_summary"]["blocking_count_total"], 1)
            self.assertEqual(evidence["repair_patch_review_summary"]["default_decision_counts"]["approve_candidate"], 2)
            self.assertEqual(evidence["repair_patch_review_summary"]["risk_counts"]["high"], 1)
            self.assertEqual(
                evidence["evidence_candidates"][0]["ocr"]["structured_formula_candidate_count"],
                3.0,
            )
            self.assertEqual(evidence["evidence_candidates"][0]["repair_patch_review"]["patch_count"], 3.0)
            self.assertEqual(evidence["evidence_candidates"][0]["repair_patch_review"]["required_count"], 1.0)
            self.assertTrue(
                evidence["evidence_candidates"][0]["repair_patch_review"]["report_file"].endswith(
                    "output/repair_patch_review.json"
                )
            )
            self.assertEqual(evidence["evidence_candidates"][0]["repair_publish"]["confirmed"], True)
            self.assertEqual(evidence["evidence_candidates"][0]["repair_publish"]["published"], False)
            self.assertEqual(evidence["evidence_candidates"][0]["repair_publish"]["open_issue_count"], 2.0)
            self.assertTrue(
                evidence["evidence_candidates"][0]["repair_publish"]["report_file"].endswith(
                    "output/repair_publish.json"
                )
            )
            self.assertEqual(evidence["evidence_candidates"][0]["reviewer"], "导师")
            self.assertIn("translated_pdf", evidence["evidence_candidates"][0]["files"])
            evidence_text = evidence_md.read_text(encoding="utf-8")
            self.assertIn("结构化表格与公式 OCR 门禁可作为证据", evidence_text)
            self.assertIn("结构化公式候选总数：3", evidence_text)
            self.assertIn("局部修复发布审核", evidence_text)
            self.assertIn("补丁审核记录行数：1", evidence_text)
            self.assertIn("approve_candidate:2", evidence_text)
            self.assertIn("开放合并问题总数：2", evidence_text)
            self.assertIn("pending_confirmation:2", evidence_text)
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
