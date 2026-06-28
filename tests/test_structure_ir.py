from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import fitz

from pdf_translate.chunking import TextChunk
from pdf_translate.chunkers.structure import build_structure_chunks
from pdf_translate.config import AppConfig
from pdf_translate.extractors.document_ir import (
    BlockIR,
    DocumentIR,
    PageIR,
    classify_text_block,
    extract_table_structure,
)
from pdf_translate.qa.repair import build_repair_plan
from pdf_translate.qa.structure import build_structure_qa
from pdf_translate.qa.translation import build_translation_qa
from pdf_translate.pipeline import init_workdir, run_split, run_translate
from pdf_translate.vision.routing import build_vision_route
from pdf_translate.zip_bundle import iter_bundle_files, map_bundle_arcname


class StructureIRTests(unittest.TestCase):
    def test_classify_table_caption_and_footnote(self) -> None:
        table_type = classify_text_block(
            "Model Acc F1 N\nA 91.2 88.1 120\nB 92.4 89.0 130",
            bbox=(40, 100, 520, 180),
            page_height=800,
            line_texts=["Model Acc F1 N", "A 91.2 88.1 120", "B 92.4 89.0 130"],
            line_xs=[[40, 150, 260, 370], [40, 150, 260, 370], [40, 150, 260, 370]],
            span_sizes=[10.0] * 12,
        )
        self.assertEqual(table_type, "table")

        caption_type = classify_text_block(
            "Table 2: Ablation results.",
            bbox=(40, 300, 520, 330),
            page_height=800,
            line_texts=["Table 2: Ablation results."],
            line_xs=[[40]],
            span_sizes=[10.0],
        )
        self.assertEqual(caption_type, "caption")

        footnote_type = classify_text_block(
            "1 Additional implementation details are available online.",
            bbox=(40, 690, 520, 720),
            page_height=800,
            line_texts=["1 Additional implementation details are available online."],
            line_xs=[[40]],
            span_sizes=[8.5],
        )
        self.assertEqual(footnote_type, "footnote")

    def test_structure_chunks_preserve_block_provenance(self) -> None:
        table_meta = {
            "rows": [["Metric", "Acc", "F1"], ["A", "91", "88"]],
            "row_count": 2,
            "column_count": 3,
            "header": ["Metric", "Acc", "F1"],
            "numeric_tokens": ["91", "88"],
            "warnings": ["numeric_dense_table"],
            "confidence": "medium",
        }
        doc_ir = DocumentIR(
            doc_id="sample",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text="Intro\n\nTable",
                    link_count=2,
                    image_count=1,
                    warnings=["table_like_content"],
                    blocks=[
                        BlockIR("p1-b0000", 1, "heading", "1 Introduction", (0, 0, 100, 20), 0),
                        BlockIR(
                            "p1-b0001",
                            1,
                            "table",
                            "Metric Acc F1\nA 91 88",
                            (0, 30, 300, 90),
                            1,
                            meta={"table": table_meta},
                        ),
                    ],
                ),
                PageIR(
                    page_no=2,
                    width=600,
                    height=800,
                    text="More",
                    blocks=[
                        BlockIR("p2-b0000", 2, "paragraph", "More discussion.", (0, 0, 300, 50), 0),
                    ],
                ),
            ],
        )
        chunks = build_structure_chunks(doc_ir, max_pages_per_chunk=1)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].block_ids, ["p1-b0000", "p1-b0001"])
        self.assertEqual(chunks[0].block_types["table"], 1)
        self.assertEqual(chunks[0].link_count, 2)
        self.assertEqual(chunks[0].image_count, 1)
        self.assertIn("table_like_content", chunks[0].warnings)
        self.assertIn("| Metric | Acc | F1 |", chunks[0].text)
        self.assertEqual(chunks[1].pages_0based, [1])

        qa = build_structure_qa(doc_ir)
        self.assertEqual(qa["summary"]["table_count"], 1)
        self.assertEqual(qa["tables"][0]["column_count"], 3)
        self.assertEqual(qa["tables"][0]["numeric_tokens"], ["91", "88"])

    def test_extract_table_structure_returns_dimensions_and_invariants(self) -> None:
        lines = [
            {"spans": [{"text": "Metric", "bbox": [40, 10, 90, 20]}, {"text": "Acc", "bbox": [140, 10, 170, 20]}]},
            {"spans": [{"text": "A", "bbox": [40, 30, 55, 40]}, {"text": "91.2", "bbox": [140, 30, 170, 40]}]},
        ]
        table = extract_table_structure(lines, ["Metric Acc", "A 91.2"])
        self.assertEqual(table["row_count"], 2)
        self.assertEqual(table["column_count"], 2)
        self.assertEqual(table["header"], ["Metric", "Acc"])
        self.assertEqual(table["numeric_tokens"], ["91.2"])

    def test_vision_route_flags_low_text_image_page_for_local_ocr(self) -> None:
        doc_ir = DocumentIR(
            doc_id="vision-sample",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text="Fig. 1",
                    image_count=1,
                    warnings=["low_text_image_heavy_page"],
                    meta={
                        "text_char_count": 6,
                        "text_area_ratio": 0.01,
                        "image_area_ratio": 0.52,
                    },
                    blocks=[
                        BlockIR("p1-b0000", 1, "image", "", (40, 80, 560, 520), 0),
                        BlockIR("p1-b0001", 1, "caption", "Fig. 1 Overview", (60, 540, 500, 570), 1),
                    ],
                )
            ],
        )
        route = build_vision_route(doc_ir)
        self.assertEqual(route["schema_version"], "vision-route-v1")
        self.assertEqual(route["summary"]["routed_page_count"], 1)
        self.assertEqual(route["pages"][0]["action"], "local_ocr")
        self.assertIn("very_low_text", route["pages"][0]["reasons"])
        self.assertEqual(route["pages"][0]["metrics"]["image_count"], 1)

    def test_translation_qa_reports_missing_invariants(self) -> None:
        root = Path.cwd() / "test-output" / "translation-qa"
        if root.exists():
            shutil.rmtree(root)
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(parents=True)
        try:
            chunks = [
                TextChunk(
                    chunk_id="c0000",
                    pages_0based=[0],
                    text=(
                        "Table 1 reports [3].\n"
                        "| Metric | Acc |\n"
                        "| --- | --- |\n"
                        "| A | 91.2% |"
                    ),
                    link_count=0,
                    image_count=0,
                )
            ]
            (chunk_dir / "c0000.md").write_text(
                "---\n{}\n---\n\n表 1 报告了结果。\n| 指标 |\n| --- |\n| A |\n",
                encoding="utf-8",
            )
            report = build_translation_qa(chunks, chunk_dir)
            issue_types = {issue["type"] for issue in report["chunks"][0]["issues"]}
            self.assertIn("missing_numbers", issue_types)
            self.assertIn("missing_references", issue_types)
            self.assertIn("table_shape_mismatch", issue_types)
            self.assertEqual(report["summary"]["issue_count"], 3)

            plan = build_repair_plan(report)
            self.assertEqual(plan["schema_version"], "repair-plan-v1")
            self.assertEqual(plan["summary"]["repair_item_count"], 3)
            actions = {item["action"] for item in plan["items"]}
            self.assertIn("rewrite_with_locked_tokens", actions)
            self.assertIn("repair_table_shape", actions)
            self.assertEqual(plan["summary"]["priority_counts"]["P0"], 3)
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_pipeline_writes_document_ir_and_structure_manifest(self) -> None:
        root = Path.cwd() / "test-output" / "structure-ir"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            pdf_path = root / "sample.pdf"
            doc = fitz.open()
            p1 = doc.new_page(width=595, height=842)
            p1.insert_text((72, 72), "1 Introduction\nThis is a short academic paragraph.")
            p2 = doc.new_page(width=595, height=842)
            p2.insert_text((72, 72), "Table 1: Results\nModel Acc F1 N\nA 91.2 88.1 120")
            pdf_path.write_bytes(doc.tobytes())
            doc.close()

            work_dir = root / "work"
            init_workdir(work_dir)
            run_split(pdf_path, work_dir)
            cfg = AppConfig.from_env()
            out = run_translate(
                work_dir,
                cfg,
                backend="echo",
                pages_per_chunk=1,
                overlap_pages=0,
                max_chunks=1,
                chunk_strategy="structure",
            )

            self.assertTrue(out.is_file())
            ir_path = work_dir / "output" / "document_ir.json"
            manifest_path = work_dir / "output" / "structure_chunks_manifest.json"
            qa_path = work_dir / "output" / "structure_qa.json"
            vision_path = work_dir / "output" / "vision_route.json"
            translation_qa_path = work_dir / "output" / "qa_report.json"
            translation_qa_md_path = work_dir / "output" / "qa_report.md"
            repair_plan_path = work_dir / "output" / "repair_plan.json"
            repair_plan_md_path = work_dir / "output" / "repair_plan.md"
            self.assertTrue(ir_path.is_file())
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(qa_path.is_file())
            self.assertTrue(vision_path.is_file())
            self.assertTrue(translation_qa_path.is_file())
            self.assertTrue(translation_qa_md_path.is_file())
            self.assertTrue(repair_plan_path.is_file())
            self.assertTrue(repair_plan_md_path.is_file())
            ir = json.loads(ir_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
            vision = json.loads(vision_path.read_text(encoding="utf-8"))
            translation_qa = json.loads(translation_qa_path.read_text(encoding="utf-8"))
            repair_plan = json.loads(repair_plan_path.read_text(encoding="utf-8"))
            self.assertEqual(ir["schema_version"], "document-ir-v1")
            self.assertGreaterEqual(len(ir["pages"]), 1)
            self.assertIn("meta", ir["pages"][0])
            self.assertGreaterEqual(len(manifest), 1)
            self.assertIn("block_ids", manifest[0])
            self.assertEqual(qa["schema_version"], "structure-qa-v1")
            self.assertIn("table_count", qa["summary"])
            self.assertGreaterEqual(qa["summary"]["table_count"], 1)
            self.assertEqual(vision["schema_version"], "vision-route-v1")
            self.assertIn("action_counts", vision["summary"])
            self.assertEqual(translation_qa["schema_version"], "translation-qa-v1")
            self.assertIn("issue_counts", translation_qa["summary"])
            self.assertEqual(repair_plan["schema_version"], "repair-plan-v1")
            self.assertIn("repair_item_count", repair_plan["summary"])
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_bundle_includes_structure_qa_and_repair_artifacts(self) -> None:
        root = Path.cwd() / "test-output" / "bundle"
        if root.exists():
            shutil.rmtree(root)
        output = root / "output"
        output.mkdir(parents=True)
        try:
            for name in [
                "translated_full.md",
                "document_ir.json",
                "structure_chunks_manifest.json",
                "structure_qa.json",
                "vision_route.json",
                "qa_report.json",
                "qa_report.md",
                "repair_plan.json",
                "repair_plan.md",
            ]:
                (output / name).write_text("{}", encoding="utf-8")
            rels = {
                path.relative_to(root).as_posix()
                for path in iter_bundle_files(root)
            }
            self.assertIn("output/repair_plan.json", rels)
            self.assertIn("output/qa_report.md", rels)
            self.assertIn("output/document_ir.json", rels)
            self.assertEqual(map_bundle_arcname("output/repair_plan.md"), "质量/局部修复计划.md")
            self.assertEqual(map_bundle_arcname("output/structure_qa.json"), "质量/结构QA.json")
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)


if __name__ == "__main__":
    unittest.main()
