from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
import httpx

from pdf_translate.chunking import TextChunk
from pdf_translate.chunkers.structure import build_structure_chunks
from pdf_translate.config import AppConfig
from pdf_translate.costing import estimate_cost, normalize_cost_profile
from pdf_translate.exporters.bilingual_html import write_bilingual_html
from pdf_translate.extractors.document_ir import (
    BlockIR,
    DocumentIR,
    PageIR,
    assign_block_parents,
    classify_text_block,
    extract_entity_candidates,
    extract_table_structure,
)
from pdf_translate.memory_store import MemoryStore
from pdf_translate.qa.chunk_boundary import build_chunk_boundary_qa, build_chunk_strategy_comparison
from pdf_translate.qa.metrics import build_experiment_metrics
from pdf_translate.qa.repair import (
    build_repair_plan,
    build_repair_requests,
    build_repair_results,
    build_repair_merge,
    build_repair_validation,
)
from pdf_translate.qa.structure import build_structure_qa
from pdf_translate.qa.table_reconstruction import build_table_reconstruction_report, build_table_translation_hints
from pdf_translate.qa.translation import build_translation_qa
from pdf_translate.pipeline import init_workdir, run_split, run_translate
from pdf_translate.run_metrics import build_run_metrics
from pdf_translate.translators.base import TranslationRequest
from pdf_translate.translators.http_retry import call_with_http_retry, capture_http_retry_events
from pdf_translate.translators.openai_compatible import _build_user_message
from pdf_translate.vision.ocr_tasks import build_ocr_task_manifest, write_ocr_task_manifest
from pdf_translate.vision.ocr_writeback import (
    build_ocr_results_payload,
    build_ocr_writeback,
    load_ocr_results,
    write_ocr_results_payload,
    write_ocr_writeback,
)
from pdf_translate.vision.routing import build_vision_route, write_vision_route
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

        figure_caption_type = classify_text_block(
            "Fig. 1 Overview.",
            bbox=(40, 340, 520, 365),
            page_height=800,
            line_texts=["Fig. 1 Overview."],
            line_xs=[[40]],
            span_sizes=[10.0],
        )
        self.assertEqual(figure_caption_type, "caption")

        footnote_type = classify_text_block(
            "1 Additional implementation details are available online.",
            bbox=(40, 690, 520, 720),
            page_height=800,
            line_texts=["1 Additional implementation details are available online."],
            line_xs=[[40]],
            span_sizes=[8.5],
        )
        self.assertEqual(footnote_type, "footnote")

    def test_extract_entity_candidates_for_academic_names(self) -> None:
        entities = extract_entity_candidates(
            "Smith et al. (2024) evaluated BERT and ImageNet at Stanford University with CNN baselines."
        )
        by_text = {item["text"]: item for item in entities}
        self.assertEqual(by_text["Smith"]["type"], "person")
        self.assertEqual(by_text["BERT"]["type"], "model_or_dataset")
        self.assertEqual(by_text["ImageNet"]["type"], "model_or_dataset")
        self.assertEqual(by_text["Stanford University"]["type"], "organization")
        self.assertEqual(by_text["CNN"]["type"], "acronym")

    def test_assign_block_parents_links_captions_and_footnotes(self) -> None:
        image = BlockIR("p1-b0000", 1, "image", "", (40, 80, 560, 300), 0)
        figure_caption = BlockIR("p1-b0001", 1, "caption", "Fig. 1 Overview.", (60, 320, 500, 345), 1)
        table_caption = BlockIR("p1-b0002", 1, "caption", "Table 1: Results", (60, 360, 500, 385), 2)
        table = BlockIR(
            "p1-b0003",
            1,
            "table",
            "Metric Acc\nA 91.2",
            (60, 390, 500, 460),
            3,
            meta={"table": {"row_count": 2, "column_count": 2}},
        )
        table_footnote = BlockIR("p1-b0004", 1, "footnote", "1 Standard deviation in parentheses.", (60, 470, 500, 490), 4)
        paragraph = BlockIR("p1-b0005", 1, "paragraph", "The method is robust.", (60, 500, 500, 545), 5)
        footnote = BlockIR("p1-b0006", 1, "footnote", "2 Additional implementation note.", (60, 700, 500, 730), 6)
        orphan_caption = BlockIR("p1-b0007", 1, "caption", "Algorithm note", (60, 745, 500, 765), 7)
        blocks = [image, figure_caption, table_caption, table, table_footnote, paragraph, footnote, orphan_caption]

        assign_block_parents(blocks)

        self.assertEqual(figure_caption.parent_id, image.block_id)
        self.assertEqual(figure_caption.meta["caption_kind"], "figure")
        self.assertEqual(figure_caption.meta["parent_relation"], "caption_for_figure")
        self.assertEqual(table_caption.parent_id, table.block_id)
        self.assertEqual(table_caption.meta["caption_kind"], "table")
        self.assertEqual(table_caption.meta["parent_relation"], "caption_for_table")
        self.assertEqual(table_footnote.parent_id, table.block_id)
        self.assertEqual(table_footnote.meta["parent_relation"], "footnote_for_table")
        self.assertTrue(table_footnote.meta["table_footnote"])
        self.assertEqual(footnote.parent_id, paragraph.block_id)
        self.assertEqual(footnote.meta["parent_relation"], "footnote_for_block")
        self.assertIsNone(orphan_caption.parent_id)
        self.assertEqual(orphan_caption.meta["parent_warning"], "orphan_caption")

        qa = build_structure_qa(
            DocumentIR(
                doc_id="relationships",
                source_pdf="sample.pdf",
                pages=[
                    PageIR(
                        page_no=1,
                        width=600,
                        height=800,
                        text="sample",
                        blocks=blocks,
                    )
                ],
            )
        )
        self.assertEqual(qa["summary"]["caption_count"], 3)
        self.assertEqual(qa["summary"]["caption_linked_count"], 2)
        self.assertEqual(qa["summary"]["caption_orphan_count"], 1)
        self.assertEqual(qa["summary"]["footnote_count"], 2)
        self.assertEqual(qa["summary"]["footnote_linked_count"], 2)
        self.assertEqual(qa["summary"]["table_footnote_count"], 1)
        self.assertEqual(qa["summary"]["relationship_count"], 4)
        self.assertEqual(qa["summary"]["relationship_warning_count"], 1)
        warnings = [item for item in qa["relationships"] if item["warning"]]
        self.assertEqual(warnings[0]["block_id"], orphan_caption.block_id)

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
        self.assertIn("page_boundary_fragment_count", qa["summary"])

    def test_structure_qa_reports_entity_candidates(self) -> None:
        doc_ir = DocumentIR(
            doc_id="entities",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text="Smith et al. (2024) evaluated BERT at Stanford University.",
                    blocks=[
                        BlockIR(
                            "p1-b0000",
                            1,
                            "paragraph",
                            "Smith et al. (2024) evaluated BERT at Stanford University.",
                            (40, 100, 520, 160),
                            0,
                            meta={
                                "entities": extract_entity_candidates(
                                    "Smith et al. (2024) evaluated BERT at Stanford University."
                                )
                            },
                        ),
                    ],
                )
            ],
        )
        qa = build_structure_qa(doc_ir)
        self.assertEqual(qa["summary"]["entity_candidate_count"], 3)
        self.assertEqual(qa["summary"]["entity_unique_count"], 3)
        self.assertEqual(qa["summary"]["entity_type_counts"]["person"], 1)
        self.assertEqual(qa["summary"]["entity_type_counts"]["model_or_dataset"], 1)
        self.assertEqual(qa["summary"]["entity_type_counts"]["organization"], 1)
        self.assertEqual({item["text"] for item in qa["entities"]}, {"Smith", "BERT", "Stanford University"})

    def test_structure_qa_reports_table_continuations(self) -> None:
        doc_ir = DocumentIR(
            doc_id="continued-table",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text="Metric Acc\nA 91",
                    blocks=[
                        BlockIR(
                            "p1-b0000",
                            1,
                            "table",
                            "Metric Acc\nA 91",
                            (40, 640, 540, 760),
                            0,
                            meta={"table": {"row_count": 2, "column_count": 2, "confidence": "medium"}},
                        ),
                    ],
                ),
                PageIR(
                    page_no=2,
                    width=600,
                    height=800,
                    text="B 92\nC 93",
                    blocks=[
                        BlockIR(
                            "p2-b0000",
                            2,
                            "table",
                            "B 92\nC 93",
                            (40, 80, 540, 180),
                            0,
                            meta={"table": {"row_count": 2, "column_count": 2, "confidence": "medium"}},
                        ),
                    ],
                ),
            ],
        )
        qa = build_structure_qa(doc_ir)
        self.assertEqual(qa["summary"]["page_boundary_fragment_count"], 1)
        self.assertEqual(qa["summary"]["table_continuation_count"], 1)
        self.assertEqual(qa["table_continuations"][0]["previous_table_block_id"], "p1-b0000")
        self.assertEqual(qa["table_continuations"][0]["next_table_block_id"], "p2-b0000")
        self.assertEqual(qa["tables"][0]["continued_to_block_id"], "p2-b0000")
        self.assertEqual(qa["tables"][1]["continued_from_block_id"], "p1-b0000")

        chunks = build_structure_chunks(
            doc_ir,
            target_chars=1000,
            max_chars=2000,
            max_pages_per_chunk=1,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].boundary_fragment_ids, ["p1-p2"])
        self.assertIn("protected_page_boundary:p1-p2", chunks[0].warnings)

    def test_table_reconstruction_report_builds_cell_contexts(self) -> None:
        table_caption = BlockIR("p1-b0000", 1, "caption", "Table 1: Results", (40, 80, 520, 100), 0)
        table = BlockIR(
            "p1-b0001",
            1,
            "table",
            "Model Acc p\nBERT 91.2% p<0.05*",
            (40, 110, 520, 180),
            1,
            meta={
                "table": {
                    "rows": [["Model", "Acc", "p"], ["BERT", "91.2%", "p<0.05*"]],
                    "row_count": 2,
                    "column_count": 3,
                    "header": ["Model", "Acc", "p"],
                    "numeric_tokens": ["91.2%", "0.05"],
                    "confidence": "medium",
                }
            },
        )
        table_footnote = BlockIR("p1-b0002", 1, "footnote", "1 * p<0.05.", (40, 190, 520, 210), 2)
        blocks = [table_caption, table, table_footnote]
        assign_block_parents(blocks)
        doc_ir = DocumentIR(
            doc_id="table-reconstruction",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text="table sample",
                    blocks=blocks,
                )
            ],
        )
        structure_qa = build_structure_qa(doc_ir)

        report = build_table_reconstruction_report(doc_ir, structure_qa)

        self.assertEqual(report["schema_version"], "table-reconstruction-v1")
        self.assertEqual(report["summary"]["table_count"], 1)
        self.assertEqual(report["summary"]["reconstructable_table_count"], 1)
        self.assertEqual(report["summary"]["cell_count"], 6)
        self.assertEqual(report["summary"]["numeric_cell_count"], 2)
        self.assertEqual(report["summary"]["unit_token_count"], 1)
        self.assertEqual(report["summary"]["significance_token_count"], 2)
        self.assertEqual(report["summary"]["caption_linked_table_count"], 1)
        self.assertEqual(report["summary"]["footnote_linked_table_count"], 1)
        self.assertEqual(report["summary"]["table_reconstruction_ready_rate"], 1.0)
        table_report = report["tables"][0]
        self.assertEqual(table_report["caption_blocks"][0]["block_id"], "p1-b0000")
        self.assertEqual(table_report["footnote_blocks"][0]["block_id"], "p1-b0002")
        acc_cell = next(cell for cell in table_report["cells"] if cell["row_index"] == 1 and cell["column_index"] == 1)
        self.assertEqual(acc_cell["column_header"], "Acc")
        self.assertEqual(acc_cell["row_header"], "BERT")
        self.assertIn("91.2%", acc_cell["locked_tokens"])
        self.assertIn("%", acc_cell["locked_tokens"])

    def test_table_translation_hints_are_chunk_scoped(self) -> None:
        page1_table = BlockIR(
            "p1-b0000",
            1,
            "table",
            "Model Acc\nBERT 91.2%",
            (40, 110, 520, 180),
            0,
            meta={
                "table": {
                    "rows": [["Model", "Acc"], ["BERT", "91.2%"]],
                    "row_count": 2,
                    "column_count": 2,
                    "header": ["Model", "Acc"],
                    "confidence": "medium",
                }
            },
        )
        page2_table = BlockIR(
            "p2-b0000",
            2,
            "table",
            "Dataset F1\nCOCO 88.1",
            (40, 110, 520, 180),
            0,
            meta={
                "table": {
                    "rows": [["Dataset", "F1"], ["COCO", "88.1"]],
                    "row_count": 2,
                    "column_count": 2,
                    "header": ["Dataset", "F1"],
                    "confidence": "medium",
                }
            },
        )
        doc_ir = DocumentIR(
            doc_id="table-hints",
            source_pdf="sample.pdf",
            pages=[
                PageIR(1, 600, 800, "Model Acc\nBERT 91.2%", [page1_table]),
                PageIR(2, 600, 800, "Dataset F1\nCOCO 88.1", [page2_table]),
            ],
        )
        report = build_table_reconstruction_report(doc_ir, build_structure_qa(doc_ir))
        chunk = TextChunk("c0000", [0], "Model Acc\nBERT 91.2%", 0, 0)

        hints = build_table_translation_hints(chunk, report)

        self.assertIn("DocumentIR", hints)
        self.assertIn("表格 p1-b0000", hints)
        self.assertIn("2 行 x 2 列", hints)
        self.assertIn("91.2%", hints)
        self.assertIn("Markdown", hints)
        self.assertNotIn("88.1", hints)

    def test_openai_user_message_includes_structure_hints(self) -> None:
        message = _build_user_message(
            TranslationRequest(
                source_text="| Model | Acc |\n| --- | --- |\n| BERT | 91.2% |",
                glossary_excerpt="",
                prior_summaries="",
                style_notes="",
                structure_hints="表格 p1-b0000：2 行 x 2 列；锁定 token：91.2%。",
            )
        )

        self.assertIn("【结构保护提示】", message)
        self.assertIn("锁定 token：91.2%", message)
        self.assertLess(message.index("【结构保护提示】"), message.index("【待译正文】"))

    def test_translation_qa_reports_table_cell_token_mismatch(self) -> None:
        root = Path.cwd() / "test-output" / "table-cell-token-qa"
        if root.exists():
            shutil.rmtree(root)
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(parents=True)
        try:
            table = BlockIR(
                "p1-b0000",
                1,
                "table",
                "Model Acc p\nBERT 91.2% p<0.05",
                (40, 110, 520, 180),
                0,
                meta={
                    "table": {
                        "rows": [["Model", "Acc", "p"], ["BERT", "91.2%", "p<0.05"]],
                        "row_count": 2,
                        "column_count": 3,
                        "header": ["Model", "Acc", "p"],
                        "confidence": "medium",
                    }
                },
            )
            doc_ir = DocumentIR(
                doc_id="table-cell-token-qa",
                source_pdf="sample.pdf",
                pages=[PageIR(1, 600, 800, table.text, [table])],
            )
            table_reconstruction = build_table_reconstruction_report(doc_ir, build_structure_qa(doc_ir))
            chunks = [
                TextChunk(
                    "c0000",
                    [0],
                    "Model Acc p\nBERT 91.2% p<0.05",
                    0,
                    0,
                )
            ]
            (chunk_dir / "c0000.md").write_text(
                (
                    "---\n{}\n---\n\n"
                    "| 模型 | 准确率 | p |\n"
                    "| --- | --- | --- |\n"
                    "| BERT | 91.2 | p<0.05 |\n\n"
                    "注：91.2% 为原始准确率。\n"
                ),
                encoding="utf-8",
            )

            report = build_translation_qa(
                chunks,
                chunk_dir,
                table_reconstruction=table_reconstruction,
            )

            issue = next(
                issue
                for issue in report["chunks"][0]["issues"]
                if issue["type"] == "table_cell_token_mismatch"
            )
            cell = issue["cells"][0]
            self.assertEqual(report["summary"]["source_table_count"], 1)
            self.assertEqual(report["summary"]["source_table_locked_token_count"], 4)
            self.assertEqual(report["summary"]["table_cell_token_error_count"], 1)
            self.assertEqual(report["summary"]["missing_table_locked_token_count"], 2)
            self.assertEqual(cell["table_id"], "p1-b0000")
            self.assertEqual(cell["row_index"], 1)
            self.assertEqual(cell["column_index"], 1)
            self.assertEqual(cell["column_header"], "Acc")
            self.assertIn("91.2%", cell["missing_tokens"])
            self.assertIn("%", cell["missing_tokens"])

            plan = build_repair_plan(report)
            repair = next(item for item in plan["items"] if item["issue_type"] == "table_cell_token_mismatch")
            self.assertEqual(repair["action"], "repair_table_cell_tokens")
            self.assertEqual(repair["scope"], "table_cell")
            self.assertEqual(repair["priority"], "P0")
            self.assertIn("cells", repair["evidence"])

            requests = build_repair_requests(plan, chunks, chunk_dir)
            self.assertEqual(requests["schema_version"], "repair-requests-v1")
            self.assertEqual(requests["summary"]["repair_request_count"], 1)
            self.assertEqual(requests["summary"]["ready_for_translation_backend_count"], 1)
            request = requests["requests"][0]
            self.assertEqual(request["action"], "repair_table_cell_tokens")
            self.assertEqual(request["status"], "ready_for_translation_backend")
            self.assertIn("91.2%", request["locked_tokens"])
            self.assertIn("对应单元格", request["instruction"])
            self.assertIn("【QA 证据】", request["backend_payload"]["user_message"])
            self.assertIn("只输出修复后的中文译文或 Markdown 表格", request["backend_payload"]["user_message"])

            skipped_results = build_repair_results(requests, execute=False)
            self.assertEqual(skipped_results["schema_version"], "repair-results-v1")
            self.assertEqual(skipped_results["summary"]["skipped_count"], 1)
            self.assertEqual(skipped_results["results"][0]["status"], "skipped_execution_disabled")
            skipped_validation = build_repair_validation(requests, skipped_results)
            self.assertEqual(skipped_validation["schema_version"], "repair-validation-v1")
            self.assertEqual(skipped_validation["summary"]["skipped_count"], 1)
            self.assertEqual(skipped_validation["validations"][0]["status"], "skipped_not_succeeded")

            class DummyRepairTranslator:
                name = "dummy-repair"

                def translate(self, req: TranslationRequest) -> str:
                    self.last_source = req.source_text
                    return "| 模型 | 准确率 | p |\n| --- | --- | --- |\n| BERT | 91.2% | p<0.05 |"

            dummy = DummyRepairTranslator()
            executed_results = build_repair_results(
                requests,
                translator=dummy,
                execute=True,
                repairs_dir=root / "repairs",
            )
            self.assertEqual(executed_results["summary"]["executed_request_count"], 1)
            self.assertEqual(executed_results["summary"]["succeeded_count"], 1)
            self.assertEqual(executed_results["results"][0]["status"], "succeeded")
            self.assertIn("91.2%", executed_results["results"][0]["result_excerpt"])
            self.assertIn("【修复目标】", dummy.last_source)
            self.assertTrue((root / "repairs" / "rq0000.md").is_file())
            validation = build_repair_validation(requests, executed_results)
            self.assertEqual(validation["summary"]["validated_result_count"], 1)
            self.assertEqual(validation["summary"]["passed_count"], 1)
            self.assertEqual(validation["summary"]["missing_locked_token_count"], 0)
            self.assertEqual(validation["summary"]["locked_token_pass_rate"], 1.0)
            self.assertEqual(validation["validations"][0]["status"], "passed")
            merge = build_repair_merge(
                requests,
                executed_results,
                validation,
                chunks,
                chunk_dir,
                repaired_chunk_dir=root / "repaired_chunks",
                repaired_full_path=root / "repaired_full.md",
            )
            self.assertEqual(merge["schema_version"], "repair-merge-v1")
            self.assertEqual(merge["summary"]["merge_candidate_count"], 1)
            self.assertEqual(merge["summary"]["applied_count"], 1)
            self.assertEqual(merge["summary"]["patched_chunk_count"], 1)
            self.assertEqual(merge["patches"][0]["strategy"], "replace_first_markdown_table")
            repaired_chunk = (root / "repaired_chunks" / "c0000.md").read_text(encoding="utf-8")
            self.assertIn("| BERT | 91.2% | p<0.05 |", repaired_chunk)
            self.assertTrue((root / "repaired_full.md").is_file())
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_structure_qa_reports_page_boundary_fragments(self) -> None:
        doc_ir = DocumentIR(
            doc_id="boundary-sample",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text="The proposed method improves",
                    blocks=[
                        BlockIR(
                            "p1-b0000",
                            1,
                            "paragraph",
                            "The proposed method improves",
                            (40, 100, 520, 180),
                            0,
                        ),
                    ],
                ),
                PageIR(
                    page_no=2,
                    width=600,
                    height=800,
                    text="accuracy under domain shift.",
                    blocks=[
                        BlockIR(
                            "p2-b0000",
                            2,
                            "paragraph",
                            "accuracy under domain shift.",
                            (40, 80, 520, 140),
                            0,
                        ),
                    ],
                ),
            ],
        )
        qa = build_structure_qa(doc_ir)
        self.assertEqual(qa["summary"]["page_boundary_fragment_count"], 1)
        self.assertEqual(qa["summary"]["page_boundary_fragment_rate"], 1.0)
        fragment = qa["page_boundary_fragments"][0]
        self.assertEqual(fragment["pages_1based"], [1, 2])
        self.assertEqual(fragment["severity"], "high")
        self.assertEqual(fragment["previous_block_id"], "p1-b0000")
        self.assertEqual(fragment["next_block_id"], "p2-b0000")
        self.assertIn("previous_page_ends_without_terminal_punctuation", fragment["reasons"])
        self.assertIn("next_page_starts_like_continuation", fragment["reasons"])
        self.assertIn("The proposed method improves", fragment["previous_tail"])
        self.assertIn("accuracy under domain shift", fragment["next_head"])

    def test_structure_chunks_protect_page_boundary_fragments(self) -> None:
        long_unfinished = "The proposed method improves " + ("robustness " * 105)
        doc_ir = DocumentIR(
            doc_id="protected-boundary-sample",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text=long_unfinished,
                    blocks=[
                        BlockIR(
                            "p1-b0000",
                            1,
                            "paragraph",
                            long_unfinished,
                            (40, 100, 520, 760),
                            0,
                        ),
                    ],
                ),
                PageIR(
                    page_no=2,
                    width=600,
                    height=800,
                    text="accuracy under domain shift.",
                    blocks=[
                        BlockIR(
                            "p2-b0000",
                            2,
                            "paragraph",
                            "accuracy under domain shift.",
                            (40, 80, 520, 140),
                            0,
                        ),
                    ],
                ),
            ],
        )
        chunks = build_structure_chunks(
            doc_ir,
            target_chars=1000,
            max_chars=2000,
            max_pages_per_chunk=1,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].pages_0based, [0, 1])
        self.assertEqual(chunks[0].boundary_fragment_ids, ["p1-p2"])
        self.assertIn("protected_page_boundary:p1-p2", chunks[0].warnings)
        self.assertIn("accuracy under domain shift", chunks[0].text)

    def test_chunk_boundary_qa_reports_split_and_protected_fragments(self) -> None:
        doc_ir = DocumentIR(
            doc_id="chunk-boundary-sample",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text="The proposed method improves",
                    blocks=[
                        BlockIR(
                            "p1-b0000",
                            1,
                            "paragraph",
                            "The proposed method improves",
                            (40, 100, 520, 180),
                            0,
                        ),
                    ],
                ),
                PageIR(
                    page_no=2,
                    width=600,
                    height=800,
                    text="accuracy under domain shift.",
                    blocks=[
                        BlockIR(
                            "p2-b0000",
                            2,
                            "paragraph",
                            "accuracy under domain shift.",
                            (40, 80, 520, 140),
                            0,
                        ),
                    ],
                ),
            ],
        )
        structure_qa = build_structure_qa(doc_ir)
        page_chunks = [
            TextChunk("c0000", [0], "The proposed method improves", 0, 0),
            TextChunk("c0001", [1], "accuracy under domain shift.", 0, 0),
        ]
        page_report = build_chunk_boundary_qa(page_chunks, structure_qa, pipeline_variant="page")
        self.assertEqual(page_report["schema_version"], "chunk-boundary-qa-v1")
        self.assertEqual(page_report["summary"]["boundary_fragment_count"], 1)
        self.assertEqual(page_report["summary"]["split_boundary_count"], 1)
        self.assertEqual(page_report["summary"]["high_risk_split_count"], 1)
        self.assertEqual(page_report["summary"]["split_boundary_rate"], 1.0)
        self.assertEqual(page_report["boundaries"][0]["status"], "split")

        structure_chunks = build_structure_chunks(
            doc_ir,
            target_chars=1000,
            max_chars=2000,
            max_pages_per_chunk=1,
        )
        structure_report = build_chunk_boundary_qa(
            structure_chunks,
            structure_qa,
            pipeline_variant="structure",
        )
        self.assertEqual(structure_report["summary"]["protected_boundary_count"], 1)
        self.assertEqual(structure_report["summary"]["split_boundary_count"], 0)
        self.assertEqual(structure_report["summary"]["protected_boundary_rate"], 1.0)
        self.assertEqual(structure_report["boundaries"][0]["status"], "protected")
        self.assertEqual(structure_report["boundaries"][0]["protected_by_chunk_ids"], ["c0000"])

        comparison = build_chunk_strategy_comparison(
            {
                "page": page_chunks,
                "structure": structure_chunks,
            },
            structure_qa,
            active_strategy="structure",
        )
        self.assertEqual(comparison["schema_version"], "chunk-strategy-comparison-v1")
        self.assertEqual(comparison["summary"]["baseline_split_boundary_count"], 1)
        self.assertEqual(comparison["summary"]["active_split_boundary_count"], 0)
        self.assertEqual(comparison["summary"]["active_split_reduction_vs_baseline"], 1)
        self.assertEqual(comparison["summary"]["active_split_reduction_rate_vs_baseline"], 1.0)
        self.assertEqual(comparison["summary"]["best_strategy_by_split_rate"], "structure")
        boundary = comparison["boundaries"][0]
        self.assertEqual(boundary["status_by_strategy"]["page"], "split")
        self.assertEqual(boundary["status_by_strategy"]["structure"], "protected")

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

    def test_write_vision_route_renders_routed_page_preview(self) -> None:
        root = Path.cwd() / "test-output" / "vision-preview"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            pdf_path = root / "sample.pdf"
            doc = fitz.open()
            page = doc.new_page(width=300, height=420)
            page.insert_text((40, 40), "Fig. 1")
            pdf_path.write_bytes(doc.tobytes())
            doc.close()

            doc_ir = DocumentIR(
                doc_id="vision-preview",
                source_pdf=str(pdf_path),
                pages=[
                    PageIR(
                        page_no=1,
                        width=300,
                        height=420,
                        text="Fig. 1",
                        image_count=1,
                        warnings=["low_text_image_heavy_page"],
                        meta={
                            "text_char_count": 6,
                            "text_area_ratio": 0.01,
                            "image_area_ratio": 0.52,
                        },
                        blocks=[
                            BlockIR("p1-b0000", 1, "image", "", (40, 80, 260, 300), 0),
                            BlockIR("p1-b0001", 1, "caption", "Fig. 1 Overview", (60, 320, 240, 345), 1),
                        ],
                    )
                ],
            )
            route_path = root / "output" / "vision_route.json"

            route = write_vision_route(doc_ir, route_path)

            self.assertTrue(route_path.is_file())
            self.assertEqual(route["summary"]["preview_page_count"], 1)
            self.assertEqual(route["summary"]["preview_crop_count"], 2)
            evidence = route["pages"][0]["evidence"]
            self.assertEqual(evidence["page_preview_status"], "rendered")
            self.assertEqual(evidence["page_preview_path"], "vision_pages/page-0001.png")
            self.assertGreater(evidence["page_preview_width"], 0)
            self.assertTrue((root / "output" / evidence["page_preview_path"]).is_file())
            self.assertEqual(evidence["region_crop_count"], 2)
            self.assertEqual(
                evidence["region_crops"][0]["crop_path"],
                "vision_crops/page-0001/p1-b0000-image.png",
            )
            self.assertTrue((root / "output" / evidence["region_crops"][0]["crop_path"]).is_file())
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_ocr_task_manifest_schedules_region_crops_for_writeback(self) -> None:
        doc_ir = DocumentIR(
            doc_id="ocr-task-sample",
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
                        BlockIR(
                            "p1-b0001",
                            1,
                            "table",
                            "Metric Acc\nA 91.2",
                            (60, 540, 500, 620),
                            1,
                            meta={"table": {"row_count": 2, "column_count": 2}},
                        ),
                    ],
                )
            ],
        )
        route = build_vision_route(doc_ir)
        route["pages"][0]["evidence"].update(
            {
                "page_preview_status": "rendered",
                "page_preview_path": "vision_pages/page-0001.png",
                "region_crop_count": 2,
                "region_crops": [
                    {
                        "block_id": "p1-b0000",
                        "block_type": "image",
                        "crop_path": "vision_crops/page-0001/p1-b0000-image.png",
                        "bbox": [40, 80, 560, 520],
                        "crop_width": 720,
                        "crop_height": 610,
                    },
                    {
                        "block_id": "p1-b0001",
                        "block_type": "table",
                        "crop_path": "vision_crops/page-0001/p1-b0001-table.png",
                        "bbox": [60, 540, 500, 620],
                        "crop_width": 720,
                        "crop_height": 140,
                    },
                ],
            }
        )

        root = Path.cwd() / "test-output" / "ocr-tasks"
        if root.exists():
            shutil.rmtree(root)
        try:
            manifest_path = root / "output" / "ocr_tasks.json"
            manifest = write_ocr_task_manifest(doc_ir, route, manifest_path)

            self.assertTrue(manifest_path.is_file())
            self.assertEqual(manifest["schema_version"], "ocr-task-manifest-v1")
            self.assertEqual(manifest["summary"]["task_count"], 2)
            self.assertEqual(manifest["summary"]["region_task_count"], 2)
            self.assertEqual(manifest["summary"]["ready_task_count"], 2)
            self.assertEqual(manifest["summary"]["blocked_by_missing_evidence_count"], 0)
            self.assertEqual(manifest["summary"]["recommended_engine_counts"]["local_table_ocr"], 1)
            self.assertEqual(manifest["tasks"][0]["input_path"], "vision_crops/page-0001/p1-b0000-image.png")
            self.assertEqual(manifest["tasks"][0]["writeback"]["target"], "document_ir.block.meta.ocr_candidates")
            self.assertTrue(manifest["tasks"][0]["block_known_in_document_ir"])
            table_task = manifest["tasks"][1]
            self.assertEqual(table_task["block_type"], "table")
            self.assertEqual(table_task["recommended_engine"], "local_table_ocr")
            self.assertEqual(table_task["writeback"]["block_id"], "p1-b0001")
            self.assertEqual(manifest["result_writeback_contract"]["schema_version"], "ocr-result-v1")
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_ocr_writeback_appends_candidates_to_augmented_ir(self) -> None:
        doc_ir = DocumentIR(
            doc_id="ocr-writeback-sample",
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
                        BlockIR(
                            "p1-b0001",
                            1,
                            "table",
                            "Metric Acc\nA 91.2",
                            (60, 540, 500, 620),
                            1,
                            meta={"table": {"row_count": 2, "column_count": 2}},
                        ),
                    ],
                )
            ],
        )
        route = build_vision_route(doc_ir)
        route["pages"][0]["evidence"].update(
            {
                "page_preview_status": "rendered",
                "page_preview_path": "vision_pages/page-0001.png",
                "region_crop_count": 2,
                "region_crops": [
                    {
                        "block_id": "p1-b0000",
                        "block_type": "image",
                        "crop_path": "vision_crops/page-0001/p1-b0000-image.png",
                        "bbox": [40, 80, 560, 520],
                        "crop_width": 720,
                        "crop_height": 610,
                    },
                    {
                        "block_id": "p1-b0001",
                        "block_type": "table",
                        "crop_path": "vision_crops/page-0001/p1-b0001-table.png",
                        "bbox": [60, 540, 500, 620],
                        "crop_width": 720,
                        "crop_height": 140,
                    },
                ],
            }
        )
        manifest = build_ocr_task_manifest(doc_ir, route)
        image_task_id = manifest["tasks"][0]["task_id"]
        table_task_id = manifest["tasks"][1]["task_id"]
        results = {
            "schema_version": "ocr-results-v1",
            "results": [
                {
                    "task_id": image_task_id,
                    "status": "succeeded",
                    "text": "Figure overview text",
                    "confidence": 0.91,
                    "engine": "unit_ocr",
                    "language": "en",
                    "bbox": [42, 82, 558, 518],
                    "warnings": [],
                },
                {
                    "task_id": table_task_id,
                    "status": "succeeded",
                    "text": "",
                    "confidence": 0.99,
                    "engine": "unit_table_ocr",
                    "language": "en",
                    "bbox": [60, 540, 500, 620],
                    "warnings": ["empty_region"],
                },
                {
                    "task_id": "missing-task",
                    "status": "succeeded",
                    "text": "ghost",
                    "confidence": 0.9,
                    "engine": "unit_ocr",
                    "language": "en",
                    "bbox": [],
                    "warnings": [],
                },
            ],
        }

        normalized_results = build_ocr_results_payload(manifest, results, source_path="manual.json")
        self.assertEqual(normalized_results["schema_version"], "ocr-results-v1")
        self.assertEqual(normalized_results["source_path"], "manual.json")
        self.assertEqual(normalized_results["summary"]["result_count"], 3)
        self.assertEqual(normalized_results["summary"]["invalid_result_count"], 0)
        self.assertEqual(normalized_results["summary"]["status_counts"]["succeeded"], 3)
        self.assertEqual(normalized_results["summary"]["engine_counts"]["unit_ocr"], 2)

        built = build_ocr_writeback(doc_ir, manifest, normalized_results)
        self.assertEqual(built["summary"]["accepted_result_count"], 1)
        self.assertIn("augmented_document_ir", built)

        root = Path.cwd() / "test-output" / "ocr-writeback"
        if root.exists():
            shutil.rmtree(root)
        try:
            report_path = root / "output" / "ocr_writeback.json"
            augmented_path = root / "output" / "document_ir_ocr.json"
            results_path = root / "output" / "ocr_results.json"
            stored_results = write_ocr_results_payload(
                manifest,
                results_path,
                results,
                source_path="manual.json",
            )
            loaded_results = load_ocr_results(results_path)
            report = write_ocr_writeback(doc_ir, manifest, report_path, augmented_path, loaded_results)

            self.assertTrue(results_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertTrue(augmented_path.is_file())
            self.assertEqual(stored_results["summary"]["result_count"], 3)
            self.assertEqual(loaded_results["summary"]["engine_counts"]["unit_table_ocr"], 1)
            self.assertEqual(report["schema_version"], "ocr-writeback-v1")
            self.assertEqual(report["summary"]["task_count"], 2)
            self.assertEqual(report["summary"]["result_count"], 3)
            self.assertEqual(report["summary"]["accepted_result_count"], 1)
            self.assertEqual(report["summary"]["rejected_result_count"], 2)
            self.assertEqual(report["summary"]["pending_task_count"], 0)
            self.assertEqual(report["summary"]["unknown_task_result_count"], 1)
            self.assertEqual(report["summary"]["block_writeback_count"], 1)
            self.assertEqual(report["summary"]["page_writeback_count"], 0)
            self.assertEqual(report["summary"]["rejection_reason_counts"]["empty_text"], 1)
            self.assertEqual(report["summary"]["rejection_reason_counts"]["unknown_task"], 1)
            self.assertEqual(report["artifacts"]["augmented_document_ir"], "output/document_ir_ocr.json")

            augmented = json.loads(augmented_path.read_text(encoding="utf-8"))
            candidates = augmented["pages"][0]["blocks"][0]["meta"]["ocr_candidates"]
            self.assertEqual(candidates[0]["task_id"], image_task_id)
            self.assertEqual(candidates[0]["text"], "Figure overview text")
            self.assertEqual(candidates[0]["confidence"], 0.91)
            self.assertNotIn("ocr_candidates", doc_ir.pages[0].blocks[0].meta)
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_http_retry_capture_records_retry_attempts(self) -> None:
        request = httpx.Request("POST", "https://example.test/chat")
        response = httpx.Response(503, request=request)
        calls = {"count": 0}

        def _op() -> str:
            calls["count"] += 1
            if calls["count"] == 1:
                raise httpx.HTTPStatusError("service unavailable", request=request, response=response)
            return "ok"

        previous = os.environ.get("PDF_TRANSLATE_HTTP_RETRIES")
        os.environ["PDF_TRANSLATE_HTTP_RETRIES"] = "2"
        try:
            with patch("pdf_translate.translators.http_retry._sleep_backoff", lambda attempt: None):
                with capture_http_retry_events() as events:
                    result = call_with_http_retry(_op, context="unit-test")
        finally:
            if previous is None:
                os.environ.pop("PDF_TRANSLATE_HTTP_RETRIES", None)
            else:
                os.environ["PDF_TRANSLATE_HTTP_RETRIES"] = previous

        self.assertEqual(result, "ok")
        self.assertEqual(calls["count"], 2)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["schema_version"], "http-retry-event-v1")
        self.assertEqual(events[0]["context"], "unit-test")
        self.assertEqual(events[0]["attempt_index"], 1)
        self.assertEqual(events[0]["status"], "retryable_error")
        self.assertEqual(events[0]["status_code"], 503)
        self.assertTrue(events[0]["will_retry"])
        self.assertEqual(events[1]["attempt_index"], 2)
        self.assertEqual(events[1]["status"], "success")
        self.assertFalse(events[1]["will_retry"])

    def test_run_metrics_aggregates_stage_chunk_and_token_evidence(self) -> None:
        metrics = build_run_metrics(
            [
                {"event_type": "stage", "phase": "document_ir", "elapsed_ms": 12},
                {"event_type": "stage", "phase": "translation_qa", "elapsed_ms": 8},
                {
                    "event_type": "chunk_translation",
                    "phase": "translation",
                    "chunk_id": "c0000",
                    "chunk_index": 1,
                    "pages_1based": [1, 1],
                    "translator": "echo",
                    "elapsed_ms": 40,
                    "source_char_count": 80,
                    "context_char_count": 40,
                    "request_char_count": 120,
                    "translated_char_count": 64,
                    "estimated_request_token_count": 30,
                    "estimated_translated_token_count": 16,
                    "http_retry_events": [
                        {
                            "schema_version": "http-retry-event-v1",
                            "context": "unit-test",
                            "attempt_index": 1,
                            "max_attempts": 2,
                            "status": "retryable_error",
                            "elapsed_ms": 3,
                            "will_retry": True,
                            "error_type": "ReadTimeout",
                            "status_code": None,
                        },
                        {
                            "schema_version": "http-retry-event-v1",
                            "context": "unit-test",
                            "attempt_index": 2,
                            "max_attempts": 2,
                            "status": "success",
                            "elapsed_ms": 37,
                            "will_retry": False,
                            "error_type": "",
                            "status_code": None,
                        },
                    ],
                },
                {
                    "event_type": "chunk_skipped",
                    "phase": "translation",
                    "chunk_id": "c0001",
                    "reason": "resume_completed",
                },
            ],
            doc_id="metrics-run",
            pipeline_variant="structure",
            backend="echo",
            translate_mode="serial",
            parallel_workers=1,
            page_count=2,
            chunk_count=2,
            completed_chunk_count=2,
            total_elapsed_ms=100,
        )
        self.assertEqual(metrics["schema_version"], "run-metrics-v1")
        self.assertEqual(metrics["doc_id"], "metrics-run")
        self.assertEqual(metrics["summary"]["total_elapsed_ms"], 100)
        self.assertEqual(metrics["summary"]["translation_request_count"], 1)
        self.assertEqual(metrics["summary"]["skipped_chunk_count"], 1)
        self.assertEqual(metrics["summary"]["source_char_count"], 80)
        self.assertEqual(metrics["summary"]["estimated_source_token_count"], 20)
        self.assertEqual(metrics["summary"]["estimated_total_token_count"], 46)
        self.assertEqual(metrics["summary"]["http_attempt_count"], 2)
        self.assertEqual(metrics["summary"]["http_retry_count"], 1)
        self.assertEqual(metrics["summary"]["http_failed_attempt_count"], 1)
        self.assertEqual(metrics["summary"]["http_retryable_error_count"], 1)
        self.assertEqual(metrics["summary"]["stage_elapsed_ms"]["document_ir"], 12)
        self.assertEqual(metrics["chunks"][0]["http_attempt_count"], 2)
        self.assertEqual(metrics["chunks"][0]["http_retry_events"][0]["error_type"], "ReadTimeout")
        self.assertEqual(metrics["breakdowns"]["translator_counts"]["echo"], 1)
        self.assertEqual(metrics["breakdowns"]["skip_reasons"]["resume_completed"], 1)

    def test_cost_estimate_uses_configured_backend_profile(self) -> None:
        run_metrics = build_run_metrics(
            [
                {
                    "event_type": "chunk_translation",
                    "phase": "translation",
                    "chunk_id": "c0000",
                    "translator": "openai_compatible",
                    "elapsed_ms": 40,
                    "source_char_count": 800,
                    "context_char_count": 200,
                    "request_char_count": 1000,
                    "translated_char_count": 600,
                    "estimated_request_token_count": 250,
                    "estimated_translated_token_count": 150,
                    "http_retry_events": [
                        {
                            "schema_version": "http-retry-event-v1",
                            "context": "deepseek",
                            "attempt_index": 1,
                            "max_attempts": 2,
                            "status": "retryable_error",
                            "elapsed_ms": 4,
                            "will_retry": True,
                            "error_type": "HTTPStatusError",
                            "status_code": 503,
                        },
                        {
                            "schema_version": "http-retry-event-v1",
                            "context": "deepseek",
                            "attempt_index": 2,
                            "max_attempts": 2,
                            "status": "success",
                            "elapsed_ms": 36,
                            "will_retry": False,
                            "error_type": "",
                            "status_code": None,
                        },
                    ],
                }
            ],
            doc_id="cost-run",
            pipeline_variant="structure",
            backend="deepseek",
            translate_mode="serial",
            chunk_count=1,
            total_elapsed_ms=100,
        )
        profile = normalize_cost_profile(
            {
                "currency": "USD",
                "backends": {
                    "deepseek": {
                        "input_per_1m_tokens": 1.0,
                        "output_per_1m_tokens": 2.0,
                        "per_request": 0.01,
                    }
                },
            },
            source="test",
        )
        estimate = estimate_cost(run_metrics, profile, backend="deepseek", model="deepseek-chat")
        self.assertEqual(estimate["schema_version"], "cost-estimate-v1")
        self.assertTrue(estimate["configured"])
        self.assertEqual(estimate["profile_key"], "deepseek")
        self.assertEqual(estimate["currency"], "USD")
        self.assertEqual(estimate["usage"]["estimated_request_token_count"], 250)
        self.assertEqual(estimate["usage"]["translation_request_count"], 1)
        self.assertEqual(estimate["usage"]["http_attempt_count"], 2)
        self.assertEqual(estimate["usage"]["http_retry_count"], 1)
        self.assertEqual(estimate["usage"]["billable_request_count"], 2)
        self.assertEqual(estimate["usage"]["billable_request_count_source"], "http_attempt_count")
        self.assertEqual(estimate["summary"]["input_token_cost"], 0.00025)
        self.assertEqual(estimate["summary"]["output_token_cost"], 0.0003)
        self.assertEqual(estimate["summary"]["request_cost"], 0.02)
        self.assertEqual(estimate["summary"]["estimated_total_cost"], 0.02055)
        self.assertEqual(len(estimate["warnings"]), 1)

    def test_experiment_metrics_aggregates_quality_evidence(self) -> None:
        metrics = build_experiment_metrics(
            {
                "schema_version": "structure-qa-v1",
                "doc_id": "metrics-sample",
                "summary": {
                    "page_count": 4,
                    "block_counts": {"paragraph": 8, "table": 2},
                    "table_count": 2,
                    "table_continuation_count": 1,
                    "table_footnote_count": 1,
                    "caption_count": 2,
                    "caption_orphan_count": 1,
                    "footnote_count": 1,
                    "footnote_orphan_count": 0,
                    "relationship_warning_count": 1,
                    "entity_candidate_count": 6,
                    "entity_unique_count": 5,
                    "entity_type_counts": {"model_or_dataset": 2, "person": 1},
                    "page_boundary_fragment_count": 2,
                    "page_boundary_fragment_rate": 0.6667,
                },
            },
            {
                "schema_version": "vision-route-v1",
                "summary": {
                    "page_count": 4,
                    "routed_page_count": 2,
                    "preview_page_count": 2,
                    "preview_crop_count": 3,
                    "action_counts": {"text_only": 2, "local_ocr": 1, "vlm_review": 1},
                    "risk_counts": {"low": 2, "medium": 1, "high": 1},
                },
            },
            {
                "schema_version": "translation-qa-v1",
                "summary": {
                    "chunk_count": 2,
                    "entity_candidate_count": 4,
                    "missing_entity_token_count": 1,
                    "source_table_count": 2,
                    "table_shape_error_count": 1,
                    "source_table_locked_token_count": 6,
                    "table_cell_token_error_count": 2,
                    "missing_table_locked_token_count": 3,
                    "issue_count": 4,
                    "issue_counts": {
                        "missing_numbers": 1,
                        "missing_entity_tokens": 2,
                        "table_cell_token_mismatch": 1,
                    },
                    "severity_counts": {"high": 2, "medium": 2},
                    "max_english_residual_ratio": 0.125,
                },
            },
            {
                "schema_version": "repair-plan-v1",
                "summary": {
                    "chunk_count": 2,
                    "repair_item_count": 4,
                    "action_counts": {
                        "rewrite_with_locked_tokens": 1,
                        "rewrite_with_entity_tokens": 2,
                        "repair_table_cell_tokens": 1,
                    },
                    "priority_counts": {"P0": 2, "P1": 2},
                    "scope_counts": {"chunk": 3, "table_cell": 1},
                },
            },
            chunk_boundary_qa={
                "schema_version": "chunk-boundary-qa-v1",
                "summary": {
                    "split_boundary_count": 1,
                    "protected_boundary_count": 1,
                    "co_located_boundary_count": 1,
                    "high_risk_split_count": 1,
                },
            },
            chunk_strategy_comparison={
                "schema_version": "chunk-strategy-comparison-v1",
                "summary": {
                    "baseline_split_boundary_count": 2,
                    "active_split_boundary_count": 1,
                    "active_split_reduction_vs_baseline": 1,
                    "active_split_reduction_rate_vs_baseline": 0.5,
                },
            },
            table_reconstruction={
                "schema_version": "table-reconstruction-v1",
                "summary": {
                    "table_count": 2,
                    "reconstructable_table_count": 1,
                    "low_confidence_table_count": 1,
                    "cell_count": 8,
                    "numeric_cell_count": 3,
                    "numeric_token_count": 4,
                    "unit_token_count": 1,
                    "significance_token_count": 2,
                    "caption_linked_table_count": 1,
                    "footnote_linked_table_count": 1,
                    "table_reconstruction_ready_rate": 0.5,
                },
            },
            ocr_tasks={
                "schema_version": "ocr-task-manifest-v1",
                "summary": {
                    "task_count": 4,
                    "region_task_count": 3,
                    "page_task_count": 1,
                    "ready_task_count": 3,
                    "blocked_by_missing_evidence_count": 1,
                    "vlm_fallback_task_count": 1,
                    "scope_counts": {"region": 3, "page": 1},
                    "status_counts": {"pending_engine": 3, "blocked_missing_visual_evidence": 1},
                    "priority_counts": {"P0": 2, "P1": 2},
                    "recommended_engine_counts": {
                        "local_ocr": 2,
                        "local_table_ocr": 1,
                        "local_formula_ocr": 1,
                    },
                    "block_type_counts": {"image": 2, "table": 1, "formula": 1},
                },
            },
            ocr_results={
                "schema_version": "ocr-results-v1",
                "summary": {
                    "result_count": 3,
                    "invalid_result_count": 1,
                    "status_counts": {"succeeded": 3},
                    "engine_counts": {"local_ocr": 1, "local_table_ocr": 1, "vlm_fallback": 1},
                },
            },
            ocr_writeback={
                "schema_version": "ocr-writeback-v1",
                "summary": {
                    "task_count": 4,
                    "result_count": 3,
                    "accepted_result_count": 2,
                    "rejected_result_count": 1,
                    "pending_task_count": 1,
                    "missing_result_task_count": 1,
                    "unknown_task_result_count": 0,
                    "block_writeback_count": 2,
                    "page_writeback_count": 0,
                    "result_status_counts": {"succeeded": 3},
                    "accepted_engine_counts": {"local_ocr": 1, "local_table_ocr": 1},
                    "rejection_reason_counts": {"low_confidence": 1},
                },
            },
            repair_requests={
                "schema_version": "repair-requests-v1",
                "summary": {
                    "repair_request_count": 4,
                    "ready_for_translation_backend_count": 3,
                    "manual_review_request_count": 1,
                },
            },
            repair_results={
                "schema_version": "repair-results-v1",
                "summary": {
                    "executed_request_count": 2,
                    "succeeded_count": 1,
                    "failed_count": 1,
                    "skipped_count": 2,
                },
            },
            repair_validation={
                "schema_version": "repair-validation-v1",
                "summary": {
                    "validated_result_count": 2,
                    "passed_count": 1,
                    "failed_count": 1,
                    "unchecked_count": 0,
                    "skipped_count": 2,
                    "checked_locked_token_count": 5,
                    "missing_locked_token_count": 1,
                    "table_shape_check_count": 2,
                    "table_shape_passed_count": 1,
                },
            },
            repair_merge={
                "schema_version": "repair-merge-v1",
                "summary": {
                    "merge_candidate_count": 2,
                    "applied_count": 1,
                    "patched_chunk_count": 1,
                    "skipped_count": 1,
                    "manual_merge_required_count": 1,
                    "conflict_count": 0,
                },
            },
            repair_merge_qa={
                "schema_version": "translation-qa-v1",
                "summary": {
                    "issue_count": 3,
                    "table_shape_error_count": 1,
                    "table_cell_token_error_count": 1,
                    "missing_table_locked_token_count": 1,
                },
            },
            run_metrics={
                "schema_version": "run-metrics-v1",
                "summary": {
                    "total_elapsed_ms": 1000,
                    "translation_elapsed_ms": 400,
                    "translation_request_count": 2,
                    "http_attempt_count": 3,
                    "http_retry_count": 1,
                    "http_failed_attempt_count": 1,
                    "http_retryable_error_count": 1,
                    "http_fatal_error_count": 0,
                    "skipped_chunk_count": 1,
                    "source_char_count": 800,
                    "context_char_count": 200,
                    "request_char_count": 1000,
                    "translated_char_count": 600,
                    "estimated_source_token_count": 200,
                    "estimated_context_token_count": 50,
                    "estimated_request_token_count": 250,
                    "estimated_translated_token_count": 150,
                    "estimated_total_token_count": 400,
                    "avg_chunk_elapsed_ms": 200,
                    "max_chunk_elapsed_ms": 240,
                    "request_chars_per_second": 2500,
                    "translated_chars_per_second": 1500,
                    "stage_elapsed_ms": {"document_ir": 50, "translation_qa": 30},
                },
                "breakdowns": {
                    "stage_counts": {"document_ir": 1, "translation_qa": 1},
                    "translator_counts": {"echo": 2},
                    "skip_reasons": {"resume_completed": 1},
                },
            },
            cost_estimate={
                "schema_version": "cost-estimate-v1",
                "configured": True,
                "currency": "USD",
                "profile_key": "deepseek",
                "usage": {
                    "translation_request_count": 2,
                    "http_attempt_count": 3,
                    "http_retry_count": 1,
                    "billable_request_count": 3,
                    "billable_request_count_source": "http_attempt_count",
                },
                "summary": {
                    "input_token_cost": 0.00025,
                    "output_token_cost": 0.0003,
                    "input_char_cost": 0,
                    "output_char_cost": 0,
                    "request_cost": 0.03,
                    "estimated_total_cost": 0.03055,
                },
                "warnings": [],
            },
            pipeline_variant="structure",
        )
        self.assertEqual(metrics["schema_version"], "experiment-metrics-v1")
        self.assertEqual(metrics["doc_id"], "metrics-sample")
        self.assertEqual(metrics["pipeline_variant"], "structure")
        self.assertEqual(metrics["performance"]["total_elapsed_ms"], 1000)
        self.assertEqual(metrics["performance"]["translation_request_count"], 2)
        self.assertEqual(metrics["performance"]["http_attempt_count"], 3)
        self.assertEqual(metrics["performance"]["http_retry_count"], 1)
        self.assertEqual(metrics["performance"]["http_failed_attempt_count"], 1)
        self.assertEqual(metrics["performance"]["billable_request_count"], 3)
        self.assertEqual(metrics["performance"]["source_char_count"], 800)
        self.assertEqual(metrics["performance"]["estimated_total_token_count"], 400)
        self.assertTrue(metrics["performance"]["cost_profile_configured"])
        self.assertEqual(metrics["performance"]["cost_profile_key"], "deepseek")
        self.assertEqual(metrics["performance"]["cost_currency"], "USD")
        self.assertEqual(metrics["performance"]["estimated_total_cost"], 0.03055)
        self.assertEqual(metrics["rates"]["translation_request_per_chunk"], 1.0)
        self.assertEqual(metrics["rates"]["http_attempt_per_translation_request"], 1.5)
        self.assertEqual(metrics["rates"]["http_retry_rate"], 0.3333)
        self.assertEqual(metrics["rates"]["billable_request_per_chunk"], 1.5)
        self.assertEqual(metrics["rates"]["estimated_request_tokens_per_chunk"], 125.0)
        self.assertEqual(metrics["rates"]["estimated_cost_per_chunk"], 0.0153)
        self.assertEqual(metrics["quality"]["ocr_candidate_page_count"], 2)
        self.assertEqual(metrics["quality"]["repair_item_count"], 4)
        self.assertEqual(metrics["quality"]["repair_request_count"], 4)
        self.assertEqual(metrics["quality"]["repair_backend_request_count"], 3)
        self.assertEqual(metrics["quality"]["repair_manual_request_count"], 1)
        self.assertEqual(metrics["quality"]["repair_executed_request_count"], 2)
        self.assertEqual(metrics["quality"]["repair_succeeded_count"], 1)
        self.assertEqual(metrics["quality"]["repair_failed_count"], 1)
        self.assertEqual(metrics["quality"]["repair_skipped_count"], 2)
        self.assertEqual(metrics["quality"]["repair_validation_checked_count"], 2)
        self.assertEqual(metrics["quality"]["repair_validation_passed_count"], 1)
        self.assertEqual(metrics["quality"]["repair_validation_failed_count"], 1)
        self.assertEqual(metrics["quality"]["repair_validation_missing_locked_token_count"], 1)
        self.assertEqual(metrics["quality"]["repair_validation_table_shape_passed_count"], 1)
        self.assertEqual(metrics["quality"]["repair_merge_candidate_count"], 2)
        self.assertEqual(metrics["quality"]["repair_merge_applied_count"], 1)
        self.assertEqual(metrics["quality"]["repair_merge_patched_chunk_count"], 1)
        self.assertEqual(metrics["quality"]["repair_merge_manual_required_count"], 1)
        self.assertEqual(metrics["quality"]["post_repair_issue_count"], 3)
        self.assertEqual(metrics["quality"]["post_repair_issue_delta"], 1)
        self.assertEqual(metrics["quality"]["post_repair_table_cell_token_error_count"], 1)
        self.assertEqual(metrics["quality"]["table_shape_error_count"], 1)
        self.assertEqual(metrics["quality"]["table_cell_token_error_count"], 2)
        self.assertEqual(metrics["quality"]["missing_table_locked_token_count"], 3)
        self.assertEqual(metrics["quality"]["split_boundary_count"], 1)
        self.assertEqual(metrics["quality"]["protected_boundary_count"], 1)
        self.assertEqual(metrics["quality"]["baseline_split_boundary_count"], 2)
        self.assertEqual(metrics["quality"]["active_split_reduction_vs_baseline"], 1)
        self.assertEqual(metrics["quality"]["reconstructable_table_count"], 1)
        self.assertEqual(metrics["quality"]["table_cell_count"], 8)
        self.assertEqual(metrics["quality"]["table_significance_token_count"], 2)
        self.assertEqual(metrics["rates"]["table_shape_error_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_cell_token_error_rate"], 0.6667)
        self.assertEqual(metrics["rates"]["table_locked_token_missing_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_reconstruction_ready_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_numeric_cell_rate"], 0.375)
        self.assertEqual(metrics["rates"]["table_caption_link_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_footnote_binding_rate"], 0.5)
        self.assertEqual(metrics["rates"]["split_boundary_rate"], 0.5)
        self.assertEqual(metrics["rates"]["protected_boundary_rate"], 0.5)
        self.assertEqual(metrics["rates"]["active_split_reduction_rate_vs_baseline"], 0.5)
        self.assertEqual(metrics["rates"]["entity_missing_rate"], 0.25)
        self.assertEqual(metrics["rates"]["repair_item_per_chunk"], 2.0)
        self.assertEqual(metrics["rates"]["repair_request_ready_rate"], 0.75)
        self.assertEqual(metrics["rates"]["repair_execution_success_rate"], 0.5)
        self.assertEqual(metrics["rates"]["repair_validation_pass_rate"], 0.5)
        self.assertEqual(metrics["rates"]["repair_locked_token_pass_rate"], 0.8)
        self.assertEqual(metrics["rates"]["repair_table_shape_validation_pass_rate"], 0.5)
        self.assertEqual(metrics["rates"]["repair_merge_apply_rate"], 0.5)
        self.assertEqual(metrics["rates"]["post_repair_issue_reduction_rate"], 0.25)
        self.assertEqual(metrics["rates"]["relationship_warning_rate"], 0.3333)
        self.assertEqual(metrics["quality"]["vision_preview_page_count"], 2)
        self.assertEqual(metrics["quality"]["vision_region_crop_count"], 3)
        self.assertEqual(metrics["quality"]["ocr_task_count"], 4)
        self.assertEqual(metrics["quality"]["ocr_region_task_count"], 3)
        self.assertEqual(metrics["quality"]["ocr_page_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_ready_task_count"], 3)
        self.assertEqual(metrics["quality"]["ocr_blocked_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_vlm_fallback_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_result_payload_count"], 3)
        self.assertEqual(metrics["quality"]["ocr_invalid_result_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_result_count"], 3)
        self.assertEqual(metrics["quality"]["ocr_accepted_result_count"], 2)
        self.assertEqual(metrics["quality"]["ocr_rejected_result_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_pending_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_missing_result_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_block_writeback_count"], 2)
        self.assertEqual(metrics["quality"]["ocr_page_writeback_count"], 0)
        self.assertEqual(metrics["rates"]["vision_preview_page_rate"], 0.5)
        self.assertEqual(metrics["rates"]["vision_region_crop_per_routed_page"], 1.5)
        self.assertEqual(metrics["rates"]["ocr_task_per_routed_page"], 2.0)
        self.assertEqual(metrics["rates"]["ocr_region_task_rate"], 0.75)
        self.assertEqual(metrics["rates"]["ocr_ready_task_rate"], 0.75)
        self.assertEqual(metrics["rates"]["ocr_result_payload_valid_rate"], 0.75)
        self.assertEqual(metrics["rates"]["ocr_task_result_coverage_rate"], 0.75)
        self.assertEqual(metrics["rates"]["ocr_result_acceptance_rate"], 0.6667)
        self.assertEqual(metrics["rates"]["ocr_writeback_apply_rate"], 0.5)
        self.assertEqual(metrics["breakdowns"]["vision_action_counts"]["local_ocr"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_task_engine_counts"]["local_table_ocr"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_result_payload_engine_counts"]["vlm_fallback"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_writeback_engine_counts"]["local_table_ocr"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_writeback_rejection_counts"]["low_confidence"], 1)
        self.assertEqual(metrics["breakdowns"]["stage_elapsed_ms"]["document_ir"], 50)
        self.assertEqual(metrics["breakdowns"]["translator_counts"]["echo"], 2)
        self.assertEqual(metrics["evidence_files"]["translation_qa"], "output/qa_report.json")
        self.assertEqual(metrics["evidence_files"]["ocr_tasks"], "output/ocr_tasks.json")
        self.assertEqual(metrics["evidence_files"]["ocr_results"], "output/ocr_results.json")
        self.assertEqual(metrics["evidence_files"]["ocr_writeback"], "output/ocr_writeback.json")
        self.assertEqual(metrics["evidence_files"]["document_ir_ocr"], "output/document_ir_ocr.json")
        self.assertEqual(metrics["evidence_files"]["repair_requests"], "output/repair_requests.json")
        self.assertEqual(metrics["evidence_files"]["repair_results"], "output/repair_results.json")
        self.assertEqual(metrics["evidence_files"]["repair_validation"], "output/repair_validation.json")
        self.assertEqual(metrics["evidence_files"]["repair_merge"], "output/repair_merge.json")
        self.assertEqual(metrics["evidence_files"]["repair_merge_qa"], "output/repair_merge_qa.json")
        self.assertEqual(metrics["evidence_files"]["run_metrics"], "output/run_metrics.json")
        self.assertEqual(metrics["evidence_files"]["run_log"], "output/run_log.jsonl")
        self.assertEqual(metrics["evidence_files"]["cost_estimate"], "output/cost_estimate.json")

    def test_memory_store_records_glossary_conflicts_for_review(self) -> None:
        root = Path.cwd() / "test-output" / "glossary-conflict-memory"
        if root.exists():
            shutil.rmtree(root)
        try:
            mem = MemoryStore(root / "memory")
            mem.ensure_files()
            added_first = mem.merge_glossary_terms_from_survey(
                [{"en": "Accuracy", "zh": "准确率"}],
                first_page_1based=1,
            )
            added_conflict = mem.merge_glossary_terms_from_survey(
                [{"en": "accuracy", "zh": "精度"}],
                first_page_1based=2,
                source="survey",
            )
            glossary = mem.load_glossary()
            pending = mem.load_pending_review()
            self.assertEqual(added_first, 1)
            self.assertEqual(added_conflict, 0)
            self.assertEqual(len(glossary["terms"]), 1)
            self.assertEqual(glossary["terms"][0]["status"], "candidate")
            conflicts = [item for item in pending["items"] if item["type"] == "glossary_conflict"]
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0]["en"], "accuracy")
            self.assertEqual(conflicts[0]["existing_zh"], ["准确率"])
            self.assertEqual(conflicts[0]["candidate_zh"], "精度")
            self.assertEqual(conflicts[0]["status"], "pending")
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_translation_qa_reports_glossary_conflicts(self) -> None:
        root = Path.cwd() / "test-output" / "translation-qa-glossary-conflict"
        if root.exists():
            shutil.rmtree(root)
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(parents=True)
        try:
            chunks = [
                TextChunk(
                    chunk_id="c0000",
                    pages_0based=[0],
                    text="Accuracy improves under domain shift.",
                    link_count=0,
                    image_count=0,
                )
            ]
            (chunk_dir / "c0000.md").write_text(
                "---\n{}\n---\n\n准确率在领域偏移下提升。\n",
                encoding="utf-8",
            )
            report = build_translation_qa(
                chunks,
                chunk_dir,
                glossary={"terms": [{"en": "Accuracy", "zh": "准确率", "first_page": 1}]},
                pending_review={
                    "items": [
                        {
                            "type": "glossary_conflict",
                            "status": "pending",
                            "en": "Accuracy",
                            "existing_zh": ["准确率"],
                            "candidate_zh": "精度",
                            "first_page": 1,
                            "source": "survey",
                        }
                    ]
                },
            )
            issue_types = {issue["type"] for issue in report["chunks"][0]["issues"]}
            self.assertIn("glossary_translation_conflict", issue_types)
            self.assertNotIn("missing_glossary_terms", issue_types)
            self.assertEqual(report["summary"]["glossary_conflict_count"], 1)
            self.assertEqual(report["summary"]["issue_count"], 1)

            plan = build_repair_plan(report)
            self.assertEqual(plan["summary"]["repair_item_count"], 1)
            self.assertEqual(plan["items"][0]["action"], "review_glossary_conflict")
            self.assertEqual(plan["items"][0]["scope"], "glossary")
            self.assertEqual(plan["items"][0]["executor"], "human_review")

            html_path = root / "bilingual.html"
            write_bilingual_html(
                chunks,
                chunk_dir,
                html_path,
                qa_report=report,
                repair_plan=plan,
                title="术语冲突样例",
            )
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("glossary_translation_conflict", html)
            self.assertIn("review_glossary_conflict", html)
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_translation_qa_reports_missing_entity_tokens(self) -> None:
        root = Path.cwd() / "test-output" / "translation-qa-entities"
        if root.exists():
            shutil.rmtree(root)
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(parents=True)
        try:
            chunks = [
                TextChunk(
                    chunk_id="c0000",
                    pages_0based=[0],
                    text="BERT improves CNN baselines on ImageNet.",
                    link_count=0,
                    image_count=0,
                )
            ]
            (chunk_dir / "c0000.md").write_text(
                "---\n{}\n---\n\n该方法提升了基线效果。\n",
                encoding="utf-8",
            )
            report = build_translation_qa(chunks, chunk_dir)
            issue_types = {issue["type"] for issue in report["chunks"][0]["issues"]}
            self.assertIn("missing_entity_tokens", issue_types)
            self.assertEqual(report["summary"]["entity_candidate_count"], 3)
            self.assertEqual(report["summary"]["missing_entity_token_count"], 3)
            entity_issue = next(
                issue for issue in report["chunks"][0]["issues"] if issue["type"] == "missing_entity_tokens"
            )
            self.assertEqual(entity_issue["severity"], "medium")
            self.assertEqual({entity["text"] for entity in entity_issue["entities"]}, {"BERT", "CNN", "ImageNet"})

            plan = build_repair_plan(report)
            self.assertEqual(plan["summary"]["repair_item_count"], 1)
            self.assertEqual(plan["items"][0]["action"], "rewrite_with_entity_tokens")
            self.assertEqual(plan["items"][0]["scope"], "chunk")
            self.assertEqual(plan["items"][0]["executor"], "translation_backend")

            html_path = root / "bilingual.html"
            write_bilingual_html(
                chunks,
                chunk_dir,
                html_path,
                qa_report=report,
                repair_plan=plan,
                title="实体缺失样例",
            )
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("missing_entity_tokens", html)
            self.assertIn("rewrite_with_entity_tokens", html)
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_translation_qa_uses_document_ir_table_invariants_for_plain_text_source(self) -> None:
        root = Path.cwd() / "test-output" / "translation-qa-document-table"
        if root.exists():
            shutil.rmtree(root)
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(parents=True)
        try:
            chunks = [
                TextChunk(
                    chunk_id="c0000",
                    pages_0based=[0],
                    text="Model Acc\nA 91.2\nB 92.4",
                    link_count=0,
                    image_count=0,
                )
            ]
            doc_ir = DocumentIR(
                doc_id="table-invariant",
                source_pdf="sample.pdf",
                pages=[
                    PageIR(
                        page_no=1,
                        width=600,
                        height=800,
                        text="Model Acc\nA 91.2\nB 92.4",
                        blocks=[
                            BlockIR(
                                "p1-b0000",
                                1,
                                "table",
                                "Model Acc\nA 91.2\nB 92.4",
                                (40, 100, 520, 170),
                                0,
                                meta={
                                    "table": {
                                        "row_count": 3,
                                        "column_count": 2,
                                        "header": ["Model", "Acc"],
                                        "numeric_tokens": ["91.2", "92.4"],
                                        "confidence": "medium",
                                    }
                                },
                            )
                        ],
                    )
                ],
            )
            (chunk_dir / "c0000.md").write_text(
                "---\n{}\n---\n\n模型 A 和 B 的准确率分别为 91.2 和 92.4。\n",
                encoding="utf-8",
            )
            report = build_translation_qa(chunks, chunk_dir, document_ir=doc_ir)
            issue = next(
                issue for issue in report["chunks"][0]["issues"] if issue["type"] == "table_shape_mismatch"
            )
            self.assertEqual(report["summary"]["source_table_count"], 1)
            self.assertEqual(report["summary"]["table_shape_error_count"], 1)
            self.assertEqual(issue["severity"], "high")
            self.assertEqual(issue["tables"][0]["block_id"], "p1-b0000")
            self.assertEqual(issue["tables"][0]["reason"], "missing_markdown_table")
            self.assertEqual(issue["tables"][0]["source"], {"row_count": 3, "column_count": 2})
            self.assertIsNone(issue["tables"][0]["target"])

            plan = build_repair_plan(report)
            table_item = next(item for item in plan["items"] if item["issue_type"] == "table_shape_mismatch")
            self.assertEqual(table_item["action"], "repair_table_shape")
            self.assertEqual(table_item["scope"], "table")
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

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
            report = build_translation_qa(
                chunks,
                chunk_dir,
                glossary={"terms": [{"en": "Acc", "zh": "准确率", "first_page": 1}]},
            )
            issue_types = {issue["type"] for issue in report["chunks"][0]["issues"]}
            self.assertIn("missing_numbers", issue_types)
            self.assertIn("missing_references", issue_types)
            self.assertIn("table_shape_mismatch", issue_types)
            self.assertIn("missing_glossary_terms", issue_types)
            self.assertEqual(report["summary"]["issue_count"], 4)
            self.assertEqual(report["summary"]["glossary_term_count"], 1)

            plan = build_repair_plan(report)
            self.assertEqual(plan["schema_version"], "repair-plan-v1")
            self.assertEqual(plan["summary"]["repair_item_count"], 4)
            actions = {item["action"] for item in plan["items"]}
            self.assertIn("rewrite_with_locked_tokens", actions)
            self.assertIn("repair_table_shape", actions)
            self.assertIn("rewrite_with_glossary_terms", actions)
            self.assertEqual(plan["summary"]["priority_counts"]["P0"], 3)
            self.assertEqual(plan["summary"]["priority_counts"]["P1"], 1)

            html_path = root / "bilingual.html"
            write_bilingual_html(
                chunks,
                chunk_dir,
                html_path,
                qa_report=report,
                repair_plan=plan,
                title="样例双语对照",
            )
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("样例双语对照", html)
            self.assertIn("Table 1 reports", html)
            self.assertIn("表 1 报告了结果", html)
            self.assertIn("missing_numbers", html)
            self.assertIn("missing_glossary_terms", html)
            self.assertIn("repair_table_shape", html)
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
            p1.insert_text((72, 72), "1 Introduction\nSmith et al. (2024) evaluated BERT at Stanford University.")
            p2 = doc.new_page(width=595, height=842)
            p2.insert_text((72, 72), "Table 1: Results\nModel Acc F1 N\nA 91.2 88.1 120")
            pdf_path.write_bytes(doc.tobytes())
            doc.close()

            work_dir = root / "work"
            init_workdir(work_dir)
            run_split(pdf_path, work_dir)
            external_ocr_results_path = root / "external_ocr_results.json"
            external_ocr_results_path.write_text(
                json.dumps(
                    {
                        "schema_version": "ocr-results-v1",
                        "source": "unit_external_file",
                        "results": [
                            {
                                "task_id": "unknown-task",
                                "status": "succeeded",
                                "text": "External OCR result",
                                "confidence": 0.93,
                                "engine": "unit_external_ocr",
                                "language": "en",
                                "bbox": [],
                                "warnings": [],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            cfg = AppConfig.from_env()
            out = run_translate(
                work_dir,
                cfg,
                backend="echo",
                pages_per_chunk=1,
                overlap_pages=0,
                max_chunks=1,
                chunk_strategy="structure",
                ocr_results_path=external_ocr_results_path,
            )

            self.assertTrue(out.is_file())
            ir_path = work_dir / "output" / "document_ir.json"
            manifest_path = work_dir / "output" / "structure_chunks_manifest.json"
            qa_path = work_dir / "output" / "structure_qa.json"
            table_reconstruction_path = work_dir / "output" / "table_reconstruction.json"
            chunk_boundary_qa_path = work_dir / "output" / "chunk_boundary_qa.json"
            chunk_strategy_comparison_path = work_dir / "output" / "chunk_strategy_comparison.json"
            vision_path = work_dir / "output" / "vision_route.json"
            ocr_tasks_path = work_dir / "output" / "ocr_tasks.json"
            ocr_results_path = work_dir / "output" / "ocr_results.json"
            ocr_writeback_path = work_dir / "output" / "ocr_writeback.json"
            document_ir_ocr_path = work_dir / "output" / "document_ir_ocr.json"
            translation_qa_path = work_dir / "output" / "qa_report.json"
            translation_qa_md_path = work_dir / "output" / "qa_report.md"
            repair_plan_path = work_dir / "output" / "repair_plan.json"
            repair_plan_md_path = work_dir / "output" / "repair_plan.md"
            repair_requests_path = work_dir / "output" / "repair_requests.json"
            repair_requests_md_path = work_dir / "output" / "repair_requests.md"
            repair_results_path = work_dir / "output" / "repair_results.json"
            repair_results_md_path = work_dir / "output" / "repair_results.md"
            repair_validation_path = work_dir / "output" / "repair_validation.json"
            repair_validation_md_path = work_dir / "output" / "repair_validation.md"
            repair_merge_path = work_dir / "output" / "repair_merge.json"
            repair_merge_md_path = work_dir / "output" / "repair_merge.md"
            repair_merge_qa_path = work_dir / "output" / "repair_merge_qa.json"
            repair_merge_qa_md_path = work_dir / "output" / "repair_merge_qa.md"
            repaired_full_path = work_dir / "output" / "repaired_full.md"
            metrics_path = work_dir / "output" / "experiment_metrics.json"
            run_metrics_path = work_dir / "output" / "run_metrics.json"
            run_log_path = work_dir / "output" / "run_log.jsonl"
            cost_estimate_path = work_dir / "output" / "cost_estimate.json"
            bilingual_path = work_dir / "output" / "bilingual.html"
            self.assertTrue(ir_path.is_file())
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(qa_path.is_file())
            self.assertTrue(table_reconstruction_path.is_file())
            self.assertTrue(chunk_boundary_qa_path.is_file())
            self.assertTrue(chunk_strategy_comparison_path.is_file())
            self.assertTrue(vision_path.is_file())
            self.assertTrue(ocr_tasks_path.is_file())
            self.assertTrue(ocr_results_path.is_file())
            self.assertTrue(ocr_writeback_path.is_file())
            self.assertTrue(document_ir_ocr_path.is_file())
            self.assertTrue(translation_qa_path.is_file())
            self.assertTrue(translation_qa_md_path.is_file())
            self.assertTrue(repair_plan_path.is_file())
            self.assertTrue(repair_plan_md_path.is_file())
            self.assertTrue(repair_requests_path.is_file())
            self.assertTrue(repair_requests_md_path.is_file())
            self.assertTrue(repair_results_path.is_file())
            self.assertTrue(repair_results_md_path.is_file())
            self.assertTrue(repair_validation_path.is_file())
            self.assertTrue(repair_validation_md_path.is_file())
            self.assertTrue(repair_merge_path.is_file())
            self.assertTrue(repair_merge_md_path.is_file())
            self.assertTrue(repair_merge_qa_path.is_file())
            self.assertTrue(repair_merge_qa_md_path.is_file())
            self.assertTrue(repaired_full_path.is_file())
            self.assertTrue(metrics_path.is_file())
            self.assertTrue(run_metrics_path.is_file())
            self.assertTrue(run_log_path.is_file())
            self.assertTrue(cost_estimate_path.is_file())
            self.assertTrue(bilingual_path.is_file())
            ir = json.loads(ir_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
            table_reconstruction = json.loads(table_reconstruction_path.read_text(encoding="utf-8"))
            chunk_boundary_qa = json.loads(chunk_boundary_qa_path.read_text(encoding="utf-8"))
            chunk_strategy_comparison = json.loads(chunk_strategy_comparison_path.read_text(encoding="utf-8"))
            vision = json.loads(vision_path.read_text(encoding="utf-8"))
            ocr_tasks = json.loads(ocr_tasks_path.read_text(encoding="utf-8"))
            ocr_results = json.loads(ocr_results_path.read_text(encoding="utf-8"))
            ocr_writeback = json.loads(ocr_writeback_path.read_text(encoding="utf-8"))
            document_ir_ocr = json.loads(document_ir_ocr_path.read_text(encoding="utf-8"))
            translation_qa = json.loads(translation_qa_path.read_text(encoding="utf-8"))
            repair_plan = json.loads(repair_plan_path.read_text(encoding="utf-8"))
            repair_requests = json.loads(repair_requests_path.read_text(encoding="utf-8"))
            repair_results = json.loads(repair_results_path.read_text(encoding="utf-8"))
            repair_validation = json.loads(repair_validation_path.read_text(encoding="utf-8"))
            repair_merge = json.loads(repair_merge_path.read_text(encoding="utf-8"))
            repair_merge_qa = json.loads(repair_merge_qa_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            run_metrics = json.loads(run_metrics_path.read_text(encoding="utf-8"))
            cost_estimate = json.loads(cost_estimate_path.read_text(encoding="utf-8"))
            run_log_lines = [
                json.loads(line)
                for line in run_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(ir["schema_version"], "document-ir-v1")
            self.assertGreaterEqual(len(ir["pages"]), 1)
            self.assertIn("meta", ir["pages"][0])
            ir_entities = [
                entity
                for page in ir["pages"]
                for block in page["blocks"]
                for entity in block.get("meta", {}).get("entities", [])
            ]
            self.assertTrue(any(entity["text"] == "BERT" for entity in ir_entities))
            self.assertGreaterEqual(len(manifest), 1)
            self.assertIn("block_ids", manifest[0])
            self.assertIn("boundary_fragment_ids", manifest[0])
            self.assertEqual(qa["schema_version"], "structure-qa-v1")
            self.assertIn("table_count", qa["summary"])
            self.assertIn("caption_orphan_count", qa["summary"])
            self.assertIn("footnote_orphan_count", qa["summary"])
            self.assertIn("table_footnote_count", qa["summary"])
            self.assertIn("table_continuation_count", qa["summary"])
            self.assertIn("entity_candidate_count", qa["summary"])
            self.assertIn("entity_type_counts", qa["summary"])
            self.assertIn("relationships", qa)
            self.assertIn("table_continuations", qa)
            self.assertIn("entities", qa)
            self.assertGreaterEqual(qa["summary"]["entity_candidate_count"], 1)
            self.assertGreaterEqual(qa["summary"]["table_count"], 1)
            self.assertEqual(table_reconstruction["schema_version"], "table-reconstruction-v1")
            self.assertGreaterEqual(table_reconstruction["summary"]["table_count"], 1)
            self.assertIn("table_reconstruction_ready_rate", table_reconstruction["summary"])
            self.assertEqual(chunk_boundary_qa["schema_version"], "chunk-boundary-qa-v1")
            self.assertEqual(chunk_boundary_qa["pipeline_variant"], "structure")
            self.assertIn("split_boundary_count", chunk_boundary_qa["summary"])
            self.assertEqual(chunk_strategy_comparison["schema_version"], "chunk-strategy-comparison-v1")
            self.assertEqual(chunk_strategy_comparison["active_strategy"], "structure")
            self.assertIn("active_split_reduction_vs_baseline", chunk_strategy_comparison["summary"])
            self.assertEqual(vision["schema_version"], "vision-route-v1")
            self.assertIn("action_counts", vision["summary"])
            self.assertEqual(ocr_tasks["schema_version"], "ocr-task-manifest-v1")
            self.assertIn("task_count", ocr_tasks["summary"])
            self.assertEqual(ocr_results["schema_version"], "ocr-results-v1")
            self.assertEqual(ocr_results["source"], "unit_external_file")
            self.assertEqual(ocr_results["summary"]["result_count"], 1)
            self.assertEqual(ocr_results["summary"]["engine_counts"]["unit_external_ocr"], 1)
            self.assertEqual(ocr_results["source_path"], str(external_ocr_results_path))
            self.assertEqual(ocr_writeback["schema_version"], "ocr-writeback-v1")
            self.assertEqual(ocr_writeback["summary"]["result_count"], 1)
            self.assertEqual(ocr_writeback["summary"]["unknown_task_result_count"], 1)
            self.assertIn("pending_task_count", ocr_writeback["summary"])
            self.assertEqual(document_ir_ocr["schema_version"], "document-ir-v1")
            self.assertEqual(translation_qa["schema_version"], "translation-qa-v1")
            self.assertIn("issue_counts", translation_qa["summary"])
            self.assertIn("entity_candidate_count", translation_qa["summary"])
            self.assertIn("missing_entity_token_count", translation_qa["summary"])
            self.assertEqual(repair_plan["schema_version"], "repair-plan-v1")
            self.assertIn("repair_item_count", repair_plan["summary"])
            self.assertEqual(repair_requests["schema_version"], "repair-requests-v1")
            self.assertIn("repair_request_count", repair_requests["summary"])
            self.assertEqual(repair_results["schema_version"], "repair-results-v1")
            self.assertIn("execution_enabled", repair_results["summary"])
            self.assertEqual(repair_validation["schema_version"], "repair-validation-v1")
            self.assertIn("validated_result_count", repair_validation["summary"])
            self.assertEqual(repair_merge["schema_version"], "repair-merge-v1")
            self.assertIn("applied_count", repair_merge["summary"])
            self.assertEqual(repair_merge_qa["schema_version"], "translation-qa-v1")
            self.assertEqual(run_metrics["schema_version"], "run-metrics-v1")
            self.assertEqual(run_metrics["pipeline_variant"], "structure")
            self.assertGreaterEqual(run_metrics["summary"]["translation_request_count"], 1)
            self.assertEqual(run_metrics["summary"]["http_attempt_count"], 0)
            self.assertEqual(run_metrics["summary"]["http_retry_count"], 0)
            self.assertGreater(run_metrics["summary"]["request_char_count"], 0)
            self.assertIn("document_ir", run_metrics["summary"]["stage_elapsed_ms"])
            self.assertTrue(any(event["event_type"] == "chunk_translation" for event in run_log_lines))
            self.assertEqual(cost_estimate["schema_version"], "cost-estimate-v1")
            self.assertTrue(cost_estimate["configured"])
            self.assertEqual(cost_estimate["profile_key"], "echo")
            self.assertIn("billable_request_count", cost_estimate["usage"])
            self.assertEqual(cost_estimate["summary"]["estimated_total_cost"], 0)
            self.assertEqual(metrics["schema_version"], "experiment-metrics-v1")
            self.assertEqual(metrics["pipeline_variant"], "structure")
            self.assertEqual(metrics["quality"]["table_count"], qa["summary"]["table_count"])
            self.assertGreaterEqual(metrics["performance"]["translation_request_count"], 1)
            self.assertEqual(metrics["performance"]["http_attempt_count"], 0)
            self.assertGreater(metrics["performance"]["estimated_total_token_count"], 0)
            self.assertEqual(metrics["performance"]["estimated_total_cost"], 0)
            self.assertIn("reconstructable_table_count", metrics["quality"])
            self.assertIn("table_reconstruction_ready_rate", metrics["rates"])
            self.assertEqual(
                metrics["breakdowns"]["vision_action_counts"],
                vision["summary"]["action_counts"],
            )
            self.assertIn("ocr_task_count", metrics["quality"])
            self.assertEqual(metrics["evidence_files"]["ocr_tasks"], "output/ocr_tasks.json")
            self.assertEqual(metrics["quality"]["ocr_result_payload_count"], 1)
            self.assertEqual(metrics["breakdowns"]["ocr_result_payload_engine_counts"]["unit_external_ocr"], 1)
            self.assertEqual(metrics["evidence_files"]["ocr_results"], "output/ocr_results.json")
            self.assertIn("ocr_pending_task_count", metrics["quality"])
            self.assertEqual(metrics["evidence_files"]["ocr_writeback"], "output/ocr_writeback.json")
            self.assertEqual(metrics["evidence_files"]["document_ir_ocr"], "output/document_ir_ocr.json")
            self.assertIn("entity_missing_rate", metrics["rates"])
            self.assertIn("split_boundary_rate", metrics["rates"])
            self.assertEqual(metrics["evidence_files"]["chunk_boundary_qa"], "output/chunk_boundary_qa.json")
            self.assertEqual(
                metrics["evidence_files"]["table_reconstruction"],
                "output/table_reconstruction.json",
            )
            self.assertEqual(
                metrics["evidence_files"]["chunk_strategy_comparison"],
                "output/chunk_strategy_comparison.json",
            )
            self.assertEqual(metrics["evidence_files"]["repair_plan"], "output/repair_plan.json")
            self.assertEqual(metrics["evidence_files"]["repair_requests"], "output/repair_requests.json")
            self.assertEqual(metrics["evidence_files"]["repair_results"], "output/repair_results.json")
            self.assertEqual(metrics["evidence_files"]["repair_validation"], "output/repair_validation.json")
            self.assertEqual(metrics["evidence_files"]["repair_merge"], "output/repair_merge.json")
            self.assertEqual(metrics["evidence_files"]["repair_merge_qa"], "output/repair_merge_qa.json")
            self.assertEqual(metrics["evidence_files"]["run_metrics"], "output/run_metrics.json")
            self.assertEqual(metrics["evidence_files"]["run_log"], "output/run_log.jsonl")
            self.assertEqual(metrics["evidence_files"]["cost_estimate"], "output/cost_estimate.json")
            self.assertIn("双语对照译文", bilingual_path.read_text(encoding="utf-8"))
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
        repairs_dir = output / "repairs"
        repairs_dir.mkdir(parents=True)
        repaired_chunks_dir = output / "repaired_chunks"
        repaired_chunks_dir.mkdir(parents=True)
        vision_pages_dir = output / "vision_pages"
        vision_pages_dir.mkdir(parents=True)
        vision_crops_dir = output / "vision_crops" / "page-0001"
        vision_crops_dir.mkdir(parents=True)
        try:
            for name in [
                "translated_full.md",
                "repaired_full.md",
                "bilingual.html",
                "document_ir.json",
                "structure_chunks_manifest.json",
                "structure_qa.json",
                "table_reconstruction.json",
                "chunk_boundary_qa.json",
                "chunk_strategy_comparison.json",
                "vision_route.json",
                "ocr_tasks.json",
                "ocr_results.json",
                "ocr_writeback.json",
                "document_ir_ocr.json",
                "qa_report.json",
                "qa_report.md",
                "repair_plan.json",
                "repair_plan.md",
                "repair_requests.json",
                "repair_requests.md",
                "repair_results.json",
                "repair_results.md",
                "repair_validation.json",
                "repair_validation.md",
                "repair_merge.json",
                "repair_merge.md",
                "repair_merge_qa.json",
                "repair_merge_qa.md",
                "experiment_metrics.json",
                "run_metrics.json",
                "cost_estimate.json",
            ]:
                (output / name).write_text("{}", encoding="utf-8")
            (repairs_dir / "rq0000.md").write_text("候选修复片段", encoding="utf-8")
            (repaired_chunks_dir / "c0000.md").write_text("修复合并分块", encoding="utf-8")
            (vision_pages_dir / "page-0001.png").write_bytes(b"fakepng")
            (vision_crops_dir / "p1-b0000-image.png").write_bytes(b"fakecrop")
            rels = {
                path.relative_to(root).as_posix()
                for path in iter_bundle_files(root)
            }
            self.assertIn("output/repair_plan.json", rels)
            self.assertIn("output/repair_requests.json", rels)
            self.assertIn("output/repair_results.json", rels)
            self.assertIn("output/repair_validation.json", rels)
            self.assertIn("output/repair_merge.json", rels)
            self.assertIn("output/repair_merge_qa.json", rels)
            self.assertIn("output/repairs/rq0000.md", rels)
            self.assertIn("output/repaired_chunks/c0000.md", rels)
            self.assertIn("output/vision_pages/page-0001.png", rels)
            self.assertIn("output/vision_crops/page-0001/p1-b0000-image.png", rels)
            self.assertIn("output/ocr_tasks.json", rels)
            self.assertIn("output/ocr_results.json", rels)
            self.assertIn("output/ocr_writeback.json", rels)
            self.assertIn("output/document_ir_ocr.json", rels)
            self.assertIn("output/repaired_full.md", rels)
            self.assertIn("output/bilingual.html", rels)
            self.assertIn("output/qa_report.md", rels)
            self.assertIn("output/document_ir.json", rels)
            self.assertIn("output/table_reconstruction.json", rels)
            self.assertIn("output/chunk_boundary_qa.json", rels)
            self.assertIn("output/chunk_strategy_comparison.json", rels)
            self.assertIn("output/experiment_metrics.json", rels)
            self.assertIn("output/run_metrics.json", rels)
            self.assertIn("output/cost_estimate.json", rels)
            self.assertEqual(map_bundle_arcname("output/bilingual.html"), "译文/双语对照.html")
            self.assertEqual(map_bundle_arcname("output/repaired_full.md"), "译文/局部修复合并译文.md")
            self.assertEqual(map_bundle_arcname("output/repair_plan.md"), "质量/局部修复计划.md")
            self.assertEqual(map_bundle_arcname("output/repair_requests.md"), "质量/局部修复请求.md")
            self.assertEqual(map_bundle_arcname("output/repair_results.md"), "质量/局部修复结果.md")
            self.assertEqual(map_bundle_arcname("output/repair_validation.md"), "质量/局部修复验证.md")
            self.assertEqual(map_bundle_arcname("output/repair_merge.md"), "质量/局部修复合并.md")
            self.assertEqual(map_bundle_arcname("output/repair_merge_qa.md"), "质量/局部修复后QA.md")
            self.assertEqual(map_bundle_arcname("output/repairs/rq0000.md"), "质量/局部修复片段/rq0000.md")
            self.assertEqual(map_bundle_arcname("output/repaired_chunks/c0000.md"), "译文/局部修复分块/c0000.md")
            self.assertEqual(
                map_bundle_arcname("output/vision_pages/page-0001.png"),
                "质量/图像OCR页面预览/page-0001.png",
            )
            self.assertEqual(
                map_bundle_arcname("output/vision_crops/page-0001/p1-b0000-image.png"),
                "质量/图像OCR区域裁剪/page-0001/p1-b0000-image.png",
            )
            self.assertEqual(map_bundle_arcname("output/ocr_tasks.json"), "质量/OCR调度任务.json")
            self.assertEqual(map_bundle_arcname("output/ocr_results.json"), "质量/OCR识别结果.json")
            self.assertEqual(map_bundle_arcname("output/ocr_writeback.json"), "质量/OCR结果回写.json")
            self.assertEqual(map_bundle_arcname("output/document_ir_ocr.json"), "设置/OCR增强文档结构IR.json")
            self.assertEqual(map_bundle_arcname("output/structure_qa.json"), "质量/结构QA.json")
            self.assertEqual(map_bundle_arcname("output/table_reconstruction.json"), "质量/表格重建证据.json")
            self.assertEqual(map_bundle_arcname("output/chunk_boundary_qa.json"), "质量/分段边界QA.json")
            self.assertEqual(map_bundle_arcname("output/chunk_strategy_comparison.json"), "质量/分段策略对比.json")
            self.assertEqual(map_bundle_arcname("output/experiment_metrics.json"), "质量/实验指标.json")
            self.assertEqual(map_bundle_arcname("output/run_metrics.json"), "质量/运行性能指标.json")
            self.assertEqual(map_bundle_arcname("output/cost_estimate.json"), "质量/成本估算.json")
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)


if __name__ == "__main__":
    unittest.main()
