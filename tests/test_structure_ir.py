from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
import httpx

from pdf_translate.chunking import TextChunk
from pdf_translate.chunkers.structure import build_structure_chunks
from pdf_translate.config import AppConfig
from pdf_translate.costing import estimate_cost, normalize_cost_profile
from pdf_translate.error_codes import PdfTranslateError, error_info_from_exception
from pdf_translate.exporters.bilingual_html import write_bilingual_html
from pdf_translate.exporters.translated_pdf import write_translated_pdf
from pdf_translate.extractors.document_ir import (
    BlockIR,
    DocumentIR,
    PageIR,
    assign_block_parents,
    classify_text_block,
    document_ir_from_json_dict,
    extract_entity_candidates,
    extract_table_structure,
)
from pdf_translate.memory_store import MemoryStore
from pdf_translate.qa.chunk_boundary import build_chunk_boundary_qa, build_chunk_strategy_comparison
from pdf_translate.qa.glossary_retranslation import (
    execute_glossary_retranslation,
    write_glossary_retranslation_publish,
    write_glossary_retranslation_plan,
    write_glossary_retranslation_rollback,
)
from pdf_translate.qa.metrics import build_experiment_metrics
from pdf_translate.qa.ocr_candidates import build_ocr_candidate_qa, write_ocr_candidate_qa
from pdf_translate.qa.repair_effectiveness import (
    build_repair_effectiveness,
    repair_effectiveness_to_markdown,
)
from pdf_translate.qa.repair import (
    apply_repair_patch_review_batch_decision,
    apply_repair_patch_review_decision,
    build_repair_plan,
    build_repair_patch_review,
    build_repair_requests,
    build_repair_results,
    build_repair_merge,
    build_repair_publish,
    build_repair_rollback,
    build_repair_formal_replace,
    build_repair_formal_rollback,
    build_repair_validation,
    repair_patch_review_to_markdown,
    write_repair_patch_review_batch_decision,
)
from pdf_translate.qa.structure import build_structure_qa
from pdf_translate.qa.table_reconstruction import (
    apply_table_merged_cell_review_batch_decision,
    apply_table_merged_cell_review_decision,
    build_confirmed_table_reconstruction,
    build_structure_hints_manifest,
    build_structure_translation_hints,
    build_table_merged_cell_review,
    build_table_reconstruction_report,
    build_table_translation_hints,
    effective_table_reconstruction_view,
    load_preferred_table_reconstruction,
    table_merged_cell_review_to_markdown,
    write_table_merged_cell_review_batch_decision,
    write_table_merged_cell_review_decision,
    write_table_structure_publish,
)
from pdf_translate.qa.translation import build_translation_qa, translation_qa_to_markdown
from pdf_translate.pipeline import _chunk_glossary_context, init_workdir, run_split, run_translate
from pdf_translate.run_metrics import build_run_metrics
from pdf_translate.translators.base import TranslationRequest
from pdf_translate.translators.factory import build_translator
from pdf_translate.translators.http_retry import call_with_http_retry, capture_http_retry_events
from pdf_translate.translators.openai_compatible import _build_user_message
from pdf_translate.vision.ocr_tasks import build_ocr_task_manifest, write_ocr_task_manifest
from pdf_translate.vision.ocr_executor import execute_ocr_tasks
from pdf_translate.vision.ocr_promotion import build_ocr_candidate_promotion, write_ocr_candidate_promotion
from pdf_translate.vision.ocr_writeback import (
    build_ocr_results_payload,
    build_ocr_writeback,
    load_ocr_results,
    write_ocr_results_payload,
    write_ocr_writeback,
)
from pdf_translate.vision.routing import build_vision_route, write_vision_route
from pdf_translate.zip_bundle import iter_bundle_files, map_bundle_arcname


def _test_app_config(**overrides) -> AppConfig:
    values = {
        "openai_api_key": None,
        "openai_base_url": "https://api.openai.com/v1",
        "openai_model": "gpt-test",
        "ollama_base_url": "http://127.0.0.1:11434/v1",
        "ollama_model": "llama-test",
        "deepl_api_key": None,
        "deepl_api_url": "https://api-free.deepl.com/v2/translate",
        "deepseek_api_key": None,
        "deepseek_base_url": "https://api.deepseek.com/v1",
        "deepseek_model": "deepseek-chat",
        "default_translator": "deepseek",
        "http_timeout_s": 120.0,
        "survey_enabled": False,
        "siliconflow_api_key": None,
        "siliconflow_base_url": "https://api.siliconflow.com/v1",
        "siliconflow_survey_model": "",
        "siliconflow_vision_model": "",
        "survey_max_text_chars": 12000,
        "planner_enabled": False,
        "planner_api_key": None,
        "planner_base_url": "https://api.siliconflow.com/v1",
        "planner_model": "",
        "cost_profile_json": "",
        "cost_profile_path": "",
        "cost_default_currency": "USD",
    }
    values.update(overrides)
    return AppConfig(**values)


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

    def test_assign_block_parents_links_cross_page_relationships(self) -> None:
        image = BlockIR("p1-b0000", 1, "image", "", (40, 80, 560, 300), 0)
        table = BlockIR(
            "p1-b0001",
            1,
            "table",
            "Metric Acc\nA 91.2",
            (60, 330, 500, 430),
            1,
            meta={"table": {"row_count": 2, "column_count": 2}},
        )
        paragraph = BlockIR("p1-b0002", 1, "paragraph", "The result is stable.", (60, 460, 500, 520), 2)
        figure_caption = BlockIR("p2-b0000", 2, "caption", "Fig. 1 Overview.", (60, 60, 500, 90), 0)
        table_caption = BlockIR("p2-b0001", 2, "caption", "Table 1: Results.", (60, 100, 500, 130), 1)
        footnote = BlockIR("p2-b0002", 2, "footnote", "1 Additional implementation note.", (60, 140, 500, 165), 2)
        far_caption = BlockIR("p3-b0000", 3, "caption", "Table 9: Far appendix.", (60, 60, 500, 90), 0)
        blocks = [image, table, paragraph, figure_caption, table_caption, footnote, far_caption]

        assign_block_parents(blocks)

        self.assertEqual(figure_caption.parent_id, image.block_id)
        self.assertTrue(figure_caption.meta["cross_page_parent"])
        self.assertTrue(figure_caption.meta["cross_page_parent_attempted"])
        self.assertEqual(figure_caption.meta["parent_page_no"], 1)
        self.assertEqual(figure_caption.meta["parent_page_gap"], 1)
        self.assertEqual(table_caption.parent_id, table.block_id)
        self.assertEqual(table_caption.meta["parent_relation"], "caption_for_table")
        self.assertTrue(table_caption.meta["cross_page_parent"])
        self.assertEqual(footnote.parent_id, paragraph.block_id)
        self.assertEqual(footnote.meta["parent_relation"], "footnote_for_block")
        self.assertTrue(footnote.meta["cross_page_parent"])
        self.assertIsNone(far_caption.parent_id)
        self.assertEqual(far_caption.meta["parent_warning"], "orphan_caption")
        self.assertTrue(far_caption.meta["cross_page_parent_attempted"])

        qa = build_structure_qa(
            DocumentIR(
                doc_id="cross-page-relationships",
                source_pdf="sample.pdf",
                pages=[
                    PageIR(1, 600, 800, "page 1", [image, table, paragraph]),
                    PageIR(2, 600, 800, "page 2", [figure_caption, table_caption, footnote]),
                    PageIR(3, 600, 800, "page 3", [far_caption]),
                ],
            )
        )
        self.assertEqual(qa["summary"]["caption_count"], 3)
        self.assertEqual(qa["summary"]["caption_linked_count"], 2)
        self.assertEqual(qa["summary"]["caption_orphan_count"], 1)
        self.assertEqual(qa["summary"]["cross_page_relationship_count"], 3)
        self.assertEqual(qa["summary"]["caption_cross_page_linked_count"], 2)
        self.assertEqual(qa["summary"]["caption_cross_page_orphan_count"], 1)
        self.assertEqual(qa["summary"]["footnote_cross_page_linked_count"], 1)
        self.assertEqual(qa["summary"]["footnote_cross_page_orphan_count"], 0)
        self.assertEqual(qa["summary"]["cross_page_parent_gap_max"], 1)
        relationships = {item["block_id"]: item for item in qa["relationships"]}
        self.assertTrue(relationships[figure_caption.block_id]["cross_page_parent"])
        self.assertEqual(relationships[figure_caption.block_id]["parent_page_no"], 1)
        self.assertTrue(relationships[far_caption.block_id]["cross_page_parent_attempted"])

    def test_structure_chunks_protect_cross_page_parent_relation(self) -> None:
        table = BlockIR(
            "p1-b0000",
            1,
            "table",
            "Metric Acc\nA 91.2",
            (60, 640, 500, 740),
            0,
            meta={"table": {"row_count": 2, "column_count": 2}},
        )
        caption = BlockIR("p2-b0000", 2, "caption", "Table 1: Results.", (60, 60, 500, 90), 0)
        assign_block_parents([table, caption])
        relation_id = "p1-b0000->p2-b0000:caption_for_table"
        doc_ir = DocumentIR(
            doc_id="cross-page-structure-chunk",
            source_pdf="sample.pdf",
            pages=[
                PageIR(1, 600, 800, table.text, [table]),
                PageIR(2, 600, 800, caption.text, [caption]),
            ],
        )

        chunks = build_structure_chunks(
            doc_ir,
            target_chars=1000,
            max_chars=2000,
            max_pages_per_chunk=1,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].block_ids, ["p1-b0000", "p2-b0000"])
        self.assertEqual(chunks[0].structural_relation_ids, [relation_id])
        self.assertIn(f"protected_structural_relation:{relation_id}", chunks[0].warnings)
        boundary_qa = build_chunk_boundary_qa(chunks, build_structure_qa(doc_ir), pipeline_variant="structure")
        self.assertEqual(boundary_qa["summary"]["structural_relation_protected_count"], 1)

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

    def test_chunk_glossary_context_uses_heading_structure_and_block_ids(self) -> None:
        doc_ir = DocumentIR(
            doc_id="glossary-context",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text="Methods\n\nAccuracy is measured.",
                    blocks=[
                        BlockIR("p1-b0000", 1, "heading", "2 Methods", (0, 0, 100, 20), 0),
                        BlockIR(
                            "p1-b0001",
                            1,
                            "paragraph",
                            "Accuracy is measured.",
                            (0, 30, 300, 90),
                            1,
                        ),
                    ],
                ),
                PageIR(
                    page_no=2,
                    width=600,
                    height=800,
                    text="Results\n\nMetric Acc F1\nA 91 88",
                    blocks=[
                        BlockIR("p2-b0000", 2, "heading", "3 Results", (0, 0, 100, 20), 0),
                        BlockIR(
                            "p2-b0001",
                            2,
                            "table",
                            "Metric Acc F1\nA 91 88",
                            (0, 30, 300, 90),
                            1,
                        ),
                    ],
                ),
            ],
        )

        chunks = build_structure_chunks(doc_ir, max_pages_per_chunk=1)
        methods_context = _chunk_glossary_context(chunks[0], doc_ir)
        results_context = _chunk_glossary_context(chunks[1], doc_ir)

        self.assertEqual(methods_context["section_scope"], ["2 Methods"])
        self.assertEqual(methods_context["block_ids"], ["p1-b0000", "p1-b0001"])
        self.assertIn("heading", methods_context["structure_types"])
        self.assertIn("paragraph", methods_context["structure_types"])
        self.assertEqual(results_context["section_scope"], ["3 Results"])
        self.assertIn("table", results_context["structure_types"])
        self.assertEqual(results_context["block_ids"], ["p2-b0000", "p2-b0001"])

    def test_structure_chunks_record_budget_metadata_and_split_reasons(self) -> None:
        first_text = ("Alpha method " * 85).strip() + "."
        second_text = "Second page starts independently."
        doc_ir = DocumentIR(
            doc_id="budget-sample",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text=first_text,
                    blocks=[
                        BlockIR("p1-b0000", 1, "paragraph", first_text, (40, 100, 520, 180), 0),
                    ],
                ),
                PageIR(
                    page_no=2,
                    width=600,
                    height=800,
                    text=second_text,
                    blocks=[
                        BlockIR("p2-b0000", 2, "paragraph", second_text, (40, 100, 520, 180), 0),
                    ],
                ),
            ],
        )
        chunks = build_structure_chunks(
            doc_ir,
            target_chars=1000,
            max_chars=2000,
            max_pages_per_chunk=3,
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].split_reason, "target_chars")
        self.assertEqual(chunks[0].budget_target_chars, 1000)
        self.assertEqual(chunks[0].budget_max_chars, 2000)
        self.assertGreater(chunks[0].approx_tokens, 0)
        manifest = chunks[0].to_manifest_entry()
        self.assertEqual(manifest["budget"]["split_reason"], "target_chars")
        self.assertEqual(manifest["budget"]["target_chars"], 1000)
        self.assertEqual(manifest["approx_tokens"], chunks[0].approx_tokens)

        boundary_qa = build_chunk_boundary_qa(
            chunks,
            build_structure_qa(doc_ir),
            pipeline_variant="structure",
        )
        self.assertEqual(boundary_qa["summary"]["budget_split_reason_counts"]["target_chars"], 1)
        self.assertIn("chunks", boundary_qa)
        self.assertEqual(boundary_qa["chunks"][0]["split_reason"], "target_chars")

    def test_structure_chunks_protect_structural_relation_under_budget_pressure(self) -> None:
        table_text = ("Metric Acc F1\n" + ("BERT 91 88\n" * 105)).strip()
        caption_text = "Table 1 explains " + ("domain shift " * 32)
        relation_id = "p1-b0000->p1-b0001:caption_for_table"
        doc_ir = DocumentIR(
            doc_id="relation-budget-sample",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text=f"{table_text}\n{caption_text}",
                    blocks=[
                        BlockIR("p1-b0000", 1, "table", table_text, (40, 100, 520, 360), 0),
                        BlockIR(
                            "p1-b0001",
                            1,
                            "caption",
                            caption_text,
                            (40, 370, 520, 420),
                            1,
                            parent_id="p1-b0000",
                            meta={"parent_relation": "caption_for_table"},
                        ),
                    ],
                ),
            ],
        )
        chunks = build_structure_chunks(
            doc_ir,
            target_chars=1000,
            max_chars=1200,
            max_pages_per_chunk=1,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].block_ids, ["p1-b0000", "p1-b0001"])
        self.assertEqual(chunks[0].structural_relation_ids, [relation_id])
        self.assertGreater(chunks[0].budget_overflow_chars, 0)
        self.assertEqual(chunks[0].budget_pressure, "over_max")
        self.assertIn(f"protected_structural_relation:{relation_id}", chunks[0].warnings)
        self.assertIn(f"budget_overflow_for_structural_relation:{relation_id}", chunks[0].warnings)

        boundary_qa = build_chunk_boundary_qa(
            chunks,
            build_structure_qa(doc_ir),
            pipeline_variant="structure",
        )
        self.assertEqual(boundary_qa["summary"]["structural_relation_protected_count"], 1)
        self.assertEqual(boundary_qa["summary"]["budget_overflow_chunk_count"], 1)
        self.assertEqual(boundary_qa["summary"]["budget_pressure_counts"]["over_max"], 1)

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
        self.assertEqual(report["summary"]["table_footnote_binding_count"], 1)
        self.assertEqual(report["summary"]["table_footnote_cell_binding_count"], 1)
        self.assertEqual(report["summary"]["table_footnote_bound_cell_count"], 1)
        self.assertEqual(report["summary"]["table_footnote_unbound_count"], 0)
        self.assertEqual(report["summary"]["merged_cell_candidate_count"], 0)
        self.assertEqual(report["summary"]["ragged_table_count"], 0)
        self.assertEqual(report["summary"]["empty_cell_count"], 0)
        self.assertEqual(report["summary"]["table_reconstruction_ready_rate"], 1.0)
        table_report = report["tables"][0]
        self.assertEqual(table_report["merged_cell_candidates"], [])
        self.assertEqual(table_report["caption_blocks"][0]["block_id"], "p1-b0000")
        self.assertEqual(table_report["footnote_blocks"][0]["block_id"], "p1-b0002")
        binding = table_report["footnote_bindings"][0]
        self.assertEqual(binding["status"], "bound_to_cells")
        self.assertEqual(binding["footnote_block_id"], "p1-b0002")
        self.assertEqual(binding["matched_cell_count"], 1)
        self.assertEqual(binding["matched_row_indices"], [1])
        self.assertEqual(binding["matched_column_indices"], [2])
        self.assertIn("*", binding["markers"])
        self.assertIn("p<0.05", binding["markers"])
        self.assertEqual(binding["matched_cells"][0]["row_index"], 1)
        self.assertEqual(binding["matched_cells"][0]["column_index"], 2)
        acc_cell = next(cell for cell in table_report["cells"] if cell["row_index"] == 1 and cell["column_index"] == 1)
        self.assertEqual(acc_cell["column_header"], "Acc")
        self.assertEqual(acc_cell["row_header"], "BERT")
        self.assertIn("91.2%", acc_cell["locked_tokens"])
        self.assertIn("%", acc_cell["locked_tokens"])
        hints = build_table_translation_hints(
            TextChunk("c0000", [0], table.text, 0, 0),
            report,
        )
        self.assertIn("footnote-cell bindings", hints)
        self.assertIn("p1-b0002", hints)
        self.assertIn("r1c2", hints)
        self.assertNotIn("合并单元格候选", hints)

    def test_table_reconstruction_reports_merged_cell_candidates(self) -> None:
        page1_table = BlockIR(
            "p1-b0000",
            1,
            "table",
            "Dataset metrics\nModel Acc F1\nA 91 88",
            (40, 640, 540, 760),
            0,
            meta={
                "table": {
                    "rows": [["Dataset metrics"], ["Model", "Acc", "F1"], ["A", "91", "88"]],
                    "row_count": 3,
                    "column_count": 3,
                    "header": ["Model", "Acc", "F1"],
                    "confidence": "medium",
                }
            },
        )
        page2_table = BlockIR(
            "p2-b0000",
            2,
            "table",
            "Model Acc F1\nB 92 89",
            (40, 80, 540, 180),
            0,
            meta={
                "table": {
                    "rows": [["Model", "Acc", "F1"], ["B", "92", "89"]],
                    "row_count": 2,
                    "column_count": 3,
                    "header": ["Model", "Acc", "F1"],
                    "confidence": "medium",
                }
            },
        )
        doc_ir = DocumentIR(
            doc_id="merged-cell-candidates",
            source_pdf="sample.pdf",
            pages=[
                PageIR(1, 600, 800, page1_table.text, [page1_table]),
                PageIR(2, 600, 800, page2_table.text, [page2_table]),
            ],
        )

        report = build_table_reconstruction_report(doc_ir, build_structure_qa(doc_ir))

        self.assertEqual(report["summary"]["table_count"], 2)
        self.assertEqual(report["summary"]["ragged_table_count"], 1)
        self.assertEqual(report["summary"]["ragged_row_count"], 1)
        self.assertEqual(report["summary"]["empty_cell_count"], 2)
        self.assertEqual(report["summary"]["merged_cell_candidate_count"], 1)
        self.assertEqual(report["summary"]["merged_cell_candidate_type_counts"]["colspan"], 1)
        self.assertEqual(report["summary"]["merged_cell_candidate_reason_counts"]["single_cell_ragged_row"], 1)
        self.assertEqual(report["summary"]["continued_table_merged_cell_candidate_count"], 1)
        table = report["tables"][0]
        self.assertIn("ragged_table_rows", table["warnings"])
        self.assertEqual(table["ragged_row_indices"], [0])
        self.assertEqual(table["empty_cell_count"], 2)
        candidate = table["merged_cell_candidates"][0]
        self.assertEqual(candidate["span_type"], "colspan")
        self.assertEqual(candidate["row_index"], 0)
        self.assertEqual(candidate["column_index"], 0)
        self.assertEqual(candidate["column_span"], 3)
        self.assertEqual(candidate["covered_cells"], [{"row_index": 0, "column_index": 1}, {"row_index": 0, "column_index": 2}])
        group = report["continued_table_groups"][0]
        self.assertEqual(group["merged_cell_candidate_count"], 1)
        self.assertEqual(group["merged_cell_candidates"][0]["source_table_id"], "p1-b0000")
        hints = build_table_translation_hints(
            TextChunk("c0000", [0, 1], page1_table.text + "\n" + page2_table.text, 0, 0),
            report,
        )
        self.assertIn("疑似合并单元格候选", hints)
        self.assertIn("未确认", hints)
        self.assertIn("不作为已确认合并结构处理", hints)
        self.assertIn("续表组内疑似合并单元格候选 1 个", hints)
        self.assertIn("p1-b0000:r0c0", hints)
        self.assertIn("疑似跨列候选(colspan 1x3)", hints)
        self.assertIn("覆盖候选空位=r0c1,r0c2", hints)
        self.assertIn("single_cell_ragged_row", hints)
        self.assertIn("Dataset metrics", hints)
        self.assertNotIn("已确认合并单元格", hints)
        hint_manifest = build_structure_hints_manifest(
            [TextChunk("c0000", [0, 1], page1_table.text + "\n" + page2_table.text, 0, 0)],
            report,
        )
        self.assertEqual(hint_manifest["schema_version"], "structure-hints-manifest-v1")
        self.assertEqual(hint_manifest["summary"]["chunk_count"], 1)
        self.assertEqual(hint_manifest["summary"]["structure_hint_chunk_count"], 1)
        self.assertEqual(hint_manifest["summary"]["structure_hint_table_count"], 2)
        self.assertEqual(hint_manifest["summary"]["structure_hint_continued_group_count"], 1)
        self.assertEqual(hint_manifest["summary"]["structure_hint_merged_cell_candidate_count"], 1)
        self.assertEqual(hint_manifest["summary"]["structure_hint_empty_chunk_count"], 0)
        self.assertEqual(hint_manifest["summary"]["structure_hint_max_char_count"], len(hint_manifest["chunks"][0]["hint_text"]))
        self.assertEqual(hint_manifest["summary"]["structure_hint_avg_char_count"], len(hint_manifest["chunks"][0]["hint_text"]))
        self.assertEqual(hint_manifest["summary"]["structure_hint_merged_cell_candidate_type_counts"]["colspan"], 1)
        self.assertEqual(
            hint_manifest["summary"]["structure_hint_merged_cell_candidate_reason_counts"]["single_cell_ragged_row"],
            1,
        )
        self.assertGreater(hint_manifest["summary"]["structure_hint_char_count"], 0)
        self.assertEqual(hint_manifest["chunks"][0]["table_ids"], ["p1-b0000", "p2-b0000"])
        self.assertEqual(hint_manifest["chunks"][0]["continued_table_group_count"], 1)
        self.assertEqual(hint_manifest["chunks"][0]["hint_char_count"], len(hint_manifest["chunks"][0]["hint_text"]))
        self.assertEqual(hint_manifest["chunks"][0]["merged_cell_candidate_type_counts"]["colspan"], 1)
        self.assertIn("疑似合并单元格候选", hint_manifest["chunks"][0]["hint_text"])

    def test_structure_translation_hints_include_relationships_and_entities(self) -> None:
        caption = BlockIR(
            "p1-b0000",
            1,
            "caption",
            "Table 1. Accuracy results on ImageNet.",
            (40, 80, 540, 110),
            0,
        )
        table = BlockIR(
            "p1-b0001",
            1,
            "table",
            "Model Acc\nBERT 91.2*",
            (40, 120, 540, 220),
            1,
            meta={
                "table": {
                    "rows": [["Model", "Acc"], ["BERT", "91.2*"]],
                    "row_count": 2,
                    "column_count": 2,
                    "header": ["Model", "Acc"],
                    "confidence": "medium",
                }
            },
        )
        footnote = BlockIR(
            "p1-b0002",
            1,
            "footnote",
            "* p < 0.05.",
            (40, 230, 540, 250),
            2,
        )
        paragraph = BlockIR(
            "p1-b0003",
            1,
            "paragraph",
            "BERT was evaluated by Stanford University.",
            (40, 270, 540, 310),
            3,
            meta={
                "entities": [
                    {
                        "type": "model_or_dataset",
                        "text": "BERT",
                        "source": "model_dataset_pattern",
                        "confidence": "medium",
                    },
                    {
                        "type": "organization",
                        "text": "Stanford University",
                        "source": "organization_suffix",
                        "confidence": "medium",
                    },
                ]
            },
        )
        blocks = [caption, table, footnote, paragraph]
        assign_block_parents(blocks)
        doc_ir = DocumentIR(
            doc_id="relationship-hints",
            source_pdf="sample.pdf",
            pages=[PageIR(1, 600, 800, "\n".join(block.text for block in blocks), blocks)],
        )
        report = build_table_reconstruction_report(doc_ir, build_structure_qa(doc_ir))
        chunk = TextChunk("c0000", [0], doc_ir.pages[0].text, 0, 0)
        chunk.block_ids = [block.block_id for block in blocks]

        hints = build_structure_translation_hints(chunk, report, doc_ir)

        self.assertIn("以下表格结构来自本地 DocumentIR", hints)
        self.assertIn("以下图注/脚注归属来自本地 DocumentIR", hints)
        self.assertIn("caption_for_table", hints)
        self.assertIn("footnote_for_table", hints)
        self.assertIn("p1-b0000(caption) -> p1-b0001(table)", hints)
        self.assertIn("p1-b0002(footnote) -> p1-b0001(table)", hints)
        self.assertIn("以下实体候选", hints)
        self.assertIn("BERT", hints)
        self.assertIn("Stanford University", hints)

        hint_manifest = build_structure_hints_manifest([chunk], report, doc_ir)
        entry = hint_manifest["chunks"][0]
        self.assertEqual(entry["relationship_count"], 2)
        self.assertEqual(entry["relationship_type_counts"]["caption_for_table"], 1)
        self.assertEqual(entry["relationship_type_counts"]["footnote_for_table"], 1)
        self.assertEqual(entry["entity_hint_count"], 2)
        self.assertEqual(hint_manifest["summary"]["structure_hint_relationship_count"], 2)
        self.assertEqual(hint_manifest["summary"]["structure_hint_entity_count"], 2)
        self.assertEqual(
            hint_manifest["summary"]["structure_hint_relationship_type_counts"]["caption_for_table"],
            1,
        )

    def test_structure_translation_hints_record_cross_page_relationships(self) -> None:
        table = BlockIR(
            "p1-b0000",
            1,
            "table",
            "Model Acc\nA 91.2",
            (40, 680, 540, 760),
            0,
            meta={
                "table": {
                    "rows": [["Model", "Acc"], ["A", "91.2"]],
                    "row_count": 2,
                    "column_count": 2,
                    "header": ["Model", "Acc"],
                    "confidence": "medium",
                }
            },
        )
        caption = BlockIR(
            "p2-b0000",
            2,
            "caption",
            "Table 1. Continued accuracy results.",
            (40, 70, 540, 100),
            0,
        )
        blocks = [table, caption]
        assign_block_parents(blocks)
        doc_ir = DocumentIR(
            doc_id="cross-page-relationship-hints",
            source_pdf="sample.pdf",
            pages=[
                PageIR(1, 600, 800, table.text, [table]),
                PageIR(2, 600, 800, caption.text, [caption]),
            ],
        )
        report = build_table_reconstruction_report(doc_ir, build_structure_qa(doc_ir))
        chunk = TextChunk("c0000", [0, 1], table.text + "\n" + caption.text, 0, 0)

        hints = build_structure_translation_hints(chunk, report, doc_ir)

        self.assertIn("caption_for_table", hints)
        self.assertIn("跨页", hints)
        self.assertIn("页差=1", hints)

        hint_manifest = build_structure_hints_manifest([chunk], report, doc_ir)
        self.assertEqual(hint_manifest["chunks"][0]["relationship_cross_page_count"], 1)
        self.assertEqual(hint_manifest["summary"]["structure_hint_relationship_count"], 1)
        self.assertEqual(hint_manifest["summary"]["structure_hint_relationship_cross_page_count"], 1)

    def test_table_reconstruction_uses_meta_merged_cell_candidates(self) -> None:
        table = BlockIR(
            "p1-b0000",
            1,
            "table",
            "Dataset metrics\nModel Acc F1\nA 91 88",
            (40, 640, 540, 760),
            0,
            meta={
                "table": {
                    "source": "ocr_candidate_promotion",
                    "source_task_id": "ocr-task-0001",
                    "source_engine": "plain_text_table_ocr",
                    "rows": [["Dataset metrics", "", ""], ["Model", "Acc", "F1"], ["A", "91", "88"]],
                    "row_count": 3,
                    "column_count": 3,
                    "header": ["Model", "Acc", "F1"],
                    "confidence": "high",
                    "merged_cell_candidates": [
                        {
                            "type": "colspan",
                            "row": 0,
                            "cols": [0, 1, 2],
                            "text": "Dataset metrics",
                            "reason": "single_cell_ragged_row",
                            "confidence": 0.91,
                            "source": "local_text_table_parser",
                        }
                    ],
                }
            },
        )
        doc_ir = DocumentIR(
            doc_id="meta-merged-cell-candidates",
            source_pdf="sample.pdf",
            pages=[PageIR(1, 600, 800, table.text, [table])],
        )

        report = build_table_reconstruction_report(doc_ir, build_structure_qa(doc_ir))

        self.assertEqual(report["summary"]["table_count"], 1)
        self.assertEqual(report["summary"]["merged_cell_candidate_count"], 1)
        self.assertEqual(report["summary"]["merged_cell_candidate_type_counts"]["colspan"], 1)
        self.assertEqual(report["summary"]["merged_cell_candidate_reason_counts"]["single_cell_ragged_row"], 1)
        table_report = report["tables"][0]
        self.assertEqual(table_report["merged_cell_candidate_count"], 1)
        candidate = table_report["merged_cell_candidates"][0]
        self.assertEqual(candidate["span_type"], "colspan")
        self.assertEqual(candidate["type"], "colspan")
        self.assertEqual(candidate["row_index"], 0)
        self.assertEqual(candidate["row"], 0)
        self.assertEqual(candidate["column_index"], 0)
        self.assertEqual(candidate["col"], 0)
        self.assertEqual(candidate["row_span"], 1)
        self.assertEqual(candidate["column_span"], 3)
        self.assertEqual(candidate["cols"], [0, 1, 2])
        self.assertEqual(
            [(cell["row_index"], cell["column_index"]) for cell in candidate["covered_cells"]],
            [(0, 1), (0, 2)],
        )
        self.assertEqual(candidate["source"], "local_text_table_parser")
        self.assertEqual(candidate["source_task_id"], "ocr-task-0001")
        self.assertEqual(candidate["engine"], "plain_text_table_ocr")
        self.assertEqual(candidate["candidate_status"], "candidate")
        self.assertEqual(candidate["visual_evidence_level"], "none")
        self.assertEqual(candidate["bbox_evidence"]["status"], "missing")

        hints = build_table_translation_hints(TextChunk("c0000", [0], table.text, 0, 0), report)
        self.assertIn("r0c0", hints)
        self.assertIn("colspan 1x3", hints)
        self.assertIn("single_cell_ragged_row", hints)
        self.assertIn("Dataset metrics", hints)
        hint_manifest = build_structure_hints_manifest([TextChunk("c0000", [0], table.text, 0, 0)], report)
        self.assertEqual(hint_manifest["summary"]["structure_hint_merged_cell_candidate_count"], 1)
        self.assertEqual(hint_manifest["summary"]["structure_hint_merged_cell_candidate_type_counts"]["colspan"], 1)
        self.assertEqual(
            hint_manifest["summary"]["structure_hint_merged_cell_candidate_reason_counts"]["single_cell_ragged_row"],
            1,
        )

    def test_table_reconstruction_marks_visual_span_bbox_evidence(self) -> None:
        table = BlockIR(
            "p1-b0000",
            1,
            "table",
            "Dataset metrics\nModel Acc F1\nA 91 88",
            (40, 640, 540, 760),
            0,
            meta={
                "table": {
                    "rows": [["Dataset metrics", "", ""], ["Model", "Acc", "F1"], ["A", "91", "88"]],
                    "row_count": 3,
                    "column_count": 3,
                    "header": ["Model", "Acc", "F1"],
                    "confidence": "high",
                    "cell_bboxes": [
                        {"row": 0, "col": 0, "bbox": [40, 640, 200, 680]},
                        {"row": 0, "col": 1, "bbox": [200, 640, 360, 680]},
                        {"row": 0, "col": 2, "bbox": [360, 640, 540, 680]},
                        {"row": 1, "col": 0, "bbox": [40, 680, 200, 720]},
                        {"row": 1, "col": 1, "bbox": [200, 680, 360, 720]},
                        {"row": 1, "col": 2, "bbox": [360, 680, 540, 720]},
                        {"row": 2, "col": 0, "bbox": [40, 720, 200, 760]},
                        {"row": 2, "col": 1, "bbox": [200, 720, 360, 760]},
                        {"row": 2, "col": 2, "bbox": [360, 720, 540, 760]},
                    ],
                    "merged_cell_candidates": [
                        {
                            "type": "colspan",
                            "row": 0,
                            "cols": [0, 1, 2],
                            "bbox": [40, 640, 540, 680],
                            "text": "Dataset metrics",
                            "reason": "visual_header_span",
                            "confidence": 0.94,
                            "source": "layout_table_ocr",
                        }
                    ],
                }
            },
        )
        doc_ir = DocumentIR(
            doc_id="visual-span-bbox-candidates",
            source_pdf="sample.pdf",
            pages=[PageIR(1, 600, 800, table.text, [table])],
        )

        report = build_table_reconstruction_report(doc_ir, build_structure_qa(doc_ir))

        self.assertEqual(report["summary"]["merged_cell_candidate_count"], 1)
        self.assertEqual(report["summary"]["merged_cell_candidate_status_counts"]["visually_supported"], 1)
        self.assertEqual(
            report["summary"]["merged_cell_candidate_visual_evidence_counts"]["visual_span_bbox"],
            1,
        )
        self.assertEqual(report["summary"]["merged_cell_candidate_bbox_evidence_counts"]["span_reported"], 1)
        candidate = report["tables"][0]["merged_cell_candidates"][0]
        self.assertEqual(candidate["candidate_status"], "visually_supported")
        self.assertEqual(candidate["visual_evidence_level"], "visual_span_bbox")
        self.assertEqual(candidate["bbox_evidence"]["status"], "span_reported")
        self.assertEqual(candidate["bbox_evidence"]["support_status"], "visual_span_supported")
        self.assertEqual(candidate["bbox_evidence"]["evidence_bbox_coverage"], 1.0)
        self.assertNotEqual(candidate["candidate_status"], "human_confirmed")

        hints = build_table_translation_hints(TextChunk("c0000", [0], table.text, 0, 0), report)
        self.assertIn("证据=span_reported/visual_span_bbox/visually_supported", hints)
        hint_manifest = build_structure_hints_manifest([TextChunk("c0000", [0], table.text, 0, 0)], report)
        self.assertEqual(
            hint_manifest["summary"]["structure_hint_merged_cell_candidate_status_counts"]["visually_supported"],
            1,
        )
        self.assertEqual(
            hint_manifest["summary"]["structure_hint_merged_cell_candidate_visual_evidence_counts"][
                "visual_span_bbox"
            ],
            1,
        )

    def test_table_merged_cell_review_summarizes_confirmation_status(self) -> None:
        review = build_table_merged_cell_review(
            {
                "schema_version": "table-reconstruction-v1",
                "doc_id": "review-doc",
                "tables": [
                    {
                        "table_id": "p1-b0000",
                        "block_id": "p1-b0000",
                        "page_no": 1,
                        "merged_cell_candidates": [
                            {
                                "span_type": "colspan",
                                "row_index": 0,
                                "column_index": 0,
                                "row_span": 1,
                                "column_span": 3,
                                "text": "Dataset metrics",
                                "reason": "visual_header_span",
                                "confidence": "0.94",
                                "source": "layout_table_ocr",
                                "candidate_status": "visually_supported",
                                "visual_evidence_level": "visual_span_bbox",
                                "bbox_evidence": {"status": "span_reported"},
                                "covered_cells": [
                                    {"row_index": 0, "column_index": 1},
                                    {"row_index": 0, "column_index": 2},
                                ],
                            },
                            {
                                "span_type": "colspan",
                                "row_index": 1,
                                "column_index": 0,
                                "row_span": 1,
                                "column_span": 2,
                                "text": "Ablation",
                                "reason": "single_cell_ragged_row",
                                "confidence": "medium",
                                "source": "local_text_table_parser",
                                "candidate_status": "candidate",
                                "visual_evidence_level": "estimated_bbox",
                                "bbox_evidence": {"status": "estimated"},
                            },
                            {
                                "span_type": "rowspan",
                                "row_index": 2,
                                "column_index": 0,
                                "row_span": 2,
                                "column_span": 1,
                                "text": "BERT",
                                "reason": "manual_review",
                                "confidence": "high",
                                "candidate_status": "human_confirmed",
                                "visual_evidence_level": "manual_verified",
                                "bbox_evidence": {"status": "manual_verified"},
                            },
                            {
                                "span_type": "colspan",
                                "row_index": 4,
                                "column_index": 0,
                                "row_span": 1,
                                "column_span": 2,
                                "text": "noise",
                                "reason": "manual_review",
                                "confidence": "low",
                                "candidate_status": "rejected",
                                "visual_evidence_level": "none",
                                "bbox_evidence": {"status": "missing"},
                            },
                        ],
                    }
                ],
            }
        )

        self.assertEqual(review["schema_version"], "table-merged-cell-review-v1")
        self.assertEqual(review["summary"]["candidate_review_count"], 4)
        self.assertEqual(review["summary"]["review_required_count"], 2)
        self.assertEqual(review["summary"]["pending_review_count"], 2)
        self.assertEqual(review["summary"]["visual_supported_count"], 1)
        self.assertEqual(review["summary"]["estimated_only_count"], 1)
        self.assertEqual(review["summary"]["human_confirmed_count"], 1)
        self.assertEqual(review["summary"]["rejected_count"], 1)
        self.assertEqual(review["summary"]["default_decision_counts"]["needs_human_confirmation"], 1)
        self.assertEqual(review["summary"]["default_decision_counts"]["needs_visual_review"], 1)
        self.assertEqual(review["summary"]["human_decision_counts"]["confirm"], 1)
        self.assertEqual(review["summary"]["human_decision_counts"]["reject"], 1)
        self.assertEqual(review["summary"]["human_decision_counts"]["pending"], 2)
        first = review["candidate_reviews"][0]
        self.assertEqual(first["review_id"], "tmc-0001-p1-b0000-r0c0")
        self.assertEqual(first["confirmation_status"], "pending_review")
        self.assertEqual(first["human_decision"], "")
        self.assertEqual(first["bbox_evidence_status"], "span_reported")
        confirmed = review["candidate_reviews"][2]
        self.assertEqual(confirmed["confirmation_status"], "human_confirmed")
        self.assertEqual(confirmed["human_decision"], "confirm")

        markdown = table_merged_cell_review_to_markdown(review)
        self.assertIn("表格合并单元格候选人工确认清单", markdown)
        self.assertIn("tmc-0001-p1-b0000-r0c0", markdown)
        self.assertIn("needs_human_confirmation", markdown)
        self.assertIn("视觉支持不等于人工确认", markdown)

        updated = apply_table_merged_cell_review_decision(
            review,
            "tmc-0001-p1-b0000-r0c0",
            decision="confirm",
            reviewer="mentor",
            comment="bbox matches header span",
            reviewed_at="2026-07-06T00:00:00+00:00",
        )
        first = updated["candidate_reviews"][0]
        self.assertEqual(first["human_decision"], "confirm")
        self.assertEqual(first["confirmation_status"], "human_confirmed")
        self.assertEqual(first["reviewed_by"], "mentor")
        self.assertEqual(updated["summary"]["review_required_count"], 1)
        self.assertEqual(updated["summary"]["pending_review_count"], 1)
        self.assertEqual(updated["summary"]["human_reviewed_count"], 3)
        self.assertEqual(updated["summary"]["human_confirmed_count"], 2)
        self.assertIn("bbox matches header span", table_merged_cell_review_to_markdown(updated))

        cleared = apply_table_merged_cell_review_decision(
            updated,
            "tmc-0001-p1-b0000-r0c0",
            decision="clear",
            reviewer="mentor",
        )
        self.assertEqual(cleared["candidate_reviews"][0]["human_decision"], "")
        self.assertEqual(cleared["candidate_reviews"][0]["confirmation_status"], "pending_review")
        self.assertEqual(cleared["summary"]["review_required_count"], 2)

        root = Path.cwd() / "test-output" / "table-merged-cell-review-decision"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            json_path = root / "table_merged_cell_review.json"
            md_path = root / "table_merged_cell_review.md"
            json_path.write_text(json.dumps(cleared, ensure_ascii=False, indent=2), encoding="utf-8")
            md_path.write_text(table_merged_cell_review_to_markdown(cleared), encoding="utf-8")
            persisted = write_table_merged_cell_review_decision(
                json_path,
                md_path,
                "tmc-0002-p1-b0000-r1c0",
                decision="needs_revision",
                reviewer="mentor",
                comment="needs another crop",
                reviewed_at="2026-07-06T00:05:00+00:00",
            )
            self.assertEqual(persisted["summary"]["needs_revision_count"], 1)
            self.assertEqual(persisted["summary"]["review_required_count"], 2)
            self.assertIn("needs another crop", md_path.read_text(encoding="utf-8"))
            stored = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["candidate_reviews"][1]["human_decision"], "needs_revision")
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_table_merged_cell_review_batch_decision_updates_atomically(self) -> None:
        review = build_table_merged_cell_review(
            {
                "schema_version": "table-reconstruction-v1",
                "doc_id": "batch-review-doc",
                "tables": [
                    {
                        "table_id": "p1-b0000",
                        "block_id": "p1-b0000",
                        "page_no": 1,
                        "merged_cell_candidates": [
                            {
                                "span_type": "colspan",
                                "row_index": 0,
                                "column_index": 0,
                                "row_span": 1,
                                "column_span": 2,
                                "text": "Dataset metrics",
                                "reason": "visual_header_span",
                                "candidate_status": "visually_supported",
                                "visual_evidence_level": "visual_span_bbox",
                                "bbox_evidence": {"status": "span_reported"},
                            },
                            {
                                "span_type": "colspan",
                                "row_index": 1,
                                "column_index": 0,
                                "row_span": 1,
                                "column_span": 2,
                                "text": "Ablation",
                                "reason": "single_cell_ragged_row",
                                "candidate_status": "candidate",
                                "visual_evidence_level": "estimated_bbox",
                                "bbox_evidence": {"status": "estimated"},
                            },
                            {
                                "span_type": "rowspan",
                                "row_index": 2,
                                "column_index": 0,
                                "row_span": 2,
                                "column_span": 1,
                                "text": "BERT",
                                "reason": "empty_cell_span",
                                "candidate_status": "candidate",
                                "visual_evidence_level": "none",
                                "bbox_evidence": {"status": "missing"},
                            },
                        ],
                    }
                ],
            }
        )

        with self.assertRaises(KeyError):
            apply_table_merged_cell_review_batch_decision(
                review,
                ["tmc-0001-p1-b0000-r0c0", "missing-review-id"],
                decision="confirm",
                reviewer="mentor",
                reviewed_at="2026-07-06T00:10:00+00:00",
            )
        self.assertEqual(review["candidate_reviews"][0]["human_decision"], "")

        updated = apply_table_merged_cell_review_batch_decision(
            review,
            ["tmc-0001-p1-b0000-r0c0", "tmc-0002-p1-b0000-r1c0"],
            decision="confirm",
            reviewer="mentor",
            comment="batch visual pass",
            reviewed_at="2026-07-06T00:11:00+00:00",
        )

        self.assertEqual(updated["summary"]["human_confirmed_count"], 2)
        self.assertEqual(updated["summary"]["human_reviewed_count"], 2)
        self.assertEqual(updated["summary"]["review_required_count"], 1)
        self.assertEqual(updated["candidate_reviews"][0]["human_comment"], "batch visual pass")
        self.assertEqual(updated["candidate_reviews"][1]["reviewed_by"], "mentor")
        self.assertEqual(updated["candidate_reviews"][1]["reviewed_at"], "2026-07-06T00:11:00+00:00")

        root = Path.cwd() / "test-output" / "table-merged-cell-review-batch-decision"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            json_path = root / "table_merged_cell_review.json"
            md_path = root / "table_merged_cell_review.md"
            json_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
            md_path.write_text(table_merged_cell_review_to_markdown(updated), encoding="utf-8")
            persisted = write_table_merged_cell_review_batch_decision(
                json_path,
                md_path,
                ["tmc-0001-p1-b0000-r0c0", "tmc-0002-p1-b0000-r1c0"],
                decision="clear",
                reviewer="mentor",
                reviewed_at="2026-07-06T00:12:00+00:00",
            )
            self.assertEqual(persisted["summary"]["human_confirmed_count"], 0)
            self.assertEqual(persisted["summary"]["review_required_count"], 3)
            stored = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["candidate_reviews"][0]["human_decision"], "")
            self.assertIn("pending", md_path.read_text(encoding="utf-8"))
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_table_structure_publish_uses_human_review_without_overwriting_source(self) -> None:
        table_reconstruction = {
            "schema_version": "table-reconstruction-v1",
            "doc_id": "publish-doc",
            "summary": {"table_count": 1},
            "tables": [
                {
                    "table_id": "p1-b0000",
                    "block_id": "p1-b0000",
                    "page_no": 1,
                    "merged_cell_candidates": [
                        {
                            "span_type": "colspan",
                            "row_index": 0,
                            "column_index": 0,
                            "row_span": 1,
                            "column_span": 3,
                            "text": "Dataset metrics",
                            "candidate_status": "visually_supported",
                        },
                        {
                            "span_type": "rowspan",
                            "row_index": 1,
                            "column_index": 0,
                            "row_span": 2,
                            "column_span": 1,
                            "text": "BERT",
                            "candidate_status": "candidate",
                        },
                    ],
                    "cells": [
                        {"row_index": 0, "column_index": 0, "text": "Dataset metrics", "role": "header"},
                        {"row_index": 0, "column_index": 1, "text": "", "role": "header"},
                        {"row_index": 0, "column_index": 2, "text": "", "role": "header"},
                        {"row_index": 1, "column_index": 0, "text": "BERT", "role": "row_header"},
                    ],
                }
            ],
        }
        review = build_table_merged_cell_review(table_reconstruction)
        review = apply_table_merged_cell_review_decision(
            review,
            "tmc-0001-p1-b0000-r0c0",
            decision="confirm",
            reviewer="mentor",
            reviewed_at="2026-07-06T01:00:00+00:00",
        )
        review = apply_table_merged_cell_review_decision(
            review,
            "tmc-0002-p1-b0000-r1c0",
            decision="reject",
            reviewer="mentor",
            reviewed_at="2026-07-06T01:05:00+00:00",
        )

        confirmed = build_confirmed_table_reconstruction(table_reconstruction, review)

        self.assertEqual(table_reconstruction["tables"][0]["merged_cell_candidates"][0]["candidate_status"], "visually_supported")
        self.assertEqual(confirmed["confirmation_schema_version"], "table-structure-publish-v1")
        self.assertEqual(confirmed["summary"]["confirmed_merged_cell_candidate_count"], 1)
        self.assertEqual(confirmed["summary"]["rejected_merged_cell_candidate_count"], 1)
        table = confirmed["tables"][0]
        self.assertEqual(table["confirmed_merged_cell_candidate_count"], 1)
        self.assertEqual(table["confirmed_merged_cell_candidates"][0]["candidate_status"], "human_confirmed")
        self.assertTrue(table["confirmed_merged_cell_candidates"][0]["effective_for_publish"])
        self.assertEqual(table["merged_cell_candidates"][1]["candidate_status"], "rejected")
        self.assertFalse(table["merged_cell_candidates"][1]["effective_for_publish"])
        self.assertEqual(confirmed["summary"]["table_structure_patch_count"], 1)
        self.assertEqual(confirmed["summary"]["table_structure_patch_applied_count"], 1)
        self.assertEqual(confirmed["summary"]["table_structure_patch_covered_cell_count"], 2)
        self.assertTrue(confirmed["summary"]["table_structure_patch_rollback_available"])
        patch = confirmed["table_structure_patches"][0]
        self.assertEqual(patch["patch_type"], "merged_cell_span")
        self.assertEqual(patch["operation"], "apply_confirmed_merged_cell_span")
        self.assertEqual(patch["source_review_id"], "tmc-0001-p1-b0000-r0c0")
        self.assertEqual(patch["span"]["span_type"], "colspan")
        self.assertEqual(patch["span"]["column_span"], 3)
        self.assertEqual(len(patch["covered_cells"]), 2)
        self.assertTrue(patch["rollback_available"])
        self.assertEqual(table["structure_patches"][0]["patch_id"], patch["patch_id"])
        self.assertEqual(table["cells"][0]["structure_patch_role"], "anchor")
        self.assertEqual(table["cells"][0]["effective_column_span"], 3)
        self.assertEqual(table["cells"][1]["structure_patch_role"], "covered")
        self.assertTrue(table["cells"][1]["suppress_in_render"])

        root = Path.cwd() / "test-output" / "table-structure-publish"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            report = write_table_structure_publish(
                table_reconstruction,
                review,
                root / "table_structure_publish.json",
                root / "table_structure_publish.md",
                confirm=True,
                published_reconstruction_path=root / "table_reconstruction_confirmed.json",
            )
            self.assertTrue(report["summary"]["published"])
            self.assertEqual(report["summary"]["applied_confirmed_count"], 1)
            self.assertEqual(report["summary"]["structure_patch_count"], 1)
            self.assertEqual(report["summary"]["structure_patch_applied_count"], 1)
            self.assertEqual(report["summary"]["structure_patch_covered_cell_count"], 2)
            self.assertTrue(report["summary"]["structure_patch_rollback_available"])
            self.assertTrue((root / "table_reconstruction_confirmed.json").is_file())
            published = json.loads((root / "table_reconstruction_confirmed.json").read_text(encoding="utf-8"))
            self.assertEqual(published["tables"][0]["confirmed_merged_cell_candidate_count"], 1)
            self.assertEqual(published["summary"]["table_structure_patch_count"], 1)
            self.assertEqual(published["tables"][0]["table_structure_patch_count"], 1)
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_preferred_table_reconstruction_uses_confirmed_effective_view(self) -> None:
        root = Path.cwd() / "test-output" / "preferred-table-reconstruction"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        raw = {
            "schema_version": "table-reconstruction-v1",
            "summary": {"merged_cell_candidate_count": 2},
            "tables": [
                {
                    "table_id": "p1-b0000",
                    "block_id": "p1-b0000",
                    "page_no": 1,
                    "merged_cell_candidates": [
                        {"span_type": "colspan", "candidate_status": "candidate", "reason": "raw_a"},
                        {"span_type": "rowspan", "candidate_status": "candidate", "reason": "raw_b"},
                    ],
                }
            ],
        }
        confirmed = {
            **raw,
            "confirmation_schema_version": "table-structure-publish-v1",
            "summary": {"confirmed_merged_cell_candidate_count": 1},
            "tables": [
                {
                    **raw["tables"][0],
                    "confirmed_merged_cell_candidates": [
                        {
                            "span_type": "colspan",
                            "candidate_status": "human_confirmed",
                            "reason": "confirmed_a",
                        }
                    ],
                }
            ],
        }
        try:
            (root / "table_reconstruction.json").write_text(json.dumps(raw), encoding="utf-8")
            (root / "table_reconstruction_confirmed.json").write_text(
                json.dumps(confirmed),
                encoding="utf-8",
            )

            preferred = load_preferred_table_reconstruction(root, fallback=raw)

            self.assertEqual(preferred["summary"]["table_structure_source"], "confirmed")
            self.assertEqual(preferred["summary"]["merged_cell_candidate_count"], 1)
            self.assertEqual(preferred["summary"]["confirmed_merged_cell_candidate_count"], 1)
            self.assertEqual(preferred["summary"]["table_structure_patch_count"], 0)
            self.assertEqual(len(preferred["tables"][0]["merged_cell_candidates"]), 1)
            self.assertEqual(preferred["tables"][0]["raw_merged_cell_candidate_count"], 2)
            self.assertEqual(raw["summary"]["merged_cell_candidate_count"], 2)

            (root / "table_reconstruction_confirmed.json").write_text("{bad", encoding="utf-8")
            fallback = load_preferred_table_reconstruction(root, fallback=raw)
            self.assertEqual(fallback["summary"]["table_structure_source"], "source")
            self.assertEqual(fallback["summary"]["merged_cell_candidate_count"], 2)

            direct = effective_table_reconstruction_view(confirmed)
            self.assertEqual(direct["summary"]["table_structure_source"], "confirmed")
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_table_reconstruction_builds_continued_table_groups(self) -> None:
        page1_table = BlockIR(
            "p1-b0000",
            1,
            "table",
            "Model Acc\nA 91\nB 92",
            (40, 640, 540, 760),
            0,
            meta={
                "table": {
                    "rows": [["Model", "Acc"], ["A", "91"], ["B", "92"]],
                    "row_count": 3,
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
            "Model Acc\nC 93\nD 94",
            (40, 80, 540, 180),
            0,
            meta={
                "table": {
                    "rows": [["Model", "Acc"], ["C", "93"], ["D", "94"]],
                    "row_count": 3,
                    "column_count": 2,
                    "header": ["Model", "Acc"],
                    "confidence": "medium",
                }
            },
        )
        doc_ir = DocumentIR(
            doc_id="continued-table-group",
            source_pdf="sample.pdf",
            pages=[
                PageIR(1, 600, 800, page1_table.text, [page1_table]),
                PageIR(2, 600, 800, page2_table.text, [page2_table]),
            ],
        )
        structure_qa = build_structure_qa(doc_ir)

        report = build_table_reconstruction_report(doc_ir, structure_qa)

        self.assertEqual(report["summary"]["continuation_group_count"], 1)
        self.assertEqual(report["summary"]["continued_table_group_count"], 1)
        self.assertEqual(report["summary"]["continued_table_segment_count"], 2)
        self.assertEqual(report["summary"]["continued_table_reconstructable_group_count"], 1)
        self.assertEqual(report["summary"]["continued_table_merged_row_count"], 5)
        self.assertEqual(report["summary"]["table_chain_candidate_count"], 1)
        self.assertEqual(report["summary"]["table_chain_merged_count"], 1)
        self.assertEqual(report["summary"]["table_chain_reject_count"], 0)
        self.assertEqual(report["summary"]["table_chain_row_gain"], 2)
        self.assertEqual(report["summary"]["table_chain_warning_count"], 0)
        group = report["continued_table_groups"][0]
        self.assertEqual(group["table_ids"], ["p1-b0000", "p2-b0000"])
        self.assertEqual(group["pages_1based"], [1, 2])
        self.assertEqual(group["merge_status"], "merged")
        self.assertEqual(group["chain_confidence"], "high")
        self.assertEqual(group["merged_row_gain"], 2)
        self.assertEqual(group["merged_row_count"], 5)
        self.assertEqual(group["merged_column_count"], 2)
        self.assertEqual(group["skipped_repeated_header_count"], 1)
        self.assertEqual(group["header"], ["Model", "Acc"])
        self.assertEqual(group["rows"][0], ["Model", "Acc"])
        self.assertEqual(group["rows"][-1], ["D", "94"])
        self.assertIn("continued_table_group", group["warnings"])

        chunk = TextChunk("c0000", [0, 1], page1_table.text + "\n" + page2_table.text, 0, 0)
        hints = build_table_translation_hints(chunk, report)
        self.assertIn("续表合并组", hints)
        self.assertIn("p1-b0000 -> p2-b0000", hints)
        self.assertIn("合并后 5 行 x 2 列", hints)

    def test_table_reconstruction_rejects_incompatible_continued_table_groups(self) -> None:
        page1_table = BlockIR(
            "p1-b0000",
            1,
            "table",
            "Model Acc\nA 91\nB 92",
            (40, 640, 540, 760),
            0,
            meta={
                "table": {
                    "rows": [["Model", "Acc"], ["A", "91"], ["B", "92"]],
                    "row_count": 3,
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
            "Dataset F1\nCOCO 88.1\nGLUE 89.0",
            (40, 80, 540, 180),
            0,
            meta={
                "table": {
                    "rows": [["Dataset", "F1"], ["COCO", "88.1"], ["GLUE", "89.0"]],
                    "row_count": 3,
                    "column_count": 2,
                    "header": ["Dataset", "F1"],
                    "confidence": "medium",
                }
            },
        )
        doc_ir = DocumentIR(
            doc_id="rejected-table-group",
            source_pdf="sample.pdf",
            pages=[
                PageIR(1, 600, 800, page1_table.text, [page1_table]),
                PageIR(2, 600, 800, page2_table.text, [page2_table]),
            ],
        )

        report = build_table_reconstruction_report(doc_ir, build_structure_qa(doc_ir))

        self.assertEqual(report["summary"]["table_chain_candidate_count"], 1)
        self.assertEqual(report["summary"]["table_chain_merged_count"], 0)
        self.assertEqual(report["summary"]["table_chain_reject_count"], 1)
        self.assertEqual(report["summary"]["table_chain_row_gain"], 0)
        self.assertEqual(report["summary"]["table_chain_reject_reason_count"], 1)
        self.assertEqual(report["summary"]["table_chain_warning_reason_count"], 0)
        self.assertEqual(
            report["summary"]["table_chain_reject_reason_counts"]["header_mismatch_segment_1"],
            1,
        )
        self.assertEqual(
            report["summary"]["table_chain_reject_reason_category_counts"]["header_mismatch"],
            1,
        )
        self.assertEqual(report["summary"]["continued_table_reconstructable_group_count"], 0)
        group = report["continued_table_groups"][0]
        self.assertEqual(group["merge_status"], "rejected")
        self.assertEqual(group["chain_confidence"], "low")
        self.assertFalse(group["reconstructable"])
        self.assertEqual(group["rows"], [])
        self.assertIn("header_mismatch_segment_1", group["compatibility"]["reject_reasons"])

        chunk = TextChunk("c0000", [0, 1], page1_table.text + "\n" + page2_table.text, 0, 0)
        hints = build_table_translation_hints(chunk, report)
        self.assertIn("续表候选", hints)
        self.assertIn("当前未安全合并", hints)
        self.assertIn("header_mismatch_segment_1", hints)

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

    def test_translation_qa_reports_caption_relation_mismatch(self) -> None:
        root = Path.cwd() / "test-output" / "caption-relation-qa"
        if root.exists():
            shutil.rmtree(root)
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(parents=True)
        try:
            caption = BlockIR(
                "p1-b0000",
                1,
                "caption",
                "Table 1: Results.",
                (40, 80, 520, 105),
                0,
            )
            table = BlockIR(
                "p1-b0001",
                1,
                "table",
                "Model Acc\nBERT 91.2%",
                (40, 110, 520, 180),
                1,
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
            assign_block_parents([caption, table])
            doc_ir = DocumentIR(
                doc_id="caption-relation-qa",
                source_pdf="sample.pdf",
                pages=[PageIR(1, 600, 800, caption.text + "\n" + table.text, [caption, table])],
            )
            chunk = TextChunk(
                "c0000",
                [0],
                "Table 1: Results.\nModel Acc\nBERT 91.2%",
                0,
                0,
            )
            chunk.block_ids = ["p1-b0000", "p1-b0001"]
            (chunk_dir / "c0000.md").write_text(
                "---\n{}\n---\n\n结果如下。\n\n| 模型 | 准确率 |\n| --- | --- |\n| BERT | 91.2% |\n",
                encoding="utf-8",
            )

            report = build_translation_qa([chunk], chunk_dir, document_ir=doc_ir)

            issue = next(
                issue
                for issue in report["chunks"][0]["issues"]
                if issue["type"] == "caption_or_footnote_relation_mismatch"
            )
            relation = issue["relations"][0]
            self.assertEqual(report["summary"]["structure_relation_check_count"], 1)
            self.assertEqual(report["summary"]["structure_relation_mismatch_count"], 1)
            self.assertEqual(report["summary"]["structure_relation_missing_anchor_count"], 1)
            self.assertEqual(relation["relation"], "caption_for_table")
            self.assertEqual(relation["parent_block_id"], "p1-b0001")
            self.assertEqual(relation["missing_anchors"][0]["token"], "Table 1")

            plan = build_repair_plan(report)
            repair = next(
                item for item in plan["items"] if item["issue_type"] == "caption_or_footnote_relation_mismatch"
            )
            self.assertEqual(repair["action"], "rewrite_with_structure_relations")
            self.assertIn("relations", repair["evidence"])
            requests = build_repair_requests(plan, [chunk], chunk_dir)
            request = next(
                item
                for item in requests["requests"]
                if item["issue_type"] == "caption_or_footnote_relation_mismatch"
            )
            self.assertIn("Table 1", request["locked_tokens"])
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_translation_qa_reports_table_footnote_binding_mismatch(self) -> None:
        root = Path.cwd() / "test-output" / "table-footnote-binding-qa"
        if root.exists():
            shutil.rmtree(root)
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(parents=True)
        try:
            table = BlockIR(
                "p1-b0000",
                1,
                "table",
                "Model Acc\nBERT 91.2% *",
                (40, 110, 520, 180),
                0,
                meta={
                    "table": {
                        "rows": [["Model", "Acc"], ["BERT", "91.2% *"]],
                        "row_count": 2,
                        "column_count": 2,
                        "header": ["Model", "Acc"],
                        "confidence": "medium",
                    }
                },
            )
            footnote = BlockIR(
                "p1-b0001",
                1,
                "footnote",
                "* p < 0.05.",
                (40, 190, 520, 210),
                1,
            )
            assign_block_parents([table, footnote])
            doc_ir = DocumentIR(
                doc_id="table-footnote-binding-qa",
                source_pdf="sample.pdf",
                pages=[PageIR(1, 600, 800, table.text + "\n" + footnote.text, [table, footnote])],
            )
            table_reconstruction = build_table_reconstruction_report(doc_ir, build_structure_qa(doc_ir))
            chunk = TextChunk(
                "c0000",
                [0],
                "Model Acc\nBERT 91.2% *\n* p < 0.05.",
                0,
                0,
            )
            chunk.block_ids = ["p1-b0000", "p1-b0001"]
            (chunk_dir / "c0000.md").write_text(
                (
                    "---\n{}\n---\n\n"
                    "| 模型 | 准确率 |\n"
                    "| --- | --- |\n"
                    "| BERT | 91.2% |\n\n"
                    "注：* p < 0.05。\n"
                ),
                encoding="utf-8",
            )

            report = build_translation_qa(
                [chunk],
                chunk_dir,
                document_ir=doc_ir,
                table_reconstruction=table_reconstruction,
            )

            issue = next(
                issue
                for issue in report["chunks"][0]["issues"]
                if issue["type"] == "table_footnote_binding_mismatch"
            )
            binding = issue["bindings"][0]
            self.assertEqual(report["summary"]["table_footnote_binding_check_count"], 1)
            self.assertEqual(report["summary"]["table_footnote_binding_mismatch_count"], 1)
            self.assertEqual(report["summary"]["table_footnote_binding_missing_cell_count"], 1)
            self.assertEqual(binding["table_id"], "p1-b0000")
            self.assertEqual(binding["footnote_block_id"], "p1-b0001")
            self.assertEqual(binding["missing_cells"][0]["row_index"], 1)
            self.assertEqual(binding["missing_cells"][0]["column_index"], 1)
            self.assertIn("*", binding["missing_cells"][0]["missing_markers"])

            plan = build_repair_plan(report)
            repair = next(item for item in plan["items"] if item["issue_type"] == "table_footnote_binding_mismatch")
            self.assertEqual(repair["action"], "repair_table_footnote_binding")
            self.assertEqual(repair["priority"], "P0")
            requests = build_repair_requests(plan, [chunk], chunk_dir)
            request = next(
                item for item in requests["requests"] if item["issue_type"] == "table_footnote_binding_mismatch"
            )
            self.assertIn("*", request["locked_tokens"])
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_translation_qa_accepts_preserved_structure_relations(self) -> None:
        root = Path.cwd() / "test-output" / "structure-relation-qa-clean"
        if root.exists():
            shutil.rmtree(root)
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(parents=True)
        try:
            caption = BlockIR("p1-b0000", 1, "caption", "Table 1: Results.", (40, 80, 520, 105), 0)
            table = BlockIR(
                "p1-b0001",
                1,
                "table",
                "Model Acc\nBERT 91.2% *",
                (40, 110, 520, 180),
                1,
                meta={
                    "table": {
                        "rows": [["Model", "Acc"], ["BERT", "91.2% *"]],
                        "row_count": 2,
                        "column_count": 2,
                        "header": ["Model", "Acc"],
                        "confidence": "medium",
                    }
                },
            )
            footnote = BlockIR("p1-b0002", 1, "footnote", "* p < 0.05.", (40, 190, 520, 210), 2)
            assign_block_parents([caption, table, footnote])
            doc_ir = DocumentIR(
                doc_id="structure-relation-qa-clean",
                source_pdf="sample.pdf",
                pages=[
                    PageIR(
                        1,
                        600,
                        800,
                        caption.text + "\n" + table.text + "\n" + footnote.text,
                        [caption, table, footnote],
                    )
                ],
            )
            table_reconstruction = build_table_reconstruction_report(doc_ir, build_structure_qa(doc_ir))
            chunk = TextChunk(
                "c0000",
                [0],
                "Table 1: Results.\nModel Acc\nBERT 91.2% *\n* p < 0.05.",
                0,
                0,
            )
            chunk.block_ids = ["p1-b0000", "p1-b0001", "p1-b0002"]
            (chunk_dir / "c0000.md").write_text(
                (
                    "---\n{}\n---\n\n"
                    "表 1：结果。\n\n"
                    "| 模型 | 准确率 |\n"
                    "| --- | --- |\n"
                    "| BERT | 91.2% * |\n\n"
                    "注：* p < 0.05。\n"
                ),
                encoding="utf-8",
            )

            report = build_translation_qa(
                [chunk],
                chunk_dir,
                document_ir=doc_ir,
                table_reconstruction=table_reconstruction,
            )
            issue_types = {issue["type"] for issue in report["chunks"][0]["issues"]}
            self.assertEqual(report["summary"]["structure_relation_check_count"], 2)
            self.assertEqual(report["summary"]["structure_relation_mismatch_count"], 0)
            self.assertEqual(report["summary"]["table_footnote_binding_check_count"], 1)
            self.assertEqual(report["summary"]["table_footnote_binding_mismatch_count"], 0)
            self.assertNotIn("caption_or_footnote_relation_mismatch", issue_types)
            self.assertNotIn("table_footnote_binding_mismatch", issue_types)
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

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
            self.assertEqual(cell["source_table_shape"], {"row_count": 2, "column_count": 3})
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
            self.assertEqual(request["merge_target"]["table_index"], 0)
            self.assertEqual(request["merge_target"]["table_id"], "p1-b0000")
            self.assertEqual(request["merge_target"]["cell_count"], 1)
            self.assertEqual(request["merge_target"]["cells"][0]["row_index"], 1)
            self.assertEqual(request["merge_target"]["cells"][0]["column_index"], 1)
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
            self.assertEqual(validation["summary"]["table_shape_check_count"], 1)
            self.assertEqual(validation["summary"]["table_shape_passed_count"], 1)
            self.assertEqual(validation["validations"][0]["status"], "passed")
            self.assertEqual(validation["validations"][0]["expected_table_shape_count"], 1)
            self.assertEqual(validation["validations"][0]["merge_target"]["table_index"], 0)
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
            self.assertEqual(merge["summary"]["table_targeted_patch_count"], 1)
            self.assertEqual(merge["patches"][0]["strategy"], "replace_markdown_table_by_evidence")
            self.assertEqual(merge["patches"][0]["merge_target"]["table_index"], 0)
            diff_preview = merge["patches"][0]["diff_preview"]
            self.assertEqual(diff_preview["preview_kind"], "table")
            self.assertIn("| BERT | 91.2 | p<0.05 |", diff_preview["current_excerpt"])
            self.assertIn("| BERT | 91.2% | p<0.05 |", diff_preview["candidate_excerpt"])
            self.assertIn("| BERT | 91.2% | p<0.05 |", diff_preview["repaired_excerpt"])
            self.assertIn("--- c0000:current", diff_preview["unified_diff"])
            self.assertIn("-| BERT | 91.2 | p<0.05 |", diff_preview["unified_diff"])
            self.assertIn("+| BERT | 91.2% | p<0.05 |", diff_preview["unified_diff"])
            patch_review = build_repair_patch_review(merge)
            self.assertEqual(patch_review["schema_version"], "repair-patch-review-v1")
            self.assertEqual(patch_review["summary"]["patch_count"], 1)
            self.assertEqual(patch_review["summary"]["auto_merge_safe_count"], 1)
            self.assertEqual(patch_review["summary"]["table_patch_review_count"], 1)
            self.assertEqual(
                patch_review["summary"]["default_decision_counts"]["approve_candidate"],
                1,
            )
            self.assertEqual(patch_review["patch_reviews"][0]["review_id"], "pr0000")
            self.assertEqual(patch_review["patch_reviews"][0]["human_decision"], "")
            self.assertEqual(patch_review["patch_reviews"][0]["default_decision"], "approve_candidate")
            self.assertEqual(patch_review["patch_reviews"][0]["merge_target"]["table_index"], 0)
            self.assertEqual(
                patch_review["patch_reviews"][0]["diff_preview"]["unified_diff"],
                diff_preview["unified_diff"],
            )
            patch_review_markdown = repair_patch_review_to_markdown(patch_review)
            self.assertIn("#### 差异预览", patch_review_markdown)
            self.assertIn("当前译文片段", patch_review_markdown)
            self.assertIn("```diff", patch_review_markdown)
            rejected_review = apply_repair_patch_review_decision(
                json.loads(json.dumps(patch_review)),
                "pr0000",
                decision="reject",
                reviewer="tester",
                comment="cell value still looks wrong",
                reviewed_at="2026-07-05T00:00:00+00:00",
            )
            self.assertEqual(rejected_review["summary"]["human_reviewed_count"], 1)
            self.assertEqual(rejected_review["summary"]["human_rejected_count"], 1)
            self.assertEqual(rejected_review["summary"]["publish_blocking_count"], 1)
            self.assertEqual(
                rejected_review["summary"]["effective_decision_counts"]["reject_candidate"],
                1,
            )
            repaired_chunk = (root / "repaired_chunks" / "c0000.md").read_text(encoding="utf-8")
            self.assertIn("| BERT | 91.2% | p<0.05 |", repaired_chunk)
            self.assertTrue((root / "repaired_full.md").is_file())
            blocked_publish = build_repair_publish(
                merge,
                confirm=True,
                source_full_path=root / "repaired_full.md",
                published_full_path=root / "blocked_published_full.md",
                original_full_path=chunk_dir / "c0000.md",
                repair_patch_review=rejected_review,
            )
            self.assertTrue(blocked_publish["summary"]["confirmed"])
            self.assertFalse(blocked_publish["summary"]["published"])
            self.assertEqual(blocked_publish["summary"]["publish_status"], "blocked_patch_review")
            self.assertEqual(blocked_publish["summary"]["patch_review_blocking_count"], 1)
            self.assertFalse((root / "blocked_published_full.md").exists())
            draft_publish = build_repair_publish(
                merge,
                source_full_path=root / "repaired_full.md",
                published_full_path=root / "published_full.md",
                original_full_path=chunk_dir / "c0000.md",
            )
            self.assertEqual(draft_publish["schema_version"], "repair-publish-v1")
            self.assertFalse(draft_publish["summary"]["confirmed"])
            self.assertFalse(draft_publish["summary"]["published"])
            self.assertEqual(draft_publish["summary"]["publish_status"], "pending_confirmation")
            self.assertFalse((root / "published_full.md").exists())
            confirmed_publish = build_repair_publish(
                merge,
                confirm=True,
                source_full_path=root / "repaired_full.md",
                published_full_path=root / "published_full.md",
                original_full_path=chunk_dir / "c0000.md",
            )
            self.assertTrue(confirmed_publish["summary"]["confirmed"])
            self.assertTrue(confirmed_publish["summary"]["published"])
            self.assertEqual(confirmed_publish["summary"]["publish_status"], "published")
            self.assertTrue(confirmed_publish["summary"]["rollback_available"])
            published_text = (root / "published_full.md").read_text(encoding="utf-8")
            self.assertIn("| BERT | 91.2% | p<0.05 |", published_text)
            draft_rollback = build_repair_rollback(
                confirmed_publish,
                original_full_path=chunk_dir / "c0000.md",
                published_full_path=root / "published_full.md",
                rollback_full_path=root / "rollback_full.md",
            )
            self.assertEqual(draft_rollback["schema_version"], "repair-rollback-v1")
            self.assertTrue(draft_rollback["summary"]["rollback_available"])
            self.assertFalse(draft_rollback["summary"]["rollback_applied"])
            self.assertEqual(draft_rollback["summary"]["rollback_status"], "pending_confirmation")
            self.assertFalse((root / "rollback_full.md").exists())
            confirmed_rollback = build_repair_rollback(
                confirmed_publish,
                confirm=True,
                original_full_path=chunk_dir / "c0000.md",
                published_full_path=root / "published_full.md",
                rollback_full_path=root / "rollback_full.md",
            )
            self.assertTrue(confirmed_rollback["summary"]["confirmed"])
            self.assertTrue(confirmed_rollback["summary"]["rollback_applied"])
            self.assertEqual(confirmed_rollback["summary"]["rollback_status"], "rolled_back")
            self.assertTrue(confirmed_rollback["summary"]["rollback_matches_original"])
            self.assertEqual(
                (root / "rollback_full.md").read_text(encoding="utf-8"),
                (chunk_dir / "c0000.md").read_text(encoding="utf-8"),
            )
            self.assertEqual((root / "published_full.md").read_text(encoding="utf-8"), published_text)
            blocked_rollback = build_repair_rollback(
                draft_publish,
                confirm=True,
                original_full_path=chunk_dir / "c0000.md",
                published_full_path=root / "not_published_full.md",
                rollback_full_path=root / "blocked_rollback_full.md",
            )
            self.assertEqual(blocked_rollback["summary"]["rollback_status"], "blocked_unpublished")
            self.assertFalse((root / "blocked_rollback_full.md").exists())
            draft_formal_replace = build_repair_formal_replace(
                confirmed_publish,
                original_full_path=chunk_dir / "c0000.md",
                published_full_path=root / "published_full.md",
                formal_full_path=root / "formal_full.md",
                backup_full_path=root / "formal_full.before_repair.md",
            )
            self.assertEqual(draft_formal_replace["schema_version"], "repair-formal-replace-v1")
            self.assertTrue(draft_formal_replace["summary"]["replace_available"])
            self.assertFalse(draft_formal_replace["summary"]["replaced"])
            self.assertEqual(draft_formal_replace["summary"]["replace_status"], "pending_confirmation")
            self.assertFalse((root / "formal_full.md").exists())
            confirmed_formal_replace = build_repair_formal_replace(
                confirmed_publish,
                confirm=True,
                original_full_path=chunk_dir / "c0000.md",
                published_full_path=root / "published_full.md",
                formal_full_path=root / "formal_full.md",
                backup_full_path=root / "formal_full.before_repair.md",
            )
            self.assertTrue(confirmed_formal_replace["summary"]["confirmed"])
            self.assertTrue(confirmed_formal_replace["summary"]["replaced"])
            self.assertTrue(confirmed_formal_replace["summary"]["formal_initialized_from_original"])
            self.assertEqual(confirmed_formal_replace["summary"]["replace_status"], "replaced")
            self.assertTrue(confirmed_formal_replace["summary"]["formal_matches_published"])
            self.assertTrue(confirmed_formal_replace["summary"]["backup_matches_formal_before"])
            self.assertEqual((root / "formal_full.md").read_text(encoding="utf-8"), published_text)
            self.assertEqual(
                (root / "formal_full.before_repair.md").read_text(encoding="utf-8"),
                (chunk_dir / "c0000.md").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (chunk_dir / "c0000.md").read_text(encoding="utf-8"),
                (root / "formal_full.before_repair.md").read_text(encoding="utf-8"),
            )
            repeated_formal_replace = build_repair_formal_replace(
                confirmed_publish,
                confirm=True,
                original_full_path=chunk_dir / "c0000.md",
                published_full_path=root / "published_full.md",
                formal_full_path=root / "formal_full.md",
                backup_full_path=root / "formal_full.before_repair.md",
            )
            self.assertTrue(repeated_formal_replace["summary"]["replaced"])
            self.assertTrue(repeated_formal_replace["summary"]["already_applied"])
            self.assertEqual(repeated_formal_replace["summary"]["replace_status"], "already_applied")
            blocked_formal_replace = build_repair_formal_replace(
                draft_publish,
                confirm=True,
                original_full_path=chunk_dir / "c0000.md",
                published_full_path=root / "missing_published_full.md",
                formal_full_path=root / "blocked_formal_full.md",
                backup_full_path=root / "blocked_formal_full.before_repair.md",
            )
            self.assertEqual(blocked_formal_replace["summary"]["replace_status"], "blocked_unpublished")
            self.assertFalse((root / "blocked_formal_full.md").exists())
            draft_formal_rollback = build_repair_formal_rollback(
                confirmed_formal_replace,
                formal_full_path=root / "formal_full.md",
                backup_full_path=root / "formal_full.before_repair.md",
                active_before_rollback_path=root / "formal_full.repair_applied.md",
            )
            self.assertEqual(draft_formal_rollback["schema_version"], "repair-formal-rollback-v1")
            self.assertTrue(draft_formal_rollback["summary"]["rollback_available"])
            self.assertFalse(draft_formal_rollback["summary"]["rollback_applied"])
            self.assertEqual(draft_formal_rollback["summary"]["rollback_status"], "pending_confirmation")
            confirmed_formal_rollback = build_repair_formal_rollback(
                confirmed_formal_replace,
                confirm=True,
                formal_full_path=root / "formal_full.md",
                backup_full_path=root / "formal_full.before_repair.md",
                active_before_rollback_path=root / "formal_full.repair_applied.md",
            )
            self.assertTrue(confirmed_formal_rollback["summary"]["confirmed"])
            self.assertTrue(confirmed_formal_rollback["summary"]["rollback_applied"])
            self.assertEqual(confirmed_formal_rollback["summary"]["rollback_status"], "rolled_back")
            self.assertTrue(confirmed_formal_rollback["summary"]["formal_matches_backup"])
            self.assertEqual(
                (root / "formal_full.md").read_text(encoding="utf-8"),
                (chunk_dir / "c0000.md").read_text(encoding="utf-8"),
            )
            self.assertEqual((root / "formal_full.repair_applied.md").read_text(encoding="utf-8"), published_text)
            self.assertEqual((root / "published_full.md").read_text(encoding="utf-8"), published_text)
            blocked_formal_rollback = build_repair_formal_rollback(
                confirmed_formal_replace,
                confirm=True,
                formal_full_path=root / "formal_full.md",
                backup_full_path=root / "missing_formal_backup.md",
                active_before_rollback_path=root / "blocked_formal_full.repair_applied.md",
            )
            self.assertEqual(blocked_formal_rollback["summary"]["rollback_status"], "blocked_missing_backup")
            self.assertFalse((root / "blocked_formal_full.repair_applied.md").exists())
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_repair_patch_review_batch_decision_updates_atomically(self) -> None:
        review = build_repair_patch_review(
            {
                "schema_version": "repair-merge-v1",
                "summary": {
                    "applied_count": 1,
                    "manual_merge_required_count": 1,
                    "conflict_count": 0,
                },
                "patches": [
                    {
                        "request_id": "rq0000",
                        "repair_id": "rp0000",
                        "chunk_id": "c0000",
                        "pages_1based": [1],
                        "priority": "medium",
                        "issue_type": "table_cell_token_mismatch",
                        "action": "replace_table",
                        "scope": "table",
                        "status": "applied",
                        "strategy": "replace_markdown_table_by_evidence",
                        "merge_target": {"table_index": 0, "cell_count": 4},
                        "reason": "table cell changed during translation",
                        "patched_chunk_path": "repaired_chunks/c0000.md",
                        "result_path": "repairs/rq0000.md",
                        "result_excerpt": "| 模型 | 准确率 |\n| --- | --- |\n| BERT | 91.2% |",
                    },
                    {
                        "request_id": "rq0001",
                        "repair_id": "rp0001",
                        "chunk_id": "c0001",
                        "pages_1based": [2, 3],
                        "priority": "high",
                        "issue_type": "caption_relation_mismatch",
                        "action": "rewrite_caption",
                        "scope": "paragraph",
                        "status": "skipped_manual_merge_required",
                        "strategy": "manual",
                        "reason": "caption target could not be located safely",
                        "patched_chunk_path": "",
                        "result_path": "",
                        "result_excerpt": "Figure 2 needs manual repair.",
                    },
                ],
            }
        )

        self.assertEqual(review["summary"]["patch_count"], 2)
        self.assertEqual(review["summary"]["publish_blocking_count"], 1)

        with self.assertRaises(KeyError):
            apply_repair_patch_review_batch_decision(
                review,
                ["pr0000", "missing-review-id"],
                decision="reject",
                reviewer="mentor",
                reviewed_at="2026-07-06T00:10:00+00:00",
            )
        self.assertEqual(review["patch_reviews"][0]["human_decision"], "")

        updated = apply_repair_patch_review_batch_decision(
            review,
            ["pr0000", "pr0001"],
            decision="approve",
            reviewer="mentor",
            comment="batch patch pass",
            reviewed_at="2026-07-06T00:11:00+00:00",
        )

        self.assertEqual(updated["summary"]["human_approved_count"], 2)
        self.assertEqual(updated["summary"]["human_reviewed_count"], 2)
        self.assertEqual(updated["summary"]["publish_blocking_count"], 0)
        self.assertEqual(updated["summary"]["effective_safe_count"], 2)
        self.assertEqual(updated["patch_reviews"][1]["human_comment"], "batch patch pass")
        self.assertEqual(updated["patch_reviews"][1]["reviewed_by"], "mentor")
        self.assertEqual(updated["patch_reviews"][1]["reviewed_at"], "2026-07-06T00:11:00+00:00")
        self.assertIn("batch patch pass", repair_patch_review_to_markdown(updated))

        root = Path.cwd() / "test-output" / "repair-patch-review-batch-decision"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            json_path = root / "repair_patch_review.json"
            md_path = root / "repair_patch_review.md"
            json_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
            md_path.write_text(repair_patch_review_to_markdown(updated), encoding="utf-8")
            persisted = write_repair_patch_review_batch_decision(
                json_path,
                md_path,
                ["pr0000", "pr0001"],
                decision="clear",
                reviewer="mentor",
                reviewed_at="2026-07-06T00:12:00+00:00",
            )
            self.assertEqual(persisted["summary"]["human_approved_count"], 0)
            self.assertEqual(persisted["summary"]["human_reviewed_count"], 0)
            self.assertEqual(persisted["summary"]["effective_safe_count"], 1)
            self.assertEqual(persisted["summary"]["publish_blocking_count"], 1)
            stored = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["patch_reviews"][0]["human_decision"], "")
            self.assertIn("pending", md_path.read_text(encoding="utf-8"))
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_repair_effectiveness_compares_before_after_qa(self) -> None:
        before = {
            "schema_version": "translation-qa-v1",
            "summary": {
                "issue_count": 3,
                "issue_counts": {
                    "missing_numbers": 1,
                    "table_cell_token_mismatch": 2,
                },
                "severity_counts": {"high": 2, "medium": 1},
                "table_cell_token_error_count": 2,
                "missing_table_locked_token_count": 2,
            },
            "chunks": [
                {
                    "chunk_id": "c0000",
                    "pages_1based": [1],
                    "issues": [
                        {"type": "missing_numbers"},
                        {"type": "table_cell_token_mismatch"},
                    ],
                },
                {
                    "chunk_id": "c0001",
                    "pages_1based": [2],
                    "issues": [{"type": "table_cell_token_mismatch"}],
                },
            ],
        }
        after = {
            "schema_version": "translation-qa-v1",
            "summary": {
                "issue_count": 2,
                "issue_counts": {
                    "table_cell_token_mismatch": 1,
                    "high_english_residual": 1,
                },
                "severity_counts": {"medium": 1, "low": 1},
                "table_cell_token_error_count": 1,
                "missing_table_locked_token_count": 1,
            },
            "chunks": [
                {
                    "chunk_id": "c0000",
                    "pages_1based": [1],
                    "issues": [{"type": "table_cell_token_mismatch"}],
                },
                {"chunk_id": "c0001", "pages_1based": [2], "issues": []},
                {
                    "chunk_id": "c0002",
                    "pages_1based": [3],
                    "issues": [{"type": "high_english_residual"}],
                },
            ],
        }

        report = build_repair_effectiveness(
            before,
            after,
            repair_merge={
                "schema_version": "repair-merge-v1",
                "summary": {
                    "applied_count": 1,
                    "manual_merge_required_count": 0,
                },
            },
        )
        summary = report["summary"]
        self.assertEqual(report["schema_version"], "repair-effectiveness-v1")
        self.assertEqual(summary["before_issue_count"], 3)
        self.assertEqual(summary["after_issue_count"], 2)
        self.assertEqual(summary["issue_delta"], 1)
        self.assertEqual(summary["issue_reduction_rate"], 0.3333)
        self.assertEqual(summary["resolved_issue_count"], 2)
        self.assertEqual(summary["persisted_issue_count"], 1)
        self.assertEqual(summary["new_issue_count"], 1)
        self.assertEqual(summary["improved_chunk_count"], 2)
        self.assertEqual(summary["regressed_chunk_count"], 1)
        self.assertEqual(summary["status"], "improved_with_regressions")
        self.assertEqual(report["issue_type_comparisons"]["missing_numbers"]["status"], "resolved")
        self.assertEqual(
            report["issue_type_comparisons"]["table_cell_token_mismatch"]["status"],
            "improved",
        )
        self.assertEqual(
            report["issue_type_comparisons"]["high_english_residual"]["status"],
            "regressed",
        )
        chunk_statuses = {
            item["chunk_id"]: item["status"]
            for item in report["chunk_comparisons"]
        }
        self.assertEqual(chunk_statuses["c0000"], "improved")
        self.assertEqual(chunk_statuses["c0001"], "resolved")
        self.assertEqual(chunk_statuses["c0002"], "regressed")

        markdown = repair_effectiveness_to_markdown(report)
        self.assertIn("局部修复效果对比", markdown)
        self.assertIn("missing_numbers", markdown)
        self.assertIn("c0000", markdown)

    def test_repair_merge_targets_markdown_table_from_cell_evidence(self) -> None:
        root = Path.cwd() / "test-output" / "targeted-table-repair-merge"
        if root.exists():
            shutil.rmtree(root)
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(parents=True)
        try:
            chunks = [
                TextChunk(
                    "c0000",
                    [0],
                    "First table\nA 10\n\nSecond table\nB 99%",
                    0,
                    0,
                )
            ]
            (chunk_dir / "c0000.md").write_text(
                (
                    "---\n{}\n---\n\n"
                    "| 名称 | 值 |\n"
                    "| --- | --- |\n"
                    "| A | 10 |\n\n"
                    "| 名称 | 值 |\n"
                    "| --- | --- |\n"
                    "| B | 99 |\n"
                ),
                encoding="utf-8",
            )
            repair_requests = {
                "schema_version": "repair-requests-v1",
                "summary": {"repair_request_count": 1},
                "requests": [
                    {
                        "request_id": "rq0000",
                        "repair_id": "r0000",
                        "chunk_id": "c0000",
                        "pages_1based": [1, 1],
                        "priority": "P0",
                        "issue_type": "table_cell_token_mismatch",
                        "action": "repair_table_cell_tokens",
                        "scope": "table_cell",
                        "executor": "translation_backend",
                        "locked_tokens": ["99%"],
                        "merge_target": {
                            "table_index": 1,
                            "table_id": "p1-b0001",
                            "cell_count": 1,
                            "cells": [{"row_index": 1, "column_index": 1, "missing_tokens": ["99%"]}],
                        },
                        "evidence": {
                            "cells": [
                                {
                                    "table_index": 1,
                                    "table_id": "p1-b0001",
                                    "row_index": 1,
                                    "column_index": 1,
                                    "missing_tokens": ["99%"],
                                    "source_table_shape": {"row_count": 2, "column_count": 2},
                                }
                            ]
                        },
                    }
                ],
            }
            repair_results = {
                "schema_version": "repair-results-v1",
                "summary": {"repair_request_count": 1, "succeeded_count": 1},
                "results": [
                    {
                        "request_id": "rq0000",
                        "chunk_id": "c0000",
                        "status": "succeeded",
                        "action": "repair_table_cell_tokens",
                        "scope": "table_cell",
                        "result_excerpt": "| 名称 | 值 |\n| --- | --- |\n| B | 99% |",
                    }
                ],
            }

            validation = build_repair_validation(repair_requests, repair_results)
            self.assertEqual(validation["summary"]["passed_count"], 1)
            self.assertEqual(validation["summary"]["table_shape_check_count"], 1)
            self.assertEqual(validation["validations"][0]["merge_target"]["table_index"], 1)
            merge = build_repair_merge(
                repair_requests,
                repair_results,
                validation,
                chunks,
                chunk_dir,
                repaired_chunk_dir=root / "repaired_chunks",
                repaired_full_path=root / "repaired_full.md",
            )

            self.assertEqual(merge["summary"]["applied_count"], 1)
            self.assertEqual(merge["summary"]["table_targeted_patch_count"], 1)
            self.assertEqual(merge["patches"][0]["strategy"], "replace_markdown_table_by_evidence")
            repaired_chunk = (root / "repaired_chunks" / "c0000.md").read_text(encoding="utf-8")
            self.assertIn("| A | 10 |", repaired_chunk)
            self.assertIn("| B | 99% |", repaired_chunk)
            self.assertNotIn("| B | 99 |", repaired_chunk)
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_repair_requests_promote_confirmed_structure_patch_into_merge_target(self) -> None:
        root = Path.cwd() / "test-output" / "repair-structure-patch-context"
        if root.exists():
            shutil.rmtree(root)
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(parents=True)
        try:
            chunks = [
                TextChunk(
                    "c0000",
                    [0],
                    "Model Result p\nBERT 91.2% p<0.05",
                    0,
                    0,
                )
            ]
            chunks[0].block_ids = ["p1-b0000"]
            (chunk_dir / "c0000.md").write_text(
                (
                    "| 模型 | 结果 |\n"
                    "| --- | --- |\n"
                    "| BERT | 91.2% |\n"
                ),
                encoding="utf-8",
            )
            structure_patch = {
                "patch_id": "tsp-0001-p1-b0000-r1c1",
                "source_review_id": "tmc-0001-p1-b0000-r1c1",
                "patch_type": "merged_cell_span",
                "operation": "apply_confirmed_merged_cell_span",
                "applied": True,
                "anchor_cell": {"row_index": 1, "column_index": 1},
                "span": {
                    "span_type": "colspan",
                    "row_span": 1,
                    "column_span": 2,
                },
                "covered_cells": [{"row_index": 1, "column_index": 2}],
            }
            table_reconstruction = {
                "schema_version": "table-reconstruction-v1",
                "confirmation_schema_version": "table-structure-publish-v1",
                "table_structure_source": "confirmed",
                "tables": [
                    {
                        "table_id": "p1-b0000",
                        "block_id": "p1-b0000",
                        "page_no": 1,
                        "row_count": 2,
                        "column_count": 3,
                        "cells": [
                            {
                                "row_index": 0,
                                "column_index": 0,
                                "text": "Model",
                                "locked_tokens": [],
                            },
                            {
                                "row_index": 0,
                                "column_index": 1,
                                "text": "Result",
                                "locked_tokens": [],
                            },
                            {
                                "row_index": 0,
                                "column_index": 2,
                                "text": "p",
                                "locked_tokens": [],
                            },
                            {
                                "row_index": 1,
                                "column_index": 0,
                                "text": "BERT",
                                "locked_tokens": ["BERT"],
                            },
                            {
                                "row_index": 1,
                                "column_index": 1,
                                "text": "91.2%",
                                "locked_tokens": ["91.2%"],
                            },
                            {
                                "row_index": 1,
                                "column_index": 2,
                                "text": "p<0.05",
                                "locked_tokens": ["p<0.05"],
                            },
                        ],
                        "structure_patches": [structure_patch],
                    }
                ],
            }

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
            cell = next(cell for cell in issue["cells"] if cell["column_index"] == 2)
            self.assertEqual(cell["matched_structure_patch_count"], 1)
            self.assertEqual(cell["matched_structure_patches"][0]["source_review_id"], "tmc-0001-p1-b0000-r1c1")
            self.assertEqual(cell["matched_structure_patches"][0]["cell_role"], "covered")

            plan = build_repair_plan(report)
            requests = build_repair_requests(plan, chunks, chunk_dir)
            self.assertEqual(requests["summary"]["structure_patch_context_request_count"], 1)
            request = next(
                request
                for request in requests["requests"]
                if request["action"] == "repair_table_cell_tokens"
            )
            merge_target = request["merge_target"]
            self.assertEqual(merge_target["structure_patch_evidence_count"], 1)
            self.assertEqual(merge_target["structure_patch_source_review_ids"], ["tmc-0001-p1-b0000-r1c1"])
            self.assertEqual(merge_target["cells"][0]["structure_patch_role"], "covered")
            self.assertEqual(merge_target["cells"][0]["render_row_index"], 1)
            self.assertEqual(merge_target["cells"][0]["render_column_index"], 1)
            self.assertIn("合并单元格", request["instruction"])
            self.assertIn("【确认表格结构补丁】", request["backend_payload"]["user_message"])

            repair_results = {
                "schema_version": "repair-results-v1",
                "summary": {"repair_request_count": 1, "succeeded_count": 1},
                "results": [
                    {
                        "request_id": request["request_id"],
                        "chunk_id": "c0000",
                        "status": "succeeded",
                        "action": "repair_table_cell_tokens",
                        "scope": "table_cell",
                        "result_excerpt": (
                            "| 模型 | 结果 | p |\n"
                            "| --- | --- | --- |\n"
                            "| BERT | 91.2% p<0.05 |  |\n"
                        ),
                    }
                ],
            }
            validation = build_repair_validation(requests, repair_results)
            self.assertEqual(validation["summary"]["passed_count"], 1)
            self.assertEqual(validation["summary"]["structure_patch_context_count"], 1)
            merge = build_repair_merge(
                requests,
                repair_results,
                validation,
                chunks,
                chunk_dir,
                repaired_chunk_dir=root / "repaired_chunks",
                repaired_full_path=root / "repaired_full.md",
            )
            self.assertEqual(merge["summary"]["structure_patch_context_candidate_count"], 1)
            self.assertEqual(merge["summary"]["applied_structure_patch_context_count"], 1)
            applied_patch = next(
                patch
                for patch in merge["patches"]
                if patch.get("request_id") == request["request_id"]
            )
            self.assertEqual(
                applied_patch["merge_target"]["structure_patch_source_review_ids"],
                ["tmc-0001-p1-b0000-r1c1"],
            )
            patch_review = build_repair_patch_review(merge)
            self.assertEqual(patch_review["summary"]["structure_patch_review_count"], 1)
            review = next(
                review
                for review in patch_review["patch_reviews"]
                if review.get("request_id") == request["request_id"]
            )
            self.assertEqual(
                review["merge_target"]["structure_patch_source_review_ids"],
                ["tmc-0001-p1-b0000-r1c1"],
            )
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
        self.assertEqual(fragment["continuation_kind"], "paragraph_continuation")
        self.assertEqual(fragment["stitch_action"], "translate_as_continuous_cross_page_text")
        self.assertEqual(fragment["previous_block_id"], "p1-b0000")
        self.assertEqual(fragment["next_block_id"], "p2-b0000")
        self.assertIn("previous_page_ends_without_terminal_punctuation", fragment["reasons"])
        self.assertIn("next_page_starts_like_continuation", fragment["reasons"])
        self.assertIn("The proposed method improves", fragment["previous_tail"])
        self.assertIn("accuracy under domain shift", fragment["next_head"])
        self.assertIn("The proposed method improves accuracy under domain shift", fragment["merged_preview"])

    def test_structure_qa_reports_hyphenated_page_boundary_fragments(self) -> None:
        doc_ir = DocumentIR(
            doc_id="hyphenated-boundary-sample",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text="The method improves cross-lingual trans-",
                    blocks=[
                        BlockIR(
                            "p1-b0000",
                            1,
                            "paragraph",
                            "The method improves cross-lingual trans-",
                            (40, 100, 520, 180),
                            0,
                        ),
                    ],
                ),
                PageIR(
                    page_no=2,
                    width=600,
                    height=800,
                    text="lation quality under domain shift.",
                    blocks=[
                        BlockIR(
                            "p2-b0000",
                            2,
                            "paragraph",
                            "lation quality under domain shift.",
                            (40, 80, 520, 140),
                            0,
                        ),
                    ],
                ),
            ],
        )
        structure_qa = build_structure_qa(doc_ir)
        fragment = structure_qa["page_boundary_fragments"][0]
        self.assertEqual(fragment["continuation_kind"], "hyphenated_word_continuation")
        self.assertEqual(fragment["stitch_action"], "join_hyphenated_word_across_page_boundary")
        self.assertEqual(fragment["joiner"], "hyphen_elision")
        self.assertIn("hyphenated_word_break_across_page", fragment["reasons"])
        self.assertIn("translation quality", fragment["merged_preview"])
        self.assertEqual(
            structure_qa["summary"]["continuation_kind_counts"]["hyphenated_word_continuation"],
            1,
        )
        self.assertEqual(
            structure_qa["summary"]["stitch_action_counts"]["join_hyphenated_word_across_page_boundary"],
            1,
        )

        page_chunks = [
            TextChunk("c0000", [0], "The method improves cross-lingual trans-", 0, 0),
            TextChunk("c0001", [1], "lation quality under domain shift.", 0, 0),
        ]
        page_report = build_chunk_boundary_qa(page_chunks, structure_qa, pipeline_variant="page")
        self.assertEqual(page_report["summary"]["hyphenated_boundary_count"], 1)
        self.assertEqual(page_report["summary"]["hyphenated_split_count"], 1)
        self.assertEqual(page_report["summary"]["hyphenated_split_rate"], 1.0)
        self.assertTrue(page_report["boundaries"][0]["is_hyphenated_continuation"])

        structure_chunks = build_structure_chunks(
            doc_ir,
            target_chars=1000,
            max_chars=2000,
            max_pages_per_chunk=1,
        )
        self.assertEqual(len(structure_chunks), 1)
        self.assertEqual(structure_chunks[0].boundary_fragment_ids, ["p1-p2"])
        self.assertIn("join_hyphenated_word_across_page_boundary", structure_chunks[0].boundary_stitch_notes[0])
        structure_report = build_chunk_boundary_qa(
            structure_chunks,
            structure_qa,
            pipeline_variant="structure",
        )
        self.assertEqual(structure_report["summary"]["hyphenated_protected_count"], 1)
        self.assertEqual(structure_report["summary"]["hyphenated_split_count"], 0)
        self.assertEqual(structure_report["summary"]["hyphenated_protected_rate"], 1.0)

        comparison = build_chunk_strategy_comparison(
            {"page": page_chunks, "structure": structure_chunks},
            structure_qa,
            active_strategy="structure",
        )
        self.assertEqual(comparison["summary"]["baseline_hyphenated_split_count"], 1)
        self.assertEqual(comparison["summary"]["active_hyphenated_split_count"], 0)
        self.assertEqual(comparison["summary"]["active_hyphenated_split_reduction_vs_baseline"], 1)

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
        self.assertTrue(chunks[0].boundary_stitch_notes)
        self.assertIn("[结构续接提示]", chunks[0].text)
        self.assertIn("accuracy under domain shift", chunks[0].text)

    def test_structure_chunks_prioritize_table_continuation_boundary(self) -> None:
        table_page_1 = ("Metric Acc F1\n" + ("BERT 91 88\n" * 90)).strip()
        table_page_2 = "RoBERTa 92 89\nXLNet 90 87"
        doc_ir = DocumentIR(
            doc_id="table-continuation-boundary",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text=table_page_1,
                    blocks=[
                        BlockIR("p1-b0000", 1, "table", table_page_1, (40, 100, 520, 760), 0),
                    ],
                ),
                PageIR(
                    page_no=2,
                    width=600,
                    height=800,
                    text=table_page_2,
                    blocks=[
                        BlockIR("p2-b0000", 2, "table", table_page_2, (40, 80, 520, 180), 0),
                    ],
                ),
            ],
        )

        structure_qa = build_structure_qa(doc_ir)
        fragment = structure_qa["page_boundary_fragments"][0]
        self.assertEqual(fragment["continuation_kind"], "table_continuation")
        self.assertEqual(fragment["stitch_action"], "preserve_table_segments_together")
        self.assertEqual(structure_qa["summary"]["table_continuation_boundary_count"], 1)
        self.assertEqual(structure_qa["summary"]["continuation_kind_counts"]["table_continuation"], 1)

        page_chunks = [
            TextChunk("c0000", [0], table_page_1, 0, 0),
            TextChunk("c0001", [1], table_page_2, 0, 0),
        ]
        page_report = build_chunk_boundary_qa(page_chunks, structure_qa, pipeline_variant="page")
        self.assertEqual(page_report["summary"]["table_continuation_boundary_count"], 1)
        self.assertEqual(page_report["summary"]["table_continuation_split_count"], 1)
        self.assertEqual(page_report["summary"]["table_continuation_split_rate"], 1.0)
        self.assertTrue(page_report["boundaries"][0]["is_table_continuation"])

        structure_chunks = build_structure_chunks(
            doc_ir,
            target_chars=1000,
            max_chars=2000,
            max_pages_per_chunk=1,
        )
        self.assertEqual(len(structure_chunks), 1)
        self.assertEqual(structure_chunks[0].boundary_fragment_ids, ["p1-p2"])
        self.assertIn("protected_page_boundary:p1-p2", structure_chunks[0].warnings)
        self.assertIn("protected_table_continuation:p1-p2", structure_chunks[0].warnings)
        self.assertIn("preserve_table_segments_together", structure_chunks[0].boundary_stitch_notes[0])
        self.assertIn("[结构续接提示]", structure_chunks[0].text)

        structure_report = build_chunk_boundary_qa(
            structure_chunks,
            structure_qa,
            pipeline_variant="structure",
        )
        self.assertEqual(structure_report["summary"]["table_continuation_protected_count"], 1)
        self.assertEqual(structure_report["summary"]["table_continuation_split_count"], 0)
        self.assertEqual(structure_report["summary"]["table_continuation_protected_rate"], 1.0)
        comparison = build_chunk_strategy_comparison(
            {"page": page_chunks, "structure": structure_chunks},
            structure_qa,
            active_strategy="structure",
        )
        self.assertEqual(comparison["summary"]["baseline_table_continuation_split_count"], 1)
        self.assertEqual(comparison["summary"]["active_table_continuation_split_count"], 0)
        self.assertEqual(comparison["summary"]["active_table_continuation_split_reduction_vs_baseline"], 1)

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
                            locked_tokens=["91.2"],
                            meta={
                                "table": {
                                    "row_count": 2,
                                    "column_count": 2,
                                    "header": ["Metric", "Acc"],
                                    "numeric_tokens": ["91.2"],
                                    "warnings": ["numeric_dense_table"],
                                    "confidence": "medium",
                                }
                            },
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
            self.assertEqual(manifest["summary"]["structured_contract_task_count"], 1)
            self.assertEqual(manifest["summary"]["table_context_task_count"], 1)
            self.assertEqual(manifest["summary"]["table_context_ready_task_count"], 1)
            self.assertEqual(manifest["summary"]["structure_target_counts"]["table"], 1)
            self.assertEqual(manifest["tasks"][0]["input_path"], "vision_crops/page-0001/p1-b0000-image.png")
            self.assertEqual(manifest["tasks"][0]["writeback"]["target"], "document_ir.block.meta.ocr_candidates")
            self.assertTrue(manifest["tasks"][0]["block_known_in_document_ir"])
            table_task = manifest["tasks"][1]
            self.assertEqual(table_task["block_type"], "table")
            self.assertEqual(table_task["layout_scope"], "table_region")
            self.assertEqual(table_task["target_structure_type"], "table")
            self.assertEqual(table_task["recommended_engine"], "local_table_ocr")
            self.assertEqual(table_task["writeback"]["block_id"], "p1-b0001")
            self.assertEqual(table_task["writeback"]["subtarget"]["type"], "table_block")
            self.assertEqual(table_task["table_context"]["table_id"], "p1-b0001")
            self.assertEqual(table_task["table_context"]["row_count"], 2)
            self.assertEqual(table_task["table_context"]["column_count"], 2)
            self.assertEqual(table_task["table_context"]["header"], ["Metric", "Acc"])
            self.assertEqual(table_task["table_context"]["numeric_tokens"], ["91.2"])
            self.assertEqual(table_task["table_context"]["locked_tokens"], ["91.2"])
            self.assertEqual(table_task["structure_contract"]["schema_version"], "ocr-structure-contract-v1")
            self.assertIn("structured_cells", table_task["structure_contract"]["optional_result_fields"])
            self.assertEqual(manifest["result_writeback_contract"]["schema_version"], "ocr-result-v1")
            self.assertIn(
                "table_context",
                manifest["result_writeback_contract"]["optional_structured_fields"],
            )
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_table_ocr_writeback_preserves_structure_context(self) -> None:
        doc_ir = DocumentIR(
            doc_id="table-ocr-context",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text="Metric Acc\nA 91.2",
                    image_count=1,
                    warnings=["low_text_image_heavy_page", "table_like_content"],
                    meta={
                        "text_char_count": 18,
                        "text_area_ratio": 0.02,
                        "image_area_ratio": 0.5,
                    },
                    blocks=[
                        BlockIR(
                            "p1-b0000",
                            1,
                            "table",
                            "Metric Acc\nA 91.2",
                            (60, 540, 500, 620),
                            0,
                            locked_tokens=["91.2"],
                            meta={
                                "table": {
                                    "row_count": 2,
                                    "column_count": 2,
                                    "header": ["Metric", "Acc"],
                                    "numeric_tokens": ["91.2"],
                                    "confidence": "medium",
                                }
                            },
                        ),
                    ],
                )
            ],
        )
        route = {
            "schema_version": "vision-route-v1",
            "doc_id": doc_ir.doc_id,
            "summary": {},
            "pages": [
                {
                    "page_no": 1,
                    "action": "local_ocr",
                    "risk_level": "high",
                    "risk_score": 0.8,
                    "reasons": ["possible_image_table"],
                    "evidence": {
                        "page_preview_path": "vision_pages/page-0001.png",
                        "region_crops": [
                            {
                                "block_id": "p1-b0000",
                                "block_type": "table",
                                "crop_path": "vision_crops/page-0001/p1-b0000-table.png",
                                "bbox": [60, 540, 500, 620],
                                "crop_width": 720,
                                "crop_height": 140,
                            }
                        ],
                    },
                }
            ],
        }
        manifest = build_ocr_task_manifest(doc_ir, route)
        table_task = manifest["tasks"][0]
        results = build_ocr_results_payload(
            manifest,
            {
                "schema_version": "ocr-results-v1",
                "results": [
                    {
                        "task_id": table_task["task_id"],
                        "status": "succeeded",
                        "text": "Metric Accuracy\nA 91.2",
                        "confidence": 0.91,
                        "engine": "unit_table_ocr",
                        "language": "en",
                        "bbox": [60, 540, 500, 620],
                        "warnings": [],
                        "structured_cells": [
                            {"row": 0, "col": 0, "text": "Metric", "role": "header"},
                            {"row": 0, "col": 1, "text": "Accuracy", "role": "header"},
                            {"row": 1, "col": 0, "text": "A"},
                            {"row": 1, "col": 1, "text": "91.2"},
                        ],
                        "cell_bboxes": [
                            {"row": 0, "col": 0, "bbox": [60, 540, 260, 580]},
                            {"row": 0, "col": 1, "bbox": [260, 540, 500, 580]},
                            {"row": 1, "col": 0, "bbox": [60, 580, 260, 620]},
                            {"row": 1, "col": 1, "bbox": [260, 580, 500, 620]},
                        ],
                        "merged_cell_candidates": [
                            {
                                "type": "colspan",
                                "row": 0,
                                "cols": [0, 1],
                                "confidence": 0.62,
                                "reason": "visual_header_span",
                            }
                        ],
                        "table_footnotes": [{"marker": "*", "text": "p < 0.05", "row": 1, "col": 1}],
                    }
                ],
            },
        )

        built = build_ocr_writeback(doc_ir, manifest, results)
        self.assertEqual(built["summary"]["accepted_result_count"], 1)
        self.assertEqual(built["summary"]["table_context_writeback_count"], 1)
        self.assertEqual(built["summary"]["structured_result_writeback_count"], 1)
        self.assertEqual(built["summary"]["structured_result_field_counts"]["structured_cells"], 1)
        self.assertEqual(built["summary"]["structured_result_item_counts"]["structured_cells"], 4)
        self.assertEqual(built["writebacks"][0]["structured_result_item_counts"]["cell_bboxes"], 4)
        candidate = built["augmented_document_ir"]["pages"][0]["blocks"][0]["meta"]["ocr_candidates"][0]
        self.assertEqual(candidate["target_structure_type"], "table")
        self.assertEqual(candidate["table_context"]["table_id"], "p1-b0000")
        self.assertEqual(candidate["table_context"]["row_count"], 2)
        self.assertEqual(candidate["subtarget"]["type"], "table_block")
        self.assertEqual(candidate["structured_cells"][3]["text"], "91.2")
        self.assertEqual(candidate["cell_bboxes"][3]["row"], 1)
        self.assertEqual(candidate["merged_cell_candidates"][0]["reason"], "visual_header_span")
        self.assertEqual(candidate["merged_cell_candidates"][0]["span_type"], "colspan")
        self.assertEqual(candidate["merged_cell_candidates"][0]["column_span"], 2)
        self.assertEqual(candidate["merged_cell_candidates"][0]["covered_cells"][0]["column_index"], 1)
        self.assertEqual(candidate["table_footnotes"][0]["marker"], "*")

        candidate_qa = build_ocr_candidate_qa(built["augmented_document_ir"], built)
        self.assertEqual(candidate_qa["summary"]["candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["table_context_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["structured_contract_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["subtarget_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["structured_result_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["structured_cells_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["cell_bboxes_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["merged_cell_candidates_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["table_footnotes_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["structured_cell_count"], 4)
        self.assertEqual(candidate_qa["summary"]["cell_bbox_count"], 4)
        self.assertEqual(candidate_qa["summary"]["result_merged_cell_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["result_table_footnote_count"], 1)
        self.assertEqual(candidate_qa["summary"]["promotable_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["structured_table_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["structured_table_gate_passed_count"], 1)
        self.assertEqual(candidate_qa["summary"]["structured_table_gate_review_count"], 0)
        self.assertEqual(candidate_qa["candidates"][0]["table_context"]["table_id"], "p1-b0000")
        self.assertEqual(candidate_qa["candidates"][0]["subtarget"]["expected_granularity"], "rows_and_cells")
        self.assertEqual(candidate_qa["candidates"][0]["status"], "candidate")
        self.assertEqual(candidate_qa["candidates"][0]["structured_table_gate"]["status"], "passed")
        self.assertEqual(candidate_qa["candidates"][0]["structured_cells"][1]["text"], "Accuracy")
        self.assertEqual(candidate_qa["candidates"][0]["table_footnotes"][0]["text"], "p < 0.05")

        promotion = build_ocr_candidate_promotion(built["augmented_document_ir"], candidate_qa)
        self.assertEqual(promotion["summary"]["promoted_candidate_count"], 1)
        self.assertEqual(promotion["summary"]["canonical_structure_promotion_count"], 1)
        self.assertEqual(promotion["summary"]["structured_table_promotion_count"], 1)
        self.assertEqual(promotion["summary"]["structured_formula_promotion_count"], 0)
        self.assertTrue(promotion["promotions"][0]["structured_table_promoted"])
        self.assertEqual(promotion["promotions"][0]["canonical_table_row_count"], 2)
        self.assertEqual(promotion["promotions"][0]["canonical_table_column_count"], 2)
        self.assertEqual(promotion["promotions"][0]["structured_cells"][3]["text"], "91.2")
        self.assertEqual(promotion["promotions"][0]["cell_bboxes"][0]["bbox"], [60, 540, 260, 580])
        self.assertEqual(promotion["promotions"][0]["merged_cell_candidates"][0]["type"], "colspan")
        self.assertEqual(
            promotion["promotions"][0]["canonical_structure_targets"],
            ["document_ir.block.meta.table"],
        )
        promoted_block = promotion["promoted_document_ir"]["pages"][0]["blocks"][0]
        promoted_table = promoted_block["meta"]["table"]
        self.assertEqual(promoted_table["source"], "ocr_candidate_promotion")
        self.assertTrue(promoted_table["ocr_structured"])
        self.assertEqual(promoted_table["rows"][1][1], "91.2")
        promoted_doc_ir = document_ir_from_json_dict(promotion["promoted_document_ir"])
        promoted_chunks = build_structure_chunks(promoted_doc_ir)
        self.assertIn("| Metric | Accuracy |", promoted_chunks[0].text)
        self.assertIn("| A | 91.2 |", promoted_chunks[0].text)
        promoted_meta = promotion["promoted_document_ir"]["pages"][0]["blocks"][0]["meta"]["ocr_promotions"][0]
        self.assertEqual(promoted_meta["table_footnotes"][0]["marker"], "*")

    def test_table_ocr_plain_text_can_be_locally_structured(self) -> None:
        doc_ir = DocumentIR(
            doc_id="plain-text-table-ocr",
            source_pdf="plain-text-table-ocr.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text="Metric Accuracy\nA 91.2",
                    image_count=1,
                    warnings=["low_text_image_heavy_page", "table_like_content"],
                    meta={"text_char_count": 18, "text_area_ratio": 0.02, "image_area_ratio": 0.5},
                    blocks=[
                        BlockIR(
                            "p1-b0000",
                            1,
                            "table",
                            "Metric Accuracy\nA 91.2",
                            (60, 540, 500, 620),
                            0,
                            locked_tokens=["91.2"],
                            meta={"table": {"row_count": 2, "column_count": 2, "confidence": "medium"}},
                        ),
                    ],
                )
            ],
        )
        route = {
            "schema_version": "vision-route-v1",
            "doc_id": doc_ir.doc_id,
            "summary": {},
            "pages": [
                {
                    "page_no": 1,
                    "action": "local_ocr",
                    "risk_level": "high",
                    "risk_score": 0.8,
                    "reasons": ["possible_image_table"],
                    "evidence": {
                        "page_preview_path": "vision_pages/page-0001.png",
                        "region_crops": [
                            {
                                "block_id": "p1-b0000",
                                "block_type": "table",
                                "crop_path": "vision_crops/page-0001/p1-b0000-table.png",
                                "bbox": [60, 540, 500, 620],
                            }
                        ],
                    },
                }
            ],
        }
        manifest = build_ocr_task_manifest(doc_ir, route)
        table_task = manifest["tasks"][0]
        results = build_ocr_results_payload(
            manifest,
            {
                "schema_version": "ocr-results-v1",
                "results": [
                    {
                        "task_id": table_task["task_id"],
                        "status": "succeeded",
                        "text": "| Metric | Accuracy |\n| --- | --- |\n| A | 91.2 |",
                        "confidence": 0.9,
                        "engine": "plain_text_table_ocr",
                        "language": "en",
                        "bbox": [60, 540, 500, 620],
                    }
                ],
            },
        )

        built = build_ocr_writeback(doc_ir, manifest, results)
        candidate = built["augmented_document_ir"]["pages"][0]["blocks"][0]["meta"]["ocr_candidates"][0]
        self.assertEqual(built["summary"]["structured_result_writeback_count"], 1)
        self.assertEqual(candidate["structured_cells"][0]["source"], "local_text_table_parser")
        self.assertEqual(candidate["structured_cells"][3]["text"], "91.2")
        self.assertEqual(candidate["cell_bboxes"][3]["row"], 1)
        self.assertIn("structured_table_inferred_from_text", candidate["warnings"])
        self.assertIn("cell_bboxes_estimated_from_region", candidate["warnings"])

        candidate_qa = build_ocr_candidate_qa(built["augmented_document_ir"], built)
        self.assertEqual(candidate_qa["summary"]["promotable_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["structured_table_gate_passed_count"], 1)
        self.assertEqual(candidate_qa["candidates"][0]["status"], "candidate")
        promotion = build_ocr_candidate_promotion(built["augmented_document_ir"], candidate_qa)
        self.assertEqual(promotion["summary"]["structured_table_promotion_count"], 1)
        promoted_table = promotion["promoted_document_ir"]["pages"][0]["blocks"][0]["meta"]["table"]
        self.assertTrue(promoted_table["ocr_structured"])
        self.assertEqual(promoted_table["rows"][1][1], "91.2")
        promoted_doc_ir = document_ir_from_json_dict(promotion["promoted_document_ir"])
        promoted_chunks = build_structure_chunks(promoted_doc_ir)
        self.assertIn("| Metric | Accuracy |", promoted_chunks[0].text)
        self.assertIn("| A | 91.2 |", promoted_chunks[0].text)

    def test_table_ocr_plain_text_infers_merged_cell_candidates(self) -> None:
        doc_ir = DocumentIR(
            doc_id="plain-text-table-ocr-merged-candidates",
            source_pdf="plain-text-table-ocr-merged-candidates.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text="Dataset metrics\nModel Acc F1\nA 91 88",
                    image_count=1,
                    warnings=["low_text_image_heavy_page", "table_like_content"],
                    meta={"text_char_count": 35, "text_area_ratio": 0.02, "image_area_ratio": 0.5},
                    blocks=[
                        BlockIR(
                            "p1-b0000",
                            1,
                            "table",
                            "Dataset metrics\nModel Acc F1\nA 91 88",
                            (60, 520, 520, 640),
                            0,
                            locked_tokens=["91", "88"],
                            meta={"table": {"row_count": 3, "column_count": 3, "confidence": "medium"}},
                        ),
                    ],
                )
            ],
        )
        route = {
            "schema_version": "vision-route-v1",
            "doc_id": doc_ir.doc_id,
            "summary": {},
            "pages": [
                {
                    "page_no": 1,
                    "action": "local_ocr",
                    "risk_level": "high",
                    "risk_score": 0.82,
                    "reasons": ["possible_image_table"],
                    "evidence": {
                        "page_preview_path": "vision_pages/page-0001.png",
                        "region_crops": [
                            {
                                "block_id": "p1-b0000",
                                "block_type": "table",
                                "crop_path": "vision_crops/page-0001/p1-b0000-table.png",
                                "bbox": [60, 520, 520, 640],
                            }
                        ],
                    },
                }
            ],
        }
        manifest = build_ocr_task_manifest(doc_ir, route)
        table_task = manifest["tasks"][0]
        results = build_ocr_results_payload(
            manifest,
            {
                "schema_version": "ocr-results-v1",
                "results": [
                    {
                        "task_id": table_task["task_id"],
                        "status": "succeeded",
                        "text": "| Dataset metrics |\n| Model | Acc | F1 |\n| A | 91 | 88 |",
                        "confidence": 0.92,
                        "engine": "plain_text_table_ocr",
                        "language": "en",
                        "bbox": [60, 520, 520, 640],
                    }
                ],
            },
        )

        built = build_ocr_writeback(doc_ir, manifest, results)
        candidate = built["augmented_document_ir"]["pages"][0]["blocks"][0]["meta"]["ocr_candidates"][0]
        self.assertEqual(candidate["structured_cells"][0]["text"], "Dataset metrics")
        self.assertEqual(candidate["structured_cells"][1]["text"], "")
        self.assertEqual(candidate["structured_cells"][8]["text"], "88")
        self.assertIn("merged_cell_candidates_inferred_from_text", candidate["warnings"])
        self.assertEqual(built["summary"]["structured_result_item_counts"]["merged_cell_candidates"], 1)
        merged = candidate["merged_cell_candidates"][0]
        self.assertEqual(merged["type"], "colspan")
        self.assertEqual(merged["span_type"], "colspan")
        self.assertEqual(merged["row"], 0)
        self.assertEqual(merged["row_index"], 0)
        self.assertEqual(merged["col"], 0)
        self.assertEqual(merged["column_span"], 3)
        self.assertEqual(merged["cols"], [0, 1, 2])
        self.assertEqual(merged["reason"], "single_cell_ragged_row")
        self.assertEqual(merged["candidate_status"], "candidate")
        self.assertEqual(merged["visual_evidence_level"], "none")
        self.assertEqual(merged["bbox_evidence"]["status"], "missing")
        self.assertEqual(
            [(cell["row_index"], cell["column_index"]) for cell in merged["covered_cells"]],
            [(0, 1), (0, 2)],
        )

        candidate_qa = build_ocr_candidate_qa(built["augmented_document_ir"], built)
        self.assertEqual(candidate_qa["summary"]["promotable_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["merged_cell_candidates_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["result_merged_cell_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["structured_table_gate_passed_count"], 1)
        self.assertEqual(candidate_qa["candidates"][0]["status"], "candidate")

        promotion = build_ocr_candidate_promotion(built["augmented_document_ir"], candidate_qa)
        self.assertEqual(promotion["summary"]["structured_table_promotion_count"], 1)
        self.assertEqual(promotion["promotions"][0]["merged_cell_candidates"][0]["reason"], "single_cell_ragged_row")
        promoted_table = promotion["promoted_document_ir"]["pages"][0]["blocks"][0]["meta"]["table"]
        self.assertEqual(promoted_table["rows"][0], ["Dataset metrics", "", ""])
        self.assertEqual(promoted_table["merged_cell_candidates"][0]["source"], "local_text_table_parser")
        promoted_doc_ir = document_ir_from_json_dict(promotion["promoted_document_ir"])
        reconstruction = build_table_reconstruction_report(promoted_doc_ir, build_structure_qa(promoted_doc_ir))
        self.assertEqual(reconstruction["summary"]["merged_cell_candidate_count"], 1)
        self.assertEqual(reconstruction["summary"]["merged_cell_candidate_type_counts"]["colspan"], 1)
        self.assertEqual(
            reconstruction["summary"]["merged_cell_candidate_reason_counts"]["single_cell_ragged_row"],
            1,
        )
        reconstruction_candidate = reconstruction["tables"][0]["merged_cell_candidates"][0]
        self.assertEqual(reconstruction_candidate["column_span"], 3)
        self.assertEqual(reconstruction_candidate["reason"], "single_cell_ragged_row")
        self.assertEqual(reconstruction_candidate["source"], "local_text_table_parser")
        self.assertEqual(reconstruction_candidate["source_task_id"], table_task["task_id"])
        self.assertEqual(reconstruction_candidate["engine"], "plain_text_table_ocr")
        self.assertEqual(reconstruction_candidate["candidate_status"], "candidate")
        self.assertEqual(reconstruction_candidate["visual_evidence_level"], "estimated_bbox")
        self.assertEqual(reconstruction_candidate["bbox_evidence"]["status"], "estimated")
        self.assertEqual(reconstruction_candidate["bbox_evidence"]["support_status"], "estimated_grid_only")
        self.assertEqual(reconstruction_candidate["confirmation_status"], "estimated_grid_only")
        self.assertNotEqual(reconstruction_candidate["candidate_status"], "visually_supported")
        self.assertNotEqual(reconstruction_candidate["candidate_status"], "human_confirmed")
        hints = build_table_translation_hints(
            TextChunk("c0000", [0], promoted_doc_ir.pages[0].blocks[0].text, 0, 0),
            reconstruction,
        )
        self.assertIn("r0c0", hints)
        self.assertIn("colspan 1x3", hints)
        self.assertIn("single_cell_ragged_row", hints)
        self.assertIn("Dataset metrics", hints)
        self.assertIn("证据=estimated/estimated_bbox/candidate", hints)

    def test_formula_ocr_writeback_preserves_structure_context(self) -> None:
        doc_ir = DocumentIR(
            doc_id="formula-ocr-context",
            source_pdf="sample.pdf",
            pages=[
                PageIR(
                    page_no=1,
                    width=600,
                    height=800,
                    text="Formula image L_i (1)",
                    image_count=1,
                    warnings=["formula_dense_low_text"],
                    meta={
                        "text_char_count": 20,
                        "text_area_ratio": 0.02,
                        "image_area_ratio": 0.45,
                    },
                    blocks=[
                        BlockIR(
                            "p1-b0000",
                            1,
                            "formula",
                            "Formula image L_i (1)",
                            (80, 220, 520, 300),
                            0,
                            locked_tokens=["L_i", "(1)"],
                        ),
                    ],
                )
            ],
        )
        route = {
            "schema_version": "vision-route-v1",
            "doc_id": doc_ir.doc_id,
            "summary": {},
            "pages": [
                {
                    "page_no": 1,
                    "action": "local_ocr",
                    "risk_level": "high",
                    "risk_score": 0.86,
                    "reasons": ["formula_dense_low_text"],
                    "evidence": {
                        "page_preview_path": "vision_pages/page-0001.png",
                        "region_crops": [
                            {
                                "block_id": "p1-b0000",
                                "block_type": "formula",
                                "crop_path": "vision_crops/page-0001/p1-b0000-formula.png",
                                "bbox": [80, 220, 520, 300],
                                "crop_width": 720,
                                "crop_height": 120,
                            }
                        ],
                    },
                }
            ],
        }

        manifest = build_ocr_task_manifest(doc_ir, route)
        formula_task = manifest["tasks"][0]
        self.assertEqual(manifest["summary"]["formula_context_task_count"], 1)
        self.assertEqual(manifest["summary"]["formula_context_ready_task_count"], 1)
        self.assertEqual(manifest["summary"]["structured_contract_task_count"], 1)
        self.assertEqual(formula_task["block_type"], "formula")
        self.assertEqual(formula_task["layout_scope"], "formula_region")
        self.assertEqual(formula_task["target_structure_type"], "formula")
        self.assertEqual(formula_task["recommended_engine"], "local_formula_ocr")
        self.assertEqual(formula_task["writeback"]["subtarget"]["type"], "formula_block")
        self.assertEqual(formula_task["formula_context"]["formula_id"], "p1-b0000")
        self.assertIn("L_i", formula_task["formula_context"]["source_tokens"])
        self.assertIn("(1)", formula_task["formula_context"]["source_tokens"])
        self.assertEqual(formula_task["structure_contract"]["target_structure_type"], "formula")
        self.assertIn("formula_latex", formula_task["structure_contract"]["optional_result_fields"])
        self.assertIn(
            "formula_tokens",
            manifest["result_writeback_contract"]["optional_structured_fields"],
        )

        results = build_ocr_results_payload(
            manifest,
            {
                "schema_version": "ocr-results-v1",
                "results": [
                    {
                        "task_id": formula_task["task_id"],
                        "status": "succeeded",
                        "text": "L_i = sum_j x_{ij} (1)",
                        "confidence": 0.94,
                        "engine": "unit_formula_ocr",
                        "language": "en",
                        "bbox": [80, 220, 520, 300],
                        "warnings": [],
                        "formula_latex": r"L_i = \sum_j x_{ij}",
                        "formula_tokens": ["L_i", "=", r"\sum", "x_{ij}", "(1)"],
                        "equation_labels": ["(1)"],
                        "formula_confidence": 0.93,
                    }
                ],
            },
        )

        built = build_ocr_writeback(doc_ir, manifest, results)
        self.assertEqual(built["summary"]["accepted_result_count"], 1)
        self.assertEqual(built["summary"]["formula_context_writeback_count"], 1)
        self.assertEqual(built["summary"]["structured_result_writeback_count"], 1)
        self.assertEqual(built["summary"]["structured_result_field_counts"]["formula_latex"], 1)
        self.assertEqual(built["summary"]["structured_result_item_counts"]["formula_tokens"], 5)
        candidate = built["augmented_document_ir"]["pages"][0]["blocks"][0]["meta"]["ocr_candidates"][0]
        self.assertEqual(candidate["target_structure_type"], "formula")
        self.assertEqual(candidate["formula_context"]["formula_id"], "p1-b0000")
        self.assertEqual(candidate["subtarget"]["type"], "formula_block")
        self.assertEqual(candidate["formula_tokens"][0], "L_i")
        self.assertEqual(candidate["equation_labels"], ["(1)"])

        candidate_qa = build_ocr_candidate_qa(built["augmented_document_ir"], built)
        self.assertEqual(candidate_qa["summary"]["candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["formula_context_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["formula_latex_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["formula_tokens_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["result_formula_token_count"], 5)
        self.assertEqual(candidate_qa["summary"]["structured_formula_candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["structured_formula_gate_passed_count"], 1)
        self.assertEqual(candidate_qa["summary"]["structured_formula_gate_review_count"], 0)
        self.assertEqual(candidate_qa["summary"]["structured_formula_token_count"], 5)
        self.assertEqual(candidate_qa["summary"]["structured_formula_equation_label_count"], 1)
        self.assertEqual(candidate_qa["candidates"][0]["status"], "candidate")
        self.assertEqual(candidate_qa["candidates"][0]["structured_formula_gate"]["status"], "passed")
        self.assertEqual(candidate_qa["candidates"][0]["formula_context"]["formula_id"], "p1-b0000")

        promotion = build_ocr_candidate_promotion(built["augmented_document_ir"], candidate_qa)
        self.assertEqual(promotion["summary"]["promoted_candidate_count"], 1)
        self.assertEqual(promotion["summary"]["canonical_structure_promotion_count"], 1)
        self.assertEqual(promotion["summary"]["structured_table_promotion_count"], 0)
        self.assertEqual(promotion["summary"]["structured_formula_promotion_count"], 1)
        self.assertTrue(promotion["promotions"][0]["structured_formula_promoted"])
        self.assertEqual(promotion["promotions"][0]["canonical_formula_token_count"], 5)
        self.assertEqual(promotion["promotions"][0]["canonical_formula_equation_label_count"], 1)
        self.assertEqual(promotion["promotions"][0]["formula_tokens"][0], "L_i")
        self.assertEqual(promotion["promotions"][0]["formula_context"]["formula_id"], "p1-b0000")
        promoted_block = promotion["promoted_document_ir"]["pages"][0]["blocks"][0]
        promoted_formula = promoted_block["meta"]["formula"]
        self.assertEqual(promoted_formula["source"], "ocr_candidate_promotion")
        self.assertEqual(promoted_formula["latex"], r"L_i = \sum_j x_{ij}")
        self.assertEqual(promoted_formula["tokens"][0], "L_i")
        self.assertEqual(promoted_formula["equation_labels"], ["(1)"])
        promoted_meta = promotion["promoted_document_ir"]["pages"][0]["blocks"][0]["meta"]["ocr_promotions"][0]
        self.assertEqual(promoted_meta["formula_latex"], r"L_i = \sum_j x_{ij}")
        self.assertEqual(promoted_meta["equation_labels"], ["(1)"])

    def test_structured_table_ocr_gate_requires_locked_tokens_and_bboxes(self) -> None:
        document_ir_ocr = {
            "doc_id": "table-ocr-gate",
            "pages": [
                {
                    "page_no": 1,
                    "text": "Table image",
                    "meta": {},
                    "blocks": [
                        {
                            "block_id": "p1-b0000",
                            "page_no": 1,
                            "type": "table",
                            "text": "Table image",
                            "meta": {
                                "ocr_candidates": [
                                    {
                                        "task_id": "ocr-task-1",
                                        "page_no": 1,
                                        "block_id": "p1-b0000",
                                        "scope": "region",
                                        "block_type": "table",
                                        "target_structure_type": "table",
                                        "text": "Metric Accuracy\nA 90.0",
                                        "confidence": 0.96,
                                        "engine": "unit_table_ocr",
                                        "language": "en",
                                        "table_context": {
                                            "table_id": "p1-b0000",
                                            "row_count": 2,
                                            "column_count": 2,
                                            "locked_tokens": ["91.2"],
                                        },
                                        "structured_cells": [
                                            {"row": 0, "col": 0, "text": "Metric"},
                                            {"row": 0, "col": 1, "text": "Accuracy"},
                                            {"row": 1, "col": 0, "text": "A"},
                                            {"row": 1, "col": 1, "text": "90.0"},
                                        ],
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        }

        qa = build_ocr_candidate_qa(document_ir_ocr, None)

        self.assertEqual(qa["summary"]["candidate_count"], 1)
        self.assertEqual(qa["summary"]["needs_review_candidate_count"], 1)
        self.assertEqual(qa["summary"]["structured_table_candidate_count"], 1)
        self.assertEqual(qa["summary"]["structured_table_gate_passed_count"], 0)
        self.assertEqual(qa["summary"]["structured_table_gate_review_count"], 1)
        self.assertEqual(qa["summary"]["structured_table_missing_locked_token_count"], 1)
        candidate = qa["candidates"][0]
        self.assertEqual(candidate["status"], "needs_review")
        self.assertEqual(candidate["structured_table_gate"]["status"], "needs_review")
        self.assertIn("structured_table_missing_locked_tokens", candidate["reasons"])
        self.assertIn("structured_table_missing_cell_bboxes", candidate["reasons"])
        self.assertIn("needs_structured_table_review", candidate["reasons"])

    def test_local_ocr_executor_runs_ready_tasks_into_results_payload(self) -> None:
        root = Path.cwd() / "test-output" / "ocr-executor"
        if root.exists():
            shutil.rmtree(root)
        try:
            crop_path = root / "output" / "vision_crops" / "page-0001" / "p1-b0000-image.png"
            crop_path.parent.mkdir(parents=True)
            crop_path.write_bytes(b"fake image")
            manifest = {
                "schema_version": "ocr-task-manifest-v1",
                "doc_id": "ocr-executor-sample",
                "tasks": [
                    {
                        "task_id": "ocr-p0001-r000",
                        "page_no": 1,
                        "scope": "region",
                        "status": "pending_engine",
                        "recommended_engine": "local_ocr",
                        "input_path": "vision_crops/page-0001/p1-b0000-image.png",
                        "block_id": "p1-b0000",
                        "bbox": [40, 80, 560, 520],
                    },
                    {
                        "task_id": "ocr-p0002-page-001",
                        "page_no": 2,
                        "scope": "page",
                        "status": "blocked_missing_visual_evidence",
                        "recommended_engine": "local_ocr",
                        "input_path": "",
                        "block_id": "",
                        "bbox": [],
                    },
                ],
            }
            seen: list[tuple[list[str], int]] = []

            def fake_runner(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
                seen.append((command, timeout_seconds))
                return subprocess.CompletedProcess(command, 0, stdout="Recognized figure text\n", stderr="")

            payload = execute_ocr_tasks(
                manifest,
                root,
                command="fake-tesseract",
                command_runner=fake_runner,
                language="eng",
                timeout_seconds=7,
            )

            self.assertEqual(payload["schema_version"], "ocr-results-v1")
            self.assertEqual(payload["source"], "local_ocr_executor")
            self.assertEqual(payload["execution"]["schema_version"], "ocr-execution-v1")
            self.assertEqual(payload["execution"]["summary"]["task_count"], 2)
            self.assertEqual(payload["execution"]["summary"]["attempted_task_count"], 1)
            self.assertEqual(payload["execution"]["summary"]["succeeded_task_count"], 1)
            self.assertEqual(payload["execution"]["summary"]["skipped_task_count"], 1)
            self.assertTrue(payload["execution"]["summary"]["engine_available"])
            self.assertEqual(payload["results"][0]["status"], "succeeded")
            self.assertEqual(payload["results"][0]["text"], "Recognized figure text")
            self.assertEqual(payload["results"][0]["confidence"], 0.6)
            self.assertIn("confidence_estimated", payload["results"][0]["warnings"])
            self.assertNotIn("structured_cells", payload["results"][0])
            self.assertEqual(payload["execution"]["commands"][0]["output_format"], "text")
            self.assertEqual(payload["results"][1]["status"], "skipped")
            self.assertEqual(seen[0][0][0], "fake-tesseract")
            self.assertEqual(Path(seen[0][0][1]), crop_path)
            self.assertEqual(seen[0][0][-1], "6")
            self.assertEqual(seen[0][1], 7)
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_local_ocr_executor_allows_task_recommended_engines(self) -> None:
        root = Path.cwd() / "test-output" / "ocr-executor-local-engines"
        if root.exists():
            shutil.rmtree(root)
        try:
            crop_path = root / "output" / "vision_crops" / "page-0001" / "p1-b0000-table.png"
            crop_path.parent.mkdir(parents=True)
            crop_path.write_bytes(b"fake table image")
            manifest = {
                "schema_version": "ocr-task-manifest-v1",
                "doc_id": "ocr-executor-local-engines",
                "tasks": [
                    {
                        "task_id": "ocr-p0001-table",
                        "page_no": 1,
                        "scope": "region",
                        "status": "pending_engine",
                        "recommended_engine": "local_table_ocr",
                        "input_path": "vision_crops/page-0001/p1-b0000-table.png",
                        "block_id": "p1-b0000",
                        "bbox": [60, 540, 500, 620],
                    }
                ],
            }

            def fake_runner(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(command, 0, stdout="Metric Accuracy\n", stderr="")

            payload = execute_ocr_tasks(
                manifest,
                root,
                engine="local_table_ocr",
                command="fake-table-ocr",
                command_runner=fake_runner,
            )

            self.assertTrue(payload["execution"]["summary"]["engine_available"])
            self.assertEqual(payload["execution"]["engine"]["type"], "local_table_ocr")
            self.assertEqual(payload["results"][0]["status"], "succeeded")
            self.assertNotIn("unsupported_ocr_engine", payload["results"][0]["warnings"])
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_local_ocr_executor_accepts_structured_json_stdout(self) -> None:
        root = Path.cwd() / "test-output" / "ocr-executor-structured-json"
        if root.exists():
            shutil.rmtree(root)
        try:
            table_crop = root / "output" / "vision_crops" / "page-0001" / "p1-b0000-table.png"
            formula_crop = root / "output" / "vision_crops" / "page-0001" / "p1-b0001-formula.png"
            table_crop.parent.mkdir(parents=True)
            table_crop.write_bytes(b"fake table image")
            formula_crop.write_bytes(b"fake formula image")
            manifest = {
                "schema_version": "ocr-task-manifest-v1",
                "doc_id": "ocr-executor-structured-json",
                "tasks": [
                    {
                        "task_id": "ocr-p0001-table",
                        "page_no": 1,
                        "scope": "region",
                        "status": "pending_engine",
                        "recommended_engine": "local_table_ocr",
                        "input_path": "vision_crops/page-0001/p1-b0000-table.png",
                        "block_id": "p1-b0000",
                        "bbox": [60, 540, 500, 620],
                    },
                    {
                        "task_id": "ocr-p0001-formula",
                        "page_no": 1,
                        "scope": "region",
                        "status": "pending_engine",
                        "recommended_engine": "local_formula_ocr",
                        "input_path": "vision_crops/page-0001/p1-b0001-formula.png",
                        "block_id": "p1-b0001",
                        "bbox": [80, 220, 520, 300],
                    },
                ],
            }

            def fake_runner(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
                input_path = str(command[1])
                if "table" in input_path:
                    stdout = json.dumps(
                        {
                            "schema_version": "ocr-results-v1",
                            "results": [
                                {
                                    "task_id": "ocr-p0001-table",
                                    "status": "succeeded",
                                    "text": "Metric Accuracy\nA 91.2",
                                    "confidence": 0.94,
                                    "engine": "unit_structured_ocr",
                                    "structured_cells": [
                                        {"row": 0, "col": 0, "text": "Metric"},
                                        {"row": 0, "col": 1, "text": "Accuracy"},
                                        {"row": 1, "col": 0, "text": "A"},
                                        {"row": 1, "col": 1, "text": "91.2"},
                                    ],
                                    "cell_bboxes": [
                                        {"row": 0, "col": 0, "bbox": [60, 540, 260, 580]},
                                        {"row": 0, "col": 1, "bbox": [260, 540, 500, 580]},
                                        {"row": 1, "col": 0, "bbox": [60, 580, 260, 620]},
                                        {"row": 1, "col": 1, "bbox": [260, 580, 500, 620]},
                                    ],
                                }
                            ],
                        }
                    )
                else:
                    stdout = json.dumps(
                        {
                            "text": "L_i = sum_j x_ij (1)",
                            "formula_latex": r"L_i = \sum_j x_{ij}",
                            "formula_tokens": ["L_i", "=", r"\sum", "x_{ij}", "(1)"],
                            "equation_labels": ["(1)"],
                            "formula_confidence": 0.92,
                        }
                    )
                return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

            payload = execute_ocr_tasks(
                manifest,
                root,
                engine="structured_json_cli",
                command="fake-structured-ocr",
                command_runner=fake_runner,
                language="eng",
                timeout_seconds=9,
            )

            self.assertEqual(payload["execution"]["summary"]["attempted_task_count"], 2)
            self.assertEqual(payload["execution"]["summary"]["succeeded_task_count"], 2)
            self.assertEqual(payload["execution"]["engine"]["type"], "structured_json_cli")
            self.assertEqual(payload["execution"]["commands"][0]["output_format"], "json")
            self.assertEqual(payload["execution"]["commands"][1]["output_format"], "json")
            table_result = payload["results"][0]
            formula_result = payload["results"][1]
            self.assertEqual(table_result["engine"], "unit_structured_ocr")
            self.assertEqual(table_result["structured_cells"][3]["text"], "91.2")
            self.assertEqual(table_result["cell_bboxes"][0]["bbox"], [60, 540, 260, 580])
            self.assertEqual(formula_result["task_id"], "ocr-p0001-formula")
            self.assertEqual(formula_result["block_id"], "p1-b0001")
            self.assertEqual(formula_result["confidence"], 0.6)
            self.assertIn("confidence_estimated", formula_result["warnings"])
            self.assertIn("structured_json_output", formula_result["warnings"])
            self.assertEqual(formula_result["formula_latex"], r"L_i = \sum_j x_{ij}")
            self.assertEqual(formula_result["formula_tokens"][0], "L_i")
            self.assertEqual(formula_result["equation_labels"], ["(1)"])
            self.assertEqual(formula_result["formula_confidence"], 0.92)
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
        candidate_qa = build_ocr_candidate_qa(built["augmented_document_ir"], built)
        self.assertEqual(candidate_qa["schema_version"], "ocr-candidate-qa-v1")
        self.assertEqual(candidate_qa["summary"]["candidate_count"], 1)
        self.assertEqual(candidate_qa["summary"]["promotable_candidate_count"], 1)
        self.assertEqual(candidate_qa["candidates"][0]["status"], "candidate")
        promotion = build_ocr_candidate_promotion(built["augmented_document_ir"], candidate_qa)
        self.assertEqual(promotion["schema_version"], "ocr-candidate-promotion-v1")
        self.assertEqual(promotion["summary"]["promoted_candidate_count"], 1)
        self.assertEqual(promotion["summary"]["block_promotion_count"], 1)
        promoted_doc_ir = document_ir_from_json_dict(promotion["promoted_document_ir"])
        promoted_chunks = build_structure_chunks(promoted_doc_ir)
        promoted_text = "\n\n".join(chunk.text for chunk in promoted_chunks)
        self.assertIn("Figure overview text", promoted_text)
        self.assertIn("p1-b0000", promoted_chunks[0].block_ids)

        root = Path.cwd() / "test-output" / "ocr-writeback"
        if root.exists():
            shutil.rmtree(root)
        try:
            report_path = root / "output" / "ocr_writeback.json"
            augmented_path = root / "output" / "document_ir_ocr.json"
            candidate_qa_path = root / "output" / "ocr_candidate_qa.json"
            candidate_qa_md_path = root / "output" / "ocr_candidate_qa.md"
            promotion_path = root / "output" / "ocr_candidate_promotion.json"
            promotion_md_path = root / "output" / "ocr_candidate_promotion.md"
            promoted_ir_path = root / "output" / "document_ir_promoted.json"
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
            candidate_qa_report = write_ocr_candidate_qa(
                augmented,
                report,
                candidate_qa_path,
                candidate_qa_md_path,
            )
            self.assertTrue(candidate_qa_path.is_file())
            self.assertTrue(candidate_qa_md_path.is_file())
            self.assertEqual(candidate_qa_report["summary"]["candidate_count"], 1)
            self.assertIn("OCR Candidate QA", candidate_qa_md_path.read_text(encoding="utf-8"))
            promotion_report = write_ocr_candidate_promotion(
                augmented,
                candidate_qa_report,
                promotion_path,
                promotion_md_path,
                promoted_ir_path,
            )
            self.assertTrue(promotion_path.is_file())
            self.assertTrue(promotion_md_path.is_file())
            self.assertTrue(promoted_ir_path.is_file())
            self.assertEqual(promotion_report["summary"]["promoted_candidate_count"], 1)
            self.assertIn(
                "OCR Candidate Promotion",
                promotion_md_path.read_text(encoding="utf-8"),
            )
            promoted = json.loads(promoted_ir_path.read_text(encoding="utf-8"))
            promoted_block = promoted["pages"][0]["blocks"][0]
            self.assertEqual(promoted_block["text"], "Figure overview text")
            self.assertEqual(promoted_block["meta"]["ocr_promotions"][0]["task_id"], image_task_id)
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
        self.assertEqual(events[0]["error_code"], "HTTP_SERVER_ERROR")
        self.assertEqual(events[0]["error_category"], "http_server")
        self.assertTrue(events[0]["error_retryable"])
        self.assertTrue(events[0]["will_retry"])
        self.assertEqual(events[1]["attempt_index"], 2)
        self.assertEqual(events[1]["status"], "success")
        self.assertEqual(events[1]["error_code"], "")
        self.assertFalse(events[1]["will_retry"])

    def test_http_retry_exhaustion_raises_structured_rate_limit_error(self) -> None:
        request = httpx.Request("POST", "https://example.test/chat")
        response = httpx.Response(429, request=request)

        def _op() -> str:
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)

        previous = os.environ.get("PDF_TRANSLATE_HTTP_RETRIES")
        os.environ["PDF_TRANSLATE_HTTP_RETRIES"] = "1"
        try:
            with self.assertRaises(PdfTranslateError) as caught:
                with capture_http_retry_events() as events:
                    call_with_http_retry(_op, context="unit-test")
        finally:
            if previous is None:
                os.environ.pop("PDF_TRANSLATE_HTTP_RETRIES", None)
            else:
                os.environ["PDF_TRANSLATE_HTTP_RETRIES"] = previous

        self.assertEqual(caught.exception.error_info.code, "HTTP_RATE_LIMIT")
        self.assertEqual(caught.exception.error_info.category, "rate_limit")
        self.assertTrue(caught.exception.error_info.retryable)
        self.assertEqual(events[0]["error_code"], "HTTP_RATE_LIMIT")
        self.assertEqual(events[0]["error_category"], "rate_limit")
        self.assertFalse(events[0]["will_retry"])

    def test_http_retry_timeout_maps_to_structured_error(self) -> None:
        request = httpx.Request("POST", "https://example.test/chat")

        def _op() -> str:
            raise httpx.ReadTimeout("slow upstream", request=request)

        previous = os.environ.get("PDF_TRANSLATE_HTTP_RETRIES")
        os.environ["PDF_TRANSLATE_HTTP_RETRIES"] = "1"
        try:
            with self.assertRaises(PdfTranslateError) as caught:
                with capture_http_retry_events() as events:
                    call_with_http_retry(_op, context="unit-test")
        finally:
            if previous is None:
                os.environ.pop("PDF_TRANSLATE_HTTP_RETRIES", None)
            else:
                os.environ["PDF_TRANSLATE_HTTP_RETRIES"] = previous

        self.assertEqual(caught.exception.error_info.code, "HTTP_TIMEOUT")
        self.assertEqual(caught.exception.error_info.category, "timeout")
        self.assertTrue(caught.exception.error_info.retryable)
        self.assertEqual(events[0]["error_code"], "HTTP_TIMEOUT")
        self.assertEqual(events[0]["error_category"], "timeout")

    def test_build_translator_missing_api_key_raises_structured_error(self) -> None:
        with self.assertRaises(PdfTranslateError) as caught:
            build_translator("deepseek", _test_app_config(deepseek_api_key=None))

        info = caught.exception.error_info
        self.assertEqual(info.code, "CONFIG_MISSING_API_KEY")
        self.assertEqual(info.category, "config")
        self.assertFalse(info.retryable)
        self.assertEqual(info.source, "translator:deepseek")

    def test_pdf_parse_exception_maps_to_structured_error(self) -> None:
        info = error_info_from_exception(fitz.FileDataError("bad pdf"), source="stage:document_ir")

        self.assertEqual(info.code, "PDF_PARSE_ERROR")
        self.assertEqual(info.category, "pdf_parse")
        self.assertFalse(info.retryable)
        self.assertEqual(info.source, "stage:document_ir")

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
                            "error_code": "HTTP_TIMEOUT",
                            "error_category": "timeout",
                            "error_retryable": True,
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
                            "error_code": "",
                            "error_category": "",
                            "error_retryable": False,
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
        self.assertEqual(metrics["summary"]["error_code_counts"]["HTTP_TIMEOUT"], 1)
        self.assertEqual(metrics["summary"]["error_category_counts"]["timeout"], 1)
        self.assertEqual(metrics["summary"]["stage_elapsed_ms"]["document_ir"], 12)
        self.assertEqual(metrics["chunks"][0]["http_attempt_count"], 2)
        self.assertEqual(metrics["chunks"][0]["error_code_counts"]["HTTP_TIMEOUT"], 1)
        self.assertEqual(metrics["chunks"][0]["http_retry_events"][0]["error_type"], "ReadTimeout")
        self.assertEqual(metrics["breakdowns"]["translator_counts"]["echo"], 1)
        self.assertEqual(metrics["breakdowns"]["skip_reasons"]["resume_completed"], 1)
        self.assertEqual(metrics["breakdowns"]["error_code_counts"]["HTTP_TIMEOUT"], 1)

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
                    "cross_page_relationship_count": 1,
                    "caption_cross_page_linked_count": 1,
                    "caption_cross_page_orphan_count": 0,
                    "footnote_cross_page_linked_count": 0,
                    "footnote_cross_page_orphan_count": 0,
                    "cross_page_parent_gap_max": 1,
                    "entity_candidate_count": 6,
                    "entity_unique_count": 5,
                    "entity_type_counts": {"model_or_dataset": 2, "person": 1},
                    "page_boundary_fragment_count": 2,
                    "page_boundary_stitch_candidate_count": 2,
                    "table_continuation_boundary_count": 1,
                    "continuation_kind_counts": {
                        "table_continuation": 1,
                        "hyphenated_word_continuation": 1,
                    },
                    "stitch_action_counts": {
                        "preserve_table_segments_together": 1,
                        "join_hyphenated_word_across_page_boundary": 1,
                    },
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
                    "source_formula_token_count": 5,
                    "missing_formula_token_count": 2,
                    "source_equation_label_count": 1,
                    "missing_equation_label_count": 1,
                    "source_table_count": 2,
                    "table_shape_error_count": 1,
                    "source_table_locked_token_count": 6,
                    "table_cell_token_error_count": 2,
                    "missing_table_locked_token_count": 3,
                    "structure_relation_check_count": 3,
                    "structure_relation_mismatch_count": 1,
                    "structure_relation_missing_anchor_count": 1,
                    "table_footnote_binding_check_count": 2,
                    "table_footnote_binding_mismatch_count": 1,
                    "table_footnote_binding_missing_cell_count": 2,
                    "issue_count": 6,
                    "issue_counts": {
                        "missing_numbers": 1,
                        "missing_entity_tokens": 2,
                        "table_cell_token_mismatch": 1,
                        "caption_or_footnote_relation_mismatch": 1,
                        "table_footnote_binding_mismatch": 1,
                    },
                    "severity_counts": {"high": 3, "medium": 3},
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
                    "table_continuation_boundary_count": 1,
                    "table_continuation_protected_count": 1,
                    "table_continuation_split_count": 0,
                    "table_continuation_co_located_count": 1,
                    "hyphenated_boundary_count": 1,
                    "hyphenated_protected_count": 1,
                    "hyphenated_split_count": 0,
                    "hyphenated_co_located_count": 1,
                    "budget_overflow_chunk_count": 1,
                    "budget_overflow_char_total": 160,
                    "structural_relation_protected_count": 2,
                    "budget_split_reason_counts": {"target_chars": 1, "end_of_document": 1},
                    "budget_pressure_counts": {"over_max": 1, "within_target": 1},
                },
            },
            chunk_strategy_comparison={
                "schema_version": "chunk-strategy-comparison-v1",
                "summary": {
                    "baseline_split_boundary_count": 2,
                    "active_split_boundary_count": 1,
                    "active_split_reduction_vs_baseline": 1,
                    "active_split_reduction_rate_vs_baseline": 0.5,
                    "baseline_table_continuation_split_count": 1,
                    "active_table_continuation_split_count": 0,
                    "active_table_continuation_split_reduction_vs_baseline": 1,
                    "active_table_continuation_split_reduction_rate_vs_baseline": 1.0,
                    "baseline_hyphenated_split_count": 1,
                    "active_hyphenated_split_count": 0,
                    "active_hyphenated_split_reduction_vs_baseline": 1,
                    "active_hyphenated_split_reduction_rate_vs_baseline": 1.0,
                },
            },
            structure_hints_manifest={
                "schema_version": "structure-hints-manifest-v1",
                "summary": {
                    "chunk_count": 2,
                    "structure_hint_chunk_count": 1,
                    "structure_hint_empty_chunk_count": 1,
                    "structure_hint_char_count": 240,
                    "structure_hint_avg_char_count": 120.0,
                    "structure_hint_max_char_count": 240,
                    "structure_hint_table_count": 2,
                    "structure_hint_continued_group_count": 1,
                    "structure_hint_merged_cell_candidate_count": 2,
                    "structure_hint_merged_cell_candidate_type_counts": {"colspan": 2},
                    "structure_hint_merged_cell_candidate_reason_counts": {
                        "single_cell_ragged_row": 1,
                        "empty_cell_right_of_nonempty_anchor": 1,
                    },
                    "structure_hint_footnote_binding_count": 1,
                    "structure_hint_relationship_count": 3,
                    "structure_hint_relationship_cross_page_count": 1,
                    "structure_hint_relationship_type_counts": {
                        "caption_for_table": 1,
                        "footnote_for_table": 1,
                        "footnote_for_block": 1,
                    },
                    "structure_hint_entity_count": 4,
                    "structure_hint_locked_token_count": 4,
                },
            },
            table_reconstruction={
                "schema_version": "table-reconstruction-v1",
                "summary": {
                    "table_count": 2,
                    "reconstructable_table_count": 1,
                    "low_confidence_table_count": 1,
                    "cell_count": 8,
                    "empty_cell_count": 2,
                    "numeric_cell_count": 3,
                    "numeric_token_count": 4,
                    "unit_token_count": 1,
                    "significance_token_count": 2,
                    "ragged_table_count": 1,
                    "ragged_row_count": 1,
                    "merged_cell_candidate_count": 2,
                    "merged_cell_candidate_type_counts": {"colspan": 2},
                    "merged_cell_candidate_reason_counts": {"single_cell_ragged_row": 1, "empty_cell_right_of_nonempty_anchor": 1},
                    "caption_linked_table_count": 1,
                    "footnote_linked_table_count": 1,
                    "table_footnote_binding_count": 2,
                    "table_footnote_cell_binding_count": 1,
                    "table_footnote_bound_cell_count": 2,
                    "table_footnote_unbound_count": 1,
                    "table_footnote_table_level_count": 0,
                    "continued_table_group_count": 2,
                    "continued_table_segment_count": 2,
                    "continued_table_merged_cell_candidate_count": 2,
                    "continued_table_reconstructable_group_count": 1,
                    "continued_table_merged_row_count": 5,
                    "table_chain_candidate_count": 2,
                    "table_chain_merged_count": 1,
                    "table_chain_reject_count": 1,
                    "table_chain_row_gain": 2,
                    "table_chain_warning_count": 1,
                    "table_chain_reject_reason_count": 1,
                    "table_chain_warning_reason_count": 1,
                    "table_chain_reject_reason_counts": {"header_mismatch_segment_1": 1},
                    "table_chain_reject_reason_category_counts": {"header_mismatch": 1},
                    "table_chain_warning_reason_counts": {"missing_header_for_segment_1": 1},
                    "table_chain_warning_reason_category_counts": {"missing_header": 1},
                    "table_reconstruction_ready_rate": 0.5,
                },
            },
            table_merged_cell_review={
                "schema_version": "table-merged-cell-review-v1",
                "summary": {
                    "candidate_review_count": 2,
                    "review_required_count": 2,
                    "pending_review_count": 2,
                    "visual_supported_count": 1,
                    "estimated_only_count": 1,
                    "missing_evidence_count": 1,
                    "human_confirmed_count": 0,
                    "rejected_count": 0,
                    "confirmation_status_counts": {"pending_review": 2},
                    "default_decision_counts": {
                        "needs_human_confirmation": 1,
                        "needs_visual_review": 1,
                    },
                    "human_decision_counts": {"pending": 2},
                    "candidate_status_counts": {"visually_supported": 1, "candidate": 1},
                    "visual_evidence_counts": {"visual_span_bbox": 1, "estimated_bbox": 1},
                    "bbox_evidence_counts": {"span_reported": 1, "estimated": 1},
                },
            },
            table_structure_publish={
                "schema_version": "table-structure-publish-v1",
                "summary": {
                    "confirmed": True,
                    "published": True,
                    "blocking_review_count": 0,
                    "applied_confirmed_count": 2,
                    "structure_patch_count": 2,
                    "structure_patch_applied_count": 2,
                    "structure_patch_table_count": 1,
                    "structure_patch_cell_count": 5,
                    "structure_patch_covered_cell_count": 3,
                    "structure_patch_rollback_available": True,
                    "structure_patch_operation_counts": {
                        "apply_confirmed_merged_cell_span": 2,
                    },
                    "structure_patch_span_type_counts": {"colspan": 1, "rowspan": 1},
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
                    "structured_contract_task_count": 2,
                    "table_context_task_count": 1,
                    "table_context_ready_task_count": 1,
                    "formula_context_task_count": 1,
                    "formula_context_ready_task_count": 1,
                    "scope_counts": {"region": 3, "page": 1},
                    "status_counts": {"pending_engine": 3, "blocked_missing_visual_evidence": 1},
                    "priority_counts": {"P0": 2, "P1": 2},
                    "recommended_engine_counts": {
                        "local_ocr": 2,
                        "local_table_ocr": 1,
                        "local_formula_ocr": 1,
                    },
                    "block_type_counts": {"image": 2, "table": 1, "formula": 1},
                    "structure_target_counts": {"image": 2, "table": 1, "formula": 1},
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
                "execution": {
                    "schema_version": "ocr-execution-v1",
                    "summary": {
                        "attempted_task_count": 3,
                        "succeeded_task_count": 2,
                        "failed_task_count": 1,
                        "skipped_task_count": 1,
                        "engine_available": True,
                        "status_counts": {"succeeded": 2, "failed": 1, "skipped": 1},
                    },
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
                    "table_context_writeback_count": 1,
                    "formula_context_writeback_count": 1,
                    "structured_result_writeback_count": 1,
                    "structured_result_field_counts": {
                        "structured_cells": 1,
                        "cell_bboxes": 1,
                        "merged_cell_candidates": 1,
                        "table_footnotes": 1,
                        "formula_latex": 1,
                        "formula_tokens": 1,
                        "equation_labels": 1,
                        "formula_confidence": 1,
                    },
                    "structured_result_item_counts": {
                        "structured_cells": 4,
                        "cell_bboxes": 2,
                        "merged_cell_candidates": 1,
                        "table_footnotes": 1,
                        "formula_latex": 1,
                        "formula_tokens": 5,
                        "equation_labels": 1,
                        "formula_confidence": 1,
                    },
                    "result_status_counts": {"succeeded": 3},
                    "accepted_engine_counts": {"local_ocr": 1, "local_table_ocr": 1},
                    "rejection_reason_counts": {"low_confidence": 1},
                },
            },
            ocr_candidate_qa={
                "schema_version": "ocr-candidate-qa-v1",
                "summary": {
                    "candidate_count": 3,
                    "promotable_candidate_count": 1,
                    "needs_review_candidate_count": 1,
                    "blocked_candidate_count": 1,
                    "candidate_text_char_count": 180,
                    "table_context_candidate_count": 1,
                    "formula_context_candidate_count": 1,
                    "structured_contract_candidate_count": 1,
                    "subtarget_candidate_count": 1,
                    "structured_result_candidate_count": 1,
                    "structured_result_field_counts": {
                        "structured_cells": 1,
                        "cell_bboxes": 1,
                        "merged_cell_candidates": 1,
                        "table_footnotes": 1,
                        "formula_latex": 1,
                        "formula_tokens": 1,
                        "equation_labels": 1,
                        "formula_confidence": 1,
                    },
                    "structured_result_item_counts": {
                        "structured_cells": 4,
                        "cell_bboxes": 2,
                        "merged_cell_candidates": 1,
                        "table_footnotes": 1,
                        "formula_latex": 1,
                        "formula_tokens": 5,
                        "equation_labels": 1,
                        "formula_confidence": 1,
                    },
                    "structured_cells_candidate_count": 1,
                    "cell_bboxes_candidate_count": 1,
                    "merged_cell_candidates_candidate_count": 1,
                    "table_footnotes_candidate_count": 1,
                    "formula_latex_candidate_count": 1,
                    "formula_tokens_candidate_count": 1,
                    "equation_labels_candidate_count": 1,
                    "formula_confidence_candidate_count": 1,
                    "structured_cell_count": 4,
                    "cell_bbox_count": 2,
                    "result_merged_cell_candidate_count": 1,
                    "result_table_footnote_count": 1,
                    "result_formula_latex_count": 1,
                    "result_formula_token_count": 5,
                    "result_equation_label_count": 1,
                    "structured_table_gate_counts": {"passed": 1, "needs_review": 1},
                    "structured_table_gate_issue_counts": {
                        "structured_table_missing_locked_tokens": 1,
                        "structured_table_missing_cell_bboxes": 1,
                    },
                    "structured_table_candidate_count": 2,
                    "structured_table_gate_passed_count": 1,
                    "structured_table_gate_review_count": 1,
                    "structured_table_gate_blocked_count": 0,
                    "structured_table_missing_locked_token_count": 1,
                    "structured_formula_gate_counts": {"passed": 1, "needs_review": 1},
                    "structured_formula_gate_issue_counts": {
                        "structured_formula_missing_equation_labels": 1,
                    },
                    "structured_formula_candidate_count": 2,
                    "structured_formula_gate_passed_count": 1,
                    "structured_formula_gate_review_count": 1,
                    "structured_formula_gate_blocked_count": 0,
                    "structured_formula_missing_locked_token_count": 0,
                    "structured_formula_token_count": 5,
                    "structured_formula_equation_label_count": 1,
                    "status_counts": {"candidate": 1, "needs_review": 1, "blocked": 1},
                    "issue_counts": {
                        "needs_table_structure_review": 1,
                        "structured_table_missing_locked_tokens": 1,
                        "structured_table_missing_cell_bboxes": 1,
                        "too_short": 1,
                    },
                },
            },
            ocr_candidate_promotion={
                "schema_version": "ocr-candidate-promotion-v1",
                "summary": {
                    "candidate_count": 3,
                    "eligible_candidate_count": 1,
                    "promoted_candidate_count": 1,
                    "skipped_candidate_count": 2,
                    "block_promotion_count": 1,
                    "page_promotion_count": 0,
                    "canonical_structure_promotion_count": 1,
                    "structured_table_promotion_count": 1,
                    "structured_formula_promotion_count": 0,
                    "promoted_text_char_count": 120,
                    "candidate_status_counts": {"candidate": 1, "needs_review": 1, "blocked": 1},
                    "skip_reason_counts": {"status_not_promotable": 2},
                },
            },
            repair_requests={
                "schema_version": "repair-requests-v1",
                "summary": {
                    "repair_request_count": 4,
                    "ready_for_translation_backend_count": 3,
                    "manual_review_request_count": 1,
                    "structure_patch_context_request_count": 1,
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
                    "structure_patch_context_count": 1,
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
                    "table_targeted_patch_count": 1,
                    "structure_patch_context_candidate_count": 1,
                    "applied_structure_patch_context_count": 1,
                    "strategy_counts": {"replace_markdown_table_by_evidence": 1, "manual_merge_required": 1},
                    "applied_strategy_counts": {"replace_markdown_table_by_evidence": 1},
                },
            },
            repair_patch_review={
                "schema_version": "repair-patch-review-v1",
                "summary": {
                    "patch_count": 1,
                    "auto_merge_safe_count": 1,
                    "effective_safe_count": 1,
                    "review_required_count": 0,
                    "publish_blocking_count": 0,
                    "human_reviewed_count": 0,
                    "table_patch_review_count": 1,
                    "structure_patch_review_count": 1,
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
            repair_effectiveness={
                "schema_version": "repair-effectiveness-v1",
                "summary": {
                    "before_issue_count": 6,
                    "after_issue_count": 3,
                    "issue_delta": 3,
                    "issue_reduction_rate": 0.5,
                    "resolved_issue_count": 4,
                    "persisted_issue_count": 2,
                    "new_issue_count": 1,
                    "improved_chunk_count": 2,
                    "regressed_chunk_count": 1,
                    "status": "improved_with_regressions",
                },
            },
            repair_publish={
                "schema_version": "repair-publish-v1",
                "summary": {
                    "confirmed": True,
                    "published": True,
                    "publish_status": "published_with_warnings",
                    "open_merge_issue_count": 1,
                },
            },
            repair_rollback={
                "schema_version": "repair-rollback-v1",
                "summary": {
                    "rollback_available": True,
                    "confirmed": True,
                    "rollback_applied": True,
                    "rollback_status": "rolled_back",
                    "rollback_matches_original": True,
                },
            },
            repair_formal_replace={
                "schema_version": "repair-formal-replace-v1",
                "summary": {
                    "replace_available": True,
                    "confirmed": True,
                    "replaced": True,
                    "replace_status": "replaced",
                    "formal_matches_published": True,
                    "rollback_available": True,
                },
            },
            repair_formal_rollback={
                "schema_version": "repair-formal-rollback-v1",
                "summary": {
                    "rollback_available": True,
                    "confirmed": True,
                    "rollback_applied": True,
                    "rollback_status": "rolled_back",
                    "formal_matches_backup": True,
                },
            },
            translated_pdf_report={
                "schema_version": "translated-pdf-report-v1",
                "summary": {
                    "generated": True,
                    "chunk_count": 2,
                    "page_count": 3,
                    "table_count": 1,
                    "qa_issue_count": 4,
                    "repair_item_count": 2,
                    "warning_count": 0,
                    "structure_context_chunk_count": 1,
                    "source_table_reference_count": 2,
                    "source_caption_reference_count": 1,
                    "source_footnote_reference_count": 1,
                    "source_footnote_cell_binding_count": 1,
                    "continued_table_group_reference_count": 1,
                    "structural_relation_reference_count": 2,
                    "table_structure_patch_reference_count": 2,
                    "table_structure_patch_covered_cell_reference_count": 3,
                    "table_structure_patch_rendered_count": 2,
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
                    "failed_event_count": 1,
                    "error_code_counts": {"HTTP_TIMEOUT": 1},
                    "error_category_counts": {"timeout": 1},
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
                    "error_code_counts": {"HTTP_TIMEOUT": 1},
                    "error_category_counts": {"timeout": 1},
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
        self.assertEqual(metrics["quality"]["structure_hint_chunk_count"], 1)
        self.assertEqual(metrics["quality"]["structure_hint_empty_chunk_count"], 1)
        self.assertEqual(metrics["quality"]["structure_hint_char_count"], 240)
        self.assertEqual(metrics["quality"]["structure_hint_avg_char_count"], 120.0)
        self.assertEqual(metrics["quality"]["structure_hint_max_char_count"], 240)
        self.assertEqual(metrics["quality"]["structure_hint_table_count"], 2)
        self.assertEqual(metrics["quality"]["structure_hint_continued_group_count"], 1)
        self.assertEqual(metrics["quality"]["structure_hint_merged_cell_candidate_count"], 2)
        self.assertEqual(metrics["quality"]["structure_hint_footnote_binding_count"], 1)
        self.assertEqual(metrics["quality"]["structure_hint_relationship_count"], 3)
        self.assertEqual(metrics["quality"]["structure_hint_relationship_cross_page_count"], 1)
        self.assertEqual(metrics["quality"]["structure_hint_entity_count"], 4)
        self.assertEqual(metrics["quality"]["structure_hint_locked_token_count"], 4)
        self.assertEqual(metrics["rates"]["structure_hint_chunk_rate"], 0.5)
        self.assertEqual(metrics["rates"]["structure_hint_table_per_chunk"], 2.0)
        self.assertEqual(metrics["rates"]["structure_hint_merged_cell_candidate_per_chunk"], 2.0)
        self.assertEqual(metrics["rates"]["structure_hint_footnote_binding_per_chunk"], 1.0)
        self.assertEqual(metrics["rates"]["structure_hint_relationship_per_chunk"], 3.0)
        self.assertEqual(metrics["rates"]["structure_hint_entity_per_chunk"], 4.0)
        self.assertEqual(metrics["rates"]["structure_hint_locked_token_per_chunk"], 4.0)
        self.assertEqual(metrics["breakdowns"]["structure_hint_merged_cell_candidate_type_counts"]["colspan"], 2)
        self.assertEqual(
            metrics["breakdowns"]["structure_hint_merged_cell_candidate_reason_counts"][
                "empty_cell_right_of_nonempty_anchor"
            ],
            1,
        )
        self.assertEqual(metrics["breakdowns"]["structure_hint_relationship_type_counts"]["caption_for_table"], 1)
        self.assertEqual(metrics["evidence_files"]["structure_hints_manifest"], "output/structure_hints_manifest.json")
        self.assertEqual(metrics["performance"]["total_elapsed_ms"], 1000)
        self.assertEqual(metrics["performance"]["translation_request_count"], 2)
        self.assertEqual(metrics["performance"]["http_attempt_count"], 3)
        self.assertEqual(metrics["performance"]["http_retry_count"], 1)
        self.assertEqual(metrics["performance"]["http_failed_attempt_count"], 1)
        self.assertEqual(metrics["performance"]["failed_event_count"], 1)
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
        self.assertEqual(metrics["breakdowns"]["error_code_counts"]["HTTP_TIMEOUT"], 1)
        self.assertEqual(metrics["breakdowns"]["error_category_counts"]["timeout"], 1)
        self.assertEqual(metrics["quality"]["ocr_candidate_page_count"], 2)
        self.assertEqual(metrics["quality"]["repair_item_count"], 4)
        self.assertEqual(metrics["quality"]["repair_request_count"], 4)
        self.assertEqual(metrics["quality"]["repair_backend_request_count"], 3)
        self.assertEqual(metrics["quality"]["repair_manual_request_count"], 1)
        self.assertEqual(metrics["quality"]["repair_request_structure_patch_context_count"], 1)
        self.assertEqual(metrics["quality"]["repair_executed_request_count"], 2)
        self.assertEqual(metrics["quality"]["repair_succeeded_count"], 1)
        self.assertEqual(metrics["quality"]["repair_failed_count"], 1)
        self.assertEqual(metrics["quality"]["repair_skipped_count"], 2)
        self.assertEqual(metrics["quality"]["repair_validation_checked_count"], 2)
        self.assertEqual(metrics["quality"]["repair_validation_passed_count"], 1)
        self.assertEqual(metrics["quality"]["repair_validation_failed_count"], 1)
        self.assertEqual(metrics["quality"]["repair_validation_missing_locked_token_count"], 1)
        self.assertEqual(metrics["quality"]["repair_validation_table_shape_passed_count"], 1)
        self.assertEqual(metrics["quality"]["repair_validation_structure_patch_context_count"], 1)
        self.assertEqual(metrics["quality"]["repair_merge_candidate_count"], 2)
        self.assertEqual(metrics["quality"]["repair_merge_applied_count"], 1)
        self.assertEqual(metrics["quality"]["repair_merge_patched_chunk_count"], 1)
        self.assertEqual(metrics["quality"]["repair_merge_manual_required_count"], 1)
        self.assertEqual(metrics["quality"]["repair_merge_table_targeted_patch_count"], 1)
        self.assertEqual(metrics["quality"]["repair_merge_structure_patch_context_candidate_count"], 1)
        self.assertEqual(metrics["quality"]["repair_merge_applied_structure_patch_context_count"], 1)
        self.assertEqual(metrics["quality"]["repair_patch_review_structure_patch_count"], 1)
        self.assertTrue(metrics["quality"]["repair_publish_confirmed"])
        self.assertTrue(metrics["quality"]["repair_publish_published"])
        self.assertEqual(metrics["quality"]["repair_publish_open_issue_count"], 1)
        self.assertTrue(metrics["quality"]["repair_rollback_available"])
        self.assertTrue(metrics["quality"]["repair_rollback_confirmed"])
        self.assertTrue(metrics["quality"]["repair_rollback_applied"])
        self.assertTrue(metrics["quality"]["repair_rollback_matches_original"])
        self.assertEqual(metrics["rates"]["repair_rollback_success_rate"], 1.0)
        self.assertEqual(metrics["breakdowns"]["repair_rollback_status_counts"]["rolled_back"], 1)
        self.assertTrue(metrics["quality"]["repair_formal_replace_available"])
        self.assertTrue(metrics["quality"]["repair_formal_replace_confirmed"])
        self.assertTrue(metrics["quality"]["repair_formal_replace_replaced"])
        self.assertTrue(metrics["quality"]["repair_formal_replace_matches_published"])
        self.assertTrue(metrics["quality"]["repair_formal_replace_rollback_available"])
        self.assertTrue(metrics["quality"]["repair_formal_rollback_available"])
        self.assertTrue(metrics["quality"]["repair_formal_rollback_confirmed"])
        self.assertTrue(metrics["quality"]["repair_formal_rollback_applied"])
        self.assertTrue(metrics["quality"]["repair_formal_rollback_matches_backup"])
        self.assertEqual(metrics["rates"]["repair_formal_replace_success_rate"], 1.0)
        self.assertEqual(metrics["rates"]["repair_formal_rollback_success_rate"], 1.0)
        self.assertEqual(metrics["breakdowns"]["repair_formal_replace_status_counts"]["replaced"], 1)
        self.assertEqual(metrics["breakdowns"]["repair_formal_rollback_status_counts"]["rolled_back"], 1)
        self.assertEqual(metrics["evidence_files"]["repair_formal_full"], "output/formal_full.md")
        self.assertEqual(
            metrics["evidence_files"]["repair_formal_backup_full"],
            "output/formal_full.before_repair.md",
        )
        self.assertEqual(metrics["quality"]["post_repair_issue_count"], 3)
        self.assertEqual(metrics["quality"]["post_repair_issue_delta"], 3)
        self.assertEqual(metrics["quality"]["post_repair_table_cell_token_error_count"], 1)
        self.assertEqual(metrics["quality"]["repair_effectiveness_before_issue_count"], 6)
        self.assertEqual(metrics["quality"]["repair_effectiveness_after_issue_count"], 3)
        self.assertEqual(metrics["quality"]["repair_effectiveness_issue_delta"], 3)
        self.assertEqual(metrics["quality"]["repair_effectiveness_resolved_issue_count"], 4)
        self.assertEqual(metrics["quality"]["repair_effectiveness_persisted_issue_count"], 2)
        self.assertEqual(metrics["quality"]["repair_effectiveness_new_issue_count"], 1)
        self.assertEqual(metrics["quality"]["repair_effectiveness_improved_chunk_count"], 2)
        self.assertEqual(metrics["quality"]["repair_effectiveness_regressed_chunk_count"], 1)
        self.assertEqual(metrics["quality"]["translation_structure_relation_check_count"], 3)
        self.assertEqual(metrics["quality"]["translation_structure_relation_mismatch_count"], 1)
        self.assertEqual(metrics["quality"]["translation_structure_relation_missing_anchor_count"], 1)
        self.assertEqual(metrics["quality"]["translation_table_footnote_binding_check_count"], 2)
        self.assertEqual(metrics["quality"]["translation_table_footnote_binding_mismatch_count"], 1)
        self.assertEqual(metrics["quality"]["translation_table_footnote_binding_missing_cell_count"], 2)
        self.assertEqual(metrics["rates"]["translation_structure_relation_mismatch_rate"], 0.3333)
        self.assertEqual(metrics["rates"]["translation_table_footnote_binding_mismatch_rate"], 0.5)
        self.assertTrue(metrics["quality"]["translated_pdf_generated"])
        self.assertEqual(metrics["quality"]["translated_pdf_page_count"], 3)
        self.assertEqual(metrics["quality"]["translated_pdf_chunk_count"], 2)
        self.assertEqual(metrics["quality"]["translated_pdf_table_count"], 1)
        self.assertEqual(metrics["quality"]["translated_pdf_qa_issue_count"], 4)
        self.assertEqual(metrics["quality"]["translated_pdf_repair_item_count"], 2)
        self.assertEqual(metrics["quality"]["translated_pdf_structure_context_chunk_count"], 1)
        self.assertEqual(metrics["quality"]["translated_pdf_source_table_reference_count"], 2)
        self.assertEqual(metrics["quality"]["translated_pdf_source_caption_reference_count"], 1)
        self.assertEqual(metrics["quality"]["translated_pdf_source_footnote_reference_count"], 1)
        self.assertEqual(metrics["quality"]["translated_pdf_source_footnote_cell_binding_count"], 1)
        self.assertEqual(metrics["quality"]["translated_pdf_continued_table_group_reference_count"], 1)
        self.assertEqual(metrics["quality"]["translated_pdf_structural_relation_reference_count"], 2)
        self.assertEqual(metrics["quality"]["translated_pdf_table_structure_patch_reference_count"], 2)
        self.assertEqual(metrics["quality"]["translated_pdf_table_structure_patch_covered_cell_reference_count"], 3)
        self.assertEqual(metrics["quality"]["translated_pdf_table_structure_patch_rendered_count"], 2)
        self.assertEqual(metrics["rates"]["translated_pdf_structure_context_chunk_rate"], 0.5)
        self.assertEqual(metrics["quality"]["table_shape_error_count"], 1)
        self.assertEqual(metrics["quality"]["table_cell_token_error_count"], 2)
        self.assertEqual(metrics["quality"]["missing_table_locked_token_count"], 3)
        self.assertEqual(metrics["quality"]["source_formula_token_count"], 5)
        self.assertEqual(metrics["quality"]["missing_formula_token_count"], 2)
        self.assertEqual(metrics["quality"]["source_equation_label_count"], 1)
        self.assertEqual(metrics["quality"]["missing_equation_label_count"], 1)
        self.assertEqual(metrics["rates"]["formula_token_missing_rate"], 0.4)
        self.assertEqual(metrics["rates"]["equation_label_missing_rate"], 1.0)
        self.assertEqual(metrics["quality"]["split_boundary_count"], 1)
        self.assertEqual(metrics["quality"]["protected_boundary_count"], 1)
        self.assertEqual(metrics["quality"]["page_boundary_stitch_candidate_count"], 2)
        self.assertEqual(metrics["quality"]["table_continuation_boundary_count"], 1)
        self.assertEqual(metrics["quality"]["table_continuation_protected_count"], 1)
        self.assertEqual(metrics["quality"]["table_continuation_split_count"], 0)
        self.assertEqual(metrics["quality"]["table_continuation_co_located_count"], 1)
        self.assertEqual(metrics["quality"]["hyphenated_boundary_count"], 1)
        self.assertEqual(metrics["quality"]["hyphenated_protected_count"], 1)
        self.assertEqual(metrics["quality"]["hyphenated_split_count"], 0)
        self.assertEqual(metrics["quality"]["hyphenated_co_located_count"], 1)
        self.assertEqual(metrics["quality"]["budget_overflow_chunk_count"], 1)
        self.assertEqual(metrics["quality"]["budget_overflow_char_total"], 160)
        self.assertEqual(metrics["quality"]["structural_relation_protected_count"], 2)
        self.assertEqual(metrics["quality"]["cross_page_relationship_count"], 1)
        self.assertEqual(metrics["quality"]["caption_cross_page_linked_count"], 1)
        self.assertEqual(metrics["quality"]["cross_page_parent_gap_max"], 1)
        self.assertEqual(metrics["quality"]["baseline_split_boundary_count"], 2)
        self.assertEqual(metrics["quality"]["active_split_reduction_vs_baseline"], 1)
        self.assertEqual(metrics["quality"]["baseline_table_continuation_split_count"], 1)
        self.assertEqual(metrics["quality"]["active_table_continuation_split_count"], 0)
        self.assertEqual(metrics["quality"]["active_table_continuation_split_reduction_vs_baseline"], 1)
        self.assertEqual(metrics["quality"]["baseline_hyphenated_split_count"], 1)
        self.assertEqual(metrics["quality"]["active_hyphenated_split_count"], 0)
        self.assertEqual(metrics["quality"]["active_hyphenated_split_reduction_vs_baseline"], 1)
        self.assertEqual(metrics["quality"]["reconstructable_table_count"], 1)
        self.assertEqual(metrics["quality"]["table_cell_count"], 8)
        self.assertEqual(metrics["quality"]["table_empty_cell_count"], 2)
        self.assertEqual(metrics["quality"]["table_ragged_table_count"], 1)
        self.assertEqual(metrics["quality"]["table_ragged_row_count"], 1)
        self.assertEqual(metrics["quality"]["table_merged_cell_candidate_count"], 2)
        self.assertEqual(metrics["quality"]["table_merged_cell_review_count"], 2)
        self.assertEqual(metrics["quality"]["table_merged_cell_review_required_count"], 2)
        self.assertEqual(metrics["quality"]["table_merged_cell_review_pending_count"], 2)
        self.assertEqual(metrics["quality"]["table_merged_cell_review_visual_supported_count"], 1)
        self.assertEqual(metrics["quality"]["table_merged_cell_review_estimated_only_count"], 1)
        self.assertEqual(metrics["quality"]["table_merged_cell_review_missing_evidence_count"], 1)
        self.assertEqual(metrics["quality"]["table_merged_cell_review_human_confirmed_count"], 0)
        self.assertEqual(metrics["quality"]["table_merged_cell_review_rejected_count"], 0)
        self.assertEqual(metrics["quality"]["table_merged_cell_review_human_reviewed_count"], 0)
        self.assertEqual(metrics["quality"]["table_merged_cell_review_needs_revision_count"], 0)
        self.assertTrue(metrics["quality"]["table_structure_publish_confirmed"])
        self.assertTrue(metrics["quality"]["table_structure_publish_published"])
        self.assertEqual(metrics["quality"]["table_structure_publish_applied_count"], 2)
        self.assertEqual(metrics["quality"]["table_structure_patch_count"], 2)
        self.assertEqual(metrics["quality"]["table_structure_patch_applied_count"], 2)
        self.assertEqual(metrics["quality"]["table_structure_patch_table_count"], 1)
        self.assertEqual(metrics["quality"]["table_structure_patch_cell_count"], 5)
        self.assertEqual(metrics["quality"]["table_structure_patch_covered_cell_count"], 3)
        self.assertTrue(metrics["quality"]["table_structure_patch_rollback_available"])
        self.assertEqual(metrics["quality"]["table_significance_token_count"], 2)
        self.assertEqual(metrics["quality"]["table_footnote_binding_count"], 2)
        self.assertEqual(metrics["quality"]["table_footnote_cell_binding_count"], 1)
        self.assertEqual(metrics["quality"]["table_footnote_bound_cell_count"], 2)
        self.assertEqual(metrics["quality"]["table_footnote_unbound_count"], 1)
        self.assertEqual(metrics["quality"]["table_footnote_table_level_count"], 0)
        self.assertEqual(metrics["quality"]["continued_table_group_count"], 2)
        self.assertEqual(metrics["quality"]["continued_table_segment_count"], 2)
        self.assertEqual(metrics["quality"]["continued_table_merged_cell_candidate_count"], 2)
        self.assertEqual(metrics["quality"]["continued_table_reconstructable_group_count"], 1)
        self.assertEqual(metrics["quality"]["continued_table_merged_row_count"], 5)
        self.assertEqual(metrics["quality"]["table_chain_candidate_count"], 2)
        self.assertEqual(metrics["quality"]["table_chain_merged_count"], 1)
        self.assertEqual(metrics["quality"]["table_chain_reject_count"], 1)
        self.assertEqual(metrics["quality"]["table_chain_row_gain"], 2)
        self.assertEqual(metrics["quality"]["table_chain_warning_count"], 1)
        self.assertEqual(metrics["quality"]["table_chain_reject_reason_count"], 1)
        self.assertEqual(metrics["quality"]["table_chain_warning_reason_count"], 1)
        self.assertEqual(metrics["rates"]["table_shape_error_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_cell_token_error_rate"], 0.6667)
        self.assertEqual(metrics["rates"]["table_locked_token_missing_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_reconstruction_ready_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_empty_cell_rate"], 0.25)
        self.assertEqual(metrics["rates"]["table_ragged_table_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_merged_cell_candidate_rate"], 1.0)
        self.assertEqual(metrics["rates"]["table_merged_cell_review_required_rate"], 1.0)
        self.assertEqual(metrics["rates"]["table_merged_cell_review_visual_supported_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_merged_cell_review_human_confirmed_rate"], 0.0)
        self.assertEqual(metrics["rates"]["table_merged_cell_review_human_reviewed_rate"], 0.0)
        self.assertEqual(metrics["rates"]["table_structure_patch_apply_rate"], 1.0)
        self.assertEqual(metrics["rates"]["table_structure_patch_per_confirmed_candidate"], 1.0)
        self.assertEqual(metrics["rates"]["continued_table_reconstruction_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_chain_merge_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_chain_reject_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_chain_reject_reason_per_rejected_chain"], 1.0)
        self.assertEqual(metrics["rates"]["table_chain_warning_reason_per_candidate_chain"], 1.0)
        self.assertEqual(metrics["rates"]["table_numeric_cell_rate"], 0.375)
        self.assertEqual(metrics["rates"]["table_caption_link_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_footnote_binding_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_footnote_cell_binding_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_footnote_unbound_rate"], 0.5)
        self.assertEqual(metrics["rates"]["split_boundary_rate"], 0.5)
        self.assertEqual(metrics["rates"]["protected_boundary_rate"], 0.5)
        self.assertEqual(metrics["rates"]["table_continuation_boundary_split_rate"], 0.0)
        self.assertEqual(metrics["rates"]["table_continuation_boundary_protected_rate"], 1.0)
        self.assertEqual(metrics["rates"]["hyphenated_boundary_split_rate"], 0.0)
        self.assertEqual(metrics["rates"]["hyphenated_boundary_protected_rate"], 1.0)
        self.assertEqual(metrics["rates"]["budget_overflow_chunk_rate"], 0.5)
        self.assertEqual(metrics["rates"]["active_split_reduction_rate_vs_baseline"], 0.5)
        self.assertEqual(metrics["rates"]["active_table_continuation_split_reduction_rate_vs_baseline"], 1.0)
        self.assertEqual(metrics["rates"]["active_hyphenated_split_reduction_rate_vs_baseline"], 1.0)
        self.assertEqual(metrics["rates"]["cross_page_relationship_rate"], 0.3333)
        self.assertEqual(metrics["rates"]["cross_page_parent_success_rate"], 1.0)
        self.assertEqual(metrics["rates"]["caption_cross_page_link_rate"], 0.5)
        self.assertEqual(metrics["rates"]["footnote_cross_page_link_rate"], 0.0)
        self.assertEqual(metrics["breakdowns"]["budget_split_reason_counts"]["target_chars"], 1)
        self.assertEqual(metrics["breakdowns"]["budget_pressure_counts"]["over_max"], 1)
        self.assertEqual(metrics["breakdowns"]["continuation_kind_counts"]["table_continuation"], 1)
        self.assertEqual(metrics["breakdowns"]["continuation_kind_counts"]["hyphenated_word_continuation"], 1)
        self.assertEqual(metrics["breakdowns"]["stitch_action_counts"]["preserve_table_segments_together"], 1)
        self.assertEqual(metrics["breakdowns"]["stitch_action_counts"]["join_hyphenated_word_across_page_boundary"], 1)
        self.assertEqual(metrics["breakdowns"]["table_merged_cell_candidate_type_counts"]["colspan"], 2)
        self.assertEqual(
            metrics["breakdowns"]["table_merged_cell_candidate_reason_counts"]["single_cell_ragged_row"],
            1,
        )
        self.assertEqual(
            metrics["breakdowns"]["table_merged_cell_review_default_decision_counts"][
                "needs_human_confirmation"
            ],
            1,
        )
        self.assertEqual(metrics["breakdowns"]["table_merged_cell_review_human_decision_counts"]["pending"], 2)
        self.assertEqual(
            metrics["breakdowns"]["table_merged_cell_review_visual_evidence_counts"]["visual_span_bbox"],
            1,
        )
        self.assertEqual(
            metrics["breakdowns"]["table_merged_cell_review_bbox_evidence_counts"]["estimated"],
            1,
        )
        self.assertEqual(
            metrics["breakdowns"]["table_structure_patch_operation_counts"][
                "apply_confirmed_merged_cell_span"
            ],
            2,
        )
        self.assertEqual(metrics["breakdowns"]["table_structure_patch_span_type_counts"]["colspan"], 1)
        self.assertEqual(metrics["breakdowns"]["table_structure_patch_span_type_counts"]["rowspan"], 1)
        self.assertEqual(
            metrics["breakdowns"]["table_chain_reject_reason_counts"]["header_mismatch_segment_1"],
            1,
        )
        self.assertEqual(metrics["breakdowns"]["table_chain_reject_reason_category_counts"]["header_mismatch"], 1)
        self.assertEqual(
            metrics["breakdowns"]["table_chain_warning_reason_counts"]["missing_header_for_segment_1"],
            1,
        )
        self.assertEqual(metrics["breakdowns"]["table_chain_warning_reason_category_counts"]["missing_header"], 1)
        self.assertEqual(metrics["rates"]["entity_missing_rate"], 0.25)
        self.assertEqual(metrics["rates"]["repair_item_per_chunk"], 2.0)
        self.assertEqual(metrics["rates"]["repair_request_ready_rate"], 0.75)
        self.assertEqual(metrics["rates"]["repair_execution_success_rate"], 0.5)
        self.assertEqual(metrics["rates"]["repair_validation_pass_rate"], 0.5)
        self.assertEqual(metrics["rates"]["repair_locked_token_pass_rate"], 0.8)
        self.assertEqual(metrics["rates"]["repair_table_shape_validation_pass_rate"], 0.5)
        self.assertEqual(metrics["rates"]["repair_merge_apply_rate"], 0.5)
        self.assertEqual(metrics["rates"]["repair_merge_table_targeted_patch_rate"], 1.0)
        self.assertEqual(metrics["rates"]["repair_publish_rate"], 1.0)
        self.assertEqual(metrics["breakdowns"]["repair_merge_strategy_counts"]["replace_markdown_table_by_evidence"], 1)
        self.assertEqual(metrics["breakdowns"]["repair_publish_status_counts"]["published_with_warnings"], 1)
        self.assertEqual(
            metrics["breakdowns"]["repair_effectiveness_status_counts"]["improved_with_regressions"],
            1,
        )
        self.assertEqual(
            metrics["breakdowns"]["repair_merge_applied_strategy_counts"]["replace_markdown_table_by_evidence"],
            1,
        )
        self.assertEqual(metrics["rates"]["post_repair_issue_reduction_rate"], 0.5)
        self.assertEqual(metrics["rates"]["repair_effectiveness_issue_reduction_rate"], 0.5)
        self.assertEqual(metrics["rates"]["relationship_warning_rate"], 0.3333)
        self.assertEqual(metrics["quality"]["vision_preview_page_count"], 2)
        self.assertEqual(metrics["quality"]["vision_region_crop_count"], 3)
        self.assertEqual(metrics["quality"]["ocr_task_count"], 4)
        self.assertEqual(metrics["quality"]["ocr_region_task_count"], 3)
        self.assertEqual(metrics["quality"]["ocr_page_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_ready_task_count"], 3)
        self.assertEqual(metrics["quality"]["ocr_blocked_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_vlm_fallback_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_contract_task_count"], 2)
        self.assertEqual(metrics["quality"]["ocr_table_context_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_table_context_ready_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_formula_context_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_formula_context_ready_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_result_payload_count"], 3)
        self.assertEqual(metrics["quality"]["ocr_invalid_result_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_executor_attempted_task_count"], 3)
        self.assertEqual(metrics["quality"]["ocr_executor_succeeded_task_count"], 2)
        self.assertEqual(metrics["quality"]["ocr_executor_failed_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_executor_skipped_task_count"], 1)
        self.assertTrue(metrics["quality"]["ocr_executor_available"])
        self.assertEqual(metrics["quality"]["ocr_result_count"], 3)
        self.assertEqual(metrics["quality"]["ocr_accepted_result_count"], 2)
        self.assertEqual(metrics["quality"]["ocr_rejected_result_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_pending_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_missing_result_task_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_block_writeback_count"], 2)
        self.assertEqual(metrics["quality"]["ocr_page_writeback_count"], 0)
        self.assertEqual(metrics["quality"]["ocr_table_context_writeback_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_formula_context_writeback_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_result_writeback_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_candidate_qa_count"], 3)
        self.assertEqual(metrics["quality"]["ocr_candidate_promotable_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_candidate_needs_review_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_candidate_blocked_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_candidate_text_char_count"], 180)
        self.assertEqual(metrics["quality"]["ocr_table_context_candidate_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_formula_context_candidate_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_contract_candidate_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_subtarget_candidate_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_result_candidate_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_cells_candidate_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_cell_bboxes_candidate_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_merged_cell_candidates_candidate_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_table_footnotes_candidate_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_formula_latex_candidate_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_formula_tokens_candidate_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_equation_labels_candidate_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_cell_count"], 4)
        self.assertEqual(metrics["quality"]["ocr_cell_bbox_count"], 2)
        self.assertEqual(metrics["quality"]["ocr_result_merged_cell_candidate_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_result_table_footnote_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_result_formula_token_count"], 5)
        self.assertEqual(metrics["quality"]["ocr_result_equation_label_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_table_candidate_count"], 2)
        self.assertEqual(metrics["quality"]["ocr_structured_table_gate_passed_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_table_gate_review_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_table_gate_blocked_count"], 0)
        self.assertEqual(metrics["quality"]["ocr_structured_table_missing_locked_token_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_table_row_col_mismatch_count"], 0)
        self.assertEqual(metrics["quality"]["ocr_structured_table_missing_cell_bboxes_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_formula_candidate_count"], 2)
        self.assertEqual(metrics["quality"]["ocr_structured_formula_gate_passed_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_formula_gate_review_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_formula_gate_blocked_count"], 0)
        self.assertEqual(metrics["quality"]["ocr_structured_formula_token_count"], 5)
        self.assertEqual(metrics["quality"]["ocr_structured_formula_equation_label_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_formula_missing_equation_label_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_candidate_promotion_eligible_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_candidate_promoted_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_candidate_promotion_skipped_count"], 2)
        self.assertEqual(metrics["quality"]["ocr_candidate_block_promotion_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_candidate_page_promotion_count"], 0)
        self.assertEqual(metrics["quality"]["ocr_canonical_structure_promotion_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_table_promotion_count"], 1)
        self.assertEqual(metrics["quality"]["ocr_structured_formula_promotion_count"], 0)
        self.assertEqual(metrics["quality"]["ocr_candidate_promoted_text_char_count"], 120)
        self.assertEqual(metrics["rates"]["vision_preview_page_rate"], 0.5)
        self.assertEqual(metrics["rates"]["vision_region_crop_per_routed_page"], 1.5)
        self.assertEqual(metrics["rates"]["ocr_task_per_routed_page"], 2.0)
        self.assertEqual(metrics["rates"]["ocr_region_task_rate"], 0.75)
        self.assertEqual(metrics["rates"]["ocr_ready_task_rate"], 0.75)
        self.assertEqual(metrics["rates"]["ocr_structured_contract_task_rate"], 0.5)
        self.assertEqual(metrics["rates"]["ocr_table_context_task_rate"], 0.25)
        self.assertEqual(metrics["rates"]["ocr_table_context_ready_rate"], 1.0)
        self.assertEqual(metrics["rates"]["ocr_formula_context_task_rate"], 0.25)
        self.assertEqual(metrics["rates"]["ocr_formula_context_ready_rate"], 1.0)
        self.assertEqual(metrics["rates"]["ocr_result_payload_valid_rate"], 0.75)
        self.assertEqual(metrics["rates"]["ocr_executor_success_rate"], 0.6667)
        self.assertEqual(metrics["rates"]["ocr_task_result_coverage_rate"], 0.75)
        self.assertEqual(metrics["rates"]["ocr_result_acceptance_rate"], 0.6667)
        self.assertEqual(metrics["rates"]["ocr_writeback_apply_rate"], 0.5)
        self.assertEqual(metrics["rates"]["ocr_structured_result_writeback_rate"], 0.5)
        self.assertEqual(metrics["rates"]["ocr_structured_result_candidate_rate"], 0.3333)
        self.assertEqual(metrics["rates"]["ocr_structured_cells_candidate_rate"], 0.3333)
        self.assertEqual(metrics["rates"]["ocr_cell_bboxes_candidate_rate"], 0.3333)
        self.assertEqual(metrics["rates"]["ocr_structured_table_gate_pass_rate"], 0.5)
        self.assertEqual(metrics["rates"]["ocr_structured_table_gate_review_rate"], 0.5)
        self.assertEqual(metrics["rates"]["ocr_structured_table_structure_review_rate"], 0.5)
        self.assertEqual(metrics["rates"]["ocr_structured_table_row_col_match_rate"], 1.0)
        self.assertEqual(metrics["rates"]["ocr_table_cell_bbox_coverage_rate"], 0.5)
        self.assertEqual(metrics["rates"]["ocr_structured_formula_gate_pass_rate"], 0.5)
        self.assertEqual(metrics["rates"]["ocr_structured_formula_gate_review_rate"], 0.5)
        self.assertEqual(metrics["rates"]["ocr_formula_token_per_candidate"], 2.5)
        self.assertEqual(metrics["rates"]["ocr_formula_equation_label_per_candidate"], 0.5)
        self.assertEqual(metrics["rates"]["ocr_candidate_promotable_rate"], 0.3333)
        self.assertEqual(metrics["rates"]["ocr_candidate_blocked_rate"], 0.3333)
        self.assertEqual(metrics["rates"]["ocr_candidate_promotion_rate"], 0.3333)
        self.assertEqual(metrics["rates"]["ocr_canonical_structure_promotion_rate"], 1.0)
        self.assertEqual(metrics["rates"]["ocr_structured_table_promotion_rate"], 1.0)
        self.assertEqual(metrics["rates"]["ocr_structured_formula_promotion_rate"], 0.0)
        self.assertEqual(metrics["rates"]["ocr_candidate_eligible_promotion_rate"], 1.0)
        self.assertEqual(metrics["breakdowns"]["vision_action_counts"]["local_ocr"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_task_engine_counts"]["local_table_ocr"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_task_structure_target_counts"]["table"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_result_payload_engine_counts"]["vlm_fallback"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_execution_status_counts"]["failed"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_writeback_engine_counts"]["local_table_ocr"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_writeback_rejection_counts"]["low_confidence"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_writeback_structured_result_field_counts"]["structured_cells"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_candidate_status_counts"]["needs_review"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_candidate_issue_counts"]["too_short"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_candidate_structured_result_field_counts"]["cell_bboxes"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_candidate_structured_table_gate_counts"]["passed"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_candidate_structured_formula_gate_counts"]["passed"], 1)
        self.assertEqual(
            metrics["breakdowns"]["ocr_candidate_structured_table_gate_issue_counts"][
                "structured_table_missing_locked_tokens"
            ],
            1,
        )
        self.assertEqual(
            metrics["breakdowns"]["ocr_candidate_structured_formula_gate_issue_counts"][
                "structured_formula_missing_equation_labels"
            ],
            1,
        )
        self.assertEqual(metrics["breakdowns"]["ocr_candidate_promotion_status_counts"]["candidate"], 1)
        self.assertEqual(metrics["breakdowns"]["ocr_candidate_promotion_skip_counts"]["status_not_promotable"], 2)
        self.assertEqual(metrics["breakdowns"]["stage_elapsed_ms"]["document_ir"], 50)
        self.assertEqual(metrics["breakdowns"]["translator_counts"]["echo"], 2)
        self.assertEqual(metrics["evidence_files"]["translation_qa"], "output/qa_report.json")
        self.assertEqual(
            metrics["evidence_files"]["table_merged_cell_review"],
            "output/table_merged_cell_review.json",
        )
        self.assertEqual(metrics["evidence_files"]["ocr_tasks"], "output/ocr_tasks.json")
        self.assertEqual(metrics["evidence_files"]["ocr_results"], "output/ocr_results.json")
        self.assertEqual(metrics["evidence_files"]["ocr_writeback"], "output/ocr_writeback.json")
        self.assertEqual(metrics["evidence_files"]["ocr_candidate_qa"], "output/ocr_candidate_qa.json")
        self.assertEqual(metrics["evidence_files"]["ocr_candidate_promotion"], "output/ocr_candidate_promotion.json")
        self.assertEqual(metrics["evidence_files"]["document_ir_ocr"], "output/document_ir_ocr.json")
        self.assertEqual(metrics["evidence_files"]["document_ir_promoted"], "output/document_ir_promoted.json")
        self.assertEqual(metrics["evidence_files"]["repair_requests"], "output/repair_requests.json")
        self.assertEqual(metrics["evidence_files"]["repair_results"], "output/repair_results.json")
        self.assertEqual(metrics["evidence_files"]["repair_validation"], "output/repair_validation.json")
        self.assertEqual(metrics["evidence_files"]["repair_merge"], "output/repair_merge.json")
        self.assertEqual(metrics["evidence_files"]["repair_publish"], "output/repair_publish.json")
        self.assertEqual(metrics["evidence_files"]["repair_published_full"], "output/published_full.md")
        self.assertEqual(metrics["evidence_files"]["repair_rollback"], "output/repair_rollback.json")
        self.assertEqual(metrics["evidence_files"]["repair_rollback_full"], "output/rollback_full.md")
        self.assertEqual(metrics["evidence_files"]["repair_merge_qa"], "output/repair_merge_qa.json")
        self.assertEqual(metrics["evidence_files"]["repair_effectiveness"], "output/repair_effectiveness.json")
        self.assertEqual(metrics["evidence_files"]["run_metrics"], "output/run_metrics.json")
        self.assertEqual(metrics["evidence_files"]["run_log"], "output/run_log.jsonl")
        self.assertEqual(metrics["evidence_files"]["cost_estimate"], "output/cost_estimate.json")
        self.assertEqual(metrics["evidence_files"]["translated_pdf"], "output/translated_full.pdf")
        self.assertEqual(metrics["evidence_files"]["translated_pdf_report"], "output/translated_pdf_report.json")

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

    def test_memory_store_applies_glossary_review_decisions(self) -> None:
        root = Path.cwd() / "test-output" / "glossary-review-decision-memory"
        if root.exists():
            shutil.rmtree(root)
        try:
            mem = MemoryStore(root / "memory")
            mem.ensure_files()
            mem.merge_glossary_terms_from_survey(
                [{"en": "Accuracy", "zh": "准确率"}],
                first_page_1based=1,
            )
            mem.merge_glossary_terms_from_survey(
                [{"en": "accuracy", "zh": "精度"}],
                first_page_1based=2,
                source="survey",
            )
            conflict = next(
                item
                for item in mem.load_pending_review()["items"]
                if item["type"] == "glossary_conflict" and item["candidate_zh"] == "精度"
            )

            with self.assertRaisesRegex(ValueError, "candidate_zh must not be empty"):
                mem.apply_glossary_review_decision(
                    conflict["dedupe_key"],
                    "confirm_candidate",
                    candidate_zh=" ",
                )

            reviewed = mem.apply_glossary_review_decision(
                conflict["dedupe_key"],
                "confirm_candidate",
                reviewer="导师",
                reviewed_at="2026-07-06T12:00:00+00:00",
                comment="按导师术语表确认",
                confidence=0.93,
                section_scope="Methods",
                candidate_zh="分类准确率",
            )

            self.assertEqual(reviewed["status"], "confirmed")
            self.assertEqual(reviewed["candidate_zh"], "分类准确率")
            self.assertEqual(reviewed["confirmed_zh"], "分类准确率")
            self.assertEqual(reviewed["original_candidate_zh"], "精度")
            self.assertEqual(reviewed["edited_candidate_zh"], "分类准确率")
            self.assertEqual(reviewed["reviewed_by"], "导师")
            self.assertEqual(reviewed["confidence"], 0.93)
            self.assertEqual(reviewed["section_scope"], "Methods")
            glossary = mem.load_glossary()
            active_terms = [
                term for term in glossary["terms"] if str(term.get("status") or "").lower() != "rejected"
            ]
            self.assertEqual(len(active_terms), 1)
            self.assertEqual(active_terms[0]["en"], "Accuracy")
            self.assertEqual(active_terms[0]["zh"], "分类准确率")
            self.assertEqual(active_terms[0]["status"], "confirmed")
            self.assertEqual(active_terms[0]["original_candidate_zh"], "精度")
            self.assertEqual(active_terms[0]["edited_candidate_zh"], "分类准确率")
            self.assertEqual(active_terms[0]["confidence"], 0.93)
            self.assertEqual(active_terms[0]["section_scope"], "Methods")
            self.assertIn("Accuracy → 分类准确率", mem.glossary_snippet_for_pages(1, 2))
            self.assertNotIn("Accuracy → 准确率", mem.glossary_snippet_for_pages(1, 2))
            self.assertIn(
                "Accuracy → 分类准确率",
                mem.glossary_snippet_for_pages(1, 2, section_scope="2 Methods"),
            )
            self.assertNotIn(
                "Accuracy → 分类准确率",
                mem.glossary_snippet_for_pages(1, 2, section_scope="3 Results"),
            )

            glossary_data = mem.load_glossary()
            glossary_data["terms"].append(
                {
                    "en": "F1 score",
                    "zh": "F1 值",
                    "first_page": 1,
                    "status": "confirmed",
                    "section_scope": "structure:table",
                }
            )
            glossary_data["terms"].append(
                {
                    "en": "Ablation",
                    "zh": "消融实验",
                    "first_page": 1,
                    "status": "confirmed",
                    "section_scope": "block:p1-b0001",
                }
            )
            mem.glossary_path.write_text(
                json.dumps(glossary_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.assertIn(
                "F1 score → F1 值",
                mem.glossary_snippet_for_pages(1, 1, structure_types=["table"]),
            )
            self.assertNotIn(
                "F1 score → F1 值",
                mem.glossary_snippet_for_pages(1, 1, structure_types=["paragraph"]),
            )
            self.assertIn(
                "Ablation → 消融实验",
                mem.glossary_snippet_for_pages(1, 1, block_ids=["p1-b0001"]),
            )
            self.assertNotIn(
                "Ablation → 消融实验",
                mem.glossary_snippet_for_pages(1, 1, block_ids=["p1-b0002"]),
            )

            with self.assertRaisesRegex(ValueError, "already been reviewed"):
                mem.apply_glossary_review_decision(
                    conflict["dedupe_key"],
                    "confirm_candidate",
                )

            chunk_dir = root / "chunks"
            chunk_dir.mkdir(parents=True)
            chunk = TextChunk(
                chunk_id="c0000",
                pages_0based=[0],
                text="Accuracy improves under domain shift.",
                link_count=0,
                image_count=0,
            )
            (chunk_dir / "c0000.md").write_text(
                "---\n{}\n---\n\n分类准确率在领域偏移下提升。\n",
                encoding="utf-8",
            )
            report = build_translation_qa(
                [chunk],
                chunk_dir,
                glossary=glossary,
                pending_review=mem.load_pending_review(),
            )
            issue_types = {issue["type"] for issue in report["chunks"][0]["issues"]}
            self.assertNotIn("glossary_translation_conflict", issue_types)
            self.assertNotIn("missing_glossary_terms", issue_types)
            self.assertEqual(report["summary"]["glossary_conflict_count"], 0)

            missing_chunk = TextChunk(
                chunk_id="c0001",
                pages_0based=[0],
                text="Accuracy improves under domain shift.",
                link_count=0,
                image_count=0,
            )
            (chunk_dir / "c0001.md").write_text(
                "---\n{}\n---\n\n准确率在领域偏移下提升。\n",
                encoding="utf-8",
            )
            missing_report = build_translation_qa(
                [missing_chunk],
                chunk_dir,
                glossary=glossary,
                pending_review=mem.load_pending_review(),
            )
            missing_issue = next(
                issue
                for issue in missing_report["chunks"][0]["issues"]
                if issue["type"] == "missing_glossary_terms"
            )
            self.assertEqual(missing_issue["terms"][0]["expected_zh"], "分类准确率")
            self.assertEqual(missing_issue["terms"][0]["confidence"], 0.93)
            self.assertEqual(missing_issue["terms"][0]["section_scope"], "Methods")

            mem.merge_glossary_terms_from_survey(
                [{"en": "Recall", "zh": "召回率"}],
                first_page_1based=3,
            )
            mem.merge_glossary_terms_from_survey(
                [{"en": "Recall", "zh": "查全率"}],
                first_page_1based=3,
                source="survey",
            )
            recall_conflict = next(
                item
                for item in mem.load_pending_review()["items"]
                if item["type"] == "glossary_conflict" and item["en"] == "Recall"
            )
            rejected = mem.apply_glossary_review_decision(
                recall_conflict["dedupe_key"],
                "reject_candidate",
                reviewer="导师",
                reviewed_at="2026-07-06T12:30:00+00:00",
                comment="保持原术语",
            )
            self.assertEqual(rejected["status"], "rejected")
            self.assertIn("Recall → 召回率", mem.glossary_snippet_for_pages(3, 3))
            self.assertNotIn("查全率", mem.glossary_snippet_for_pages(3, 3))
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_glossary_retranslation_plan_detects_stale_scoped_chunks(self) -> None:
        root = Path.cwd() / "test-output" / "glossary-retranslation-plan"
        if root.exists():
            shutil.rmtree(root)
        try:
            mem = MemoryStore(root / "memory")
            mem.ensure_files()
            mem.merge_glossary_terms_from_survey(
                [{"en": "Accuracy", "zh": "准确率"}],
                first_page_1based=1,
            )
            mem.merge_glossary_terms_from_survey(
                [{"en": "Accuracy", "zh": "精度"}],
                first_page_1based=1,
            )
            conflict = next(
                item
                for item in mem.load_pending_review()["items"]
                if item["type"] == "glossary_conflict"
            )
            mem.apply_glossary_review_decision(
                conflict["dedupe_key"],
                "confirm_candidate",
                candidate_zh="分类准确率",
                section_scope="Methods",
                reviewer="导师",
            )

            out = root / "output"
            chunk_dir = out / "chunks"
            chunk_dir.mkdir(parents=True)
            (out / "chunks_manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "chunk_id": "c0000",
                            "pages_1based": [1],
                            "text": "Accuracy improves under domain shift.",
                            "section_scopes": ["2 Methods"],
                            "block_ids": ["p1-b0001"],
                            "block_types": {"paragraph": 1},
                        },
                        {
                            "chunk_id": "c0001",
                            "pages_1based": [2],
                            "text": "Accuracy is reported in the ablation.",
                            "section_scopes": ["3 Results"],
                            "block_ids": ["p2-b0001"],
                            "block_types": {"paragraph": 1},
                        },
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (chunk_dir / "c0000.md").write_text(
                "---\n{}\n---\n\n准确率在领域偏移下提升。\n",
                encoding="utf-8",
            )
            (chunk_dir / "c0001.md").write_text(
                "---\n{}\n---\n\n准确率在消融实验中报告。\n",
                encoding="utf-8",
            )

            plan = write_glossary_retranslation_plan(out, root / "memory")

            self.assertEqual(plan["schema_version"], "glossary-retranslation-plan-v1")
            self.assertEqual(plan["summary"]["status"], "needs_retranslation")
            self.assertEqual(plan["summary"]["confirmed_review_count"], 1)
            self.assertEqual(plan["summary"]["matched_chunk_count"], 1)
            self.assertEqual(plan["summary"]["stale_chunk_count"], 1)
            self.assertEqual(plan["terms"][0]["matched_chunk_ids"], ["c0000"])
            self.assertEqual(plan["terms"][0]["stale_chunk_ids"], ["c0000"])
            self.assertEqual(plan["chunks"][0]["recommended_action"], "retranslate_chunk")
            self.assertIn("contains_previous_translation", plan["chunks"][0]["stale_reasons"])
            self.assertIn("missing_confirmed_translation", plan["chunks"][0]["stale_reasons"])
            self.assertTrue((out / "glossary_retranslation_plan.json").is_file())
            md = (out / "glossary_retranslation_plan.md").read_text(encoding="utf-8")
            self.assertIn("术语确认重译计划", md)
            self.assertIn("c0000", md)
            self.assertNotIn("c0001 |", md)
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_glossary_retranslation_plan_reads_source_text_path(self) -> None:
        root = Path.cwd() / "test-output" / "glossary-retranslation-source-path"
        if root.exists():
            shutil.rmtree(root)
        try:
            mem = MemoryStore(root / "memory")
            mem.ensure_files()
            mem.merge_glossary_terms_from_survey(
                [{"en": "Accuracy", "zh": "准确率"}],
                first_page_1based=1,
            )
            mem.merge_glossary_terms_from_survey(
                [{"en": "Accuracy", "zh": "精度"}],
                first_page_1based=1,
            )
            conflict = next(
                item
                for item in mem.load_pending_review()["items"]
                if item["type"] == "glossary_conflict"
            )
            mem.apply_glossary_review_decision(
                conflict["dedupe_key"],
                "confirm_candidate",
                candidate_zh="分类准确率",
                reviewer="导师",
            )

            out = root / "output"
            chunk_dir = out / "chunks"
            source_dir = out / "source_chunks"
            chunk_dir.mkdir(parents=True)
            source_dir.mkdir(parents=True)
            (source_dir / "c0000.txt").write_text(
                "Accuracy improves under domain shift.",
                encoding="utf-8",
            )
            (out / "chunks_manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "chunk_id": "c0000",
                            "pages_1based": [1, 1],
                            "source_text_path": "output/source_chunks/c0000.txt",
                            "section_scopes": [],
                            "block_ids": [],
                            "block_types": {"paragraph": 1},
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (chunk_dir / "c0000.md").write_text(
                "---\n{}\n---\n\n准确率在领域偏移下提升。\n",
                encoding="utf-8",
            )

            plan = write_glossary_retranslation_plan(out, root / "memory")

            self.assertEqual(plan["summary"]["status"], "needs_retranslation")
            self.assertEqual(plan["summary"]["matched_chunk_count"], 1)
            self.assertEqual(plan["chunks"][0]["source_text_path"], "output/source_chunks/c0000.txt")
            self.assertEqual(plan["chunks"][0]["recommended_action"], "retranslate_chunk")
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_glossary_retranslation_execution_writes_candidate_artifacts(self) -> None:
        root = Path.cwd() / "test-output" / "glossary-retranslation-execution"
        if root.exists():
            shutil.rmtree(root)

        class FakeGlossaryTranslator:
            name = "fake_glossary"

            def __init__(self) -> None:
                self.requests = []

            def translate(self, req) -> str:
                self.requests.append(req)
                return "分类准确率在领域偏移下提升。\n\n| 指标 | 值 |\n| --- | --- |\n| Accuracy | 0.91 |"

        try:
            mem = MemoryStore(root / "memory")
            mem.ensure_files()
            mem.merge_glossary_terms_from_survey(
                [{"en": "Accuracy", "zh": "准确率"}],
                first_page_1based=1,
            )
            mem.merge_glossary_terms_from_survey(
                [{"en": "Accuracy", "zh": "精度"}],
                first_page_1based=1,
            )
            conflict = next(
                item
                for item in mem.load_pending_review()["items"]
                if item["type"] == "glossary_conflict"
            )
            mem.apply_glossary_review_decision(
                conflict["dedupe_key"],
                "confirm_candidate",
                candidate_zh="分类准确率",
                reviewer="导师",
            )

            out = root / "output"
            chunk_dir = out / "chunks"
            source_dir = out / "source_chunks"
            chunk_dir.mkdir(parents=True)
            source_dir.mkdir(parents=True)
            (source_dir / "c0000.txt").write_text(
                "Accuracy improves under domain shift.",
                encoding="utf-8",
            )
            (source_dir / "c0001.txt").write_text(
                "Recall is reported in the ablation.",
                encoding="utf-8",
            )
            (out / "chunks_manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "chunk_id": "c0000",
                            "pages_1based": [1, 1],
                            "source_text_path": "output/source_chunks/c0000.txt",
                            "section_scopes": [],
                            "block_ids": ["p1-b0001"],
                            "block_types": {"paragraph": 1, "table": 1},
                            "link_count": 0,
                            "image_count": 0,
                        },
                        {
                            "chunk_id": "c0001",
                            "pages_1based": [2, 2],
                            "source_text_path": "output/source_chunks/c0001.txt",
                            "section_scopes": [],
                            "block_ids": ["p2-b0001"],
                            "block_types": {"paragraph": 1},
                            "link_count": 0,
                            "image_count": 0,
                        },
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (out / "structure_hints_manifest.json").write_text(
                json.dumps(
                    {
                        "chunks": [
                            {
                                "chunk_id": "c0000",
                                "hint_text": "表格 p1-b0001：保持 Markdown 表格。",
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (chunk_dir / "c0000.md").write_text(
                "---\n{}\n---\n\n准确率在领域偏移下提升。\n",
                encoding="utf-8",
            )
            (chunk_dir / "c0001.md").write_text(
                "---\n{}\n---\n\n召回率在消融实验中报告。\n",
                encoding="utf-8",
            )
            translator = FakeGlossaryTranslator()

            result = execute_glossary_retranslation(
                out,
                root / "memory",
                translator,
                backend="fake",
            )

            self.assertEqual(result["schema_version"], "glossary-retranslation-execution-v1")
            self.assertEqual(result["summary"]["status"], "executed")
            self.assertEqual(result["summary"]["executed_chunk_count"], 1)
            self.assertEqual(result["chunks"][0]["confirmed_term_present_count"], 1)
            self.assertIn("分类准确率", (out / "glossary_retranslated_chunks" / "c0000.md").read_text(encoding="utf-8"))
            self.assertIn("准确率在领域偏移下提升", (chunk_dir / "c0000.md").read_text(encoding="utf-8"))
            full = (out / "glossary_retranslated_full.md").read_text(encoding="utf-8")
            self.assertIn("分类准确率在领域偏移下提升", full)
            self.assertIn("召回率在消融实验中报告", full)
            self.assertTrue((out / "glossary_retranslation_result.json").is_file())
            self.assertTrue((out / "glossary_retranslation_result.md").is_file())
            self.assertEqual(len(translator.requests), 1)
            self.assertIn("Accuracy → 分类准确率", translator.requests[0].glossary_excerpt)
            self.assertIn("保持 Markdown 表格", translator.requests[0].structure_hints)
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_glossary_retranslation_publish_and_rollback_preserve_baselines(self) -> None:
        root = Path.cwd() / "test-output" / "glossary-retranslation-publish"
        if root.exists():
            shutil.rmtree(root)
        try:
            out = root / "output"
            out.mkdir(parents=True)
            original = out / "translated_full.md"
            candidate = out / "glossary_retranslated_full.md"
            original.write_text("original translation", encoding="utf-8")
            candidate.write_text("candidate glossary translation", encoding="utf-8")
            execution_result = {
                "schema_version": "glossary-retranslation-execution-v1",
                "summary": {
                    "status": "executed",
                    "requested_chunk_count": 1,
                    "executed_chunk_count": 1,
                    "failed_chunk_count": 0,
                    "skipped_chunk_count": 0,
                },
                "artifacts": {
                    "retranslated_full_path": candidate.as_posix(),
                },
                "requested_chunk_ids": ["c0000"],
            }

            publish = write_glossary_retranslation_publish(
                execution_result,
                out / "glossary_retranslation_publish.json",
                out / "glossary_retranslation_publish.md",
                confirm=True,
                candidate_full_path=candidate,
                published_full_path=out / "glossary_retranslation_published_full.md",
                original_full_path=original,
            )

            published = out / "glossary_retranslation_published_full.md"
            self.assertEqual(publish["schema_version"], "glossary-retranslation-publish-v1")
            self.assertTrue(publish["summary"]["published"])
            self.assertEqual(publish["summary"]["publish_status"], "published")
            self.assertTrue(publish["summary"]["published_matches_candidate"])
            self.assertEqual(published.read_text(encoding="utf-8"), "candidate glossary translation")
            self.assertEqual(original.read_text(encoding="utf-8"), "original translation")

            rollback = write_glossary_retranslation_rollback(
                publish,
                out / "glossary_retranslation_rollback.json",
                out / "glossary_retranslation_rollback.md",
                confirm=True,
                original_full_path=original,
                published_full_path=published,
                rollback_full_path=out / "glossary_retranslation_rollback_full.md",
            )

            rollback_full = out / "glossary_retranslation_rollback_full.md"
            self.assertEqual(rollback["schema_version"], "glossary-retranslation-rollback-v1")
            self.assertTrue(rollback["summary"]["rollback_applied"])
            self.assertEqual(rollback["summary"]["rollback_status"], "rolled_back")
            self.assertTrue(rollback["summary"]["rollback_matches_original"])
            self.assertTrue(rollback["summary"]["published_preserved"])
            self.assertEqual(rollback_full.read_text(encoding="utf-8"), "original translation")
            self.assertEqual(published.read_text(encoding="utf-8"), "candidate glossary translation")
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

    def test_translation_qa_reports_formula_invariant_mismatch(self) -> None:
        root = Path.cwd() / "test-output" / "translation-qa-formula"
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
                        "The objective is L_i = \\sum_j x_{ij} + beta (1). "
                        "The result keeps F1-score with p < 0.05."
                    ),
                    link_count=0,
                    image_count=0,
                )
            ]
            (chunk_dir / "c0000.md").write_text(
                "---\n{}\n---\n\n目标函数只保留结论，统计显著。\n",
                encoding="utf-8",
            )

            report = build_translation_qa(chunks, chunk_dir)
            chunk_report = report["chunks"][0]
            issue = next(issue for issue in chunk_report["issues"] if issue["type"] == "formula_mismatch")
            formula_tokens = {item["token"] for item in issue["formulas"]}
            formula_kinds = {item["kind"] for item in issue["formulas"]}

            self.assertEqual(issue["severity"], "high")
            self.assertIn("L_i", formula_tokens)
            self.assertIn("(1)", formula_tokens)
            self.assertIn("F1-score", formula_tokens)
            self.assertIn("variable", formula_kinds)
            self.assertIn("equation_label", formula_kinds)
            self.assertEqual(report["summary"]["source_equation_label_count"], 1)
            self.assertEqual(report["summary"]["missing_equation_label_count"], 1)
            self.assertGreater(report["summary"]["source_formula_token_count"], 0)
            self.assertGreater(report["summary"]["missing_formula_token_count"], 0)

            plan = build_repair_plan(report)
            item = next(item for item in plan["items"] if item["issue_type"] == "formula_mismatch")
            self.assertEqual(item["priority"], "P0")
            self.assertEqual(item["action"], "rewrite_formula_context")
            self.assertEqual(item["scope"], "paragraph")
            self.assertIn("formulas", item["evidence"])
            requests = build_repair_requests(plan, chunks, chunk_dir)
            request = next(request for request in requests["requests"] if request["issue_type"] == "formula_mismatch")
            locked_tokens = set(request["locked_tokens"])
            self.assertIn("L_i", locked_tokens)
            self.assertIn("(1)", locked_tokens)

            markdown = translation_qa_to_markdown(report)
            self.assertIn("formula_mismatch", markdown)
            self.assertIn("equation_label:(1)", markdown)
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_bilingual_html_applies_confirmed_table_structure_patches(self) -> None:
        root = Path.cwd() / "test-output" / "bilingual-html-table-patches"
        if root.exists():
            shutil.rmtree(root)
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(parents=True)
        try:
            chunk = TextChunk(
                chunk_id="c0000",
                pages_0based=[0],
                text="Source paragraph without a markdown table.",
                link_count=0,
                image_count=0,
            )
            chunk.block_ids = ["p1-b0000"]
            (chunk_dir / "c0000.md").write_text(
                "| Dataset group | Covered metric | Score |\n"
                "| --- | --- | --- |\n"
                "| A | Accuracy | 91.2 |\n",
                encoding="utf-8",
            )
            reconstruction = {
                "schema_version": "table-reconstruction-v1",
                "confirmation_schema_version": "table-structure-publish-v1",
                "tables": [
                    {
                        "table_id": "p1-b0000",
                        "block_id": "p1-b0000",
                        "page_no": 1,
                        "confirmed_merged_cell_candidates": [
                            {"span_type": "colspan", "candidate_status": "human_confirmed"}
                        ],
                        "structure_patches": [
                            {
                                "patch_id": "tsp-0001-p1-b0000-r0c0",
                                "patch_type": "merged_cell_span",
                                "operation": "apply_confirmed_merged_cell_span",
                                "applied": True,
                                "table_id": "p1-b0000",
                                "anchor_cell": {"row_index": 0, "column_index": 0},
                                "span": {
                                    "span_type": "colspan",
                                    "row_span": 1,
                                    "column_span": 2,
                                },
                                "covered_cells": [{"row_index": 0, "column_index": 1}],
                            }
                        ],
                    }
                ],
            }

            confirmed = effective_table_reconstruction_view(reconstruction)
            confirmed_html_path = root / "confirmed.html"
            write_bilingual_html(
                [chunk],
                chunk_dir,
                confirmed_html_path,
                table_reconstruction=confirmed,
                title="Confirmed table patches",
            )
            confirmed_html = confirmed_html_path.read_text(encoding="utf-8")
            self.assertIn('class="structure-patched"', confirmed_html)
            self.assertIn('colspan="2"', confirmed_html)
            self.assertIn('data-structure-patch-id="tsp-0001-p1-b0000-r0c0"', confirmed_html)
            self.assertNotIn("<th>Covered metric</th>", confirmed_html)

            source_view = effective_table_reconstruction_view(
                {
                    **reconstruction,
                    "confirmation_schema_version": "",
                }
            )
            source_html_path = root / "source.html"
            write_bilingual_html(
                [chunk],
                chunk_dir,
                source_html_path,
                table_reconstruction=source_view,
                title="Source table patches",
            )
            source_html = source_html_path.read_text(encoding="utf-8")
            self.assertNotIn('colspan="2"', source_html)
            self.assertIn("<th>Covered metric</th>", source_html)
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_translated_pdf_exporter_writes_readable_pdf_and_report(self) -> None:
        root = Path.cwd() / "test-output" / "translated-pdf-exporter"
        if root.exists():
            shutil.rmtree(root)
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(parents=True)
        try:
            chunks = [
                TextChunk(
                    chunk_id="c0000",
                    pages_0based=[0],
                    text="Table 1 reports model accuracy.",
                    link_count=0,
                    image_count=0,
                )
            ]
            chunks[0].block_ids = ["p1-b0001", "p1-b0002", "p1-b0003"]
            chunks[0].structural_relation_ids = ["p1-b0001->p1-b0002:caption_for_table"]
            (chunk_dir / "c0000.md").write_text(
                "---\n{}\n---\n\n"
                "# Results\n\n"
                "结构化 PDF 译文保留表格与 QA 提示。\n\n"
                "| Model | Acc |\n"
                "| --- | --- |\n"
                "| A | 91.2% |\n",
                encoding="utf-8",
            )
            qa_report = {
                "chunks": [
                    {
                        "chunk_id": "c0000",
                        "issues": [{"type": "missing_numbers", "severity": "high", "tokens": ["120"]}],
                    }
                ]
            }
            repair_plan = {
                "items": [
                    {
                        "chunk_id": "c0000",
                        "repair_id": "r0001",
                        "priority": "P0",
                        "action": "rewrite_with_locked_tokens",
                        "reason": "missing number",
                    }
                ]
            }
            structure_qa = {
                "summary": {
                    "caption_count": 1,
                    "caption_linked_count": 1,
                    "caption_orphan_count": 0,
                    "footnote_count": 1,
                    "footnote_linked_count": 1,
                    "footnote_orphan_count": 0,
                    "table_footnote_count": 1,
                    "cross_page_relationship_count": 1,
                    "caption_cross_page_linked_count": 1,
                    "caption_cross_page_orphan_count": 0,
                    "footnote_cross_page_linked_count": 0,
                    "footnote_cross_page_orphan_count": 0,
                    "cross_page_parent_gap_max": 1,
                }
            }
            table_reconstruction = {
                "summary": {
                    "table_footnote_binding_count": 1,
                    "table_footnote_cell_binding_count": 1,
                    "table_footnote_bound_cell_count": 1,
                    "table_footnote_unbound_count": 0,
                    "table_footnote_table_level_count": 0,
                },
                "tables": [
                    {
                        "table_id": "p1-b0001",
                        "block_id": "p1-b0001",
                        "page_no": 1,
                        "caption_blocks": [{"block_id": "p1-b0002", "text": "Table 1: Results"}],
                        "footnote_blocks": [{"block_id": "p1-b0003", "text": "* p < 0.05"}],
                        "footnote_bindings": [
                            {
                                "status": "bound_to_cells",
                                "matched_cell_count": 1,
                                "matched_cells": [{"row_index": 1, "column_index": 1}],
                            }
                        ],
                        "merged_cell_candidates": [
                            {"span_type": "colspan", "row_index": 0, "column_index": 0}
                        ],
                    }
                ],
                "continued_table_groups": [
                    {"group_id": "ctg0001", "table_ids": ["p1-b0001"], "merge_status": "merged"}
                ],
            }
            pdf_path = root / "translated_full.pdf"
            report_path = root / "translated_pdf_report.json"

            report = write_translated_pdf(
                chunks,
                chunk_dir,
                pdf_path,
                qa_report=qa_report,
                repair_plan=repair_plan,
                structure_qa=structure_qa,
                table_reconstruction=table_reconstruction,
                title="Unit Structured PDF",
                source_pdf="sample.pdf",
                report_path=report_path,
            )

            self.assertTrue(pdf_path.is_file())
            self.assertTrue(report_path.is_file())
            doc = fitz.open(pdf_path)
            try:
                text = "\n".join(page.get_text("text") for page in doc)
                self.assertGreaterEqual(len(doc), 1)
            finally:
                doc.close()
            self.assertIn("Unit Structured PDF", text)
            self.assertIn("91.2", text)
            self.assertIn("missing_numbers", text)
            self.assertIn("Structure context", text)
            self.assertIn("Source tables", text)
            self.assertTrue(report["summary"]["generated"])
            self.assertEqual(report["summary"]["chunk_count"], 1)
            self.assertEqual(report["summary"]["table_count"], 1)
            self.assertEqual(report["summary"]["caption_count"], 1)
            self.assertEqual(report["summary"]["footnote_linked_count"], 1)
            self.assertEqual(report["summary"]["table_footnote_cell_binding_count"], 1)
            self.assertEqual(report["summary"]["structure_context_chunk_count"], 1)
            self.assertEqual(report["summary"]["source_table_reference_count"], 1)
            self.assertEqual(report["summary"]["source_caption_reference_count"], 1)
            self.assertEqual(report["summary"]["source_footnote_cell_binding_count"], 1)
            self.assertEqual(report["summary"]["continued_table_group_reference_count"], 1)
            self.assertEqual(report["summary"]["structural_relation_reference_count"], 1)
            self.assertTrue(report["chunks"][0]["has_structure_context"])
            self.assertEqual(report["chunks"][0]["source_table_ids"], ["p1-b0001"])
            self.assertEqual(report["chunks"][0]["continued_table_group_ids"], ["ctg0001"])
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], "translated-pdf-report-v1")
            self.assertEqual(saved["summary"]["qa_issue_count"], 1)
            self.assertEqual(saved["summary"]["repair_item_count"], 1)
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_translated_pdf_report_marks_confirmed_table_reconstruction(self) -> None:
        root = Path.cwd() / "test-output" / "translated-pdf-confirmed-source"
        if root.exists():
            shutil.rmtree(root)
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(parents=True)
        try:
            chunk = TextChunk(
                chunk_id="c0000",
                pages_0based=[0],
                text="Table text",
                link_count=0,
                image_count=0,
            )
            chunk.block_ids = ["p1-b0000"]
            (chunk_dir / "c0000.md").write_text(
                "| Dataset metrics | Covered metric |\n| --- | --- |\n| A | 1 |\n",
                encoding="utf-8",
            )
            confirmed = effective_table_reconstruction_view(
                {
                    "schema_version": "table-reconstruction-v1",
                    "confirmation_schema_version": "table-structure-publish-v1",
                    "summary": {"confirmed_merged_cell_candidate_count": 1},
                    "tables": [
                        {
                            "table_id": "p1-b0000",
                            "block_id": "p1-b0000",
                            "page_no": 1,
                            "merged_cell_candidates": [
                                {"span_type": "colspan", "reason": "raw"},
                                {"span_type": "rowspan", "reason": "raw"},
                            ],
                            "confirmed_merged_cell_candidates": [
                                {
                                    "span_type": "colspan",
                                    "candidate_status": "human_confirmed",
                                    "reason": "confirmed",
                                }
                            ],
                            "structure_patches": [
                                {
                                    "patch_id": "tsp-0001-p1-b0000-r0c0",
                                    "patch_type": "merged_cell_span",
                                    "operation": "apply_confirmed_merged_cell_span",
                                    "applied": True,
                                    "table_id": "p1-b0000",
                                    "anchor_cell": {"row_index": 0, "column_index": 0},
                                    "span": {
                                        "span_type": "colspan",
                                        "row_span": 1,
                                        "column_span": 2,
                                    },
                                    "covered_cells": [{"row_index": 0, "column_index": 1}],
                                }
                            ],
                        }
                    ],
                }
            )

            report = write_translated_pdf(
                [chunk],
                chunk_dir,
                root / "translated_full.pdf",
                table_reconstruction=confirmed,
                title="Confirmed PDF",
                report_path=root / "translated_pdf_report.json",
            )

            self.assertEqual(report["table_reconstruction_source"], "confirmed")
            self.assertEqual(report["summary"]["merged_cell_candidate_reference_count"], 1)
            self.assertEqual(report["summary"]["confirmed_merged_cell_candidate_reference_count"], 1)
            self.assertEqual(report["summary"]["confirmed_merged_cell_candidate_count"], 1)
            self.assertEqual(report["summary"]["table_structure_patch_count"], 1)
            self.assertEqual(report["summary"]["table_structure_patch_reference_count"], 1)
            self.assertEqual(report["summary"]["table_structure_patch_covered_cell_reference_count"], 1)
            self.assertEqual(report["summary"]["table_structure_patch_rendered_count"], 1)
            self.assertEqual(report["chunks"][0]["table_structure_patch_rendered_count"], 1)
            doc = fitz.open(root / "translated_full.pdf")
            try:
                text = "\n".join(page.get_text("text") for page in doc)
            finally:
                doc.close()
            self.assertIn("Dataset metrics", text)
            self.assertNotIn("Covered metric", text)
            saved = json.loads((root / "translated_pdf_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["table_reconstruction_source"], "confirmed")
            self.assertEqual(saved["summary"]["table_structure_patch_reference_count"], 1)
            self.assertEqual(saved["summary"]["table_structure_patch_rendered_count"], 1)

            source_view = effective_table_reconstruction_view(
                {
                    "schema_version": "table-reconstruction-v1",
                    "tables": confirmed["tables"],
                }
            )
            source_report = write_translated_pdf(
                [chunk],
                chunk_dir,
                root / "source_view.pdf",
                table_reconstruction=source_view,
                title="Source PDF",
                report_path=root / "source_translated_pdf_report.json",
            )
            self.assertEqual(source_report["table_reconstruction_source"], "source")
            self.assertEqual(source_report["summary"]["table_structure_patch_rendered_count"], 0)
            source_doc = fitz.open(root / "source_view.pdf")
            try:
                source_text = "\n".join(page.get_text("text") for page in source_doc)
            finally:
                source_doc.close()
            self.assertIn("Covered metric", source_text)
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)

    def test_translated_pdf_renders_confirmed_rowspan_patches(self) -> None:
        root = Path.cwd() / "test-output" / "translated-pdf-rowspan-patch"
        if root.exists():
            shutil.rmtree(root)
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(parents=True)
        try:
            chunk = TextChunk(
                chunk_id="c0000",
                pages_0based=[0],
                text="Table text",
                link_count=0,
                image_count=0,
            )
            chunk.block_ids = ["p1-b0000"]
            (chunk_dir / "c0000.md").write_text(
                "| Group | Metric | Score |\n"
                "| --- | --- | --- |\n"
                "| Shared group | Accuracy | 91.2 |\n"
                "| SHOULD_HIDE | F1 | 88.1 |\n",
                encoding="utf-8",
            )
            confirmed = effective_table_reconstruction_view(
                {
                    "schema_version": "table-reconstruction-v1",
                    "confirmation_schema_version": "table-structure-publish-v1",
                    "summary": {"confirmed_merged_cell_candidate_count": 1},
                    "tables": [
                        {
                            "table_id": "p1-b0000",
                            "block_id": "p1-b0000",
                            "page_no": 1,
                            "confirmed_merged_cell_candidates": [
                                {
                                    "span_type": "rowspan",
                                    "candidate_status": "human_confirmed",
                                }
                            ],
                            "structure_patches": [
                                {
                                    "patch_id": "tsp-0002-p1-b0000-r1c0",
                                    "patch_type": "merged_cell_span",
                                    "operation": "apply_confirmed_merged_cell_span",
                                    "applied": True,
                                    "table_id": "p1-b0000",
                                    "anchor_cell": {"row_index": 1, "column_index": 0},
                                    "span": {
                                        "span_type": "rowspan",
                                        "row_span": 2,
                                        "column_span": 1,
                                    },
                                    "covered_cells": [{"row_index": 2, "column_index": 0}],
                                }
                            ],
                        }
                    ],
                }
            )

            report = write_translated_pdf(
                [chunk],
                chunk_dir,
                root / "translated_full.pdf",
                table_reconstruction=confirmed,
                title="Rowspan PDF",
                report_path=root / "translated_pdf_report.json",
            )

            self.assertEqual(report["summary"]["table_structure_patch_rendered_count"], 1)
            doc = fitz.open(root / "translated_full.pdf")
            try:
                text = "\n".join(page.get_text("text") for page in doc)
            finally:
                doc.close()
            self.assertIn("Shared group", text)
            self.assertIn("Accuracy", text)
            self.assertIn("F1", text)
            self.assertNotIn("SHOULD_HIDE", text)
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
            active_manifest_path = work_dir / "output" / "chunks_manifest.json"
            qa_path = work_dir / "output" / "structure_qa.json"
            table_reconstruction_path = work_dir / "output" / "table_reconstruction.json"
            table_merged_cell_review_path = work_dir / "output" / "table_merged_cell_review.json"
            table_merged_cell_review_md_path = work_dir / "output" / "table_merged_cell_review.md"
            table_structure_publish_path = work_dir / "output" / "table_structure_publish.json"
            table_structure_publish_md_path = work_dir / "output" / "table_structure_publish.md"
            table_reconstruction_confirmed_path = work_dir / "output" / "table_reconstruction_confirmed.json"
            structure_hints_manifest_path = work_dir / "output" / "structure_hints_manifest.json"
            chunk_boundary_qa_path = work_dir / "output" / "chunk_boundary_qa.json"
            chunk_strategy_comparison_path = work_dir / "output" / "chunk_strategy_comparison.json"
            vision_path = work_dir / "output" / "vision_route.json"
            ocr_tasks_path = work_dir / "output" / "ocr_tasks.json"
            ocr_results_path = work_dir / "output" / "ocr_results.json"
            ocr_writeback_path = work_dir / "output" / "ocr_writeback.json"
            ocr_candidate_qa_path = work_dir / "output" / "ocr_candidate_qa.json"
            ocr_candidate_qa_md_path = work_dir / "output" / "ocr_candidate_qa.md"
            document_ir_ocr_path = work_dir / "output" / "document_ir_ocr.json"
            ocr_candidate_promotion_path = work_dir / "output" / "ocr_candidate_promotion.json"
            ocr_candidate_promotion_md_path = work_dir / "output" / "ocr_candidate_promotion.md"
            document_ir_promoted_path = work_dir / "output" / "document_ir_promoted.json"
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
            repair_patch_review_path = work_dir / "output" / "repair_patch_review.json"
            repair_patch_review_md_path = work_dir / "output" / "repair_patch_review.md"
            repair_publish_path = work_dir / "output" / "repair_publish.json"
            repair_publish_md_path = work_dir / "output" / "repair_publish.md"
            repair_rollback_path = work_dir / "output" / "repair_rollback.json"
            repair_rollback_md_path = work_dir / "output" / "repair_rollback.md"
            repair_formal_replace_path = work_dir / "output" / "repair_formal_replace.json"
            repair_formal_replace_md_path = work_dir / "output" / "repair_formal_replace.md"
            repair_formal_rollback_path = work_dir / "output" / "repair_formal_rollback.json"
            repair_formal_rollback_md_path = work_dir / "output" / "repair_formal_rollback.md"
            repair_merge_qa_path = work_dir / "output" / "repair_merge_qa.json"
            repair_merge_qa_md_path = work_dir / "output" / "repair_merge_qa.md"
            repair_effectiveness_path = work_dir / "output" / "repair_effectiveness.json"
            repair_effectiveness_md_path = work_dir / "output" / "repair_effectiveness.md"
            repaired_full_path = work_dir / "output" / "repaired_full.md"
            published_full_path = work_dir / "output" / "published_full.md"
            rollback_full_path = work_dir / "output" / "rollback_full.md"
            formal_full_path = work_dir / "output" / "formal_full.md"
            formal_backup_full_path = work_dir / "output" / "formal_full.before_repair.md"
            formal_active_before_rollback_path = work_dir / "output" / "formal_full.repair_applied.md"
            metrics_path = work_dir / "output" / "experiment_metrics.json"
            run_metrics_path = work_dir / "output" / "run_metrics.json"
            run_log_path = work_dir / "output" / "run_log.jsonl"
            cost_estimate_path = work_dir / "output" / "cost_estimate.json"
            bilingual_path = work_dir / "output" / "bilingual.html"
            translated_pdf_path = work_dir / "output" / "translated_full.pdf"
            translated_pdf_report_path = work_dir / "output" / "translated_pdf_report.json"
            self.assertTrue(ir_path.is_file())
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(active_manifest_path.is_file())
            self.assertTrue(qa_path.is_file())
            self.assertTrue(table_reconstruction_path.is_file())
            self.assertTrue(table_merged_cell_review_path.is_file())
            self.assertTrue(table_merged_cell_review_md_path.is_file())
            self.assertTrue(table_structure_publish_path.is_file())
            self.assertTrue(table_structure_publish_md_path.is_file())
            self.assertFalse(table_reconstruction_confirmed_path.exists())
            self.assertTrue(structure_hints_manifest_path.is_file())
            self.assertTrue(chunk_boundary_qa_path.is_file())
            self.assertTrue(chunk_strategy_comparison_path.is_file())
            self.assertTrue(vision_path.is_file())
            self.assertTrue(ocr_tasks_path.is_file())
            self.assertTrue(ocr_results_path.is_file())
            self.assertTrue(ocr_writeback_path.is_file())
            self.assertTrue(ocr_candidate_qa_path.is_file())
            self.assertTrue(ocr_candidate_qa_md_path.is_file())
            self.assertTrue(document_ir_ocr_path.is_file())
            self.assertTrue(ocr_candidate_promotion_path.is_file())
            self.assertTrue(ocr_candidate_promotion_md_path.is_file())
            self.assertTrue(document_ir_promoted_path.is_file())
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
            self.assertTrue(repair_patch_review_path.is_file())
            self.assertTrue(repair_patch_review_md_path.is_file())
            self.assertTrue(repair_publish_path.is_file())
            self.assertTrue(repair_publish_md_path.is_file())
            self.assertTrue(repair_rollback_path.is_file())
            self.assertTrue(repair_rollback_md_path.is_file())
            self.assertTrue(repair_formal_replace_path.is_file())
            self.assertTrue(repair_formal_replace_md_path.is_file())
            self.assertTrue(repair_formal_rollback_path.is_file())
            self.assertTrue(repair_formal_rollback_md_path.is_file())
            self.assertTrue(repair_merge_qa_path.is_file())
            self.assertTrue(repair_merge_qa_md_path.is_file())
            self.assertTrue(repair_effectiveness_path.is_file())
            self.assertTrue(repair_effectiveness_md_path.is_file())
            self.assertTrue(repaired_full_path.is_file())
            self.assertFalse(published_full_path.exists())
            self.assertFalse(rollback_full_path.exists())
            self.assertFalse(formal_full_path.exists())
            self.assertFalse(formal_backup_full_path.exists())
            self.assertFalse(formal_active_before_rollback_path.exists())
            self.assertTrue(metrics_path.is_file())
            self.assertTrue(run_metrics_path.is_file())
            self.assertTrue(run_log_path.is_file())
            self.assertTrue(cost_estimate_path.is_file())
            self.assertTrue(bilingual_path.is_file())
            self.assertTrue(translated_pdf_path.is_file())
            self.assertTrue(translated_pdf_report_path.is_file())
            ir = json.loads(ir_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            active_manifest = json.loads(active_manifest_path.read_text(encoding="utf-8"))
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
            table_reconstruction = json.loads(table_reconstruction_path.read_text(encoding="utf-8"))
            table_merged_cell_review = json.loads(table_merged_cell_review_path.read_text(encoding="utf-8"))
            table_structure_publish = json.loads(table_structure_publish_path.read_text(encoding="utf-8"))
            structure_hints_manifest = json.loads(structure_hints_manifest_path.read_text(encoding="utf-8"))
            chunk_boundary_qa = json.loads(chunk_boundary_qa_path.read_text(encoding="utf-8"))
            chunk_strategy_comparison = json.loads(chunk_strategy_comparison_path.read_text(encoding="utf-8"))
            vision = json.loads(vision_path.read_text(encoding="utf-8"))
            ocr_tasks = json.loads(ocr_tasks_path.read_text(encoding="utf-8"))
            ocr_results = json.loads(ocr_results_path.read_text(encoding="utf-8"))
            ocr_writeback = json.loads(ocr_writeback_path.read_text(encoding="utf-8"))
            ocr_candidate_qa = json.loads(ocr_candidate_qa_path.read_text(encoding="utf-8"))
            document_ir_ocr = json.loads(document_ir_ocr_path.read_text(encoding="utf-8"))
            ocr_candidate_promotion = json.loads(ocr_candidate_promotion_path.read_text(encoding="utf-8"))
            document_ir_promoted = json.loads(document_ir_promoted_path.read_text(encoding="utf-8"))
            translation_qa = json.loads(translation_qa_path.read_text(encoding="utf-8"))
            repair_plan = json.loads(repair_plan_path.read_text(encoding="utf-8"))
            repair_requests = json.loads(repair_requests_path.read_text(encoding="utf-8"))
            repair_results = json.loads(repair_results_path.read_text(encoding="utf-8"))
            repair_validation = json.loads(repair_validation_path.read_text(encoding="utf-8"))
            repair_merge = json.loads(repair_merge_path.read_text(encoding="utf-8"))
            repair_patch_review = json.loads(repair_patch_review_path.read_text(encoding="utf-8"))
            repair_publish = json.loads(repair_publish_path.read_text(encoding="utf-8"))
            repair_rollback = json.loads(repair_rollback_path.read_text(encoding="utf-8"))
            repair_formal_replace = json.loads(repair_formal_replace_path.read_text(encoding="utf-8"))
            repair_formal_rollback = json.loads(repair_formal_rollback_path.read_text(encoding="utf-8"))
            repair_merge_qa = json.loads(repair_merge_qa_path.read_text(encoding="utf-8"))
            repair_effectiveness = json.loads(repair_effectiveness_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            run_metrics = json.loads(run_metrics_path.read_text(encoding="utf-8"))
            cost_estimate = json.loads(cost_estimate_path.read_text(encoding="utf-8"))
            translated_pdf_report = json.loads(translated_pdf_report_path.read_text(encoding="utf-8"))
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
            self.assertIn("structural_relation_ids", manifest[0])
            self.assertIn("approx_tokens", manifest[0])
            self.assertIn("budget", manifest[0])
            self.assertGreaterEqual(len(active_manifest), 1)
            self.assertIn("budget", active_manifest[0])
            self.assertIn("structural_relation_ids", active_manifest[0])
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
            self.assertIn("continued_table_group_count", table_reconstruction["summary"])
            self.assertIn("merged_cell_candidate_count", table_reconstruction["summary"])
            self.assertIn("merged_cell_candidate_type_counts", table_reconstruction["summary"])
            self.assertIn("table_footnote_cell_binding_count", table_reconstruction["summary"])
            self.assertIn("table_footnote_unbound_count", table_reconstruction["summary"])
            self.assertIn("table_chain_reject_reason_counts", table_reconstruction["summary"])
            self.assertIn("table_chain_reject_reason_category_counts", table_reconstruction["summary"])
            self.assertIn("continued_table_groups", table_reconstruction)
            self.assertEqual(table_merged_cell_review["schema_version"], "table-merged-cell-review-v1")
            self.assertIn("candidate_review_count", table_merged_cell_review["summary"])
            self.assertIn("review_required_count", table_merged_cell_review["summary"])
            self.assertIn("candidate_reviews", table_merged_cell_review)
            self.assertIn(
                "表格合并单元格候选人工确认清单",
                table_merged_cell_review_md_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(table_structure_publish["schema_version"], "table-structure-publish-v1")
            self.assertFalse(table_structure_publish["summary"]["confirmed"])
            self.assertFalse(table_structure_publish["summary"]["published"])
            self.assertEqual(table_structure_publish["summary"]["publish_status"], "pending_confirmation")
            self.assertIn(
                "表格结构人工确认发布",
                table_structure_publish_md_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(structure_hints_manifest["schema_version"], "structure-hints-manifest-v1")
            self.assertGreaterEqual(structure_hints_manifest["summary"]["chunk_count"], 1)
            self.assertIn("structure_hint_chunk_count", structure_hints_manifest["summary"])
            self.assertIn("structure_hint_empty_chunk_count", structure_hints_manifest["summary"])
            self.assertIn("structure_hint_avg_char_count", structure_hints_manifest["summary"])
            self.assertIn("structure_hint_max_char_count", structure_hints_manifest["summary"])
            self.assertIn("structure_hint_merged_cell_candidate_type_counts", structure_hints_manifest["summary"])
            self.assertIn("structure_hint_merged_cell_candidate_reason_counts", structure_hints_manifest["summary"])
            self.assertIn("structure_hint_relationship_count", structure_hints_manifest["summary"])
            self.assertIn("structure_hint_relationship_type_counts", structure_hints_manifest["summary"])
            self.assertIn("structure_hint_entity_count", structure_hints_manifest["summary"])
            self.assertGreaterEqual(len(structure_hints_manifest["chunks"]), 1)
            self.assertIn("has_structure_hints", structure_hints_manifest["chunks"][0])
            self.assertIn("hint_text", structure_hints_manifest["chunks"][0])
            self.assertIn("relationship_count", structure_hints_manifest["chunks"][0])
            self.assertIn("entity_hint_count", structure_hints_manifest["chunks"][0])
            self.assertEqual(
                structure_hints_manifest["chunks"][0]["hint_char_count"],
                len(structure_hints_manifest["chunks"][0]["hint_text"]),
            )
            self.assertEqual(chunk_boundary_qa["schema_version"], "chunk-boundary-qa-v1")
            self.assertEqual(chunk_boundary_qa["pipeline_variant"], "structure")
            self.assertIn("split_boundary_count", chunk_boundary_qa["summary"])
            self.assertIn("budget_pressure_counts", chunk_boundary_qa["summary"])
            self.assertIn("chunks", chunk_boundary_qa)
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
            self.assertEqual(ocr_candidate_qa["schema_version"], "ocr-candidate-qa-v1")
            self.assertIn("candidate_count", ocr_candidate_qa["summary"])
            self.assertEqual(document_ir_ocr["schema_version"], "document-ir-v1")
            self.assertEqual(ocr_candidate_promotion["schema_version"], "ocr-candidate-promotion-v1")
            self.assertIn("promoted_candidate_count", ocr_candidate_promotion["summary"])
            self.assertEqual(document_ir_promoted["schema_version"], "document-ir-v1")
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
            self.assertEqual(repair_patch_review["schema_version"], "repair-patch-review-v1")
            self.assertIn("patch_count", repair_patch_review["summary"])
            self.assertIn("default_decision_counts", repair_patch_review["summary"])
            self.assertEqual(repair_publish["schema_version"], "repair-publish-v1")
            self.assertFalse(repair_publish["summary"]["confirmed"])
            self.assertFalse(repair_publish["summary"]["published"])
            self.assertEqual(repair_publish["summary"]["publish_status"], "pending_confirmation")
            self.assertEqual(repair_rollback["schema_version"], "repair-rollback-v1")
            self.assertFalse(repair_rollback["summary"]["confirmed"])
            self.assertFalse(repair_rollback["summary"]["rollback_applied"])
            self.assertEqual(repair_rollback["summary"]["rollback_status"], "not_ready")
            self.assertEqual(repair_formal_replace["schema_version"], "repair-formal-replace-v1")
            self.assertFalse(repair_formal_replace["summary"]["confirmed"])
            self.assertFalse(repair_formal_replace["summary"]["replaced"])
            self.assertEqual(repair_formal_replace["summary"]["replace_status"], "not_ready")
            self.assertEqual(repair_formal_rollback["schema_version"], "repair-formal-rollback-v1")
            self.assertFalse(repair_formal_rollback["summary"]["confirmed"])
            self.assertFalse(repair_formal_rollback["summary"]["rollback_applied"])
            self.assertEqual(repair_formal_rollback["summary"]["rollback_status"], "not_ready")
            self.assertEqual(repair_merge_qa["schema_version"], "translation-qa-v1")
            self.assertEqual(repair_effectiveness["schema_version"], "repair-effectiveness-v1")
            self.assertIn("before_issue_count", repair_effectiveness["summary"])
            self.assertIn("after_issue_count", repair_effectiveness["summary"])
            self.assertEqual(run_metrics["schema_version"], "run-metrics-v1")
            self.assertEqual(run_metrics["pipeline_variant"], "structure")
            self.assertGreaterEqual(run_metrics["summary"]["translation_request_count"], 1)
            self.assertEqual(run_metrics["summary"]["http_attempt_count"], 0)
            self.assertEqual(run_metrics["summary"]["http_retry_count"], 0)
            self.assertGreater(run_metrics["summary"]["request_char_count"], 0)
            self.assertIn("document_ir", run_metrics["summary"]["stage_elapsed_ms"])
            self.assertIn("ocr_candidate_promotion", run_metrics["summary"]["stage_elapsed_ms"])
            self.assertIn("table_merged_cell_review", run_metrics["summary"]["stage_elapsed_ms"])
            self.assertIn("table_structure_publish", run_metrics["summary"]["stage_elapsed_ms"])
            self.assertIn("repair_patch_review", run_metrics["summary"]["stage_elapsed_ms"])
            self.assertIn("repair_effectiveness", run_metrics["summary"]["stage_elapsed_ms"])
            self.assertIn("repair_publish", run_metrics["summary"]["stage_elapsed_ms"])
            self.assertIn("repair_rollback", run_metrics["summary"]["stage_elapsed_ms"])
            self.assertIn("repair_formal_replace", run_metrics["summary"]["stage_elapsed_ms"])
            self.assertIn("repair_formal_rollback", run_metrics["summary"]["stage_elapsed_ms"])
            self.assertIn("translated_pdf", run_metrics["summary"]["stage_elapsed_ms"])
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
            self.assertIn("table_merged_cell_candidate_count", metrics["quality"])
            self.assertIn("table_merged_cell_review_count", metrics["quality"])
            self.assertIn("table_merged_cell_review_required_rate", metrics["rates"])
            self.assertIn("table_merged_cell_review_default_decision_counts", metrics["breakdowns"])
            self.assertFalse(metrics["quality"]["table_structure_publish_confirmed"])
            self.assertFalse(metrics["quality"]["table_structure_publish_published"])
            self.assertIn("table_structure_publish_rate", metrics["rates"])
            self.assertIn("structure_hint_chunk_count", metrics["quality"])
            self.assertIn("structure_hint_empty_chunk_count", metrics["quality"])
            self.assertIn("structure_hint_avg_char_count", metrics["quality"])
            self.assertIn("structure_hint_max_char_count", metrics["quality"])
            self.assertIn("structure_hint_relationship_count", metrics["quality"])
            self.assertIn("structure_hint_entity_count", metrics["quality"])
            self.assertIn("structure_hint_chunk_rate", metrics["rates"])
            self.assertIn("structure_hint_relationship_per_chunk", metrics["rates"])
            self.assertIn("structure_hint_entity_per_chunk", metrics["rates"])
            self.assertIn("structure_hint_merged_cell_candidate_type_counts", metrics["breakdowns"])
            self.assertIn("structure_hint_relationship_type_counts", metrics["breakdowns"])
            self.assertIn("table_merged_cell_candidate_rate", metrics["rates"])
            self.assertIn("table_merged_cell_candidate_type_counts", metrics["breakdowns"])
            self.assertIn("continued_table_group_count", metrics["quality"])
            self.assertIn("table_footnote_cell_binding_count", metrics["quality"])
            self.assertIn("table_footnote_cell_binding_rate", metrics["rates"])
            self.assertIn("table_chain_reject_reason_count", metrics["quality"])
            self.assertIn("table_chain_reject_reason_counts", metrics["breakdowns"])
            self.assertIn("table_reconstruction_ready_rate", metrics["rates"])
            self.assertIn("continued_table_reconstruction_rate", metrics["rates"])
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
            self.assertIn("ocr_candidate_qa_count", metrics["quality"])
            self.assertEqual(metrics["evidence_files"]["ocr_candidate_qa"], "output/ocr_candidate_qa.json")
            self.assertIn("ocr_candidate_promoted_count", metrics["quality"])
            self.assertEqual(metrics["evidence_files"]["ocr_candidate_promotion"], "output/ocr_candidate_promotion.json")
            self.assertEqual(metrics["evidence_files"]["document_ir_ocr"], "output/document_ir_ocr.json")
            self.assertEqual(metrics["evidence_files"]["document_ir_promoted"], "output/document_ir_promoted.json")
            self.assertFalse(metrics["quality"]["repair_publish_published"])
            self.assertIn("repair_patch_review_count", metrics["quality"])
            self.assertIn("repair_patch_review_required_count", metrics["quality"])
            self.assertIn("repair_patch_review_safe_rate", metrics["rates"])
            self.assertIn("repair_patch_review_default_decision_counts", metrics["breakdowns"])
            self.assertEqual(metrics["evidence_files"]["repair_patch_review"], "output/repair_patch_review.json")
            self.assertIn("repair_effectiveness_issue_delta", metrics["quality"])
            self.assertIn("repair_effectiveness_issue_reduction_rate", metrics["rates"])
            self.assertIn("repair_effectiveness_status_counts", metrics["breakdowns"])
            self.assertEqual(metrics["breakdowns"]["repair_publish_status_counts"]["pending_confirmation"], 1)
            self.assertEqual(metrics["evidence_files"]["repair_publish"], "output/repair_publish.json")
            self.assertEqual(metrics["evidence_files"]["repair_published_full"], "output/published_full.md")
            self.assertEqual(metrics["evidence_files"]["repair_formal_replace"], "output/repair_formal_replace.json")
            self.assertEqual(metrics["evidence_files"]["repair_formal_rollback"], "output/repair_formal_rollback.json")
            self.assertEqual(metrics["evidence_files"]["repair_formal_full"], "output/formal_full.md")
            self.assertEqual(
                metrics["evidence_files"]["repair_formal_backup_full"],
                "output/formal_full.before_repair.md",
            )
            self.assertIn("entity_missing_rate", metrics["rates"])
            self.assertIn("split_boundary_rate", metrics["rates"])
            self.assertIn("budget_overflow_chunk_rate", metrics["rates"])
            self.assertEqual(
                metrics["evidence_files"]["structure_chunks_manifest"],
                "output/structure_chunks_manifest.json",
            )
            self.assertEqual(metrics["evidence_files"]["chunk_boundary_qa"], "output/chunk_boundary_qa.json")
            self.assertEqual(
                metrics["evidence_files"]["table_reconstruction"],
                "output/table_reconstruction.json",
            )
            self.assertEqual(
                metrics["evidence_files"]["table_merged_cell_review"],
                "output/table_merged_cell_review.json",
            )
            self.assertEqual(
                metrics["evidence_files"]["table_structure_publish"],
                "output/table_structure_publish.json",
            )
            self.assertEqual(
                metrics["evidence_files"]["table_reconstruction_confirmed"],
                "output/table_reconstruction_confirmed.json",
            )
            self.assertEqual(
                metrics["evidence_files"]["structure_hints_manifest"],
                "output/structure_hints_manifest.json",
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
            self.assertEqual(metrics["evidence_files"]["repair_effectiveness"], "output/repair_effectiveness.json")
            self.assertEqual(metrics["evidence_files"]["run_metrics"], "output/run_metrics.json")
            self.assertEqual(metrics["evidence_files"]["run_log"], "output/run_log.jsonl")
            self.assertEqual(metrics["evidence_files"]["cost_estimate"], "output/cost_estimate.json")
            self.assertTrue(metrics["quality"]["translated_pdf_generated"])
            self.assertGreaterEqual(metrics["quality"]["translated_pdf_page_count"], 1)
            self.assertEqual(metrics["evidence_files"]["translated_pdf"], "output/translated_full.pdf")
            self.assertEqual(metrics["evidence_files"]["translated_pdf_report"], "output/translated_pdf_report.json")
            self.assertEqual(translated_pdf_report["schema_version"], "translated-pdf-report-v1")
            self.assertTrue(translated_pdf_report["summary"]["generated"])
            self.assertIn("caption_count", translated_pdf_report["summary"])
            self.assertIn("table_footnote_cell_binding_count", translated_pdf_report["summary"])
            self.assertIn("structure_context_chunk_count", translated_pdf_report["summary"])
            self.assertIn("translated_pdf_structure_context_chunk_rate", metrics["rates"])
            pdf_doc = fitz.open(translated_pdf_path)
            try:
                pdf_text = "\n".join(page.get_text("text") for page in pdf_doc)
                self.assertGreaterEqual(len(pdf_doc), 1)
            finally:
                pdf_doc.close()
            self.assertIn("ECHO", pdf_text)
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
                "translated_full.pdf",
                "translated_pdf_report.json",
                "repaired_full.md",
                "bilingual.html",
                "glossary_retranslation_plan.json",
                "glossary_retranslation_plan.md",
                "document_ir.json",
                "structure_chunks_manifest.json",
                "structure_hints_manifest.json",
                "structure_qa.json",
                "table_reconstruction.json",
                "table_merged_cell_review.json",
                "table_merged_cell_review.md",
                "table_structure_publish.json",
                "table_structure_publish.md",
                "table_reconstruction_confirmed.json",
                "chunk_boundary_qa.json",
                "chunk_strategy_comparison.json",
                "vision_route.json",
                "ocr_tasks.json",
                "ocr_results.json",
                "ocr_writeback.json",
                "ocr_candidate_qa.json",
                "ocr_candidate_qa.md",
                "document_ir_ocr.json",
                "ocr_candidate_promotion.json",
                "ocr_candidate_promotion.md",
                "document_ir_promoted.json",
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
                "repair_patch_review.json",
                "repair_patch_review.md",
                "repair_publish.json",
                "repair_publish.md",
                "repair_rollback.json",
                "repair_rollback.md",
                "repair_formal_replace.json",
                "repair_formal_replace.md",
                "repair_formal_rollback.json",
                "repair_formal_rollback.md",
                "repair_merge_qa.json",
                "repair_merge_qa.md",
                "repair_effectiveness.json",
                "repair_effectiveness.md",
                "published_full.md",
                "rollback_full.md",
                "formal_full.md",
                "formal_full.before_repair.md",
                "formal_full.repair_applied.md",
                "experiment_metrics.json",
                "run_metrics.json",
                "cost_estimate.json",
                "glossary_retranslation_result.json",
                "glossary_retranslation_result.md",
                "glossary_retranslated_full.md",
                "glossary_retranslation_publish.json",
                "glossary_retranslation_publish.md",
                "glossary_retranslation_published_full.md",
                "glossary_retranslation_rollback.json",
                "glossary_retranslation_rollback.md",
                "glossary_retranslation_rollback_full.md",
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
            self.assertIn("output/repair_patch_review.json", rels)
            self.assertIn("output/repair_patch_review.md", rels)
            self.assertIn("output/repair_publish.json", rels)
            self.assertIn("output/repair_publish.md", rels)
            self.assertIn("output/repair_rollback.json", rels)
            self.assertIn("output/repair_rollback.md", rels)
            self.assertIn("output/repair_formal_replace.json", rels)
            self.assertIn("output/repair_formal_replace.md", rels)
            self.assertIn("output/repair_formal_rollback.json", rels)
            self.assertIn("output/repair_formal_rollback.md", rels)
            self.assertIn("output/repair_merge_qa.json", rels)
            self.assertIn("output/repair_effectiveness.json", rels)
            self.assertIn("output/repair_effectiveness.md", rels)
            self.assertIn("output/repairs/rq0000.md", rels)
            self.assertIn("output/repaired_chunks/c0000.md", rels)
            self.assertIn("output/vision_pages/page-0001.png", rels)
            self.assertIn("output/vision_crops/page-0001/p1-b0000-image.png", rels)
            self.assertIn("output/ocr_tasks.json", rels)
            self.assertIn("output/ocr_results.json", rels)
            self.assertIn("output/ocr_writeback.json", rels)
            self.assertIn("output/ocr_candidate_qa.json", rels)
            self.assertIn("output/ocr_candidate_qa.md", rels)
            self.assertIn("output/document_ir_ocr.json", rels)
            self.assertIn("output/ocr_candidate_promotion.json", rels)
            self.assertIn("output/ocr_candidate_promotion.md", rels)
            self.assertIn("output/document_ir_promoted.json", rels)
            self.assertIn("output/repaired_full.md", rels)
            self.assertIn("output/published_full.md", rels)
            self.assertIn("output/rollback_full.md", rels)
            self.assertIn("output/formal_full.md", rels)
            self.assertIn("output/formal_full.before_repair.md", rels)
            self.assertIn("output/formal_full.repair_applied.md", rels)
            self.assertIn("output/translated_full.pdf", rels)
            self.assertIn("output/translated_pdf_report.json", rels)
            self.assertIn("output/bilingual.html", rels)
            self.assertIn("output/glossary_retranslation_plan.json", rels)
            self.assertIn("output/glossary_retranslation_plan.md", rels)
            self.assertIn("output/glossary_retranslation_result.json", rels)
            self.assertIn("output/glossary_retranslation_result.md", rels)
            self.assertIn("output/glossary_retranslated_full.md", rels)
            self.assertIn("output/glossary_retranslation_publish.json", rels)
            self.assertIn("output/glossary_retranslation_publish.md", rels)
            self.assertIn("output/glossary_retranslation_published_full.md", rels)
            self.assertIn("output/glossary_retranslation_rollback.json", rels)
            self.assertIn("output/glossary_retranslation_rollback.md", rels)
            self.assertIn("output/glossary_retranslation_rollback_full.md", rels)
            self.assertIn("output/qa_report.md", rels)
            self.assertIn("output/document_ir.json", rels)
            self.assertIn("output/table_reconstruction.json", rels)
            self.assertIn("output/table_merged_cell_review.json", rels)
            self.assertIn("output/table_merged_cell_review.md", rels)
            self.assertIn("output/table_structure_publish.json", rels)
            self.assertIn("output/table_structure_publish.md", rels)
            self.assertIn("output/table_reconstruction_confirmed.json", rels)
            self.assertIn("output/structure_hints_manifest.json", rels)
            self.assertIn("output/chunk_boundary_qa.json", rels)
            self.assertIn("output/chunk_strategy_comparison.json", rels)
            self.assertIn("output/experiment_metrics.json", rels)
            self.assertIn("output/run_metrics.json", rels)
            self.assertIn("output/cost_estimate.json", rels)
            self.assertEqual(map_bundle_arcname("output/repair_formal_replace.json"), "质量/局部修复正式替换.json")
            self.assertEqual(map_bundle_arcname("output/repair_formal_replace.md"), "质量/局部修复正式替换.md")
            self.assertEqual(map_bundle_arcname("output/repair_formal_rollback.json"), "质量/局部修复正式回滚.json")
            self.assertEqual(map_bundle_arcname("output/repair_formal_rollback.md"), "质量/局部修复正式回滚.md")
            self.assertEqual(map_bundle_arcname("output/formal_full.md"), "译文/正式译文.md")
            self.assertEqual(map_bundle_arcname("output/formal_full.before_repair.md"), "译文/正式译文修复前备份.md")
            self.assertEqual(map_bundle_arcname("output/formal_full.repair_applied.md"), "译文/正式译文回滚前修复稿.md")
            self.assertEqual(map_bundle_arcname("output/bilingual.html"), "译文/双语对照.html")
            self.assertEqual(
                map_bundle_arcname("output/glossary_retranslation_plan.json"),
                "质量/术语确认重译计划.json",
            )
            self.assertEqual(
                map_bundle_arcname("output/glossary_retranslation_plan.md"),
                "质量/术语确认重译计划.md",
            )
            self.assertEqual(
                map_bundle_arcname("output/glossary_retranslation_result.json"),
                "质量/术语重译执行报告.json",
            )
            self.assertEqual(
                map_bundle_arcname("output/glossary_retranslation_result.md"),
                "质量/术语重译执行报告.md",
            )
            self.assertEqual(
                map_bundle_arcname("output/glossary_retranslated_full.md"),
                "译文/术语候选重译全文.md",
            )
            self.assertEqual(
                map_bundle_arcname("output/glossary_retranslation_publish.json"),
                "质量/术语重译发布确认.json",
            )
            self.assertEqual(
                map_bundle_arcname("output/glossary_retranslation_publish.md"),
                "质量/术语重译发布确认.md",
            )
            self.assertEqual(
                map_bundle_arcname("output/glossary_retranslation_published_full.md"),
                "译文/术语重译发布稿.md",
            )
            self.assertEqual(
                map_bundle_arcname("output/glossary_retranslation_rollback.json"),
                "质量/术语重译回滚演练.json",
            )
            self.assertEqual(
                map_bundle_arcname("output/glossary_retranslation_rollback.md"),
                "质量/术语重译回滚演练.md",
            )
            self.assertEqual(
                map_bundle_arcname("output/glossary_retranslation_rollback_full.md"),
                "译文/术语重译回滚演练稿.md",
            )
            self.assertEqual(map_bundle_arcname("output/translated_full.pdf"), "译文/结构化译文.pdf")
            self.assertEqual(map_bundle_arcname("output/translated_pdf_report.json"), "质量/PDF译文生成报告.json")
            self.assertEqual(map_bundle_arcname("output/repaired_full.md"), "译文/局部修复合并译文.md")
            self.assertEqual(map_bundle_arcname("output/repair_plan.md"), "质量/局部修复计划.md")
            self.assertEqual(map_bundle_arcname("output/repair_requests.md"), "质量/局部修复请求.md")
            self.assertEqual(map_bundle_arcname("output/repair_results.md"), "质量/局部修复结果.md")
            self.assertEqual(map_bundle_arcname("output/repair_validation.md"), "质量/局部修复验证.md")
            self.assertEqual(map_bundle_arcname("output/repair_merge.md"), "质量/局部修复合并.md")
            self.assertEqual(map_bundle_arcname("output/repair_patch_review.json"), "质量/局部修复补丁审核.json")
            self.assertEqual(map_bundle_arcname("output/repair_patch_review.md"), "质量/局部修复补丁审核.md")
            self.assertEqual(map_bundle_arcname("output/repair_publish.json"), "质量/局部修复发布确认.json")
            self.assertEqual(map_bundle_arcname("output/repair_publish.md"), "质量/局部修复发布确认.md")
            self.assertEqual(map_bundle_arcname("output/repair_rollback.json"), "质量/局部修复回滚演练.json")
            self.assertEqual(map_bundle_arcname("output/repair_rollback.md"), "质量/局部修复回滚演练.md")
            self.assertEqual(map_bundle_arcname("output/repair_merge_qa.md"), "质量/局部修复后QA.md")
            self.assertEqual(map_bundle_arcname("output/repair_effectiveness.json"), "质量/局部修复效果对比.json")
            self.assertEqual(map_bundle_arcname("output/repair_effectiveness.md"), "质量/局部修复效果对比.md")
            self.assertEqual(map_bundle_arcname("output/published_full.md"), "译文/人工确认修复发布稿.md")
            self.assertEqual(map_bundle_arcname("output/rollback_full.md"), "译文/局部修复回滚演练稿.md")
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
            self.assertEqual(map_bundle_arcname("output/ocr_candidate_qa.json"), "质量/OCR候选文本QA.json")
            self.assertEqual(map_bundle_arcname("output/ocr_candidate_qa.md"), "质量/OCR候选文本QA.md")
            self.assertEqual(map_bundle_arcname("output/document_ir_ocr.json"), "设置/OCR增强文档结构IR.json")
            self.assertEqual(map_bundle_arcname("output/ocr_candidate_promotion.json"), "质量/OCR候选文本晋级.json")
            self.assertEqual(map_bundle_arcname("output/ocr_candidate_promotion.md"), "质量/OCR候选文本晋级.md")
            self.assertEqual(map_bundle_arcname("output/document_ir_promoted.json"), "设置/OCR晋级文档结构IR.json")
            self.assertEqual(map_bundle_arcname("output/structure_qa.json"), "质量/结构QA.json")
            self.assertEqual(map_bundle_arcname("output/table_reconstruction.json"), "质量/表格重建证据.json")
            self.assertEqual(
                map_bundle_arcname("output/table_merged_cell_review.json"),
                "质量/表格合并候选人工确认.json",
            )
            self.assertEqual(
                map_bundle_arcname("output/table_merged_cell_review.md"),
                "质量/表格合并候选人工确认.md",
            )
            self.assertEqual(map_bundle_arcname("output/table_structure_publish.json"), "质量/表格结构确认发布.json")
            self.assertEqual(map_bundle_arcname("output/table_structure_publish.md"), "质量/表格结构确认发布.md")
            self.assertEqual(
                map_bundle_arcname("output/table_reconstruction_confirmed.json"),
                "质量/表格重建确认副本.json",
            )
            self.assertEqual(map_bundle_arcname("output/structure_hints_manifest.json"), "设置/结构提示清单.json")
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
