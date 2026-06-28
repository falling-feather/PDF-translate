from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import fitz

from pdf_translate.chunkers.structure import build_structure_chunks
from pdf_translate.config import AppConfig
from pdf_translate.extractors.document_ir import BlockIR, DocumentIR, PageIR, classify_text_block
from pdf_translate.pipeline import init_workdir, run_split, run_translate


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
                        BlockIR("p1-b0001", 1, "table", "Metric Acc F1\nA 91 88", (0, 30, 300, 90), 1),
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
        self.assertEqual(chunks[1].pages_0based, [1])

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
            self.assertTrue(ir_path.is_file())
            self.assertTrue(manifest_path.is_file())
            ir = json.loads(ir_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(ir["schema_version"], "document-ir-v1")
            self.assertGreaterEqual(len(ir["pages"]), 1)
            self.assertGreaterEqual(len(manifest), 1)
            self.assertIn("block_ids", manifest[0])
        finally:
            if root.exists():
                shutil.rmtree(root)
            parent = root.parent
            if parent.is_dir() and not any(parent.iterdir()):
                shutil.rmtree(parent)


if __name__ == "__main__":
    unittest.main()
