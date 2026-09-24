from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from reference.evidence_gate import EXPECTED_SPLIT
from runner.build_retrieval_queries import build_retrieval_queries
from runner.materialize_teacher_scores import EXPECTED_CHECKPOINT_SHA256, sha256
from runner.retrieval_runner import load_queries


class RetrievalQueryBuilderTest(unittest.TestCase):
    def inputs(self, root: Path, *, mismatch: bool = False) -> Path:
        raw_rows = []
        teacher_rows = []
        for date_index, date_value in enumerate(EXPECTED_SPLIT["test"]):
            for rank, product in enumerate((101, 102), start=1):
                common = {
                    "snapshot_id": f"snapshot-{date_index}",
                    "snapshot_date": date.fromisoformat(date_value),
                    "user_id": f"user-{date_index}",
                    "product_id": product,
                    "candidate_rank": rank,
                }
                raw_rows.append(
                    {
                        **common,
                        "click_seq_product_id": "102;101",
                        "purchase_seq_product_id": "",
                        "first_category": "category-a",
                        "label_click": int(product == 101),
                        "label_purchase": int(product == 102),
                    }
                )
                teacher_rows.append({**common, "teacher_purchase": 0.1 * rank})
        if mismatch:
            teacher_rows[0]["product_id"] = 999
        raw_path = root / "raw.parquet"
        teacher_path = root / "teacher.parquet"
        pq.write_table(pa.Table.from_pylist(raw_rows), raw_path)
        pq.write_table(pa.Table.from_pylist(teacher_rows), teacher_path)
        manifest = {
            "contract_version": "ptd-frozen-teacher-scores/v1",
            "status": "complete",
            "teacher_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "label_columns_read": [],
            "source_files": [
                {
                    "path": str(raw_path),
                    "sha256": sha256(raw_path),
                    "rows": len(raw_rows),
                }
            ],
            "output": {"path": str(teacher_path), "sha256": sha256(teacher_path)},
        }
        manifest_path = root / "teacher-manifest.json"
        manifest_path.write_text(json.dumps(manifest) + "\n")
        return manifest_path

    def test_builds_five_date_query_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.inputs(root)
            output = root / "queries.jsonl"
            output_manifest = root / "query-manifest.json"
            manifest = build_retrieval_queries(
                teacher_manifest_path=manifest_path,
                output_path=output,
                output_manifest_path=output_manifest,
                batch_size=3,
            )
            queries = load_queries(output)
        self.assertEqual(len(queries), 5)
        self.assertEqual(manifest["queries"], 5)
        self.assertEqual(manifest["test_candidate_rows"], 10)
        self.assertTrue(all(manifest["checks"].values()))

    def test_misaligned_teacher_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.inputs(root, mismatch=True)
            with self.assertRaisesRegex(ValueError, "raw/teacher key mismatch"):
                build_retrieval_queries(
                    teacher_manifest_path=manifest_path,
                    output_path=root / "queries.jsonl",
                    output_manifest_path=root / "query-manifest.json",
                )


if __name__ == "__main__":
    unittest.main()
