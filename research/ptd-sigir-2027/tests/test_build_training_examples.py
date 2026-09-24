from __future__ import annotations

import json
import math
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from reference.evidence_gate import EXPECTED_SPLIT
from runner.build_training_examples import build_training_examples
from runner.materialize_teacher_scores import EXPECTED_CHECKPOINT_SHA256, sha256


class TrainingExampleBuilderTest(unittest.TestCase):
    def inputs(self, root: Path, *, mismatch: bool = False) -> tuple[Path, Path, Path]:
        dates = (
            EXPECTED_SPLIT["train"]
            + EXPECTED_SPLIT["validation"]
            + EXPECTED_SPLIT["test"]
        )
        products = (101, 102, 103, 104)
        depth = 13
        leaf_start = 2**depth - 1
        catalog_path = root / "catalog.parquet"
        eligibility_path = root / "eligibility.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "product_id": product,
                        "first_category": "category-a" if product < 103 else "category-b",
                        "leaf_node_id": leaf_start + index,
                        "path_bits": format(index, f"0{depth}b"),
                    }
                    for index, product in enumerate(products)
                ]
            ),
            catalog_path,
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "snapshot_date": date.fromisoformat(snapshot_date),
                        "product_id": product,
                        "leaf_node_id": leaf_start + index,
                    }
                    for snapshot_date in dates
                    for index, product in enumerate(products)
                ]
            ),
            eligibility_path,
        )

        sources = []
        teacher_rows = []
        for date_index, snapshot_date in enumerate(dates):
            raw_rows = []
            for rank, product in enumerate(products, start=1):
                common = {
                    "snapshot_id": f"snapshot-{date_index}",
                    "snapshot_date": date.fromisoformat(snapshot_date),
                    "user_id": f"user-{date_index}",
                    "product_id": product,
                    "candidate_rank": rank,
                }
                raw_rows.append(
                    {
                        **common,
                        "click_seq_product_id": "104;101",
                        "purchase_seq_product_id": "102",
                        "first_category": "category-a" if product < 103 else "category-b",
                        # Invalid test values prove those columns are never opened.
                        "label_purchase": 99
                        if snapshot_date in EXPECTED_SPLIT["test"]
                        else int(product == 101),
                    }
                )
                teacher_rows.append(
                    {
                        **common,
                        "teacher_purchase": 0.1 * rank,
                    }
                )
            raw_path = root / f"raw-{date_index}.parquet"
            pq.write_table(pa.Table.from_pylist(raw_rows), raw_path)
            sources.append(
                {"path": str(raw_path), "sha256": sha256(raw_path), "rows": len(raw_rows)}
            )
        if mismatch:
            teacher_rows[0]["product_id"] = 999
        teacher_path = root / "teacher.parquet"
        pq.write_table(pa.Table.from_pylist(teacher_rows), teacher_path)
        teacher_manifest = {
            "contract_version": "ptd-frozen-teacher-scores/v1",
            "status": "complete",
            "teacher_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "label_columns_read": [],
            "source_files": sources,
            "output": {"path": str(teacher_path), "sha256": sha256(teacher_path)},
        }
        manifest_path = root / "teacher-manifest.json"
        manifest_path.write_text(json.dumps(teacher_manifest) + "\n")
        return manifest_path, catalog_path, eligibility_path

    def test_builds_complete_paths_without_reading_test_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher, catalog, eligibility = self.inputs(root)
            output = root / "training.parquet"
            output_manifest = root / "training-manifest.json"
            manifest = build_training_examples(
                teacher_manifest_path=teacher,
                catalog_path=catalog,
                date_eligibility_path=eligibility,
                output_path=output,
                output_manifest_path=output_manifest,
                batch_size=3,
            )
            rows = pq.read_table(output).to_pylist()

        expected_paths = len(EXPECTED_SPLIT["train"] + EXPECTED_SPLIT["validation"])
        self.assertEqual(manifest["path_groups"], expected_paths)
        self.assertEqual(manifest["sibling_rows"], expected_paths * 13)
        self.assertTrue(all(manifest["checks"].values()))
        self.assertTrue(
            all(
                not source["outcome_columns_read"]
                for source in manifest["source_files"]
                if source["snapshot_date"] in EXPECTED_SPLIT["test"]
            )
        )
        for row in rows:
            self.assertTrue(math.isfinite(row["teacher_left"]))
            self.assertTrue(math.isfinite(row["teacher_right"]))
            self.assertAlmostEqual(row["teacher_left"] + row["teacher_right"], 1.0)
        leaf_row = next(row for row in rows if row["level"] == 13)
        self.assertEqual(leaf_row["positive_child"], 0)
        self.assertEqual(leaf_row["left_item_key"], "101")
        self.assertEqual(leaf_row["right_item_key"], "102")
        expected_left = (0.1 / 0.9) / ((0.1 / 0.9) + (0.2 / 0.8))
        self.assertAlmostEqual(leaf_row["teacher_left"], expected_left)

    def test_rejects_aligned_teacher_key_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher, catalog, eligibility = self.inputs(root, mismatch=True)
            with self.assertRaisesRegex(ValueError, "raw/teacher key mismatch"):
                build_training_examples(
                    teacher_manifest_path=teacher,
                    catalog_path=catalog,
                    date_eligibility_path=eligibility,
                    output_path=root / "training.parquet",
                    output_manifest_path=root / "training-manifest.json",
                )


if __name__ == "__main__":
    unittest.main()
