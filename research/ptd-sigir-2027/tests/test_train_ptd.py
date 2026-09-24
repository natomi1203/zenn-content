from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import torch

    from runner.materialize_teacher_scores import sha256
    from runner.ptd_model import PTDModelConfig
    from runner.train_ptd import train_from_examples
except ImportError:  # pragma: no cover - optional production dependency
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional production dependency")
class TrainPTDTest(unittest.TestCase):
    def examples(self, root: Path) -> Path:
        rows = []
        dates = ("2026-07-18", "2026-07-19", "2026-07-21")
        for group_id, snapshot_date in enumerate(dates):
            for level in range(1, 14):
                rows.append(
                    {
                        "snapshot_date": date.fromisoformat(snapshot_date),
                        "snapshot_id": f"snapshot-{group_id}",
                        "user_id": f"user-{group_id}",
                        "positive_product_id": 101,
                        "path_group_id": group_id,
                        "level": level,
                        "positive_child": level % 2,
                        "click_seq_product_id": "103;102;101",
                        "purchase_seq_product_id": "101",
                        "left_item_key": f"node:{level * 2}",
                        "right_item_key": f"node:{level * 2 + 1}",
                        "left_category": "__internal__" if level < 13 else "category-a",
                        "right_category": "__internal__" if level < 13 else "category-b",
                        "parent_node_id": level,
                        "teacher_left": 0.7,
                        "teacher_right": 0.3,
                        "distillation_kind": 1 if level == 13 else 2,
                    }
                )
        examples_path = root / "examples.parquet"
        pq.write_table(pa.Table.from_pylist(rows), examples_path)
        manifest = {
            "contract_version": "ptd-training-examples/v1",
            "status": "complete",
            "output": {"path": str(examples_path), "sha256": sha256(examples_path)},
            "checks": {
                "raw_teacher_keys_match_for_included_dates": True,
                "candidate_sets_match_locked_date_masks": True,
                "complete_depth_13_paths": True,
                "teacher_targets_finite_and_normalized": True,
                "test_outcome_columns_excluded": True,
                "test_teacher_scores_excluded": True,
                "no_overwrite": True,
            },
        }
        manifest_path = root / "examples-manifest.json"
        manifest_path.write_text(json.dumps(manifest) + "\n")
        return manifest_path

    def test_trains_two_epochs_and_writes_immutable_checkpoint(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = self.examples(root)
            checkpoint = root / "checkpoint.pt"
            output_manifest = root / "fit-manifest.json"
            output = train_from_examples(
                examples_manifest_path=examples,
                variant="ptd_combined",
                seed=16630,
                temperature=2.0,
                lambda_item=0.3,
                lambda_node=0.3,
                checkpoint_path=checkpoint,
                output_manifest_path=output_manifest,
                device="cpu",
                config=PTDModelConfig(
                    item_hash_bucket_size=64,
                    user_hash_bucket_size=32,
                    category_hash_bucket_size=16,
                ),
                test_only_config_override=True,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["variant"], "ptd_combined")
            self.assertEqual(payload["seed"], 16630)
            self.assertEqual(len(payload["history"]), 2)
            self.assertTrue(all(output["checks"].values()))
            self.assertTrue(output["test_only_config_override"])
            with self.assertRaises(FileExistsError):
                train_from_examples(
                    examples_manifest_path=examples,
                    variant="ptd_combined",
                    seed=16630,
                    temperature=2.0,
                    lambda_item=0.3,
                    lambda_node=0.3,
                    checkpoint_path=checkpoint,
                    output_manifest_path=output_manifest,
                    device="cpu",
                    config=PTDModelConfig(
                        item_hash_bucket_size=64,
                        user_hash_bucket_size=32,
                        category_hash_bucket_size=16,
                    ),
                    test_only_config_override=True,
                )


if __name__ == "__main__":
    unittest.main()
