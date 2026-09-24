from __future__ import annotations

import json
import math
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import jsonschema
except ImportError:  # pragma: no cover - optional schema-validation dependency
    jsonschema = None

from reference.evidence_gate import EXPECTED_SPLIT
from runner.build_assignment_queries import build_assignment_queries
from runner.materialize_teacher_scores import EXPECTED_CHECKPOINT_SHA256, sha256
from runner.retrieval_runner import load_queries

try:
    import torch

    from runner.alternating_solver import build_alternating_bundle, validate_weight_manifest
    from runner.materialize_assignment_weights import materialize_assignment_weights
except ImportError:  # pragma: no cover - optional production dependencies
    torch = None


class AssignmentFixture:
    @staticmethod
    def inputs(root: Path) -> tuple[Path, Path, Path]:
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
                        "first_registered_date": date.fromisoformat("2026-07-18"),
                        "catalog_index": index,
                        "path_bits": format(index, f"0{depth}b"),
                        "leaf_node_id": leaf_start + index,
                        "train_seen": product < 103,
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
                        "catalog_index": index,
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
            is_train = snapshot_date in EXPECTED_SPLIT["train"]
            is_validation = snapshot_date in EXPECTED_SPLIT["validation"]
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
                        "label_click": int(product == 102)
                        if (is_train or is_validation)
                        else 99,
                        "label_purchase": int(product == 101)
                        if (is_train or is_validation)
                        else 99,
                    }
                )
                teacher_rows.append({**common, "teacher_purchase": 0.1 * rank})
            raw_path = root / f"raw-{date_index}.parquet"
            pq.write_table(pa.Table.from_pylist(raw_rows), raw_path)
            sources.append(
                {"path": str(raw_path), "sha256": sha256(raw_path), "rows": len(raw_rows)}
            )
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
        teacher_manifest_path = root / "teacher-manifest.json"
        teacher_manifest_path.write_text(json.dumps(teacher_manifest) + "\n")
        return teacher_manifest_path, catalog_path, eligibility_path


class AssignmentQueryBuilderTest(unittest.TestCase):
    def test_builds_train_only_queries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher, catalog, eligibility = AssignmentFixture.inputs(root)
            output = root / "assignment-queries.jsonl"
            manifest_path = root / "assignment-query-manifest.json"
            manifest = build_assignment_queries(
                teacher_manifest_path=teacher,
                catalog_path=catalog,
                date_eligibility_path=eligibility,
                output_path=output,
                output_manifest_path=manifest_path,
                batch_size=3,
            )
            queries = load_queries(output, allowed_dates=EXPECTED_SPLIT["train"])
            if jsonschema is not None:
                schema = json.loads(
                    (Path(__file__).parents[1] / "artifact" / "assignment_query_row.schema.json").read_text()
                )
                validator = jsonschema.Draft202012Validator(schema)
                for line in output.read_text().splitlines():
                    validator.validate(json.loads(line))
        self.assertEqual(len(queries), 3)
        self.assertEqual(manifest["candidate_rows"], 12)
        self.assertTrue(all(manifest["checks"].values()))
        self.assertTrue(
            all(
                source["outcome_columns_read"]
                == (source["snapshot_date"] in EXPECTED_SPLIT["train"])
                for source in manifest["source_files"]
            )
        )


class UniformBackend:
    def score_sibling_pairs(self, *, pairs, **_kwargs):
        return [(0.0, 0.0)] * len(pairs)

    def synchronize(self) -> None:
        return None


@unittest.skipIf(torch is None, "PyTorch/SciPy are optional production dependencies")
class AssignmentWeightMaterializerTest(unittest.TestCase):
    def test_materializes_complete_matrix_and_hands_off_to_exact_solver(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher, catalog, eligibility = AssignmentFixture.inputs(root)
            queries = root / "assignment-queries.jsonl"
            query_manifest = root / "assignment-query-manifest.json"
            build_assignment_queries(
                teacher_manifest_path=teacher,
                catalog_path=catalog,
                date_eligibility_path=eligibility,
                output_path=queries,
                output_manifest_path=query_manifest,
                batch_size=3,
            )
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"synthetic-checkpoint")
            fit_manifest = root / "fit-manifest.json"
            fit_manifest.write_text(
                json.dumps(
                    {
                        "contract_version": "ptd-single-fit/v1",
                        "status": "complete",
                        "variant": "alternating_ptd",
                        "seed": 16630,
                        "test_only_config_override": True,
                        "checkpoint": {
                            "path": str(checkpoint),
                            "sha256": sha256(checkpoint),
                        },
                        "checks": {"complete": True},
                    }
                )
                + "\n"
            )
            weights = root / "weights.parquet"
            weight_manifest = root / "weight-manifest.json"
            manifest = materialize_assignment_weights(
                assignment_queries_manifest_path=query_manifest,
                catalog_path=catalog,
                date_eligibility_path=eligibility,
                fit_manifest_path=fit_manifest,
                output_path=weights,
                output_manifest_path=weight_manifest,
                device="cpu",
                query_block_size=2,
                score_chunk_size=1024,
                backend_factory=lambda _fit, _tree, _device: UniformBackend(),
                allow_test_only_fit=True,
            )
            table = pq.read_table(weights)
            self.assertEqual(manifest["matrix_shape"], [2, 8190])
            self.assertEqual(table.num_rows, 2 * 8190)
            expected_path = -13.0 * math.log(2.0)
            values = table.to_pydict()
            product_101 = [
                value
                for product, value in zip(
                    values["product_id"], values["assignment_weight"], strict=True
                )
                if product == 101
            ]
            product_102 = [
                value
                for product, value in zip(
                    values["product_id"], values["assignment_weight"], strict=True
                )
                if product == 102
            ]
            self.assertTrue(all(abs(value - 3.3 * expected_path) < 1e-10 for value in product_101))
            self.assertTrue(all(abs(value - 0.6 * expected_path) < 1e-10 for value in product_102))
            validate_weight_manifest(
                json.loads(weight_manifest.read_text()),
                weights_path=weights,
                catalog_path=catalog,
            )
            alternating = build_alternating_bundle(
                catalog_path=catalog,
                date_eligibility_path=eligibility,
                weights_path=weights,
                weight_manifest_path=weight_manifest,
                output_dir=root / "alternating",
                depth=13,
            )
            self.assertTrue(all(alternating["checks"].values()))
            if jsonschema is not None:
                schema = json.loads(
                    (Path(__file__).parents[1] / "artifact" / "assignment_weights.schema.json").read_text()
                )
                jsonschema.Draft202012Validator(schema).validate(manifest)
            with self.assertRaises(FileExistsError):
                materialize_assignment_weights(
                    assignment_queries_manifest_path=query_manifest,
                    catalog_path=catalog,
                    date_eligibility_path=eligibility,
                    fit_manifest_path=fit_manifest,
                    output_path=weights,
                    output_manifest_path=weight_manifest,
                    device="cpu",
                    backend_factory=lambda _fit, _tree, _device: UniformBackend(),
                    allow_test_only_fit=True,
                )


if __name__ == "__main__":
    unittest.main()
