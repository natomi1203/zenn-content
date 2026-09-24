from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import json

try:
    import jsonschema
except ImportError:  # pragma: no cover - optional schema-validation dependency
    jsonschema = None

try:
    import torch

    from runner.build_assignment_queries import build_assignment_queries
    from runner.build_validation_queries import build_validation_queries
    from runner.ptd_model import PTDModelConfig
    from runner.run_alternating_cycles import run_alternating_cycles
    from tests.test_assignment_weights import AssignmentFixture
except ImportError:  # pragma: no cover - optional production dependencies
    torch = None


@unittest.skipIf(torch is None, "PyTorch/SciPy are optional production dependencies")
class AlternatingCycleOrchestratorTest(unittest.TestCase):
    def test_runs_four_warm_started_cycles_and_locks_validation_winner(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher, catalog, eligibility = AssignmentFixture.inputs(root)
            assignment_queries = root / "assignment-queries.jsonl"
            assignment_manifest = root / "assignment-query-manifest.json"
            build_assignment_queries(
                teacher_manifest_path=teacher,
                catalog_path=catalog,
                date_eligibility_path=eligibility,
                output_path=assignment_queries,
                output_manifest_path=assignment_manifest,
                batch_size=3,
            )
            validation_queries = root / "validation-queries.jsonl"
            validation_manifest = root / "validation-query-manifest.json"
            build_validation_queries(
                teacher_manifest_path=teacher,
                catalog_path=catalog,
                date_eligibility_path=eligibility,
                output_path=validation_queries,
                output_manifest_path=validation_manifest,
                batch_size=3,
            )
            output_dir = root / "cycles"
            output = run_alternating_cycles(
                teacher_manifest_path=teacher,
                assignment_queries_manifest_path=assignment_manifest,
                validation_queries_manifest_path=validation_manifest,
                initial_catalog_path=catalog,
                initial_date_eligibility_path=eligibility,
                variant="alternating_ptd",
                seed=16630,
                temperature=2.0,
                lambda_item=0.3,
                lambda_node=0.3,
                output_dir=output_dir,
                device="cpu",
                model_config=PTDModelConfig(
                    item_hash_bucket_size=64,
                    user_hash_bucket_size=32,
                    category_hash_bucket_size=16,
                ),
                test_only_config_override=True,
            )
            self.assertEqual(len(output["cycles"]), 4)
            self.assertIn(output["selected_cycle"], range(4))
            self.assertTrue(all(output["checks"].values()))
            self.assertIsNone(output["cycles"][0]["warm_started_from_cycle"])
            self.assertTrue(
                all(
                    record["warm_started_from_cycle"] == record["cycle"] - 1
                    for record in output["cycles"][1:]
                )
            )
            self.assertTrue((output_dir / "manifest.json").is_file())
            if jsonschema is not None:
                schema = json.loads(
                    (
                        Path(__file__).parents[1]
                        / "artifact"
                        / "alternating_cycle_selection.schema.json"
                    ).read_text()
                )
                jsonschema.Draft202012Validator(schema).validate(output)
            with self.assertRaises(FileExistsError):
                run_alternating_cycles(
                    teacher_manifest_path=teacher,
                    assignment_queries_manifest_path=assignment_manifest,
                    validation_queries_manifest_path=validation_manifest,
                    initial_catalog_path=catalog,
                    initial_date_eligibility_path=eligibility,
                    variant="alternating_ptd",
                    seed=16630,
                    temperature=2.0,
                    lambda_item=0.3,
                    lambda_node=0.3,
                    output_dir=output_dir,
                    device="cpu",
                    model_config=PTDModelConfig(
                        item_hash_bucket_size=64,
                        user_hash_bucket_size=32,
                        category_hash_bucket_size=16,
                    ),
                    test_only_config_override=True,
                )


if __name__ == "__main__":
    unittest.main()
