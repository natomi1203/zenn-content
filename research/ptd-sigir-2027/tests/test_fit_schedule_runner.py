from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runner.build_fit_schedule import build_fit_schedule
from runner.materialize_teacher_scores import sha256
from runner.run_fit_schedule import run_fit_schedule

try:
    import jsonschema
except ImportError:  # pragma: no cover - optional schema dependency
    jsonschema = None


class FitScheduleRunnerTest(unittest.TestCase):
    revision = "a" * 40
    selected = (1.0, 0.1, 0.1)

    def write(self, path: Path, value: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return path

    def schedule(self, root: Path) -> Path:
        inputs = {
            name: self.write(root / "inputs" / f"{name}.data", f"{name}\n")
            for name in (
                "teacher_scores_manifest",
                "training_examples_manifest",
                "validation_queries_manifest",
                "assignment_queries_manifest",
                "fixed_catalog",
                "fixed_date_eligibility",
            )
        }
        path = root / "fit-schedule.json"
        build_fit_schedule(
            code_revision=self.revision,
            output_root=root / "outputs",
            teacher_scores_manifest_path=inputs["teacher_scores_manifest"],
            training_examples_manifest_path=inputs["training_examples_manifest"],
            validation_queries_manifest_path=inputs["validation_queries_manifest"],
            assignment_queries_manifest_path=inputs["assignment_queries_manifest"],
            fixed_catalog_path=inputs["fixed_catalog"],
            fixed_date_eligibility_path=inputs["fixed_date_eligibility"],
            output_path=path,
        )
        return path

    def test_executes_exact_registered_fit_dag_before_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = self.schedule(root)
            train_calls: list[dict] = []
            preselect_counts: list[int] = []
            select_counts: list[int] = []
            alternating_calls: list[tuple[str, int]] = []

            def fake_train(**kwargs):
                train_calls.append(kwargs)
                checkpoint = kwargs["checkpoint_path"]
                manifest = kwargs["output_manifest_path"]
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(
                    f'{kwargs["variant"]}/{kwargs["seed"]}/{len(train_calls)}\n'
                )
                payload = {
                    "contract_version": "ptd-single-fit/v1",
                    "status": "complete",
                    "variant": kwargs["variant"],
                    "seed": kwargs["seed"],
                    "test_only_config_override": False,
                    "configuration": {
                        "temperature": kwargs["temperature"],
                        "lambda_item": kwargs["lambda_item"],
                        "lambda_node": kwargs["lambda_node"],
                    },
                    "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
                    "checks": {"complete": True},
                }
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text(json.dumps(payload) + "\n")
                return payload

            def fake_preselect(**kwargs):
                preselect_counts.append(len(kwargs["fit_manifest_paths"]))
                return {
                    "selected_key": self.selected,
                    "grid_scores": [],
                    "purchase_positive_units": 1,
                }

            def fake_select(**kwargs):
                manifests = kwargs["fit_manifest_paths"]
                select_counts.append(len(manifests))
                fits = [json.loads(path.read_text()) | {"path": path} for path in manifests]
                grids = [value for value in fits if value["variant"] == "ptd_combined"]
                singles = [value for value in fits if value["variant"] in {"ptd_item", "ptd_node"}]
                payload = {
                    "contract_version": "ptd-validation-selection/v1",
                    "status": "complete",
                    "selected_hyperparameters": {
                        "temperature": self.selected[0],
                        "lambda_item": self.selected[1],
                        "lambda_node": self.selected[2],
                    },
                    "grid_scores": [
                        {
                            **value["configuration"],
                            "fit_manifest": {
                                "path": str(value["path"]),
                                "sha256": sha256(value["path"]),
                            },
                        }
                        for value in grids
                    ],
                    "single_variant_scores": [
                        {
                            "variant": value["variant"],
                            "fit_manifest": {
                                "path": str(value["path"]),
                                "sha256": sha256(value["path"]),
                            },
                        }
                        for value in singles
                    ],
                    "best_single_variant": "ptd_node",
                    "checks": {"complete": True},
                }
                kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
                kwargs["output_path"].write_text(json.dumps(payload) + "\n")
                return payload

            def fake_alternating(**kwargs):
                identity = (kwargs["variant"], kwargs["seed"])
                alternating_calls.append(identity)
                output_dir = kwargs["output_dir"]
                output_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "contract_version": "ptd-alternating-cycle-selection/v1",
                    "status": "complete",
                    "variant": identity[0],
                    "seed": identity[1],
                    "selected_cycle": 0,
                    "checks": {"complete": True},
                }
                (output_dir / "manifest.json").write_text(json.dumps(payload) + "\n")
                return payload

            selection = root / "validation-selection.json"
            output = root / "fit-execution.json"
            with (
                patch("runner.run_fit_schedule._train_from_examples", fake_train),
                patch("runner.run_fit_schedule.preselect_combined_grid", fake_preselect),
                patch("runner.run_fit_schedule.select_validation", fake_select),
                patch("runner.run_fit_schedule._run_alternating_cycles", fake_alternating),
            ):
                result = run_fit_schedule(
                    fit_schedule_path=schedule,
                    validation_selection_path=selection,
                    output_path=output,
                    device="cpu",
                )

            self.assertEqual(len(train_calls), 41)
            self.assertEqual(preselect_counts, [27])
            self.assertEqual(select_counts, [29])
            self.assertEqual(len(alternating_calls), 6)
            self.assertEqual(result["counts"]["total_fit_executions"], 65)
            self.assertEqual(len(result["final_fixed_tree_fit_manifests"]), 15)
            self.assertEqual(len(result["alternating_cycle_manifests"]), 6)
            self.assertTrue(all(result["checks"].values()))
            if jsonschema is not None:
                schema = json.loads(
                    (Path(__file__).parents[1] / "artifact" / "fit_execution.schema.json").read_text()
                )
                jsonschema.Draft202012Validator(schema).validate(result)
            output.unlink()
            with (
                patch("runner.run_fit_schedule._train_from_examples", fake_train),
                patch("runner.run_fit_schedule.preselect_combined_grid", fake_preselect),
                patch("runner.run_fit_schedule.select_validation", fake_select),
                patch("runner.run_fit_schedule._run_alternating_cycles", fake_alternating),
            ):
                resumed = run_fit_schedule(
                    fit_schedule_path=schedule,
                    validation_selection_path=selection,
                    output_path=output,
                    device="cpu",
                )
            self.assertEqual(len(train_calls), 41)
            self.assertEqual(preselect_counts, [27, 27])
            self.assertEqual(select_counts, [29])
            self.assertEqual(len(alternating_calls), 6)
            self.assertTrue(resumed["checks"]["complete_outputs_reusable_on_restart"])
            with self.assertRaises(FileExistsError):
                run_fit_schedule(
                    fit_schedule_path=schedule,
                    validation_selection_path=selection,
                    output_path=output,
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
