from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import jsonschema
except ImportError:  # pragma: no cover - optional schema dependency
    jsonschema = None

from reference.evidence_gate import EXPECTED_SEEDS, EXPECTED_SPLIT, EXPECTED_VARIANTS
from runner.assemble_run_bundle import finalize_run_manifest, lock_retrieval_plan
from runner.build_fit_schedule import (
    ALTERNATING_VARIANTS,
    FIXED_TREE_VARIANTS,
    build_fit_schedule,
)
from runner.retrieval_runner import load_plan

EXPECTED_CHECKPOINT_SHA256 = (
    "5a435e4ea2579ca226f26fd8dfa5ad48a7be016f3d1a8e61798ce1b2d6ed1540"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


class RunBundleAssemblerTest(unittest.TestCase):
    revision = "a" * 40
    selected = {"temperature": 2.0, "lambda_item": 0.3, "lambda_node": 0.3}

    def write(self, path: Path, value: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return path

    def validation_selection(
        self, root: Path, validation_queries_manifest: Path
    ) -> Path:
        path = root / "validation-selection.json"
        payload = {
            "contract_version": "ptd-validation-selection/v1",
            "status": "complete",
            "selection_seed": 16630,
            "selection_dates": EXPECTED_SPLIT["validation"],
            "selection_metric": "purchase_ndcg_at_50",
            "best_single_variant": "ptd_node",
            "validation_queries_manifest_sha256": sha256(
                validation_queries_manifest
            ),
            "selected_hyperparameters": {
                **self.selected,
                "tie_window_absolute_ndcg": 0.001,
                "tie_break": (
                    "within_0.001_of_best_then_lambda_sum_temperature_"
                    "lambda_item_lambda_node_asc"
                ),
            },
            "checks": {
                "exact_27_cell_combined_grid": True,
                "selection_seed_16630_only": True,
                "validation_date_only": True,
                "purchase_positive_macro": True,
                "item_and_node_use_selected_hyperparameters": True,
                "teacher_oracle_not_used": True,
                "test_queries_not_read": True,
                "deterministic_tie_break": True,
                "no_overwrite": True,
            },
        }
        path.write_text(json.dumps(payload) + "\n")
        return path

    def teacher_manifest(self, root: Path) -> Path:
        output = self.write(root / "teacher.parquet", "synthetic teacher rows\n")
        path = root / "teacher-manifest.json"
        payload = {
            "contract_version": "ptd-frozen-teacher-scores/v1",
            "status": "complete",
            "teacher_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "label_columns_read": [],
            "output": artifact(output),
            "checks": {
                "checkpoint_hash_match": True,
                "feature_order_match": True,
                "all_scores_finite_and_bounded": True,
                "row_count_match": True,
                "labels_read": False,
            },
        }
        path.write_text(json.dumps(payload) + "\n")
        return path

    def query_manifest(self, root: Path, teacher_manifest: Path) -> Path:
        output = self.write(root / "queries.jsonl", "synthetic locked queries\n")
        path = root / "query-manifest.json"
        payload = {
            "contract_version": "ptd-retrieval-queries/v1",
            "status": "complete",
            "teacher_manifest_sha256": sha256(teacher_manifest),
            "queries_by_date": {date: 201 for date in EXPECTED_SPLIT["test"]},
            "output": artifact(output),
            "checks": {
                "raw_teacher_keys_match": True,
                "teacher_labels_not_read": True,
                "query_rows_contiguous": True,
                "unique_date_user_units": True,
                "all_test_dates_present": True,
                "labels_read_only_for_evaluation": True,
                "no_overwrite": True,
            },
        }
        path.write_text(json.dumps(payload) + "\n")
        return path

    def fixed_fits(self, root: Path, training_examples_manifest: Path) -> list[Path]:
        paths = []
        for variant in FIXED_TREE_VARIANTS:
            for seed in EXPECTED_SEEDS:
                directory = root / "fits" / variant / str(seed)
                checkpoint = self.write(directory / "checkpoint.pt", f"{variant}/{seed}\n")
                manifest = directory / "fit_manifest.json"
                payload = {
                    "contract_version": "ptd-single-fit/v1",
                    "status": "complete",
                    "variant": variant,
                    "seed": seed,
                    "configuration": dict(self.selected),
                    "training_examples_manifest_sha256": sha256(
                        training_examples_manifest
                    ),
                    "test_only_config_override": False,
                    "initial_fit": None,
                    "checkpoint": artifact(checkpoint),
                    "checks": {
                        "registered_variant": True,
                        "registered_seed": True,
                        "exact_two_epoch_budget": True,
                        "finite_losses": True,
                        "teacher_not_serving_feature": True,
                        "test_examples_absent": True,
                        "checkpoint_no_overwrite": True,
                        "warm_start_contract_valid": True,
                    },
                }
                manifest.write_text(json.dumps(payload) + "\n")
                paths.append(manifest)
        return paths

    def alternating_cycles(
        self,
        root: Path,
        *,
        teacher_manifest: Path,
        assignment_queries_manifest: Path,
        validation_queries_manifest: Path,
        fixed_catalog: Path,
        fixed_eligibility: Path,
    ) -> list[Path]:
        paths = []
        for variant in ALTERNATING_VARIANTS:
            for seed in EXPECTED_SEEDS:
                directory = root / "cycles" / variant / str(seed)
                records = []
                for cycle in range(4):
                    cycle_dir = directory / f"cycle-{cycle}"
                    catalog = self.write(cycle_dir / "catalog.parquet", f"catalog {cycle}\n")
                    eligibility = self.write(
                        cycle_dir / "eligibility.parquet", f"eligibility {cycle}\n"
                    )
                    checkpoint = self.write(
                        cycle_dir / "checkpoint.pt", f"{variant}/{seed}/{cycle}\n"
                    )
                    fit_manifest = self.write(
                        cycle_dir / "fit_manifest.json", f"fit {variant}/{seed}/{cycle}\n"
                    )
                    tree_manifest = self.write(
                        cycle_dir / "tree_manifest.json", f"tree {cycle}\n"
                    )
                    weights = self.write(cycle_dir / "weights.parquet", f"weights {cycle}\n")
                    weights_manifest = self.write(
                        cycle_dir / "weights_manifest.json", f"weight manifest {cycle}\n"
                    )
                    examples = self.write(
                        cycle_dir / "examples_manifest.json", f"examples {cycle}\n"
                    )
                    records.append(
                        {
                            "cycle": cycle,
                            "warm_started_from_cycle": None if cycle == 0 else cycle - 1,
                            "catalog": artifact(catalog),
                            "date_eligibility": artifact(eligibility),
                            "tree_manifest": None if cycle == 0 else artifact(tree_manifest),
                            "assignment_weights": None
                            if cycle == 0
                            else {
                                "weights": artifact(weights),
                                "manifest": artifact(weights_manifest),
                            },
                            "training_examples_manifest": artifact(examples),
                            "fit_manifest": artifact(fit_manifest),
                            "checkpoint": artifact(checkpoint),
                            "state_sha256": hashlib.sha256(
                                f"state/{variant}/{seed}/{cycle}".encode()
                            ).hexdigest(),
                            "validation_purchase_ndcg_at_50": 0.1 + cycle * 0.01,
                            "validation_purchase_positive_units": 5,
                        }
                    )
                selected_cycle = 3
                selected = records[selected_cycle]
                bundle_hash = hashlib.sha256(
                    (
                        "\n".join(
                            selected[key]["sha256"]
                            for key in (
                                "catalog",
                                "date_eligibility",
                                "fit_manifest",
                                "checkpoint",
                            )
                        )
                        + "\n"
                    ).encode()
                ).hexdigest()
                payload = {
                    "contract_version": "ptd-alternating-cycle-selection/v1",
                    "status": "complete",
                    "variant": variant,
                    "seed": seed,
                    "maximum_cycles": 3,
                    "selected_cycle": selected_cycle,
                    "hyperparameters": dict(self.selected),
                    "teacher_manifest_sha256": sha256(teacher_manifest),
                    "assignment_queries_manifest_sha256": sha256(
                        assignment_queries_manifest
                    ),
                    "validation_queries_manifest_sha256": sha256(
                        validation_queries_manifest
                    ),
                    "initial_catalog_sha256": sha256(fixed_catalog),
                    "initial_date_eligibility_sha256": sha256(fixed_eligibility),
                    "cycles": records,
                    "selected": selected,
                    "selected_bundle_sha256": bundle_hash,
                    "checks": {
                        "all_four_cycles_complete": True,
                        "cycle_zero_fixed_tree": True,
                        "cycles_one_to_three_reassigned": True,
                        "model_parameters_warm_started": True,
                        "optimizers_reset_each_fit": True,
                        "validation_only_cycle_selection": True,
                        "lower_cycle_exact_tie_break": True,
                        "test_queries_not_read": True,
                        "selected_bundle_locked": True,
                        "no_overwrite": True,
                    },
                }
                manifest = directory / "manifest.json"
                manifest.write_text(json.dumps(payload) + "\n")
                paths.append(manifest)
        return paths

    def test_builds_schedule_locks_plan_and_finalizes_run_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = self.teacher_manifest(root)
            training_examples = self.write(
                root / "training-examples-manifest.json", "training examples\n"
            )
            validation_queries = self.write(
                root / "validation-queries-manifest.json", "validation queries\n"
            )
            assignment_queries = self.write(
                root / "assignment-queries-manifest.json", "assignment queries\n"
            )
            catalog = self.write(root / "fixed-catalog.parquet", "fixed catalog\n")
            eligibility = self.write(
                root / "fixed-eligibility.parquet", "fixed eligibility\n"
            )
            schedule_path = root / "fit-schedule.json"
            schedule = build_fit_schedule(
                code_revision=self.revision,
                output_root=root / "outputs",
                teacher_scores_manifest_path=teacher,
                training_examples_manifest_path=training_examples,
                validation_queries_manifest_path=validation_queries,
                assignment_queries_manifest_path=assignment_queries,
                fixed_catalog_path=catalog,
                fixed_date_eligibility_path=eligibility,
                output_path=schedule_path,
            )
            self.assertEqual(schedule["counts"]["total_fit_executions"], 65)
            self.assertEqual(
                sum(
                    task["action"] == "run_selected_hyperparameters"
                    for task in schedule["final_fixed_tree_models"]
                ),
                12,
            )
            selection = self.validation_selection(root, validation_queries)
            queries = self.query_manifest(root, teacher)
            plan_path = root / "retrieval-plan.json"
            plan = lock_retrieval_plan(
                code_revision=self.revision,
                created_at="2026-09-25T10:00:00+09:00",
                locked_at="2026-09-25T11:00:00+09:00",
                fit_schedule_path=schedule_path,
                validation_selection_path=selection,
                teacher_scores_manifest_path=teacher,
                retrieval_queries_manifest_path=queries,
                fixed_catalog_path=catalog,
                fixed_date_eligibility_path=eligibility,
                fit_manifest_paths=self.fixed_fits(root, training_examples),
                alternating_cycle_manifest_paths=self.alternating_cycles(
                    root,
                    teacher_manifest=teacher,
                    assignment_queries_manifest=assignment_queries,
                    validation_queries_manifest=validation_queries,
                    fixed_catalog=catalog,
                    fixed_eligibility=eligibility,
                ),
                device="cuda",
                hardware="NVIDIA L4 on g2-standard-16",
                software="registered container image",
                timer="CUDA synchronized perf_counter_ns",
                output_path=plan_path,
            )
            self.assertEqual(len(plan["entries"]), 24)
            self.assertTrue(all(plan["checks"].values()))
            self.assertEqual(load_plan(plan_path), plan)
            metrics_rows = self.write(root / "retrieval-metrics.jsonl", "synthetic\n")
            metrics_manifest = root / "retrieval-metrics-manifest.json"
            metrics = {
                "contract_version": "ptd-retrieval-metrics/v1",
                "status": "complete",
                "plan_sha256": sha256(plan_path),
                "queries_sha256": plan["queries"]["sha256"],
                "test_dates": EXPECTED_SPLIT["test"],
                "seeds": EXPECTED_SEEDS,
                "variants": list(EXPECTED_VARIANTS),
                "measured_rows_by_variant": {
                    variant: 1005 for variant in EXPECTED_VARIANTS
                },
                "output": artifact(metrics_rows),
                "checks": {
                    "all_variant_seed_entries_complete": True,
                    "minimum_latency_queries_per_variant": True,
                    "candidate_sets_match_date_masks": True,
                    "concurrency_one": True,
                    "teacher_score_used_only_by_oracle": True,
                    "labels_used_only_after_retrieval": True,
                    "no_overwrite": True,
                },
            }
            metrics_manifest.write_text(json.dumps(metrics) + "\n")
            run_path = root / "run-manifest.json"
            run = finalize_run_manifest(
                retrieval_plan_path=plan_path,
                retrieval_metrics_manifest_path=metrics_manifest,
                run_id="synthetic-five-day-run",
                test_scoring_started_at="2026-09-25T12:00:00+09:00",
                completed_at="2026-09-25T13:00:00+09:00",
                output_path=run_path,
            )
            self.assertEqual(run["status"], "complete")
            self.assertEqual(
                run["execution"]["retrieval_plan"]["sha256"], sha256(plan_path)
            )
            self.assertEqual(
                set(run["variants"]), set(EXPECTED_VARIANTS)
            )
            if jsonschema is not None:
                for schema_name, payload in (
                    ("fit_schedule.schema.json", schedule),
                    ("retrieval_run_plan.schema.json", plan),
                    ("run_manifest.schema.json", run),
                ):
                    schema = json.loads(
                        (Path(__file__).parents[1] / "artifact" / schema_name).read_text()
                    )
                    jsonschema.Draft202012Validator(schema).validate(payload)
            with self.assertRaises(FileExistsError):
                finalize_run_manifest(
                    retrieval_plan_path=plan_path,
                    retrieval_metrics_manifest_path=metrics_manifest,
                    run_id="synthetic-five-day-run",
                    test_scoring_started_at="2026-09-25T12:00:00+09:00",
                    completed_at="2026-09-25T13:00:00+09:00",
                    output_path=run_path,
                )


if __name__ == "__main__":
    unittest.main()
