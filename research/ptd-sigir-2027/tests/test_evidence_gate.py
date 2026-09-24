from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from reference.evaluation import (
    PairedScoreRow,
    guardrail_report,
    primary_metric_summaries,
    registered_primary_contrasts,
)
from reference.evidence_gate import (
    ASSERTIONS,
    EXPECTED_SEEDS,
    EXPECTED_SPLIT,
    EXPECTED_VARIANTS,
    METRIC_BOUNDS,
    SOURCE_CONTRACT_SHA256,
    SOURCE_MANIFEST_SHA256,
    admission_report,
)


def metrics() -> dict[str, float]:
    return {
        name: 10.0 if name.startswith("latency_") or name == "candidates_scored_mean" else 0.2
        for name in METRIC_BOUNDS
    }


def paired_score_rows() -> list[dict]:
    rows: list[dict] = []
    for date in EXPECTED_SPLIT["test"]:
        for user_index, user in enumerate(("u1", "u2")):
            for seed in EXPECTED_SEEDS:
                baseline = 0.20 + user_index * 0.01 + (seed - EXPECTED_SEEDS[0]) * 0.001
                rows.append(
                    {
                        "date": date,
                        "user_id": user,
                        "seed": seed,
                        "scores": {
                            "fixed_tdm": baseline,
                            "ptd_item": baseline + 0.01,
                            "ptd_node": baseline + 0.015,
                            "ptd_combined": baseline + 0.02,
                            "alternating_tdm": baseline + 0.005,
                            "alternating_ptd": baseline + 0.025,
                            "ptd_combined_baseline_encoder": baseline + 0.012,
                            "teacher_oracle": baseline + 0.04,
                        },
                    }
                )
    return rows


def write_paired_score_rows(path: Path) -> list[PairedScoreRow]:
    values = paired_score_rows()
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))
    return [
        PairedScoreRow(
            date=value["date"],
            user_id=value["user_id"],
            seed=value["seed"],
            scores=value["scores"],
        )
        for value in values
    ]


def valid_run() -> dict:
    artifact = {"uri": "gs://example.invalid/artifact", "sha256": "b" * 64}
    return {
        "schema_version": "ptd-run-manifest/v1",
        "status": "complete",
        "run_id": "prospective-five-day-example",
        "created_at": "2026-09-24T00:00:00Z",
        "test_scoring_started_at": "2026-09-24T02:00:00Z",
        "code_revision": "a" * 40,
        "source_contract": {
            "contract_version": "shared_bottom_esmm_v2_source",
            "manifest_uri": "gs://example.invalid/source/manifest.json",
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        },
        "teacher": {
            "teacher_id": "legacy_loss_shared_bottom_esmm_v2",
            "frozen": True,
            "score": "pCTR*pCVR",
            "artifact": artifact,
        },
        "split": copy.deepcopy(EXPECTED_SPLIT),
        "seeds": list(EXPECTED_SEEDS),
        "assertions": {name: True for name in ASSERTIONS},
        "variants": {
            variant: {
                "status": "complete",
                "seed_artifacts": {str(seed): artifact for seed in EXPECTED_SEEDS},
            }
            for variant in EXPECTED_VARIANTS
        },
        "selected_hyperparameters": {
            "selection_seed": 16630,
            "temperature": 2.0,
            "lambda_item": 0.3,
            "lambda_node": 0.3,
            "epsilon_item": 1e-6,
            "epsilon_node": 1e-12,
            "assignment_weight": "y_plus_teacher_times_path_log_probability",
            "selection_artifact": artifact,
        },
        "validation_selection": {
            "selection_seed": 16630,
            "selection_metric": "purchase_ndcg_at_50",
            "best_single_variant": "ptd_node",
            "artifact": artifact,
        },
        "tree": {
            "branching_factor": 2,
            "depth": 12,
            "leaf_capacity": 1,
            "beam_width": 50,
            "alternating_cycles_selected": 1,
            "locked_before_test": True,
            "locked_at": "2026-09-24T01:00:00Z",
            "locked_tree_sha256": "c" * 64,
        },
        "latency_protocol": {
            "baseline_variant": "fixed_tdm",
            "p95_relative_ceiling": 1.20,
            "warmup_queries": 100,
            "measured_queries": 1000,
            "concurrency": 1,
            "hardware": "documented test CPU",
            "software": "documented runtime",
            "timer": "monotonic wall clock",
        },
        "deviations": [],
    }


def valid_evaluation(
    run_manifest_sha256: str,
    paired_observations_sha256: str,
    paired_rows: list[PairedScoreRow],
) -> dict:
    metric_values = metrics()
    variants = {
        variant: {
            "status": "complete",
            "n_users": 1109,
            "metrics": dict(metric_values),
            "by_seed": {str(seed): dict(metric_values) for seed in EXPECTED_SEEDS},
            "by_date": {date: dict(metric_values) for date in EXPECTED_SPLIT["test"]},
        }
        for variant in EXPECTED_VARIANTS
    }
    summaries = primary_metric_summaries(paired_rows)
    for variant, summary in summaries.items():
        variants[variant]["n_users"] = summary["n_users"]
        variants[variant]["metrics"]["purchase_ndcg_at_50"] = summary["metrics"]
        for seed, value in summary["by_seed"].items():
            variants[variant]["by_seed"][seed]["purchase_ndcg_at_50"] = value
        for date, value in summary["by_date"].items():
            variants[variant]["by_date"][date]["purchase_ndcg_at_50"] = value
    contrasts = registered_primary_contrasts(paired_rows, best_single_variant="ptd_node")
    for contrast in contrasts.values():
        contrast["guardrails_pass"] = guardrail_report(
            variants[contrast["numerator"]]["metrics"],
            variants["fixed_tdm"]["metrics"],
        )["passed"]
    return {
        "schema_version": "ptd-evaluation/v1",
        "status": "complete",
        "generated_at": "2026-09-24T03:00:00Z",
        "code_revision": "a" * 40,
        "run_manifest_uri": "gs://example.invalid/run/manifest.json",
        "run_manifest_sha256": run_manifest_sha256,
        "split": copy.deepcopy(EXPECTED_SPLIT),
        "seeds": list(EXPECTED_SEEDS),
        "assertions": {name: True for name in ASSERTIONS},
        "variants": variants,
        "paired_observations": {
            "uri": "gs://example.invalid/run/paired_observations.jsonl",
            "sha256": paired_observations_sha256,
            "row_count": len(paired_rows),
            "format": "jsonl",
            "row_schema": "ptd-paired-observation-row/v1",
        },
        "primary_contrasts": contrasts,
        "inference": {
            "unit": "date_user_after_seed_average",
            "stratification": "test_date",
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 20_260_925,
            "interval": "percentile_2.5_97.5",
            "raw_p": "two_sided_bootstrap_sign",
            "multiplicity": "holm_four_primary_contrasts",
        },
        "guardrails": {
            "click_ndcg_relative_floor": 0.95,
            "category_coverage_relative_floor": 0.95,
            "max_category_share_relative_ceiling": 1.05,
            "latency_p95_relative_ceiling": 1.20,
        },
        "deviations": [],
    }


class EvidenceGateTest(unittest.TestCase):
    def check(self, mutate_run=None, mutate_evaluation=None) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = valid_run()
            if mutate_run:
                mutate_run(run)
            run_path = root / "run.json"
            run_path.write_text(json.dumps(run, sort_keys=True))
            run_hash = hashlib.sha256(run_path.read_bytes()).hexdigest()
            paired_path = root / "paired_observations.jsonl"
            paired_rows = write_paired_score_rows(paired_path)
            paired_hash = hashlib.sha256(paired_path.read_bytes()).hexdigest()
            evaluation = valid_evaluation(run_hash, paired_hash, paired_rows)
            if mutate_evaluation:
                mutate_evaluation(evaluation)
            evaluation_path = root / "evaluation.json"
            evaluation_path.write_text(json.dumps(evaluation, sort_keys=True))
            return admission_report(run_path, evaluation_path, paired_path)

    def test_complete_registered_pair_is_admissible(self) -> None:
        report = self.check()
        self.assertTrue(report["admissible"], report["errors"])

    def test_false_assertion_fails_closed(self) -> None:
        report = self.check(mutate_evaluation=lambda value: value["assertions"].__setitem__("point_in_time_safe", False))
        self.assertFalse(report["admissible"])
        self.assertTrue(any("point_in_time_safe" in error for error in report["errors"]))

    def test_manifest_hash_mismatch_fails_closed(self) -> None:
        report = self.check(mutate_evaluation=lambda value: value.__setitem__("run_manifest_sha256", "0" * 64))
        self.assertFalse(report["admissible"])
        self.assertTrue(any("run_manifest_sha256" in error for error in report["errors"]))

    def test_paired_observation_hash_mismatch_fails_closed(self) -> None:
        report = self.check(
            mutate_evaluation=lambda value: value["paired_observations"].__setitem__("sha256", "0" * 64)
        )
        self.assertFalse(report["admissible"])
        self.assertTrue(any("paired observation hash" in error for error in report["errors"]))

    def test_tampered_contrast_fails_recomputation(self) -> None:
        report = self.check(
            mutate_evaluation=lambda value: value["primary_contrasts"]["rq1_combined_vs_tdm"].__setitem__(
                "mean_delta", 0.99
            )
        )
        self.assertFalse(report["admissible"])
        self.assertTrue(any("paired-observation recomputation" in error for error in report["errors"]))

    def test_tampered_aggregate_metric_fails_recomputation(self) -> None:
        report = self.check(
            mutate_evaluation=lambda value: value["variants"]["ptd_combined"]["metrics"].__setitem__(
                "purchase_ndcg_at_50", 0.99
            )
        )
        self.assertFalse(report["admissible"])
        self.assertTrue(any("purchase NDCG does not match paired observations" in error for error in report["errors"]))

    def test_missing_variant_fails_closed(self) -> None:
        report = self.check(mutate_evaluation=lambda value: value["variants"].pop("ptd_node"))
        self.assertFalse(report["admissible"])
        self.assertTrue(any("eight registered variants" in error for error in report["errors"]))

    def test_deviation_requires_manual_review(self) -> None:
        report = self.check(mutate_evaluation=lambda value: value["deviations"].append("hardware changed"))
        self.assertFalse(report["admissible"])
        self.assertTrue(any("deviations" in error for error in report["errors"]))

    def test_naive_timestamp_fails_closed(self) -> None:
        report = self.check(mutate_run=lambda value: value.__setitem__("created_at", "2026-09-24T00:00:00"))
        self.assertFalse(report["admissible"])
        self.assertTrue(any("timestamps" in error for error in report["errors"]))

    def test_unregistered_hyperparameter_fails_closed(self) -> None:
        report = self.check(
            mutate_run=lambda value: value["selected_hyperparameters"].__setitem__("temperature", 3.0)
        )
        self.assertFalse(report["admissible"])
        self.assertTrue(any("temperature" in error for error in report["errors"]))

    def test_complete_guardrail_failure_is_admissible_evidence(self) -> None:
        def fail_latency_guardrail(value: dict) -> None:
            value["variants"]["ptd_combined"]["metrics"]["latency_p95_ms"] = 13.0
            for name in (
                "rq1_combined_vs_tdm",
                "rq2_combined_vs_best_single",
                "rq4_hstu_vs_baseline_encoder",
            ):
                value["primary_contrasts"][name]["guardrails_pass"] = False

        report = self.check(mutate_evaluation=fail_latency_guardrail)
        self.assertTrue(report["admissible"], report["errors"])


if __name__ == "__main__":
    unittest.main()
