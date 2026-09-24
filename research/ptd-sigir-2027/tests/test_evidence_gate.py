from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from reference.evidence_gate import (
    ASSERTIONS,
    EXPECTED_CONTRASTS,
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


def valid_evaluation(run_manifest_sha256: str) -> dict:
    metric_values = metrics()
    variants = {
        variant: {
            "status": "complete",
            "n_users": 1109,
            "metrics": metric_values,
            "by_seed": {str(seed): metric_values for seed in EXPECTED_SEEDS},
            "by_date": {date: metric_values for date in EXPECTED_SPLIT["test"]},
        }
        for variant in EXPECTED_VARIANTS
    }
    contrasts = {
        name: {
            "numerator": numerator,
            "denominator": denominator,
            "metric": "purchase_ndcg_at_50",
            "mean_delta": 0.01,
            "ci95": [0.001, 0.02],
            "raw_p": 0.01,
            "holm_adjusted_p": 0.04,
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 20_260_925,
            "guardrails_pass": True,
        }
        for name, (numerator, denominator) in EXPECTED_CONTRASTS.items()
    }
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
        "primary_contrasts": contrasts,
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
            evaluation = valid_evaluation(run_hash)
            if mutate_evaluation:
                mutate_evaluation(evaluation)
            evaluation_path = root / "evaluation.json"
            evaluation_path.write_text(json.dumps(evaluation, sort_keys=True))
            return admission_report(run_path, evaluation_path)

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


if __name__ == "__main__":
    unittest.main()
