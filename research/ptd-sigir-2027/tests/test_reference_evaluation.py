from __future__ import annotations

import unittest

from reference.evaluation import (
    PairedObservation,
    date_stratified_paired_bootstrap,
    guardrail_report,
    holm_adjust,
    ndcg_at_k,
    recall_at_k,
)
from reference.evidence_gate import EXPECTED_SEEDS, EXPECTED_SPLIT


def positive_observations() -> list[PairedObservation]:
    rows: list[PairedObservation] = []
    for date in EXPECTED_SPLIT["test"]:
        for user in ("u1", "u2"):
            for seed in EXPECTED_SEEDS:
                rows.append(PairedObservation(date, user, seed, treatment=0.3, baseline=0.2))
    return rows


class ReferenceEvaluationTest(unittest.TestCase):
    def test_ranking_metrics(self) -> None:
        self.assertAlmostEqual(ndcg_at_k([1, 0, 1], 3), 0.9197207891)
        self.assertEqual(ndcg_at_k([0, 0], 2), 0.0)
        self.assertAlmostEqual(recall_at_k([1, 0, 1], 2), 0.5)

    def test_bootstrap_is_seeded_and_stratified(self) -> None:
        first = date_stratified_paired_bootstrap(positive_observations(), resamples=200)
        second = date_stratified_paired_bootstrap(positive_observations(), resamples=200)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_delta"], 0.1)
        self.assertAlmostEqual(first["ci95"][0], 0.1)
        self.assertAlmostEqual(first["ci95"][1], 0.1)
        self.assertEqual(first["paired_units"], 10)

    def test_incomplete_seed_pairing_is_rejected(self) -> None:
        rows = positive_observations()
        rows.pop()
        with self.assertRaisesRegex(ValueError, "incomplete seed pairing"):
            date_stratified_paired_bootstrap(rows, resamples=10)

    def test_holm_adjustment_is_monotone(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertEqual(adjusted, {"a": 0.03, "c": 0.06, "b": 0.06})

    def test_guardrails_pass_and_fail(self) -> None:
        baseline = {
            "click_ndcg_at_50": 0.20,
            "category_coverage_at_50": 0.40,
            "max_category_share_at_50": 0.30,
            "latency_p95_ms": 10.0,
        }
        passing = {
            "click_ndcg_at_50": 0.19,
            "category_coverage_at_50": 0.38,
            "max_category_share_at_50": 0.315,
            "latency_p95_ms": 12.0,
        }
        failing = dict(passing, latency_p95_ms=12.01)
        self.assertTrue(guardrail_report(passing, baseline)["passed"])
        self.assertFalse(guardrail_report(failing, baseline)["passed"])


if __name__ == "__main__":
    unittest.main()
