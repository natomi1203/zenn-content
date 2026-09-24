from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reference.evaluation import (
    PairedObservation,
    date_stratified_paired_bootstrap,
    guardrail_report,
    holm_adjust,
    load_paired_score_rows,
    ndcg_at_k,
    recall_at_k,
    registered_primary_contrasts,
)
from reference.evidence_gate import EXPECTED_SEEDS, EXPECTED_SPLIT, EXPECTED_VARIANTS


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

    def test_jsonl_rows_recompute_registered_contrasts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.jsonl"
            records = []
            for date in EXPECTED_SPLIT["test"]:
                for seed in EXPECTED_SEEDS:
                    scores = {variant: 0.2 for variant in EXPECTED_VARIANTS}
                    scores.update(
                        ptd_item=0.21,
                        ptd_node=0.22,
                        ptd_combined=0.23,
                        alternating_ptd=0.24,
                        ptd_combined_baseline_encoder=0.225,
                    )
                    records.append({"date": date, "user_id": "u1", "seed": seed, "scores": scores})
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            rows = load_paired_score_rows(path)
            contrasts = registered_primary_contrasts(rows, best_single_variant="ptd_node", resamples=200)
        self.assertEqual(contrasts["rq2_combined_vs_best_single"]["resolved_denominator"], "ptd_node")
        self.assertAlmostEqual(contrasts["rq1_combined_vs_tdm"]["mean_delta"], 0.03)
        self.assertEqual(contrasts["rq1_combined_vs_tdm"]["paired_units"], 5)

    def test_jsonl_rejects_missing_variant_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.jsonl"
            scores = {variant: 0.2 for variant in EXPECTED_VARIANTS}
            scores.pop("ptd_node")
            path.write_text(
                json.dumps(
                    {
                        "date": EXPECTED_SPLIT["test"][0],
                        "user_id": "u1",
                        "seed": EXPECTED_SEEDS[0],
                        "scores": scores,
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "eight registered variant"):
                load_paired_score_rows(path)

    def test_registered_contrasts_are_row_order_invariant(self) -> None:
        records = []
        for date in EXPECTED_SPLIT["test"]:
            for user_index, user_id in enumerate(("u1", "u2")):
                for seed_index, seed in enumerate(EXPECTED_SEEDS):
                    base = 0.10 + user_index * 0.01 + seed_index * 0.001
                    scores = {variant: base for variant in EXPECTED_VARIANTS}
                    scores.update(ptd_item=base + 0.01, ptd_node=base + 0.02, ptd_combined=base + 0.03)
                    records.append({"date": date, "user_id": user_id, "seed": seed, "scores": scores})
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.jsonl"
            second_path = Path(directory) / "second.jsonl"
            first_path.write_text("".join(json.dumps(record) + "\n" for record in records))
            second_path.write_text("".join(json.dumps(record) + "\n" for record in reversed(records)))
            first = registered_primary_contrasts(
                load_paired_score_rows(first_path), best_single_variant="ptd_node", resamples=200
            )
            second = registered_primary_contrasts(
                load_paired_score_rows(second_path), best_single_variant="ptd_node", resamples=200
            )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
