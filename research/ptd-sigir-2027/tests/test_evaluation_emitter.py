from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reference.evidence_gate import EXPECTED_SEEDS, EXPECTED_SPLIT, EXPECTED_VARIANTS
from runner.emit_evaluation import aggregate_variants, load_retrieval_metric_rows


def row(date: str, user: str, seed: int, variant: str) -> dict:
    return {
        "date": date,
        "user_id": user,
        "seed": seed,
        "variant": variant,
        "purchase_ndcg_at_50": 0.2,
        "purchase_recall_at_50": 0.4,
        "purchase_ndcg_at_10": 0.1,
        "purchase_ndcg_at_100": 0.3,
        "purchase_auc": 0.6,
        "click_ndcg_at_50": 0.25,
        "category_coverage_at_50": 0.5,
        "max_category_share_at_50": 0.2,
        "latency_ms": 10.0,
        "candidates_scored": 100.0,
    }


def complete_rows() -> list[dict]:
    return [
        row(date, "u1", seed, variant)
        for date in EXPECTED_SPLIT["test"]
        for seed in EXPECTED_SEEDS
        for variant in EXPECTED_VARIANTS
    ]


class EvaluationEmitterTest(unittest.TestCase):
    def write(self, values: list[dict]) -> tuple[tempfile.TemporaryDirectory, Path]:
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "rows.jsonl"
        path.write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in values)
        )
        return directory, path

    def test_complete_rows_aggregate_every_registered_slice(self) -> None:
        directory, path = self.write(complete_rows())
        self.addCleanup(directory.cleanup)
        rows = load_retrieval_metric_rows(path)
        variants = aggregate_variants(rows)
        self.assertEqual(set(variants), set(EXPECTED_VARIANTS))
        self.assertEqual(variants["fixed_tdm"]["n_users"], 1)
        self.assertEqual(
            set(variants["fixed_tdm"]["by_seed"]),
            {str(value) for value in EXPECTED_SEEDS},
        )
        self.assertEqual(
            set(variants["fixed_tdm"]["by_date"]), set(EXPECTED_SPLIT["test"])
        )
        self.assertEqual(variants["fixed_tdm"]["metrics"]["latency_p95_ms"], 10.0)

    def test_missing_variant_is_rejected(self) -> None:
        values = complete_rows()
        values.pop()
        directory, path = self.write(values)
        self.addCleanup(directory.cleanup)
        with self.assertRaisesRegex(ValueError, "incomplete variant pairing"):
            load_retrieval_metric_rows(path)

    def test_duplicate_row_is_rejected(self) -> None:
        values = complete_rows()
        values.append(dict(values[0]))
        directory, path = self.write(values)
        self.addCleanup(directory.cleanup)
        with self.assertRaisesRegex(ValueError, "duplicate retrieval row"):
            load_retrieval_metric_rows(path)


if __name__ == "__main__":
    unittest.main()
