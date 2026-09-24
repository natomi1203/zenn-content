from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from reference.evidence_gate import EXPECTED_SPLIT
from runner.retrieval_runner import (
    CandidateLabel,
    CatalogTree,
    RetrievalQuery,
    beam_retrieve,
    retrieval_metrics,
)


class ZeroBackend:
    def score_sibling_pairs(self, **kwargs):
        return [(0.0, 0.0) for _ in kwargs["pairs"]]

    def synchronize(self) -> None:
        return None


class RetrievalRunnerTest(unittest.TestCase):
    def tree(self, root: Path) -> CatalogTree:
        depth = 3
        leaf_start = 2**depth - 1
        products = list(range(101, 107))
        catalog = pa.table(
            {
                "product_id": products,
                "first_category": ["a", "a", "b", "b", "c", "c"],
                "path_bits": [format(index, "03b") for index in range(6)],
                "leaf_node_id": [leaf_start + index for index in range(6)],
            }
        )
        catalog_path = root / "catalog.parquet"
        pq.write_table(catalog, catalog_path)
        eligibility = pa.table(
            {
                "snapshot_date": [
                    date.fromisoformat(value)
                    for value in EXPECTED_SPLIT["test"]
                    for _ in products
                ],
                "product_id": products * len(EXPECTED_SPLIT["test"]),
                "leaf_node_id": [leaf_start + index for index in range(6)]
                * len(EXPECTED_SPLIT["test"]),
            }
        )
        eligibility_path = root / "eligibility.parquet"
        pq.write_table(eligibility, eligibility_path)
        return CatalogTree(catalog_path, eligibility_path, depth=depth)

    def query(self) -> RetrievalQuery:
        return RetrievalQuery(
            date="2026-08-12",
            user_id="u1",
            click_history_most_recent_first=(106, 105),
            purchase_history_most_recent_first=(),
            candidates={
                product: CandidateLabel(
                    product_id=product,
                    category=("a", "a", "b", "b", "c", "c")[index],
                    click_label=1 if product == 101 else 0,
                    purchase_label=1 if product == 102 else 0,
                    teacher_purchase=0.1 * (index + 1),
                )
                for index, product in enumerate(range(101, 107))
            },
        )

    def test_beam_returns_only_eligible_products(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = beam_retrieve(
                self.query(),
                self.tree(Path(directory)),
                ZeroBackend(),
                beam_width=600,
                top_k=600,
            )
        self.assertEqual(set(result.product_ids), set(range(101, 107)))
        self.assertEqual(len(result.product_ids), 6)
        self.assertGreater(result.candidates_scored, 0)

    def test_metrics_use_all_eligible_items_for_ideal_and_recall(self) -> None:
        query = self.query()
        from runner.retrieval_runner import RetrievalResult

        result = RetrievalResult(
            product_ids=(101, 103), path_scores=(1.0, 0.5), candidates_scored=10
        )
        metrics = retrieval_metrics(query, result, latency_ms=2.0)
        self.assertEqual(metrics["purchase_recall_at_50"], 0.0)
        self.assertEqual(metrics["latency_ms"], 2.0)
        self.assertEqual(metrics["candidates_scored"], 10.0)
        self.assertLess(metrics["purchase_ndcg_at_50"], 1.0)


if __name__ == "__main__":
    unittest.main()
