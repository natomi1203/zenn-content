from __future__ import annotations

import math
import unittest

from reference.ptd import (
    balanced_paths,
    capacity_balanced_assignment,
    item_sibling_distribution,
    kl_divergence,
    node_sibling_distribution,
    ptd_objective,
    softmax,
)


class ReferencePTDTest(unittest.TestCase):
    def test_softmax_is_stable_and_normalized(self) -> None:
        result = softmax({"b": 1001.0, "a": 1000.0}, temperature=2.0)
        self.assertEqual(list(result), ["a", "b"])
        self.assertAlmostEqual(sum(result.values()), 1.0)
        self.assertGreater(result["b"], result["a"])

    def test_item_targets_use_purchase_logits(self) -> None:
        result = item_sibling_distribution({"a": 0.8, "b": 0.2}, ["a", "b"])
        self.assertAlmostEqual(result["a"], 16.0 / 17.0, places=8)
        self.assertAlmostEqual(result["b"], 1.0 / 17.0, places=8)

    def test_node_targets_aggregate_eligible_descendants(self) -> None:
        result = node_sibling_distribution(
            {"a": 0.8, "b": 0.2, "c": 0.1, "d": 0.1},
            {"left": ["a", "b"], "right": ["c", "d"]},
            ["a", "b", "c", "d"],
        )
        self.assertAlmostEqual(result["left"], 5.0 / 6.0, places=8)
        self.assertAlmostEqual(result["right"], 1.0 / 6.0, places=8)

    def test_kl_is_zero_for_equal_distributions(self) -> None:
        distribution = {"a": 0.25, "b": 0.75}
        self.assertAlmostEqual(kl_divergence(distribution, distribution), 0.0)

    def test_objective_matches_registered_weighting(self) -> None:
        teacher = softmax({"a": 1.0, "b": 0.0}, temperature=2.0)
        result = ptd_objective(
            0.5,
            item_teacher=teacher,
            item_student_logits={"a": 1.0, "b": 0.0},
            node_teacher=teacher,
            node_student_logits={"a": 0.0, "b": 1.0},
            temperature=2.0,
            lambda_item=0.3,
            lambda_node=0.1,
        )
        self.assertAlmostEqual(result["item_kl"], 0.0)
        self.assertGreater(result["node_kl"], 0.0)
        self.assertAlmostEqual(result["total"], 0.5 + 0.4 * result["node_kl"])

    def test_balanced_paths_are_deterministic_and_unique(self) -> None:
        paths = balanced_paths(["d", "b", "a", "c", "e"], branching_factor=2)
        self.assertEqual(paths, balanced_paths(reversed(["d", "b", "a", "c", "e"]), 2))
        self.assertEqual(len(set(paths.values())), 5)
        self.assertEqual({len(path) for path in paths.values()}, {3})

    def test_capacity_assignment_respects_limit_and_ties(self) -> None:
        assignments = capacity_balanced_assignment(
            {
                "a": {"left": 2.0, "right": 0.0},
                "b": {"left": 1.0, "right": 1.0},
                "c": {"left": 0.0, "right": 2.0},
                "d": {"left": 0.5, "right": 0.4},
            },
            ["left", "right"],
            capacity=2,
        )
        counts = {node: sum(item.node == node for item in assignments) for node in ("left", "right")}
        self.assertEqual(counts, {"left": 2, "right": 2})
        self.assertTrue(all(math.isfinite(item.weight) for item in assignments))


if __name__ == "__main__":
    unittest.main()
