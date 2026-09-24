from __future__ import annotations

import unittest

try:
    import numpy as np

    from runner.alternating_solver import solve_anchored_assignment
except ImportError:  # pragma: no cover - optional production dependencies
    np = None


@unittest.skipIf(np is None, "NumPy/SciPy/PyArrow are optional production dependencies")
class AlternatingSolverTest(unittest.TestCase):
    def catalog(self) -> list[dict]:
        return [
            {
                "product_id": 100 + index,
                "catalog_index": index,
                "leaf_node_id": 7 + index,
                "train_seen": index < 2,
            }
            for index in range(4)
        ]

    def test_exact_assignment_moves_only_train_items(self) -> None:
        # Anchors occupy leaves 9 and 10; free leaves are 7, 8, 11, 12, 13, 14.
        weights = np.array(
            [
                [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assignment, objective = solve_anchored_assignment(
            self.catalog(), weights, depth=3
        )
        self.assertEqual(assignment[100], 8)
        self.assertEqual(assignment[101], 7)
        self.assertEqual(assignment[102], 9)
        self.assertEqual(assignment[103], 10)
        self.assertEqual(objective["optimized_objective"], 20.0)

    def test_matrix_shape_is_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            solve_anchored_assignment(self.catalog(), np.zeros((2, 2)), depth=3)


if __name__ == "__main__":
    unittest.main()
