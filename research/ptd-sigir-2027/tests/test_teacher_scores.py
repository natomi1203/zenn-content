from __future__ import annotations

import unittest

import numpy as np

from runner.materialize_teacher_scores import (
    EXPECTED_SOURCE_REVISION,
    EXPECTED_TRIAL_CONFIG_SHA256,
    FEATURE_COLUMNS,
    normalized_feature_matrix,
    validate_checkpoint_metadata,
)


def metadata() -> dict:
    return {
        "contract_version": "shared_bottom_esmm_v2",
        "source_revision": EXPECTED_SOURCE_REVISION,
        "trial_config_sha256": EXPECTED_TRIAL_CONFIG_SHA256,
        "checkpoint_metric": "valid_loss",
        "checkpoint_epoch": 4,
        "seed": 16630,
        "trial": {"shared_hidden_sizes": (32, 32), "tower_hidden_sizes": (16,)},
        "feature_stats": {
            "columns": FEATURE_COLUMNS,
            "mean": tuple(float(value) for value in range(8)),
            "scale": (2.0,) * 8,
        },
    }


class TeacherScoreContractTest(unittest.TestCase):
    def test_metadata_and_normalization_match_frozen_order(self) -> None:
        mean, scale = validate_checkpoint_metadata(metadata())
        columns = {
            name: [float(index), None]
            for index, name in enumerate(FEATURE_COLUMNS)
        }
        matrix = normalized_feature_matrix(columns, mean, scale)
        self.assertEqual(matrix.shape, (2, 8))
        np.testing.assert_allclose(matrix[0], np.zeros(8, dtype=np.float32))
        np.testing.assert_allclose(matrix[1], -mean / scale)

    def test_feature_order_is_fail_closed(self) -> None:
        mean, scale = validate_checkpoint_metadata(metadata())
        reversed_columns = {
            name: [0.0]
            for name in reversed(FEATURE_COLUMNS)
        }
        with self.assertRaises(ValueError):
            normalized_feature_matrix(reversed_columns, mean, scale)

    def test_checkpoint_identity_is_fail_closed(self) -> None:
        value = metadata()
        value["checkpoint_epoch"] = 3
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata(value)


if __name__ == "__main__":
    unittest.main()
