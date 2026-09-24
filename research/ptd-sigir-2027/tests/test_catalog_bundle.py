from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from runner.build_catalog_bundle import build_bundle, contract_summary, path_bits


class CatalogBundleTest(unittest.TestCase):
    def test_bundle_order_paths_masks_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            output = root / "bundle"
            pq.write_table(
                pa.table(
                    {
                        "snapshot_date": [date(2026, 7, 18), date(2026, 7, 18), date(2026, 7, 21), date(2026, 7, 21)],
                        "product_id": [2, 10, 3, 2],
                        "first_category": ["b", "a", "a", "b"],
                    }
                ),
                source,
            )
            manifest = build_bundle([source], output, depth=2, verify_registered=False, batch_size=2)
            catalog = pq.read_table(output / "catalog.parquet")
            self.assertEqual(catalog["product_id"].to_pylist(), [10, 3, 2])
            self.assertEqual(catalog["path_bits"].to_pylist(), ["00", "01", "10"])
            self.assertEqual(manifest["padding_leaf_count"], 1)
            self.assertEqual(manifest["checks"]["users_or_outcomes_read"], False)
            with self.assertRaises(FileExistsError):
                build_bundle([source], output, depth=2, verify_registered=False)

    def test_contract_summary_uses_lexical_product_ids(self) -> None:
        catalog = [(10, "a", "2026-07-18"), (2, "a", "2026-07-18")]
        summary = contract_summary(catalog, {"2026-07-18": {2, 10}})
        self.assertEqual(summary["item_count"], 2)
        self.assertNotEqual(summary["catalog_order_sha256"], summary["date_eligibility_sha256"]["2026-07-18"])

    def test_path_bits_rejects_capacity_overflow(self) -> None:
        self.assertEqual(path_bits(3, 2), "11")
        with self.assertRaises(ValueError):
            path_bits(4, 2)


if __name__ == "__main__":
    unittest.main()
