from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import jsonschema
except ImportError:  # pragma: no cover - optional schema-validation dependency
    jsonschema = None

from runner.materialize_teacher_scores import sha256
from runner.select_validation import (
    LAMBDAS,
    TEMPERATURES,
    preselect_combined_grid,
    select_validation,
)


class TargetBackend:
    def __init__(self, *, target_leaf: int, favor_target: bool, depth: int) -> None:
        self.target_leaf = target_leaf
        self.favor_target = favor_target
        self.depth = depth
        self.leaf_start = 2**depth - 1

    def _contains_target(self, node_id: int, level: int) -> bool:
        offset = node_id - (2**level - 1)
        width = 2 ** (self.depth - level)
        target_offset = self.target_leaf - self.leaf_start
        return offset * width <= target_offset < (offset + 1) * width

    def score_sibling_pairs(self, *, pairs, **_kwargs):
        direction = 10.0 if self.favor_target else -10.0
        return [
            tuple(
                direction if self._contains_target(child.node_id, child.level) else 0.0
                for child in pair
            )
            for pair in pairs
        ]

    def synchronize(self) -> None:
        return None


class ValidationSelectionTest(unittest.TestCase):
    def inputs(self, root: Path, *, single_key=(2.0, 0.3, 1.0)):
        products = (101, 102, 103, 104)
        depth = 13
        leaf_start = 2**depth - 1
        catalog = root / "catalog.parquet"
        eligibility = root / "eligibility.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "product_id": product,
                        "first_category": "category-a",
                        "leaf_node_id": leaf_start + index,
                        "path_bits": format(index, f"0{depth}b"),
                    }
                    for index, product in enumerate(products)
                ]
            ),
            catalog,
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "snapshot_date": date.fromisoformat("2026-07-21"),
                        "product_id": product,
                        "leaf_node_id": leaf_start + index,
                    }
                    for index, product in enumerate(products)
                ]
            ),
            eligibility,
        )
        queries = root / "queries.jsonl"
        queries.write_text(
            json.dumps(
                {
                    "date": "2026-07-21",
                    "user_id": "user-1",
                    "click_history_most_recent_first": [101],
                    "purchase_history_most_recent_first": [],
                    "candidates": [
                        {
                            "product_id": product,
                            "category": "category-a",
                            "click_label": 0,
                            "purchase_label": int(product == 104),
                            "teacher_purchase": 0.1,
                        }
                        for product in products
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        query_manifest = root / "query-manifest.json"
        query_manifest.write_text(
            json.dumps(
                {
                    "contract_version": "ptd-validation-queries/v1",
                    "status": "complete",
                    "dates": ["2026-07-21"],
                    "output": {"path": str(queries), "sha256": sha256(queries)},
                    "checks": {"validation_only": True},
                }
            )
            + "\n"
        )

        fit_manifests = []
        selected = (2.0, 0.3, 1.0)
        fits = [("ptd_combined", *values) for values in itertools.product(TEMPERATURES, LAMBDAS, LAMBDAS)]
        fits.extend((variant, *single_key) for variant in ("ptd_item", "ptd_node"))
        for index, (variant, temperature, lambda_item, lambda_node) in enumerate(fits):
            checkpoint = root / f"checkpoint-{index}.pt"
            checkpoint.write_bytes(f"checkpoint-{index}".encode())
            fit_manifest = root / f"fit-{index}.json"
            fit_manifest.write_text(
                json.dumps(
                    {
                        "contract_version": "ptd-single-fit/v1",
                        "status": "complete",
                        "variant": variant,
                        "seed": 16630,
                        "test_only_config_override": True,
                        "configuration": {
                            "temperature": temperature,
                            "lambda_item": lambda_item,
                            "lambda_node": lambda_node,
                        },
                        "checkpoint": {
                            "path": str(checkpoint),
                            "sha256": sha256(checkpoint),
                        },
                        "checks": {"complete": True},
                    }
                )
                + "\n"
            )
            fit_manifests.append(fit_manifest)
        return query_manifest, catalog, eligibility, fit_manifests, leaf_start + 3, selected

    def test_selects_grid_and_single_variant_without_test_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_manifest, catalog, eligibility, fits, target_leaf, selected = self.inputs(root)

            def factory(fit, _tree, _device):
                key = tuple(fit["selection_key"])
                favor = key == selected
                if fit["variant"] == "ptd_item":
                    favor = False
                if fit["variant"] == "ptd_node":
                    favor = True
                return TargetBackend(target_leaf=target_leaf, favor_target=favor, depth=13)

            output_path = root / "selection.json"
            output = select_validation(
                validation_queries_manifest_path=query_manifest,
                catalog_path=catalog,
                date_eligibility_path=eligibility,
                fit_manifest_paths=fits,
                output_path=output_path,
                device="cpu",
                backend_factory=factory,
                allow_test_only_fits=True,
            )
            self.assertEqual(
                (
                    output["selected_hyperparameters"]["temperature"],
                    output["selected_hyperparameters"]["lambda_item"],
                    output["selected_hyperparameters"]["lambda_node"],
                ),
                selected,
            )
            self.assertEqual(output["best_single_variant"], "ptd_node")
            self.assertEqual(len(output["grid_scores"]), 27)
            self.assertTrue(all(output["checks"].values()))
            if jsonschema is not None:
                schema = json.loads(
                    (Path(__file__).parents[1] / "artifact" / "validation_selection.schema.json").read_text()
                )
                jsonschema.Draft202012Validator(schema).validate(output)
            with self.assertRaises(FileExistsError):
                select_validation(
                    validation_queries_manifest_path=query_manifest,
                    catalog_path=catalog,
                    date_eligibility_path=eligibility,
                    fit_manifest_paths=fits,
                    output_path=output_path,
                    device="cpu",
                    backend_factory=factory,
                    allow_test_only_fits=True,
                )

    def test_preselects_grid_before_single_level_fits_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_manifest, catalog, eligibility, fits, target_leaf, selected = self.inputs(root)

            def factory(fit, _tree, _device):
                return TargetBackend(
                    target_leaf=target_leaf,
                    favor_target=tuple(fit["selection_key"]) == selected,
                    depth=13,
                )

            output = preselect_combined_grid(
                validation_queries_manifest_path=query_manifest,
                catalog_path=catalog,
                date_eligibility_path=eligibility,
                fit_manifest_paths=fits[:27],
                device="cpu",
                backend_factory=factory,
                allow_test_only_fits=True,
            )
            self.assertEqual(output["selected_key"], selected)
            self.assertEqual(len(output["grid_scores"]), 27)
            self.assertEqual(output["purchase_positive_units"], 1)

    def test_applies_registered_grid_and_single_variant_tie_breaks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_manifest, catalog, eligibility, fits, target_leaf, _selected = self.inputs(
                root, single_key=(1.0, 0.1, 0.1)
            )

            def factory(_fit, _tree, _device):
                return TargetBackend(target_leaf=target_leaf, favor_target=True, depth=13)

            output = select_validation(
                validation_queries_manifest_path=query_manifest,
                catalog_path=catalog,
                date_eligibility_path=eligibility,
                fit_manifest_paths=fits,
                output_path=root / "selection.json",
                device="cpu",
                backend_factory=factory,
                allow_test_only_fits=True,
            )
            self.assertEqual(
                (
                    output["selected_hyperparameters"]["temperature"],
                    output["selected_hyperparameters"]["lambda_item"],
                    output["selected_hyperparameters"]["lambda_node"],
                ),
                (1.0, 0.1, 0.1),
            )
            self.assertEqual(output["best_single_variant"], "ptd_item")


if __name__ == "__main__":
    unittest.main()
