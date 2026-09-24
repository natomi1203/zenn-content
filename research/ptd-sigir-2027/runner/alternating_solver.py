#!/usr/bin/env python3
"""Exact train-only anchored reassignment for the registered PTD tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment

from runner.build_catalog_bundle import TRAIN_DATES, file_sha256, path_bits

WEIGHT_FORMULA = (
    "sum_train_requests((purchase_label+teacher_purchase)*path_log_probability)"
)
CANDIDATE_LEAF_POLICY = "all_physical_leaves_except_anchored"


def _digest_lines(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def validate_weight_manifest(
    manifest: Mapping[str, Any],
    *,
    weights_path: Path,
    catalog_path: Path,
) -> None:
    """Reject weight metadata that does not prove the registered train-only boundary."""
    if manifest.get("contract_version") != "ptd-assignment-weights/v1":
        raise ValueError("assignment-weight contract mismatch")
    if manifest.get("status") != "complete":
        raise ValueError("assignment weights are incomplete")
    if manifest.get("source_split") != "train":
        raise ValueError("assignment weights must come from the train split")
    if tuple(manifest.get("source_dates", ())) != TRAIN_DATES:
        raise ValueError(
            "assignment weights must use exactly the registered train dates"
        )
    if manifest.get("formula") != WEIGHT_FORMULA:
        raise ValueError("assignment-weight formula mismatch")
    if manifest.get("candidate_leaf_policy") != CANDIDATE_LEAF_POLICY:
        raise ValueError("assignment candidate-leaf policy mismatch")
    if manifest.get("validation_or_test_fields_read") != []:
        raise ValueError("validation/test evidence is forbidden in reassignment")
    if manifest.get("weights", {}).get("sha256") != file_sha256(weights_path):
        raise ValueError("assignment-weight file hash mismatch")
    if manifest.get("catalog_sha256") != file_sha256(catalog_path):
        raise ValueError("assignment-weight catalog hash mismatch")


def _catalog_rows(catalog_path: Path, depth: int) -> list[dict[str, Any]]:
    table = pq.read_table(catalog_path)
    required = {
        "product_id",
        "first_category",
        "first_registered_date",
        "catalog_index",
        "path_bits",
        "leaf_node_id",
        "train_seen",
    }
    if missing := required - set(table.column_names):
        raise ValueError(f"catalog is missing columns: {sorted(missing)}")
    rows = table.to_pylist()
    leaf_start = 2**depth - 1
    product_ids = [int(row["product_id"]) for row in rows]
    leaf_ids = [int(row["leaf_node_id"]) for row in rows]
    if len(set(product_ids)) != len(rows) or len(set(leaf_ids)) != len(rows):
        raise ValueError("catalog products and occupied leaves must be unique")
    for row in rows:
        index = int(row["catalog_index"])
        if int(row["leaf_node_id"]) != leaf_start + index:
            raise ValueError("fixed catalog index/leaf mismatch")
        if str(row["path_bits"]) != path_bits(index, depth):
            raise ValueError("fixed catalog path mismatch")
    return sorted(rows, key=lambda row: int(row["catalog_index"]))


def _weight_matrix(
    weights_path: Path,
    *,
    train_products: list[int],
    available_leaves: list[int],
) -> np.ndarray:
    parquet = pq.ParquetFile(weights_path)
    required = {"product_id", "leaf_node_id", "assignment_weight"}
    if missing := required - set(parquet.schema_arrow.names):
        raise ValueError(f"assignment weights are missing columns: {sorted(missing)}")
    product_index = {product: index for index, product in enumerate(train_products)}
    leaf_index = {leaf: index for index, leaf in enumerate(available_leaves)}
    matrix = np.full(
        (len(train_products), len(available_leaves)), np.nan, dtype=np.float64
    )
    rows = 0
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=131_072):
        values = batch.to_pydict()
        for raw_product, raw_leaf, raw_weight in zip(
            values["product_id"],
            values["leaf_node_id"],
            values["assignment_weight"],
            strict=True,
        ):
            product = int(raw_product)
            leaf = int(raw_leaf)
            if product not in product_index:
                raise ValueError("weights contain a non-train product")
            if leaf not in leaf_index:
                raise ValueError("weights contain an anchored or invalid leaf")
            weight = float(raw_weight)
            if not math.isfinite(weight):
                raise ValueError("assignment weights must be finite")
            row = product_index[product]
            column = leaf_index[leaf]
            if not math.isnan(matrix[row, column]):
                raise ValueError("duplicate product/leaf assignment weight")
            matrix[row, column] = weight
            rows += 1
    expected = len(train_products) * len(available_leaves)
    if rows != expected or np.isnan(matrix).any():
        raise ValueError(
            "assignment weights must cover every train-product/available-leaf pair"
        )
    return matrix


def solve_anchored_assignment(
    catalog_rows: list[dict[str, Any]],
    weights: np.ndarray,
    *,
    depth: int,
) -> tuple[dict[int, int], dict[str, float]]:
    """Solve the exact rectangular maximum-weight matching with fixed anchors."""
    leaf_start = 2**depth - 1
    physical_leaves = list(range(leaf_start, leaf_start + 2**depth))
    train_rows = [row for row in catalog_rows if bool(row["train_seen"])]
    anchored_rows = [row for row in catalog_rows if not bool(row["train_seen"])]
    train_products = [int(row["product_id"]) for row in train_rows]
    anchored_leaves = {int(row["leaf_node_id"]) for row in anchored_rows}
    available_leaves = [leaf for leaf in physical_leaves if leaf not in anchored_leaves]
    if weights.shape != (len(train_products), len(available_leaves)):
        raise ValueError(
            "assignment matrix shape does not match train products and free leaves"
        )
    if len(available_leaves) < len(train_products):
        raise ValueError("insufficient unanchored leaf capacity")
    row_indices, column_indices = linear_sum_assignment(weights, maximize=True)
    if len(row_indices) != len(train_products) or set(row_indices) != set(
        range(len(train_products))
    ):
        raise ValueError("exact assignment did not cover every train product")
    assignment = {
        train_products[int(row)]: available_leaves[int(column)]
        for row, column in zip(row_indices, column_indices, strict=True)
    }
    assignment.update(
        {int(row["product_id"]): int(row["leaf_node_id"]) for row in anchored_rows}
    )
    if len(assignment) != len(catalog_rows) or len(set(assignment.values())) != len(
        catalog_rows
    ):
        raise ValueError("assignment violates one-item-per-leaf")

    product_row = {product: index for index, product in enumerate(train_products)}
    leaf_column = {leaf: index for index, leaf in enumerate(available_leaves)}
    fixed_objective = sum(
        float(
            weights[
                product_row[int(row["product_id"])],
                leaf_column[int(row["leaf_node_id"])],
            ]
        )
        for row in train_rows
    )
    optimized_objective = sum(
        float(weights[product_row[product], leaf_column[leaf]])
        for product, leaf in assignment.items()
        if product in product_row
    )
    tolerance = 1e-9 * max(1.0, abs(fixed_objective), abs(optimized_objective))
    if optimized_objective + tolerance < fixed_objective:
        raise ValueError("exact assignment is worse than the feasible fixed assignment")
    return assignment, {
        "fixed_objective": fixed_objective,
        "optimized_objective": optimized_objective,
        "absolute_gain": optimized_objective - fixed_objective,
    }


def _validate_subtree_capacity(assigned_leaves: list[int], depth: int) -> None:
    leaf_start = 2**depth - 1
    offsets = [leaf - leaf_start for leaf in assigned_leaves]
    for level in range(depth + 1):
        capacity = 2 ** (depth - level)
        counts: dict[int, int] = {}
        for offset in offsets:
            prefix = offset >> (depth - level) if level else 0
            counts[prefix] = counts.get(prefix, 0) + 1
        if any(count > capacity for count in counts.values()):
            raise ValueError("assignment exceeds a subtree's physical leaf capacity")


def build_alternating_bundle(
    *,
    catalog_path: Path,
    date_eligibility_path: Path,
    weights_path: Path,
    weight_manifest_path: Path,
    output_dir: Path,
    depth: int = 13,
) -> dict[str, Any]:
    """Write an immutable alternating catalog and remapped date masks."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    weight_manifest = json.loads(weight_manifest_path.read_text())
    validate_weight_manifest(
        weight_manifest,
        weights_path=weights_path,
        catalog_path=catalog_path,
    )
    catalog = _catalog_rows(catalog_path, depth)
    leaf_start = 2**depth - 1
    train_products = [
        int(row["product_id"]) for row in catalog if bool(row["train_seen"])
    ]
    anchored_leaves = {
        int(row["leaf_node_id"]) for row in catalog if not bool(row["train_seen"])
    }
    available_leaves = [
        leaf
        for leaf in range(leaf_start, leaf_start + 2**depth)
        if leaf not in anchored_leaves
    ]
    weights = _weight_matrix(
        weights_path,
        train_products=train_products,
        available_leaves=available_leaves,
    )
    assignment, objectives = solve_anchored_assignment(catalog, weights, depth=depth)
    _validate_subtree_capacity(list(assignment.values()), depth)

    output_dir.mkdir(parents=True)
    assigned_rows = []
    fixed_leaf_by_product = {}
    train_seen_by_product = {}
    for row in catalog:
        product = int(row["product_id"])
        fixed_leaf = int(row["leaf_node_id"])
        assigned_leaf = assignment[product]
        assigned_index = assigned_leaf - leaf_start
        fixed_leaf_by_product[product] = fixed_leaf
        train_seen_by_product[product] = bool(row["train_seen"])
        assigned_rows.append(
            {
                "product_id": product,
                "first_category": row["first_category"],
                "first_registered_date": row["first_registered_date"],
                "fixed_catalog_index": int(row["catalog_index"]),
                "fixed_leaf_node_id": fixed_leaf,
                "catalog_index": assigned_index,
                "path_bits": path_bits(assigned_index, depth),
                "leaf_node_id": assigned_leaf,
                "train_seen": bool(row["train_seen"]),
            }
        )
    assigned_rows.sort(key=lambda row: int(row["catalog_index"]))
    catalog_output = output_dir / "alternating_catalog.parquet"
    pq.write_table(
        pa.Table.from_pylist(assigned_rows), catalog_output, compression="zstd"
    )

    eligibility = pq.read_table(date_eligibility_path)
    required = {"snapshot_date", "product_id", "catalog_index", "leaf_node_id"}
    if missing := required - set(eligibility.column_names):
        raise ValueError(f"date eligibility is missing columns: {sorted(missing)}")
    eligibility_rows = []
    for row in eligibility.to_pylist():
        product = int(row["product_id"])
        if product not in assignment:
            raise ValueError("date eligibility contains product outside catalog")
        leaf = assignment[product]
        eligibility_rows.append(
            {
                "snapshot_date": row["snapshot_date"],
                "product_id": product,
                "catalog_index": leaf - leaf_start,
                "leaf_node_id": leaf,
            }
        )
    eligibility_rows.sort(key=lambda row: (row["snapshot_date"], row["catalog_index"]))
    eligibility_output = output_dir / "alternating_date_eligibility.parquet"
    pq.write_table(
        pa.Table.from_pylist(eligibility_rows), eligibility_output, compression="zstd"
    )

    anchored = [product for product, seen in train_seen_by_product.items() if not seen]
    anchored_unchanged = all(
        assignment[product] == fixed_leaf_by_product[product] for product in anchored
    )
    assignment_hash = _digest_lines(
        [f"{product}\t{assignment[product]}" for product in sorted(assignment)]
    )
    manifest = {
        "contract_version": "ptd-alternating-tree-bundle/v1",
        "status": "complete",
        "depth": depth,
        "physical_leaf_count": 2**depth,
        "item_count": len(catalog),
        "train_seen_item_count": len(train_products),
        "anchored_item_count": len(anchored),
        "available_leaf_count": len(available_leaves),
        "source_catalog_sha256": file_sha256(catalog_path),
        "source_date_eligibility_sha256": file_sha256(date_eligibility_path),
        "weight_manifest_sha256": file_sha256(weight_manifest_path),
        "weights_sha256": file_sha256(weights_path),
        "assignment_sha256": assignment_hash,
        "objective": objectives,
        "artifacts": {
            "alternating_catalog": {
                "path": str(catalog_output),
                "sha256": file_sha256(catalog_output),
            },
            "alternating_date_eligibility": {
                "path": str(eligibility_output),
                "sha256": file_sha256(eligibility_output),
            },
        },
        "checks": {
            "train_only_weight_manifest": True,
            "anchored_items_unchanged": anchored_unchanged,
            "one_item_per_leaf": len(set(assignment.values())) == len(assignment),
            "subtree_capacities_respected": True,
            "fixed_assignment_feasible": True,
            "optimized_objective_not_below_fixed": objectives["absolute_gain"] >= -1e-9,
            "validation_or_test_outcomes_or_teacher_scores_not_read": True,
        },
    }
    if not all(manifest["checks"].values()):
        raise ValueError("alternating bundle failed an invariant")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--date-eligibility", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weight-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=13)
    args = parser.parse_args()
    manifest = build_alternating_bundle(
        catalog_path=args.catalog,
        date_eligibility_path=args.date_eligibility,
        weights_path=args.weights,
        weight_manifest_path=args.weight_manifest,
        output_dir=args.output_dir,
        depth=args.depth,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "assignment_sha256": manifest["assignment_sha256"],
                "objective": manifest["objective"],
            }
        )
    )


if __name__ == "__main__":
    main()
