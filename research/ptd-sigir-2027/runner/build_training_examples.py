#!/usr/bin/env python3
"""Materialize registered depth-13 PTD training examples without reading test labels."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from reference.evidence_gate import EXPECTED_SPLIT
from runner.build_retrieval_queries import KEY_COLUMNS, TEACHER_COLUMNS, _records
from runner.materialize_teacher_scores import EXPECTED_CHECKPOINT_SHA256, sha256

EPSILON_ITEM = 1e-6
EPSILON_NODE = 1e-12
DISTILLATION_ITEM = 1
DISTILLATION_NODE = 2
INTERNAL_CATEGORY = "__internal__"
PADDING_CATEGORY = "__padding__"
ALLOWED_DATES = tuple(EXPECTED_SPLIT["train"] + EXPECTED_SPLIT["validation"])
ALL_REGISTERED_DATES = tuple(
    EXPECTED_SPLIT["train"]
    + EXPECTED_SPLIT["validation"]
    + EXPECTED_SPLIT["test"]
)
RAW_COLUMNS = (
    "snapshot_id",
    "snapshot_date",
    "user_id",
    "product_id",
    "candidate_rank",
    "click_seq_product_id",
    "purchase_seq_product_id",
    "first_category",
    "label_purchase",
)


def _date(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _history(value: Any, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{label} history must be a semicolon-delimited string")
    tokens = [token.strip() for token in value.split(";") if token.strip()]
    if len(tokens) > 30:
        raise ValueError(f"{label} history exceeds 30 items")
    for token in tokens:
        int(token)
    return ";".join(tokens)


def _source_date(path: Path) -> tuple[str, int]:
    """Read only the date column so excluded shards never expose outcome fields."""
    parquet = pq.ParquetFile(path)
    observed: set[str] = set()
    rows = 0
    for batch in parquet.iter_batches(columns=["snapshot_date"], batch_size=131_072):
        rows += batch.num_rows
        observed.update(_date(value) for value in batch.column(0).to_pylist())
        if len(observed) > 1:
            raise ValueError(f"raw shard spans multiple dates: {path}")
    if len(observed) != 1:
        raise ValueError(f"raw shard has no snapshot date: {path}")
    value = next(iter(observed))
    if value not in ALL_REGISTERED_DATES:
        raise ValueError(f"raw shard has an unregistered date: {value}")
    return value, rows


def _logit_sibling_target(left: float, right: float) -> tuple[float, float]:
    logits = []
    for value in (left, right):
        clipped = min(1.0 - EPSILON_ITEM, max(EPSILON_ITEM, value))
        logits.append(math.log(clipped / (1.0 - clipped)))
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = sum(weights)
    return weights[0] / total, weights[1] / total


class LockedCatalog:
    def __init__(self, catalog_path: Path, eligibility_path: Path) -> None:
        catalog = pq.read_table(catalog_path)
        required = {"product_id", "first_category", "leaf_node_id", "path_bits"}
        if missing := required - set(catalog.column_names):
            raise ValueError(f"catalog is missing columns: {sorted(missing)}")
        rows = catalog.to_pylist()
        if not rows:
            raise ValueError("catalog is empty")
        depths = {len(str(row["path_bits"])) for row in rows}
        if len(depths) != 1:
            raise ValueError("catalog paths have inconsistent depths")
        self.depth = next(iter(depths))
        if self.depth != 13:
            raise ValueError("registered training examples require depth 13")
        self.leaf_start = 2**self.depth - 1
        self.leaf_by_product: dict[int, int] = {}
        self.product_by_leaf: dict[int, int] = {}
        self.category_by_product: dict[int, str] = {}
        for row in rows:
            product = int(row["product_id"])
            leaf = int(row["leaf_node_id"])
            offset = leaf - self.leaf_start
            if offset < 0 or offset >= 2**self.depth:
                raise ValueError("catalog leaf is outside the complete tree")
            if str(row["path_bits"]) != format(offset, f"0{self.depth}b"):
                raise ValueError("catalog path and leaf ID disagree")
            if product in self.leaf_by_product or leaf in self.product_by_leaf:
                raise ValueError("catalog product and leaf assignments must be unique")
            self.leaf_by_product[product] = leaf
            self.product_by_leaf[leaf] = product
            self.category_by_product[product] = str(row["first_category"] or "")

        eligibility = pq.read_table(eligibility_path)
        required = {"snapshot_date", "product_id", "leaf_node_id"}
        if missing := required - set(eligibility.column_names):
            raise ValueError(f"date eligibility is missing columns: {sorted(missing)}")
        self.eligible_products: dict[str, set[int]] = defaultdict(set)
        for row in eligibility.to_pylist():
            snapshot_date = _date(row["snapshot_date"])
            product = int(row["product_id"])
            leaf = int(row["leaf_node_id"])
            if self.leaf_by_product.get(product) != leaf:
                raise ValueError("date eligibility product/leaf differs from catalog")
            self.eligible_products[snapshot_date].add(product)
        if set(self.eligible_products) != set(ALL_REGISTERED_DATES):
            raise ValueError("date eligibility must cover exactly the registered dates")

    def descriptor(self, node_id: int, level: int) -> tuple[str, str]:
        if level < self.depth:
            return f"node:{node_id}", INTERNAL_CATEGORY
        product = self.product_by_leaf.get(node_id)
        if product is None:
            return f"padding:{node_id}", PADDING_CATEGORY
        return str(product), self.category_by_product[product]


def _paired_allowed_records(
    teacher_manifest: Mapping[str, Any], *, batch_size: int
) -> tuple[Iterator[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    source_files = teacher_manifest.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("teacher manifest has no raw source files")
    teacher_output = teacher_manifest.get("output", {})
    teacher_path = Path(teacher_output.get("path", ""))
    if not teacher_path.is_file() or sha256(teacher_path) != teacher_output.get("sha256"):
        raise ValueError("teacher score output path/hash mismatch")

    normalized_sources: list[dict[str, Any]] = []
    for source in source_files:
        path = Path(source["path"])
        if not path.is_file() or sha256(path) != source["sha256"]:
            raise ValueError("raw source path/hash mismatch")
        snapshot_date, rows = _source_date(path)
        if rows != int(source["rows"]):
            raise ValueError("raw source row count mismatch")
        normalized_sources.append(
            {
                "path": path,
                "sha256": source["sha256"],
                "rows": rows,
                "snapshot_date": snapshot_date,
                "included": snapshot_date in ALLOWED_DATES,
            }
        )

    def iterator() -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        teachers = iter(_records([teacher_path], TEACHER_COLUMNS, batch_size=batch_size))
        sentinel = object()
        for source in normalized_sources:
            if source["included"]:
                raw_rows = _records([source["path"]], RAW_COLUMNS, batch_size=batch_size)
                for raw in raw_rows:
                    teacher = next(teachers, sentinel)
                    if teacher is sentinel:
                        raise ValueError("teacher scores end before raw rows")
                    for name in KEY_COLUMNS:
                        raw_value = _date(raw[name]) if name == "snapshot_date" else raw[name]
                        teacher_value = (
                            _date(teacher[name]) if name == "snapshot_date" else teacher[name]
                        )
                        if raw_value != teacher_value:
                            raise ValueError(f"raw/teacher key mismatch in {name}")
                    yield raw, teacher
            else:
                for _ in range(source["rows"]):
                    teacher = next(teachers, sentinel)
                    if teacher is sentinel:
                        raise ValueError("teacher scores end before excluded raw shard")
                    if _date(teacher["snapshot_date"]) != source["snapshot_date"]:
                        raise ValueError("excluded raw shard and teacher dates differ")
        if next(teachers, sentinel) is not sentinel:
            raise ValueError("teacher scores contain rows beyond raw sources")

    return iterator(), normalized_sources


def _query_rows(
    *,
    catalog: LockedCatalog,
    snapshot_date: str,
    snapshot_id: str,
    user_id: str,
    click_history: str,
    purchase_history: str,
    candidates: Mapping[int, Mapping[str, Any]],
    first_path_group_id: int,
) -> list[dict[str, Any]]:
    if set(candidates) != catalog.eligible_products[snapshot_date]:
        raise ValueError("query candidate set differs from the locked date mask")
    node_mass: dict[int, float] = defaultdict(float)
    for product, candidate in candidates.items():
        if catalog.category_by_product[product] != candidate["category"]:
            raise ValueError("query category differs from the locked catalog")
        score = float(candidate["teacher_purchase"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("teacher purchase score must be finite and bounded")
        offset = catalog.leaf_by_product[product] - catalog.leaf_start
        for level in range(1, catalog.depth + 1):
            prefix = offset >> (catalog.depth - level)
            node_mass[(2**level - 1) + prefix] += score

    positives = sorted(
        product for product, candidate in candidates.items() if candidate["label_purchase"] == 1
    )
    output: list[dict[str, Any]] = []
    for path_index, product in enumerate(positives):
        leaf = catalog.leaf_by_product[product]
        bits = format(leaf - catalog.leaf_start, f"0{catalog.depth}b")
        parent = 0
        path_group_id = first_path_group_id + path_index
        for level, raw_bit in enumerate(bits, start=1):
            left = 2 * parent + 1
            right = left + 1
            left_key, left_category = catalog.descriptor(left, level)
            right_key, right_category = catalog.descriptor(right, level)
            if level == catalog.depth:
                left_product = catalog.product_by_leaf.get(left)
                right_product = catalog.product_by_leaf.get(right)
                teacher_left, teacher_right = _logit_sibling_target(
                    float(candidates.get(left_product, {}).get("teacher_purchase", 0.0)),
                    float(candidates.get(right_product, {}).get("teacher_purchase", 0.0)),
                )
                distillation_kind = DISTILLATION_ITEM
            else:
                left_mass = node_mass.get(left, 0.0) + EPSILON_NODE
                right_mass = node_mass.get(right, 0.0) + EPSILON_NODE
                total = left_mass + right_mass
                teacher_left, teacher_right = left_mass / total, right_mass / total
                distillation_kind = DISTILLATION_NODE
            output.append(
                {
                    "snapshot_date": date.fromisoformat(snapshot_date),
                    "snapshot_id": snapshot_id,
                    "user_id": user_id,
                    "positive_product_id": product,
                    "path_group_id": path_group_id,
                    "level": level,
                    "positive_child": int(raw_bit),
                    "click_seq_product_id": click_history,
                    "purchase_seq_product_id": purchase_history,
                    "left_item_key": left_key,
                    "right_item_key": right_key,
                    "left_category": left_category,
                    "right_category": right_category,
                    "parent_node_id": parent,
                    "teacher_left": teacher_left,
                    "teacher_right": teacher_right,
                    "distillation_kind": distillation_kind,
                }
            )
            parent = right if raw_bit == "1" else left
    return output


def build_training_examples(
    *,
    teacher_manifest_path: Path,
    catalog_path: Path,
    date_eligibility_path: Path,
    output_path: Path,
    output_manifest_path: Path,
    batch_size: int = 65_536,
) -> dict[str, Any]:
    """Write immutable train/validation path rows from aligned raw/teacher inputs."""
    if output_path.exists() or output_manifest_path.exists():
        raise FileExistsError("refusing to overwrite training-example outputs")
    teacher_manifest = json.loads(teacher_manifest_path.read_text())
    if teacher_manifest.get("contract_version") != "ptd-frozen-teacher-scores/v1":
        raise ValueError("frozen teacher score manifest contract mismatch")
    if teacher_manifest.get("teacher_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen teacher checkpoint mismatch")
    if teacher_manifest.get("label_columns_read") != []:
        raise ValueError("teacher materialization must not read labels")
    catalog = LockedCatalog(catalog_path, date_eligibility_path)
    paired, sources = _paired_allowed_records(teacher_manifest, batch_size=batch_size)

    rows: list[dict[str, Any]] = []
    paths_by_date: Counter[str] = Counter()
    current_key: tuple[str, str, str] | None = None
    current_click = ""
    current_purchase = ""
    current_candidates: dict[int, dict[str, Any]] = {}
    finalized: set[tuple[str, str, str]] = set()
    path_group_id = 0

    def flush() -> None:
        nonlocal path_group_id
        if current_key is None:
            return
        snapshot_date, user_id, snapshot_id = current_key
        emitted = _query_rows(
            catalog=catalog,
            snapshot_date=snapshot_date,
            snapshot_id=snapshot_id,
            user_id=user_id,
            click_history=current_click,
            purchase_history=current_purchase,
            candidates=current_candidates,
            first_path_group_id=path_group_id,
        )
        rows.extend(emitted)
        count = len(emitted) // catalog.depth
        path_group_id += count
        paths_by_date[snapshot_date] += count

    for raw, teacher in paired:
        snapshot_date = _date(raw["snapshot_date"])
        if snapshot_date not in ALLOWED_DATES:
            raise ValueError("excluded date entered training materialization")
        if not raw["snapshot_id"] or not raw["user_id"]:
            raise ValueError("training row has null/empty snapshot or user ID")
        key = (snapshot_date, str(raw["user_id"]), str(raw["snapshot_id"]))
        click = _history(raw["click_seq_product_id"], "click")
        purchase = _history(raw["purchase_seq_product_id"], "purchase")
        if key != current_key:
            if current_key is not None:
                flush()
                finalized.add(current_key)
            if key in finalized:
                raise ValueError("raw rows are not contiguous by training query")
            current_key = key
            current_click = click
            current_purchase = purchase
            current_candidates = {}
        elif click != current_click or purchase != current_purchase:
            raise ValueError("history fields differ within one training query")
        product = raw["product_id"]
        rank = raw["candidate_rank"]
        label = raw["label_purchase"]
        if product is None or rank is None or label not in (0, 1):
            raise ValueError("training candidate key/label is invalid")
        product = int(product)
        if product in current_candidates:
            raise ValueError("training query has a duplicate product")
        current_candidates[product] = {
            "candidate_rank": int(rank),
            "category": str(raw["first_category"] or ""),
            "label_purchase": int(label),
            "teacher_purchase": float(teacher["teacher_purchase"]),
        }
    if current_key is not None:
        flush()

    if not rows or path_group_id == 0:
        raise ValueError("training materialization produced no purchase-positive paths")
    if len(rows) != path_group_id * catalog.depth:
        raise ValueError("training paths are not complete depth-13 groups")
    if not paths_by_date or not set(paths_by_date) <= set(ALLOWED_DATES):
        raise ValueError("training example dates are invalid")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        staged_output = Path(directory) / "training_examples.parquet"
        pq.write_table(pa.Table.from_pylist(rows), staged_output, compression="zstd")
        output_hash = sha256(staged_output)
        manifest = {
            "contract_version": "ptd-training-examples/v1",
            "status": "complete",
            "teacher_manifest_sha256": sha256(teacher_manifest_path),
            "teacher_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "catalog_sha256": sha256(catalog_path),
            "date_eligibility_sha256": sha256(date_eligibility_path),
            "depth": catalog.depth,
            "included_dates": list(ALLOWED_DATES),
            "excluded_dates": list(EXPECTED_SPLIT["test"]),
            "path_groups": path_group_id,
            "sibling_rows": len(rows),
            "paths_by_date": dict(sorted(paths_by_date.items())),
            "source_files": [
                {
                    "path": str(source["path"]),
                    "sha256": source["sha256"],
                    "rows": source["rows"],
                    "snapshot_date": source["snapshot_date"],
                    "outcome_columns_read": bool(source["included"]),
                }
                for source in sources
            ],
            "output": {"path": str(output_path), "sha256": output_hash},
            "checks": {
                "raw_teacher_keys_match_for_included_dates": True,
                "candidate_sets_match_locked_date_masks": True,
                "complete_depth_13_paths": True,
                "teacher_targets_finite_and_normalized": True,
                "test_outcome_columns_excluded": True,
                "test_teacher_scores_excluded": True,
                "no_overwrite": True,
            },
        }
        staged_manifest = Path(directory) / "manifest.json"
        staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(staged_output, output_path)
        os.replace(staged_manifest, output_manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--date-eligibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=65_536)
    args = parser.parse_args()
    manifest = build_training_examples(
        teacher_manifest_path=args.teacher_manifest,
        catalog_path=args.catalog,
        date_eligibility_path=args.date_eligibility,
        output_path=args.output,
        output_manifest_path=args.manifest,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "path_groups": manifest["path_groups"],
                "output": manifest["output"],
            }
        )
    )


if __name__ == "__main__":
    main()
