#!/usr/bin/env python3
"""Audit PTD item-set overlap without persisting item or user payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from inventory_raw_inputs import GsutilRangeFile, ROOT

INVENTORY_PATH = ROOT / "artifact" / "verified" / "raw_input_inventory.json"
TRAIN_DATES = ("2026-07-18", "2026-07-19", "2026-07-20")
VALID_DATES = ("2026-07-21",)
TEST_DATES = ("2026-08-12", "2026-08-14", "2026-08-25", "2026-08-26", "2026-08-28")
ALL_DATES = TRAIN_DATES + VALID_DATES + TEST_DATES


def digest_lines(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def inspect_object(record: dict[str, Any]) -> dict[str, Any]:
    table = pq.read_table(
        GsutilRangeFile(record["uri"], int(record["size_bytes"])),
        columns=["snapshot_date", "product_id", "is_new_product", "first_category"],
        use_threads=False,
    )
    dates = table["snapshot_date"].to_pylist()
    products = table["product_id"].to_pylist()
    new_flags = table["is_new_product"].to_pylist()
    categories = table["first_category"].to_pylist()
    rows_by_date: Counter[str] = Counter()
    new_rows_by_date: Counter[str] = Counter()
    items_by_date: dict[str, set[str]] = defaultdict(set)
    new_items_by_date: dict[str, set[str]] = defaultdict(set)
    categories_by_item_date: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for raw_date, raw_product, raw_new, raw_category in zip(
        dates, products, new_flags, categories, strict=True
    ):
        snapshot_date = raw_date.isoformat() if hasattr(raw_date, "isoformat") else str(raw_date)
        product_id = str(raw_product)
        category = "" if raw_category is None else str(raw_category)
        rows_by_date[snapshot_date] += 1
        items_by_date[snapshot_date].add(product_id)
        categories_by_item_date[product_id][snapshot_date].add(category)
        if bool(raw_new):
            new_rows_by_date[snapshot_date] += 1
            new_items_by_date[snapshot_date].add(product_id)
    return {
        "rows_by_date": dict(rows_by_date),
        "new_rows_by_date": dict(new_rows_by_date),
        "items_by_date": {key: sorted(value) for key, value in items_by_date.items()},
        "new_items_by_date": {key: sorted(value) for key, value in new_items_by_date.items()},
        "categories_by_item_date": {
            item: {date: sorted(values) for date, values in by_date.items()}
            for item, by_date in categories_by_item_date.items()
        },
    }


def build_audit(max_workers: int) -> dict[str, Any]:
    inventory_bytes = INVENTORY_PATH.read_bytes()
    inventory = json.loads(inventory_bytes)
    objects = inventory["objects"]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        shards = list(executor.map(inspect_object, objects))

    rows_by_date: Counter[str] = Counter()
    new_rows_by_date: Counter[str] = Counter()
    items_by_date: dict[str, set[str]] = defaultdict(set)
    new_items_by_date: dict[str, set[str]] = defaultdict(set)
    categories_by_item_date: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for shard in shards:
        rows_by_date.update(shard["rows_by_date"])
        new_rows_by_date.update(shard["new_rows_by_date"])
        for snapshot_date, products in shard["items_by_date"].items():
            items_by_date[snapshot_date].update(products)
        for snapshot_date, products in shard["new_items_by_date"].items():
            new_items_by_date[snapshot_date].update(products)
        for product_id, by_date in shard["categories_by_item_date"].items():
            for snapshot_date, categories in by_date.items():
                categories_by_item_date[product_id][snapshot_date].update(categories)

    if set(items_by_date) != set(ALL_DATES):
        raise ValueError(f"unexpected snapshot dates: {sorted(items_by_date)}")
    if dict(sorted(rows_by_date.items())) != inventory["rows_by_date"]:
        raise ValueError("item audit row counts do not match the immutable raw inventory")

    split_sets = {
        "train": set().union(*(items_by_date[value] for value in TRAIN_DATES)),
        "valid": set().union(*(items_by_date[value] for value in VALID_DATES)),
        "test": set().union(*(items_by_date[value] for value in TEST_DATES)),
    }
    pretest = split_sets["train"] | split_sets["valid"]
    all_items = pretest | split_sets["test"]

    canonical_category: dict[str, str] = {}
    within_date_category_conflicts = 0
    cross_date_category_conflicts = 0
    for product_id, by_date in categories_by_item_date.items():
        ordered_dates = sorted(by_date)
        if any(len(by_date[value]) != 1 for value in ordered_dates):
            within_date_category_conflicts += 1
        all_categories = set().union(*(by_date[value] for value in ordered_dates))
        if len(all_categories) != 1:
            cross_date_category_conflicts += 1
        # The first observed point-in-time category is deterministic and uses no outcome field.
        canonical_category[product_id] = sorted(by_date[ordered_dates[0]])[0]

    canonical_catalog_lines = [
        f"{canonical_category[product_id]}\t{product_id}" for product_id in sorted(all_items)
    ]
    category_order_lines = sorted(canonical_catalog_lines)
    per_date = {}
    for snapshot_date in ALL_DATES:
        products = items_by_date[snapshot_date]
        per_date[snapshot_date] = {
            "rows": rows_by_date[snapshot_date],
            "unique_items": len(products),
            "new_flag_rows": new_rows_by_date[snapshot_date],
            "new_flag_items": len(new_items_by_date[snapshot_date]),
            "eligible_item_set_sha256": digest_lines(sorted(products)),
        }

    return {
        "contract_version": "ptd-item-universe-audit/v1",
        "status": "VERIFIED",
        "raw_input_inventory": {
            "path": "artifact/verified/raw_input_inventory.json",
            "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        },
        "columns_read": ["snapshot_date", "product_id", "is_new_product", "first_category"],
        "outcome_columns_read": [],
        "user_columns_read": [],
        "per_date": per_date,
        "split_unique_items": {key: len(value) for key, value in split_sets.items()},
        "split_item_set_sha256": {
            key: digest_lines(sorted(value)) for key, value in split_sets.items()
        },
        "pretest_unique_items": len(pretest),
        "all_unique_items": len(all_items),
        "overlap": {
            "validation_only_vs_train": len(split_sets["valid"] - split_sets["train"]),
            "test_only_vs_train": len(split_sets["test"] - split_sets["train"]),
            "test_only_vs_pretest": len(split_sets["test"] - pretest),
            "test_overlap_pretest": len(split_sets["test"] & pretest),
        },
        "catalog_envelope": {
            "scope": "union_of_fixed_candidate_items_across_registered_train_valid_test_dates",
            "item_count": len(all_items),
            "canonical_category_rule": "category_at_earliest_registered_snapshot_then_lexical_tie_break",
            "canonical_mapping_sha256": digest_lines(canonical_catalog_lines),
            "binary_category_order_sha256": digest_lines(category_order_lines),
            "uses_labels": False,
            "uses_teacher_scores": False,
            "transductive_item_identity_and_category_only": True,
        },
        "category_consistency": {
            "items_with_within_date_conflict": within_date_category_conflicts,
            "items_with_cross_date_conflict": cross_date_category_conflicts,
        },
        "checks": {
            "row_count_matches_raw_inventory": True,
            "row_payloads_persisted_locally": False,
            "identifiers_persisted_locally": False,
            "outcomes_or_teacher_scores_read": False,
        },
        "scope_note": (
            "The artifact stores only counts and canonical SHA-256 fingerprints. Product identifiers and "
            "categories are held in memory for set auditing and are not emitted; user IDs and outcomes are not read."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifact" / "verified" / "item_universe_audit.json",
    )
    parser.add_argument("--max-workers", type=int, default=12)
    args = parser.parse_args()
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")
    audit = build_audit(args.max_workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        "item universe audit: PASS "
        f"({audit['all_unique_items']} items; {audit['overlap']['test_only_vs_pretest']} test-only vs pretest)"
    )


if __name__ == "__main__":
    main()
