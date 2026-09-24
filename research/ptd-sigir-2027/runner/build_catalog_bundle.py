#!/usr/bin/env python3
"""Build the immutable PTD catalog tree and date masks from non-outcome columns."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN_DATES = ("2026-07-18", "2026-07-19", "2026-07-20")
VALID_DATES = ("2026-07-21",)
TEST_DATES = ("2026-08-12", "2026-08-14", "2026-08-25", "2026-08-26", "2026-08-28")
ALL_DATES = TRAIN_DATES + VALID_DATES + TEST_DATES
EXPECTED_ITEM_COUNT = 5584
EXPECTED_CATALOG_ORDER_SHA256 = "dd41695bb4de9a7d09bae0237cdb2f0c5f1a08b572a5647cdba9c5165bb31d61"
EXPECTED_DATE_HASHES = {
    "2026-07-18": "b989f2147fd6b31498e50c1a58ebd1a7a4b2760901a391140bef64f030c11586",
    "2026-07-19": "2122897fbaee8c18284d4375231af2153c88324211983cd7cf940b78c95e1f36",
    "2026-07-20": "3989b880ff98c5babe13f2893c0534de9863a84d737fb6a8e02f2d287e6b58d6",
    "2026-07-21": "4c84fc3319cd6231568d0bbabe400ec3aeae16d90c274348422a6f3e38fc6313",
    "2026-08-12": "5e71dc506230819f22f35527310c5c241a705300c0e61bfb5a6ee5ef537a03d4",
    "2026-08-14": "5600039a543faa8235b26b3a052f870acb2e64ecdbd36c950390f65519367927",
    "2026-08-25": "e1fc0e68422812a637a6ba1eb46b25c0e7cbe9ac1460b71096712910ba156dd8",
    "2026-08-26": "bb63904e1b8863f0632381a26d1f5917feabdeacbbbb3625886a6980826f7861",
    "2026-08-28": "fcc577e51c932f81b5f7c99f7f8a1a754c8013e8dd329c338be195919b7ab46a",
}


def digest_lines(lines: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_bits(index: int, depth: int) -> str:
    if index < 0 or index >= 2**depth:
        raise ValueError("leaf index is outside fixed tree capacity")
    return format(index, f"0{depth}b")


def collect_catalog(
    batches: Iterable[pa.RecordBatch],
) -> tuple[dict[int, dict[str, set[str]]], dict[str, set[int]], int]:
    """Collect item/category/date sets from batches containing only three fields."""
    categories: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    items_by_date: dict[str, set[int]] = defaultdict(set)
    rows = 0
    for batch in batches:
        required = {"snapshot_date", "product_id", "first_category"}
        if missing := required - set(batch.schema.names):
            raise ValueError(f"catalog batch missing columns: {sorted(missing)}")
        dates = batch.column(batch.schema.get_field_index("snapshot_date")).to_pylist()
        products = batch.column(batch.schema.get_field_index("product_id")).to_pylist()
        raw_categories = batch.column(batch.schema.get_field_index("first_category")).to_pylist()
        for raw_date, raw_product, raw_category in zip(
            dates, products, raw_categories, strict=True
        ):
            snapshot_date = raw_date.isoformat() if hasattr(raw_date, "isoformat") else str(raw_date)
            if raw_product is None:
                raise ValueError("catalog input contains null product_id")
            product_id = int(raw_product)
            category = "" if raw_category is None else str(raw_category)
            categories[product_id][snapshot_date].add(category)
            items_by_date[snapshot_date].add(product_id)
            rows += 1
    return categories, items_by_date, rows


def canonical_catalog(
    categories: dict[int, dict[str, set[str]]],
) -> list[tuple[int, str, str]]:
    """Return product, first point-in-time category, and first date in fixed order."""
    values: list[tuple[int, str, str]] = []
    for product_id, by_date in categories.items():
        if not by_date:
            raise ValueError("catalog item has no observed date")
        all_categories = set().union(*by_date.values())
        if len(all_categories) != 1 or any(len(value) != 1 for value in by_date.values()):
            raise ValueError(f"category conflict for product {product_id}")
        first_date = min(by_date)
        category = next(iter(by_date[first_date]))
        values.append((product_id, category, first_date))
    return sorted(values, key=lambda value: (value[1], str(value[0])))


def contract_summary(
    catalog: Sequence[tuple[int, str, str]],
    items_by_date: dict[str, set[int]],
) -> dict:
    catalog_lines = [f"{category}\t{product_id}" for product_id, category, _ in catalog]
    date_hashes = {
        snapshot_date: digest_lines(sorted(str(product) for product in products))
        for snapshot_date, products in sorted(items_by_date.items())
    }
    train = set().union(*(items_by_date.get(value, set()) for value in TRAIN_DATES))
    valid = set().union(*(items_by_date.get(value, set()) for value in VALID_DATES))
    test = set().union(*(items_by_date.get(value, set()) for value in TEST_DATES))
    return {
        "catalog_order_sha256": digest_lines(catalog_lines),
        "date_eligibility_sha256": date_hashes,
        "item_count": len(catalog),
        "split_unique_items": {"train": len(train), "valid": len(valid), "test": len(test)},
        "pretest_unique_items": len(train | valid),
        "test_only_vs_pretest": len(test - (train | valid)),
    }


def validate_registered_contract(summary: dict) -> None:
    if summary["item_count"] != EXPECTED_ITEM_COUNT:
        raise ValueError("registered catalog item count mismatch")
    if summary["catalog_order_sha256"] != EXPECTED_CATALOG_ORDER_SHA256:
        raise ValueError("registered catalog order hash mismatch")
    if summary["date_eligibility_sha256"] != EXPECTED_DATE_HASHES:
        raise ValueError("registered date eligibility hashes mismatch")
    if summary["split_unique_items"] != {"train": 3126, "valid": 2815, "test": 4592}:
        raise ValueError("registered split item counts mismatch")
    if summary["pretest_unique_items"] != 3245 or summary["test_only_vs_pretest"] != 2339:
        raise ValueError("registered cold-item overlap mismatch")


def _batches(paths: Sequence[Path], batch_size: int) -> Iterable[pa.RecordBatch]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["snapshot_date", "product_id", "first_category"],
            use_threads=True,
        ):
            yield batch


def build_bundle(
    input_paths: Sequence[Path],
    output_dir: Path,
    *,
    depth: int = 13,
    verify_registered: bool = True,
    batch_size: int = 131_072,
) -> dict:
    """Build catalog, complete-tree nodes, date masks, and a hash manifest."""
    if not input_paths:
        raise ValueError("catalog builder needs at least one input")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    categories, items_by_date, rows = collect_catalog(_batches(input_paths, batch_size))
    source_files = [
        {"path": str(path), "sha256": file_sha256(path), "rows": pq.ParquetFile(path).metadata.num_rows}
        for path in input_paths
    ]
    return _write_bundle(
        categories,
        items_by_date,
        rows,
        source_files,
        output_dir,
        depth=depth,
        verify_registered=verify_registered,
    )


def build_bundle_from_inventory(
    inventory_path: Path,
    output_dir: Path,
    *,
    depth: int = 13,
    batch_size: int = 131_072,
    max_workers: int = 16,
) -> dict:
    """Build the registered bundle directly from generation-pinned GCS objects."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    from scripts.inventory_raw_inputs import GsutilRangeFile

    inventory_bytes = inventory_path.read_bytes()
    inventory = json.loads(inventory_bytes)
    if inventory.get("contract_version") != "ptd-raw-input-inventory/v1":
        raise ValueError("raw input inventory contract mismatch")

    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    def inspect_remote(item: dict) -> tuple[dict[int, dict[str, set[str]]], dict[str, set[int]], int]:
        parquet = pq.ParquetFile(GsutilRangeFile(item["uri"], int(item["size_bytes"])))
        return collect_catalog(
            parquet.iter_batches(
                batch_size=batch_size,
                columns=["snapshot_date", "product_id", "first_category"],
                use_threads=False,
            )
        )

    categories: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    items_by_date: dict[str, set[int]] = defaultdict(set)
    rows = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        partials = executor.map(inspect_remote, inventory["objects"])
        for partial_categories, partial_dates, partial_rows in partials:
            for product_id, by_date in partial_categories.items():
                for snapshot_date, values in by_date.items():
                    categories[product_id][snapshot_date].update(values)
            for snapshot_date, products in partial_dates.items():
                items_by_date[snapshot_date].update(products)
            rows += partial_rows
    source_files = [
        {
            key: item[key]
            for key in ("uri", "generation", "crc32c_base64", "md5_base64", "rows")
        }
        for item in inventory["objects"]
    ]
    manifest = _write_bundle(
        categories,
        items_by_date,
        rows,
        source_files,
        output_dir,
        depth=depth,
        verify_registered=True,
    )
    manifest["raw_input_inventory_sha256"] = hashlib.sha256(inventory_bytes).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _write_bundle(
    categories: dict[int, dict[str, set[str]]],
    items_by_date: dict[str, set[int]],
    rows: int,
    source_files: list[dict],
    output_dir: Path,
    *,
    depth: int,
    verify_registered: bool,
) -> dict:
    if verify_registered and set(items_by_date) != set(ALL_DATES):
        raise ValueError(f"registered snapshot dates mismatch: {sorted(items_by_date)}")
    catalog = canonical_catalog(categories)
    if len(catalog) > 2**depth:
        raise ValueError("catalog exceeds tree capacity")
    summary = contract_summary(catalog, items_by_date)
    if verify_registered:
        validate_registered_contract(summary)

    output_dir.mkdir(parents=True)
    train_items = set().union(*(items_by_date.get(value, set()) for value in TRAIN_DATES))
    catalog_table = pa.table(
        {
            "product_id": pa.array([value[0] for value in catalog], type=pa.int64()),
            "first_category": pa.array([value[1] for value in catalog], type=pa.string()),
            "first_registered_date": pa.array(
                [date.fromisoformat(value[2]) for value in catalog], type=pa.date32()
            ),
            "catalog_index": pa.array(range(len(catalog)), type=pa.int32()),
            "path_bits": pa.array(
                [path_bits(index, depth) for index in range(len(catalog))], type=pa.string()
            ),
            "leaf_node_id": pa.array(
                [(2**depth - 1) + index for index in range(len(catalog))], type=pa.int32()
            ),
            "train_seen": pa.array([value[0] in train_items for value in catalog], type=pa.bool_()),
        }
    )
    catalog_path = output_dir / "catalog.parquet"
    pq.write_table(catalog_table, catalog_path, compression="zstd")

    index_by_product = {value[0]: index for index, value in enumerate(catalog)}
    eligibility_rows = [
        {
            "snapshot_date": date.fromisoformat(snapshot_date),
            "product_id": product_id,
            "catalog_index": index_by_product[product_id],
            "leaf_node_id": (2**depth - 1) + index_by_product[product_id],
        }
        for snapshot_date in sorted(items_by_date)
        for product_id in sorted(items_by_date[snapshot_date], key=lambda value: index_by_product[value])
    ]
    eligibility_path = output_dir / "date_eligibility.parquet"
    pq.write_table(pa.Table.from_pylist(eligibility_rows), eligibility_path, compression="zstd")

    nodes = []
    for level in range(depth + 1):
        for offset in range(2**level):
            node_id = (2**level - 1) + offset
            nodes.append(
                {
                    "node_id": node_id,
                    "level": level,
                    "path_bits": path_bits(offset, level) if level else "",
                    "parent_node_id": None if level == 0 else (node_id - 1) // 2,
                    "is_leaf": level == depth,
                    "is_padding_leaf": level == depth and offset >= len(catalog),
                }
            )
    nodes_path = output_dir / "complete_binary_nodes.parquet"
    pq.write_table(pa.Table.from_pylist(nodes), nodes_path, compression="zstd")

    manifest = {
        "contract_version": "ptd-catalog-tree-bundle/v1",
        "status": "complete",
        "depth": depth,
        "branching_factor": 2,
        "physical_leaf_count": 2**depth,
        "padding_leaf_count": 2**depth - len(catalog),
        "rows_scanned": rows,
        **summary,
        "columns_read": ["snapshot_date", "product_id", "first_category"],
        "user_or_outcome_columns_read": [],
        "source_files": source_files,
        "artifacts": {
            "catalog": {"path": str(catalog_path), "sha256": file_sha256(catalog_path)},
            "date_eligibility": {
                "path": str(eligibility_path),
                "sha256": file_sha256(eligibility_path),
            },
            "complete_binary_nodes": {
                "path": str(nodes_path),
                "sha256": file_sha256(nodes_path),
            },
        },
        "checks": {
            "registered_contract_match": verify_registered,
            "category_conflicts": 0,
            "one_item_per_leaf": True,
            "date_masks_fixed": True,
            "users_or_outcomes_read": False,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_paths(patterns: Sequence[str]) -> list[Path]:
    paths = sorted({Path(value) for pattern in patterns for value in glob.glob(pattern)})
    if not paths:
        raise ValueError("input patterns matched no local parquet files")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", action="append", help="Local Parquet glob; repeatable")
    source.add_argument("--inventory", type=Path, help="Generation-pinned raw GCS inventory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=131_072)
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()
    if args.inventory:
        manifest = build_bundle_from_inventory(
            args.inventory,
            args.output_dir,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
        )
    else:
        manifest = build_bundle(
            parse_paths(args.input),
            args.output_dir,
            batch_size=args.batch_size,
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "item_count": manifest["item_count"],
                "catalog_order_sha256": manifest["catalog_order_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
