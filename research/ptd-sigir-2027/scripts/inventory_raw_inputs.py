#!/usr/bin/env python3
"""Inventory immutable raw PTD input shards without persisting row payloads."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST_URI = (
    "gs://kauche-app-lab-product-recommend/home-feed-cvr-lab/offline_gate_full/"
    "f065bd20db-five-day-esmm-cold-start-v1/shared_bottom_esmm_v2/source/manifest.json"
)
SOURCE_MANIFEST_SHA256 = "33e97c72a3925b5b1ffe5550fbf625e82cacf78f18e073c0be9a2718b4fa1748"
RAW_INPUT_GLOB = (
    "gs://kauche-app-lab-product-recommend/home-feed-cvr-lab/offline_gate_full/"
    "f065bd20db-five-day-esmm-cold-start-v1/input/*.parquet"
)
REQUIRED_COLUMNS = (
    "snapshot_id",
    "snapshot_date",
    "snapshot_timestamp_utc",
    "user_id",
    "product_id",
    "candidate_rank",
    "click_seq_product_id",
    "purchase_seq_product_id",
    "first_category",
    "feature_source_max_timestamp_utc",
    "label_click",
    "label_purchase",
)


class GsutilRangeFile(io.RawIOBase):
    """Seekable range reader that lets PyArrow inspect only Parquet metadata."""

    def __init__(self, uri: str, size: int) -> None:
        self.uri = uri
        self.size = size
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        end = self.size - 1 if size is None or size < 0 else min(self.size, self.position + size) - 1
        if end < self.position:
            return b""
        completed = subprocess.run(
            ["gsutil", "cat", "-r", f"{self.position}-{end}", self.uri],
            check=True,
            capture_output=True,
        )
        data = completed.stdout
        self.position += len(data)
        return data


def command_json(arguments: list[str]) -> Any:
    completed = subprocess.run(arguments, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def gsutil_bytes(uri: str) -> bytes:
    return subprocess.run(["gsutil", "cat", uri], check=True, capture_output=True).stdout


def json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    return value


def column_bounds(parquet_file: pq.ParquetFile, column_name: str) -> tuple[Any | None, Any | None]:
    column_index = parquet_file.schema_arrow.get_field_index(column_name)
    if column_index < 0:
        return None, None
    minima: list[Any] = []
    maxima: list[Any] = []
    for row_group in range(parquet_file.num_row_groups):
        statistics = parquet_file.metadata.row_group(row_group).column(column_index).statistics
        if statistics is not None and statistics.has_min_max:
            minima.append(statistics.min)
            maxima.append(statistics.max)
    if not minima:
        return None, None
    return json_value(min(minima)), json_value(max(maxima))


def inspect_object(value: dict[str, Any]) -> dict[str, Any]:
    metadata = value["metadata"]
    uri = f"gs://{metadata['bucket']}/{metadata['name']}"
    size = int(metadata["size"])
    parquet_file = pq.ParquetFile(GsutilRangeFile(uri, size))
    schema = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in parquet_file.schema_arrow
    ]
    schema_sha256 = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    minimum_date, maximum_date = column_bounds(parquet_file, "snapshot_date")
    if minimum_date == maximum_date:
        snapshot_date_rows = {str(minimum_date): parquet_file.metadata.num_rows}
    else:
        dates = pq.read_table(
            GsutilRangeFile(uri, size),
            columns=["snapshot_date"],
            use_threads=False,
        )["snapshot_date"].to_pylist()
        snapshot_date_rows = dict(sorted(Counter(value.isoformat() for value in dates).items()))
    return {
        "uri": uri,
        "generation": metadata["generation"],
        "size_bytes": size,
        "crc32c_base64": metadata["crc32c"],
        "md5_base64": metadata.get("md5Hash"),
        "rows": parquet_file.metadata.num_rows,
        "row_groups": parquet_file.num_row_groups,
        "snapshot_date_min": minimum_date,
        "snapshot_date_max": maximum_date,
        "snapshot_date_rows": snapshot_date_rows,
        "schema_sha256": schema_sha256,
        "schema": schema,
    }


def build_inventory(max_workers: int) -> dict[str, Any]:
    source_bytes = gsutil_bytes(SOURCE_MANIFEST_URI)
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_MANIFEST_SHA256:
        raise ValueError("source manifest hash mismatch")
    source = json.loads(source_bytes)
    objects = command_json(["gcloud", "storage", "ls", "--json", RAW_INPUT_GLOB])
    if not isinstance(objects, list) or not objects:
        raise ValueError("raw input listing is empty")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        inspected = list(executor.map(inspect_object, objects))
    inspected.sort(key=lambda item: item["uri"])
    schema_hashes = {item["schema_sha256"] for item in inspected}
    if len(schema_hashes) != 1:
        raise ValueError(f"raw input schemas differ: {sorted(schema_hashes)}")
    schema = inspected[0]["schema"]
    columns = {field["name"] for field in schema}
    missing = sorted(set(REQUIRED_COLUMNS) - columns)
    if missing:
        raise ValueError(f"raw input schema is missing PTD columns: {missing}")
    expected_rows = sum(int(value) for value in source["row_counts"].values())
    observed_rows = sum(item["rows"] for item in inspected)
    if observed_rows != expected_rows:
        raise ValueError(f"raw/source row counts differ: {observed_rows} != {expected_rows}")
    rows_by_date: dict[str, int] = {}
    objects_by_date: dict[str, int] = {}
    for item in inspected:
        for snapshot_date, rows in item["snapshot_date_rows"].items():
            rows_by_date[snapshot_date] = rows_by_date.get(snapshot_date, 0) + int(rows)
            objects_by_date[snapshot_date] = objects_by_date.get(snapshot_date, 0) + 1
    rows_by_split = {
        split: sum(rows_by_date[date] for date in dates)
        for split, dates in source["split_dates"].items()
    }
    if rows_by_split != {key: int(value) for key, value in source["row_counts"].items()}:
        raise ValueError(f"raw/source split row counts differ: {rows_by_split} != {source['row_counts']}")
    object_identity = [
        {
            key: item[key]
            for key in ("uri", "generation", "size_bytes", "crc32c_base64", "md5_base64", "rows", "schema_sha256")
        }
        for item in inspected
    ]
    return {
        "contract_version": "ptd-raw-input-inventory/v1",
        "status": "VERIFIED",
        "source_manifest": {
            "uri": SOURCE_MANIFEST_URI,
            "sha256": SOURCE_MANIFEST_SHA256,
            "source_contract_sha256": source["source_contract_sha256"],
        },
        "raw_input_glob": RAW_INPUT_GLOB,
        "object_count": len(inspected),
        "boundary_shard_count": sum(
            item["snapshot_date_min"] != item["snapshot_date_max"] for item in inspected
        ),
        "total_rows": observed_rows,
        "total_size_bytes": sum(item["size_bytes"] for item in inspected),
        "rows_by_date": dict(sorted(rows_by_date.items())),
        "objects_by_date": dict(sorted(objects_by_date.items())),
        "rows_by_split": rows_by_split,
        "schema_sha256": inspected[0]["schema_sha256"],
        "schema": schema,
        "required_ptd_columns": list(REQUIRED_COLUMNS),
        "object_identity_sha256": hashlib.sha256(
            json.dumps(object_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "checks": {
            "uniform_schema": True,
            "required_columns_present": True,
            "row_count_matches_verified_source": True,
            "split_row_counts_match_verified_source": True,
            "row_payloads_persisted_locally": False,
        },
        "objects": [{key: value for key, value in item.items() if key != "schema"} for item in inspected],
        "scope_note": (
            "Metadata-only inventory. It verifies immutable object identity, schema, and row-count alignment; "
            "it does not by itself prove semantic point-in-time correctness."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifact" / "verified" / "raw_input_inventory.json",
    )
    parser.add_argument("--max-workers", type=int, default=12)
    args = parser.parse_args()
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")
    inventory = build_inventory(args.max_workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(
        "raw input inventory: PASS "
        f"({inventory['object_count']} objects, {inventory['total_rows']} rows, no row payloads persisted)"
    )


if __name__ == "__main__":
    main()
