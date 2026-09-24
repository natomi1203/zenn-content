#!/usr/bin/env python3
"""Validate a real catalog bundle and persist only non-identifying evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "artifact" / "verified" / "raw_input_inventory.json"
BUILDER = ROOT / "runner" / "build_catalog_bundle.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(bundle_dir: Path, rerun_bundle_dir: Path | None = None) -> dict:
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("contract_version") != "ptd-catalog-tree-bundle/v1":
        raise ValueError("catalog bundle contract mismatch")
    if manifest.get("raw_input_inventory_sha256") != sha256(INVENTORY):
        raise ValueError("catalog bundle raw inventory hash mismatch")
    if manifest.get("item_count") != 5584 or manifest.get("depth") != 13:
        raise ValueError("catalog bundle shape mismatch")
    if manifest.get("catalog_order_sha256") != (
        "dd41695bb4de9a7d09bae0237cdb2f0c5f1a08b572a5647cdba9c5165bb31d61"
    ):
        raise ValueError("catalog order hash mismatch")
    if manifest.get("checks") != {
        "registered_contract_match": True,
        "category_conflicts": 0,
        "one_item_per_leaf": True,
        "date_masks_fixed": True,
        "users_or_outcomes_read": False,
    }:
        raise ValueError("catalog bundle checks are incomplete")

    expected_files = {
        "catalog": (bundle_dir / "catalog.parquet", 5584),
        "date_eligibility": (bundle_dir / "date_eligibility.parquet", None),
        "complete_binary_nodes": (bundle_dir / "complete_binary_nodes.parquet", 16383),
    }
    artifacts = {}
    for name, (path, expected_rows) in expected_files.items():
        rows = pq.ParquetFile(path).metadata.num_rows
        if expected_rows is not None and rows != expected_rows:
            raise ValueError(f"{name} row count mismatch")
        if manifest["artifacts"][name]["sha256"] != sha256(path):
            raise ValueError(f"{name} hash mismatch")
        artifacts[name] = {"sha256": sha256(path), "rows": rows}

    deterministic_rerun_verified = False
    if rerun_bundle_dir is not None:
        rerun_manifest = json.loads((rerun_bundle_dir / "manifest.json").read_text())
        if rerun_manifest.get("catalog_order_sha256") != manifest["catalog_order_sha256"]:
            raise ValueError("catalog rerun contract differs")
        for name, value in artifacts.items():
            rerun_path = rerun_bundle_dir / expected_files[name][0].name
            if sha256(rerun_path) != value["sha256"]:
                raise ValueError(f"catalog rerun {name} bytes differ")
        deterministic_rerun_verified = True

    eligibility = pq.read_table(
        expected_files["date_eligibility"][0], columns=["snapshot_date"]
    )["snapshot_date"].to_pylist()
    rows_by_date = dict(sorted(Counter(value.isoformat() for value in eligibility).items()))
    if set(rows_by_date) != set(manifest["date_eligibility_sha256"]):
        raise ValueError("date eligibility rows do not cover every registered date")
    return {
        "contract_version": "ptd-catalog-bundle-evidence/v1",
        "status": "VERIFIED",
        "raw_input_inventory_sha256": manifest["raw_input_inventory_sha256"],
        "builder_sha256": sha256(BUILDER),
        "internal_bundle_manifest_sha256": sha256(manifest_path),
        "item_count": manifest["item_count"],
        "depth": manifest["depth"],
        "physical_leaf_count": manifest["physical_leaf_count"],
        "padding_leaf_count": manifest["padding_leaf_count"],
        "rows_scanned": manifest["rows_scanned"],
        "split_unique_items": manifest["split_unique_items"],
        "pretest_unique_items": manifest["pretest_unique_items"],
        "test_only_vs_pretest": manifest["test_only_vs_pretest"],
        "catalog_order_sha256": manifest["catalog_order_sha256"],
        "date_eligibility_sha256": manifest["date_eligibility_sha256"],
        "eligible_items_by_date": rows_by_date,
        "internal_artifacts": artifacts,
        "columns_read": manifest["columns_read"],
        "user_or_outcome_columns_read": manifest["user_or_outcome_columns_read"],
        "checks": manifest["checks"],
        "deterministic_rerun_verified": deterministic_rerun_verified,
        "identifiers_persisted_in_this_evidence": False,
        "scope_note": (
            "This public evidence contains only counts and hashes. The keyed catalog/mask bundle remains "
            "access-controlled and is not part of the anonymous artifact."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--rerun-bundle-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifact" / "verified" / "catalog_bundle_evidence.json",
    )
    args = parser.parse_args()
    payload = record(args.bundle_dir, args.rerun_bundle_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"catalog bundle evidence: PASS ({payload['item_count']} items, "
        f"{payload['internal_artifacts']['date_eligibility']['rows']} eligibility rows)"
    )


if __name__ == "__main__":
    main()
