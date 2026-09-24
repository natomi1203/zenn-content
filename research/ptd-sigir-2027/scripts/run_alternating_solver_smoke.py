#!/usr/bin/env python3
"""Exercise exact anchored reassignment on a deterministic synthetic tree."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import scipy  # noqa: E402

from runner.alternating_solver import (  # noqa: E402
    CANDIDATE_LEAF_POLICY,
    WEIGHT_FORMULA,
    build_alternating_bundle,
)
from runner.build_catalog_bundle import file_sha256, path_bits  # noqa: E402


def _write_inputs(
    root: Path,
) -> tuple[Path, Path, Path, Path, dict[int, dict[int, float]]]:
    depth = 3
    leaf_start = 2**depth - 1
    products = [101, 102, 103, 104, 105, 106]
    train_products = products[:4]
    catalog = pa.table(
        {
            "product_id": products,
            "first_category": ["a", "a", "b", "b", "c", "c"],
            "first_registered_date": [date(2026, 7, 18)] * 6,
            "catalog_index": list(range(6)),
            "path_bits": [path_bits(index, depth) for index in range(6)],
            "leaf_node_id": [leaf_start + index for index in range(6)],
            "train_seen": [True, True, True, True, False, False],
        }
    )
    catalog_path = root / "catalog.parquet"
    pq.write_table(catalog, catalog_path, compression="zstd")
    eligibility = pa.table(
        {
            "snapshot_date": [date(2026, 7, 18)] * 6 + [date(2026, 8, 12)] * 6,
            "product_id": products * 2,
            "catalog_index": list(range(6)) * 2,
            "leaf_node_id": [leaf_start + index for index in range(6)] * 2,
        }
    )
    eligibility_path = root / "date_eligibility.parquet"
    pq.write_table(eligibility, eligibility_path, compression="zstd")

    anchored_leaves = {leaf_start + 4, leaf_start + 5}
    available = [
        leaf
        for leaf in range(leaf_start, leaf_start + 2**depth)
        if leaf not in anchored_leaves
    ]
    preferred = dict(zip(train_products, reversed(available[:4]), strict=True))
    weights = {
        product: {
            leaf: (10.0 if leaf == preferred[product] else 0.0) for leaf in available
        }
        for product in train_products
    }
    rows = [
        {"product_id": product, "leaf_node_id": leaf, "assignment_weight": weight}
        for product in train_products
        for leaf, weight in weights[product].items()
    ]
    weights_path = root / "weights.parquet"
    pq.write_table(pa.Table.from_pylist(rows), weights_path, compression="zstd")
    weight_manifest = {
        "contract_version": "ptd-assignment-weights/v1",
        "status": "complete",
        "source_split": "train",
        "source_dates": ["2026-07-18", "2026-07-19", "2026-07-20"],
        "formula": WEIGHT_FORMULA,
        "candidate_leaf_policy": CANDIDATE_LEAF_POLICY,
        "validation_or_test_fields_read": [],
        "catalog_sha256": file_sha256(catalog_path),
        "weights": {
            "path": str(weights_path),
            "sha256": file_sha256(weights_path),
            "rows": len(rows),
        },
    }
    manifest_path = root / "weight_manifest.json"
    manifest_path.write_text(
        json.dumps(weight_manifest, indent=2, sort_keys=True) + "\n"
    )
    return catalog_path, eligibility_path, weights_path, manifest_path, weights


def _brute_force_objective(weights: dict[int, dict[int, float]]) -> float:
    products = sorted(weights)
    leaves = sorted(next(iter(weights.values())))
    return max(
        sum(
            weights[product][leaf]
            for product, leaf in zip(products, chosen, strict=True)
        )
        for chosen in itertools.permutations(leaves, len(products))
    )


def run() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        catalog, eligibility, weights_path, weight_manifest, weights = _write_inputs(
            root
        )
        first_dir = root / "first"
        second_dir = root / "second"
        first = build_alternating_bundle(
            catalog_path=catalog,
            date_eligibility_path=eligibility,
            weights_path=weights_path,
            weight_manifest_path=weight_manifest,
            output_dir=first_dir,
            depth=3,
        )
        second = build_alternating_bundle(
            catalog_path=catalog,
            date_eligibility_path=eligibility,
            weights_path=weights_path,
            weight_manifest_path=weight_manifest,
            output_dir=second_dir,
            depth=3,
        )
        brute_force = _brute_force_objective(weights)
        if first["objective"]["optimized_objective"] != brute_force:
            raise ValueError("exact solver differs from brute-force optimum")
        if first["assignment_sha256"] != second["assignment_sha256"]:
            raise ValueError("assignment changed across identical reruns")
        first_hashes = {
            key: value["sha256"] for key, value in first["artifacts"].items()
        }
        second_hashes = {
            key: value["sha256"] for key, value in second["artifacts"].items()
        }
        if first_hashes != second_hashes:
            raise ValueError("alternating Parquet artifacts changed across reruns")
        no_overwrite = False
        try:
            build_alternating_bundle(
                catalog_path=catalog,
                date_eligibility_path=eligibility,
                weights_path=weights_path,
                weight_manifest_path=weight_manifest,
                output_dir=first_dir,
                depth=3,
            )
        except FileExistsError:
            no_overwrite = True
        contaminated = json.loads(weight_manifest.read_text())
        contaminated["source_dates"][-1] = "2026-08-12"
        contaminated_path = root / "contaminated.json"
        contaminated_path.write_text(json.dumps(contaminated) + "\n")
        rejects_test_date = False
        try:
            build_alternating_bundle(
                catalog_path=catalog,
                date_eligibility_path=eligibility,
                weights_path=weights_path,
                weight_manifest_path=contaminated_path,
                output_dir=root / "contaminated-output",
                depth=3,
            )
        except ValueError:
            rejects_test_date = True
    return {
        "contract_version": "ptd-alternating-solver-smoke/v1",
        "status": "SMOKE_ONLY",
        "empirical_claim_allowed": False,
        "synthetic_tree": {
            "depth": 3,
            "items": 6,
            "train_seen_items": 4,
            "anchored_items": 2,
            "physical_leaves": 8,
        },
        "objective": {
            **first["objective"],
            "brute_force_optimum": brute_force,
        },
        "assignment_sha256": first["assignment_sha256"],
        "checks": {
            **first["checks"],
            "matches_brute_force_optimum": True,
            "deterministic_rerun": True,
            "deterministic_parquet_hashes": True,
            "no_overwrite": no_overwrite,
            "rejects_test_date_weight_manifest": rejects_test_date,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "scipy": scipy.__version__,
            "platform": platform.system(),
        },
        "scope_note": "Synthetic solver evidence only; no PTD metric, real assignment, or efficacy claim is present.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifact" / "smoke" / "alternating_solver_smoke.json",
    )
    args = parser.parse_args()
    payload = run()
    if not all(payload["checks"].values()):
        raise ValueError("alternating solver smoke failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("alternating solver smoke: PASS (exact, anchored, deterministic, train-only)")


if __name__ == "__main__":
    main()
