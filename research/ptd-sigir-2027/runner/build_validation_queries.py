#!/usr/bin/env python3
"""Build immutable validation queries while excluding train/test outcome columns."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from reference.evidence_gate import EXPECTED_SPLIT
from runner.build_retrieval_queries import KEY_COLUMNS, TEACHER_COLUMNS, _records
from runner.build_training_examples import LockedCatalog, _date, _history, _source_date
from runner.materialize_teacher_scores import EXPECTED_CHECKPOINT_SHA256, sha256

VALIDATION_DATES = tuple(EXPECTED_SPLIT["validation"])
RAW_COLUMNS = (
    "snapshot_id",
    "snapshot_date",
    "user_id",
    "product_id",
    "candidate_rank",
    "click_seq_product_id",
    "purchase_seq_product_id",
    "first_category",
    "label_click",
    "label_purchase",
)


def _selected_records(
    teacher_manifest: Mapping[str, Any], *, selected_dates: Sequence[str], batch_size: int
) -> tuple[Iterator[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    source_files = teacher_manifest.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("teacher manifest has no raw source files")
    teacher_output = teacher_manifest.get("output", {})
    teacher_path = Path(teacher_output.get("path", ""))
    if not teacher_path.is_file() or sha256(teacher_path) != teacher_output.get("sha256"):
        raise ValueError("teacher score output path/hash mismatch")
    sources: list[dict[str, Any]] = []
    for source in source_files:
        path = Path(source["path"])
        if not path.is_file() or sha256(path) != source["sha256"]:
            raise ValueError("raw source path/hash mismatch")
        snapshot_date, rows = _source_date(path)
        if rows != int(source["rows"]):
            raise ValueError("raw source row count mismatch")
        sources.append(
            {
                "path": path,
                "sha256": source["sha256"],
                "rows": rows,
                "snapshot_date": snapshot_date,
                "included": snapshot_date in selected_dates,
            }
        )

    def iterator() -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        teachers = iter(_records([teacher_path], TEACHER_COLUMNS, batch_size=batch_size))
        sentinel = object()
        for source in sources:
            if source["included"]:
                for raw in _records([source["path"]], RAW_COLUMNS, batch_size=batch_size):
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

    return iterator(), sources


def build_split_queries(
    *,
    teacher_manifest_path: Path,
    catalog_path: Path,
    date_eligibility_path: Path,
    output_path: Path,
    output_manifest_path: Path,
    selected_dates: Sequence[str],
    contract_version: str,
    split_label: str,
    batch_size: int = 65_536,
) -> dict[str, Any]:
    selected_dates = tuple(selected_dates)
    registered_dates = tuple(
        EXPECTED_SPLIT["train"]
        + EXPECTED_SPLIT["validation"]
        + EXPECTED_SPLIT["test"]
    )
    if (
        not selected_dates
        or len(set(selected_dates)) != len(selected_dates)
        or not set(selected_dates) <= set(registered_dates)
    ):
        raise ValueError("query materialization needs unique registered dates")
    if not contract_version or not split_label:
        raise ValueError("query materialization contract/split labels are required")
    if output_path.exists() or output_manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {split_label}-query outputs")
    teacher_manifest = json.loads(teacher_manifest_path.read_text())
    if teacher_manifest.get("contract_version") != "ptd-frozen-teacher-scores/v1":
        raise ValueError("frozen teacher score manifest contract mismatch")
    if teacher_manifest.get("teacher_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen teacher checkpoint mismatch")
    if teacher_manifest.get("label_columns_read") != []:
        raise ValueError("teacher materialization must not read labels")
    catalog = LockedCatalog(catalog_path, date_eligibility_path)
    paired, sources = _selected_records(
        teacher_manifest, selected_dates=selected_dates, batch_size=batch_size
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    current_key: tuple[str, str, str] | None = None
    current_click = ""
    current_purchase = ""
    candidates: list[tuple[int, dict[str, Any]]] = []
    finalized: set[tuple[str, str, str]] = set()
    date_users: set[tuple[str, str]] = set()
    query_count = 0
    candidate_rows = 0

    with tempfile.TemporaryDirectory() as directory:
        staged_output = Path(directory) / f"{split_label}_queries.jsonl"
        with staged_output.open("x") as handle:

            def flush() -> None:
                nonlocal query_count
                if current_key is None:
                    return
                snapshot_date, user_id, _snapshot_id = current_key
                products = [candidate["product_id"] for _, candidate in candidates]
                ranks = [rank for rank, _ in candidates]
                if len(set(products)) != len(products) or len(set(ranks)) != len(ranks):
                    raise ValueError(f"{split_label} query has duplicate products or ranks")
                if set(products) != catalog.eligible_products[snapshot_date]:
                    raise ValueError(f"{split_label} candidates differ from the locked date mask")
                for _, candidate in candidates:
                    if catalog.category_by_product[candidate["product_id"]] != candidate["category"]:
                        raise ValueError(f"{split_label} category differs from the locked catalog")
                value = {
                    "date": snapshot_date,
                    "user_id": user_id,
                    "click_history_most_recent_first": (
                        [] if not current_click else [int(value) for value in current_click.split(";")]
                    ),
                    "purchase_history_most_recent_first": (
                        []
                        if not current_purchase
                        else [int(value) for value in current_purchase.split(";")]
                    ),
                    "candidates": [candidate for _, candidate in sorted(candidates)],
                }
                handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
                query_count += 1

            for raw, teacher in paired:
                snapshot_date = _date(raw["snapshot_date"])
                if snapshot_date not in selected_dates:
                    raise ValueError(f"non-{split_label} row entered query materialization")
                if not raw["snapshot_id"] or not raw["user_id"]:
                    raise ValueError(f"{split_label} row has null/empty snapshot or user ID")
                key = (snapshot_date, str(raw["user_id"]), str(raw["snapshot_id"]))
                click = _history(raw["click_seq_product_id"], "click")
                purchase = _history(raw["purchase_seq_product_id"], "purchase")
                if key != current_key:
                    if current_key is not None:
                        flush()
                        finalized.add(current_key)
                    if key in finalized:
                        raise ValueError(f"raw rows are not contiguous by {split_label} query")
                    if (key[0], key[1]) in date_users:
                        raise ValueError(f"multiple {split_label} snapshots exist for one date-user")
                    date_users.add((key[0], key[1]))
                    current_key = key
                    current_click = click
                    current_purchase = purchase
                    candidates = []
                elif click != current_click or purchase != current_purchase:
                    raise ValueError(f"history fields differ within one {split_label} query")
                product = raw["product_id"]
                rank = raw["candidate_rank"]
                click_label = raw["label_click"]
                purchase_label = raw["label_purchase"]
                teacher_score = teacher["teacher_purchase"]
                if product is None or rank is None:
                    raise ValueError(f"{split_label} candidate key is null")
                if click_label not in (0, 1) or purchase_label not in (0, 1):
                    raise ValueError(f"{split_label} labels must be binary")
                if (
                    teacher_score is None
                    or not math.isfinite(float(teacher_score))
                    or not 0.0 <= float(teacher_score) <= 1.0
                ):
                    raise ValueError("teacher score must be finite and bounded")
                candidates.append(
                    (
                        int(rank),
                        {
                            "product_id": int(product),
                            "category": str(raw["first_category"] or ""),
                            "click_label": int(click_label),
                            "purchase_label": int(purchase_label),
                            "teacher_purchase": float(teacher_score),
                        },
                    )
                )
                candidate_rows += 1
            if current_key is not None:
                flush()
        if query_count == 0:
            raise ValueError(f"{split_label} query materialization produced no queries")
        output_hash = sha256(staged_output)
        manifest = {
            "contract_version": contract_version,
            "status": "complete",
            "dates": list(selected_dates),
            "queries": query_count,
            "candidate_rows": candidate_rows,
            "teacher_manifest_sha256": sha256(teacher_manifest_path),
            "teacher_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "catalog_sha256": sha256(catalog_path),
            "date_eligibility_sha256": sha256(date_eligibility_path),
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
                "raw_teacher_keys_match": True,
                "candidate_sets_match_locked_date_mask": True,
                "unique_date_user_units": True,
                "non_selected_outcome_columns_excluded": True,
                f"{split_label}_only": True,
                "no_overwrite": True,
            },
        }
        staged_manifest = Path(directory) / "manifest.json"
        staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(staged_output, output_path)
        os.replace(staged_manifest, output_manifest_path)
    return manifest


def build_validation_queries(
    *,
    teacher_manifest_path: Path,
    catalog_path: Path,
    date_eligibility_path: Path,
    output_path: Path,
    output_manifest_path: Path,
    batch_size: int = 65_536,
) -> dict[str, Any]:
    return build_split_queries(
        teacher_manifest_path=teacher_manifest_path,
        catalog_path=catalog_path,
        date_eligibility_path=date_eligibility_path,
        output_path=output_path,
        output_manifest_path=output_manifest_path,
        selected_dates=VALIDATION_DATES,
        contract_version="ptd-validation-queries/v1",
        split_label="validation",
        batch_size=batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--date-eligibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=65_536)
    args = parser.parse_args()
    result = build_validation_queries(
        teacher_manifest_path=args.teacher_manifest,
        catalog_path=args.catalog,
        date_eligibility_path=args.date_eligibility,
        output_path=args.output,
        output_manifest_path=args.manifest,
        batch_size=args.batch_size,
    )
    print(json.dumps({"status": result["status"], "queries": result["queries"]}))


if __name__ == "__main__":
    main()
