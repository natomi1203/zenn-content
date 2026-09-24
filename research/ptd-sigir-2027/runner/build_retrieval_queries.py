#!/usr/bin/env python3
"""Join raw candidate rows to frozen teacher scores and emit grouped test queries."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow.parquet as pq

from reference.evidence_gate import EXPECTED_SPLIT
from runner.materialize_teacher_scores import EXPECTED_CHECKPOINT_SHA256, sha256

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
TEACHER_COLUMNS = (
    "snapshot_id",
    "snapshot_date",
    "user_id",
    "product_id",
    "candidate_rank",
    "teacher_purchase",
)
KEY_COLUMNS = TEACHER_COLUMNS[:-1]


def _records(
    paths: Sequence[Path], columns: Sequence[str], *, batch_size: int
) -> Iterator[dict[str, Any]]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        if missing := set(columns) - set(parquet.schema_arrow.names):
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for batch in parquet.iter_batches(columns=list(columns), batch_size=batch_size):
            values = batch.to_pydict()
            for index in range(batch.num_rows):
                yield {name: values[name][index] for name in columns}


def _history(value: Any, label: str) -> list[int]:
    if value is None or value == "":
        return []
    if not isinstance(value, str):
        raise ValueError(f"{label} history must be a semicolon-delimited string")
    tokens = [token.strip() for token in value.split(";") if token.strip()]
    if len(tokens) > 30:
        raise ValueError(f"{label} history exceeds 30 items")
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"{label} history contains a non-integer product ID") from exc


def _date(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _flush_query(
    handle,
    key: tuple[str, str, str],
    click_history: list[int],
    purchase_history: list[int],
    candidates: list[tuple[int, dict[str, Any]]],
) -> None:
    if not candidates:
        raise ValueError("retrieval query has no candidates")
    ranks = [rank for rank, _ in candidates]
    products = [candidate["product_id"] for _, candidate in candidates]
    if len(set(ranks)) != len(ranks) or len(set(products)) != len(products):
        raise ValueError("retrieval query has duplicate candidate ranks or products")
    date, user_id, _snapshot_id = key
    value = {
        "date": date,
        "user_id": user_id,
        "click_history_most_recent_first": click_history,
        "purchase_history_most_recent_first": purchase_history,
        "candidates": [candidate for _, candidate in sorted(candidates)],
    }
    handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def build_retrieval_queries(
    *,
    teacher_manifest_path: Path,
    output_path: Path,
    output_manifest_path: Path,
    batch_size: int = 65_536,
) -> dict[str, Any]:
    """Build immutable five-day query JSONL from row-aligned raw and teacher Parquet."""
    if output_path.exists() or output_manifest_path.exists():
        raise FileExistsError("refusing to overwrite retrieval query outputs")
    teacher_manifest = json.loads(teacher_manifest_path.read_text())
    if teacher_manifest.get("contract_version") != "ptd-frozen-teacher-scores/v1":
        raise ValueError("frozen teacher score manifest contract mismatch")
    if teacher_manifest.get("teacher_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen teacher checkpoint mismatch")
    if teacher_manifest.get("label_columns_read") != []:
        raise ValueError("teacher materialization must not read labels")
    source_files = teacher_manifest.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("teacher manifest has no raw source files")
    raw_paths: list[Path] = []
    for source in source_files:
        path = Path(source["path"])
        if not path.is_file() or sha256(path) != source["sha256"]:
            raise ValueError("raw source path/hash mismatch")
        raw_paths.append(path)
    teacher_output = teacher_manifest.get("output", {})
    teacher_path = Path(teacher_output.get("path", ""))
    if not teacher_path.is_file() or sha256(teacher_path) != teacher_output.get(
        "sha256"
    ):
        raise ValueError("teacher score output path/hash mismatch")

    raw_rows = _records(raw_paths, RAW_COLUMNS, batch_size=batch_size)
    teacher_rows = _records([teacher_path], TEACHER_COLUMNS, batch_size=batch_size)
    sentinel = object()
    current_key: tuple[str, str, str] | None = None
    current_click: list[int] = []
    current_purchase: list[int] = []
    current_candidates: list[tuple[int, dict[str, Any]]] = []
    finalized_keys: set[tuple[str, str, str]] = set()
    paired_rows = 0
    test_rows = 0
    query_count = 0
    queries_by_date: Counter[str] = Counter()
    date_user_keys: set[tuple[str, str]] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        staged = Path(directory) / "queries.jsonl"
        with staged.open("x") as handle:
            for raw, teacher in itertools.zip_longest(
                raw_rows, teacher_rows, fillvalue=sentinel
            ):
                if raw is sentinel or teacher is sentinel:
                    raise ValueError("raw and teacher row counts differ")
                paired_rows += 1
                for name in KEY_COLUMNS:
                    raw_value = (
                        _date(raw[name]) if name == "snapshot_date" else raw[name]
                    )
                    teacher_value = (
                        _date(teacher[name])
                        if name == "snapshot_date"
                        else teacher[name]
                    )
                    if raw_value != teacher_value:
                        raise ValueError(f"raw/teacher key mismatch in {name}")
                snapshot_date = _date(raw["snapshot_date"])
                if snapshot_date not in EXPECTED_SPLIT["test"]:
                    continue
                test_rows += 1
                if not raw["snapshot_id"] or not raw["user_id"]:
                    raise ValueError("test row has null/empty snapshot or user ID")
                key = (snapshot_date, str(raw["user_id"]), str(raw["snapshot_id"]))
                click = _history(raw["click_seq_product_id"], "click")
                purchase = _history(raw["purchase_seq_product_id"], "purchase")
                if current_key != key:
                    if current_key is not None:
                        _flush_query(
                            handle,
                            current_key,
                            current_click,
                            current_purchase,
                            current_candidates,
                        )
                        finalized_keys.add(current_key)
                        query_count += 1
                        queries_by_date[current_key[0]] += 1
                    if key in finalized_keys:
                        raise ValueError(
                            "raw rows are not contiguous by retrieval query"
                        )
                    if (key[0], key[1]) in date_user_keys:
                        raise ValueError(
                            "multiple snapshots exist for one date-user evaluation unit"
                        )
                    date_user_keys.add((key[0], key[1]))
                    current_key = key
                    current_click = click
                    current_purchase = purchase
                    current_candidates = []
                elif click != current_click or purchase != current_purchase:
                    raise ValueError("history fields differ within one retrieval query")
                product = raw["product_id"]
                rank = raw["candidate_rank"]
                click_label = raw["label_click"]
                purchase_label = raw["label_purchase"]
                teacher_score = teacher["teacher_purchase"]
                if product is None or rank is None:
                    raise ValueError("test candidate key is null")
                if click_label not in (0, 1) or purchase_label not in (0, 1):
                    raise ValueError("test labels must be binary")
                if (
                    teacher_score is None
                    or not math.isfinite(float(teacher_score))
                    or not 0.0 <= float(teacher_score) <= 1.0
                ):
                    raise ValueError("teacher score must be finite and bounded")
                current_candidates.append(
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
            if current_key is not None:
                _flush_query(
                    handle,
                    current_key,
                    current_click,
                    current_purchase,
                    current_candidates,
                )
                query_count += 1
                queries_by_date[current_key[0]] += 1
        if set(queries_by_date) != set(EXPECTED_SPLIT["test"]):
            raise ValueError("grouped queries do not cover all five test dates")
        output_hash = sha256(staged)
        manifest = {
            "contract_version": "ptd-retrieval-queries/v1",
            "status": "complete",
            "teacher_manifest_sha256": sha256(teacher_manifest_path),
            "teacher_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "raw_source_files": len(raw_paths),
            "paired_rows": paired_rows,
            "test_candidate_rows": test_rows,
            "queries": query_count,
            "queries_by_date": dict(sorted(queries_by_date.items())),
            "output": {"path": str(output_path), "sha256": output_hash},
            "columns_read": list(RAW_COLUMNS) + ["teacher_purchase"],
            "checks": {
                "raw_teacher_keys_match": True,
                "teacher_labels_not_read": True,
                "query_rows_contiguous": True,
                "unique_date_user_units": True,
                "all_test_dates_present": True,
                "labels_read_only_for_evaluation": True,
                "no_overwrite": True,
            },
        }
        staged_manifest = Path(directory) / "manifest.json"
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staged, output_path)
        os.replace(staged_manifest, output_manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=65_536)
    args = parser.parse_args()
    manifest = build_retrieval_queries(
        teacher_manifest_path=args.teacher_manifest,
        output_path=args.output,
        output_manifest_path=args.manifest,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "queries": manifest["queries"],
                "output": manifest["output"],
            }
        )
    )


if __name__ == "__main__":
    main()
