#!/usr/bin/env python3
"""Build immutable train-only queries for alternating assignment weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reference.evidence_gate import EXPECTED_SPLIT
from runner.build_validation_queries import build_split_queries


def build_assignment_queries(
    *,
    teacher_manifest_path: Path,
    catalog_path: Path,
    date_eligibility_path: Path,
    output_path: Path,
    output_manifest_path: Path,
    batch_size: int = 65_536,
):
    return build_split_queries(
        teacher_manifest_path=teacher_manifest_path,
        catalog_path=catalog_path,
        date_eligibility_path=date_eligibility_path,
        output_path=output_path,
        output_manifest_path=output_manifest_path,
        selected_dates=EXPECTED_SPLIT["train"],
        contract_version="ptd-assignment-queries/v1",
        split_label="train",
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
    result = build_assignment_queries(
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
