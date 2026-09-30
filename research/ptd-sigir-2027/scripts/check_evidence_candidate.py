#!/usr/bin/env python3
"""Check whether a run manifest and evaluation may enter the PTD claim ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference.evidence_gate import admission_report
from reference.public_evidence_gate import admission_report as public_admission_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--paired-observations", required=True, type=Path)
    args = parser.parse_args()
    try:
        run = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        run = {}
    checker = public_admission_report if run.get("schema_version") == "ptd-public-run-manifest/v1" else admission_report
    report = checker(args.run_manifest, args.evaluation, args.paired_observations)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["admissible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
