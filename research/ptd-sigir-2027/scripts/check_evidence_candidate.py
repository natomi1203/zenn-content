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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    args = parser.parse_args()
    report = admission_report(args.run_manifest, args.evaluation)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["admissible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
