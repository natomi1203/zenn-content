#!/usr/bin/env python3
"""Run a synthetic row-level smoke check against the exact frozen ESMM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner.materialize_teacher_scores import EXPECTED_CHECKPOINT_SHA256, materialize, sha256
EXPECTED_PURCHASE = (0.33971071243286133, 0.00027223789948038757)
ABSOLUTE_TOLERANCE = 1e-6


def synthetic_rows() -> dict:
    return {
        "snapshot_id": ["synthetic-request", "synthetic-request"],
        "snapshot_date": [date(2026, 7, 18)] * 2,
        "user_id": ["synthetic-user", "synthetic-user"],
        "product_id": [101, 102],
        "candidate_rank": [1, 2],
        "conversion_score": [0.9, 0.2],
        "cosine_similarity": [0.8, 0.1],
        "old_score": [0.7, 0.3],
        "repeat_click_match": [1, 0],
        "repeat_purchase_match": [0, 0],
        "is_new_user": [False, False],
        "is_new_product": [False, True],
    }


def run(checkpoint: Path) -> dict:
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen teacher checkpoint hash mismatch")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "input.parquet"
        output = root / "scores.parquet"
        manifest = root / "manifest.json"
        pq.write_table(pa.table(synthetic_rows()), source)
        result = materialize(
            [source], checkpoint, output, manifest, device="cpu", batch_size=1
        )
        scored = pq.read_table(output)
        purchase = tuple(float(value) for value in scored["teacher_purchase"].to_pylist())
        if len(purchase) != len(EXPECTED_PURCHASE) or any(
            abs(observed - expected) > ABSOLUTE_TOLERANCE
            for observed, expected in zip(purchase, EXPECTED_PURCHASE, strict=True)
        ):
            raise ValueError(f"frozen teacher smoke values changed: {purchase}")
        no_overwrite = False
        try:
            materialize([source], checkpoint, output, manifest, device="cpu", batch_size=1)
        except FileExistsError:
            no_overwrite = True
        if not no_overwrite:
            raise ValueError("teacher materializer did not refuse overwrite")
    import torch

    return {
        "contract_version": "ptd-frozen-teacher-smoke/v1",
        "status": "SMOKE_ONLY",
        "empirical_claim_allowed": False,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "synthetic_input_sha256": hashlib.sha256(
            json.dumps(synthetic_rows(), default=str, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": result["rows"],
        "teacher_purchase": list(purchase),
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "checks": {
            "factorization_matches": all(
                abs(float(a) * float(b) - float(c)) <= 1e-7
                for a, b, c in zip(
                    scored["teacher_pctr"].to_pylist(),
                    scored["teacher_pcvr"].to_pylist(),
                    scored["teacher_purchase"].to_pylist(),
                    strict=True,
                )
            ),
            "labels_read": result["checks"]["labels_read"],
            "no_overwrite": no_overwrite,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.system(),
        },
        "scope_note": "Synthetic compatibility check only; it is not PTD efficacy evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifact" / "smoke" / "frozen_teacher_score_smoke.json",
    )
    args = parser.parse_args()
    payload = run(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"frozen teacher score smoke: PASS ({payload['rows']} synthetic rows)")


if __name__ == "__main__":
    main()
