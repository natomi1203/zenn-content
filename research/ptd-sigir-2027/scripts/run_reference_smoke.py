#!/usr/bin/env python3
"""Generate a deterministic, non-empirical PTD reference smoke record."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference.ptd import (
    balanced_paths,
    capacity_balanced_assignment,
    item_sibling_distribution,
    node_sibling_distribution,
    ptd_objective,
)

OUTPUT = ROOT / "artifact" / "smoke" / "reference_smoke_evaluation.json"


def main() -> None:
    purchase = {"i1": 0.7, "i2": 0.2, "i3": 0.08, "i4": 0.02}
    item_teacher = item_sibling_distribution(purchase, ["i1", "i2"], temperature=2.0)
    node_teacher = node_sibling_distribution(
        purchase,
        {"n0": ["i1", "i2"], "n1": ["i3", "i4"]},
        purchase,
        temperature=2.0,
    )
    objective = ptd_objective(
        0.4,
        item_teacher=item_teacher,
        item_student_logits={"i1": 0.4, "i2": -0.1},
        node_teacher=node_teacher,
        node_student_logits={"n0": 0.5, "n1": -0.2},
        temperature=2.0,
        lambda_item=0.3,
        lambda_node=0.3,
    )
    assignments = capacity_balanced_assignment(
        {
            "i1": {"n0": 3.0, "n1": 0.0},
            "i2": {"n0": 2.0, "n1": 0.2},
            "i3": {"n0": 0.1, "n1": 2.0},
            "i4": {"n0": 0.0, "n1": 3.0},
        },
        ["n0", "n1"],
        capacity=2,
    )
    payload = {
        "contract_version": "ptd-reference-smoke/v1",
        "status": "SMOKE_ONLY",
        "empirical_claim_allowed": False,
        "teacher": {"factorization": "pCTR*pCVR", "purchase_probabilities": purchase},
        "item_sibling_distribution": item_teacher,
        "node_sibling_distribution": node_teacher,
        "objective": objective,
        "balanced_paths": {item: list(path) for item, path in balanced_paths(purchase, 2).items()},
        "assignments": [assignment.__dict__ for assignment in assignments],
        "note": "Synthetic arithmetic check only; it cannot populate a manuscript result table.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"reference smoke: PASS ({OUTPUT.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
