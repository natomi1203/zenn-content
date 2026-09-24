#!/usr/bin/env python3
"""Fail-closed checks for the paper artifact and claim status."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str):
    return json.loads((ROOT / path).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest() -> None:
    manifest = load("artifact/manifest.json")
    assert manifest["schema_version"] == "ptd-artifact-manifest/v1"
    assert manifest["paper_id"] == "ptd-sigir-2027"
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["git"]["base_revision"])
    assert {item["status"] for item in manifest["artifacts"]} <= {"VERIFIED", "PREREGISTERED", "PENDING", "GENERATED"}
    for item in manifest["artifacts"]:
        if item["status"] == "PENDING":
            assert item["sha256"] is None
            continue
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert item["sha256"] == sha256(path), item["id"]


def validate_verified_evidence() -> None:
    evidence = load("artifact/verified/legacy_esmm_evidence.json")
    assert evidence["status"] == "VERIFIED"
    assert re.fullmatch(r"[0-9a-f]{64}", evidence["manifest"]["sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", evidence["evaluation"]["sha256"])
    assert len(evidence["split"]["test"]) == 5
    assert evidence["summary"]["test_rows"] == 12_844_800
    assert evidence["summary"]["legacy_esmm"]["purchase_ndcg_at_50"] == 0.1762586881279632
    assert "no PTD outcome" in evidence["scope_note"]


def validate_ledger() -> None:
    with (ROOT / "artifact" / "claim_evidence_ledger.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and len({row["claim_id"] for row in rows}) == len(rows)
    assert {row["status"] for row in rows} <= {"VERIFIED", "PREREGISTERED", "PENDING"}
    pending = [row for row in rows if row["status"] == "PENDING"]
    assert len(pending) >= 4
    tex = (ROOT / "paper" / "main.tex").read_text().lower()
    forbidden = ["ptd significantly improves", "ptd outperforms", "our results show that ptd"]
    assert not any(phrase in tex for phrase in forbidden)
    assert "pending" in tex


def validate_generated() -> None:
    expected = [
        "related_work_table.tex",
        "legacy_evidence_table.tex",
        "claim_status_table.tex",
        "pending_results_table.tex",
        "ptd_architecture.tex",
        "evidence_flow.tex",
    ]
    for name in expected:
        path = ROOT / "paper" / "generated" / name
        assert path.is_file() and path.stat().st_size > 50, name


def validate_reference_smoke() -> None:
    smoke = load("artifact/smoke/reference_smoke_evaluation.json")
    assert smoke["contract_version"] == "ptd-reference-smoke/v1"
    assert smoke["status"] == "SMOKE_ONLY"
    assert smoke["empirical_claim_allowed"] is False
    assert abs(sum(smoke["item_sibling_distribution"].values()) - 1.0) < 1e-9
    assert abs(sum(smoke["node_sibling_distribution"].values()) - 1.0) < 1e-9
    assert smoke["objective"]["total"] >= smoke["objective"]["supervised"]


if __name__ == "__main__":
    validate_manifest()
    validate_verified_evidence()
    validate_ledger()
    validate_generated()
    validate_reference_smoke()
    tracked = [
        ROOT / "artifact" / "preregistration.md",
        ROOT / "artifact" / "claim_evidence_ledger.csv",
        ROOT / "artifact" / "verified" / "legacy_esmm_evidence.json",
    ]
    print("artifact validation: PASS")
    for path in tracked:
        print(f"{sha256(path)}  {path.relative_to(ROOT)}")
