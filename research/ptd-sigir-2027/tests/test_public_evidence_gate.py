from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from reference.public_evidence_gate import admission_report


REVISION = "1" * 40
HYPOTHESES = (
    "primary_recall_1200",
    "hit_rate_1_1200",
    "beam_interaction_300_to_2400",
    "absolute_gap_shrinkage_300_to_2400",
)


class PublicEvidenceGateTest(unittest.TestCase):
    def write_candidate(self, root: Path, *, tamper_primary: bool = False) -> tuple[Path, Path, Path]:
        paired_path = root / "paired.jsonl"
        with paired_path.open("w", encoding="utf-8") as stream:
            for user in range(116):
                for seed in (16630, 16631, 16632):
                    for width in (300, 600, 1200, 2400):
                        stream.write(
                            json.dumps(
                                {
                                    "user_id": f"u{user}",
                                    "seed": seed,
                                    "search_topk": width,
                                    "purchase_recall_600": 0.0,
                                    "purchase_recall_600_max": 0.0,
                                    "recall_600_difference": 0.0,
                                    "purchase_hit_rate_1": 0.0,
                                    "purchase_hit_rate_1_max": 0.0,
                                    "hit_rate_1_difference": 0.0,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
        contrast = {
            "estimate": 0.0,
            "ci95": [-0.1, 0.1],
            "p_value_two_sided": 1.0,
            "samples": 10_000,
            "seed": 16630,
        }
        evaluation = {
            "contract_version": "ptd_public_retailrocket_confirmatory_evaluation_v1",
            "amendment": 15,
            "source_revision": REVISION,
            "cells": 24,
            "search_topks": [300, 600, 1200, 2400],
            "target_users": 65_624,
            "purchase_users": 116,
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 16630,
            "hypotheses": {name: dict(contrast) for name in HYPOTHESES},
            "holm_bonferroni": {
                name: {"raw_p_value": 1.0, "adjusted_p_value": 1.0, "rejected": False}
                for name in HYPOTHESES
            },
            "decision": {"integrity_failure": False, "superiority": False, "test_opened": False},
            "sealed_test_opened": False,
            "validation_only": True,
        }
        if tamper_primary:
            evaluation["hypotheses"]["primary_recall_1200"]["estimate"] = 0.5
        evaluation_path = root / "evaluation.json"
        evaluation_path.write_text(json.dumps(evaluation, sort_keys=True), encoding="utf-8")
        artifact = {
            "uri": "gs://bucket/object",
            "generation": 1,
            "metageneration": 1,
            "sha256": "0" * 64,
        }
        run = {
            "schema_version": "ptd-public-run-manifest/v1",
            "status": "complete",
            "category": "C",
            "dataset": "RetailRocket",
            "amendment": 15,
            "code_revision": REVISION,
            "arms": ["internal_max_descendant", "internal_true_sum_mass"],
            "seeds": [16630, 16631, 16632],
            "search_topks": [300, 600, 1200, 2400],
            "return_k": 600,
            "cell_count": 24,
            "validation_only": True,
            "immutable": True,
            "test_opened": False,
            "sealed_test_opened": False,
            "artifacts": {name: dict(artifact) for name in (
                "audit",
                "input_identities",
                "evaluation",
                "paired_observations",
                "per_user_outcomes",
                "candidate_jaccard",
            )},
            "excluded_non_evidence": {"width": 4800, "job_ids": [str(index) for index in range(6)]},
        }
        run["artifacts"]["evaluation"]["sha256"] = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
        run["artifacts"]["paired_observations"]["sha256"] = hashlib.sha256(paired_path.read_bytes()).hexdigest()
        run_path = root / "run.json"
        run_path.write_text(json.dumps(run, sort_keys=True), encoding="utf-8")
        return run_path, evaluation_path, paired_path

    def test_complete_negative_public_evidence_is_admissible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = admission_report(*self.write_candidate(Path(directory)))
        self.assertTrue(report["admissible"], report["errors"])

    def test_tampered_public_primary_estimate_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = admission_report(*self.write_candidate(Path(directory), tamper_primary=True))
        self.assertFalse(report["admissible"])
        self.assertTrue(any("primary_recall_1200" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
