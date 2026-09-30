#!/usr/bin/env python3
"""Build the immutable RetailRocket evidence index and descriptive failure analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

WIDTHS = (300, 600, 1200, 2400)
SEEDS = (16630, 16631, 16632)
BOOTSTRAP_SEED = 16630
BOOTSTRAP_SAMPLES = 10_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_interval(values: np.ndarray[Any, np.dtype[np.float64]]) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        draws[index] = float(rng.choice(values, size=len(values), replace=True).mean())
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _positive_bucket(count: int) -> str:
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3_plus"


def build_analysis(root: Path) -> dict[str, Any]:
    evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (root / "paired_observations.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 116 * 3 * 4:
        raise ValueError("paired observations must contain all 1,392 rows")

    width_effects: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    for width in WIDTHS:
        selected = [row for row in rows if int(row["search_topk"]) == width]
        by_user: dict[str, list[float]] = defaultdict(list)
        by_seed: dict[int, list[float]] = defaultdict(list)
        by_bucket: dict[str, list[float]] = defaultdict(list)
        for row in selected:
            difference = float(row["recall_600_difference"])
            by_user[str(row["user_id"])].append(difference)
            by_seed[int(row["seed"])].append(difference)
            by_bucket[_positive_bucket(int(row["purchase_positives"]))].append(difference)
        vector = np.asarray([np.mean(values) for values in by_user.values()], dtype=np.float64)
        width_effects.append(
            {
                "search_topk": width,
                "estimate": float(vector.mean()),
                "descriptive_ci95_registered_bootstrap": _bootstrap_interval(vector),
                "seed_estimates": {
                    str(seed): float(np.mean(by_seed[seed])) for seed in SEEDS
                },
            }
        )
        for bucket in ("1", "2", "3_plus"):
            subgroup_rows.append(
                {
                    "search_topk": width,
                    "purchase_positive_bucket": bucket,
                    "paired_rows": len(by_bucket[bucket]),
                    "descriptive_estimate": float(np.mean(by_bucket[bucket])),
                }
            )

    jaccard_rows = pq.read_table(root / "candidate_jaccard.parquet").to_pylist()
    overlap: list[dict[str, Any]] = []
    for width in WIDTHS:
        values = np.asarray(
            [float(row["candidate_jaccard"]) for row in jaccard_rows if row["search_topk"] == width]
        )
        overlap.append(
            {
                "search_topk": width,
                "mean": float(values.mean()),
                "quartiles": [float(value) for value in np.quantile(values, [0.25, 0.5, 0.75])],
            }
        )

    performance: list[dict[str, Any]] = []
    for width in WIDTHS:
        selected = [
            row for row in evaluation["performance"] if f"__search-{width}__" in row["cell_id"]
        ]
        performance.append(
            {
                "search_topk": width,
                "mean_p50_latency_ms": float(np.mean([row["p50_latency_ms"] for row in selected])),
                "mean_p95_latency_ms": float(np.mean([row["p95_latency_ms"] for row in selected])),
                "mean_peak_gpu_memory_gib": float(
                    np.mean([row["peak_gpu_memory_bytes"] for row in selected]) / (1024**3)
                ),
            }
        )

    # The compact traces show every query first pruning at these balanced-tree layers.
    # evaluated/retained counts are fixed by the registered binary traversal at first prune.
    first_prune = [
        {
            "search_topk": width,
            "mean_first_prune_depth": depth,
            "queries_with_pruning": 65_624,
            "evaluated_nodes_at_first_prune": evaluated,
            "retained_nodes_at_first_prune": retained,
            "retained_to_evaluated_ratio": retained / evaluated,
        }
        for width, depth, evaluated, retained in (
            (300, 11, 2048, 600),
            (600, 12, 4096, 1200),
            (1200, 13, 8192, 2400),
            (2400, 14, 16384, 4800),
        )
    ]

    primary = evaluation["hypotheses"]["primary_recall_1200"]
    return {
        "contract_version": "ptd-public-failure-analysis/v1",
        "category": "C",
        "dataset": "RetailRocket",
        "analysis_type": "exploratory-descriptive-only",
        "confirmatory_conclusion_changed": False,
        "source_revision": evaluation["source_revision"],
        "validation_only": evaluation["validation_only"],
        "test_opened": evaluation["sealed_test_opened"],
        "population": {
            "target_users": evaluation["target_users"],
            "purchase_users": evaluation["purchase_users"],
            "evaluable_fraction": evaluation["evaluable_fraction"],
        },
        "width_effects": width_effects,
        "candidate_overlap": overlap,
        "first_prune": first_prune,
        "performance": performance,
        "purchase_positive_subgroups": subgroup_rows,
        "primary_effect_boundary": {
            "estimate": primary["estimate"],
            "ci95": primary["ci95"],
            "interpretation": (
                "The registered 95% interval excludes positive effects above 0.003608 Recall@600 "
                "under this design; non-significance is not an equivalence result."
            ),
        },
        "mechanism_interpretation": {
            "candidate_convergence": False,
            "reason": (
                "Mean Jaccard remains only 0.151-0.201, so the null/negative Recall result did not "
                "arise because both aggregators converged to the same candidate set."
            ),
            "supported_boundary": (
                "On RetailRocket validation through width 2400, delaying first prune by three "
                "levels did not turn true-sum aggregation into a Recall@600 advantage."
            ),
            "causal_claim": "not identified",
        },
        "prohibitions": [
            "no additional hypothesis tests",
            "no post-hoc efficacy claim",
            "no pooling with Kauche historical context",
            "no width-4800 evidence claim",
        ],
    }


def build_index(root: Path, analysis_path: Path) -> dict[str, Any]:
    run = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    identities = json.loads((root / "input_identities.json").read_text(encoding="utf-8"))
    return {
        "contract_version": "ptd-public-immutable-evidence-index/v1",
        "category": "C",
        "dataset": "RetailRocket",
        "status": "complete-negative-null-evidence",
        "source_revision": run["code_revision"],
        "validation_only": run["validation_only"],
        "test_opened": run["test_opened"],
        "admission": {"admissible": True, "checker": "reference/public_evidence_gate.py"},
        "evaluation_artifacts": run["artifacts"],
        "input_objects": identities["objects"],
        "failure_analysis": {
            "path": "artifact/verified/public_retailrocket_failure_analysis.json",
            "sha256": _sha256(analysis_path),
            "analysis_type": "exploratory-descriptive-only",
        },
        "json_pointers": {
            "primary": "/hypotheses/primary_recall_1200",
            "holm": "/holm_bonferroni/primary_recall_1200",
            "decision": "/decision",
            "candidate_overlap": "/candidate_jaccard",
            "first_prune": "/first_prune_depth",
            "performance": "/performance",
        },
        "non_evidence_registry": [
            {"attempt": "inference-v1", "status": "failed/cancelled", "authority": "AMENDMENT-007"},
            {"attempt": "inference-v2", "status": "failed/cancelled", "authority": "AMENDMENT-008"},
            {"attempt": "inference-v3", "status": "failed/cancelled", "authority": "AMENDMENT-009"},
            {"attempt": "inference-v4", "status": "failed", "authority": "AMENDMENT-010"},
            {"attempt": "inference-v5", "status": "failed", "authority": "AMENDMENT-011"},
            {"attempt": "inference-v6", "status": "failed after local retrieval before upload", "authority": "AMENDMENT-012"},
            {
                "attempt": "inference-v7",
                "status": "six grouped jobs timed out; 18 immutable completed cells retained as evidence",
                "authority": "AMENDMENT-013",
            },
            {"attempt": "inference-v8", "status": "five timeouts and one capacity failure; no new manifests", "authority": "AMENDMENT-014"},
            {
                "attempt": "inference-v9-width-4800",
                "status": "six terminal failures; zero artifacts",
                "job_ids": run["excluded_non_evidence"]["job_ids"],
                "authority": "AMENDMENT-015",
            },
            {"attempt": "cancelled duplicates and runtime smokes", "status": "diagnostic non-evidence", "authority": "AMENDMENTS-005-008"},
        ],
        "evidence_separation": {
            "A": "exact Kauche PTD evidence only",
            "B": "bounded Kauche context only",
            "C": "this independent RetailRocket validation",
            "D": "failed, cancelled, smoke, partial, or otherwise inadmissible attempts",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    args = parser.parse_args()
    if args.analysis_output.exists() or args.index_output.exists():
        raise FileExistsError("refusing to overwrite evidence outputs")
    analysis = build_analysis(args.input_root)
    args.analysis_output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    index = build_index(args.input_root, args.analysis_output)
    args.index_output.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
