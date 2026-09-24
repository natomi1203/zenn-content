#!/usr/bin/env python3
"""Run the paired-evidence emitter end to end on deterministic synthetic rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference.evidence_gate import (  # noqa: E402
    ASSERTIONS,
    CATALOG_ORDER_SHA256,
    DATE_ELIGIBILITY_SHA256,
    EXPECTED_SEEDS,
    EXPECTED_SPLIT,
    EXPECTED_VARIANTS,
    ITEM_UNIVERSE_AUDIT_SHA256,
    METHOD_CONTRACT_SHA256,
    RAW_INPUT_INVENTORY_SHA256,
    RAW_SEQUENCE_CONTRACT_SHA256,
    SOURCE_CONTRACT_SHA256,
    SOURCE_MANIFEST_SHA256,
    TEACHER_CHECKPOINT_SHA256,
)
from runner.emit_evaluation import emit_evaluation, load_retrieval_metric_rows  # noqa: E402

USER_COUNT = 67
GENERATED_AT = "2026-09-25T12:00:00+09:00"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_run_manifest() -> dict:
    artifact = {"uri": "synthetic://artifact", "sha256": "b" * 64}
    return {
        "schema_version": "ptd-run-manifest/v1",
        "status": "complete",
        "run_id": "synthetic-emitter-smoke",
        "created_at": "2026-09-25T09:00:00+09:00",
        "test_scoring_started_at": "2026-09-25T11:00:00+09:00",
        "code_revision": "a" * 40,
        "source_contract": {
            "contract_version": "shared_bottom_esmm_v2_source",
            "manifest_uri": "synthetic://source/manifest.json",
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_contract_sha256": SOURCE_CONTRACT_SHA256,
            "raw_input_inventory_uri": "artifact/verified/raw_input_inventory.json",
            "raw_input_inventory_sha256": RAW_INPUT_INVENTORY_SHA256,
            "raw_sequence_contract_uri": "artifact/verified/raw_sequence_contract.json",
            "raw_sequence_contract_sha256": RAW_SEQUENCE_CONTRACT_SHA256,
            "item_universe_audit_uri": "artifact/verified/item_universe_audit.json",
            "item_universe_audit_sha256": ITEM_UNIVERSE_AUDIT_SHA256,
        },
        "method_contract": {
            "uri": "artifact/preregistered_method.json",
            "sha256": METHOD_CONTRACT_SHA256,
        },
        "teacher": {
            "teacher_id": "legacy_loss_shared_bottom_esmm_v2",
            "frozen": True,
            "score": "pCTR*pCVR",
            "artifact": {
                "uri": "synthetic://frozen-teacher.pt",
                "sha256": TEACHER_CHECKPOINT_SHA256,
            },
        },
        "split": EXPECTED_SPLIT,
        "seeds": EXPECTED_SEEDS,
        "assertions": {name: True for name in ASSERTIONS},
        "variants": {
            variant: {
                "status": "complete",
                "seed_artifacts": {
                    str(seed): {
                        "uri": f"synthetic://{variant}/{seed}",
                        "sha256": "b" * 64,
                    }
                    for seed in EXPECTED_SEEDS
                },
            }
            for variant in EXPECTED_VARIANTS
        },
        "selected_hyperparameters": {
            "selection_seed": 16630,
            "temperature": 2.0,
            "lambda_item": 0.3,
            "lambda_node": 0.3,
            "epsilon_item": 1e-6,
            "epsilon_node": 1e-12,
            "assignment_weight": "y_plus_teacher_times_path_log_probability",
            "selection_artifact": artifact,
        },
        "validation_selection": {
            "selection_seed": 16630,
            "selection_metric": "purchase_ndcg_at_50",
            "best_single_variant": "ptd_node",
            "artifact": artifact,
        },
        "tree": {
            "branching_factor": 2,
            "depth": 13,
            "leaf_capacity": 1,
            "beam_width": 600,
            "catalog_order_sha256": CATALOG_ORDER_SHA256,
            "item_universe_audit_sha256": ITEM_UNIVERSE_AUDIT_SHA256,
            "date_eligibility_sha256": DATE_ELIGIBILITY_SHA256,
            "alternating_cycles_selected_by_seed": {
                str(seed): 1 for seed in EXPECTED_SEEDS
            },
            "locked_before_test": True,
            "locked_at": "2026-09-25T10:00:00+09:00",
            "locked_bundle_sha256": "c" * 64,
        },
        "latency_protocol": {
            "baseline_variant": "fixed_tdm",
            "p95_relative_ceiling": 1.20,
            "warmup_queries": 100,
            "measured_queries": 1000,
            "concurrency": 1,
            "hardware": "synthetic CPU smoke",
            "software": "synthetic emitter runtime",
            "timer": "synthetic monotonic durations",
        },
        "deviations": [],
    }


def write_retrieval_rows(path: Path) -> int:
    offsets = {
        "fixed_tdm": 0.000,
        "ptd_item": 0.005,
        "ptd_node": 0.007,
        "ptd_combined": 0.010,
        "alternating_tdm": 0.003,
        "alternating_ptd": 0.012,
        "ptd_combined_baseline_encoder": 0.006,
        "teacher_oracle": 0.020,
    }
    count = 0
    with path.open("x") as handle:
        for date_index, date in enumerate(EXPECTED_SPLIT["test"]):
            for user_index in range(USER_COUNT):
                user = f"synthetic-user-{user_index:03d}"
                for seed_index, seed in enumerate(EXPECTED_SEEDS):
                    baseline = (
                        0.20
                        + date_index * 0.001
                        + (user_index % 5) * 0.0001
                        + seed_index * 0.0002
                    )
                    for variant_index, variant in enumerate(EXPECTED_VARIANTS):
                        offset = offsets[variant]
                        value = {
                            "date": date,
                            "user_id": user,
                            "seed": seed,
                            "variant": variant,
                            "purchase_ndcg_at_50": baseline + offset,
                            "purchase_recall_at_50": 0.40 + offset,
                            "purchase_ndcg_at_10": 0.15 + offset,
                            "purchase_ndcg_at_100": 0.25 + offset,
                            "purchase_auc": 0.60 + offset,
                            "click_ndcg_at_50": 0.30,
                            "category_coverage_at_50": 0.50,
                            "max_category_share_at_50": 0.20,
                            "latency_ms": 10.0
                            + ((date_index * USER_COUNT + user_index + seed_index) % 10)
                            * 0.01
                            + variant_index * 0.001,
                            "candidates_scored": 100.0 + variant_index,
                        }
                        handle.write(
                            json.dumps(value, sort_keys=True, separators=(",", ":"))
                            + "\n"
                        )
                        count += 1
    return count


def run() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run_manifest_path = root / "run_manifest.json"
        run_manifest_path.write_text(
            json.dumps(synthetic_run_manifest(), indent=2, sort_keys=True) + "\n"
        )
        metrics_path = root / "retrieval_metrics.jsonl"
        metric_rows = write_retrieval_rows(metrics_path)
        first_paired = root / "first" / "paired.jsonl"
        first_evaluation = root / "first" / "evaluation.json"
        first = emit_evaluation(
            run_manifest_path=run_manifest_path,
            retrieval_metrics_path=metrics_path,
            paired_output_path=first_paired,
            evaluation_output_path=first_evaluation,
            run_manifest_uri="synthetic://run_manifest.json",
            paired_observations_uri="synthetic://paired_observations.jsonl",
            generated_at=GENERATED_AT,
        )
        second_paired = root / "second" / "paired.jsonl"
        second_evaluation = root / "second" / "evaluation.json"
        second = emit_evaluation(
            run_manifest_path=run_manifest_path,
            retrieval_metrics_path=metrics_path,
            paired_output_path=second_paired,
            evaluation_output_path=second_evaluation,
            run_manifest_uri="synthetic://run_manifest.json",
            paired_observations_uri="synthetic://paired_observations.jsonl",
            generated_at=GENERATED_AT,
        )
        no_overwrite = False
        try:
            emit_evaluation(
                run_manifest_path=run_manifest_path,
                retrieval_metrics_path=metrics_path,
                paired_output_path=first_paired,
                evaluation_output_path=first_evaluation,
                run_manifest_uri="synthetic://run_manifest.json",
                paired_observations_uri="synthetic://paired_observations.jsonl",
                generated_at=GENERATED_AT,
            )
        except FileExistsError:
            no_overwrite = True
        duplicate_path = root / "duplicate.jsonl"
        first_line = metrics_path.read_text().splitlines()[0]
        duplicate_path.write_text(metrics_path.read_text() + first_line + "\n")
        duplicate_rejected = False
        try:
            load_retrieval_metric_rows(duplicate_path)
        except ValueError:
            duplicate_rejected = True

        evaluation = first["evaluation"]
        contrast_deltas = {
            name: value["mean_delta"]
            for name, value in evaluation["primary_contrasts"].items()
        }
        checks = {
            "self_admission_passed": first["admission"]["admissible"],
            "all_eight_variants": set(evaluation["variants"]) == set(EXPECTED_VARIANTS),
            "five_dates_three_seeds": evaluation["split"] == EXPECTED_SPLIT
            and evaluation["seeds"] == EXPECTED_SEEDS,
            "minimum_latency_queries_per_variant": metric_rows // len(EXPECTED_VARIANTS)
            >= 1000,
            "deterministic_paired_hash": first["paired_observations_sha256"]
            == second["paired_observations_sha256"],
            "deterministic_evaluation_hash": first["evaluation_sha256"]
            == second["evaluation_sha256"],
            "no_overwrite": no_overwrite,
            "duplicate_row_rejected": duplicate_rejected,
            "no_deviations": evaluation["deviations"] == [],
        }
        if not all(checks.values()):
            raise ValueError(f"evaluation emitter smoke failed: {checks}")
        payload = {
            "contract_version": "ptd-evaluation-emitter-smoke/v1",
            "status": "SMOKE_ONLY",
            "empirical_claim_allowed": False,
            "synthetic_counts": {
                "users": USER_COUNT,
                "dates": len(EXPECTED_SPLIT["test"]),
                "seeds": len(EXPECTED_SEEDS),
                "variants": len(EXPECTED_VARIANTS),
                "retrieval_metric_rows": metric_rows,
                "paired_observation_rows": first["paired_observation_rows"],
                "latency_rows_per_variant": metric_rows // len(EXPECTED_VARIANTS),
            },
            "input_sha256": {
                "run_manifest": sha256(run_manifest_path),
                "retrieval_metrics": sha256(metrics_path),
            },
            "output_sha256": {
                "paired_observations": first["paired_observations_sha256"],
                "evaluation": first["evaluation_sha256"],
            },
            "synthetic_primary_mean_deltas": contrast_deltas,
            "checks": checks,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.system(),
            },
            "scope_note": (
                "Synthetic emitter evidence only. Positive synthetic deltas are fixtures, not PTD "
                "results, and cannot populate the manuscript."
            ),
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifact" / "smoke" / "evaluation_emitter_smoke.json",
    )
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "evaluation emitter smoke: PASS (paired JSONL, bootstrap, Holm, guardrails, admission)"
    )


if __name__ == "__main__":
    main()
