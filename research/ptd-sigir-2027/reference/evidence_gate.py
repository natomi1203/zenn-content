"""Fail-closed admission checks for prospective PTD five-day evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

EXPECTED_SPLIT = {
    "train": ["2026-07-18", "2026-07-19", "2026-07-20"],
    "validation": ["2026-07-21"],
    "test": ["2026-08-12", "2026-08-14", "2026-08-25", "2026-08-26", "2026-08-28"],
}
EXPECTED_SEEDS = [16630, 16631, 16632]
EXPECTED_VARIANTS = (
    "fixed_tdm",
    "ptd_item",
    "ptd_node",
    "ptd_combined",
    "alternating_tdm",
    "alternating_ptd",
    "ptd_combined_baseline_encoder",
    "teacher_oracle",
)
EXPECTED_CONTRASTS = {
    "rq1_combined_vs_tdm": ("ptd_combined", "fixed_tdm"),
    "rq2_combined_vs_best_single": ("ptd_combined", "best_single_validation_selected"),
    "rq3_alternating_vs_fixed": ("alternating_ptd", "ptd_combined"),
    "rq4_hstu_vs_baseline_encoder": ("ptd_combined", "ptd_combined_baseline_encoder"),
}
ASSERTIONS = (
    "point_in_time_safe",
    "candidate_universe_match",
    "unique_keys",
    "finite_scores",
    "tree_locked_before_test",
    "teacher_frozen",
    "code_hash_match",
)
METRIC_BOUNDS = {
    "purchase_ndcg_at_50": (0.0, 1.0),
    "purchase_recall_at_50": (0.0, 1.0),
    "purchase_ndcg_at_10": (0.0, 1.0),
    "purchase_ndcg_at_100": (0.0, 1.0),
    "purchase_auc": (0.0, 1.0),
    "click_ndcg_at_50": (0.0, 1.0),
    "category_coverage_at_50": (0.0, 1.0),
    "max_category_share_at_50": (0.0, 1.0),
    "latency_p50_ms": (0.0, None),
    "latency_p95_ms": (0.0, None),
    "candidates_scored_mean": (0.0, None),
}
SOURCE_MANIFEST_SHA256 = "33e97c72a3925b5b1ffe5550fbf625e82cacf78f18e073c0be9a2718b4fa1748"
SOURCE_CONTRACT_SHA256 = "900637a45b8cbb84f4bdd15400db3f1a4eae8c0228b4a4b812c2968493abba91"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")


def canonical_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else None
    except ValueError:
        return None


def _check_artifact(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    if not isinstance(value.get("uri"), str) or not value["uri"]:
        errors.append(f"{label}.uri must be non-empty")
    if not isinstance(value.get("sha256"), str) or not SHA256_RE.fullmatch(value["sha256"]):
        errors.append(f"{label}.sha256 must be 64 lowercase hex characters")


def _check_metrics(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    if set(value) != set(METRIC_BOUNDS):
        errors.append(f"{label} must contain exactly the registered metrics")
        return
    for metric, (lower, upper) in METRIC_BOUNDS.items():
        observed = value[metric]
        if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed):
            errors.append(f"{label}.{metric} must be finite")
        elif observed < lower or (upper is not None and observed > upper):
            errors.append(f"{label}.{metric} is outside [{lower}, {upper}]")


def _check_exact_assertions(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict) or set(value) != set(ASSERTIONS):
        errors.append(f"{label} must contain exactly the registered assertions")
        return
    for assertion in ASSERTIONS:
        if value[assertion] is not True:
            errors.append(f"{label}.{assertion} must be true")


def _check_split_and_seeds(payload: dict[str, Any], label: str, errors: list[str]) -> None:
    if payload.get("split") != EXPECTED_SPLIT:
        errors.append(f"{label}.split does not match the preregistered dates")
    if payload.get("seeds") != EXPECTED_SEEDS:
        errors.append(f"{label}.seeds must equal {EXPECTED_SEEDS}")


def validate_evidence(run_manifest_path: Path, evaluation_path: Path) -> list[str]:
    """Return every admission error; an empty list means automatic admission is allowed."""
    errors: list[str] = []
    try:
        run = json.loads(run_manifest_path.read_text())
        evaluation = json.loads(evaluation_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read candidate evidence: {exc}"]
    if not isinstance(run, dict) or not isinstance(evaluation, dict):
        return ["run manifest and evaluation must both be JSON objects"]

    if run.get("schema_version") != "ptd-run-manifest/v1" or run.get("status") != "complete":
        errors.append("run manifest must be complete ptd-run-manifest/v1")
    if evaluation.get("schema_version") != "ptd-evaluation/v1" or evaluation.get("status") != "complete":
        errors.append("evaluation must be complete ptd-evaluation/v1")
    for payload, label in ((run, "run"), (evaluation, "evaluation")):
        revision = payload.get("code_revision")
        if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
            errors.append(f"{label}.code_revision must be 40 lowercase hex characters")
        _check_split_and_seeds(payload, label, errors)
    if run.get("code_revision") != evaluation.get("code_revision"):
        errors.append("code revisions do not match")

    expected_manifest_hash = canonical_sha256(run_manifest_path)
    if evaluation.get("run_manifest_sha256") != expected_manifest_hash:
        errors.append("evaluation.run_manifest_sha256 does not match the supplied run manifest")
    if not isinstance(evaluation.get("run_manifest_uri"), str) or not evaluation["run_manifest_uri"]:
        errors.append("evaluation.run_manifest_uri must be non-empty")

    source = run.get("source_contract")
    if not isinstance(source, dict):
        errors.append("run.source_contract must be an object")
    else:
        if source.get("contract_version") != "shared_bottom_esmm_v2_source":
            errors.append("source contract version mismatch")
        if source.get("manifest_sha256") != SOURCE_MANIFEST_SHA256:
            errors.append("source manifest hash mismatch")
        if source.get("source_contract_sha256") != SOURCE_CONTRACT_SHA256:
            errors.append("source contract hash mismatch")
        if not isinstance(source.get("manifest_uri"), str) or not source["manifest_uri"]:
            errors.append("source manifest URI is missing")

    teacher = run.get("teacher")
    if not isinstance(teacher, dict):
        errors.append("run.teacher must be an object")
    else:
        if teacher.get("teacher_id") != "legacy_loss_shared_bottom_esmm_v2":
            errors.append("teacher identity mismatch")
        if teacher.get("frozen") is not True or teacher.get("score") != "pCTR*pCVR":
            errors.append("teacher must be frozen and use pCTR*pCVR")
        _check_artifact(teacher.get("artifact"), "run.teacher.artifact", errors)

    _check_exact_assertions(run.get("assertions"), "run.assertions", errors)
    _check_exact_assertions(evaluation.get("assertions"), "evaluation.assertions", errors)

    run_variants = run.get("variants")
    if not isinstance(run_variants, dict) or set(run_variants) != set(EXPECTED_VARIANTS):
        errors.append("run.variants must contain exactly the eight registered variants")
    else:
        expected_seed_keys = {str(seed) for seed in EXPECTED_SEEDS}
        for variant in EXPECTED_VARIANTS:
            record = run_variants[variant]
            if not isinstance(record, dict) or record.get("status") != "complete":
                errors.append(f"run variant {variant} must be complete")
                continue
            artifacts = record.get("seed_artifacts")
            if not isinstance(artifacts, dict) or set(artifacts) != expected_seed_keys:
                errors.append(f"run variant {variant} must have all three seed artifacts")
                continue
            for seed, artifact in artifacts.items():
                _check_artifact(artifact, f"run.variants.{variant}.seed_artifacts.{seed}", errors)

    eval_variants = evaluation.get("variants")
    if not isinstance(eval_variants, dict) or set(eval_variants) != set(EXPECTED_VARIANTS):
        errors.append("evaluation.variants must contain exactly the eight registered variants")
    else:
        expected_seed_keys = {str(seed) for seed in EXPECTED_SEEDS}
        expected_dates = set(EXPECTED_SPLIT["test"])
        for variant in EXPECTED_VARIANTS:
            record = eval_variants[variant]
            if not isinstance(record, dict) or record.get("status") != "complete":
                errors.append(f"evaluation variant {variant} must be complete")
                continue
            if not isinstance(record.get("n_users"), int) or record["n_users"] <= 0:
                errors.append(f"evaluation variant {variant}.n_users must be positive")
            _check_metrics(record.get("metrics"), f"evaluation.variants.{variant}.metrics", errors)
            by_seed = record.get("by_seed")
            if not isinstance(by_seed, dict) or set(by_seed) != expected_seed_keys:
                errors.append(f"evaluation variant {variant} must report every seed")
            else:
                for seed, metrics in by_seed.items():
                    _check_metrics(metrics, f"evaluation.variants.{variant}.by_seed.{seed}", errors)
            by_date = record.get("by_date")
            if not isinstance(by_date, dict) or set(by_date) != expected_dates:
                errors.append(f"evaluation variant {variant} must report every test date")
            else:
                for date, metrics in by_date.items():
                    _check_metrics(metrics, f"evaluation.variants.{variant}.by_date.{date}", errors)

    contrasts = evaluation.get("primary_contrasts")
    if not isinstance(contrasts, dict) or set(contrasts) != set(EXPECTED_CONTRASTS):
        errors.append("evaluation.primary_contrasts must contain exactly RQ1--RQ4")
    else:
        for name, (numerator, denominator) in EXPECTED_CONTRASTS.items():
            contrast = contrasts[name]
            if not isinstance(contrast, dict):
                errors.append(f"contrast {name} must be an object")
                continue
            expected = {
                "numerator": numerator,
                "denominator": denominator,
                "metric": "purchase_ndcg_at_50",
                "bootstrap_resamples": 10_000,
                "bootstrap_seed": 20_260_925,
            }
            for key, value in expected.items():
                if contrast.get(key) != value:
                    errors.append(f"contrast {name}.{key} must equal {value}")
            ci = contrast.get("ci95")
            if (
                not isinstance(ci, list)
                or len(ci) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in ci)
                or ci[0] > ci[1]
            ):
                errors.append(f"contrast {name}.ci95 must be two ordered finite numbers")
            for key in ("mean_delta", "raw_p", "holm_adjusted_p"):
                value = contrast.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    errors.append(f"contrast {name}.{key} must be finite")
                elif key.endswith("_p") and not 0.0 <= value <= 1.0:
                    errors.append(f"contrast {name}.{key} must be in [0, 1]")
            if isinstance(contrast.get("raw_p"), (int, float)) and isinstance(contrast.get("holm_adjusted_p"), (int, float)):
                if contrast["holm_adjusted_p"] < contrast["raw_p"]:
                    errors.append(f"contrast {name} has Holm p below raw p")
            if contrast.get("guardrails_pass") is not True:
                errors.append(f"contrast {name}.guardrails_pass must be true for automatic admission")

    guardrails = evaluation.get("guardrails")
    expected_guardrails = {
        "click_ndcg_relative_floor": 0.95,
        "category_coverage_relative_floor": 0.95,
        "max_category_share_relative_ceiling": 1.05,
        "latency_p95_relative_ceiling": 1.20,
    }
    if guardrails != expected_guardrails:
        errors.append("evaluation.guardrails do not match the preregistered thresholds")

    latency = run.get("latency_protocol")
    if not isinstance(latency, dict):
        errors.append("run.latency_protocol must be an object")
    else:
        if latency.get("baseline_variant") != "fixed_tdm" or latency.get("p95_relative_ceiling") != 1.20:
            errors.append("latency protocol must use fixed_tdm and a 1.20 p95 ceiling")
        if latency.get("concurrency") != 1:
            errors.append("latency protocol concurrency must equal one")
        if not isinstance(latency.get("warmup_queries"), int) or latency["warmup_queries"] < 100:
            errors.append("latency protocol needs at least 100 warm-up queries")
        if not isinstance(latency.get("measured_queries"), int) or latency["measured_queries"] < 1000:
            errors.append("latency protocol needs at least 1000 measured queries")
        for key in ("hardware", "software", "timer"):
            if not isinstance(latency.get(key), str) or not latency[key]:
                errors.append(f"latency protocol {key} is missing")

    tree = run.get("tree")
    if not isinstance(tree, dict) or tree.get("locked_before_test") is not True:
        errors.append("tree must be locked before test")
    elif not isinstance(tree.get("locked_tree_sha256"), str) or not SHA256_RE.fullmatch(tree["locked_tree_sha256"]):
        errors.append("locked tree hash is missing")

    created_at = _timestamp(run.get("created_at"))
    test_started_at = _timestamp(run.get("test_scoring_started_at"))
    tree_locked_at = _timestamp(tree.get("locked_at")) if isinstance(tree, dict) else None
    if created_at is None or test_started_at is None or tree_locked_at is None:
        errors.append("run timestamps must be valid ISO-8601 values")
    elif not (created_at <= tree_locked_at <= test_started_at):
        errors.append("run manifest and tree lock must precede test scoring")

    for payload, label in ((run, "run"), (evaluation, "evaluation")):
        deviations = payload.get("deviations")
        if deviations != []:
            errors.append(f"{label}.deviations must be empty for automatic admission")

    return errors


def admission_report(run_manifest_path: Path, evaluation_path: Path) -> dict[str, Any]:
    errors = validate_evidence(run_manifest_path, evaluation_path)
    return {
        "contract_version": "ptd-evidence-admission/v1",
        "admissible": not errors,
        "automatic_claim_promotion_allowed": not errors,
        "run_manifest_sha256": canonical_sha256(run_manifest_path) if run_manifest_path.is_file() else None,
        "evaluation_sha256": canonical_sha256(evaluation_path) if evaluation_path.is_file() else None,
        "errors": errors,
    }
