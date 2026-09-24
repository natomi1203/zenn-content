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
RAW_INPUT_INVENTORY_SHA256 = "d28d69f602b3782f923190768b0d2efc64104c456c9b7b961ea457ed04a31db3"
RAW_SEQUENCE_CONTRACT_SHA256 = "c22332825f6baeab6375040f03fad5ed599f893d37e1ba78e215b0016d8ef4b7"
ITEM_UNIVERSE_AUDIT_SHA256 = "11293abab08c86bf386963d92c27daf116612d590c1a54d65771bdc442147415"
METHOD_CONTRACT_SHA256 = "8fd008bcfddfaeda74f6c6cddfab5b644e664bb5aa645ca778a94a788c5fcfee"
TEACHER_CHECKPOINT_SHA256 = "5a435e4ea2579ca226f26fd8dfa5ad48a7be016f3d1a8e61798ce1b2d6ed1540"
CATALOG_ORDER_SHA256 = "dd41695bb4de9a7d09bae0237cdb2f0c5f1a08b572a5647cdba9c5165bb31d61"
DATE_ELIGIBILITY_SHA256 = {
    "2026-08-12": "5e71dc506230819f22f35527310c5c241a705300c0e61bfb5a6ee5ef537a03d4",
    "2026-08-14": "5600039a543faa8235b26b3a052f870acb2e64ecdbd36c950390f65519367927",
    "2026-08-25": "e1fc0e68422812a637a6ba1eb46b25c0e7cbe9ac1460b71096712910ba156dd8",
    "2026-08-26": "bb63904e1b8863f0632381a26d1f5917feabdeacbbbb3625886a6980826f7861",
    "2026-08-28": "fcc577e51c932f81b5f7c99f7f8a1a754c8013e8dd329c338be195919b7ab46a",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
FLOAT_ABS_TOLERANCE = 1e-12


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


def _same_number(observed: Any, expected: float) -> bool:
    return (
        not isinstance(observed, bool)
        and isinstance(observed, (int, float))
        and math.isfinite(observed)
        and math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=FLOAT_ABS_TOLERANCE)
    )


def validate_evidence(
    run_manifest_path: Path,
    evaluation_path: Path,
    paired_observations_path: Path | None = None,
) -> list[str]:
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
        if source.get("raw_input_inventory_sha256") != RAW_INPUT_INVENTORY_SHA256:
            errors.append("raw input inventory hash mismatch")
        if not isinstance(source.get("raw_input_inventory_uri"), str) or not source["raw_input_inventory_uri"]:
            errors.append("raw input inventory URI is missing")
        if source.get("raw_sequence_contract_sha256") != RAW_SEQUENCE_CONTRACT_SHA256:
            errors.append("raw sequence contract hash mismatch")
        if not isinstance(source.get("raw_sequence_contract_uri"), str) or not source["raw_sequence_contract_uri"]:
            errors.append("raw sequence contract URI is missing")
        if source.get("item_universe_audit_sha256") != ITEM_UNIVERSE_AUDIT_SHA256:
            errors.append("item universe audit hash mismatch")
        if not isinstance(source.get("item_universe_audit_uri"), str) or not source["item_universe_audit_uri"]:
            errors.append("item universe audit URI is missing")

    method_contract = run.get("method_contract")
    _check_artifact(method_contract, "run.method_contract", errors)
    if isinstance(method_contract, dict) and method_contract.get("sha256") != METHOD_CONTRACT_SHA256:
        errors.append("preregistered method contract hash mismatch")

    teacher = run.get("teacher")
    if not isinstance(teacher, dict):
        errors.append("run.teacher must be an object")
    else:
        if teacher.get("teacher_id") != "legacy_loss_shared_bottom_esmm_v2":
            errors.append("teacher identity mismatch")
        if teacher.get("frozen") is not True or teacher.get("score") != "pCTR*pCVR":
            errors.append("teacher must be frozen and use pCTR*pCVR")
        _check_artifact(teacher.get("artifact"), "run.teacher.artifact", errors)
        if isinstance(teacher.get("artifact"), dict):
            if teacher["artifact"].get("sha256") != TEACHER_CHECKPOINT_SHA256:
                errors.append("teacher checkpoint hash mismatch")

    execution = run.get("execution")
    if not isinstance(execution, dict) or set(execution) != {
        "fit_schedule",
        "retrieval_plan",
        "retrieval_metrics_manifest",
    }:
        errors.append("run.execution must link the schedule, plan, and retrieval manifest")
    else:
        for key in ("fit_schedule", "retrieval_plan", "retrieval_metrics_manifest"):
            _check_artifact(execution.get(key), f"run.execution.{key}", errors)

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

    best_single_variant: str | None = None
    selection = run.get("validation_selection")
    if not isinstance(selection, dict):
        errors.append("run.validation_selection must be an object")
    else:
        if selection.get("selection_seed") != 16630:
            errors.append("best-single selection seed must equal 16630")
        if selection.get("selection_metric") != "purchase_ndcg_at_50":
            errors.append("best-single selection metric must be purchase_ndcg_at_50")
        if selection.get("best_single_variant") not in {"ptd_item", "ptd_node"}:
            errors.append("best-single variant must be ptd_item or ptd_node")
        else:
            best_single_variant = selection["best_single_variant"]
        _check_artifact(selection.get("artifact"), "run.validation_selection.artifact", errors)

    paired_rows = None
    paired_record = evaluation.get("paired_observations")
    if not isinstance(paired_record, dict):
        errors.append("evaluation.paired_observations must be an object")
    else:
        if set(paired_record) != {"uri", "sha256", "row_count", "format", "row_schema"}:
            errors.append("evaluation.paired_observations must contain exactly the registered fields")
        _check_artifact(paired_record, "evaluation.paired_observations", errors)
        if paired_record.get("format") != "jsonl" or paired_record.get("row_schema") != "ptd-paired-observation-row/v1":
            errors.append("paired observations must use the registered JSONL row schema")
        if not isinstance(paired_record.get("row_count"), int) or paired_record["row_count"] <= 0:
            errors.append("paired observation row_count must be positive")
    if paired_observations_path is None:
        errors.append("paired observation evidence path is required for recomputation")
    elif not paired_observations_path.is_file():
        errors.append("paired observation evidence file does not exist")
    elif isinstance(paired_record, dict):
        if paired_record.get("sha256") != canonical_sha256(paired_observations_path):
            errors.append("paired observation hash does not match the supplied file")
        try:
            from reference.evaluation import load_paired_score_rows

            paired_rows = load_paired_score_rows(paired_observations_path)
            if paired_record.get("row_count") != len(paired_rows):
                errors.append("paired observation row_count does not match the supplied file")
        except (OSError, ValueError) as exc:
            errors.append(f"invalid paired observation evidence: {exc}")

    recomputed_contrasts = None
    if paired_rows is not None and best_single_variant is not None:
        try:
            from reference.evaluation import primary_metric_summaries, registered_primary_contrasts

            recomputed_contrasts = registered_primary_contrasts(
                paired_rows,
                best_single_variant=best_single_variant,
            )
            summaries = primary_metric_summaries(paired_rows)
            if isinstance(eval_variants, dict) and set(eval_variants) == set(EXPECTED_VARIANTS):
                for variant, summary in summaries.items():
                    record = eval_variants[variant]
                    if record.get("n_users") != summary["n_users"]:
                        errors.append(f"evaluation variant {variant}.n_users does not match paired observations")
                    if not _same_number(record.get("metrics", {}).get("purchase_ndcg_at_50"), float(summary["metrics"])):
                        errors.append(f"evaluation variant {variant}.metrics purchase NDCG does not match paired observations")
                    for seed, expected_value in summary["by_seed"].items():
                        observed = record.get("by_seed", {}).get(seed, {}).get("purchase_ndcg_at_50")
                        if not _same_number(observed, float(expected_value)):
                            errors.append(f"evaluation variant {variant}.by_seed.{seed} purchase NDCG mismatch")
                    for date, expected_value in summary["by_date"].items():
                        observed = record.get("by_date", {}).get(date, {}).get("purchase_ndcg_at_50")
                        if not _same_number(observed, float(expected_value)):
                            errors.append(f"evaluation variant {variant}.by_date.{date} purchase NDCG mismatch")
        except ValueError as exc:
            errors.append(f"cannot recompute primary contrasts: {exc}")

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
                "resolved_denominator": best_single_variant if denominator == "best_single_validation_selected" else denominator,
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
            if not isinstance(contrast.get("paired_units"), int) or contrast["paired_units"] <= 0:
                errors.append(f"contrast {name}.paired_units must be positive")
            if not isinstance(contrast.get("guardrails_pass"), bool):
                errors.append(f"contrast {name}.guardrails_pass must be boolean")
            if recomputed_contrasts is not None:
                recomputed = recomputed_contrasts[name]
                for key in ("mean_delta", "raw_p", "holm_adjusted_p"):
                    if not _same_number(contrast.get(key), float(recomputed[key])):
                        errors.append(f"contrast {name}.{key} does not match paired-observation recomputation")
                observed_ci = contrast.get("ci95")
                expected_ci = recomputed["ci95"]
                if (
                    not isinstance(observed_ci, list)
                    or len(observed_ci) != 2
                    or not all(_same_number(observed_ci[index], float(expected_ci[index])) for index in range(2))
                ):
                    errors.append(f"contrast {name}.ci95 does not match paired-observation recomputation")
                if contrast.get("paired_units") != recomputed["paired_units"]:
                    errors.append(f"contrast {name}.paired_units does not match paired-observation recomputation")
            if (
                isinstance(eval_variants, dict)
                and set(eval_variants) == set(EXPECTED_VARIANTS)
                and isinstance(contrast.get("resolved_denominator"), str)
                and contrast.get("numerator") in eval_variants
                and contrast["resolved_denominator"] in eval_variants
            ):
                from reference.evaluation import guardrail_report

                treatment_metrics = eval_variants[contrast["numerator"]].get("metrics", {})
                baseline_metrics = eval_variants["fixed_tdm"].get("metrics", {})
                try:
                    expected_guardrail_pass = guardrail_report(treatment_metrics, baseline_metrics)["passed"]
                    if contrast.get("guardrails_pass") is not expected_guardrail_pass:
                        errors.append(f"contrast {name}.guardrails_pass does not match aggregate metrics")
                except (KeyError, ValueError) as exc:
                    errors.append(f"cannot recompute contrast {name} guardrails: {exc}")

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
    elif not isinstance(tree.get("locked_bundle_sha256"), str) or not SHA256_RE.fullmatch(tree["locked_bundle_sha256"]):
        errors.append("locked tree bundle hash is missing")
    if isinstance(tree, dict):
        expected_tree_constants = {
            "branching_factor": 2,
            "depth": 13,
            "leaf_capacity": 1,
            "beam_width": 600,
            "catalog_order_sha256": CATALOG_ORDER_SHA256,
            "item_universe_audit_sha256": ITEM_UNIVERSE_AUDIT_SHA256,
        }
        for key, expected in expected_tree_constants.items():
            if tree.get(key) != expected:
                errors.append(f"tree.{key} does not match the preregistered method")
        if tree.get("date_eligibility_sha256") != DATE_ELIGIBILITY_SHA256:
            errors.append("tree date eligibility hashes do not match the fixed candidate sets")
        cycles_by_variant = tree.get("alternating_cycles_selected_by_variant_and_seed")
        expected_seed_keys = {str(seed) for seed in EXPECTED_SEEDS}
        expected_variants = {"alternating_tdm", "alternating_ptd"}
        if not isinstance(cycles_by_variant, dict) or set(cycles_by_variant) != expected_variants:
            errors.append("alternating cycle selection must cover both alternating variants")
        else:
            for variant, cycles_by_seed in cycles_by_variant.items():
                if not isinstance(cycles_by_seed, dict) or set(cycles_by_seed) != expected_seed_keys:
                    errors.append(f"alternating cycle selection for {variant} must cover every seed")
                    continue
                for seed, cycles in cycles_by_seed.items():
                    if isinstance(cycles, bool) or not isinstance(cycles, int) or not 0 <= cycles <= 3:
                        errors.append(
                            f"alternating cycles for {variant}/{seed} must be an integer in [0, 3]"
                        )

    hyperparameters = run.get("selected_hyperparameters")
    if not isinstance(hyperparameters, dict):
        errors.append("run.selected_hyperparameters must be an object")
    else:
        if hyperparameters.get("selection_seed") != 16630:
            errors.append("hyperparameter selection seed must equal 16630")
        if hyperparameters.get("temperature") not in (1.0, 2.0, 4.0):
            errors.append("temperature is outside the registered grid")
        for key in ("lambda_item", "lambda_node"):
            if hyperparameters.get(key) not in (0.1, 0.3, 1.0):
                errors.append(f"{key} is outside the registered grid")
        if hyperparameters.get("epsilon_item") != 1e-6 or hyperparameters.get("epsilon_node") != 1e-12:
            errors.append("teacher-target epsilon values do not match the registration")
        if hyperparameters.get("assignment_weight") != "y_plus_teacher_times_path_log_probability":
            errors.append("assignment weight definition mismatch")
        _check_artifact(
            hyperparameters.get("selection_artifact"),
            "run.selected_hyperparameters.selection_artifact",
            errors,
        )

    expected_inference = {
        "unit": "date_user_after_seed_average",
        "stratification": "test_date",
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 20_260_925,
        "interval": "percentile_2.5_97.5",
        "raw_p": "two_sided_bootstrap_sign",
        "multiplicity": "holm_four_primary_contrasts",
    }
    if evaluation.get("inference") != expected_inference:
        errors.append("evaluation.inference does not match the registered procedure")

    created_at = _timestamp(run.get("created_at"))
    test_started_at = _timestamp(run.get("test_scoring_started_at"))
    completed_at = _timestamp(run.get("completed_at"))
    tree_locked_at = _timestamp(tree.get("locked_at")) if isinstance(tree, dict) else None
    if (
        created_at is None
        or test_started_at is None
        or completed_at is None
        or tree_locked_at is None
    ):
        errors.append("run timestamps must be valid ISO-8601 values")
    elif not (created_at <= tree_locked_at <= test_started_at <= completed_at):
        errors.append("run creation/tree lock/test start/completion timestamps are out of order")

    for payload, label in ((run, "run"), (evaluation, "evaluation")):
        deviations = payload.get("deviations")
        if deviations != []:
            errors.append(f"{label}.deviations must be empty for automatic admission")

    return errors


def admission_report(
    run_manifest_path: Path,
    evaluation_path: Path,
    paired_observations_path: Path | None = None,
) -> dict[str, Any]:
    errors = validate_evidence(run_manifest_path, evaluation_path, paired_observations_path)
    return {
        "contract_version": "ptd-evidence-admission/v1",
        "admissible": not errors,
        "automatic_evidence_admission_allowed": not errors,
        "run_manifest_sha256": canonical_sha256(run_manifest_path) if run_manifest_path.is_file() else None,
        "evaluation_sha256": canonical_sha256(evaluation_path) if evaluation_path.is_file() else None,
        "paired_observations_sha256": (
            canonical_sha256(paired_observations_path)
            if paired_observations_path is not None and paired_observations_path.is_file()
            else None
        ),
        "errors": errors,
    }
