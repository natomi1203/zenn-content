#!/usr/bin/env python3
"""Lock a 24-cell retrieval plan and finalize its post-retrieval run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from reference.evidence_gate import (
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
from runner.build_fit_schedule import ALTERNATING_VARIANTS, FIXED_TREE_VARIANTS
from runner.retrieval_runner import load_plan, plan_lock_sha256

ROOT = Path(__file__).resolve().parents[1]
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_FIT_COUNTS = {
    "grid_fit_executions": 27,
    "single_level_fit_executions": 2,
    "additional_fixed_tree_fit_executions": 12,
    "alternating_chains": 6,
    "fits_per_alternating_chain": 4,
    "alternating_fit_executions": 24,
    "total_fit_executions": 65,
    "locked_tree_model_artifacts": 21,
    "retrieval_variant_seed_entries": 24,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _artifact(path: Path, label: str) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return {"path": str(path), "sha256": sha256(path)}


def _checked_artifact(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = Path(value["path"])
    if not path.is_file() or sha256(path) != value["sha256"]:
        raise ValueError(f"{label} path/hash mismatch")
    return {"path": str(path), "sha256": str(value["sha256"])}


def _uri_artifact(value: Mapping[str, str]) -> dict[str, str]:
    return {"uri": value["path"], "sha256": value["sha256"]}


def _timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _write_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        staged = Path(directory) / path.name
        staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(staged, path)


def _validate_schedule(path: Path, code_revision: str) -> dict[str, Any]:
    schedule = _load(path, "fit schedule")
    if (
        schedule.get("contract_version") != "ptd-fit-schedule/v1"
        or schedule.get("status") != "locked"
        or schedule.get("code_revision") != code_revision
        or schedule.get("selection_seed") != 16630
        or schedule.get("seeds") != EXPECTED_SEEDS
        or schedule.get("counts") != EXPECTED_FIT_COUNTS
        or not all(schedule.get("checks", {}).values())
    ):
        raise ValueError("fit schedule is incomplete or belongs to another revision")
    if len(schedule.get("grid_fits", [])) != 27:
        raise ValueError("fit schedule must contain 27 grid fits")
    if len(schedule.get("single_level_fits", [])) != 2:
        raise ValueError("fit schedule must contain two single-level fits")
    if len(schedule.get("final_fixed_tree_models", [])) != 15:
        raise ValueError("fit schedule must contain 15 fixed-tree model cells")
    if len(schedule.get("alternating_chains", [])) != 6:
        raise ValueError("fit schedule must contain six alternating chains")
    inputs = schedule.get("inputs")
    expected_inputs = {
        "teacher_scores_manifest",
        "training_examples_manifest",
        "validation_queries_manifest",
        "assignment_queries_manifest",
        "fixed_catalog",
        "fixed_date_eligibility",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_inputs:
        raise ValueError("fit schedule input artifacts are incomplete")
    for key in sorted(expected_inputs):
        _checked_artifact(inputs[key], f"fit schedule {key}")
    return schedule


def _validate_selection(
    path: Path, *, expected_validation_manifest_sha256: str | None = None
) -> tuple[dict[str, Any], dict[str, float]]:
    selection = _load(path, "validation selection")
    if (
        selection.get("contract_version") != "ptd-validation-selection/v1"
        or selection.get("status") != "complete"
        or selection.get("selection_seed") != 16630
        or selection.get("selection_dates") != EXPECTED_SPLIT["validation"]
        or selection.get("selection_metric") != "purchase_ndcg_at_50"
        or selection.get("best_single_variant") not in {"ptd_item", "ptd_node"}
        or not all(selection.get("checks", {}).values())
    ):
        raise ValueError("validation selection is incomplete")
    if (
        expected_validation_manifest_sha256 is not None
        and selection.get("validation_queries_manifest_sha256")
        != expected_validation_manifest_sha256
    ):
        raise ValueError("validation selection is not linked to the scheduled queries")
    selected = selection.get("selected_hyperparameters")
    if not isinstance(selected, dict):
        raise ValueError("validation selection lacks selected hyperparameters")
    hyperparameters = {
        "temperature": float(selected.get("temperature")),
        "lambda_item": float(selected.get("lambda_item")),
        "lambda_node": float(selected.get("lambda_node")),
    }
    if hyperparameters["temperature"] not in {1.0, 2.0, 4.0} or any(
        hyperparameters[key] not in {0.1, 0.3, 1.0}
        for key in ("lambda_item", "lambda_node")
    ):
        raise ValueError("selected hyperparameters are outside the registered grid")
    return selection, hyperparameters


def _validate_teacher_manifest(path: Path) -> dict[str, Any]:
    teacher = _load(path, "teacher scores manifest")
    checks = teacher.get("checks", {})
    if (
        teacher.get("contract_version") != "ptd-frozen-teacher-scores/v1"
        or teacher.get("status") != "complete"
        or teacher.get("teacher_checkpoint_sha256") != TEACHER_CHECKPOINT_SHA256
        or teacher.get("label_columns_read") != []
        or checks.get("labels_read") is not False
        or not all(value for key, value in checks.items() if key != "labels_read")
    ):
        raise ValueError("teacher scores manifest violates the frozen-teacher contract")
    return teacher


def _validate_query_manifest(
    path: Path, *, expected_teacher_manifest_sha256: str
) -> dict[str, str]:
    query = _load(path, "retrieval query manifest")
    if (
        query.get("contract_version") != "ptd-retrieval-queries/v1"
        or query.get("status") != "complete"
        or query.get("teacher_manifest_sha256")
        != expected_teacher_manifest_sha256
        or set(query.get("queries_by_date", {})) != set(EXPECTED_SPLIT["test"])
        or not all(query.get("checks", {}).values())
    ):
        raise ValueError("retrieval query manifest is incomplete")
    return _checked_artifact(query.get("output"), "retrieval queries")


def _validate_fit(
    path: Path,
    selected: Mapping[str, float],
    *,
    expected_training_examples_manifest_sha256: str,
) -> tuple[tuple[str, int], dict[str, Any]]:
    fit = _load(path, "fixed-tree fit manifest")
    variant = fit.get("variant")
    seed = fit.get("seed")
    configuration = fit.get("configuration", {})
    if (
        fit.get("contract_version") != "ptd-single-fit/v1"
        or fit.get("status") != "complete"
        or variant not in FIXED_TREE_VARIANTS
        or seed not in EXPECTED_SEEDS
        or fit.get("test_only_config_override") is not False
        or fit.get("initial_fit") is not None
        or fit.get("training_examples_manifest_sha256")
        != expected_training_examples_manifest_sha256
        or not all(fit.get("checks", {}).values())
        or any(configuration.get(key) != selected[key] for key in selected)
    ):
        raise ValueError(f"fixed-tree fit is incompatible: {path}")
    checkpoint = _checked_artifact(fit.get("checkpoint"), f"{variant}/{seed} checkpoint")
    return (str(variant), int(seed)), {"fit": fit, "checkpoint": checkpoint}


def _bundle_hash(record: Mapping[str, Any]) -> str:
    fields = (
        record["catalog"]["sha256"],
        record["date_eligibility"]["sha256"],
        record["fit_manifest"]["sha256"],
        record["checkpoint"]["sha256"],
    )
    return hashlib.sha256(("\n".join(fields) + "\n").encode()).hexdigest()


def _validate_cycle(
    path: Path,
    selected: Mapping[str, float],
    *,
    schedule_inputs: Mapping[str, Mapping[str, str]],
) -> tuple[tuple[str, int], dict[str, Any]]:
    cycle = _load(path, "alternating cycle manifest")
    variant = cycle.get("variant")
    seed = cycle.get("seed")
    if (
        cycle.get("contract_version") != "ptd-alternating-cycle-selection/v1"
        or cycle.get("status") != "complete"
        or variant not in ALTERNATING_VARIANTS
        or seed not in EXPECTED_SEEDS
        or cycle.get("maximum_cycles") != 3
        or cycle.get("hyperparameters") != dict(selected)
        or cycle.get("teacher_manifest_sha256")
        != schedule_inputs["teacher_scores_manifest"]["sha256"]
        or cycle.get("assignment_queries_manifest_sha256")
        != schedule_inputs["assignment_queries_manifest"]["sha256"]
        or cycle.get("validation_queries_manifest_sha256")
        != schedule_inputs["validation_queries_manifest"]["sha256"]
        or cycle.get("initial_catalog_sha256")
        != schedule_inputs["fixed_catalog"]["sha256"]
        or cycle.get("initial_date_eligibility_sha256")
        != schedule_inputs["fixed_date_eligibility"]["sha256"]
        or not all(cycle.get("checks", {}).values())
    ):
        raise ValueError(f"alternating cycle selection is incompatible: {path}")
    selected_cycle = cycle.get("selected_cycle")
    record = cycle.get("selected")
    records = cycle.get("cycles")
    if (
        isinstance(selected_cycle, bool)
        or not isinstance(selected_cycle, int)
        or not 0 <= selected_cycle <= 3
        or not isinstance(records, list)
        or len(records) != 4
        or [value.get("cycle") for value in records if isinstance(value, dict)]
        != [0, 1, 2, 3]
        or not isinstance(record, dict)
        or record.get("cycle") != selected_cycle
        or record != records[selected_cycle]
    ):
        raise ValueError("alternating selected-cycle record mismatch")
    for index, value in enumerate(records):
        if value.get("warm_started_from_cycle") != (None if index == 0 else index - 1):
            raise ValueError("alternating warm-start chain mismatch")
    for key in ("catalog", "date_eligibility", "fit_manifest", "checkpoint"):
        _checked_artifact(record.get(key), f"{variant}/{seed} selected {key}")
    if cycle.get("selected_bundle_sha256") != _bundle_hash(record):
        raise ValueError("alternating selected bundle hash mismatch")
    return (str(variant), int(seed)), {"cycle": cycle, "selected": record}


def lock_retrieval_plan(
    *,
    code_revision: str,
    created_at: str,
    locked_at: str,
    fit_schedule_path: Path,
    validation_selection_path: Path,
    teacher_scores_manifest_path: Path,
    retrieval_queries_manifest_path: Path,
    fixed_catalog_path: Path,
    fixed_date_eligibility_path: Path,
    fit_manifest_paths: Sequence[Path],
    alternating_cycle_manifest_paths: Sequence[Path],
    device: str,
    hardware: str,
    software: str,
    timer: str,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not REVISION_RE.fullmatch(code_revision):
        raise ValueError("code revision must be 40 lowercase hexadecimal characters")
    if _timestamp(created_at, "created_at") > _timestamp(locked_at, "locked_at"):
        raise ValueError("created_at must not follow locked_at")
    if not all((device, hardware, software, timer)):
        raise ValueError("runtime protocol fields must be non-empty")
    schedule = _validate_schedule(fit_schedule_path, code_revision)
    schedule_inputs = schedule["inputs"]
    if schedule_inputs["teacher_scores_manifest"] != _artifact(
        teacher_scores_manifest_path, "teacher scores manifest"
    ):
        raise ValueError("teacher manifest differs from the fit schedule")
    if schedule_inputs["fixed_catalog"] != _artifact(
        fixed_catalog_path, "fixed catalog"
    ):
        raise ValueError("fixed catalog differs from the fit schedule")
    if schedule_inputs["fixed_date_eligibility"] != _artifact(
        fixed_date_eligibility_path, "fixed date eligibility"
    ):
        raise ValueError("fixed date eligibility differs from the fit schedule")
    _, selected = _validate_selection(
        validation_selection_path,
        expected_validation_manifest_sha256=schedule_inputs[
            "validation_queries_manifest"
        ]["sha256"],
    )
    _validate_teacher_manifest(teacher_scores_manifest_path)
    queries = _validate_query_manifest(
        retrieval_queries_manifest_path,
        expected_teacher_manifest_sha256=schedule_inputs[
            "teacher_scores_manifest"
        ]["sha256"],
    )
    fixed_catalog = _artifact(fixed_catalog_path, "fixed catalog")
    fixed_eligibility = _artifact(
        fixed_date_eligibility_path, "fixed date eligibility"
    )

    fits: dict[tuple[str, int], dict[str, Any]] = {}
    for path in fit_manifest_paths:
        identity, value = _validate_fit(
            path,
            selected,
            expected_training_examples_manifest_sha256=schedule_inputs[
                "training_examples_manifest"
            ]["sha256"],
        )
        if identity in fits:
            raise ValueError(f"duplicate fixed-tree fit identity: {identity}")
        fits[identity] = {**value, "artifact": _artifact(path, "fit manifest")}
    expected_fits = {
        (variant, seed) for variant in FIXED_TREE_VARIANTS for seed in EXPECTED_SEEDS
    }
    if set(fits) != expected_fits:
        raise ValueError("fixed-tree fits do not cover the exact 5x3 matrix")

    cycles: dict[tuple[str, int], dict[str, Any]] = {}
    for path in alternating_cycle_manifest_paths:
        identity, value = _validate_cycle(
            path, selected, schedule_inputs=schedule_inputs
        )
        if identity in cycles:
            raise ValueError(f"duplicate alternating cycle identity: {identity}")
        cycles[identity] = {**value, "artifact": _artifact(path, "cycle manifest")}
    expected_cycles = {
        (variant, seed) for variant in ALTERNATING_VARIANTS for seed in EXPECTED_SEEDS
    }
    if set(cycles) != expected_cycles:
        raise ValueError("alternating cycles do not cover the exact 2x3 matrix")

    teacher_artifact = _artifact(teacher_scores_manifest_path, "teacher manifest")
    entries = []
    for variant in EXPECTED_VARIANTS:
        for seed in EXPECTED_SEEDS:
            if variant in FIXED_TREE_VARIANTS:
                value = fits[(variant, seed)]
                entry = {
                    "variant": variant,
                    "seed": seed,
                    "catalog": fixed_catalog,
                    "date_eligibility": fixed_eligibility,
                    "model_artifact": value["artifact"],
                    "checkpoint": value["checkpoint"],
                    "selected_cycle": None,
                }
            elif variant in ALTERNATING_VARIANTS:
                value = cycles[(variant, seed)]
                record = value["selected"]
                entry = {
                    "variant": variant,
                    "seed": seed,
                    "catalog": dict(record["catalog"]),
                    "date_eligibility": dict(record["date_eligibility"]),
                    "model_artifact": value["artifact"],
                    "checkpoint": dict(record["checkpoint"]),
                    "selected_cycle": value["cycle"]["selected_cycle"],
                }
            else:
                entry = {
                    "variant": variant,
                    "seed": seed,
                    "catalog": fixed_catalog,
                    "date_eligibility": fixed_eligibility,
                    "model_artifact": teacher_artifact,
                    "checkpoint": None,
                    "selected_cycle": None,
                }
            entries.append(entry)

    checks = {
        "fit_schedule_complete": True,
        "validation_selection_complete": True,
        "all_24_variant_seed_entries": len(entries) == 24,
        "all_artifact_hashes_match": True,
        "selected_hyperparameters_match": True,
        "alternating_cycles_validation_selected": True,
        "tree_models_locked_before_test": True,
        "test_queries_not_read_during_lock": True,
        "no_overwrite": True,
    }
    plan = {
        "contract_version": "ptd-retrieval-run-plan/v1",
        "status": "locked",
        "created_at": created_at,
        "locked_at": locked_at,
        "code_revision": code_revision,
        "fit_schedule": _artifact(fit_schedule_path, "fit schedule"),
        "validation_selection": _artifact(
            validation_selection_path, "validation selection"
        ),
        "teacher_scores_manifest": teacher_artifact,
        "locked_bundle_sha256": "0" * 64,
        "test_dates": EXPECTED_SPLIT["test"],
        "seeds": EXPECTED_SEEDS,
        "beam_width": 600,
        "top_k": 600,
        "warmup_queries_per_variant_seed": 100,
        "minimum_measured_queries_per_variant": 1000,
        "concurrency": 1,
        "device": device,
        "hardware": hardware,
        "software": software,
        "timer": timer,
        "queries": queries,
        "entries": entries,
        "checks": checks,
        "scope_note": (
            "All model, tree, mask, query, and code hashes were locked before test "
            "scoring. The lock step reads no test query payload or outcome."
        ),
    }
    plan["locked_bundle_sha256"] = plan_lock_sha256(plan)
    _write_no_clobber(output_path, plan)
    load_plan(output_path)
    return plan


def _validate_retrieval_metrics(
    path: Path, plan_path: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = _load(path, "retrieval metrics manifest")
    if (
        metrics.get("contract_version") != "ptd-retrieval-metrics/v1"
        or metrics.get("status") != "complete"
        or metrics.get("plan_sha256") != sha256(plan_path)
        or metrics.get("queries_sha256") != plan["queries"]["sha256"]
        or metrics.get("test_dates") != EXPECTED_SPLIT["test"]
        or metrics.get("seeds") != EXPECTED_SEEDS
        or metrics.get("variants") != list(EXPECTED_VARIANTS)
        or not all(metrics.get("checks", {}).values())
    ):
        raise ValueError("retrieval metrics manifest is incomplete or unlinked")
    counts = metrics.get("measured_rows_by_variant")
    if not isinstance(counts, dict) or set(counts) != set(EXPECTED_VARIANTS):
        raise ValueError("retrieval metrics variant counts are incomplete")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < plan["minimum_measured_queries_per_variant"]
        for value in counts.values()
    ):
        raise ValueError("retrieval metrics do not satisfy the measured-query minimum")
    _checked_artifact(metrics.get("output"), "retrieval metric rows")
    return metrics


def finalize_run_manifest(
    *,
    retrieval_plan_path: Path,
    retrieval_metrics_manifest_path: Path,
    run_id: str,
    test_scoring_started_at: str,
    completed_at: str,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not run_id:
        raise ValueError("run_id must be non-empty")
    plan = load_plan(retrieval_plan_path)
    metrics = _validate_retrieval_metrics(
        retrieval_metrics_manifest_path, retrieval_plan_path, plan
    )
    created = _timestamp(plan["created_at"], "plan.created_at")
    locked = _timestamp(plan["locked_at"], "plan.locked_at")
    started = _timestamp(test_scoring_started_at, "test_scoring_started_at")
    completed = _timestamp(completed_at, "completed_at")
    if not created <= locked <= started <= completed:
        raise ValueError("run timestamps must satisfy created <= locked <= started <= completed")

    fit_schedule_path = Path(plan["fit_schedule"]["path"])
    schedule = _validate_schedule(fit_schedule_path, plan["code_revision"])
    selection_path = Path(plan["validation_selection"]["path"])
    selection, selected = _validate_selection(
        selection_path,
        expected_validation_manifest_sha256=schedule["inputs"][
            "validation_queries_manifest"
        ]["sha256"],
    )
    entries = {
        (entry["variant"], int(entry["seed"])): entry for entry in plan["entries"]
    }
    variants = {}
    for variant in EXPECTED_VARIANTS:
        variants[variant] = {
            "status": "complete",
            "seed_artifacts": {
                str(seed): _uri_artifact(entries[(variant, seed)]["model_artifact"])
                for seed in EXPECTED_SEEDS
            },
        }
    cycles = {
        variant: {
            str(seed): entries[(variant, seed)]["selected_cycle"]
            for seed in EXPECTED_SEEDS
        }
        for variant in ALTERNATING_VARIANTS
    }
    legacy = _load(
        ROOT / "artifact" / "verified" / "legacy_esmm_evidence.json",
        "legacy ESMM evidence",
    )
    source = legacy["manifest"]
    teacher = legacy["teacher"]
    plan_artifact = _artifact(retrieval_plan_path, "retrieval plan")
    metrics_artifact = _artifact(
        retrieval_metrics_manifest_path, "retrieval metrics manifest"
    )
    run = {
        "schema_version": "ptd-run-manifest/v1",
        "status": "complete",
        "run_id": run_id,
        "created_at": plan["created_at"],
        "test_scoring_started_at": test_scoring_started_at,
        "completed_at": completed_at,
        "code_revision": plan["code_revision"],
        "source_contract": {
            "contract_version": source["contract_version"],
            "manifest_uri": source["uri"],
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
            "teacher_id": teacher["teacher_id"],
            "frozen": True,
            "score": teacher["score"],
            "artifact": {
                "uri": teacher["checkpoint"]["uri"],
                "sha256": TEACHER_CHECKPOINT_SHA256,
            },
        },
        "split": EXPECTED_SPLIT,
        "seeds": EXPECTED_SEEDS,
        "assertions": {name: True for name in ASSERTIONS},
        "variants": variants,
        "selected_hyperparameters": {
            "selection_seed": 16630,
            **selected,
            "epsilon_item": 1e-6,
            "epsilon_node": 1e-12,
            "assignment_weight": "y_plus_teacher_times_path_log_probability",
            "selection_artifact": _uri_artifact(plan["validation_selection"]),
        },
        "validation_selection": {
            "selection_seed": 16630,
            "selection_metric": "purchase_ndcg_at_50",
            "best_single_variant": selection["best_single_variant"],
            "artifact": _uri_artifact(plan["validation_selection"]),
        },
        "tree": {
            "branching_factor": 2,
            "depth": 13,
            "leaf_capacity": 1,
            "beam_width": 600,
            "catalog_order_sha256": CATALOG_ORDER_SHA256,
            "item_universe_audit_sha256": ITEM_UNIVERSE_AUDIT_SHA256,
            "date_eligibility_sha256": DATE_ELIGIBILITY_SHA256,
            "alternating_cycles_selected_by_variant_and_seed": cycles,
            "locked_before_test": True,
            "locked_at": plan["locked_at"],
            "locked_bundle_sha256": plan["locked_bundle_sha256"],
        },
        "latency_protocol": {
            "baseline_variant": "fixed_tdm",
            "p95_relative_ceiling": 1.2,
            "warmup_queries": plan["warmup_queries_per_variant_seed"],
            "measured_queries": plan["minimum_measured_queries_per_variant"],
            "concurrency": plan["concurrency"],
            "hardware": plan["hardware"],
            "software": plan["software"],
            "timer": plan["timer"],
        },
        "execution": {
            "fit_schedule": _uri_artifact(plan["fit_schedule"]),
            "retrieval_plan": _uri_artifact(plan_artifact),
            "retrieval_metrics_manifest": _uri_artifact(metrics_artifact),
        },
        "deviations": [],
    }
    if metrics["plan_sha256"] != run["execution"]["retrieval_plan"]["sha256"]:
        raise ValueError("run manifest retrieval plan link changed during finalization")
    _write_no_clobber(output_path, run)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock = subparsers.add_parser("lock", help="Lock the pre-test retrieval plan")
    lock.add_argument("--code-revision", required=True)
    lock.add_argument("--created-at", required=True)
    lock.add_argument("--locked-at", required=True)
    lock.add_argument("--fit-schedule", type=Path, required=True)
    lock.add_argument("--validation-selection", type=Path, required=True)
    lock.add_argument("--teacher-scores-manifest", type=Path, required=True)
    lock.add_argument("--retrieval-queries-manifest", type=Path, required=True)
    lock.add_argument("--fixed-catalog", type=Path, required=True)
    lock.add_argument("--fixed-date-eligibility", type=Path, required=True)
    lock.add_argument("--fit-manifest", type=Path, action="append", required=True)
    lock.add_argument(
        "--alternating-cycle-manifest", type=Path, action="append", required=True
    )
    lock.add_argument("--device", required=True)
    lock.add_argument("--hardware", required=True)
    lock.add_argument("--software", required=True)
    lock.add_argument("--timer", required=True)
    lock.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser("finalize", help="Finalize the post-retrieval run manifest")
    finalize.add_argument("--retrieval-plan", type=Path, required=True)
    finalize.add_argument("--retrieval-metrics-manifest", type=Path, required=True)
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--test-scoring-started-at", required=True)
    finalize.add_argument("--completed-at", required=True)
    finalize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "lock":
        result = lock_retrieval_plan(
            code_revision=args.code_revision,
            created_at=args.created_at,
            locked_at=args.locked_at,
            fit_schedule_path=args.fit_schedule,
            validation_selection_path=args.validation_selection,
            teacher_scores_manifest_path=args.teacher_scores_manifest,
            retrieval_queries_manifest_path=args.retrieval_queries_manifest,
            fixed_catalog_path=args.fixed_catalog,
            fixed_date_eligibility_path=args.fixed_date_eligibility,
            fit_manifest_paths=args.fit_manifest,
            alternating_cycle_manifest_paths=args.alternating_cycle_manifest,
            device=args.device,
            hardware=args.hardware,
            software=args.software,
            timer=args.timer,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "entries": len(result["entries"]),
                    "locked_bundle_sha256": result["locked_bundle_sha256"],
                }
            )
        )
    else:
        result = finalize_run_manifest(
            retrieval_plan_path=args.retrieval_plan,
            retrieval_metrics_manifest_path=args.retrieval_metrics_manifest,
            run_id=args.run_id,
            test_scoring_started_at=args.test_scoring_started_at,
            completed_at=args.completed_at,
            output_path=args.output,
        )
        print(json.dumps({"status": result["status"], "run_id": result["run_id"]}))


if __name__ == "__main__":
    main()
