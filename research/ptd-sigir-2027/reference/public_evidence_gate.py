"""Fail-closed admission for independent RetailRocket Category C evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
EXPECTED_WIDTHS = [300, 600, 1200, 2400]
EXPECTED_SEEDS = [16630, 16631, 16632]
EXPECTED_ARMS = ["internal_max_descendant", "internal_true_sum_mass"]
EXPECTED_HYPOTHESES = {
    "primary_recall_1200",
    "hit_rate_1_1200",
    "beam_interaction_300_to_2400",
    "absolute_gap_shrinkage_300_to_2400",
}
ABS_TOLERANCE = 1e-12


def admission_report(
    run_manifest_path: Path, evaluation_path: Path, paired_observations_path: Path
) -> dict[str, Any]:
    errors = validate_evidence(run_manifest_path, evaluation_path, paired_observations_path)
    return {
        "contract_version": "ptd-public-evidence-admission/v1",
        "category": "C",
        "admissible": not errors,
        "automatic_evidence_admission_allowed": not errors,
        "run_manifest_sha256": _sha256(run_manifest_path) if run_manifest_path.is_file() else None,
        "evaluation_sha256": _sha256(evaluation_path) if evaluation_path.is_file() else None,
        "paired_observations_sha256": (
            _sha256(paired_observations_path) if paired_observations_path.is_file() else None
        ),
        "errors": errors,
    }


def validate_evidence(
    run_manifest_path: Path, evaluation_path: Path, paired_observations_path: Path
) -> list[str]:
    errors: list[str] = []
    try:
        run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in paired_observations_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read public candidate evidence: {exc}"]

    if run.get("schema_version") != "ptd-public-run-manifest/v1" or run.get("status") != "complete":
        errors.append("public run manifest must be complete ptd-public-run-manifest/v1")
    if run.get("category") != "C" or run.get("dataset") != "RetailRocket":
        errors.append("public run must be RetailRocket Category C evidence")
    if run.get("amendment") != 15 or evaluation.get("amendment") != 15:
        errors.append("public evidence must use Amendment 015")
    revision = run.get("code_revision")
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        errors.append("public run code_revision must be a full Git SHA")
    if evaluation.get("source_revision") != revision:
        errors.append("public run and evaluation revisions differ")
    if run.get("arms") != EXPECTED_ARMS or run.get("seeds") != EXPECTED_SEEDS:
        errors.append("public run arm or seed registry differs")
    if run.get("search_topks") != EXPECTED_WIDTHS or evaluation.get("search_topks") != EXPECTED_WIDTHS:
        errors.append("public confirmatory widths must equal 300/600/1200/2400")
    if run.get("return_k") != 600 or run.get("cell_count") != 24 or evaluation.get("cells") != 24:
        errors.append("public evidence must contain all 24 return-600 cells")
    for payload, label in ((run, "run"), (evaluation, "evaluation")):
        if payload.get("validation_only") is not True:
            errors.append(f"{label} must be validation-only")
        if payload.get("sealed_test_opened") is not False:
            errors.append(f"{label} must keep sealed test closed")
    if run.get("immutable") is not True or run.get("test_opened") is not False:
        errors.append("public run immutability/test flags differ")
    if evaluation.get("contract_version") != "ptd_public_retailrocket_confirmatory_evaluation_v1":
        errors.append("public evaluation contract differs")
    if evaluation.get("target_users") != 65_624 or evaluation.get("purchase_users") != 116:
        errors.append("public target or evaluable population differs")
    if evaluation.get("bootstrap_samples") != 10_000 or evaluation.get("bootstrap_seed") != 16630:
        errors.append("public bootstrap contract differs")
    decision = evaluation.get("decision")
    if not isinstance(decision, dict) or decision.get("integrity_failure") is not False:
        errors.append("public evaluation must report no integrity failure")
    if not isinstance(decision, dict) or decision.get("test_opened") is not False:
        errors.append("public evaluation decision opened sealed test")

    artifacts = run.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("public run artifacts must be an object")
    else:
        for name in (
            "audit",
            "input_identities",
            "evaluation",
            "paired_observations",
            "per_user_outcomes",
            "candidate_jaccard",
        ):
            _check_artifact(artifacts.get(name), f"run.artifacts.{name}", errors)
        if isinstance(artifacts.get("evaluation"), dict) and artifacts["evaluation"].get("sha256") != _sha256(
            evaluation_path
        ):
            errors.append("public evaluation hash differs from run manifest")
        if isinstance(artifacts.get("paired_observations"), dict) and artifacts["paired_observations"].get(
            "sha256"
        ) != _sha256(paired_observations_path):
            errors.append("public paired-observation hash differs from run manifest")

    excluded = run.get("excluded_non_evidence")
    if (
        not isinstance(excluded, dict)
        or excluded.get("width") != 4800
        or not isinstance(excluded.get("job_ids"), list)
        or len(excluded["job_ids"]) != 6
    ):
        errors.append("six width-4800 failures must remain registered non-evidence")

    recomputed = _validate_rows(rows, errors)
    hypotheses = evaluation.get("hypotheses")
    if not isinstance(hypotheses, dict) or set(hypotheses) != EXPECTED_HYPOTHESES:
        errors.append("public evaluation hypothesis family differs")
    else:
        for name, expected in recomputed.items():
            observed = hypotheses[name].get("estimate") if isinstance(hypotheses[name], dict) else None
            if not _same_number(observed, expected):
                errors.append(f"{name} estimate does not match paired observations")
            _check_contrast(hypotheses[name], name, errors)
    _check_holm(evaluation, errors)
    if isinstance(decision, dict) and isinstance(hypotheses, dict):
        primary = hypotheses.get("primary_recall_1200", {})
        holm = evaluation.get("holm_bonferroni", {}).get("primary_recall_1200", {})
        expected_success = (
            isinstance(primary, dict)
            and isinstance(primary.get("estimate"), (int, float))
            and primary["estimate"] > 0
            and isinstance(holm, dict)
            and holm.get("rejected") is True
        )
        if decision.get("superiority") is not expected_success:
            errors.append("public superiority decision does not match primary evidence")
    return errors


def _validate_rows(rows: list[dict[str, Any]], errors: list[str]) -> dict[str, float]:
    if len(rows) != 116 * 3 * 4:
        errors.append("paired observations must contain 1,392 complete rows")
        return {name: math.nan for name in EXPECTED_HYPOTHESES}
    keys: set[tuple[str, int, int]] = set()
    by_width: dict[int, list[float]] = defaultdict(list)
    hit_1200: list[float] = []
    by_user_seed: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    users: set[str] = set()
    for row in rows:
        try:
            user = str(row["user_id"])
            seed = int(row["seed"])
            width = int(row["search_topk"])
            recall_sum = float(row["purchase_recall_600"])
            recall_max = float(row["purchase_recall_600_max"])
            recall_difference = float(row["recall_600_difference"])
            hit_sum = float(row["purchase_hit_rate_1"])
            hit_max = float(row["purchase_hit_rate_1_max"])
            hit_difference = float(row["hit_rate_1_difference"])
        except (KeyError, TypeError, ValueError):
            errors.append("paired observation row is malformed")
            continue
        key = (user, seed, width)
        if key in keys:
            errors.append("paired observations contain duplicate user/seed/width rows")
        keys.add(key)
        users.add(user)
        if seed not in EXPECTED_SEEDS or width not in EXPECTED_WIDTHS:
            errors.append("paired observation seed or width is unregistered")
        if not _same_number(recall_difference, recall_sum - recall_max):
            errors.append("paired recall difference is inconsistent")
        if not _same_number(hit_difference, hit_sum - hit_max):
            errors.append("paired Hit@1 difference is inconsistent")
        if not all(math.isfinite(value) and -1 <= value <= 1 for value in (recall_difference, hit_difference)):
            errors.append("paired differences must be finite values in [-1, 1]")
        by_width[width].append(recall_difference)
        by_user_seed[(user, seed)][width] = recall_difference
        if width == 1200:
            hit_1200.append(hit_difference)
    if len(users) != 116 or any(len(widths) != 4 for widths in by_user_seed.values()):
        errors.append("every evaluable user/seed block must contain all four widths")
    means = {width: sum(values) / len(values) for width, values in by_width.items() if values}
    return {
        "primary_recall_1200": means.get(1200, math.nan),
        "hit_rate_1_1200": sum(hit_1200) / len(hit_1200) if hit_1200 else math.nan,
        "beam_interaction_300_to_2400": means.get(2400, math.nan) - means.get(300, math.nan),
        "absolute_gap_shrinkage_300_to_2400": abs(means.get(300, math.nan))
        - abs(means.get(2400, math.nan)),
    }


def _check_holm(evaluation: dict[str, Any], errors: list[str]) -> None:
    hypotheses = evaluation.get("hypotheses")
    observed = evaluation.get("holm_bonferroni")
    if not isinstance(hypotheses, dict) or not isinstance(observed, dict):
        errors.append("public Holm family must be present")
        return
    p_values = {name: float(hypotheses[name]["p_value_two_sided"]) for name in EXPECTED_HYPOTHESES}
    expected = _holm(p_values)
    if set(observed) != EXPECTED_HYPOTHESES:
        errors.append("public Holm result keys differ")
        return
    for name in EXPECTED_HYPOTHESES:
        row = observed[name]
        if not _same_number(row.get("raw_p_value"), p_values[name]):
            errors.append(f"{name} Holm raw p-value differs")
        if not _same_number(row.get("adjusted_p_value"), expected[name]["adjusted_p_value"]):
            errors.append(f"{name} Holm adjusted p-value differs")
        if row.get("rejected") is not expected[name]["rejected"]:
            errors.append(f"{name} Holm rejection differs")


def _holm(p_values: dict[str, float]) -> dict[str, dict[str, float | bool]]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running = 0.0
    reject_open = True
    output: dict[str, dict[str, float | bool]] = {}
    for index, (name, value) in enumerate(ordered):
        multiplier = count - index
        running = max(running, min(1.0, multiplier * value))
        reject = reject_open and value <= 0.05 / multiplier
        output[name] = {"adjusted_p_value": running, "rejected": reject}
        if not reject:
            reject_open = False
    return output


def _check_artifact(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    if not isinstance(value.get("uri"), str) or not value["uri"].startswith("gs://"):
        errors.append(f"{label}.uri must be a GCS URI")
    if not isinstance(value.get("generation"), int) or value["generation"] <= 0:
        errors.append(f"{label}.generation must be positive")
    if value.get("metageneration") != 1:
        errors.append(f"{label}.metageneration must equal one")
    if not isinstance(value.get("sha256"), str) or not SHA256_RE.fullmatch(value["sha256"]):
        errors.append(f"{label}.sha256 must be lowercase SHA-256")


def _check_contrast(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    if value.get("samples") != 10_000 or value.get("seed") != 16630:
        errors.append(f"{label} bootstrap contract differs")
    ci = value.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2 or not all(isinstance(item, (int, float)) for item in ci):
        errors.append(f"{label} confidence interval is malformed")
    p_value = value.get("p_value_two_sided")
    if not isinstance(p_value, (int, float)) or not math.isfinite(p_value) or not 0 <= p_value <= 1:
        errors.append(f"{label} p-value is invalid")


def _same_number(observed: Any, expected: float) -> bool:
    return (
        not isinstance(observed, bool)
        and isinstance(observed, (int, float))
        and math.isfinite(observed)
        and math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=ABS_TOLERANCE)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
