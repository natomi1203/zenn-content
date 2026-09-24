#!/usr/bin/env python3
"""Emit paired PTD JSONL and evaluation JSON from complete retrieval metric rows."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence

from reference.evaluation import (
    PairedScoreRow,
    guardrail_report,
    percentile,
    registered_primary_contrasts,
)
from reference.evidence_gate import (
    ASSERTIONS,
    EXPECTED_SEEDS,
    EXPECTED_SPLIT,
    EXPECTED_VARIANTS,
    METRIC_BOUNDS,
    admission_report,
    canonical_sha256,
)

ROW_METRICS = (
    "purchase_ndcg_at_50",
    "purchase_recall_at_50",
    "purchase_ndcg_at_10",
    "purchase_ndcg_at_100",
    "purchase_auc",
    "click_ndcg_at_50",
    "category_coverage_at_50",
    "max_category_share_at_50",
    "latency_ms",
    "candidates_scored",
)
AVERAGED_METRICS = ROW_METRICS[:8]


@dataclass(frozen=True)
class RetrievalMetricRow:
    date: str
    user_id: str
    seed: int
    variant: str
    metrics: dict[str, float]


def _timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("generated_at must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("generated_at must include a timezone")


def _validate_metric(name: str, value: Any, line_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric at line {line_number}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite at line {line_number}")
    if name in AVERAGED_METRICS:
        if not 0.0 <= normalized <= 1.0:
            raise ValueError(f"{name} must be in [0, 1] at line {line_number}")
    elif normalized < 0.0:
        raise ValueError(f"{name} must be non-negative at line {line_number}")
    return normalized


def load_retrieval_metric_rows(path: Path) -> list[RetrievalMetricRow]:
    """Load strict complete per-query metric evidence for all registered variants."""
    expected_keys = {"date", "user_id", "seed", "variant", *ROW_METRICS}
    rows: list[RetrievalMetricRow] = []
    seen: set[tuple[str, str, int, str]] = set()
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"blank retrieval row at line {line_number}")
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict) or set(value) != expected_keys:
                raise ValueError(
                    f"line {line_number} must contain exactly the registered fields"
                )
            date = value["date"]
            user_id = value["user_id"]
            seed = value["seed"]
            variant = value["variant"]
            if date not in EXPECTED_SPLIT["test"]:
                raise ValueError(f"unexpected test date at line {line_number}")
            if not isinstance(user_id, str) or not user_id:
                raise ValueError(f"user_id must be non-empty at line {line_number}")
            if seed not in EXPECTED_SEEDS:
                raise ValueError(f"unexpected seed at line {line_number}")
            if variant not in EXPECTED_VARIANTS:
                raise ValueError(f"unexpected variant at line {line_number}")
            key = (date, user_id, seed, variant)
            if key in seen:
                raise ValueError(f"duplicate retrieval row at line {line_number}")
            seen.add(key)
            rows.append(
                RetrievalMetricRow(
                    date=date,
                    user_id=user_id,
                    seed=seed,
                    variant=variant,
                    metrics={
                        name: _validate_metric(name, value[name], line_number)
                        for name in ROW_METRICS
                    },
                )
            )
    if not rows:
        raise ValueError("retrieval metric evidence must not be empty")
    _validate_complete_pairing(rows)
    return rows


def _validate_complete_pairing(rows: Sequence[RetrievalMetricRow]) -> None:
    variants_by_unit: dict[tuple[str, str, int], set[str]] = {}
    seeds_by_date_user: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        variants_by_unit.setdefault((row.date, row.user_id, row.seed), set()).add(
            row.variant
        )
        seeds_by_date_user.setdefault((row.date, row.user_id), set()).add(row.seed)
    expected_variants = set(EXPECTED_VARIANTS)
    expected_seeds = set(EXPECTED_SEEDS)
    incomplete_variants = [
        key for key, values in variants_by_unit.items() if values != expected_variants
    ]
    if incomplete_variants:
        raise ValueError(f"incomplete variant pairing for {incomplete_variants[0]}")
    incomplete_seeds = [
        key for key, values in seeds_by_date_user.items() if values != expected_seeds
    ]
    if incomplete_seeds:
        raise ValueError(f"incomplete seed pairing for {incomplete_seeds[0]}")
    observed_dates = {date for date, _ in seeds_by_date_user}
    if observed_dates != set(EXPECTED_SPLIT["test"]):
        raise ValueError("retrieval rows must cover all five test dates")


def _aggregate_metric_group(rows: Sequence[RetrievalMetricRow]) -> dict[str, float]:
    if not rows:
        raise ValueError("metric aggregation group must not be empty")
    metrics = {
        name: fmean(row.metrics[name] for row in rows) for name in AVERAGED_METRICS
    }
    latencies = [row.metrics["latency_ms"] for row in rows]
    metrics.update(
        {
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
            "candidates_scored_mean": fmean(
                row.metrics["candidates_scored"] for row in rows
            ),
        }
    )
    if set(metrics) != set(METRIC_BOUNDS):
        raise ValueError("emitter metric set differs from the registered schema")
    return metrics


def aggregate_variants(rows: Sequence[RetrievalMetricRow]) -> dict[str, dict[str, Any]]:
    """Compute complete aggregate, per-seed, and per-date metric records."""
    records: dict[str, dict[str, Any]] = {}
    for variant in EXPECTED_VARIANTS:
        variant_rows = [row for row in rows if row.variant == variant]
        records[variant] = {
            "status": "complete",
            "n_users": len({row.user_id for row in variant_rows}),
            "metrics": _aggregate_metric_group(variant_rows),
            "by_seed": {
                str(seed): _aggregate_metric_group(
                    [row for row in variant_rows if row.seed == seed]
                )
                for seed in EXPECTED_SEEDS
            },
            "by_date": {
                date: _aggregate_metric_group(
                    [row for row in variant_rows if row.date == date]
                )
                for date in EXPECTED_SPLIT["test"]
            },
        }
    return records


def paired_score_rows(rows: Sequence[RetrievalMetricRow]) -> list[PairedScoreRow]:
    grouped: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in rows:
        grouped.setdefault((row.date, row.user_id, row.seed), {})[row.variant] = (
            row.metrics["purchase_ndcg_at_50"]
        )
    result = [
        PairedScoreRow(
            date=date, user_id=user, seed=seed, scores=dict(sorted(scores.items()))
        )
        for (date, user, seed), scores in sorted(grouped.items())
    ]
    if any(set(row.scores) != set(EXPECTED_VARIANTS) for row in result):
        raise ValueError("paired rows do not contain all variants")
    return result


def _write_paired(path: Path, rows: Iterable[PairedScoreRow]) -> int:
    count = 0
    with path.open("x") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "date": row.date,
                        "user_id": row.user_id,
                        "seed": row.seed,
                        "scores": row.scores,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    return count


def _load_run(run_manifest_path: Path) -> dict[str, Any]:
    try:
        run = json.loads(run_manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read run manifest: {exc}") from exc
    if not isinstance(run, dict):
        raise ValueError("run manifest must be an object")
    if (
        run.get("status") != "complete"
        or run.get("schema_version") != "ptd-run-manifest/v1"
    ):
        raise ValueError("run manifest must be complete ptd-run-manifest/v1")
    if run.get("split") != EXPECTED_SPLIT or run.get("seeds") != EXPECTED_SEEDS:
        raise ValueError("run manifest split/seeds differ from registration")
    if run.get("deviations") != []:
        raise ValueError("non-empty run deviations require manual evaluation emission")
    assertions = run.get("assertions")
    if (
        not isinstance(assertions, dict)
        or set(assertions) != set(ASSERTIONS)
        or not all(assertions.values())
    ):
        raise ValueError("run assertions are incomplete or false")
    return run


def emit_evaluation(
    *,
    run_manifest_path: Path,
    retrieval_metrics_path: Path,
    paired_output_path: Path,
    evaluation_output_path: Path,
    run_manifest_uri: str,
    paired_observations_uri: str,
    generated_at: str,
) -> dict[str, Any]:
    """Build, self-admit, and atomically publish the two evaluation artifacts."""
    if paired_output_path.exists() or evaluation_output_path.exists():
        raise FileExistsError("refusing to overwrite evaluation outputs")
    if not run_manifest_uri or not paired_observations_uri:
        raise ValueError("artifact URIs must be non-empty")
    _timestamp(generated_at)
    run = _load_run(run_manifest_path)
    rows = load_retrieval_metric_rows(retrieval_metrics_path)
    variants = aggregate_variants(rows)
    measured_required = run.get("latency_protocol", {}).get("measured_queries")
    if not isinstance(measured_required, int) or measured_required < 1_000:
        raise ValueError(
            "run manifest latency protocol must require at least 1,000 queries"
        )
    for variant in EXPECTED_VARIANTS:
        measured = sum(row.variant == variant for row in rows)
        if measured < measured_required:
            raise ValueError(f"variant {variant} has fewer latency rows than required")

    paired_rows = paired_score_rows(rows)
    selection = run.get("validation_selection", {})
    best_single = selection.get("best_single_variant")
    if best_single not in {"ptd_item", "ptd_node"}:
        raise ValueError("run manifest lacks a valid locked best-single selection")

    paired_output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        staging = Path(directory)
        staged_paired = staging / "paired_observations.jsonl"
        paired_count = _write_paired(staged_paired, paired_rows)
        paired_hash = canonical_sha256(staged_paired)
        contrasts = registered_primary_contrasts(
            paired_rows,
            best_single_variant=best_single,
        )
        for contrast in contrasts.values():
            contrast["guardrails_pass"] = guardrail_report(
                variants[contrast["numerator"]]["metrics"],
                variants["fixed_tdm"]["metrics"],
            )["passed"]
        evaluation = {
            "schema_version": "ptd-evaluation/v1",
            "status": "complete",
            "generated_at": generated_at,
            "code_revision": run["code_revision"],
            "run_manifest_uri": run_manifest_uri,
            "run_manifest_sha256": canonical_sha256(run_manifest_path),
            "split": EXPECTED_SPLIT,
            "seeds": EXPECTED_SEEDS,
            "assertions": {name: True for name in ASSERTIONS},
            "variants": variants,
            "paired_observations": {
                "uri": paired_observations_uri,
                "sha256": paired_hash,
                "row_count": paired_count,
                "format": "jsonl",
                "row_schema": "ptd-paired-observation-row/v1",
            },
            "primary_contrasts": contrasts,
            "inference": {
                "unit": "date_user_after_seed_average",
                "stratification": "test_date",
                "bootstrap_resamples": 10_000,
                "bootstrap_seed": 20_260_925,
                "interval": "percentile_2.5_97.5",
                "raw_p": "two_sided_bootstrap_sign",
                "multiplicity": "holm_four_primary_contrasts",
            },
            "guardrails": {
                "click_ndcg_relative_floor": 0.95,
                "category_coverage_relative_floor": 0.95,
                "max_category_share_relative_ceiling": 1.05,
                "latency_p95_relative_ceiling": 1.20,
            },
            "deviations": [],
        }
        staged_evaluation = staging / "evaluation.json"
        staged_evaluation.write_text(
            json.dumps(evaluation, indent=2, sort_keys=True) + "\n"
        )
        report = admission_report(run_manifest_path, staged_evaluation, staged_paired)
        if not report["admissible"]:
            raise ValueError(f"emitted evidence failed admission: {report['errors']}")
        os.replace(staged_paired, paired_output_path)
        os.replace(staged_evaluation, evaluation_output_path)
    return {
        "evaluation": evaluation,
        "evaluation_sha256": canonical_sha256(evaluation_output_path),
        "paired_observations_sha256": canonical_sha256(paired_output_path),
        "paired_observation_rows": len(paired_rows),
        "retrieval_metric_rows": len(rows),
        "admission": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--retrieval-metrics", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument("--run-manifest-uri", required=True)
    parser.add_argument("--paired-observations-uri", required=True)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args()
    result = emit_evaluation(
        run_manifest_path=args.run_manifest,
        retrieval_metrics_path=args.retrieval_metrics,
        paired_output_path=args.paired_output,
        evaluation_output_path=args.evaluation_output,
        run_manifest_uri=args.run_manifest_uri,
        paired_observations_uri=args.paired_observations_uri,
        generated_at=args.generated_at,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "evaluation_sha256": result["evaluation_sha256"],
                "paired_observations_sha256": result["paired_observations_sha256"],
                "paired_observation_rows": result["paired_observation_rows"],
            }
        )
    )


if __name__ == "__main__":
    main()
