"""Dependency-free reference statistics for the registered PTD evaluation."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from reference.evidence_gate import EXPECTED_SEEDS, EXPECTED_SPLIT, EXPECTED_VARIANTS

PRIMARY_CONTRASTS = {
    "rq1_combined_vs_tdm": ("ptd_combined", "fixed_tdm"),
    "rq2_combined_vs_best_single": ("ptd_combined", "best_single_validation_selected"),
    "rq3_alternating_vs_fixed": ("alternating_ptd", "ptd_combined"),
    "rq4_hstu_vs_baseline_encoder": ("ptd_combined", "ptd_combined_baseline_encoder"),
}


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Discounted cumulative gain using exponential gain and log2 discount."""
    if k <= 0:
        raise ValueError("k must be positive")
    values = [float(value) for value in relevances[:k]]
    if any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("relevances must be finite and non-negative")
    return sum((2.0**value - 1.0) / math.log2(rank + 2.0) for rank, value in enumerate(values))


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    """NDCG@k; a query with no relevant item receives zero."""
    observed = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted((float(value) for value in relevances), reverse=True), k)
    return observed / ideal if ideal > 0 else 0.0


def recall_at_k(relevances: Sequence[float], k: int, total_relevant: int | None = None) -> float:
    """Binary Recall@k with an optional known relevant-item denominator."""
    if k <= 0:
        raise ValueError("k must be positive")
    labels = [float(value) for value in relevances]
    if any(value not in (0.0, 1.0) for value in labels):
        raise ValueError("recall expects binary relevance labels")
    denominator = sum(labels) if total_relevant is None else total_relevant
    if denominator < 0 or int(denominator) != denominator:
        raise ValueError("total_relevant must be a non-negative integer")
    if denominator == 0:
        return 0.0
    return sum(labels[:k]) / float(denominator)


def percentile(values: Sequence[float], probability: float) -> float:
    """Linear-interpolated percentile matching the registered bootstrap output."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("values must be finite")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class PairedObservation:
    date: str
    user_id: str
    seed: int
    treatment: float
    baseline: float


@dataclass(frozen=True)
class PairedScoreRow:
    """One immutable date-user-seed row containing every registered variant."""

    date: str
    user_id: str
    seed: int
    scores: Mapping[str, float]


def load_paired_score_rows(path: Path) -> list[PairedScoreRow]:
    """Load and strictly validate the canonical JSONL primary-metric evidence."""
    rows: list[PairedScoreRow] = []
    seen: set[tuple[str, str, int]] = set()
    expected_dates = set(EXPECTED_SPLIT["test"])
    expected_seeds = set(EXPECTED_SEEDS)
    expected_variants = set(EXPECTED_VARIANTS)
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"blank JSONL record at line {line_number}")
            try:
                value: Any = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc.msg}") from exc
            if not isinstance(value, dict) or set(value) != {"date", "user_id", "seed", "scores"}:
                raise ValueError(f"line {line_number} must contain exactly date, user_id, seed, and scores")
            date = value["date"]
            user_id = value["user_id"]
            seed = value["seed"]
            scores = value["scores"]
            if date not in expected_dates:
                raise ValueError(f"unexpected date at line {line_number}: {date}")
            if not isinstance(user_id, str) or not user_id:
                raise ValueError(f"user_id must be non-empty at line {line_number}")
            if seed not in expected_seeds:
                raise ValueError(f"unexpected seed at line {line_number}: {seed}")
            if not isinstance(scores, dict) or set(scores) != expected_variants:
                raise ValueError(f"line {line_number} must contain exactly the eight registered variant scores")
            normalized: dict[str, float] = {}
            for variant, score in scores.items():
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError(f"non-finite score for {variant} at line {line_number}")
                if not 0.0 <= score <= 1.0:
                    raise ValueError(f"score outside [0, 1] for {variant} at line {line_number}")
                normalized[variant] = float(score)
            key = (date, user_id, seed)
            if key in seen:
                raise ValueError(f"duplicate date-user-seed at line {line_number}: {date}/{user_id}/{seed}")
            seen.add(key)
            rows.append(PairedScoreRow(date=date, user_id=user_id, seed=seed, scores=normalized))
    if not rows:
        raise ValueError("paired score evidence must not be empty")
    return rows


def registered_primary_contrasts(
    rows: Sequence[PairedScoreRow],
    *,
    best_single_variant: str,
    resamples: int = 10_000,
    seed: int = 20_260_925,
) -> dict[str, dict[str, object]]:
    """Recompute all four preregistered primary contrasts from paired rows."""
    if best_single_variant not in {"ptd_item", "ptd_node"}:
        raise ValueError("best_single_variant must be ptd_item or ptd_node")
    results: dict[str, dict[str, object]] = {}
    for name, (numerator, registered_denominator) in PRIMARY_CONTRASTS.items():
        denominator = best_single_variant if registered_denominator == "best_single_validation_selected" else registered_denominator
        observations = [
            PairedObservation(
                date=row.date,
                user_id=row.user_id,
                seed=row.seed,
                treatment=row.scores[numerator],
                baseline=row.scores[denominator],
            )
            for row in rows
        ]
        statistics = date_stratified_paired_bootstrap(observations, resamples=resamples, seed=seed)
        results[name] = {
            "numerator": numerator,
            "denominator": registered_denominator,
            "resolved_denominator": denominator,
            "metric": "purchase_ndcg_at_50",
            **statistics,
        }
    adjusted = holm_adjust({name: float(result["raw_p"]) for name, result in results.items()})
    for name, value in adjusted.items():
        results[name]["holm_adjusted_p"] = value
    return results


def primary_metric_summaries(rows: Sequence[PairedScoreRow]) -> dict[str, dict[str, object]]:
    """Aggregate purchase NDCG@50 exactly as required by the evaluation JSON."""
    if not rows:
        raise ValueError("paired score evidence must not be empty")
    summaries: dict[str, dict[str, object]] = {}
    for variant in EXPECTED_VARIANTS:
        by_seed = {
            str(seed): fmean(row.scores[variant] for row in rows if row.seed == seed)
            for seed in EXPECTED_SEEDS
        }
        by_date = {
            date: fmean(row.scores[variant] for row in rows if row.date == date)
            for date in EXPECTED_SPLIT["test"]
        }
        summaries[variant] = {
            "n_users": len({row.user_id for row in rows}),
            "metrics": fmean(row.scores[variant] for row in rows),
            "by_seed": by_seed,
            "by_date": by_date,
        }
    return summaries


def paired_unit_deltas(
    observations: Iterable[PairedObservation],
    *,
    expected_dates: Sequence[str] = tuple(EXPECTED_SPLIT["test"]),
    expected_seeds: Sequence[int] = tuple(EXPECTED_SEEDS),
) -> dict[str, list[float]]:
    """Average seed-paired deltas for each date-user unit, rejecting incomplete pairs."""
    expected_date_set = set(expected_dates)
    expected_seed_set = set(expected_seeds)
    if len(expected_date_set) != len(expected_dates) or len(expected_seed_set) != len(expected_seeds):
        raise ValueError("expected dates and seeds must be unique")
    grouped: dict[tuple[str, str], dict[int, float]] = {}
    for observation in observations:
        if observation.date not in expected_date_set:
            raise ValueError(f"unexpected date: {observation.date}")
        if observation.seed not in expected_seed_set:
            raise ValueError(f"unexpected seed: {observation.seed}")
        if not math.isfinite(observation.treatment) or not math.isfinite(observation.baseline):
            raise ValueError("paired observations must be finite")
        unit = grouped.setdefault((observation.date, observation.user_id), {})
        if observation.seed in unit:
            raise ValueError(f"duplicate date-user-seed: {observation.date}/{observation.user_id}/{observation.seed}")
        unit[observation.seed] = observation.treatment - observation.baseline
    if not grouped:
        raise ValueError("observations must not be empty")

    deltas = {date: [] for date in expected_dates}
    for (date, user_id), seed_deltas in sorted(grouped.items()):
        if set(seed_deltas) != expected_seed_set:
            raise ValueError(f"incomplete seed pairing for {date}/{user_id}")
        deltas[date].append(fmean(seed_deltas[seed] for seed in expected_seeds))
    missing_dates = [date for date, values in deltas.items() if not values]
    if missing_dates:
        raise ValueError(f"missing date strata: {missing_dates}")
    return deltas


def date_stratified_paired_bootstrap(
    observations: Iterable[PairedObservation],
    *,
    resamples: int = 10_000,
    seed: int = 20_260_925,
) -> dict[str, float | int | list[float]]:
    """Bootstrap date-user units within date after averaging the three paired seeds."""
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    strata = paired_unit_deltas(observations)
    observed = [delta for date in EXPECTED_SPLIT["test"] for delta in strata[date]]
    point_estimate = fmean(observed)
    generator = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        draw: list[float] = []
        for date in EXPECTED_SPLIT["test"]:
            values = strata[date]
            draw.extend(values[generator.randrange(len(values))] for _ in range(len(values)))
        samples.append(fmean(draw))
    lower = percentile(samples, 0.025)
    upper = percentile(samples, 0.975)
    non_positive = (sum(value <= 0.0 for value in samples) + 1.0) / (resamples + 1.0)
    non_negative = (sum(value >= 0.0 for value in samples) + 1.0) / (resamples + 1.0)
    raw_p = min(1.0, 2.0 * min(non_positive, non_negative))
    return {
        "mean_delta": point_estimate,
        "ci95": [lower, upper],
        "raw_p": raw_p,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "paired_units": len(observed),
    }


def holm_adjust(raw_p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down family-wise error adjustment with deterministic tie-breaking."""
    if not raw_p_values:
        raise ValueError("raw_p_values must not be empty")
    for name, value in raw_p_values.items():
        if isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid p-value for {name}")
    ordered = sorted(raw_p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def guardrail_report(treatment: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, object]:
    """Evaluate the four preregistered relative guardrails."""
    rules = {
        "click_ndcg_at_50": ("floor", 0.95),
        "category_coverage_at_50": ("floor", 0.95),
        "max_category_share_at_50": ("ceiling", 1.05),
        "latency_p95_ms": ("ceiling", 1.20),
    }
    checks: dict[str, dict[str, float | bool | str]] = {}
    for metric, (direction, threshold) in rules.items():
        if metric not in treatment or metric not in baseline:
            raise KeyError(f"missing guardrail metric: {metric}")
        candidate = float(treatment[metric])
        reference = float(baseline[metric])
        if candidate < 0 or reference < 0 or not math.isfinite(candidate) or not math.isfinite(reference):
            raise ValueError(f"guardrail metric {metric} must be finite and non-negative")
        if reference == 0.0:
            ratio = 1.0 if candidate == 0.0 else math.inf
        else:
            ratio = candidate / reference
        passed = ratio >= threshold if direction == "floor" else ratio <= threshold
        checks[metric] = {
            "direction": direction,
            "threshold": threshold,
            "ratio": ratio,
            "passed": passed,
        }
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}
