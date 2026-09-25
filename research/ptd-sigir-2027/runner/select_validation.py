#!/usr/bin/env python3
"""Select preregistered PTD hyperparameters and the better single-level variant."""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from reference.evidence_gate import EXPECTED_SPLIT
from runner.materialize_teacher_scores import sha256
from runner.retrieval_runner import (
    CatalogTree,
    RetrievalQuery,
    SiblingScoringBackend,
    TorchSiblingBackend,
    beam_retrieve,
    load_queries,
    retrieval_metrics,
)

SELECTION_SEED = 16630
TEMPERATURES = (1.0, 2.0, 4.0)
LAMBDAS = (0.1, 0.3, 1.0)
COMBINED_VARIANT = "ptd_combined"
SINGLE_VARIANTS = ("ptd_item", "ptd_node")
SELECTION_METRIC = "purchase_ndcg_at_50"


class BackendFactory(Protocol):
    def __call__(
        self, fit: Mapping[str, Any], tree: CatalogTree, device: str
    ) -> SiblingScoringBackend: ...


def _default_backend_factory(
    fit: Mapping[str, Any], tree: CatalogTree, device: str
) -> SiblingScoringBackend:
    del tree
    return TorchSiblingBackend(
        Path(fit["checkpoint"]["path"]),
        variant=str(fit["variant"]),
        seed=int(fit["seed"]),
        device=device,
    )


def _load_fit(path: Path, *, allow_test_only_fits: bool) -> dict[str, Any]:
    fit = json.loads(path.read_text())
    if fit.get("contract_version") != "ptd-single-fit/v1" or fit.get("status") != "complete":
        raise ValueError("single-fit manifest contract/status mismatch")
    if fit.get("variant") not in {COMBINED_VARIANT, *SINGLE_VARIANTS}:
        raise ValueError("validation selection received an unsupported fit variant")
    if fit.get("seed") != SELECTION_SEED:
        raise ValueError("validation selection must use seed 16630")
    if fit.get("test_only_config_override") and not allow_test_only_fits:
        raise ValueError("production validation selection rejects test-only model configs")
    if not all(fit.get("checks", {}).values()):
        raise ValueError("single-fit manifest checks are incomplete")
    configuration = fit.get("configuration", {})
    key = (
        float(configuration.get("temperature", math.nan)),
        float(configuration.get("lambda_item", math.nan)),
        float(configuration.get("lambda_node", math.nan)),
    )
    if key[0] not in TEMPERATURES or key[1] not in LAMBDAS or key[2] not in LAMBDAS:
        raise ValueError("single-fit hyperparameters are outside the registered grid")
    checkpoint = fit.get("checkpoint", {})
    checkpoint_path = Path(checkpoint.get("path", ""))
    if not checkpoint_path.is_file() or sha256(checkpoint_path) != checkpoint.get("sha256"):
        raise ValueError("single-fit checkpoint path/hash mismatch")
    return {
        **fit,
        "manifest_path": str(path),
        "manifest_sha256": sha256(path),
        "selection_key": key,
    }


def _score_fit(
    fit: Mapping[str, Any],
    *,
    tree: CatalogTree,
    queries: Sequence[RetrievalQuery],
    device: str,
    backend_factory: BackendFactory,
) -> tuple[float, int]:
    backend = backend_factory(fit, tree, device)
    values: list[float] = []
    for query in queries:
        if not any(candidate.purchase_label == 1 for candidate in query.candidates.values()):
            continue
        result = beam_retrieve(query, tree, backend)
        values.append(
            retrieval_metrics(query, result, latency_ms=0.0)["purchase_ndcg_at_50"]
        )
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("validation selection has no finite purchase-positive scores")
    return sum(values) / len(values), len(values)


def _select_grid_record(grid_scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the registered near-tie rule to a complete scored 27-cell grid."""
    if len(grid_scores) != 27:
        raise ValueError("grid selection requires exactly 27 scored cells")
    observed = {
        (
            float(value["temperature"]),
            float(value["lambda_item"]),
            float(value["lambda_node"]),
        )
        for value in grid_scores
    }
    if observed != set(itertools.product(TEMPERATURES, LAMBDAS, LAMBDAS)):
        raise ValueError("scored grid does not cover the registered 27 cells")
    metrics = [float(value[SELECTION_METRIC]) for value in grid_scores]
    if any(not math.isfinite(value) for value in metrics):
        raise ValueError("grid selection metrics must be finite")
    best_metric = max(metrics)
    tied_grid = [
        value
        for value in grid_scores
        if best_metric - float(value[SELECTION_METRIC]) <= 0.001
    ]
    return dict(
        min(
            tied_grid,
            key=lambda value: (
                float(value["lambda_item"]) + float(value["lambda_node"]),
                float(value["temperature"]),
                float(value["lambda_item"]),
                float(value["lambda_node"]),
            ),
        )
    )


def preselect_combined_grid(
    *,
    validation_queries_manifest_path: Path,
    catalog_path: Path,
    date_eligibility_path: Path,
    fit_manifest_paths: Sequence[Path],
    device: str,
    backend_factory: BackendFactory | None = None,
    allow_test_only_fits: bool = False,
) -> dict[str, Any]:
    """Score only the combined grid so single-level fits can use its winner."""
    query_manifest = json.loads(validation_queries_manifest_path.read_text())
    if (
        query_manifest.get("contract_version") != "ptd-validation-queries/v1"
        or query_manifest.get("status") != "complete"
        or not all(query_manifest.get("checks", {}).values())
        or query_manifest.get("dates") != EXPECTED_SPLIT["validation"]
    ):
        raise ValueError("validation-query manifest is incomplete")
    query_artifact = query_manifest.get("output", {})
    query_path = Path(query_artifact.get("path", ""))
    if not query_path.is_file() or sha256(query_path) != query_artifact.get("sha256"):
        raise ValueError("validation-query path/hash mismatch")
    fits = [
        _load_fit(path, allow_test_only_fits=allow_test_only_fits)
        for path in fit_manifest_paths
    ]
    if len(fits) != 27 or any(fit["variant"] != COMBINED_VARIANT for fit in fits):
        raise ValueError("grid preselection accepts exactly 27 combined fits")
    if {fit["selection_key"] for fit in fits} != set(
        itertools.product(TEMPERATURES, LAMBDAS, LAMBDAS)
    ):
        raise ValueError("combined PTD fits must cover the exact 27-cell grid")
    tree = CatalogTree(
        catalog_path,
        date_eligibility_path,
        depth=13,
        required_dates=EXPECTED_SPLIT["validation"],
    )
    queries = load_queries(query_path, allowed_dates=EXPECTED_SPLIT["validation"])
    for query in queries:
        tree.validate_query(query)
    factory = backend_factory or _default_backend_factory
    grid_scores = []
    positive_units: int | None = None
    for fit in sorted(fits, key=lambda value: value["selection_key"]):
        metric, units = _score_fit(
            fit,
            tree=tree,
            queries=queries,
            device=device,
            backend_factory=factory,
        )
        if positive_units is None:
            positive_units = units
        elif units != positive_units:
            raise ValueError("validation fits used different purchase-positive units")
        grid_scores.append(
            {
                "temperature": fit["selection_key"][0],
                "lambda_item": fit["selection_key"][1],
                "lambda_node": fit["selection_key"][2],
                SELECTION_METRIC: metric,
                "fit_manifest": {
                    "path": fit["manifest_path"],
                    "sha256": fit["manifest_sha256"],
                },
                "checkpoint_sha256": fit["checkpoint"]["sha256"],
            }
        )
    selected = _select_grid_record(grid_scores)
    return {
        "grid_scores": grid_scores,
        "selected_key": (
            float(selected["temperature"]),
            float(selected["lambda_item"]),
            float(selected["lambda_node"]),
        ),
        "purchase_positive_units": positive_units,
    }


def select_validation(
    *,
    validation_queries_manifest_path: Path,
    catalog_path: Path,
    date_eligibility_path: Path,
    fit_manifest_paths: Sequence[Path],
    output_path: Path,
    device: str,
    backend_factory: BackendFactory | None = None,
    allow_test_only_fits: bool = False,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    query_manifest = json.loads(validation_queries_manifest_path.read_text())
    if (
        query_manifest.get("contract_version") != "ptd-validation-queries/v1"
        or query_manifest.get("status") != "complete"
        or not all(query_manifest.get("checks", {}).values())
    ):
        raise ValueError("validation-query manifest is incomplete")
    if query_manifest.get("dates") != EXPECTED_SPLIT["validation"]:
        raise ValueError("validation-query manifest date mismatch")
    query_artifact = query_manifest.get("output", {})
    query_path = Path(query_artifact.get("path", ""))
    if not query_path.is_file() or sha256(query_path) != query_artifact.get("sha256"):
        raise ValueError("validation-query path/hash mismatch")
    if not fit_manifest_paths:
        raise ValueError("validation selection needs fit manifests")
    fits = [
        _load_fit(path, allow_test_only_fits=allow_test_only_fits)
        for path in fit_manifest_paths
    ]
    identity = [(fit["variant"], *fit["selection_key"]) for fit in fits]
    if len(set(identity)) != len(identity):
        raise ValueError("validation selection has duplicate fit identities")
    combined = [fit for fit in fits if fit["variant"] == COMBINED_VARIANT]
    expected_grid = set(itertools.product(TEMPERATURES, LAMBDAS, LAMBDAS))
    if {fit["selection_key"] for fit in combined} != expected_grid:
        raise ValueError("combined PTD fits must cover the exact 27-cell grid")

    tree = CatalogTree(
        catalog_path,
        date_eligibility_path,
        depth=13,
        required_dates=EXPECTED_SPLIT["validation"],
    )
    queries = load_queries(query_path, allowed_dates=EXPECTED_SPLIT["validation"])
    for query in queries:
        tree.validate_query(query)
    factory = backend_factory or _default_backend_factory
    grid_scores = []
    positive_units: int | None = None
    for fit in sorted(combined, key=lambda value: value["selection_key"]):
        metric, units = _score_fit(
            fit,
            tree=tree,
            queries=queries,
            device=device,
            backend_factory=factory,
        )
        if positive_units is None:
            positive_units = units
        elif units != positive_units:
            raise ValueError("validation fits used different purchase-positive units")
        grid_scores.append(
            {
                "temperature": fit["selection_key"][0],
                "lambda_item": fit["selection_key"][1],
                "lambda_node": fit["selection_key"][2],
                "purchase_ndcg_at_50": metric,
                "fit_manifest": {
                    "path": fit["manifest_path"],
                    "sha256": fit["manifest_sha256"],
                },
                "checkpoint_sha256": fit["checkpoint"]["sha256"],
            }
        )
    selected_grid = _select_grid_record(grid_scores)
    selected_key = (
        selected_grid["temperature"],
        selected_grid["lambda_item"],
        selected_grid["lambda_node"],
    )
    singles = [
        fit
        for fit in fits
        if fit["variant"] in SINGLE_VARIANTS and fit["selection_key"] == selected_key
    ]
    if {fit["variant"] for fit in singles} != set(SINGLE_VARIANTS) or len(singles) != 2:
        raise ValueError("item/node fits must both use the selected hyperparameters")
    single_scores = []
    for fit in sorted(singles, key=lambda value: value["variant"]):
        metric, units = _score_fit(
            fit,
            tree=tree,
            queries=queries,
            device=device,
            backend_factory=factory,
        )
        if units != positive_units:
            raise ValueError("single-level fits used different validation units")
        single_scores.append(
            {
                "variant": fit["variant"],
                "purchase_ndcg_at_50": metric,
                "fit_manifest": {
                    "path": fit["manifest_path"],
                    "sha256": fit["manifest_sha256"],
                },
                "checkpoint_sha256": fit["checkpoint"]["sha256"],
            }
        )
    selected_single = min(
        single_scores,
        key=lambda value: (-value["purchase_ndcg_at_50"], value["variant"]),
    )
    output = {
        "contract_version": "ptd-validation-selection/v1",
        "status": "complete",
        "selection_seed": SELECTION_SEED,
        "selection_dates": EXPECTED_SPLIT["validation"],
        "selection_metric": SELECTION_METRIC,
        "selection_population": "purchase_positive_date_user_units",
        "purchase_positive_units": positive_units,
        "validation_queries_manifest_sha256": sha256(
            validation_queries_manifest_path
        ),
        "validation_queries_sha256": query_artifact["sha256"],
        "catalog_sha256": sha256(catalog_path),
        "date_eligibility_sha256": sha256(date_eligibility_path),
        "grid_scores": grid_scores,
        "selected_hyperparameters": {
            "temperature": selected_key[0],
            "lambda_item": selected_key[1],
            "lambda_node": selected_key[2],
            "tie_window_absolute_ndcg": 0.001,
            "tie_break": (
                "within_0.001_of_best_then_lambda_sum_temperature_"
                "lambda_item_lambda_node_asc"
            ),
        },
        "single_variant_scores": single_scores,
        "best_single_variant": selected_single["variant"],
        "single_variant_tie_break": "metric_desc_then_variant_lexical_asc",
        "checks": {
            "exact_27_cell_combined_grid": True,
            "selection_seed_16630_only": True,
            "validation_date_only": True,
            "purchase_positive_macro": True,
            "item_and_node_use_selected_hyperparameters": True,
            "teacher_oracle_not_used": True,
            "test_queries_not_read": True,
            "deterministic_tie_break": True,
            "no_overwrite": True,
        },
        "scope_note": (
            "Validation-only model selection. No test query, efficacy estimate, or latency "
            "measurement is admitted by this artifact."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        staged = Path(directory) / "validation_selection.json"
        staged.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        os.replace(staged, output_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-queries-manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--date-eligibility", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, action="append", default=[])
    parser.add_argument(
        "--fit-manifest-glob",
        action="append",
        default=[],
        help="Local fit-manifest glob; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    fit_manifests = sorted(
        {
            *args.fit_manifest,
            *(Path(value) for pattern in args.fit_manifest_glob for value in glob.glob(pattern)),
        }
    )
    output = select_validation(
        validation_queries_manifest_path=args.validation_queries_manifest,
        catalog_path=args.catalog,
        date_eligibility_path=args.date_eligibility,
        fit_manifest_paths=fit_manifests,
        output_path=args.output,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "selected_hyperparameters": output["selected_hyperparameters"],
                "best_single_variant": output["best_single_variant"],
            }
        )
    )


if __name__ == "__main__":
    main()
