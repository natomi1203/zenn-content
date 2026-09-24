#!/usr/bin/env python3
"""Run and validation-select cycles 0..3 for one alternating variant and seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from reference.evidence_gate import EXPECTED_SEEDS, EXPECTED_SPLIT
from runner.alternating_solver import build_alternating_bundle
from runner.build_training_examples import build_training_examples
from runner.materialize_assignment_weights import materialize_assignment_weights
from runner.materialize_teacher_scores import sha256
from runner.ptd_model import PTDModelConfig
from runner.retrieval_runner import (
    CatalogTree,
    TorchSiblingBackend,
    beam_retrieve,
    load_queries,
    retrieval_metrics,
)
from runner.train_ptd import train_from_examples

ALTERNATING_VARIANTS = ("alternating_tdm", "alternating_ptd")
MAXIMUM_CYCLES = 3


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def _validation_score(
    *,
    validation_queries_manifest_path: Path,
    catalog_path: Path,
    date_eligibility_path: Path,
    fit_manifest_path: Path,
    device: str,
) -> tuple[float, int]:
    query_manifest = json.loads(validation_queries_manifest_path.read_text())
    if (
        query_manifest.get("contract_version") != "ptd-validation-queries/v1"
        or query_manifest.get("status") != "complete"
        or query_manifest.get("dates") != EXPECTED_SPLIT["validation"]
        or not all(query_manifest.get("checks", {}).values())
    ):
        raise ValueError("validation-query manifest is incomplete")
    query_artifact = query_manifest.get("output", {})
    query_path = Path(query_artifact.get("path", ""))
    if not query_path.is_file() or sha256(query_path) != query_artifact.get("sha256"):
        raise ValueError("validation-query path/hash mismatch")
    fit = json.loads(fit_manifest_path.read_text())
    checkpoint = fit.get("checkpoint", {})
    checkpoint_path = Path(checkpoint.get("path", ""))
    if (
        fit.get("contract_version") != "ptd-single-fit/v1"
        or fit.get("status") != "complete"
        or not all(fit.get("checks", {}).values())
        or not checkpoint_path.is_file()
        or sha256(checkpoint_path) != checkpoint.get("sha256")
    ):
        raise ValueError("cycle fit manifest/checkpoint is incomplete")
    tree = CatalogTree(
        catalog_path,
        date_eligibility_path,
        depth=13,
        required_dates=EXPECTED_SPLIT["validation"],
    )
    queries = load_queries(query_path, allowed_dates=EXPECTED_SPLIT["validation"])
    backend = TorchSiblingBackend(
        checkpoint_path,
        variant=str(fit["variant"]),
        seed=int(fit["seed"]),
        device=device,
    )
    values = []
    for query in queries:
        tree.validate_query(query)
        if not any(candidate.purchase_label == 1 for candidate in query.candidates.values()):
            continue
        result = beam_retrieve(query, tree, backend)
        values.append(
            retrieval_metrics(query, result, latency_ms=0.0)["purchase_ndcg_at_50"]
        )
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("cycle validation has no finite purchase-positive metric")
    return sum(values) / len(values), len(values)


def _selected_bundle_sha256(record: dict[str, Any]) -> str:
    fields = (
        record["catalog"]["sha256"],
        record["date_eligibility"]["sha256"],
        record["fit_manifest"]["sha256"],
        record["checkpoint"]["sha256"],
    )
    return hashlib.sha256(("\n".join(fields) + "\n").encode()).hexdigest()


def run_alternating_cycles(
    *,
    teacher_manifest_path: Path,
    assignment_queries_manifest_path: Path,
    validation_queries_manifest_path: Path,
    initial_catalog_path: Path,
    initial_date_eligibility_path: Path,
    variant: str,
    seed: int,
    temperature: float,
    lambda_item: float,
    lambda_node: float,
    output_dir: Path,
    device: str,
    model_config: PTDModelConfig | None = None,
    test_only_config_override: bool = False,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if variant not in ALTERNATING_VARIANTS:
        raise ValueError("alternating orchestrator received a non-alternating variant")
    if seed not in EXPECTED_SEEDS:
        raise ValueError("alternating orchestrator received an unregistered seed")
    for path, label in (
        (teacher_manifest_path, "teacher manifest"),
        (assignment_queries_manifest_path, "assignment-query manifest"),
        (validation_queries_manifest_path, "validation-query manifest"),
        (initial_catalog_path, "initial catalog"),
        (initial_date_eligibility_path, "initial date eligibility"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing")
    output_dir.mkdir(parents=True)
    current_catalog = initial_catalog_path
    current_eligibility = initial_date_eligibility_path
    previous_fit_manifest: Path | None = None
    cycles: list[dict[str, Any]] = []

    for cycle in range(MAXIMUM_CYCLES + 1):
        cycle_dir = output_dir / f"cycle-{cycle}"
        cycle_dir.mkdir()
        assignment_weights = None
        tree_manifest_artifact = None
        if cycle > 0:
            weights_path = cycle_dir / "assignment_weights.parquet"
            weights_manifest_path = cycle_dir / "assignment_weights_manifest.json"
            materialize_assignment_weights(
                assignment_queries_manifest_path=assignment_queries_manifest_path,
                catalog_path=current_catalog,
                date_eligibility_path=current_eligibility,
                fit_manifest_path=previous_fit_manifest,
                output_path=weights_path,
                output_manifest_path=weights_manifest_path,
                device=device,
                allow_test_only_fit=test_only_config_override,
            )
            tree_dir = cycle_dir / "tree"
            build_alternating_bundle(
                catalog_path=current_catalog,
                date_eligibility_path=current_eligibility,
                weights_path=weights_path,
                weight_manifest_path=weights_manifest_path,
                output_dir=tree_dir,
                depth=13,
            )
            assignment_weights = {
                "weights": _artifact(weights_path),
                "manifest": _artifact(weights_manifest_path),
            }
            tree_manifest_path = tree_dir / "manifest.json"
            tree_manifest_artifact = _artifact(tree_manifest_path)
            current_catalog = tree_dir / "alternating_catalog.parquet"
            current_eligibility = tree_dir / "alternating_date_eligibility.parquet"

        examples_path = cycle_dir / "training_examples.parquet"
        examples_manifest_path = cycle_dir / "training_examples_manifest.json"
        build_training_examples(
            teacher_manifest_path=teacher_manifest_path,
            catalog_path=current_catalog,
            date_eligibility_path=current_eligibility,
            output_path=examples_path,
            output_manifest_path=examples_manifest_path,
        )
        checkpoint_path = cycle_dir / "checkpoint.pt"
        fit_manifest_path = cycle_dir / "fit_manifest.json"
        fit = train_from_examples(
            examples_manifest_path=examples_manifest_path,
            variant=variant,
            seed=seed,
            temperature=temperature,
            lambda_item=lambda_item,
            lambda_node=lambda_node,
            checkpoint_path=checkpoint_path,
            output_manifest_path=fit_manifest_path,
            device=device,
            initial_fit_manifest_path=previous_fit_manifest,
            config=model_config,
            test_only_config_override=test_only_config_override,
        )
        metric, units = _validation_score(
            validation_queries_manifest_path=validation_queries_manifest_path,
            catalog_path=current_catalog,
            date_eligibility_path=current_eligibility,
            fit_manifest_path=fit_manifest_path,
            device=device,
        )
        cycles.append(
            {
                "cycle": cycle,
                "warm_started_from_cycle": None if cycle == 0 else cycle - 1,
                "catalog": _artifact(current_catalog),
                "date_eligibility": _artifact(current_eligibility),
                "tree_manifest": tree_manifest_artifact,
                "assignment_weights": assignment_weights,
                "training_examples_manifest": _artifact(examples_manifest_path),
                "fit_manifest": _artifact(fit_manifest_path),
                "checkpoint": _artifact(checkpoint_path),
                "state_sha256": fit["state_sha256"],
                "validation_purchase_ndcg_at_50": metric,
                "validation_purchase_positive_units": units,
            }
        )
        previous_fit_manifest = fit_manifest_path

    selected = min(
        cycles,
        key=lambda record: (-record["validation_purchase_ndcg_at_50"], record["cycle"]),
    )
    manifest = {
        "contract_version": "ptd-alternating-cycle-selection/v1",
        "status": "complete",
        "variant": variant,
        "seed": seed,
        "maximum_cycles": MAXIMUM_CYCLES,
        "selected_cycle": selected["cycle"],
        "selection_metric": "purchase_ndcg_at_50",
        "selection_population": "purchase_positive_date_user_units",
        "cycle_tie_break": "metric_desc_then_cycle_asc",
        "hyperparameters": {
            "temperature": temperature,
            "lambda_item": lambda_item,
            "lambda_node": lambda_node,
        },
        "teacher_manifest_sha256": sha256(teacher_manifest_path),
        "assignment_queries_manifest_sha256": sha256(
            assignment_queries_manifest_path
        ),
        "validation_queries_manifest_sha256": sha256(
            validation_queries_manifest_path
        ),
        "initial_catalog_sha256": sha256(initial_catalog_path),
        "initial_date_eligibility_sha256": sha256(
            initial_date_eligibility_path
        ),
        "cycles": cycles,
        "selected": selected,
        "selected_bundle_sha256": _selected_bundle_sha256(selected),
        "checks": {
            "all_four_cycles_complete": len(cycles) == 4,
            "cycle_zero_fixed_tree": cycles[0]["tree_manifest"] is None,
            "cycles_one_to_three_reassigned": all(
                record["tree_manifest"] is not None for record in cycles[1:]
            ),
            "model_parameters_warm_started": all(
                record["warm_started_from_cycle"] == record["cycle"] - 1
                for record in cycles[1:]
            ),
            "optimizers_reset_each_fit": True,
            "validation_only_cycle_selection": True,
            "lower_cycle_exact_tie_break": True,
            "test_queries_not_read": True,
            "selected_bundle_locked": True,
            "no_overwrite": True,
        },
        "scope_note": (
            "Validation-only cycle selection for one alternating variant/seed. No test "
            "query, efficacy estimate, or production latency is read or emitted."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--assignment-queries-manifest", type=Path, required=True)
    parser.add_argument("--validation-queries-manifest", type=Path, required=True)
    parser.add_argument("--initial-catalog", type=Path, required=True)
    parser.add_argument("--initial-date-eligibility", type=Path, required=True)
    parser.add_argument("--variant", choices=ALTERNATING_VARIANTS, required=True)
    parser.add_argument("--seed", choices=EXPECTED_SEEDS, type=int, required=True)
    parser.add_argument("--temperature", choices=(1.0, 2.0, 4.0), type=float, required=True)
    parser.add_argument("--lambda-item", choices=(0.1, 0.3, 1.0), type=float, required=True)
    parser.add_argument("--lambda-node", choices=(0.1, 0.3, 1.0), type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    output = run_alternating_cycles(
        teacher_manifest_path=args.teacher_manifest,
        assignment_queries_manifest_path=args.assignment_queries_manifest,
        validation_queries_manifest_path=args.validation_queries_manifest,
        initial_catalog_path=args.initial_catalog,
        initial_date_eligibility_path=args.initial_date_eligibility,
        variant=args.variant,
        seed=args.seed,
        temperature=args.temperature,
        lambda_item=args.lambda_item,
        lambda_node=args.lambda_node,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "variant": output["variant"],
                "seed": output["seed"],
                "selected_cycle": output["selected_cycle"],
                "selected_bundle_sha256": output["selected_bundle_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
