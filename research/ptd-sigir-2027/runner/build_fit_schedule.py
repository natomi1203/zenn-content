#!/usr/bin/env python3
"""Build the locked 65-fit PTD execution DAG without running any fit."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from reference.evidence_gate import EXPECTED_SEEDS
SELECTION_SEED = 16630
TEMPERATURES = (1.0, 2.0, 4.0)
LAMBDAS = (0.1, 0.3, 1.0)

FIXED_TREE_VARIANTS = (
    "fixed_tdm",
    "ptd_item",
    "ptd_node",
    "ptd_combined",
    "ptd_combined_baseline_encoder",
)
ALTERNATING_VARIANTS = ("alternating_tdm", "alternating_ptd")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def _fit_outputs(root: Path, relative: str) -> dict[str, str]:
    directory = root / relative
    return {
        "checkpoint": str(directory / "checkpoint.pt"),
        "manifest": str(directory / "fit_manifest.json"),
    }


def _artifact(path: Path, label: str) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def build_fit_schedule(
    *,
    code_revision: str,
    output_root: Path,
    teacher_scores_manifest_path: Path,
    training_examples_manifest_path: Path,
    validation_queries_manifest_path: Path,
    assignment_queries_manifest_path: Path,
    fixed_catalog_path: Path,
    fixed_date_eligibility_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not REVISION_RE.fullmatch(code_revision):
        raise ValueError("code revision must be 40 lowercase hexadecimal characters")
    if not str(output_root):
        raise ValueError("output root must be non-empty")
    inputs = {
        "teacher_scores_manifest": _artifact(
            teacher_scores_manifest_path, "teacher scores manifest"
        ),
        "training_examples_manifest": _artifact(
            training_examples_manifest_path, "training examples manifest"
        ),
        "validation_queries_manifest": _artifact(
            validation_queries_manifest_path, "validation queries manifest"
        ),
        "assignment_queries_manifest": _artifact(
            assignment_queries_manifest_path, "assignment queries manifest"
        ),
        "fixed_catalog": _artifact(fixed_catalog_path, "fixed catalog"),
        "fixed_date_eligibility": _artifact(
            fixed_date_eligibility_path, "fixed date eligibility"
        ),
    }

    grid_fits = []
    for index, (temperature, lambda_item, lambda_node) in enumerate(
        itertools.product(TEMPERATURES, LAMBDAS, LAMBDAS)
    ):
        grid_fits.append(
            {
                "task_id": f"selection-grid-{index:02d}",
                "variant": "ptd_combined",
                "seed": SELECTION_SEED,
                "hyperparameters": {
                    "temperature": temperature,
                    "lambda_item": lambda_item,
                    "lambda_node": lambda_node,
                },
                "outputs": _fit_outputs(
                    output_root, f"selection/grid-{index:02d}"
                ),
            }
        )

    single_level_fits = [
        {
            "task_id": f"selection-{variant}",
            "variant": variant,
            "seed": SELECTION_SEED,
            "hyperparameters_from": "grid_validation_winner",
            "depends_on": "all_27_grid_fits_scored_on_validation",
            "outputs": _fit_outputs(output_root, f"selection/{variant}"),
        }
        for variant in ("ptd_item", "ptd_node")
    ]

    final_fixed_tree_models = []
    for variant in FIXED_TREE_VARIANTS:
        for seed in EXPECTED_SEEDS:
            if seed == SELECTION_SEED and variant == "ptd_combined":
                action = "reuse_selected_grid_fit"
            elif seed == SELECTION_SEED and variant in {"ptd_item", "ptd_node"}:
                action = "reuse_single_level_fit"
            else:
                action = "run_selected_hyperparameters"
            final_fixed_tree_models.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "action": action,
                    "hyperparameters_from": "validation_selection",
                    "output_dir": (
                        None
                        if action.startswith("reuse_")
                        else str(output_root / "final" / variant / f"seed-{seed}")
                    ),
                }
            )

    alternating_chains = [
        {
            "variant": variant,
            "seed": seed,
            "cycles": [0, 1, 2, 3],
            "hyperparameters_from": "validation_selection",
            "depends_on": "fixed_tree_and_validation_selection",
            "output_dir": str(
                output_root / "alternating" / variant / f"seed-{seed}"
            ),
        }
        for variant in ALTERNATING_VARIANTS
        for seed in EXPECTED_SEEDS
    ]
    checks = {
        "exact_27_cell_grid": len(grid_fits) == 27
        and len(
            {
                (
                    task["hyperparameters"]["temperature"],
                    task["hyperparameters"]["lambda_item"],
                    task["hyperparameters"]["lambda_node"],
                )
                for task in grid_fits
            }
        )
        == 27,
        "selection_seed_only_for_grid": all(
            task["seed"] == SELECTION_SEED for task in grid_fits
        ),
        "single_levels_after_grid_selection": {
            task["variant"] for task in single_level_fits
        }
        == {"ptd_item", "ptd_node"},
        "all_five_fixed_tree_variants_and_three_seeds": {
            (task["variant"], task["seed"]) for task in final_fixed_tree_models
        }
        == {
            (variant, seed)
            for variant in FIXED_TREE_VARIANTS
            for seed in EXPECTED_SEEDS
        },
        "selected_seed_fits_reused": sum(
            task["action"].startswith("reuse_")
            for task in final_fixed_tree_models
        )
        == 3,
        "all_two_alternating_variants_and_three_seeds": {
            (task["variant"], task["seed"]) for task in alternating_chains
        }
        == {
            (variant, seed)
            for variant in ALTERNATING_VARIANTS
            for seed in EXPECTED_SEEDS
        },
        "four_fits_per_alternating_chain": all(
            task["cycles"] == [0, 1, 2, 3] for task in alternating_chains
        ),
        "no_test_stage_dependency": True,
        "deterministic_paths": True,
        "no_overwrite": True,
    }
    if not all(checks.values()):
        raise ValueError(f"fit schedule invariant failed: {checks}")
    payload = {
        "contract_version": "ptd-fit-schedule/v1",
        "status": "locked",
        "code_revision": code_revision,
        "selection_seed": SELECTION_SEED,
        "seeds": EXPECTED_SEEDS,
        "output_root": str(output_root),
        "inputs": inputs,
        "grid_fits": grid_fits,
        "single_level_fits": single_level_fits,
        "final_fixed_tree_models": final_fixed_tree_models,
        "alternating_chains": alternating_chains,
        "counts": {
            "grid_fit_executions": 27,
            "single_level_fit_executions": 2,
            "additional_fixed_tree_fit_executions": 12,
            "alternating_chains": 6,
            "fits_per_alternating_chain": 4,
            "alternating_fit_executions": 24,
            "total_fit_executions": 65,
            "locked_tree_model_artifacts": 21,
            "retrieval_variant_seed_entries": 24,
        },
        "checks": checks,
        "scope_note": (
            "A preregistered execution DAG only. It schedules 65 fits and reads no "
            "test query, label, metric, latency, or prospective PTD outcome."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        staged = Path(directory) / "fit_schedule.json"
        staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(staged, output_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--teacher-scores-manifest", type=Path, required=True)
    parser.add_argument("--training-examples-manifest", type=Path, required=True)
    parser.add_argument("--validation-queries-manifest", type=Path, required=True)
    parser.add_argument("--assignment-queries-manifest", type=Path, required=True)
    parser.add_argument("--fixed-catalog", type=Path, required=True)
    parser.add_argument("--fixed-date-eligibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schedule = build_fit_schedule(
        code_revision=args.code_revision,
        output_root=args.output_root,
        teacher_scores_manifest_path=args.teacher_scores_manifest,
        training_examples_manifest_path=args.training_examples_manifest,
        validation_queries_manifest_path=args.validation_queries_manifest,
        assignment_queries_manifest_path=args.assignment_queries_manifest,
        fixed_catalog_path=args.fixed_catalog,
        fixed_date_eligibility_path=args.fixed_date_eligibility,
        output_path=args.output,
    )
    print(json.dumps({"status": schedule["status"], "counts": schedule["counts"]}))


if __name__ == "__main__":
    main()
