#!/usr/bin/env python3
"""Execute the locked 65-fit PTD pre-test schedule without opening test queries."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from reference.evidence_gate import EXPECTED_SEEDS
from runner.build_fit_schedule import ALTERNATING_VARIANTS, FIXED_TREE_VARIANTS
from runner.materialize_teacher_scores import sha256
from runner.select_validation import preselect_combined_grid, select_validation


def _train_from_examples(**kwargs):
    from runner.train_ptd import train_from_examples

    return train_from_examples(**kwargs)


def _run_alternating_cycles(**kwargs):
    from runner.run_alternating_cycles import run_alternating_cycles

    return run_alternating_cycles(**kwargs)


def _progress(stage: str, **details: Any) -> None:
    print(json.dumps({"stage": stage, **details}, sort_keys=True), flush=True)


def _artifact(path: Path, label: str) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return {"path": str(path), "sha256": sha256(path)}


def _checked_input(value: Any, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain path and sha256")
    path = Path(value["path"])
    if not path.is_file() or sha256(path) != value["sha256"]:
        raise ValueError(f"{label} path/hash mismatch")
    return path


def _load_schedule(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    schedule = json.loads(path.read_text())
    if (
        schedule.get("contract_version") != "ptd-fit-schedule/v1"
        or schedule.get("status") != "locked"
        or not all(schedule.get("checks", {}).values())
    ):
        raise ValueError("fit schedule is not a complete locked schedule")
    counts = schedule.get("counts", {})
    if counts != {
        "grid_fit_executions": 27,
        "single_level_fit_executions": 2,
        "additional_fixed_tree_fit_executions": 12,
        "alternating_chains": 6,
        "fits_per_alternating_chain": 4,
        "alternating_fit_executions": 24,
        "total_fit_executions": 65,
        "locked_tree_model_artifacts": 21,
        "retrieval_variant_seed_entries": 24,
    }:
        raise ValueError("fit schedule counts differ from the registered 65-fit DAG")
    expected_inputs = {
        "teacher_scores_manifest",
        "training_examples_manifest",
        "validation_queries_manifest",
        "assignment_queries_manifest",
        "fixed_catalog",
        "fixed_date_eligibility",
    }
    values = schedule.get("inputs")
    if not isinstance(values, dict) or set(values) != expected_inputs:
        raise ValueError("fit schedule input set is incomplete")
    inputs = {key: _checked_input(values[key], key) for key in expected_inputs}
    if len(schedule.get("grid_fits", [])) != 27:
        raise ValueError("fit schedule must contain 27 grid fits")
    if len(schedule.get("single_level_fits", [])) != 2:
        raise ValueError("fit schedule must contain two single-level fits")
    if len(schedule.get("final_fixed_tree_models", [])) != 15:
        raise ValueError("fit schedule must contain 15 final fixed-tree cells")
    if len(schedule.get("alternating_chains", [])) != 6:
        raise ValueError("fit schedule must contain six alternating chains")
    return schedule, inputs


def _fit_identity(path: Path, *, variant: str, seed: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("contract_version") != "ptd-single-fit/v1"
        or payload.get("status") != "complete"
        or payload.get("variant") != variant
        or payload.get("seed") != seed
        or payload.get("test_only_config_override") is not False
        or not all(payload.get("checks", {}).values())
    ):
        raise ValueError(f"fit output identity/checks mismatch: {path}")
    checkpoint = payload.get("checkpoint", {})
    checkpoint_path = Path(checkpoint.get("path", ""))
    if (
        not checkpoint_path.is_file()
        or sha256(checkpoint_path) != checkpoint.get("sha256")
    ):
        raise ValueError(f"fit checkpoint path/hash mismatch: {path}")
    return payload


def _train(
    *,
    examples_manifest: Path,
    task: Mapping[str, Any],
    hyperparameters: tuple[float, float, float],
    device: str,
) -> Path:
    outputs = task.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"checkpoint", "manifest"}:
        raise ValueError("fit task outputs are incomplete")
    checkpoint = Path(outputs["checkpoint"])
    manifest = Path(outputs["manifest"])
    if checkpoint.exists() or manifest.exists():
        if not checkpoint.is_file() or not manifest.is_file():
            raise ValueError("partial fit output cannot be resumed")
        _fit_identity(manifest, variant=str(task["variant"]), seed=int(task["seed"]))
        return manifest
    _train_from_examples(
        examples_manifest_path=examples_manifest,
        variant=str(task["variant"]),
        seed=int(task["seed"]),
        temperature=hyperparameters[0],
        lambda_item=hyperparameters[1],
        lambda_node=hyperparameters[2],
        checkpoint_path=checkpoint,
        output_manifest_path=manifest,
        device=device,
    )
    _fit_identity(manifest, variant=str(task["variant"]), seed=int(task["seed"]))
    return manifest


def run_fit_schedule(
    *,
    fit_schedule_path: Path,
    validation_selection_path: Path,
    output_path: Path,
    device: str,
) -> dict[str, Any]:
    """Execute selection, final fixed-tree fits, and six alternating chains."""
    if output_path.exists():
        raise FileExistsError("refusing to overwrite fit-execution output")
    schedule, inputs = _load_schedule(fit_schedule_path)
    examples_manifest = inputs["training_examples_manifest"]
    validation_manifest = inputs["validation_queries_manifest"]
    catalog = inputs["fixed_catalog"]
    eligibility = inputs["fixed_date_eligibility"]

    grid_manifests: list[Path] = []
    for task in schedule["grid_fits"]:
        hyperparameters = task.get("hyperparameters", {})
        grid_manifests.append(
            _train(
                examples_manifest=examples_manifest,
                task=task,
                hyperparameters=(
                    float(hyperparameters["temperature"]),
                    float(hyperparameters["lambda_item"]),
                    float(hyperparameters["lambda_node"]),
                ),
                device=device,
            )
        )
        _progress("grid_fit_complete", task_id=task["task_id"])

    preselection = preselect_combined_grid(
        validation_queries_manifest_path=validation_manifest,
        catalog_path=catalog,
        date_eligibility_path=eligibility,
        fit_manifest_paths=grid_manifests,
        device=device,
    )
    selected = tuple(float(value) for value in preselection["selected_key"])
    if len(selected) != 3:
        raise ValueError("grid preselection returned an invalid key")
    _progress(
        "grid_preselection_complete",
        temperature=selected[0],
        lambda_item=selected[1],
        lambda_node=selected[2],
    )

    single_manifests = []
    for task in schedule["single_level_fits"]:
        single_manifests.append(
            _train(
                examples_manifest=examples_manifest,
                task=task,
                hyperparameters=selected,
                device=device,
            )
        )
        _progress("single_level_fit_complete", variant=task["variant"])
    if validation_selection_path.exists():
        selection = json.loads(validation_selection_path.read_text())
        if (
            selection.get("contract_version") != "ptd-validation-selection/v1"
            or selection.get("status") != "complete"
            or not all(selection.get("checks", {}).values())
        ):
            raise ValueError("existing validation selection is incomplete")
    else:
        selection = select_validation(
            validation_queries_manifest_path=validation_manifest,
            catalog_path=catalog,
            date_eligibility_path=eligibility,
            fit_manifest_paths=[*grid_manifests, *single_manifests],
            output_path=validation_selection_path,
            device=device,
        )
    selected_payload = selection.get("selected_hyperparameters", {})
    confirmed = (
        float(selected_payload.get("temperature")),
        float(selected_payload.get("lambda_item")),
        float(selected_payload.get("lambda_node")),
    )
    if confirmed != selected:
        raise ValueError("full validation selection differs from grid preselection")
    _progress(
        "validation_selection_locked",
        best_single_variant=selection["best_single_variant"],
    )

    selected_combined = Path(
        next(
            value["fit_manifest"]["path"]
            for value in selection["grid_scores"]
            if (
                float(value["temperature"]),
                float(value["lambda_item"]),
                float(value["lambda_node"]),
            )
            == selected
        )
    )
    selected_singles = {
        value["variant"]: Path(value["fit_manifest"]["path"])
        for value in selection["single_variant_scores"]
    }
    final_manifests: dict[tuple[str, int], Path] = {}
    additional_fixed_fits = 0
    for task in schedule["final_fixed_tree_models"]:
        variant = str(task["variant"])
        seed = int(task["seed"])
        action = task["action"]
        if action == "reuse_selected_grid_fit":
            manifest = selected_combined
        elif action == "reuse_single_level_fit":
            manifest = selected_singles[variant]
        elif action == "run_selected_hyperparameters":
            directory = Path(task["output_dir"])
            manifest = _train(
                examples_manifest=examples_manifest,
                task={
                    "variant": variant,
                    "seed": seed,
                    "outputs": {
                        "checkpoint": str(directory / "checkpoint.pt"),
                        "manifest": str(directory / "fit_manifest.json"),
                    },
                },
                hyperparameters=selected,
                device=device,
            )
            additional_fixed_fits += 1
        else:
            raise ValueError(f"unsupported final fixed-tree action: {action}")
        _fit_identity(manifest, variant=variant, seed=seed)
        identity = (variant, seed)
        if identity in final_manifests:
            raise ValueError(f"duplicate final fit identity: {identity}")
        final_manifests[identity] = manifest
        _progress(
            "final_fixed_tree_cell_complete",
            variant=variant,
            seed=seed,
            action=action,
        )
    expected_fixed = {
        (variant, seed) for variant in FIXED_TREE_VARIANTS for seed in EXPECTED_SEEDS
    }
    if set(final_manifests) != expected_fixed or additional_fixed_fits != 12:
        raise ValueError("final fixed-tree matrix/count is incomplete")

    cycle_manifests: dict[tuple[str, int], Path] = {}
    for task in schedule["alternating_chains"]:
        variant = str(task["variant"])
        seed = int(task["seed"])
        output_dir = Path(task["output_dir"])
        manifest = output_dir / "manifest.json"
        if output_dir.exists():
            if not manifest.is_file():
                raise ValueError("partial alternating output cannot be resumed")
            cycle = json.loads(manifest.read_text())
        else:
            cycle = _run_alternating_cycles(
                teacher_manifest_path=inputs["teacher_scores_manifest"],
                assignment_queries_manifest_path=inputs[
                    "assignment_queries_manifest"
                ],
                validation_queries_manifest_path=validation_manifest,
                initial_catalog_path=catalog,
                initial_date_eligibility_path=eligibility,
                variant=variant,
                seed=seed,
                temperature=selected[0],
                lambda_item=selected[1],
                lambda_node=selected[2],
                output_dir=output_dir,
                device=device,
            )
        if (
            cycle.get("contract_version") != "ptd-alternating-cycle-selection/v1"
            or cycle.get("status") != "complete"
            or cycle.get("variant") != variant
            or cycle.get("seed") != seed
            or not all(cycle.get("checks", {}).values())
        ):
            raise ValueError("alternating cycle output identity/checks mismatch")
        cycle_manifests[(variant, seed)] = manifest
        _progress(
            "alternating_chain_complete",
            variant=variant,
            seed=seed,
            selected_cycle=cycle["selected_cycle"],
        )
    expected_cycles = {
        (variant, seed) for variant in ALTERNATING_VARIANTS for seed in EXPECTED_SEEDS
    }
    if set(cycle_manifests) != expected_cycles:
        raise ValueError("alternating cycle matrix is incomplete")

    payload = {
        "contract_version": "ptd-fit-execution/v1",
        "status": "complete",
        "code_revision": schedule["code_revision"],
        "fit_schedule": _artifact(fit_schedule_path, "fit schedule"),
        "validation_selection": _artifact(
            validation_selection_path, "validation selection"
        ),
        "selected_hyperparameters": {
            "temperature": selected[0],
            "lambda_item": selected[1],
            "lambda_node": selected[2],
        },
        "grid_fit_manifests": [
            _artifact(path, "grid fit manifest") for path in grid_manifests
        ],
        "single_level_fit_manifests": [
            _artifact(path, "single-level fit manifest") for path in single_manifests
        ],
        "final_fixed_tree_fit_manifests": [
            {
                "variant": variant,
                "seed": seed,
                **_artifact(final_manifests[(variant, seed)], "final fit manifest"),
            }
            for variant, seed in sorted(final_manifests)
        ],
        "alternating_cycle_manifests": [
            {
                "variant": variant,
                "seed": seed,
                **_artifact(cycle_manifests[(variant, seed)], "cycle manifest"),
            }
            for variant, seed in sorted(cycle_manifests)
        ],
        "counts": {
            "grid_fits": 27,
            "single_level_fits": 2,
            "additional_fixed_tree_fits": 12,
            "alternating_chains": 6,
            "fits_per_alternating_chain": 4,
            "total_fit_executions": 65,
        },
        "checks": {
            "fit_schedule_hash_verified": True,
            "grid_selected_before_single_level_fits": True,
            "full_validation_selection_matches_preselection": True,
            "exact_fixed_tree_matrix": True,
            "exact_alternating_matrix": True,
            "test_queries_not_opened": True,
            "complete_outputs_reusable_on_restart": True,
            "no_overwrite": True,
        },
        "scope_note": (
            "Complete pre-test fit execution only. Test queries, test outcomes, efficacy, "
            "and production latency are not opened or emitted by this driver."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        staged = Path(directory) / "fit-execution.json"
        staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(staged, output_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-schedule", type=Path, required=True)
    parser.add_argument("--validation-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = run_fit_schedule(
        fit_schedule_path=args.fit_schedule,
        validation_selection_path=args.validation_selection,
        output_path=args.output,
        device=args.device,
    )
    print(json.dumps({"status": result["status"], "counts": result["counts"]}))


if __name__ == "__main__":
    main()
