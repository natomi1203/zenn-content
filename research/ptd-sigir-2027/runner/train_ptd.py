#!/usr/bin/env python3
"""Train one registered PTD variant/seed from immutable depth-13 examples."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import torch

from reference.evidence_gate import EXPECTED_SEEDS, EXPECTED_SPLIT
from runner.materialize_teacher_scores import sha256
from runner.ptd_model import (
    LossConfiguration,
    ModelInputs,
    PTDModelConfig,
    PTDTrainer,
    TrainingBatch,
    build_registered_model,
    configure_determinism,
    count_trainable_parameters,
    parse_most_recent_first_history,
    save_checkpoint_no_clobber,
    stable_hash_bucket,
    state_sha256,
)

REQUIRED_COLUMNS = {
    "snapshot_date",
    "snapshot_id",
    "user_id",
    "positive_product_id",
    "path_group_id",
    "level",
    "positive_child",
    "click_seq_product_id",
    "purchase_seq_product_id",
    "left_item_key",
    "right_item_key",
    "left_category",
    "right_category",
    "parent_node_id",
    "teacher_left",
    "teacher_right",
    "distillation_kind",
}
TRAINABLE_VARIANTS = (
    "fixed_tdm",
    "ptd_item",
    "ptd_node",
    "ptd_combined",
    "alternating_tdm",
    "alternating_ptd",
    "ptd_combined_baseline_encoder",
)


def _date(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _group_rows(path: Path) -> list[list[dict[str, Any]]]:
    table = pq.read_table(path)
    if missing := REQUIRED_COLUMNS - set(table.column_names):
        raise ValueError(f"training examples are missing columns: {sorted(missing)}")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in table.to_pylist():
        grouped[int(row["path_group_id"])].append(row)
    if not grouped:
        raise ValueError("training examples contain no path groups")
    output: list[list[dict[str, Any]]] = []
    for group_id in sorted(grouped):
        rows = sorted(grouped[group_id], key=lambda row: int(row["level"]))
        if [int(row["level"]) for row in rows] != list(range(1, 14)):
            raise ValueError("every training path group must contain levels 1 through 13")
        invariant_fields = (
            "snapshot_date",
            "snapshot_id",
            "user_id",
            "positive_product_id",
            "click_seq_product_id",
            "purchase_seq_product_id",
        )
        for field in invariant_fields:
            if len({str(row[field]) for row in rows}) != 1:
                raise ValueError(f"{field} differs within one training path")
        output.append(rows)
    return output


def _batch(
    path_groups: Sequence[Sequence[Mapping[str, Any]]], config: PTDModelConfig
) -> TrainingBatch:
    user_ids: list[int] = []
    click_ids: list[list[int]] = []
    click_lengths: list[int] = []
    purchase_ids: list[list[int]] = []
    purchase_lengths: list[int] = []
    child_items: list[list[int]] = []
    child_categories: list[list[int]] = []
    parent_nodes: list[list[int]] = []
    child_levels: list[list[int]] = []
    path_ids: list[int] = []
    positive_children: list[int] = []
    teacher: list[list[float]] = []
    kinds: list[int] = []
    for local_group_id, rows in enumerate(path_groups):
        first = rows[0]
        click, click_length = parse_most_recent_first_history(
            str(first["click_seq_product_id"]),
            bucket_size=config.item_hash_bucket_size,
            maximum_length=config.maximum_length,
        )
        purchase, purchase_length = parse_most_recent_first_history(
            str(first["purchase_seq_product_id"]),
            bucket_size=config.item_hash_bucket_size,
            maximum_length=config.maximum_length,
        )
        for row in rows:
            user_ids.append(
                stable_hash_bucket(first["user_id"], config.user_hash_bucket_size)
            )
            click_ids.append(click)
            click_lengths.append(click_length)
            purchase_ids.append(purchase)
            purchase_lengths.append(purchase_length)
            child_items.append(
                [
                    stable_hash_bucket(row["left_item_key"], config.item_hash_bucket_size),
                    stable_hash_bucket(row["right_item_key"], config.item_hash_bucket_size),
                ]
            )
            child_categories.append(
                [
                    stable_hash_bucket(row["left_category"], config.category_hash_bucket_size),
                    stable_hash_bucket(row["right_category"], config.category_hash_bucket_size),
                ]
            )
            parent_key = stable_hash_bucket(
                f"node:{int(row['parent_node_id'])}", config.item_hash_bucket_size
            )
            parent_nodes.append([parent_key, parent_key])
            level = int(row["level"])
            child_levels.append([level, level])
            path_ids.append(local_group_id)
            positive_children.append(int(row["positive_child"]))
            probabilities = [float(row["teacher_left"]), float(row["teacher_right"])]
            if not all(math.isfinite(value) for value in probabilities):
                raise ValueError("teacher probabilities must be finite")
            teacher.append(probabilities)
            kinds.append(int(row["distillation_kind"]))
    batch = TrainingBatch(
        inputs=ModelInputs(
            user_ids=torch.tensor(user_ids, dtype=torch.long),
            click_history_ids=torch.tensor(click_ids, dtype=torch.long),
            click_lengths=torch.tensor(click_lengths, dtype=torch.long),
            purchase_history_ids=torch.tensor(purchase_ids, dtype=torch.long),
            purchase_lengths=torch.tensor(purchase_lengths, dtype=torch.long),
            child_item_ids=torch.tensor(child_items, dtype=torch.long),
            child_category_ids=torch.tensor(child_categories, dtype=torch.long),
            parent_node_ids=torch.tensor(parent_nodes, dtype=torch.long),
            child_levels=torch.tensor(child_levels, dtype=torch.long),
        ),
        path_group_ids=torch.tensor(path_ids, dtype=torch.long),
        positive_child=torch.tensor(positive_children, dtype=torch.long),
        teacher_probabilities=torch.tensor(teacher, dtype=torch.float32),
        distillation_kind=torch.tensor(kinds, dtype=torch.long),
    )
    batch.validate(config.maximum_length)
    return batch


def load_batches(
    path: Path, *, config: PTDModelConfig, paths_per_batch: int
) -> tuple[list[TrainingBatch], list[TrainingBatch], dict[str, int]]:
    if paths_per_batch != 64:
        raise ValueError("the registered batch size is exactly 64 paths")
    groups = _group_rows(path)
    train_dates = set(EXPECTED_SPLIT["train"])
    validation_dates = set(EXPECTED_SPLIT["validation"])
    train_groups = [group for group in groups if _date(group[0]["snapshot_date"]) in train_dates]
    validation_groups = [
        group for group in groups if _date(group[0]["snapshot_date"]) in validation_dates
    ]
    admitted = train_dates | validation_dates
    if any(_date(group[0]["snapshot_date"]) not in admitted for group in groups):
        raise ValueError("training examples contain a test or unregistered date")
    if not train_groups or not validation_groups:
        raise ValueError("training and validation must both contain purchase-positive paths")

    def batches(values: list[list[dict[str, Any]]]) -> list[TrainingBatch]:
        return [
            _batch(values[index : index + paths_per_batch], config)
            for index in range(0, len(values), paths_per_batch)
        ]

    return batches(train_groups), batches(validation_groups), {
        "train_paths": len(train_groups),
        "validation_paths": len(validation_groups),
        "train_sibling_rows": len(train_groups) * 13,
        "validation_sibling_rows": len(validation_groups) * 13,
    }


def _weighted_validation_loss(
    trainer: PTDTrainer, batches: Sequence[TrainingBatch]
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    paths = 0
    for batch in batches:
        batch_paths = len(torch.unique(batch.path_group_ids))
        values = trainer.evaluate_batch(batch)
        for name, value in values.items():
            totals[name] += value * batch_paths
        paths += batch_paths
    return {name: value / paths for name, value in totals.items()}


def train_from_examples(
    *,
    examples_manifest_path: Path,
    variant: str,
    seed: int,
    temperature: float,
    lambda_item: float,
    lambda_node: float,
    checkpoint_path: Path,
    output_manifest_path: Path,
    device: str,
    config: PTDModelConfig | None = None,
    test_only_config_override: bool = False,
) -> dict[str, Any]:
    """Fit one immutable model; validation NDCG selection remains a later gate."""
    if checkpoint_path.exists() or output_manifest_path.exists():
        raise FileExistsError("refusing to overwrite PTD training outputs")
    if variant not in TRAINABLE_VARIANTS:
        raise ValueError("variant is not a registered trainable PTD variant")
    if seed not in EXPECTED_SEEDS:
        raise ValueError("seed is not registered")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    manifest = json.loads(examples_manifest_path.read_text())
    if manifest.get("contract_version") != "ptd-training-examples/v1":
        raise ValueError("training-example manifest contract mismatch")
    if manifest.get("status") != "complete" or not all(manifest.get("checks", {}).values()):
        raise ValueError("training-example manifest is incomplete")
    artifact = manifest.get("output", {})
    examples_path = Path(artifact.get("path", ""))
    if not examples_path.is_file() or sha256(examples_path) != artifact.get("sha256"):
        raise ValueError("training-example path/hash mismatch")
    model_config = config or PTDModelConfig()
    if model_config != PTDModelConfig() and not test_only_config_override:
        raise ValueError("production training forbids bucket-size overrides")

    configure_determinism(seed)
    model, switches = build_registered_model(variant, model_config)
    loss_configuration = LossConfiguration(
        temperature=temperature,
        lambda_item=lambda_item,
        lambda_node=lambda_node,
        enable_item=switches[0],
        enable_node=switches[1],
    )
    train_batches, validation_batches, counts = load_batches(
        examples_path, config=model_config, paths_per_batch=64
    )
    trainer = PTDTrainer(model, loss_configuration, device=device)
    history = trainer.fit(train_batches, epochs=2)
    validation_loss = _weighted_validation_loss(trainer, validation_batches)
    if not all(
        math.isfinite(value)
        for record in [*history, validation_loss]
        for name, value in record.items()
        if name != "epoch"
    ):
        raise ValueError("training produced a non-finite loss")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        staged_checkpoint = Path(directory) / "checkpoint.pt"
        save_checkpoint_no_clobber(
            staged_checkpoint,
            model=model,
            variant=variant,
            seed=seed,
            history=history,
        )
        checkpoint_hash = sha256(staged_checkpoint)
        output = {
            "contract_version": "ptd-single-fit/v1",
            "status": "complete",
            "variant": variant,
            "seed": seed,
            "device": device,
            "training_examples_manifest_sha256": sha256(examples_manifest_path),
            "training_examples_sha256": artifact["sha256"],
            "configuration": {
                "temperature": temperature,
                "lambda_item": lambda_item,
                "lambda_node": lambda_node,
                "item_distillation_enabled": switches[0],
                "node_distillation_enabled": switches[1],
                "batch_size_paths": 64,
                "epochs": 2,
            },
            "model_config": model_config.__dict__,
            "test_only_config_override": test_only_config_override,
            "counts": counts,
            "train_history": history,
            "validation_loss": validation_loss,
            "trainable_parameters": count_trainable_parameters(model),
            "state_sha256": state_sha256(model),
            "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_hash},
            "checks": {
                "registered_variant": True,
                "registered_seed": True,
                "exact_two_epoch_budget": len(history) == 2,
                "finite_losses": True,
                "teacher_not_serving_feature": True,
                "test_examples_absent": True,
                "checkpoint_no_overwrite": True,
            },
            "scope_note": (
                "A single train/validation fit only. Purchase-NDCG hyperparameter selection, "
                "alternating-tree cycles, test retrieval, latency, and efficacy remain pending."
            ),
        }
        staged_manifest = Path(directory) / "manifest.json"
        staged_manifest.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        os.replace(staged_checkpoint, checkpoint_path)
        os.replace(staged_manifest, output_manifest_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-manifest", type=Path, required=True)
    parser.add_argument("--variant", choices=TRAINABLE_VARIANTS, required=True)
    parser.add_argument("--seed", choices=EXPECTED_SEEDS, type=int, required=True)
    parser.add_argument("--temperature", choices=(1.0, 2.0, 4.0), type=float, required=True)
    parser.add_argument("--lambda-item", choices=(0.1, 0.3, 1.0), type=float, required=True)
    parser.add_argument("--lambda-node", choices=(0.1, 0.3, 1.0), type=float, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    output = train_from_examples(
        examples_manifest_path=args.examples_manifest,
        variant=args.variant,
        seed=args.seed,
        temperature=args.temperature,
        lambda_item=args.lambda_item,
        lambda_node=args.lambda_node,
        checkpoint_path=args.checkpoint,
        output_manifest_path=args.manifest,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "variant": output["variant"],
                "seed": output["seed"],
                "checkpoint": output["checkpoint"],
            }
        )
    )


if __name__ == "__main__":
    main()
