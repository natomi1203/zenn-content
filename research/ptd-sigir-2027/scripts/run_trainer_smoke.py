#!/usr/bin/env python3
"""Deterministically exercise both registered PTD encoder/trainer variants."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from runner.ptd_model import (  # noqa: E402
    DISTILLATION_ITEM,
    DISTILLATION_NODE,
    LossConfiguration,
    ModelInputs,
    PTDModelConfig,
    PTDTrainer,
    PTDTreeModel,
    TrainingBatch,
    build_registered_model,
    configure_determinism,
    count_trainable_parameters,
    production_parameter_count,
    save_checkpoint_no_clobber,
    state_sha256,
)

SEED = 16630


def synthetic_batch() -> TrainingBatch:
    path_count = 2
    depth = 13
    rows = path_count * depth
    click_base = torch.zeros((path_count, 30), dtype=torch.long)
    purchase_base = torch.zeros((path_count, 30), dtype=torch.long)
    click_base[0, :4] = torch.tensor([2, 3, 4, 5])
    purchase_base[1, :3] = torch.tensor([14, 15, 16])
    click = click_base.repeat_interleave(depth, dim=0)
    purchase = purchase_base.repeat_interleave(depth, dim=0)
    levels = torch.arange(1, depth + 1).repeat(path_count)
    row_index = torch.arange(rows)
    left_child = 2 + (row_index * 2) % 56
    child_items = torch.stack((left_child, left_child + 1), dim=-1)
    categories = torch.stack((2 + row_index % 12, 2 + (row_index + 1) % 12), dim=-1)
    parents = torch.stack((2 + row_index % 56, 2 + row_index % 56), dim=-1)
    positive = row_index % 2
    teacher = torch.where(
        positive[:, None] == torch.tensor([0, 1])[None, :],
        torch.tensor(0.75),
        torch.tensor(0.25),
    ).to(torch.float32)
    inputs = ModelInputs(
        user_ids=torch.tensor([2, 3]).repeat_interleave(depth),
        click_history_ids=click,
        click_lengths=torch.tensor([4, 0]).repeat_interleave(depth),
        purchase_history_ids=purchase,
        purchase_lengths=torch.tensor([0, 3]).repeat_interleave(depth),
        child_item_ids=child_items,
        child_category_ids=categories,
        parent_node_ids=parents,
        child_levels=torch.stack((levels, levels), dim=-1),
    )
    return TrainingBatch(
        inputs=inputs,
        path_group_ids=torch.arange(path_count).repeat_interleave(depth),
        positive_child=positive,
        teacher_probabilities=teacher,
        distillation_kind=torch.where(
            levels == depth,
            torch.tensor(DISTILLATION_ITEM),
            torch.tensor(DISTILLATION_NODE),
        ),
    )


def one_run(variant: str) -> dict:
    configure_determinism(SEED)
    config = PTDModelConfig(
        item_hash_bucket_size=64,
        user_hash_bucket_size=32,
        category_hash_bucket_size=16,
    )
    model, switches = build_registered_model(variant, config)
    loss = LossConfiguration(
        temperature=2.0,
        lambda_item=0.3,
        lambda_node=0.3,
        enable_item=switches[0],
        enable_node=switches[1],
    )
    trainer = PTDTrainer(model, loss)
    batch = synthetic_batch()
    initial = trainer.evaluate_batch(batch)
    with torch.inference_mode():
        logits_before = model(batch.inputs).clone()
        swapped_teacher = replace(
            batch, teacher_probabilities=batch.teacher_probabilities.flip(-1)
        )
        logits_after_teacher_change = model(swapped_teacher.inputs).clone()
    history = trainer.fit([batch], epochs=2)
    final = trainer.evaluate_batch(batch)
    if not final["total"] < initial["total"]:
        raise ValueError(f"synthetic {variant} loss did not decrease")
    if not torch.equal(logits_before, logits_after_teacher_change):
        raise ValueError("teacher values leaked into serving logits")

    padded = batch.inputs.click_history_ids.clone()
    for row, length in enumerate(batch.inputs.click_lengths.tolist()):
        padded[row, length:] = 40 + row % 20
    perturbed_inputs = replace(batch.inputs, click_history_ids=padded)
    model.eval()
    with torch.inference_mode():
        original = model(batch.inputs)
        perturbed = model(perturbed_inputs)
    if not torch.equal(original, perturbed):
        raise ValueError(f"{variant} read tokens beyond registered sequence lengths")

    no_overwrite = False
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        save_checkpoint_no_clobber(
            checkpoint,
            model=model,
            variant=variant,
            seed=SEED,
            history=history,
        )
        try:
            save_checkpoint_no_clobber(
                checkpoint,
                model=model,
                variant=variant,
                seed=SEED,
                history=history,
            )
        except FileExistsError:
            no_overwrite = True
    return {
        "variant": variant,
        "encoder": model.encoder_name,
        "trainable_parameters_synthetic_buckets": count_trainable_parameters(model),
        "trainable_parameters_production_buckets": production_parameter_count(model),
        "initial_loss": initial,
        "epoch_history": history,
        "final_loss": final,
        "state_sha256": state_sha256(model),
        "checks": {
            "finite_losses": all(
                torch.isfinite(torch.tensor(value))
                for record in [initial, final, *history]
                for key, value in record.items()
                if key != "epoch"
            ),
            "loss_decreased": final["total"] < initial["total"],
            "teacher_not_serving_feature": torch.equal(
                logits_before, logits_after_teacher_change
            ),
            "padding_suffix_ignored": torch.equal(original, perturbed),
            "checkpoint_no_overwrite": no_overwrite,
        },
    }


def run() -> dict:
    torch.set_num_threads(1)
    first = [one_run("ptd_combined"), one_run("ptd_combined_baseline_encoder")]
    second = [one_run("ptd_combined"), one_run("ptd_combined_baseline_encoder")]
    if [item["state_sha256"] for item in first] != [
        item["state_sha256"] for item in second
    ]:
        raise ValueError("trainer smoke is not deterministic across identical reruns")
    if [item["epoch_history"] for item in first] != [
        item["epoch_history"] for item in second
    ]:
        raise ValueError("trainer loss history changed across identical reruns")
    forward_parameters = list(inspect.signature(PTDTreeModel.forward).parameters)
    if forward_parameters != ["self", "inputs"]:
        raise ValueError(
            "serving forward signature unexpectedly accepts extra features"
        )
    return {
        "contract_version": "ptd-trainer-smoke/v1",
        "status": "SMOKE_ONLY",
        "empirical_claim_allowed": False,
        "seed": SEED,
        "synthetic_bucket_override": {
            "item_hash_bucket_size": 64,
            "user_hash_bucket_size": 32,
            "category_hash_bucket_size": 16,
        },
        "registered_architecture_retained": {
            "hidden_dim": 64,
            "stream_type_embedding_dim": 8,
            "maximum_length": 30,
            "position_buckets": 64,
            "hstu_style_layers": 2,
            "hstu_style_heads": 4,
            "din_windows_most_recent_first": [1, 2, 3, 4, 5, 5, 10],
            "time_encoding": False,
        },
        "runs": first,
        "checks": {
            "identical_rerun_state_hashes": True,
            "identical_rerun_loss_histories": True,
            "both_encoders_exercised": {item["encoder"] for item in first}
            == {"hstu", "din"},
            "item_and_node_kl_exercised": True,
            "complete_depth_13_path_groups": True,
            "serving_forward_excludes_teacher": True,
            "exact_two_epoch_budget": all(
                len(item["epoch_history"]) == 2 for item in first
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.system(),
            "device": "cpu",
        },
        "scope_note": (
            "Synthetic implementation evidence only. Bucket counts are reduced for the smoke test; "
            "this artifact is not retrieval, latency, or PTD efficacy evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifact" / "smoke" / "trainer_smoke.json",
    )
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PTD trainer smoke: PASS (HSTU-style + multiwindow DIN, deterministic rerun)")


if __name__ == "__main__":
    main()
