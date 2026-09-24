#!/usr/bin/env python3
"""Materialize frozen ESMM row scores without reading labels or overwriting outputs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FEATURE_COLUMNS = (
    "conversion_score",
    "cosine_similarity",
    "old_score",
    "candidate_rank",
    "repeat_click_match",
    "repeat_purchase_match",
    "is_new_user",
    "is_new_product",
)
KEY_COLUMNS = ("snapshot_id", "snapshot_date", "user_id", "product_id", "candidate_rank")
OUTPUT_COLUMNS = KEY_COLUMNS + ("teacher_pctr", "teacher_pcvr", "teacher_purchase")
EXPECTED_CHECKPOINT_SHA256 = "5a435e4ea2579ca226f26fd8dfa5ad48a7be016f3d1a8e61798ce1b2d6ed1540"
EXPECTED_SOURCE_REVISION = "f065bd20db568ee440569e65627b1598256bd079"
EXPECTED_TRIAL_CONFIG_SHA256 = "2ac2e44f4b37b19737fb0a4472a93c48bca1e922411ccd7828fe069d89a642b5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_checkpoint_metadata(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Validate the self-describing teacher checkpoint except for tensor values."""
    if payload.get("contract_version") != "shared_bottom_esmm_v2":
        raise ValueError("teacher checkpoint contract mismatch")
    if payload.get("source_revision") != EXPECTED_SOURCE_REVISION:
        raise ValueError("teacher source revision mismatch")
    if payload.get("trial_config_sha256") != EXPECTED_TRIAL_CONFIG_SHA256:
        raise ValueError("teacher trial hash mismatch")
    if payload.get("checkpoint_metric") != "valid_loss" or payload.get("checkpoint_epoch") != 4:
        raise ValueError("teacher selection checkpoint mismatch")
    if payload.get("seed") != 16630:
        raise ValueError("teacher seed mismatch")
    trial = payload.get("trial")
    if not isinstance(trial, Mapping):
        raise ValueError("teacher trial metadata is missing")
    if tuple(trial.get("shared_hidden_sizes", ())) != (32, 32):
        raise ValueError("teacher shared architecture mismatch")
    if tuple(trial.get("tower_hidden_sizes", ())) != (16,):
        raise ValueError("teacher tower architecture mismatch")
    stats = payload.get("feature_stats")
    if not isinstance(stats, Mapping) or tuple(stats.get("columns", ())) != FEATURE_COLUMNS:
        raise ValueError("teacher feature order mismatch")
    mean = np.asarray(stats.get("mean"), dtype=np.float32)
    scale = np.asarray(stats.get("scale"), dtype=np.float32)
    if mean.shape != (len(FEATURE_COLUMNS),) or scale.shape != mean.shape:
        raise ValueError("teacher feature-stat shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("teacher feature stats must be finite with positive scales")
    return mean, scale


def normalized_feature_matrix(
    columns: Mapping[str, Sequence[Any]],
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Apply the frozen train-only transform in the checkpoint's exact feature order."""
    if tuple(columns) != FEATURE_COLUMNS:
        raise ValueError("feature mapping must preserve the exact ESMM feature order")
    lengths = {len(columns[name]) for name in FEATURE_COLUMNS}
    if len(lengths) != 1:
        raise ValueError("feature columns have different lengths")
    raw = np.column_stack(
        [
            np.asarray(
                [0.0 if value is None else float(value) for value in columns[name]],
                dtype=np.float32,
            )
            for name in FEATURE_COLUMNS
        ]
    )
    transformed = (raw - mean.astype(np.float32, copy=False)) / scale.astype(np.float32, copy=False)
    return np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )


def _model_from_checkpoint(path: Path, device: str):
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("PyTorch is required to materialize teacher scores") from exc

    if sha256(path) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("teacher checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mean, scale = validate_checkpoint_metadata(payload)

    def mlp(input_size: int, hidden_sizes: tuple[int, ...], output_size: int | None = None):
        layers: list[nn.Module] = []
        previous = input_size
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(previous, hidden), nn.ReLU()))
            previous = hidden
        if output_size is not None:
            layers.append(nn.Linear(previous, output_size))
        return nn.Sequential(*layers)

    class FrozenESMM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = mlp(8, (32, 32))
            self.ctr_tower = mlp(32, (16,), 1)
            self.cvr_tower = mlp(32, (16,), 1)

        def forward(self, features):
            shared = self.shared(features)
            pctr = torch.sigmoid(self.ctr_tower(shared).squeeze(-1))
            pcvr = torch.sigmoid(self.cvr_tower(shared).squeeze(-1))
            return pctr, pcvr, pctr * pcvr

    model = FrozenESMM()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(torch.device(device)), mean, scale, torch


def _feature_columns(batch: pa.RecordBatch) -> dict[str, list[Any]]:
    return {name: batch.column(batch.schema.get_field_index(name)).to_pylist() for name in FEATURE_COLUMNS}


def _key_arrays(batch: pa.RecordBatch) -> list[pa.Array]:
    arrays: list[pa.Array] = []
    for name in KEY_COLUMNS:
        index = batch.schema.get_field_index(name)
        if index < 0:
            raise ValueError(f"input batch is missing key column {name}")
        arrays.append(batch.column(index))
    return arrays


def materialize(
    input_paths: Sequence[Path],
    checkpoint: Path,
    output: Path,
    manifest_path: Path,
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Score fixed local Parquet inputs and write one immutable keyed output."""
    if not input_paths:
        raise ValueError("at least one input parquet is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for path in (output, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    model, mean, scale, torch = _model_from_checkpoint(checkpoint, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("snapshot_id", pa.string(), nullable=False),
            pa.field("snapshot_date", pa.date32(), nullable=False),
            pa.field("user_id", pa.string(), nullable=False),
            pa.field("product_id", pa.int64(), nullable=False),
            pa.field("candidate_rank", pa.int64(), nullable=False),
            pa.field("teacher_pctr", pa.float32(), nullable=False),
            pa.field("teacher_pcvr", pa.float32(), nullable=False),
            pa.field("teacher_purchase", pa.float32(), nullable=False),
        ]
    )
    rows = 0
    rows_by_date: Counter[str] = Counter()
    source_files = []
    with pq.ParquetWriter(output, schema=schema, compression="zstd") as writer:
        for input_path in input_paths:
            parquet = pq.ParquetFile(input_path)
            required = set(KEY_COLUMNS) | set(FEATURE_COLUMNS)
            missing = required - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{input_path} missing columns: {sorted(missing)}")
            source_files.append(
                {"path": str(input_path), "sha256": sha256(input_path), "rows": parquet.metadata.num_rows}
            )
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=list(dict.fromkeys(KEY_COLUMNS + FEATURE_COLUMNS)),
                use_threads=True,
            ):
                features = normalized_feature_matrix(_feature_columns(batch), mean, scale)
                with torch.inference_mode():
                    pctr, pcvr, purchase = model(torch.from_numpy(features).to(device))
                arrays = [
                    *[pc.cast(value, schema[index].type) for index, value in enumerate(_key_arrays(batch))],
                    pa.array(pctr.detach().cpu().numpy(), type=pa.float32()),
                    pa.array(pcvr.detach().cpu().numpy(), type=pa.float32()),
                    pa.array(purchase.detach().cpu().numpy(), type=pa.float32()),
                ]
                scored = pa.RecordBatch.from_arrays(arrays, schema=schema)
                purchase_values = scored.column(scored.schema.get_field_index("teacher_purchase")).to_numpy()
                if not np.isfinite(purchase_values).all() or np.any((purchase_values < 0) | (purchase_values > 1)):
                    raise ValueError("teacher produced invalid purchase probabilities")
                for value in scored.column(scored.schema.get_field_index("snapshot_date")).to_pylist():
                    rows_by_date[value.isoformat()] += 1
                writer.write_batch(scored)
                rows += scored.num_rows
    if rows != sum(item["rows"] for item in source_files):
        raise ValueError("teacher score output row count differs from inputs")
    manifest = {
        "contract_version": "ptd-frozen-teacher-scores/v1",
        "status": "complete",
        "teacher_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "teacher_contract": "pCTR*pCVR",
        "feature_columns": list(FEATURE_COLUMNS),
        "key_columns": list(KEY_COLUMNS),
        "label_columns_read": [],
        "device": device,
        "rows": rows,
        "rows_by_date": dict(sorted(rows_by_date.items())),
        "source_files": source_files,
        "output": {"path": str(output), "sha256": sha256(output)},
        "checks": {
            "checkpoint_hash_match": True,
            "feature_order_match": True,
            "all_scores_finite_and_bounded": True,
            "row_count_match": True,
            "labels_read": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_paths(patterns: Iterable[str]) -> list[Path]:
    paths = sorted({Path(value) for pattern in patterns for value in glob.glob(pattern)})
    if not paths:
        raise ValueError("input patterns matched no local files")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Local Parquet glob; repeatable")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=65_536)
    args = parser.parse_args()
    manifest = materialize(
        parse_paths(args.input),
        args.checkpoint,
        args.output,
        args.manifest,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps({"status": "complete", "rows": manifest["rows"], "output": manifest["output"]}))


if __name__ == "__main__":
    main()
