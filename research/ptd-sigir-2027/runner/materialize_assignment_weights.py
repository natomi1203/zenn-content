#!/usr/bin/env python3
"""Materialize exact train-only alternating assignment weights for every free leaf."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from reference.evidence_gate import EXPECTED_SEEDS, EXPECTED_SPLIT
from runner.alternating_solver import CANDIDATE_LEAF_POLICY, WEIGHT_FORMULA
from runner.materialize_teacher_scores import sha256
from runner.ptd_model import configure_determinism
from runner.retrieval_runner import (
    CatalogTree,
    RetrievalQuery,
    SiblingScoringBackend,
    TorchSiblingBackend,
    load_queries,
)

ALTERNATING_VARIANTS = ("alternating_tdm", "alternating_ptd")


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


def _load_fit(path: Path, *, allow_test_only_fit: bool) -> dict[str, Any]:
    fit = json.loads(path.read_text())
    if fit.get("contract_version") != "ptd-single-fit/v1" or fit.get("status") != "complete":
        raise ValueError("single-fit manifest contract/status mismatch")
    if fit.get("variant") not in ALTERNATING_VARIANTS:
        raise ValueError("assignment weights require an alternating variant")
    if fit.get("seed") not in EXPECTED_SEEDS:
        raise ValueError("assignment-weight fit uses an unregistered seed")
    if fit.get("test_only_config_override") and not allow_test_only_fit:
        raise ValueError("production assignment weights reject test-only configs")
    if not all(fit.get("checks", {}).values()):
        raise ValueError("single-fit manifest checks are incomplete")
    checkpoint = fit.get("checkpoint", {})
    checkpoint_path = Path(checkpoint.get("path", ""))
    if not checkpoint_path.is_file() or sha256(checkpoint_path) != checkpoint.get("sha256"):
        raise ValueError("single-fit checkpoint path/hash mismatch")
    return fit


def _catalog_assignment_axes(
    catalog_path: Path, *, depth: int
) -> tuple[list[int], list[int]]:
    table = pq.read_table(catalog_path)
    required = {"product_id", "leaf_node_id", "catalog_index", "train_seen"}
    if missing := required - set(table.column_names):
        raise ValueError(f"catalog is missing columns: {sorted(missing)}")
    rows = sorted(table.to_pylist(), key=lambda row: int(row["catalog_index"]))
    leaf_start = 2**depth - 1
    train_products = [int(row["product_id"]) for row in rows if bool(row["train_seen"])]
    anchored_leaves = {
        int(row["leaf_node_id"]) for row in rows if not bool(row["train_seen"])
    }
    available_leaves = [
        leaf
        for leaf in range(leaf_start, leaf_start + 2**depth)
        if leaf not in anchored_leaves
    ]
    if not train_products or len(set(train_products)) != len(train_products):
        raise ValueError("catalog train-product axis is empty or duplicated")
    if len(available_leaves) < len(train_products):
        raise ValueError("catalog has insufficient free leaves")
    return train_products, available_leaves


def _all_leaf_path_scores(
    query: RetrievalQuery,
    *,
    tree: CatalogTree,
    backend: SiblingScoringBackend,
    available_leaves: Sequence[int],
    score_chunk_size: int,
) -> list[float]:
    if score_chunk_size <= 0:
        raise ValueError("score_chunk_size must be positive")
    cumulative: dict[int, float] = {0: 0.0}
    for level in range(1, tree.depth + 1):
        parents = sorted(cumulative)
        pairs = [
            (
                tree.descriptor(2 * parent + 1, parent, level),
                tree.descriptor(2 * parent + 2, parent, level),
            )
            for parent in parents
        ]
        logits: list[tuple[float, float]] = []
        for start in range(0, len(pairs), score_chunk_size):
            logits.extend(
                backend.score_sibling_pairs(
                    user_id=query.user_id,
                    click_history_most_recent_first=query.click_history_most_recent_first,
                    purchase_history_most_recent_first=query.purchase_history_most_recent_first,
                    pairs=pairs[start : start + score_chunk_size],
                )
            )
        if len(logits) != len(pairs):
            raise ValueError("assignment scorer returned the wrong sibling shape")
        next_scores: dict[int, float] = {}
        for parent, pair, raw_logits in zip(parents, pairs, logits, strict=True):
            if len(raw_logits) != 2:
                raise ValueError("assignment scorer returned a non-binary sibling score")
            left, right = (float(value) for value in raw_logits)
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError("assignment scorer returned a non-finite logit")
            maximum = max(left, right)
            denominator = math.exp(left - maximum) + math.exp(right - maximum)
            log_denominator = maximum + math.log(denominator)
            next_scores[pair[0].node_id] = cumulative[parent] + left - log_denominator
            next_scores[pair[1].node_id] = cumulative[parent] + right - log_denominator
        cumulative = next_scores
    try:
        return [cumulative[leaf] for leaf in available_leaves]
    except KeyError as exc:
        raise ValueError("available leaf is absent from the complete-tree scores") from exc


def _write_weights(
    path: Path,
    *,
    train_products: Sequence[int],
    available_leaves: Sequence[int],
    weights: np.ndarray,
    product_chunk_size: int = 32,
) -> None:
    if weights.shape != (len(train_products), len(available_leaves)):
        raise ValueError("assignment weight matrix shape mismatch")
    schema = pa.schema(
        [
            pa.field("product_id", pa.int64(), nullable=False),
            pa.field("leaf_node_id", pa.int32(), nullable=False),
            pa.field("assignment_weight", pa.float64(), nullable=False),
        ]
    )
    leaf_values = np.asarray(available_leaves, dtype=np.int32)
    with pq.ParquetWriter(path, schema=schema, compression="zstd") as writer:
        for start in range(0, len(train_products), product_chunk_size):
            stop = min(start + product_chunk_size, len(train_products))
            block = weights[start:stop]
            products = np.repeat(
                np.asarray(train_products[start:stop], dtype=np.int64), len(available_leaves)
            )
            leaves = np.tile(leaf_values, stop - start)
            values = block.reshape(-1)
            if not np.isfinite(values).all():
                raise ValueError("assignment weights contain non-finite values")
            writer.write_table(
                pa.table(
                    {
                        "product_id": products,
                        "leaf_node_id": leaves,
                        "assignment_weight": values,
                    },
                    schema=schema,
                )
            )


def materialize_assignment_weights(
    *,
    assignment_queries_manifest_path: Path,
    catalog_path: Path,
    date_eligibility_path: Path,
    fit_manifest_path: Path,
    output_path: Path,
    output_manifest_path: Path,
    device: str,
    query_block_size: int = 16,
    score_chunk_size: int = 2048,
    backend_factory: BackendFactory | None = None,
    allow_test_only_fit: bool = False,
) -> dict[str, Any]:
    if output_path.exists() or output_manifest_path.exists():
        raise FileExistsError("refusing to overwrite assignment-weight outputs")
    if query_block_size <= 0:
        raise ValueError("query_block_size must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    query_manifest = json.loads(assignment_queries_manifest_path.read_text())
    if (
        query_manifest.get("contract_version") != "ptd-assignment-queries/v1"
        or query_manifest.get("status") != "complete"
        or query_manifest.get("dates") != EXPECTED_SPLIT["train"]
        or not all(query_manifest.get("checks", {}).values())
    ):
        raise ValueError("assignment-query manifest is incomplete")
    query_artifact = query_manifest.get("output", {})
    query_path = Path(query_artifact.get("path", ""))
    if not query_path.is_file() or sha256(query_path) != query_artifact.get("sha256"):
        raise ValueError("assignment-query path/hash mismatch")
    fit = _load_fit(fit_manifest_path, allow_test_only_fit=allow_test_only_fit)
    seed = int(fit["seed"])
    configure_determinism(seed)
    tree = CatalogTree(
        catalog_path,
        date_eligibility_path,
        depth=13,
        required_dates=EXPECTED_SPLIT["train"],
    )
    queries = load_queries(query_path, allowed_dates=EXPECTED_SPLIT["train"])
    for query in queries:
        tree.validate_query(query)
    train_products, available_leaves = _catalog_assignment_axes(catalog_path, depth=13)
    product_index = {product: index for index, product in enumerate(train_products)}
    accumulation_device = torch.device(device)
    weights = torch.zeros(
        (len(train_products), len(available_leaves)),
        dtype=torch.float64,
        device=accumulation_device,
    )
    factory = backend_factory or _default_backend_factory
    backend = factory(fit, tree, device)
    coefficient_rows: list[list[float]] = []
    path_rows: list[list[float]] = []

    def flush_block() -> None:
        if not coefficient_rows:
            return
        coefficients = torch.tensor(
            coefficient_rows, dtype=torch.float64, device=accumulation_device
        )
        paths = torch.tensor(path_rows, dtype=torch.float64, device=accumulation_device)
        weights.addmm_(coefficients.transpose(0, 1), paths)
        coefficient_rows.clear()
        path_rows.clear()

    for query in queries:
        coefficients = [0.0] * len(train_products)
        for product, candidate in query.candidates.items():
            index = product_index.get(product)
            if index is not None:
                coefficients[index] = (
                    float(candidate.purchase_label) + candidate.teacher_purchase
                )
        coefficient_rows.append(coefficients)
        path_rows.append(
            _all_leaf_path_scores(
                query,
                tree=tree,
                backend=backend,
                available_leaves=available_leaves,
                score_chunk_size=score_chunk_size,
            )
        )
        if len(coefficient_rows) == query_block_size:
            flush_block()
    flush_block()
    values = weights.detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("assignment weight accumulation produced non-finite values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        staged_output = Path(directory) / "assignment_weights.parquet"
        _write_weights(
            staged_output,
            train_products=train_products,
            available_leaves=available_leaves,
            weights=values,
        )
        output_hash = sha256(staged_output)
        manifest = {
            "contract_version": "ptd-assignment-weights/v1",
            "status": "complete",
            "source_split": "train",
            "source_dates": EXPECTED_SPLIT["train"],
            "formula": WEIGHT_FORMULA,
            "candidate_leaf_policy": CANDIDATE_LEAF_POLICY,
            "validation_or_test_fields_read": [],
            "assignment_queries_manifest_sha256": sha256(
                assignment_queries_manifest_path
            ),
            "assignment_queries_sha256": query_artifact["sha256"],
            "fit_manifest_sha256": sha256(fit_manifest_path),
            "checkpoint_sha256": fit["checkpoint"]["sha256"],
            "variant": fit["variant"],
            "seed": seed,
            "catalog_sha256": sha256(catalog_path),
            "date_eligibility_sha256": sha256(date_eligibility_path),
            "train_request_count": len(queries),
            "train_product_count": len(train_products),
            "available_leaf_count": len(available_leaves),
            "matrix_shape": [len(train_products), len(available_leaves)],
            "accumulation_dtype": "float64",
            "weights": {"path": str(output_path), "sha256": output_hash},
            "checks": {
                "train_dates_only": True,
                "validation_or_test_fields_excluded": True,
                "all_train_products_present": True,
                "all_free_leaves_present": True,
                "complete_rectangular_matrix": True,
                "finite_weights": True,
                "temperature_one_path_probabilities": True,
                "no_overwrite": True,
            },
            "scope_note": (
                "Train-only reassignment weights. This artifact contains no validation/test "
                "outcome and is not efficacy or latency evidence."
            ),
        }
        staged_manifest = Path(directory) / "manifest.json"
        staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(staged_output, output_path)
        os.replace(staged_manifest, output_manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-queries-manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--date-eligibility", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--query-block-size", type=int, default=16)
    parser.add_argument("--score-chunk-size", type=int, default=2048)
    args = parser.parse_args()
    output = materialize_assignment_weights(
        assignment_queries_manifest_path=args.assignment_queries_manifest,
        catalog_path=args.catalog,
        date_eligibility_path=args.date_eligibility,
        fit_manifest_path=args.fit_manifest,
        output_path=args.output,
        output_manifest_path=args.manifest,
        device=args.device,
        query_block_size=args.query_block_size,
        score_chunk_size=args.score_chunk_size,
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "matrix_shape": output["matrix_shape"],
                "weights": output["weights"],
            }
        )
    )


if __name__ == "__main__":
    main()
