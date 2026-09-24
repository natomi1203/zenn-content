#!/usr/bin/env python3
"""Exercise the full 8-variant x 3-seed retrieval/latency matrix synthetically."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

from reference.evidence_gate import EXPECTED_SEEDS, EXPECTED_SPLIT, EXPECTED_VARIANTS  # noqa: E402
from runner.emit_evaluation import load_retrieval_metric_rows  # noqa: E402
from runner.ptd_model import (  # noqa: E402
    PTDModelConfig,
    build_registered_model,
    configure_determinism,
    save_checkpoint_no_clobber,
)
from runner.retrieval_runner import (  # noqa: E402
    CatalogTree,
    TorchSiblingBackend,
    beam_retrieve,
    load_queries,
    run_from_plan,
    sha256,
)

USER_COUNT = 67


class SyntheticBackend:
    def __init__(self, variant: str, seed: int) -> None:
        self.variant = EXPECTED_VARIANTS.index(variant)
        self.seed = EXPECTED_SEEDS.index(seed)

    def score_sibling_pairs(self, **kwargs):
        user_value = sum(ord(value) for value in kwargs["user_id"])
        result = []
        for pair in kwargs["pairs"]:
            values = tuple(
                (
                    (
                        child.node_id * 37
                        + user_value * 11
                        + self.variant * 101
                        + self.seed * 17
                    )
                    % 997
                )
                / 100.0
                for child in pair
            )
            result.append(values)
        return result

    def synchronize(self) -> None:
        return None


class DeterministicClock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 1_000_000
        return self.value


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def write_tree(root: Path) -> tuple[Path, Path]:
    depth = 13
    leaf_start = 2**depth - 1
    products = list(range(1001, 1013))
    categories = [f"category-{index % 4}" for index in range(len(products))]
    catalog = pa.table(
        {
            "product_id": products,
            "first_category": categories,
            "path_bits": [
                format(index, f"0{depth}b") for index in range(len(products))
            ],
            "leaf_node_id": [leaf_start + index for index in range(len(products))],
        }
    )
    catalog_path = root / "catalog.parquet"
    pq.write_table(catalog, catalog_path, compression="zstd")
    eligibility = pa.table(
        {
            "snapshot_date": [
                date.fromisoformat(value)
                for value in EXPECTED_SPLIT["test"]
                for _ in products
            ],
            "product_id": products * len(EXPECTED_SPLIT["test"]),
            "leaf_node_id": [leaf_start + index for index in range(len(products))]
            * len(EXPECTED_SPLIT["test"]),
        }
    )
    eligibility_path = root / "eligibility.parquet"
    pq.write_table(eligibility, eligibility_path, compression="zstd")
    return catalog_path, eligibility_path


def write_queries(path: Path) -> int:
    products = list(range(1001, 1013))
    count = 0
    with path.open("x") as handle:
        for date_index, test_date in enumerate(EXPECTED_SPLIT["test"]):
            for user_index in range(USER_COUNT):
                purchase_product = products[(date_index + user_index) % len(products)]
                click_product = products[(date_index + user_index + 1) % len(products)]
                value = {
                    "date": test_date,
                    "user_id": f"synthetic-user-{user_index:03d}",
                    "click_history_most_recent_first": [1001, 1002],
                    "purchase_history_most_recent_first": [1003],
                    "candidates": [
                        {
                            "product_id": product,
                            "category": f"category-{index % 4}",
                            "click_label": int(product == click_product),
                            "purchase_label": int(product == purchase_product),
                            "teacher_purchase": (index + 1) / 20.0,
                        }
                        for index, product in enumerate(products)
                    ],
                }
                handle.write(
                    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
                )
                count += 1
    return count


def write_plan(
    path: Path,
    *,
    queries: Path,
    catalog: Path,
    eligibility: Path,
    checkpoint: Path,
) -> None:
    entries = []
    for variant in EXPECTED_VARIANTS:
        for seed in EXPECTED_SEEDS:
            entries.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "catalog": _artifact(catalog),
                    "date_eligibility": _artifact(eligibility),
                    "checkpoint": None
                    if variant == "teacher_oracle"
                    else _artifact(checkpoint),
                }
            )
    plan = {
        "contract_version": "ptd-retrieval-run-plan/v1",
        "status": "locked",
        "test_dates": EXPECTED_SPLIT["test"],
        "seeds": EXPECTED_SEEDS,
        "beam_width": 600,
        "top_k": 600,
        "warmup_queries_per_variant_seed": 100,
        "minimum_measured_queries_per_variant": 1000,
        "concurrency": 1,
        "device": "synthetic-cpu",
        "queries": _artifact(queries),
        "entries": entries,
    }
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")


def torch_adapter_checks(
    root: Path,
    *,
    catalog: Path,
    eligibility: Path,
    queries: Path,
) -> tuple[dict[str, bool], Path]:
    torch.set_num_threads(1)
    tree = CatalogTree(catalog, eligibility, depth=13)
    query = load_queries(queries)[0]
    config = PTDModelConfig(
        item_hash_bucket_size=64,
        user_hash_bucket_size=64,
        category_hash_bucket_size=16,
    )
    checks: dict[str, bool] = {}
    first_checkpoint: Path | None = None
    for variant, label in (
        ("ptd_combined", "hstu"),
        ("ptd_combined_baseline_encoder", "din"),
    ):
        configure_determinism(16630)
        model, _ = build_registered_model(variant, config)
        checkpoint = root / f"{label}-checkpoint.pt"
        save_checkpoint_no_clobber(
            checkpoint,
            model=model,
            variant=variant,
            seed=16630,
            history=[{"epoch": 1.0}, {"epoch": 2.0}],
        )
        if first_checkpoint is None:
            first_checkpoint = checkpoint
        backend = TorchSiblingBackend(
            checkpoint,
            variant=variant,
            seed=16630,
            device="cpu",
        )
        result = beam_retrieve(query, tree, backend)
        checks[f"torch_{label}_checkpoint_loaded"] = True
        checks[f"torch_{label}_beam_finite"] = len(result.product_ids) == len(
            query.candidates
        ) and all(
            value == value and abs(value) != float("inf")
            for value in result.path_scores
        )
        if label == "hstu":
            checks["torch_hstu_user_state_cached_across_levels"] = (
                backend._cached_hstu_key is not None
                and backend._cached_hstu_state is not None
            )
    if first_checkpoint is None:
        raise ValueError("torch adapter smoke created no checkpoint")
    return checks, first_checkpoint


def run() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        catalog, eligibility = write_tree(root)
        queries = root / "queries.jsonl"
        query_count = write_queries(queries)
        torch_checks, checkpoint = torch_adapter_checks(
            root,
            catalog=catalog,
            eligibility=eligibility,
            queries=queries,
        )
        plan = root / "plan.json"
        write_plan(
            plan,
            queries=queries,
            catalog=catalog,
            eligibility=eligibility,
            checkpoint=checkpoint,
        )

        def factory(entry: dict, tree: CatalogTree, device: str):
            del tree, device
            return SyntheticBackend(entry["variant"], entry["seed"])

        first_output = root / "first" / "metrics.jsonl"
        first_manifest = root / "first" / "manifest.json"
        first = run_from_plan(
            plan,
            first_output,
            first_manifest,
            backend_factory=factory,
            clock_ns=DeterministicClock(),
        )
        second_output = root / "second" / "metrics.jsonl"
        second_manifest = root / "second" / "manifest.json"
        second = run_from_plan(
            plan,
            second_output,
            second_manifest,
            backend_factory=factory,
            clock_ns=DeterministicClock(),
        )
        parsed = load_retrieval_metric_rows(first_output)
        no_overwrite = False
        try:
            run_from_plan(
                plan,
                first_output,
                first_manifest,
                backend_factory=factory,
                clock_ns=DeterministicClock(),
            )
        except FileExistsError:
            no_overwrite = True
        expected_rows = query_count * len(EXPECTED_SEEDS) * len(EXPECTED_VARIANTS)
        checks = {
            **first["checks"],
            **torch_checks,
            "retrieval_observation_contract_passed": len(parsed) == expected_rows,
            "deterministic_metric_rows": first["output"]["sha256"]
            == second["output"]["sha256"],
            "no_overwrite_reverified": no_overwrite,
            "all_metrics_finite": all(
                all(value >= 0 for value in row.metrics.values()) for row in parsed
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"retrieval runner smoke failed: {checks}")
        payload = {
            "contract_version": "ptd-retrieval-runner-smoke/v1",
            "status": "SMOKE_ONLY",
            "empirical_claim_allowed": False,
            "synthetic_counts": {
                "users": USER_COUNT,
                "dates": len(EXPECTED_SPLIT["test"]),
                "queries": query_count,
                "seeds": len(EXPECTED_SEEDS),
                "variants": len(EXPECTED_VARIANTS),
                "retrieval_metric_rows": expected_rows,
                "measured_rows_per_variant": query_count * len(EXPECTED_SEEDS),
                "warmup_queries_per_variant_seed": 100,
            },
            "output_sha256": first["output"]["sha256"],
            "checks": checks,
            "runtime": {
                "python": platform.python_version(),
                "pyarrow": pa.__version__,
                "torch": torch.__version__,
                "platform": platform.system(),
            },
            "scope_note": (
                "Synthetic beam/latency implementation evidence only. The deterministic scorer, "
                "clock, labels, and metrics are fixtures and are not PTD results."
            ),
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifact" / "smoke" / "retrieval_runner_smoke.json",
    )
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "retrieval runner smoke: PASS (5 dates x 3 seeds x 8 variants, beam + latency)"
    )


if __name__ == "__main__":
    main()
