#!/usr/bin/env python3
"""Five-date, three-seed PTD beam retrieval and matched latency runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import pyarrow.parquet as pq

from reference.evaluation import dcg_at_k, recall_at_k
from reference.evidence_gate import EXPECTED_SEEDS, EXPECTED_SPLIT, EXPECTED_VARIANTS

BEAM_WIDTH = 600
TOP_K = 600
WARMUP_QUERIES = 100
MIN_MEASURED_QUERIES_PER_VARIANT = 1_000
INTERNAL_CATEGORY = "__internal__"
PADDING_CATEGORY = "__padding__"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CandidateLabel:
    product_id: int
    category: str
    click_label: int
    purchase_label: int
    teacher_purchase: float


@dataclass(frozen=True)
class RetrievalQuery:
    date: str
    user_id: str
    click_history_most_recent_first: tuple[int, ...]
    purchase_history_most_recent_first: tuple[int, ...]
    candidates: Mapping[int, CandidateLabel]


@dataclass(frozen=True)
class NodeDescriptor:
    node_id: int
    parent_node_id: int
    level: int
    item_key: str
    category: str
    product_id: int | None


@dataclass(frozen=True)
class RetrievalResult:
    product_ids: tuple[int, ...]
    path_scores: tuple[float, ...]
    candidates_scored: int


class SiblingScoringBackend(Protocol):
    """Scores binary child pairs without seeing labels or teacher probabilities."""

    def score_sibling_pairs(
        self,
        *,
        user_id: str,
        click_history_most_recent_first: Sequence[int],
        purchase_history_most_recent_first: Sequence[int],
        pairs: Sequence[tuple[NodeDescriptor, NodeDescriptor]],
    ) -> list[tuple[float, float]]: ...

    def synchronize(self) -> None: ...


class CatalogTree:
    """Complete binary tree plus immutable date eligibility sets."""

    def __init__(
        self, catalog_path: Path, date_eligibility_path: Path, *, depth: int
    ) -> None:
        if depth <= 0:
            raise ValueError("tree depth must be positive")
        self.depth = depth
        self.leaf_start = 2**depth - 1
        catalog = pq.read_table(catalog_path)
        required_catalog = {"product_id", "first_category", "leaf_node_id", "path_bits"}
        if missing := required_catalog - set(catalog.column_names):
            raise ValueError(f"catalog missing columns: {sorted(missing)}")
        self.product_by_leaf: dict[int, int] = {}
        self.leaf_by_product: dict[int, int] = {}
        self.category_by_product: dict[int, str] = {}
        for row in catalog.to_pylist():
            product = int(row["product_id"])
            leaf = int(row["leaf_node_id"])
            if not self.leaf_start <= leaf < self.leaf_start + 2**depth:
                raise ValueError("catalog leaf is outside the complete tree")
            if product in self.leaf_by_product or leaf in self.product_by_leaf:
                raise ValueError("catalog products and occupied leaves must be unique")
            offset = leaf - self.leaf_start
            if str(row["path_bits"]) != format(offset, f"0{depth}b"):
                raise ValueError("catalog path_bits do not match leaf_node_id")
            self.product_by_leaf[leaf] = product
            self.leaf_by_product[product] = leaf
            self.category_by_product[product] = str(row["first_category"] or "")

        eligibility = pq.read_table(date_eligibility_path)
        required_eligibility = {"snapshot_date", "product_id", "leaf_node_id"}
        if missing := required_eligibility - set(eligibility.column_names):
            raise ValueError(f"date eligibility missing columns: {sorted(missing)}")
        self.eligible_products: dict[str, set[int]] = {}
        self.eligible_nodes: dict[str, dict[int, set[int]]] = {}
        for row in eligibility.to_pylist():
            date_value = row["snapshot_date"]
            date = (
                date_value.isoformat()
                if hasattr(date_value, "isoformat")
                else str(date_value)
            )
            product = int(row["product_id"])
            leaf = int(row["leaf_node_id"])
            if self.leaf_by_product.get(product) != leaf:
                raise ValueError("date eligibility product/leaf differs from catalog")
            self.eligible_products.setdefault(date, set()).add(product)
        test_dates = set(EXPECTED_SPLIT["test"])
        if not test_dates <= set(self.eligible_products):
            raise ValueError("tree date eligibility is missing a registered test date")
        self.eligible_products = {
            date: products
            for date, products in self.eligible_products.items()
            if date in test_dates
        }
        for date, products in self.eligible_products.items():
            by_level = {level: set() for level in range(depth + 1)}
            for product in products:
                offset = self.leaf_by_product[product] - self.leaf_start
                for level in range(depth + 1):
                    prefix = offset >> (depth - level) if level else 0
                    by_level[level].add((2**level - 1) + prefix)
            self.eligible_nodes[date] = by_level

    def descriptor(
        self, node_id: int, parent_node_id: int, level: int
    ) -> NodeDescriptor:
        if level == self.depth:
            product = self.product_by_leaf.get(node_id)
            if product is None:
                return NodeDescriptor(
                    node_id=node_id,
                    parent_node_id=parent_node_id,
                    level=level,
                    item_key=f"padding:{node_id}",
                    category=PADDING_CATEGORY,
                    product_id=None,
                )
            return NodeDescriptor(
                node_id=node_id,
                parent_node_id=parent_node_id,
                level=level,
                item_key=str(product),
                category=self.category_by_product[product],
                product_id=product,
            )
        return NodeDescriptor(
            node_id=node_id,
            parent_node_id=parent_node_id,
            level=level,
            item_key=f"node:{node_id}",
            category=INTERNAL_CATEGORY,
            product_id=None,
        )

    def validate_query(self, query: RetrievalQuery) -> None:
        if query.date not in self.eligible_products:
            raise ValueError(f"query date has no eligibility mask: {query.date}")
        if set(query.candidates) != self.eligible_products[query.date]:
            raise ValueError(
                f"query candidate set differs from date mask for {query.date}/{query.user_id}"
            )
        for product, candidate in query.candidates.items():
            if candidate.category != self.category_by_product[product]:
                raise ValueError("query candidate category differs from locked catalog")


def _parse_history(value: Any, label: str, line_number: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) > 30:
        raise ValueError(
            f"{label} must be an array of at most 30 IDs at line {line_number}"
        )
    normalized: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{label} must contain integer IDs at line {line_number}")
        normalized.append(item)
    return tuple(normalized)


def load_queries(path: Path) -> list[RetrievalQuery]:
    """Load exact five-day query rows; candidate labels are evaluation-only."""
    rows: list[RetrievalQuery] = []
    seen: set[tuple[str, str]] = set()
    expected_keys = {
        "date",
        "user_id",
        "click_history_most_recent_first",
        "purchase_history_most_recent_first",
        "candidates",
    }
    candidate_keys = {
        "product_id",
        "category",
        "click_label",
        "purchase_label",
        "teacher_purchase",
    }
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"blank query row at line {line_number}")
            value = json.loads(raw_line)
            if not isinstance(value, dict) or set(value) != expected_keys:
                raise ValueError(f"query line {line_number} has unexpected fields")
            date = value["date"]
            user_id = value["user_id"]
            if date not in EXPECTED_SPLIT["test"]:
                raise ValueError(f"unexpected query date at line {line_number}")
            if not isinstance(user_id, str) or not user_id:
                raise ValueError(f"invalid user_id at line {line_number}")
            key = (date, user_id)
            if key in seen:
                raise ValueError(f"duplicate date-user query at line {line_number}")
            seen.add(key)
            raw_candidates = value["candidates"]
            if not isinstance(raw_candidates, list) or not raw_candidates:
                raise ValueError(f"candidates must be non-empty at line {line_number}")
            candidates: dict[int, CandidateLabel] = {}
            for candidate in raw_candidates:
                if not isinstance(candidate, dict) or set(candidate) != candidate_keys:
                    raise ValueError(
                        f"candidate has unexpected fields at line {line_number}"
                    )
                product = candidate["product_id"]
                if isinstance(product, bool) or not isinstance(product, int):
                    raise ValueError(
                        f"candidate product_id must be integer at line {line_number}"
                    )
                if product in candidates:
                    raise ValueError(
                        f"duplicate candidate product at line {line_number}"
                    )
                click = candidate["click_label"]
                purchase = candidate["purchase_label"]
                teacher = candidate["teacher_purchase"]
                if click not in (0, 1) or purchase not in (0, 1):
                    raise ValueError(
                        f"candidate labels must be binary at line {line_number}"
                    )
                if (
                    isinstance(teacher, bool)
                    or not isinstance(teacher, (int, float))
                    or not math.isfinite(teacher)
                    or not 0.0 <= teacher <= 1.0
                ):
                    raise ValueError(
                        f"teacher score must be in [0,1] at line {line_number}"
                    )
                category = candidate["category"]
                if not isinstance(category, str):
                    raise ValueError(
                        f"candidate category must be a string at line {line_number}"
                    )
                candidates[product] = CandidateLabel(
                    product_id=product,
                    category=category,
                    click_label=int(click),
                    purchase_label=int(purchase),
                    teacher_purchase=float(teacher),
                )
            rows.append(
                RetrievalQuery(
                    date=date,
                    user_id=user_id,
                    click_history_most_recent_first=_parse_history(
                        value["click_history_most_recent_first"],
                        "click history",
                        line_number,
                    ),
                    purchase_history_most_recent_first=_parse_history(
                        value["purchase_history_most_recent_first"],
                        "purchase history",
                        line_number,
                    ),
                    candidates=candidates,
                )
            )
    if not rows or {row.date for row in rows} != set(EXPECTED_SPLIT["test"]):
        raise ValueError("queries must cover all five test dates")
    return sorted(rows, key=lambda row: (row.date, row.user_id))


def beam_retrieve(
    query: RetrievalQuery,
    tree: CatalogTree,
    backend: SiblingScoringBackend,
    *,
    beam_width: int = BEAM_WIDTH,
    top_k: int = TOP_K,
) -> RetrievalResult:
    """Run eligible-node-masked sibling-softmax beam retrieval."""
    if beam_width != 600 or top_k != 600:
        raise ValueError("registered beam width and top-K are both 600")
    tree.validate_query(query)
    beam: list[tuple[int, float]] = [(0, 0.0)]
    scored = 0
    for level in range(1, tree.depth + 1):
        pairs: list[tuple[NodeDescriptor, NodeDescriptor]] = []
        for parent, _ in beam:
            left = 2 * parent + 1
            right = left + 1
            pairs.append(
                (
                    tree.descriptor(left, parent, level),
                    tree.descriptor(right, parent, level),
                )
            )
        logits = backend.score_sibling_pairs(
            user_id=query.user_id,
            click_history_most_recent_first=query.click_history_most_recent_first,
            purchase_history_most_recent_first=query.purchase_history_most_recent_first,
            pairs=pairs,
        )
        if len(logits) != len(pairs) or any(len(value) != 2 for value in logits):
            raise ValueError("scoring backend returned the wrong sibling shape")
        scored += 2 * len(pairs)
        expanded: list[tuple[int, float]] = []
        eligible = tree.eligible_nodes[query.date][level]
        for (_, parent_score), pair, pair_logits in zip(
            beam, pairs, logits, strict=True
        ):
            valid = [descriptor.node_id in eligible for descriptor in pair]
            if not any(valid):
                continue
            finite_logits = [float(value) for value in pair_logits]
            if any(not math.isfinite(value) for value in finite_logits):
                raise ValueError("scoring backend returned non-finite logits")
            maximum = max(
                value for value, keep in zip(finite_logits, valid, strict=True) if keep
            )
            denominator = sum(
                math.exp(value - maximum)
                for value, keep in zip(finite_logits, valid, strict=True)
                if keep
            )
            for descriptor, value, keep in zip(pair, finite_logits, valid, strict=True):
                if keep:
                    log_probability = value - maximum - math.log(denominator)
                    expanded.append(
                        (descriptor.node_id, parent_score + log_probability)
                    )
        beam = sorted(expanded, key=lambda value: (-value[1], value[0]))[:beam_width]
        if not beam:
            raise ValueError("beam became empty despite a non-empty eligibility set")
    products: list[int] = []
    scores: list[float] = []
    for leaf, score in beam[:top_k]:
        product = tree.product_by_leaf.get(leaf)
        if product is None or product not in query.candidates:
            raise ValueError("beam returned padding or ineligible leaf")
        products.append(product)
        scores.append(score)
    if len(products) != len(set(products)):
        raise ValueError("beam returned duplicate products")
    return RetrievalResult(tuple(products), tuple(scores), scored)


def teacher_oracle_retrieve(
    query: RetrievalQuery, *, top_k: int = TOP_K
) -> RetrievalResult:
    ordered = sorted(
        query.candidates.values(),
        key=lambda candidate: (-candidate.teacher_purchase, candidate.product_id),
    )[:top_k]
    return RetrievalResult(
        product_ids=tuple(candidate.product_id for candidate in ordered),
        path_scores=tuple(candidate.teacher_purchase for candidate in ordered),
        candidates_scored=len(query.candidates),
    )


def _rank_auc(query: RetrievalQuery, ranked_products: Sequence[int]) -> float:
    """AUC over all eligible items using retrieval rank; unretrieved items tie below top-K."""
    rank_score = {
        product: float(len(ranked_products) - rank)
        for rank, product in enumerate(ranked_products)
    }
    positives = [
        rank_score.get(product, 0.0)
        for product, candidate in query.candidates.items()
        if candidate.purchase_label == 1
    ]
    negatives = [
        rank_score.get(product, 0.0)
        for product, candidate in query.candidates.items()
        if candidate.purchase_label == 0
    ]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def _retrieval_ndcg(
    query: RetrievalQuery,
    ranked_products: Sequence[int],
    *,
    label: str,
    k: int,
) -> float:
    observed = [
        getattr(query.candidates[product], label) for product in ranked_products[:k]
    ]
    eligible = sorted(
        (getattr(candidate, label) for candidate in query.candidates.values()),
        reverse=True,
    )
    ideal = dcg_at_k(eligible, k)
    return dcg_at_k(observed, k) / ideal if ideal > 0 else 0.0


def retrieval_metrics(
    query: RetrievalQuery,
    result: RetrievalResult,
    *,
    latency_ms: float,
) -> dict[str, float]:
    retrieved = list(result.product_ids)
    purchase_relevance = [
        query.candidates[product].purchase_label for product in retrieved
    ]
    top50 = retrieved[:50]
    eligible_categories = {
        candidate.category for candidate in query.candidates.values()
    }
    top_categories = [query.candidates[product].category for product in top50]
    category_denominator = min(50, len(eligible_categories))
    coverage = (
        len(set(top_categories)) / category_denominator if category_denominator else 0.0
    )
    max_share = (
        max(top_categories.count(category) for category in set(top_categories))
        / len(top_categories)
        if top_categories
        else 0.0
    )
    total_purchase = sum(
        candidate.purchase_label for candidate in query.candidates.values()
    )
    return {
        "purchase_ndcg_at_50": _retrieval_ndcg(
            query, retrieved, label="purchase_label", k=50
        ),
        "purchase_recall_at_50": recall_at_k(
            purchase_relevance, 50, total_relevant=total_purchase
        ),
        "purchase_ndcg_at_10": _retrieval_ndcg(
            query, retrieved, label="purchase_label", k=10
        ),
        "purchase_ndcg_at_100": _retrieval_ndcg(
            query, retrieved, label="purchase_label", k=100
        ),
        "purchase_auc": _rank_auc(query, retrieved),
        "click_ndcg_at_50": _retrieval_ndcg(
            query, retrieved, label="click_label", k=50
        ),
        "category_coverage_at_50": coverage,
        "max_category_share_at_50": max_share,
        "latency_ms": latency_ms,
        "candidates_scored": float(result.candidates_scored),
    }


def _artifact(value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = Path(value["path"])
    digest = value["sha256"]
    if not path.is_file() or sha256(path) != digest:
        raise ValueError(f"{label} path/hash mismatch")
    return path, digest


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text())
    if (
        plan.get("contract_version") != "ptd-retrieval-run-plan/v1"
        or plan.get("status") != "locked"
    ):
        raise ValueError("retrieval plan must be locked ptd-retrieval-run-plan/v1")
    constants = {
        "test_dates": EXPECTED_SPLIT["test"],
        "seeds": EXPECTED_SEEDS,
        "beam_width": 600,
        "top_k": 600,
        "concurrency": 1,
    }
    for key, expected in constants.items():
        if plan.get(key) != expected:
            raise ValueError(f"retrieval plan {key} differs from registration")
    for key, minimum in (
        ("warmup_queries_per_variant_seed", WARMUP_QUERIES),
        ("minimum_measured_queries_per_variant", MIN_MEASURED_QUERIES_PER_VARIANT),
    ):
        value = plan.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"retrieval plan {key} must be at least {minimum}")
    _artifact(plan.get("queries"), "queries")
    entries = plan.get("entries")
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_VARIANTS) * len(
        EXPECTED_SEEDS
    ):
        raise ValueError("retrieval plan must contain all 24 variant-seed entries")
    seen: set[tuple[str, int]] = set()
    for entry in entries:
        required = {"variant", "seed", "catalog", "date_eligibility", "checkpoint"}
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError("retrieval plan entry fields mismatch")
        variant = entry["variant"]
        seed = entry["seed"]
        if variant not in EXPECTED_VARIANTS or seed not in EXPECTED_SEEDS:
            raise ValueError("retrieval plan has unknown variant or seed")
        if (variant, seed) in seen:
            raise ValueError("retrieval plan has duplicate variant-seed entry")
        seen.add((variant, seed))
        _artifact(entry["catalog"], f"{variant}/{seed} catalog")
        _artifact(entry["date_eligibility"], f"{variant}/{seed} eligibility")
        if variant == "teacher_oracle":
            if entry["checkpoint"] is not None:
                raise ValueError("teacher oracle must not have a tree-model checkpoint")
        else:
            _artifact(entry["checkpoint"], f"{variant}/{seed} checkpoint")
    expected_pairs = {
        (variant, seed) for variant in EXPECTED_VARIANTS for seed in EXPECTED_SEEDS
    }
    if seen != expected_pairs:
        raise ValueError("retrieval plan does not cover the registered matrix")
    if not isinstance(plan.get("device"), str) or not plan["device"]:
        raise ValueError("retrieval plan device is missing")
    return plan


class TorchSiblingBackend:
    """Registered PTD checkpoint adapter; teacher scores and labels are not accepted."""

    def __init__(
        self, checkpoint: Path, *, variant: str, seed: int, device: str
    ) -> None:
        import torch

        from runner.ptd_model import ModelInputs, PTDModelConfig, build_registered_model

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("contract_version") != "ptd-trainer-checkpoint/v1":
            raise ValueError("PTD checkpoint contract mismatch")
        if payload.get("variant") != variant or payload.get("seed") != seed:
            raise ValueError("PTD checkpoint variant/seed mismatch")
        config_values = dict(payload["config"])
        if "windows" in config_values:
            config_values["windows"] = tuple(config_values["windows"])
        self.config = PTDModelConfig(**config_values)
        self.model, _ = build_registered_model(variant, self.config)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.torch = torch
        self.ModelInputs = ModelInputs
        self._cached_hstu_key: tuple[str, tuple[int, ...], tuple[int, ...]] | None = (
            None
        )
        self._cached_hstu_state = None

    def _history(self, values: Sequence[int]) -> tuple[list[int], int]:
        from runner.ptd_model import stable_hash_bucket

        selected = list(values[:30])
        encoded = [
            stable_hash_bucket(value, self.config.item_hash_bucket_size)
            for value in reversed(selected)
        ]
        return encoded + [0] * (30 - len(encoded)), len(encoded)

    def _inputs(
        self,
        user_id: str,
        click_history: Sequence[int],
        purchase_history: Sequence[int],
        pairs: Sequence[tuple[NodeDescriptor, NodeDescriptor]],
    ):
        from runner.ptd_model import stable_hash_bucket

        torch = self.torch
        click, click_length = self._history(click_history)
        purchase, purchase_length = self._history(purchase_history)
        batch = len(pairs)
        return self.ModelInputs(
            user_ids=torch.tensor(
                [stable_hash_bucket(user_id, self.config.user_hash_bucket_size)]
                * batch,
                dtype=torch.long,
                device=self.device,
            ),
            click_history_ids=torch.tensor(
                [click] * batch, dtype=torch.long, device=self.device
            ),
            click_lengths=torch.tensor(
                [click_length] * batch, dtype=torch.long, device=self.device
            ),
            purchase_history_ids=torch.tensor(
                [purchase] * batch, dtype=torch.long, device=self.device
            ),
            purchase_lengths=torch.tensor(
                [purchase_length] * batch, dtype=torch.long, device=self.device
            ),
            child_item_ids=torch.tensor(
                [
                    [
                        stable_hash_bucket(
                            child.item_key, self.config.item_hash_bucket_size
                        )
                        for child in pair
                    ]
                    for pair in pairs
                ],
                dtype=torch.long,
                device=self.device,
            ),
            child_category_ids=torch.tensor(
                [
                    [
                        stable_hash_bucket(
                            child.category, self.config.category_hash_bucket_size
                        )
                        for child in pair
                    ]
                    for pair in pairs
                ],
                dtype=torch.long,
                device=self.device,
            ),
            parent_node_ids=torch.tensor(
                [
                    [
                        stable_hash_bucket(
                            f"node:{child.parent_node_id}",
                            self.config.item_hash_bucket_size,
                        )
                        for child in pair
                    ]
                    for pair in pairs
                ],
                dtype=torch.long,
                device=self.device,
            ),
            child_levels=torch.tensor(
                [[child.level for child in pair] for pair in pairs],
                dtype=torch.long,
                device=self.device,
            ),
        )

    def score_sibling_pairs(
        self,
        *,
        user_id: str,
        click_history_most_recent_first: Sequence[int],
        purchase_history_most_recent_first: Sequence[int],
        pairs: Sequence[tuple[NodeDescriptor, NodeDescriptor]],
    ) -> list[tuple[float, float]]:
        inputs = self._inputs(
            user_id,
            click_history_most_recent_first,
            purchase_history_most_recent_first,
            pairs,
        )
        torch = self.torch
        with torch.inference_mode():
            if self.model.encoder_name == "hstu":
                cache_key = (
                    user_id,
                    tuple(click_history_most_recent_first),
                    tuple(purchase_history_most_recent_first),
                )
                if cache_key != self._cached_hstu_key:
                    one = self._inputs(
                        user_id,
                        click_history_most_recent_first,
                        purchase_history_most_recent_first,
                        pairs[:1],
                    )
                    user_value = self.model.user(one.user_ids)
                    self._cached_hstu_state = self.model.encoder(one, user_value)
                    self._cached_hstu_key = cache_key
                state = self._cached_hstu_state
                logits = self.model.scorer(
                    state[:, None, :].expand(len(pairs), 2, -1), inputs
                )
            else:
                logits = self.model(inputs)
        return [
            tuple(float(value) for value in row)
            for row in logits.detach().cpu().tolist()
        ]

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)


BackendFactory = Callable[[Mapping[str, Any], CatalogTree, str], SiblingScoringBackend]


def _default_backend_factory(
    entry: Mapping[str, Any], tree: CatalogTree, device: str
) -> SiblingScoringBackend:
    del tree
    checkpoint, _ = _artifact(entry["checkpoint"], "checkpoint")
    return TorchSiblingBackend(
        checkpoint,
        variant=str(entry["variant"]),
        seed=int(entry["seed"]),
        device=device,
    )


def run_from_plan(
    plan_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    backend_factory: BackendFactory | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Run all 24 locked entries serially and atomically emit retrieval metric rows."""
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite retrieval outputs")
    plan = load_plan(plan_path)
    queries_path, queries_hash = _artifact(plan["queries"], "queries")
    queries = load_queries(queries_path)
    queries_per_variant = len(queries) * len(EXPECTED_SEEDS)
    required_measured = int(plan["minimum_measured_queries_per_variant"])
    warmup_count = int(plan["warmup_queries_per_variant_seed"])
    if queries_per_variant < required_measured:
        raise ValueError("five-day query count cannot satisfy the latency protocol")
    factory = backend_factory or _default_backend_factory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {variant: 0 for variant in EXPECTED_VARIANTS}
    with tempfile.TemporaryDirectory() as directory:
        staged_output = Path(directory) / "retrieval_metrics.jsonl"
        with staged_output.open("x") as handle:
            for entry in sorted(
                plan["entries"], key=lambda item: (item["variant"], item["seed"])
            ):
                variant = str(entry["variant"])
                seed = int(entry["seed"])
                catalog_path, _ = _artifact(
                    entry["catalog"], f"{variant}/{seed} catalog"
                )
                eligibility_path, _ = _artifact(
                    entry["date_eligibility"], f"{variant}/{seed} eligibility"
                )
                tree = CatalogTree(catalog_path, eligibility_path, depth=13)
                for query in queries:
                    tree.validate_query(query)
                backend = (
                    None
                    if variant == "teacher_oracle"
                    else factory(entry, tree, plan["device"])
                )
                warmups = [
                    queries[index % len(queries)] for index in range(warmup_count)
                ]
                for query in warmups:
                    if backend is None:
                        teacher_oracle_retrieve(query)
                    else:
                        beam_retrieve(query, tree, backend)
                if backend is not None:
                    backend.synchronize()
                for query in queries:
                    if backend is not None:
                        backend.synchronize()
                    started = clock_ns()
                    result = (
                        teacher_oracle_retrieve(query)
                        if backend is None
                        else beam_retrieve(query, tree, backend)
                    )
                    if backend is not None:
                        backend.synchronize()
                    elapsed_ms = (clock_ns() - started) / 1_000_000.0
                    if elapsed_ms < 0 or not math.isfinite(elapsed_ms):
                        raise ValueError("latency clock returned an invalid duration")
                    metrics = retrieval_metrics(query, result, latency_ms=elapsed_ms)
                    row = {
                        "date": query.date,
                        "user_id": query.user_id,
                        "seed": seed,
                        "variant": variant,
                        **metrics,
                    }
                    handle.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    )
                    counts[variant] += 1
        output_hash = sha256(staged_output)
        manifest = {
            "contract_version": "ptd-retrieval-metrics/v1",
            "status": "complete",
            "plan_sha256": sha256(plan_path),
            "queries_sha256": queries_hash,
            "test_dates": EXPECTED_SPLIT["test"],
            "seeds": EXPECTED_SEEDS,
            "variants": list(EXPECTED_VARIANTS),
            "beam_width": BEAM_WIDTH,
            "top_k": TOP_K,
            "warmup_queries_per_variant_seed": warmup_count,
            "measured_rows_by_variant": counts,
            "output": {"path": str(output_path), "sha256": output_hash},
            "checks": {
                "all_variant_seed_entries_complete": all(
                    count == queries_per_variant for count in counts.values()
                ),
                "minimum_latency_queries_per_variant": all(
                    count >= required_measured for count in counts.values()
                ),
                "candidate_sets_match_date_masks": True,
                "concurrency_one": True,
                "teacher_score_used_only_by_oracle": True,
                "labels_used_only_after_retrieval": True,
                "no_overwrite": True,
            },
            "scope_note": (
                "This manifest records retrieval observations. It is not admissible efficacy evidence "
                "until the paired evaluation emitter and evidence gate accept the complete run bundle."
            ),
        }
        if not all(manifest["checks"].values()):
            raise ValueError("retrieval run failed a completeness invariant")
        staged_manifest = Path(directory) / "manifest.json"
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staged_output, output_path)
        os.replace(staged_manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_from_plan(args.plan, args.output, args.manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": manifest["output"],
                "rows": sum(manifest["measured_rows_by_variant"].values()),
            }
        )
    )


if __name__ == "__main__":
    main()
