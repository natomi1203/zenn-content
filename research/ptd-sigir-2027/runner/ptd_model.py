#!/usr/bin/env python3
"""Registered PTD encoders, sibling scorer, objective, and two-epoch trainer.

The module is the production model/training core. Data materialization, tree
alternation, retrieval, and evaluation remain separate fail-closed components.
Teacher probabilities enter :func:`compute_ptd_loss` only; the serving forward
path accepts no teacher feature.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

WINDOWS_MOST_RECENT_FIRST = (1, 2, 3, 4, 5, 5, 10)
REGISTERED_VARIANTS = {
    "fixed_tdm": ("hstu", False, False),
    "ptd_item": ("hstu", True, False),
    "ptd_node": ("hstu", False, True),
    "ptd_combined": ("hstu", True, True),
    "alternating_tdm": ("hstu", False, False),
    "alternating_ptd": ("hstu", True, True),
    "ptd_combined_baseline_encoder": ("din", True, True),
}
DISTILLATION_NONE = 0
DISTILLATION_ITEM = 1
DISTILLATION_NODE = 2


def stable_hash_bucket(value: object, bucket_size: int) -> int:
    """Map an identifier to ``[1, bucket_size)``; zero is reserved for padding."""
    if bucket_size < 2:
        raise ValueError("bucket_size must leave one non-padding bucket")
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (bucket_size - 1)


def parse_most_recent_first_history(
    value: str | None,
    *,
    bucket_size: int,
    maximum_length: int = 30,
) -> tuple[list[int], int]:
    """Reverse the registered semicolon stream into oldest-to-newest order."""
    if maximum_length <= 0:
        raise ValueError("maximum_length must be positive")
    tokens = (
        []
        if value is None
        else [token.strip() for token in value.split(";") if token.strip()]
    )
    tokens = tokens[:maximum_length]
    encoded = [stable_hash_bucket(token, bucket_size) for token in reversed(tokens)]
    return encoded + [0] * (maximum_length - len(encoded)), len(encoded)


@dataclass(frozen=True)
class PTDModelConfig:
    """Registered architecture, with bucket sizes overridable for synthetic tests."""

    item_hash_bucket_size: int = 200_000
    user_hash_bucket_size: int = 1_000_000
    category_hash_bucket_size: int = 10_000
    hidden_dim: int = 64
    stream_type_embedding_dim: int = 8
    maximum_length: int = 30
    position_buckets: int = 64
    num_layers: int = 2
    num_heads: int = 4
    tree_depth: int = 13
    windows: tuple[int, ...] = WINDOWS_MOST_RECENT_FIRST

    def __post_init__(self) -> None:
        if (
            min(
                self.item_hash_bucket_size,
                self.user_hash_bucket_size,
                self.category_hash_bucket_size,
            )
            < 2
        ):
            raise ValueError("hash bucket sizes must be at least two")
        if self.hidden_dim != 64 or self.stream_type_embedding_dim != 8:
            raise ValueError("the registered embedding dimensions are 64 and 8")
        if self.maximum_length != 30 or self.position_buckets != 64:
            raise ValueError("the registered history/position sizes are 30 and 64")
        if self.num_layers != 2 or self.num_heads != 4:
            raise ValueError(
                "the registered HSTU-style stack has two layers and four heads"
            )
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if (
            tuple(self.windows) != WINDOWS_MOST_RECENT_FIRST
            or sum(self.windows) != self.maximum_length
        ):
            raise ValueError("DIN windows must be exactly [1,2,3,4,5,5,10]")
        if self.tree_depth != 13:
            raise ValueError("the registered tree depth is 13")


@dataclass(frozen=True)
class ModelInputs:
    """Tensor contract shared by the HSTU-style and DIN serving paths."""

    user_ids: Tensor
    click_history_ids: Tensor
    click_lengths: Tensor
    purchase_history_ids: Tensor
    purchase_lengths: Tensor
    child_item_ids: Tensor
    child_category_ids: Tensor
    parent_node_ids: Tensor
    child_levels: Tensor

    def to(self, device: torch.device | str) -> "ModelInputs":
        return ModelInputs(
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
            }
        )

    def validate(self, maximum_length: int) -> None:
        batch_size = self.user_ids.shape[0]
        if self.user_ids.ndim != 1:
            raise ValueError("user_ids must have shape [batch]")
        for name in ("click_history_ids", "purchase_history_ids"):
            value = getattr(self, name)
            if value.shape != (batch_size, maximum_length):
                raise ValueError(f"{name} must have shape [batch, {maximum_length}]")
        for name in ("click_lengths", "purchase_lengths"):
            value = getattr(self, name)
            if value.shape != (batch_size,):
                raise ValueError(f"{name} must have shape [batch]")
            if torch.any(value < 0) or torch.any(value > maximum_length):
                raise ValueError(f"{name} values must be in [0, {maximum_length}]")
        sibling_shape = (batch_size, 2)
        for name in (
            "child_item_ids",
            "child_category_ids",
            "parent_node_ids",
            "child_levels",
        ):
            if getattr(self, name).shape != sibling_shape:
                raise ValueError(f"{name} must have shape [batch, 2]")
        if torch.any(self.child_levels < 1) or torch.any(self.child_levels > 13):
            raise ValueError("child_levels must be in [1, 13]")


@dataclass(frozen=True)
class TrainingBatch:
    inputs: ModelInputs
    path_group_ids: Tensor
    positive_child: Tensor
    teacher_probabilities: Tensor
    distillation_kind: Tensor

    def to(self, device: torch.device | str) -> "TrainingBatch":
        return TrainingBatch(
            inputs=self.inputs.to(device),
            path_group_ids=self.path_group_ids.to(device),
            positive_child=self.positive_child.to(device),
            teacher_probabilities=self.teacher_probabilities.to(device),
            distillation_kind=self.distillation_kind.to(device),
        )

    def validate(self, maximum_length: int) -> None:
        self.inputs.validate(maximum_length)
        batch_size = self.inputs.user_ids.shape[0]
        if self.path_group_ids.shape != (batch_size,):
            raise ValueError("path_group_ids must have shape [sibling groups]")
        if torch.any(self.path_group_ids < 0):
            raise ValueError("path_group_ids must be non-negative")
        if self.positive_child.shape != (batch_size,):
            raise ValueError("positive_child must have shape [batch]")
        if torch.any((self.positive_child < 0) | (self.positive_child > 1)):
            raise ValueError("positive_child must contain binary sibling indices")
        if self.teacher_probabilities.shape != (batch_size, 2):
            raise ValueError("teacher_probabilities must have shape [batch, 2]")
        if not torch.isfinite(self.teacher_probabilities).all():
            raise ValueError("teacher probabilities must be finite")
        if torch.any(self.teacher_probabilities < 0):
            raise ValueError("teacher probabilities must be non-negative")
        totals = self.teacher_probabilities.sum(dim=-1)
        if not torch.allclose(totals, torch.ones_like(totals), atol=1e-6, rtol=0):
            raise ValueError("teacher sibling probabilities must sum to one")
        if self.distillation_kind.shape != (batch_size,):
            raise ValueError("distillation_kind must have shape [batch]")
        valid = (self.distillation_kind >= DISTILLATION_NONE) & (
            self.distillation_kind <= DISTILLATION_NODE
        )
        if not torch.all(valid):
            raise ValueError("distillation_kind values must be none, item, or node")
        _, inverse, counts = torch.unique(
            self.path_group_ids, sorted=True, return_inverse=True, return_counts=True
        )
        if torch.any(counts != 13):
            raise ValueError(
                "each path group must contain exactly 13 sibling decisions"
            )
        item_counts = torch.zeros_like(counts).scatter_add_(
            0, inverse, (self.distillation_kind == DISTILLATION_ITEM).to(counts.dtype)
        )
        node_counts = torch.zeros_like(counts).scatter_add_(
            0, inverse, (self.distillation_kind == DISTILLATION_NODE).to(counts.dtype)
        )
        if torch.any(item_counts != 1) or torch.any(node_counts != 12):
            raise ValueError(
                "each path must contain one item and twelve internal-node groups"
            )


class TokenGroupNorm(nn.Module):
    """Group-normalize each token independently, preserving causal semantics."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.normalization = nn.GroupNorm(8, hidden_dim)

    def forward(self, values: Tensor) -> Tensor:
        batch, length, width = values.shape
        normalized = self.normalization(values.reshape(batch * length, width, 1))
        return normalized.reshape(batch, length, width)


class GatedSequentialTransductionBlock(nn.Module):
    """A compact causal, gated sequential-transduction block (HSTU-style)."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.input_norm = TokenGroupNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attention_gate = nn.Linear(hidden_dim, hidden_dim)
        self.attention_output = nn.Linear(hidden_dim, hidden_dim)
        self.feedforward_norm = TokenGroupNorm(hidden_dim)
        self.feedforward_input = nn.Linear(hidden_dim, hidden_dim * 2)
        self.feedforward_output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, values: Tensor, valid_tokens: Tensor) -> Tensor:
        batch, length, width = values.shape
        normalized = self.input_norm(values)
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)

        def heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(
                batch, length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        scores = torch.matmul(heads(q), heads(k).transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        causal = torch.ones(
            length, length, dtype=torch.bool, device=values.device
        ).tril()
        allowed = causal[None, None, :, :] & valid_tokens[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        attended = (
            torch.matmul(probabilities, heads(v))
            .transpose(1, 2)
            .reshape(batch, length, width)
        )
        gate = torch.sigmoid(self.attention_gate(normalized))
        values = values + gate * self.attention_output(attended)
        normalized = self.feedforward_norm(values)
        left, right = self.feedforward_input(normalized).chunk(2, dim=-1)
        values = values + self.feedforward_output(F.silu(left) * right)
        return values


class SharedHistoryEmbeddings(nn.Module):
    """Shared item table plus learned stream type projection."""

    def __init__(self, config: PTDModelConfig) -> None:
        super().__init__()
        self.item = nn.Embedding(
            config.item_hash_bucket_size,
            config.hidden_dim,
            padding_idx=0,
            sparse=True,
        )
        self.stream_type = nn.Embedding(2, config.stream_type_embedding_dim)
        self.input_projection = nn.Linear(
            config.hidden_dim + config.stream_type_embedding_dim,
            config.hidden_dim,
        )

    def forward(self, identifiers: Tensor, stream_index: int) -> Tensor:
        item_values = self.item(identifiers)
        stream_ids = torch.full_like(identifiers, stream_index)
        stream_values = self.stream_type(stream_ids)
        return self.input_projection(torch.cat((item_values, stream_values), dim=-1))


def _length_mask(lengths: Tensor, length: int) -> Tensor:
    return torch.arange(length, device=lengths.device)[None, :] < lengths[:, None]


class TwoStreamHSTUStyleEncoder(nn.Module):
    def __init__(
        self, config: PTDModelConfig, histories: SharedHistoryEmbeddings
    ) -> None:
        super().__init__()
        self.config = config
        self.histories = histories
        self.position = nn.Embedding(config.position_buckets, config.hidden_dim)
        self.empty_stream = nn.Parameter(torch.empty(2, config.hidden_dim))
        self.blocks = nn.ModuleList(
            GatedSequentialTransductionBlock(config.hidden_dim, config.num_heads)
            for _ in range(config.num_layers)
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        nn.init.normal_(self.empty_stream, std=0.02)

    def encode_stream(
        self, identifiers: Tensor, lengths: Tensor, stream_index: int
    ) -> Tensor:
        batch, sequence_length = identifiers.shape
        valid = _length_mask(lengths, sequence_length)
        values = self.histories(identifiers, stream_index)
        positions = torch.arange(sequence_length, device=identifiers.device)
        values = values + self.position(positions)[None, :, :]
        empty = lengths == 0
        if torch.any(empty):
            first_position = (
                torch.arange(sequence_length, device=identifiers.device)[None, :] == 0
            )
            selector = empty[:, None] & first_position
            token = (
                self.empty_stream[stream_index][None, None, :]
                + self.position.weight[0][None, None, :]
            )
            values = torch.where(selector[:, :, None], token, values)
            valid = valid | selector
        for block in self.blocks:
            values = block(values, valid)
        final_index = torch.clamp(lengths - 1, min=0)
        return values[torch.arange(batch, device=values.device), final_index]

    def forward(self, inputs: ModelInputs, user_values: Tensor) -> Tensor:
        click = self.encode_stream(inputs.click_history_ids, inputs.click_lengths, 0)
        purchase = self.encode_stream(
            inputs.purchase_history_ids, inputs.purchase_lengths, 1
        )
        return self.fusion(torch.cat((user_values, click, purchase), dim=-1))


class MultiwindowDINEncoder(nn.Module):
    """Node-conditioned two-stream DIN over the seven registered recency windows."""

    def __init__(
        self, config: PTDModelConfig, histories: SharedHistoryEmbeddings
    ) -> None:
        super().__init__()
        self.config = config
        self.histories = histories
        self.empty_stream = nn.Parameter(torch.empty(2, config.hidden_dim))
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.stream_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * len(config.windows), config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.user_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        nn.init.normal_(self.empty_stream, std=0.02)

    def encode_stream(
        self,
        identifiers: Tensor,
        lengths: Tensor,
        stream_index: int,
        target_values: Tensor,
    ) -> Tensor:
        history = self.histories(identifiers, stream_index)
        target = target_values[:, None, :].expand_as(history)
        attention_input = torch.cat(
            (history, target, history - target, history * target), dim=-1
        )
        logits = self.attention(attention_input).squeeze(-1)
        positions = torch.arange(identifiers.shape[1], device=identifiers.device)[
            None, :
        ]
        age = lengths[:, None] - 1 - positions
        valid = _length_mask(lengths, identifiers.shape[1])
        pooled: list[Tensor] = []
        start = 0
        for width in self.config.windows:
            window = valid & (age >= start) & (age < start + width)
            masked_logits = logits.masked_fill(~window, torch.finfo(logits.dtype).min)
            maximum = masked_logits.max(dim=-1, keepdim=True).values
            weights = torch.exp(masked_logits - maximum) * window.to(logits.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            pooled.append(torch.sum(weights[:, :, None] * history, dim=1))
            start += width
        stream = self.stream_fusion(torch.cat(pooled, dim=-1))
        return torch.where(
            (lengths == 0)[:, None],
            self.empty_stream[stream_index][None, :],
            stream,
        )

    def forward(
        self, inputs: ModelInputs, user_values: Tensor, target_values: Tensor
    ) -> Tensor:
        click = self.encode_stream(
            inputs.click_history_ids, inputs.click_lengths, 0, target_values
        )
        purchase = self.encode_stream(
            inputs.purchase_history_ids, inputs.purchase_lengths, 1, target_values
        )
        return self.user_fusion(torch.cat((user_values, click, purchase), dim=-1))


class TreeSiblingScorer(nn.Module):
    def __init__(self, config: PTDModelConfig, item_embedding: nn.Embedding) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = item_embedding
        self.category = nn.Embedding(
            config.category_hash_bucket_size,
            config.hidden_dim,
            padding_idx=0,
            sparse=True,
        )
        layers: list[nn.Module] = []
        previous = config.hidden_dim * 4 + 1
        for width in (256, 128, 64, 32):
            layers.extend((nn.Linear(previous, width), nn.SiLU()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_state: Tensor, inputs: ModelInputs) -> Tensor:
        item = self.item_embedding(inputs.child_item_ids)
        category = self.category(inputs.child_category_ids)
        parent = self.item_embedding(inputs.parent_node_ids)
        level = (
            inputs.child_levels.to(user_state.dtype)[:, :, None]
            / self.config.tree_depth
        )
        features = torch.cat((user_state, item, category, parent, level), dim=-1)
        return self.mlp(features).squeeze(-1)


class PTDTreeModel(nn.Module):
    """Two-stream tree model whose forward signature excludes teacher values."""

    def __init__(self, config: PTDModelConfig, *, encoder: str) -> None:
        super().__init__()
        if encoder not in {"hstu", "din"}:
            raise ValueError("encoder must be 'hstu' or 'din'")
        self.config = config
        self.encoder_name = encoder
        self.histories = SharedHistoryEmbeddings(config)
        self.user = nn.Embedding(
            config.user_hash_bucket_size, config.hidden_dim, sparse=True
        )
        if encoder == "hstu":
            self.encoder = TwoStreamHSTUStyleEncoder(config, self.histories)
        else:
            self.encoder = MultiwindowDINEncoder(config, self.histories)
        self.scorer = TreeSiblingScorer(config, self.histories.item)

    def forward(self, inputs: ModelInputs) -> Tensor:
        inputs.validate(self.config.maximum_length)
        user_values = self.user(inputs.user_ids)
        if self.encoder_name == "hstu":
            state = self.encoder(inputs, user_values)[:, None, :].expand(-1, 2, -1)
        else:
            target_values = self.histories.item(inputs.child_item_ids)
            states = [
                self.encoder(inputs, user_values, target_values[:, sibling, :])
                for sibling in range(2)
            ]
            state = torch.stack(states, dim=1)
        return self.scorer(state, inputs)

    def sparse_parameters(self) -> list[nn.Parameter]:
        return [
            self.histories.item.weight,
            self.user.weight,
            self.scorer.category.weight,
        ]

    def dense_parameters(self) -> list[nn.Parameter]:
        sparse_ids = {id(parameter) for parameter in self.sparse_parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in sparse_ids
        ]


@dataclass(frozen=True)
class LossConfiguration:
    temperature: float
    lambda_item: float
    lambda_node: float
    enable_item: bool
    enable_node: bool

    def __post_init__(self) -> None:
        if self.temperature not in {1.0, 2.0, 4.0}:
            raise ValueError("temperature must be in the registered grid")
        if self.lambda_item not in {0.1, 0.3, 1.0}:
            raise ValueError("lambda_item must be in the registered grid")
        if self.lambda_node not in {0.1, 0.3, 1.0}:
            raise ValueError("lambda_node must be in the registered grid")


def _mean_path_sum(
    values: Tensor, path_group_ids: Tensor, mask: Tensor | None = None
) -> Tensor:
    """Sum sibling-group terms within each depth-13 path, then average paths."""
    _, inverse = torch.unique(path_group_ids, sorted=True, return_inverse=True)
    selected = values if mask is None else values * mask.to(values.dtype)
    path_sums = torch.zeros(
        int(inverse.max().item()) + 1,
        dtype=values.dtype,
        device=values.device,
    ).scatter_add_(0, inverse, selected)
    return path_sums.mean()


def compute_ptd_loss(
    logits: Tensor,
    batch: TrainingBatch,
    configuration: LossConfiguration,
) -> dict[str, Tensor]:
    """Compute supervised path CE plus separate temperature-scaled item/node KL."""
    batch.validate(30)
    if logits.shape != batch.teacher_probabilities.shape:
        raise ValueError("logits must match the binary teacher-probability shape")
    supervised_per_group = F.cross_entropy(
        logits, batch.positive_child, reduction="none"
    )
    supervised = _mean_path_sum(supervised_per_group, batch.path_group_ids)
    log_student = F.log_softmax(logits / configuration.temperature, dim=-1)
    teacher = batch.teacher_probabilities
    per_row_kl = torch.sum(
        torch.xlogy(teacher, teacher) - teacher * log_student, dim=-1
    )
    item_mask = batch.distillation_kind == DISTILLATION_ITEM
    node_mask = batch.distillation_kind == DISTILLATION_NODE
    item_kl = _mean_path_sum(per_row_kl, batch.path_group_ids, item_mask)
    node_kl = _mean_path_sum(per_row_kl, batch.path_group_ids, node_mask)
    scale = configuration.temperature * configuration.temperature
    item_term = (
        configuration.lambda_item * scale * item_kl
        if configuration.enable_item
        else item_kl * 0
    )
    node_term = (
        configuration.lambda_node * scale * node_kl
        if configuration.enable_node
        else node_kl * 0
    )
    total = supervised + item_term + node_term
    return {
        "total": total,
        "supervised": supervised,
        "item_kl": item_kl,
        "node_kl": node_kl,
        "item_term": item_term,
        "node_term": node_term,
    }


def build_registered_model(
    variant: str,
    config: PTDModelConfig | None = None,
) -> tuple[PTDTreeModel, tuple[bool, bool]]:
    if variant not in REGISTERED_VARIANTS:
        raise ValueError(f"unsupported trainable variant: {variant}")
    encoder, item, node = REGISTERED_VARIANTS[variant]
    return PTDTreeModel(config or PTDModelConfig(), encoder=encoder), (item, node)


def configure_determinism(seed: int) -> None:
    if seed not in {16630, 16631, 16632}:
        raise ValueError("seed must be one of the three registered seeds")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class PTDTrainer:
    """Registered two-epoch optimizer loop with sparse/dense parameter separation."""

    def __init__(
        self,
        model: PTDTreeModel,
        loss_configuration: LossConfiguration,
        *,
        device: str = "cpu",
        sparse_learning_rate: float = 0.001,
        dense_learning_rate: float = 0.001,
    ) -> None:
        if sparse_learning_rate != 0.001 or dense_learning_rate != 0.001:
            raise ValueError(
                "registered sparse and dense learning rates are both 0.001"
            )
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_configuration = loss_configuration
        sparse = model.sparse_parameters()
        dense = model.dense_parameters()
        sparse_ids = {id(parameter) for parameter in sparse}
        dense_ids = {id(parameter) for parameter in dense}
        all_ids = {id(parameter) for parameter in model.parameters()}
        if sparse_ids & dense_ids or sparse_ids | dense_ids != all_ids:
            raise ValueError("sparse/dense optimizer partition is not exact")
        self.sparse_optimizer = torch.optim.Adagrad(sparse, lr=sparse_learning_rate)
        self.dense_optimizer = torch.optim.Adam(dense, lr=dense_learning_rate)

    def train_step(self, batch: TrainingBatch) -> dict[str, float]:
        self.model.train()
        batch = batch.to(self.device)
        batch.validate(self.model.config.maximum_length)
        self.sparse_optimizer.zero_grad(set_to_none=True)
        self.dense_optimizer.zero_grad(set_to_none=True)
        losses = compute_ptd_loss(
            self.model(batch.inputs), batch, self.loss_configuration
        )
        losses["total"].backward()
        self.sparse_optimizer.step()
        self.dense_optimizer.step()
        return {name: float(value.detach().cpu()) for name, value in losses.items()}

    def evaluate_batch(self, batch: TrainingBatch) -> dict[str, float]:
        self.model.eval()
        batch = batch.to(self.device)
        with torch.inference_mode():
            losses = compute_ptd_loss(
                self.model(batch.inputs), batch, self.loss_configuration
            )
        return {name: float(value.detach().cpu()) for name, value in losses.items()}

    def fit(
        self, batches: Sequence[TrainingBatch], *, epochs: int = 2
    ) -> list[dict[str, float]]:
        if epochs != 2:
            raise ValueError("the preregistered fit budget is exactly two epochs")
        if not batches:
            raise ValueError("training requires at least one batch")
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            totals: dict[str, float] = {}
            for batch in batches:
                values = self.train_step(batch)
                for name, value in values.items():
                    totals[name] = totals.get(name, 0.0) + value
            history.append(
                {
                    "epoch": float(epoch + 1),
                    **{name: value / len(batches) for name, value in totals.items()},
                }
            )
        return history


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def production_parameter_count(model: PTDTreeModel) -> int:
    """Extrapolate the exact production count when only smoke bucket sizes differ."""
    current = model.config
    registered = PTDModelConfig()
    bucket_delta = (
        registered.item_hash_bucket_size
        - current.item_hash_bucket_size
        + registered.user_hash_bucket_size
        - current.user_hash_bucket_size
        + registered.category_hash_bucket_size
        - current.category_hash_bucket_size
    ) * current.hidden_dim
    return count_trainable_parameters(model) + bucket_delta


def state_sha256(model: nn.Module) -> str:
    """Hash model tensors canonically rather than relying on archive metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().clone().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(bytes(tensor.untyped_storage()))
    return digest.hexdigest()


def save_checkpoint_no_clobber(
    path: Path,
    *,
    model: PTDTreeModel,
    variant: str,
    seed: int,
    history: Sequence[Mapping[str, float]],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract_version": "ptd-trainer-checkpoint/v1",
            "variant": variant,
            "seed": seed,
            "config": model.config.__dict__,
            "state_dict": model.state_dict(),
            "history": list(history),
            "teacher_score_as_serving_feature": False,
        },
        path,
    )
