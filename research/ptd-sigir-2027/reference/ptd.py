"""Executable reference mathematics for Purchase-aware Tree Distillation.

This module is intentionally dependency-free. It is not the production trainer;
it specifies the transformations that a production implementation must match:
local sibling distributions, internal-node mass aggregation, the PTD objective,
deterministic balanced paths, and capacity-constrained reassignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, isfinite, log
from typing import Iterable, Mapping, Sequence


def softmax(logits: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Return a stable softmax in deterministic key order."""
    if temperature <= 0 or not isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if not logits:
        raise ValueError("logits must not be empty")
    ordered = sorted(logits)
    scaled = {key: float(logits[key]) / temperature for key in ordered}
    if not all(isfinite(value) for value in scaled.values()):
        raise ValueError("all logits must be finite")
    maximum = max(scaled.values())
    weights = {key: exp(value - maximum) for key, value in scaled.items()}
    denominator = sum(weights.values())
    return {key: weights[key] / denominator for key in ordered}


def _logit(probability: float, epsilon: float) -> float:
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    clipped = min(max(float(probability), epsilon), 1.0 - epsilon)
    return log(clipped / (1.0 - clipped))


def item_sibling_distribution(
    purchase_probabilities: Mapping[str, float],
    siblings: Iterable[str],
    *,
    temperature: float = 1.0,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Normalize frozen ESMM purchase logits over one leaf sibling set."""
    sibling_ids = sorted(set(siblings))
    if not sibling_ids:
        raise ValueError("siblings must not be empty")
    missing = [item for item in sibling_ids if item not in purchase_probabilities]
    if missing:
        raise KeyError(f"missing teacher probabilities for {missing}")
    logits = {item: _logit(purchase_probabilities[item], epsilon) for item in sibling_ids}
    return softmax(logits, temperature)


def node_sibling_distribution(
    purchase_probabilities: Mapping[str, float],
    child_descendants: Mapping[str, Iterable[str]],
    eligible_items: Iterable[str],
    *,
    temperature: float = 1.0,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Aggregate frozen descendant purchase mass and normalize across children."""
    if not child_descendants:
        raise ValueError("child_descendants must not be empty")
    eligible = set(eligible_items)
    masses: dict[str, float] = {}
    for child in sorted(child_descendants):
        descendants = set(child_descendants[child]) & eligible
        missing = [item for item in sorted(descendants) if item not in purchase_probabilities]
        if missing:
            raise KeyError(f"missing teacher probabilities for {missing}")
        mass = sum(float(purchase_probabilities[item]) for item in descendants)
        if mass < 0 or not isfinite(mass):
            raise ValueError("teacher mass must be finite and non-negative")
        masses[child] = mass
    logits = {child: log(mass + epsilon) for child, mass in masses.items()}
    return softmax(logits, temperature)


def kl_divergence(teacher: Mapping[str, float], student: Mapping[str, float], epsilon: float = 1e-12) -> float:
    """Compute KL(teacher || student) on exactly matching supports."""
    if set(teacher) != set(student) or not teacher:
        raise ValueError("teacher and student supports must be equal and non-empty")
    if any(value < 0 or not isfinite(value) for value in teacher.values()):
        raise ValueError("teacher probabilities must be finite and non-negative")
    if any(value < 0 or not isfinite(value) for value in student.values()):
        raise ValueError("student probabilities must be finite and non-negative")
    if abs(sum(teacher.values()) - 1.0) > 1e-9 or abs(sum(student.values()) - 1.0) > 1e-9:
        raise ValueError("teacher and student distributions must sum to one")
    return sum(
        teacher[key] * log(max(teacher[key], epsilon) / max(student[key], epsilon))
        for key in sorted(teacher)
        if teacher[key] > 0
    )


def ptd_objective(
    supervised_loss: float,
    *,
    item_teacher: Mapping[str, float],
    item_student_logits: Mapping[str, float],
    node_teacher: Mapping[str, float],
    node_student_logits: Mapping[str, float],
    temperature: float,
    lambda_item: float,
    lambda_node: float,
) -> dict[str, float]:
    """Evaluate Equation 6 from the manuscript for one item and node group."""
    if supervised_loss < 0 or lambda_item < 0 or lambda_node < 0:
        raise ValueError("loss and weights must be non-negative")
    item_student = softmax(item_student_logits, temperature)
    node_student = softmax(node_student_logits, temperature)
    item_kl = kl_divergence(item_teacher, item_student)
    node_kl = kl_divergence(node_teacher, node_student)
    item_term = lambda_item * temperature * temperature * item_kl
    node_term = lambda_node * temperature * temperature * node_kl
    return {
        "supervised": float(supervised_loss),
        "item_kl": item_kl,
        "node_kl": node_kl,
        "item_term": item_term,
        "node_term": node_term,
        "total": float(supervised_loss) + item_term + node_term,
    }


def balanced_paths(items: Iterable[str], branching_factor: int) -> dict[str, tuple[int, ...]]:
    """Assign sorted items to deterministic leaves of the smallest complete b-ary tree."""
    ordered = sorted(set(items))
    if not ordered:
        raise ValueError("items must not be empty")
    if branching_factor < 2:
        raise ValueError("branching_factor must be at least two")
    depth = 0
    capacity = 1
    while capacity < len(ordered):
        capacity *= branching_factor
        depth += 1
    paths: dict[str, tuple[int, ...]] = {}
    for index, item in enumerate(ordered):
        digits = [0] * depth
        value = index
        for position in range(depth - 1, -1, -1):
            digits[position] = value % branching_factor
            value //= branching_factor
        paths[item] = tuple(digits)
    return paths


def ordered_fixed_depth_paths(
    ordered_items: Sequence[str],
    *,
    branching_factor: int,
    depth: int,
) -> dict[str, tuple[int, ...]]:
    """Encode a preordered catalog into fixed-depth paths, leaving tail leaves as padding."""
    if branching_factor < 2:
        raise ValueError("branching_factor must be at least two")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if not ordered_items:
        raise ValueError("ordered_items must not be empty")
    if len(set(ordered_items)) != len(ordered_items):
        raise ValueError("ordered_items must be unique")
    capacity = branching_factor**depth
    if len(ordered_items) > capacity:
        raise ValueError("fixed-depth tree capacity is insufficient")
    paths: dict[str, tuple[int, ...]] = {}
    for index, item in enumerate(ordered_items):
        digits = [0] * depth
        value = index
        for position in range(depth - 1, -1, -1):
            digits[position] = value % branching_factor
            value //= branching_factor
        paths[item] = tuple(digits)
    return paths


@dataclass(frozen=True)
class Assignment:
    item: str
    node: str
    weight: float


def capacity_balanced_assignment(
    weights: Mapping[str, Mapping[str, float]],
    nodes: Sequence[str],
    *,
    capacity: int | None = None,
) -> list[Assignment]:
    """Deterministic greedy reference for train-only capacity-constrained reassignment.

    Production may use an exact matching solver. This routine fixes tie-breaking and
    validates the capacity invariant for unit tests and small examples.
    """
    node_ids = sorted(set(nodes))
    item_ids = sorted(weights)
    if not node_ids or not item_ids:
        raise ValueError("weights and nodes must not be empty")
    limit = capacity if capacity is not None else ceil(len(item_ids) / len(node_ids))
    if limit <= 0 or limit * len(node_ids) < len(item_ids):
        raise ValueError("node capacity is insufficient")
    for item in item_ids:
        if set(weights[item]) != set(node_ids):
            raise ValueError(f"item {item} must have one finite weight for every node")
        if not all(isfinite(float(value)) for value in weights[item].values()):
            raise ValueError("assignment weights must be finite")

    remaining = {node: limit for node in node_ids}
    assignments: list[Assignment] = []
    # Place the most decisive items first; ties resolve by item ID then node ID.
    ranked_items = sorted(
        item_ids,
        key=lambda item: (
            -(
                max(float(value) for value in weights[item].values())
                - min(float(value) for value in weights[item].values())
            ),
            item,
        ),
    )
    for item in ranked_items:
        available = [node for node in node_ids if remaining[node] > 0]
        node = min(available, key=lambda candidate: (-float(weights[item][candidate]), candidate))
        assignments.append(Assignment(item=item, node=node, weight=float(weights[item][node])))
        remaining[node] -= 1
    return sorted(assignments, key=lambda assignment: assignment.item)
