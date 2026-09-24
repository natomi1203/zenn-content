# Preregistration: Purchase-aware Tree Distillation

Registered in this repository before inspecting any prospective five-day PTD result. The authors have already seen legacy ranker/ESMM results on the five evaluation dates and one-day tree-family diagnostics. This is therefore an **analysis-plan registration**, not a claim of blinded preregistration. One-day diagnostics may be used only for implementation debugging; they may not support efficacy claims or select a five-day winner.

## Research questions

- **RQ1:** On a fixed balanced tree, does sibling-level distillation from a frozen purchase teacher improve purchase NDCG@50 over an otherwise identical TDM-style student?
- **RQ2:** Are item-sibling and internal-node-sibling signals complementary?
- **RQ3:** Does train-only alternating tree optimization improve PTD without introducing test leakage?
- **RQ4:** Does an HSTU-style action-sequence encoder improve the effectiveness/latency frontier relative to a matched non-HSTU encoder?

## Frozen evidence universe

The source contract is the verified `shared_bottom_esmm_v2_source` manifest. Train dates are 2026-07-18--20, validation is 2026-07-21, and the prospective PTD evaluation dates are 2026-08-12, 08-14, 08-25, 08-26, and 08-28. Features must stop before each snapshot; labels are restricted to `[snapshot, snapshot+24h)`. The candidate universe and row keys must match the source manifest. Users/items may be represented only by information available before the snapshot.

## Teacher and students

The teacher is the frozen `legacy_loss` shared-bottom ESMM selected without PTD test outcomes. Its purchase score is `pCTR * pCVR`. Teacher parameters, calibration, and feature transforms do not update during student or tree training.

All tree students use the same branching factor, depth/capacity policy, beam width, negative sampler, HSTU input sequence, optimizer budget, and seed set `{16630,16631,16632}` unless the factor itself requires a change. The primary comparison is paired by user, date, and seed.

## Registered variants

1. Fixed-tree TDM-style supervised loss, no distillation.
2. Fixed-tree PTD with item-sibling distillation only.
3. Fixed-tree PTD with node-sibling distillation only.
4. Fixed-tree PTD with both sibling losses (primary PTD variant).
5. Train-only alternating TDM/JTM-style tree optimization without distillation.
6. Train-only alternating PTD with both sibling losses.
7. Primary PTD with the HSTU-style encoder replaced by the matched baseline encoder.
8. Teacher-score reranking oracle over the same candidate universe (diagnostic upper anchor, not deployable retrieval).

## Fixed objectives

For a sibling set `S`, temperature `tau`, frozen teacher logits `a_T`, and student logits `a_theta`, define `q_T = softmax(a_T/tau)` and `p_theta = softmax(a_theta/tau)`. Item siblings use direct ESMM purchase logits. An internal child aggregates the frozen teacher mass of eligible descendant items in the training candidate universe, then normalizes across siblings. The student objective is

`L = L_tree + lambda_item tau^2 KL(q_T_item || p_theta_item) + lambda_node tau^2 KL(q_T_node || p_theta_node)`.

`L_tree` is the sum of negative log sibling probabilities along the positive item's root-to-leaf path for purchase-positive training pairs. Item probabilities are clipped with `epsilon_item = 1e-6`; internal descendant masses use `epsilon_node = 1e-12`. A zero-mass child remains in the sibling support with finite near-zero mass.

The primary hyperparameters are selected once on validation from a predeclared grid: `tau in {1,2,4}`, `lambda_item in {0.1,0.3,1.0}`, and `lambda_node in {0.1,0.3,1.0}`. The grid is evaluated only for seed 16630; the selected tuple is then rerun for all three seeds. Ties within 0.001 validation NDCG@50 use the lower total distillation weight, then lower temperature.

## Tree optimization

The fixed tree is a deterministic balanced hierarchy built from train-only item representations. Alternating optimization runs at most three cycles: optimize model parameters, compute train-only assignment weights, solve capacity-constrained item reassignment, then refit. For item `i` and candidate leaf `l`, the assignment weight is the sum over eligible training requests of `(y_ui + q_T_purchase(u,i))` times the model log probability of the candidate leaf path at temperature one. The assignment maximizes total recorded weight subject to one leaf per item and declared subtree capacities. Validation chooses the cycle; the tree is locked before test scoring. No test label, test teacher score, or test metric may affect tree construction, stopping, or model selection.

## Outcomes and tests

- **Primary:** macro user purchase NDCG@50, pooled across the five dates, averaged over the three seeds.
- **Secondary:** Recall@50, NDCG@10/100, purchase AUC, click NDCG@50, category coverage@50, max category share@50, p50/p95 retrieval latency, and candidates scored.
- **Inference:** first average paired treatment-minus-baseline metric differences over the three seeds for each date--user unit. Then resample those units with replacement within each test date, preserving each date's unit count, for 10,000 replicates. Pool the resampled units using their original count weights. Report mean delta, the 2.5th/97.5th percentile interval, and a two-sided bootstrap sign p-value. A user appearing on multiple dates contributes one unit per date. The primary success criterion is an interval lower bound above zero and Holm-adjusted `p < 0.05`, with no registered guardrail failure.
- **Bootstrap determinism:** the bootstrap seed is `20260925`. RQ2 compares combined PTD against the better single-level variant selected on validation seed 16630 only; that denominator is locked before any test readout.
- **Multiplicity:** Holm correction across the four RQ1--RQ4 primary contrasts; uncorrected intervals remain descriptive.
- **Guardrails:** no decrease greater than 5% relative in click NDCG@50 or category coverage@50; no increase greater than 5% relative in max category share@50; p95 retrieval latency must be at most 1.20 times the fixed-tree TDM baseline measured on identical hardware/software with concurrency one, at least 100 warm-up queries, and at least 1,000 measured queries. This is a relative experimental guardrail, not a production SLO.

The admissible evidence shape is frozen in `run_manifest.schema.json` and `evaluation.schema.json`. Automatic claim promotion additionally requires `scripts/check_evidence_candidate.py` to return `admissible: true`; any listed deviation forces manual review and leaves claims pending.

## Exclusions and deviations

Runs with failed point-in-time assertions, duplicate keys, incomplete candidate coverage, non-finite scores, code/hash mismatch, or post-test tree changes are invalid. Every deviation must be listed in the evaluation JSON before interpretation. Missing or invalid variants stay `pending`; they are not silently omitted.

See `preregistration_amendments.md` for the dated amendment record. Git history is the immutable ordering record.
