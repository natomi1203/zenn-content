# PTD preregistration amendments

## A1 - 2026-09-25 JST - evidence admission and latency guardrail

This amendment was made while every prospective five-day PTD claim remained `PENDING`; no five-day PTD evaluation is present in the research artifact.

Changes:

1. Replaced the unresolved p95 latency placeholder with a relative ceiling of `1.20 x` the fixed-tree TDM baseline under matched hardware/software, concurrency one, at least 100 warm-up queries, and at least 1,000 measured queries.
2. Fixed the paired bootstrap seed to `20260925`.
3. Made the RQ2 denominator explicit: the better item-only or node-only variant is selected using validation seed 16630 only and locked before test readout.
4. Added exact run-manifest/evaluation schemas and a fail-closed admission checker. Non-empty deviations require manual review and cannot automatically promote a claim.

Rationale: the original text required the latency budget to be set before execution but did not set it. A relative matched-system guardrail is auditable without inventing an unavailable production SLO. This amendment narrows discretion; it does not add or reinterpret an outcome.

## A2 - 2026-09-25 JST - executable estimand and assignment specification

This amendment was made while the prospective five-day evaluation remained absent and every efficacy claim remained `PENDING`.

Changes:

1. Defined the supervised path loss, item/node epsilon values, and the train-only capacity-constrained assignment weight used by the alternating variant.
2. Defined the inferential unit as a date--user pair after averaging the three paired seeds.
3. Fixed within-date resampling, percentile interpolation, the two-sided bootstrap sign p-value, and Holm-adjusted `p < 0.05` as part of the primary success rule.
4. Added dependency-free reference implementations and negative-path tests for ranking metrics, paired bootstrap, Holm adjustment, and guardrails.

Rationale: these details operationalize the existing method and analysis plan so independent implementations can produce the same estimand. They do not select a result, add a test date, or change a pending claim.

## A3 - 2026-09-25 JST - outcome-independent evidence admission

This amendment was made after a read-only artifact inventory confirmed that no prospective five-day PTD evaluation exists. Only the previously disclosed one-day tree-family diagnostics were visible; they were not used to select a five-day result.

Changes:

1. Separated evidence validity from efficacy success. A complete null result or guardrail failure is admissible evidence; it cannot support a positive claim.
2. Added a canonical JSONL row contract containing all eight purchase-NDCG@50 scores for every date--user--seed unit.
3. Required the admission checker to hash and parse that file, recompute aggregate/seed/date primary metrics, all four paired bootstraps, Holm-adjusted p-values, paired-unit counts, and guardrail flags, and reject any mismatch.
4. Required hash-linked validation records for hyperparameter selection and the RQ2 best-single denominator.
5. Clarified that `alternating_cycles_selected` is a cycle count in `[0,3]`, consistent with the original at-most-three-cycle rule.
6. Resolved the guardrail reference unambiguously: every candidate's quality, diversity, concentration, and latency guardrails use fixed-tree TDM.

Rationale: the earlier contract rejected an otherwise valid experiment when a guardrail failed and accepted reported contrast values without row-level recomputation. That would create success bias and weak provenance. This amendment removes both defects without changing the split, variants, metric, bootstrap, or success threshold.
