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

## A4 - 2026-09-25 JST - raw sequence inventory and exact teacher identity

This amendment was made while no prospective five-day PTD evaluation existed. A read-only inspection used Parquet footer metadata and the non-identifying `snapshot_date` column only; no user, item, sequence, feature, or label payload was persisted locally.

Changes:

1. Fixed all 216 raw source objects by URI, GCS generation, CRC32C, MD5, schema fingerprint, and row count in `verified/raw_input_inventory.json`.
2. Verified a uniform 24-column schema, 23,761,140 total rows, and exact train/validation/test row-count agreement with the already verified ESMM source contract.
3. Registered `click_seq_product_id`, `purchase_seq_product_id`, `first_category`, and point-in-time timestamps as the only additional raw fields used for the HSTU-style encoder and diversity metrics. They must join on existing source keys and cannot change the candidate universe.
4. Fixed the frozen ESMM validation-loss checkpoint by SHA-256 `5a435e4ea2579ca226f26fd8dfa5ad48a7be016f3d1a8e61798ce1b2d6ed1540`.
5. Added both hashes to the fail-closed prospective run contract.

Rationale: the prior source contract fixed row keys and eight ESMM features but did not individually freeze the raw sequence/category shards needed by PTD. The inventory closes that provenance gap without reading a PTD test outcome or changing any registered comparison.

## A5 - 2026-09-25 JST - implementable sequence encoder and catalog envelope

This amendment was made while no prospective five-day PTD evaluation existed and every PTD efficacy claim remained `PENDING`. It follows read-only audits of input fields and item-set overlap; neither audit read outcome columns or teacher scores.

Changes:

1. Recorded the immutable source SQL hash and its actual sequence semantics: separate most-recent-first click and purchase streams, maximum length 30, with no retained per-event timestamp or cross-stream total order.
2. Removed the infeasible per-token time-gap embedding and relative-time claim. Registered independent stream reversal, a shared two-layer/four-head/64-dimensional HSTU-style stack with positional but no time encoding, and a two-stream multiwindow-DIN ablation. Exact parameter equality is not claimed; trainable counts and measured latency must be reported.
3. Audited item identity/category only across all fixed dates. The train, validation, and test unions contain 3,126, 2,815, and 4,592 items; 2,339 test items are absent from train and validation. No identifier or category payload is persisted in the public audit, only counts and SHA-256 fingerprints.
4. Fixed a disclosed transductive catalog envelope containing all 5,584 registered candidate items, ordered by earliest point-in-time category then product ID in a binary depth-13 tree. Each date's immutable item-set hash is an eligibility mask applied before beam search.
5. Restricted alternating reassignment to train-seen items; validation/test-only items remain anchored. Cycle selection is recorded per seed, and one bundle hash covers the fixed/alternating trees and date masks before test scoring.
6. Added a hash-pinned machine-readable method contract and made its source, sequence, item-universe, tree, and mask hashes mandatory in the fail-closed run manifest.

Rationale: the earlier prose assumed timestamps that the frozen input does not contain and did not say how a train-only tree represents later catalog items. Leaving either ambiguity would make the experiment non-reproducible or silently exclude 2,339 items. This amendment makes the retrospective comparison executable while explicitly limiting its external validity; it does not use or reinterpret a PTD result.
