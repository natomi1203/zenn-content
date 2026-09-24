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

## A6 - 2026-09-25 JST - canonical retrieval-to-evaluation handoff

This amendment was made while no prospective five-day PTD evaluation existed and every PTD efficacy, ablation, and latency claim remained `PENDING`. Only synthetic rows were used to test the emitter.

Changes:

1. Added `artifact/retrieval_observation_row.schema.json` as the exact handoff from retrieval to evaluation: one row per date--user--seed--variant with all registered quality/diversity metrics, one measured latency, and the number of scored candidates.
2. Required identical date--user--seed support across all eight variants, complete three-seed pairing for each date--user, and coverage of all five test dates. Duplicate, missing, non-finite, or out-of-range rows fail closed.
3. Fixed aggregate, per-seed, and per-date quality metrics to arithmetic means over their rows; latency uses the registered linear-interpolated p50/p95; candidates scored uses the arithmetic mean.
4. Required at least the run manifest's preregistered number of measured latency rows for every variant and prohibited automatic emission when the run manifest contains a deviation.
5. Required the emitter to regenerate canonical paired JSONL, all four 10,000-resample contrasts, Holm values, and guardrail flags, then pass the existing evidence-admission checker before atomically publishing outputs.

Rationale: the earlier schemas fixed final evidence but left the retrieval-to-evaluation handoff implicit. The new contract removes discretion in slicing and aggregation and prevents partially paired runs from being promoted. It changes no date, seed, variant, threshold, or observed result.

## A7 - 2026-09-25 JST - executable beam retrieval and secondary metrics

This amendment was made while no prospective five-day PTD evaluation existed and all PTD efficacy, ablation, and latency claims remained `PENDING`. The complete matrix was exercised only with synthetic scorers, labels, clocks, and small-bucket random checkpoints.

Changes:

1. Fixed internal-node identity to `node:<heap_node_id>`, padding identity to `padding:<heap_node_id>`, internal/padding category sentinels to `__internal__`/`__padding__`, and leaf identity/category to the locked product and canonical category. All identities use the preregistered shared hash tables.
2. Fixed beam traversal to binary sibling softmax with every child lacking an eligible descendant masked before normalization, accumulated root-to-leaf log probability, deterministic node-ID tie-breaking, and beam/top-K 600. The HSTU-style user state is cached once per query; the registered DIN baseline remains node-conditioned.
3. Fixed purchase/click NDCG ideals and purchase-recall denominators over all eligible items, not only retrieved items. Purchase AUC uses retrieval rank over every eligible item, with all unretrieved items tied below top-K and value 0.5 for a single-class query.
4. Fixed category coverage@50 to the number of distinct top-50 categories divided by `min(50, eligible distinct categories)`, and max category share@50 to the largest category frequency divided by the returned top-50 length.
5. Required at least 100 serial warm-up queries per variant--seed, timing around retrieval with device synchronization and concurrency one, and at least 1,000 measured rows per variant across the three seeds.
6. Added hash-locked query and 24-entry run-plan schemas. Frozen teacher scores are available to the explicit teacher oracle only; outcome labels are accessed only after retrieval for metric computation.

Rationale: the earlier registration fixed the model, beam width, metric names, and latency thresholds but not every serving and secondary-metric convention. These choices make the runner executable and fail closed without changing an outcome, comparison, or threshold.

## A8 - 2026-09-25 JST - executable validation selection

This amendment was made while no prospective five-day PTD evaluation existed and every PTD efficacy, ablation, and latency claim remained `PENDING`. The selector was tested only with synthetic validation queries, synthetic checkpoints, and an injected deterministic scorer.

Changes:

1. Required the combined PTD validation sweep to contain exactly all 27 `temperature x lambda_item x lambda_node` cells at seed 16630.
2. Clarified the pre-existing 0.001 tie rule: every cell within an absolute 0.001 purchase NDCG@50 of the maximum enters the tie set; selection then minimizes `lambda_item + lambda_node`, temperature, `lambda_item`, and `lambda_node`, in that order.
3. Fixed the validation population to purchase-positive date--user units, matching the registered primary macro metric.
4. Required item-only and node-only fits to use the selected tuple. Their higher validation purchase NDCG@50 wins; exact equality uses lexical variant name.
5. Required an immutable validation-query artifact that reads outcome columns only for the validation shard, plus a hash-linked selection artifact fixed before any PTD test scoring.

Rationale: the original registration declared the grid, seed, 0.001 tie window, and first two tie-break dimensions but did not completely order residual ties or define the executable query artifact. This amendment removes those degrees of freedom without inspecting or changing a PTD result.

## A9 - 2026-09-25 JST - train-only assignment-weight materialization

This amendment was made while no prospective five-day PTD evaluation or real alternating-tree result existed and every efficacy, ablation, and latency claim remained `PENDING`. Only synthetic train queries, a uniform scorer, and a four-item depth-13 tree were used to test the path-to-solver handoff.

Changes:

1. Added an immutable assignment-query artifact covering exactly the three training dates. Outcome columns from validation/test shards are never opened; no validation/test teacher score enters the artifact.
2. Fixed candidate assignment positions to every physical leaf except leaves occupied by anchored non-training items. Currently unused capacity leaves remain valid positions.
3. Fixed assignment path affinity to complete-tree binary sibling softmax at temperature one, without date-eligibility masking. Date masks continue to apply to validation/test retrieval, not to structural candidate leaf positions during reassignment.
4. Required a complete `train_seen_item x available_leaf` rectangular matrix accumulated in float64 from `(purchase_label + frozen_teacher_purchase) * root_to_leaf_log_probability`.
5. Required the matrix manifest to pass the exact anchored solver's existing hash, split, formula, coverage, and no-validation/test checks before reassignment.

Rationale: the registered equation fixed the weight but did not fully specify whether unused leaves were candidates or whether date masks applied to structural paths. These choices make the exact solver handoff reproducible without inspecting an alternating result or changing the registered objective.

## A10 - 2026-09-25 JST - alternating cycle orchestration and tree-lock shape

This amendment was made while no prospective five-day PTD evaluation or real alternating-tree result existed and every efficacy, ablation, and latency claim remained `PENDING`. The complete cycle graph was exercised only on synthetic four-item depth-13 data with reduced hash buckets.

Changes:

1. Required cycles `0,1,2,3` to be completed for each alternating variant and seed; cycle 0 uses the fixed tree, and cycles 1--3 each materialize train-only weights, solve a new anchored tree, rebuild sibling targets, and refit.
2. Fixed refitting to warm-start model parameters from the preceding cycle while resetting both registered optimizers before each exact two-epoch fit.
3. Fixed cycle selection to validation purchase NDCG@50 over purchase-positive date--user units. The maximum wins; exact equality selects the lower cycle.
4. Corrected the run-manifest shape from one cycle count per seed to one count per alternating-variant--seed pair. `alternating_tdm` and `alternating_ptd` may produce different trees and therefore cannot share an implicit count.
5. Required the selected catalog, date masks, fit manifest, checkpoint, and their combined bundle hash to be locked before test scoring.

Rationale: the prior registration allowed up to three cycles and validation-only selection but did not specify warm-start/reset behavior, residual ties, or how two alternating variants map into one run manifest. This amendment removes those ambiguities without reading a PTD outcome.

## A11 - 2026-09-25 JST - staged fit DAG and pre-test bundle lock

This amendment was made while no prospective five-day PTD result existed and every PTD efficacy, ablation, and latency claim remained `PENDING`. The complete assembly path was exercised only with synthetic text artifacts and synthetic retrieval/evaluation records.

Changes:

1. Fixed the staged execution DAG to 27 combined-grid fits at seed 16630, two selected-tuple single-level fits, 12 additional final fixed-tree fits after reusing the three applicable seed-16630 selection fits, and six four-fit alternating chains. This is 65 fit executions and 21 final tree-model artifacts.
2. Required the fit schedule to hash-link the frozen-teacher, fixed training examples, validation queries, train-only assignment queries, fixed catalog, and fixed date-eligibility artifacts before fitting.
3. Required the pre-test retrieval plan to cover exactly eight variants by three seeds and to bind every model manifest, checkpoint, catalog, date mask, query artifact, code revision, hardware/software/timer description, and selected alternating cycle in one combined lock hash.
4. Required the lock timestamp to precede test scoring. The lock phase may inspect manifests and hashes but not the test-query payload or any test outcome.
5. Required the final run manifest to be created only after a complete retrieval-metrics manifest exists, to hash-link the fit schedule, retrieval plan, and retrieval metrics manifest, and to record ordered creation, lock, test-start, and completion timestamps.

Rationale: the earlier contracts specified every component but left the full fit count, reuse rules, and pre-test/post-test assembly boundary implicit. This amendment makes the paid execution graph and evidence handoff auditable without changing a model, comparison, threshold, or observed result.
