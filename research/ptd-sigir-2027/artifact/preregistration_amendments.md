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
