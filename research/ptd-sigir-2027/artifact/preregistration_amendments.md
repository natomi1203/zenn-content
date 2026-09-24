# PTD preregistration amendments

## A1 - 2026-09-25 JST - evidence admission and latency guardrail

This amendment was made while every prospective five-day PTD claim remained `PENDING`; no five-day PTD evaluation is present in the research artifact.

Changes:

1. Replaced the unresolved p95 latency placeholder with a relative ceiling of `1.20 x` the fixed-tree TDM baseline under matched hardware/software, concurrency one, at least 100 warm-up queries, and at least 1,000 measured queries.
2. Fixed the paired bootstrap seed to `20260925`.
3. Made the RQ2 denominator explicit: the better item-only or node-only variant is selected using validation seed 16630 only and locked before test readout.
4. Added exact run-manifest/evaluation schemas and a fail-closed admission checker. Non-empty deviations require manual review and cannot automatically promote a claim.

Rationale: the original text required the latency budget to be set before execution but did not set it. A relative matched-system guardrail is auditable without inventing an unavailable production SLO. This amendment narrows discretion; it does not add or reinterpret an outcome.
