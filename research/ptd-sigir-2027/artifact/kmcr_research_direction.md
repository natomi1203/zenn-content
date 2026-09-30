# KMCR research direction and preregistered hypothesis

## Status and evidence boundary

This document fixes the next research direction after admission of the RetailRocket 24-cell PTD result. That result is immutable Category C negative/null evidence: at width 1200, true-sum minus max-descendant transaction Recall@600 is -0.0231 (95% CI [-0.0526, 0.00361], Holm-adjusted p=0.378; superiority=false). Width 4800 is not part of that confirmatory result and, if run, is an exploratory sensitivity analysis that cannot change the 24-cell conclusion.

No Kauche label, hidden test, production system, Spanner database, serving path, or A/B experiment was opened to prepare this document. Existing failures and null results remain recorded. Kauche evidence, public-dataset evidence, and implementation-only canaries must remain separately identified.

## Investment allocation

- PTD: 20--30%. Work is limited to evidence preservation, reproducibility, failure analysis, and optionally the separately registered width-4800 sensitivity analysis.
- KMCR: 70--80%. New empirical investment targets recovery of candidates omitted early in retrieval.

This allocation is a research-priority decision, not an efficacy claim.

## Failure mechanism motivating KMCR

The PTD result is consistent with a structural limitation: changing the utility aggregation inside an existing tree can redistribute probability among reachable branches, but cannot recover an item once the branch containing it has been pruned. Increasing the registered search width through 2400 did not establish the preregistered beam-gap-shrinkage mechanism. This interpretation is a hypothesis derived from the negative/null result, not a proven causal explanation.

## Preregistered KMCR hypothesis

Before opening outcome labels, freeze a KMCR candidate-complementation rule that operates outside the PTD tree path and injects a bounded, deduplicated complement set into the candidate union.

Primary hypothesis: under the same point-in-time user population, eligible item universe, frozen ranker, candidate budget, and evaluation protocol, KMCR increases purchase Recall@600 relative to the matched base retriever because complementary candidate injection recovers relevant items absent from the base retriever's pre-union candidate set.

Mechanism hypothesis: the gain is mediated by recovered purchase-positive items that are absent from the base candidate set before union. It is not sufficient for KMCR merely to reorder items already reachable by the base retriever.

## Required contrasts and diagnostics

1. Primary paired contrast: KMCR-union minus matched base purchase Recall@600, evaluated on the preregistered purchase-positive unit with the frozen paired-bootstrap and multiplicity procedure.
2. Recovery contrast: purchase-positive items newly present after KMCR union but absent from the base pre-union candidates.
3. Budget control: compare at the same final 600-candidate budget; deduplicate before deterministic truncation.
4. Guardrails: retain the registered quality, diversity, coverage, and latency checks. A primary gain does not pass if a registered guardrail fails.
5. Attribution: report overlap, KMCR-only, base-only, and neither-retrieved strata without dropping losing seeds, dates, or cohorts.

The exact KMCR generator, complement budget, union ordering, tie-breaking, seeds, dates, hashes, and stopping rules must be fixed in an immutable amendment before a full candidate run. Canary artifacts establish execution only and are not efficacy evidence.

## Fail-closed admission

KMCR evidence is admissible only when all registered arms and paired units are present; immutable source, data, model, and artifact identities match; coverage and candidate-count checks pass; and `test_opened` remains false until the registered opening step. Missing arms, capacity failures, partial unions, or implementation canaries are non-evidence and must remain visible in the ledger.
