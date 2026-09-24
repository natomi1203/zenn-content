# PTD SIGIR 2027 research progress

## Status (2026-09-25 JST)

The repository rules, isolated worktree, paper scaffold, preregistration, claim--evidence ledger, related-work matrix, artifact contracts, and evidence-bounded manuscript are complete. PTD one-day tree-family diagnostics and the prospective five-day evaluation remain **pending** and are not used for efficacy claims.

The official SIGIR 2027 full-paper call is available. It requires English PDF submissions in the current ACM `sigconf` two-column format, at most nine pages excluding references, anonymous review, CCS concepts, and keywords. The dates on the call are labelled "PROPOSED" as of the access date. See `artifact/venue_requirements.md`.

## Evidence policy

`VERIFIED` means that a stable URI, a manifest, a content hash, and a machine-readable evaluation record are available. `PREREGISTERED` is a method or analysis fixed before the prospective PTD five-day readout. `PENDING` must not be converted into a result sentence, abstract claim, conclusion, or table value. The machine-checkable source of truth is `artifact/claim_evidence_ledger.csv`.

The included legacy ESMM anchor is restricted internal evidence. It is useful for defining the teacher and the common point-in-time-safe evaluation universe, but it is not evidence that PTD works. Before anonymous submission, mirror public artifacts to an anonymous repository and redact organization-identifying URIs without changing content hashes.

## Reproduce the manuscript artifact

Requirements: Python 3.10+ and Tectonic (or a TeX Live installation containing `acmart`).

```bash
make all
```

This regenerates all LaTeX tables and figures, runs the dependency-free PTD reference tests and synthetic smoke check, validates citations, JSON/CSV contracts, hashes, and claim status, then compiles both PDFs with Tectonic. `paper/main.pdf` is the anonymous SIGIR review build; `paper/main-author.pdf` visibly identifies **NAOYUKI TOMITA** (given name NAOYUKI, family name TOMITA). Run the stages separately with `make generate`, `make test`, `make smoke`, `make validate`, `make paper-review`, and `make paper-author`.

The Makefile pins `SOURCE_DATE_EPOCH` to the artifact manifest timestamp so repeated builds from identical inputs produce the same PDF hash. Override it only when intentionally minting a new artifact epoch.

`reference/ptd.py` is an executable specification of the paper's local sibling distributions, internal-node teacher aggregation, temperature-scaled KL objective, deterministic balanced paths, and capacity-constrained reassignment. It does not replace the production trainer or constitute empirical evidence.

`reference/evaluation.py` fixes binary NDCG/Recall, the date-stratified paired bootstrap, percentile interval, two-sided bootstrap sign test, Holm adjustment, and quality/diversity/latency guardrails. Its tests use synthetic values only and cannot populate a manuscript result table.

## Evidence-ingestion gate

To promote PTD results from `PENDING` to `VERIFIED`, all of the following are required:

1. An immutable evaluation JSON conforming to `artifact/evaluation.schema.json`.
2. A prospective run manifest conforming to `artifact/run_manifest.schema.json`, plus the research artifact manifest conforming to `artifact/manifest.schema.json`, with SHA-256 hashes.
3. The preregistered five-date split, three fixed seeds, and no test-driven tree rebuilding or hyperparameter selection.
4. Point-in-time assertions and candidate-universe checks passing.
5. All eight variants, per-seed/per-date metrics, four registered contrasts, and matched latency protocol present; non-empty deviations require manual review.
6. The fail-closed checker returning `admissible: true`, followed by ledger updates with exact JSON pointers and manuscript regeneration.

Run the read-only admission check with:

```bash
make check-evidence RUN_MANIFEST=/path/to/run_manifest.json EVALUATION=/path/to/evaluation.json
```

The checker verifies immutable hash linkage, code revision, source/teacher identity, exact splits and seeds, tree-lock timing, metric completeness, bootstrap settings, Holm-adjusted contrasts, and quality/diversity/latency guardrails. It never edits the ledger or manuscript.

## Submission hygiene

This development repository identifies its owner and therefore cannot be linked from an anonymous SIGIR submission. Create an anonymous snapshot only after the evidence gate passes. Affiliations, acknowledgements, and public artifact links remain withheld or placeholders until the review/camera-ready transition.

The paper source records the author as **NAOYUKI TOMITA**, with NAOYUKI as the given name and TOMITA as the family name. The review build keeps ACM's `anonymous=true`, which suppresses that identity in `paper/main.pdf`; `paper/main-author.pdf` displays it. Affiliation remains omitted because it has not yet been supplied.
