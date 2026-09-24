# PTD SIGIR 2027 research progress

## Status (2026-09-25 JST)

The repository rules, isolated worktree, paper scaffold, preregistration, claim--evidence ledger, related-work matrix, artifact contracts, and evidence-bounded manuscript are complete. A metadata-only inventory fixes the 216 raw sequence/category shards and exact frozen teacher checkpoint. An outcome-free item audit and a machine-readable method contract now also fix the 5,584-item catalog envelope, date masks, implementable two-stream encoder, and tree/training settings. PTD one-day tree-family diagnostics and the prospective five-day evaluation remain **pending** and are not used for efficacy claims.

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

`artifact/verified/raw_input_inventory.json` records GCS generations, checksums, Parquet footer schemas, and row counts without storing user/item/sequence/label payloads. With read-only GCS credentials and PyArrow, regenerate it using `uv run --with pyarrow python scripts/inventory_raw_inputs.py`; the resulting SHA-256 must remain `d28d69f602b3782f923190768b0d2efc64104c456c9b7b961ea457ed04a31db3`.

`artifact/verified/raw_sequence_contract.json` pins the source SQL and proves that click and purchase histories are separate, most-recent-first, length-30 ID streams without per-event timestamps. `artifact/verified/item_universe_audit.json` stores only counts and hashes: train/test contain 3,126/4,592 unique items, 2,339 test items are absent from train and validation, and the fixed envelope contains 5,584 items. Reproduce the latter with `uv run --with pyarrow python scripts/audit_item_universe.py`; its SHA-256 must remain `11293abab08c86bf386963d92c27daf116612d590c1a54d65771bdc442147415`.

`artifact/preregistered_method.json` is the exact pre-result method contract (SHA-256 `8fd008bcfddfaeda74f6c6cddfab5b644e664bb5aa645ca778a94a788c5fcfee`). It records the no-time-encoding two-stream HSTU-style model, matched-input multiwindow-DIN ablation, binary depth-13 tree, eligibility masks, optimizer/sampling budget, and L4 runtime. Cloud execution is intentionally not launched by this repository workflow without explicit cost authorization.

`artifact/verified/execution_readiness_audit.json` hashes the recovered one-day TDM runner and records why it cannot be relabelled as PTD: its teacher value is static per node, it has one KD term and one history stream, and it lacks the registered catalog masks and five-day paired-evidence emitter. Four of the six production components are now implemented. `runner/materialize_teacher_scores.py` streams the exact frozen checkpoint over label-free feature batches; `artifact/smoke/frozen_teacher_score_smoke.json` records its real-weight compatibility test. `runner/build_catalog_bundle.py` generated the real 5,584-item tree and 25,394 date-eligibility rows; the public, identifier-free proof is `artifact/verified/catalog_bundle_evidence.json`. `runner/ptd_model.py` implements the shared two-stream HSTU-style encoder, the node-conditioned seven-window DIN baseline, the common tree scorer, root-to-leaf aggregation with separate item/node KL terms, and the registered sparse/dense optimizer loop. `artifact/smoke/trainer_smoke.json` proves deterministic two-epoch updates for both encoders with complete depth-13 synthetic paths and reduced hash buckets. `runner/alternating_solver.py` performs exact maximum-weight matching over all non-anchored physical leaves, keeps every non-train item fixed, remaps date masks, and rejects weight manifests containing validation/test evidence. Its smoke artifact matches a brute-force optimum on a small tree. These are implementation checks, not efficacy, real-tree, or production-latency evidence. Two production components remain before a paid launch can be proposed.

The optional restricted-data checks are:

```bash
uv run --with torch --with pyarrow --with numpy python scripts/run_teacher_score_smoke.py --checkpoint /path/to/checkpoint-valid_loss.pt
uv run --with torch --with numpy python scripts/run_trainer_smoke.py
uv run --with torch --with numpy python -m unittest tests.test_ptd_model -v
uv run --with numpy --with scipy --with pyarrow python scripts/run_alternating_solver_smoke.py
uv run --with numpy --with scipy --with pyarrow python -m unittest tests.test_alternating_solver -v
uv run --with pyarrow --with numpy python runner/build_catalog_bundle.py --inventory artifact/verified/raw_input_inventory.json --output-dir /restricted/output/catalog-bundle
uv run --with pyarrow --with numpy python runner/build_catalog_bundle.py --inventory artifact/verified/raw_input_inventory.json --output-dir /restricted/output/catalog-bundle-rerun
python3 scripts/record_catalog_bundle_evidence.py --bundle-dir /restricted/output/catalog-bundle --rerun-bundle-dir /restricted/output/catalog-bundle-rerun
```

## Evidence-ingestion gate

To promote PTD results from `PENDING` to `VERIFIED`, all of the following are required:

1. An immutable evaluation JSON conforming to `artifact/evaluation.schema.json` and a hash-linked JSONL file whose rows conform to `artifact/paired_observation_row.schema.json`.
2. A prospective run manifest conforming to `artifact/run_manifest.schema.json`, plus the research artifact manifest conforming to `artifact/manifest.schema.json`, with the exact raw-input, sequence, item-universe, method-contract, candidate-mask, and frozen-teacher hashes.
3. The preregistered five-date split, three fixed seeds, and no test-driven tree rebuilding or hyperparameter selection.
4. Point-in-time assertions and candidate-universe checks passing.
5. All eight variants, per-seed/per-date metrics, four registered contrasts, and matched latency protocol present; non-empty deviations require manual review.
6. Aggregate, per-seed, and per-date primary metrics, plus contrasts, Holm values, paired-unit counts, and guardrail flags, reproducing exactly from the supplied row-level evidence. A completed guardrail failure remains admissible evidence but cannot support a success claim.
7. The fail-closed checker returning `admissible: true`, followed by outcome-aware ledger updates with exact JSON pointers and manuscript regeneration.

Run the read-only admission check with:

```bash
make check-evidence RUN_MANIFEST=/path/to/run_manifest.json EVALUATION=/path/to/evaluation.json PAIRED_OBSERVATIONS=/path/to/paired_observations.jsonl
```

The checker verifies immutable hash linkage, code revision, source/teacher identity, exact splits and seeds, tree-lock timing, metric completeness, recomputed aggregate/seed/date primary metrics and bootstrap/Holm contrasts, and recomputed quality/diversity/latency guardrail flags. Evidence validity is separate from whether an efficacy claim succeeds. The checker never edits the ledger or manuscript.

## Submission hygiene

This development repository identifies its owner and therefore cannot be linked from an anonymous SIGIR submission. Create an anonymous snapshot only after the evidence gate passes. Affiliations, acknowledgements, and public artifact links remain withheld or placeholders until the review/camera-ready transition.

The paper source records the author as **NAOYUKI TOMITA**, with NAOYUKI as the given name and TOMITA as the family name. The review build keeps ACM's `anonymous=true`, which suppresses that identity in `paper/main.pdf`; `paper/main-author.pdf` displays it. Affiliation remains omitted because it has not yet been supplied.
