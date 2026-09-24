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

This regenerates all LaTeX tables and figures, validates JSON/CSV contracts and claim status, then compiles `paper/main.pdf` with Tectonic. Run the stages separately with `make generate`, `make validate`, and `make paper`.

## Evidence-ingestion gate

To promote PTD results from `PENDING` to `VERIFIED`, all of the following are required:

1. An immutable evaluation JSON conforming to `artifact/evaluation.schema.json`.
2. An artifact manifest conforming to `artifact/manifest.schema.json` with SHA-256 hashes.
3. The preregistered five-date split, three fixed seeds, and no test-driven tree rebuilding or hyperparameter selection.
4. Point-in-time assertions and candidate-universe checks passing.
5. The ledger updated with exact JSON pointers and the manuscript regenerated.

## Submission hygiene

This development repository identifies its owner and therefore cannot be linked from an anonymous SIGIR submission. Create an anonymous snapshot only after the evidence gate passes. Author names, affiliations, acknowledgements, and public artifact links remain placeholders until the review/camera-ready transition.
