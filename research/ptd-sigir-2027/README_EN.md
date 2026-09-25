# PTD SIGIR 2027 research progress

## Status (2026-09-25 JST)

The repository rules, isolated worktree, paper scaffold, preregistration, claim--evidence ledger, related-work matrix, artifact contracts, and evidence-bounded manuscript are complete. A metadata-only inventory fixes the 216 raw sequence/category shards and exact frozen teacher checkpoint. An outcome-free item audit and a machine-readable method contract now also fix the 5,584-item catalog envelope, date masks, implementable two-stream encoder, and tree/training settings. PTD one-day tree-family diagnostics and the prospective five-day evaluation remain **pending** and are not used for efficacy claims.

The official SIGIR 2027 full-paper call is available. It requires English PDF submissions in the current ACM `sigconf` two-column format, at most nine pages excluding references, anonymous review, CCS concepts, and keywords. The dates on the call are labelled "PROPOSED" as of the access date. See `artifact/venue_requirements.md`.

## Evidence policy

`VERIFIED` means that a stable URI, a manifest, a content hash, and a machine-readable evaluation record are available. `PREREGISTERED` is a method or analysis fixed before the prospective PTD five-day readout. `PENDING` must not be converted into a result sentence, abstract claim, conclusion, or table value. The machine-checkable source of truth is `artifact/claim_evidence_ledger.csv`.

The included legacy ESMM anchor is restricted internal evidence. It is useful for defining the teacher and the common point-in-time-safe evaluation universe, but it is not evidence that PTD works. Before anonymous submission, mirror public artifacts to an anonymous repository and redact organization-identifying URIs without changing content hashes.

## Objective-conditional aggregation extension

Amendment A14 now records the primary mechanism hypothesis as objective-conditional: max targets best-one retrieval, sum mass targets additive top-$K$ purchase utility, and finite beam may amplify their difference through irreversible early pruning. Sum is not assumed to win universally. P0 requires matched hard/no-KD, leaf-only, max, full-mean, and sum-mass targets plus beam interaction and depth-wise survival. P1 adds cardinality matching, shuffled/uniform/popularity controls, and sparsity-by-teacher-quality analysis. P2 adds log-sum-exp, teacher-quality comparisons, and public-data replication. Existing one-day max/log-sum-exp/top-8-mean diagnostics are context only and cannot satisfy this extension. No extension job may start until exact beam widths, stopping rules, and immutable identities are locked before outcomes.

## Reproduce the manuscript artifact

Requirements: Python 3.10+ and Tectonic (or a TeX Live installation containing `acmart`).

```bash
make all
```

This regenerates all LaTeX tables and figures, runs the dependency-free PTD reference tests and synthetic smoke check, validates citations, JSON/CSV contracts, hashes, and claim status, then compiles both PDFs with Tectonic. `paper/main.pdf` is the anonymous SIGIR review build; `paper/main-author.pdf` visibly identifies **Naoyuki Tomita** (given name Naoyuki, family name Tomita). Run the stages separately with `make generate`, `make test`, `make smoke`, `make validate`, `make paper-review`, and `make paper-author`.

The Makefile pins `SOURCE_DATE_EPOCH` to the artifact manifest timestamp so repeated builds from identical inputs produce the same PDF hash. Override it only when intentionally minting a new artifact epoch.

`reference/ptd.py` is an executable specification of the paper's local sibling distributions, internal-node teacher aggregation, temperature-scaled KL objective, deterministic balanced paths, and capacity-constrained reassignment. It does not replace the production trainer or constitute empirical evidence.

`reference/evaluation.py` fixes binary NDCG/Recall, the date-stratified paired bootstrap, percentile interval, two-sided bootstrap sign test, Holm adjustment, and quality/diversity/latency guardrails. Its tests use synthetic values only and cannot populate a manuscript result table.

`artifact/verified/raw_input_inventory.json` records GCS generations, checksums, Parquet footer schemas, and row counts without storing user/item/sequence/label payloads. With read-only GCS credentials and PyArrow, regenerate it using `uv run --with pyarrow python scripts/inventory_raw_inputs.py`; the resulting SHA-256 must remain `d28d69f602b3782f923190768b0d2efc64104c456c9b7b961ea457ed04a31db3`.

`artifact/verified/raw_sequence_contract.json` pins the source SQL and proves that click and purchase histories are separate, most-recent-first, length-30 ID streams without per-event timestamps. `artifact/verified/item_universe_audit.json` stores only counts and hashes: train/test contain 3,126/4,592 unique items, 2,339 test items are absent from train and validation, and the fixed envelope contains 5,584 items. Reproduce the latter with `uv run --with pyarrow python scripts/audit_item_universe.py`; its SHA-256 must remain `11293abab08c86bf386963d92c27daf116612d590c1a54d65771bdc442147415`.

`artifact/preregistered_method.json` is the exact pre-result method contract (SHA-256 `8fd008bcfddfaeda74f6c6cddfab5b644e664bb5aa645ca778a94a788c5fcfee`). It records the no-time-encoding two-stream HSTU-style model, matched-input multiwindow-DIN ablation, binary depth-13 tree, eligibility masks, optimizer/sampling budget, and L4 runtime. Cloud execution is intentionally not launched by this repository workflow without explicit cost authorization.

`artifact/verified/execution_readiness_audit.json` hashes the recovered one-day TDM runner and records why it cannot be relabelled as PTD. All six core components have deterministic synthetic implementation evidence: frozen teacher materialization; the real 5,584-item catalog and date masks; the shared two-stream HSTU-style/multiwindow-DIN trainer; exact train-only anchored reassignment; five-date/three-seed beam retrieval and latency measurement; and the paired evaluation emitter. `runner/build_training_examples.py` now joins row-aligned raw and frozen-teacher rows, verifies every train/validation candidate set against its locked date mask, emits one complete depth-13 group per observed purchase, and opens no outcome column from test shards. `runner/train_ptd.py` consumes that immutable artifact for one registered variant/seed, enforces batch size 64 and exactly two epochs, and emits a no-clobber checkpoint plus validation-loss audit. `runner/build_assignment_queries.py` and `runner/materialize_assignment_weights.py` isolate the three train dates and emit the complete float64 train-item by free-leaf matrix accepted by the exact anchored solver. `runner/run_alternating_cycles.py` completes cycles 0--3 for one alternating variant/seed, warm-starts model parameters while resetting both optimizers, selects by validation purchase NDCG@50 with a lower-cycle exact-tie rule, and locks the selected bundle without opening test queries. `runner/build_fit_schedule.py` fixes the complete 65-fit staged DAG and hash-links all pre-test inputs; `runner/run_fit_schedule.py` executes that DAG, including validation-only grid preselection before the two single-level fits, and emits the exact 15 fixed-tree plus six alternating manifests without opening test queries. `runner/assemble_run_bundle.py` verifies those artifacts, locks the exact 24-cell retrieval plan before test scoring, and finalizes the run manifest only after complete retrieval metrics exist. `runner/build_retrieval_queries.py` and `runner/retrieval_runner.py` cover the test-side handoff and beam/latency execution. These are implementation checks, not efficacy, real-tree, or production-latency evidence.

Paid execution is authorized only in `kauche-app-lab`. The complete 65-fit executor is frozen in the `b3f2468` bundle, which completed its exact synthetic L4 Vertex preflight as CustomJob `215211130346274816`; its GCS output includes `_SUCCESS` and the explicit scope marker `preflight-only:no-ptd-result`. The first production job encountered regional L4 insufficiency before program start and was cancelled with an empty output prefix. The hash-identical Flex Start replacement, CustomJob `5503281517809369088`, subsequently failed before training and left the immutable output prefix empty. The driver copied only two repeated Parquet basenames from the registered 216-object inventory before its exact-count check terminated; this operational failure is not empirical evidence. No empirical readout is promoted until a complete immutable output passes admission.

`artifact/verified/launch_bundle_evidence.json` records a byte-identical two-run `git archive`/`gzip -n` bundle for completed pipeline revision `b3f2468ae88d95658c8bfb6d1b13b0ecf31e8093` (SHA-256 `f1f510cd105b255496e04307714b024a1eb3ab7a082c0b693abaddfe7745542e`). The extracted bundle passed all 72 tests with the optional dependencies installed, citation checks, and artifact validation, then was uploaded once with no-clobber semantics and passed its exact synthetic Vertex preflight. `artifact/verified/vertex_preflight_evidence.json` records the immutable job and output-object identities.

The bridge and single-fit commands are intentionally separate and fail closed on existing outputs:

```bash
uv run --with pyarrow --with numpy python runner/build_training_examples.py \
  --teacher-manifest /restricted/teacher/manifest.json \
  --catalog /restricted/catalog/catalog.parquet \
  --date-eligibility /restricted/catalog/date_eligibility.parquet \
  --output /restricted/training/examples.parquet \
  --manifest /restricted/training/manifest.json

uv run --with torch --with pyarrow --with numpy python runner/train_ptd.py \
  --examples-manifest /restricted/training/manifest.json \
  --variant ptd_combined --seed 16630 \
  --temperature 2 --lambda-item 0.3 --lambda-node 0.3 \
  --checkpoint /restricted/fits/ptd_combined-16630/checkpoint.pt \
  --manifest /restricted/fits/ptd_combined-16630/manifest.json \
  --device cuda

uv run --with pyarrow --with numpy python runner/build_validation_queries.py \
  --teacher-manifest /restricted/teacher/manifest.json \
  --catalog /restricted/catalog/catalog.parquet \
  --date-eligibility /restricted/catalog/date_eligibility.parquet \
  --output /restricted/validation/queries.jsonl \
  --manifest /restricted/validation/query-manifest.json

uv run --with torch --with pyarrow --with numpy python runner/select_validation.py \
  --validation-queries-manifest /restricted/validation/query-manifest.json \
  --catalog /restricted/catalog/catalog.parquet \
  --date-eligibility /restricted/catalog/date_eligibility.parquet \
  --fit-manifest-glob '/restricted/fits/*/manifest.json' \
  --output /restricted/validation/selection.json \
  --device cuda

uv run --with pyarrow --with numpy python runner/build_assignment_queries.py \
  --teacher-manifest /restricted/teacher/manifest.json \
  --catalog /restricted/catalog/catalog.parquet \
  --date-eligibility /restricted/catalog/date_eligibility.parquet \
  --output /restricted/alternating/train-queries.jsonl \
  --manifest /restricted/alternating/train-query-manifest.json

uv run --with torch --with pyarrow --with numpy --with scipy \
  python runner/materialize_assignment_weights.py \
  --assignment-queries-manifest /restricted/alternating/train-query-manifest.json \
  --catalog /restricted/catalog/catalog.parquet \
  --date-eligibility /restricted/catalog/date_eligibility.parquet \
  --fit-manifest /restricted/fits/alternating_ptd-16630/manifest.json \
  --output /restricted/alternating/weights.parquet \
  --manifest /restricted/alternating/weight-manifest.json \
  --device cuda

uv run --with numpy --with scipy --with pyarrow python runner/alternating_solver.py \
  --catalog /restricted/catalog/catalog.parquet \
  --date-eligibility /restricted/catalog/date_eligibility.parquet \
  --weights /restricted/alternating/weights.parquet \
  --weight-manifest /restricted/alternating/weight-manifest.json \
  --output-dir /restricted/alternating/tree-cycle-1

uv run --with torch --with pyarrow --with numpy --with scipy \
  python runner/run_alternating_cycles.py \
  --teacher-manifest /restricted/teacher/manifest.json \
  --assignment-queries-manifest /restricted/alternating/train-query-manifest.json \
  --validation-queries-manifest /restricted/validation/query-manifest.json \
  --initial-catalog /restricted/catalog/catalog.parquet \
  --initial-date-eligibility /restricted/catalog/date_eligibility.parquet \
  --variant alternating_ptd --seed 16630 \
  --temperature 2 --lambda-item 0.3 --lambda-node 0.3 \
  --output-dir /restricted/alternating/alternating_ptd-16630 \
  --device cuda
```

The individual fit and solver commands are component-level entry points. The cycle orchestrator performs the registered four-fit chain for one alternating variant/seed, not the complete prospective experiment. The hyperparameter selector still requires all 27 combined-grid manifests plus item-only and node-only manifests at the selected tuple before all six alternating chains are scheduled.

The full execution boundary is materialized in four fail-closed stages. The fit executor consumes the locked schedule and emits an inventory conforming to `artifact/fit_execution.schema.json`. The lock command then requires exactly 15 final fixed-tree fit manifests and six alternating-cycle manifests from that inventory; repeat the corresponding flags for every registered cell.

```bash
python3 runner/build_fit_schedule.py \
  --code-revision <40-hex-revision> \
  --output-root /restricted/ptd-run \
  --teacher-scores-manifest /restricted/teacher/manifest.json \
  --training-examples-manifest /restricted/training/manifest.json \
  --validation-queries-manifest /restricted/validation/query-manifest.json \
  --assignment-queries-manifest /restricted/alternating/train-query-manifest.json \
  --fixed-catalog /restricted/catalog/catalog.parquet \
  --fixed-date-eligibility /restricted/catalog/date_eligibility.parquet \
  --output /restricted/ptd-run/fit-schedule.json

uv run --with torch --with pyarrow --with numpy --with scipy \
  python runner/run_fit_schedule.py \
  --fit-schedule /restricted/ptd-run/fit-schedule.json \
  --validation-selection /restricted/validation/selection.json \
  --output /restricted/ptd-run/fit-execution.json \
  --device cuda

uv run --with pyarrow python runner/assemble_run_bundle.py lock \
  --code-revision <40-hex-revision> \
  --created-at <ISO-8601> --locked-at <ISO-8601> \
  --fit-schedule /restricted/ptd-run/fit-schedule.json \
  --validation-selection /restricted/validation/selection.json \
  --teacher-scores-manifest /restricted/teacher/manifest.json \
  --retrieval-queries-manifest /restricted/test/query-manifest.json \
  --fixed-catalog /restricted/catalog/catalog.parquet \
  --fixed-date-eligibility /restricted/catalog/date_eligibility.parquet \
  --fit-manifest <repeat-15-times> \
  --alternating-cycle-manifest <repeat-6-times> \
  --device cuda --hardware <matched-hardware> \
  --software <locked-runtime> --timer <synchronized-timer> \
  --output /restricted/ptd-run/retrieval-plan.json

uv run --with pyarrow python runner/assemble_run_bundle.py finalize \
  --retrieval-plan /restricted/ptd-run/retrieval-plan.json \
  --retrieval-metrics-manifest /restricted/ptd-run/retrieval-metrics-manifest.json \
  --run-id <run-id> --test-scoring-started-at <ISO-8601> \
  --completed-at <ISO-8601> \
  --output /restricted/ptd-run/run-manifest.json
```

`artifact/vertex_preflight_job_spec.json` remains a non-launchable Vertex `CustomJobSpec` template for synthetic runtime compatibility checks. It fixes the registered L4 machine and container, exact bundle hash, empty-output/no-clobber checks, and an in-container `PTD_COST_AUTHORIZED=true` gate; service account and GCS locations remain placeholders and authorization defaults to false. Validate it safely with `make vertex-preflight-validate`. `scripts/validate_vertex_preflight.py` renders a spec only with the explicit `--authorize-cost` flag and concrete service-account/GCS values; it never calls `gcloud` or submits a job. The authorized rendered instance was used for the verified preflight and is retained outside the source artifact because it contains launch-specific locations.

`artifact/vertex_production_job_spec.json` and `scripts/run_vertex_production.sh` define the separate production boundary. The template defaults to cost authorization false and requires exact hashes for the driver, code bundle, raw prefix, and frozen teacher; a rendered instance also requires an empty unique GCS output prefix and Lab service account. The driver completes pre-test fitting before test-query materialization, locks the retrieval plan before scoring, runs the evidence-admission gate, and publishes with no-clobber semantics before `_SUCCESS`. Validate or render it with `scripts/validate_vertex_production.py`; the validator itself never submits a job.

The optional restricted-data checks are:

```bash
uv run --with torch --with pyarrow --with numpy python scripts/run_teacher_score_smoke.py --checkpoint /path/to/checkpoint-valid_loss.pt
uv run --with torch --with numpy python scripts/run_trainer_smoke.py
uv run --with torch --with numpy python -m unittest tests.test_ptd_model -v
uv run --with numpy --with scipy --with pyarrow python scripts/run_alternating_solver_smoke.py
uv run --with numpy --with scipy --with pyarrow python -m unittest tests.test_alternating_solver -v
uv run --with torch --with pyarrow --with numpy --with scipy --with jsonschema python -m unittest tests.test_alternating_cycles tests.test_train_ptd -v
uv run --with pyarrow --with jsonschema python -m unittest tests.test_run_bundle_assembler -v
python3 scripts/run_evaluation_emitter_smoke.py
python3 -m unittest tests.test_evaluation_emitter -v
uv run --with torch --with pyarrow --with numpy python scripts/run_retrieval_runner_smoke.py
uv run --with torch --with pyarrow --with numpy python -m unittest tests.test_build_retrieval_queries tests.test_retrieval_runner -v
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

The paper source records the author as **Naoyuki Tomita**, with Naoyuki as the given name and Tomita as the family name. The author build lists **X Pipelines Co., Ltd., Ho Chi Minh City, Vietnam** as the current affiliation and states in the acknowledgments that the work was initiated during a one-month research visit to Fudan University. Fudan University is not listed as an affiliation. The review build keeps ACM's `anonymous=true` and omits the author identity, affiliation, and acknowledgment from `paper/main.pdf`; `paper/main-author.pdf` displays them.
