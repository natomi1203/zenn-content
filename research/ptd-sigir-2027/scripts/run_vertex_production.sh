#!/usr/bin/env bash
set -euo pipefail

for PTD_REQUIRED_NAME in \
  PTD_COST_AUTHORIZED \
  PTD_CODE_BUNDLE_URI \
  PTD_CODE_BUNDLE_SHA256 \
  PTD_CODE_REVISION \
  PTD_RAW_INPUT_PREFIX \
  PTD_TEACHER_CHECKPOINT_URI \
  PTD_TEACHER_CHECKPOINT_SHA256 \
  PTD_RUN_OUTPUT_PREFIX \
  PTD_RUN_ID; do
  if [[ -z "${!PTD_REQUIRED_NAME:-}" ]]; then
    echo "missing required environment variable: ${PTD_REQUIRED_NAME}" >&2
    exit 10
  fi
done
test "${PTD_COST_AUTHORIZED}" = "true"
test "${PTD_CODE_REVISION}" = "b3f2468ae88d95658c8bfb6d1b13b0ecf31e8093"
test "${PTD_CODE_BUNDLE_SHA256}" = "f1f510cd105b255496e04307714b024a1eb3ab7a082c0b693abaddfe7745542e"
test "${PTD_TEACHER_CHECKPOINT_SHA256}" = "5a435e4ea2579ca226f26fd8dfa5ad48a7be016f3d1a8e61798ce1b2d6ed1540"

if gsutil ls "${PTD_RUN_OUTPUT_PREFIX}/**" >/dev/null 2>&1; then
  echo "refusing non-empty production output prefix" >&2
  exit 20
fi

PTD_WORK_ROOT=/workspace/ptd
PTD_INPUT_ROOT=${PTD_WORK_ROOT}/input
PTD_RUN_ROOT=${PTD_WORK_ROOT}/run
PTD_SOURCE_ROOT=${PTD_WORK_ROOT}/source
PTD_RAW_ROOT=${PTD_INPUT_ROOT}/raw
PTD_TEACHER_CHECKPOINT=${PTD_INPUT_ROOT}/checkpoint-valid_loss.pt
PTD_RAW_GLOB="${PTD_RAW_ROOT}/*.parquet"

mkdir -p "${PTD_SOURCE_ROOT}" "${PTD_RAW_ROOT}" "${PTD_RUN_ROOT}"
gsutil cp "${PTD_CODE_BUNDLE_URI}" "${PTD_WORK_ROOT}/code.tar.gz"
echo "${PTD_CODE_BUNDLE_SHA256}  ${PTD_WORK_ROOT}/code.tar.gz" | sha256sum -c -
tar -xzf "${PTD_WORK_ROOT}/code.tar.gz" -C "${PTD_SOURCE_ROOT}"
gsutil -m cp "${PTD_RAW_INPUT_PREFIX}/*.parquet" "${PTD_RAW_ROOT}/"
PTD_RAW_COUNT=$(find "${PTD_RAW_ROOT}" -maxdepth 1 -type f -name '*.parquet' | wc -l | tr -d ' ')
test "${PTD_RAW_COUNT}" = "216"
gsutil cp "${PTD_TEACHER_CHECKPOINT_URI}" "${PTD_TEACHER_CHECKPOINT}"
echo "${PTD_TEACHER_CHECKPOINT_SHA256}  ${PTD_TEACHER_CHECKPOINT}" | sha256sum -c -

cd "${PTD_SOURCE_ROOT}/research/ptd-sigir-2027"
nvidia-smi
python -c 'import pyarrow, scipy, torch; assert torch.cuda.is_available()'

PTD_CREATED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "${PTD_CREATED_AT}" > "${PTD_RUN_ROOT}/created-at.txt"

python runner/materialize_teacher_scores.py \
  --input "${PTD_RAW_GLOB}" \
  --checkpoint "${PTD_TEACHER_CHECKPOINT}" \
  --output "${PTD_RUN_ROOT}/teacher/scores.parquet" \
  --manifest "${PTD_RUN_ROOT}/teacher/manifest.json" \
  --device cuda

python runner/build_catalog_bundle.py \
  --input "${PTD_RAW_GLOB}" \
  --output-dir "${PTD_RUN_ROOT}/catalog"

python runner/build_training_examples.py \
  --teacher-manifest "${PTD_RUN_ROOT}/teacher/manifest.json" \
  --catalog "${PTD_RUN_ROOT}/catalog/catalog.parquet" \
  --date-eligibility "${PTD_RUN_ROOT}/catalog/date_eligibility.parquet" \
  --output "${PTD_RUN_ROOT}/training/examples.parquet" \
  --manifest "${PTD_RUN_ROOT}/training/manifest.json"

python runner/build_validation_queries.py \
  --teacher-manifest "${PTD_RUN_ROOT}/teacher/manifest.json" \
  --catalog "${PTD_RUN_ROOT}/catalog/catalog.parquet" \
  --date-eligibility "${PTD_RUN_ROOT}/catalog/date_eligibility.parquet" \
  --output "${PTD_RUN_ROOT}/validation/queries.jsonl" \
  --manifest "${PTD_RUN_ROOT}/validation/query-manifest.json"

python runner/build_assignment_queries.py \
  --teacher-manifest "${PTD_RUN_ROOT}/teacher/manifest.json" \
  --catalog "${PTD_RUN_ROOT}/catalog/catalog.parquet" \
  --date-eligibility "${PTD_RUN_ROOT}/catalog/date_eligibility.parquet" \
  --output "${PTD_RUN_ROOT}/alternating/train-queries.jsonl" \
  --manifest "${PTD_RUN_ROOT}/alternating/train-query-manifest.json"

python runner/build_fit_schedule.py \
  --code-revision "${PTD_CODE_REVISION}" \
  --output-root "${PTD_RUN_ROOT}/fits" \
  --teacher-scores-manifest "${PTD_RUN_ROOT}/teacher/manifest.json" \
  --training-examples-manifest "${PTD_RUN_ROOT}/training/manifest.json" \
  --validation-queries-manifest "${PTD_RUN_ROOT}/validation/query-manifest.json" \
  --assignment-queries-manifest "${PTD_RUN_ROOT}/alternating/train-query-manifest.json" \
  --fixed-catalog "${PTD_RUN_ROOT}/catalog/catalog.parquet" \
  --fixed-date-eligibility "${PTD_RUN_ROOT}/catalog/date_eligibility.parquet" \
  --output "${PTD_RUN_ROOT}/fit-schedule.json"

python runner/run_fit_schedule.py \
  --fit-schedule "${PTD_RUN_ROOT}/fit-schedule.json" \
  --validation-selection "${PTD_RUN_ROOT}/validation/selection.json" \
  --output "${PTD_RUN_ROOT}/fit-execution.json" \
  --device cuda

python runner/build_retrieval_queries.py \
  --teacher-manifest "${PTD_RUN_ROOT}/teacher/manifest.json" \
  --output "${PTD_RUN_ROOT}/test/queries.jsonl" \
  --manifest "${PTD_RUN_ROOT}/test/query-manifest.json"

mapfile -t PTD_FIXED_MANIFESTS < <(
  python -c 'import json,sys; print("\n".join(v["path"] for v in json.load(open(sys.argv[1]))["final_fixed_tree_fit_manifests"]))' \
    "${PTD_RUN_ROOT}/fit-execution.json"
)
mapfile -t PTD_ALTERNATING_MANIFESTS < <(
  python -c 'import json,sys; print("\n".join(v["path"] for v in json.load(open(sys.argv[1]))["alternating_cycle_manifests"]))' \
    "${PTD_RUN_ROOT}/fit-execution.json"
)
test "${#PTD_FIXED_MANIFESTS[@]}" = "15"
test "${#PTD_ALTERNATING_MANIFESTS[@]}" = "6"

PTD_LOCKED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
PTD_LOCK_COMMAND=(
  python runner/assemble_run_bundle.py lock
  --code-revision "${PTD_CODE_REVISION}"
  --created-at "${PTD_CREATED_AT}"
  --locked-at "${PTD_LOCKED_AT}"
  --fit-schedule "${PTD_RUN_ROOT}/fit-schedule.json"
  --validation-selection "${PTD_RUN_ROOT}/validation/selection.json"
  --teacher-scores-manifest "${PTD_RUN_ROOT}/teacher/manifest.json"
  --retrieval-queries-manifest "${PTD_RUN_ROOT}/test/query-manifest.json"
  --fixed-catalog "${PTD_RUN_ROOT}/catalog/catalog.parquet"
  --fixed-date-eligibility "${PTD_RUN_ROOT}/catalog/date_eligibility.parquet"
  --device cuda
  --hardware g2-standard-16_NVIDIA-L4x1
  --software tzrec-mmoe-training_1.3.9-cu130-py312
  --timer synchronized_time.perf_counter_ns
  --output "${PTD_RUN_ROOT}/retrieval-plan.json"
)
for PTD_MANIFEST in "${PTD_FIXED_MANIFESTS[@]}"; do
  PTD_LOCK_COMMAND+=(--fit-manifest "${PTD_MANIFEST}")
done
for PTD_MANIFEST in "${PTD_ALTERNATING_MANIFESTS[@]}"; do
  PTD_LOCK_COMMAND+=(--alternating-cycle-manifest "${PTD_MANIFEST}")
done
"${PTD_LOCK_COMMAND[@]}"

PTD_TEST_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python runner/retrieval_runner.py \
  --plan "${PTD_RUN_ROOT}/retrieval-plan.json" \
  --output "${PTD_RUN_ROOT}/retrieval-metrics.jsonl" \
  --manifest "${PTD_RUN_ROOT}/retrieval-metrics-manifest.json"

PTD_COMPLETED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python runner/assemble_run_bundle.py finalize \
  --retrieval-plan "${PTD_RUN_ROOT}/retrieval-plan.json" \
  --retrieval-metrics-manifest "${PTD_RUN_ROOT}/retrieval-metrics-manifest.json" \
  --run-id "${PTD_RUN_ID}" \
  --test-scoring-started-at "${PTD_TEST_STARTED_AT}" \
  --completed-at "${PTD_COMPLETED_AT}" \
  --output "${PTD_RUN_ROOT}/run-manifest.json"

python runner/emit_evaluation.py \
  --run-manifest "${PTD_RUN_ROOT}/run-manifest.json" \
  --retrieval-metrics "${PTD_RUN_ROOT}/retrieval-metrics.jsonl" \
  --paired-output "${PTD_RUN_ROOT}/paired-observations.jsonl" \
  --evaluation-output "${PTD_RUN_ROOT}/evaluation.json" \
  --run-manifest-uri "${PTD_RUN_OUTPUT_PREFIX}/run-manifest.json" \
  --paired-observations-uri "${PTD_RUN_OUTPUT_PREFIX}/paired-observations.jsonl" \
  --generated-at "${PTD_COMPLETED_AT}"

python scripts/check_evidence_candidate.py \
  --run-manifest "${PTD_RUN_ROOT}/run-manifest.json" \
  --evaluation "${PTD_RUN_ROOT}/evaluation.json" \
  --paired-observations "${PTD_RUN_ROOT}/paired-observations.jsonl"

printf '%s\n' "prospective-ptd-five-day" > "${PTD_RUN_ROOT}/SCOPE"
printf '%s\n' "${PTD_CODE_BUNDLE_SHA256}" > "${PTD_RUN_ROOT}/code-bundle-sha256.txt"
gsutil -m cp -n -r "${PTD_RUN_ROOT}/"* "${PTD_RUN_OUTPUT_PREFIX}/"
printf '%s\n' "complete" > "${PTD_WORK_ROOT}/PRODUCTION_SUCCESS"
gsutil cp -n "${PTD_WORK_ROOT}/PRODUCTION_SUCCESS" "${PTD_RUN_OUTPUT_PREFIX}/_SUCCESS"
