#!/usr/bin/env python3
"""Fail-closed checks for the paper artifact and claim status."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str):
    return json.loads((ROOT / path).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest() -> None:
    manifest = load("artifact/manifest.json")
    assert manifest["schema_version"] == "ptd-artifact-manifest/v1"
    assert manifest["paper_id"] == "ptd-sigir-2027"
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["git"]["base_revision"])
    assert {item["status"] for item in manifest["artifacts"]} <= {"VERIFIED", "PREREGISTERED", "PENDING", "GENERATED"}
    assert len({item["id"] for item in manifest["artifacts"]}) == len(manifest["artifacts"])
    assert len({item["path"] for item in manifest["artifacts"]}) == len(manifest["artifacts"])
    for item in manifest["artifacts"]:
        relative = Path(item["path"])
        assert not relative.is_absolute() and ".." not in relative.parts, item["path"]
        if item["status"] == "PENDING":
            assert item["sha256"] is None
            continue
        path = ROOT / relative
        assert path.is_file(), item["path"]
        assert item["sha256"] == sha256(path), item["id"]


def validate_verified_evidence() -> None:
    evidence = load("artifact/verified/legacy_esmm_evidence.json")
    assert evidence["status"] == "VERIFIED"
    assert re.fullmatch(r"[0-9a-f]{64}", evidence["manifest"]["sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", evidence["evaluation"]["sha256"])
    assert len(evidence["split"]["test"]) == 5
    assert evidence["summary"]["test_rows"] == 12_844_800
    assert evidence["summary"]["legacy_esmm"]["purchase_ndcg_at_50"] == 0.1762586881279632
    assert evidence["teacher"]["frozen"] is True
    assert evidence["teacher"]["score"] == "pCTR*pCVR"
    assert evidence["teacher"]["checkpoint"]["sha256"] == (
        "5a435e4ea2579ca226f26fd8dfa5ad48a7be016f3d1a8e61798ce1b2d6ed1540"
    )
    assert "no PTD outcome" in evidence["scope_note"]


def validate_raw_input_inventory() -> None:
    inventory = load("artifact/verified/raw_input_inventory.json")
    assert inventory["contract_version"] == "ptd-raw-input-inventory/v1"
    assert inventory["status"] == "VERIFIED"
    assert inventory["object_count"] == 216
    assert inventory["boundary_shard_count"] == 8
    assert inventory["total_rows"] == 23_761_140
    assert inventory["rows_by_split"] == {"train": 8_367_000, "valid": 2_549_340, "test": 12_844_800}
    assert inventory["checks"] == {
        "uniform_schema": True,
        "required_columns_present": True,
        "row_count_matches_verified_source": True,
        "split_row_counts_match_verified_source": True,
        "row_payloads_persisted_locally": False,
    }
    columns = {field["name"] for field in inventory["schema"]}
    assert set(inventory["required_ptd_columns"]) <= columns
    assert {"click_seq_product_id", "purchase_seq_product_id", "first_category"} <= columns
    objects = inventory["objects"]
    assert len(objects) == inventory["object_count"]
    assert len({item["uri"] for item in objects}) == len(objects)
    assert sum(item["rows"] for item in objects) == inventory["total_rows"]
    assert all(re.fullmatch(r"[0-9]+", item["generation"]) for item in objects)
    serialized = json.dumps(inventory)
    assert '"acl"' not in serialized and '"owner"' not in serialized


def validate_sequence_and_item_contracts() -> None:
    sequence = load("artifact/verified/raw_sequence_contract.json")
    assert sequence["contract_version"] == "ptd-raw-sequence-contract/v1"
    assert sequence["status"] == "VERIFIED"
    assert sequence["source_sql"]["sha256"] == (
        "cda3ca1312d1060bb9690d3a9da0edd763b178ea4914f107ea58f540b9c98f65"
    )
    assert sequence["streams"]["click"]["maximum_length"] == 30
    assert sequence["streams"]["purchase"]["maximum_length"] == 30
    assert sequence["checks"]["per_event_time_gap_encoding_supported"] is False
    assert sequence["checks"]["cross_stream_merge_supported"] is False

    audit = load("artifact/verified/item_universe_audit.json")
    assert audit["contract_version"] == "ptd-item-universe-audit/v1"
    assert audit["status"] == "VERIFIED"
    assert audit["raw_input_inventory"]["sha256"] == sha256(
        ROOT / "artifact" / "verified" / "raw_input_inventory.json"
    )
    assert audit["split_unique_items"] == {"train": 3126, "valid": 2815, "test": 4592}
    assert audit["pretest_unique_items"] == 3245
    assert audit["all_unique_items"] == 5584
    assert audit["overlap"] == {
        "validation_only_vs_train": 119,
        "test_only_vs_train": 2399,
        "test_only_vs_pretest": 2339,
        "test_overlap_pretest": 2253,
    }
    assert audit["catalog_envelope"]["binary_category_order_sha256"] == (
        "dd41695bb4de9a7d09bae0237cdb2f0c5f1a08b572a5647cdba9c5165bb31d61"
    )
    assert audit["category_consistency"] == {
        "items_with_within_date_conflict": 0,
        "items_with_cross_date_conflict": 0,
    }
    assert audit["checks"]["outcomes_or_teacher_scores_read"] is False
    assert audit["checks"]["identifiers_persisted_locally"] is False

    method = load("artifact/preregistered_method.json")
    assert method["contract_version"] == "ptd-preregistered-method/v1"
    assert method["status"] == "PREREGISTERED"
    assert method["result_readout_present_when_frozen"] is False
    assert method["source"]["raw_sequence_contract_sha256"] == sha256(
        ROOT / "artifact" / "verified" / "raw_sequence_contract.json"
    )
    assert method["source"]["item_universe_audit_sha256"] == sha256(
        ROOT / "artifact" / "verified" / "item_universe_audit.json"
    )
    assert method["catalog_and_tree"]["depth"] == 13
    assert method["catalog_and_tree"]["padding_leaf_count"] == 2608
    assert method["hstu_style_encoder"]["use_time_encoding"] is False
    assert method["baseline_encoder"]["parameter_equality_claimed"] is False
    assert sum(method["baseline_encoder"]["windows_most_recent_first"]) == 30


def validate_execution_readiness() -> None:
    audit = load("artifact/verified/execution_readiness_audit.json")
    assert audit["contract_version"] == "ptd-execution-readiness-audit/v1"
    assert audit["status"] == "VERIFIED"
    assert audit["prospective_ptd_result_found"] is False
    assert audit["paid_cloud_job_launched_by_this_audit"] is False
    assert audit["prior_runner"]["successful_tree_sha256"] == (
        "d9d702729230a5dceb103693433585a74f281155da421308fee481af00c236e3"
    )
    assert len(audit["prior_runner"]["source_sha256"]) == 4
    assert len(audit["prior_runner"]["incompatibilities_with_registered_ptd"]) == 6
    assert all(audit["registered_inputs_ready"].values())
    assert audit["ready_for_paid_launch"] is False
    assert sum(audit["production_components_ready"].values()) == 6
    assert audit["production_components_ready"]["per_row_frozen_teacher_score_materializer"] is True
    assert audit["production_components_ready"]["catalog_envelope_and_date_mask_builder"] is True
    assert audit["production_components_ready"]["two_stream_HSTU_and_multiwindow_DIN_trainer"] is True
    assert audit["production_components_ready"]["train_only_anchored_alternating_solver"] is True
    assert audit["production_components_ready"]["paired_JSONL_and_evaluation_emitter"] is True
    assert audit["production_components_ready"]["five_day_three_seed_retrieval_and_latency_runner"] is True
    assert audit["implemented_component_evidence"] == {
        "frozen_teacher_materializer_sha256": sha256(ROOT / "runner" / "materialize_teacher_scores.py"),
        "frozen_teacher_smoke_sha256": sha256(
            ROOT / "artifact" / "smoke" / "frozen_teacher_score_smoke.json"
        ),
        "catalog_bundle_builder_sha256": sha256(ROOT / "runner" / "build_catalog_bundle.py"),
        "catalog_bundle_evidence_sha256": sha256(
            ROOT / "artifact" / "verified" / "catalog_bundle_evidence.json"
        ),
        "ptd_model_trainer_sha256": sha256(ROOT / "runner" / "ptd_model.py"),
        "ptd_trainer_smoke_sha256": sha256(
            ROOT / "artifact" / "smoke" / "trainer_smoke.json"
        ),
        "training_example_builder_sha256": sha256(
            ROOT / "runner" / "build_training_examples.py"
        ),
        "training_example_builder_tests_sha256": sha256(
            ROOT / "tests" / "test_build_training_examples.py"
        ),
        "single_fit_runner_sha256": sha256(ROOT / "runner" / "train_ptd.py"),
        "single_fit_runner_tests_sha256": sha256(ROOT / "tests" / "test_train_ptd.py"),
        "validation_query_builder_sha256": sha256(
            ROOT / "runner" / "build_validation_queries.py"
        ),
        "validation_query_builder_tests_sha256": sha256(
            ROOT / "tests" / "test_build_validation_queries.py"
        ),
        "validation_selector_sha256": sha256(ROOT / "runner" / "select_validation.py"),
        "validation_selector_tests_sha256": sha256(
            ROOT / "tests" / "test_select_validation.py"
        ),
        "assignment_query_builder_sha256": sha256(
            ROOT / "runner" / "build_assignment_queries.py"
        ),
        "assignment_weight_materializer_sha256": sha256(
            ROOT / "runner" / "materialize_assignment_weights.py"
        ),
        "assignment_weight_tests_sha256": sha256(
            ROOT / "tests" / "test_assignment_weights.py"
        ),
        "alternating_solver_sha256": sha256(ROOT / "runner" / "alternating_solver.py"),
        "alternating_solver_smoke_sha256": sha256(
            ROOT / "artifact" / "smoke" / "alternating_solver_smoke.json"
        ),
        "alternating_cycle_selection_schema_sha256": sha256(
            ROOT / "artifact" / "alternating_cycle_selection.schema.json"
        ),
        "alternating_cycle_orchestrator_sha256": sha256(
            ROOT / "runner" / "run_alternating_cycles.py"
        ),
        "alternating_cycle_orchestrator_tests_sha256": sha256(
            ROOT / "tests" / "test_alternating_cycles.py"
        ),
        "evaluation_emitter_sha256": sha256(ROOT / "runner" / "emit_evaluation.py"),
        "evaluation_emitter_smoke_sha256": sha256(
            ROOT / "artifact" / "smoke" / "evaluation_emitter_smoke.json"
        ),
        "retrieval_query_builder_sha256": sha256(
            ROOT / "runner" / "build_retrieval_queries.py"
        ),
        "retrieval_runner_sha256": sha256(ROOT / "runner" / "retrieval_runner.py"),
        "retrieval_runner_smoke_sha256": sha256(
            ROOT / "artifact" / "smoke" / "retrieval_runner_smoke.json"
        ),
    }
    assert audit["production_pipeline_ready"] is False
    assert audit["launch_blockers"] == [
        "assemble_and_validate_the_full_run_manifest_and_retrieval_plan",
        "mint_a_new_deterministic_launch_bundle_from_the_completed_pipeline_revision",
        "obtain_explicit_authorization_for_Vertex_AI_cost",
    ]
    assert "obtain_explicit_authorization_for_Vertex_AI_cost" in audit["launch_blockers"]

    bundle = load("artifact/verified/launch_bundle_evidence.json")
    assert bundle["contract_version"] == "ptd-launch-bundle-evidence/v1"
    assert bundle["status"] == "VERIFIED"
    assert re.fullmatch(r"[0-9a-f]{40}", bundle["code_revision"])
    assert bundle["code_revision"] == "f8158afdc84b8831c1dac6c5c7893e0bec0d5e51"
    assert bundle["bundle"] == {
        "filename": "ptd-sigir-2027-f8158af.tar.gz",
        "sha256": "ce5c399287192ff5a5a723aa7540b4533e0a90f6906f23634c5ecdea32ce5be9",
        "size_bytes": 379782,
        "tar_entries": 91,
    }
    assert bundle["checks"] == {
        "independent_rerun_byte_identical": True,
        "revision_is_full_commit_sha": True,
        "no_git_metadata": True,
        "no_python_cache": True,
        "unit_tests_passed": True,
        "unit_tests_run": 54,
        "unit_tests_skipped_optional_torch": 5,
        "citation_check_passed": True,
        "artifact_validation_passed": True,
        "external_upload_performed": False,
        "paid_cloud_job_launched": False,
    }

    preflight = load("artifact/verified/vertex_preflight_evidence.json")
    assert preflight["contract_version"] == "ptd-vertex-preflight-evidence/v1"
    assert preflight["status"] == "VERIFIED"
    assert preflight["spec_sha256"] == sha256(
        ROOT / "artifact" / "vertex_preflight_job_spec.json"
    )
    assert preflight["validator_sha256"] == sha256(
        ROOT / "scripts" / "validate_vertex_preflight.py"
    )
    assert preflight["tests_sha256"] == sha256(ROOT / "tests" / "test_vertex_preflight.py")
    assert preflight["bundle_sha256"] == bundle["bundle"]["sha256"]
    assert preflight["runtime"] == {
        "project": "kauche-app-lab",
        "region": "us-central1",
        "machine_type": "g2-standard-16",
        "accelerator_type": "NVIDIA_L4",
        "accelerator_count": 1,
        "container_image": (
            "us-central1-docker.pkg.dev/kauche-app-lab/kauche-app/"
            "tzrec-mmoe-training:1.3.9-cu130-py312"
        ),
    }
    assert preflight["checks"]["custom_job_spec_structure_valid"] is True
    assert preflight["checks"]["template_cost_authorization_false"] is True
    assert preflight["checks"]["remote_output_no_clobber"] is True
    assert preflight["checks"]["rendered_launchable_spec_created"] is False
    assert preflight["checks"]["external_upload_performed"] is False
    assert preflight["checks"]["vertex_job_created"] is False
    assert preflight["checks"]["paid_cloud_job_launched"] is False


def validate_production_component_evidence() -> None:
    smoke = load("artifact/smoke/frozen_teacher_score_smoke.json")
    assert smoke["contract_version"] == "ptd-frozen-teacher-smoke/v1"
    assert smoke["status"] == "SMOKE_ONLY"
    assert smoke["empirical_claim_allowed"] is False
    assert smoke["checkpoint_sha256"] == (
        "5a435e4ea2579ca226f26fd8dfa5ad48a7be016f3d1a8e61798ce1b2d6ed1540"
    )
    assert smoke["rows"] == 2
    assert smoke["checks"] == {
        "factorization_matches": True,
        "labels_read": False,
        "no_overwrite": True,
    }
    expected = [0.33971071243286133, 0.00027223789948038757]
    assert all(abs(observed - value) <= smoke["absolute_tolerance"] for observed, value in zip(
        smoke["teacher_purchase"], expected, strict=True
    ))

    catalog = load("artifact/verified/catalog_bundle_evidence.json")
    assert catalog["contract_version"] == "ptd-catalog-bundle-evidence/v1"
    assert catalog["status"] == "VERIFIED"
    assert catalog["raw_input_inventory_sha256"] == sha256(
        ROOT / "artifact" / "verified" / "raw_input_inventory.json"
    )
    assert catalog["builder_sha256"] == sha256(ROOT / "runner" / "build_catalog_bundle.py")
    assert catalog["item_count"] == 5584
    assert catalog["depth"] == 13
    assert catalog["padding_leaf_count"] == 2608
    assert catalog["rows_scanned"] == 23_761_140
    assert catalog["test_only_vs_pretest"] == 2339
    assert catalog["catalog_order_sha256"] == (
        "dd41695bb4de9a7d09bae0237cdb2f0c5f1a08b572a5647cdba9c5165bb31d61"
    )
    assert catalog["internal_artifacts"]["catalog"]["rows"] == 5584
    assert catalog["internal_artifacts"]["complete_binary_nodes"]["rows"] == 16383
    assert catalog["internal_artifacts"]["date_eligibility"]["rows"] == 25394
    assert catalog["deterministic_rerun_verified"] is True
    assert catalog["user_or_outcome_columns_read"] == []
    assert catalog["identifiers_persisted_in_this_evidence"] is False

    trainer = load("artifact/smoke/trainer_smoke.json")
    assert trainer["contract_version"] == "ptd-trainer-smoke/v1"
    assert trainer["status"] == "SMOKE_ONLY"
    assert trainer["empirical_claim_allowed"] is False
    assert trainer["seed"] == 16630
    assert trainer["registered_architecture_retained"] == {
        "din_windows_most_recent_first": [1, 2, 3, 4, 5, 5, 10],
        "hidden_dim": 64,
        "hstu_style_heads": 4,
        "hstu_style_layers": 2,
        "maximum_length": 30,
        "position_buckets": 64,
        "stream_type_embedding_dim": 8,
        "time_encoding": False,
    }
    assert trainer["checks"] == {
        "both_encoders_exercised": True,
        "complete_depth_13_path_groups": True,
        "exact_two_epoch_budget": True,
        "identical_rerun_loss_histories": True,
        "identical_rerun_state_hashes": True,
        "item_and_node_kl_exercised": True,
        "serving_forward_excludes_teacher": True,
    }
    assert [run["variant"] for run in trainer["runs"]] == [
        "ptd_combined",
        "ptd_combined_baseline_encoder",
    ]
    assert [run["trainable_parameters_production_buckets"] for run in trainer["runs"]] == [
        77_637_777,
        77_611_986,
    ]
    assert all(all(run["checks"].values()) for run in trainer["runs"])
    assert all(run["final_loss"]["total"] < run["initial_loss"]["total"] for run in trainer["runs"])

    alternating = load("artifact/smoke/alternating_solver_smoke.json")
    assert alternating["contract_version"] == "ptd-alternating-solver-smoke/v1"
    assert alternating["status"] == "SMOKE_ONLY"
    assert alternating["empirical_claim_allowed"] is False
    assert alternating["synthetic_tree"] == {
        "anchored_items": 2,
        "depth": 3,
        "items": 6,
        "physical_leaves": 8,
        "train_seen_items": 4,
    }
    assert alternating["objective"] == {
        "absolute_gain": 40.0,
        "brute_force_optimum": 40.0,
        "fixed_objective": 0.0,
        "optimized_objective": 40.0,
    }
    assert all(alternating["checks"].values())
    assert alternating["checks"]["anchored_items_unchanged"] is True
    assert alternating["checks"]["matches_brute_force_optimum"] is True
    assert alternating["checks"]["rejects_test_date_weight_manifest"] is True

    emitter = load("artifact/smoke/evaluation_emitter_smoke.json")
    assert emitter["contract_version"] == "ptd-evaluation-emitter-smoke/v1"
    assert emitter["status"] == "SMOKE_ONLY"
    assert emitter["empirical_claim_allowed"] is False
    assert emitter["synthetic_counts"] == {
        "dates": 5,
        "latency_rows_per_variant": 1005,
        "paired_observation_rows": 1005,
        "retrieval_metric_rows": 8040,
        "seeds": 3,
        "users": 67,
        "variants": 8,
    }
    assert all(emitter["checks"].values())
    assert emitter["checks"]["self_admission_passed"] is True
    assert emitter["checks"]["deterministic_evaluation_hash"] is True
    assert emitter["checks"]["deterministic_paired_hash"] is True

    retrieval_schema = load("artifact/retrieval_observation_row.schema.json")
    assert retrieval_schema["type"] == "object"
    assert retrieval_schema["additionalProperties"] is False
    assert set(retrieval_schema["properties"]["date"]["enum"]) == {
        "2026-08-12",
        "2026-08-14",
        "2026-08-25",
        "2026-08-26",
        "2026-08-28",
    }
    assert set(retrieval_schema["properties"]["seed"]["enum"]) == {16630, 16631, 16632}
    assert len(retrieval_schema["properties"]["variant"]["enum"]) == 8

    retrieval_runner = load("artifact/smoke/retrieval_runner_smoke.json")
    assert retrieval_runner["contract_version"] == "ptd-retrieval-runner-smoke/v1"
    assert retrieval_runner["status"] == "SMOKE_ONLY"
    assert retrieval_runner["empirical_claim_allowed"] is False
    assert retrieval_runner["synthetic_counts"] == {
        "dates": 5,
        "measured_rows_per_variant": 1005,
        "queries": 335,
        "retrieval_metric_rows": 8040,
        "seeds": 3,
        "users": 67,
        "variants": 8,
        "warmup_queries_per_variant_seed": 100,
    }
    assert all(retrieval_runner["checks"].values())
    assert retrieval_runner["checks"]["torch_hstu_checkpoint_loaded"] is True
    assert retrieval_runner["checks"]["torch_din_checkpoint_loaded"] is True
    assert retrieval_runner["checks"]["torch_hstu_user_state_cached_across_levels"] is True

    query_schema = load("artifact/retrieval_query_row.schema.json")
    assert query_schema["additionalProperties"] is False
    assert query_schema["properties"]["click_history_most_recent_first"]["maxItems"] == 30
    assert query_schema["properties"]["purchase_history_most_recent_first"]["maxItems"] == 30
    validation_query_schema = load("artifact/validation_query_row.schema.json")
    assert validation_query_schema["additionalProperties"] is False
    assert validation_query_schema["properties"]["date"]["const"] == "2026-07-21"
    validation_selection_schema = load("artifact/validation_selection.schema.json")
    assert validation_selection_schema["additionalProperties"] is False
    assert validation_selection_schema["properties"]["selection_seed"]["const"] == 16630
    assert validation_selection_schema["properties"]["grid_scores"]["minItems"] == 27
    assert validation_selection_schema["properties"]["grid_scores"]["maxItems"] == 27
    assert (
        validation_selection_schema["properties"]["selected_hyperparameters"]
        ["properties"]["tie_window_absolute_ndcg"]["const"]
        == 0.001
    )
    assignment_query_schema = load("artifact/assignment_query_row.schema.json")
    assert assignment_query_schema["additionalProperties"] is False
    assert assignment_query_schema["properties"]["date"]["enum"] == [
        "2026-07-18",
        "2026-07-19",
        "2026-07-20",
    ]
    assignment_weights_schema = load("artifact/assignment_weights.schema.json")
    assert assignment_weights_schema["additionalProperties"] is False
    assert (
        assignment_weights_schema["properties"]["candidate_leaf_policy"]["const"]
        == "all_physical_leaves_except_anchored"
    )
    assert assignment_weights_schema["properties"]["accumulation_dtype"]["const"] == "float64"
    alternating_cycle_schema = load("artifact/alternating_cycle_selection.schema.json")
    assert alternating_cycle_schema["additionalProperties"] is False
    assert alternating_cycle_schema["properties"]["maximum_cycles"]["const"] == 3
    assert alternating_cycle_schema["properties"]["variant"]["enum"] == [
        "alternating_tdm",
        "alternating_ptd",
    ]
    assert alternating_cycle_schema["properties"]["selection_metric"]["const"] == (
        "purchase_ndcg_at_50"
    )
    assert alternating_cycle_schema["properties"]["cycle_tie_break"]["const"] == (
        "metric_desc_then_cycle_asc"
    )
    assert alternating_cycle_schema["properties"]["cycles"]["minItems"] == 4
    assert alternating_cycle_schema["properties"]["cycles"]["maxItems"] == 4
    cycle_prefixes = alternating_cycle_schema["properties"]["cycles"]["prefixItems"]
    assert len(cycle_prefixes) == 4
    for cycle, prefix in enumerate(cycle_prefixes):
        fixed = prefix["allOf"][1]["properties"]
        assert fixed["cycle"]["const"] == cycle
        assert fixed["warm_started_from_cycle"]["const"] == (
            None if cycle == 0 else cycle - 1
        )
    assert alternating_cycle_schema["properties"]["cycles"]["items"] is False
    assert set(alternating_cycle_schema["properties"]["checks"]["required"]) == {
        "all_four_cycles_complete",
        "cycle_zero_fixed_tree",
        "cycles_one_to_three_reassigned",
        "model_parameters_warm_started",
        "optimizers_reset_each_fit",
        "validation_only_cycle_selection",
        "lower_cycle_exact_tie_break",
        "test_queries_not_read",
        "selected_bundle_locked",
        "no_overwrite",
    }
    plan_schema = load("artifact/retrieval_run_plan.schema.json")
    assert plan_schema["properties"]["entries"]["minItems"] == 24
    assert plan_schema["properties"]["entries"]["maxItems"] == 24
    assert plan_schema["properties"]["beam_width"]["const"] == 600
    assert plan_schema["properties"]["top_k"]["const"] == 600


def validate_ledger() -> None:
    with (ROOT / "artifact" / "claim_evidence_ledger.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and len({row["claim_id"] for row in rows}) == len(rows)
    assert {row["status"] for row in rows} <= {"VERIFIED", "PREREGISTERED", "PENDING"}
    pending = [row for row in rows if row["status"] == "PENDING"]
    assert len(pending) >= 4
    tex = (ROOT / "paper" / "main.tex").read_text().lower()
    forbidden = ["ptd significantly improves", "ptd outperforms", "our results show that ptd"]
    assert not any(phrase in tex for phrase in forbidden)
    assert "pending" in tex


def validate_generated() -> None:
    expected = [
        "related_work_table.tex",
        "legacy_evidence_table.tex",
        "claim_status_table.tex",
        "pending_results_table.tex",
        "ptd_architecture.tex",
        "evidence_flow.tex",
    ]
    for name in expected:
        path = ROOT / "paper" / "generated" / name
        assert path.is_file() and path.stat().st_size > 50, name


def validate_reference_smoke() -> None:
    smoke = load("artifact/smoke/reference_smoke_evaluation.json")
    assert smoke["contract_version"] == "ptd-reference-smoke/v1"
    assert smoke["status"] == "SMOKE_ONLY"
    assert smoke["empirical_claim_allowed"] is False
    assert abs(sum(smoke["item_sibling_distribution"].values()) - 1.0) < 1e-9
    assert abs(sum(smoke["node_sibling_distribution"].values()) - 1.0) < 1e-9
    assert smoke["objective"]["total"] >= smoke["objective"]["supervised"]


def validate_evidence_contracts() -> None:
    artifact_schema = load("artifact/manifest.schema.json")
    run_schema = load("artifact/run_manifest.schema.json")
    evaluation_schema = load("artifact/evaluation.schema.json")
    paired_row_schema = load("artifact/paired_observation_row.schema.json")
    for schema in (artifact_schema, run_schema, evaluation_schema, paired_row_schema):
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False
    assert run_schema["properties"]["seeds"]["const"] == [16630, 16631, 16632]
    assert run_schema["properties"]["latency_protocol"]["properties"]["p95_relative_ceiling"]["const"] == 1.2
    assert evaluation_schema["properties"]["guardrails"]["const"]["latency_p95_relative_ceiling"] == 1.2
    assert evaluation_schema["$defs"]["contrast"]["properties"]["bootstrap_resamples"]["const"] == 10_000
    assert evaluation_schema["$defs"]["contrast"]["properties"]["guardrails_pass"]["type"] == "boolean"
    assert evaluation_schema["properties"]["inference"]["const"]["unit"] == "date_user_after_seed_average"
    assert run_schema["properties"]["selected_hyperparameters"]["properties"]["epsilon_item"]["const"] == 1e-6
    assert (
        run_schema["properties"]["source_contract"]["properties"]["raw_input_inventory_sha256"]["const"]
        == sha256(ROOT / "artifact" / "verified" / "raw_input_inventory.json")
    )
    assert (
        run_schema["properties"]["source_contract"]["properties"]["raw_sequence_contract_sha256"]["const"]
        == sha256(ROOT / "artifact" / "verified" / "raw_sequence_contract.json")
    )
    assert (
        run_schema["properties"]["source_contract"]["properties"]["item_universe_audit_sha256"]["const"]
        == sha256(ROOT / "artifact" / "verified" / "item_universe_audit.json")
    )
    assert run_schema["properties"]["method_contract"]["properties"]["sha256"]["const"] == sha256(
        ROOT / "artifact" / "preregistered_method.json"
    )
    assert (
        run_schema["properties"]["teacher"]["properties"]["artifact"]["properties"]["sha256"]["const"]
        == "5a435e4ea2579ca226f26fd8dfa5ad48a7be016f3d1a8e61798ce1b2d6ed1540"
    )
    cycles = run_schema["properties"]["tree"]["properties"][
        "alternating_cycles_selected_by_variant_and_seed"
    ]
    assert set(cycles["required"]) == {"alternating_tdm", "alternating_ptd"}
    assert set(run_schema["$defs"]["cycle_map"]["required"]) == {
        "16630",
        "16631",
        "16632",
    }
    assert all(
        value["minimum"] == 0 and value["maximum"] == 3
        for value in run_schema["$defs"]["cycle_map"]["properties"].values()
    )
    assert run_schema["properties"]["tree"]["properties"]["depth"]["const"] == 13
    assert run_schema["properties"]["tree"]["properties"]["beam_width"]["const"] == 600
    assert set(paired_row_schema["properties"]["scores"]["required"]) == {
        "fixed_tdm",
        "ptd_item",
        "ptd_node",
        "ptd_combined",
        "alternating_tdm",
        "alternating_ptd",
        "ptd_combined_baseline_encoder",
        "teacher_oracle",
    }
    preregistration = (ROOT / "artifact" / "preregistration.md").read_text()
    assert "to be filled" not in preregistration
    assert "scripts/check_evidence_candidate.py" in preregistration
    assert (ROOT / "artifact" / "preregistration_amendments.md").is_file()


if __name__ == "__main__":
    validate_manifest()
    validate_verified_evidence()
    validate_raw_input_inventory()
    validate_sequence_and_item_contracts()
    validate_production_component_evidence()
    validate_execution_readiness()
    validate_ledger()
    validate_generated()
    validate_reference_smoke()
    validate_evidence_contracts()
    tracked = [
        ROOT / "artifact" / "preregistration.md",
        ROOT / "artifact" / "preregistration_amendments.md",
        ROOT / "artifact" / "claim_evidence_ledger.csv",
        ROOT / "artifact" / "verified" / "legacy_esmm_evidence.json",
    ]
    print("artifact validation: PASS")
    for path in tracked:
        print(f"{sha256(path)}  {path.relative_to(ROOT)}")
