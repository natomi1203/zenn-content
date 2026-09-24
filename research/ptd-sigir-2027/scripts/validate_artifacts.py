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
    assert not any(audit["production_components_ready"].values())
    assert "obtain_explicit_authorization_for_Vertex_AI_cost" in audit["launch_blockers"]


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
    cycles = run_schema["properties"]["tree"]["properties"]["alternating_cycles_selected_by_seed"]
    assert set(cycles["required"]) == {"16630", "16631", "16632"}
    assert all(value["maximum"] == 3 for value in cycles["properties"].values())
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
