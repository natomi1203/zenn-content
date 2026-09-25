#!/usr/bin/env python3
"""Validate or render the fail-closed Vertex PTD production CustomJobSpec."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "artifact" / "vertex_production_job_spec.json"
PLACEHOLDER = "__REQUIRED_AT_AUTHORIZED_LAUNCH__"
EXPECTED_IMAGE = (
    "us-central1-docker.pkg.dev/kauche-app-lab/kauche-app/"
    "tzrec-mmoe-training:1.3.9-cu130-py312"
)
EXPECTED_DRIVER_SHA256 = (
    "89a2189acf062fa9cc0e75bf87783ae49d04611b6ce9039c462ebacd3f4efb0c"
)
EXPECTED_BUNDLE_SHA256 = (
    "f1f510cd105b255496e04307714b024a1eb3ab7a082c0b693abaddfe7745542e"
)
EXPECTED_REVISION = "b3f2468ae88d95658c8bfb6d1b13b0ecf31e8093"
EXPECTED_CHECKPOINT_SHA256 = (
    "5a435e4ea2579ca226f26fd8dfa5ad48a7be016f3d1a8e61798ce1b2d6ed1540"
)
EXPECTED_RAW_PREFIX = (
    "gs://kauche-app-lab-product-recommend/home-feed-cvr-lab/offline_gate_full/"
    "f065bd20db-five-day-esmm-cold-start-v1/input"
)
EXPECTED_CHECKPOINT_URI = (
    "gs://kauche-app-lab-product-recommend/home-feed-cvr-lab/offline_gate_full/"
    "f065bd20db-five-day-esmm-cold-start-v1/shared_bottom_esmm_v2/search/"
    "legacy_loss/seed-16630/checkpoint-valid_loss.pt"
)


def _env_map(container: dict[str, Any]) -> dict[str, str]:
    values = container.get("env")
    if not isinstance(values, list):
        raise ValueError("containerSpec.env must be an array")
    result: dict[str, str] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {"name", "value"}:
            raise ValueError("every environment entry must contain name and value")
        name = str(value["name"])
        if name in result:
            raise ValueError("duplicate production environment variable")
        result[name] = str(value["value"])
    return result


def validate_spec(payload: dict[str, Any], *, rendered: bool) -> dict[str, Any]:
    if set(payload) != {"serviceAccount", "workerPoolSpecs"}:
        raise ValueError("CustomJobSpec fields differ from the production contract")
    pools = payload["workerPoolSpecs"]
    if not isinstance(pools, list) or len(pools) != 1:
        raise ValueError("production requires exactly one worker pool")
    pool = pools[0]
    if set(pool) != {"machineSpec", "replicaCount", "diskSpec", "containerSpec"}:
        raise ValueError("worker pool fields differ from the production contract")
    if pool["machineSpec"] != {
        "machineType": "g2-standard-16",
        "acceleratorType": "NVIDIA_L4",
        "acceleratorCount": 1,
    }:
        raise ValueError("production machine/GPU differs from the method contract")
    if pool["replicaCount"] != 1:
        raise ValueError("production replica count must equal one")
    if pool["diskSpec"] != {"bootDiskType": "pd-ssd", "bootDiskSizeGb": 300}:
        raise ValueError("production disk specification mismatch")
    container = pool["containerSpec"]
    if set(container) != {"imageUri", "command", "args", "env"}:
        raise ValueError("production container fields differ from the contract")
    if container["imageUri"] != EXPECTED_IMAGE:
        raise ValueError("production image differs from the registered runtime")
    if container["command"] != ["bash", "-lc"]:
        raise ValueError("production command must use bash -lc")
    if not isinstance(container["args"], list) or len(container["args"]) != 1:
        raise ValueError("production must contain one fail-closed launch program")
    program = container["args"][0]
    required_fragments = (
        'test "${PTD_COST_AUTHORIZED}" = "true"',
        'gsutil ls "${PTD_RUN_OUTPUT_PREFIX}/**"',
        'gsutil cp "${PTD_DRIVER_URI}"',
        'echo "${PTD_DRIVER_SHA256}',
        "sha256sum -c -",
        "chmod 0500",
        "exec /workspace/launch/run_vertex_production.sh",
    )
    missing = [fragment for fragment in required_fragments if fragment not in program]
    if missing:
        raise ValueError(f"production launch program is missing safeguards: {missing}")
    if "gcloud ai custom-jobs create" in program:
        raise ValueError("production container must not recursively submit jobs")
    env = _env_map(container)
    expected_names = {
        "PTD_COST_AUTHORIZED",
        "PTD_DRIVER_URI",
        "PTD_DRIVER_SHA256",
        "PTD_CODE_BUNDLE_URI",
        "PTD_CODE_BUNDLE_SHA256",
        "PTD_CODE_REVISION",
        "PTD_RAW_INPUT_PREFIX",
        "PTD_TEACHER_CHECKPOINT_URI",
        "PTD_TEACHER_CHECKPOINT_SHA256",
        "PTD_RUN_OUTPUT_PREFIX",
        "PTD_RUN_ID",
    }
    if set(env) != expected_names:
        raise ValueError("production environment fields mismatch")
    fixed = {
        "PTD_DRIVER_SHA256": EXPECTED_DRIVER_SHA256,
        "PTD_CODE_BUNDLE_SHA256": EXPECTED_BUNDLE_SHA256,
        "PTD_CODE_REVISION": EXPECTED_REVISION,
        "PTD_RAW_INPUT_PREFIX": EXPECTED_RAW_PREFIX,
        "PTD_TEACHER_CHECKPOINT_URI": EXPECTED_CHECKPOINT_URI,
        "PTD_TEACHER_CHECKPOINT_SHA256": EXPECTED_CHECKPOINT_SHA256,
    }
    if any(env[name] != value for name, value in fixed.items()):
        raise ValueError("production fixed input identity mismatch")
    dynamic_names = (
        "PTD_DRIVER_URI",
        "PTD_CODE_BUNDLE_URI",
        "PTD_RUN_OUTPUT_PREFIX",
        "PTD_RUN_ID",
    )
    if rendered:
        if env["PTD_COST_AUTHORIZED"] != "true":
            raise ValueError("rendered production spec requires cost authorization")
        if payload["serviceAccount"] == PLACEHOLDER:
            raise ValueError("rendered production spec requires a service account")
        for name in dynamic_names[:3]:
            if env[name] == PLACEHOLDER or not env[name].startswith("gs://"):
                raise ValueError(f"rendered production spec requires gs:// for {name}")
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,62})", env["PTD_RUN_ID"]):
            raise ValueError("rendered production run ID is invalid")
    else:
        if payload["serviceAccount"] != PLACEHOLDER:
            raise ValueError("template service account must remain a placeholder")
        if env["PTD_COST_AUTHORIZED"] != "false":
            raise ValueError("template must fail closed on cost authorization")
        if any(env[name] != PLACEHOLDER for name in dynamic_names):
            raise ValueError("template dynamic launch values must remain placeholders")
    return {
        "contract_version": "ptd-vertex-production-validation/v1",
        "valid": True,
        "rendered": rendered,
        "machine_type": pool["machineSpec"]["machineType"],
        "accelerator_type": pool["machineSpec"]["acceleratorType"],
        "accelerator_count": pool["machineSpec"]["acceleratorCount"],
        "container_image": container["imageUri"],
        "driver_sha256": env["PTD_DRIVER_SHA256"],
        "bundle_sha256": env["PTD_CODE_BUNDLE_SHA256"],
        "paid_job_launched": False,
    }


def render_authorized(
    payload: dict[str, Any],
    *,
    service_account: str,
    driver_uri: str,
    bundle_uri: str,
    output_prefix: str,
    run_id: str,
) -> dict[str, Any]:
    if not re.fullmatch(
        r"[^@\s]+@[^@\s]+\.iam\.gserviceaccount\.com", service_account
    ):
        raise ValueError("service_account must be a Google service-account email")
    if any(
        not value.startswith("gs://")
        for value in (driver_uri, bundle_uri, output_prefix)
    ):
        raise ValueError("driver, bundle, and output locations must use gs://")
    rendered = json.loads(json.dumps(payload))
    rendered["serviceAccount"] = service_account
    replacements = {
        "PTD_COST_AUTHORIZED": "true",
        "PTD_DRIVER_URI": driver_uri,
        "PTD_CODE_BUNDLE_URI": bundle_uri,
        "PTD_RUN_OUTPUT_PREFIX": output_prefix.rstrip("/"),
        "PTD_RUN_ID": run_id,
    }
    for value in rendered["workerPoolSpecs"][0]["containerSpec"]["env"]:
        if value["name"] in replacements:
            value["value"] = replacements[value["name"]]
    validate_spec(rendered, rendered=True)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--render-output", type=Path)
    parser.add_argument("--authorize-cost", action="store_true")
    parser.add_argument("--service-account")
    parser.add_argument("--driver-uri")
    parser.add_argument("--bundle-uri")
    parser.add_argument("--output-prefix")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    payload = json.loads(args.spec.read_text())
    report = validate_spec(payload, rendered=False)
    if args.render_output is None:
        print(json.dumps(report, sort_keys=True))
        return
    if not args.authorize_cost:
        raise SystemExit("refusing to render production spec without --authorize-cost")
    if args.render_output.exists():
        raise SystemExit(f"refusing to overwrite {args.render_output}")
    rendered = render_authorized(
        payload,
        service_account=args.service_account or "",
        driver_uri=args.driver_uri or "",
        bundle_uri=args.bundle_uri or "",
        output_prefix=args.output_prefix or "",
        run_id=args.run_id or "",
    )
    args.render_output.parent.mkdir(parents=True, exist_ok=True)
    args.render_output.write_text(json.dumps(rendered, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rendered": str(args.render_output), "paid_job_launched": False}))


if __name__ == "__main__":
    main()
