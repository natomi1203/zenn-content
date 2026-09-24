#!/usr/bin/env python3
"""Validate or render the no-submit Vertex PTD preflight CustomJobSpec."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "artifact" / "vertex_preflight_job_spec.json"
EXPECTED_BUNDLE_SHA256 = (
    "6d9da3e853c51511821d450e2a83a0db64ac11375bf2124011e09e4a8a1b2507"
)
EXPECTED_IMAGE = (
    "us-central1-docker.pkg.dev/kauche-app-lab/kauche-app/"
    "tzrec-mmoe-training:1.3.9-cu130-py312"
)
PLACEHOLDER = "__REQUIRED_AT_AUTHORIZED_LAUNCH__"


def _env_map(container: dict[str, Any]) -> dict[str, str]:
    values = container.get("env")
    if not isinstance(values, list):
        raise ValueError("containerSpec.env must be an array")
    result: dict[str, str] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {"name", "value"}:
            raise ValueError("every environment entry must contain name and value")
        if value["name"] in result:
            raise ValueError("duplicate preflight environment variable")
        result[str(value["name"])] = str(value["value"])
    return result


def validate_spec(payload: dict[str, Any], *, rendered: bool) -> dict[str, Any]:
    if set(payload) != {"serviceAccount", "workerPoolSpecs"}:
        raise ValueError(
            "CustomJobSpec must contain exactly serviceAccount and workerPoolSpecs"
        )
    pools = payload["workerPoolSpecs"]
    if not isinstance(pools, list) or len(pools) != 1:
        raise ValueError("preflight must contain exactly one worker pool")
    pool = pools[0]
    if set(pool) != {"machineSpec", "replicaCount", "diskSpec", "containerSpec"}:
        raise ValueError("worker pool fields differ from the locked preflight")
    if pool["machineSpec"] != {
        "machineType": "g2-standard-16",
        "acceleratorType": "NVIDIA_L4",
        "acceleratorCount": 1,
    }:
        raise ValueError("preflight machine/GPU differs from the method contract")
    if pool["replicaCount"] != 1:
        raise ValueError("preflight replica count must equal one")
    if pool["diskSpec"] != {"bootDiskType": "pd-ssd", "bootDiskSizeGb": 200}:
        raise ValueError("preflight disk specification mismatch")
    container = pool["containerSpec"]
    if set(container) != {"imageUri", "command", "args", "env"}:
        raise ValueError("container fields differ from the locked preflight")
    if container["imageUri"] != EXPECTED_IMAGE:
        raise ValueError("preflight image differs from the preregistered runtime")
    if container["command"] != ["bash", "-lc"]:
        raise ValueError("preflight command must use bash -lc")
    if not isinstance(container["args"], list) or len(container["args"]) != 1:
        raise ValueError("preflight must contain one fail-closed shell program")
    program = container["args"][0]
    required_fragments = (
        'test "${PTD_COST_AUTHORIZED}" = "true"',
        'gsutil ls "${PTD_PREFLIGHT_OUTPUT_PREFIX}/**"',
        "sha256sum -c -",
        "python -m unittest discover -s tests -v",
        "python scripts/validate_artifacts.py",
        "python scripts/run_trainer_smoke.py",
        "python scripts/run_alternating_solver_smoke.py",
        "python scripts/run_retrieval_runner_smoke.py",
        "python scripts/run_evaluation_emitter_smoke.py",
        "gsutil -m cp -n",
        "preflight-only:no-ptd-result",
    )
    missing = [fragment for fragment in required_fragments if fragment not in program]
    if missing:
        raise ValueError(f"preflight program is missing safeguards: {missing}")
    if "gcloud ai custom-jobs create" in program:
        raise ValueError("preflight container must not recursively submit jobs")
    env = _env_map(container)
    if set(env) != {
        "PTD_COST_AUTHORIZED",
        "PTD_CODE_BUNDLE_URI",
        "PTD_CODE_BUNDLE_SHA256",
        "PTD_PREFLIGHT_OUTPUT_PREFIX",
    }:
        raise ValueError("preflight environment fields mismatch")
    if env["PTD_CODE_BUNDLE_SHA256"] != EXPECTED_BUNDLE_SHA256:
        raise ValueError("preflight bundle hash mismatch")
    if rendered:
        if env["PTD_COST_AUTHORIZED"] != "true":
            raise ValueError("rendered spec requires explicit cost authorization")
        if payload["serviceAccount"] == PLACEHOLDER:
            raise ValueError("rendered spec requires a service account")
        for name in ("PTD_CODE_BUNDLE_URI", "PTD_PREFLIGHT_OUTPUT_PREFIX"):
            if env[name] == PLACEHOLDER or not env[name].startswith("gs://"):
                raise ValueError(f"rendered spec requires a gs:// value for {name}")
    else:
        if payload["serviceAccount"] != PLACEHOLDER:
            raise ValueError("template service account must remain a placeholder")
        if env["PTD_COST_AUTHORIZED"] != "false":
            raise ValueError("template must fail closed on cost authorization")
        if env["PTD_CODE_BUNDLE_URI"] != PLACEHOLDER:
            raise ValueError("template bundle URI must remain a placeholder")
        if env["PTD_PREFLIGHT_OUTPUT_PREFIX"] != PLACEHOLDER:
            raise ValueError("template output prefix must remain a placeholder")
    return {
        "contract_version": "ptd-vertex-preflight-validation/v1",
        "valid": True,
        "rendered": rendered,
        "machine_type": pool["machineSpec"]["machineType"],
        "accelerator_type": pool["machineSpec"]["acceleratorType"],
        "accelerator_count": pool["machineSpec"]["acceleratorCount"],
        "container_image": container["imageUri"],
        "bundle_sha256": env["PTD_CODE_BUNDLE_SHA256"],
        "paid_job_launched": False,
    }


def render_authorized(
    payload: dict[str, Any],
    *,
    service_account: str,
    bundle_uri: str,
    output_prefix: str,
) -> dict[str, Any]:
    if not service_account or not re.fullmatch(
        r"[^@\s]+@[^@\s]+\.iam\.gserviceaccount\.com", service_account
    ):
        raise ValueError("service_account must be a Google service-account email")
    if not bundle_uri.startswith("gs://") or not output_prefix.startswith("gs://"):
        raise ValueError("bundle URI and output prefix must use gs://")
    rendered = json.loads(json.dumps(payload))
    rendered["serviceAccount"] = service_account
    replacements = {
        "PTD_COST_AUTHORIZED": "true",
        "PTD_CODE_BUNDLE_URI": bundle_uri,
        "PTD_PREFLIGHT_OUTPUT_PREFIX": output_prefix.rstrip("/"),
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
    parser.add_argument("--bundle-uri")
    parser.add_argument("--output-prefix")
    args = parser.parse_args()
    payload = json.loads(args.spec.read_text())
    report = validate_spec(payload, rendered=False)
    if args.render_output is None:
        print(json.dumps(report, sort_keys=True))
        return
    if not args.authorize_cost:
        raise SystemExit(
            "refusing to render a launchable spec without --authorize-cost"
        )
    if args.render_output.exists():
        raise SystemExit(f"refusing to overwrite {args.render_output}")
    rendered = render_authorized(
        payload,
        service_account=args.service_account or "",
        bundle_uri=args.bundle_uri or "",
        output_prefix=args.output_prefix or "",
    )
    args.render_output.parent.mkdir(parents=True, exist_ok=True)
    args.render_output.write_text(json.dumps(rendered, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rendered": str(args.render_output), "paid_job_launched": False}))


if __name__ == "__main__":
    main()
