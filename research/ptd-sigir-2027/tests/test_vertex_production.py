from __future__ import annotations

import json
import unittest
from pathlib import Path

from runner.materialize_teacher_scores import sha256
from scripts.validate_vertex_production import (
    DEFAULT_SPEC,
    EXPECTED_BUNDLE_SHA256,
    EXPECTED_DRIVER_SHA256,
    render_authorized,
    validate_spec,
)


class VertexProductionTest(unittest.TestCase):
    def payload(self):
        return json.loads(DEFAULT_SPEC.read_text())

    def test_template_and_driver_are_fail_closed(self) -> None:
        report = validate_spec(self.payload(), rendered=False)
        self.assertTrue(report["valid"])
        self.assertFalse(report["rendered"])
        self.assertEqual(report["bundle_sha256"], EXPECTED_BUNDLE_SHA256)
        self.assertEqual(report["driver_sha256"], EXPECTED_DRIVER_SHA256)
        driver = Path(__file__).parents[1] / "scripts" / "run_vertex_production.sh"
        self.assertEqual(sha256(driver), EXPECTED_DRIVER_SHA256)
        program = driver.read_text()
        for fragment in (
            "runner/run_fit_schedule.py",
            "runner/build_retrieval_queries.py",
            "runner/assemble_run_bundle.py lock",
            "runner/retrieval_runner.py",
            "runner/emit_evaluation.py",
            "scripts/check_evidence_candidate.py",
            "d28d69f602b3782f923190768b0d2efc64104c456c9b7b961ea457ed04a31db3",
            'uris.append(f"{uri}#{generation}")',
            'gsutil -m cp -I "${PTD_RAW_ROOT}/"',
            'gsutil -m cp -n -r "${PTD_RUN_ROOT}/"*',
        ):
            self.assertIn(fragment, program)

    def test_authorized_render_requires_exact_dynamic_values(self) -> None:
        rendered = render_authorized(
            self.payload(),
            service_account=(
                "product-recommend-pipelines@kauche-app-lab.iam.gserviceaccount.com"
            ),
            driver_uri="gs://bucket/driver.sh",
            bundle_uri="gs://bucket/code.tar.gz",
            output_prefix="gs://bucket/run-1/",
            run_id="ptd-five-day-b3f2468",
        )
        report = validate_spec(rendered, rendered=True)
        self.assertTrue(report["rendered"])
        with self.assertRaises(ValueError):
            render_authorized(
                self.payload(),
                service_account="invalid@example.com",
                driver_uri="gs://bucket/driver.sh",
                bundle_uri="gs://bucket/code.tar.gz",
                output_prefix="gs://bucket/run-1",
                run_id="ptd-five-day-b3f2468",
            )
        with self.assertRaises(ValueError):
            render_authorized(
                self.payload(),
                service_account=(
                    "product-recommend-pipelines@kauche-app-lab.iam.gserviceaccount.com"
                ),
                driver_uri="gs://bucket/driver.sh",
                bundle_uri="gs://bucket/code.tar.gz",
                output_prefix="gs://bucket/run-1",
                run_id="INVALID_RUN_ID",
            )

    def test_machine_or_fixed_input_change_is_rejected(self) -> None:
        payload = self.payload()
        payload["workerPoolSpecs"][0]["machineSpec"]["machineType"] = "n1-standard-4"
        with self.assertRaises(ValueError):
            validate_spec(payload, rendered=False)
        payload = self.payload()
        payload["workerPoolSpecs"][0]["containerSpec"]["env"][4]["value"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_spec(payload, rendered=False)


if __name__ == "__main__":
    unittest.main()
