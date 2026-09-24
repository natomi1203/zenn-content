from __future__ import annotations

import json
import unittest

from scripts.validate_vertex_preflight import (
    DEFAULT_SPEC,
    render_authorized,
    validate_spec,
)


class VertexPreflightTest(unittest.TestCase):
    def payload(self) -> dict:
        return json.loads(DEFAULT_SPEC.read_text())

    def test_template_is_valid_and_fail_closed(self) -> None:
        report = validate_spec(self.payload(), rendered=False)
        self.assertTrue(report["valid"])
        self.assertFalse(report["paid_job_launched"])

    def test_machine_change_is_rejected(self) -> None:
        payload = self.payload()
        payload["workerPoolSpecs"][0]["machineSpec"]["machineType"] = "a2-highgpu-1g"
        with self.assertRaisesRegex(ValueError, "machine/GPU"):
            validate_spec(payload, rendered=False)

    def test_authorized_render_requires_explicit_values(self) -> None:
        rendered = render_authorized(
            self.payload(),
            service_account="ptd-runner@example.iam.gserviceaccount.com",
            bundle_uri="gs://example/code.tar.gz",
            output_prefix="gs://example/preflight/one",
        )
        report = validate_spec(rendered, rendered=True)
        self.assertTrue(report["valid"])
        self.assertFalse(report["paid_job_launched"])


if __name__ == "__main__":
    unittest.main()
