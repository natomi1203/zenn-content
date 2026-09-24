#!/usr/bin/env python3
"""Refresh local artifact hashes after deterministic generation and PDF build."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifact" / "manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    payload = json.loads(MANIFEST.read_text())
    for artifact in payload["artifacts"]:
        if artifact["status"] == "PENDING":
            artifact["sha256"] = None
            continue
        path = ROOT / artifact["path"]
        if not path.is_file():
            raise SystemExit(f"cannot finalize missing artifact: {artifact['path']}")
        artifact["sha256"] = sha256(path)
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"manifest finalization: PASS ({len(payload['artifacts'])} entries)")


if __name__ == "__main__":
    main()
