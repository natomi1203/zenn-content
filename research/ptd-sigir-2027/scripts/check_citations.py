#!/usr/bin/env python3
"""Check that every LaTeX citation key resolves to one BibTeX entry."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / "paper" / "main.tex").read_text()
BIB = (ROOT / "paper" / "references.bib").read_text()

cited: set[str] = set()
for group in re.findall(r"\\cite(?:p|t|author|year)?\{([^}]+)\}", TEX):
    cited.update(key.strip() for key in group.split(","))

defined = set(re.findall(r"@[A-Za-z]+\{([^,\s]+),", BIB))
missing = sorted(cited - defined)
duplicate_entries = sorted(key for key in defined if len(re.findall(rf"@[A-Za-z]+\{{{re.escape(key)},", BIB)) != 1)

if missing or duplicate_entries:
    raise SystemExit(f"citation check failed: missing={missing}, duplicate_entries={duplicate_entries}")

print(f"citation check: PASS ({len(cited)} cited keys, {len(defined)} bibliography entries)")
