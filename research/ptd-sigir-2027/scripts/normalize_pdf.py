#!/usr/bin/env python3
"""Normalize volatile XeTeX/XMP identifiers without changing PDF layout."""

from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

DATE_PATTERN = re.compile(
    rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?P<zone>Z|[+-]\d{2}:\d{2})?"
)
UUID_PATTERN = re.compile(rb"uuid:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
ID_PATTERN = re.compile(
    rb"(/ID\s*\[\s*<)([0-9a-fA-F]{32})(>\s*<)([0-9a-fA-F]{32})(>\s*\])"
)
ZERO_UUID = b"uuid:00000000-0000-0000-0000-000000000000"
ZERO_ID = b"0" * 32


def normalize(pdf: Path, source_date_epoch: int) -> str:
    data = pdf.read_bytes()
    start = data.find(b"<x:xmpmeta")
    end = data.find(b"</x:xmpmeta>", start)
    if start < 0 or end < 0:
        raise SystemExit("PDF normalization failed: XMP packet not found")
    end += len(b"</x:xmpmeta>")

    xmp = data[start:end]
    fixed_date = datetime.fromtimestamp(source_date_epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S").encode()

    def set_date(match: re.Match[bytes]) -> bytes:
        zone = match.group("zone")
        if zone is None:
            return fixed_date
        if zone == b"Z":
            return fixed_date + b"Z"
        return fixed_date + b"+00:00"

    xmp, date_count = DATE_PATTERN.subn(set_date, xmp)
    xmp, uuid_count = UUID_PATTERN.subn(ZERO_UUID, xmp)
    if date_count < 1 or uuid_count < 1:
        raise SystemExit(
            f"PDF normalization failed: expected XMP date and UUID (dates={date_count}, uuids={uuid_count})"
        )
    data = data[:start] + xmp + data[end:]

    def zero_identifier(match: re.Match[bytes]) -> bytes:
        return match.group(1) + ZERO_ID + match.group(3) + ZERO_ID + match.group(5)

    data, id_count = ID_PATTERN.subn(zero_identifier, data)
    if id_count != 1:
        raise SystemExit(f"PDF normalization failed: expected one trailer ID, found {id_count}")

    digest = hashlib.sha256(data).hexdigest()
    uuid_hex = digest[:32]
    fixed_uuid = (
        f"uuid:{uuid_hex[:8]}-{uuid_hex[8:12]}-4{uuid_hex[13:16]}-"
        f"a{uuid_hex[17:20]}-{uuid_hex[20:32]}"
    ).encode()
    identifier = digest[:32].encode()
    data = data.replace(ZERO_UUID, fixed_uuid)

    def set_identifier(match: re.Match[bytes]) -> bytes:
        return match.group(1) + identifier + match.group(3) + identifier + match.group(5)

    data, id_count = ID_PATTERN.subn(set_identifier, data)
    if id_count != 1 or len(data) != pdf.stat().st_size:
        raise SystemExit("PDF normalization failed: output structure changed unexpectedly")
    pdf.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    args = parser.parse_args()
    digest = normalize(args.pdf, args.source_date_epoch)
    print(f"PDF normalization: PASS ({digest})")


if __name__ == "__main__":
    main()
