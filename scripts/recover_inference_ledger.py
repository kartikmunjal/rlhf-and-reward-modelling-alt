#!/usr/bin/env python3
"""Auditable removal of preregistered trial rows invalidated before analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-removed", type=int, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    original = args.ledger.read_bytes()
    rows = [json.loads(line) for line in original.decode("utf-8").splitlines() if line.strip()]
    removed, retained = [], []
    for row in rows:
        matches = (
            row.get("system") == args.system
            and row.get("target") == args.target
            and row.get("precision") == args.precision
        )
        (removed if matches else retained).append(row)
    if len(removed) != args.expected_removed:
        raise RuntimeError(f"Expected {args.expected_removed} rows, matched {len(removed)}; ledger unchanged")

    replacement = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for row in retained
    )
    audit = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "ledger": str(args.ledger),
        "reason": args.reason,
        "selector": {"system": args.system, "target": args.target, "precision": args.precision},
        "original_rows": len(rows),
        "removed_rows": len(removed),
        "retained_rows": len(retained),
        "original_sha256": digest(original),
        "replacement_sha256": digest(replacement),
        "removed": removed,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary = args.ledger.with_suffix(args.ledger.suffix + ".tmp")
    temporary.write_bytes(replacement)
    temporary.replace(args.ledger)
    print(json.dumps({key: audit[key] for key in ("original_rows", "removed_rows", "retained_rows", "original_sha256", "replacement_sha256")}, indent=2))


if __name__ == "__main__":
    main()
