#!/usr/bin/env python3
"""Audit per-case result tables against an authoritative test split."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_test_ids(path: Path) -> list[str]:
    split = json.loads(path.read_text())
    entries = split.get("test")
    if not isinstance(entries, list):
        raise ValueError(f"{path} does not contain a test list")
    return [str(entry.get("id")) if isinstance(entry, dict) else str(entry) for entry in entries]


def audit_rows(rows: list[dict[str, str]], test_ids: list[str], case_column: str = "case_id") -> dict[str, Any]:
    ids = [str(row.get(case_column, "")).strip() for row in rows]
    if any(not case_id for case_id in ids):
        raise ValueError(f"One or more rows have an empty {case_column}")
    counts = Counter(ids)
    test_set = set(test_ids)
    failed = [row for row in rows if row.get("status", "ok").lower() not in {"", "ok", "success"}]
    return {
        "rows": len(rows),
        "unique_ids": len(counts),
        "duplicate_ids": sorted(case_id for case_id, count in counts.items() if count > 1),
        "missing_ids": sorted(test_set - set(ids)),
        "extra_ids": sorted(set(ids) - test_set),
        "failed_rows": len(failed),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit one or more result CSVs against the exact test cohort.")
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="Named result source; repeat for multiple files",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=250)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--allow-documented-duplicates", action="store_true")
    args = parser.parse_args()

    test_ids = load_test_ids(args.split_json)
    if len(test_ids) != args.expected_cases or len(set(test_ids)) != args.expected_cases:
        raise SystemExit(f"Expected {args.expected_cases} unique test IDs, got {len(test_ids)} rows/{len(set(test_ids))} unique")
    output_rows = []
    invalid = []
    for source in args.source:
        if "=" not in source:
            parser.error(f"Invalid --source {source!r}; expected NAME=CSV")
        name, raw_path = source.split("=", 1)
        path = Path(raw_path)
        audit = audit_rows(read_csv(path), test_ids, args.case_column)
        duplicates = ";".join(audit["duplicate_ids"])
        output_rows.append(
            {
                "source": name,
                "path": str(path),
                "rows": audit["rows"],
                "unique_ids": audit["unique_ids"],
                "duplicate_ids": duplicates,
                "missing_ids": ";".join(audit["missing_ids"]),
                "extra_ids": ";".join(audit["extra_ids"]),
                "failed_rows": audit["failed_rows"],
            }
        )
        valid = (
            audit["unique_ids"] == args.expected_cases
            and not audit["missing_ids"]
            and not audit["extra_ids"]
            and not audit["failed_rows"]
            and (not audit["duplicate_ids"] or args.allow_documented_duplicates)
        )
        if not valid:
            invalid.append(name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    if invalid:
        raise SystemExit(f"Cohort audit failed for: {', '.join(invalid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
