#!/usr/bin/env python3
"""
aggregate_mesh_qc.py
--------------------
Aggregate per-case Phase B mesh-QC reports into cohort-level tables.

Use this when your pipeline (run_phase_b_qc=ON) writes a qc_report.json PER CASE but no
cohort summary. It is schema-agnostic: it flattens whatever keys each report contains, so
it works regardless of the exact QC fields, and it summarizes every numeric and boolean
column it finds.

Inputs:
  - a directory searched recursively for files named qc_report.json
    (override the filename with --qc-name, or pass an explicit --qc-glob)

Outputs:
  - per_case_mesh_qc.csv     one row per case, all flattened fields
  - summary_mesh_qc.csv      numeric: count/mean/median/std/min/max ; bool: %True ; cat: top value
  - summary_mesh_qc.json     same summary as JSON

Dependencies: pandas (CPU; runs anywhere). Example:
  python aggregate_mesh_qc.py --qc-dir outputs/v11_test --out-prefix v11_mesh
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd


def find_reports(qc_dir, qc_name, qc_glob):
    if qc_glob:
        return sorted(glob.glob(qc_glob, recursive=True))
    return sorted(glob.glob(os.path.join(qc_dir, "**", qc_name), recursive=True))


def case_id_from_path(path, qc_name):
    """Best-effort case id = name of the directory containing the report."""
    d = os.path.dirname(path)
    cid = os.path.basename(d)
    if not cid or cid in (".", ""):
        cid = os.path.basename(path).replace(qc_name, "").strip("._") or path
    return cid


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-case mesh QC reports.")
    ap.add_argument("--qc-dir", default=".",
                    help="Directory searched recursively for qc_report.json files")
    ap.add_argument("--qc-name", default="qc_report.json",
                    help="Report filename to search for (default qc_report.json)")
    ap.add_argument("--qc-glob", default=None,
                    help="Explicit glob (overrides --qc-dir/--qc-name), e.g. 'out/**/qc.json'")
    ap.add_argument("--out-prefix", default="mesh_qc",
                    help="Prefix for output files (default 'mesh_qc')")
    args = ap.parse_args()

    reports = find_reports(args.qc_dir, args.qc_name, args.qc_glob)
    if not reports:
        sys.exit(f"No '{args.qc_name}' files found under {args.qc_dir}")

    rows = []
    for path in reports:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"skip {path}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            data = {"value": data}
        flat = pd.json_normalize(data, sep=".").to_dict(orient="records")[0]
        flat = {"case_id": flat.get("case_id", case_id_from_path(path, args.qc_name)),
                **{k: v for k, v in flat.items() if k != "case_id"}}
        flat["_report_path"] = path
        rows.append(flat)

    per_case = pd.DataFrame(rows)
    per_case_path = f"{args.out_prefix}_per_case.csv"
    per_case.to_csv(per_case_path, index=False)

    # ---- build summary ----
    summary = {"n_cases": int(len(per_case))}
    drop = {"case_id", "_report_path"}
    for col in [c for c in per_case.columns if c not in drop]:
        s = per_case[col]
        non_null = s.dropna()
        if non_null.empty:
            continue
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().all() and not set(non_null.unique()).issubset({True, False, 0, 1}):
            summary[col] = {
                "type": "numeric", "count": int(numeric.count()),
                "mean": float(numeric.mean()), "median": float(numeric.median()),
                "std": float(numeric.std(ddof=1)) if numeric.count() > 1 else 0.0,
                "min": float(numeric.min()), "max": float(numeric.max()),
            }
        elif set(non_null.astype(str).str.lower().unique()).issubset(
                {"true", "false", "0", "1", "0.0", "1.0"}):
            as_bool = non_null.astype(str).str.lower().isin({"true", "1", "1.0"})
            summary[col] = {
                "type": "boolean", "count": int(as_bool.count()),
                "n_true": int(as_bool.sum()),
                "pct_true": round(100.0 * as_bool.mean(), 2),
            }
        else:
            vc = non_null.astype(str).value_counts()
            summary[col] = {
                "type": "categorical", "count": int(non_null.count()),
                "top": vc.index[0], "top_count": int(vc.iloc[0]),
                "n_unique": int(non_null.nunique()),
            }

    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # flat CSV version of the summary
    srows = []
    for k, v in summary.items():
        if isinstance(v, dict):
            srows.append({"field": k, **v})
    pd.DataFrame(srows).to_csv(f"{args.out_prefix}_summary.csv", index=False)

    # ---- console highlights ----
    print(f"cases: {summary['n_cases']}")
    for key in ("watertight", "is_watertight", "repaired_watertight",
                "overall_pass", "qc_pass", "passed"):
        if key in summary and summary[key].get("type") == "boolean":
            print(f"{key}: {summary[key]['pct_true']}% True "
                  f"({summary[key]['n_true']}/{summary[key]['count']})")
    print(f"written: {per_case_path}, {args.out_prefix}_summary.csv, {args.out_prefix}_summary.json")


if __name__ == "__main__":
    main()
