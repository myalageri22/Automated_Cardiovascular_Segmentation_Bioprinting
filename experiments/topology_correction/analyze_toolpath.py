#!/usr/bin/env python3
"""Paired analysis of the secondary toolpath check.

Input: ``outputs/topology_correction/toolpath/per_case_toolpath.csv`` produced by
``run_toolpath_experiment.py``. Every corrected variant is compared against the
S0 control, paired by case, with the same statistical machinery the main
experiment uses (exact McNemar for binary outcomes, Wilcoxon/t with
Hodges-Lehmann effect and bootstrap CI for continuous ones, Benjamini-Hochberg
within endpoint families).

This is a software-level slicing check on the reconstructed surfaces. It is not
evidence of physical printability and must not be reported as such.

    python analyze_toolpath.py --config config/experiment_config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from topocorr import stats_tests as st
from topocorr.io_utils import ensure_dir, load_config, repo_root, write_csv, write_json

CONTROL = "s0_original"

BINARY_ENDPOINTS = [("toolpath_success", "Toolpath generation success"),
                    ("gcode_generated", "G-code generated")]
CONTINUOUS_ENDPOINTS = [("layer_count", "Layer count"),
                        ("empty_layer_count", "Layers without extrusion"),
                        ("warning_count", "Slicer warning count"),
                        ("estimated_print_time_min", "Estimated print time (min)"),
                        ("filament_used_g", "Filament used (g)"),
                        ("filament_used_mm", "Filament used (mm)"),
                        ("gcode_bytes", "G-code size (bytes)")]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).parent / "config" / "experiment_config.yaml"))
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args(argv)

    root = repo_root()
    cfg = load_config(args.config)
    out_root = Path(args.output_root) if args.output_root else (root / cfg["paths"]["output_root"])
    tp_dir = ensure_dir(out_root / "toolpath")
    srcs = sorted(tp_dir.glob("per_case_toolpath*.csv"))
    if not srcs:
        print(f"no per_case_toolpath*.csv in {tp_dir}", file=sys.stderr)
        return 2
    allrows = pd.concat([pd.read_csv(s_) for s_ in srcs], ignore_index=True)
    if "profile_name" not in allrows.columns:
        allrows["profile_name"] = "unnamed_profile"
    rc = 0
    for profile, df in allrows.groupby("profile_name"):
        rc |= _analyse_one(df.copy(), str(profile), tp_dir, cfg)
    return rc


def _analyse_one(df: pd.DataFrame, profile: str, tp_dir: Path, cfg: Dict[str, Any]) -> int:
    sfx = f"__{profile}"
    df["case_id"] = df["case_id"].astype(str)
    for c, _ in BINARY_ENDPOINTS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(("true", "1", "yes"))

    # ------------------------------------------------------------- cohort summary
    agg: List[Dict[str, Any]] = []
    for vid, sub in df.groupby("variant_id"):
        row: Dict[str, Any] = {"variant_id": vid, "n_cases": len(sub),
                               "n_toolpath_success": int(sub["toolpath_success"].sum()),
                               "toolpath_success_rate": float(sub["toolpath_success"].mean())}
        for col, label in CONTINUOUS_ENDPOINTS:
            if col in sub.columns:
                s = pd.to_numeric(sub[col], errors="coerce")
                row[f"{col}_mean"] = float(s.mean())
                row[f"{col}_median"] = float(s.median())
        row["cases_with_any_warning"] = int((pd.to_numeric(sub.get("warning_count"), errors="coerce") > 0).sum())
        row["cases_with_empty_layers"] = int((pd.to_numeric(sub.get("empty_layer_count"), errors="coerce") > 0).sum())
        agg.append(row)
    summary = pd.DataFrame(agg).sort_values("variant_id")
    summary.insert(0, "profile_name", profile)
    summary.to_csv(tp_dir / f"toolpath_summary{sfx}.csv", index=False)

    # distinct warning strings and how often each fires, per variant
    wrows: List[Dict[str, Any]] = []
    for vid, sub in df.groupby("variant_id"):
        counts: Dict[str, int] = {}
        for w in sub["warnings"].fillna(""):
            for part in [p.strip() for p in str(w).split(";") if p.strip()]:
                counts[part] = counts.get(part, 0) + 1
        for w, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            wrows.append({"variant_id": vid, "warning": w, "n_cases": n,
                          "fraction": n / len(sub)})
    if wrows:
        for r in wrows:
            r["profile_name"] = profile
        write_csv(tp_dir / f"toolpath_warnings{sfx}.csv", wrows)

    # ------------------------------------------------------------- paired tests
    ctrl = df[df["variant_id"] == CONTROL].set_index("case_id")
    rows: List[Dict[str, Any]] = []
    stat_cfg = cfg["statistics"]
    for vid, sub in df.groupby("variant_id"):
        if vid == CONTROL:
            continue
        sub = sub.set_index("case_id")
        cases = sorted(set(ctrl.index) & set(sub.index))
        for col, label in BINARY_ENDPOINTS:
            if col not in df.columns:
                continue
            r = st.paired_binary(ctrl.loc[cases, col].tolist(), sub.loc[cases, col].tolist(),
                                 label, vid, iterations=int(stat_cfg["bootstrap_iterations"]),
                                 ci=float(stat_cfg["bootstrap_ci"]),
                                 seed=int(cfg["experiment"]["seed"]))
            r.update({"variant_id": vid, "profile_name": profile,
                      "family": f"toolpath_binary::{profile}::{vid}"})
            rows.append(r)
        for col, label in CONTINUOUS_ENDPOINTS:
            if col not in df.columns:
                continue
            a = pd.to_numeric(ctrl.loc[cases, col], errors="coerce")
            b = pd.to_numeric(sub.loc[cases, col], errors="coerce")
            keep = a.notna() & b.notna()
            if keep.sum() < 2:
                continue
            r = st.paired_continuous(a[keep].tolist(), b[keep].tolist(), label, vid,
                                     mode=stat_cfg["paired_test_continuous"],
                                     shapiro_alpha=float(stat_cfg["shapiro_alpha"]),
                                     iterations=int(stat_cfg["bootstrap_iterations"]),
                                     ci=float(stat_cfg["bootstrap_ci"]),
                                     seed=int(cfg["experiment"]["seed"]))
            r.update({"variant_id": vid, "profile_name": profile,
                      "family": f"toolpath_continuous::{profile}::{vid}"})
            rows.append(r)
    if rows:
        rows = st.apply_fdr_by_family(rows, "family", float(stat_cfg["fdr_alpha"]))
        write_csv(tp_dir / f"toolpath_paired_tests{sfx}.csv", rows)

    failures = df[~df["toolpath_success"]]
    if len(failures):
        failures.to_csv(tp_dir / f"toolpath_failures{sfx}.csv", index=False)
    write_json(tp_dir / f"toolpath_qc{sfx}.json", {
        "profile_name": profile,
        "n_rows": int(len(df)),
        "variants": sorted(df["variant_id"].unique().tolist()),
        "cases_per_variant": {k: int(v) for k, v in df.groupby("variant_id")["case_id"].nunique().items()},
        "n_failures": int(len(failures)),
        "failed": failures[["variant_id", "case_id", "return_code", "stderr_tail"]].to_dict("records") if len(failures) else [],
        "failure_reasons": (failures["stderr_tail"].fillna("").value_counts().to_dict()
                            if len(failures) else {}),
        "control_variant": CONTROL,
        "claim_boundary": "Software-level slicing outcome only. Not evidence of physical "
                          "printability, print quality, or fabrication success.",
    })
    print(f"--- {profile} ---")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
