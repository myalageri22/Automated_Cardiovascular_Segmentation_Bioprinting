#!/usr/bin/env python3
"""Primary endpoint and integrity-group analysis (manuscript Section 3.8, Table 10).

Inputs (both in this repository):
  outputs/final_test_250/per_case_metrics.csv   per-case segmentation metrics (250 held-out cases)
  outputs/phase_b_mesh_qc/per_case_mesh_qc.csv   per-case repaired-mesh QC

Outputs (written to outputs/final_test_250/):
  primary_endpoint_bootstrap.json
      Primary endpoint: |rho(clDice@0.5, components)| - |rho(Dice@0.5, components)|,
      Spearman rho against repaired-mesh connected-component count, with a paired
      case-level percentile bootstrap (10,000 resamples, seed 42).
  segmentation_component_correlations.csv
      Table 10 Panel A: Spearman rho of each metric vs component count, BH across the 5 metrics.
  integrity_group_comparison.csv
      Table 10 Panel B: segmentation metrics in integrity-pass vs integrity-fail cases
      (two-sided Mann-Whitney U, rank-biserial r, Benjamini-Hochberg across 5 metrics).

HD95: the original evaluation (evaluate_full_test_a40.py before the spacing fix) scored
HD95 with the ORIGINAL header spacing on 0.6 mm resampled arrays, underestimating it.
When outputs/final_test_250/hd95_corrected_per_case.csv exists, its hd95_mm (0.6 mm grid)
replaces the hd95@0.5 column here.

Mesh integrity = watertight repaired mesh AND zero non-manifold edges.

Usage:
  python scripts/evaluation/segmentation_geometry_association.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
METRICS = ["dice@0.5", "cldice@0.5", "precision@0.5", "recall@0.5", "hd95@0.5"]


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def load(seg_csv: Path, mesh_csv: Path) -> pd.DataFrame:
    seg = pd.read_csv(seg_csv)
    mesh = pd.read_csv(mesh_csv)[["case_id", "connected_component_count", "watertight", "non_manifold_edge_count"]]
    df = seg.merge(mesh, on="case_id", how="inner", validate="one_to_one")
    if len(df) != 250:
        raise SystemExit(f"expected 250 joined cases, got {len(df)}")
    corr = seg_csv.parent / "hd95_corrected_per_case.csv"
    if corr.exists():
        h = pd.read_csv(corr)[["case_id", "hd95_mm"]]
        df = df.drop(columns=["hd95@0.5"]).merge(h.rename(columns={"hd95_mm": "hd95@0.5"}), on="case_id", validate="one_to_one")
    df["integrity_pass"] = (df["watertight"].astype(str) == "True") & (df["non_manifold_edge_count"] == 0)
    return df


def primary_endpoint(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    cc = df["connected_component_count"].to_numpy()
    cl = df["cldice@0.5"].to_numpy()
    di = df["dice@0.5"].to_numpy()
    rho_cl = stats.spearmanr(cl, cc)[0]
    rho_di = stats.spearmanr(di, cc)[0]
    rng = np.random.default_rng(seed)
    n = len(df)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        boot[b] = abs(stats.spearmanr(cl[i], cc[i])[0]) - abs(stats.spearmanr(di[i], cc[i])[0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "endpoint": "|rho(clDice@0.5, components)| - |rho(Dice@0.5, components)|",
        "outcome": "repaired-mesh connected_component_count",
        "n_cases": int(n),
        "rho_cldice": float(rho_cl),
        "rho_dice": float(rho_di),
        "delta_abs_rho": float(abs(rho_cl) - abs(rho_di)),
        "ci95_percentile": [float(lo), float(hi)],
        "bootstrap_resamples": n_boot,
        "bootstrap_seed": seed,
        "bootstrap_unit": "case (paired: both correlations recomputed on the same resample)",
    }


def component_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METRICS:
        r, p = stats.spearmanr(df[m], df["connected_component_count"])
        rows.append({"metric": m, "spearman_rho": r, "p": p})
    out = pd.DataFrame(rows)
    out["bh_q"] = bh(out["p"].to_numpy())
    return out


def integrity_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    passed, failed = df[df.integrity_pass], df[~df.integrity_pass]
    for m in METRICS:
        x, y = passed[m].to_numpy(), failed[m].to_numpy()
        u = stats.mannwhitneyu(x, y, alternative="two-sided")
        r = 2 * u.statistic / (len(x) * len(y)) - 1
        q = lambda s: (np.median(s), np.percentile(s, 25), np.percentile(s, 75))
        mp, mf = q(x), q(y)
        rows.append({
            "metric": m, "n_pass": len(x), "n_fail": len(y),
            "pass_median": mp[0], "pass_q1": mp[1], "pass_q3": mp[2],
            "fail_median": mf[0], "fail_q1": mf[1], "fail_q3": mf[2],
            "mann_whitney_u": u.statistic, "rank_biserial_r": r, "p": u.pvalue,
        })
    out = pd.DataFrame(rows)
    out["bh_q"] = bh(out["p"].to_numpy())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-csv", type=Path, default=ROOT / "outputs/final_test_250/per_case_metrics.csv")
    ap.add_argument("--mesh-csv", type=Path, default=ROOT / "outputs/phase_b_mesh_qc/per_case_mesh_qc.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/final_test_250")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = load(a.seg_csv, a.mesh_csv)
    pe = primary_endpoint(df, a.n_boot, a.seed)
    (a.out_dir / "primary_endpoint_bootstrap.json").write_text(json.dumps(pe, indent=2) + "\n")
    cor = component_correlations(df)
    cor.to_csv(a.out_dir / "segmentation_component_correlations.csv", index=False)
    print(cor.to_string(index=False))
    grp = integrity_groups(df)
    grp.to_csv(a.out_dir / "integrity_group_comparison.csv", index=False)
    print(json.dumps(pe, indent=2))
    print(grp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
