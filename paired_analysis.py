#!/usr/bin/env python3
"""
paired_analysis.py
------------------
The paired segmentation -> mesh analysis (NOT in your pipeline). Tests whether per-case
segmentation quality predicts per-case mesh / printability quality, which is the empirical
core of the "Dice does not imply print-ready" argument.

It merges:
  - per-case segmentation metrics (per_case_metrics.csv: case_id, dice, precision, recall, hd95, ...)
  - per-case mesh QC          (per_case_mesh_qc.csv from aggregate_mesh_qc.py)
  - optionally clDice         (cldice_per_case.csv from compute_cldice.py)
on case_id, then for every (segmentation metric) x (mesh metric) pair computes Spearman and
Pearson correlations with p-values, applies Benjamini-Hochberg FDR correction across all
tests, and writes scatterplots for the strongest associations.

Dependencies: numpy, pandas, scipy, matplotlib (CPU). Example:
  python paired_analysis.py \
      --seg-csv per_case_metrics.csv \
      --mesh-csv v11_mesh_per_case.csv \
      --cldice-csv cldice_per_case.csv \
      --out-prefix paired --top-k 6
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def coerce_numeric_frame(df, key):
    """Keep the key column; convert booleans to 0/1; keep only numeric feature columns."""
    out = pd.DataFrame({key: df[key].astype(str)})
    for col in df.columns:
        if col == key:
            continue
        s = df[col]
        if s.dtype == bool:
            out[col] = s.astype(int)
        else:
            low = s.astype(str).str.lower()
            if set(low.dropna().unique()).issubset({"true", "false", "nan"}):
                out[col] = low.map({"true": 1, "false": 0})
            else:
                num = pd.to_numeric(s, errors="coerce")
                if num.notna().any():
                    out[col] = num
    return out


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="Paired segmentation-to-mesh correlation analysis.")
    ap.add_argument("--seg-csv", required=True, help="per_case_metrics.csv")
    ap.add_argument("--mesh-csv", required=True, help="per_case mesh QC csv")
    ap.add_argument("--cldice-csv", default=None, help="optional cldice_per_case.csv to merge in")
    ap.add_argument("--key", default="case_id", help="join column (default case_id)")
    ap.add_argument("--out-prefix", default="paired")
    ap.add_argument("--top-k", type=int, default=6, help="how many scatterplots to emit")
    ap.add_argument("--min-n", type=int, default=8, help="skip pairs with fewer paired points")
    args = ap.parse_args()

    seg = pd.read_csv(args.seg_csv)
    mesh = pd.read_csv(args.mesh_csv)
    for name, df in (("seg", seg), ("mesh", mesh)):
        if args.key not in df.columns:
            sys.exit(f"'{args.key}' not in {name} csv. Columns: {list(df.columns)}")

    seg_n = coerce_numeric_frame(seg, args.key)
    mesh_n = coerce_numeric_frame(mesh, args.key)

    seg_cols = [c for c in seg_n.columns if c != args.key]
    mesh_cols = [c for c in mesh_n.columns if c != args.key]

    merged = seg_n.merge(mesh_n, on=args.key, how="inner", suffixes=("_seg", "_mesh"))

    if args.cldice_csv and os.path.exists(args.cldice_csv):
        cl = pd.read_csv(args.cldice_csv)
        cl[args.key] = cl[args.key].astype(str)
        keep = [args.key] + [c for c in ("cldice", "tprec", "tsens") if c in cl.columns]
        merged = merged.merge(cl[keep], on=args.key, how="left")
        seg_cols += [c for c in ("cldice", "tprec", "tsens") if c in cl.columns]

    # handle suffix collisions from the merge
    def resolved(col):
        if col in merged.columns:
            return col
        for suf in ("_seg", "_mesh"):
            if col + suf in merged.columns:
                return col + suf
        return None

    print(f"merged cases: {len(merged)}")
    if len(merged) < args.min_n:
        sys.exit(f"Only {len(merged)} paired cases (< --min-n {args.min_n}). "
                 f"Check that case_id values match across files.")

    results = []
    for sc in seg_cols:
        rs = resolved(sc)
        if rs is None:
            continue
        for mc in mesh_cols:
            rm = resolved(mc)
            if rm is None or rm == rs:
                continue
            pair = merged[[rs, rm]].dropna()
            if len(pair) < args.min_n:
                continue
            x, y = pair[rs].to_numpy(), pair[rm].to_numpy()
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            sr, sp = stats.spearmanr(x, y)
            pr, pp = stats.pearsonr(x, y)
            results.append({
                "seg_metric": sc, "mesh_metric": mc, "n": len(pair),
                "spearman_r": sr, "spearman_p": sp,
                "pearson_r": pr, "pearson_p": pp,
            })

    if not results:
        sys.exit("No valid metric pairs to correlate (after dropna / variance checks).")

    res = pd.DataFrame(results)
    res["spearman_p_fdr"] = bh_fdr(res["spearman_p"].to_numpy())
    res["abs_spearman"] = res["spearman_r"].abs()
    res = res.sort_values("abs_spearman", ascending=False).reset_index(drop=True)

    table_path = f"{args.out_prefix}_correlations.csv"
    res.drop(columns="abs_spearman").to_csv(table_path, index=False)

    # scatterplots for the strongest associations
    plot_dir = f"{args.out_prefix}_scatter"
    os.makedirs(plot_dir, exist_ok=True)
    made = []
    for _, row in res.head(args.top_k).iterrows():
        sc, mc = row["seg_metric"], row["mesh_metric"]
        rs, rm = resolved(sc), resolved(mc)
        pair = merged[[rs, rm]].dropna()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(pair[rs], pair[rm], s=18, alpha=0.7, edgecolor="none")
        ax.set_xlabel(sc)
        ax.set_ylabel(mc)
        ax.set_title(f"rho={row['spearman_r']:.2f} (p_fdr={row['spearman_p_fdr']:.3g}, n={int(row['n'])})")
        fig.tight_layout()
        fn = os.path.join(plot_dir, f"{sc}__vs__{mc}.png".replace("/", "_"))
        fig.savefig(fn, dpi=150)
        plt.close(fig)
        made.append(fn)

    print("\n=== top associations (|Spearman|) ===")
    show = res.head(min(args.top_k, len(res)))
    for _, r in show.iterrows():
        sig = "*" if r["spearman_p_fdr"] < 0.05 else " "
        print(f"{sig} {r['seg_metric']:>14s} ~ {r['mesh_metric']:<22s} "
              f"rho={r['spearman_r']:+.3f}  p_fdr={r['spearman_p_fdr']:.3g}  n={int(r['n'])}")
    print(f"\nwritten: {table_path}  and {len(made)} plots in {plot_dir}/")


if __name__ == "__main__":
    main()
