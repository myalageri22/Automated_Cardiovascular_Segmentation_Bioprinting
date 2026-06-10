#!/usr/bin/env python3
"""Generate reviewer-ready figures, tables, and representative case manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _imports():
    missing = {}
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        pd = None  # type: ignore
        missing["pandas"] = str(exc)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        plt = None  # type: ignore
        missing["matplotlib"] = str(exc)
    return pd, plt, missing


def ensure_dirs() -> tuple[Path, Path]:
    fig_dir = Path("outputs/figures_for_paper")
    table_dir = Path("outputs/tables_for_paper")
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, table_dir


def copy_table(src: Path, dest: Path) -> None:
    if src.exists():
        dest.write_text(src.read_text())
    else:
        dest.write_text("status,reason\nmissing,input_not_found\n")


def generate(eval_dir: str | Path = "outputs/full_test_eval_a40") -> Dict[str, str]:
    pd, plt, missing = _imports()
    fig_dir, table_dir = ensure_dirs()
    eval_dir = Path(eval_dir)
    outputs: Dict[str, str] = {}
    copy_table(eval_dir / "summary_metrics.csv", table_dir / "cohort_summary.csv")
    copy_table(Path("outputs/phase_b_mesh_qc/summary_mesh_qc.csv"), table_dir / "mesh_qc_summary.csv")
    copy_table(Path("outputs/baselines/baseline_comparison.csv"), table_dir / "baseline_comparison_summary.csv")
    copy_table(Path("outputs/phase_b_mesh_qc/segmentation_mesh_linkage.csv"), table_dir / "segmentation_to_mesh_linkage_summary.csv")
    if missing:
        (fig_dir / "missing_plot_dependencies.json").write_text(json.dumps(missing, indent=2))
        return {"status": "missing_dependency", **missing}
    per_case = eval_dir / "per_case_metrics.csv"
    if not per_case.exists():
        (fig_dir / "representative_cases.csv").write_text("status,reason\nmissing,per_case_metrics_not_found\n")
        return {"status": "missing_inputs"}
    df = pd.read_csv(per_case)
    numeric_cols = [c for c in df.columns if c.startswith(("dice@", "precision@", "recall@", "hd95", "cldice", "runtime"))]
    if "dice@0.5" in df.columns:
        metric = "dice@0.5"
    elif "dice@0.5" not in df.columns and "dice@0.5" in numeric_cols:
        metric = "dice@0.5"
    else:
        metric = next((c for c in df.columns if c.startswith("dice@")), None)
    if metric:
        plt.figure(figsize=(6, 4))
        df.boxplot(column=metric)
        plt.ylabel(metric)
        plt.title("Held-out Test Dice Distribution")
        plt.tight_layout()
        plt.savefig(fig_dir / "dice_distribution_boxplot.png", dpi=200)
        plt.close()
        ranked = df.sort_values(metric)
        reps = []
        if len(ranked):
            reps.append({"role": "worst", **ranked.iloc[0].to_dict()})
            reps.append({"role": "median", **ranked.iloc[len(ranked) // 2].to_dict()})
            reps.append({"role": "best", **ranked.iloc[-1].to_dict()})
        pd.DataFrame(reps).to_csv(fig_dir / "representative_cases.csv", index=False)
    threshold_rows: List[Dict[str, float]] = []
    for thr in ["0.1", "0.2", "0.3", "0.4", "0.5"]:
        row: Dict[str, float] = {"threshold": float(thr)}
        for prefix in ("dice", "precision", "recall"):
            col = f"{prefix}@{thr}"
            if col in df.columns:
                row[prefix] = float(df[col].mean())
        threshold_rows.append(row)
    if threshold_rows:
        tdf = pd.DataFrame(threshold_rows)
        tdf.to_csv(table_dir / "threshold_summary.csv", index=False)
        for cols, name, ylabel in [
            (["dice"], "threshold_performance_curves.png", "Dice"),
            (["precision", "recall"], "precision_recall_curves.png", "Value"),
        ]:
            plt.figure(figsize=(6, 4))
            for col in cols:
                if col in tdf:
                    plt.plot(tdf["threshold"], tdf[col], marker="o", label=col)
            plt.xlabel("Threshold")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / name, dpi=200)
            plt.close()
    if "hd95@0.5" in df.columns:
        vals = pd.to_numeric(df["hd95@0.5"], errors="coerce").dropna()
        if len(vals):
            plt.figure(figsize=(6, 4))
            vals.plot(kind="hist", bins=20)
            plt.xlabel("HD95")
            plt.tight_layout()
            plt.savefig(fig_dir / "hd95_distribution.png", dpi=200)
            plt.close()
    runtime_col = "runtime_seconds" if "runtime_seconds" in df.columns else None
    if runtime_col:
        plt.figure(figsize=(6, 4))
        pd.to_numeric(df[runtime_col], errors="coerce").dropna().plot(kind="hist", bins=20)
        plt.xlabel("Runtime seconds")
        plt.tight_layout()
        plt.savefig(fig_dir / "runtime_distribution.png", dpi=200)
        plt.close()
    mesh_path = Path("outputs/phase_b_mesh_qc/per_case_mesh_qc.csv")
    if mesh_path.exists() and metric:
        mesh = pd.read_csv(mesh_path)
        merged = df.merge(mesh, on="case_id", how="inner")
        if "non_manifold_edge_count" in merged.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(pd.to_numeric(merged[metric], errors="coerce"), pd.to_numeric(merged["non_manifold_edge_count"], errors="coerce"))
            plt.xlabel(metric)
            plt.ylabel("Non-manifold edge count")
            plt.tight_layout()
            plt.savefig(fig_dir / "correlation_dice_mesh_qc.png", dpi=200)
            plt.close()
    return {"status": "ok", "figures": str(fig_dir), "tables": str(table_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paper figures/tables from evaluation outputs")
    parser.add_argument("--eval_dir", default="outputs/full_test_eval_a40")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(generate(args.eval_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
