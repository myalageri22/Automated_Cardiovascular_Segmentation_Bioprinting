"""Publication figures. Every figure writes the exact data it plots to CSV.

No figure may contain a number that is not present in its ``*_source_data.csv``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "original": "#4C6EF5",
    "corrected": "#E8590C",
    "neutral": "#868E96",
    "accent": "#0CA678",
    "warn": "#C92A2A",
}


def _save(fig: plt.Figure, out_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(str(p))
    plt.close(fig)
    return written


def _source(df: pd.DataFrame, out_dir: Path, stem: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{stem}_source_data.csv"
    df.to_csv(p, index=False)
    return str(p)


# ---------------------------------------------------------------------------
# Figure A: pipeline schematic
# ---------------------------------------------------------------------------
def figure_a_pipeline(out_dir: Path, formats=("png", "pdf"), dpi: int = 300) -> Dict[str, Any]:
    stages = [
        ("CCTA volume", "input"),
        ("Frozen\nAttention U-Net", "frozen"),
        ("Original\nsegmentation", "control"),
        ("Topology\ndiagnosis\n(Stage A)", "new"),
        ("Conservative\ncorrection\n(S1-S3)", "new"),
        ("Phase B\nreconstruction", "shared"),
        ("Geometry QC\n+ fidelity", "eval"),
    ]
    colors = {"input": "#DEE2E6", "frozen": "#BAC8FF", "control": "#A5D8FF",
              "new": "#FFD8A8", "shared": "#B2F2BB", "eval": "#E9ECEF"}
    fig, ax = plt.subplots(figsize=(14, 3.2))
    x = 0.0
    w, h = 1.7, 1.0
    for label, kind in stages:
        ax.add_patch(plt.Rectangle((x, 0), w, h, facecolor=colors[kind],
                                   edgecolor="#343A40", linewidth=1.2, zorder=2))
        ax.text(x + w / 2, h / 2, label, ha="center", va="center", fontsize=9, zorder=3)
        if x > 0:
            ax.annotate("", xy=(x, h / 2), xytext=(x - 0.35, h / 2),
                        arrowprops=dict(arrowstyle="->", color="#343A40", lw=1.4), zorder=1)
        x += w + 0.35
    ax.text(0, -0.42, "Orange stages are introduced by this experiment. "
                      "The network and the reconstruction pipeline are unchanged.",
            fontsize=8.5, color="#495057")
    ax.set_xlim(-0.4, x); ax.set_ylim(-0.6, h + 0.25); ax.axis("off")
    files = _save(fig, out_dir, "figure_A_pipeline", formats, dpi)
    src = _source(pd.DataFrame({"stage_order": range(1, len(stages) + 1),
                                "stage": [s for s, _ in stages],
                                "kind": [k for _, k in stages]}),
                  out_dir, "figure_A_pipeline")
    return {"files": files, "source_data": src}


# ---------------------------------------------------------------------------
# Figure B: paired connected-component count
# ---------------------------------------------------------------------------
def figure_b_paired_components(
    paired: pd.DataFrame, strategy: str, out_dir: Path,
    formats=("png", "pdf"), dpi: int = 300,
) -> Dict[str, Any]:
    sub = paired[paired["strategy"] == strategy].copy()
    a = pd.to_numeric(sub["components_original"], errors="coerce")
    b = pd.to_numeric(sub["components_corrected"], errors="coerce")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for ai, bi in zip(a, b):
        ax.plot([0, 1], [ai, bi], color=PALETTE["neutral"], alpha=0.25, lw=0.8, zorder=1)
    ax.scatter(np.zeros(len(a)), a, s=14, color=PALETTE["original"], zorder=2, label="original")
    ax.scatter(np.ones(len(b)), b, s=14, color=PALETTE["corrected"], zorder=2, label="corrected")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Original", "Corrected"])
    ax.set_ylabel("Connected components (voxel mask)")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title(f"Paired components, n={len(a)}")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    lim = max(1, int(np.nanmax([a.max() if len(a) else 1, b.max() if len(b) else 1])) + 1)
    ax.plot([0, lim], [0, lim], ls="--", color=PALETTE["neutral"], lw=1)
    ax.scatter(a, b, s=18, alpha=0.65, color=PALETTE["accent"], edgecolor="none")
    ax.set_xlabel("Original components"); ax.set_ylabel("Corrected components")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title("Below the diagonal = fewer components")
    for a_ in axes:
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Figure B - connected-component count: {strategy}", fontsize=11)
    files = _save(fig, out_dir, f"figure_B_components_{strategy}", formats, dpi)
    src = _source(sub[["case_id", "strategy", "components_original", "components_corrected"]],
                  out_dir, f"figure_B_components_{strategy}")
    return {"files": files, "source_data": src}


# ---------------------------------------------------------------------------
# Figure C: topology-fidelity tradeoff
# ---------------------------------------------------------------------------
def figure_c_tradeoff(
    summary: pd.DataFrame, out_dir: Path, formats=("png", "pdf"), dpi: int = 300,
) -> Dict[str, Any]:
    """One point per strategy variant: connectivity benefit against anatomical cost."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    cost_specs = [
        ("delta_dice_mean", "Change in Dice (corrected - original)"),
        ("delta_hd95_mean", "Change in HD95 (mm)"),
        ("surface_deviation_mean_mm", "Symmetric surface deviation vs original (mm)"),
    ]
    strategies = sorted(summary["strategy"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    color_of = {s: cmap(i % 10) for i, s in enumerate(strategies)}

    for ax, (cost_col, xlabel) in zip(axes, cost_specs):
        if cost_col not in summary.columns:
            ax.text(0.5, 0.5, f"{cost_col}\nnot computed", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color=PALETTE["warn"])
            ax.axis("off")
            continue
        for s in strategies:
            sub = summary[summary["strategy"] == s]
            x = pd.to_numeric(sub[cost_col], errors="coerce")
            y = pd.to_numeric(sub["delta_components_median"], errors="coerce")
            ax.scatter(x, y, s=70, color=color_of[s], label=s, edgecolor="#212529", linewidth=0.5)
            for _, r in sub.iterrows():
                ax.annotate(str(r.get("variant_id", "")).split("__")[-1],
                            (pd.to_numeric(r[cost_col], errors="coerce"),
                             pd.to_numeric(r["delta_components_median"], errors="coerce")),
                            fontsize=6.5, xytext=(4, 3), textcoords="offset points")
        ax.axvline(0, color=PALETTE["neutral"], lw=0.8, ls="--")
        ax.axhline(0, color=PALETTE["neutral"], lw=0.8, ls="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Median change in components")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5, loc="best")
    fig.suptitle("Figure C - topology benefit versus anatomical cost "
                 "(down-left quadrant = fewer components at lower fidelity cost)", fontsize=10.5)
    files = _save(fig, out_dir, "figure_C_tradeoff", formats, dpi)
    src = _source(summary, out_dir, "figure_C_tradeoff")
    return {"files": files, "source_data": src}


# ---------------------------------------------------------------------------
# Figure E: benefit as a function of baseline quality
# ---------------------------------------------------------------------------
def figure_e_benefit_vs_baseline(
    paired: pd.DataFrame, strategy: str, out_dir: Path,
    formats=("png", "pdf"), dpi: int = 300,
) -> Dict[str, Any]:
    sub = paired[paired["strategy"] == strategy].copy()
    sub["delta_components"] = (pd.to_numeric(sub["components_corrected"], errors="coerce")
                               - pd.to_numeric(sub["components_original"], errors="coerce"))
    specs = [("cldice_original", "Baseline clDice"),
             ("dice_original", "Baseline Dice"),
             ("components_original", "Baseline components")]
    available = [(c, l) for c, l in specs if c in sub.columns]
    if not available:
        return {"files": [], "source_data": None, "note": "no baseline covariates available"}

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.4), squeeze=False)
    from scipy import stats as _st
    for ax, (col, label) in zip(axes[0], available):
        x = pd.to_numeric(sub[col], errors="coerce")
        y = sub["delta_components"]
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=18, alpha=0.65, color=PALETTE["corrected"], edgecolor="none")
        if ok.sum() >= 3:
            rho, p = _st.spearmanr(x[ok], y[ok])
            ax.set_title(f"Spearman rho = {rho:.3f}, P = {p:.3g}", fontsize=9)
        ax.axhline(0, color=PALETTE["neutral"], lw=0.8, ls="--")
        ax.set_xlabel(label); ax.set_ylabel("Change in components")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Figure E - correction benefit versus baseline quality: {strategy} (exploratory)",
                 fontsize=10.5)
    files = _save(fig, out_dir, f"figure_E_benefit_vs_baseline_{strategy}", formats, dpi)
    keep = ["case_id", "strategy", "delta_components"] + [c for c, _ in available]
    src = _source(sub[[c for c in keep if c in sub.columns]], out_dir,
                  f"figure_E_benefit_vs_baseline_{strategy}")
    return {"files": files, "source_data": src}


# ---------------------------------------------------------------------------
# Sensitivity curves (ablation)
# ---------------------------------------------------------------------------
def figure_sensitivity(
    summary: pd.DataFrame, out_dir: Path, formats=("png", "pdf"), dpi: int = 300,
) -> Dict[str, Any]:
    """Correction strength against every primary endpoint, one line per strategy."""
    endpoints = [
        ("delta_components_median", "Median change in components"),
        ("delta_dice_mean", "Mean change in Dice"),
        ("delta_cldice_mean", "Mean change in clDice"),
        ("delta_hd95_mean", "Mean change in HD95 (mm)"),
        ("mesh_integrity_rate_corrected", "Mesh-integrity pass rate"),
        ("surface_deviation_mean_mm", "Surface deviation vs original (mm)"),
    ]
    present = [(c, l) for c, l in endpoints if c in summary.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    strategies = [s for s in sorted(summary["strategy"].dropna().unique()) if s != "s0_original"]
    cmap = plt.get_cmap("tab10")
    for ax, (col, label) in zip(axes.ravel(), present):
        for i, s in enumerate(strategies):
            sub = summary[summary["strategy"] == s].copy()
            if "strength" not in sub.columns:
                continue
            sub = sub.sort_values("strength")
            ax.plot(pd.to_numeric(sub["strength"], errors="coerce"),
                    pd.to_numeric(sub[col], errors="coerce"),
                    marker="o", ms=4, color=cmap(i % 10), label=s, lw=1.4)
        ax.set_xlabel("Correction strength (strategy-specific parameter)")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(present):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=4)
    fig.suptitle("Sensitivity analysis - correction strength against every endpoint", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    files = _save(fig, out_dir, "figure_S1_sensitivity", formats, dpi)
    src = _source(summary, out_dir, "figure_S1_sensitivity")
    return {"files": files, "source_data": src}


def _resolve_panel(path: Any, category: Any, panel_root: Path) -> Optional[Path]:
    """Locate a rendered panel.

    ``rendered_cases.csv`` stores absolute paths from the machine that produced
    them. The same results tree can be read from a different mount point, so an
    absolute miss falls back to the panel's location relative to the results
    directory before the panel is declared missing.
    """
    if not isinstance(path, str) or not path:
        return None
    p = Path(path)
    if p.exists():
        return p
    if isinstance(category, str):
        cand = panel_root / category / p.name
        if cand.exists():
            return cand
    for cand in panel_root.rglob(p.name):
        return cand
    return None


# ---------------------------------------------------------------- Figure D
# Representative 3D cases: original vs corrected reconstruction across the
# success AND failure categories selected by failure_analysis.select_cases.
# The per-case panels rendered there are the source images; this function only
# assembles them into one publication figure. No case is re-selected here, so
# the figure cannot be made to show a friendlier subset than the audit trail.
FIGURE_D_CATEGORY_ORDER = [
    "high_dice_high_fragmentation",
    "moderate_dice_good_topology",
    "largest_component_reduction",
    "no_improvement",
    "largest_fidelity_loss",
]

FIGURE_D_CATEGORY_LABEL = {
    "high_dice_high_fragmentation": "High Dice / high fragmentation",
    "moderate_dice_good_topology": "Moderate Dice / good topology",
    "largest_component_reduction": "Largest component reduction",
    "no_improvement": "Little or no improvement",
    "largest_fidelity_loss": "Largest fidelity loss (worst case)",
}


def figure_d_representative_cases(
    rendered: pd.DataFrame,
    paired: pd.DataFrame,
    variant_id: str,
    out_dir: Path,
    formats=("png", "pdf"),
    dpi: int = 300,
    n_per_category: int = 1,
) -> Dict[str, Any]:
    """Assemble Figure D from the already-rendered representative-case panels.

    One row per selection category (successes and failures alike), in the fixed
    order above. Left column: CTA / ground truth / original prediction /
    corrected prediction slices. Right column: original vs corrected mesh.
    """
    import matplotlib.image as mpimg

    sel = rendered[rendered["variant_id"] == variant_id].copy() if "variant_id" in rendered.columns \
        else rendered.copy()
    if sel.empty:
        return {"files": [], "source_data": None, "note": "no rendered cases for this variant"}

    rows: List[Dict[str, Any]] = []
    for cat in FIGURE_D_CATEGORY_ORDER:
        sub = sel[sel["category"] == cat]
        for _, r in sub.head(n_per_category).iterrows():
            rows.append(r.to_dict())
    for _, r in sel.iterrows():                       # any category not in the fixed list
        if r["category"] not in FIGURE_D_CATEGORY_ORDER:
            rows.append(r.to_dict())
    if not rows:
        return {"files": [], "source_data": None, "note": "no rows"}

    pv = paired[paired["variant_id"] == variant_id].set_index(
        paired[paired["variant_id"] == variant_id]["case_id"].astype(str))

    panel_root = Path(out_dir).parent / "failure_analysis"
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(13.5, 3.1 * n),
                             gridspec_kw={"width_ratios": [2.0, 1.0]})
    axes = np.atleast_2d(axes)
    src_rows: List[Dict[str, Any]] = []

    for i, r in enumerate(rows):
        case = str(r["case_id"])
        for j, key in enumerate(("slices_png", "meshes_png")):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            p = _resolve_panel(r.get(key), r.get("category"), panel_root)
            if p is not None:
                ax.imshow(mpimg.imread(str(p)))
            else:
                ax.text(0.5, 0.5, "panel unavailable", ha="center", va="center",
                        fontsize=8, transform=ax.transAxes)

        m = pv.loc[case] if case in pv.index else None
        if m is not None and isinstance(m, pd.DataFrame):
            m = m.iloc[0]

        def g(col):
            try:
                return float(m[col]) if m is not None and col in m.index else float("nan")
            except Exception:
                return float("nan")

        cc_o, cc_c = g("components_original"), g("components_corrected")
        d_o, d_c = g("dice_original"), g("dice_corrected")
        cl_o, cl_c = g("cldice_original"), g("cldice_corrected")
        surf = g("geom_vs_original_symmetric_mean_surface_distance_mm")
        cap = (f"Case {case} — {FIGURE_D_CATEGORY_LABEL.get(r['category'], r['category'])}\n"
               f"components {cc_o:.0f}→{cc_c:.0f}   "
               f"Dice {d_o:.3f}→{d_c:.3f}   "
               f"clDice {cl_o:.3f}→{cl_c:.3f}   "
               f"surface Δ {surf:.3f} mm")
        axes[i, 0].set_title(cap, fontsize=8, loc="left")
        axes[i, 1].set_title("original vs corrected mesh", fontsize=8, loc="left")

        src_rows.append({
            "case_id": case, "variant_id": variant_id, "category": r["category"],
            "selection_rule": r.get("selection_rule"),
            "panel_slices": r.get("slices_png"), "panel_meshes": r.get("meshes_png"),
            "components_original": cc_o, "components_corrected": cc_c,
            "dice_original": d_o, "dice_corrected": d_c,
            "cldice_original": cl_o, "cldice_corrected": cl_c,
            "hd95_original": g("hd95_original"), "hd95_corrected": g("hd95_corrected"),
            "mesh_integrity_pass_original": g("mesh_integrity_pass_original"),
            "mesh_integrity_pass_corrected": g("mesh_integrity_pass_corrected"),
            "symmetric_mean_surface_distance_mm": surf,
            "chamfer_mm": g("geom_vs_original_chamfer_mm"),
            "hausdorff_mm": g("geom_vs_original_hausdorff_mm"),
            "centroid_displacement_mm": g("geom_vs_original_centroid_displacement_mm"),
        })

    fig.suptitle(f"Figure D — representative cases, original vs corrected reconstruction "
                 f"({variant_id}). Rows are prespecified selection categories, "
                 f"successes and failures alike.", fontsize=9, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    files = _save(fig, out_dir, f"figure_D_representative_{variant_id}", formats, dpi)
    src = _source(pd.DataFrame(src_rows), out_dir, f"figure_D_representative_{variant_id}")
    return {"files": files, "source_data": src,
            "selection_rules_source": "failure_analysis/selection_rules.json"}
