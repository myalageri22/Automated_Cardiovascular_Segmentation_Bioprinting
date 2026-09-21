"""Statistical assembly, the manuscript table, and neutral outcome classification."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .stats_tests import apply_fdr_by_family, paired_binary, paired_continuous

# Endpoint families. FDR correction is applied WITHIN a family.
PRIMARY_ENDPOINTS: List[Tuple[str, str, bool]] = [
    # (base metric name, human label, is_binary)
    ("components", "Connected components", False),
    ("cldice", "clDice", False),
    ("dice", "Dice", False),
    ("hd95", "HD95 (mm)", False),
    ("mesh_integrity_pass", "Mesh integrity", True),
]

SECONDARY_ENDPOINTS: List[Tuple[str, str, bool]] = [
    ("precision", "Precision", False),
    ("recall", "Recall", False),
    ("assd", "Average symmetric surface distance (mm)", False),
    ("centerline_mean_mm", "Centreline distance (mm)", False),
    ("pred_volume_mm3", "Predicted foreground volume (mm^3)", False),
]

# Prespecified interpretation thresholds. Fixed before results were seen. They
# classify an outcome; they never alter a parameter or a reported number.
INTERPRETATION_RULES = {
    "meaningful_component_reduction_median": -1.0,   # at least one component fewer, median
    "negligible_dice_loss": -0.01,                   # Dice may fall by at most 0.01
    "negligible_hd95_increase": 0.5,                 # HD95 may rise by at most 0.5 mm
    "negligible_surface_deviation_mm": 0.6,          # <= one voxel of mean surface displacement
}


def run_statistics(paired: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Paired tests for every variant across primary and secondary endpoints."""
    if paired.empty:
        return pd.DataFrame()
    scfg = cfg["statistics"]
    seed = int(cfg["experiment"]["seed"])
    rows: List[Dict[str, Any]] = []

    for vid, sub in paired.groupby("variant_id", sort=True):
        strategy = str(sub["strategy"].iloc[0])
        for family, endpoints in (("primary", PRIMARY_ENDPOINTS), ("secondary", SECONDARY_ENDPOINTS)):
            for base, label, is_binary in endpoints:
                o, c = f"{base}_original", f"{base}_corrected"
                if o not in sub.columns or c not in sub.columns:
                    continue
                if is_binary:
                    ok = sub[[o, c]].notna().all(axis=1)
                    if not ok.any():
                        continue
                    to_bool = lambda s: s.map(lambda v: str(v).strip().lower() in {"true", "1", "yes"})
                    r = paired_binary(to_bool(sub.loc[ok, o]), to_bool(sub.loc[ok, c]),
                                      metric=label, strategy=strategy,
                                      iterations=int(scfg["bootstrap_iterations"]),
                                      ci=float(scfg["bootstrap_ci"]), seed=seed)
                else:
                    r = paired_continuous(
                        pd.to_numeric(sub[o], errors="coerce"),
                        pd.to_numeric(sub[c], errors="coerce"),
                        metric=label, strategy=strategy,
                        mode=str(scfg["paired_test_continuous"]),
                        shapiro_alpha=float(scfg["shapiro_alpha"]),
                        iterations=int(scfg["bootstrap_iterations"]),
                        ci=float(scfg["bootstrap_ci"]), seed=seed)
                r["variant_id"] = vid
                r["endpoint_base"] = base
                r["endpoint_family"] = family
                r["primary_variant"] = bool(sub["primary_variant"].iloc[0])
                # FDR family: endpoints of one family within one variant.
                r["family"] = f"{vid}|{family}"
                rows.append(r)

    rows = apply_fdr_by_family(rows, family_key="family", alpha=float(scfg["fdr_alpha"]))
    return pd.DataFrame(rows)


def build_primary_table(
    paired: pd.DataFrame, stats_df: pd.DataFrame, variant_id: str
) -> pd.DataFrame:
    """The manuscript-ready comparison table for one variant.

    Rows with no computed metric are emitted as 'not computed' rather than
    fabricated.
    """
    sub = paired[paired["variant_id"] == variant_id]
    st = stats_df[stats_df["variant_id"] == variant_id] if not stats_df.empty else pd.DataFrame()

    def fmt(v: Any, nd: int = 4) -> str:
        try:
            f = float(v)
            return "not computed" if not np.isfinite(f) else f"{f:.{nd}f}"
        except Exception:
            return "not computed"

    spec = [("Dice", "dice", 4), ("clDice", "cldice", 4), ("HD95 (mm)", "hd95", 3),
            ("Components", "components", 2), ("Mesh integrity", "mesh_integrity_pass", 4)]
    rows: List[Dict[str, Any]] = []

    for label, base, nd in spec:
        o, c = f"{base}_original", f"{base}_corrected"
        if o not in sub.columns or c not in sub.columns:
            rows.append({"Metric": label, "Original": "not computed", "Corrected": "not computed",
                         "Paired effect": "not computed", "95% CI": "not computed",
                         "Adjusted P": "not computed", "Test": "not computed", "n": 0})
            continue
        srow = st[st["endpoint_base"] == base]
        if base == "mesh_integrity_pass":
            to_bool = lambda s: s.map(lambda v: str(v).strip().lower() in {"true", "1", "yes"})
            ok = sub[[o, c]].notna().all(axis=1)
            ov = f"{to_bool(sub.loc[ok, o]).mean():.4f}" if ok.any() else "not computed"
            cv = f"{to_bool(sub.loc[ok, c]).mean():.4f}" if ok.any() else "not computed"
        else:
            ov = fmt(pd.to_numeric(sub[o], errors="coerce").mean(), nd)
            cv = fmt(pd.to_numeric(sub[c], errors="coerce").mean(), nd)
        if srow.empty:
            rows.append({"Metric": label, "Original": ov, "Corrected": cv,
                         "Paired effect": "not computed", "95% CI": "not computed",
                         "Adjusted P": "not computed", "Test": "not computed", "n": 0})
            continue
        s0 = srow.iloc[0]
        eff = f"{fmt(s0.get('effect_size'), 4)} ({s0.get('effect_size_name', '')})"
        ci = f"[{fmt(s0.get('ci_low'), nd)}, {fmt(s0.get('ci_high'), nd)}]"
        rows.append({
            "Metric": label, "Original": ov, "Corrected": cv, "Paired effect": eff,
            "95% CI": ci, "Adjusted P": fmt(s0.get("p_adjusted_bh"), 6),
            "Test": s0.get("test", ""), "n": int(s0.get("n_pairs", 0)),
        })

    # Geometry rows come from the paired table, not from a hypothesis test.
    for label, col, nd in (("Centroid displacement (mm)", "geom_vs_original_centroid_displacement_mm", 4),
                           ("Surface deviation (mm)", "geom_vs_original_symmetric_mean_surface_distance_mm", 4)):
        if col in sub.columns and pd.to_numeric(sub[col], errors="coerce").notna().any():
            v = pd.to_numeric(sub[col], errors="coerce")
            rows.append({"Metric": label, "Original": "0 (reference)",
                         "Corrected": fmt(v.mean(), nd),
                         "Paired effect": f"{fmt(v.mean(), nd)} (mean displacement vs original geometry)",
                         "95% CI": f"[{fmt(v.quantile(0.025), nd)}, {fmt(v.quantile(0.975), nd)}]",
                         "Adjusted P": "not applicable", "Test": "descriptive", "n": int(v.notna().sum())})
        else:
            rows.append({"Metric": label, "Original": "0 (reference)", "Corrected": "not computed",
                         "Paired effect": "not computed", "95% CI": "not computed",
                         "Adjusted P": "not computed", "Test": "not computed", "n": 0})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def classify_outcome(summary_row: pd.Series) -> Dict[str, Any]:
    """Neutral classification into Outcome A / B / C using prespecified rules.

    A  topology improves substantially at negligible geometric cost
    B  topology improves but anatomical fidelity degrades
    C  correction provides little benefit

    All three are scientifically valid results. The rules are fixed; they are
    never adjusted to produce a particular label.
    """
    R = INTERPRETATION_RULES
    d_comp = float(summary_row.get("delta_components_median", np.nan))
    d_dice = float(summary_row.get("delta_dice_mean", np.nan))
    d_hd95 = float(summary_row.get("delta_hd95_mean", np.nan))
    surf = float(summary_row.get("surface_deviation_mean_mm", np.nan))

    improved = np.isfinite(d_comp) and d_comp <= R["meaningful_component_reduction_median"]
    dice_ok = (not np.isfinite(d_dice)) or d_dice >= R["negligible_dice_loss"]
    hd95_ok = (not np.isfinite(d_hd95)) or d_hd95 <= R["negligible_hd95_increase"]
    surf_ok = (not np.isfinite(surf)) or surf <= R["negligible_surface_deviation_mm"]
    cost_ok = dice_ok and hd95_ok and surf_ok

    if improved and cost_ok:
        outcome, text = "A", "Topology improves substantially with negligible geometric cost."
    elif improved and not cost_ok:
        outcome, text = "B", "Topology improves but anatomical fidelity degrades: a genuine tradeoff."
    else:
        outcome, text = "C", "Correction provides little topological benefit."

    return {
        "variant_id": summary_row.get("variant_id"),
        "strategy": summary_row.get("strategy"),
        "outcome": outcome,
        "interpretation": text,
        "delta_components_median": d_comp,
        "delta_dice_mean": d_dice,
        "delta_hd95_mean": d_hd95,
        "surface_deviation_mean_mm": surf,
        "criterion_components_met": bool(improved),
        "criterion_dice_met": bool(dice_ok),
        "criterion_hd95_met": bool(hd95_ok),
        "criterion_surface_met": bool(surf_ok),
        "rules": dict(R),
    }
