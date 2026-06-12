#!/usr/bin/env python3
"""Generate the additional manuscript figures from the real 250-case test results.

All figures are derived strictly from committed artifacts:
  - outputs/final_test_250/per_case_metrics.csv          (per-case Dice/clDice/HD95/precision/recall)
  - outputs/final_test_250/seg_to_mesh_correlations.csv  (paired seg->mesh Spearman matrix)
  - outputs/final_test_250/case_outputs/<id>/seg_mask_0.5.nii.gz  (per-case prediction, 0.6mm grid)
  - Data/all/<range>/<id>.{img,label}.nii.gz             (ground-truth CTA + vessel label)

No metrics are recomputed for the headline; the qualitative panel only *displays* the
already-evaluated prediction next to the ground truth, with an alignment self-check that
reproduces the reported per-case Dice so the overlay is provably correct.

Run in the `isef` conda env:
  source ~/opt/miniconda3/etc/profile.d/conda.sh && conda activate isef
  python paper/figures/make_extra_figures.py
"""
from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGDIR = os.path.join(REPO, "paper", "figures")
FINAL = os.path.join(REPO, "outputs", "final_test_250")
PER_CASE = os.path.join(FINAL, "per_case_metrics.csv")
CORR = os.path.join(FINAL, "seg_to_mesh_correlations.csv")
CASE_OUT = os.path.join(FINAL, "case_outputs")

plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "savefig.bbox": "tight"})


def _load_metrics() -> pd.DataFrame:
    df = pd.read_csv(PER_CASE)
    df = df[df["status"] == "ok"] if "status" in df.columns else df
    df["case_id"] = df["case_id"].astype(int)
    return df


# --------------------------------------------------------------------------------------
# Figure: qualitative segmentation overlay (3 representative cases)
# --------------------------------------------------------------------------------------
def _find_gt_paths(case_id: int):
    img = glob.glob(os.path.join(REPO, "Data", "all", "*", f"{case_id}.img.nii.gz"))
    lab = glob.glob(os.path.join(REPO, "Data", "all", "*", f"{case_id}.label.nii.gz"))
    return (img[0] if img else None), (lab[0] if lab else None)


def _pick_representative(df: pd.DataFrame):
    """Pick lower / median / higher accuracy cases among reasonably sized vessel trees."""
    have_gt = df["case_id"].apply(lambda c: _find_gt_paths(c)[1] is not None)
    cand = df[have_gt].copy()
    vol_floor = cand["gt_voxels"].quantile(0.40)
    cand = cand[cand["gt_voxels"] >= vol_floor]
    cand = cand.sort_values("dice@0.5").reset_index(drop=True)
    picks = []
    for q, lab in [(0.15, "lower"), (0.50, "median"), (0.85, "higher")]:
        target = cand["dice@0.5"].quantile(q)
        row = cand.iloc[(cand["dice@0.5"] - target).abs().argmin()]
        picks.append((int(row["case_id"]), float(row["dice@0.5"]),
                      float(row["cldice@0.5"]), lab))
    return picks


def fig_qualitative(df: pd.DataFrame):
    import nibabel as nib
    from nibabel.processing import resample_from_to

    picks = _pick_representative(df)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))

    for col, (cid, dice, cldice, acclab) in enumerate(picks):
        pred_p = os.path.join(CASE_OUT, str(cid), "seg_mask_0.5.nii.gz")
        img_p, lab_p = _find_gt_paths(cid)
        pred = nib.load(pred_p)
        pred_a = np.asarray(pred.dataobj) > 0
        # Resample GT + CTA onto the prediction's 0.6 mm grid (affine-aware).
        gt = np.asarray(resample_from_to(nib.load(lab_p), (pred.shape, pred.affine), order=0).dataobj) > 0
        cta = np.asarray(resample_from_to(nib.load(img_p), (pred.shape, pred.affine), order=1).dataobj)

        # Alignment self-check: Dice of resampled GT vs prediction must track the reported value.
        inter = np.logical_and(pred_a, gt).sum()
        check_dice = 2 * inter / (pred_a.sum() + gt.sum() + 1e-8)
        if abs(check_dice - dice) > 0.06:
            print(f"  WARNING case {cid}: alignment-check Dice {check_dice:.3f} "
                  f"vs reported {dice:.3f} (>0.06 gap)")
        else:
            print(f"  case {cid} ({acclab}): alignment OK "
                  f"(check Dice {check_dice:.3f} vs reported {dice:.3f})")

        # Axial maximum-intensity projections (project over z) show the whole vessel tree.
        cta_w = np.clip(cta, -200, 700)
        cta_w = (cta_w + 200) / 900.0
        cta_mip = np.rot90(cta_w.max(axis=2))
        gt_mip = np.rot90(gt.max(axis=2))
        pred_mip = np.rot90(pred_a.max(axis=2))

        ax = axes[0, col]
        ax.imshow(cta_mip, cmap="gray", origin="lower")
        ax.set_title(f"Case {cid} — {acclab} accuracy\nCTA (axial MIP)")
        ax.axis("off")

        ax = axes[1, col]
        rgb = np.zeros((*gt_mip.shape, 3))
        rgb[..., 1] = gt_mip      # ground truth -> green
        rgb[..., 0] = pred_mip    # prediction   -> red (overlap -> yellow)
        ax.imshow(rgb, origin="lower")
        ax.set_title(f"Dice {dice:.3f}  |  clDice {cldice:.3f}")
        ax.axis("off")

    # Legend.
    from matplotlib.patches import Patch
    handles = [Patch(color=(0, 1, 0), label="Ground truth"),
               Patch(color=(1, 0, 0), label="Prediction"),
               Patch(color=(1, 1, 0), label="Overlap (TP)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Qualitative coronary vessel segmentation on held-out test cases",
                 y=1.00, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.99))
    out = os.path.join(FIGDIR, "fig_qualitative_segmentation.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------------------
# Figure: seg-metric x mesh-QC Spearman correlation heatmap
# --------------------------------------------------------------------------------------
def fig_corr_heatmap():
    df = pd.read_csv(CORR)
    df = df[df["seg_metric"] != "seg_metric"].copy()  # drop any stray duplicated header row
    df["spearman_r"] = df["spearman_r"].astype(float)
    df["spearman_p_fdr"] = df["spearman_p_fdr"].astype(float)

    seg_order = ["Dice@0.5", "clDice@0.5", "precision@0.5", "recall@0.5", "HD95"]
    mesh_pretty = {
        "connected_component_count": "Connected\ncomponents",
        "wall_thickness_compliance_fraction": "Wall-thickness\ncompliance",
        "non_manifold_edge_count": "Non-manifold\nedges",
        "surface_roughness_post_mean": "Surface\nroughness",
    }
    mesh_order = list(mesh_pretty.keys())
    seg_order = [s for s in seg_order if s in df["seg_metric"].unique()]

    R = np.full((len(seg_order), len(mesh_order)), np.nan)
    P = np.full_like(R, np.nan)
    for i, s in enumerate(seg_order):
        for j, m in enumerate(mesh_order):
            sub = df[(df["seg_metric"] == s) & (df["mesh_metric"] == m)]
            if len(sub):
                R[i, j] = sub["spearman_r"].iloc[0]
                P[i, j] = sub["spearman_p_fdr"].iloc[0]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(mesh_order)))
    ax.set_xticklabels([mesh_pretty[m] for m in mesh_order])
    ax.set_yticks(range(len(seg_order)))
    ax.set_yticklabels(seg_order)
    for i in range(len(seg_order)):
        for j in range(len(mesh_order)):
            if np.isnan(R[i, j]):
                continue
            star = "*" if P[i, j] < 0.05 else ""
            ax.text(j, i, f"{R[i, j]:.2f}{star}", ha="center", va="center",
                    color="white" if abs(R[i, j]) > 0.33 else "black", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman $\\rho$")
    ax.set_title("Segmentation accuracy vs mesh-quality coupling (250-case test)\n"
                 "* FDR-corrected $p<0.05$", fontsize=11)
    out = os.path.join(FIGDIR, "fig_metric_mesh_heatmap.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------------------
# Figure: HD95 distribution
# --------------------------------------------------------------------------------------
def fig_hd95(df: pd.DataFrame):
    hd = df["hd95@0.5"].astype(float).dropna()
    hd = hd[np.isfinite(hd)]
    med, mean = float(hd.median()), float(hd.mean())
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(np.clip(hd, 0, 25), bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(med, color="#C44E52", lw=2, label=f"median = {med:.2f} mm")
    ax.axvline(mean, color="#55A868", lw=2, ls="--", label=f"mean = {mean:.2f} mm")
    ax.set_xlabel("HD95 @ 0.5 (mm)")
    ax.set_ylabel("Number of test cases")
    ax.set_title("Distribution of 95th-percentile Hausdorff distance (250-case test)")
    ax.legend()
    out = os.path.join(FIGDIR, "fig_hd95_distribution.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------------------
# Figure: clDice vs Dice per-case scatter
# --------------------------------------------------------------------------------------
def fig_cldice_vs_dice(df: pd.DataFrame):
    d = df["dice@0.5"].astype(float)
    c = df["cldice@0.5"].astype(float)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(d, c, s=18, alpha=0.6, color="#4C72B0", edgecolor="none")
    lo, hi = 0.45, 1.0
    ax.plot([lo, hi], [lo, hi], color="0.4", ls="--", lw=1, label="identity ($y=x$)")
    ax.axhline(c.mean(), color="#C44E52", lw=1, ls=":")
    ax.axvline(d.mean(), color="#C44E52", lw=1, ls=":")
    ax.scatter([d.mean()], [c.mean()], color="#C44E52", s=70, zorder=5,
               label=f"mean ({d.mean():.3f}, {c.mean():.3f})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Dice@0.5 (voxel overlap)")
    ax.set_ylabel("clDice@0.5 (topology-aware)")
    ax.set_title("Topology-aware vs voxel-overlap accuracy\nper held-out test case")
    ax.legend(loc="lower right")
    out = os.path.join(FIGDIR, "fig_cldice_vs_dice.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------------------
# Figure: precision vs recall scatter colored by Dice
# --------------------------------------------------------------------------------------
def fig_precision_recall(df: pd.DataFrame):
    p = df["precision@0.5"].astype(float)
    r = df["recall@0.5"].astype(float)
    d = df["dice@0.5"].astype(float)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    sc = ax.scatter(p, r, c=d, cmap="viridis", s=22, alpha=0.85, edgecolor="none")
    ax.plot([0.4, 1.0], [0.4, 1.0], color="0.4", ls="--", lw=1, label="precision = recall")
    ax.scatter([p.mean()], [r.mean()], color="#C44E52", s=80, marker="X", zorder=5,
               label=f"mean ({p.mean():.3f}, {r.mean():.3f})")
    ax.set_xlabel("Precision@0.5")
    ax.set_ylabel("Recall@0.5")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Per-case precision vs recall (250-case test)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Dice@0.5")
    ax.legend(loc="lower left")
    out = os.path.join(FIGDIR, "fig_precision_recall.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main():
    df = _load_metrics()
    print(f"loaded {len(df)} cases from {PER_CASE}")
    print("[1/5] qualitative overlay")
    fig_qualitative(df)
    print("[2/5] correlation heatmap")
    fig_corr_heatmap()
    print("[3/5] HD95 distribution")
    fig_hd95(df)
    print("[4/5] clDice vs Dice")
    fig_cldice_vs_dice(df)
    print("[5/5] precision vs recall")
    fig_precision_recall(df)
    print("done")


if __name__ == "__main__":
    main()
