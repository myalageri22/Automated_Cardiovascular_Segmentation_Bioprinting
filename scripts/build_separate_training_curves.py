#!/usr/bin/env python3
"""Build one standalone training/validation curve per run.

The output is intended for manuscript figure selection. Each run is plotted as
its own single-panel PNG/PDF. The v14 panel is built from the terminal excerpt
because its CSV was not present in the local export.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_combined_training_curve import (
    FINAL_TEST,
    RUN_LABELS,
    SOURCE_ROOT,
    add_v14_excerpt,
    load_exported_runs,
)


ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = ROOT / "figures" / "training_curves_separate"
AUDIT_FIG_ROOT = ROOT / "outputs" / "paper_update_audit" / "figures" / "training_curves_separate"

RUN_ORDER = [
    "bioprint_v6_a40_resume",
    "bioprint_v7_a40_roi96_finetune",
    "bioprint_v8_a40_roi96_lr1e5",
    "bioprint_v9_a40_roi96_lr2e5_wd1e4",
    "bioprint_v10_a40_roi96_freshopt_lr3e5",
    "bioprint_v10_a40_roi96_cleanopt_lr3e5",
    "bioprint_v11_a40_roi96_posneg3to1_lr2e5",
    "bioprint_v14_tversky_fn",
]

RUN_SHORT = {
    "bioprint_v6_a40_resume": "v6_a40_resume",
    "bioprint_v7_a40_roi96_finetune": "v7_roi96_finetune",
    "bioprint_v8_a40_roi96_lr1e5": "v8_roi96_lr1e5",
    "bioprint_v9_a40_roi96_lr2e5_wd1e4": "v9_roi96_lr2e5_wd1e4",
    "bioprint_v10_a40_roi96_freshopt_lr3e5": "v10_freshopt_lr3e5",
    "bioprint_v10_a40_roi96_cleanopt_lr3e5": "v10_cleanopt_lr3e5",
    "bioprint_v11_a40_roi96_posneg3to1_lr2e5": "v11_posneg3to1_lr2e5",
    "bioprint_v14_tversky_fn": "v14_tversky_fn_excerpt",
}

RUN_LABELS_EXTENDED = {
    **RUN_LABELS,
    "bioprint_v14_tversky_fn": "v14 Tversky fine-tune",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.6,
            "legend.fontsize": 6.3,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 450,
        }
    )


def line_or_point(ax, x, y, *, label: str, color: str, linestyle: str = "-", marker: str = "o", zorder: int = 3):
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if clean.empty:
        return
    if len(clean) == 1:
        ax.scatter(clean["x"], clean["y"], s=38, color=color, marker=marker, label=label, zorder=zorder)
    else:
        ax.plot(
            clean["x"],
            clean["y"],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            marker=marker,
            markersize=3.8,
            markerfacecolor="white" if marker == "o" else color,
            markeredgewidth=1.1,
            label=label,
            zorder=zorder,
        )


def plot_single_run(run_df: pd.DataFrame, run: str) -> dict[str, object]:
    run_df = run_df.sort_values("epoch").copy()
    label = RUN_LABELS_EXTENDED.get(run, run)
    source_type = str(run_df["source_type"].iloc[0])
    source_file = str(run_df["source_file"].iloc[0])
    out_stem = f"{RUN_SHORT.get(run, run)}_training_curve"

    fig, ax = plt.subplots(figsize=(5.5, 4.05), constrained_layout=False)

    ax.axhspan(
        FINAL_TEST["ci_low"],
        FINAL_TEST["ci_high"],
        color="#d95f02",
        alpha=0.11,
        linewidth=0,
        label="Final test Dice@0.5 95% CI",
        zorder=0,
    )
    ax.axhline(
        FINAL_TEST["dice_mean"],
        color="#d95f02",
        linewidth=1.05,
        linestyle="--",
        alpha=0.72,
        label=f"Final test mean {FINAL_TEST['dice_mean']:.3f}",
        zorder=1,
    )

    line_or_point(
        ax,
        run_df["epoch"],
        run_df.get("val_dice@0.5"),
        label="Validation Dice@0.5",
        color="#005ea8",
        linestyle="-",
        marker="o",
        zorder=5,
    )
    if "val_soft_dice" in run_df:
        line_or_point(
            ax,
            run_df["epoch"],
            run_df["val_soft_dice"],
            label="Validation soft Dice",
            color="#7a3db8",
            linestyle=":",
            marker="s",
            zorder=4,
        )
    if "train_dice" in run_df:
        line_or_point(
            ax,
            run_df["epoch"],
            run_df["train_dice"],
            label="Training Dice",
            color="#2f7f4f",
            linestyle="--",
            marker="^",
            zorder=3,
        )

    metric = pd.to_numeric(run_df.get("val_dice@0.5"), errors="coerce")
    best_idx = metric.idxmax() if metric.notna().any() else None
    best_epoch = int(run_df.loc[best_idx, "epoch"]) if best_idx is not None else None
    best_dice = float(run_df.loc[best_idx, "val_dice@0.5"]) if best_idx is not None else np.nan
    if best_idx is not None:
        ax.scatter(
            [best_epoch],
            [best_dice],
            s=54,
            color="#005ea8",
            edgecolor="white",
            linewidth=1.0,
            zorder=8,
        )

    x_min = int(run_df["epoch"].min())
    x_max = int(run_df["epoch"].max())
    pad = max(1, int(round((x_max - x_min) * 0.12))) if x_max > x_min else 1
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(0.20, 0.83)
    ax.set_yticks(np.arange(0.20, 0.85, 0.10))
    ax.set_xlabel("Training epoch", labelpad=7)
    ax.set_ylabel("Dice coefficient")
    ax.set_title(label, pad=18)
    ax.grid(True, axis="y", color="#d9dde3", linewidth=0.7, alpha=0.78)
    ax.grid(True, axis="x", color="#eef0f3", linewidth=0.5, alpha=0.55)

    subtitle = f"epochs {x_min}-{x_max}; best val Dice@0.5 = {best_dice:.3f}" if best_idx is not None else f"epochs {x_min}-{x_max}"
    if source_type == "terminal_excerpt":
        subtitle += " (terminal excerpt)"
    ax.text(
        0.5,
        1.018,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#30343a",
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#d5d5d5", "lw": 0.55, "alpha": 0.92},
    )

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.52),
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#d7d7d7",
        ncol=3,
        columnspacing=0.72,
        handlelength=1.35,
    )
    legend.get_frame().set_linewidth(0.6)

    fig.subplots_adjust(left=0.12, right=0.985, top=0.80, bottom=0.55)

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    AUDIT_FIG_ROOT.mkdir(parents=True, exist_ok=True)
    png = FIG_ROOT / f"{out_stem}.png"
    pdf = FIG_ROOT / f"{out_stem}.pdf"
    audit_png = AUDIT_FIG_ROOT / f"{out_stem}.png"
    audit_pdf = AUDIT_FIG_ROOT / f"{out_stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(audit_png, bbox_inches="tight")
    fig.savefig(audit_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "run": run,
        "label": label,
        "source_type": source_type,
        "source_file": source_file,
        "epoch_min": x_min,
        "epoch_max": x_max,
        "best_epoch": best_epoch,
        "best_val_dice@0.5": best_dice,
        "png": str(png.relative_to(ROOT)),
        "pdf": str(pdf.relative_to(ROOT)),
    }


def main() -> None:
    configure_style()
    exported = load_exported_runs()
    all_runs = add_v14_excerpt(exported)

    rows = []
    for run in RUN_ORDER:
        run_df = all_runs[all_runs["run"].eq(run)]
        if run_df.empty:
            continue
        rows.append(plot_single_run(run_df, run))

    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(SOURCE_ROOT / "separate_training_curve_manifest.csv", index=False)

    notes = [
        "# Separate Training Curve Figures",
        "",
        "Generated by `scripts/build_separate_training_curves.py`.",
        "",
        "Each run is plotted as its own standalone PNG/PDF. The final held-out test mean and 95% CI are shown as outcome context, not as validation data.",
        "",
        "The v14 figure uses the terminal excerpt because the v14 `val_epoch_log.csv` was not present in the local export.",
        "",
        "## Files",
        "",
    ]
    for row in rows:
        notes.append(f"- {row['label']}: `{row['png']}`")
    (SOURCE_ROOT / "separate_training_curve_notes.md").write_text("\n".join(notes) + "\n")

    print(f"Wrote {len(rows)} separate training curve figures")
    print(FIG_ROOT)
    print(SOURCE_ROOT / "separate_training_curve_manifest.csv")
    print(SOURCE_ROOT / "separate_training_curve_notes.md")


if __name__ == "__main__":
    main()
