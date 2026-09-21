#!/usr/bin/env python3
"""Build a combined publication-style training curve from exported run logs.

This intentionally excludes projection-only files. It uses exported A40
train/validation CSVs as the primary source and adds the final held-out test
result as a distinct marker rather than treating it as validation data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs" / "train_runs"
AUDIT_ROOT = ROOT / "outputs" / "paper_update_audit"
SOURCE_ROOT = AUDIT_ROOT / "curve_sources"
AUDIT_FIG_ROOT = AUDIT_ROOT / "figures"
FIG_ROOT = ROOT / "figures"

FINAL_TEST = {
    "epoch_plot": 82,
    "dice_mean": 0.7952709310697139,
    "ci_low": 0.789609509870251,
    "ci_high": 0.8009323522691768,
    "median": 0.7985364227392779,
    "n": 250,
}

# The local export contains the final test outputs, but not the v14
# val_epoch_log.csv. These values are from the terminal log excerpt the user
# pasted for bioprint_v14_tversky_fn and are kept as a separate source type.
V14_TERMINAL_EXCERPT = pd.DataFrame(
    [
        {"epoch": 74, "train_dice": 0.7465, "val_soft_dice": 0.7835, "val_dice@0.5": 0.7939},
        {"epoch": 75, "train_dice": 0.7577, "val_soft_dice": 0.7842, "val_dice@0.5": 0.7922},
        {"epoch": 76, "train_dice": 0.7608, "val_soft_dice": 0.7880, "val_dice@0.5": 0.7952},
        {"epoch": 77, "train_dice": 0.7616, "val_soft_dice": 0.7850, "val_dice@0.5": 0.7914},
        {"epoch": 78, "train_dice": 0.7611, "val_soft_dice": 0.7900, "val_dice@0.5": 0.7960},
        {"epoch": 79, "train_dice": 0.7640, "val_soft_dice": 0.7918, "val_dice@0.5": 0.7972},
        {"epoch": 80, "train_dice": 0.7658, "val_soft_dice": 0.7916, "val_dice@0.5": 0.7967},
        {"epoch": 81, "train_dice": 0.7653, "val_soft_dice": 0.7917, "val_dice@0.5": 0.7963},
    ]
)

RUN_LABELS = {
    "bioprint_v6_a40_resume": "v6 A40 resume",
    "bioprint_v7_a40_roi96_finetune": "v7 ROI96 fine-tune",
    "bioprint_v8_a40_roi96_lr1e5": "v8 LR 1e-5",
    "bioprint_v9_a40_roi96_lr2e5_wd1e4": "v9 LR 2e-5",
    "bioprint_v10_a40_roi96_freshopt_lr3e5": "v10 fresh optimizer",
    "bioprint_v10_a40_roi96_cleanopt_lr3e5": "v10 clean optimizer",
    "bioprint_v11_a40_roi96_posneg3to1_lr2e5": "v11 3:1 pos-neg",
}


def load_exported_runs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for val_path in sorted(RUN_ROOT.glob("*/val_epoch_log.csv")):
        run_dir = val_path.parent
        run = run_dir.name
        train_path = run_dir / "train_epoch_log.csv"
        val = pd.read_csv(val_path)
        val["epoch"] = pd.to_numeric(val["epoch"], errors="coerce")
        val = val.dropna(subset=["epoch"]).copy()
        val["epoch"] = val["epoch"].astype(int)
        val["run"] = run
        val["run_label"] = RUN_LABELS.get(run, run)
        val["source_type"] = "exported_csv"
        val["source_file"] = str(val_path.relative_to(ROOT))

        if train_path.exists():
            train = pd.read_csv(train_path)
            train["epoch"] = pd.to_numeric(train["epoch"], errors="coerce")
            train = train.dropna(subset=["epoch"]).copy()
            train["epoch"] = train["epoch"].astype(int)
            keep = [c for c in ["epoch", "train_loss", "train_dice", "train_bce", "lr", "seconds"] if c in train]
            val = val.merge(train[keep], on="epoch", how="left")

        frames.append(val)

    if not frames:
        raise FileNotFoundError(f"No val_epoch_log.csv files found under {RUN_ROOT}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    # De-duplicate exact repeated rows from restarted tails, but preserve
    # genuinely different runs at the same epoch.
    subset = ["run", "epoch", "val_soft_dice", "val_dice@0.5", "train_dice"]
    subset = [c for c in subset if c in combined.columns]
    combined = combined.drop_duplicates(subset=subset, keep="last")
    return combined


def add_v14_excerpt(combined: pd.DataFrame) -> pd.DataFrame:
    v14 = V14_TERMINAL_EXCERPT.copy()
    v14["run"] = "bioprint_v14_tversky_fn"
    v14["run_label"] = "v14 Tversky fine-tune"
    v14["source_type"] = "terminal_excerpt"
    v14["source_file"] = "user-provided terminal excerpt; v14 CSV not present in local export"
    for col in combined.columns:
        if col not in v14.columns:
            v14[col] = np.nan
    return pd.concat([combined, v14[combined.columns]], ignore_index=True, sort=False)


def best_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for epoch, group in df.dropna(subset=["val_dice@0.5"]).groupby("epoch"):
        idx = group["val_dice@0.5"].astype(float).idxmax()
        rows.append(df.loc[idx])
    out = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    out["best_so_far_val_dice@0.5"] = out["val_dice@0.5"].cummax()
    return out


def write_source_notes(local_only: pd.DataFrame, with_v14: pd.DataFrame, best: pd.DataFrame) -> None:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    notes = f"""# Combined Training Curve Source Notes

Generated by `scripts/build_combined_training_curve.py`.

## Included sources

- Exported A40 run CSVs under `outputs/train_runs/*/train_epoch_log.csv` and `outputs/train_runs/*/val_epoch_log.csv`.
- Final held-out test summary from `outputs/final_test_250/summary_metrics.json`, plotted as a separate marker/shaded CI rather than as validation.
- `bioprint_v14_tversky_fn` epochs 74-81 are included only in the `*_with_v14_excerpt*` outputs from the terminal log excerpt pasted by the user. The original v14 `val_epoch_log.csv` is not present in the local export.

## Excluded sources

- `epoch_metrics_with_projection.csv` was excluded because it contains projected future values, not observed measurements.
- Top-level `epoch_metrics.csv` was not merged into the main A40 curve because its run provenance does not match the exported A40 training sessions used for final model selection.

## Generated data

- Local exported rows: {len(local_only)}
- Rows with v14 excerpt: {len(with_v14)}
- Consolidated best-by-epoch rows: {len(best)}
- Best validation Dice@0.5 in plotted training history: {best['val_dice@0.5'].max():.4f}
- Final held-out test Dice@0.5: {FINAL_TEST['dice_mean']:.4f} (95% CI {FINAL_TEST['ci_low']:.4f}-{FINAL_TEST['ci_high']:.4f}, N={FINAL_TEST['n']})
"""
    (SOURCE_ROOT / "training_curve_source_notes.md").write_text(notes)
    caption = (
        "Validation Dice@0.5 trajectory across A40 fine-tuning. Faint gray points show raw logged "
        "validation measurements from exported training sessions, while the blue curve shows an "
        "exponentially smoothed selected validation trajectory for visual clarity. The dotted blue "
        "line marks the running best validation Dice. The orange line and shaded band indicate the "
        f"full held-out 250-case test Dice@0.5 mean ({FINAL_TEST['dice_mean']:.4f}) and 95% "
        f"confidence interval ({FINAL_TEST['ci_low']:.4f}-{FINAL_TEST['ci_high']:.4f}); the test "
        "result is displayed as outcome context and was not used as a validation point."
    )
    (SOURCE_ROOT / "combined_training_curve_caption.md").write_text(caption + "\n")


def plot_curve(df: pd.DataFrame, best: pd.DataFrame, out_name: str, include_v14: bool) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.labelsize": 9.2,
            "axes.titlesize": 10.2,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 450,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.15), constrained_layout=False)

    raw = df.dropna(subset=["val_dice@0.5"]).sort_values(["epoch", "run"])
    ax.scatter(
        raw["epoch"],
        raw["val_dice@0.5"],
        s=13,
        color="#6b7280",
        alpha=0.23,
        linewidth=0,
        label="Logged validation Dice@0.5",
        zorder=1,
    )

    # At each epoch, if multiple logged branches exist, select the strongest
    # validation value. The displayed blue curve is smoothed only for visual
    # readability; the faint gray points preserve the raw measurements.
    selected = best.dropna(subset=["val_dice@0.5"]).sort_values("epoch").copy()
    selected["smoothed_val_dice@0.5"] = (
        selected["val_dice@0.5"].astype(float).ewm(span=4, adjust=False).mean()
    )
    ax.plot(
        selected["epoch"],
        selected["smoothed_val_dice@0.5"],
        color="#005ea8",
        linewidth=2.45,
        solid_capstyle="round",
        label="Smoothed selected validation Dice@0.5",
        zorder=5,
    )
    ax.plot(
        selected["epoch"],
        selected["best_so_far_val_dice@0.5"],
        color="#005ea8",
        linewidth=1.25,
        linestyle=":",
        alpha=0.78,
        label="Running best validation Dice@0.5",
        zorder=4,
    )

    best_idx = selected["val_dice@0.5"].astype(float).idxmax()
    best_row = selected.loc[best_idx]
    ax.scatter(
        [best_row["epoch"]],
        [best_row["val_dice@0.5"]],
        s=48,
        color="#005ea8",
        edgecolor="white",
        linewidth=1.0,
        label=f"Best validation checkpoint ({best_row['val_dice@0.5']:.3f})",
        zorder=7,
    )

    ax.axhspan(
        FINAL_TEST["ci_low"],
        FINAL_TEST["ci_high"],
        color="#d95f02",
        alpha=0.10,
        linewidth=0,
        label="Held-out test Dice@0.5 95% CI",
        zorder=0,
    )
    ax.axhline(
        FINAL_TEST["dice_mean"],
        color="#d95f02",
        linewidth=1.45,
        linestyle="--",
        alpha=0.78,
        label=f"Held-out test mean {FINAL_TEST['dice_mean']:.3f} (N={FINAL_TEST['n']})",
        zorder=3,
    )

    ax.set_title("Validation Dice Trajectory Across A40 Fine-Tuning", pad=10)
    ax.set_xlabel("Training epoch", labelpad=7)
    ax.set_ylabel("Dice coefficient")
    ax.set_xlim(33, 84)
    ax.set_ylim(0.20, 0.83)
    ax.set_yticks(np.arange(0.20, 0.85, 0.10))
    ax.grid(True, axis="y", color="#d9dde3", linewidth=0.7, alpha=0.75)
    ax.grid(True, axis="x", color="#eef0f3", linewidth=0.5, alpha=0.55)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#d7d7d7",
        ncol=2,
        columnspacing=0.9,
        handlelength=2.2,
    )
    legend.get_frame().set_linewidth(0.6)
    fig.subplots_adjust(left=0.095, right=0.99, top=0.88, bottom=0.31)

    for target in [FIG_ROOT, AUDIT_FIG_ROOT]:
        target.mkdir(parents=True, exist_ok=True)
        png = target / f"{out_name}.png"
        pdf = target / f"{out_name}.pdf"
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    AUDIT_FIG_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    local_only = load_exported_runs()
    with_v14 = add_v14_excerpt(local_only)
    best_local = best_by_epoch(local_only)
    best_with_v14 = best_by_epoch(with_v14)

    local_only.to_csv(SOURCE_ROOT / "combined_a40_training_history_local_exported_only.csv", index=False)
    with_v14.to_csv(SOURCE_ROOT / "combined_a40_training_history_with_v14_excerpt.csv", index=False)
    best_local.to_csv(SOURCE_ROOT / "combined_a40_training_best_by_epoch_local_exported_only.csv", index=False)
    best_with_v14.to_csv(SOURCE_ROOT / "combined_a40_training_best_by_epoch_with_v14_excerpt.csv", index=False)

    plot_curve(local_only, best_local, "figure4_training_curves_combined_local_only", include_v14=False)
    plot_curve(with_v14, best_with_v14, "figure4_training_curves_combined_with_v14_excerpt", include_v14=True)

    # Keep the manuscript-facing figure name pointed to the richest curve, but
    # keep both variants available for audit/review.
    for suffix in ["png", "pdf"]:
        src = FIG_ROOT / f"figure4_training_curves_combined_with_v14_excerpt.{suffix}"
        dst = FIG_ROOT / f"figure4_training_curves.{suffix}"
        dst.write_bytes(src.read_bytes())

    write_source_notes(local_only, with_v14, best_with_v14)

    print("Wrote:")
    print(FIG_ROOT / "figure4_training_curves.png")
    print(FIG_ROOT / "figure4_training_curves_combined_with_v14_excerpt.png")
    print(FIG_ROOT / "figure4_training_curves_combined_local_only.png")
    print(SOURCE_ROOT / "combined_a40_training_history_with_v14_excerpt.csv")
    print(SOURCE_ROOT / "training_curve_source_notes.md")
    print(SOURCE_ROOT / "combined_training_curve_caption.md")


if __name__ == "__main__":
    main()
