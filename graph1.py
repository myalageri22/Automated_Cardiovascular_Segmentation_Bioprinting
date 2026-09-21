import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ====== CONFIG ======
csv_path = "epoch_metrics.csv"     # <-- change if needed
TOTAL_EPOCHS = 60
TARGET_VAL_DICE = 0.84
DICE_COL = "val_soft_dice"         # or "val_dice_0_5"
LOSS_COLS = ["train_loss", "val_loss"]

# ====== LOAD ======
df = pd.read_csv(csv_path)

def recent_linear_slope(x, y):
    """Least-squares slope of y vs x."""
    A = np.vstack([x, np.ones_like(x)]).T
    m, _b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m

def anchored_saturating_projection(epochs_full, e0, y0, yT, slope0=None, monotone=None):
    """
    Project from (e0,y0) to (TOTAL_EPOCHS,yT) using a normalized saturating exponential:
      y(t) = y0 + (yT-y0) * (1-exp(-k*t)) / (1-exp(-k*T)),
    with t = epoch - e0 and T = TOTAL_EPOCHS - e0.

    If slope0 is provided, choose k so dy/dt at t=0 roughly matches slope0.
    """
    T = int(epochs_full.max() - e0)
    if T <= 0:
        return np.full_like(epochs_full, y0, dtype=float)

    delta = (yT - y0)
    avg = delta / T  # slope if near-linear

    # If slope0 has wrong sign (noise), fall back to avg
    if slope0 is None or (delta > 0 and slope0 < 0) or (delta < 0 and slope0 > 0) or slope0 == 0:
        slope0 = avg

    def slope_at_0(k):
        denom = 1.0 - np.exp(-k * T)
        return (delta * k) / denom

    # Solve slope_at_0(k) = slope0
    k = 0.05
    try:
        if delta != 0:
            if abs(slope0) <= abs(avg) + 1e-12:
                k = 1e-6  # near-linear
            else:
                f = lambda kk: slope_at_0(kk) - slope0
                k_lo, k_hi = 1e-8, 10.0
                while f(k_hi) * f(k_lo) > 0 and k_hi < 1e6:
                    k_hi *= 10
                if f(k_hi) * f(k_lo) < 0:
                    k = brentq(f, k_lo, k_hi, maxiter=200)
    except Exception:
        k = 0.05

    t = np.clip(epochs_full - e0, 0, None)
    denom = 1.0 - np.exp(-k * T)

    if abs(denom) < 1e-12:  # linear fallback
        y = y0 + delta * (t / T)
    else:
        y = y0 + delta * (1.0 - np.exp(-k * t)) / denom

    if monotone == "increasing":
        y = np.maximum.accumulate(y)
    elif monotone == "decreasing":
        y = np.minimum.accumulate(y)

    return y

epochs_full = np.arange(1, TOTAL_EPOCHS + 1)
proj = pd.DataFrame({"epoch": epochs_full}).merge(df, on="epoch", how="left")
last_actual_epoch = int(df["epoch"].max())

# ====== DICE PROJECTION (force chosen val dice to reach 0.84 by epoch 60) ======
mask = df[DICE_COL].notna()
e0 = int(df.loc[mask, "epoch"].max())
y0 = float(df.loc[df["epoch"] == e0, DICE_COL].iloc[0])

tail_n = min(5, mask.sum())
e_tail = df.loc[mask, "epoch"].to_numpy()[-tail_n:]
y_tail = df.loc[mask, DICE_COL].to_numpy()[-tail_n:]
slope0 = recent_linear_slope(e_tail, y_tail) if len(e_tail) >= 2 else None

yT = max(TARGET_VAL_DICE, y0)

proj[f"proj_{DICE_COL}"] = anchored_saturating_projection(
    epochs_full=epochs_full, e0=e0, y0=y0, yT=yT, slope0=slope0, monotone="increasing"
)
proj[f"{DICE_COL}_plot"] = proj[DICE_COL].where(proj[DICE_COL].notna(), proj[f"proj_{DICE_COL}"])

# Optional: also project train_dice (not forced)
if "train_dice" in df.columns:
    mask = df["train_dice"].notna()
    e0 = int(df.loc[mask, "epoch"].max())
    y0 = float(df.loc[df["epoch"] == e0, "train_dice"].iloc[0])

    tail_n = min(5, mask.sum())
    e_tail = df.loc[mask, "epoch"].to_numpy()[-tail_n:]
    y_tail = df.loc[mask, "train_dice"].to_numpy()[-tail_n:]
    slope0 = recent_linear_slope(e_tail, y_tail) if len(e_tail) >= 2 else None

    remaining = TOTAL_EPOCHS - e0
    yT = np.clip(y0 + (slope0 or 0) * remaining, 0, 0.99)

    proj["proj_train_dice"] = anchored_saturating_projection(
        epochs_full=epochs_full, e0=e0, y0=y0, yT=float(yT), slope0=slope0, monotone="increasing"
    )
    proj["train_dice_plot"] = proj["train_dice"].where(proj["train_dice"].notna(), proj["proj_train_dice"])

# ====== LOSS PROJECTION (continue recent trend smoothly to epoch 60) ======
for col in LOSS_COLS:
    if col not in df.columns:
        continue
    mask = df[col].notna()
    e0 = int(df.loc[mask, "epoch"].max())
    y0 = float(df.loc[df["epoch"] == e0, col].iloc[0])

    tail_n = min(5, mask.sum())
    e_tail = df.loc[mask, "epoch"].to_numpy()[-tail_n:]
    y_tail = df.loc[mask, col].to_numpy()[-tail_n:]
    slope0 = recent_linear_slope(e_tail, y_tail) if len(e_tail) >= 2 else None

    remaining = TOTAL_EPOCHS - e0
    yT = max(0.0, y0 + (slope0 or 0.0) * remaining)  # clamp to non-negative

    proj[f"proj_{col}"] = anchored_saturating_projection(
        epochs_full=epochs_full, e0=e0, y0=y0, yT=float(yT), slope0=slope0, monotone="decreasing"
    )
    proj[f"{col}_plot"] = proj[col].where(proj[col].notna(), proj[f"proj_{col}"])

# ====== SAVE TABLE ======
proj.to_csv("epoch_metrics_with_projection.csv", index=False)

# ====== PLOTS (no subplots) ======
# Dice plot
plt.figure()
if "train_dice_plot" in proj.columns:
    plt.plot(proj["epoch"], proj["train_dice_plot"], label="train_dice (actual+proj)")
plt.plot(proj["epoch"], proj[f"{DICE_COL}_plot"], label=f"{DICE_COL} (actual+proj)")
plt.axvline(last_actual_epoch, linestyle="--", label="projection start")
plt.xlabel("epoch"); plt.ylabel("dice"); plt.title("Dice progression (actual + projected to 60 epochs)")
plt.legend()
plt.show()

# Loss plot
plt.figure()
plt.plot(proj["epoch"], proj["train_loss_plot"], label="train_loss (actual+proj)")
plt.plot(proj["epoch"], proj["val_loss_plot"], label="val_loss (actual+proj)")
plt.axvline(last_actual_epoch, linestyle="--", label="projection start")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss progression (actual + projected to 60 epochs)")
plt.legend()
plt.show()
