import os
import re
from glob import glob

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Find all label files in your structure
# -----------------------------
BASE = os.environ.get("IMAGECAS_PROCESSED_ROOT", "data/processed/all")

label_paths = []
label_paths += glob(os.path.join(BASE, "*.label.nii.gz"))                # e.g., 1000.label.nii.gz at root
label_paths += glob(os.path.join(BASE, "*", "*.label.nii.gz"))          # e.g., 801-1000/801.label.nii.gz

# Optional: keep only those that match "<id>.label.nii.gz"
label_paths = sorted(label_paths)

print(f"Found {len(label_paths)} label files.")

# -----------------------------
# 2) Helper to extract case id for nicer plots
# -----------------------------
def case_id_from_path(p: str):
    m = re.search(r"[/\\](\d+)\.label\.nii\.gz$", p)
    return int(m.group(1)) if m else None

# -----------------------------
# 3) Compute sparsity (positive voxel fraction) for N cases
# -----------------------------
N = 200  # change to 500, 1000, etc.
fractions = []
case_ids = []
used_paths = []

for p in label_paths[:N]:
    try:
        lab = nib.load(p).get_fdata()
        pos = np.sum(lab > 0)
        total = lab.size
        frac = pos / total

        fractions.append(frac)
        cid = case_id_from_path(p)
        case_ids.append(cid if cid is not None else len(case_ids))
        used_paths.append(p)

    except Exception as e:
        print(f"[WARN] Skipping {p} due to error: {e}")

fractions = np.array(fractions, dtype=np.float64)
case_ids = np.array(case_ids)

print("\nSparsity stats (vessel voxels as % of volume):")
print("  N cases:", len(fractions))
print("  Min %:", np.min(fractions) * 100)
print("  Median %:", np.median(fractions) * 100)
print("  Mean %:", np.mean(fractions) * 100)
print("  Max %:", np.max(fractions) * 100)

# -----------------------------
# 4) Plot A: Histogram (distribution across dataset)
# -----------------------------
plt.figure(figsize=(7, 4))
plt.hist(fractions * 100, bins=30)
plt.xlabel("Vessel voxels (% of volume)")
plt.ylabel("Number of cases")
plt.title(f"Dataset sparsity (N={len(fractions)} cases)")
plt.tight_layout()
plt.show()

# -----------------------------
# 5) Plot B: Per-case scatter (nice “wow it’s tiny” visualization)
# -----------------------------
order = np.argsort(case_ids)  # sorts by case id if available
plt.figure(figsize=(8, 4))
plt.scatter(case_ids[order], fractions[order] * 100, s=10)
plt.xlabel("Case ID")
plt.ylabel("Vessel voxels (% of volume)")
plt.title("Per-case sparsity (lower = more imbalance)")
plt.tight_layout()
plt.show()

# -----------------------------
# 6) Plot C: CDF (cumulative curve: “X% of cases have < Y% vessels”)
# -----------------------------
sorted_frac = np.sort(fractions * 100)
cdf = np.arange(1, len(sorted_frac) + 1) / len(sorted_frac)

plt.figure(figsize=(7, 4))
plt.plot(sorted_frac, cdf)
plt.xlabel("Vessel voxels (% of volume)")
plt.ylabel("Fraction of cases ≤ x")
plt.title("Cumulative sparsity distribution (CDF)")
plt.tight_layout()
plt.show()
