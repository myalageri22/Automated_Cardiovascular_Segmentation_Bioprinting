"""Representative-case selection and rendering.

Cases are selected by PRESPECIFIED rules covering successes and failures alike.
The rules are evaluated on the cohort table; no case is hand-picked. Categories
deliberately include cases where correction achieved nothing and cases where it
cost the most fidelity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAX_RENDER_FACES = 8000

SELECTION_RULES: Dict[str, str] = {
    "high_dice_high_fragmentation":
        "highest original component count among cases with original Dice above the cohort median",
    "moderate_dice_good_topology":
        "original components <= 2 and original Dice closest to the cohort median",
    "largest_component_reduction":
        "largest reduction in component count (original - corrected)",
    "no_improvement":
        "zero change in component count, tie-broken by highest original component count",
    "largest_fidelity_loss":
        "largest decrease in Dice (corrected - original), i.e. the worst anatomical cost",
}


def select_cases(paired: pd.DataFrame, strategy: str, n_per_category: int = 3) -> pd.DataFrame:
    df = paired[paired["strategy"] == strategy].copy()
    for col in ("dice_original", "dice_corrected", "components_original", "components_corrected"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["delta_components"] = df["components_corrected"] - df["components_original"]
    df["delta_dice"] = df["dice_corrected"] - df["dice_original"]

    picks: List[pd.DataFrame] = []
    dice_median = df["dice_original"].median()

    hi = df[df["dice_original"] > dice_median].nlargest(n_per_category, "components_original")
    picks.append(hi.assign(category="high_dice_high_fragmentation"))

    good = df[df["components_original"] <= 2].copy()
    if len(good):
        good["dist"] = (good["dice_original"] - dice_median).abs()
        picks.append(good.nsmallest(n_per_category, "dist").drop(columns=["dist"])
                     .assign(category="moderate_dice_good_topology"))

    picks.append(df.nsmallest(n_per_category, "delta_components")
                 .assign(category="largest_component_reduction"))

    flat = df[df["delta_components"] == 0]
    picks.append(flat.nlargest(n_per_category, "components_original")
                 .assign(category="no_improvement"))

    picks.append(df.nsmallest(n_per_category, "delta_dice")
                 .assign(category="largest_fidelity_loss"))

    out = pd.concat([p for p in picks if len(p)], ignore_index=True)
    out["selection_rule"] = out["category"].map(SELECTION_RULES)
    out["strategy"] = strategy
    return out


def _slice_indices(mask: np.ndarray, n: int) -> List[int]:
    """Axial slices through the densest part of the mask, deterministic."""
    if not mask.any():
        return [mask.shape[2] // 2] * n
    per_slice = mask.sum(axis=(0, 1))
    order = np.argsort(-per_slice)
    chosen = sorted(order[:max(n * 4, n)][:n].tolist())
    return chosen if chosen else [mask.shape[2] // 2] * n


def render_case(
    case_id: str,
    ct_path: Optional[str],
    gt: Optional[np.ndarray],
    original: np.ndarray,
    corrected: np.ndarray,
    out_dir: Path,
    original_mesh: Optional[str] = None,
    corrected_mesh: Optional[str] = None,
    n_slices: int = 3,
    dpi: int = 200,
    title_suffix: str = "",
    stem: Optional[str] = None,
) -> Dict[str, Any]:
    """Render CTA / GT / original / corrected slices plus both reconstructions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ct = None
    if ct_path and Path(ct_path).exists():
        try:
            import nibabel as nib
            ct = np.asarray(nib.load(str(ct_path)).get_fdata())
        except Exception:
            ct = None

    idxs = _slice_indices(np.asarray(original, dtype=bool) | (
        np.asarray(gt, dtype=bool) if gt is not None else np.zeros_like(original, dtype=bool)), n_slices)

    panels = [("CTA", ct), ("Ground truth", gt), ("Original prediction", original),
              ("Corrected prediction", corrected)]
    fig, axes = plt.subplots(len(idxs), 4, figsize=(13, 3.3 * len(idxs)), squeeze=False)
    for r, z in enumerate(idxs):
        for c, (label, vol) in enumerate(panels):
            ax = axes[r][c]
            if vol is None:
                ax.text(0.5, 0.5, f"{label}\nunavailable", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#C92A2A")
            else:
                sl = np.asarray(vol)[:, :, min(z, vol.shape[2] - 1)]
                if label == "CTA":
                    ax.imshow(sl.T, cmap="gray", origin="lower")
                else:
                    ax.imshow(sl.T, cmap="magma", origin="lower", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(label, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"z = {z}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Case {case_id} {title_suffix}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    stem = stem or str(case_id)
    slice_path = out_dir / f"{stem}_slices.png"
    fig.savefig(slice_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    mesh_path = None
    if original_mesh or corrected_mesh:
        mesh_path = _render_mesh_pair(case_id, original_mesh, corrected_mesh, out_dir, dpi, stem)

    return {"case_id": case_id, "slices_png": str(slice_path),
            "meshes_png": str(mesh_path) if mesh_path else None}


def _render_mesh_pair(case_id: str, a: Optional[str], b: Optional[str],
                      out_dir: Path, dpi: int, stem: Optional[str] = None) -> Optional[Path]:
    try:
        import trimesh
    except Exception:
        return None
    fig = plt.figure(figsize=(11, 5.2))
    for i, (label, path) in enumerate((("Original mesh", a), ("Corrected mesh", b))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        if not path or not Path(str(path)).exists():
            ax.text2D(0.5, 0.5, f"{label}\nunavailable", ha="center", transform=ax.transAxes,
                      fontsize=9, color="#C92A2A")
            ax.axis("off")
            continue
        try:
            mesh = trimesh.load(str(path), process=False)
            v = np.asarray(mesh.vertices, dtype=float)
            f = np.asarray(mesh.faces, dtype=int)
            # Matplotlib's trisurf cost grows steeply with face count and a full
            # coronary surface can carry hundreds of thousands of faces. Cap it
            # deterministically: rendering is illustrative, never a measurement.
            if len(f) > MAX_RENDER_FACES:
                f = f[:: int(np.ceil(len(f) / MAX_RENDER_FACES))]
            ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2], linewidth=0,
                            antialiased=False, color="#E8590C" if i else "#4C6EF5", alpha=0.9)
            ax.set_title(label, fontsize=10)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        except Exception as exc:
            ax.text2D(0.5, 0.5, f"{label}\nrender failed:\n{exc}", ha="center",
                      transform=ax.transAxes, fontsize=7, color="#C92A2A")
            ax.axis("off")
    fig.suptitle(f"Case {case_id} - reconstruction comparison", fontsize=11)
    path = out_dir / f"{stem or case_id}_meshes.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def remap_to_root(path: Optional[str], root: Path) -> Optional[str]:
    """Rebase a recorded absolute path onto the current repository root.

    Result CSVs record absolute paths from the machine that produced them. The
    same tree read from a different mount point would otherwise look like a
    missing file, which is how a mesh panel silently becomes "unavailable".
    """
    if not path:
        return None
    p = Path(str(path))
    if p.exists():
        return str(p)
    root = Path(root)
    marker = root.name + "/"
    sp = str(p)
    if marker in sp:
        cand = root / sp.split(marker, 1)[1]
        if cand.exists():
            return str(cand)
    return None
