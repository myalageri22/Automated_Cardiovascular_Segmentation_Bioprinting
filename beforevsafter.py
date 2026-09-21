#!/usr/bin/env python3
"""beforevsafter.py

End-to-end *visual* pipeline demo for your poster:
CT (raw HU) -> window/normalize -> resample -> (optional) model prob_map -> threshold -> postprocess -> mesh -> STL.

Outputs (by default under ./pipeline_outputs/<case_id>/):
- pipeline_panels.png  (multi-panel figure for poster)
- pipeline_map.png     (simple pipeline flow diagram)
- binary_mask_post.nii.gz
- mesh_raw.stl
- mesh_smoothed.stl (if trimesh available)

Notes:
- Uses the same default paths you already used (801.*).
- If you have a model probability map NIfTI (e.g., saved from validation), pass --pred_prob.
  Otherwise, it will use the GT label as a stand-in for visualization so you can still demo the pipeline.

Install helpers (recommended):
  pip install nibabel numpy matplotlib scipy scikit-image trimesh

Run examples:
  python beforevsafter.py
  python beforevsafter.py --case_id 801 \
    --ct "data/processed/all/801-1000/801.img.nii.gz" \
    --label "data/processed/all/801-1000/801.label.nii.gz" \
    --pred_prob "/path/to/801_prob.nii.gz" \
    --out_dir "pipeline_outputs"

"""

import argparse
import os
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# -----------------------------
# Defaults: match your repo
# -----------------------------
DEFAULT_CASE_ID = "801"
DEFAULT_CT_PATH = "data/processed/all/801-1000/801.img.nii.gz"
DEFAULT_LBL_PATH = "data/processed/all/801-1000/801.label.nii.gz"


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def hu_window_to_unit(x: np.ndarray, a_min: float = -200.0, a_max: float = 700.0) -> np.ndarray:
    """Match MONAI ScaleIntensityRanged for CT: clip then scale to [0,1]."""
    x = x.astype(np.float32)
    x = np.clip(x, a_min, a_max)
    return (x - a_min) / (a_max - a_min)


def resample_to_spacing(vol: np.ndarray, in_sp: tuple[float, float, float], out_sp: tuple[float, float, float], order: int) -> np.ndarray:
    """Resample a 3D volume using scipy.ndimage.zoom.

    zoom_factors = in_spacing / out_spacing per axis.
    - order=1 for images (linear)
    - order=0 for labels (nearest)
    """
    try:
        from scipy.ndimage import zoom
    except Exception as e:
        raise RuntimeError(
            "scipy is required for resampling. Install with: pip install scipy"
        ) from e

    zoom_factors = (in_sp[0] / out_sp[0], in_sp[1] / out_sp[1], in_sp[2] / out_sp[2])
    # scipy zoom expects (x,y,z) factors aligned with array axes; nibabel data is typically (X,Y,Z)
    return zoom(vol, zoom=zoom_factors, order=order)


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than min_size voxels."""
    if min_size <= 0:
        return mask

    try:
        from scipy.ndimage import label
    except Exception as e:
        raise RuntimeError(
            "scipy is required for connected-component cleanup. Install with: pip install scipy"
        ) from e

    lab, n = label(mask.astype(np.uint8))
    if n == 0:
        return mask

    # Count voxels per component id
    counts = np.bincount(lab.ravel())
    # Keep background (0) off
    keep_ids = np.where(counts >= min_size)[0]
    keep_ids = keep_ids[keep_ids != 0]

    cleaned = np.isin(lab, keep_ids)
    return cleaned.astype(np.uint8)


def postprocess_mask(mask: np.ndarray, min_cc_size: int, closing_iters: int, gaussian_sigma: float) -> np.ndarray:
    """Topology-friendly postprocessing for vessel masks.

    Steps:
    1) remove small connected components
    2) 3D binary closing (optional)
    3) fill holes (optional)
    4) gaussian smoothing + re-threshold (optional)
    """
    from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter

    m = (mask > 0).astype(np.uint8)

    # 1) remove tiny specks
    m = remove_small_components(m, min_cc_size)

    # 2) closing (bridges tiny gaps)
    if closing_iters > 0:
        # Use a small 3x3x3 structure
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        for _ in range(closing_iters):
            m = binary_closing(m, structure=structure).astype(np.uint8)

    # 3) fill holes (careful: can overfill in vessels, but okay for demo)
    m = binary_fill_holes(m).astype(np.uint8)

    # 4) smooth boundary (soften stair-steps), then threshold back
    if gaussian_sigma > 0:
        m_f = gaussian_filter(m.astype(np.float32), sigma=gaussian_sigma)
        m = (m_f >= 0.5).astype(np.uint8)

    return m


def mask_to_mesh(mask: np.ndarray, spacing: tuple[float, float, float]):
    """Extract a surface mesh from a binary mask using marching cubes."""
    try:
        from skimage.measure import marching_cubes
    except Exception as e:
        raise RuntimeError(
            "scikit-image is required for marching cubes. Install with: pip install scikit-image"
        ) from e

    # marching_cubes expects values; use float with level=0.5
    verts, faces, normals, values = marching_cubes(mask.astype(np.float32), level=0.5, spacing=spacing)
    return verts, faces


def export_stl(verts: np.ndarray, faces: np.ndarray, out_path: Path, smooth: bool = True):
    """Export STL. If trimesh is available, do basic cleanup + optional Laplacian smoothing."""
    try:
        import trimesh
    except Exception:
        trimesh = None

    if trimesh is None:
        # Minimal ASCII STL writer (no smoothing). Works even without trimesh.
        # NOTE: This is for visualization; for production meshes, install trimesh.
        out_path = out_path.with_suffix(".stl")
        with open(out_path, "w") as f:
            f.write("solid mesh\n")
            for tri in faces:
                v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
                # compute normal
                n = np.cross(v1 - v0, v2 - v0)
                nn = np.linalg.norm(n) + 1e-12
                n = n / nn
                f.write(f" facet normal {n[0]} {n[1]} {n[2]}\n")
                f.write("  outer loop\n")
                f.write(f"   vertex {v0[0]} {v0[1]} {v0[2]}\n")
                f.write(f"   vertex {v1[0]} {v1[1]} {v1[2]}\n")
                f.write(f"   vertex {v2[0]} {v2[1]} {v2[2]}\n")
                f.write("  endloop\n")
                f.write(" endfacet\n")
            f.write("endsolid mesh\n")
        return {
            "engine": "ascii_stl_writer",
            "watertight": None,
            "n_verts": int(verts.shape[0]),
            "n_faces": int(faces.shape[0]),
            "out": str(out_path),
        }

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # Basic cleanup
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()

    raw_path = out_path.with_name(out_path.stem + "_raw.stl")
    mesh.export(raw_path)

    info = {
        "engine": "trimesh",
        "watertight_raw": bool(mesh.is_watertight),
        "n_verts_raw": int(len(mesh.vertices)),
        "n_faces_raw": int(len(mesh.faces)),
        "out_raw": str(raw_path),
    }

    if smooth:
        try:
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=10)
        except Exception:
            # smoothing is optional
            pass
        smooth_path = out_path.with_name(out_path.stem + "_smoothed.stl")
        mesh.export(smooth_path)
        info.update({
            "watertight_smoothed": bool(mesh.is_watertight),
            "n_verts_smoothed": int(len(mesh.vertices)),
            "n_faces_smoothed": int(len(mesh.faces)),
            "out_smoothed": str(smooth_path),
        })

    return info


def save_nifti_like(ref_img: nib.Nifti1Image, data: np.ndarray, out_path: Path):
    """Save a new NIfTI with the same affine as a reference image."""
    nii = nib.Nifti1Image(data.astype(np.uint8), ref_img.affine)
    nib.save(nii, str(out_path))


def make_pipeline_panels(
    case_id: str,
    raw_ct: np.ndarray,
    windowed_ct: np.ndarray,
    resampled_ct: np.ndarray,
    gt_lbl: np.ndarray,
    prob_map: np.ndarray,
    binary_mask: np.ndarray,
    post_mask: np.ndarray,
    out_path: Path,
    spacing_in: tuple[float, float, float],
    spacing_out: tuple[float, float, float],
    threshold: float,
):
    """Save a clean multi-panel figure for your poster."""

    # Choose a representative axial slice
    z_raw = raw_ct.shape[2] // 2
    z_res = resampled_ct.shape[2] // 2

    fig = plt.figure(figsize=(18, 10))

    # Panel 1: raw
    ax1 = plt.subplot2grid((2, 4), (0, 0))
    ax1.imshow(raw_ct[:, :, z_raw].T, cmap="gray", origin="lower")
    ax1.set_title(f"Raw CT (HU)\nspacing={tuple(np.round(spacing_in, 3))}")
    ax1.axis("off")

    # Panel 2: windowed
    ax2 = plt.subplot2grid((2, 4), (0, 1))
    ax2.imshow(windowed_ct[:, :, z_raw].T, cmap="gray", origin="lower")
    ax2.set_title("Windowed [-200,700] → [0,1]")
    ax2.axis("off")

    # Panel 3: resampled + windowed
    ax3 = plt.subplot2grid((2, 4), (0, 2))
    ax3.imshow(resampled_ct[:, :, z_res].T, cmap="gray", origin="lower")
    ax3.set_title(f"Resampled to {tuple(spacing_out)} mm")
    ax3.axis("off")

    # Panel 4: GT overlay (on resampled CT)
    ax4 = plt.subplot2grid((2, 4), (0, 3))
    ax4.imshow(resampled_ct[:, :, z_res].T, cmap="gray", origin="lower")
    # resample label to match resampled_ct if needed (assumes caller already aligned)
    ax4.imshow(gt_lbl[:, :, z_res].T, alpha=0.35, origin="lower")
    ax4.set_title("Ground Truth vessel mask (overlay)")
    ax4.axis("off")

    # Panel 5: probability map
    ax5 = plt.subplot2grid((2, 4), (1, 0))
    ax5.imshow(prob_map[:, :, z_res].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax5.set_title("Model probability map (or GT stand-in)")
    ax5.axis("off")

    # Panel 6: thresholded binary
    ax6 = plt.subplot2grid((2, 4), (1, 1))
    ax6.imshow(binary_mask[:, :, z_res].T, cmap="gray", origin="lower")
    ax6.set_title(f"Thresholded mask\n(prob > {threshold:.2f})")
    ax6.axis("off")

    # Panel 7: postprocessed
    ax7 = plt.subplot2grid((2, 4), (1, 2))
    ax7.imshow(post_mask[:, :, z_res].T, cmap="gray", origin="lower")
    ax7.set_title("Postprocessed mask\n(CC cleanup + closing + smooth)")
    ax7.axis("off")

    # Panel 8: text summary (mesh / print readiness)
    ax8 = plt.subplot2grid((2, 4), (1, 3))
    ax8.axis("off")

    # Key stats
    pos_frac = float(post_mask.sum() / np.prod(post_mask.shape))
    ax8.text(0.0, 0.95, f"Case {case_id} summary", fontsize=14, fontweight="bold")
    ax8.text(0.0, 0.82, f"Vessel voxels (post): {int(post_mask.sum()):,}")
    ax8.text(0.0, 0.74, f"Positive fraction: {pos_frac*100:.4f}%")
    ax8.text(0.0, 0.66, "Next: marching cubes → STL → slicer", fontsize=11)
    ax8.text(0.0, 0.52, "Poster tip:", fontsize=12, fontweight="bold")
    ax8.text(0.0, 0.44, "Show this figure + exported STL in slicer preview.")

    plt.suptitle("End-to-End CT → Bioprintable Vessel STL Pipeline", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_pipeline_map(case_id: str, out_path: Path):
    """Simple pipeline flow diagram image for the poster."""
    fig = plt.figure(figsize=(16, 3.5))
    ax = plt.gca()
    ax.axis("off")

    steps = [
        "Raw CT (NIfTI)",
        "Window +\nNormalize",
        "Resample\n(0.6mm)",
        "3D U-Net\n(prob map)",
        "Threshold", 
        "Postprocess\n(CC/closing/smooth)",
        "Marching Cubes\n(mesh)",
        "STL Export\n(watertight/slice)",
        "3D Bioprinting\nworkflow",
    ]

    n = len(steps)
    xs = np.linspace(0.05, 0.95, n)
    y = 0.5

    for i, (x, s) in enumerate(zip(xs, steps)):
        ax.text(
            x,
            y,
            s,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="#F3F3F3", ec="#333333"),
        )
        if i < n - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.03, y),
                xytext=(x + 0.03, y),
                arrowprops=dict(arrowstyle="->", lw=2),
            )

    ax.text(0.5, 0.92, f"Pipeline Map (Example Case: {case_id})", ha="center", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="End-to-end pipeline visualizer for poster")
    p.add_argument("--case_id", default=DEFAULT_CASE_ID)
    p.add_argument("--ct", default=DEFAULT_CT_PATH, help="Path to CT NIfTI")
    p.add_argument("--label", default=DEFAULT_LBL_PATH, help="Path to label NIfTI")
    p.add_argument("--pred_prob", default=None, help="Optional: model prob_map NIfTI for this case")
    p.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "pipeline_outputs"))

    # Match your training config defaults
    p.add_argument("--ct_window", default="-200,700")
    p.add_argument("--target_spacing", default="0.6,0.6,0.6")
    p.add_argument("--threshold", type=float, default=0.10, help="Mask threshold for printable binary")

    # Postprocess knobs (keep conservative for vessels)
    p.add_argument("--min_cc_size", type=int, default=50)
    p.add_argument("--closing_iters", type=int, default=1)
    p.add_argument("--gaussian_sigma", type=float, default=0.8)

    # Mesh export
    p.add_argument("--no_mesh_smooth", action="store_true", help="Disable trimesh smoothing")

    args = p.parse_args()

    ct_window = tuple(map(float, args.ct_window.split(",")))
    target_spacing = tuple(map(float, args.target_spacing.split(",")))

    out_root = Path(args.out_dir)
    out_case = out_root / str(args.case_id)
    _safe_mkdir(out_case)

    print("=" * 80)
    print("PIPELINE DEMO")
    print("=" * 80)
    print("CT:    ", args.ct)
    print("Label: ", args.label)
    if args.pred_prob:
        print("Prob:  ", args.pred_prob)
    print("Out:   ", out_case)

    # ---- Load ----
    ct_img = nib.load(args.ct)
    lbl_img = nib.load(args.label)

    ct = ct_img.get_fdata().astype(np.float32)
    lbl = (lbl_img.get_fdata() > 0).astype(np.uint8)

    spacing_in = ct_img.header.get_zooms()[:3]
    print("Original spacing (pixdim):", spacing_in)

    # ---- Window/normalize ----
    ct_win = hu_window_to_unit(ct, a_min=ct_window[0], a_max=ct_window[1])

    # ---- Resample (image: linear, label: nearest) ----
    ct_res = resample_to_spacing(ct_win, spacing_in, target_spacing, order=1)
    lbl_res = resample_to_spacing(lbl, spacing_in, target_spacing, order=0)
    lbl_res = (lbl_res > 0).astype(np.uint8)

    # ---- Probability map (optional) ----
    if args.pred_prob and os.path.exists(args.pred_prob):
        prob_img = nib.load(args.pred_prob)
        prob = prob_img.get_fdata().astype(np.float32)
        # If prob map spacing differs, resample it too
        prob_sp = prob_img.header.get_zooms()[:3]
        if tuple(np.round(prob_sp, 6)) != tuple(np.round(target_spacing, 6)) or prob.shape != ct_res.shape:
            prob = resample_to_spacing(prob, prob_sp, target_spacing, order=1)
        # clamp to [0,1]
        prob = np.clip(prob, 0.0, 1.0)
        prob_source = "model"
    else:
        # Stand-in for visualization: use GT as a pseudo-probability
        prob = lbl_res.astype(np.float32)
        prob_source = "gt_standin"

    print(f"Probability map source: {prob_source}")

    # ---- Threshold ----
    binary = (prob > float(args.threshold)).astype(np.uint8)

    # ---- Postprocess ----
    try:
        post = postprocess_mask(
            binary,
            min_cc_size=int(args.min_cc_size),
            closing_iters=int(args.closing_iters),
            gaussian_sigma=float(args.gaussian_sigma),
        )
    except Exception as e:
        print("[WARN] Postprocessing failed, using raw binary mask. Error:", e)
        post = binary

    # Save post mask NIfTI in the *original* CT affine (for poster/demo only)
    # NOTE: Because we resampled, this is mainly for visualization. For strict spatial correctness,
    # you'd store an updated affine. For poster purposes, this is fine as a pipeline illustration.
    mask_out = out_case / f"{args.case_id}_binary_mask_post.nii.gz"
    save_nifti_like(ct_img, post, mask_out)
    print("Saved postprocessed mask:", mask_out)

    # ---- Mesh + STL ----
    try:
        verts, faces = mask_to_mesh(post, spacing=target_spacing)
        print(f"Mesh extracted: verts={len(verts):,} faces={len(faces):,}")

        stl_base = out_case / f"{args.case_id}_mesh"
        info = export_stl(verts, faces, stl_base, smooth=(not args.no_mesh_smooth))
        print("STL export:")
        for k, v in info.items():
            print(f"  - {k}: {v}")
    except Exception as e:
        print("[WARN] Mesh/STL step failed. Install scikit-image + trimesh. Error:", e)

    # ---- Figures ----
    panels_path = out_case / "pipeline_panels.png"
    map_path = out_case / "pipeline_map.png"

    make_pipeline_panels(
        case_id=str(args.case_id),
        raw_ct=ct,
        windowed_ct=ct_win,
        resampled_ct=ct_res,
        gt_lbl=lbl_res,
        prob_map=prob,
        binary_mask=binary,
        post_mask=post,
        out_path=panels_path,
        spacing_in=spacing_in,
        spacing_out=target_spacing,
        threshold=float(args.threshold),
    )

    make_pipeline_map(str(args.case_id), map_path)

    print("Saved figure:", panels_path)
    print("Saved map:", map_path)
    print("Done ✅")


if __name__ == "__main__":
    main()
