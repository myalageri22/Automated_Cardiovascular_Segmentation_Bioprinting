"""Report generation utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import trimesh


def _normalize_ct(ct_array: np.ndarray) -> np.ndarray:
    vmin, vmax = np.percentile(ct_array, [0.5, 99.5])
    vmax = vmax if vmax > vmin else vmin + 1.0
    normed = np.clip((ct_array - vmin) / (vmax - vmin), 0, 1)
    return normed


def generate_slice_overlays(ct_image: sitk.Image, mask: np.ndarray, outdir: str, case_id: str) -> None:
    ct_array = sitk.GetArrayFromImage(ct_image)
    normed = _normalize_ct(ct_array.astype(np.float32))
    if mask.shape != normed.shape:
        common_shape = tuple(min(m, n) for m, n in zip(mask.shape, normed.shape))
        mask = mask[: common_shape[0], : common_shape[1], : common_shape[2]]
        normed = normed[: common_shape[0], : common_shape[1], : common_shape[2]]
    indices = [ax // 2 for ax in mask.shape]
    preview_dir = Path(outdir) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    slices = {
        "axial": (mask[indices[0]], normed[indices[0]]),
        "coronal": (mask[:, indices[1], :], normed[:, indices[1], :]),
        "sagittal": (mask[:, :, indices[2]], normed[:, :, indices[2]]),
    }

    for name, (mask_slice, ct_slice) in slices.items():
        plt.figure(figsize=(4, 4))
        plt.imshow(ct_slice, cmap="gray")
        plt.imshow(np.ma.masked_where(mask_slice == 0, mask_slice), cmap="Reds", alpha=0.4)
        plt.axis("off")
        plt.title(f"{case_id} {name}")
        plt.tight_layout()
        plt.savefig(preview_dir / f"{case_id}_{name}.png", dpi=200)
        plt.close()


def generate_mesh_preview(mesh: trimesh.Trimesh, outdir: str, case_id: str) -> None:
    preview_dir = Path(outdir) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    try:
        scene = mesh.scene()
        png = scene.save_image(resolution=(600, 600), visible=True)
        if png is not None:
            with open(preview_dir / f"{case_id}_mesh.png", "wb") as f:
                f.write(png)
            return
    except Exception:
        pass
    # Fallback to matplotlib rendering
    try:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            color="lightblue",
            linewidth=0.1,
            alpha=0.8,
        )
        ax.view_init(elev=20, azim=30)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(preview_dir / f"{case_id}_mesh.png", dpi=200)
        plt.close()
    except Exception:
        pass


def write_json(report: Dict, outdir: str) -> None:
    path = Path(outdir) / "qc_report.json"
    with path.open("w") as f:
        json.dump(report, f, indent=2)


def write_text_summary(report: Dict, outdir: str) -> None:
    lines = []
    lines.append(f"Case: {report.get('case_id')}")
    lines.append(f"Seg type: {report['segmentation']['type']} threshold={report['segmentation'].get('threshold')}")
    mesh_metrics = report.get("mesh_repaired", {})
    mask_metrics = report.get("mask_metrics", {})
    lines.append(f"Mask volume (mm^3): {mask_metrics.get('volume_mm3'):.2f}")
    lines.append(f"Components after cleanup: {report.get('topology_metrics', {}).get('component_count_after_cleanup')}")
    lines.append(f"Mesh surface area (mm^2): {mesh_metrics.get('surface_area_mm2')}")
    lines.append(f"Mesh watertight: {mesh_metrics.get('is_watertight')}")
    pr = report.get("printability", {})
    lines.append(
        "Radii (min/p5/p50 mm): "
        f"{pr.get('min_radius_mm')} / {pr.get('p5_radius_mm')} / {pr.get('p50_radius_mm')}"
    )

    pf, reasons = _pass_fail(report)
    lines.append(f"Pass: {pf}")
    if reasons:
        lines.append("Reasons: " + "; ".join(reasons))

    path = Path(outdir) / "qc_summary.txt"
    with path.open("w") as f:
        f.write("\n".join(lines))


def write_csv_summary(report: Dict, outdir: str) -> None:
    path = Path(outdir) / "qc_summary.csv"
    header_needed = not path.exists()
    cols = [
        "case_id",
        "threshold",
        "volume_mm3",
        "component_count",
        "mesh_surface_area_mm2",
        "mesh_watertight",
        "min_radius_mm",
        "percent_below_radius",
        "pass",
    ]
    pf, _ = _pass_fail(report)
    row = {
        "case_id": report.get("case_id"),
        "threshold": report.get("segmentation", {}).get("threshold"),
        "volume_mm3": report.get("mask_metrics", {}).get("volume_mm3"),
        "component_count": report.get("mask_metrics", {}).get("component_count"),
        "mesh_surface_area_mm2": report.get("mesh_repaired", {}).get("surface_area_mm2"),
        "mesh_watertight": report.get("mesh_repaired", {}).get("is_watertight"),
        "min_radius_mm": report.get("printability", {}).get("min_radius_mm"),
        "percent_below_radius": report.get("printability", {}).get("percent_below_threshold"),
        "pass": pf,
    }
    with path.open("a") as f:
        if header_needed:
            f.write(",".join(cols) + "\n")
        f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")


def _pass_fail(report: Dict) -> Tuple[bool, list]:
    reasons = []
    mask_metrics = report.get("mask_metrics", {})
    printability = report.get("printability", {})
    if mask_metrics.get("voxel_count", 0) == 0:
        reasons.append("empty mask")
    if printability.get("percent_below_threshold") is not None and printability.get("percent_below_threshold") > 30:
        reasons.append("large fraction below min radius")
    if report.get("mesh_repaired", {}).get("component_count", 0) > 5:
        reasons.append("many mesh components")
    return len(reasons) == 0, reasons


def generate_previews(
    ct_image: sitk.Image,
    mask: np.ndarray,
    mesh: Optional[trimesh.Trimesh],
    outdir: str,
    case_id: str,
    enable_mesh_preview: bool = True,
) -> None:
    generate_slice_overlays(ct_image, mask, outdir, case_id)
    if mesh is not None and enable_mesh_preview:
        generate_mesh_preview(mesh, outdir, case_id)
