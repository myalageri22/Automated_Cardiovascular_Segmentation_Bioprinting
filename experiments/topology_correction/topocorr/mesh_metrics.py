"""Phase B reconstruction wrapper.

The reconstruction itself is NOT reimplemented here. ``phaseb_mesh_qc.py`` from
the repository root is imported and ``run_phaseb_for_case`` is called unchanged,
so marching cubes, affine handling, repair rules, smoothing and every QC
definition are identical to the authoritative experiment. This module only:

  * routes outputs into the experiment directory;
  * additionally evaluates the already-implemented
    ``phaseb_mesh_qc.slicability_plane_check`` cross-sectional contour-closure
    metric on the repaired mesh;
  * computes mesh-to-mask centroid displacement in world coordinates.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

_PHASEB_QC = None
PHASEB_QC_SOURCE = "phaseb_mesh_qc.run_phaseb_for_case (repository implementation, unmodified)"


def load_phaseb_qc():
    """Import the repository's mesh-QC module by path, without copying it."""
    global _PHASEB_QC
    if _PHASEB_QC is not None:
        return _PHASEB_QC
    from .io_utils import repo_root

    path = repo_root() / "phaseb_mesh_qc.py"
    spec = importlib.util.spec_from_file_location("_repo_phaseb_mesh_qc", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_repo_phaseb_mesh_qc"] = mod
    spec.loader.exec_module(mod)
    _PHASEB_QC = mod
    return mod


def phaseb_source_sha256() -> str:
    from .io_utils import repo_root, sha256_file

    return sha256_file(repo_root() / "phaseb_mesh_qc.py")


def reconstruct(
    case_id: str,
    mask_path: str | Path,
    output_root: str | Path,
    target_wall_thickness_mm: float = 3.0,
    minimal: bool = False,
    slicability_planes: int = 50,
) -> Dict[str, Any]:
    """Run the authoritative Phase B reconstruction on one mask.

    Returns the QC row exactly as ``run_phaseb_for_case`` produced it, plus the
    additional contour-closure and centroid fields.
    """
    qc = load_phaseb_qc()
    row = qc.run_phaseb_for_case(
        case_id=str(case_id),
        mask_path=str(mask_path),
        output_root=str(output_root),
        target_wall_thickness_mm=float(target_wall_thickness_mm),
        minimal=bool(minimal),
    )
    row = dict(row)
    row["phaseb_source"] = PHASEB_QC_SOURCE

    repaired = row.get("repaired_stl")
    if row.get("status") == "ok" and repaired and Path(str(repaired)).exists():
        try:
            import trimesh

            mesh = trimesh.load(str(repaired), process=False)
            row.update(qc.slicability_plane_check(mesh, n_planes=int(slicability_planes)))
            row.update(_mesh_to_mask_centroid(mesh, mask_path))
        except Exception as exc:  # pragma: no cover
            row["slicability_error"] = repr(exc)
    return row


def _mesh_to_mask_centroid(mesh: Any, mask_path: str | Path) -> Dict[str, float]:
    """Displacement between the mesh centroid and the mask centroid, in world mm."""
    import nibabel as nib

    img = nib.load(str(mask_path))
    arr = np.asarray(img.get_fdata() > 0)
    out: Dict[str, float] = {}
    if not arr.any() or not len(getattr(mesh, "faces", [])):
        return {"mesh_to_mask_centroid_mm": float("nan")}
    idx = np.argwhere(arr).astype(float)
    mask_centroid_world = nib.affines.apply_affine(img.affine, idx.mean(axis=0))
    try:
        mesh_centroid = np.asarray(mesh.centroid, dtype=float)
    except Exception:
        mesh_centroid = np.asarray(mesh.vertices, dtype=float).mean(axis=0)
    delta = mesh_centroid - mask_centroid_world
    out["mask_centroid_world_x"] = float(mask_centroid_world[0])
    out["mask_centroid_world_y"] = float(mask_centroid_world[1])
    out["mask_centroid_world_z"] = float(mask_centroid_world[2])
    out["mesh_centroid_world_x"] = float(mesh_centroid[0])
    out["mesh_centroid_world_y"] = float(mesh_centroid[1])
    out["mesh_centroid_world_z"] = float(mesh_centroid[2])
    out["mesh_to_mask_centroid_mm"] = float(np.linalg.norm(delta))
    return out


def mesh_integrity_pass(row: Dict[str, Any]) -> Optional[bool]:
    """The existing mesh-integrity criterion, evaluated from a QC row.

    Definition preserved from the authoritative pipeline: a mesh passes when
    reconstruction succeeded, repair succeeded, the repaired surface is
    watertight and no non-manifold edges remain.
    """
    if row.get("status") != "ok":
        return False

    def _b(v: Any) -> bool:
        return str(v).strip().lower() in {"true", "1", "yes"}

    try:
        nm = int(float(row.get("non_manifold_edge_count", 1)))
    except Exception:
        return False
    return bool(_b(row.get("watertight")) and _b(row.get("repair_success")) and nm == 0)
