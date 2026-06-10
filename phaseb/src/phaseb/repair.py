"""Mesh repair helpers."""
from __future__ import annotations

from typing import Dict, Tuple

import trimesh

try:
    import pymeshfix  # type: ignore
except Exception:  # pragma: no cover
    pymeshfix = None


def _call_mesh_method(mesh: trimesh.Trimesh, method: str) -> None:
    func = getattr(mesh, method, None)
    if callable(func):
        try:
            func()
        except Exception:
            pass


def repair_mesh(mesh: trimesh.Trimesh, use_pymeshfix: bool = True) -> Tuple[trimesh.Trimesh, Dict]:
    info: Dict = {"used_pymeshfix": False, "messages": []}
    repaired = mesh.copy()

    # Basic cleanup with compatibility across trimesh versions
    _call_mesh_method(repaired, "remove_duplicate_faces")
    try:
        trimesh.repair.remove_duplicate_faces(repaired)
    except Exception:
        pass

    _call_mesh_method(repaired, "remove_degenerate_faces")
    try:
        trimesh.repair.remove_degenerate_faces(repaired)
    except Exception:
        pass

    _call_mesh_method(repaired, "remove_unreferenced_vertices")
    _call_mesh_method(repaired, "remove_infinite_values")
    _call_mesh_method(repaired, "rezero")
    _call_mesh_method(repaired, "merge_vertices")

    if use_pymeshfix and pymeshfix is not None:
        try:
            fixer = pymeshfix.MeshFix(repaired.vertices, repaired.faces)
            fixer.repair(verbose=False, joincomp=True, remove_smallest_components=False)
            repaired = trimesh.Trimesh(fixer.v, fixer.f, process=False)
            info["used_pymeshfix"] = True
        except Exception as exc:  # pragma: no cover - optional path
            info["messages"].append(f"pymeshfix repair failed: {exc}")

    _call_mesh_method(repaired, "fix_normals")
    _call_mesh_method(repaired, "remove_unreferenced_vertices")
    _call_mesh_method(repaired, "merge_vertices")
    info["is_watertight"] = bool(repaired.is_watertight)
    info["components"] = len(repaired.split(only_watertight=False))
    info["euler_number"] = repaired.euler_number if repaired.faces.size > 0 else None
    return repaired, info
