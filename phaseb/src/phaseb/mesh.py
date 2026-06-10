"""Mesh extraction utilities."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import SimpleITK as sitk
import trimesh
from skimage import measure


def _index_to_physical_points(image: sitk.Image, vertices_zyx: np.ndarray) -> np.ndarray:
    """Convert marching cubes vertices (z, y, x) to physical mm using SimpleITK metadata."""
    spacing = np.asarray(image.GetSpacing(), dtype=float)  # (x, y, z)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    ijk = np.stack([vertices_zyx[:, 2], vertices_zyx[:, 1], vertices_zyx[:, 0]], axis=1)
    scaled = ijk * spacing
    physical = (direction @ scaled.T).T + origin
    return physical


def mask_to_mesh(mask: np.ndarray, reference_image: sitk.Image, step_size: int = 1) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    if mask.sum() == 0:
        raise ValueError("Segmentation mask is empty; cannot build mesh")
    verts, faces, normals, values = measure.marching_cubes(mask.astype(np.float32), level=0.5, step_size=step_size)
    physical_vertices = _index_to_physical_points(reference_image, verts)
    mesh = trimesh.Trimesh(vertices=physical_vertices, faces=faces.astype(np.int64), process=False)
    return mesh, physical_vertices, faces


def export_mesh(mesh: trimesh.Trimesh, path: str) -> None:
    mesh.export(path)
