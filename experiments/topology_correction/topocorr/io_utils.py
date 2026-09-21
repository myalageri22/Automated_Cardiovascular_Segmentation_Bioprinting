"""IO, configuration and repository-location helpers."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

LOGGER = logging.getLogger("topocorr")

# Authoritative artifacts that this experiment must never write to.
PROTECTED_PATHS: Tuple[str, ...] = (
    "outputs/final_test_250",
    "outputs/phase_b_mesh_qc",
    "outputs/cmig_robustness",
    "outputs/fabrication_readiness",
    "outputs/_superseded",
    "analysis/cmig_robustness",
    "paper",
    "checkpoints",
    "extra_information",
)


def repo_root(start: Optional[Path] = None) -> Path:
    """Locate the repository root by walking up for a known marker."""
    here = Path(start or __file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "phaseb_mesh_qc.py").exists() and (parent / "outputs").exists():
            return parent
    raise RuntimeError("Could not locate repository root (no phaseb_mesh_qc.py found)")


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping")
    return cfg


def resolve(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (root / p)


def assert_not_protected(root: Path, target: Path) -> None:
    """Refuse to write anywhere near an authoritative artifact."""
    target = target.resolve()
    for prot in PROTECTED_PATHS:
        prot_path = (root / prot).resolve()
        try:
            target.relative_to(prot_path)
        except ValueError:
            continue
        raise PermissionError(
            f"Refusing to write inside protected authoritative path: {prot_path} (target {target})"
        )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def load_split_entries(splits_json: str | Path, key: str = "test") -> List[Dict[str, Any]]:
    """Return the split entries as dicts with at least an ``id``.

    ``dataset_splits.json`` stores each case as
    ``{"id": "1000", "image": "Data/all/1000.img.nii.gz", "label": ...}``.
    Older layouts stored bare id strings, so both forms are accepted.
    """
    with open(splits_json, "r") as fh:
        splits = json.load(fh)
    cases = splits.get(key)
    if cases is None:
        raise KeyError(f"Split key '{key}' not present in {splits_json}; keys={list(splits)}")
    out: List[Dict[str, Any]] = []
    for c in cases:
        if isinstance(c, dict):
            if "id" not in c:
                raise ValueError(f"Split entry without an 'id': {c}")
            out.append({"id": str(c["id"]), "image": c.get("image"), "label": c.get("label")})
        else:
            out.append({"id": str(c), "image": None, "label": None})
    return out


def load_test_cases(splits_json: str | Path, key: str = "test") -> List[str]:
    """Case identifiers of a split, in the split's own order."""
    return [e["id"] for e in load_split_entries(splits_json, key)]


@dataclass
class Volume:
    """A binary volume plus the geometry needed to stay in world coordinates."""

    array: np.ndarray          # bool, nibabel axis order
    affine: np.ndarray         # 4x4
    spacing: Tuple[float, float, float]
    header: Any = None
    path: Optional[str] = None

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))

    def voxel_to_world(self, idx_xyz: np.ndarray) -> np.ndarray:
        import nibabel as nib

        return nib.affines.apply_affine(self.affine, np.asarray(idx_xyz, dtype=float))


def load_binary_volume(path: str | Path, threshold: Optional[float] = None) -> Volume:
    """Load a NIfTI as a boolean volume.

    ``threshold`` is applied only when the file holds a probability map. It is
    always supplied from the frozen configuration and never tuned here.
    """
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asarray(img.get_fdata())
    if threshold is None:
        arr = data > 0
    else:
        arr = data >= float(threshold)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return Volume(array=arr, affine=np.asarray(img.affine, dtype=float), spacing=spacing,
                  header=img.header, path=str(path))


def save_binary_volume(vol: Volume, mask: np.ndarray, path: str | Path) -> Path:
    """Write a corrected mask preserving the source affine and header geometry."""
    import nibabel as nib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), vol.affine)
    if vol.header is not None:
        out.header.set_zooms(tuple(vol.spacing))
        try:
            out.header.set_qform(vol.affine, code=int(vol.header["qform_code"]))
            out.header.set_sform(vol.affine, code=int(vol.header["sform_code"]))
        except Exception:
            pass
    nib.save(out, str(path))
    return path


def resample_gt_to(pred_path: str | Path, gt_path: str | Path) -> np.ndarray:
    """Resample a ground-truth label onto the prediction grid, world-aware, nearest.

    Mirrors ``compute_cldice.gt_on_pred_grid`` so ground truth handling is
    identical to the authoritative clDice computation.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to

    pred_img = nib.load(str(pred_path))
    gt_img = nib.load(str(gt_path))
    if gt_img.shape[:3] == pred_img.shape[:3] and np.allclose(gt_img.affine, pred_img.affine, atol=1e-4):
        return np.asarray(gt_img.dataobj) > 0
    matched = resample_from_to(gt_img, (pred_img.shape[:3], pred_img.affine), order=0)
    return np.asarray(matched.dataobj) > 0


def setup_logging(log_dir: str | Path, name: str = "topocorr", level: str = "INFO") -> logging.Logger:
    log_dir = ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> Path:
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=default))
    return path
