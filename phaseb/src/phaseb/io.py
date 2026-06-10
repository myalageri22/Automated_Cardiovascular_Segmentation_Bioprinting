"""IO utilities for Phase B.

Handles loading CT volumes and segmentations with SimpleITK, segmentation type detection,
manifest parsing, and output utilities.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml


class SegmentationType:
    PROBABILITY = "prob"
    MASK = "mask"
    AUTO = "auto"


@dataclass
class LoadedData:
    image: sitk.Image
    array: np.ndarray


def load_image(path: str) -> sitk.Image:
    """Load an image from a file or DICOM directory using SimpleITK."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")

    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(path)
        if not series_ids:
            raise ValueError(f"No DICOM series found in {path}")
        file_names = reader.GetGDCMSeriesFileNames(path, series_ids[0])
        reader.SetFileNames(file_names)
        return reader.Execute()

    if os.path.isfile(path) and os.path.getsize(path) == 0:
        raise ValueError(f"Input file is empty: {path}")

    return sitk.ReadImage(path)


def image_to_array(img: sitk.Image) -> np.ndarray:
    """Convert SimpleITK image to numpy array with z, y, x ordering."""
    return sitk.GetArrayFromImage(img)


def load_volume(path: str) -> LoadedData:
    image = load_image(path)
    return LoadedData(image=image, array=image_to_array(image))


def detect_segmentation_type(array: np.ndarray) -> str:
    """Infer whether the segmentation is probability map or binary mask."""
    if np.issubdtype(array.dtype, np.floating):
        max_val = float(np.nanmax(array)) if array.size else 0.0
        min_val = float(np.nanmin(array)) if array.size else 0.0
        if 0.0 <= min_val <= 1.0 and max_val <= 1.0 + 1e-3:
            return SegmentationType.PROBABILITY
    # default
    return SegmentationType.MASK


def ensure_outdir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_used_config(config: Dict, outdir: str) -> None:
    path = Path(outdir) / "used_config.yaml"
    with path.open("w") as f:
        yaml.safe_dump(config, f)


def setup_logging(outdir: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("phaseb")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logger.level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(Path(outdir) / "phaseb.log")
    fh.setLevel(logger.level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


def load_manifest(path: str) -> List[Dict[str, str]]:
    """Load a batch manifest from CSV or JSON."""
    ext = Path(path).suffix.lower()
    records: List[Dict[str, str]]
    if ext in {".csv"}:
        df = pd.read_csv(path)
        records = df.to_dict(orient="records")
    elif ext in {".json"}:
        with open(path, "r") as f:
            data = json.load(f)
            records = data if isinstance(data, list) else data.get("cases", [])
    else:
        raise ValueError("Unsupported manifest format; use CSV or JSON")
    for rec in records:
        if "case_id" not in rec:
            rec["case_id"] = Path(rec.get("ct_path", "case")).stem
    return records


def serialize_config(config: Dict) -> Dict:
    """Prepare config dict for logging/JSON serialization."""
    serializable = {}
    for key, val in config.items():
        if isinstance(val, (list, dict, str, int, float, bool)) or val is None:
            serializable[key] = val
        else:
            serializable[key] = str(val)
    return serializable
