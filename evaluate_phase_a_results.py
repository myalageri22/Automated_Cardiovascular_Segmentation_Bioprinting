#!/usr/bin/env python3
"""Phase A paper-grade evaluation/results suite for cardiac vascular segmentation.

This script is designed for this repository's flat layout and supports:
- evaluation-only mode from existing predictions
- split parsing from `splits.json` and/or `imageCAS_data_split.xlsx`
- validation-only threshold selection, then locked test evaluation
- per-case metrics, summary statistics + bootstrap CIs, plots, qualitative outputs
- strict preflight sanity checks and verification artifacts for defensible reporting

Examples (repo-root):

1) Use existing predictions auto-discovered under `eval_outputs/`/`pipeline_outputs/`:
   python evaluate_phase_a_results.py \
     --splits_json splits.json \
     --split_xlsx imageCAS_data_split.xlsx

2) Explicit prediction + GT roots:
   python evaluate_phase_a_results.py \
     --pred_dir eval_outputs/my_preds \
     --gt_dir data/processed/all \
     --splits_json splits.json \
     --sweep_thresholds 0.05:0.95:0.05 \
     --primary_threshold_metric cldice \
     --run_assd

3) Prefer XLSX split selection (e.g., Split-1):
   python evaluate_phase_a_results.py \
     --pred_dir eval_outputs/my_preds \
     --split_xlsx imageCAS_data_split.xlsx \
     --id_key Split-1

4) Add optional Phase B correlation:
   python evaluate_phase_a_results.py \
     --pred_dir eval_outputs/my_preds \
     --splits_json splits.json \
     --phase_b_qc_csv phaseb_outputs/720/qc_summary.csv \
     --export_markdown_report
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import random
import re
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    _tmp_base = Path(tempfile.gettempdir())
except Exception:
    _tmp_base = Path.cwd()
os.environ.setdefault("MPLCONFIGDIR", str((_tmp_base / "mplconfig").resolve()))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

try:
    import nibabel as nib  # type: ignore
except Exception:
    nib = None

try:
    import SimpleITK as sitk  # type: ignore
except Exception:
    sitk = None

try:
    from scipy import ndimage as ndi  # type: ignore
except Exception:
    ndi = None

try:
    from skimage.morphology import skeletonize_3d  # type: ignore
except Exception:
    skeletonize_3d = None

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    import openpyxl  # type: ignore
except Exception:
    openpyxl = None

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    import monai  # type: ignore
except Exception:
    monai = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from sanity_check_eval_inputs import run_sanity_checks  # type: ignore
except Exception:
    run_sanity_checks = None  # type: ignore


ALLOWED_EXTS = (".nii.gz", ".nii", ".npy")
GT_NAME_HINTS = ("label", "mask", "seg")
PRED_NAME_HINTS = ("pred", "prob", "logit", "mask", "binary", "output")


@dataclass
class CaseEntry:
    case_id: str
    split: str
    pred_path: Path
    gt_path: Path
    image_path: Optional[Path]


@dataclass
class LoadedVolume:
    array: np.ndarray
    spacing: Optional[Tuple[float, float, float]]
    affine: Optional[np.ndarray]
    path: Path


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("phase_a_results")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def suppress_noisy_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*Orientationd\.__init__:labels.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Using a non-tuple sequence for multidimensional indexing is deprecated.*",
        category=UserWarning,
    )


def set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def strip_known_ext(name: str) -> str:
    lower = name.lower()
    for ext in ALLOWED_EXTS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def canonical_case_id(raw: Any) -> str:
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    # Handle numeric strings from csv/xlsx such as "801.0"
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    if re.fullmatch(r"\d+", s):
        return str(int(s))
    return s.lower()


def case_id_candidates_from_text(text: str) -> List[str]:
    candidates: List[str] = []
    base = strip_known_ext(text)
    cleaned = re.sub(r"(?:_|-)?(?:pred|prob|logit|mask|binary|post|output|seg)$", "", base, flags=re.IGNORECASE)
    for token in [base, cleaned]:
        cid = canonical_case_id(token)
        if cid:
            candidates.append(cid)
    for m in re.findall(r"\d+", base):
        cid = canonical_case_id(m)
        if cid:
            candidates.append(cid)
    # Preserve order while de-duplicating.
    out: List[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def case_id_candidates_from_path(path: Path) -> List[str]:
    cands: List[str] = []
    cands.extend(case_id_candidates_from_text(path.name))
    cands.extend(case_id_candidates_from_text(path.parent.name))
    if path.parent.parent != path.parent:
        cands.extend(case_id_candidates_from_text(path.parent.parent.name))
    out: List[str] = []
    seen: set[str] = set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def optional_import_repo_eval_utils(logger: logging.Logger) -> Dict[str, Any]:
    try:
        import evaluate_checkpoints as ec  # type: ignore

        logger.info("Reusing split/path utilities from evaluate_checkpoints.py")
        return {
            "autodetect_data_root": getattr(ec, "autodetect_data_root", None),
            "build_data_index": getattr(ec, "build_data_index", None),
            "remap_case_paths_by_id": getattr(ec, "remap_case_paths_by_id", None),
            "load_split_cases": getattr(ec, "load_split_cases", None),
            "resolve_device": getattr(ec, "resolve_device", None),
            "_mps_available": getattr(ec, "_mps_available", None),
        }
    except Exception as e:
        logger.warning(f"Could not import evaluate_checkpoints.py utilities; using local fallback ({e})")
        return {}


def local_mps_available() -> bool:
    if torch is None or not hasattr(torch.backends, "mps"):
        return False
    try:
        return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())
    except Exception:
        return False


def resolve_device(
    device_arg: str,
    strict_device: bool,
    logger: logging.Logger,
    repo_utils: Dict[str, Any],
) -> str:
    fn = repo_utils.get("resolve_device")
    if callable(fn):
        try:
            return fn(device_arg, strict_device, logger)
        except Exception:
            pass

    cuda_ok = bool(torch is not None and torch.cuda.is_available())
    mps_ok = local_mps_available()

    if device_arg == "auto":
        if platform.system() == "Darwin" and mps_ok:
            return "mps"
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"

    if device_arg == "cuda" and not cuda_ok:
        msg = "Requested --device cuda, but CUDA is unavailable."
        if strict_device:
            raise RuntimeError(msg)
        if mps_ok:
            logger.warning(f"{msg} Falling back to mps.")
            return "mps"
        logger.warning(f"{msg} Falling back to cpu.")
        return "cpu"

    if device_arg == "mps" and not mps_ok:
        msg = "Requested --device mps, but MPS is unavailable."
        if strict_device:
            raise RuntimeError(msg)
        logger.warning(f"{msg} Falling back to cpu.")
        return "cpu"
    return device_arg


def autodetect_data_root(
    explicit_data_root: Optional[str],
    logger: logging.Logger,
    repo_utils: Dict[str, Any],
) -> Optional[Path]:
    fn = repo_utils.get("autodetect_data_root")
    if callable(fn):
        try:
            root = fn(explicit_data_root, logger)
            return Path(root) if root else None
        except Exception:
            pass

    if explicit_data_root:
        p = Path(explicit_data_root).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--gt_dir/--data_root path does not exist: {p}")
        return p
    candidates = [
        (Path.cwd() / "data" / "processed" / "all").resolve(),
        (Path(__file__).resolve().parent / "data" / "processed" / "all").resolve(),
        Path("/workspace/cloud_bundle/data/processed/all"),
    ]
    for c in candidates:
        if c.exists():
            logger.info(f"Auto-detected data root: {c}")
            return c
    logger.warning("Could not auto-detect data root.")
    return None


def build_data_index(
    data_root: Optional[Path],
    logger: logging.Logger,
    repo_utils: Dict[str, Any],
) -> Dict[str, str]:
    fn = repo_utils.get("build_data_index")
    if callable(fn):
        try:
            return dict(fn(data_root, logger))
        except Exception:
            pass
    if data_root is None or not data_root.exists():
        return {}
    out: Dict[str, str] = {}
    for p in data_root.rglob("*.nii.gz"):
        out.setdefault(p.name, str(p.resolve()))
    for p in data_root.rglob("*.nii"):
        out.setdefault(p.name, str(p.resolve()))
    logger.info(f"Indexed {len(out)} files under {data_root}")
    return out


def local_remap_case_paths_by_id(case_id: str, data_index: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    if not data_index:
        return None, None
    cids = [case_id]
    if case_id.isdigit():
        cids.append(str(int(case_id)))
    image_candidates: List[str] = []
    label_candidates: List[str] = []
    for cid in cids:
        image_candidates.extend([f"{cid}.img.nii.gz", f"{cid}_img.nii.gz", f"{cid}.image.nii.gz", f"{cid}_image.nii.gz"])
        label_candidates.extend([f"{cid}.label.nii.gz", f"{cid}_label.nii.gz", f"{cid}-label.nii.gz", f"{cid}.mask.nii.gz", f"{cid}_mask.nii.gz"])
    image = next((data_index[n] for n in image_candidates if n in data_index), None)
    label = next((data_index[n] for n in label_candidates if n in data_index), None)
    return image, label


def remap_case_paths_by_id(
    case_id: str,
    data_index: Dict[str, str],
    repo_utils: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    fn = repo_utils.get("remap_case_paths_by_id")
    if callable(fn):
        try:
            return fn(case_id, data_index)
        except Exception:
            pass
    return local_remap_case_paths_by_id(case_id, data_index)


def read_ids_file(path: Path) -> List[str]:
    ids: List[str] = []
    if not path.exists():
        return ids
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in re.split(r"[,;\t ]+", line) if p.strip()]
            if not parts:
                continue
            cid = canonical_case_id(parts[0])
            if cid:
                ids.append(cid)
    return ids


def normalize_split_value(value: Any) -> Optional[str]:
    s = str(value).strip().lower()
    if not s or s in {"nan", "none"}:
        return None
    if s in {"train", "training"}:
        return "train"
    if s in {"val", "valid", "validation"}:
        return "val"
    if s in {"test", "testing"}:
        return "test"
    return None


def parse_splits_json_records(
    splits_json: Path,
    logger: logging.Logger,
    data_index: Dict[str, str],
    repo_utils: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, Optional[str]]]]:
    out: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {"val": {}, "test": {}}
    if not splits_json.exists():
        return out

    # Prefer existing loader from evaluate_checkpoints.py to avoid duplicating repo-specific remap behavior.
    loader = repo_utils.get("load_split_cases")
    if callable(loader):
        try:
            loaded_total = 0
            for split_name in ("val", "test"):
                recs = loader(splits_json, split_name, None, data_index, True, logger)
                for r in recs:
                    cid = canonical_case_id(r.get("id"))
                    if not cid:
                        continue
                    image = str(r.get("image", "")).strip() or None
                    label = str(r.get("label", "")).strip() or None
                    out[split_name][cid] = {
                        "image": image,
                        "label": label,
                    }
                loaded_total += len(out[split_name])
            if loaded_total > 0:
                return out
            logger.warning(
                "evaluate_checkpoints.load_split_cases returned zero val/test records; "
                "falling back to local split parser for ID recovery."
            )
            out = {"val": {}, "test": {}}
        except Exception as e:
            logger.warning(f"evaluate_checkpoints.load_split_cases failed; falling back to local parser ({e})")

    with open(splits_json, "r") as f:
        payload = json.load(f)
    for split_name in ("val", "test"):
        recs = payload.get(split_name, [])
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            cid = canonical_case_id(rec.get("id"))
            if not cid:
                continue
            image = str(rec.get("image", "")).strip() or None
            label = str(rec.get("label", "")).strip() or None

            if label and not Path(label).exists():
                remap_img, remap_lbl = remap_case_paths_by_id(cid, data_index, repo_utils)
                if remap_lbl:
                    label = remap_lbl
                if remap_img and (not image or not Path(image).exists()):
                    image = remap_img
            out[split_name][cid] = {"image": image, "label": label}
    return out


def parse_split_xlsx_ids(
    split_xlsx: Path,
    id_key: Optional[str],
    logger: logging.Logger,
) -> Dict[str, List[str]]:
    results = {"val": [], "test": []}
    if not split_xlsx.exists():
        return results

    def add_ids(split_val: str, ids: Iterable[Any]) -> None:
        for raw in ids:
            cid = canonical_case_id(raw)
            if cid:
                results[split_val].append(cid)

    if pd is not None:
        try:
            xls = pd.ExcelFile(split_xlsx)
            chosen_sheet = xls.sheet_names[0]
            for sheet in xls.sheet_names:
                df = pd.read_excel(split_xlsx, sheet_name=sheet)
                if df.empty:
                    continue
                # Common pattern in this repo: first data row contains headers (FileName, Split-1,...)
                if any(str(c).lower().startswith("unnamed") for c in df.columns):
                    first_row = [str(v).strip() for v in df.iloc[0].tolist()]
                    if any("split" in v.lower() for v in first_row):
                        df.columns = first_row
                        df = df.iloc[1:].copy()
                cols = [str(c).strip() for c in df.columns]
                id_col = None
                for c in cols:
                    lc = c.lower()
                    if any(k in lc for k in ("filename", "file", "id", "case")):
                        id_col = c
                        break
                if id_col is None:
                    id_col = cols[0]
                split_cols = [c for c in cols if "split" in c.lower()]
                if not split_cols:
                    # Fallback: inspect columns with split-like values.
                    for c in cols:
                        sample = [normalize_split_value(v) for v in df[c].head(30).tolist()]
                        if any(v in {"train", "val", "test"} for v in sample):
                            split_cols.append(c)
                if not split_cols:
                    continue
                chosen_split_col = id_key if (id_key and id_key in cols) else split_cols[0]
                tmp_val = df.loc[df[chosen_split_col].apply(normalize_split_value) == "val", id_col].tolist()
                tmp_test = df.loc[df[chosen_split_col].apply(normalize_split_value) == "test", id_col].tolist()
                if tmp_val or tmp_test:
                    add_ids("val", tmp_val)
                    add_ids("test", tmp_test)
                    chosen_sheet = sheet
                    logger.info(f"Parsed split IDs from XLSX sheet='{chosen_sheet}', split_col='{chosen_split_col}', id_col='{id_col}'")
                    break
            return {
                "val": sorted(set(results["val"])),
                "test": sorted(set(results["test"])),
            }
        except Exception as e:
            logger.warning(f"pandas XLSX parser failed ({e}); trying openpyxl fallback.")

    if openpyxl is not None:
        wb = openpyxl.load_workbook(split_xlsx, data_only=True)
        for ws in wb.worksheets:
            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                continue
            header = [str(v).strip() if v is not None else "" for v in rows[0]]
            if any("split" in h.lower() for h in header):
                data_rows = rows[1:]
            elif len(rows) > 1:
                header2 = [str(v).strip() if v is not None else "" for v in rows[1]]
                if any("split" in h.lower() for h in header2):
                    header = header2
                    data_rows = rows[2:]
                else:
                    data_rows = rows[1:]
            else:
                data_rows = rows[1:]

            id_col_idx = 0
            for i, h in enumerate(header):
                if any(k in h.lower() for k in ("filename", "file", "id", "case")):
                    id_col_idx = i
                    break

            split_col_idxs = [i for i, h in enumerate(header) if "split" in h.lower()]
            if not split_col_idxs:
                continue
            chosen_idx = split_col_idxs[0]
            if id_key:
                for i, h in enumerate(header):
                    if h == id_key:
                        chosen_idx = i
                        break
            val_ids: List[Any] = []
            test_ids: List[Any] = []
            for row in data_rows:
                if row is None:
                    continue
                split_val = normalize_split_value(row[chosen_idx] if chosen_idx < len(row) else None)
                if split_val not in {"val", "test"}:
                    continue
                cid_raw = row[id_col_idx] if id_col_idx < len(row) else None
                if split_val == "val":
                    val_ids.append(cid_raw)
                else:
                    test_ids.append(cid_raw)
            if val_ids or test_ids:
                add_ids("val", val_ids)
                add_ids("test", test_ids)
                logger.info(f"Parsed split IDs from XLSX sheet='{ws.title}'")
                break
        return {
            "val": sorted(set(results["val"])),
            "test": sorted(set(results["test"])),
        }

    logger.warning("Neither pandas nor openpyxl is available; cannot parse split_xlsx.")
    return {"val": [], "test": []}


def scan_files(root: Path, allowed_exts: Sequence[str] = ALLOWED_EXTS) -> List[Path]:
    if not root.exists():
        return []
    paths: List[Path] = []
    for ext in allowed_exts:
        paths.extend(root.rglob(f"*{ext}"))
    return sorted({p.resolve() for p in paths if p.is_file()})


def discover_prediction_files(pred_dir: Optional[str], logger: logging.Logger) -> List[Path]:
    if pred_dir:
        root = Path(pred_dir).expanduser().resolve()
        files = scan_files(root)
        logger.info(f"Using --pred_dir={root} with {len(files)} candidate files")
        return files

    candidate_roots = [
        Path("eval_outputs").resolve(),
        Path("pipeline_outputs").resolve(),
        Path("checkpoints/val_preds").resolve(),
    ]
    discovered: List[Path] = []
    for root in candidate_roots:
        if not root.exists():
            continue
        files = scan_files(root)
        for p in files:
            low = p.as_posix().lower()
            if any(k in low for k in PRED_NAME_HINTS) or "phasea" in low:
                discovered.append(p)
    discovered = sorted({p.resolve() for p in discovered})
    if not discovered:
        # Conservative fallback to any NIfTI/NPY under eval_outputs + pipeline_outputs only.
        for root in candidate_roots[:2]:
            if root.exists():
                discovered.extend(scan_files(root))
        discovered = sorted({p.resolve() for p in discovered})
    logger.info(f"Auto-discovered {len(discovered)} prediction candidate files")
    return discovered


def build_id_to_paths(files: Sequence[Path]) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for p in files:
        for cid in case_id_candidates_from_path(p):
            out.setdefault(cid, []).append(p)
    return out


def choose_path(paths: Sequence[Path], role: str, pred_type: str = "auto") -> Optional[Path]:
    if not paths:
        return None

    def score(path: Path) -> Tuple[int, int]:
        low = path.as_posix().lower()
        s = 0
        if role == "pred":
            if any(k in low for k in ("pred", "prob", "logit", "mask", "binary", "phasea")):
                s += 8
            if "label" in low:
                s -= 10
            if pred_type == "prob":
                if "prob" in low or ".pred." in low or low.endswith(".pred.npy"):
                    s += 10
                if "binary" in low or ".binary." in low:
                    s -= 4
            elif pred_type == "logit" and "logit" in low:
                s += 8
            elif pred_type == "binary":
                if "binary" in low or ".binary." in low or "mask" in low:
                    s += 10
                elif ".pred." in low or low.endswith(".pred.npy") or "prob" in low:
                    s -= 4
        elif role == "gt":
            if any(k in low for k in ("label", "mask", "seg")):
                s += 10
            if any(k in low for k in ("pred", "prob", "logit")):
                s -= 8
        elif role == "image":
            if any(k in low for k in ("img", "image", "ct")):
                s += 6
            if any(k in low for k in ("label", "mask", "seg", "pred", "prob", "logit")):
                s -= 6
        ext_score = 0
        if str(path).lower().endswith(".nii.gz"):
            ext_score = 3
        elif str(path).lower().endswith(".nii"):
            ext_score = 2
        elif str(path).lower().endswith(".npy"):
            ext_score = 1
        return (s, ext_score)

    return sorted(paths, key=score, reverse=True)[0]


def load_volume(path: Path) -> LoadedVolume:
    low = path.name.lower()
    arr: np.ndarray
    spacing: Optional[Tuple[float, float, float]] = None
    affine: Optional[np.ndarray] = None
    if low.endswith(".npy"):
        arr = np.load(path)
    elif low.endswith(".nii.gz") or low.endswith(".nii"):
        if nib is None:
            if sitk is None:
                raise RuntimeError("nibabel and SimpleITK are both unavailable for NIfTI IO.")
            img = sitk.ReadImage(str(path))
            arr = sitk.GetArrayFromImage(img)
            sp = img.GetSpacing()
            spacing = (float(sp[2]), float(sp[1]), float(sp[0]))  # sitk spacing is x,y,z
            affine = np.eye(4)
        else:
            img = nib.load(str(path))
            arr = np.asarray(img.get_fdata())
            if hasattr(img, "header"):
                zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
                if len(zooms) == 3:
                    spacing = zooms
            affine = np.asarray(img.affine) if hasattr(img, "affine") else None
    else:
        raise ValueError(f"Unsupported file type: {path}")

    arr = np.asarray(arr)
    if arr.ndim > 3:
        arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume after squeeze, got shape={arr.shape} for {path}")
    return LoadedVolume(array=arr.astype(np.float32), spacing=spacing, affine=affine, path=path)


def infer_pred_type(arr: np.ndarray) -> str:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "binary"
    unique = np.unique(finite)
    if unique.size <= 2 and set(np.round(unique).astype(int).tolist()).issubset({0, 1}):
        return "binary"
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin < 0.0 or vmax > 1.0:
        return "logit"
    return "prob"


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def prediction_to_binary(
    pred_arr: np.ndarray,
    pred_type_setting: str,
    threshold: float,
) -> Tuple[np.ndarray, str, float]:
    inferred = infer_pred_type(pred_arr) if pred_type_setting == "auto" else pred_type_setting
    if inferred == "binary":
        return (pred_arr > 0).astype(np.uint8), "binary", 0.0
    if inferred == "logit":
        probs = sigmoid(pred_arr.astype(np.float32))
        return (probs >= threshold).astype(np.uint8), "logit", threshold
    # prob
    return (pred_arr >= threshold).astype(np.uint8), "prob", threshold


def postprocess_binary_mask(mask: np.ndarray, min_component_voxels: int, do_hole_fill: bool) -> np.ndarray:
    if ndi is None:
        return mask.astype(np.uint8)
    out = mask.astype(bool)
    if min_component_voxels > 0:
        labeled, n_cc = ndi.label(out)
        if n_cc > 0:
            counts = np.bincount(labeled.ravel())
            keep = np.where(counts >= min_component_voxels)[0]
            keep = keep[keep != 0]
            out = np.isin(labeled, keep)
    if do_hole_fill:
        out = ndi.binary_fill_holes(out)
    return out.astype(np.uint8)


def confusion_from_binary(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    tn = int(np.logical_and(~pred_b, ~gt_b).sum())
    return tp, fp, fn, tn


def compute_hd95_assd(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Optional[Tuple[float, float, float]],
    run_assd: bool,
) -> Tuple[float, float, str]:
    unit = "mm" if spacing is not None else "vox"
    sampling = spacing if spacing is not None else (1.0, 1.0, 1.0)

    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    if not pred_b.any() and not gt_b.any():
        return 0.0, 0.0 if run_assd else float("nan"), unit
    if not pred_b.any() or not gt_b.any():
        return float("inf"), float("inf") if run_assd else float("nan"), unit
    if ndi is None:
        return float("nan"), float("nan"), unit

    structure = np.ones((3, 3, 3), dtype=bool)
    pred_eroded = ndi.binary_erosion(pred_b, structure=structure, border_value=0)
    gt_eroded = ndi.binary_erosion(gt_b, structure=structure, border_value=0)
    pred_surf = np.logical_xor(pred_b, pred_eroded)
    gt_surf = np.logical_xor(gt_b, gt_eroded)
    if not pred_surf.any():
        pred_surf = pred_b
    if not gt_surf.any():
        gt_surf = gt_b

    dt_gt = ndi.distance_transform_edt(~gt_surf, sampling=sampling)
    dt_pred = ndi.distance_transform_edt(~pred_surf, sampling=sampling)
    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    hd95 = float(np.percentile(all_d, 95)) if all_d.size else float("nan")
    assd = float(np.mean(all_d)) if (run_assd and all_d.size) else float("nan")
    return hd95, assd, unit


def compute_cldice(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, bool]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    if not pred_b.any() and not gt_b.any():
        return 1.0, False
    if skeletonize_3d is None:
        # graceful fallback: use Dice as clDice surrogate
        tp = np.logical_and(pred_b, gt_b).sum()
        denom = pred_b.sum() + gt_b.sum()
        dice = (2.0 * tp) / denom if denom > 0 else 1.0
        return float(dice), True

    skel_pred = skeletonize_3d(pred_b).astype(bool)
    skel_gt = skeletonize_3d(gt_b).astype(bool)
    skel_pred_n = int(skel_pred.sum())
    skel_gt_n = int(skel_gt.sum())

    tprec = float(np.logical_and(skel_pred, gt_b).sum() / skel_pred_n) if skel_pred_n > 0 else 0.0
    tsens = float(np.logical_and(skel_gt, pred_b).sum() / skel_gt_n) if skel_gt_n > 0 else 0.0
    if tprec + tsens == 0:
        return (1.0 if (not pred_b.any() and not gt_b.any()) else 0.0), False
    return float(2.0 * tprec * tsens / (tprec + tsens)), False


def connected_component_stats(mask: np.ndarray) -> Tuple[float, float]:
    if ndi is None:
        return float("nan"), float("nan")
    mask_b = mask.astype(bool)
    if not mask_b.any():
        return 0.0, 0.0
    labeled, n_cc = ndi.label(mask_b)
    counts = np.bincount(labeled.ravel())
    largest = int(counts[1:].max()) if counts.size > 1 else 0
    frac = float(largest / mask_b.sum()) if mask_b.sum() > 0 else 0.0
    return float(n_cc), frac


def compute_case_metrics(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    spacing: Optional[Tuple[float, float, float]],
    threshold_used: float,
    run_assd: bool,
) -> Dict[str, Any]:
    tp, fp, fn, tn = confusion_from_binary(pred_bin, gt_bin)
    pred_fg = int(pred_bin.sum())
    gt_fg = int(gt_bin.sum())
    gt_empty = (gt_fg == 0)
    pred_empty = (pred_fg == 0)

    # Explicit empty-mask conventions (documented in metric_conventions.json):
    # - GT empty & Pred empty => perfect agreement
    # - GT empty & Pred non-empty => clear false positive failure
    # - GT non-empty & Pred empty => complete miss of target
    if gt_empty and pred_empty:
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
        specificity = 1.0
    elif gt_empty and not pred_empty:
        dice = 0.0
        iou = 0.0
        precision = 0.0
        recall = 1.0
        specificity = 0.0
    elif (not gt_empty) and pred_empty:
        dice = 0.0
        iou = 0.0
        precision = 1.0
        recall = 0.0
        specificity = 1.0
    else:
        dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0

    voxel_vol = float(np.prod(spacing)) if spacing is not None else 1.0
    abs_vol_diff = abs(pred_fg - gt_fg) * voxel_vol
    if gt_fg > 0:
        rel_vol_err = (abs(pred_fg - gt_fg) / gt_fg) * 100.0
    elif pred_fg == 0:
        rel_vol_err = 0.0
    else:
        rel_vol_err = float("nan")

    hd95, assd, hd_unit = compute_hd95_assd(pred_bin, gt_bin, spacing, run_assd=run_assd)
    cldice, cldice_fallback = compute_cldice(pred_bin, gt_bin)
    pred_cc, pred_lcc_frac = connected_component_stats(pred_bin)
    gt_cc, _ = connected_component_stats(gt_bin)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_fg_voxels": pred_fg,
        "gt_fg_voxels": gt_fg,
        "pred_empty": int(pred_empty),
        "gt_empty": int(gt_empty),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "iou": float(iou),
        "abs_volume_difference": float(abs_vol_diff),
        "relative_volume_error_percent": float(rel_vol_err),
        "voxel_volume": float(voxel_vol),
        "hd95": float(hd95),
        "hd95_unit": hd_unit,
        "assd": float(assd),
        "cldice": float(cldice),
        "cldice_fallback_used": bool(cldice_fallback),
        "pred_cc_count": float(pred_cc),
        "gt_cc_count": float(gt_cc),
        "pred_lcc_fraction": float(pred_lcc_frac),
        "fragmentation_proxy": float(pred_cc - gt_cc) if np.isfinite(pred_cc) and np.isfinite(gt_cc) else float("nan"),
        "threshold_used": float(threshold_used),
        "spacing_available": bool(spacing is not None),
    }


def parse_thresholds(spec: str) -> List[float]:
    spec = spec.strip()
    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Invalid threshold range: {spec}")
        start, stop, step = map(float, parts)
        if step <= 0:
            raise ValueError("Threshold step must be > 0")
        vals = list(np.arange(start, stop + 1e-12, step))
    else:
        vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    cleaned = sorted({float(np.clip(v, 0.0, 1.0)) for v in vals})
    return cleaned


def finite_values(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_ci_mean(values: np.ndarray, n_bootstrap: int, seed: int) -> Tuple[float, float]:
    vals = finite_values(values)
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(vals, size=vals.size, replace=True)
        means[i] = float(np.mean(sample))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_metric(values: Sequence[float], n_bootstrap: int, seed: int) -> Dict[str, float]:
    vals = finite_values(values)
    if vals.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "iqr": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    q1 = float(np.percentile(vals, 25))
    q3 = float(np.percentile(vals, 75))
    ci_low, ci_high = bootstrap_ci_mean(vals, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "median": float(np.median(vals)),
        "iqr": float(q3 - q1),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def pick_summary_metrics() -> List[str]:
    return [
        "dice",
        "cldice",
        "hd95",
        "assd",
        "precision",
        "recall",
        "specificity",
        "iou",
        "abs_volume_difference",
        "relative_volume_error_percent",
        "pred_cc_count",
        "gt_cc_count",
        "pred_lcc_fraction",
        "fragmentation_proxy",
    ]


def evaluate_loaded_case(
    pred_vol: LoadedVolume,
    gt_vol: LoadedVolume,
    threshold: float,
    pred_type_setting: str,
    apply_postprocess: bool,
    min_component_voxels: int,
    run_assd: bool,
) -> Tuple[Dict[str, Any], str]:
    pred_arr = pred_vol.array
    gt_arr = gt_vol.array
    pred_arr, gt_arr, _ = align_shapes_if_safe(pred_arr, gt_arr)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(f"Shape mismatch pred={pred_arr.shape}, gt={gt_arr.shape}")

    gt_bin = (gt_arr > 0).astype(np.uint8)
    pred_bin, inferred_type, used_thr = prediction_to_binary(pred_arr, pred_type_setting, threshold)
    if apply_postprocess:
        pred_bin = postprocess_binary_mask(
            pred_bin,
            min_component_voxels=min_component_voxels,
            do_hole_fill=True,
        )
    spacing = gt_vol.spacing if gt_vol.spacing is not None else pred_vol.spacing
    metrics = compute_case_metrics(pred_bin, gt_bin, spacing=spacing, threshold_used=used_thr, run_assd=run_assd)
    return metrics, inferred_type


def align_shapes_if_safe(pred_arr: np.ndarray, gt_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    if pred_arr.shape == gt_arr.shape:
        return pred_arr, gt_arr, False

    # Safe fix: squeeze singleton dims only.
    pred_s = np.squeeze(pred_arr)
    gt_s = np.squeeze(gt_arr)
    if pred_s.shape == gt_s.shape and pred_s.ndim == 3 and gt_s.ndim == 3:
        return pred_s, gt_s, True

    return pred_arr, gt_arr, False


def choose_threshold_from_validation(
    val_cases: Sequence[CaseEntry],
    thresholds: Sequence[float],
    pred_type: str,
    primary_metric: str,
    apply_postprocess: bool,
    min_component_voxels: int,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not val_cases:
        return rows, {
            "selection_source": "no_validation_cases",
            "primary_metric": primary_metric,
            "selected_threshold": 0.5,
            "validation_case_count": 0,
            "note": "Validation set unavailable for threshold sweep; used default 0.5.",
        }

    # Load once per case to avoid repeated NIfTI IO per threshold.
    loaded: List[Tuple[CaseEntry, LoadedVolume, LoadedVolume]] = []
    for case in val_cases:
        try:
            pred_vol = load_volume(case.pred_path)
            gt_vol = load_volume(case.gt_path)
            loaded.append((case, pred_vol, gt_vol))
        except Exception as e:
            logger.warning(f"[val] skipping {case.case_id} during threshold sweep: {e}")

    if not loaded:
        return rows, {
            "selection_source": "validation_load_failure",
            "primary_metric": primary_metric,
            "selected_threshold": 0.5,
            "validation_case_count": 0,
            "note": "No validation cases were loadable; used default 0.5.",
        }

    for thr in thresholds:
        per_thr: Dict[str, List[float]] = {"dice": [], "cldice": [], "hd95": [], "assd": []}
        used_cases = 0
        for case, pred_vol, gt_vol in loaded:
            try:
                metrics, _ = evaluate_loaded_case(
                    pred_vol,
                    gt_vol,
                    threshold=thr,
                    pred_type_setting=pred_type,
                    apply_postprocess=apply_postprocess,
                    min_component_voxels=min_component_voxels,
                    run_assd=False,
                )
                for k in per_thr.keys():
                    v = float(metrics.get(k, float("nan")))
                    if np.isfinite(v):
                        per_thr[k].append(v)
                used_cases += 1
            except Exception:
                continue

        row = {
            "threshold": float(thr),
            "n_cases": int(used_cases),
            "mean_dice": float(np.mean(per_thr["dice"])) if per_thr["dice"] else float("nan"),
            "mean_cldice": float(np.mean(per_thr["cldice"])) if per_thr["cldice"] else float("nan"),
            "mean_hd95": float(np.mean(per_thr["hd95"])) if per_thr["hd95"] else float("nan"),
            "mean_assd": float(np.mean(per_thr["assd"])) if per_thr["assd"] else float("nan"),
        }
        rows.append(row)

    metric_key = {
        "dice": "mean_dice",
        "cldice": "mean_cldice",
        "hd95": "mean_hd95",
        "assd": "mean_assd",
    }.get(primary_metric.lower(), "mean_dice")

    valid_rows = [r for r in rows if np.isfinite(r.get(metric_key, float("nan")))]
    if not valid_rows:
        selected = thresholds[0] if thresholds else 0.5
        selection = {
            "selection_source": "validation_metric_all_nan",
            "primary_metric": primary_metric,
            "primary_metric_column": metric_key,
            "selected_threshold": float(selected),
            "validation_case_count": int(len(loaded)),
            "note": "No finite validation metric values; used first threshold.",
        }
        return rows, selection

    minimize = metric_key in {"mean_hd95", "mean_assd"}
    if minimize:
        best = min(valid_rows, key=lambda r: float(r[metric_key]))
    else:
        best = max(valid_rows, key=lambda r: float(r[metric_key]))

    selection = {
        "selection_source": "validation_sweep",
        "primary_metric": primary_metric,
        "primary_metric_column": metric_key,
        "selected_threshold": float(best["threshold"]),
        "selected_metric_value": float(best[metric_key]),
        "validation_case_count": int(len(loaded)),
    }
    return rows, selection


def format_spacing(spacing: Optional[Tuple[float, float, float]]) -> str:
    if spacing is None:
        return "unknown"
    return f"{spacing[0]:.4f},{spacing[1]:.4f},{spacing[2]:.4f}"


def environment_info(device: str, repo_utils: Dict[str, Any]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "os": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "device_requested_or_selected": device,
        "timestamp": datetime.now().isoformat(),
        "numpy_version": getattr(np, "__version__", None),
        "torch_version": None,
        "monai_version": None,
        "nibabel_version": getattr(nib, "__version__", None) if nib is not None else None,
        "scipy_version": getattr(sys.modules.get("scipy"), "__version__", None) if "scipy" in sys.modules else None,
        "skimage_version": getattr(sys.modules.get("skimage"), "__version__", None) if "skimage" in sys.modules else None,
        "cuda_available": False,
        "mps_available": False,
        "gpu_name": None,
        "dependency_warnings": [],
    }
    mps_fn = repo_utils.get("_mps_available")
    if torch is not None:
        info["torch_version"] = getattr(torch, "__version__", None)
        try:
            info["cuda_available"] = bool(torch.cuda.is_available())
            if info["cuda_available"]:
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        if callable(mps_fn):
            try:
                info["mps_available"] = bool(mps_fn())
            except Exception:
                info["mps_available"] = local_mps_available()
        else:
            info["mps_available"] = local_mps_available()
    if monai is not None:
        info["monai_version"] = getattr(monai, "__version__", None)
    dep_warnings: List[str] = []
    if nib is None and sitk is None:
        dep_warnings.append("nibabel and SimpleITK missing: NIfTI IO is unavailable.")
    if ndi is None:
        dep_warnings.append("scipy.ndimage missing: connected-component and HD95/ASSD fallbacks are limited.")
    if skeletonize_3d is None:
        dep_warnings.append("skimage skeletonize_3d missing: clDice falls back to Dice surrogate.")
    info["dependency_warnings"] = dep_warnings
    return info


def plot_threshold_sweep(rows: Sequence[Dict[str, Any]], out_path: Path, selected_threshold: float) -> None:
    if plt is None or not rows:
        return
    xs = [r["threshold"] for r in rows]
    dice = [r.get("mean_dice", float("nan")) for r in rows]
    cldice = [r.get("mean_cldice", float("nan")) for r in rows]
    hd95 = [r.get("mean_hd95", float("nan")) for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(xs, dice, marker="o", label="Mean Dice")
    ax1.plot(xs, cldice, marker="s", label="Mean clDice")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score (higher better)")
    ax1.axvline(selected_threshold, color="k", linestyle="--", label=f"Selected={selected_threshold:.3f}")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, hd95, marker="^", color="tab:red", label="Mean HD95")
    ax2.set_ylabel("HD95 (lower better)")

    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(lab)
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_core_metrics(per_case_rows: Sequence[Dict[str, Any]], plots_dir: Path) -> None:
    if plt is None or not per_case_rows:
        return
    dice = finite_values(r.get("dice", float("nan")) for r in per_case_rows)
    hd95 = finite_values(r.get("hd95", float("nan")) for r in per_case_rows)
    cldice_vals = finite_values(r.get("cldice", float("nan")) for r in per_case_rows)
    frag = finite_values(r.get("fragmentation_proxy", float("nan")) for r in per_case_rows)

    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    for name, vals in [("Dice", dice), ("HD95", hd95), ("clDice", cldice_vals)]:
        if vals.size > 0:
            data.append(vals)
            labels.append(name)
    if data:
        ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title("Metrics Boxplot")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "metrics_boxplot.png", dpi=180)
    plt.close(fig)

    # 2) dice histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    if dice.size > 0:
        ax.hist(dice, bins=20)
    ax.set_xlabel("Dice")
    ax.set_ylabel("Case count")
    ax.set_title("Dice Histogram")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "dice_histogram.png", dpi=180)
    plt.close(fig)

    # 3) dice vs hd95
    fig, ax = plt.subplots(figsize=(8, 5))
    pairs = [(float(r["dice"]), float(r["hd95"])) for r in per_case_rows if np.isfinite(r.get("dice", np.nan)) and np.isfinite(r.get("hd95", np.nan))]
    if pairs:
        x, y = zip(*pairs)
        ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel("Dice")
    ax.set_ylabel("HD95")
    ax.set_title("Dice vs HD95")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "dice_vs_hd95.png", dpi=180)
    plt.close(fig)

    # 4) dice vs cldice
    fig, ax = plt.subplots(figsize=(8, 5))
    pairs = [(float(r["dice"]), float(r["cldice"])) for r in per_case_rows if np.isfinite(r.get("dice", np.nan)) and np.isfinite(r.get("cldice", np.nan))]
    if pairs:
        x, y = zip(*pairs)
        ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel("Dice")
    ax.set_ylabel("clDice")
    ax.set_title("Dice vs clDice")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "dice_vs_cldice.png", dpi=180)
    plt.close(fig)

    # 5) cldice vs fragmentation
    fig, ax = plt.subplots(figsize=(8, 5))
    pairs = [(float(r["cldice"]), float(r["fragmentation_proxy"])) for r in per_case_rows if np.isfinite(r.get("cldice", np.nan)) and np.isfinite(r.get("fragmentation_proxy", np.nan))]
    if pairs:
        x, y = zip(*pairs)
        ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel("clDice")
    ax.set_ylabel("Fragmentation proxy (pred_cc - gt_cc)")
    ax.set_title("clDice vs Fragmentation")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "cldice_vs_fragmentation.png", dpi=180)
    plt.close(fig)


def choose_slice_idx(gt: np.ndarray, pred: np.ndarray) -> int:
    combined = (gt.astype(np.uint8) + pred.astype(np.uint8))
    per_slice = combined.reshape(combined.shape[0], -1).sum(axis=1)
    return int(np.argmax(per_slice)) if per_slice.size > 0 else int(gt.shape[0] // 2)


def load_optional_image(image_path: Optional[Path], fallback_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    if image_path is None or not image_path.exists():
        return None
    try:
        v = load_volume(image_path)
        if v.array.shape == fallback_shape:
            return v.array
        return None
    except Exception:
        return None


def normalize_slice(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    p1, p99 = np.percentile(x, [1, 99])
    if p99 <= p1:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x - p1) / (p99 - p1), 0, 1)
    return y.astype(np.float32)


def build_error_map(pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    pred_b = pred_bin.astype(bool)
    gt_b = gt_bin.astype(bool)
    err = np.zeros(pred_bin.shape, dtype=np.uint8)
    err[np.logical_and(pred_b, ~gt_b)] = 1  # FP
    err[np.logical_and(~pred_b, gt_b)] = 2  # FN
    err[np.logical_and(pred_b, gt_b)] = 3   # TP
    return err


def save_qualitative_panel(
    case_id: str,
    category: str,
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    image_vol: Optional[np.ndarray],
    out_path: Path,
    title_suffix: str,
) -> None:
    if plt is None:
        return
    z = choose_slice_idx(gt_bin, pred_bin)
    pred_sl = pred_bin[z]
    gt_sl = gt_bin[z]
    err_sl = build_error_map(pred_bin, gt_bin)[z]

    if image_vol is not None:
        img_sl = normalize_slice(image_vol[z])
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img_sl, cmap="gray")
        axes[0].imshow(gt_sl, cmap="Reds", alpha=0.35)
        axes[0].set_title("GT overlay")

        axes[1].imshow(img_sl, cmap="gray")
        axes[1].imshow(pred_sl, cmap="Greens", alpha=0.35)
        axes[1].set_title("Pred overlay")

        axes[2].imshow(img_sl, cmap="gray")
        axes[2].imshow((err_sl == 1).astype(np.uint8), cmap="Blues", alpha=0.45)
        axes[2].imshow((err_sl == 2).astype(np.uint8), cmap="Oranges", alpha=0.45)
        axes[2].set_title("Error overlay (FP blue / FN orange)")

        axes[3].imshow(err_sl, cmap="viridis")
        axes[3].set_title("Error map")
        for ax in axes:
            ax.axis("off")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(gt_sl, cmap="gray")
        axes[0].set_title("GT mask")
        axes[1].imshow(pred_sl, cmap="gray")
        axes[1].set_title("Pred mask")
        axes[2].imshow(err_sl, cmap="viridis")
        axes[2].set_title("Error map")
        for ax in axes:
            ax.axis("off")

    fig.suptitle(f"{category.upper()} | Case {case_id} | {title_suffix}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_error_nifti(
    out_path: Path,
    error_map: np.ndarray,
    reference_gt: LoadedVolume,
) -> None:
    if nib is None:
        return
    affine = reference_gt.affine if reference_gt.affine is not None else np.eye(4)
    nii = nib.Nifti1Image(error_map.astype(np.uint8), affine)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nii.to_filename(str(out_path))


def build_failure_flags(per_case_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    hd95_vals = finite_values(r.get("hd95", np.nan) for r in per_case_rows)
    frag_vals = finite_values(r.get("fragmentation_proxy", np.nan) for r in per_case_rows)
    hd95_cut = float(np.percentile(hd95_vals, 90)) if hd95_vals.size > 0 else float("inf")
    frag_cut = float(np.percentile(frag_vals, 90)) if frag_vals.size > 0 else 3.0
    frag_cut = max(frag_cut, 3.0)

    flags: List[Dict[str, Any]] = []
    for r in per_case_rows:
        pred_fg = float(r.get("pred_fg_voxels", 0))
        gt_fg = float(r.get("gt_fg_voxels", 0))
        precision = float(r.get("precision", np.nan))
        recall = float(r.get("recall", np.nan))
        cldice = float(r.get("cldice", np.nan))
        hd95 = float(r.get("hd95", np.nan))
        frag = float(r.get("fragmentation_proxy", np.nan))
        spacing_avail = bool(r.get("spacing_available", False))

        row = {
            "case_id": r.get("case_id"),
            "empty_prediction": bool(pred_fg == 0 and gt_fg > 0),
            "false_positive_heavy": bool(np.isfinite(precision) and precision < 0.25 and pred_fg > (1.5 * max(gt_fg, 1.0))),
            "boundary_error_high": bool(np.isfinite(hd95) and hd95 > hd95_cut),
            "fragmentation_high": bool(np.isfinite(frag) and frag > frag_cut),
            "possible_distal_miss": bool(np.isfinite(recall) and np.isfinite(cldice) and recall < 0.6 and cldice < 0.6),
            "shape_or_spacing_issue": bool(not spacing_avail),
            "heuristic_note": "Heuristic flags; interpret with qualitative review.",
        }
        flags.append(row)

    summary: Dict[str, Any] = {"n_cases": len(flags), "thresholds": {"hd95_cut": hd95_cut, "frag_cut": frag_cut}}
    for key in ["empty_prediction", "false_positive_heavy", "boundary_error_high", "fragmentation_high", "possible_distal_miss", "shape_or_spacing_issue"]:
        summary[key] = int(sum(1 for row in flags if row.get(key)))
    return flags, summary


def maybe_phaseb_correlation(
    per_case_rows: Sequence[Dict[str, Any]],
    phase_b_qc_csv: Optional[str],
    out_dir: Path,
) -> None:
    if not phase_b_qc_csv:
        return
    path = Path(phase_b_qc_csv).expanduser().resolve()
    if not path.exists():
        return

    phaseb_rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = canonical_case_id(row.get("case_id"))
            if cid:
                row["case_id"] = cid
                phaseb_rows.append(row)
    if not phaseb_rows:
        return

    phasea_by_id = {canonical_case_id(r.get("case_id")): r for r in per_case_rows}
    merged: List[Dict[str, Any]] = []
    for b in phaseb_rows:
        cid = b["case_id"]
        if cid in phasea_by_id:
            row = {}
            row.update({f"phaseb_{k}": v for k, v in b.items()})
            row.update({f"phasea_{k}": v for k, v in phasea_by_id[cid].items()})
            row["case_id"] = cid
            merged.append(row)
    if not merged:
        return

    write_rows_csv(out_dir / "phase_a_phase_b_merged.csv", merged)

    # Numeric correlations against key Phase A metrics.
    num_cols: List[str] = []
    for k in merged[0].keys():
        if not k.startswith("phaseb_"):
            continue
        vals = []
        for r in merged:
            try:
                vals.append(float(r[k]))
            except Exception:
                pass
        if len(vals) >= 3:
            num_cols.append(k)

    corr: Dict[str, Any] = {"n_merged_cases": len(merged), "correlations": {}}
    for key_a in ("phasea_dice", "phasea_cldice", "phasea_hd95"):
        x = []
        for r in merged:
            try:
                x.append(float(r[key_a]))
            except Exception:
                x.append(float("nan"))
        x = np.asarray(x, dtype=float)
        corr["correlations"][key_a] = {}
        for key_b in num_cols:
            y = []
            for r in merged:
                try:
                    y.append(float(r[key_b]))
                except Exception:
                    y.append(float("nan"))
            y = np.asarray(y, dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) < 3:
                corr["correlations"][key_a][key_b] = float("nan")
            else:
                c = float(np.corrcoef(x[mask], y[mask])[0, 1])
                corr["correlations"][key_a][key_b] = c

    save_json(out_dir / "phase_a_phase_b_correlation.json", corr)

    if plt is None:
        return
    # One grouped plot: Phase A Dice by Phase B pass/fail if available.
    pass_key = None
    for k in ("phaseb_pass", "phaseb_success", "phaseb_mesh_watertight"):
        if k in merged[0]:
            pass_key = k
            break
    if pass_key is None:
        return
    grp_true = []
    grp_false = []
    for r in merged:
        val = str(r.get(pass_key, "")).strip().lower()
        try:
            dice = float(r["phasea_dice"])
        except Exception:
            continue
        if val in {"true", "1", "yes", "pass"}:
            grp_true.append(dice)
        elif val in {"false", "0", "no", "fail"}:
            grp_false.append(dice)
    if not grp_true and not grp_false:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    data = []
    labels = []
    if grp_false:
        data.append(np.asarray(grp_false))
        labels.append("PhaseB Fail/False")
    if grp_true:
        data.append(np.asarray(grp_true))
        labels.append("PhaseB Pass/True")
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("Phase A Dice")
    ax.set_title("Phase A Dice grouped by Phase B outcome")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_a_phase_b_grouped.png", dpi=180)
    plt.close(fig)


def save_markdown_report(
    out_path: Path,
    selection: Dict[str, Any],
    summary_rows: Sequence[Dict[str, Any]],
    n_test_cases: int,
    n_skipped: int,
) -> None:
    summary_map = {r["metric"]: r for r in summary_rows if "metric" in r}
    def line(metric: str) -> str:
        r = summary_map.get(metric, {})
        if not r:
            return f"- {metric}: n/a"
        return (
            f"- {metric}: mean={r.get('mean'):.4f}, std={r.get('std'):.4f}, "
            f"median={r.get('median'):.4f}, IQR={r.get('iqr'):.4f}, "
            f"95% CI [{r.get('ci95_low'):.4f}, {r.get('ci95_high'):.4f}]"
        )

    lines = [
        "# Phase A Evaluation Results",
        "",
        "## Threshold Selection",
        f"- Source: {selection.get('selection_source')}",
        f"- Primary metric: {selection.get('primary_metric')}",
        f"- Selected threshold: {selection.get('selected_threshold')}",
        f"- Validation cases used: {selection.get('validation_case_count')}",
        "",
        "## Test Set Summary",
        f"- Evaluated test cases: {n_test_cases}",
        f"- Skipped cases: {n_skipped}",
        line("dice"),
        line("cldice"),
        line("hd95"),
        line("precision"),
        line("recall"),
        line("iou"),
        "",
        "_Note: Failure mode labels are heuristics and should be interpreted with qualitative inspection._",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def metric_conventions() -> Dict[str, Any]:
    return {
        "gt_binarization": "gt_bin = (gt > 0)",
        "pred_binarization_binary": "pred_bin = (pred > 0)",
        "pred_binarization_prob": "pred_bin = (pred >= threshold)",
        "pred_binarization_logit": "pred_bin = (sigmoid(pred) >= threshold)",
        "empty_mask_conventions": {
            "gt_empty_pred_empty": {
                "dice": 1.0,
                "iou": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "specificity": 1.0,
                "hd95": 0.0,
            },
            "gt_empty_pred_nonempty": {
                "dice": 0.0,
                "iou": 0.0,
                "precision": 0.0,
                "recall": 1.0,
                "specificity": 0.0,
                "hd95": "inf",
            },
            "gt_nonempty_pred_empty": {
                "dice": 0.0,
                "iou": 0.0,
                "precision": 1.0,
                "recall": 0.0,
                "specificity": 1.0,
                "hd95": "inf",
            },
        },
    }


def run_invariant_checks(per_case_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {"warnings": [], "passed": True}
    if not per_case_rows:
        checks["warnings"].append("No per-case rows available for invariant checks.")
        checks["passed"] = False
        return checks

    dice = np.asarray([float(r.get("dice", np.nan)) for r in per_case_rows], dtype=float)
    recall = np.asarray([float(r.get("recall", np.nan)) for r in per_case_rows], dtype=float)
    specificity = np.asarray([float(r.get("specificity", np.nan)) for r in per_case_rows], dtype=float)
    pred_fg = np.asarray([float(r.get("pred_fg_voxels", np.nan)) for r in per_case_rows], dtype=float)
    gt_fg = np.asarray([float(r.get("gt_fg_voxels", np.nan)) for r in per_case_rows], dtype=float)
    gt_empty = np.asarray([int(r.get("gt_empty", 0)) for r in per_case_rows], dtype=int)
    pred_empty = np.asarray([int(r.get("pred_empty", 0)) for r in per_case_rows], dtype=int)

    nonempty_both = (gt_fg > 0) & (pred_fg > 0) & np.isfinite(dice)
    tiny_dice_rate = float(np.mean(dice[nonempty_both] < 1e-6)) if int(nonempty_both.sum()) > 0 else float("nan")
    recall_one_rate = float(np.mean(np.isclose(recall, 1.0, atol=1e-8))) if recall.size else float("nan")
    gt_empty_rate = float(np.mean(gt_empty)) if gt_empty.size else float("nan")
    pred_foreground_fraction = np.asarray(
        [float(r.get("pred_fg_voxels", np.nan)) / max(float(r.get("pred_fg_voxels", 0.0)) + float(r.get("tn", 0.0)) + float(r.get("fn", 0.0)), 1.0) for r in per_case_rows],
        dtype=float,
    )
    pred_fg_frac_mean = float(np.nanmean(pred_foreground_fraction)) if pred_foreground_fraction.size else float("nan")

    checks.update(
        {
            "n_cases": len(per_case_rows),
            "n_nonempty_both": int(nonempty_both.sum()),
            "tiny_dice_rate_nonempty_both": tiny_dice_rate,
            "recall_one_rate": recall_one_rate,
            "gt_empty_rate": gt_empty_rate,
            "pred_empty_rate": float(np.mean(pred_empty)) if pred_empty.size else float("nan"),
            "specificity_mean": float(np.nanmean(specificity)) if specificity.size else float("nan"),
            "pred_foreground_fraction_mean": pred_fg_frac_mean,
        }
    )

    if int(nonempty_both.sum()) >= 10 and np.isfinite(tiny_dice_rate) and tiny_dice_rate > 0.95:
        checks["warnings"].append(
            "Dice is ~0 for >95% of cases even when both GT and prediction are non-empty. "
            "This strongly suggests thresholding/type mismatch or ID misalignment."
        )
    if np.isfinite(recall_one_rate) and recall_one_rate > 0.95 and np.isfinite(gt_empty_rate) and gt_empty_rate < 0.30:
        checks["warnings"].append(
            "Recall is 1.0 for almost all cases while GT empty rate is low; this pattern is suspicious."
        )
    spec_mean = checks.get("specificity_mean")
    if np.isfinite(spec_mean) and 0.30 <= float(spec_mean) <= 0.55:
        checks["warnings"].append(
            "Specificity is around 0.4; verify whether predictions are near full-foreground."
        )
    if np.isfinite(pred_fg_frac_mean) and pred_fg_frac_mean > 0.50:
        checks["warnings"].append(
            f"Predicted foreground fraction is high on average ({pred_fg_frac_mean:.3f})."
        )

    checks["passed"] = len(checks["warnings"]) == 0
    return checks


def save_verification_report(
    out_path: Path,
    sanity_report: Dict[str, Any],
    threshold_selection: Dict[str, Any],
    summary_rows: Sequence[Dict[str, Any]],
    skipped_rows: Sequence[Dict[str, Any]],
    shape_spacing_rows: Sequence[Dict[str, Any]],
    invariant_checks: Dict[str, Any],
) -> None:
    metrics = {str(r.get("metric")): r for r in summary_rows}

    def metric_line(name: str) -> str:
        r = metrics.get(name, {})
        if not r:
            return f"- {name}: n/a"
        return (
            f"- {name}: mean={r.get('mean')}, median={r.get('median')}, iqr={r.get('iqr')}, "
            f"ci95=[{r.get('ci95_low')}, {r.get('ci95_high')}], n={r.get('n')}"
        )

    id_rep = sanity_report.get("id_matching_report", {})
    gt_rep = sanity_report.get("gt_sanity_summary", {})
    pred_rep = sanity_report.get("pred_type_detection", {})
    hard_fail = sanity_report.get("hard_fail_reasons", [])
    warnings_list = invariant_checks.get("warnings", [])

    lines = [
        "# Phase A Verification Report",
        "",
        "## Input Verification",
        f"- GT files: {id_rep.get('gt_file_count')}",
        f"- Pred files: {id_rep.get('pred_file_count')}",
        f"- GT unique IDs: {id_rep.get('gt_unique_ids')}",
        f"- Pred unique IDs: {id_rep.get('pred_unique_ids')}",
        f"- Matched ID intersection: {id_rep.get('intersection_count')}",
        f"- Missing in GT: {id_rep.get('missing_in_gt_count')}",
        f"- Missing in Pred: {id_rep.get('missing_in_pred_count')}",
        f"- Duplicate pred IDs: {id_rep.get('duplicate_pred_id_count')}",
        f"- Duplicate GT IDs: {id_rep.get('duplicate_gt_id_count')}",
        "",
        "## GT/Prediction Sanity",
        f"- GT sampled cases: {gt_rep.get('sampled_cases')}",
        f"- GT empty fraction (sampled): {gt_rep.get('gt_empty_fraction')}",
        f"- Pred type selected: {pred_rep.get('selected_type_for_binarization')}",
        f"- Pred type auto choice: {pred_rep.get('auto_selected_type')}",
        f"- Pred type confidence: {pred_rep.get('confidence')}",
        "",
        "## Thresholding",
        f"- Selection source: {threshold_selection.get('selection_source')}",
        f"- Primary metric: {threshold_selection.get('primary_metric')}",
        f"- Selected threshold: {threshold_selection.get('selected_threshold')}",
        f"- Validation case count: {threshold_selection.get('validation_case_count')}",
        "",
        "## Test Metrics Summary",
        metric_line("dice"),
        metric_line("cldice"),
        metric_line("hd95"),
        metric_line("precision"),
        metric_line("recall"),
        metric_line("specificity"),
        metric_line("iou"),
        "",
        "## Quality/Failure Diagnostics",
        f"- Skipped cases: {len(skipped_rows)}",
        f"- Shape/spacing issues: {len(shape_spacing_rows)}",
        f"- Invariant checks passed: {invariant_checks.get('passed')}",
    ]
    if warnings_list:
        lines.append("- Invariant warnings:")
        for w in warnings_list:
            lines.append(f"  - {w}")
    if hard_fail:
        lines.append("- Preflight hard-fail reasons encountered:")
        for r in hard_fail:
            lines.append(f"  - {r}")
    lines.extend(
        [
            "",
            "## Notes",
            "- Threshold selection was performed on validation IDs only, then locked for test evaluation.",
            "- Empty-mask metric conventions are documented in `metric_conventions.json`.",
            "- Failure-mode labels are heuristic and must be interpreted with qualitative inspection.",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Professional Phase A evaluation/results suite")
    parser.add_argument("--pred_dir", default=None, help="Directory of prediction files (auto-detect if omitted)")
    parser.add_argument("--gt_dir", default=None, help="Ground-truth masks directory")
    parser.add_argument("--image_dir", default=None, help="Optional source image directory for qualitative panels")
    parser.add_argument("--split_xlsx", default="imageCAS_data_split.xlsx", help="Path to split Excel")
    parser.add_argument("--splits_json", default="splits.json", help="Path to split JSON")
    parser.add_argument("--val_ids_file", default=None, help="Optional explicit validation IDs file")
    parser.add_argument("--test_ids_file", default=None, help="Optional explicit test IDs file")
    parser.add_argument("--id_key", default=None, help="Split column/key override (e.g., Split-1)")
    parser.add_argument("--output_dir", default=None, help="Output root (default: eval_outputs/verification_runs/<run_name>)")
    parser.add_argument("--run_name", default=None, help="Run name suffix for default output directory")
    parser.add_argument("--pred_type", default="auto", choices=["prob", "logit", "binary", "auto"], help="Prediction representation")
    parser.add_argument("--sweep_thresholds", default="0.05:0.95:0.05", help="Threshold sweep list or range")
    parser.add_argument("--primary_threshold_metric", default="cldice", choices=["cldice", "dice", "hd95", "assd"], help="Threshold selection objective")
    parser.add_argument("--postprocess", default="on", choices=["on", "off"], help="Enable/disable postprocessing")
    parser.add_argument("--min_component_voxels", type=int, default=50, help="Remove components smaller than this voxel count")
    parser.add_argument("--run_assd", action="store_true", help="Compute ASSD when possible")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Bootstrap iterations for CI")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--phase_b_qc_csv", default=None, help="Optional Phase B QC CSV for correlation analysis")
    parser.add_argument("--num_qual_cases", type=int, default=3, help="Number of representative qualitative cases")
    parser.add_argument("--save_nifti_errors", action="store_true", help="Save NIfTI error volumes for worst cases")
    parser.add_argument("--top_k_worst", type=int, default=5, help="Number of worst cases for failure examples")
    parser.add_argument("--top_k_best", type=int, default=5, help="Number of best cases for quick reference")
    parser.add_argument("--export_markdown_report", action="store_true", help="Export markdown results report")
    parser.add_argument("--run_inference", action="store_true", help="Optional inference mode (adapter/TODO)")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_best.pt", help="Checkpoint path for optional inference adapter")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "mps", "auto"], help="Device for optional inference adapter")
    parser.add_argument("--strict_device", action="store_true", help="Fail if requested device is unavailable")
    parser.add_argument("--allow_empty_eval", action="store_true", help="Allow writing outputs even when zero val/test cases are matched")
    parser.add_argument("--sanity_sample_n", type=int, default=50, help="Number of matched cases to sample for preflight sanity checks")
    parser.add_argument(
        "--allow_many_empty_gt",
        action="store_true",
        help="Allow >30%% empty GT masks in sanity sample without hard failure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    suppress_noisy_warnings()
    set_reproducible_seeds(args.random_seed)

    repo_utils = optional_import_repo_eval_utils(logger)
    selected_device = resolve_device(args.device, args.strict_device, logger, repo_utils)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        run_name = args.run_name or f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_phase_a_verification"
        output_dir = Path("eval_outputs") / "verification_runs" / run_name
        output_dir = output_dir.resolve()
    plots_dir = output_dir / "plots"
    qual_dir = output_dir / "qualitative"
    failure_dir = output_dir / "failure_examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    qual_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    env_info = environment_info(selected_device, repo_utils)
    save_json(output_dir / "environment_info.json", env_info)
    for w in env_info.get("dependency_warnings", []):
        logger.warning(w)

    # Discover data root early so preflight sanity checks and evaluation use the same GT base.
    data_root = autodetect_data_root(args.gt_dir, logger, repo_utils)

    sanity_report: Dict[str, Any] = {}
    if callable(run_sanity_checks):
        try:
            sanity_report = run_sanity_checks(
                pred_dir=args.pred_dir,
                gt_dir=(args.gt_dir if args.gt_dir else (str(data_root) if data_root else None)),
                image_dir=args.image_dir,
                output_dir=output_dir,
                pred_type=args.pred_type,
                sample_n=max(1, int(args.sanity_sample_n)),
                random_seed=int(args.random_seed),
                logger=logger,
                allow_many_empty_gt=bool(args.allow_many_empty_gt),
            )
        except Exception as e:
            logger.error(f"Preflight sanity checks failed: {e}")
            raise
    else:
        logger.warning("sanity_check_eval_inputs.py could not be imported; proceeding without preflight module.")

    save_json(output_dir / "metric_conventions.json", metric_conventions())

    apply_post = (args.postprocess.lower() == "on")
    if args.run_inference:
        logger.warning(
            "--run_inference requested. Adapter is intentionally conservative in this repo layout; "
            "using existing predictions for evaluation. TODO: integrate a robust inference runner per dataset manifest."
        )
        logger.warning(f"Checkpoint requested for adapter: {args.checkpoint} | device={selected_device}")

    # Build lookup index for optional path remapping.
    data_index = build_data_index(data_root, logger, repo_utils)

    # Split sources
    split_json_path = Path(args.splits_json).resolve()
    split_xlsx_path = Path(args.split_xlsx).resolve()
    json_records = parse_splits_json_records(split_json_path, logger, data_index, repo_utils)
    xlsx_ids = parse_split_xlsx_ids(split_xlsx_path, args.id_key, logger) if split_xlsx_path.exists() else {"val": [], "test": []}

    explicit_val_ids = read_ids_file(Path(args.val_ids_file).resolve()) if args.val_ids_file else []
    explicit_test_ids = read_ids_file(Path(args.test_ids_file).resolve()) if args.test_ids_file else []

    # Precedence: explicit files > splits.json > xlsx.
    val_ids: List[str]
    test_ids: List[str]
    split_source = {"val": None, "test": None}

    if explicit_val_ids:
        val_ids = sorted(set(explicit_val_ids))
        split_source["val"] = "val_ids_file"
    elif json_records["val"]:
        val_ids = sorted(json_records["val"].keys())
        split_source["val"] = "splits_json"
    else:
        val_ids = sorted(set(xlsx_ids.get("val", [])))
        split_source["val"] = "split_xlsx"

    if explicit_test_ids:
        test_ids = sorted(set(explicit_test_ids))
        split_source["test"] = "test_ids_file"
    elif json_records["test"]:
        test_ids = sorted(json_records["test"].keys())
        split_source["test"] = "splits_json"
    else:
        test_ids = sorted(set(xlsx_ids.get("test", [])))
        split_source["test"] = "split_xlsx"

    logger.info(f"Validation IDs: {len(val_ids)} | Test IDs: {len(test_ids)}")
    logger.info(f"Split source => val={split_source['val']} | test={split_source['test']}")

    pred_files = discover_prediction_files(args.pred_dir, logger)
    if not pred_files and not args.allow_empty_eval:
        hint = ""
        if args.pred_dir and args.pred_dir.startswith("/REAL/"):
            hint = " You used a placeholder path. Replace --pred_dir with your actual predictions directory."
        raise RuntimeError(
            f"No prediction files were found in --pred_dir={args.pred_dir!r}.{hint}"
        )
    pred_idx = build_id_to_paths(pred_files)

    gt_files: List[Path] = []
    if args.gt_dir:
        gt_files = scan_files(Path(args.gt_dir).expanduser().resolve())
    elif data_root is not None:
        gt_files = scan_files(data_root)
    gt_idx = build_id_to_paths(gt_files)

    image_files: List[Path] = []
    if args.image_dir:
        image_files = scan_files(Path(args.image_dir).expanduser().resolve())
    elif data_root is not None:
        image_files = scan_files(data_root)
    image_idx = build_id_to_paths(image_files)

    skipped_rows: List[Dict[str, Any]] = []
    shape_spacing_rows: List[Dict[str, Any]] = []

    def select_gt_for_case(cid: str, split_rec: Dict[str, Dict[str, Optional[str]]]) -> Optional[Path]:
        rec = split_rec.get(cid, {})
        lbl = rec.get("label")
        if lbl and Path(lbl).exists():
            return Path(lbl)
        gt_candidates = gt_idx.get(cid, [])
        gt_choice = choose_path(gt_candidates, role="gt", pred_type=args.pred_type)
        if gt_choice:
            return gt_choice
        _, remap_lbl = remap_case_paths_by_id(cid, data_index, repo_utils)
        if remap_lbl and Path(remap_lbl).exists():
            return Path(remap_lbl)
        return None

    def select_img_for_case(cid: str, split_rec: Dict[str, Dict[str, Optional[str]]]) -> Optional[Path]:
        rec = split_rec.get(cid, {})
        img = rec.get("image")
        if img and Path(img).exists():
            return Path(img)
        img_candidates = image_idx.get(cid, [])
        return choose_path(img_candidates, role="image", pred_type=args.pred_type)

    def build_cases(ids: Sequence[str], split_name: str) -> List[CaseEntry]:
        cases: List[CaseEntry] = []
        rec_map = json_records.get(split_name, {})
        for cid in ids:
            pred_candidates = pred_idx.get(cid, [])
            pred_path = choose_path(pred_candidates, role="pred", pred_type=args.pred_type)
            gt_path = select_gt_for_case(cid, rec_map)
            image_path = select_img_for_case(cid, rec_map)
            if pred_path is None:
                skipped_rows.append(
                    {"split": split_name, "case_id": cid, "reason": "missing_prediction", "pred_candidates": len(pred_candidates), "gt_path": str(gt_path) if gt_path else ""}
                )
                continue
            if gt_path is None:
                skipped_rows.append(
                    {"split": split_name, "case_id": cid, "reason": "missing_ground_truth", "pred_path": str(pred_path), "pred_candidates": len(pred_candidates)}
                )
                continue
            cases.append(CaseEntry(case_id=cid, split=split_name, pred_path=pred_path, gt_path=gt_path, image_path=image_path))
        return cases

    val_cases = build_cases(val_ids, "val")
    test_cases = build_cases(test_ids, "test")
    logger.info(f"Matched cases => val={len(val_cases)}, test={len(test_cases)}")
    if not args.allow_empty_eval:
        if len(val_cases) == 0:
            write_rows_csv(output_dir / "skipped_cases.csv", skipped_rows)
            raise RuntimeError(
                "Matched 0 validation cases. Cannot perform validation-only threshold selection. "
                "Provide validation predictions in --pred_dir."
            )
        if len(test_cases) == 0:
            write_rows_csv(output_dir / "skipped_cases.csv", skipped_rows)
            raise RuntimeError(
                "Matched 0 test cases. Check prediction case IDs/naming in --pred_dir."
            )

    thresholds = parse_thresholds(args.sweep_thresholds)
    primary_metric = args.primary_threshold_metric.lower()
    if primary_metric == "cldice" and skeletonize_3d is None:
        logger.warning("clDice skeleton dependency unavailable (skimage). Falling back threshold objective to dice.")
        primary_metric = "dice"

    sweep_rows, selection = choose_threshold_from_validation(
        val_cases=val_cases,
        thresholds=thresholds,
        pred_type=args.pred_type,
        primary_metric=primary_metric,
        apply_postprocess=apply_post,
        min_component_voxels=args.min_component_voxels,
        logger=logger,
    )
    selected_threshold = float(selection.get("selected_threshold", 0.5))
    write_rows_csv(output_dir / "threshold_sweep_validation.csv", sweep_rows)
    save_json(output_dir / "threshold_selection.json", selection)
    plot_threshold_sweep(sweep_rows, plots_dir / "threshold_sweep.png", selected_threshold)

    per_case_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []

    eval_start = time.perf_counter()
    for case in test_cases:
        t0 = time.perf_counter()
        load_t = 0.0
        proc_t = 0.0
        metric_t = 0.0
        inferred_pred_type = "unknown"
        try:
            t_load = time.perf_counter()
            pred_vol = load_volume(case.pred_path)
            gt_vol = load_volume(case.gt_path)
            load_t = time.perf_counter() - t_load

            shape_fixed = False
            pred_arr_aligned, gt_arr_aligned, shape_fixed = align_shapes_if_safe(pred_vol.array, gt_vol.array)
            if pred_arr_aligned.shape != gt_arr_aligned.shape:
                shape_spacing_rows.append(
                    {
                        "case_id": case.case_id,
                        "issue": "shape_mismatch",
                        "pred_shape": str(tuple(int(x) for x in pred_vol.array.shape)),
                        "gt_shape": str(tuple(int(x) for x in gt_vol.array.shape)),
                        "pred_path": str(case.pred_path),
                        "gt_path": str(case.gt_path),
                    }
                )
                skipped_rows.append(
                    {
                        "split": "test",
                        "case_id": case.case_id,
                        "reason": "shape_mismatch",
                        "pred_shape": str(tuple(int(x) for x in pred_vol.array.shape)),
                        "gt_shape": str(tuple(int(x) for x in gt_vol.array.shape)),
                        "pred_path": str(case.pred_path),
                        "gt_path": str(case.gt_path),
                    }
                )
                continue

            if shape_fixed:
                shape_spacing_rows.append(
                    {
                        "case_id": case.case_id,
                        "issue": "shape_fixed_by_squeeze_singletons",
                        "pred_shape": str(tuple(int(x) for x in pred_vol.array.shape)),
                        "gt_shape": str(tuple(int(x) for x in gt_vol.array.shape)),
                        "fixed_shape": str(tuple(int(x) for x in pred_arr_aligned.shape)),
                        "pred_path": str(case.pred_path),
                        "gt_path": str(case.gt_path),
                    }
                )

            pred_vol = LoadedVolume(array=pred_arr_aligned, spacing=pred_vol.spacing, affine=pred_vol.affine, path=pred_vol.path)
            gt_vol = LoadedVolume(array=gt_arr_aligned, spacing=gt_vol.spacing, affine=gt_vol.affine, path=gt_vol.path)

            if pred_vol.spacing is not None and gt_vol.spacing is not None:
                ps = np.asarray(pred_vol.spacing, dtype=float)
                gs = np.asarray(gt_vol.spacing, dtype=float)
                if ps.shape == gs.shape and not np.allclose(ps, gs, atol=1e-3):
                    shape_spacing_rows.append(
                        {
                            "case_id": case.case_id,
                            "issue": "spacing_mismatch",
                            "pred_spacing": format_spacing(pred_vol.spacing),
                            "gt_spacing": format_spacing(gt_vol.spacing),
                            "pred_path": str(case.pred_path),
                            "gt_path": str(case.gt_path),
                        }
                    )

            t_proc = time.perf_counter()
            metrics, inferred_pred_type = evaluate_loaded_case(
                pred_vol=pred_vol,
                gt_vol=gt_vol,
                threshold=selected_threshold,
                pred_type_setting=args.pred_type,
                apply_postprocess=apply_post,
                min_component_voxels=args.min_component_voxels,
                run_assd=args.run_assd,
            )
            proc_t = time.perf_counter() - t_proc

            t_met = time.perf_counter()
            # metrics already computed above; retain a dedicated "metric time" bucket for reporting consistency.
            metric_t = time.perf_counter() - t_met

            row = {
                "case_id": case.case_id,
                "split": case.split,
                "pred_path": str(case.pred_path),
                "gt_path": str(case.gt_path),
                "image_path": str(case.image_path) if case.image_path else "",
                "pred_type_inferred": inferred_pred_type,
                "threshold_used": metrics.get("threshold_used"),
                "spacing": format_spacing(gt_vol.spacing if gt_vol.spacing is not None else pred_vol.spacing),
            }
            row.update(metrics)
            per_case_rows.append(row)
        except Exception as e:
            skipped_rows.append(
                {
                    "split": "test",
                    "case_id": case.case_id,
                    "reason": f"evaluation_error: {e}",
                    "pred_path": str(case.pred_path),
                    "gt_path": str(case.gt_path),
                }
            )
            continue
        finally:
            total_t = time.perf_counter() - t0
            runtime_rows.append(
                {
                    "case_id": case.case_id,
                    "split": case.split,
                    "load_time_s": load_t,
                    "threshold_postprocess_time_s": proc_t,
                    "metric_time_s": metric_t,
                    "total_eval_time_s": total_t,
                    "inference_time_s": float("nan"),
                    "device": selected_device,
                }
            )

    total_eval_time = time.perf_counter() - eval_start
    logger.info(f"Completed test evaluation for {len(per_case_rows)} cases in {total_eval_time:.2f}s")
    if len(per_case_rows) == 0 and not args.allow_empty_eval:
        write_rows_csv(output_dir / "skipped_cases.csv", skipped_rows)
        raise RuntimeError(
            "0 test cases were successfully evaluated. See skipped_cases.csv for details (likely shape mismatch or invalid predictions)."
        )

    write_rows_csv(output_dir / "test_metrics_per_case.csv", per_case_rows)
    write_rows_csv(output_dir / "runtime_per_case.csv", runtime_rows)
    write_rows_csv(output_dir / "skipped_cases.csv", skipped_rows)
    write_rows_csv(output_dir / "shape_spacing_issues.csv", shape_spacing_rows)

    summary_rows: List[Dict[str, Any]] = []
    summary_json: Dict[str, Any] = {"n_test_cases": len(per_case_rows), "n_skipped_cases": len(skipped_rows), "metrics": {}}
    for metric in pick_summary_metrics():
        vals = [float(r.get(metric, float("nan"))) for r in per_case_rows]
        stats = summarize_metric(vals, n_bootstrap=args.n_bootstrap, seed=args.random_seed)
        row = {"metric": metric}
        row.update(stats)
        summary_rows.append(row)
        summary_json["metrics"][metric] = stats
    write_rows_csv(output_dir / "test_metrics_summary.csv", summary_rows)
    save_json(output_dir / "test_metrics_summary.json", summary_json)

    plot_core_metrics(per_case_rows, plots_dir)

    flags_rows, flags_summary = build_failure_flags(per_case_rows)
    write_rows_csv(output_dir / "failure_mode_flags.csv", flags_rows)
    save_json(output_dir / "failure_mode_summary.json", flags_summary)

    invariant_checks = run_invariant_checks(per_case_rows)
    save_json(output_dir / "invariant_checks.json", invariant_checks)
    for w in invariant_checks.get("warnings", []):
        logger.warning(f"[invariant] {w}")

    # Qualitative selection: best / typical / hard + failure_examples.
    qual_manifest: List[Dict[str, Any]] = []
    if per_case_rows:
        rows_sorted = sorted(per_case_rows, key=lambda r: float(r.get("dice", -1.0)))
        worst = rows_sorted[0]
        best = rows_sorted[-1]
        med_dice = float(median([float(r.get("dice", float("nan"))) for r in per_case_rows if np.isfinite(r.get("dice", np.nan))])) if per_case_rows else float("nan")
        typical = min(per_case_rows, key=lambda r: abs(float(r.get("dice", float("nan"))) - med_dice)) if np.isfinite(med_dice) else rows_sorted[len(rows_sorted) // 2]
        representative = [("hard", worst), ("typical", typical), ("best", best)]

        # honor num_qual_cases by truncating/expanding with quantile picks
        if args.num_qual_cases > 3:
            n_extra = args.num_qual_cases - 3
            idxs = np.linspace(0, len(rows_sorted) - 1, n_extra + 2, dtype=int)[1:-1]
            for i, idx in enumerate(idxs):
                representative.append((f"extra_{i+1}", rows_sorted[int(idx)]))
        representative = representative[: max(1, args.num_qual_cases)]

        by_id = {r["case_id"]: r for r in per_case_rows}
        seen_case_ids: set[str] = set()
        for category, row in representative:
            cid = str(row["case_id"])
            if cid in seen_case_ids:
                continue
            seen_case_ids.add(cid)
            case = CaseEntry(
                case_id=cid,
                split="test",
                pred_path=Path(row["pred_path"]),
                gt_path=Path(row["gt_path"]),
                image_path=Path(row["image_path"]) if row.get("image_path") else None,
            )
            try:
                pred_vol = load_volume(case.pred_path)
                gt_vol = load_volume(case.gt_path)
                pred_bin, _, _ = prediction_to_binary(pred_vol.array, args.pred_type, selected_threshold)
                if apply_post:
                    pred_bin = postprocess_binary_mask(pred_bin, args.min_component_voxels, True)
                gt_bin = (gt_vol.array > 0).astype(np.uint8)
                if pred_bin.shape != gt_bin.shape:
                    continue
                img_vol = load_optional_image(case.image_path, gt_bin.shape)
                out_path = qual_dir / f"{category}_{cid}.png"
                save_qualitative_panel(
                    case_id=cid,
                    category=category,
                    pred_bin=pred_bin,
                    gt_bin=gt_bin,
                    image_vol=img_vol,
                    out_path=out_path,
                    title_suffix=f"Dice={float(row.get('dice', np.nan)):.4f}",
                )
                qual_manifest.append(
                    {
                        "case_id": cid,
                        "category": category,
                        "dice": row.get("dice"),
                        "cldice": row.get("cldice"),
                        "hd95": row.get("hd95"),
                        "panel_path": str(out_path),
                    }
                )
            except Exception as e:
                skipped_rows.append({"split": "test", "case_id": cid, "reason": f"qualitative_error: {e}"})

        # failure_examples for worst-K
        worst_k = rows_sorted[: max(0, args.top_k_worst)]
        for rank, row in enumerate(worst_k, start=1):
            cid = str(row["case_id"])
            case = CaseEntry(
                case_id=cid,
                split="test",
                pred_path=Path(row["pred_path"]),
                gt_path=Path(row["gt_path"]),
                image_path=Path(row["image_path"]) if row.get("image_path") else None,
            )
            try:
                pred_vol = load_volume(case.pred_path)
                gt_vol = load_volume(case.gt_path)
                pred_bin, _, _ = prediction_to_binary(pred_vol.array, args.pred_type, selected_threshold)
                if apply_post:
                    pred_bin = postprocess_binary_mask(pred_bin, args.min_component_voxels, True)
                gt_bin = (gt_vol.array > 0).astype(np.uint8)
                if pred_bin.shape != gt_bin.shape:
                    continue
                img_vol = load_optional_image(case.image_path, gt_bin.shape)
                out_path = failure_dir / f"worst_{rank:02d}_{cid}.png"
                save_qualitative_panel(
                    case_id=cid,
                    category=f"worst_{rank}",
                    pred_bin=pred_bin,
                    gt_bin=gt_bin,
                    image_vol=img_vol,
                    out_path=out_path,
                    title_suffix=f"Dice={float(row.get('dice', np.nan)):.4f}",
                )
                if args.save_nifti_errors and nib is not None:
                    err = build_error_map(pred_bin, gt_bin)
                    save_error_nifti(failure_dir / f"worst_{rank:02d}_{cid}_error.nii.gz", err, gt_vol)
            except Exception as e:
                skipped_rows.append({"split": "test", "case_id": cid, "reason": f"failure_example_error: {e}"})

        # best-K examples (useful for paper appendix/qualitative sanity checks)
        best_k = list(reversed(rows_sorted[-max(0, args.top_k_best):]))
        for rank, row in enumerate(best_k, start=1):
            cid = str(row["case_id"])
            case = CaseEntry(
                case_id=cid,
                split="test",
                pred_path=Path(row["pred_path"]),
                gt_path=Path(row["gt_path"]),
                image_path=Path(row["image_path"]) if row.get("image_path") else None,
            )
            try:
                pred_vol = load_volume(case.pred_path)
                gt_vol = load_volume(case.gt_path)
                pred_bin, _, _ = prediction_to_binary(pred_vol.array, args.pred_type, selected_threshold)
                if apply_post:
                    pred_bin = postprocess_binary_mask(pred_bin, args.min_component_voxels, True)
                gt_bin = (gt_vol.array > 0).astype(np.uint8)
                if pred_bin.shape != gt_bin.shape:
                    continue
                img_vol = load_optional_image(case.image_path, gt_bin.shape)
                out_path = qual_dir / f"best_{rank:02d}_{cid}.png"
                save_qualitative_panel(
                    case_id=cid,
                    category=f"best_{rank}",
                    pred_bin=pred_bin,
                    gt_bin=gt_bin,
                    image_vol=img_vol,
                    out_path=out_path,
                    title_suffix=f"Dice={float(row.get('dice', np.nan)):.4f}",
                )
                qual_manifest.append(
                    {
                        "case_id": cid,
                        "category": f"best_{rank}",
                        "dice": row.get("dice"),
                        "cldice": row.get("cldice"),
                        "hd95": row.get("hd95"),
                        "panel_path": str(out_path),
                    }
                )
            except Exception as e:
                skipped_rows.append({"split": "test", "case_id": cid, "reason": f"best_example_error: {e}"})

    write_rows_csv(output_dir / "qualitative_manifest.csv", qual_manifest)

    maybe_phaseb_correlation(per_case_rows, args.phase_b_qc_csv, output_dir)

    run_config = {
        "args": vars(args),
        "selected_device": selected_device,
        "split_source": split_source,
        "preflight_sanity_ok": sanity_report.get("ok") if sanity_report else None,
        "preflight_intersection_count": sanity_report.get("id_matching_report", {}).get("intersection_count") if sanity_report else None,
        "counts": {
            "val_ids": len(val_ids),
            "test_ids": len(test_ids),
            "val_matched_cases": len(val_cases),
            "test_matched_cases": len(test_cases),
            "test_evaluated_cases": len(per_case_rows),
            "skipped_cases": len(skipped_rows),
            "shape_spacing_issues": len(shape_spacing_rows),
        },
        "selected_threshold": selected_threshold,
        "threshold_selection": selection,
        "postprocess_applied": apply_post,
        "postprocess_details": {
            "remove_small_components": apply_post,
            "min_component_voxels": args.min_component_voxels,
            "hole_fill": apply_post,
        },
        "total_eval_time_s": total_eval_time,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(output_dir / "run_config.json", run_config)

    save_verification_report(
        output_dir / "verification_report.md",
        sanity_report=sanity_report,
        threshold_selection=selection,
        summary_rows=summary_rows,
        skipped_rows=skipped_rows,
        shape_spacing_rows=shape_spacing_rows,
        invariant_checks=invariant_checks,
    )

    if args.export_markdown_report:
        save_markdown_report(
            output_dir / "results_report.md",
            selection=selection,
            summary_rows=summary_rows,
            n_test_cases=len(per_case_rows),
            n_skipped=len(skipped_rows),
        )

    # Write a short plain-text README for reproducibility.
    readme_txt = [
        "Phase A results suite outputs",
        f"Run time: {datetime.now().isoformat()}",
        f"Selected threshold: {selected_threshold}",
        f"Validation threshold metric: {selection.get('primary_metric')}",
        f"Evaluated test cases: {len(per_case_rows)}",
        f"Skipped cases: {len(skipped_rows)}",
        "Important: threshold selection used validation IDs only.",
    ]
    (output_dir / "README_results.txt").write_text("\n".join(readme_txt))

    # Refresh skipped CSV after qualitative/failure generation logs.
    write_rows_csv(output_dir / "skipped_cases.csv", skipped_rows)
    logger.info(f"Results suite complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
