#!/usr/bin/env python3
"""Checkpoint portability, integrity, and selection helpers.

The helpers are intentionally CPU-safe so A40 setup can validate MPS-created
checkpoints before launching expensive GPU training. They do not modify
checkpoints unless explicitly asked to create backups.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

try:
    import torch
except Exception as exc:  # pragma: no cover - import guard for diagnostics
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


class Config:  # noqa: D101 - compatibility shim for checkpoints saved from script __main__
    pass


def _install_checkpoint_compat_globals() -> None:
    """Install pickle globals needed by repo checkpoints saved from scripts."""
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "Config"):
        setattr(main_mod, "Config", Config)


COMMON_STATE_KEYS = ("model_state_dict", "state_dict", "model", "model_state", "net")
COMMON_OPTIMIZER_KEYS = ("optimizer_state_dict", "optimizer")
COMMON_SCHEDULER_KEYS = ("scheduler_state_dict", "scheduler")
COMMON_SCALER_KEYS = ("scaler_state_dict", "scaler")
# Keep MONAI module paths such as "model.0.conv..." intact. "model." can be a
# real state_dict prefix for MONAI networks, not just a wrapper prefix.
WRAPPER_PREFIXES = ("module.", "_orig_mod.", "net.")


def list_checkpoints(root: str | Path = ".") -> List[str]:
    """Return checkpoint-like files under root, excluding virtualenv caches."""
    root = Path(root)
    excluded = {".git", ".venv", "node_modules", ".next", "Data"}
    out: List[str] = []
    for pattern in ("*.pt", "*.pth", "*.ckpt"):
        for path in root.rglob(pattern):
            if any(part in excluded for part in path.parts):
                continue
            if "outputs" in path.parts and "checkpoint_backups" in path.parts:
                continue
            if path.is_file():
                out.append(str(path))
    return sorted(out)


def normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip common DataParallel/compile/wrapper prefixes from state dict keys."""
    normalized: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        # Whole-model gradient checkpointing wraps the MONAI network in a module
        # whose child is also named "model", yielding keys like
        # "model.model.0.conv...". MONAI AttentionUnet itself expects
        # "model.0.conv...", so strip exactly one duplicated "model." prefix.
        if new_key.startswith("model.model."):
            new_key = new_key[len("model.") :]
        changed = True
        while changed:
            changed = False
            for prefix in WRAPPER_PREFIXES:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        normalized[new_key] = value
    return normalized


def safe_torch_load(path: str | Path, map_location: str = "cpu") -> Any:
    """Load a checkpoint with readable diagnostics."""
    if torch is None:
        raise RuntimeError(f"torch import failed: {_TORCH_IMPORT_ERROR}")
    _install_checkpoint_compat_globals()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Could not load checkpoint {path}: {exc}") from exc


def _extract_state_dict(obj: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    if isinstance(obj, Mapping):
        if obj and all(hasattr(v, "shape") for v in obj.values()):
            return normalize_state_dict_keys(obj), "state_dict"
        for key in COMMON_STATE_KEYS:
            value = obj.get(key)
            if isinstance(value, Mapping):
                return normalize_state_dict_keys(value), "checkpoint"
    return None, "unknown"


def _metric_from_checkpoint(obj: Any) -> Optional[float]:
    if not isinstance(obj, Mapping):
        return None
    metric_keys = (
        "best_metric",
        "best_dice05",
        "best_val_dice",
        "val_dice@0.5",
        "dice@0.5",
        "val_soft_dice",
    )
    for key in metric_keys:
        value = obj.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    best_metrics = obj.get("best_metrics")
    if isinstance(best_metrics, Mapping):
        for key in metric_keys:
            value = best_metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _epoch_from_name(path: str | Path) -> Optional[int]:
    match = re.search(r"(?:epoch_|checkpoint_epoch_)(\d+)", str(path))
    return int(match.group(1)) if match else None


def test_checkpoint(path: str | Path) -> Dict[str, Any]:
    """Return a structured integrity report for one checkpoint."""
    path = Path(path)
    report: Dict[str, Any] = {
        "path": str(path),
        "readable": False,
        "type": "unknown",
        "contains_model_keys": False,
        "contains_optimizer": False,
        "contains_scheduler": False,
        "contains_scaler": False,
        "contains_epoch": False,
        "epoch": None,
        "best_metric": None,
        "sample_param_shapes": [],
        "load_error": None,
        "size_bytes": path.stat().st_size if path.exists() else None,
    }
    try:
        obj = safe_torch_load(path, map_location="cpu")
        report["readable"] = True
        state_dict, obj_type = _extract_state_dict(obj)
        report["type"] = obj_type
        report["contains_model_keys"] = bool(state_dict)
        if isinstance(obj, Mapping):
            report["contains_optimizer"] = any(k in obj for k in COMMON_OPTIMIZER_KEYS)
            report["contains_scheduler"] = any(k in obj for k in COMMON_SCHEDULER_KEYS)
            report["contains_scaler"] = any(k in obj for k in COMMON_SCALER_KEYS)
            epoch = obj.get("epoch", _epoch_from_name(path))
            report["contains_epoch"] = epoch is not None
            report["epoch"] = int(epoch) if isinstance(epoch, int) else epoch
            report["best_metric"] = _metric_from_checkpoint(obj)
        else:
            report["epoch"] = _epoch_from_name(path)
            report["contains_epoch"] = report["epoch"] is not None
        if state_dict:
            samples = []
            for key, value in list(state_dict.items())[:5]:
                shape = list(value.shape) if hasattr(value, "shape") else None
                samples.append({"key": key, "shape": shape})
            report["sample_param_shapes"] = samples
    except Exception as exc:
        report["load_error"] = str(exc)
    return report


def safe_backup_checkpoint(path: str | Path, dest_dir: str | Path = "outputs/checkpoint_backups/") -> str:
    """Back up a checkpoint using a hardlink when possible, copy otherwise."""
    path = Path(path)
    dest_dir = Path(dest_dir)
    rel = path if path.is_absolute() else path
    safe_rel = Path(*[part for part in rel.parts if part not in (os.sep, "")])
    dest = dest_dir / safe_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return str(dest)
    try:
        os.link(path, dest)
    except Exception:
        shutil.copy2(path, dest)
    return str(dest)


def load_checkpoint_for_model(
    model: Any,
    path: str | Path,
    strict: bool = False,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load raw or wrapped checkpoint weights into model with diagnostics."""
    obj = safe_torch_load(path, map_location=device)
    state_dict, obj_type = _extract_state_dict(obj)
    if not state_dict:
        raise RuntimeError(f"No model state_dict found in {path}")
    result = model.load_state_dict(state_dict, strict=strict)
    return {
        "path": str(path),
        "type": obj_type,
        "strict": strict,
        "missing_keys": list(getattr(result, "missing_keys", [])),
        "unexpected_keys": list(getattr(result, "unexpected_keys", [])),
        "epoch": obj.get("epoch") if isinstance(obj, Mapping) else None,
        "best_metric": _metric_from_checkpoint(obj),
        "raw_checkpoint": obj,
    }


def select_best_checkpoint(checkpoint_paths: Iterable[str | Path]) -> Tuple[Optional[str], Dict[str, Any]]:
    """Select the most useful resume checkpoint with a documented rationale."""
    reports = [test_checkpoint(path) for path in checkpoint_paths]
    candidate_priority = "outputs/checkpoints/bioprint_v5_mps_retrain_evalprep_lowmem/checkpoint_best.pt"

    def rank(report: Dict[str, Any]) -> Tuple[int, float, int, int, int]:
        readable = 1 if report.get("readable") else 0
        has_model = 1 if report.get("contains_model_keys") else 0
        metric = report.get("best_metric")
        metric_score = float(metric) if isinstance(metric, (int, float)) else -1.0
        epoch = report.get("epoch")
        epoch_score = int(epoch) if isinstance(epoch, int) else (_epoch_from_name(report["path"]) or -1)
        priority = 1 if Path(report["path"]).as_posix().endswith(candidate_priority) or report["path"] == candidate_priority else 0
        return (readable, has_model, priority, metric_score, epoch_score)

    valid = [r for r in reports if r.get("readable") and r.get("contains_model_keys")]
    selected = max(valid, key=rank) if valid else None
    rationale = {
        "selected": selected["path"] if selected else None,
        "reason": (
            "selected highest-ranked readable checkpoint with model weights; current local checkpoint_best.pt is prioritized when valid"
            if selected
            else "no readable checkpoint with model weights found"
        ),
        "candidate_count": len(reports),
        "valid_model_checkpoint_count": len(valid),
        "reports": reports,
    }
    return (selected["path"] if selected else None), rationale


def _write_reports(rationale: Dict[str, Any], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "checkpoint_integrity_report.json"
    md_path = outdir / "checkpoint_integrity_report.md"
    json_path.write_text(json.dumps(rationale, indent=2, default=str))
    lines = [
        "# Checkpoint Integrity Report",
        "",
        f"- selected: `{rationale.get('selected')}`",
        f"- reason: {rationale.get('reason')}",
        f"- candidate_count: {rationale.get('candidate_count')}",
        f"- valid_model_checkpoint_count: {rationale.get('valid_model_checkpoint_count')}",
        "",
        "| Path | Readable | Type | Epoch | Best Metric | Model Keys | Error |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for r in rationale.get("reports", []):
        err = str(r.get("load_error") or "").replace("|", "/")
        lines.append(
            f"| `{r.get('path')}` | {r.get('readable')} | {r.get('type')} | {r.get('epoch')} | "
            f"{r.get('best_metric')} | {r.get('contains_model_keys')} | {err[:120]} |"
        )
    md_path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and select PyTorch checkpoints")
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="outputs/checkpoint_backups")
    parser.add_argument("--backup", action="store_true", help="Back up discovered checkpoints using hardlinks when possible")
    args = parser.parse_args(argv)

    paths = list_checkpoints(args.root)
    backup_manifest = []
    if args.backup:
        for path in paths:
            try:
                backup_manifest.append({"source": path, "backup": safe_backup_checkpoint(path, args.outdir)})
            except Exception as exc:
                backup_manifest.append({"source": path, "backup": None, "error": str(exc)})
    selected, rationale = select_best_checkpoint(paths)
    outdir = Path(args.outdir)
    _write_reports(rationale, outdir)
    if backup_manifest:
        (outdir / "backup_manifest.json").write_text(json.dumps(backup_manifest, indent=2))
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/selected_checkpoint.txt").write_text((selected or "fallback_init_unspecified") + "\n")
    print(json.dumps({"selected": selected, "report": str(outdir / "checkpoint_integrity_report.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
