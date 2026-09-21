"""Reproducibility capture: environment, provenance, inputs, outputs."""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .io_utils import sha256_file


def _run(cmd: Sequence[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None,
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_provenance(root: Path) -> Dict[str, Any]:
    """Record the git commit if the working copy is a repository.

    The audited working copy has no ``.git`` directory. That is recorded
    explicitly rather than silently omitted, because the experiment's
    reproducibility claim depends on knowing which code state produced it.
    """
    if not (root / ".git").exists():
        return {
            "is_git_repository": False,
            "commit": None,
            "note": "No .git directory in the working copy; commit hash unavailable. "
                    "Record the upstream commit manually before publication.",
        }
    return {
        "is_git_repository": True,
        "commit": _run(["git", "rev-parse", "HEAD"], root),
        "commit_short": _run(["git", "rev-parse", "--short", "HEAD"], root),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": bool(_run(["git", "status", "--porcelain"], root)),
        "describe": _run(["git", "describe", "--always", "--dirty"], root),
    }


def environment() -> Dict[str, Any]:
    versions: Dict[str, Optional[str]] = {}
    for name in ("numpy", "scipy", "skimage", "nibabel", "trimesh", "pandas",
                 "matplotlib", "yaml", "SimpleITK", "pymeshfix"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": versions,
    }


def code_provenance(root: Path) -> Dict[str, Any]:
    """Hash the repository code this experiment depends on but does not modify."""
    tracked = [
        "phaseb_mesh_qc.py",
        "compute_cldice.py",
        "phaseb/src/phaseb/postprocess.py",
        "phaseb/src/phaseb/mesh.py",
        "phaseb/src/phaseb/repair.py",
    ]
    out: Dict[str, Any] = {}
    for rel in tracked:
        p = root / rel
        out[rel] = sha256_file(p) if p.exists() else None
    experiment_files = sorted((root / "experiments" / "topology_correction").rglob("*.py"))
    out["experiment_code"] = {
        str(p.relative_to(root)): sha256_file(p) for p in experiment_files
    }
    return out


def authoritative_fingerprint(root: Path, paths: Sequence[str]) -> Dict[str, Any]:
    """Fingerprint authoritative artifacts so it can be proven they were untouched."""
    out: Dict[str, Any] = {}
    for rel in paths:
        p = root / rel
        if p.exists():
            st = p.stat()
            out[rel] = {"sha256": sha256_file(p), "size_bytes": st.st_size,
                        "mtime_iso": _dt.datetime.fromtimestamp(st.st_mtime).isoformat()}
        else:
            out[rel] = None
    return out


def input_manifest(cases: Sequence[str], resolved: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per-case input file inventory with hashes."""
    return {
        "n_cases_requested": len(cases),
        "n_cases_resolved": sum(1 for c in cases if resolved.get(c, {}).get("complete")),
        "cases": {c: resolved.get(c, {"complete": False, "missing": ["unresolved"]}) for c in cases},
    }


def build_run_manifest(
    root: Path,
    config: Dict[str, Any],
    config_path: Path,
    started_iso: str,
    finished_iso: str,
    outputs: Dict[str, Any],
    authoritative_paths: Sequence[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = {
        "experiment": config.get("experiment", {}),
        "timestamp_started": started_iso,
        "timestamp_finished": finished_iso,
        "timezone": str(_dt.datetime.now().astimezone().tzinfo),
        "seed": config.get("experiment", {}).get("seed"),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path) if Path(config_path).exists() else None,
        "config_resolved": config,
        "git": git_provenance(root),
        "environment": environment(),
        "code_provenance": code_provenance(root),
        "authoritative_artifacts_fingerprint": authoritative_fingerprint(root, authoritative_paths),
        "outputs": outputs,
    }
    if extra:
        manifest.update(extra)
    return manifest


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat()
