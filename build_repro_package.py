#!/usr/bin/env python3
"""Build an anonymized reproducibility package for review."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Iterable, Optional


EXCLUDE_DIRS = {"Data", ".git", ".venv", "node_modules", ".next", "outputs/checkpoints", "checkpoints"}
EXCLUDE_SUFFIXES = {".nii", ".gz", ".pt", ".pth", ".ckpt", ".stl", ".npy"}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unspecified"


def pip_freeze() -> str:
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL)
    except Exception as exc:
        return f"pip_freeze_unavailable: {exc}\n"


def should_exclude(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if any(str(path).startswith(d + os.sep) for d in EXCLUDE_DIRS):
        return True
    if path.suffix in EXCLUDE_SUFFIXES or path.name.endswith(".nii.gz"):
        return True
    if "CloudStorage" in path.parts or "OneDrive" in path.parts:
        return True
    return False


def sanitize_text(text: str, root: Path) -> str:
    text = text.replace(str(root), "<repo_root>")
    text = re.sub(r"/Users/[^/\\s]+", "/Users/<user>", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", "<email>", text)
    return text


def build_package(output: str | Path = "outputs/repro_package_anonymized.tar.gz") -> Path:
    root = Path.cwd()
    outputs = root / "outputs"
    outputs.mkdir(exist_ok=True)
    (outputs / "git_commit.txt").write_text(git_commit() + "\n")
    (outputs / "pip_freeze.txt").write_text(pip_freeze())
    readme = outputs / "README_anonymized_package.md"
    readme.write_text(
        "# Anonymized Reproducibility Package\n\n"
        "This archive excludes raw datasets, large checkpoints, personal paths, and generated mesh/volume artifacts. "
        "It includes code, commands, environment records, inventory, summary tables, and configuration files needed to reproduce the A40 campaign.\n"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tar:
        for path in root.rglob("*"):
            rel = path.relative_to(root)
            if path == output or should_exclude(rel):
                continue
            if path.is_dir():
                continue
            if rel.parts and rel.parts[0] == "outputs":
                allowed = rel.name.endswith((".txt", ".md", ".json", ".csv", ".yaml", ".yml"))
                if not allowed:
                    continue
            try:
                if path.suffix.lower() in {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".csv"}:
                    data = sanitize_text(path.read_text(errors="ignore"), root).encode()
                    info = tarfile.TarInfo(str(rel))
                    info.size = len(data)
                    tar.addfile(info, fileobj=__import__("io").BytesIO(data))
                else:
                    tar.add(path, arcname=str(rel))
            except Exception:
                continue
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build anonymized reproducibility package")
    parser.add_argument("--output", default="outputs/repro_package_anonymized.tar.gz")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(build_package(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
