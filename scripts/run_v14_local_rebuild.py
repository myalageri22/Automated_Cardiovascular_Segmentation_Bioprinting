#!/usr/bin/env python3
"""Run the full local v14 rebuild workflow.

Workflow:
1. Fine-tune from the archived v11 checkpoint with the v14 Tversky command.
2. Build an epoch 75-80 uniform checkpoint soup.
3. Run the held-out 250-case test on the reproduced soup.
4. Write provenance and SHA256 files.

This does not claim to recover the deleted RunPod checkpoint. It creates a new,
locally reproduced v14-style checkpoint with complete provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_nii(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.nii.gz")) if root.exists() else 0


def run(cmd: list[str], log_path: Path | None = None) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=f"bioprint_v14_tversky_fn_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--imagecas-root", default="Data/all")
    parser.add_argument("--splits-json", default="splits.json")
    parser.add_argument(
        "--resume-checkpoint",
        default="outputs/train_runs/bioprint_v11_a40_roi96_posneg3to1_lr2e5/checkpoints/best_dice05.pt",
    )
    parser.add_argument("--output-root", default="outputs/train_runs")
    parser.add_argument("--eval-output-root", default="outputs/final_test_250_local_reproduced")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-training", action="store_true", help="Use an existing run folder and only soup/evaluate.")
    parser.add_argument("--skip-eval", action="store_true", help="Train and soup only.")
    args = parser.parse_args()

    imagecas_root = Path(args.imagecas_root)
    resume = Path(args.resume_checkpoint)
    out_root = Path(args.output_root)
    run_dir = out_root / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    soup_path = ckpt_dir / "soup_ep75-80.pt"
    provenance_dir = Path("outputs/v14_local_rebuild_provenance") / args.run_name
    provenance_dir.mkdir(parents=True, exist_ok=True)
    workflow_log = provenance_dir / "workflow.log"

    data_count = count_nii(imagecas_root)
    if data_count < 2000:
        raise SystemExit(
            f"ImageCAS data not found or incomplete at {imagecas_root} ({data_count} .nii.gz files). "
            "Restore Data/all or pass --imagecas-root /path/to/Data/all."
        )
    if not resume.exists():
        raise SystemExit(f"Resume checkpoint not found: {resume}")

    status = {
        "run_name": args.run_name,
        "imagecas_root": str(imagecas_root),
        "splits_json": args.splits_json,
        "resume_checkpoint": str(resume),
        "resume_checkpoint_sha256": sha256_file(resume),
        "output_root": str(out_root),
        "run_dir": str(run_dir),
        "soup_checkpoint": str(soup_path),
        "device_requested": args.device,
        "started_at": datetime.now().isoformat(),
        "note": "New local v14-style reproduction. Not the deleted original RunPod checkpoint.",
    }
    (provenance_dir / "rebuild_status_start.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    if not args.skip_training:
        train_cmd = [
            sys.executable,
            "train_v14_local_mps.py",
            "--experiment_name",
            args.run_name,
            "--resume_checkpoint",
            str(resume),
            "--strict_load",
            "--allow_partial_load",
            "--model_type",
            "attention_unet",
            "--roi_size",
            "96,192,192",
            "--batch_size",
            "1",
            "--accumulation_steps",
            "2",
            "--epochs",
            "100",
            "--lr",
            "2e-5",
            "--weight_decay",
            "1e-5",
            "--grad_clip_norm",
            "1.0",
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
            "--splits_json",
            args.splits_json,
            "--imagecas_root",
            str(imagecas_root),
            "--cache_rate_train",
            "0.0",
            "--cache_rate_val",
            "0.0",
            "--scheduler",
            "reduce_on_plateau",
            "--early_stopping_patience",
            "15",
            "--pos_neg_ratio",
            "3,1",
            "--num_samples",
            "4",
            "--min_pos_voxels",
            "1",
            "--pos_weight_fixed",
            "0",
            "--loss_mode",
            "tversky_dice",
            "--output_dir",
            str(out_root),
        ]
        run(train_cmd, workflow_log)

    soup_cmd = [
        sys.executable,
        "soup_checkpoints.py",
        "--checkpoint_dir",
        str(ckpt_dir),
        "--output",
        str(soup_path),
        "--epoch_start",
        "75",
        "--epoch_end",
        "80",
    ]
    run(soup_cmd, workflow_log)

    if not soup_path.exists():
        raise SystemExit(f"Soup checkpoint was not created: {soup_path}")

    if not args.skip_eval:
        eval_out = Path(args.eval_output_root) / args.run_name
        eval_cmd = [
            sys.executable,
            "evaluate_v14_local.py",
            "--checkpoint",
            str(soup_path),
            "--splits_json",
            args.splits_json,
            "--imagecas_root",
            str(imagecas_root),
            "--split",
            "test",
            "--roi_size",
            "96,192,192",
            "--sw_overlap",
            "0.625",
            "--sw_batch_size",
            "1",
            "--threshold",
            "0.5",
            "--output_dir",
            str(eval_out),
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
            "--resume",
            "--skip_existing",
        ]
        run(eval_cmd, workflow_log)
        status["eval_output_dir"] = str(eval_out)

    status["soup_checkpoint_sha256"] = sha256_file(soup_path)
    status["completed_at"] = datetime.now().isoformat()
    (provenance_dir / "rebuild_status_complete.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    (provenance_dir / "README.md").write_text(
        "# v14 Local Rebuild Provenance\n\n"
        "This folder documents a new local reproduction of the v14-style checkpoint. "
        "It does not recover the deleted original RunPod `/tmp` checkpoint.\n\n"
        f"- Run: `{args.run_name}`\n"
        f"- Soup checkpoint: `{soup_path}`\n"
        f"- Workflow log: `{workflow_log}`\n",
        encoding="utf-8",
    )
    print(f"\nDone. Provenance: {provenance_dir}")
    print(f"Soup checkpoint: {soup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
