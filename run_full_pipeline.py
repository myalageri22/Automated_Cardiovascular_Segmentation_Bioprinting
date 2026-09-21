#!/usr/bin/env python3
"""Run Phase A (model inference) -> Phase B (mesh + STL) end-to-end.

Example:
  python run_full_pipeline.py \
    --ct /path/to/ct.nii.gz \
    --checkpoint checkpoints/checkpoint_best.pt \
    --outdir pipeline_outputs \
    --case-id demo
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import nibabel as nib
except ImportError as e:
    raise SystemExit("nibabel is required. Install with: pip install nibabel") from e

try:
    import SimpleITK as sitk
except ImportError as e:
    raise SystemExit("SimpleITK is required. Install with: pip install SimpleITK") from e

try:
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
    )
except ImportError as e:
    raise SystemExit("monai is required. Install with: pip install monai[all]") from e

from train_updated import Config, build_model
from phaseb.src.phaseb import io as phaseb_io
from phaseb.src.phaseb import pipeline as phaseb_pipeline


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("full_pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _load_checkpoint(checkpoint_path: Path, device: str, logger: logging.Logger):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    torch.serialization.add_safe_globals([
        np._core.multiarray.scalar,
        np.dtype,
        Config,
    ])
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = state.get("config")
    if not isinstance(cfg, Config):
        logger.warning("Checkpoint missing Config; using defaults.")
        cfg = Config()

    cfg.device = device
    return state, cfg


def _infer_transforms(config: Config):
    transforms = [
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=config.pixdim, mode=("bilinear")),
    ]
    if config.modality == "ct":
        transforms.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config.ct_window[0],
                a_max=config.ct_window[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    else:
        transforms.append(
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=True,
            )
        )
    transforms.append(EnsureTyped(keys=["image"], dtype=torch.float32))
    return Compose(transforms)


def _dicom_to_nifti(ct_path: Path, out_path: Path, logger: logging.Logger) -> Path:
    if not ct_path.is_dir():
        return ct_path
    logger.info("Detected DICOM directory; converting to NIfTI for inference.")
    image = phaseb_io.load_image(str(ct_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(out_path))
    return out_path


def _extract_affine(meta: dict) -> np.ndarray:
    affine = meta.get("affine", None)
    if affine is None:
        return np.eye(4)
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    if affine.shape == (1, 4, 4):
        return affine[0]
    if affine.shape != (4, 4):
        return np.eye(4)
    return affine


def run_phase_a(
    ct_path: Path,
    checkpoint_path: Path,
    outdir: Path,
    case_id: str,
    device: str,
    prob_threshold: Optional[float],
    logger: logging.Logger,
) -> tuple[Path, Path, Optional[Path]]:
    state, config = _load_checkpoint(checkpoint_path, device, logger)
    model = build_model(config, logger)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    infer_t = _infer_transforms(config)
    data = infer_t({"image": str(ct_path)})
    image = data["image"].unsqueeze(0).to(config.device)
    meta = data.get("image_meta_dict", {})
    affine = _extract_affine(meta)

    with torch.no_grad():
        logits = sliding_window_inference(
            image,
            roi_size=config.roi_size,
            sw_batch_size=config.sw_batch_size,
            predictor=model,
            overlap=config.sw_overlap,
            mode="gaussian",
        )
    probs = torch.sigmoid(logits).cpu()

    phasea_dir = outdir / case_id / "phasea"
    phasea_dir.mkdir(parents=True, exist_ok=True)
    ct_out = phasea_dir / "ct_preprocessed.nii.gz"
    prob_out = phasea_dir / "seg_prob.nii.gz"
    mask_out = phasea_dir / "seg_mask.nii.gz"

    nib.Nifti1Image(image[0, 0].cpu().numpy(), affine).to_filename(str(ct_out))
    nib.Nifti1Image(probs[0, 0].numpy(), affine).to_filename(str(prob_out))

    if prob_threshold is not None:
        mask = (probs > prob_threshold).float()
        nib.Nifti1Image(mask[0, 0].numpy().astype(np.uint8), affine).to_filename(str(mask_out))
    else:
        mask_out = None

    logger.info(f"Phase A outputs: {prob_out}")
    return ct_out, prob_out, mask_out


def run_phase_b(
    ct_path: Path,
    seg_path: Path,
    outdir: Path,
    case_id: str,
    config_path: Path,
    threshold: Optional[float],
    threshold_sweep: bool,
    generate_previews: bool,
    logger: logging.Logger,
) -> None:
    cfg = phaseb_pipeline.load_config(str(config_path))
    logger.info("Starting Phase B")
    phaseb_pipeline.run_case(
        ct_path=str(ct_path),
        seg_path=str(seg_path),
        seg_type=phaseb_io.SegmentationType.PROBABILITY,
        case_id=case_id,
        outdir=outdir,
        config=cfg,
        threshold_override=threshold,
        threshold_sweep=threshold_sweep,
        resample_mm=None,
        min_component_mm3=None,
        min_radius_mm=None,
        generate_previews=generate_previews,
        seed_point=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase A inference + Phase B STL pipeline")
    parser.add_argument("--ct", required=True, help="Path to CT NIfTI or DICOM directory")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_best.pt", help="Phase A checkpoint path")
    parser.add_argument("--outdir", default="pipeline_outputs", help="Output directory root")
    parser.add_argument("--case-id", default=None, help="Case identifier (defaults to CT filename stem)")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu", "mps"], help="Force device (default: auto)")
    parser.add_argument("--prob-threshold", type=float, default=None, help="Optional threshold to save a binary mask")
    parser.add_argument("--phaseb-config", default=str(Path("phaseb/configs/default.yaml")), help="Phase B config YAML")
    parser.add_argument("--phaseb-threshold", type=float, default=None, help="Override Phase B threshold for prob map")
    parser.add_argument("--phaseb-threshold-sweep", action="store_true", help="Run Phase B threshold sweep")
    parser.add_argument("--phaseb-previews", action="store_true", help="Generate Phase B preview images")

    args = parser.parse_args()
    logger = _setup_logger()

    ct_path = Path(args.ct)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    case_id = args.case_id or ct_path.stem
    if ct_path.is_dir():
        case_id = args.case_id or ct_path.name

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare CT (convert DICOM to NIfTI if needed)
    ct_nifti = _dicom_to_nifti(ct_path, outdir / case_id / "phasea" / "ct_input.nii.gz", logger)

    # Phase A inference
    ct_pre, seg_prob, _ = run_phase_a(
        ct_path=ct_nifti,
        checkpoint_path=Path(args.checkpoint),
        outdir=outdir,
        case_id=case_id,
        device=device,
        prob_threshold=args.prob_threshold,
        logger=logger,
    )

    # Phase B
    run_phase_b(
        ct_path=ct_pre,
        seg_path=seg_prob,
        outdir=outdir,
        case_id=case_id,
        config_path=Path(args.phaseb_config),
        threshold=args.phaseb_threshold,
        threshold_sweep=args.phaseb_threshold_sweep,
        generate_previews=args.phaseb_previews,
        logger=logger,
    )

    logger.info(f"Pipeline complete. Outputs in: {outdir / case_id}")


if __name__ == "__main__":
    main()
