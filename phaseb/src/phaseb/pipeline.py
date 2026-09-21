"""Phase B pipeline orchestration."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import yaml

from . import io
from . import mesh as mesh_utils
from . import postprocess, preprocess, qc, report, repair, resample

np.random.seed(0)

_DEFAULT_ROOT = Path(__file__).resolve().parents[3]  # repo root
DEFAULT_OUTDIR = Path(os.environ.get("PHASEB_OUTPUT_DIR") or (_DEFAULT_ROOT / "phaseb_outputs"))


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_seed_point(seed: Optional[str], reference_image: sitk.Image) -> Optional[Tuple[int, int, int]]:
    if seed is None:
        return None
    parts = [p.strip() for p in seed.split(",")]
    if len(parts) != 3:
        raise ValueError("Seed point must be x,y,z in mm")
    physical = tuple(float(p) for p in parts)
    continuous = reference_image.TransformPhysicalPointToContinuousIndex(physical)
    zyx = (int(round(continuous[2])), int(round(continuous[1])), int(round(continuous[0])))
    return zyx


def prepare_case_output(base_outdir: Path, case_id: str) -> Path:
    case_dir = base_outdir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _select_seg_type(seg_type_arg: str, seg_array: np.ndarray) -> str:
    if seg_type_arg != io.SegmentationType.AUTO:
        return seg_type_arg
    return io.detect_segmentation_type(seg_array)


def run_case(
    ct_path: str,
    seg_path: str,
    seg_type: str,
    case_id: Optional[str],
    outdir: Path,
    config: Dict,
    threshold_override: Optional[float] = None,
    threshold_sweep: bool = False,
    resample_mm: Optional[float] = None,
    min_component_mm3: Optional[float] = None,
    min_radius_mm: Optional[float] = None,
    generate_previews: bool = False,
    seed_point: Optional[str] = None,
) -> Dict:
    case_id = case_id or Path(seg_path if seg_path else ct_path).stem
    case_dir = prepare_case_output(outdir, case_id)
    logger = io.setup_logging(str(case_dir), level=config.get("logging", {}).get("level", "INFO"))
    logger.info(f"Starting Phase B for case {case_id}")

    ct = io.load_volume(ct_path)
    seg = io.load_volume(seg_path)

    seg_type_resolved = _select_seg_type(seg_type, seg.array)
    logger.info(f"Segmentation type: {seg_type_resolved}")

    if resample_mm is None:
        resample_mm = config.get("resample", {}).get("target_spacing_mm")
    if resample_mm:
        logger.info(f"Resampling to {resample_mm} mm isotropic")
        ct.image, ct.array = resample.resample_image(ct.image, resample_mm, is_label=False)
        seg.image, seg.array = resample.resample_image(
            seg.image, resample_mm, is_label=(seg_type_resolved != io.SegmentationType.PROBABILITY)
        )

    spacing = tuple(float(s) for s in seg.image.GetSpacing())

    if seg_type_resolved == io.SegmentationType.PROBABILITY:
        seg_array = seg.array.astype(np.float32)
        prob_array = seg_array
    else:
        seg_array = (seg.array > 0).astype(np.uint8)
        prob_array = None

    if min_component_mm3 is None:
        min_component_mm3 = config.get("postprocess", {}).get("min_component_mm3", 0.0)
    if min_radius_mm is None:
        min_radius_mm = config.get("qc", {}).get("min_radius_mm", 0.6)

    post_cfg = copy.deepcopy(config.get("postprocess", {}))
    post_cfg["min_component_mm3"] = min_component_mm3
    if seed_point is not None:
        post_cfg["keep_mode"] = "seed"
    seed_index = parse_seed_point(seed_point, seg.image) if seed_point else None

    threshold = threshold_override or config.get("segmentation", {}).get("default_threshold", 0.5)
    sweep_results = None
    if threshold_sweep:
        if seg_type_resolved != io.SegmentationType.PROBABILITY:
            logger.warning("Threshold sweep requested but segmentation is not probability map; skipping sweep")
        else:
            candidates = config.get("segmentation", {}).get("threshold_candidates", [0.2, 0.4, 0.6, 0.8])
            threshold, sweep_results = preprocess.run_threshold_sweep(
                prob_array, spacing=spacing, thresholds=[float(c) for c in candidates], post_cfg=post_cfg, seed_index_zyx=seed_index
            )
            logger.info(f"Selected threshold {threshold} from sweep")

    cleaned_mask = preprocess.threshold_and_postprocess(
        seg_array,
        seg_type=seg_type_resolved,
        threshold=threshold,
        spacing=spacing,
        post_cfg=post_cfg,
        seed_index_zyx=seed_index,
    )
    if cleaned_mask.sum() == 0:
        raise ValueError("Processed segmentation is empty after cleanup")

    step_size = int(config.get("mesh", {}).get("marching_cubes_step", 1) or 1)
    raw_mesh, _, _ = mesh_utils.mask_to_mesh(cleaned_mask, seg.image, step_size=step_size)
    raw_mesh_path = case_dir / "vessels_raw.stl"
    mesh_utils.export_mesh(raw_mesh, raw_mesh_path)

    repaired_mesh, repair_info = repair.repair_mesh(raw_mesh, use_pymeshfix=config.get("repair", {}).get("use_pymeshfix", True))
    repaired_mesh_path = case_dir / "vessels_repaired.stl"
    mesh_utils.export_mesh(repaired_mesh, repaired_mesh_path)

    robustness = qc.robustness_checks(
        prob_array=prob_array,
        reference_image=seg.image,
        base_threshold=threshold,
        spacing=spacing,
        post_cfg=post_cfg,
        base_volume_mm3=postprocess.component_metrics(cleaned_mask, spacing)["volume_mm3"],
        base_components=postprocess.component_metrics(cleaned_mask, spacing)["component_count"],
        base_watertight=bool(repaired_mesh.is_watertight),
    )

    used_config = copy.deepcopy(config)
    used_config.setdefault("segmentation", {})["selected_threshold"] = float(threshold)
    used_config.setdefault("postprocess", {})["min_component_mm3"] = float(min_component_mm3)
    used_config.setdefault("qc", {})["min_radius_mm"] = float(min_radius_mm)

    qc_report = qc.assemble_qc_report(
        case_id=case_id,
        ct_path=os.path.abspath(ct_path),
        seg_path=os.path.abspath(seg_path),
        seg_type=seg_type_resolved,
        ct_image=ct.image,
        probability_array=prob_array,
        cleaned_mask=cleaned_mask,
        raw_mesh=raw_mesh,
        repaired_mesh=repaired_mesh,
        threshold=float(threshold),
        post_cfg=post_cfg,
        min_radius_mm=float(min_radius_mm),
        sweep_results=sweep_results,
        robustness=robustness,
        used_config=io.serialize_config(used_config),
    )

    report.write_json(qc_report, str(case_dir))
    report.write_text_summary(qc_report, str(case_dir))
    report.write_csv_summary(qc_report, str(case_dir))

    if generate_previews or config.get("previews", {}).get("enable", False):
        report.generate_previews(ct.image, cleaned_mask, repaired_mesh, str(case_dir), case_id)

    io.save_used_config(used_config, str(case_dir))
    logger.info(f"Completed Phase B for {case_id}")
    return qc_report


def run_batch(
    manifest_path: str,
    outdir: Path,
    config: Dict,
    threshold_override: Optional[float] = None,
    threshold_sweep: bool = False,
    resample_mm: Optional[float] = None,
    min_component_mm3: Optional[float] = None,
    min_radius_mm: Optional[float] = None,
    generate_previews: bool = False,
) -> List[Dict]:
    manifest = io.load_manifest(manifest_path)
    reports = []
    for rec in manifest:
        ct_path = rec.get("ct_path") or rec.get("ct")
        seg_path = rec.get("seg_path") or rec.get("seg")
        seg_type = rec.get("seg_type", io.SegmentationType.AUTO)
        case_id = rec.get("case_id")
        qc_report = run_case(
            ct_path=ct_path,
            seg_path=seg_path,
            seg_type=seg_type,
            case_id=case_id,
            outdir=outdir,
            config=config,
            threshold_override=threshold_override,
            threshold_sweep=threshold_sweep,
            resample_mm=resample_mm,
            min_component_mm3=min_component_mm3,
            min_radius_mm=min_radius_mm,
            generate_previews=generate_previews,
            seed_point=None,
        )
        reports.append(qc_report)
    return reports
