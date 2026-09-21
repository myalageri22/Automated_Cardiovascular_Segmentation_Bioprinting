#!/usr/bin/env python3
"""
ImageCAS 3D Vascular Segmentation Training Pipeline

Features:
- Patch-based training with RandCropByPosNegLabeld
- Sliding window validation with Gaussian blending
- Composite loss (Dice + BCEWithLogits + Tversky/Focal)
- Robust class balancing for extreme sparsity
- Built-in sanity checks: --overfit_one, --dry_run
- Safe checkpoint resume (absolute/relative/"latest") - **FIXED for PyTorch 2.6+**
- Automatic mixed precision (AMP) with GradScaler
- Comprehensive logging and probability diagnostics
- Training history tracking and plotting

# Train from scratch:
python "train_vascular.py" \
  --dataset_preset imagecas \
  --imagecas_root /workspace/datasets/imagecas_v3/Cloud_bundle1111/data/processed/imagecas_extracted \
  --use_official_split \
  --split_xlsx /workspace/datasets/imagecas_v3/Cloud_bundle1111/imageCAS_data_split_ids1to1000.xlsx \
  --split_id 1 \
  --epochs 100 \
  --batch_size 1 \
  --roi_size 80,160,160 \
  --pos_neg_ratio 1,0 \
  --force_pos_patches \
  --min_pos_voxels 500 \
  --save_val_preds \
  --experiment_name "bioprint_v1"
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import re

try:
    import monai
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRanged, NormalizeIntensityd, RandCropByPosNegLabeld,
        SpatialPadd, RandFlipd, RandRotate90d, Rand3DElasticd,
        RandAdjustContrastd, RandScaleIntensityd, RandGaussianNoised,
        RandGaussianSmoothd, ResizeWithPadOrCropd, EnsureTyped, Lambdad,
        RandRotated, RandShiftIntensityd
    )
    from monai.networks.nets import UNet, AttentionUnet
    from monai.inferers import sliding_window_inference
    from monai.data import CacheDataset, DataLoader
    from monai.data.utils import pad_list_data_collate
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    from monai.utils import set_determinism
except ImportError as e:
    print(f"MONAI not installed: {e}\nInstall with: pip install monai[all]")
    sys.exit(1)

try:
    import nibabel as nib
except ImportError:
    print("nibabel not installed: pip install nibabel")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("pandas not installed: pip install pandas openpyxl")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning, message=".*epoch parameter.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*min_size.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*always_return_as_numpy.*")

# Plotting Function

def plot_training_history(history_path: Path, output_path: Optional[Path] = None, logger: Optional[logging.Logger] = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    if not history_path.exists():
        print(f"History file not found: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Training History: {history_path.stem}", fontsize=14, fontweight='bold')
    
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    
    # Plot 1: Loss curves
    ax = axes[0, 0]
    if "train_loss" in history and history["train_loss"]:
        ax.plot(epochs, history["train_loss"], 'b-', label='Train Loss', linewidth=2)
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Dice scores
    ax = axes[0, 1]
    dice_key = None
    for key in ["dice@0.5", "dice@0.1", "soft_dice"]:
        if key in history.get("val_metrics", {}) and history["val_metrics"][key]:
            dice_key = key
            break
    
    if dice_key:
        dice_scores = history["val_metrics"][dice_key]
        ax.plot(epochs, dice_scores, 'g-', label=f'Val {dice_key}', linewidth=2, marker='o', markersize=3)
        best_idx = np.argmax(dice_scores)
        best_score = dice_scores[best_idx]
        ax.axhline(y=best_score, color='g', linestyle='--', alpha=0.5, label=f'Best: {best_score:.4f} @ epoch {best_idx+1}')
        ax.scatter([best_idx+1], [best_score], color='red', s=100, zorder=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice Score')
    ax.set_title(f'Validation Dice ({dice_key or "N/A"})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 3: Learning rate
    ax = axes[1, 0]
    if "lr" in history and history["lr"]:
        ax.plot(epochs, history["lr"], 'm-', label='Learning Rate', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Probability statistics
    ax = axes[1, 1]
    if "val_metrics" in history:
        if "prob_mean" in history["val_metrics"] and history["val_metrics"]["prob_mean"]:
            ax.plot(epochs, history["val_metrics"]["prob_mean"], 'c-', label='Prob Mean', linewidth=2)
        if "voxels>0.5" in history["val_metrics"] and history["val_metrics"]["voxels>0.5"]:
            # Normalize voxels>0.5 for visibility
            voxels = np.array(history["val_metrics"]["voxels>0.5"])
            ax.plot(epochs, voxels / (voxels.max() + 1e-6), 'orange', label='Voxels>0.5 (norm)', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('Probability Statistics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = history_path.parent / f"{history_path.stem}_plot.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total epochs: {len(epochs)}")
    if "train_loss" in history:
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if "val_loss" in history and history["val_loss"]:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    if dice_key:
        dice_scores = history["val_metrics"][dice_key]
        best_idx = np.argmax(dice_scores)
        print(f"Best {dice_key}: {dice_scores[best_idx]:.4f} at epoch {best_idx+1}")
        print(f"Final {dice_key}: {dice_scores[-1]:.4f}")
    print("="*60)
    
    if logger:
        logger.info(f"Training plot saved: {output_path}")

# Configuration Dataclass

@dataclass
class Config:
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    imagecas_root: Path = None
    checkpoint_dir: Path = None
    log_dir: Path = None
    
    # Hardware
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "auto"
    
    # Data
    dataset_preset: str = "custom"
    modality: str = "ct"
    ct_window: Tuple[float, float] = (-200.0, 700.0)
    pixdim: Tuple[float, float, float] = (0.6, 0.6, 0.6)
    roi_size: Tuple[int, int, int] = (96, 192, 192)
    roi_multiple: int = 16
    
    # Training
    epochs: int = 100
    batch_size: int = 1
    accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    
    # Sampling
    pos_ratio: int = 1
    neg_ratio: int = 0
    pos_neg_ratio: Tuple[int, int] = (1, 0)
    num_samples: int = 2
    force_pos_patches: bool = False
    min_pos_voxels: int = 400
    
    # Loss
    loss_mode: str = "all"
    pos_weight_cap: float = 10.0
    pos_weight_fixed: Optional[float] = None
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    
    # Metrics
    prob_threshold: float = 0.15
    save_val_threshold: float = 0.15
    min_cc_size: int = 50
    
    # Validation
    sw_batch_size: int = 1
    sw_overlap: float = 0.5
    save_val_preds: bool = False
    save_val_every: int = 1
    save_val_preds_n: int = 5
    
    # Data loading
    cache_rate_train: float = 0.1
    cache_rate_val: float = 0.25
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 4
    
    # Model
    unet_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024)
    unet_strides: Tuple[int, ...] = (2, 2, 2, 2)
    unet_num_res_units: int = 3
    unet_dropout: float = 0.1
    unet_norm: str = "batch"
    use_attention: bool = True
    compile_model: bool = False
    
    # Misc
    split_file: Path = None
    split_id: int = 1
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    seed: int = 42
    deterministic: bool = True
    skip_nonfinite_batches: bool = True
    dice_fp32: bool = True
    
    # Sanity modes
    overfit_one: bool = False
    dry_run: bool = False
    verify_data: bool = False
    verify_n: int = 3
    limit_train: Optional[int] = None
    limit_val: Optional[int] = 30
    
    def __post_init__(self):
        """Post-initialization to set derived paths and validate."""
        if self.imagecas_root is None:
            self.imagecas_root = self.project_root / "data"
        
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_root / "checkpoints"
            
        if self.log_dir is None:
            self.log_dir = self.project_root / "logs"
            
        if self.split_file is None:
            self.split_file = self.project_root / "splits.json"
            
        for d in [self.checkpoint_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        if self.dataset_preset == "imagecas":
            self.modality = "ct"
            self.pixdim = (0.6, 0.6, 0.6)
            self.ct_window = (-200.0, 700.0)
            if not hasattr(self, '_batch_size_set'):
                self.batch_size = 1
            if not hasattr(self, '_learning_rate_set'):
                self.learning_rate = 2e-4
            
        self.roi_size = tuple(
            ((dim + self.roi_multiple - 1) // self.roi_multiple) * self.roi_multiple
            for dim in self.roi_size
        )



def set_all_seeds(seed: int, deterministic: bool = True):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        set_determinism(seed=seed)
    else:
        torch.backends.cudnn.benchmark = True

def setup_logging(log_dir: Path, experiment_name: str) -> logging.Logger:
    """Setup comprehensive logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"
    
    logger = logging.getLogger("VascularSeg")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized: {log_file}")
    return logger

def discover_cases(data_root: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Recursively discover image files and pair them with label files."""
    logger.info(f"Scanning for cases in: {data_root}")

    patterns = ["*_img.nii.gz", "*.img.nii.gz", "*_image.nii.gz", "*.nii.gz"]
    found = []
    for p in patterns:
        for f in data_root.rglob(p):
            if not f.is_file():
                continue
            stem_l = f.stem.lower()
            if re.search(r"label|mask", stem_l):
                continue
            found.append(f)

    image_files = sorted(list({p.resolve(): p for p in found}.values()))

    if not image_files:
        raise FileNotFoundError(
            f"No image files found in {data_root}. Expected patterns: {patterns}"
        )

    cases = []
    for img_path in image_files:
        name = img_path.name
        if name.lower().endswith('.nii.gz'):
            base = name[:-7]
        else:
            base = img_path.stem
        case_id = re.sub(r"[\._-]?(?:img|image)$", "", base, flags=re.IGNORECASE)

        label_candidates = [
            f"{case_id}_label.nii.gz",
            f"{case_id}.label.nii.gz",
            f"{case_id}-label.nii.gz",
            f"{case_id}_mask.nii.gz",
            f"{case_id}.mask.nii.gz",
        ]

        label_path = None
        for name in label_candidates:
            cand = img_path.with_name(name)
            if cand.exists():
                label_path = cand
                break

        if label_path is None:
            for f in img_path.parent.iterdir():
                if not f.is_file():
                    continue
                fname = f.name
                if fname.lower().endswith('.nii.gz'):
                    fbase = fname[:-7]
                else:
                    fbase = f.stem
                if fbase.lower().startswith(case_id.lower()) and re.search(r"label|mask", fbase.lower()):
                    label_path = f
                    break

        if label_path is None:
            logger.warning(f"Label missing for {case_id}: looked for {label_candidates}")
            continue

        cases.append({
            "id": case_id,
            "image": str(img_path.resolve()),
            "label": str(label_path.resolve())
        })

    logger.info(f"Discovered {len(cases)} valid cases")
    return cases

def load_official_split(
    split_xlsx: Path, 
    split_id: int, 
    logger: logging.Logger
) -> Tuple[set, set, set]:
    """Load ImageCAS official split from Excel."""
    logger.info(f"Loading split from {split_xlsx}, split_id={split_id}")
    
    df = pd.read_excel(split_xlsx)
    
    id_col = None
    for col in df.columns:
        if any(x in str(col).lower() for x in ["id", "file", "case"]):
            id_col = col
            break
    if id_col is None:
        raise ValueError("Could not find ID column in split file")
    
    split_col_name = f"split-{split_id}"
    split_col = None
    for col in df.columns:
        if str(col).lower() == split_col_name.lower():
            split_col = col
            break
    if split_col is None:
        raise ValueError(f"Could not find {split_col_name} column")
    
    train_ids, val_ids, test_ids = set(), set(), set()
    
    for _, row in df.iterrows():
        case_id = str(row[id_col]).split('.')[0]
        split_val = str(row[split_col]).strip().lower()
        
        if split_val in ("train", "training"):
            train_ids.add(case_id)
        elif split_val in ("val", "valid", "validation"):
            val_ids.add(case_id)
        elif split_val == "test":
            test_ids.add(case_id)
    
    logger.info(f"Split loaded: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    return train_ids, val_ids, test_ids

def split_dataset(
    cases: List[Dict],
    split_path: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    logger: logging.Logger,
    official_split: Optional[Tuple[set, set, set]] = None,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset either from official split or randomly."""
    case_dict = {c["id"]: c for c in cases}
    
    if official_split:
        train_ids, val_ids, test_ids = official_split
        logger.info("Using official split from Excel")
    else:
        logger.info(f"Creating random split with val_ratio={val_ratio}, test_ratio={test_ratio}, seed={seed}")
        case_ids = list(case_dict.keys())
        random.seed(seed)
        random.shuffle(case_ids)
        
        n_test = int(len(case_ids) * test_ratio)
        n_val = int(len(case_ids) * val_ratio)
        
        test_ids = set(case_ids[:n_test])
        val_ids = set(case_ids[n_test:n_test + n_val])
        train_ids = set(case_ids[n_test + n_val:])
    
    train_cases = [case_dict[i] for i in sorted(train_ids) if i in case_dict]
    val_cases = [case_dict[i] for i in sorted(val_ids) if i in case_dict]
    test_cases = [case_dict[i] for i in sorted(test_ids) if i in case_dict]
    
    missing_train = train_ids - set(case_dict.keys())
    missing_val = val_ids - set(case_dict.keys())
    if missing_train or missing_val:
        logger.warning(f"Missing from dataset - train: {sorted(missing_train)[:10]}, val: {sorted(missing_val)[:10]}")
    
    if limit_train:
        train_cases = train_cases[:limit_train]
    if limit_val:
        val_cases = val_cases[:limit_val]
    
    split_data = {"train": train_cases, "val": val_cases, "test": test_cases}
    with open(split_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    logger.info(f"Final split: train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}")
    return train_cases, val_cases, test_cases

# Transforms

def get_transforms(config: Config, mode: str = "train"):
    """Build MONAI transforms pipeline."""
    assert mode in ["train", "val"], f"Invalid mode: {mode}"
    
    transforms = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=config.pixdim,
            mode=("bilinear", "nearest"),
        ),
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
    
    transforms.append(Lambdad(keys=["label"], func=lambda x: (x > 0).astype(np.float32)))
    
    if mode == "train":
        transforms.extend([
            SpatialPadd(keys=["image", "label"], spatial_size=config.roi_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=config.roi_size,
                pos=config.pos_neg_ratio[0],
                neg=config.pos_neg_ratio[1],
                num_samples=config.num_samples,
                image_key="image",
                allow_smaller=True,
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandRotated(
                keys=["image", "label"],
                prob=0.3,
                range_x=0.2,
                range_y=0.2,
                range_z=0.2,
                mode=("bilinear", "nearest"),
            ),
            Rand3DElasticd(
                keys=["image", "label"],
                prob=0.2,
                sigma_range=(4, 6),
                magnitude_range=(50, 150),
                mode=("bilinear", "nearest"),
            ),
            RandAdjustContrastd(keys=["image"], prob=0.4, gamma=(0.8, 1.2)),
            RandScaleIntensityd(keys=["image"], factors=(0.85, 1.15), prob=0.5),
            RandShiftIntensityd(
                keys=["image"],
                prob=0.3,
                offsets=0.1,
            ),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.3,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config.roi_size),
        ])
    
    transforms.append(EnsureTyped(keys=["image", "label"], dtype=torch.float32))
    
    return Compose(transforms)

# MODEL

def build_model(config: Config, logger: logging.Logger):
    """Build 3D U-Net or Attention U-Net model."""
    logger.info(
        f"Building model: channels={config.unet_channels}, "
        f"res_units={config.unet_num_res_units}, dropout={config.unet_dropout}"
    )
    
    if config.use_attention:
        if config.device == "mps":
            logger.warning("Attention U-Net disabled on MPS (memory constraints)")
            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=config.unet_channels,
                strides=config.unet_strides,
                num_res_units=config.unet_num_res_units,
                dropout=config.unet_dropout,
                norm=config.unet_norm,
            )
        else:
            try:
                model = AttentionUnet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=1,
                    channels=config.unet_channels,
                    strides=config.unet_strides,
                    dropout=config.unet_dropout,
                )
                logger.info(" Using Attention U-Net")
            except Exception as e:
                logger.warning(f"Attention U-Net failed: {e}, using standard U-Net")
                model = UNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=1,
                    channels=config.unet_channels,
                    strides=config.unet_strides,
                    num_res_units=config.unet_num_res_units,
                    dropout=config.unet_dropout,
                    norm=config.unet_norm,
                )
    else:
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=config.unet_channels,
            strides=config.unet_strides,
            num_res_units=config.unet_num_res_units,
            dropout=config.unet_dropout,
            norm=config.unet_norm,
        )
    
    model = model.to(config.device, dtype=torch.float32)
    
    try:
        for module in model.modules():
            if isinstance(module, nn.Conv3d) and module.out_channels == 1:
                nn.init.constant_(module.bias, -4.0)
                logger.info("Initialized final conv bias for stable sigmoid")
                break
    except Exception as e:
        logger.warning(f"Could not initialize final bias: {e}")
    
    if config.compile_model and hasattr(torch, "compile") and config.device.startswith("cuda"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")
    
    return model

# Loss and Optimization

class DiceLoss(nn.Module):
    """Dice loss with numerical stability."""
    def __init__(self, eps: float = 1e-6, include_background: bool = False):
        super().__init__()
        self.eps = eps
        self.include_background = include_background
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target_bin = (target > 0.5).float()
        
        if not self.include_background:
            probs = probs.flatten(2)
            target_bin = target_bin.flatten(2)
        
        intersection = (probs * target_bin).sum()
        dice_score = (2.0 * intersection + self.eps) / (probs.sum() + target_bin.sum() + self.eps)
        return 1.0 - dice_score

class TverskyLoss(nn.Module):
    """Tversky loss for imbalance."""
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        target_bin = (target > 0.5).float()
        
        tp = (probs * target_bin).sum()
        fp = (probs * (1 - target_bin)).sum()
        fn = ((1 - probs) * target_bin).sum()
        
        tversky = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        return 1.0 - tversky

class FocalLoss(nn.Module):
    """Focal loss for hard examples."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction='none'
        )
        probs = torch.sigmoid(logits)
        p_t = probs * target + (1 - probs) * (1 - target)
        loss = bce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        
        return loss.mean()

def compute_class_weights(
    train_cases: List[Dict],
    config: Config,
    logger: logging.Logger
) -> torch.Tensor:
    """Compute positive weight from dataset statistics."""
    if config.pos_weight_fixed is not None:
        logger.info(f"Using fixed pos_weight: {config.pos_weight_fixed}")
        return torch.tensor([config.pos_weight_fixed], device=config.device)
    
    logger.info("Computing class weights from training set...")
    pos_voxels = 0
    total_voxels = 0
    
    for case in tqdm(train_cases, desc="Analyzing labels", unit="case"):
        try:
            label = nib.load(case["label"]).get_fdata()
            pos = np.sum(label > 0)
            pos_voxels += pos
            total_voxels += label.size
        except Exception as e:
            logger.warning(f"Failed to load {case['label']}: {e}")
    
    if pos_voxels == 0:
        logger.warning("No positive voxels found! Using pos_weight=1.0")
        return torch.tensor([1.0], device=config.device)
    
    neg_pos_ratio = (total_voxels - pos_voxels) / pos_voxels
    pos_weight = min(max(1.0, neg_pos_ratio), config.pos_weight_cap)
    
    logger.info(
        f"Pos weight: pos_voxels={pos_voxels:,}, neg_pos_ratio={neg_pos_ratio:.2f}, "
        f"capped_pos_weight={pos_weight:.2f}"
    )
    return torch.tensor([pos_weight], device=config.device, dtype=torch.float32)

def build_optimizer_scheduler(model: nn.Module, config: Config, logger: logging.Logger = None):
    """Build AdamW optimizer with EXPONENTIAL decay."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    if logger:
        logger.info("Using ExponentialLR scheduler (gamma=0.95) - stable decay, no oscillation")
    
    return optimizer, scheduler

def build_loss_fn(config: Config, pos_weight: torch.Tensor):
    """Build ENHANCED composite loss function with Dice + BCE + Focal."""
    components = []
    
    dice_loss = DiceLoss(eps=1e-6)
    components.append(("dice", dice_loss, 0.5))
    
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    components.append(("bce", bce_loss, 0.3))
    
    if config.use_focal_loss:
        focal_loss = FocalLoss(alpha=0.25, gamma=config.focal_gamma, pos_weight=pos_weight)
        components.append(("focal", focal_loss, 0.2))
    else:
        tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
        components.append(("tversky", tversky_loss, 0.2))
    
    def loss_fn(logits, target):
        total_loss = 0.0
        stats = {}
        
        for name, loss_obj, weight in components:
            loss_val = loss_obj(logits, target)
            total_loss += weight * loss_val
            stats[f"{name}_loss"] = loss_val.detach()
        
        stats.update({
            "logits_min": logits.min().detach(),
            "logits_max": logits.max().detach(),
            "probs_mean": torch.sigmoid(logits).mean().detach(),
        })
        return total_loss, stats
    
    return loss_fn

# ============================================================================
# CHECKPOINTING
# ============================================================================

def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find newest checkpoint_epoch_*.pt file."""
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return checkpoints[0] if checkpoints else None

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_metric: float,
    config: Config,
    scaler: Optional[torch.amp.GradScaler] = None,
    is_best: bool = False,
    logger: Optional[logging.Logger] = None
):
    """Save training checkpoint."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
        "config": config,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    
    if is_best:
        best_path = path.parent / "checkpoint_best.pt"
        best_path.unlink(missing_ok=True)
        best_path.symlink_to(path.name)
        if logger:
            logger.info(f"Saved best checkpoint: {path}")
    else:
        if logger:
            logger.info(f"Saved checkpoint: {path}")

def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: Config,
    scaler: Optional[torch.amp.GradScaler] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Load checkpoint with proper path handling."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if logger:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    torch.serialization.add_safe_globals([
        np._core.multiarray.scalar,
        np.dtype,
    ])
    
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    model.load_state_dict(state["model_state_dict"])
    model.to(config.device)
    
    optimizer.load_state_dict(state["optimizer_state_dict"])
    
    if scheduler and state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
    
    if scaler and state.get("scaler_state_dict"):
        try:
            scaler.load_state_dict(state["scaler_state_dict"])
        except Exception as e:
            if logger:
                logger.warning(f"Could not load scaler state: {e}")
    
    return state

def resolve_checkpoint_path(resume: str, checkpoint_dir: Path) -> Path:
    """Resolve checkpoint path safely."""
    if not resume or resume == "":
        raise ValueError("No resume path specified")
    
    if resume.lower() == "latest":
        latest = find_latest_checkpoint(checkpoint_dir)
        if not latest:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        return latest
    
    path = Path(resume)
    if path.is_absolute():
        return path
    
    cwd_path = Path.cwd() / resume
    if cwd_path.exists():
        return cwd_path.resolve()
    
    fallback = checkpoint_dir / resume
    if fallback.exists():
        return fallback
    
    raise FileNotFoundError(f"Checkpoint not found: {resume} (tried {cwd_path} and {fallback})")


# Validation and metrics

def cleanup_connected_components(
    mask: torch.Tensor,
    min_size: int,
    logger: logging.Logger
) -> torch.Tensor:
    """Remove small connected components."""
    if min_size <= 0:
        return mask
    
    try:
        from scipy.ndimage import label, sum as ndi_sum
        from monai.transforms import RemoveSmallObjects
    except ImportError:
        logger.warning("scipy/monai unavailable for CC cleanup")
        return mask
    
    cleaned = mask.clone()
    for b in range(mask.shape[0]):
        arr = mask[b, 0].cpu().numpy()
        cleaned_arr = RemoveSmallObjects(min_size=min_size)(arr > 0)
        cleaned[b, 0] = torch.from_numpy(cleaned_arr.astype(np.float32))
    
    return cleaned

def compute_validation_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    metrics: Dict[str, Any],
    min_cc_size: int,
    logger: logging.Logger
) -> Dict[str, float]:
    """Compute validation metrics."""
    results = {}
    
    probs = torch.sigmoid(preds).clamp(1e-6, 1-1e-6)
    soft_dice = (2.0 * (probs * labels).sum() + 1e-6) / (probs.sum() + labels.sum() + 1e-6)
    results["soft_dice"] = soft_dice.item()
    
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        binary_preds = (probs > thresh).float()
        dice = (2.0 * (binary_preds * labels).sum() + 1e-6) / (binary_preds.sum() + labels.sum() + 1e-6)
        results[f"dice@{thresh:.1f}"] = dice.item()
    
    cleaned_preds = cleanup_connected_components(binary_preds, min_cc_size, logger)
    
    if "hd95" in metrics:
        try:
            preds_onehot = torch.cat([1-cleaned_preds, cleaned_preds], dim=1)
            labels_onehot = torch.cat([1-labels, labels], dim=1)
            metrics["hd95"](preds_onehot, labels_onehot)
            hd95_val = metrics["hd95"].aggregate()
            if isinstance(hd95_val, tuple):
                hd95_val = hd95_val[0]
            results["hd95"] = hd95_val.item()
        except Exception as e:
            logger.debug(f"HD95 computation failed: {e}")
            results["hd95"] = float("nan")
    
    flat_probs = probs.flatten()
    results["prob_mean"] = flat_probs.mean().item()
    results["prob_median"] = flat_probs.median().item()
    results["voxels>0.1"] = (flat_probs > 0.1).sum().item()
    results["voxels>0.5"] = (flat_probs > 0.5).sum().item()
    
    return results

# Training

class Trainer:
    """Main training engine with AMP and history tracking."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = config.device
        self.use_amp = config.amp and self.device.startswith("cuda")
        
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp) if self.use_amp else None
        self.autocast_dtype = torch.float16
        if config.amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
            self.autocast_dtype = torch.bfloat16
        elif config.amp_dtype == "fp16":
            self.autocast_dtype = torch.float16
        
        self.best_metric = -float("inf")
        self.best_epoch = -1
        self.epoch = 0
        self.global_step = 0
        
    def train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        overfit_one: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """Train one epoch."""
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        grad_norms = []
        all_loss_stats = defaultdict(list)
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.autocast_dtype):
                logits = model(images)
                loss, loss_stats = loss_fn(logits, labels)
            
            if not torch.isfinite(loss):
                self.logger.error(f"Non-finite loss at batch {batch_idx}: {loss}")
                if self.config.skip_nonfinite_batches:
                    continue
                else:
                    raise RuntimeError("Non-finite loss encountered")
            
            loss = loss / self.config.accumulation_steps
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.grad_clip_norm > 0:
                    if self.scaler:
                        self.scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.grad_clip_norm
                    )
                    grad_norms.append(grad_norm.item())
                
                if self.scaler:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            
            epoch_loss += loss.item() * self.config.accumulation_steps
            num_batches += 1
            
            for k, v in loss_stats.items():
                all_loss_stats[k].append(v.item() if torch.is_tensor(v) else v)
            
            pbar.set_postfix({
                "loss": f"{loss_stats.get('bce_loss', loss):.4f}",
                "dice": f"{1-loss_stats.get('dice_loss', 0):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })
            
            if overfit_one and batch_idx >= 0:
                break
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR):
            scheduler.step()
        
        if grad_norms:
            self.logger.info(
                f"Epoch {epoch} grad_norm: mean={np.mean(grad_norms):.2f}, "
                f"max={np.max(grad_norms):.2f}"
            )
        
        # Aggregate training stats
        train_stats = {k: np.mean(v) for k, v in all_loss_stats.items()}
        train_stats["epoch_loss"] = epoch_loss / max(1, num_batches)
        
        return train_stats["epoch_loss"], train_stats
    
    def validate(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_fn,
        metrics: Dict[str, Any],
        epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        """Validate with sliding window inference."""
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Epoch {epoch} Val", leave=False):
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
                
                logits = sliding_window_inference(
                    images,
                    roi_size=self.config.roi_size,
                    sw_batch_size=self.config.sw_batch_size,
                    predictor=model,
                    overlap=self.config.sw_overlap,
                    mode="gaussian",
                )
                
                loss, _ = loss_fn(logits, labels)
                val_loss += loss.item()
                val_batches += 1
                
                metrics_dict = compute_validation_metrics(
                    logits, labels, metrics, self.config.min_cc_size, self.logger
                )
                for k, v in metrics_dict.items():
                    if np.isfinite(v):
                        all_metrics[k].append(v)
                
                if self.config.overfit_one:
                    break
        
        final_metrics = {k: np.mean(vs) for k, vs in all_metrics.items()}
        final_metrics["val_loss"] = val_loss / max(1, val_batches)
        
        return final_metrics["val_loss"], final_metrics
    
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn,
        metrics: Dict[str, Any],
        start_epoch: int = 0,
        overfit_one: bool = False
    ) -> Dict[str, Any]:
        """Main training loop with detailed history tracking."""
        self.epoch = start_epoch
        
        # Detailed history tracking
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": defaultdict(list),
            "val_metrics": defaultdict(list),
            "lr": [],
            "best_metric": [],
            "best_epoch": [],
        }
        
        for epoch in range(start_epoch + 1, self.config.epochs + 1):
            self.epoch = epoch
            
            # Train
            train_loss, train_stats = self.train_epoch(
                model, train_loader, optimizer, loss_fn, scheduler, epoch, overfit_one
            )
            
            history["train_loss"].append(train_loss)
            history["lr"].append(optimizer.param_groups[0]['lr'])
            for k, v in train_stats.items():
                if k != "epoch_loss":
                    history["train_metrics"][k].append(v)
            
            # Validate
            val_loss = float("nan")
            val_metrics = {}
            
            if val_loader and not overfit_one:
                val_loss, val_metrics = self.validate(
                    model, val_loader, loss_fn, metrics, epoch
                )
                
                history["val_loss"].append(val_loss)
                for k, v in val_metrics.items():
                    history["val_metrics"][k].append(v)
                
                if self.config.save_val_preds and epoch % self.config.save_val_every == 0:
                    self.save_validation_predictions(
                        model, val_loader, epoch, self.config.checkpoint_dir, self.logger
                    )
            
            # Logging
            self.log_epoch_summary(epoch, train_loss, train_stats, val_loss, val_metrics, optimizer)
            
            # Checkpoint
            is_best = False
            current_metric = val_metrics.get("dice@0.5", -float("inf"))
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                is_best = True
            
            history["best_metric"].append(self.best_metric)
            history["best_epoch"].append(self.best_epoch)
            
            save_checkpoint(
                self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
                model, optimizer, scheduler, epoch, self.best_metric,
                self.config, self.scaler, is_best, self.logger
            )
            
            if overfit_one:
                if train_loss < 0.01:
                    self.logger.info("✅ Overfit test PASSED (loss < 0.01)")
                    break
                if epoch >= 30 and train_loss > 0.1:
                    self.logger.error("❌ Overfit test FAILED (loss > 0.1 after 30 epochs)")
                    break
        
        # Convert defaultdict to regular dict for JSON serialization
        history["train_metrics"] = dict(history["train_metrics"])
        history["val_metrics"] = dict(history["val_metrics"])
        
        return history
    
    def log_epoch_summary(self, epoch, train_loss, train_stats, val_loss, val_metrics, optimizer):
        """Log epoch summary with key metrics."""
        parts = [
            f"Epoch {epoch:03d}",
            f"train_loss={train_loss:.4f}",
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
        ]
        
        # Add training stats
        if "dice_loss" in train_stats:
            parts.append(f"train_dice={1-train_stats['dice_loss']:.4f}")
        if "bce_loss" in train_stats:
            parts.append(f"train_bce={train_stats['bce_loss']:.4f}")
        
        if val_metrics:
            parts.append(f"val_loss={val_loss:.4f}")
            if "soft_dice" in val_metrics:
                parts.append(f"val_soft_dice={val_metrics['soft_dice']:.4f}")
            if "dice@0.5" in val_metrics:
                parts.append(f"val_dice@0.5={val_metrics['dice@0.5']:.4f}")
            if "dice@0.1" in val_metrics:
                parts.append(f"val_dice@0.1={val_metrics['dice@0.1']:.4f}")
            if "prob_mean" in val_metrics:
                parts.append(f"prob_mean={val_metrics['prob_mean']:.4f}")
        
        self.logger.info(" | ".join(parts))
    
    def save_validation_predictions(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        epoch: int,
        checkpoint_dir: Path,
        logger: logging.Logger
    ):
        """Save validation predictions as NIfTI."""
        save_dir = checkpoint_dir / "val_preds" / f"epoch_{epoch:04d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if count >= self.config.save_val_preds_n:
                    break
                
                images = batch["image"].to(self.device)
                labels = batch["label"]
                
                logits = sliding_window_inference(
                    images,
                    roi_size=self.config.roi_size,
                    sw_batch_size=self.config.sw_batch_size,
                    predictor=model,
                    overlap=self.config.sw_overlap,
                    mode="gaussian",
                )
                
                probs = torch.sigmoid(logits)
                preds = (probs > self.config.save_val_threshold).float()
                
                meta = batch.get("image_meta_dict", {})
                affine = meta.get("affine", None)
                
                if affine is not None:
                    if isinstance(affine, torch.Tensor):
                        affine = affine.detach().cpu().numpy()
                    if affine.shape == (1, 4, 4):
                        affine = affine[0]
                    elif affine.shape != (4, 4):
                        affine = np.eye(4)
                else:
                    affine = np.eye(4)
                
                case_id = batch.get("id", [f"case_{count}"])[0]
                pred_path = save_dir / f"{case_id}_pred.nii.gz"
                prob_path = save_dir / f"{case_id}_prob.nii.gz"
                
                nib.Nifti1Image(preds[0, 0].cpu().numpy(), affine).to_filename(str(pred_path))
                nib.Nifti1Image(probs[0, 0].cpu().numpy(), affine).to_filename(str(prob_path))
                
                count += 1
        
        logger.info(f"Saved {count} validation predictions to {save_dir}")

# Main Pipeline

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ImageCAS 3D Vascular Segmentation Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Plotting argument
    parser.add_argument("--plot_history", type=Path, help="Plot training history from JSON file and exit")
    
    # Dataset preset
    parser.add_argument("--dataset_preset", type=str, default="custom", choices=["custom", "imagecas"])
    
    # Data arguments
    parser.add_argument("--imagecas_root", type=Path)
    parser.add_argument("--use_official_split", action="store_true")
    parser.add_argument("--split_xlsx", type=Path)
    parser.add_argument("--split_id", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument("--limit_train", type=int)
    parser.add_argument("--limit_val", type=int, default=30)
    
    # Preprocessing arguments
    parser.add_argument("--modality", type=str, default="ct", choices=["ct", "mri"])
    parser.add_argument("--ct_window", type=str, default="-200,700")
    parser.add_argument("--pixdim", type=str, default="0.6,0.6,0.6")
    parser.add_argument("--roi_size", type=str, default="96,192,192")
    parser.add_argument("--roi_multiple", type=int, default=16)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    
    # Sampling arguments
    parser.add_argument("--pos_ratio", type=int, default=1)
    parser.add_argument("--neg_ratio", type=int, default=0)
    parser.add_argument("--pos_neg_ratio", type=str)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--force_pos_patches", action="store_true")
    parser.add_argument("--min_pos_voxels", type=int, default=400)
    
    # Model arguments
    parser.add_argument("--unet_channels", type=str, default="64,128,256,512,1024")
    parser.add_argument("--unet_res_units", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_attention", action="store_true", default=True)
    parser.add_argument("--compile_model", action="store_true")
    
    # Loss arguments
    parser.add_argument("--loss_mode", type=str, default="all", choices=["all", "dice", "bce", "dice_focal"])
    parser.add_argument("--use_focal_loss", action="store_true", default=True)
    parser.add_argument("--pos_weight_cap", type=float, default=10.0)
    parser.add_argument("--pos_weight_fixed", type=float)
    
    # Metrics arguments
    parser.add_argument("--prob_threshold", type=float, default=0.15)
    parser.add_argument("--save_val_threshold", type=float, default=0.15)
    parser.add_argument("--min_cc_size", type=int, default=50)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    
    # Data loading
    parser.add_argument("--cache_rate_train", type=float, default=0.1)
    parser.add_argument("--cache_rate_val", type=float, default=0.25)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--resume", type=str)
    
    # Validation outputs
    parser.add_argument("--save_val_preds", action="store_true")
    parser.add_argument("--save_val_preds_n", type=int, default=5)
    
    # Sanity modes
    parser.add_argument("--overfit_one", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verify_data", action="store_true")
    parser.add_argument("--verify_n", type=int, default=3)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--skip_nonfinite_batches", action="store_true", default=True)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    parser.add_argument("--experiment_name", type=str)
    
    args = parser.parse_args()
    
    # Handle plotting mode
    if args.plot_history:
        plot_training_history(args.plot_history)
        return
    
    # Build config from args
    config = Config()
    
    if hasattr(args, 'dataset_preset') and args.dataset_preset:
        config.dataset_preset = args.dataset_preset
    
    config.imagecas_root = args.imagecas_root
    config.checkpoint_dir = args.checkpoint_dir
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.grad_clip_norm = args.grad_clip_norm
    
    if hasattr(args, 'roi_size') and args.roi_size:
        config.roi_size = tuple(map(int, args.roi_size.split(',')))
    
    if hasattr(args, 'ct_window') and args.ct_window:
        config.ct_window = tuple(map(float, args.ct_window.split(',')))
    
    if hasattr(args, 'pixdim') and args.pixdim:
        config.pixdim = tuple(map(float, args.pixdim.split(',')))
    
    if args.pos_neg_ratio:
        pos, neg = map(int, args.pos_neg_ratio.split(','))
        config.pos_neg_ratio = (pos, neg)
    else:
        config.pos_neg_ratio = (args.pos_ratio, args.neg_ratio)
    
    config.num_samples = args.num_samples
    config.force_pos_patches = args.force_pos_patches
    config.min_pos_voxels = args.min_pos_voxels
    
    if args.unet_channels:
        config.unet_channels = tuple(map(int, args.unet_channels.split(',')))
    
    config.unet_num_res_units = args.unet_res_units
    config.unet_dropout = args.dropout
    config.use_attention = args.use_attention
    config.compile_model = args.compile_model
    
    config.loss_mode = args.loss_mode
    config.use_focal_loss = args.use_focal_loss
    config.pos_weight_cap = args.pos_weight_cap
    config.pos_weight_fixed = args.pos_weight_fixed
    
    config.prob_threshold = args.prob_threshold
    config.save_val_threshold = args.save_val_threshold
    config.min_cc_size = args.min_cc_size
    config.sw_batch_size = args.sw_batch_size
    
    config.cache_rate_train = args.cache_rate_train
    config.cache_rate_val = args.cache_rate_val
    config.num_workers = args.num_workers
    
    config.seed = args.seed
    config.deterministic = args.deterministic
    config.skip_nonfinite_batches = args.skip_nonfinite_batches
    config.amp = args.amp
    config.amp_dtype = args.amp_dtype
    
    config.split_id = args.split_id
    config.val_ratio = args.val_ratio
    config.test_ratio = args.test_ratio
    config.limit_train = args.limit_train
    config.limit_val = args.limit_val
    config.save_val_preds = args.save_val_preds
    config.save_val_preds_n = args.save_val_preds_n
    
    config.overfit_one = args.overfit_one
    config.dry_run = args.dry_run
    config.verify_data = args.verify_data
    config.verify_n = args.verify_n
    
    config.checkpoint_dir = config.checkpoint_dir or config.project_root / "checkpoints"
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir = config.log_dir or config.project_root / "logs"
    config.log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(config.log_dir, args.experiment_name)
    
    set_all_seeds(config.seed, config.deterministic)
    
    logger.info("=" * 80)
    logger.info("STARTING VASCULAR SEGMENTATION TRAINING")
    logger.info("=" * 80)
    logger.info(f"Experiment: {args.experiment_name or 'unnamed'}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Dataset preset: {config.dataset_preset}")
    logger.info(f"Modality: {config.modality}")
    logger.info(f"ROI size: {config.roi_size}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Pos/neg ratio: {config.pos_neg_ratio}")
    logger.info(f"Force pos patches: {config.force_pos_patches}")
    logger.info(f"Min pos voxels: {config.min_pos_voxels}")
    logger.info(f"AMP: {config.amp}, dtype: {config.amp_dtype}")
    logger.info(f"Checkpoint dir: {config.checkpoint_dir}")
    
    logger.info("Phase 1: Data discovery")
    all_cases = discover_cases(config.imagecas_root, logger)
    
    if not all_cases:
        logger.error("No valid cases found!")
        sys.exit(1)
    
    official_split = None
    if args.use_official_split:
        if not args.split_xlsx or not args.split_xlsx.exists():
            logger.error("Official split requested but split_xlsx not found")
            sys.exit(1)
        official_split = load_official_split(args.split_xlsx, config.split_id, logger)
    
    logger.info("Phase 2: Dataset splitting")
    split_path = config.checkpoint_dir / "splits.json"
    train_cases, val_cases, test_cases = split_dataset(
        all_cases, split_path, config.val_ratio, config.test_ratio,
        config.seed, logger, official_split,
        config.limit_train, config.limit_val
    )
    
    if config.overfit_one:
        logger.warning("🔬 OVERFIT ONE MODE: Using single case for train/val")
        train_cases = train_cases[:1]
        val_cases = train_cases[:1]
        config.epochs = 50
        config.save_val_preds = True
    
    if config.verify_data:
        logger.info("Phase 3: Data verification")
        try:
            from monai.transforms import Compose
            verify_transform = get_transforms(config, mode="train")
            for case in train_cases[:config.verify_n]:
                sample = verify_transform({"image": case["image"], "label": case["label"]})
                logger.info(f"✅ Verified: {case['id']} -> {sample['image'].shape}")
            logger.info("Data verification PASSED")
        except Exception as e:
            logger.error(f"Data verification FAILED: {e}")
            sys.exit(1)
        return
    
    logger.info("Phase 3: Building dataloaders")
    
    train_transform = get_transforms(config, mode="train")
    val_transform = get_transforms(config, mode="val")
    
    logger.info(f"Creating training dataset (cache_rate={config.cache_rate_train})...")
    train_ds = CacheDataset(
        data=train_cases,
        transform=train_transform,
        cache_rate=config.cache_rate_train,
        num_workers=config.num_workers,
    )
    
    logger.info(f"Creating validation dataset (cache_rate={config.cache_rate_val})...")
    val_ds = CacheDataset(
        data=val_cases,
        transform=val_transform,
        cache_rate=config.cache_rate_val,
        num_workers=max(1, config.num_workers // 2),
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=pad_list_data_collate,
        pin_memory=config.device.startswith("cuda"),
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        collate_fn=pad_list_data_collate,
        pin_memory=config.device.startswith("cuda"),
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    logger.info("Phase 4: Building model and optimizer")
    model = build_model(config, logger)
    
    pos_weight = compute_class_weights(train_cases, config, logger)
    
    loss_fn = build_loss_fn(config, pos_weight)
    
    optimizer, scheduler = build_optimizer_scheduler(model, config, logger)
    
    metrics = {}
    if not config.overfit_one:
        metrics = {
            "hd95": HausdorffDistanceMetric(
                include_background=False,
                percentile=95,
                reduction="mean",
            )
        }
    
    start_epoch = 0
    
    if args.resume:
        try:
            ckpt_path = resolve_checkpoint_path(args.resume, config.checkpoint_dir)
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
            
            state = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, config, None, logger
            )
            start_epoch = state.get("epoch", 0)
            logger.info(f"Checkpoint loaded (epoch {start_epoch})")
            
        except Exception as e:
            logger.error(f"Failed to resume: {e}")
            if "does not match" in str(e):
                logger.error("Model architecture mismatch. Check --unet_channels and similar flags.")
            sys.exit(1)
    
    if config.dry_run:
        logger.info("🔍 DRY RUN: Testing forward/backward pass")
        model.train()
        batch = next(iter(train_loader))
        
        images = batch["image"].to(config.device)
        labels = batch["label"].to(config.device)
        
        logger.info(f"Input shape: {images.shape}, dtype: {images.dtype}")
        logger.info(f"Label shape: {labels.shape}, dtype: {labels.dtype}")
        
        with torch.amp.autocast("cuda", enabled=config.amp):
            logits = model(images)
            loss, stats = loss_fn(logits, labels)
        
        logger.info(f"Output shape: {logits.shape}")
        logger.info(f"Loss: {loss.item():.4f}, finite: {torch.isfinite(loss)}")
        logger.info(f"Loss components: {list(stats.keys())}")
        
        loss.backward()
        logger.info("✅ Backward pass successful")
        
        model.eval()
        with torch.no_grad():
            val_batch = next(iter(val_loader))
            val_images = val_batch["image"].to(config.device)
            val_logits = sliding_window_inference(
                val_images, config.roi_size, config.sw_batch_size, model,
                overlap=config.sw_overlap, mode="gaussian",
            )
            logger.info(f"Validation logits shape: {val_logits.shape}")
        
        logger.info("Dry run completed successfully")
        return
    
    logger.info("Phase 5: Starting training loop")
    trainer = Trainer(config, logger)
    
    try:
        history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            metrics=metrics,
            start_epoch=start_epoch,
            overfit_one=config.overfit_one,
        )
        
        # Save detailed history
        history_path = config.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Best validation dice@0.5: {trainer.best_metric:.4f} at epoch {trainer.best_epoch}")
        logger.info(f"Final checkpoint: {config.checkpoint_dir / 'checkpoint_best.pt'}")
        logger.info(f"Training history: {history_path}")
        
        # Auto-plot at the end
        try:
            plot_path = history_path.parent / f"{history_path.stem}_plot.png"
            plot_training_history(history_path, plot_path, logger)
        except Exception as e:
            logger.warning(f"Could not auto-plot: {e}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
