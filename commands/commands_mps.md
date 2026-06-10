# MPS Commands For Reviewer Item 1

## 2-case smoke test

```bash
cd /path/to/cloud_bundle
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.68 MPLCONFIGDIR=/tmp/mplconfig \
python evaluate_full_imagecas_test_mps.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --split-file splits.json \
  --data-root Data/all \
  --outdir outputs/full_test_eval_mps_smoke \
  --device auto \
  --limit 2 \
  --roi-size 96,192,192 \
  --sw-batch-size 1 \
  --num-workers 0
```

## Full 250-case test evaluation

```bash
cd /path/to/cloud_bundle
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.68 MPLCONFIGDIR=/tmp/mplconfig \
python evaluate_full_imagecas_test_mps.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --split-file splits.json \
  --data-root Data/all \
  --outdir outputs/full_test_eval_mps \
  --device auto \
  --roi-size 96,192,192 \
  --sw-batch-size 1 \
  --num-workers 0 \
  --resume \
  --skip-existing
```

Lower-memory fallback:

```bash
cd /path/to/cloud_bundle
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.68 MPLCONFIGDIR=/tmp/mplconfig \
python evaluate_full_imagecas_test_mps.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --split-file splits.json \
  --data-root Data/all \
  --outdir outputs/full_test_eval_mps \
  --device auto \
  --roi-size 64,128,128 \
  --sw-batch-size 1 \
  --num-workers 0 \
  --resume \
  --skip-existing
```

## Local Mac MPS training/resume

The repo-local checkpoints are currently truncated, so this is a fresh retrain command. The training log evidence for the submitted Attention U-Net is 23,625,953 trainable parameters; with this MONAI version that corresponds to `--unet_channels 32,64,128,256,512` for `AttentionUnet`. The command writes into a new checkpoint directory and reuses the existing `splits.json` so the held-out 250-case test partition is preserved.

One-epoch checkpoint smoke test:

```bash
cd /path/to/cloud_bundle
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.68 MPLCONFIGDIR=/tmp/mplconfig \
python "train_updated copy.py" \
  --dataset_preset imagecas \
  --imagecas_root Data/all \
  --split_file splits.json \
  --checkpoint_dir outputs/checkpoints/bioprint_v5_mps_checkpoint_smoke \
  --experiment_name bioprint_v5_mps_checkpoint_smoke \
  --device mps \
  --epochs 1 \
  --limit_train 2 \
  --limit_val 1 \
  --batch_size 1 \
  --roi_size 64,128,128 \
  --unet_channels 32,64,128,256,512 \
  --unet_res_units 3 \
  --use_attention \
  --grad_checkpoint \
  --pixdim 0.6,0.6,0.6 \
  --ct_window=-200,700 \
  --pos_neg_ratio 1,0 \
  --force_pos_patches \
  --min_pos_voxels 400 \
  --pos_weight_cap 200 \
  --sw_batch_size 1 \
  --val_output_device cpu \
  --num_workers 0 \
  --cache_rate_train 0.0 \
  --cache_rate_val 0.0
```

Mac MPS fresh retrain:

Use this as the primary local Mac command. The original `96,192,192` training crop OOMs on MPS even with gradient checkpointing, so this keeps the same split, preprocessing, loss weighting, and Attention U-Net architecture while reducing only the training crop size.

```bash
cd /path/to/cloud_bundle
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.68 MPLCONFIGDIR=/tmp/mplconfig \
python "train_updated copy.py" \
  --dataset_preset imagecas \
  --imagecas_root Data/all \
  --split_file splits.json \
  --checkpoint_dir outputs/checkpoints/bioprint_v5_mps_retrain_evalprep_lowmem \
  --experiment_name bioprint_v5_mps_retrain_evalprep_lowmem \
  --device mps \
  --epochs 100 \
  --batch_size 1 \
  --roi_size 64,128,128 \
  --unet_channels 32,64,128,256,512 \
  --unet_res_units 3 \
  --use_attention \
  --grad_checkpoint \
  --pixdim 0.6,0.6,0.6 \
  --ct_window=-200,700 \
  --pos_neg_ratio 1,0 \
  --force_pos_patches \
  --min_pos_voxels 400 \
  --pos_weight_cap 200 \
  --sw_batch_size 1 \
  --val_output_device cpu \
  --num_workers 0 \
  --cache_rate_train 0.0 \
  --cache_rate_val 0.0
```

Original ROI command for a larger-memory Mac:

```bash
cd /path/to/cloud_bundle
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.68 MPLCONFIGDIR=/tmp/mplconfig \
python "train_updated copy.py" \
  --dataset_preset imagecas \
  --imagecas_root Data/all \
  --split_file splits.json \
  --checkpoint_dir outputs/checkpoints/bioprint_v5_mps_retrain_evalprep_original_roi \
  --experiment_name bioprint_v5_mps_retrain_evalprep_original_roi \
  --device mps \
  --epochs 100 \
  --batch_size 1 \
  --roi_size 96,192,192 \
  --unet_channels 32,64,128,256,512 \
  --unet_res_units 3 \
  --use_attention \
  --grad_checkpoint \
  --pixdim 0.6,0.6,0.6 \
  --ct_window=-200,700 \
  --pos_neg_ratio 1,0 \
  --force_pos_patches \
  --min_pos_voxels 400 \
  --pos_weight_cap 200 \
  --sw_batch_size 1 \
  --val_output_device cpu \
  --num_workers 0 \
  --cache_rate_train 0.0 \
  --cache_rate_val 0.0
```
