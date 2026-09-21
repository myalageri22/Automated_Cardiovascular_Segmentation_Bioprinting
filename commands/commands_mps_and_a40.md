# MPS and A40 Commands

Use these from the repository root. The current working model architecture is
Attention U-Net with channels `(32,64,128,256,512)`. Do not change architecture
when resuming from the current Attention U-Net checkpoint. Changing ROI size is
allowed because ROI size does not change checkpoint weight shapes.

Headline manuscript metrics should come from the full 250-case held-out test
set, not training Dice.

## Selected Local Resume Candidate

```text
outputs/checkpoints/bioprint_v5_mps_retrain_evalprep_lowmem/checkpoint_best.pt
```

The checkpoint integrity utility selected this because it is readable, contains
model/optimizer/scheduler/scaler state, and has the best valid local Dice-based
metadata among discovered checkpoints.

## A40 Resume Training

```bash
cd /workspace/cloud_bundle
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128 \
python train_a40_resume.py \
  --experiment_name bioprint_v6_a40_resume \
  --resume_checkpoint auto \
  --model_type attention_unet \
  --roi_size 96,192,192 \
  --batch_size 1 \
  --accumulation_steps 2 \
  --epochs 200 \
  --lr 2e-4 \
  --weight_decay 1e-2 \
  --grad_clip_norm 1.0 \
  --device cuda \
  --num_workers 4 \
  --pin_memory \
  --splits_json splits.json \
  --imagecas_root Data/all \
  --output_dir outputs/train_runs
```

## A40 Clean Restart Fallback

```bash
cd /workspace/cloud_bundle
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128 \
python train_a40_resume.py \
  --experiment_name bioprint_v6_a40_restart_pretrained \
  --resume_checkpoint auto \
  --clean_restart_if_invalid_checkpoint \
  --init_from_pretrained_encoder \
  --model_type attention_unet \
  --roi_size 96,192,192 \
  --batch_size 1 \
  --accumulation_steps 2 \
  --epochs 200 \
  --lr 2e-4 \
  --device cuda \
  --splits_json splits.json \
  --imagecas_root Data/all \
  --output_dir outputs/train_runs
```

## A40 Low-Memory Fallback

```bash
cd /workspace/cloud_bundle
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128 \
python train_a40_resume.py \
  --experiment_name bioprint_v6_a40_lowmem \
  --resume_checkpoint auto \
  --model_type attention_unet \
  --roi_size 64,128,128 \
  --batch_size 1 \
  --accumulation_steps 2 \
  --epochs 200 \
  --device cuda \
  --splits_json splits.json \
  --imagecas_root Data/all \
  --output_dir outputs/train_runs
```

## 2-Case Smoke-Test Evaluation

```bash
cd /workspace/cloud_bundle
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128 \
python evaluate_full_test_a40.py \
  --checkpoint outputs/train_runs/bioprint_v6_a40_resume/checkpoints/best_dice05.pt \
  --splits_json splits.json \
  --data_root Data/all \
  --output_dir outputs/full_test_eval_a40 \
  --device cuda \
  --roi_size 96,192,192 \
  --sw_batch_size 1 \
  --smoke_test \
  --limit 2 \
  --save_case_outputs
```

## Full 250-Case Evaluation

```bash
cd /workspace/cloud_bundle
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128 \
python evaluate_full_test_a40.py \
  --checkpoint outputs/train_runs/bioprint_v6_a40_resume/checkpoints/best_dice05.pt \
  --splits_json splits.json \
  --data_root Data/all \
  --output_dir outputs/full_test_eval_a40 \
  --device cuda \
  --roi_size 96,192,192 \
  --sw_batch_size 1 \
  --resume \
  --skip_existing \
  --save_case_outputs
```

## Full 250-Case Evaluation With Phase B Mesh QC

```bash
cd /workspace/cloud_bundle
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128 \
python evaluate_full_test_a40.py \
  --checkpoint outputs/train_runs/bioprint_v6_a40_resume/checkpoints/best_dice05.pt \
  --splits_json splits.json \
  --data_root Data/all \
  --output_dir outputs/full_test_eval_a40 \
  --device cuda \
  --roi_size 96,192,192 \
  --sw_batch_size 1 \
  --resume \
  --skip_existing \
  --save_case_outputs \
  --run_phase_b_qc
```

## Plain UNet Baseline Training

```bash
cd /workspace/cloud_bundle
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128 \
python train_a40_resume.py \
  --experiment_name bioprint_v6_plain_unet_baseline \
  --resume_checkpoint none \
  --model_type plain_unet \
  --roi_size 96,192,192 \
  --batch_size 1 \
  --accumulation_steps 2 \
  --epochs 200 \
  --device cuda \
  --splits_json splits.json \
  --imagecas_root Data/all \
  --output_dir outputs/baselines/plain_unet
```

## Plain UNet Baseline Evaluation

```bash
cd /workspace/cloud_bundle
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128 \
python evaluate_full_test_a40.py \
  --checkpoint outputs/baselines/plain_unet/bioprint_v6_plain_unet_baseline/checkpoints/best_dice05.pt \
  --splits_json splits.json \
  --data_root Data/all \
  --output_dir outputs/baselines/plain_unet/full_test_eval \
  --device cuda \
  --roi_size 96,192,192 \
  --sw_batch_size 1 \
  --resume \
  --skip_existing \
  --model_type plain_unet
```

## Optional nnU-Net-Style Baseline

No nnU-Net environment was detected or configured in this repo. Treat this as
optional/unavailable unless you explicitly install and configure nnU-Net.
