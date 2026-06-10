# v14 Checkpoint Reconstruction Commands

These commands preserve the original code. They create new outputs only.

## 0. Current local blocker

As of this check, the local repository has:

```text
data/all: nested ImageCAS folders such as `1-200`, `201-400`, etc.
```

The v11 checkpoint is present locally, but the ImageCAS NIfTI data must be
restored locally before any Mac rebuild or 250-case evaluation can run.

The clean local v11 checkpoint is:

```text
outputs/train_runs/bioprint_v11_a40_roi96_posneg3to1_lr2e5/checkpoints/best_dice05.pt
```

## 1. Audit whether true v14 reconstruction is possible locally

Run from the repository root:

```bash
cd /path/to/cloud_bundle
python scripts/reconstruct_v14_checkpoint.py --allow-missing
```

Outputs:

- `outputs/v14_reconstruction/V14_RECONSTRUCTION_STATUS.md`
- `outputs/v14_reconstruction/v14_reconstruction_status.json`

If this reports missing epochs 75-80, the true v14 checkpoint cannot be
reconstructed locally from existing files.

## 2. Rebuild v14 locally from v11 on Mac

This creates a new local reproduction. It does not recover the deleted RunPod
checkpoint. It is the defensible path if you cannot use RunPod but can afford
local runtime.

First ensure `data/all` contains the ImageCAS files:

```bash
cd /path/to/cloud_bundle
find data/all -type f -name "*.nii.gz" | wc -l
```

Expected count is about `2002`.

Then run the complete local rebuild:

```bash
cd /path/to/cloud_bundle
export MPLCONFIGDIR=/tmp/mplconfig
export PYTORCH_ENABLE_MPS_FALLBACK=1

python scripts/run_v14_local_rebuild.py \
  --run-name bioprint_v14_tversky_fn_local_rebuild \
  --imagecas-root data/all \
  --resume-checkpoint outputs/train_runs/bioprint_v11_a40_roi96_posneg3to1_lr2e5/checkpoints/best_dice05.pt \
  --output-root outputs/train_runs \
  --eval-output-root outputs/final_test_250_local_reproduced \
  --device auto \
  --num-workers 0
```

Outputs:

- `outputs/train_runs/bioprint_v14_tversky_fn_local_rebuild/`
- `outputs/train_runs/bioprint_v14_tversky_fn_local_rebuild/checkpoints/soup_ep75-80.pt`
- `outputs/final_test_250_local_reproduced/bioprint_v14_tversky_fn_local_rebuild/summary_metrics.json`
- `outputs/v14_local_rebuild_provenance/bioprint_v14_tversky_fn_local_rebuild/`

Use the local result in the paper only after this reproduced checkpoint is
evaluated on the 250-case held-out test and the metrics match or are clearly
reported as the new reproduced result.

## 3. If you later recover the real v14 ingredient folder

If an old disk/export contains:

```text
bioprint_v14_tversky_fn/checkpoints/epoch_075.pt
bioprint_v14_tversky_fn/checkpoints/epoch_076.pt
bioprint_v14_tversky_fn/checkpoints/epoch_077.pt
bioprint_v14_tversky_fn/checkpoints/epoch_078.pt
bioprint_v14_tversky_fn/checkpoints/epoch_079.pt
bioprint_v14_tversky_fn/checkpoints/epoch_080.pt
```

then reconstruct the soup with:

```bash
cd /path/to/cloud_bundle
python scripts/reconstruct_v14_checkpoint.py \
  --source-dir /path/to/bioprint_v14_tversky_fn/checkpoints \
  --output-root outputs/v14_reconstruction
```

The reconstructed checkpoint will be:

```text
outputs/v14_reconstruction/bioprint_v14_tversky_fn/checkpoints/soup_ep75-80.pt
```

## 4. Scientifically valid rerun command for a new A40 RunPod

Use this only if the true v14 ingredients are not recoverable. This is a new
attempt to reproduce the v14 result, not a reconstruction of the missing old
checkpoint.

Important: write to `/workspace/cloud_bundle/outputs`, not `/tmp`, so the
checkpoint survives export.

```bash
cd /workspace/cloud_bundle
source .venv/bin/activate

RUN_NAME="bioprint_v14_tversky_fn_reproduce_$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="/workspace/cloud_bundle/outputs/train_runs"
RESUME_CKPT="/workspace/cloud_bundle/outputs/train_runs/bioprint_v11_a40_roi96_posneg3to1_lr2e5/checkpoints/best_dice05.pt"

mkdir -p "${OUT_ROOT}/${RUN_NAME}/logs"

nohup bash -lc "MPLCONFIGDIR=/tmp/mplconfig \
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128 \
python train_a40_resume.py \
  --experiment_name ${RUN_NAME} \
  --resume_checkpoint ${RESUME_CKPT} \
  --strict_load \
  --allow_partial_load \
  --model_type attention_unet \
  --roi_size 96,192,192 \
  --batch_size 1 \
  --accumulation_steps 2 \
  --epochs 100 \
  --lr 2e-5 \
  --weight_decay 1e-5 \
  --grad_clip_norm 1.0 \
  --device cuda \
  --num_workers 4 \
  --pin_memory \
  --splits_json splits.json \
  --imagecas_root Data/all \
  --cache_rate_train 0.0 \
  --cache_rate_val 0.0 \
  --scheduler reduce_on_plateau \
  --early_stopping_patience 10 \
  --pos_neg_ratio 3,1 \
  --num_samples 2 \
  --pos_weight_fixed 0 \
  --loss_mode tversky_dice \
  --output_dir ${OUT_ROOT}" \
  > "${OUT_ROOT}/${RUN_NAME}_console.log" 2>&1 &

tail -f "${OUT_ROOT}/${RUN_NAME}_console.log"
```

After epochs 75-80 exist, create the soup:

```bash
cd /workspace/cloud_bundle
source .venv/bin/activate

python soup_checkpoints.py \
  --checkpoint_dir "${OUT_ROOT}/${RUN_NAME}/checkpoints" \
  --output "${OUT_ROOT}/${RUN_NAME}/checkpoints/soup_ep75-80.pt" \
  --epoch_start 75 \
  --epoch_end 80

python scripts/reconstruct_v14_checkpoint.py \
  --expected-run "${RUN_NAME}" \
  --source-dir "${OUT_ROOT}/${RUN_NAME}/checkpoints" \
  --output-root "/workspace/cloud_bundle/outputs/v14_reconstruction_${RUN_NAME}"
```

Then rerun the 250-case held-out test using the reproduced soup. Do not claim
the old 0.7953 result unless this reproduced checkpoint is evaluated and
matches the reported result.
