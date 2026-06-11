#!/usr/bin/env bash
#
# run_full_revision.sh
# ---------------------
# One command to run the entire blocked critical path for the manuscript revision
# once the ImageCAS dataset is available on a GPU machine.
#
# It re-evaluates the 250-case held-out test set from an ACCOUNTABLE checkpoint
# (checkpoints/best_dice05.pt) with per-case predictions saved and clDice + Phase B
# mesh QC enabled, then runs cohort mesh-QC aggregation and the paired
# segmentation-to-mesh analysis. This unblocks manuscript TODO items 1, 2, 3 and 6.
#
# USAGE:
#   1. Download ImageCAS so that <DATA_ROOT> contains the *.img.nii.gz / *.label.nii.gz
#      volumes (case IDs 1..1000). The stale absolute paths in the splits file are
#      remapped to <DATA_ROOT> by case ID automatically.
#   2. Edit the variables below (or pass them as environment variables), then run:
#        bash run_full_revision.sh
#   3. Requires a CUDA GPU (set DEVICE=cuda). For a CPU/MPS dry run, set DEVICE=cpu.
#
# All paths are relative to the repository root.
set -euo pipefail

# ---- Configuration (override via environment) --------------------------------
DATA_ROOT="${DATA_ROOT:-Data/all}"                      # <-- point at downloaded ImageCAS
SPLITS="${SPLITS:-extra_information/data_information/dataset_splits.json}"
CHECKPOINT="${CHECKPOINT:-checkpoints/best_dice05.pt}"  # accountable epoch-79 Attention U-Net
OUTDIR="${OUTDIR:-outputs/final_test_250}"
DEVICE="${DEVICE:-cuda}"                                # cuda | cpu | mps
ROI="${ROI:-96,192,192}"
SW_OVERLAP="${SW_OVERLAP:-0.625}"                       # matches manuscript inference setting
# On Apple-Silicon MPS the sliding-window aggregation buffer must live on CPU,
# otherwise the full-volume gaussian map exhausts unified memory (OOM). On CUDA
# leave this as "same" for speed.
if [ "${DEVICE}" = "mps" ]; then
  VAL_OUTPUT_DEVICE="${VAL_OUTPUT_DEVICE:-cpu}"
  export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
else
  VAL_OUTPUT_DEVICE="${VAL_OUTPUT_DEVICE:-same}"
fi
# -----------------------------------------------------------------------------

echo "=============================================================="
echo " Full manuscript-revision run"
echo "   data_root  : ${DATA_ROOT}"
echo "   splits     : ${SPLITS}"
echo "   checkpoint : ${CHECKPOINT}"
echo "   output_dir : ${OUTDIR}"
echo "   device     : ${DEVICE}"
echo "=============================================================="

# ---- Step 0: sanity checks ---------------------------------------------------
[ -f "${CHECKPOINT}" ] || { echo "ERROR: checkpoint not found: ${CHECKPOINT}"; exit 1; }
[ -f "${SPLITS}" ]     || { echo "ERROR: splits file not found: ${SPLITS}"; exit 1; }
[ -d "${DATA_ROOT}" ]  || { echo "ERROR: data_root not found: ${DATA_ROOT} (download ImageCAS first)"; exit 1; }

# ---- Step 1: full 250-case held-out evaluation -------------------------------
# Enables per-case prediction saving, clDice, and per-case Phase B mesh QC.
# --resume --skip_existing makes the run idempotent and restartable.
echo ">>> Step 1/3: 250-case held-out evaluation (this is the long step)"
python evaluate_full_test_a40.py \
  --checkpoint "${CHECKPOINT}" \
  --splits_json "${SPLITS}" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTDIR}" \
  --device "${DEVICE}" \
  --roi_size "${ROI}" \
  --sw_overlap "${SW_OVERLAP}" \
  --threshold 0.5 \
  --val_output_device "${VAL_OUTPUT_DEVICE}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --compute_hd95 \
  --compute_cldice \
  --save_case_outputs \
  --run_phase_b_qc \
  --resume --skip_existing

PER_CASE="${OUTDIR}/per_case_metrics.csv"
[ -f "${PER_CASE}" ] || { echo "ERROR: ${PER_CASE} not produced; check the eval log."; exit 1; }
echo ">>> Per-case metrics (with clDice + mesh QC columns): ${PER_CASE}"

# ---- Step 2: cohort-level Phase B mesh-QC aggregation ------------------------
# The eval already writes outputs/phase_b_mesh_qc/ summaries; this regenerates
# cohort tables from the per-case qc_report.json files as a cross-check.
echo ">>> Step 2/3: cohort mesh-QC aggregation"
if [ -d "outputs/phase_b_mesh_qc" ]; then
  python aggregate_mesh_qc.py \
    --qc-dir "outputs/phase_b_mesh_qc" \
    --qc-glob "**/qc_report.json" \
    --out-prefix "outputs/phase_b_mesh_qc/summary_mesh_qc" \
    || echo "WARN: mesh-QC aggregation skipped (check qc_report.json locations)."
else
  echo "WARN: outputs/phase_b_mesh_qc not found; skipping aggregation."
fi

# ---- Step 3: paired segmentation-to-mesh analysis ---------------------------
# Joins per-case segmentation metrics (incl. clDice) to mesh outcomes by case_id
# and reports Spearman/Pearson correlations + scatterplots. The per_case_metrics.csv
# already contains the appended mesh-QC columns, so it serves as both inputs.
echo ">>> Step 3/3: paired segmentation-to-mesh correlation analysis"
python paired_analysis.py \
  --seg-csv "${PER_CASE}" \
  --mesh-csv "${PER_CASE}" \
  --key case_id \
  --out-prefix "${OUTDIR}/paired" \
  || echo "WARN: paired analysis skipped (inspect ${PER_CASE} columns/keys)."

echo "=============================================================="
echo " DONE. Next manuscript actions (now unblocked):"
echo "   - Table 6 / Table 9: refresh from ${OUTDIR}/summary_metrics.json"
echo "   - Table 9 clDice row: from cldice@0.5 in ${PER_CASE}"
echo "   - Table 4 (Phase B QC): from outputs/phase_b_mesh_qc/summary_mesh_qc.*"
echo "   - Sec 3.10 paired analysis: from ${OUTDIR}/paired_*"
echo "   - Reconcile Sec 4.1 + Appendix B to the accountable checkpoint:"
echo "       ${CHECKPOINT}"
echo "   NOTE: headline Dice@0.5 will differ from the lost-soup 0.7953;"
echo "         regenerate every per-case figure (Fig 4-7) and update the text."
echo "=============================================================="
