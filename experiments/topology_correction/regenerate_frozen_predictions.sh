#!/usr/bin/env bash
# Regenerate the per-case predictions the topology-correction experiment needs.
#
# This is Step 2 of analysis/cmig_robustness/RECOVERY_RUNBOOK.md, wrapped so it
# is resumable and so the provenance gate runs automatically at the end.
#
# It writes ONLY into a new directory. It never touches outputs/final_test_250/
# or outputs/phase_b_mesh_qc/.
#
# Runtime: 118-199 s per case on Apple MPS (mean 196.7 s recorded), so budget
# 13-14 h for 250 cases. --resume --skip_existing means it survives interruption;
# re-running continues where it stopped.
#
#   bash experiments/topology_correction/regenerate_frozen_predictions.sh
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

CHECKPOINT="${CHECKPOINT:-checkpoints/best_dice05.pt}"
DATA_ROOT="${DATA_ROOT:-Data/all}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/reinference_2026}"
DEVICE="${DEVICE:-mps}"
SPLITS="extra_information/data_information/dataset_splits.json"

# FROZEN. Must equal the threshold of the reported evaluation. Never tune this.
THRESHOLD=0.5

# Interpreter used for every step. Override to pin a conda env explicitly:
#   PYTHON=$HOME/opt/miniconda3/envs/vascseg/bin/python
PY_BIN="${PYTHON:-python3}"

# Optional smoke test: SMOKE=2 runs only the first N cases, then exits before
# the provenance gate. Results are written to the same directory and are reused
# by the full run, so a smoke test is never wasted work.
SMOKE="${SMOKE:-}"

echo "repo:       $REPO"
echo "checkpoint: $CHECKPOINT"
echo "data root:  $DATA_ROOT"
echo "output:     $OUTPUT_DIR   (new directory; archived metrics are never overwritten)"
echo "threshold:  $THRESHOLD    (frozen)"
echo "python:     $("$PY_BIN" -c 'import sys; print(sys.executable)' 2>/dev/null || echo "NOT FOUND: $PY_BIN")"
echo
"$PY_BIN" - <<'PYCHK' || { echo "Dependency check failed. Activate the right conda env, or set PYTHON=/path/to/env/bin/python."; exit 2; }
import importlib, sys
missing = []
for m in ("torch", "monai", "numpy", "nibabel", "scipy", "skimage"):
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m} ({type(e).__name__})")
import torch
print(f"torch {torch.__version__}   mps_available={torch.backends.mps.is_available()}")
try:
    import monai; print(f"monai {monai.__version__}")
except Exception: pass
if missing:
    print("MISSING:", ", ".join(missing)); sys.exit(1)
PYCHK
echo

# ---------------------------------------------------------------- preflight
fail=0
if [ ! -f "$CHECKPOINT" ]; then
  echo "MISSING: $CHECKPOINT"
  echo "  If it is a Git-LFS pointer:  git lfs install && git lfs pull --include=\"$CHECKPOINT\""
  fail=1
elif [ "$(wc -c < "$CHECKPOINT")" -lt 100000 ]; then
  echo "SUSPECT: $CHECKPOINT is $(wc -c < "$CHECKPOINT") bytes - that is an LFS pointer, not weights."
  fail=1
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "MISSING: $DATA_ROOT"
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo
  echo "Preflight failed. Nothing was run."
  exit 2
fi

# Verify the checkpoint is the one behind the reported numbers.
"$PY_BIN" - "$CHECKPOINT" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu")
epoch, best = c.get("epoch"), c.get("best_metric")
print(f"checkpoint epoch={epoch} best_metric={best}")
if epoch != 79 or best is None or abs(float(best) - 0.7888903796672821) > 1e-6:
    print("WARNING: does not match the recorded epoch 79 / best_metric 0.7888903796672821.")
    print("         This may be a different checkpoint. Do not mix its outputs with the reported cohort.")
PY

# Confirm every test case is present. A partial dataset means a partial cohort,
# which must be reported as such, never silently.
"$PY_BIN" - "$SPLITS" "$DATA_ROOT" <<'PY'
import json, sys, glob, os
splits = json.load(open(sys.argv[1]))
root = sys.argv[2]
cases = [str(c["id"]) if isinstance(c, dict) else str(c) for c in splits["test"]]
missing = []
for c in cases:
    hits = glob.glob(os.path.join(root, f"{c}.img.nii.gz")) + \
           glob.glob(os.path.join(root, "*", f"{c}.img.nii.gz"))
    lab  = glob.glob(os.path.join(root, f"{c}.label.nii.gz")) + \
           glob.glob(os.path.join(root, "*", f"{c}.label.nii.gz"))
    if not hits or not lab:
        missing.append(c)
print(f"test cases in split: {len(cases)}   present on disk: {len(cases)-len(missing)}   missing: {len(missing)}")
if missing:
    print("missing case ids:", " ".join(missing[:25]), "..." if len(missing) > 25 else "")
    open("outputs/reinference_missing_cases.txt", "w").write("\n".join(missing))
    print("full list written to outputs/reinference_missing_cases.txt")
    print()
    print("A partial cohort is not the reported cohort. Either obtain the rest, or")
    print("report the reduced n explicitly - do not quietly evaluate a subset.")
PY

echo
# Under nohup there is no TTY, so a prompt would read EOF and abort the run.
# Confirm only when a human is actually attached, or when ASSUME_YES=1.
if [ -t 0 ] && [ -t 1 ] && [ "${ASSUME_YES:-0}" != "1" ]; then
  read -r -p "Proceed with inference? [y/N] " ok
  [ "$ok" = "y" ] || { echo "aborted"; exit 0; }
else
  echo "non-interactive (or ASSUME_YES=1): proceeding"
fi

# 2. disk guard. --save_case_outputs writes a float32 probability map plus a
# mask per case; budget roughly 25-30 GB for 250 cases. Stop early rather than
# dying 200 cases in with a half-written volume.
FREE_GB=$(df -g "$REPO" | tail -1 | awk '{print $4}')
echo "free disk: ${FREE_GB} GB"
if [ "$FREE_GB" -lt 35 ]; then
  echo "WARNING: under 35 GB free. The run needs roughly 25-30 GB for 250 cases."
  echo "         Free space first, or expect it to fail partway."
  if [ -t 0 ] && [ -t 1 ] && [ "${ASSUME_YES:-0}" != "1" ]; then
    read -r -p "Continue anyway? [y/N] " ok2
    [ "$ok2" = "y" ] || { echo "aborted"; exit 0; }
  fi
fi

# ---------------------------------------------------------------- inference
mkdir -p "$OUTPUT_DIR"
"$PY_BIN" evaluate_full_test_a40.py \
  --checkpoint "$CHECKPOINT" \
  --splits_json "$SPLITS" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --roi_size 96,192,192 \
  --sw_overlap 0.625 \
  --threshold "$THRESHOLD" \
  --val_output_device cpu \
  --num_workers 0 \
  --compute_hd95 --compute_cldice \
  --save_case_outputs --run_phase_b_qc \
  --resume --skip_existing \
  ${SMOKE:+--limit "$SMOKE"} \
  2>&1 | tee -a "$OUTPUT_DIR/inference.log"

if [ -n "$SMOKE" ]; then
  echo
  echo "SMOKE TEST of $SMOKE cases complete. Provenance gate skipped (it needs all 250)."
  echo "Inspect $OUTPUT_DIR/per_case_metrics.csv, then re-run without SMOKE for the full cohort."
  exit 0
fi

# ---------------------------------------------------------------- gate
echo
echo "=== provenance gate ==="
"$PY_BIN" - "$OUTPUT_DIR" <<'PY'
import sys, os, csv, statistics as st
newcsv = os.path.join(sys.argv[1], "per_case_metrics.csv")
if not os.path.exists(newcsv):
    print(f"no per-case metrics at {newcsv}"); raise SystemExit(2)
new = {r["case_id"]: r for r in csv.DictReader(open(newcsv))}
old = {r["case_id"]: r for r in csv.DictReader(open("outputs/final_test_250/per_case_metrics.csv"))}
f = lambda r: float(r["dice@0.5"])
common = sorted(set(new) & set(old))
print(f"cases regenerated: {len(new)}   overlapping with the archived run: {len(common)}")
if common:
    deltas = [abs(f(new[c]) - f(old[c])) for c in common]
    worst = max(range(len(common)), key=lambda i: deltas[i])
    mean_new = st.mean(f(new[c]) for c in common)
    print(f"mean Dice recomputed: {mean_new:.4f}   archived interval: 0.7820-0.7935")
    print(f"max per-case |delta Dice|: {deltas[worst]:.4f}  (case {common[worst]})")
    ok = 0.7820 <= mean_new <= 0.7935
    print("GATE:", "PASS" if ok else "FAIL - these outputs are not the reported run; stop and investigate")
PY

cat <<'EOS'

Next:
  1. point paths.pred_root in experiments/topology_correction/config/experiment_config.yaml
     at this OUTPUT_DIR/case_outputs
  2. python3 experiments/topology_correction/run_topology_correction_experiment.py --dry-run
  3. python3 experiments/topology_correction/run_topology_correction_experiment.py --mesh-variants primary
EOS
