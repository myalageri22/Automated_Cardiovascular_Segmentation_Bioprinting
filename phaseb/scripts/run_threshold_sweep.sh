#!/usr/bin/env bash
set -euo pipefail

python -m phaseb.src.phaseb.cli \
  --ct /path/to/ct.nii.gz \
  --seg /path/to/seg_prob.nii.gz \
  --case-id demo_case \
  --threshold-sweep \
  --outdir /workspace/cloud_bundle/phaseb_outputs
