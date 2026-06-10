#!/usr/bin/env bash
set -euo pipefail

python -m phaseb.src.phaseb.cli \
  --batch-manifest /path/to/manifest.csv \
  --outdir /workspace/cloud_bundle/phaseb_outputs
