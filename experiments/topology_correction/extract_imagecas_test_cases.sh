#!/usr/bin/env bash
# Reassemble the split ImageCAS archives and extract ONLY the 250 held-out test cases.
#
# Data/archive-2 holds five ranges, each a split Info-ZIP set:
#   <range>.z01 .. <range>.z04   (4 GB parts, in order)
#   <range>.change2zip           (final part; a .zip with its extension changed)
#
# NOTE ON REASSEMBLY. A split zip cannot be rebuilt with `cat`. Each part is a
# separate "disk", and the central directory in the final part stores offsets
# RELATIVE TO THE DISK a member lives on, not relative to the whole archive.
# Concatenating leaves every offset wrong, which surfaces as
#   "N extra bytes at beginning or within zipfile"
#   "file #k: bad zipfile offset (local header sig)"
# `zip -s 0 <last-part> --out <single>` is the supported conversion: it walks the
# parts and rewrites every offset into a single-file archive. It is slower than
# cat because it re-reads and rewrites each entry, but it is the correct tool.
#
# The 250 test ids span 751-1000, so only ranges 601-800 (50 cases) and
# 801-1000 (200 cases) are touched. The other three ranges are left alone.
#
# Extracts to Data/all/<range>/<id>.img.nii.gz - the layout that
# outputs/final_test_250/per_case_metrics.csv records for the reported run.
#
# Source archives are never renamed, moved or modified: the parts are symlinked
# into a scratch directory under the names zip expects.
#
#   bash experiments/topology_correction/extract_imagecas_test_cases.sh
#
# Peak extra disk: ~18 GB (one reassembled range at a time) + ~23 GB extracted.
# Resumable: re-running skips ranges already extracted.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCH="$REPO/Data/archive-2"
DEST="$REPO/Data/all"
WORK="$REPO/Data/_reassemble"
SPLITS="$REPO/extra_information/data_information/dataset_splits.json"

mkdir -p "$DEST" "$WORK"

echo "[$(date +%H:%M:%S)] test ids needed per range:"
python3 - "$SPLITS" "$WORK" <<'PY'
import json, sys, os
splits = json.load(open(sys.argv[1])); work = sys.argv[2]
ids = sorted(int(c["id"]) if isinstance(c, dict) else int(c) for c in splits["test"])
for name, (lo, hi) in {"601-800": (601, 800), "801-1000": (801, 1000)}.items():
    want = [i for i in ids if lo <= i <= hi]
    # Trailing newline is required: `while read` drops a final unterminated line.
    open(os.path.join(work, f"{name}.ids"), "w").write("".join(f"{i}\n" for i in want))
    print(f"   {name}: {len(want)}")
print(f"   total: {len(ids)}")
PY

for RANGE in 601-800 801-1000; do
  echo
  echo "[$(date +%H:%M:%S)] ===== $RANGE ====="
  if [ -f "$WORK/.${RANGE}.done" ]; then
    echo "[$(date +%H:%M:%S)] already extracted, skipping"
    continue
  fi

  COMBINED="$WORK/${RANGE}_full.zip"
  LINKS="$WORK/links_$RANGE"

  # Discard anything left by an earlier attempt, including a bad `cat` result.
  rm -f "$COMBINED"
  rm -rf "$LINKS"
  mkdir -p "$LINKS"
  for p in "$ARCH/$RANGE".z0*; do ln -sf "$p" "$LINKS/$(basename "$p")"; done
  ln -sf "$ARCH/$RANGE.change2zip" "$LINKS/$RANGE.zip"

  echo "[$(date +%H:%M:%S)] reassembling with zip -s 0 (several minutes, rewrites all offsets)"
  if ! zip -q -s 0 "$LINKS/$RANGE.zip" --out "$COMBINED"; then
    echo "[$(date +%H:%M:%S)] zip -s 0 failed; retrying with the archive-repair path (zip -FF)"
    rm -f "$COMBINED"
    zip -q -FF "$LINKS/$RANGE.zip" --out "$COMBINED" </dev/null
  fi
  echo "[$(date +%H:%M:%S)] combined: $(du -h "$COMBINED" | cut -f1)"

  # A correctly rebuilt archive lists cleanly. Refuse to continue if it does not,
  # rather than extracting whatever happens to survive.
  if unzip -l "$COMBINED" 2>&1 >/dev/null | grep -qiE "extra bytes|bad zipfile offset|cannot find"; then
    echo "[$(date +%H:%M:%S)] ERROR: reassembled archive still reports offset problems."
    unzip -l "$COMBINED" 2>&1 >/dev/null | head -5
    echo "Stopping. Do not use a partial extraction."
    exit 3
  fi
  echo "[$(date +%H:%M:%S)] archive verifies clean, entries: $(unzip -Z1 "$COMBINED" | wc -l)"

  echo "[$(date +%H:%M:%S)] extracting test cases"
  PATTERNS=()
  while read -r id; do
    [ -n "$id" ] || continue
    PATTERNS+=("$RANGE/$id.img.nii.gz" "$RANGE/$id.label.nii.gz")
  done < <(cat "$WORK/$RANGE.ids"; echo)
  want=$(grep -c . "$WORK/$RANGE.ids")
  if [ "${#PATTERNS[@]}" -ne "$((want * 2))" ]; then
    echo "[$(date +%H:%M:%S)] ERROR: built ${#PATTERNS[@]} patterns for $want ids (expected $((want * 2)))."
    exit 4
  fi
  echo "[$(date +%H:%M:%S)] requesting $want cases (${#PATTERNS[@]} members)"
  unzip -q -o "$COMBINED" "${PATTERNS[@]}" -d "$DEST"

  n=$(ls "$DEST/$RANGE"/*.img.nii.gz 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] extracted $n image volumes into $DEST/$RANGE"
  if [ "$n" -lt "$want" ]; then
    echo "[$(date +%H:%M:%S)] ERROR: expected $want image volumes, found $n. Not marking this range done."
    exit 5
  fi
  touch "$WORK/.${RANGE}.done"

  rm -f "$COMBINED"; rm -rf "$LINKS"
  echo "[$(date +%H:%M:%S)] freed the reassembled copy"
done

echo
echo "[$(date +%H:%M:%S)] ===== verification ====="
python3 - "$SPLITS" "$DEST" <<'PY'
import json, sys, os, glob, gzip
splits = json.load(open(sys.argv[1])); dest = sys.argv[2]
ids = [str(c["id"]) if isinstance(c, dict) else str(c) for c in splits["test"]]
missing, truncated = [], []
for i in ids:
    img = glob.glob(os.path.join(dest, "*", f"{i}.img.nii.gz"))
    lab = glob.glob(os.path.join(dest, "*", f"{i}.label.nii.gz"))
    if not img or not lab:
        missing.append(i); continue
    # cheap integrity check: a truncated gzip fails on the first read
    for p in (img[0], lab[0]):
        try:
            with gzip.open(p, "rb") as fh: fh.read(1024)
        except Exception:
            truncated.append(os.path.basename(p)); break
print(f"test cases required: {len(ids)}   present: {len(ids)-len(missing)}   missing: {len(missing)}")
if truncated:
    print(f"CORRUPT/TRUNCATED: {len(truncated)} ->", " ".join(truncated[:10]))
if not missing and not truncated:
    print("ALL 250 HELD-OUT TEST CASES PRESENT AND READABLE")
elif missing:
    print("missing:", " ".join(missing[:30]))
PY

echo
echo "Scratch you can remove once verification passes:  $WORK"
echo "Next:  bash experiments/topology_correction/regenerate_frozen_predictions.sh"
