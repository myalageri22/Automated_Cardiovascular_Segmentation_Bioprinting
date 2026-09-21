# Final Baseline Artifact Inventory

## Committed compact artifacts

- Exact-250 plain and Attention per-case CSV files.
- Exact-250 paired comparison CSV.
- Model summary CSV/JSON files.
- Paired bootstrap, Wilcoxon, and Benjamini-Hochberg FDR statistics.
- Historical Attention source and duplicate audits.
- Sanitized ID-only split manifest with authoritative source hash.
- Checkpoint/config provenance and environment metadata.
- clDice, cohort-audit, and paired-comparison utilities with tests.
- Publication-ready report, table, and manuscript paragraphs.

## Intentionally omitted local artifacts

| Artifact | Expected local path pattern | Approximate size | Reason omitted |
|---|---|---:|---|
| ImageCAS images and labels | `Data/all/**` | Dataset-scale | Dataset privacy/licensing and repository size |
| Plain prediction masks | `outputs/final_baseline_comparison/plain_unet/mask_regeneration/case_outputs/{case_id}/seg_mask_0.5.nii.gz` | 27 MB total | Regenerable binary NIfTI payload |
| Attention prediction masks | `outputs/final_test_250_phaseb_traceable_v14/case_outputs/{case_id}/seg_mask_0.5.nii.gz` | 27 MB total | Regenerable binary NIfTI payload |
| Plain checkpoint | `outputs/baselines/plain_unet/bioprint_v6_plain_unet_baseline/checkpoints/best_dice05.pt` | 328 MB | Large model binary; hash retained |
| Attention checkpoint | `outputs/train_runs/bioprint_v14_tversky_fn_local_rebuild_mps_lowmem/checkpoints/best_dice05.pt` | 271 MB | Large model binary; hash retained |
| Mesh, STL, G-code, logs, and caches | Local output directories | Variable | Unrelated large/generated payloads |

Plain checkpoint SHA-256: `f48269a225c20b0f1eb173b92308bd215d34a7dcded05fca46e21dfb2a9afba0`.

Attention checkpoint SHA-256: `792178759e9e579eb65f89a5a63f6538d7d0202da7e8eeeefda4825bfffc743b`.

Authoritative source split SHA-256: `70e847d64dc020cc8233016052772bc9b4d119c56685f6333dd4a1495b72d2db`.

## Provenance notes

The plain masks were regenerated through one explicitly authorized inference-only pass from the frozen epoch-30 checkpoint. A three-case CPU pilot and the complete 250-case run reproduced the frozen scalar metrics exactly. No training, fine-tuning, threshold sweep, split modification, or Attention inference was performed.

The historical Attention table with 252 rows remains local and unchanged. Its duplicated IDs were 751 and 1000; duplicate scientific fields and paths were identical, while runtime differed. The committed final Attention table comes from a separate exact-250 intended-checkpoint source.
