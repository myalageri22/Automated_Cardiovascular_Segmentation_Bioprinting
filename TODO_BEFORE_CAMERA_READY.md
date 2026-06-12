# Actions before camera-ready / archive deposition

These are intentionally deferred until after double-anonymous review. They were removed from the
manuscript PDF (no in-text TODO markers) and tracked here instead.

1. **Restore author metadata.** Restore/replace the author names that remain in `LICENSE` and
   `CITATION.cff` as appropriate once the review is complete. The manuscript `\author{}` /
   `\affil{}` fields in `paper/main.tex` also still carry the
   `[... removed for double-anonymous review]` placeholders — fill these in for camera-ready.

2. **Final anonymity sweep.** Re-run a grep over the full code/results archive for names,
   institution, email addresses, usernames, and absolute home paths. Confirm zero hits, then
   update the "Data and code availability" paragraph in `paper/main.tex` to state that the
   deposited archive is fully anonymized (it currently describes a *partial* scrub).

Done already (for reference): username + absolute home paths stripped from training logs and
`dataset_splits.json`; figures inspected for baked-in paths/usernames; manuscript text/figures
carry no emails or institutional names.
