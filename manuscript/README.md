# Canonical manuscript

- Source: `finalv2_production_slicer.tex`
- Compiled publication artifact: `finalv2_production_slicer.pdf`
- Bibliography: `references.bib`
- IOP class: `iopjournal.cls`
- Figure assets: `figures/`

Compile from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error finalv2_production_slicer.tex
```

The canonical source incorporates the complete 250-case production-slicer result. The separate case-720 discussion remains an illustrative validation-partition analysis and is not part of the held-out cohort.

The compiled 22-page PDF resolves every citation, reference, table, and figure target. The authentic supplied HD95 and precision-recall figures are `figures/fig_hd95_distribution.png` and `figures/fig_precision_recall.png`.
