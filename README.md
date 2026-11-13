# Tissue-aging scores from PAS histology

Code, fitted model parameters and per-biopsy evaluation data for the PAS-based estimation of
control-derived kidney tissue-aging scores described in:

> Hong SG, Ryu D, et al. *A healthy donor kidney aging signature estimable from PAS histology and
> its association with progression of diabetic kidney disease.* medRxiv 2026.
> doi:[10.64898/2026.11.05.26367347](https://doi.org/10.64898/2026.11.05.26367347)

The four compartment scores (collecting duct, tubulointerstitial, tubular epithelial, filtration
barrier) are defined from single-cell spatial transcriptomics of healthy donor kidneys (genes and
weights in the manuscript's Supplementary Table S1). This repository covers the estimation of
those scores from a routine periodic acid–Schiff (PAS) slide.

## Model

| Step | What is done |
|---|---|
| Tiling | 128 × 128 µm non-overlapping tiles on tissue (≥ 50 % tissue), resized to 224 × 224 px |
| Stain normalization | Reinhard L\*a\*b\* matching to one fixed target (`model/stain_target_lab.json`); statistics from tissue pixels only |
| Features | UNI2-h pathology foundation model, frozen, 1,536 values per tile |
| Biopsy summary | 25th, 50th, 75th percentiles and mean over tiles → 6,144 values |
| Regression | StandardScaler → PCA (16 components) → multi-output ridge, penalty chosen by internal leave-one-out CV on a grid of 25 values from 10⁻² to 10⁴ |
| Targets | The four compartment scores, centred within disease group (DKD, IgA nephropathy) inside each training fold |
| Validation | Patient-grouped 5-fold cross-validation repeated 20 times; each training biopsy's estimate is the mean of its 20 out-of-fold estimates |

Only the feature scaler, the PCA and the ridge regression are fitted. The foundation model is a fixed feature
extractor. Training used 48 biopsies (24 DKD, 24 IgA nephropathy) from the first staining batch;
three additional DKD biopsies from a second staining batch were scored with the model fitted to
all 48 after a constant feature shift that aligned the two batches (mean of IgA nephropathy
biopsies in the training batch minus their mean in the new batch; `model/batch2_feature_shift.csv`).
The penalty selected for the model fitted to all 48 biopsies is 10,000, the largest value of the
grid; extending the grid to 10⁶ or 10⁸ selects the same value and leaves the cross-validated
results unchanged (mean Spearman 0.619 vs 0.621).

## Contents

```
pas_aging/            library: tiling, stain, embed, pooling, model, evaluate
scripts/              command-line steps (see below)
model/                fitted parameters (CC BY-NC 4.0)
data/                 per-biopsy features and scores used to evaluate the model (CC BY-NC 4.0)
```

`model/`

| File | Meaning |
|---|---|
| `stain_target_lab.json` | Reinhard target: L\*a\*b\* channel means and standard deviations |
| `scaler_mean.csv`, `scaler_scale.csv` | StandardScaler fitted on the 48 training biopsies (6,144 each) |
| `pca_mean.csv`, `pca_components.csv` | PCA mean (6,144) and loadings (16 × 6,144) |
| `ridge_coef.csv`, `ridge_intercept.csv`, `ridge_alpha.json` | Ridge coefficients (4 × 16), intercepts (4), selected penalty |
| `target_centering.json` | Disease-group means subtracted from the targets before fitting |
| `batch2_feature_shift.csv` | Constant added to second-batch features (6,144) |
| `linear_weights.csv`, `linear_intercept.csv` | The three stages collapsed into one linear map (6,144 × 4) and intercept (4): estimate = features · weights + intercept |

All parameters are plain CSV text with 12 significant digits.

`data/`

| File | Meaning |
|---|---|
| `biopsy_features.csv` | One row per biopsy (random identifier `KB###`, disease, batch) and 6,144 pooled features (`p25_0000 … mean_1535`). Second-batch rows already include the batch shift. |
| `biopsy_scores.csv` | Transcriptomic compartment scores (`*__transcriptomic`) and PAS estimates (`*__pas_estimate`); `estimate_type` is `out_of_fold` for the 48 training biopsies and `held_out_batch2` for the three second-batch DKD biopsies. `fold_r01` … `fold_r20` give each training biopsy's fold (0–4) in each of the 20 cross-validation repetitions, so the study's exact splits can be reused. |

No images, tile-level data, individual-level transcriptomic data or clinical data are included.
Identifiers are random and cannot be linked to patients. The eGFR-slope analyses in the manuscript
require clinical data that are available only through a collaboration agreement (see the
manuscript's Data Sharing Statement); the statistics used for them are implemented in
`pas_aging/evaluate.py`.

## Reproducing the estimation results

```bash
pip install numpy pandas scipy scikit-learn
python scripts/reproduce_from_release.py
```

This refits the cross-validated model from `data/biopsy_features.csv`, compares the out-of-fold
estimates with `data/biopsy_scores.csv`, reports the within-DKD Spearman agreement with the
transcriptomic scores (manuscript: mean 0.62 over repetitions; filtration barrier 0.72 and
collecting duct 0.59 for the repetition-averaged estimates) and checks the released `model/`
against the held-out second-batch estimates.

The study results were computed with scikit-learn 1.7.2 and NumPy 1.26.4; the check below was
also run with scikit-learn 1.9.0. Expected output: within-DKD Spearman 0.621 (mean over repetitions);
0.594 / 0.632 / 0.618 / 0.719 for collecting duct / tubulointerstitial / tubular epithelial /
filtration barrier; held-out second-batch estimates reproduced to 4e-7. The refitted out-of-fold
estimates agree with the released ones to a median of 8e-6 and a maximum of 5e-4 (about 3 % of
their standard deviation; rank correlation ≥ 0.9999) when a different scikit-learn version is
used. The residual comes from the ridge penalty being selected at an adjacent grid value
(5,623 vs 10,000) in a few folds, not from the data.

## Applying the model to new slides

Requires `openslide`, `torch`, `timm` and access to the gated UNI2-h weights
(`huggingface-cli login` after accepting the terms of `MahmoodLab/UNI2-h`).

```bash
python scripts/tile_slides.py      --svs-dir slides/ --out-dir work/tiles
python scripts/embed_tiles.py      --tile-dir work/tiles --stain-target model/stain_target_lab.json --out work/tile_features.csv
python scripts/pool_biopsies.py    --features work/tile_features.csv --out work/biopsy_features.csv   # add --tile-map if a slide holds several biopsies
python scripts/apply_model.py      --model-dir model --features work/biopsy_features.csv --out work/estimates.csv
```

Estimates are deviations from the training-group mean and are comparable within one staining
and scanning batch. A new batch differs from the training batch by a constant feature offset that
the linear model turns into a constant offset of the estimates; ranks within the batch are not
affected, absolute values are. The model has not been validated outside the study cohort.

To fit the model on your own cohort with the same procedure:

```bash
python scripts/train_oof.py --features work/biopsy_features.csv --targets work/scores.csv --out-dir work/fit
```

## Notes

* Tile-level maps in the manuscript (Figure 6, Supplementary Figure S6) were drawn from a
  per-compartment refit of the ridge step on the tile features and illustrate within-biopsy
  variation only; all biopsy-level results use the model described above.
* UNI2-h is distributed by its authors under their own terms (non-commercial research use,
  access approval on Hugging Face). Those terms apply to any use of this pipeline that runs
  the foundation model.

## License

Code: MIT (`LICENSE`). Model parameters and data: CC BY-NC 4.0 (`data/LICENSE`).
