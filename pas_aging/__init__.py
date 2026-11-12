"""Estimation of control-derived tissue-aging scores from PAS histology.

Pipeline (see README.md):
  tiling   -> 128 x 128 um tiles from PAS whole-slide images (tissue >= 50 %)
  stain    -> Reinhard L*a*b* normalization to a fixed target (tissue pixels only)
  embed    -> UNI2-h features (1,536 per tile)
  pooling  -> per-biopsy p25 / p50 / p75 / mean over tiles (6,144 values)
  model    -> StandardScaler -> PCA(16) -> multi-output RidgeCV, targets centred within disease
  evaluate -> Spearman agreement with transcriptomic scores; partial Spearman with eGFR slope
"""

DOMAINS = [
    "Domain_collecting_duct",
    "Domain_tubulointerstitial",
    "Domain_tubular_epithelial",
    "Domain_glomerular_barrier",
]
DOMAIN_LABELS = {
    "Domain_collecting_duct": "collecting duct",
    "Domain_tubulointerstitial": "tubulointerstitial",
    "Domain_tubular_epithelial": "tubular epithelial",
    "Domain_glomerular_barrier": "filtration barrier",
}
TILE_UM = 128.0
TILE_PX = 224
MIN_TISSUE_FRACTION = 0.5
BACKGROUND_RGB_THRESHOLD = 200
N_COMPONENTS = 16
