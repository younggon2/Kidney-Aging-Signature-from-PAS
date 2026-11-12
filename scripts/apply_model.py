#!/usr/bin/env python3
"""Score new biopsies with the released model (model/).

Input: a CSV of pooled biopsy features (pool_biopsies.py output; 6,144 columns in the released order).
Output: the four compartment-score estimates, in the same units as data/biopsy_scores.csv
(deviation from the training-group mean). Estimates are comparable within one staining batch;
--apply-shift adds the stored constant that aligned the study's second staining batch and is
only meaningful for slides from that batch.

python scripts/apply_model.py --model-dir model --features work/biopsy_features.csv --out work/estimates.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pas_aging.model import LinearModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--features", required=True)
    ap.add_argument("--apply-shift", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F = pd.read_csv(a.features, index_col=0)
    num = F.select_dtypes("number")
    m = LinearModel(a.model_dir)
    if num.shape[1] != m.W.shape[0]:
        raise SystemExit("expected %d feature columns, got %d" % (m.W.shape[0], num.shape[1]))
    E = m.predict(num.to_numpy(float), a.apply_shift)
    E.index = F.index
    E.to_csv(a.out, float_format="%.6f")
    print("[apply] %d biopsies -> %s" % (len(E), a.out))


if __name__ == "__main__":
    main()
