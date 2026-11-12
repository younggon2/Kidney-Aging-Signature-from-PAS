#!/usr/bin/env python3
"""Reproduce the manuscript's PAS-estimation results from the released per-biopsy data (CPU, ~1 min).

1. Refits the model with repeated patient-grouped cross-validation on the 48 training biopsies in
   data/biopsy_features.csv and compares the out-of-fold estimates with data/biopsy_scores.csv.
2. Applies the released model/ to the three held-out second-batch DKD biopsies and compares.
3. Prints the Spearman agreement between PAS-estimated and transcriptomic scores in DKD.

python scripts/reproduce_from_release.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pas_aging import DOMAINS  # noqa: E402
from pas_aging.evaluate import agreement  # noqa: E402
from pas_aging.model import LinearModel, out_of_fold  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap.add_argument("--features", default=os.path.join(root, "data", "biopsy_features.csv"))
    ap.add_argument("--scores", default=os.path.join(root, "data", "biopsy_scores.csv"))
    ap.add_argument("--model-dir", default=os.path.join(root, "model"))
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    F = pd.read_csv(a.features).set_index("release_id")
    S = pd.read_csv(a.scores).set_index("release_id")
    fcols = [c for c in F.columns if c not in ("disease", "batch")]
    est_cols = [d + "__pas_estimate" for d in DOMAINS]
    meas_cols = [d + "__transcriptomic" for d in DOMAINS]

    tr = S.index[S["estimate_type"] == "out_of_fold"].tolist()
    X = F.loc[tr, fcols].to_numpy(float)
    Y = S.loc[tr, meas_cols].to_numpy(float)
    dis = S.loc[tr, "disease"].astype(str).to_numpy()
    print("[repro] training biopsies %d: %s" % (len(tr), dict(zip(*np.unique(dis, return_counts=True)))))
    # fold_r01..fold_r20 hold the study's fold assignments, so the exact splits are reused
    fcol = sorted(c for c in S.columns if c.startswith("fold_r"))[:a.reps]
    fold_ids = S.loc[tr, fcol].astype(int).to_numpy().T if fcol else None
    oof, per_rep = out_of_fold(X, Y, tr, disease=dis, patient=np.array(tr), n_reps=a.reps, fold_ids=fold_ids)
    diff = np.abs(oof.to_numpy() - S.loc[tr, est_cols].to_numpy(float))
    sd = S.loc[tr, est_cols].to_numpy(float).std(0).mean()
    print("[repro] out-of-fold estimates vs released: max |diff| %.2e, median %.2e (estimate SD %.2e). "
          "Differences up to ~1e-3 arise when the ridge penalty is selected at an adjacent grid value by a different "
          "scikit-learn version; rank correlation per domain: %s"
          % (diff.max(), np.median(diff), sd,
             np.round([spearmanr(oof.iloc[:, j], S.loc[tr, est_cols[j]]).correlation for j in range(4)], 4)))
    ev = dis == "DKD"
    print("[repro] within-DKD Spearman, mean over repetitions: %.3f (manuscript 0.62)" % per_rep.mean())
    print("[repro] within-DKD Spearman of rep-averaged estimates per domain: %s (manuscript: filtration barrier 0.72, collecting duct 0.59)"
          % dict(zip(DOMAINS, np.round(agreement(oof.to_numpy()[ev], Y[ev]), 3))))

    ho = S.index[S["estimate_type"] == "held_out_batch2"].tolist()
    if ho:
        m = LinearModel(a.model_dir)
        pred = m.predict(F.loc[ho, fcols].to_numpy(float)).to_numpy()   # features are released already batch-aligned
        d2 = np.abs(pred - S.loc[ho, est_cols].to_numpy(float)).max()
        print("[repro] held-out batch-2 DKD (%d): released model vs released estimates max |diff| %.2e" % (len(ho), d2))


if __name__ == "__main__":
    main()
