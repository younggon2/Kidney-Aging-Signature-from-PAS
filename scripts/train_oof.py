#!/usr/bin/env python3
"""Train the PAS -> compartment-score model and produce out-of-fold estimates.

Inputs
  --features  CSV from pool_biopsies.py (biopsy_id + 6,144 pooled features)
  --targets   CSV with biopsy_id and the four compartment scores
              (Domain_collecting_duct, Domain_tubulointerstitial, Domain_tubular_epithelial, Domain_glomerular_barrier)
  --disease-col / --patient-col  optional columns in --targets; by default both are read from the identifier
Outputs (in --out-dir)
  oof_estimates.csv   rep-averaged out-of-fold estimates (centred within disease)
  oof_summary.csv     per-domain Spearman within the evaluation group, both definitions used in the paper
  model/              parameters of the model fitted to all biopsies

python scripts/train_oof.py --features work/biopsy_features.csv --targets work/scores.csv --out-dir work/fit
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pas_aging import DOMAINS  # noqa: E402
from pas_aging.evaluate import agreement  # noqa: E402
from pas_aging.model import disease_of, export, fit, out_of_fold, patient_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--disease-col", default=None)
    ap.add_argument("--patient-col", default=None)
    ap.add_argument("--eval-group", default="DKD")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stain-target", default=None, help="copied into model/ for completeness")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    F = pd.read_csv(a.features, index_col=0)
    F.index = F.index.astype(str)
    T = pd.read_csv(a.targets)
    T["biopsy_id"] = T["biopsy_id"].astype(str)
    T = T.drop_duplicates("biopsy_id").set_index("biopsy_id")
    ids = [s for s in F.index if s in T.index and T.loc[s, DOMAINS].notna().all()]
    X = F.loc[ids].to_numpy(float)
    Y = T.loc[ids, DOMAINS].to_numpy(float)
    dis = T.loc[ids, a.disease_col].astype(str).to_numpy() if a.disease_col else np.array([disease_of(s) for s in ids])
    pat = T.loc[ids, a.patient_col].astype(str).to_numpy() if a.patient_col else np.array([patient_key(s) for s in ids])
    print("[train] %d biopsies, %s" % (len(ids), dict(zip(*np.unique(dis, return_counts=True)))))

    oof, per_rep = out_of_fold(X, Y, ids, dis, pat, a.folds, a.reps, eval_group=a.eval_group)
    ev = dis == a.eval_group
    rho_avg = agreement(oof.to_numpy()[ev], Y[ev])
    os.makedirs(a.out_dir, exist_ok=True)
    oof.to_csv(os.path.join(a.out_dir, "oof_estimates.csv"), float_format="%.6f")
    S = pd.DataFrame({"domain": DOMAINS, "rho_mean_over_reps": per_rep.mean(0), "rho_of_rep_averaged_oof": rho_avg})
    S.to_csv(os.path.join(a.out_dir, "oof_summary.csv"), index=False, float_format="%.4f")
    print("[train] within-%s Spearman: mean over reps %.3f; rep-averaged OOF per domain %s"
          % (a.eval_group, per_rep.mean(), np.round(rho_avg, 3)))

    pipe, means = fit(X, Y, dis)
    stain = None
    if a.stain_target:
        from pas_aging.stain import load_target
        stain = load_target(a.stain_target)
    export(pipe, means, os.path.join(a.out_dir, "model"), stain_target=stain, n_train=len(ids), feature_names=F.columns)
    print("[train] model fitted to all %d biopsies -> %s/model" % (len(ids), a.out_dir))


if __name__ == "__main__":
    main()
