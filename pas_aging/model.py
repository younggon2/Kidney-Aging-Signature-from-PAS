"""PAS -> compartment-score model: StandardScaler -> PCA(16) -> multi-output RidgeCV.

Targets (four compartment scores) are centred within each disease group before
fitting, so the model estimates deviation from the group mean. Out-of-fold
estimates use patient-grouped 5-fold cross-validation repeated 20 times
(fold assignment seeds 1000..1019); a biopsy's estimate is the mean over the
20 repetitions. Scaler, PCA and ridge are all refit inside each training fold.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import DOMAINS, N_COMPONENTS

ALPHAS = np.logspace(-2, 4, 25)


def disease_of(biopsy_id):
    """Leading letters of the identifier (e.g. 'DKD12' -> 'DKD')."""
    m = re.match(r"^([A-Za-z]+)", str(biopsy_id))
    return m.group(1) if m else str(biopsy_id)


def patient_key(biopsy_id):
    """Disease prefix + first number; a section suffix such as '_2' is dropped so that
    two sections of one patient never fall into different folds."""
    m = re.match(r"^([A-Za-z]+)(\d+)", str(biopsy_id))
    return (m.group(1) + m.group(2)) if m else re.sub(r"[^A-Za-z0-9]", "", str(biopsy_id))


def center_by_group(Y, groups):
    Y = np.array(Y, dtype=float, copy=True)
    means = {}
    for g in np.unique(groups):
        means[g] = Y[groups == g].mean(0)
        Y[groups == g] -= means[g]
    return Y, means


def grouped_folds(groups, n_folds, rng):
    """Shuffle the unique groups and deal them round-robin into n_folds folds."""
    order = rng.permutation(np.unique(groups))
    fold_of = {g: i % n_folds for i, g in enumerate(order)}
    fid = np.array([fold_of[g] for g in groups])
    return [(np.where(fid != f)[0], np.where(fid == f)[0]) for f in range(n_folds)]


def make_pipeline(n_components=N_COMPONENTS):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline as mk
    from sklearn.preprocessing import StandardScaler
    return mk(StandardScaler(), PCA(n_components, random_state=0), RidgeCV(alphas=ALPHAS))


def fit(X, Y, groups):
    """Fit on all biopsies. Returns (pipeline, per-group target means)."""
    Yc, means = center_by_group(Y, np.asarray(groups))
    pipe = make_pipeline(min(N_COMPONENTS, len(X) - 1)).fit(X, Yc)
    return pipe, means


def folds_from_ids(fold_ids):
    """fold_ids: (n,) integer fold number per biopsy -> list of (train_idx, test_idx)."""
    fold_ids = np.asarray(fold_ids, int)
    return [(np.where(fold_ids != f)[0], np.where(fold_ids == f)[0]) for f in np.unique(fold_ids)]


def out_of_fold(X, Y, biopsy_ids, disease=None, patient=None, n_folds=5, n_reps=20, seed0=1000,
                eval_group="DKD", fold_ids=None):
    """Repeated patient-grouped cross-validation.
    disease / patient default to disease_of(id) / patient_key(id); pass them explicitly when the
    identifiers do not encode them. fold_ids, an (n_reps, n) array of fold numbers, replaces the
    random fold assignment (used to reproduce the study's exact splits from data/biopsy_scores.csv).
    Returns (oof DataFrame of rep-averaged estimates, per-rep Spearman array (n_reps, 4) within eval_group)."""
    ids = [str(s) for s in biopsy_ids]
    dis = np.array([str(d) for d in disease]) if disease is not None else np.array([disease_of(s) for s in ids])
    pk = np.asarray(patient) if patient is not None else np.array([patient_key(s) for s in ids])
    Y = np.asarray(Y, float)
    if fold_ids is not None:
        fold_ids = np.asarray(fold_ids, int)
        n_reps = fold_ids.shape[0]
    acc = np.zeros((n_reps,) + Y.shape)
    for r in range(n_reps):
        splits = folds_from_ids(fold_ids[r]) if fold_ids is not None else grouped_folds(pk, n_folds, np.random.default_rng(seed0 + r))
        for i_tr, i_va in splits:
            Yc, _ = center_by_group(Y[i_tr], dis[i_tr])
            acc[r][i_va] = make_pipeline(min(N_COMPONENTS, len(i_tr) - 1)).fit(X[i_tr], Yc).predict(X[i_va])
    ev = dis == eval_group
    per_rep = np.array([[spearmanr(acc[r][ev, j], Y[ev, j]).correlation for j in range(Y.shape[1])]
                        for r in range(n_reps)])
    oof = pd.DataFrame(acc.mean(0), index=pd.Index(ids, name="biopsy_id"), columns=list(DOMAINS)[:Y.shape[1]])
    return oof, per_rep


def batch_shift(X_ref, X_new):
    """Constant feature shift that aligns a new staining batch with the training batch:
    mean of reference-group biopsies in the training batch minus mean of the same group in the new batch."""
    return np.asarray(X_ref, float).mean(0) - np.asarray(X_new, float).mean(0)


# ----------------------------------------------------------------------------- export / load
def _wcsv(path, arr, index, columns):
    pd.DataFrame(np.asarray(arr, dtype=float), index=index, columns=columns).to_csv(path, float_format="%.12g")


def export(pipe, means, out_dir, stain_target=None, shift=None, n_train=None, feature_names=None):
    """Write all fitted parameters as CSV text files (exact to 12 significant digits)."""
    sc, pca, rg = pipe.named_steps["standardscaler"], pipe.named_steps["pca"], pipe.named_steps["ridgecv"]
    os.makedirs(out_dir, exist_ok=True)
    d = len(sc.mean_)
    fcols = list(feature_names) if feature_names is not None else ["f%04d" % i for i in range(d)]
    pcs = ["PC%02d" % (k + 1) for k in range(pca.components_.shape[0])]
    doms = list(DOMAINS)[:rg.coef_.shape[0]]
    j = lambda n: os.path.join(out_dir, n)  # noqa: E731
    _wcsv(j("scaler_mean.csv"), sc.mean_[None, :], ["mean"], fcols)
    _wcsv(j("scaler_scale.csv"), sc.scale_[None, :], ["scale"], fcols)
    _wcsv(j("pca_mean.csv"), pca.mean_[None, :], ["mean"], fcols)
    _wcsv(j("pca_components.csv"), pca.components_, pcs, fcols)
    _wcsv(j("ridge_coef.csv"), rg.coef_, doms, pcs)
    _wcsv(j("ridge_intercept.csv"), np.atleast_1d(rg.intercept_)[None, :], ["intercept"], doms)
    W = (pca.components_.T @ rg.coef_.T) / sc.scale_[:, None]
    b = np.atleast_1d(rg.intercept_) - ((sc.mean_ / sc.scale_) + pca.mean_) @ (pca.components_.T @ rg.coef_.T)
    _wcsv(j("linear_weights.csv"), W, fcols, doms)
    _wcsv(j("linear_intercept.csv"), b[None, :], ["intercept"], doms)
    if shift is not None:
        _wcsv(j("batch2_feature_shift.csv"), np.asarray(shift, float)[None, :], ["shift"], fcols)
    with open(os.path.join(out_dir, "ridge_alpha.json"), "w") as fh:
        json.dump({"alpha": float(rg.alpha_), "alpha_grid": "logspace(-2, 4, 25)",
                   "n_components": int(pca.n_components_), "training_biopsies": n_train, "domains": DOMAINS}, fh, indent=1)
    with open(os.path.join(out_dir, "target_centering.json"), "w") as fh:
        json.dump({d: {k: float(v[j]) for k, v in means.items()} for j, d in enumerate(DOMAINS)}, fh, indent=1)
    if stain_target is not None:
        from .stain import save_target
        save_target(stain_target, os.path.join(out_dir, "stain_target_lab.json"))


class LinearModel:
    """Closed-form equivalent of the fitted pipeline: y = x @ W + b (x = 6,144 pooled features)."""

    def __init__(self, model_dir):
        Wd = pd.read_csv(os.path.join(model_dir, "linear_weights.csv"), index_col=0)
        self.feature_names = list(Wd.index)
        self.W = Wd.to_numpy(float)
        self.b = pd.read_csv(os.path.join(model_dir, "linear_intercept.csv"), index_col=0).to_numpy(float)[0]
        p = os.path.join(model_dir, "batch2_feature_shift.csv")
        self.shift = pd.read_csv(p, index_col=0).to_numpy(float)[0] if os.path.exists(p) else None
        with open(os.path.join(model_dir, "ridge_alpha.json")) as fh:
            self.meta = json.load(fh)

    def predict(self, X, apply_shift=False):
        X = np.asarray(X, float)
        if apply_shift:
            if self.shift is None:
                raise ValueError("no batch shift stored in model directory")
            X = X + self.shift
        return pd.DataFrame(X @ self.W + self.b, columns=self.meta["domains"])
