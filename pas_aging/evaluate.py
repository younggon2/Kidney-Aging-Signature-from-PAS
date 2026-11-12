"""Evaluation statistics used in the manuscript.

* Spearman correlation between PAS-estimated and transcriptomic scores within a disease group.
* Partial Spearman correlation with eGFR slope: exposure and outcome are rank-transformed and
  standardized, each is regressed on the covariates (age and baseline eGFR standardized, sex as 0/1),
  and the residuals are correlated. P values come from Freedman-Lane permutation of the outcome
  residuals (two-sided, 10,000 permutations); a family-wise P uses the maximum |rho| across exposures.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr


def zrank(v):
    r = rankdata(v).astype(float)
    return (r - r.mean()) / (r.std() + 1e-12)


def _z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def covariate_design(age, male, baseline_egfr):
    return np.column_stack([np.ones(len(age)), _z(age), np.asarray(male, float), _z(baseline_egfr)])


def _resid(v, Z):
    return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]


def partial_spearman(x, y, Z):
    return float(np.corrcoef(_resid(zrank(x), Z), _resid(zrank(y), Z))[0, 1])


def freedman_lane(X, y, Z, n_perm=10000, seed=20261014):
    """X: (n, m) exposures sharing one permutation. Returns (rho (m,), two-sided P (m,), family-wise P (m,))."""
    X = np.asarray(X, float)
    ex = np.column_stack([_resid(zrank(X[:, j]), Z) for j in range(X.shape[1])])
    ex = (ex - ex.mean(0)) / (ex.std(0) + 1e-12)
    ry = zrank(y)
    ey = _resid(ry, Z)
    fit = ry - ey
    H = Z @ np.linalg.pinv(Z)

    def corr(v):
        e = v - H @ v
        e = (e - e.mean()) / (e.std() + 1e-12)
        return ex.T @ e / len(e)

    obs = corr(ry)
    rng = np.random.default_rng(seed)
    k_ind = np.zeros(X.shape[1])
    k_fam = np.zeros(X.shape[1])
    for _ in range(n_perm):
        c = np.abs(corr(fit + rng.permutation(ey)))
        k_ind += c >= np.abs(obs) - 1e-12
        k_fam += c.max() >= np.abs(obs) - 1e-12
    return obs, (k_ind + 1) / (n_perm + 1), (k_fam + 1) / (n_perm + 1)


def agreement(est, meas):
    """Spearman rho per column between two aligned arrays (n, m)."""
    est, meas = np.asarray(est, float), np.asarray(meas, float)
    return np.array([spearmanr(est[:, j], meas[:, j]).correlation for j in range(est.shape[1])])
