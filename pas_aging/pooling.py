"""Per-biopsy summary of tile features: 25th, 50th and 75th percentiles and the mean across tiles."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pool_iqr_mean(A):
    """A: (n_tiles, d) -> (4 d,) = [p25, p50, p75, mean]."""
    A = np.asarray(A, dtype=np.float32)
    return np.concatenate([np.percentile(A, 25, 0), np.percentile(A, 50, 0),
                           np.percentile(A, 75, 0), A.mean(0)]).astype(float)


def feature_names(d):
    return ["%s_%04d" % (p, i) for p in ("p25", "p50", "p75", "mean") for i in range(d)]


def pool_biopsies(tiles, biopsy_col="biopsy_id", min_tiles=10):
    """tiles: DataFrame with emb__* columns and a biopsy identifier column.
    Returns a DataFrame indexed by biopsy with 4 x d pooled features (biopsies with < min_tiles tiles are dropped)."""
    ec = [c for c in tiles.columns if c.startswith("emb__")]
    X = tiles[ec].to_numpy(np.float32)
    ids = tiles[biopsy_col].astype(str).to_numpy()
    rows, keep = [], []
    for s in pd.unique(ids):
        idx = np.where(ids == s)[0]
        if len(idx) < min_tiles:
            continue
        rows.append(pool_iqr_mean(X[idx]))
        keep.append(s)
    return pd.DataFrame(np.vstack(rows), index=pd.Index(keep, name=biopsy_col), columns=feature_names(len(ec)))
