"""Reinhard stain normalization in OpenCV L*a*b* space.

Each tile's per-channel mean and standard deviation are matched to a fixed
target. Statistics are computed from tissue pixels only (a pixel whose R, G and
B values are all above `bg_thr` is treated as background), but the transform is
applied to every pixel. The target is the median of the per-tile statistics of
up to 3,000 tiles sampled at random from the study slides.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

KEYS = ["L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std"]


def _lab(rgb):
    import cv2
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _tissue_mask(rgb, bg_thr):
    if not bg_thr:
        return None
    m = ~(rgb > bg_thr).all(2)
    return m if m.sum() >= 100 else None


def tile_stats(rgb, bg_thr=200):
    """Per-channel mean and std of the L*a*b* tissue pixels of one RGB tile."""
    lab = _lab(rgb)
    m = _tissue_mask(rgb, bg_thr)
    sel = [(lab[:, :, i] if m is None else lab[:, :, i][m]) for i in range(3)]
    return [float(s.mean()) for s in sel] + [float(s.std()) for s in sel]


def reinhard(pil_img, target, bg_thr=200):
    """Return a PIL image whose L*a*b* tissue statistics match `target` (6 values)."""
    import cv2
    from PIL import Image
    rgb = np.asarray(pil_img.convert("RGB"))
    lab = _lab(rgb)
    m = _tissue_mask(rgb, bg_thr)
    out = lab.copy()
    for i in range(3):
        ch = lab[:, :, i]
        src = ch if m is None else ch[m]
        mu, sd = src.mean(), src.std() + 1e-6
        out[:, :, i] = (ch - mu) / sd * target[3 + i] + target[i]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_LAB2RGB))


def make_target(tile_dir, n_sample=3000, bg_thr=200, seed=0):
    """Median per-tile L*a*b* statistics over a random sample of tiles (PNG files under tile_dir)."""
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    files = sorted(glob.glob(os.path.join(tile_dir, "**", "*.png"), recursive=True))
    rng = np.random.default_rng(seed)
    if len(files) > n_sample:
        files = [files[i] for i in sorted(rng.choice(len(files), n_sample, replace=False))]
    st = [tile_stats(np.asarray(Image.open(f).convert("RGB")), bg_thr) for f in files]
    if not st:
        raise SystemExit("no tiles found under %s" % tile_dir)
    return np.median(np.array(st), axis=0)


def load_target(path):
    """Read a target from .npy (6 values) or from model/stain_target_lab.json."""
    if path.endswith(".json"):
        with open(path) as fh:
            d = json.load(fh)
        return np.array([d[k] for k in KEYS], dtype=float)
    return np.load(path).astype(float)


def save_target(target, path, bg_thr=200):
    if path.endswith(".json"):
        d = {k: float(v) for k, v in zip(KEYS, target)}
        d["note"] = ("OpenCV L*a*b* (8-bit scale); statistics from tissue pixels only "
                     "(pixels with R,G,B all > %d treated as background)" % bg_thr)
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1)
    else:
        np.save(path, np.asarray(target, dtype=float))
