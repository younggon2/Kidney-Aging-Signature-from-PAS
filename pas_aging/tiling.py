"""Uniform grid tiling of PAS whole-slide images.

Tissue is detected on a 32-fold downsampled thumbnail (grayscale < 220 and
saturation > 15). Non-overlapping 128 x 128 um tiles are placed on a grid at
level 0, and tiles with at least 50 % tissue are kept, resized to 224 x 224
pixels (LANCZOS) and written as PNG. The slide's own microns-per-pixel value is
required; slides without it are skipped rather than assigned a default.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import MIN_TISSUE_FRACTION, TILE_PX, TILE_UM


def tissue_mask(slide, ds=32):
    from PIL import Image  # noqa: F401
    W, H = slide.dimensions
    thumb = slide.get_thumbnail((max(1, W // ds), max(1, H // ds))).convert("RGB")
    a = np.asarray(thumb, dtype=np.int16)
    gray = a.mean(2)
    sat = a.max(2) - a.min(2)
    return (gray < 220) & (sat > 15), (W / thumb.size[0], H / thumb.size[1])


def tile_slide(svs_path, out_dir, tile_um=TILE_UM, out_px=TILE_PX, min_tissue=MIN_TISSUE_FRACTION, level=0):
    """Write tiles of one slide to out_dir/<slide_id>/<slide_id>__r<idx>.png and return a manifest DataFrame."""
    import openslide
    from PIL import Image, ImageFile
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    sid = os.path.splitext(os.path.basename(svs_path))[0]
    sl = openslide.OpenSlide(svs_path)
    mpp = sl.properties.get("openslide.mpp-x")
    if mpp is None:
        sl.close()
        raise ValueError("%s: no microns-per-pixel property; skipped" % sid)
    mpp = float(mpp)
    win0 = tile_um / mpp
    scale = sl.level_downsamples[level]
    size = (max(1, int(round(win0 / scale))),) * 2
    M, (sx, sy) = tissue_mask(sl)
    W, H = sl.dimensions
    odir = os.path.join(out_dir, sid)
    os.makedirs(odir, exist_ok=True)
    rows, idx = [], 0
    for y0 in np.arange(0, H - win0, win0):
        for x0 in np.arange(0, W - win0, win0):
            r0, r1 = int(y0 / sy), int((y0 + win0) / sy)
            c0, c1 = int(x0 / sx), int((x0 + win0) / sx)
            sub = M[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)]
            frac = float(sub.mean()) if sub.size else 0.0
            idx += 1
            if frac < min_tissue:
                continue
            img = sl.read_region((int(x0), int(y0)), level, size).convert("RGB")
            img = img.resize((out_px, out_px), Image.LANCZOS)
            img.info.pop("icc_profile", None)
            img.save(os.path.join(odir, "%s__r%d.png" % (sid, idx)), optimize=True)
            rows.append({"slide_id": sid, "region_index": idx, "cx": float(x0 + win0 / 2),
                         "cy": float(y0 + win0 / 2), "tissue_frac": frac, "mpp": mpp})
    sl.close()
    return pd.DataFrame(rows)
