#!/usr/bin/env python3
"""Summarize tile features per biopsy (p25, p50, p75, mean over tiles -> 6,144 values).

A tile-to-biopsy map is needed when a slide holds several biopsy cores (as the study slides did).
It is a CSV with columns slide_id, region_index, biopsy_id. Without it, one slide = one biopsy.

python scripts/pool_biopsies.py --features work/tile_features.csv [--tile-map work/tile_map.csv] --out work/biopsy_features.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pas_aging.pooling import pool_biopsies  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="tile feature CSV from embed_tiles.py")
    ap.add_argument("--tile-map", default=None, help="CSV: slide_id, region_index, biopsy_id")
    ap.add_argument("--min-tiles", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = pd.read_csv(a.features)
    if a.tile_map:
        m = pd.read_csv(a.tile_map)
        t = t.merge(m[["slide_id", "region_index", "biopsy_id"]], on=["slide_id", "region_index"], how="inner")
    else:
        t["biopsy_id"] = t["slide_id"].astype(str)
    F = pool_biopsies(t, "biopsy_id", a.min_tiles)
    F.to_csv(a.out, float_format="%.6g")
    print("[pool] %d biopsies x %d features -> %s" % (F.shape[0], F.shape[1], a.out))


if __name__ == "__main__":
    main()
