#!/usr/bin/env python3
"""Tile PAS whole-slide images into 128 x 128 um tiles (224 px PNG, tissue >= 50 %).

python scripts/tile_slides.py --svs-dir /path/to/slides --out-dir work/tiles
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pas_aging.tiling import tile_slide  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--svs-dir", help="directory searched recursively for *.svs")
    ap.add_argument("--svs-list", help="text file with one slide path per line (alternative to --svs-dir)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tile-um", type=float, default=128.0)
    ap.add_argument("--out-px", type=int, default=224)
    ap.add_argument("--min-tissue", type=float, default=0.5)
    a = ap.parse_args()
    if a.svs_list:
        slides = [l.strip() for l in open(a.svs_list) if l.strip()]
    elif a.svs_dir:
        slides = sorted(glob.glob(os.path.join(a.svs_dir, "**", "*.svs"), recursive=True))
    else:
        raise SystemExit("give --svs-dir or --svs-list")
    os.makedirs(a.out_dir, exist_ok=True)
    frames = []
    for i, p in enumerate(slides, 1):
        try:
            m = tile_slide(p, a.out_dir, a.tile_um, a.out_px, a.min_tissue)
        except ValueError as e:
            print("[tile] %s" % e)
            continue
        frames.append(m)
        print("[tile] %d/%d %s: %d tiles" % (i, len(slides), os.path.basename(p), len(m)), flush=True)
    M = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    M.to_csv(os.path.join(a.out_dir, "tile_manifest.csv"), index=False)
    print("[tile] %d tiles from %d slides -> %s/tile_manifest.csv" % (len(M), len(frames), a.out_dir))


if __name__ == "__main__":
    main()
