#!/usr/bin/env python3
"""Compute the Reinhard stain-normalization target from a tile directory.

To reproduce the study, use the released target (model/stain_target_lab.json) instead of
recomputing it; this script is for new cohorts that are analysed on their own.

python scripts/make_stain_target.py --tile-dir work/tiles --out work/stain_target.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pas_aging import BACKGROUND_RGB_THRESHOLD  # noqa: E402
from pas_aging.stain import make_target, save_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile-dir", required=True)
    ap.add_argument("--out", required=True, help=".json or .npy")
    ap.add_argument("--n-sample", type=int, default=3000)
    ap.add_argument("--bg-thr", type=int, default=BACKGROUND_RGB_THRESHOLD)
    a = ap.parse_args()
    t = make_target(a.tile_dir, a.n_sample, a.bg_thr)
    save_target(t, a.out, a.bg_thr)
    print("[stain] target", [round(float(v), 2) for v in t], "->", a.out)


if __name__ == "__main__":
    main()
