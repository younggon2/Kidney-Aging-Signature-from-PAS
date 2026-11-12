#!/usr/bin/env python3
"""Stain-normalize tiles and extract UNI2-h features (GPU recommended).

python scripts/embed_tiles.py --tile-dir work/tiles --stain-target model/stain_target_lab.json --out work/tile_features.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pas_aging import BACKGROUND_RGB_THRESHOLD  # noqa: E402
from pas_aging.embed import embed_tiles, load_encoder  # noqa: E402
from pas_aging.stain import load_target, reinhard  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile-dir", required=True)
    ap.add_argument("--stain-target", required=True, help="model/stain_target_lab.json (or a .npy of 6 values)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="hf-hub:MahmoodLab/UNI2-h")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--bg-thr", type=int, default=BACKGROUND_RGB_THRESHOLD)
    a = ap.parse_args()
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    target = load_target(a.stain_target)
    model, transform = load_encoder(a.model, device)
    df = embed_tiles(a.tile_dir, model, transform, device, a.batch, stain_fn=lambda im: reinhard(im, target, a.bg_thr))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print("[embed] %d tiles x %d features -> %s" % (len(df), sum(c.startswith("emb__") for c in df.columns), a.out))


if __name__ == "__main__":
    main()
