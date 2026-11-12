"""Tile features from the UNI2-h pathology foundation model (frozen).

UNI2-h (MahmoodLab/UNI2-h on Hugging Face) is gated: accept the model terms and
run `huggingface-cli login` before use. The model is used only as a fixed
feature extractor; no weights are trained or distributed here.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

TILE_RE = re.compile(r"(?P<slide>.+)__r(?P<idx>\d+)\.png$")


def load_encoder(model_name="hf-hub:MahmoodLab/UNI2-h", device="cuda"):
    import timm
    import torch
    kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
              embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
              mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
              reg_tokens=8, dynamic_img_size=True)
    model = timm.create_model(model_name, pretrained=True, **kw)
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    model.eval().to(device)
    return model, transform


def embed_tiles(tile_dir, model, transform, device="cuda", batch=64, stain_fn=None):
    """Encode every PNG under tile_dir. Returns a DataFrame: slide_id, region_index, emb__0..emb__1535."""
    import torch
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    files = sorted(glob.glob(os.path.join(tile_dir, "**", "*.png"), recursive=True))
    meta, embs, buf = [], [], []

    def flush():
        if not buf:
            return
        with torch.inference_mode():
            embs.append(model(torch.stack(buf).to(device)).float().cpu().numpy())
        buf.clear()

    for i, fp in enumerate(files, 1):
        img = Image.open(fp).convert("RGB")
        if stain_fn is not None:
            img = stain_fn(img)
        buf.append(transform(img))
        m = TILE_RE.search(os.path.basename(fp))
        meta.append({"slide_id": m.group("slide"), "region_index": int(m.group("idx"))} if m
                    else {"slide_id": os.path.basename(fp), "region_index": -1})
        if len(buf) >= batch:
            flush()
            if i % (batch * 50) == 0:
                print("[embed] %d/%d" % (i, len(files)), flush=True)
    flush()
    E = np.vstack(embs) if embs else np.zeros((0, 1536), np.float32)
    out = pd.DataFrame(E, columns=["emb__%d" % k for k in range(E.shape[1])])
    return pd.concat([pd.DataFrame(meta).reset_index(drop=True), out], axis=1)
