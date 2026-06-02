"""Lever ② — feature fusion. Concatenate two per-frame feature sets (e.g.
ResNet50 2048-d + EndoViT 768-d) into one richer 2816-d sequence per video, so
the temporal head sees complementary "ImageNet" and "surgical-domain" views.

Both feature sets must come from the SAME 1fps frames (same length per video);
we align by min() defensively.

Run:
  python fuse_features.py --a features --b features_endovit_ft --out features_fused
"""
import argparse
from pathlib import Path

import torch

from splits import ALL_IDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for i in ALL_IDS:
        pa, pb = args.a / f"video{i:02d}.pt", args.b / f"video{i:02d}.pt"
        if not (pa.exists() and pb.exists()):
            continue
        da, db = torch.load(pa), torch.load(pb)
        n = min(da["feats"].shape[0], db["feats"].shape[0])
        feats = torch.cat([da["feats"][:n], db["feats"][:n]], dim=1)  # (n, Ca+Cb)
        torch.save({"feats": feats, "labels": da["labels"][:n]},
                   args.out / f"video{i:02d}.pt")
    # report dim
    sample = next(args.out.glob("video*.pt"))
    print(f"fused dim: {torch.load(sample)['feats'].shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
