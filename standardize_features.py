"""Standardize a feature directory: per-dimension z-score using TRAIN-set
statistics only (no test leakage). Makes the temporal head invariant to each
backbone's feature scale (the confound that collapsed surgMAE training).

Run:  python standardize_features.py --src features.raw --dst features
"""
import argparse
from pathlib import Path
import numpy as np
import torch

from splits import TRAIN_IDS, ALL_IDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    args = ap.parse_args()

    # per-dim mean/std from TRAIN videos only
    tr = np.concatenate([
        torch.load(args.src / f"video{i:02d}.pt")["feats"].numpy()
        for i in TRAIN_IDS if (args.src / f"video{i:02d}.pt").exists()])
    mu = tr.mean(0)
    sd = tr.std(0) + 1e-6

    args.dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in ALL_IDS:
        p = args.src / f"video{i:02d}.pt"
        if not p.exists():
            continue
        d = torch.load(p)
        f = (d["feats"].numpy() - mu) / sd
        torch.save({"feats": torch.tensor(f, dtype=torch.float32),
                    "labels": d["labels"]}, args.dst / f"video{i:02d}.pt")
        n += 1
    np.savez(args.dst / "stats.npz", mean=mu, std=sd)
    sample = np.linalg.norm(
        torch.load(args.dst / f"video{ALL_IDS[0]:02d}.pt")["feats"].numpy(), axis=1).mean()
    print(f"standardized {n} videos {args.src} -> {args.dst}  (L2-norm/frame now ~{sample:.1f})")


if __name__ == "__main__":
    main()
