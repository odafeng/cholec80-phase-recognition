"""Evaluate a trained temporal model on the Cholec80 TEST set (videos 41-80).

Reports the standard phase-recognition metrics:
  - Accuracy           (frame-level)
  - per-phase Precision / Recall / Jaccard(IoU)  and their macro mean
  - video-averaged accuracy (mean over videos, the common Cholec80 number)

Run:
  python evaluate.py --features features --ckpt checkpoints/tecno.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (precision_score, recall_score, jaccard_score,
                             confusion_matrix)

from mstcn import MultiStageTCN
from lovit import LoViT
from phases import NUM_PHASES, PHASES
from splits import TEST_IDS
from train_tcn import load_features


def build_from_ckpt(ckpt, device):
    cfg = ckpt["cfg"]
    if cfg.get("arch", "mstcn") == "lovit":
        model = LoViT(in_dim=2048, num_classes=NUM_PHASES, d=cfg["d"],
                      heads=cfg["heads"], layers=cfg["layers"],
                      num_stages=cfg["stages"], causal=cfg["causal"]).to(device)
    else:
        model = MultiStageTCN(cfg["stages"], cfg["layers"], cfg["fmaps"],
                              in_dim=2048, num_classes=NUM_PHASES,
                              causal=cfg["causal"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # our own trusted checkpoint; weights_only=False also loads older checkpoints
    # that accidentally stored pathlib.Path objects in cfg.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_from_ckpt(ckpt, device)
    arch = ckpt["cfg"].get("arch", "mstcn").upper()
    name = f"{arch} ({'causal/online' if ckpt.get('causal') else 'non-causal/offline'})"

    test = load_features(args.features, TEST_IDS)
    all_pred, all_gt, vid_acc = [], [], []
    for feats, labels in test:
        pred = model(feats.to(device))[-1].argmax(1).squeeze(0).cpu().numpy()
        gt = labels.numpy()
        vid_acc.append((pred == gt).mean())
        all_pred.append(pred)
        all_gt.append(gt)
    P = np.concatenate(all_pred)
    G = np.concatenate(all_gt)

    acc = (P == G).mean()
    labels_range = list(range(NUM_PHASES))
    prec = precision_score(G, P, labels=labels_range, average=None, zero_division=0)
    rec = recall_score(G, P, labels=labels_range, average=None, zero_division=0)
    jac = jaccard_score(G, P, labels=labels_range, average=None, zero_division=0)

    print(f"\n===== {name} =====")
    print(f"frame accuracy      : {acc*100:.2f}%")
    print(f"video-averaged acc  : {np.mean(vid_acc)*100:.2f}%  (+/- {np.std(vid_acc)*100:.2f})")
    print(f"\n{'phase':<26}{'prec':>8}{'rec':>8}{'jacc':>8}")
    for i, ph in enumerate(PHASES):
        print(f"{ph:<26}{prec[i]*100:>7.1f}{rec[i]*100:>8.1f}{jac[i]*100:>8.1f}")
    print(f"{'MEAN':<26}{prec.mean()*100:>7.1f}{rec.mean()*100:>8.1f}{jac.mean()*100:>8.1f}")


if __name__ == "__main__":
    main()
