"""Operating-point sweep for the uncertainty-aware smoothing (BUA).

The smoothing is a deploy-time knob (no retraining). We run the model ONCE per
VAL video to get calibrated probabilities, then sweep (gamma, conf_floor) purely in
numpy to trace the accuracy / over-segmentation / boundary-ECE trade-off. The
operating point is SELECTED on val and then applied unchanged on test -- so the
knob is never tuned on the test set. This recovers accuracy on weak backbones
(e.g. EndoViT-pretrained) and yields the paper's trade-off curve.

Runs on CPU by default (tiny temporal head) so it doesn't disturb GPU training.

Run:
  CUDA_VISIBLE_DEVICES="" python sweep_smoothing.py \
      --ckpt checkpoints/rel_endovit/lovit_causal_bua_s0.pt --features features_endovit
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import build_from_ckpt, _load_temperature
from train_tcn import load_features
from smooth import online_uncertainty_smooth
from metrics import accuracy, over_segmentation, calibration
from splits import VAL_IDS   # operating point is chosen on VAL, never test

GAMMAS = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
FLOORS = [0.3, 0.4, 0.5]


@torch.no_grad()
def raw_probs(ckpt_path, feat_dir):
    """One forward per test video -> list of (calibrated probs (T,C), gt)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_from_ckpt(ckpt, "cpu")
    T = _load_temperature("auto", ckpt_path) or 1.0
    # VAL set: the smoothing operating point (gamma, conf_floor) is SELECTED here,
    # then applied unchanged on test -> no test-set hyperparameter tuning.
    ids = [i for i in VAL_IDS if (Path(feat_dir) / f"video{i:02d}.pt").exists()]
    out = []
    for feats, labels in load_features(Path(feat_dir), ids):
        logits = model(feats)[-1].squeeze(0).t().numpy() / T
        p = torch.softmax(torch.as_tensor(logits), 1).numpy()
        out.append((p, labels.numpy()))
    return out


def agg(probs_list, gamma, floor):
    accs, oseg, bece = [], [], []
    for p, gt in probs_list:
        sp = online_uncertainty_smooth(p, gamma, floor) if gamma > 0 else p
        n = min(len(gt), sp.shape[0]); sp, g = sp[:n], gt[:n]
        pred = sp.argmax(1)
        accs.append(accuracy(pred, g))
        oseg.append(over_segmentation(pred, g)["ratio"])
        bece.append(calibration(sp, g)["boundary_ece"])
    return np.mean(accs) * 100, np.mean(oseg), np.mean(bece) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pl = raw_probs(args.ckpt, args.features)
    base_acc = agg(pl, 0.0, 0.4)[0]
    print(f"# {args.ckpt.name} on {args.features}  (no-smooth acc={base_acc:.2f}%)")
    print(f"{'gamma':>6}{'floor':>6}{'acc%':>8}{'dAcc':>7}{'over-seg':>9}{'bECE':>7}")
    grid = {}
    for f in FLOORS:
        for g in GAMMAS:
            a, o, b = agg(pl, g, f)
            grid[f"{g}_{f}"] = {"acc": a, "over_seg": o, "bece": b}
            tag = ""
            if g > 0 and a >= base_acc - 0.3 and o < 3.0:
                tag = "  <- non-inferior acc & low over-seg"
            print(f"{g:>6.1f}{f:>6.1f}{a:>8.2f}{a-base_acc:>+7.2f}{o:>9.2f}{b:>7.2f}{tag}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"base_acc": base_acc, "grid": grid}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
