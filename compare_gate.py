"""Decisive test for Path B: does the boundary head EARN its place at inference?

For a +BUA checkpoint (boundary head already trained), compute test metrics under:
  raw            : argmax, no smoothing
  conf-smooth    : plain confidence smoothing (current BUA inference)
  bgate(kappa)   : boundary-GATED smoothing (boundary head used at inference)
The question: can bgate match conf-smooth's over-segmentation at LOWER latency
(or beat over-seg at equal latency)? If yes, the boundary head is a real
contribution, not a decoration. Runs on CPU. kappa is meant to be selected on VAL.

Run:
  CUDA_VISIBLE_DEVICES="" python compare_gate.py \
      --ckpt checkpoints/rel_rn50/tecno_bua_s0.pt --features features [--split val|test]
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from evaluate import build_from_ckpt, _load_temperature
from train_tcn import load_features
from smooth import online_uncertainty_smooth, boundary_gated_smooth
from metrics import accuracy, over_segmentation, transition_latency, calibration, segmental_f1
from splits import VAL_IDS, TEST_IDS

BETAS = [0.5, 1.0, 2.0]


@torch.no_grad()
def collect(ckpt_path, feat_dir, ids):
    """Per video -> (class_probs (T,C), boundary_prob (T,), gt). One forward each."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not ckpt["cfg"].get("boundary", False):
        raise SystemExit("checkpoint has no boundary head (not a +BUA model)")
    model = build_from_ckpt(ckpt, "cpu")
    T = _load_temperature("auto", ckpt_path) or 1.0
    ids = [i for i in ids if (Path(feat_dir) / f"video{i:02d}.pt").exists()]
    out = []
    for i, (feats, labels) in zip(ids, load_features(Path(feat_dir), ids)):
        outputs, blogits = model(feats, return_boundary=True)
        logits = outputs[-1].squeeze(0).t().numpy() / T
        probs = torch.softmax(torch.as_tensor(logits), 1).numpy()
        bprob = torch.sigmoid(blogits.squeeze()).numpy()
        out.append((probs, bprob, labels.numpy()))
    return out


def metrics_of(seqs, mode, beta=1.0, gamma=0.5, conf_floor=0.4):
    acc, oseg, lat, f1, bece = [], [], [], [], []
    for probs, bprob, gt in seqs:
        if mode == "raw":
            sp = probs
        elif mode == "conf":
            sp = online_uncertainty_smooth(probs, gamma, conf_floor)
        else:  # bgate
            sp = boundary_gated_smooth(probs, bprob, gamma, conf_floor, beta)
        n = min(len(gt), sp.shape[0]); sp, g = sp[:n], gt[:n]
        pred = sp.argmax(1)
        acc.append(accuracy(pred, g))
        oseg.append(over_segmentation(pred, g)["ratio"])
        lat.append(transition_latency(pred, g)["median_latency"])
        f1.append(segmental_f1(pred, g, 0.10)[0])
        bece.append(calibration(sp, g)["boundary_ece"])
    return (np.nanmean(acc) * 100, np.nanmean(oseg), np.nanmean(lat),
            np.nanmean(f1), np.nanmean(bece) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()
    ids = VAL_IDS if args.split == "val" else TEST_IDS
    seqs = collect(args.ckpt, args.features, ids)
    print(f"# {args.ckpt.name} on {args.features} [{args.split}]")
    print(f"{'mode':16s}{'acc%':>7}{'over-seg':>9}{'lat(s)':>8}{'F1@10':>7}{'bECE%':>7}")
    for mode, kap in [("raw", None), ("conf", None)] + [("bgate", k) for k in BETAS]:
        a, o, l, f, b = metrics_of(seqs, mode, beta=kap or 1.0)
        tag = f"bgate b={kap}" if mode == "bgate" else mode
        print(f"{tag:16s}{a:>7.1f}{o:>9.2f}{l:>8.1f}{f:>7.1f}{b:>7.1f}")


if __name__ == "__main__":
    main()
