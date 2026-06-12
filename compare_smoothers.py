"""Fairness experiment the reviewer demands: does OUR smoothing beat the standard
online post-processor? On a BASELINE checkpoint, trace the over-seg-vs-latency
Pareto frontier of:
  - causal mode filter (standard baseline; sweep window)
  - confidence smoothing (ours; sweep gamma)
If ours doesn't dominate the mode filter, "smoothing" is not a contribution and the
paper is purely a benchmark. Selection should be on val; prints chosen split. CPU.
"""
import argparse
from pathlib import Path
import numpy as np
import torch

from evaluate import build_from_ckpt, _load_temperature
from train_tcn import load_features
from smooth import online_uncertainty_smooth, causal_mode_filter
from metrics import accuracy, over_segmentation, transition_latency, segmental_f1
from splits import VAL_IDS, TEST_IDS

WINDOWS = [3, 5, 9, 15, 25, 41]
GAMMAS = [0.2, 0.3, 0.5, 0.7, 1.0]


@torch.no_grad()
def collect(ckpt_path, feat_dir, ids):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_from_ckpt(ckpt, "cpu")
    T = _load_temperature("auto", ckpt_path) or 1.0
    ids = [i for i in ids if (Path(feat_dir) / f"video{i:02d}.pt").exists()]
    out = []
    for feats, labels in load_features(Path(feat_dir), ids):
        logits = model(feats)[-1].squeeze(0).t().numpy() / T
        probs = torch.softmax(torch.as_tensor(logits), 1).numpy()
        out.append((probs, labels.numpy()))
    return out


def stats(seqs, kind, knob):
    o, l, f = [], [], []
    for probs, gt in seqs:
        if kind == "mode":
            pred = causal_mode_filter(probs.argmax(1), window=knob)
        else:
            pred = online_uncertainty_smooth(probs, gamma=knob).argmax(1)
        n = min(len(gt), len(pred)); pred, g = pred[:n], gt[:n]
        o.append(over_segmentation(pred, g)["ratio"])
        l.append(transition_latency(pred, g)["median_latency"])
        f.append(segmental_f1(pred, g, 0.10)[0])
    return np.nanmean(o), np.nanmean(l), np.nanmean(f)


def pareto(points):  # (over-seg, latency) both lower better
    pts = sorted(points, key=lambda p: (p[0], p[1])); out, best = [], 1e9
    for p in pts:
        if p[1] < best:
            out.append(p); best = p[1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    ids = VAL_IDS if args.split == "val" else TEST_IDS
    seqs = collect(args.ckpt, args.features, ids)
    mode = [(*stats(seqs, "mode", w)[:2], w) for w in WINDOWS]
    conf = [(*stats(seqs, "conf", g)[:2], g) for g in GAMMAS]
    print(f"# {args.ckpt.name} [{args.split}]  over-seg / latency(s)")
    print("standard causal MODE filter frontier:")
    for o, l, w in pareto(mode):
        print(f"    over-seg {o:5.2f}  lat {l:5.1f}  (window={w})")
    print("OUR confidence smoothing frontier:")
    for o, l, g in pareto(conf):
        print(f"    over-seg {o:5.2f}  lat {l:5.1f}  (gamma={g})")
    for thr in (3.0, 4.0):
        cm = min([l for o, l, _ in mode if o <= thr], default=float("nan"))
        cc = min([l for o, l, _ in conf if o <= thr], default=float("nan"))
        v = "OURS wins" if cc < cm - 0.2 else ("mode wins" if cm < cc - 0.2 else "tie")
        print(f"  @over-seg<={thr}: mode lat={cm:.1f}  ours lat={cc:.1f}  -> {v}")


if __name__ == "__main__":
    main()
