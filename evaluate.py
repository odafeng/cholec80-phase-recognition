"""Evaluate a trained temporal model on the Cholec80 TEST set (videos 41-80).

Reports the standard phase-recognition metrics:
  - Accuracy           (frame-level)
  - per-phase Precision / Recall / Jaccard(IoU)  and their macro mean
  - video-averaged accuracy (mean over videos, the common Cholec80 number)

Run:
  python evaluate.py --features features --ckpt checkpoints/tecno.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (precision_score, recall_score, jaccard_score,
                             confusion_matrix)

from mstcn import MultiStageTCN
from lovit import LoViT
from asformer import ASFormer
from transsvnet import TransSVNet
from surgformer import Surgformer
from phases import NUM_PHASES, PHASES
from splits import TEST_IDS
from metrics import evaluate_video
from smooth import online_uncertainty_smooth
from train_tcn import load_features


def build_from_ckpt(ckpt, device):
    cfg = ckpt["cfg"]
    in_dim = cfg.get("in_dim", 2048)   # 2048 = ResNet50, 768 = EndoViT
    boundary = cfg.get("boundary", False)
    arch = cfg.get("arch", "mstcn")
    if arch == "lovit":
        model = LoViT(in_dim=in_dim, num_classes=NUM_PHASES, d=cfg["d"],
                      heads=cfg["heads"], layers=cfg["layers"],
                      num_stages=cfg["stages"], causal=cfg["causal"],
                      boundary=boundary).to(device)
    elif arch == "asformer":
        model = ASFormer(in_dim=in_dim, num_classes=NUM_PHASES, num_stages=cfg["stages"],
                         num_layers=cfg["layers"], d=cfg["d"], heads=cfg["heads"],
                         causal=cfg["causal"], boundary=boundary).to(device)
    elif arch == "transsvnet":
        model = TransSVNet(in_dim=in_dim, num_classes=NUM_PHASES, num_stages=cfg["stages"],
                           num_layers=cfg["layers"], d=cfg["d"], heads=cfg["heads"],
                           causal=cfg["causal"], boundary=boundary).to(device)
    elif arch == "surgformer":
        model = Surgformer(in_dim=in_dim, num_classes=NUM_PHASES, num_stages=cfg["stages"],
                           num_layers=cfg["layers"], d=cfg["d"], heads=cfg["heads"],
                           causal=cfg["causal"], boundary=boundary).to(device)
    else:
        model = MultiStageTCN(cfg["stages"], cfg["layers"], cfg["fmaps"],
                              in_dim=in_dim, num_classes=NUM_PHASES,
                              causal=cfg["causal"], boundary=boundary).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def _load_temperature(arg, ckpt_path):
    """--temp: a float, or 'auto' to read the <ckpt>.temp.json sidecar, or None."""
    if arg is None:
        return None
    if str(arg).lower() == "auto":
        side = Path(str(ckpt_path) + ".temp.json")
        if not side.exists():
            print(f"[warn] --temp auto but no sidecar {side}; skipping")
            return None
        return float(json.loads(side.read_text())["temperature"])
    return float(arg)


@torch.no_grad()
def predict_probs(model, feats, device, temp=None, do_smooth=False,
                  gamma=0.5, conf_floor=0.4):
    """Return per-frame probabilities (T, C), optionally temperature-scaled and
    online-smoothed. Strictly causal-safe: smoothing only looks at the past."""
    logits = model(feats.to(device))[-1].squeeze(0).t().cpu().numpy()   # (T, C)
    if temp is not None:
        logits = logits / float(temp)
    probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    if do_smooth:
        probs = online_uncertainty_smooth(probs, gamma, conf_floor)
    return probs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--relaxed", action="store_true", help="also show relaxed-boundary acc")
    ap.add_argument("--temp", default=None, help="temperature: float or 'auto' (sidecar)")
    ap.add_argument("--smooth", action="store_true", help="uncertainty-aware causal smoothing")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--conf_floor", type=float, default=0.4)
    ap.add_argument("--tol", type=int, default=10, help="relaxed/boundary tolerance (s)")
    ap.add_argument("--stable_k", type=int, default=3, help="latency stability frames")
    ap.add_argument("--out", type=Path, default=None, help="results dir: per-video json + npz")
    ap.add_argument("--tag", default=None, help="npz filename stem (e.g. tecno_baseline_s0)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # our own trusted checkpoint; weights_only=False also loads older checkpoints
    # that accidentally stored pathlib.Path objects in cfg.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_from_ckpt(ckpt, device)
    arch = ckpt["cfg"].get("arch", "mstcn").upper()
    name = f"{arch} ({'causal/online' if ckpt.get('causal') else 'non-causal/offline'})"
    temp = _load_temperature(args.temp, args.ckpt)

    test_ids = [i for i in TEST_IDS if (args.features / f"video{i:02d}.pt").exists()]
    data = load_features(args.features, test_ids)

    # per-video metric accumulators (one value per test video) for npz/significance
    per_vid = {k: [] for k in [
        "accuracy", "relaxed_accuracy", "jaccard", "f1@10", "f1@25", "f1@50",
        "edit", "median_latency", "miss_rate", "false_start_count",
        "over_segmentation_ratio", "ece", "mce", "boundary_ece", "interior_ece"]}
    all_pred, all_gt, vid_json = [], [], {}

    for vid, (feats, labels) in zip(test_ids, data):
        probs = predict_probs(model, feats, device, temp, args.smooth,
                              args.gamma, args.conf_floor)
        gt = labels.numpy()
        n = min(len(gt), probs.shape[0])
        probs, gt = probs[:n], gt[:n]
        pred = probs.argmax(1)
        rep = evaluate_video(pred, gt, probs, tol=args.tol, stable_k=args.stable_k)
        per_vid["accuracy"].append(rep["accuracy"])
        per_vid["relaxed_accuracy"].append(rep["relaxed_accuracy"])
        per_vid["jaccard"].append(rep["perphase_mean_jaccard"])
        per_vid["f1@10"].append(rep["f1@10"])
        per_vid["f1@25"].append(rep["f1@25"])
        per_vid["f1@50"].append(rep["f1@50"])
        per_vid["edit"].append(rep["edit"])
        per_vid["median_latency"].append(rep["latency"]["median_latency"])
        per_vid["miss_rate"].append(rep["latency"]["miss_rate"])
        per_vid["false_start_count"].append(rep["latency"]["false_start_count"])
        per_vid["over_segmentation_ratio"].append(rep["over_segmentation"]["ratio"])
        per_vid["ece"].append(rep["calibration"]["ece"])
        per_vid["mce"].append(rep["calibration"]["mce"])
        per_vid["boundary_ece"].append(rep["calibration"]["boundary_ece"])
        per_vid["interior_ece"].append(rep["calibration"]["interior_ece"])
        vid_json[f"video{vid:02d}"] = {k: (v if not isinstance(v, dict) else v)
                                       for k, v in rep.items() if k != "calibration"}
        all_pred.append(pred); all_gt.append(gt)

    P, G = np.concatenate(all_pred), np.concatenate(all_gt)
    labels_range = list(range(NUM_PHASES))
    prec = precision_score(G, P, labels=labels_range, average=None, zero_division=0)
    rec = recall_score(G, P, labels=labels_range, average=None, zero_division=0)
    jac = jaccard_score(G, P, labels=labels_range, average=None, zero_division=0)

    def m(key):  # nan-safe mean over videos
        return float(np.nanmean(per_vid[key]))

    cfg_bits = []
    if temp is not None: cfg_bits.append(f"T={temp:.3f}")
    if args.smooth: cfg_bits.append(f"smooth(g={args.gamma},f={args.conf_floor})")
    suffix = ("  [" + ", ".join(cfg_bits) + "]") if cfg_bits else ""

    # ---- backward-compatible core numbers ----
    print(f"\n===== {name}{suffix} =====")
    print(f"frame accuracy      : {(P == G).mean()*100:.2f}%")
    print(f"video-averaged acc  : {m('accuracy')*100:.2f}%  (+/- {np.std(per_vid['accuracy'])*100:.2f})")
    if args.relaxed:
        print(f"relaxed acc (±{args.tol}s)  : {m('relaxed_accuracy')*100:.2f}%")
    print(f"\n{'phase':<26}{'prec':>8}{'rec':>8}{'jacc':>8}")
    for i, ph in enumerate(PHASES):
        print(f"{ph:<26}{prec[i]*100:>7.1f}{rec[i]*100:>8.1f}{jac[i]*100:>8.1f}")
    print(f"{'MEAN':<26}{prec.mean()*100:>7.1f}{rec.mean()*100:>8.1f}{jac.mean()*100:>8.1f}")

    # ---- reliability suite ----
    print(f"\n--- reliability suite ---")
    print(f"segmental F1@10/25/50 : {m('f1@10'):.1f} / {m('f1@25'):.1f} / {m('f1@50'):.1f}")
    print(f"segmental edit        : {m('edit'):.1f}")
    print(f"over-segmentation     : {m('over_segmentation_ratio'):.2f}x  "
          f"(pred/GT segment ratio)")
    print(f"transition latency    : median {m('median_latency'):.1f}s   "
          f"miss-rate {m('miss_rate')*100:.1f}%   false-starts {m('false_start_count'):.1f}/vid")
    print(f"calibration ECE / MCE : {m('ece')*100:.2f} / {m('mce')*100:.2f}")
    print(f"  interior-ECE        : {m('interior_ece')*100:.2f}")
    print(f"  boundary-ECE        : {m('boundary_ece')*100:.2f}   "
          f"<<< headline (boundary >> interior == over-confident at transitions)")

    # ---- dump for significance ----
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        stem = args.tag or args.ckpt.stem
        np.savez(args.out / f"{stem}.npz",
                 **{k: np.asarray(v, dtype=float) for k, v in per_vid.items()})
        def _js(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return float(o)
        (args.out / f"{stem}.json").write_text(json.dumps(vid_json, indent=2, default=_js))
        print(f"\nwrote {args.out / (stem + '.npz')}  (+ .json)")


if __name__ == "__main__":
    main()
