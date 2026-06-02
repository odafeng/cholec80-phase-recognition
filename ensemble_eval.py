"""Lever ④ — ensemble. Average the per-frame softmax of several trained temporal
models (each with its own feature set) and evaluate the averaged prediction.
Ensembling is the most reliable accuracy lever on small datasets: it averages out
the per-model variance we saw (lucky/unlucky seeds, checkpoint-selection noise).

Each model is given as  CKPT:FEATURE_DIR  so models trained on different feature
sets (ResNet50 / EndoViT / fused) can be combined.

Run:
  python ensemble_eval.py --models \
     checkpoints/mstcn.pt:features \
     checkpoints/endovitft_mstcn.pt:features_endovit_ft \
     checkpoints/fused_mstcn.pt:features_fused
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, jaccard_score

from evaluate import build_from_ckpt
from phases import NUM_PHASES, PHASES
from splits import TEST_IDS


@torch.no_grad()
def model_probs(ckpt_path, feat_dir, device):
    """Return {video_id: softmax probs (T, C)} for one model on its features."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_from_ckpt(ckpt, device)
    out = {}
    for i in TEST_IDS:
        p = Path(feat_dir) / f"video{i:02d}.pt"
        if not p.exists():
            continue
        d = torch.load(p)
        feats = d["feats"].t().unsqueeze(0).to(device)        # (1, C, T)
        logits = model(feats)[-1]                              # (1, K, T)
        out[i] = (F.softmax(logits, dim=1).squeeze(0).t().cpu(),  # (T, K)
                  d["labels"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="CKPT:FEATDIR ...")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = [m.split(":") for m in args.models]
    per_model = [model_probs(c, f, device) for c, f in pairs]
    print(f"ensembling {len(pairs)} models:")
    for c, f in pairs:
        print(f"  {Path(c).name}  on  {f}")

    all_pred, all_gt, vid_acc = [], [], []
    for i in TEST_IDS:
        if not all(i in pm for pm in per_model):
            continue
        n = min(pm[i][0].shape[0] for pm in per_model)
        avg = sum(pm[i][0][:n] for pm in per_model) / len(per_model)   # (n, K)
        pred = avg.argmax(1).numpy()
        gt = per_model[0][i][1][:n].numpy()
        vid_acc.append((pred == gt).mean())
        all_pred.append(pred); all_gt.append(gt)
    P, G = np.concatenate(all_pred), np.concatenate(all_gt)

    rng = list(range(NUM_PHASES))
    prec = precision_score(G, P, labels=rng, average=None, zero_division=0)
    rec = recall_score(G, P, labels=rng, average=None, zero_division=0)
    jac = jaccard_score(G, P, labels=rng, average=None, zero_division=0)
    print(f"\n===== ENSEMBLE ({len(pairs)} models) =====")
    print(f"frame accuracy     : {(P == G).mean()*100:.2f}%")
    print(f"video-averaged acc : {np.mean(vid_acc)*100:.2f}%  (+/- {np.std(vid_acc)*100:.2f})")
    print(f"mean P / R / Jacc  : {prec.mean()*100:.1f} / {rec.mean()*100:.1f} / {jac.mean()*100:.1f}")


if __name__ == "__main__":
    main()
