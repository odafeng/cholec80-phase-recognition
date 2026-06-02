"""Evaluate one of OUR end-to-end trained checkpoints on the Cholec80 test set
(41-80), online per-video protocol. Optionally dump per-video accuracies (--out)
for the cross-seed significance test."""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import CausalSurgicalMamba
from data.dataset import VideoClipDataset

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--backbone", default="convnextv2_tiny")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default=None, help="npz path to save per-video acc/jacc")
args = ap.parse_args()

DEVICE = "cuda"
TEST = list(range(41, 81))
NP = 7

model = CausalSurgicalMamba(
    num_phases=NP, backbone=args.backbone,
    head_chunk_size=32, chunk_size_block=64,
    chunk_size_fast_block=64, chunk_size_slow_block=64,
).eval().to(DEVICE)
c = torch.load(args.ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(c["model"] if isinstance(c, dict) and "model" in c else c)


@torch.no_grad()
def infer(vid):
    ds = VideoClipDataset(video_id=vid, data_root="cholec80_preprocessed",
                          phase_dir=os.path.abspath("phase_ann_pp"), tool_dir="_no_tools",
                          seq_len=128, img_size=224, tag_format="video{:02d}")
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    slow, logits, labels = None, [], []
    for frames, _, lab, mask, _ in ld:
        out, slow, _, _ = model.forward_clip(frames.to(DEVICE), slow_states=slow)
        if slow is not None:
            slow = [tuple(s.detach() for s in st) if st is not None else None for st in slow]
        msq = mask[0].cpu()
        logits.append(out[0].cpu()[msq]); labels.append(lab[0][msq])
    return torch.cat(logits).argmax(-1), torch.cat(labels)


def metrics(pred, lab):
    P, R, J = [], [], []
    for cph in range(NP):
        tp = ((pred == cph) & (lab == cph)).sum().item()
        fp = ((pred == cph) & (lab != cph)).sum().item()
        fn = ((pred != cph) & (lab == cph)).sum().item()
        P.append(tp / (tp + fp) if tp + fp else 0.0)
        R.append(tp / (tp + fn) if tp + fn else 0.0)
        J.append(tp / (tp + fp + fn) if tp + fp + fn else 0.0)
    return (pred == lab).float().mean().item(), sum(J) / NP


accs, jacs = [], []
for v in TEST:
    pred, lab = infer(v)
    a, j = metrics(pred, lab)
    accs.append(a); jacs.append(j)
    print(f"video{v:02d}: acc={a*100:5.1f} jacc={j*100:5.1f}", flush=True)

accs, jacs = np.array(accs), np.array(jacs)
if args.out:
    np.savez(args.out, accs=accs, jacs=jacs, videos=np.array(TEST))
    print(f"saved per-video metrics -> {args.out}")

print("\n" + "=" * 60)
print(f"[{args.tag}] Cholec80 test (40 videos) — backbone={args.backbone}")
print(f"  Accuracy: {accs.mean()*100:.2f}%  (+/- {accs.std()*100:.2f})")
print(f"  Jaccard : {jacs.mean()*100:.2f}")
print("=" * 60)
