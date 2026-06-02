import os
"""Standalone evaluation of the released SurgicalMamba Cholec80 checkpoint on the
test set (videos 41-80), replicating the repo's online per-video metric
(_evaluate_videos: per-video acc/precision/recall/jaccard, averaged over videos).
No wandb / no cfg machinery."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import CausalSurgicalMamba
from data.dataset import VideoClipDataset

DEVICE = "cuda"
TEST = list(range(41, 81))
NUM_PHASES = 7

model = CausalSurgicalMamba(num_phases=NUM_PHASES, chunk_size_fast_block=64,
                            chunk_size_slow_block=64).eval().to(DEVICE)
sd = torch.load("ckpts/cholec80_release.pt", map_location="cpu")
model.load_state_dict(sd["model"] if "model" in sd else sd)


@torch.no_grad()
def infer(vid):
    ds = VideoClipDataset(video_id=vid, data_root="cholec80_preprocessed",
                          phase_dir=os.path.abspath("phase_pp"), tool_dir="_no_tools",
                          seq_len=128, img_size=224, tag_format="video{:02d}")
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    slow, logits, labels = None, [], []
    for frames, _, lab, mask, _ in ld:
        frames = frames.to(DEVICE)
        out, slow, _, _ = model.forward_clip(frames, slow_states=slow)
        if slow is not None:
            slow = [tuple(s.detach() for s in st) if st is not None else None for st in slow]
        msq = mask[0].cpu()
        logits.append(out[0].cpu()[msq]); labels.append(lab[0][msq])
    return torch.cat(logits).argmax(-1), torch.cat(labels)


def per_video_metrics(pred, lab):
    P, R, J = [], [], []
    for c in range(NUM_PHASES):
        tp = ((pred == c) & (lab == c)).sum().item()
        fp = ((pred == c) & (lab != c)).sum().item()
        fn = ((pred != c) & (lab == c)).sum().item()
        P.append(tp / (tp + fp) if tp + fp else 0.0)
        R.append(tp / (tp + fn) if tp + fn else 0.0)
        J.append(tp / (tp + fp + fn) if tp + fp + fn else 0.0)
    return ((pred == lab).float().mean().item(),
            sum(P) / NUM_PHASES, sum(R) / NUM_PHASES, sum(J) / NUM_PHASES)


accs, precs, recs, jacs = [], [], [], []
for v in TEST:
    pred, lab = infer(v)
    a, p, r, j = per_video_metrics(pred, lab)
    accs.append(a); precs.append(p); recs.append(r); jacs.append(j)
    print(f"video{v:02d}: acc={a*100:5.1f}  jacc={j*100:5.1f}", flush=True)

print("\n" + "=" * 64)
print("SurgicalMamba (official weights) — Cholec80 test (40 videos)")
print(f"  Accuracy : {np.mean(accs)*100:.2f}%  (+/- {np.std(accs)*100:.2f})")
print(f"  Precision: {np.mean(precs)*100:.2f}")
print(f"  Recall   : {np.mean(recs)*100:.2f}")
print(f"  Jaccard  : {np.mean(jacs)*100:.2f}")
print("  (paper reports 94.6% acc / 82.7% jaccard)")
print("=" * 64)
