"""Stage 1.5 — dump per-video 2048-d feature time-series from the trained CNN.

For each video we run the ResNet50 (minus its fc layer) over frames IN TIME
ORDER and save:
    features/videoXX.pt = { "feats": Tensor(T, 2048), "labels": Tensor(T) }

These per-video sequences are the input to the temporal models (Stage 2).
We use the eval transform (resize+centercrop, NO augmentation/shuffle).

Run:
  python extract_features.py --frames data/frames --anno data/phase_annotations \
        --ckpt checkpoints/resnet50.pt --out features
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_index, Cholec80FrameDataset
from splits import ALL_IDS
from train_cnn import build_model, make_transforms


def feature_extractor(ckpt_path, device):
    model = build_model()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.fc = nn.Identity()        # output the 2048-d pooled features
    return model.to(device).eval()


@torch.no_grad()
def extract_video(model, pairs, transform, device, bs=128, workers=8):
    ds = Cholec80FrameDataset(pairs, transform)
    ld = DataLoader(ds, bs, shuffle=False, num_workers=workers, pin_memory=True)
    feats, labels = [], []
    for x, y in ld:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda"):
            f = model(x)
        feats.append(f.float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--anno", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("features"))
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, eval_t = make_transforms()
    model = feature_extractor(args.ckpt, device)
    args.out.mkdir(parents=True, exist_ok=True)

    # build_index per single video so we keep per-video sequences
    for i in tqdm(ALL_IDS, desc="videos"):
        _, per_video = build_index(args.frames, args.anno, [i])
        if i not in per_video:
            continue
        feats, labels = extract_video(model, per_video[i], eval_t, device,
                                      args.bs, args.workers)
        torch.save({"feats": feats, "labels": labels}, args.out / f"video{i:02d}.pt")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
