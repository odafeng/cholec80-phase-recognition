"""Stage 1 — fine-tune ResNet50 for per-frame phase classification.

Plain PyTorch (no Lightning) so every step is visible. Uses:
  - ImageNet-pretrained ResNet50, fc -> 7 classes
  - class-weighted CrossEntropy (phases are very imbalanced)
  - AMP (mixed precision) for speed/memory on the T4
  - saves the checkpoint with best VAL accuracy

Run:
  python train_cnn.py --frames data/frames --anno data/phase_annotations \
                      --epochs 5 --bs 64 --out checkpoints/resnet50.pt
"""
import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

from dataset import build_index, Cholec80FrameDataset
from phases import NUM_PHASES
from splits import TRAIN_IDS, VAL_IDS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_transforms():
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_t = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_t, eval_t


def class_weights(index, num_classes):
    cnt = Counter(lab for _, lab in index)
    total = sum(cnt.values())
    # inverse-frequency weights, normalized to mean 1
    w = torch.tensor([total / (num_classes * cnt.get(c, 1)) for c in range(num_classes)])
    return w / w.mean()


def build_model():
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, NUM_PHASES)
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda"):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--anno", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/resnet50.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_t, eval_t = make_transforms()

    print("building index...")
    train_idx, _ = build_index(args.frames, args.anno, TRAIN_IDS)
    val_idx, _ = build_index(args.frames, args.anno, VAL_IDS)
    print(f"train frames: {len(train_idx)} | val frames: {len(val_idx)}")

    train_ds = Cholec80FrameDataset(train_idx, train_t)
    val_ds = Cholec80FrameDataset(val_idx, eval_t)
    train_ld = DataLoader(train_ds, args.bs, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=True)
    val_ld = DataLoader(val_ds, args.bs, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=True)

    model = build_model().to(device)
    w = class_weights(train_idx, NUM_PHASES).to(device)
    print("class weights:", [round(x, 2) for x in w.tolist()])
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_ld, desc=f"epoch {ep}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running = 0.9 * running + 0.1 * loss.item()
            pbar.set_postfix(loss=f"{running:.3f}")

        acc = evaluate(model, val_ld, device)
        print(f"epoch {ep}: val_acc = {acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "val_acc": acc, "epoch": ep}, args.out)
            print(f"  -> saved best to {args.out} (val_acc={acc:.4f})")

    print(f"done. best val_acc = {best:.4f}")


if __name__ == "__main__":
    main()
