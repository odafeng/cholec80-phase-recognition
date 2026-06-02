"""Lever ① — multi-task Stage-1 backbone (phase + tool).

Cholec80 ships 7 binary tool-presence labels per frame that we never used. Adding
them as an auxiliary task gives the backbone MORE supervision from the SAME 32
videos, which regularises the features and makes them more phase-discriminative
(this is what the original TeCNO feature extractor did).

Architecture: ResNet50 backbone -> phase head (7-way CE) + tool head (7 BCE).
Saved in the SAME format as train_cnn.py (a resnet50 whose fc IS the phase head),
so extract_features.py loads it unchanged.

Run:
  python train_cnn_mtl.py --frames data/frames --phase data/phase_annotations \
        --tool data/tool_annotations --epochs 5 --bs 64 --out checkpoints/resnet50_mtl.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from dataset import build_mtl_index, Cholec80MTLDataset
from phases import NUM_PHASES
from splits import TRAIN_IDS, VAL_IDS
from train_cnn import class_weights, make_transforms

NUM_TOOLS = 7


class MTLResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()              # backbone -> 2048
        self.phase_head = nn.Linear(2048, NUM_PHASES)
        self.tool_head = nn.Linear(2048, NUM_TOOLS)

    def forward(self, x):
        feat = self.backbone(x)
        return self.phase_head(feat), self.tool_head(feat)

    def export_resnet_state(self):
        """Return a resnet50(fc=phase_head) state_dict == train_cnn.py format."""
        m = resnet50()
        m.fc = nn.Linear(2048, NUM_PHASES)
        m.load_state_dict(self.backbone.state_dict(), strict=False)  # all but fc
        m.fc.load_state_dict(self.phase_head.state_dict())
        return m.state_dict()


@torch.no_grad()
def val_phase_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for x, yp, _ in loader:
        x, yp = x.to(device), yp.to(device)
        with torch.autocast("cuda"):
            pred = model(x)[0].argmax(1)
        correct += (pred == yp).sum().item()
        total += yp.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--phase", required=True, type=Path)
    ap.add_argument("--tool", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tool_w", type=float, default=1.0, help="tool loss weight")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/resnet50_mtl.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_t, eval_t = make_transforms()

    print("building MTL index...")
    train_idx = build_mtl_index(args.frames, args.phase, args.tool, TRAIN_IDS)
    val_idx = build_mtl_index(args.frames, args.phase, args.tool, VAL_IDS)
    print(f"train {len(train_idx)} | val {len(val_idx)}")

    train_ld = DataLoader(Cholec80MTLDataset(train_idx, train_t), args.bs,
                          shuffle=True, num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    val_ld = DataLoader(Cholec80MTLDataset(val_idx, eval_t), args.bs, shuffle=False,
                        num_workers=args.workers, pin_memory=True, persistent_workers=True)

    model = MTLResNet().to(device)
    pw = class_weights([(p, ph) for p, ph, _ in train_idx], NUM_PHASES).to(device)
    phase_crit = nn.CrossEntropyLoss(weight=pw)
    tool_crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        from tqdm import tqdm
        pbar = tqdm(train_ld, desc=f"epoch {ep}/{args.epochs}")
        for x, yp, yt in pbar:
            x, yp, yt = x.to(device), yp.to(device), yt.to(device)
            opt.zero_grad()
            with torch.autocast("cuda"):
                lp, lt = model(x)
                loss = phase_crit(lp, yp) + args.tool_w * tool_crit(lt, yt)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run = 0.9 * run + 0.1 * loss.item()
            pbar.set_postfix(loss=f"{run:.3f}")
        acc = val_phase_acc(model, val_ld, device)
        print(f"epoch {ep}: phase val_acc = {acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"model": model.export_resnet_state(), "val_acc": acc,
                        "epoch": ep}, args.out)
            print(f"  -> saved best to {args.out} (val_acc={acc:.4f})")
    print(f"done. best phase val_acc = {best:.4f}")


if __name__ == "__main__":
    main()
